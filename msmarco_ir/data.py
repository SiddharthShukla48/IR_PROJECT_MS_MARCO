"""Load, validate, de-duplicate, clean and cache the MS MARCO files.

Everything heavy (TSV parsing, regex cleaning, filtering) runs inside Arrow's multithreaded C++ kernels.
The cleaned corpus is cached as parquet so later stages stream it in row-group sized pieces instead of
re-parsing a 3 GB TSV or holding 8.8M Python strings in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .config import Config
from .utils import free_memory, log, read_json, timed, write_json


@dataclass
class EvalData:
    queries: pd.DataFrame   # qid:int64, query:str      (evaluation queries only)
    qrels: pd.DataFrame     # qid:int64, pid:int64      (unique pairs, all pids exist in the corpus)
    stats: dict

    @property
    def qids(self) -> np.ndarray:
        return self.queries["qid"].to_numpy(dtype=np.int64)

    @property
    def texts(self) -> list[str]:
        return self.queries["query"].tolist()


# --------------------------------------------------------------------------- reading & cleaning
def _read_tsv(path, columns: list[str], types: list[pa.DataType]) -> tuple[pa.Table, int]:
    """Multithreaded TSV read. Quoting is disabled (passages contain raw quote characters) and malformed
    rows (wrong number of tabs) are skipped and counted instead of aborting the whole load."""
    skipped: list[int] = []
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=columns, block_size=1 << 26),
        parse_options=pacsv.ParseOptions(
            delimiter="\t", quote_char=False,
            invalid_row_handler=lambda row: (skipped.append(1) or "skip"),
        ),
        convert_options=pacsv.ConvertOptions(column_types=dict(zip(columns, types))),
    )
    return table, len(skipped)


def _clean_text(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Cleaning: control chars / newlines / runs of whitespace -> single space, trim ends."""
    col = pc.replace_substring_regex(col, pattern=r"[\p{Cc}\s]+", replacement=" ")
    return pc.utf8_trim_whitespace(col)


def _drop_empty(table: pa.Table, text_col: str) -> tuple[pa.Table, int]:
    keep = pc.fill_null(pc.greater(pc.utf8_length(table[text_col]), 0), False)
    n_bad = len(table) - (pc.sum(keep.cast(pa.int64())).as_py() or 0)
    return (table.filter(keep) if n_bad else table), n_bad


def _dedupe_first(table: pa.Table, key: str) -> tuple[pa.Table, int]:
    """Keep the first row of every id. np.unique is a single C sort, no Python loop."""
    ids = table[key].to_numpy()
    _, first = np.unique(ids, return_index=True)
    n_dup = len(ids) - len(first)
    if n_dup:
        table = table.take(pa.array(np.sort(first)))
    return table, n_dup


def _load_corpus(cfg: Config, stats: dict) -> pa.Table:
    table, bad_rows = _read_tsv(cfg.data_dir / cfg.collection_file, ["pid", "text"], [pa.int64(), pa.string()])
    stats["collection_rows_read"] = len(table)
    stats["collection_malformed_rows_skipped"] = bad_rows
    table = table.set_column(1, "text", _clean_text(table["text"]))
    table, stats["collection_empty_text_dropped"] = _drop_empty(table, "text")
    table, stats["collection_duplicate_pids_dropped"] = _dedupe_first(table, "pid")
    stats["collection_passages"] = len(table)
    return table


def _load_queries(cfg: Config, stats: dict) -> pd.DataFrame:
    table, bad_rows = _read_tsv(cfg.data_dir / cfg.queries_file, ["qid", "query"], [pa.int64(), pa.string()])
    stats["queries_rows_read"] = len(table)
    stats["queries_malformed_rows_skipped"] = bad_rows
    table = table.set_column(1, "query", _clean_text(table["query"]))
    table, stats["queries_empty_text_dropped"] = _drop_empty(table, "query")
    table, stats["queries_duplicate_qids_dropped"] = _dedupe_first(table, "qid")
    stats["queries_total"] = len(table)
    return table.to_pandas()


def _load_qrels(cfg: Config, stats: dict) -> pd.DataFrame:
    """qrels columns: qid, (always 0 -> ignored), pid, relevance (ignored: everything listed is relevant)."""
    table, bad_rows = _read_tsv(
        cfg.data_dir / cfg.qrels_file, ["qid", "_unused", "pid", "_rel"], [pa.int64()] * 4)
    stats["qrels_rows_read"] = len(table)
    stats["qrels_malformed_rows_skipped"] = bad_rows
    df = table.select(["qid", "pid"]).to_pandas()
    before = len(df)
    df = df.drop_duplicates(["qid", "pid"], ignore_index=True)
    stats["qrels_duplicate_pairs_dropped"] = before - len(df)
    return df


# --------------------------------------------------------------------------- selection
def _select_eval_subset(queries: pd.DataFrame, qrels: pd.DataFrame, cfg: Config, rng, stats: dict):
    judged = np.intersect1d(queries["qid"].to_numpy(), qrels["qid"].to_numpy())
    stats["queries_with_judgements"] = int(len(judged))
    if cfg.eval_queries and cfg.eval_queries < len(judged):
        judged = np.sort(rng.choice(judged, size=cfg.eval_queries, replace=False))
    queries = queries[queries["qid"].isin(judged)].sort_values("qid", ignore_index=True)
    qrels = qrels[qrels["qid"].isin(judged)].sort_values(["qid", "pid"], ignore_index=True)
    return queries, qrels


def _subset_corpus(table: pa.Table, gold_pids: np.ndarray, cfg: Config, rng, stats: dict) -> pa.Table:
    """Optional smaller corpus: every gold passage of the eval queries + random distractors."""
    if not cfg.corpus_size or cfg.corpus_size >= len(table):
        return table
    all_pids = table["pid"].to_numpy()
    if cfg.corpus_size <= len(gold_pids):
        log.warning("corpus_size (%d) <= #gold passages (%d): keeping gold passages only",
                    cfg.corpus_size, len(gold_pids))
        keep = gold_pids
    else:
        others = np.setdiff1d(all_pids, gold_pids, assume_unique=True)
        extra = rng.choice(others, size=cfg.corpus_size - len(gold_pids), replace=False)
        keep = np.concatenate([gold_pids, extra])
    stats["corpus_subset_size"] = int(len(keep))
    return table.filter(pa.array(np.isin(all_pids, keep)))


# --------------------------------------------------------------------------- public API
def _fingerprint(cfg: Config) -> dict:
    files = {}
    for name in (cfg.collection_file, cfg.queries_file, cfg.qrels_file):
        st = (cfg.data_dir / name).stat()
        files[name] = [st.st_size, st.st_mtime_ns]
    return {"eval_queries": cfg.eval_queries, "corpus_size": cfg.corpus_size,
            "seed": cfg.seed, "chunk": cfg.corpus_chunk_docs, "files": files}


def prepare_data(cfg: Config) -> EvalData:
    """Idempotent: re-uses the cached parquet files when inputs and sampling settings are unchanged."""
    meta_path = cfg.prepared_dir / "prepare_meta.json"
    queries_path = cfg.prepared_dir / "eval_queries.parquet"
    qrels_path = cfg.prepared_dir / "eval_qrels.parquet"
    fp = _fingerprint(cfg)

    if meta_path.exists() and cfg.corpus_parquet.exists() and queries_path.exists() and qrels_path.exists():
        meta = read_json(meta_path)
        if meta.get("fingerprint") == fp:
            log.info("Using cached prepared data in %s", cfg.prepared_dir)
            return EvalData(pd.read_parquet(queries_path), pd.read_parquet(qrels_path), meta["stats"])
        log.info("Input files or sampling settings changed -> re-preparing data "
                 "(existing indexes were built on the old corpus: rerun with --rebuild-indexes)")

    stats: dict = {}
    rng = np.random.default_rng(cfg.seed)
    cfg.prepared_dir.mkdir(parents=True, exist_ok=True)

    with timed("load + clean + dedupe collection"):
        corpus = _load_corpus(cfg, stats)
    with timed("load queries + qrels"):
        queries = _load_queries(cfg, stats)
        qrels = _load_qrels(cfg, stats)

    # consistency between the three files
    corpus_pids = corpus["pid"].to_numpy()
    in_corpus = np.isin(qrels["pid"].to_numpy(), corpus_pids)
    stats["qrels_pairs_with_unknown_pid_dropped"] = int((~in_corpus).sum())
    qrels = qrels[in_corpus]
    in_queries = qrels["qid"].isin(queries["qid"])
    stats["qrels_pairs_with_unknown_qid_dropped"] = int((~in_queries).sum())
    qrels = qrels[in_queries]

    per_q = qrels.groupby("qid").size()
    stats["queries_with_multiple_relevant"] = int((per_q > 1).sum())
    stats["max_relevant_per_query"] = int(per_q.max())

    queries, qrels = _select_eval_subset(queries, qrels, cfg, rng, stats)
    corpus = _subset_corpus(corpus, qrels["pid"].unique(), cfg, rng, stats)
    stats["eval_queries"] = len(queries)
    stats["eval_qrel_pairs"] = len(qrels)
    stats["corpus_passages_used"] = len(corpus)

    with timed("write prepared parquet"):
        pq.write_table(corpus, cfg.corpus_parquet, row_group_size=cfg.corpus_chunk_docs, compression="zstd")
        queries.to_parquet(queries_path, index=False)
        qrels.to_parquet(qrels_path, index=False)
        write_json(meta_path, {"fingerprint": fp, "stats": stats})

    for key, value in stats.items():
        log.info("  data | %-42s %s", key, f"{value:,}" if isinstance(value, int) else value)

    del corpus, corpus_pids, in_corpus, in_queries, per_q   # big Arrow table / arrays are dead from here on
    free_memory()
    return EvalData(queries, qrels, stats)


def corpus_num_docs(cfg: Config) -> int:
    return pq.ParquetFile(cfg.corpus_parquet).metadata.num_rows


def iter_corpus_batches(cfg: Config) -> Iterator[tuple[np.ndarray, list[str]]]:
    """Stream the cleaned corpus as (pids, texts) pieces of ~corpus_chunk_docs passages."""
    pf = pq.ParquetFile(cfg.corpus_parquet)
    for batch in pf.iter_batches(batch_size=cfg.corpus_chunk_docs, columns=["pid", "text"]):
        yield batch.column("pid").to_numpy(zero_copy_only=False), batch.column("text").to_pylist()
