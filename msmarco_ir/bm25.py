"""BM25 via Pyserini / Anserini (Lucene).

Indexing runs Anserini's multithreaded Java `IndexCollection` (one thread per JSONL shard).
Tokenisation, lower-casing, English stop-words and Porter stemming are done by Lucene's analyzer at
both index time and query time, so no custom text processing is needed here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import Config
from .data import iter_corpus_batches
from .runs import Run, empty_result
from .utils import free_memory, is_complete, log, mark_complete, read_json, timed, write_json

NAME = "bm25"


# --------------------------------------------------------------------------- index build
def _export_jsonl(cfg: Config) -> None:
    """Corpus parquet -> {"id": ..., "contents": ...} JSONL shards (Anserini's JsonCollection format).
    Several shards = several indexing threads."""
    out = cfg.jsonl_dir
    if is_complete(out):
        log.info("bm25: reusing JSONL shards in %s", out)
        return
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for i, (pids, texts) in enumerate(iter_corpus_batches(cfg)):
        lines = [f'{{"id":"{pid}","contents":{json.dumps(text, ensure_ascii=False)}}}'
                 for pid, text in zip(pids.tolist(), texts)]
        (out / f"shard_{i:05d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mark_complete(out)


def build_index(cfg: Config) -> None:
    final = cfg.index_dir(NAME)
    building = final.with_name(final.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with timed("bm25: export JSONL"):
        _export_jsonl(cfg)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(cfg.jsonl_dir),
        "--index", str(building),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(cfg.threads),
        "--storeRaw",            # raw JSON kept so the search UI can show passage text
    ]
    env = os.environ.copy()
    if cfg.jvm_heap_gb:
        env["JAVA_TOOL_OPTIONS"] = f"-Xmx{cfg.jvm_heap_gb}g"
    with timed("bm25: Anserini indexing"):
        log.info("bm25: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    write_json(building / "meta.json", {"build_seconds": time.perf_counter() - t0})
    mark_complete(building)
    shutil.rmtree(final, ignore_errors=True)
    building.rename(final)
    if not cfg.keep_jsonl:
        shutil.rmtree(cfg.jsonl_dir, ignore_errors=True)
    free_memory()


# --------------------------------------------------------------------------- retrieval
class Bm25Retriever:
    name = NAME

    def __init__(self, cfg: Config):
        from pyserini.search.lucene import LuceneSearcher   # starts the JVM on import

        self.cfg = cfg
        d = cfg.index_dir(NAME)
        if not is_complete(d):
            raise FileNotFoundError(f"BM25 index missing/incomplete at {d}")
        self.searcher = LuceneSearcher(str(d))
        self.searcher.set_bm25(cfg.bm25_k1, cfg.bm25_b)
        meta = d / "meta.json"
        self.build_seconds = read_json(meta).get("build_seconds") if meta.exists() else None

    def search(self, qids: np.ndarray, queries: list[str], k: int) -> Run:
        n = len(qids)
        pids, scores = empty_result(n, k)
        t0 = time.perf_counter()
        for start in range(0, n, self.cfg.search_chunk):
            stop = min(start + self.cfg.search_chunk, n)
            id_strs = [str(q) for q in qids[start:stop]]
            # one call, all queries searched concurrently by Lucene's own thread pool
            results = self.searcher.batch_search(queries[start:stop], id_strs, k=k, threads=self.cfg.threads)
            for row, qid in enumerate(id_strs, start=start):
                hits = results.get(qid) or ()
                if hits:
                    pids[row, :len(hits)] = [int(h.docid) for h in hits]
                    scores[row, :len(hits)] = [h.score for h in hits]
            del results
        dt = time.perf_counter() - t0
        return Run(NAME, np.asarray(qids), pids, scores,
                   {"search_seconds": dt, "queries_per_second": n / max(dt, 1e-9),
                    "index_build_seconds": self.build_seconds})

    def fetch_text(self, pid: int) -> str:
        doc = self.searcher.doc(str(pid))
        return json.loads(doc.raw())["contents"] if doc is not None else ""
