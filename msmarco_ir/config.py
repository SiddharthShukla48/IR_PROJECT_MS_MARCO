"""All tunables live here so every other module stays free of magic numbers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CPU_COUNT = os.cpu_count() or 1


@dataclass(frozen=True)
class DenseSpec:
    """A dense bi-encoder: which checkpoints to use and how to pool them."""

    key: str                 # short name used for folders / run files
    kind: str                # "dpr" (CLS pooler output, dot product) | "sbert" (mean pool + L2 norm)
    doc_model: str
    query_model: str
    dim: int
    doc_max_len: int = 256
    query_max_len: int = 64
    label: str = ""


DENSE_SPECS: dict[str, DenseSpec] = {
    "dpr": DenseSpec(
        key="dpr", kind="dpr", dim=768, label="DPR (single-nq-base)",
        doc_model="facebook/dpr-ctx_encoder-single-nq-base",
        query_model="facebook/dpr-question_encoder-single-nq-base",
    ),
    "st": DenseSpec(
        key="st", kind="sbert", dim=384, label="Sentence-Transformer (all-MiniLM-L6-v2)",
        doc_model="sentence-transformers/all-MiniLM-L6-v2",
        query_model="sentence-transformers/all-MiniLM-L6-v2",
    ),
}

SYSTEM_LABELS = {
    "tfidf": "TF-IDF (cosine)",
    "bm25": "BM25 (Lucene)",
    **{k: v.label for k, v in DENSE_SPECS.items()},
}

# Order in which systems run / appear in tables.
BASE_SYSTEMS = ("tfidf", "bm25", "dpr", "st")


@dataclass(frozen=True)
class Config:
    # ---- input files (as downloaded from MS MARCO) ----
    data_dir: Path = Path("data/msmarco")
    collection_file: str = "collection.tsv"      # pid <TAB> passage
    queries_file: str = "queries.train.tsv"      # qid <TAB> query
    qrels_file: str = "qrels.train.tsv"          # qid <TAB> 0 <TAB> pid <TAB> 1

    # ---- outputs ----
    artifacts_dir: Path = Path("artifacts")

    # ---- experiment size ----
    eval_queries: int | None = 10_000   # None = every query that has a judgement (~500k)
    corpus_size: int | None = None      # None = full 8.8M passages; else subset (gold passages always kept)
    seed: int = 42

    # ---- retrieval ----
    k: int = 10             # evaluation cut-off (all metrics @k)
    retrieve_k: int = 10    # depth fetched per system; >k only helps RRF, metrics still use top-k
    search_chunk: int = 20_000   # queries per batch call (bounds RAM of result objects)

    # ---- hardware ----
    threads: int = CPU_COUNT          # Lucene / FAISS / sparse_dot_topn threads
    workers: int = min(CPU_COUNT, 16) # processes for TF-IDF tokenisation
    device: str = "auto"              # "auto" | "cpu" | "cuda" | "cuda:1"
    fp16: bool = True                 # half-precision encoders on GPU
    faiss_device: str = "cpu"         # "cpu" or "cuda:N" (needs faiss-gpu)
    faiss_fp16_storage: bool = False  # True halves index RAM (scalar-quantised fp16), tiny accuracy cost
    jvm_heap_gb: int | None = None    # sets -Xmx for Lucene indexing + searching if given

    # ---- corpus streaming ----
    corpus_chunk_docs: int = 250_000  # passages per parquet row-group / JSONL shard / TF-IDF shard / embedding chunk

    # ---- BM25 (Anserini MS MARCO-tuned values; Anserini default is 0.9 / 0.4) ----
    bm25_k1: float = 0.82
    bm25_b: float = 0.68
    keep_jsonl: bool = False          # keep the Lucene input JSONL after indexing

    # ---- dense ----
    dense_batch_size: int = 256
    query_batch_size: int = 512

    # ---- RRF ----
    rrf_k: int = 60
    hybrids: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "rrf_bm25_dpr": ("bm25", "dpr"),
        "rrf_bm25_st": ("bm25", "st"),
        "rrf_bm25_dpr_st": ("bm25", "dpr", "st"),
    })

    # ---- behaviour ----
    rebuild_indexes: bool = False
    rerun_search: bool = False

    # ---------------- derived paths ----------------
    @property
    def prepared_dir(self) -> Path:
        return self.artifacts_dir / "prepared"

    @property
    def corpus_parquet(self) -> Path:
        return self.prepared_dir / "corpus.parquet"

    @property
    def jsonl_dir(self) -> Path:
        return self.prepared_dir / "lucene_jsonl"

    def index_dir(self, name: str) -> Path:
        return self.artifacts_dir / "indexes" / name

    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"

    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "eval"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
