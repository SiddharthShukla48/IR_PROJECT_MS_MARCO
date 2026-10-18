"""CLI entry point.

    uv run python run.py                                  # everything, 10k sampled eval queries
    uv run python run.py --systems tfidf bm25             # only the sparse systems
    uv run python run.py --systems rrf                    # re-fuse saved runs only
    uv run python run.py --eval-queries 0                 # all judged queries (slow for dense on CPU)
    uv run python run.py --corpus-size 1000000            # smaller corpus (gold passages always kept)
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from msmarco_ir.config import BASE_SYSTEMS, Config
from msmarco_ir.utils import configure_jvm, log, setup_logging


def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", default=[*BASE_SYSTEMS, "rrf"], choices=[*BASE_SYSTEMS, "rrf"])
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--artifacts-dir", type=Path, default=d.artifacts_dir)
    p.add_argument("--eval-queries", type=int, default=d.eval_queries, help="0 = all judged queries")
    p.add_argument("--corpus-size", type=int, default=0, help="0 = full corpus")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--k", type=int, default=d.k, help="evaluation cut-off")
    p.add_argument("--retrieve-k", type=int, default=d.retrieve_k, help="depth fetched per system (>=k)")
    p.add_argument("--threads", type=int, default=d.threads)
    p.add_argument("--workers", type=int, default=d.workers, help="TF-IDF tokenisation processes")
    p.add_argument("--device", default=d.device)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--faiss-device", default=d.faiss_device)
    p.add_argument("--faiss-fp16-storage", action="store_true", help="halve dense-index RAM")
    p.add_argument("--jvm-heap-gb", type=int, default=None)
    p.add_argument("--rrf-k", type=int, default=d.rrf_k)
    p.add_argument("--bm25-k1", type=float, default=d.bm25_k1)
    p.add_argument("--bm25-b", type=float, default=d.bm25_b)
    p.add_argument("--rebuild-indexes", action="store_true")
    p.add_argument("--rerun-search", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = replace(
        Config(), data_dir=a.data_dir, artifacts_dir=a.artifacts_dir,
        eval_queries=a.eval_queries or None, corpus_size=a.corpus_size or None, seed=a.seed,
        k=a.k, retrieve_k=max(a.retrieve_k, a.k), threads=a.threads, workers=a.workers, device=a.device,
        fp16=not a.no_fp16, faiss_device=a.faiss_device, faiss_fp16_storage=a.faiss_fp16_storage,
        jvm_heap_gb=a.jvm_heap_gb, rrf_k=a.rrf_k, bm25_k1=a.bm25_k1, bm25_b=a.bm25_b,
        rebuild_indexes=a.rebuild_indexes, rerun_search=a.rerun_search,
    )
    setup_logging(cfg.logs_dir / "run.log")
    configure_jvm(cfg.jvm_heap_gb)          # before anything imports pyserini
    log.info("config: %s", cfg)

    from msmarco_ir.pipeline import run_pipeline   # imported late so JAVA_TOOL_OPTIONS is already set
    run_pipeline(cfg, a.systems)


if __name__ == "__main__":
    main()
