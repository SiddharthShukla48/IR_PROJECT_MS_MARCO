"""Orchestration: for every system -> build or load index -> search -> save run -> evaluate -> save JSON.

Memory discipline: each system runs in its own function, so its index/searcher/encoder die with the stack
frame; `free_memory()` (gc + Arrow pool + CUDA cache + malloc_trim) is called once per stage boundary,
never inside a loop.
"""
from __future__ import annotations

import time

from .config import BASE_SYSTEMS, DENSE_SPECS, Config
from .data import EvalData, prepare_data
from .evaluation import Qrels, evaluate_run
from .fusion import rrf_fuse
from .reporting import plot_pr_curves, write_table
from .runs import Run
from .utils import free_memory, is_complete, log, read_json, timed, write_json


# --------------------------------------------------------------------------- per-system stage
def _module_for(system: str):
    """(build_index, make_retriever) for a base system; heavy imports stay lazy."""
    if system == "tfidf":
        from . import tfidf
        return tfidf.build_index, tfidf.TfidfRetriever
    if system == "bm25":
        from . import bm25
        return bm25.build_index, bm25.Bm25Retriever
    from . import dense
    spec = DENSE_SPECS[system]
    return (lambda cfg: dense.build_index(cfg, spec)), (lambda cfg: dense.DenseRetriever(cfg, spec))


def run_base_system(system: str, cfg: Config, data: EvalData) -> Run:
    if not cfg.rerun_search and Run.exists(cfg.runs_dir, system) and not cfg.rebuild_indexes:
        run = Run.load(cfg.runs_dir, system)
        if len(run.qids) == len(data.qids) and (run.qids == data.qids).all() and run.k >= cfg.retrieve_k:
            log.info("[%s] reusing saved run", system)
            return run
    build_index, make_retriever = _module_for(system)

    if cfg.rebuild_indexes or not is_complete(cfg.index_dir(system)):
        log.info("[%s] building index -> %s", system, cfg.index_dir(system))
        build_index(cfg)
        free_memory()                       # encoder / token tables / JSONL are done with
    else:
        log.info("[%s] loading existing index from %s", system, cfg.index_dir(system))

    retriever = make_retriever(cfg)
    with timed(f"[{system}] search {len(data.qids):,} queries (top {cfg.retrieve_k})"):
        run = retriever.search(data.qids, data.texts, cfg.retrieve_k)
    run.save(cfg.runs_dir)                  # persist immediately
    del retriever
    free_memory()
    return run


# --------------------------------------------------------------------------- evaluation + reporting
def _evaluate_and_store(run: Run, qrels: Qrels, cfg: Config) -> dict:
    result = evaluate_run(run, qrels, cfg.k)
    write_json(cfg.eval_dir / f"{run.name}.json", result)
    m = result["metrics"]
    log.info("[%s] P@%d=%.4f R@%d=%.4f MAP@%d=%.4f nDCG@%d=%.4f MRR@%d=%.4f", run.name,
             cfg.k, m[f"precision@{cfg.k}"], cfg.k, m[f"recall@{cfg.k}"], cfg.k, m[f"map@{cfg.k}"],
             cfg.k, m[f"ndcg@{cfg.k}"], cfg.k, m[f"mrr@{cfg.k}"])
    return result


def run_pipeline(cfg: Config, systems: list[str]) -> None:
    data = prepare_data(cfg)
    qrels = Qrels(data.qrels)
    run_hybrids = "rrf" in systems
    base = [s for s in BASE_SYSTEMS if s in systems]

    results: list[dict] = []
    runs: dict[str, Run] = {}
    for system in base:
        runs[system] = run_base_system(system, cfg, data)
        results.append(_evaluate_and_store(runs[system], qrels, cfg))

    if run_hybrids:
        for name, parts in cfg.hybrids.items():
            for p in parts:                                   # components may come from an earlier invocation
                if p not in runs and Run.exists(cfg.runs_dir, p):
                    runs[p] = Run.load(cfg.runs_dir, p)
            if not all(p in runs for p in parts):
                log.warning("[%s] skipped: missing runs for %s", name, [p for p in parts if p not in runs])
                continue
            t0 = time.perf_counter()
            fused = rrf_fuse([runs[p] for p in parts], k_out=cfg.k, rrf_k=cfg.rrf_k, name=name)
            fusion_s = time.perf_counter() - t0
            log.info("[%s] RRF fusion: %.2fs", name, fusion_s)
            # end-to-end cost of serving the hybrid = all component searches + the fusion itself
            total_s = fusion_s + sum(runs[p].meta.get("search_seconds", 0.0) for p in parts)
            fused.meta.update({"fusion_seconds": fusion_s, "search_seconds": total_s,
                               "queries_per_second": len(fused.qids) / max(total_s, 1e-9),
                               "index_build_seconds": sum(runs[p].meta.get("index_build_seconds") or 0.0 for p in parts)})
            fused.save(cfg.runs_dir)
            results.append(_evaluate_and_store(fused, qrels, cfg))

    if not results:
        log.warning("nothing was evaluated")
        return
    # also fold in systems evaluated in earlier invocations so the table / plot always show everything
    # (only if they were scored on the same query set)
    seen = {r["system"] for r in results}
    for path in sorted(cfg.eval_dir.glob("*.json")):
        if path.stem in seen or path.stem == "summary":
            continue
        earlier = read_json(path)
        if earlier.get("num_queries") == len(data.qids) and earlier.get("k") == cfg.k:
            results.append(earlier)
    order = list(BASE_SYSTEMS) + list(cfg.hybrids)
    results.sort(key=lambda r: order.index(r["system"]) if r["system"] in order else len(order))

    write_json(cfg.eval_dir / "summary.json", {"data": data.stats, "k": cfg.k, "results": results})
    table = write_table(results, cfg.eval_dir)
    plot_pr_curves(results, cfg.figures_dir / "precision_recall.png")
    log.info("\n%s", table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log.info("Wrote %s, %s", cfg.eval_dir / "summary.json", cfg.figures_dir / "precision_recall.png")
    del runs, results, data, qrels
    free_memory()
