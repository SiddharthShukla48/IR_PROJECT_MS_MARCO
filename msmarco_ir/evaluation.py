"""Vectorised IR metrics @k computed from one (n_queries x k) relevance matrix. No per-query Python loops.

Definitions (binary relevance, macro-averaged over queries):
  Precision@k = hits / k
  Recall@k    = hits / R                      R = number of relevant passages of the query
  MAP@k       = sum_i P@i * rel_i / R         (trec_eval `map_cut` convention: divide by R)
  nDCG@k      = DCG / IDCG,  gain = rel / log2(rank + 1),  IDCG assumes min(R, k) relevant passages on top
  MRR@k       = 1 / rank of the first relevant passage (0 if none)   [extra: the official MS MARCO metric]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .runs import PAD_PID, Run


class Qrels:
    """Qrels packed as sorted int64 keys (qid * base + pid) so membership is one vectorised np.isin."""

    def __init__(self, qrels: pd.DataFrame):
        qid = qrels["qid"].to_numpy(dtype=np.int64)
        pid = qrels["pid"].to_numpy(dtype=np.int64)
        self.base = int(pid.max()) + 2
        self.keys = np.unique(qid * self.base + pid)               # also guards against duplicate pairs
        self.qids, self.counts = np.unique(self.keys // self.base, return_counts=True)

    def relevance(self, run: Run) -> tuple[np.ndarray, np.ndarray]:
        """Returns rel (n, k) bool and R (n,) number of relevant passages for each run query."""
        run_keys = run.qids[:, None].astype(np.int64) * self.base + run.pids
        rel = np.isin(run_keys, self.keys) & (run.pids != PAD_PID)
        pos = np.searchsorted(self.qids, run.qids)
        if not np.array_equal(self.qids[np.minimum(pos, len(self.qids) - 1)], run.qids):
            raise ValueError("run contains queries that have no judgement")
        return rel, self.counts[pos]


def compute_metrics(rel: np.ndarray, n_rel: np.ndarray, k: int) -> dict:
    rel = rel[:, :k].astype(np.float64)
    ranks = np.arange(1, k + 1)
    hits_cum = np.cumsum(rel, axis=1)
    precision_at = hits_cum / ranks                           # (n, k) precision after each rank
    recall_at = hits_cum / n_rel[:, None]                     # (n, k) recall after each rank
    discount = 1.0 / np.log2(ranks + 1)
    dcg = (rel * discount).sum(axis=1)
    idcg = np.cumsum(discount)[np.minimum(n_rel, k) - 1]
    first_hit = rel.argmax(axis=1)
    found = rel.any(axis=1)

    return {
        "metrics": {
            f"precision@{k}": float(precision_at[:, -1].mean()),
            f"recall@{k}": float(recall_at[:, -1].mean()),
            f"map@{k}": float(((precision_at * rel).sum(axis=1) / n_rel).mean()),
            f"ndcg@{k}": float((dcg / idcg).mean()),
            f"mrr@{k}": float(np.where(found, 1.0 / (first_hit + 1), 0.0).mean()),
            f"hit_rate@{k}": float(found.mean()),
        },
        # macro-averaged precision/recall after each rank 1..k -> the precision-recall graph
        "pr_curve": {"rank": ranks.tolist(),
                     "precision": precision_at.mean(axis=0).tolist(),
                     "recall": recall_at.mean(axis=0).tolist()},
    }


def evaluate_run(run: Run, qrels: Qrels, k: int) -> dict:
    rel, n_rel = qrels.relevance(run.top(k))
    result = compute_metrics(rel, n_rel, k)
    result.update({
        "system": run.name,
        "k": k,
        "num_queries": int(len(run.qids)),
        "queries_with_no_results": int((run.pids[:, 0] == PAD_PID).sum()),
        "timing": {key: run.meta[key] for key in ("index_build_seconds", "search_seconds", "queries_per_second")
                   if key in run.meta},
    })
    if "components" in run.meta:
        result["components"] = run.meta["components"]
        result["rrf_k"] = run.meta["rrf_k"]
    return result
