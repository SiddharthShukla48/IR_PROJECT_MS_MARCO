"""Reciprocal Rank Fusion, fully vectorised in numpy (no per-query Python loop).

    RRF(d) = sum over systems s of 1 / (rrf_k + rank_s(d))        (rank starts at 1; rrf_k = 60 in the paper)

All runs are flattened into (query, doc, contribution) triples; np.unique over a packed int64 key groups
identical (query, doc) pairs, np.bincount sums their contributions, one lexsort orders the result.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .runs import PAD_PID, Run, empty_result


def rrf_fuse(runs: Sequence[Run], k_out: int, rrf_k: int = 60, name: str = "rrf") -> Run:
    base = runs[0]
    for r in runs[1:]:
        if not np.array_equal(r.qids, base.qids):
            raise ValueError(f"run '{r.name}' was produced for a different query order than '{base.name}'")
    n = len(base.qids)

    pids = np.concatenate([r.pids for r in runs], axis=1)                         # (n, sum of depths)
    ranks = np.concatenate([np.broadcast_to(np.arange(1, r.k + 1), r.pids.shape) for r in runs], axis=1)
    rows = np.broadcast_to(np.arange(n)[:, None], pids.shape)

    valid = (pids != PAD_PID).ravel()          # row-major ravel => order is (query, system, rank)
    p, rk, rw = pids.ravel()[valid], ranks.ravel()[valid], rows.ravel()[valid]
    contrib = 1.0 / (rrf_k + rk)

    out_p, out_s = empty_result(n, k_out)
    if p.size:
        base_mult = int(p.max()) + 1
        keys = rw.astype(np.int64) * base_mult + p
        uniq, first_pos, inverse = np.unique(keys, return_index=True, return_inverse=True)
        score = np.bincount(inverse.ravel(), weights=contrib, minlength=len(uniq))
        u_rows, u_pids = uniq // base_mult, uniq % base_mult

        # row asc, fused score desc; ties -> whichever doc appeared first (earlier system, better rank)
        order = np.lexsort((first_pos, -score, u_rows))
        u_rows, u_pids, score = u_rows[order], u_pids[order], score[order]

        row_start = np.flatnonzero(np.r_[True, u_rows[1:] != u_rows[:-1]])
        row_len = np.diff(np.r_[row_start, len(u_rows)])
        pos = np.arange(len(u_rows)) - np.repeat(row_start, row_len)
        keep = pos < k_out
        out_p[u_rows[keep], pos[keep]] = u_pids[keep]
        out_s[u_rows[keep], pos[keep]] = score[keep]

    return Run(name, base.qids, out_p, out_s, {"components": [r.name for r in runs], "rrf_k": rrf_k})
