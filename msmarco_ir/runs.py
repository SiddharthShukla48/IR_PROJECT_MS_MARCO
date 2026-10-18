"""`Run` = top-k retrieval output of one system for a fixed, ordered list of queries."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .utils import read_json, write_json

PAD_PID = -1


@dataclass
class Run:
    name: str
    qids: np.ndarray      # (n,) int64
    pids: np.ndarray      # (n, k) int64, PAD_PID where fewer than k hits
    scores: np.ndarray    # (n, k) float32, NaN where padded
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.pids.shape[1]

    def top(self, k: int) -> "Run":
        return Run(self.name, self.qids, self.pids[:, :k], self.scores[:, :k], self.meta)

    def save(self, runs_dir: Path) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        tmp = runs_dir / f"{self.name}.tmp.npz"
        np.savez_compressed(tmp, qids=self.qids, pids=self.pids, scores=self.scores)
        tmp.replace(runs_dir / f"{self.name}.npz")
        write_json(runs_dir / f"{self.name}.meta.json", self.meta)

    @classmethod
    def exists(cls, runs_dir: Path, name: str) -> bool:
        return (runs_dir / f"{name}.npz").exists()

    @classmethod
    def load(cls, runs_dir: Path, name: str) -> "Run":
        with np.load(runs_dir / f"{name}.npz") as z:
            meta_path = runs_dir / f"{name}.meta.json"
            return cls(name, z["qids"], z["pids"], z["scores"], read_json(meta_path) if meta_path.exists() else {})


def empty_result(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((n, k), PAD_PID, dtype=np.int64), np.full((n, k), np.nan, dtype=np.float32)


def csr_topk_to_arrays(mat: sp.csr_matrix, id_lookup: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised unpack of a CSR matrix holding <=k entries per row (rows already sorted by score)."""
    n = mat.shape[0]
    pids, scores = empty_result(n, k)
    counts = np.diff(mat.indptr)
    rows = np.repeat(np.arange(n), counts)
    pos = np.arange(mat.nnz) - np.repeat(mat.indptr[:-1], counts)
    pids[rows, pos] = id_lookup[mat.indices]
    scores[rows, pos] = mat.data
    return pids, scores
