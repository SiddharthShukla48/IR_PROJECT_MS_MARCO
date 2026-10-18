"""Query-time service used by the search UI: lazy-loads indexes from disk and answers single queries."""
from __future__ import annotations

import numpy as np

from .config import BASE_SYSTEMS, DENSE_SPECS, Config
from .fusion import rrf_fuse
from .reporting import label_of
from .runs import PAD_PID, Run
from .utils import is_complete


class SearchService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._retrievers: dict = {}

    # -- which systems can be served right now (index exists on disk)
    def available_base(self) -> list[str]:
        return [s for s in BASE_SYSTEMS if is_complete(self.cfg.index_dir(s))]

    def available_systems(self) -> dict[str, str]:
        base = self.available_base()
        systems = {s: label_of(s) for s in base}
        for name, parts in self.cfg.hybrids.items():
            if all(p in base for p in parts):
                systems[name] = label_of(name)
        return systems

    def _retriever(self, system: str):
        if system not in self._retrievers:                       # index is loaded once, on first use
            if system == "tfidf":
                from .tfidf import TfidfRetriever
                self._retrievers[system] = TfidfRetriever(self.cfg)
            elif system == "bm25":
                from .bm25 import Bm25Retriever
                self._retrievers[system] = Bm25Retriever(self.cfg)
            else:
                from .dense import DenseRetriever
                self._retrievers[system] = DenseRetriever(self.cfg, DENSE_SPECS[system])
        return self._retrievers[system]

    def _run(self, system: str, query: str, depth: int) -> Run:
        return self._retriever(system).search(np.array([0], dtype=np.int64), [query], depth)

    def passage_text(self, pid: int) -> str:
        return self._retriever("bm25").fetch_text(pid)        # raw text is stored inside the BM25 index

    def search(self, system: str, query: str, k: int = 10) -> list[dict]:
        depth = max(self.cfg.retrieve_k, k)
        if system in self.cfg.hybrids:
            run = rrf_fuse([self._run(p, query, depth) for p in self.cfg.hybrids[system]],
                           k_out=k, rrf_k=self.cfg.rrf_k, name=system)
        else:
            run = self._run(system, query, depth).top(k)
        return [{"rank": r + 1, "pid": int(p), "score": float(s), "text": self.passage_text(int(p))}
                for r, (p, s) in enumerate(zip(run.pids[0], run.scores[0])) if p != PAD_PID]
