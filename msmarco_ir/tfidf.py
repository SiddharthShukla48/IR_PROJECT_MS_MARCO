"""TF-IDF cosine retrieval.

Speed design (nothing per-token happens in Python):
  * tokenisation + stop-words: sklearn CountVectorizer in parallel processes, one corpus shard each
  * stemming: PyStemmer over the *vocabulary* only (~1-3M unique terms) instead of ~500M tokens; the
    raw-term -> stem remap is a numpy fancy-index on the CSR column array
  * weighting (sublinear tf, idf, L2 norm): in-place vectorised numpy / scipy kernels
  * search: sparse_dot_topn = multithreaded C++ (Q @ D^T) with the top-k selection fused into the product,
    so the (queries x 8.8M) score matrix is never materialised
"""
from __future__ import annotations

import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .config import Config
from .data import iter_corpus_batches
from .preprocessing import count_vectorizer, new_stemmer, query_analyzer
from .runs import Run, csr_topk_to_arrays, empty_result
from .utils import free_memory, is_complete, log, mark_complete, ordered_parallel_map, read_json, timed, write_json
from sklearn.feature_extraction.text import CountVectorizer

NAME = "tfidf"
MATRIX_FILE, VOCAB_FILE, IDF_FILE, DOCIDS_FILE, META_FILE = (
    "doc_term_T.npz", "vocab.txt", "idf.npy", "docids.npy", "meta.json")


# --------------------------------------------------------------------------- weighting
def _tf_weight_inplace(counts: sp.csr_matrix) -> None:
    """Sublinear tf: 1 + ln(tf)."""
    np.log(counts.data, out=counts.data)
    counts.data += 1.0


# --------------------------------------------------------------------------- index build
def _count_shard(texts: list[str]) -> tuple[list[str], sp.csr_matrix]:
    """Worker process: lower-case, tokenise, drop stop-words, count terms for one shard."""
    cv = count_vectorizer(dtype=np.float32)
    try:
        counts = cv.fit_transform(texts)
    except ValueError:  # shard contained nothing but stop-words
        return [], sp.csr_matrix((len(texts), 0), dtype=np.float32)
    return cv.get_feature_names_out().tolist(), counts


def build_index(cfg: Config) -> None:
    final = cfg.index_dir(NAME)
    building = final.with_name(final.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    t0 = time.perf_counter()

    stemmer = new_stemmer()
    stem_ids: dict[str, int] = {}     # stem  -> global column id
    term_to_col: dict[str, int] = {}  # raw term -> global column id (cache across shards)
    blocks: list[sp.csr_matrix] = []
    pid_parts: list[np.ndarray] = []

    def shards():
        for pids, texts in iter_corpus_batches(cfg):
            pid_parts.append(pids)           # appended in submission order == result order
            yield texts

    with timed("tfidf: tokenise + count shards"):
        ctx = mp.get_context("spawn")        # spawn: safe even if a JVM / CUDA context already exists
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=ctx) as pool:
            for terms, counts in ordered_parallel_map(pool, _count_shard, shards(), cfg.workers + 2):
                new_terms = [t for t in terms if t not in term_to_col]
                if new_terms:
                    for term, stem in zip(new_terms, stemmer.stemWords(new_terms)):
                        sid = stem_ids.get(stem)
                        if sid is None:
                            sid = stem_ids[stem] = len(stem_ids)
                        term_to_col[term] = sid
                col_map = np.fromiter((term_to_col[t] for t in terms), dtype=np.int32, count=len(terms))
                block = sp.csr_matrix(
                    (counts.data, col_map[counts.indices], counts.indptr),
                    shape=(counts.shape[0], max(len(stem_ids), 1)))
                block.sum_duplicates()        # two surface forms of one stem inside a passage -> add counts
                blocks.append(block)
                log.info("  tfidf | shards done: %d | stems so far: %s", len(blocks), f"{len(stem_ids):,}")
    del term_to_col

    n_stems = len(stem_ids)
    with timed("tfidf: stack + weight + transpose"):
        blocks = [sp.csr_matrix((b.data, b.indices, b.indptr), shape=(b.shape[0], n_stems)) for b in blocks]
        docs = sp.vstack(blocks, format="csr", dtype=np.float32)
        del blocks
        n_docs = docs.shape[0]
        df = np.bincount(docs.indices, minlength=n_stems)
        idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)   # smoothed idf
        _tf_weight_inplace(docs)
        docs.data *= idf[docs.indices]
        normalize(docs, norm="l2", copy=False)
        docs_t = docs.T.tocsr()               # (stems x docs): row = posting list, what Q @ D^T walks
        del docs

    with timed("tfidf: save index"):
        sp.save_npz(building / MATRIX_FILE, docs_t, compressed=False)
        np.save(building / IDF_FILE, idf)
        np.save(building / DOCIDS_FILE, np.concatenate(pid_parts))
        vocab = sorted(stem_ids, key=stem_ids.get)
        (building / VOCAB_FILE).write_text("\n".join(vocab), encoding="utf-8")
        write_json(building / META_FILE, {
            "docs": n_docs, "stems": n_stems, "nnz": int(docs_t.nnz),
            "build_seconds": time.perf_counter() - t0})
        mark_complete(building)

    shutil.rmtree(final, ignore_errors=True)
    building.rename(final)
    del docs_t, idf, df, vocab, stem_ids
    free_memory()


# --------------------------------------------------------------------------- retrieval
class TfidfRetriever:
    name = NAME

    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = cfg.index_dir(NAME)
        if not is_complete(d):
            raise FileNotFoundError(f"TF-IDF index missing/incomplete at {d}")
        with timed("tfidf: load index"):
            self.docs_t: sp.csr_matrix = sp.load_npz(d / MATRIX_FILE).tocsr()
            self.docids = np.load(d / DOCIDS_FILE)
            self.idf = np.load(d / IDF_FILE)
            vocab = (d / VOCAB_FILE).read_text(encoding="utf-8").split("\n")
        self.meta = read_json(d / META_FILE)
        self._vectorizer = CountVectorizer(
            analyzer=query_analyzer(), vocabulary={t: i for i, t in enumerate(vocab)}, dtype=np.float32)
        del vocab

    def _vectorize(self, queries: list[str]) -> sp.csr_matrix:
        q = self._vectorizer.transform(queries).tocsr()
        _tf_weight_inplace(q)
        q.data *= self.idf[q.indices]
        normalize(q, norm="l2", copy=False)
        return q

    def search(self, qids: np.ndarray, queries: list[str], k: int) -> Run:
        n = len(qids)
        pids, scores = empty_result(n, k)
        t0 = time.perf_counter()
        for start in range(0, n, self.cfg.search_chunk):
            sl = slice(start, min(start + self.cfg.search_chunk, n))
            q = self._vectorize(queries[sl])
            top = sp_matmul_topn(q, self.docs_t, top_n=k, sort=True, n_threads=self.cfg.threads)
            pids[sl], scores[sl] = csr_topk_to_arrays(top, self.docids, k)
        dt = time.perf_counter() - t0
        return Run(NAME, np.asarray(qids), pids, scores,
                   {"search_seconds": dt, "queries_per_second": n / max(dt, 1e-9),
                    "index_build_seconds": self.meta.get("build_seconds")})
