"""Dense retrieval: DPR and Sentence-Transformer bi-encoders, FAISS index, Pyserini `FaissSearcher`.

Why encoding is done here instead of `python -m pyserini.encode`: Pyserini's corpus encoder loads the whole
JSONL into Python lists, does not sort by length (so every batch is padded to its longest passage), ignores
fp16 and encodes queries one at a time. Here passages are streamed from parquet, length-sorted inside each
chunk, run in fp16 on GPU, and every chunk is checkpointed to disk so a multi-hour encode can be resumed.
The resulting index is written in Pyserini's on-disk layout (`index` + `docid`) and searched with Pyserini's
own `FaissSearcher.batch_search`.
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np

from .config import Config, DenseSpec
from .data import corpus_num_docs, iter_corpus_batches
from .runs import Run, empty_result
from .utils import free_memory, is_complete, log, mark_complete, read_json, timed, write_json


# --------------------------------------------------------------------------- batched text encoder
class TextEncoder:
    """Batched, length-sorted, mixed-precision encoder for one checkpoint."""

    def __init__(self, spec: DenseSpec, role: str, cfg: Config):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.spec = spec
        self.device = cfg.resolve_device()
        self.batch_size = cfg.dense_batch_size if role == "doc" else cfg.query_batch_size
        name = spec.doc_model if role == "doc" else spec.query_model
        self.max_length = spec.doc_max_len if role == "doc" else spec.query_max_len

        if spec.kind == "dpr":
            from transformers import DPRContextEncoder, DPRQuestionEncoder
            model_cls = DPRContextEncoder if role == "doc" else DPRQuestionEncoder
        else:
            model_cls = AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = model_cls.from_pretrained(name).to(self.device).eval()
        self.half = bool(cfg.fp16 and self.device.startswith("cuda"))
        if self.half:
            self.model.half()
        self.out_dtype = np.float16 if self.half else np.float32
        log.info("%s %s encoder: %s on %s (%s)", spec.key, role, name, self.device, "fp16" if self.half else "fp32")

    def _embed(self, texts: list[str]):
        torch = self.torch
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        if self.spec.kind == "sbert":
            enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc)
            if self.spec.kind == "dpr":
                emb = out.pooler_output                                   # CLS vector, scored by dot product
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean pooling
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)                    # cosine == dot
        return emb

    def encode(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        out = np.empty((n, self.spec.dim), dtype=self.out_dtype)
        lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n)
        order = np.argsort(-lengths, kind="stable")      # similar lengths share a batch; longest first
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            emb = self._embed([texts[i] for i in idx])
            out[idx] = emb.to("cpu").numpy().astype(self.out_dtype, copy=False)
        return out


# --------------------------------------------------------------------------- index build
def _save_npy_atomic(path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _new_faiss_index(dim: int, fp16_storage: bool):
    import faiss
    if fp16_storage:   # scalar-quantised fp16: half the RAM, no training needed
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    return faiss.IndexFlatIP(dim)   # exact inner-product search


def build_index(cfg: Config, spec: DenseSpec) -> None:
    import faiss

    final = cfg.index_dir(spec.key)
    building = final.with_name(final.name + ".building")   # kept between runs => resumable
    chunks = building / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_docs = corpus_num_docs(cfg)

    encoder = None
    n_chunks = 0
    with timed(f"{spec.key}: encode corpus"):
        for i, (pids, texts) in enumerate(iter_corpus_batches(cfg)):
            n_chunks += 1
            emb_path, pid_path = chunks / f"emb_{i:05d}.npy", chunks / f"pid_{i:05d}.npy"
            if emb_path.exists() and pid_path.exists():
                continue                                    # finished in an earlier run
            if encoder is None:
                encoder = TextEncoder(spec, "doc", cfg)
            t_chunk = time.perf_counter()
            emb = encoder.encode(texts)
            _save_npy_atomic(pid_path, pids)
            _save_npy_atomic(emb_path, emb)                 # written last: its presence marks the chunk done
            done = min((i + 1) * cfg.corpus_chunk_docs, n_docs)
            log.info("  %s | %s / %s passages | %.0f passages/s", spec.key, f"{done:,}", f"{n_docs:,}",
                     len(texts) / (time.perf_counter() - t_chunk))
    del encoder
    free_memory()                                           # drop model + CUDA cache before the big index

    with timed(f"{spec.key}: assemble FAISS index"):
        index = _new_faiss_index(spec.dim, cfg.faiss_fp16_storage)
        pid_parts = []
        for i in range(n_chunks):
            emb = np.load(chunks / f"emb_{i:05d}.npy")
            index.add(np.ascontiguousarray(emb, dtype=np.float32))
            pid_parts.append(np.load(chunks / f"pid_{i:05d}.npy"))
            del emb
        docids = np.concatenate(pid_parts)
        assert index.ntotal == len(docids) == n_docs, "chunk files do not cover the corpus"
        faiss.write_index(index, str(building / "index"))     # Pyserini layout: <dir>/index + <dir>/docid
        (building / "docid").write_text("\n".join(map(str, docids.tolist())) + "\n", encoding="utf-8")
    del index, docids, pid_parts

    write_json(building / "meta.json", {"build_seconds": time.perf_counter() - t0, "docs": n_docs,
                                        "model": spec.doc_model, "dim": spec.dim,
                                        "fp16_storage": cfg.faiss_fp16_storage})
    shutil.rmtree(chunks)
    mark_complete(building)
    shutil.rmtree(final, ignore_errors=True)
    building.rename(final)
    free_memory()


# --------------------------------------------------------------------------- retrieval
class DenseRetriever:
    def __init__(self, cfg: Config, spec: DenseSpec):
        from pyserini.search.faiss import FaissSearcher

        self.cfg, self.spec, self.name = cfg, spec, spec.key
        d = cfg.index_dir(spec.key)
        if not is_complete(d):
            raise FileNotFoundError(f"{spec.key} index missing/incomplete at {d}")
        with timed(f"{spec.key}: load FAISS index"):
            # queries are pre-encoded in big GPU batches, so Pyserini needs no query encoder of its own
            self.searcher = FaissSearcher(str(d), query_encoder=None, faiss_device=cfg.faiss_device)
        self.encoder = TextEncoder(spec, "query", cfg)
        self.build_seconds = read_json(d / "meta.json").get("build_seconds")

    def search(self, qids: np.ndarray, queries: list[str], k: int) -> Run:
        n = len(qids)
        pids, scores = empty_result(n, k)
        t0 = time.perf_counter()
        for start in range(0, n, self.cfg.search_chunk):
            stop = min(start + self.cfg.search_chunk, n)
            q_embs = self.encoder.encode(queries[start:stop]).astype(np.float32, copy=False)
            id_strs = [str(q) for q in qids[start:stop]]
            results = self.searcher.batch_search(q_embs, id_strs, k=k, threads=self.cfg.threads)
            for row, qid in enumerate(id_strs, start=start):
                hits = results.get(qid) or ()
                if hits:
                    pids[row, :len(hits)] = [int(h.docid) for h in hits]
                    scores[row, :len(hits)] = [h.score for h in hits]
            del results, q_embs
        dt = time.perf_counter() - t0
        return Run(self.name, np.asarray(qids), pids, scores,
                   {"search_seconds": dt, "queries_per_second": n / max(dt, 1e-9),
                    "index_build_seconds": self.build_seconds})
