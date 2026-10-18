"""Search interface.  uv run --extra ui python app.py   (after run.py has built the indexes)"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from msmarco_ir.config import Config
from msmarco_ir.utils import configure_jvm, setup_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, default=Config().artifacts_dir)
    p.add_argument("--device", default="auto")
    p.add_argument("--faiss-device", default="cpu")
    p.add_argument("--jvm-heap-gb", type=int, default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    a = p.parse_args()

    cfg = replace(Config(), artifacts_dir=a.artifacts_dir, device=a.device, faiss_device=a.faiss_device,
                  jvm_heap_gb=a.jvm_heap_gb)
    setup_logging()
    configure_jvm(cfg.jvm_heap_gb)

    import gradio as gr
    from msmarco_ir.service import SearchService

    service = SearchService(cfg)
    systems = service.available_systems()
    if "bm25" not in systems:
        raise SystemExit("Build the BM25 index first (python run.py --systems bm25): it also stores passage text.")
    label_to_key = {label: key for key, label in systems.items()}

    def search(query: str, system_label: str, k: int):
        query = (query or "").strip()
        if not query:
            return []
        hits = service.search(label_to_key[system_label], query, int(k))
        return [[h["rank"], h["pid"], round(h["score"], 4), h["text"]] for h in hits]

    with gr.Blocks(title="MS MARCO passage search") as demo:
        gr.Markdown("# MS MARCO passage search\nTF-IDF, BM25, DPR, Sentence-Transformer and RRF hybrids over the indexed corpus.")
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="what is the capital of france", scale=4)
            system = gr.Dropdown(list(label_to_key), value=list(label_to_key)[1] if len(label_to_key) > 1 else None,
                                 label="Retrieval system", scale=2)
            k = gr.Slider(1, 10, value=10, step=1, label="Top-k", scale=1)
        results = gr.Dataframe(headers=["rank", "passage id", "score", "passage"], wrap=True, interactive=False)
        go = gr.Button("Search", variant="primary")
        go.click(search, [query, system, k], results)
        query.submit(search, [query, system, k], results)
    demo.launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
