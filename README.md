# MS MARCO passage retrieval: TF-IDF, BM25, DPR, Sentence-Transformer, RRF

## Setup
```bash
# Java 21 must be on PATH (Pyserini/Anserini requirement):  java -version
uv python install 3.12            # current Pyserini (2.x) requires Python 3.12
uv sync                           # add `--extra ui` for the Gradio search page
mkdir -p data/msmarco             # put collection.tsv, queries.train.tsv, qrels.train.tsv here
```

## Run
```bash
uv run python run.py                              # all systems, 10,000 sampled eval queries
uv run python run.py --systems tfidf bm25         # subset of systems
uv run python run.py --systems rrf                # re-fuse saved runs (no search)
uv run python run.py --eval-queries 0             # every judged query (~500k)
uv run python run.py --corpus-size 1000000        # smaller corpus for a quick dry run
uv run python run.py --retrieve-k 100             # deeper lists for RRF; metrics still use top-10
uv run --extra ui python app.py                   # search interface
```
Re-running skips everything already on disk (prepared data, indexes, runs); `--rebuild-indexes` / `--rerun-search` force.

## Layout of `artifacts/`
```
prepared/   corpus.parquet (cleaned, de-duplicated), eval_queries.parquet, eval_qrels.parquet, prepare_meta.json (dup/consistency stats)
indexes/    tfidf/  bm25/  dpr/  st/        (each has a .complete marker; partial builds are never loaded)
runs/       <system>.npz + .meta.json       (top-k pids/scores per query, written right after search)
eval/       <system>.json, summary.json, comparison_table.{md,csv}
figures/    precision_recall.png
logs/run.log
```
