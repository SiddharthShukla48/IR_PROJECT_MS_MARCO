"""Performance comparison table (md + csv) and the precision-recall graph."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import SYSTEM_LABELS  # noqa: E402


def label_of(name: str) -> str:
    if name in SYSTEM_LABELS:
        return SYSTEM_LABELS[name]
    if name.startswith("rrf_"):
        return "Hybrid RRF: " + " + ".join(part.upper() for part in name.removeprefix("rrf_").split("_"))
    return name


def comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"system": label_of(r["system"])}
        row.update(r["metrics"])
        row["queries/s"] = r["timing"].get("queries_per_second")
        rows.append(row)
    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [str(row[c]) if c == "system" else ("" if pd.isna(row[c]) else f"{row[c]:.4f}" if c != "queries/s"
                 else f"{row[c]:.1f}") for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def write_table(results: list[dict], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = comparison_table(results)
    df.to_csv(out_dir / "comparison_table.csv", index=False, float_format="%.6f")
    (out_dir / "comparison_table.md").write_text(_to_markdown(df), encoding="utf-8")
    return df


def plot_pr_curves(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        pr = r["pr_curve"]
        ax.plot(pr["recall"], pr["precision"], marker="o", markersize=3.5, linewidth=1.6, label=label_of(r["system"]))
    ax.set_xlabel(f"Recall@rank (macro-avg over {results[0]['num_queries']:,} queries)")
    ax.set_ylabel("Precision@rank")
    ax.set_title(f"Precision-Recall curve, ranks 1..{results[0]['k']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
