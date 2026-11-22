"""
Make results/training_curves.png from the training logs (for the "how each model
was trained" slide). One row per model: training loss | validation macro-F1.
The best epoch (the one that was saved) is marked.

Run:  python scripts/plot_training_curves.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SERIES, SURFACE, INK, INK_2, GRID = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LOGS = [("train_history_w2v_ffn.csv", "Word2Vec + FFN (SGD)"), ("train_history_bert.csv", "BERT fine-tuning (AdamW)")]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    logs = [(config.RESULTS_DIR / f, name) for f, name in LOGS if (config.RESULTS_DIR / f).exists()]
    if not logs:
        sys.exit("No training logs found in results/ - train the models first.")
    fig, axes = plt.subplots(len(logs), 2, figsize=(10, 3.2 * len(logs)), squeeze=False, facecolor=SURFACE)
    for row, (path, name) in enumerate(logs):
        h = pd.read_csv(path)
        best = h.loc[h["val_macro_f1"].idxmax()]
        for col, (column, label) in enumerate([("train_loss", "training loss"), ("val_macro_f1", "validation macro-F1")]):
            ax = axes[row][col]
            ax.plot(h["epoch"], h[column], color=SERIES, linewidth=2, marker="o" if len(h) <= 10 else None,
                    markersize=6)
            style(ax, f"{name} - {label}", label)
            if column == "val_macro_f1":
                ax.plot([best["epoch"]], [best[column]], "o", color=SERIES, markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2)
                ax.annotate(f"best: epoch {int(best['epoch'])} ({best[column]:.3f})",
                            (best["epoch"], best[column]), textcoords="offset points", xytext=(6, -14),
                            fontsize=8, color=INK_2)
            ax.set_xticks(h["epoch"] if len(h) <= 10 else ax.get_xticks())
    fig.tight_layout()
    out = config.RESULTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
