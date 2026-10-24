"""Generate a publication-friendly static evaluation figure from aggregate CSVs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(source, output):
    source, output = Path(source), Path(output)
    risk = pd.read_csv(source / "risk_summary_weekly.csv")
    performance = pd.read_csv(source / "model_performance_weekly.csv")
    quality = pd.read_csv(source / "data_quality_weekly.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.patch.set_facecolor("#f4f7fa")
    colors = {"full": "#087f8c", "internal": "#e19a38"}
    for mode, part in performance.groupby("mode"):
        axes[0, 0].plot(part.WEEK_NUM, part.gini, marker="o", label=mode, color=colors[mode])
        axes[0, 1].plot(part.WEEK_NUM, part.brier, marker="o", label=mode, color=colors[mode])
    full = risk[risk["mode"] == "full"]
    axes[1, 0].bar(full.WEEK_NUM, full.applications, color=colors["full"])
    q = quality.groupby("WEEK_NUM").psi.max()
    axes[1, 1].plot(q.index, q.values, marker="o", color="#634da0")
    axes[1, 1].axhline(0.2, color="#bd5245", linestyle="--", label="Demo alert threshold")
    titles = [
        "Weekly discrimination · Gini",
        "Probability error · Brier score",
        "Applications in final holdout",
        "Largest numeric-feature drift · PSI",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontweight="bold", color="#183a4c")
        ax.set_xlabel("Competition week")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.suptitle(
        "HOME CREDIT  /  Risk & Model Stability\nReal-data local holdout · Curated features · No leaderboard claim",
        fontsize=17,
        fontweight="bold",
        color="#183a4c",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    plot(args.source, args.output)
