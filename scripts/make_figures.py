"""Generate README-ready figures from curated CalibraCT metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBSET_ORDER = ["0.5%", "1%", "10%", "50%", "100%"]
SUBSET_MAP = {0.005: "0.5%", 0.01: "1%", 0.1: "10%", 0.5: "50%", 1.0: "100%"}
PALETTE = [
    "#01B8AA",
    "#374649",
    "#FD625E",
    "#F2C80F",
    "#5F6B6D",
    "#8AD4EB",
    "#FE9666",
    "#A66999",
    "#3599B8",
    "#DFBFBF",
]


def subset_label(value) -> str:
    value = float(value)
    for frac, label in SUBSET_MAP.items():
        if abs(value - frac) < 1e-9:
            return label
    return str(value)


def as_pct(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric > 1.0, numeric * 100.0)


def prepare_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = df["Name"].str.replace("Relu", "ReLU", regex=False)
    df["subset"] = df["data_subset"].apply(subset_label)
    df["accuracy"] = as_pct(df["test_accuracy"])
    df["AUC"] = as_pct(df["AUC_micro"])
    df["ECE"] = as_pct(df["ECE_l1"])
    df["params"] = pd.to_numeric(df["trainable_params"], errors="coerce")
    return df


def heatmap(df: pd.DataFrame, value_col: str, title: str, label: str, out_path: Path, low_is_good: bool = False) -> None:
    pivot = df.pivot_table(index="method", columns="subset", values=value_col, aggfunc="mean")
    pivot = pivot[[c for c in SUBSET_ORDER if c in pivot.columns]]
    pivot = pivot.loc[sorted(pivot.index, key=lambda idx: -np.nanmean(pivot.loc[idx].values))]

    cmap_colors = ["#63BE7B", "#FFEB84", "#F8696B"] if low_is_good else ["#F8696B", "#FFEB84", "#63BE7B"]
    cmap = mcolors.LinearSegmentedColormap.from_list("calibract_heatmap", cmap_colors)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_ylabel("Method")
    ax.set_title(title, pad=24)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.1f}%", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pareto(df: pd.DataFrame, subset: str, title: str, out_path: Path) -> None:
    data = df[df["subset"] == subset].dropna(subset=["AUC", "ECE", "params"])
    if data.empty:
        return

    param_values = data["params"].to_numpy()
    sizes = 80 + 520 * (param_values - param_values.min()) / max(1.0, param_values.max() - param_values.min())

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, (_, row) in enumerate(data.iterrows()):
        ax.scatter(row["AUC"], row["ECE"], s=sizes[i], color=PALETTE[i % len(PALETTE)], alpha=0.88, label=row["method"])

    ax.set_xlabel("AUC (%)")
    ax.set_ylabel("ECE (%) - lower is better")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def binary_heatmap(path: Path, out_path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path).rename(columns={"Method": "method"})
    long = df.melt(id_vars="method", value_vars=[c for c in SUBSET_ORDER if c in df.columns], var_name="subset", value_name="binary_accuracy")
    heatmap(long, "binary_accuracy", "Class-Agnostic Tumor Accuracy", "Binary accuracy (%)", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/final_metrics.csv"))
    parser.add_argument("--binary-metrics", type=Path, default=Path("results/binary_accuracy.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = prepare_metrics(args.metrics)

    heatmap(metrics, "accuracy", "Test Accuracy by Method vs Dataset Subset", "Accuracy (%)", args.output_dir / "fig3_accuracy.png")
    heatmap(metrics, "ECE", "Expected Calibration Error by Method vs Dataset Subset", "ECE (%)", args.output_dir / "fig4_ece.png", low_is_good=True)
    pareto(metrics, "0.5%", "AUC vs ECE at 0.5% Training Subset", args.output_dir / "fig5_pareto_lowdata.png")
    pareto(metrics, "100%", "AUC vs ECE at 100% Training Subset", args.output_dir / "fig6_pareto_fulldata.png")
    binary_heatmap(args.binary_metrics, args.output_dir / "fig7_binary_accuracy.png")


if __name__ == "__main__":
    main()
