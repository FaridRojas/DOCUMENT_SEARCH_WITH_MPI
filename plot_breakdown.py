#!/usr/bin/env python3
"""
Plot time breakdown (index/scatter/search/merge) as stacked bars.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


TimingRow = Dict[str, float]


def read_timing_table(path: Path) -> Dict[int, List[TimingRow]]:
    data: Dict[int, List[TimingRow]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_docs = int(row["num_docs"])
            data.setdefault(num_docs, []).append({
                "procs": int(row["procs"]),
                "index_time_mean": float(row["index_time_mean"]),
                "scatter_time_mean": float(row["scatter_time_mean"]),
                "search_time_mean": float(row["search_time_mean"]),
                "merge_time_mean": float(row["merge_time_mean"]),
            })

    for nd in data:
        data[nd].sort(key=lambda r: r["procs"])
    return data


def plot_breakdown(timing: Dict[int, List[TimingRow]], out_dir: Path) -> None:
    num_docs_list = sorted(timing.keys())
    ncols = 2
    nrows = math.ceil(len(num_docs_list) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 7), sharex=False)
    axes = axes.flatten() if len(num_docs_list) > 1 else [axes]

    phases = [
        ("Index", "index_time_mean", "#1f77b4"),
        ("Scatter", "scatter_time_mean", "#ff7f0e"),
        ("Search", "search_time_mean", "#2ca02c"),
        ("Merge", "merge_time_mean", "#d62728"),
    ]

    for ax, num_docs in zip(axes, num_docs_list):
        rows = timing[num_docs]
        procs = [r["procs"] for r in rows]
        bottom = [0.0] * len(procs)
        for label, key, color in phases:
            vals = [r[key] for r in rows]
            ax.bar(procs, vals, bottom=bottom, label=label, color=color)
            bottom = [b + v for b, v in zip(bottom, vals)]

        ax.set_title(f"{num_docs} documentos")
        ax.set_xlabel("Procesos (P)")
        ax.set_ylabel("Tiempo [s]")
        ax.set_xticks(procs)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(num_docs_list):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Desglose de tiempos por fase")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "time_breakdown_stacked.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stacked time breakdown plots."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory with timing_table.csv")
    parser.add_argument("--output-dir", default="figuras",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timing = read_timing_table(results_dir / "timing_table.csv")
    plot_breakdown(timing, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
