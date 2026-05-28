#!/usr/bin/env python3
"""
Plot scaling metrics (total time, speedup, efficiency) from results CSVs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

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
                "index_time_std": float(row["index_time_std"]),
                "scatter_time_mean": float(row["scatter_time_mean"]),
                "scatter_time_std": float(row["scatter_time_std"]),
                "search_time_mean": float(row["search_time_mean"]),
                "search_time_std": float(row["search_time_std"]),
                "merge_time_mean": float(row["merge_time_mean"]),
                "merge_time_std": float(row["merge_time_std"]),
                "total_time_mean": float(row["total_time_mean"]),
                "total_time_std": float(row["total_time_std"]),
            })

    for nd in data:
        data[nd].sort(key=lambda r: r["procs"])
    return data


def read_matrix_table(path: Path) -> Tuple[List[int], Dict[int, List[float | None]]]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        procs = [int(h.split("=")[1]) for h in header[1:]]
        values: Dict[int, List[float | None]] = {}
        for row in reader:
            num_docs = int(row[0])
            vals: List[float | None] = []
            for v in row[1:]:
                vals.append(None if v == "N/A" else float(v))
            values[num_docs] = vals
    return procs, values


def plot_total_time(timing: Dict[int, List[TimingRow]], out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for num_docs, rows in sorted(timing.items()):
        procs = [r["procs"] for r in rows]
        total = [r["total_time_mean"] for r in rows]
        std = [r["total_time_std"] for r in rows]
        plt.errorbar(procs, total, yerr=std, marker="o", capsize=3, label=f"{num_docs} docs")

    plt.title("Tiempo total promedio vs procesos")
    plt.xlabel("Procesos (P)")
    plt.ylabel("Tiempo total [s]")
    plt.xticks(sorted({r["procs"] for rows in timing.values() for r in rows}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "total_time_vs_procs.png", dpi=200)
    plt.close()


def plot_speedup(speedup_path: Path, out_dir: Path) -> None:
    procs, speedups = read_matrix_table(speedup_path)

    plt.figure(figsize=(8, 5))
    plt.plot(procs, procs, "k--", label="Ideal")
    for num_docs, vals in sorted(speedups.items()):
        xs, ys = [], []
        for p, v in zip(procs, vals):
            if v is None:
                continue
            xs.append(p)
            ys.append(v)
        plt.plot(xs, ys, marker="o", label=f"{num_docs} docs")

    plt.title("Speedup vs procesos")
    plt.xlabel("Procesos (P)")
    plt.ylabel("Speedup (T1 / Tp)")
    plt.xticks(procs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_procs.png", dpi=200)
    plt.close()


def plot_efficiency(eff_path: Path, out_dir: Path) -> None:
    procs, efficiencies = read_matrix_table(eff_path)

    plt.figure(figsize=(8, 5))
    plt.axhline(1.0, color="k", linestyle="--", label="Ideal")
    for num_docs, vals in sorted(efficiencies.items()):
        xs, ys = [], []
        for p, v in zip(procs, vals):
            if v is None:
                continue
            xs.append(p)
            ys.append(v)
        plt.plot(xs, ys, marker="o", label=f"{num_docs} docs")

    plt.title("Eficiencia vs procesos")
    plt.xlabel("Procesos (P)")
    plt.ylabel("Eficiencia (Speedup / P)")
    plt.xticks(procs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_vs_procs.png", dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate scaling plots from results tables."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory with timing/speedup/efficiency CSVs")
    parser.add_argument("--output-dir", default="figuras",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timing = read_timing_table(results_dir / "timing_table.csv")
    plot_total_time(timing, output_dir)
    plot_speedup(results_dir / "speedup_table.csv", output_dir)
    plot_efficiency(results_dir / "efficiency_table.csv", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
