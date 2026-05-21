#!/usr/bin/env python3
"""
generate_tables.py — Performance Analysis for MPI Document Search Engine

Reads raw_results.csv and generates summary tables:
  - Average timing by (procs, num_docs)
  - Speedup table
  - Efficiency table

Usage:
    python3 generate_tables.py [--input results/raw_results.csv]
                               [--output-dir results]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def read_results(filepath: str) -> list[dict]:
    """Read raw CSV results into a list of dicts."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "procs":        int(row["procs"]),
                "num_docs":     int(row["num_docs"]),
                "vocab_size":   int(row["vocab_size"]),
                "top_k":        int(row["top_k"]),
                "index_time":   float(row["index_time"]),
                "scatter_time": float(row["scatter_time"]),
                "search_time":  float(row["search_time"]),
                "merge_time":   float(row["merge_time"]),
                "total_time":   float(row["total_time"]),
            })
    return rows


def compute_averages(rows: list[dict]) -> dict:
    """
    Group by (procs, num_docs) and compute mean/std of each timing field.
    Returns dict[(procs, num_docs)] → {field: (mean, std)}
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["procs"], r["num_docs"])
        groups[key].append(r)

    averages = {}
    for key, group in groups.items():
        n = len(group)
        result = {}
        for field in ["index_time", "scatter_time", "search_time",
                      "merge_time", "total_time"]:
            values = [g[field] for g in group]
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
            std = variance ** 0.5
            result[field] = (mean, std)
        averages[key] = result

    return averages


def write_timing_table(averages: dict, output_dir: str):
    """Write average timing table."""
    filepath = os.path.join(output_dir, "timing_table.csv")
    with open(filepath, "w") as f:
        f.write("procs,num_docs,index_time_mean,index_time_std,"
                "scatter_time_mean,scatter_time_std,"
                "search_time_mean,search_time_std,"
                "merge_time_mean,merge_time_std,"
                "total_time_mean,total_time_std\n")

        for (procs, num_docs) in sorted(averages.keys()):
            a = averages[(procs, num_docs)]
            f.write(f"{procs},{num_docs},"
                    f"{a['index_time'][0]:.6f},{a['index_time'][1]:.6f},"
                    f"{a['scatter_time'][0]:.6f},{a['scatter_time'][1]:.6f},"
                    f"{a['search_time'][0]:.6f},{a['search_time'][1]:.6f},"
                    f"{a['merge_time'][0]:.6f},{a['merge_time'][1]:.6f},"
                    f"{a['total_time'][0]:.6f},{a['total_time'][1]:.6f}\n")

    print(f"  Timing table:     {filepath}")


def write_speedup_table(averages: dict, output_dir: str):
    """Write speedup table (S = T1 / Tp)."""
    filepath = os.path.join(output_dir, "speedup_table.csv")

    # Group by num_docs to get T1 baselines
    docs_set = sorted(set(nd for _, nd in averages.keys()))
    procs_set = sorted(set(p for p, _ in averages.keys()))

    with open(filepath, "w") as f:
        f.write("num_docs," + ",".join(f"P={p}" for p in procs_set) + "\n")

        for nd in docs_set:
            # Get T1 (single process)
            t1_key = (1, nd)
            if t1_key not in averages:
                # Use smallest proc count as baseline
                min_p = min(p for p, d in averages if d == nd)
                t1_key = (min_p, nd)

            t1 = averages[t1_key]["total_time"][0]

            speedups = []
            for p in procs_set:
                key = (p, nd)
                if key in averages:
                    tp = averages[key]["total_time"][0]
                    speedup = t1 / tp if tp > 0 else 0
                    speedups.append(f"{speedup:.4f}")
                else:
                    speedups.append("N/A")

            f.write(f"{nd}," + ",".join(speedups) + "\n")

    print(f"  Speedup table:    {filepath}")


def write_efficiency_table(averages: dict, output_dir: str):
    """Write efficiency table (E = S / P)."""
    filepath = os.path.join(output_dir, "efficiency_table.csv")

    docs_set = sorted(set(nd for _, nd in averages.keys()))
    procs_set = sorted(set(p for p, _ in averages.keys()))

    with open(filepath, "w") as f:
        f.write("num_docs," + ",".join(f"P={p}" for p in procs_set) + "\n")

        for nd in docs_set:
            t1_key = (1, nd)
            if t1_key not in averages:
                min_p = min(p for p, d in averages if d == nd)
                t1_key = (min_p, nd)

            t1 = averages[t1_key]["total_time"][0]

            efficiencies = []
            for p in procs_set:
                key = (p, nd)
                if key in averages:
                    tp = averages[key]["total_time"][0]
                    speedup = t1 / tp if tp > 0 else 0
                    efficiency = speedup / p
                    efficiencies.append(f"{efficiency:.4f}")
                else:
                    efficiencies.append("N/A")

            f.write(f"{nd}," + ",".join(efficiencies) + "\n")

    print(f"  Efficiency table: {filepath}")


def write_summary(averages: dict, output_dir: str):
    """Write a human-readable summary."""
    filepath = os.path.join(output_dir, "summary.txt")

    docs_set = sorted(set(nd for _, nd in averages.keys()))
    procs_set = sorted(set(p for p, _ in averages.keys()))

    with open(filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  MPI Document Search Engine — Performance Summary\n")
        f.write("=" * 70 + "\n\n")

        for nd in docs_set:
            f.write(f"--- Corpus Size: {nd} documents ---\n\n")
            f.write(f"{'Procs':>6}  {'Total (s)':>10}  {'Index (s)':>10}  "
                    f"{'Search (s)':>10}  {'Speedup':>8}  {'Effic.':>8}\n")
            f.write("-" * 62 + "\n")

            t1_key = (1, nd)
            if t1_key not in averages:
                min_p = min(p for p, d in averages if d == nd)
                t1_key = (min_p, nd)
            t1 = averages[t1_key]["total_time"][0]

            for p in procs_set:
                key = (p, nd)
                if key not in averages:
                    continue
                a = averages[key]
                total = a["total_time"][0]
                index = a["index_time"][0]
                search = a["search_time"][0]
                speedup = t1 / total if total > 0 else 0
                eff = speedup / p

                f.write(f"{p:>6}  {total:>10.4f}  {index:>10.4f}  "
                        f"{search:>10.4f}  {speedup:>8.4f}  {eff:>8.4f}\n")

            f.write("\n")

    print(f"  Summary:          {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance analysis tables from raw results"
    )
    parser.add_argument("--input", type=str, default="results/raw_results.csv",
                        help="Input CSV file (default: results/raw_results.csv)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading results from '{args.input}'...")
    rows = read_results(args.input)
    print(f"  {len(rows)} data points loaded.")

    averages = compute_averages(rows)

    print(f"\nGenerating tables in '{args.output_dir}/'...")
    write_timing_table(averages, args.output_dir)
    write_speedup_table(averages, args.output_dir)
    write_efficiency_table(averages, args.output_dir)
    write_summary(averages, args.output_dir)

    print(f"\nDone! All tables written to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
