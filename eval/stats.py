"""Statistical significance tests (paired Wilcoxon signed-rank) for E1.

Compares proposed method vs each baseline on per-volume SSIM / NMSE.

Usage:
    python -m eval.stats \
        --results_dir results/ \
        --metric ssim
"""

import argparse
import csv
import os
import numpy as np
from scipy import stats
from typing import Dict, List


def load_per_sample_metrics(csv_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load a CSV with columns: method, sample_id, psnr, ssim, nmse.

    Returns: {method: {metric: [values]}}
    """
    data: Dict[str, Dict[str, List[float]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["method"]
            if m not in data:
                data[m] = {"psnr": [], "ssim": [], "nmse": []}
            for k in ["psnr", "ssim", "nmse"]:
                data[m][k].append(float(row[k]))
    return data


def wilcoxon_vs_all(
    data: Dict[str, Dict[str, List[float]]],
    proposed: str,
    baselines: List[str],
    metric: str = "ssim",
    alpha: float = 0.05,
) -> None:
    """Run paired Wilcoxon test: proposed vs each baseline, print table."""
    if proposed not in data:
        print(f"[ERROR] Proposed method '{proposed}' not found in results.")
        return

    y_proposed = np.array(data[proposed][metric])
    print(f"\nPaired Wilcoxon signed-rank test: {proposed} vs baselines on {metric.upper()}")
    print(f"{'Baseline':20s}  {'p-value':>10s}  {'sig?':>6s}  {'mean Δ':>10s}")
    print("-" * 55)

    for bl in baselines:
        if bl not in data:
            print(f"{bl:20s}  {'N/A':>10s}  {'N/A':>6s}  {'N/A':>10s}")
            continue
        y_bl = np.array(data[bl][metric])
        n = min(len(y_proposed), len(y_bl))
        result = stats.wilcoxon(y_proposed[:n], y_bl[:n], alternative="greater")
        sig = "YES" if result.pvalue < alpha else "NO"
        delta = float(y_proposed[:n].mean() - y_bl[:n].mean())
        print(f"{bl:20s}  {result.pvalue:10.4e}  {sig:>6s}  {delta:+10.4f}")


def aggregate_table(
    data: Dict[str, Dict[str, List[float]]],
    metrics: List[str] = None,
) -> None:
    """Print mean ± std table for all methods × metrics."""
    if metrics is None:
        metrics = ["psnr", "ssim", "nmse"]
    print(f"\n{'Method':20s}", end="")
    for m in metrics:
        print(f"  {m.upper():>14s}", end="")
    print()
    print("-" * (20 + 16 * len(metrics)))
    for method, mdict in sorted(data.items()):
        print(f"{method:20s}", end="")
        for metric in metrics:
            vals = np.array(mdict[metric])
            print(f"  {vals.mean():6.3f}±{vals.std():.3f}", end="")
        print()


def main(args):
    csv_path = os.path.join(args.results_dir, "per_sample_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Per-sample metrics file not found: {csv_path}")
        print("Run make_tables.py first to generate per-sample metrics.")
        return

    data = load_per_sample_metrics(csv_path)
    aggregate_table(data)
    wilcoxon_vs_all(
        data,
        proposed="codesign",
        baselines=["full_nex1", "uniform_avg", "cs_vda", "fixed_poisson", "loupe"],
        metric=args.metric,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--metric", default="ssim", choices=["ssim", "nmse", "psnr"])
    p.add_argument("--alpha", type=float, default=0.05)
    main(p.parse_args())
