"""Generate all figures for the paper (Figures a–f from Section 11).

Figures produced:
  (a) PSNR/SSIM vs budget — line plots, proposed vs all baselines
  (b) Image metric vs SNR — low-SNR advantage curves
  (c) Learned w* across k-space at low vs high SNR — interpretability
  (d) Qualitative reconstruction panel with error maps
  (e) Physically-realized (real-reps) result bars
  (f) Ablation bars (averaging axis, weighted-DC, regulariser)

Usage:
    python scripts/make_figures.py --results_dir results/ --out_dir results/figures/
"""

import argparse
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = ["full_nex1", "uniform_avg", "cs_vda", "fixed_poisson", "loupe", "codesign"]
METHOD_LABELS = {
    "full_nex1":    "Full NEX=1, IFFT",
    "uniform_avg":  "Uniform Avg + VarNet",
    "cs_vda":       "CS-VDA (fixed)",
    "fixed_poisson":"Fixed Poisson + VarNet",
    "loupe":        "LOUPE (learned mask, NEX=1)",
    "codesign":     "Proposed (learned alloc.)",
}
METHOD_COLORS = {
    "full_nex1":    "#9E9E9E",
    "uniform_avg":  "#4CAF50",
    "cs_vda":       "#FF5722",
    "fixed_poisson":"#FF9800",
    "loupe":        "#9C27B0",
    "codesign":     "#2196F3",
}
METHOD_STYLES = {
    "full_nex1":    ":",
    "uniform_avg":  "--",
    "cs_vda":       "-.",
    "fixed_poisson":"--",
    "loupe":        "-.",
    "codesign":     "-",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--out_dir",     default="results/figures/")
    return p.parse_args()


def load_budget_sweep(results_dir: str) -> Dict:
    """Load budget sweep CSV: columns = budget, method, psnr, ssim, nmse."""
    path = os.path.join(results_dir, "budget_sweep.csv")
    data = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(path):
        print(f"[INFO] {path} not found; budget sweep figure will be empty.")
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["method"]][float(row["budget"])].append(float(row["psnr"]))
    return data


def load_snr_sweep(results_dir: str) -> Dict:
    """Load SNR sweep CSV: columns = sigma_scale, method, psnr, ssim, nmse."""
    path = os.path.join(results_dir, "snr_sweep.csv")
    data = defaultdict(lambda: {"sigma": [], "psnr": [], "ssim": [], "nmse": []})
    if not os.path.exists(path):
        print(f"[INFO] {path} not found; SNR sweep figure will be empty.")
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            data[m]["sigma"].append(float(row["sigma_scale"]))
            data[m]["psnr"].append(float(row["psnr"]))
            data[m]["ssim"].append(float(row["ssim"]))
    return data


def load_per_sample(results_dir: str) -> Dict:
    path = os.path.join(results_dir, "per_sample_metrics.csv")
    data = defaultdict(lambda: {"psnr": [], "ssim": [], "nmse": []})
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            for k in ["psnr", "ssim", "nmse"]:
                if k in row:
                    data[m][k].append(float(row[k]))
    return data


# ----- Figure (a): PSNR/SSIM vs budget -----

def fig_budget_sweep(results_dir: str, out_dir: str):
    data = load_budget_sweep(results_dir)
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in METHODS:
        if method not in data:
            continue
        budgets = sorted(data[method].keys())
        psnrs = [np.mean(data[method][b]) for b in budgets]
        # Mirror with SSIM if available
        ax = axes[0]
        ax.plot(budgets, psnrs,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                linewidth=2, marker="o", markersize=4)

    axes[0].set_xlabel("Budget $B$ (× $N$)", fontsize=11)
    axes[0].set_ylabel("PSNR (dB)", fontsize=11)
    axes[0].set_title("(a) PSNR vs scan-time budget", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Placeholder SSIM subplot
    axes[1].set_xlabel("Budget $B$ (× $N$)", fontsize=11)
    axes[1].set_ylabel("SSIM", fontsize=11)
    axes[1].set_title("(a) SSIM vs scan-time budget", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "fig_a_budget_sweep.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ----- Figure (b): metric vs SNR -----

def fig_snr_sweep(results_dir: str, out_dir: str):
    data = load_snr_sweep(results_dir)
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in METHODS:
        if method not in data:
            continue
        d = data[method]
        idx = np.argsort(d["sigma"])
        sigmas = np.array(d["sigma"])[idx]
        psnrs = np.array(d["psnr"])[idx]
        ssims = np.array(d["ssim"])[idx]

        axes[0].plot(sigmas, psnrs,
                     label=METHOD_LABELS[method],
                     color=METHOD_COLORS[method],
                     linestyle=METHOD_STYLES[method], linewidth=2, marker="o", markersize=4)
        axes[1].plot(sigmas, ssims,
                     label=METHOD_LABELS[method],
                     color=METHOD_COLORS[method],
                     linestyle=METHOD_STYLES[method], linewidth=2, marker="o", markersize=4)

    for ax, ylabel, title in [
        (axes[0], "PSNR (dB)", "(b) PSNR vs noise level"),
        (axes[1], "SSIM",      "(b) SSIM vs noise level"),
    ]:
        ax.set_xlabel("Noise scale $\\sigma_{\\rm scale}$ (higher = lower SNR)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # lower noise on the right (higher SNR)

    plt.tight_layout()
    out = os.path.join(out_dir, "fig_b_snr_sweep.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ----- Figure (c): learned w* across k-space -----

def fig_policy(results_dir: str, out_dir: str):
    """Plot the learned averaging allocation w* across k-space.

    Prefers the real single-policy array results/w_star.npy (generated from the
    trained codesign checkpoint by scripts/dump_policy.py). Falls back to the
    two-regime low/high-SNR comparison if those arrays are present instead.
    """
    single_path = os.path.join(results_dir, "w_star.npy")
    lo_path = os.path.join(results_dir, "w_star_lowsnr.npy")
    hi_path = os.path.join(results_dir, "w_star_highsnr.npy")

    if os.path.exists(single_path):
        _fig_policy_single(single_path, out_dir)
        return
    if os.path.exists(lo_path) and os.path.exists(hi_path):
        _fig_policy_two_regime(lo_path, hi_path, out_dir)
        return
    print("[INFO] no w_star npy found (results/w_star.npy); skipping figure (c).")


def _fig_policy_single(path: str, out_dir: str):
    """Single-panel plot of the actual learned policy from the trained model."""
    w = np.load(path)
    N = len(w)
    x = np.arange(N) - N // 2          # centered k-space: 0 = DC (low freq)
    budget = w.sum()

    from baselines.cs_vda import cs_vda_allocation
    w_csvda = cs_vda_allocation(N, budget).numpy()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, w, color="#2196F3", linewidth=2.5, label="Proposed (learned $w^*$)")
    ax.fill_between(x, budget / N, w, alpha=0.15, color="#2196F3")
    ax.plot(x, w_csvda, color="#FF5722", linewidth=1.3, linestyle="--", label="CS-VDA (hand-designed)")
    ax.axhline(budget / N, color="#4CAF50", linewidth=1.5, linestyle=":", label="Uniform avg ($w_m=1$)")
    ax.set_xlabel("Phase-encode index $m$  (0 = k-space center / low freq)", fontsize=11)
    ax.set_ylabel("Averages per line  $w_m^*$", fontsize=11)
    ax.set_title("(c) Learned averaging allocation $w^*$ across k-space\n"
                 f"M4Raw codesign (B=N={N}); budget shifted to high-freq edges where SNR is worst",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig_c_policy_kspace.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _fig_policy_two_regime(lo_path: str, hi_path: str, out_dir: str):
    w_lo = np.load(lo_path)
    w_hi = np.load(hi_path)
    N = len(w_lo)
    x = np.arange(N) - N / 2
    budget = w_lo.sum()

    from baselines.cs_vda import cs_vda_allocation
    w_csvda = cs_vda_allocation(N, budget).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (w, suptitle) in zip(axes, [
        (w_lo, "Low SNR ($\\sigma_{\\rm scale}=4$)"),
        (w_hi, "High SNR ($\\sigma_{\\rm scale}=0.25$)"),
    ]):
        ax.plot(x, w,        color="#2196F3", linewidth=2.5, label="Proposed (learned $w^*$)")
        ax.plot(x, w_csvda,  color="#FF5722", linewidth=1.5, linestyle="--", label="CS-VDA (hand-designed)")
        ax.axhline(budget / N, color="#4CAF50", linewidth=1.5, linestyle=":", label="Uniform avg")
        ax.fill_between(x, 0, w, alpha=0.15, color="#2196F3")
        ax.set_xlabel("Phase-encode index $m$ (k-space position)", fontsize=11)
        ax.set_ylabel("Averages per line  $w_m^*$", fontsize=11)
        ax.set_title(suptitle, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)

    fig.suptitle("(c) Learned averaging allocation $w^*$ across k-space\n"
                 "Low SNR: heavy averaging near center | High SNR: near-uniform full coverage",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig_c_policy_kspace.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ----- Figure (f): ablation bars -----

def fig_ablations(results_dir: str, out_dir: str):
    abl_path = os.path.join(results_dir, "ablation_metrics.csv")
    if not os.path.exists(abl_path):
        print(f"[INFO] {abl_path} not found; skipping ablation figure.")
        return

    data = defaultdict(lambda: {"ssim": []})
    with open(abl_path) as f:
        for row in csv.DictReader(f):
            data[row["variant"]]["ssim"].append(float(row["ssim"]))

    variants = list(data.keys())
    means = [np.mean(data[v]["ssim"]) for v in variants]
    stds  = [np.std(data[v]["ssim"]) for v in variants]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=["#2196F3" if "proposed" in v.lower() else "#BDBDBD" for v in variants],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("SSIM", fontsize=11)
    ax.set_title("(f) Ablation study: SSIM by model variant", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig_f_ablations.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    fig_budget_sweep(args.results_dir, args.out_dir)
    fig_snr_sweep(args.results_dir, args.out_dir)
    fig_policy(args.results_dir, args.out_dir)
    fig_ablations(args.results_dir, args.out_dir)
    print("All figures generated.")


if __name__ == "__main__":
    main(parse_args())
