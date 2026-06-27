"""Learned policy visualisation (E3): plot w* across k-space at low/high SNR.

Produces the signature interpretability figure:
  - Center-concentrated heavy averaging at low SNR.
  - Broader, nearer-uniform coverage at high SNR.

Usage:
    python -m eval.policy_plots \
        --checkpoint_lo checkpoints/codesign_lowsnr.pt \
        --checkpoint_hi checkpoints/codesign_highsnr.pt \
        --n_lines 128 \
        --budget 128 \
        --out results/figures/policy_w_vs_kspace.pdf
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acq.policy import AcquisitionPolicy


def load_policy(ckpt_path: str, n_lines: int, budget: float) -> AcquisitionPolicy:
    policy = AcquisitionPolicy(n_lines, budget)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("policy", ckpt)
        policy.load_state_dict(state)
    policy.eval()
    return policy


def plot_policy_comparison(
    w_low: np.ndarray,
    w_high: np.ndarray,
    w_csvda: np.ndarray,
    w_uniform: np.ndarray,
    out_path: str,
    budget: float,
):
    """Plot w* vs phase-encode line index for low/high SNR and baselines."""
    N = len(w_low)
    x = np.arange(N) - N / 2  # centred k-space coordinates

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (w, title) in zip(axes, [
        (w_low,  "Low SNR (σ·4)"),
        (w_high, "High SNR (σ·0.25)"),
    ]):
        ax.plot(x, w,       color="#2196F3", linewidth=2, label="Proposed (learned)")
        ax.plot(x, w_csvda, color="#FF5722", linewidth=1.5, linestyle="--", label="CS-VDA (fixed)")
        ax.axhline(budget / N, color="#4CAF50", linewidth=1.5, linestyle=":", label="Uniform avg")
        ax.set_xlabel("Phase-encode index (k-space position)", fontsize=11)
        ax.set_ylabel("Averages per line  $w_m$", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Learned averaging allocation $w^*$ across k-space\n"
        "at low vs high SNR — proposed vs hand-designed baselines",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Policy plot saved to {out_path}")


def plot_single_policy(
    w: np.ndarray,
    sigma: np.ndarray,
    title: str,
    out_path: str,
):
    """Plot a single learned allocation with precision overlay."""
    N = len(w)
    x = np.arange(N) - N / 2
    rho = w / (sigma ** 2 + 1e-12)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(x, w, width=0.8, color="#2196F3", alpha=0.7, label="Averages $w_m$")
    ax2.plot(x, rho, color="#FF5722", linewidth=1.5, label="Precision $\\rho_m = w_m/\\sigma_m^2$")
    ax1.set_xlabel("Phase-encode index (k-space position)", fontsize=11)
    ax1.set_ylabel("Averages per line  $w_m$", fontsize=11, color="#2196F3")
    ax2.set_ylabel("Precision $\\rho_m$", fontsize=11, color="#FF5722")
    ax1.set_title(title, fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Single policy plot saved to {out_path}")


def main(args):
    n_lines = args.n_lines
    budget = args.budget if args.budget > 0 else float(n_lines)

    policy_lo = load_policy(args.checkpoint_lo, n_lines, budget)
    policy_hi = load_policy(args.checkpoint_hi, n_lines, budget)

    with torch.no_grad():
        w_lo = policy_lo().numpy()
        w_hi = policy_hi().numpy()

    from baselines.cs_vda import cs_vda_allocation
    w_csvda = cs_vda_allocation(n_lines, budget).numpy()
    w_uniform = np.full(n_lines, budget / n_lines)

    plot_policy_comparison(w_lo, w_hi, w_csvda, w_uniform, args.out, budget)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_lo", default="checkpoints/best_codesign.pt")
    p.add_argument("--checkpoint_hi", default="checkpoints/best_codesign.pt")
    p.add_argument("--n_lines", type=int, default=128)
    p.add_argument("--budget",  type=float, default=-1)
    p.add_argument("--out", default="results/figures/policy_w_vs_kspace.pdf")
    main(p.parse_args())
