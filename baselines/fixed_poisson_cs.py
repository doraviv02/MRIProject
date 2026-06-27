"""Baseline 4: Fixed variable-density Poisson-disc undersampling + deep recon.

NEX = 1 on a fixed Poisson-disc mask, combined with the same unrolled network
as the proposed method. This simulates the "Accelerating Low-field MRI" strategy
(arXiv:2411.06704): mild undersampling + deep learning reconstruction.

The key difference from Proposed: the mask is FIXED (not learned) and every
acquired line is averaged exactly once (no averaging allocation axis).
"""

import numpy as np
import torch
from typing import Optional
from baselines.cs_vda import poisson_disc_mask


def get_poisson_allocation(
    n_lines: int,
    budget: float,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Binary {0,1} allocation from a Poisson-disc mask scaled to budget.

    Lines with value 1 are acquired once; lines with value 0 are dropped.
    Total acquisitions ≈ budget (may differ slightly due to rounding).

    Returns:
        w: [N] float tensor with values in {0.0, 1.0}
    """
    mask = poisson_disc_mask(n_lines, budget, device=device)
    return mask.float()


def apply_fixed_mask(
    kspace_real: torch.Tensor,
    mask: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple:
    """
    Apply a fixed binary mask to k-space and compute precision map.

    Args:
        kspace_real: [B, 2, H, W] k-space
        mask:        [N] binary (0/1) — N = W (phase-encode lines)
        sigma:       [N] per-line noise std
        epsilon:     stability constant

    Returns:
        (y_masked, rho) where y_masked: [B, 2, H, W], rho: [B, W]
    """
    B, C, H, W = kspace_real.shape
    mask_2d = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    y_masked = kspace_real * mask_2d

    # Precision: ρ_m = w_m / σ_m²; w_m ∈ {0, 1} here
    rho = mask.float() / (sigma ** 2 + epsilon)  # [W]
    rho = rho.unsqueeze(0).expand(B, -1)          # [B, W]

    return y_masked, rho
