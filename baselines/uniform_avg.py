"""Baseline 2: Uniform averaging + precision-weighted DC reconstruction.

All lines receive the same number of averages: w_m = B/N ∀m.
Uses the same unrolled reconstruction backbone as the proposed method.
This isolates the benefit of *non-uniform* allocation (Proposed) over
simply averaging everything equally.
"""

import torch
from typing import Optional


def get_uniform_w(n_lines: int, budget: float, device: str = "cpu") -> torch.Tensor:
    """
    Return uniform averaging allocation w_m = B/N.

    Args:
        n_lines: N (number of phase-encode lines)
        budget:  B (total averaging budget)
        device:  torch device

    Returns:
        w: [N] float tensor, all equal to B/N
    """
    return torch.full((n_lines,), budget / n_lines, dtype=torch.float32, device=device)


def get_uniform_rho(
    n_lines: int,
    budget: float,
    sigma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute precision for uniform averaging.

    ρ_m = (B/N) / σ_m²

    Args:
        n_lines: N
        budget:  B
        sigma:   [N] per-line noise std
        epsilon: stability

    Returns:
        rho: [N]
    """
    w = get_uniform_w(n_lines, budget, device=sigma.device)
    return w / (sigma ** 2 + epsilon)
