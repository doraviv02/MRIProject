"""Baseline 1: Full coverage, NEX = 1, zero-filled IFFT reconstruction.

w_m = 1 ∀m, B = N. The conventional reference: all lines sampled once,
simple IFFT with no deep learning.
"""

import torch
import numpy as np
from acq.measurement import ifft2c


def full_nex1_reconstruct(kspace_real: torch.Tensor) -> torch.Tensor:
    """
    Zero-filled IFFT reconstruction from fully-sampled NEX=1 k-space.

    Args:
        kspace_real: [B, 2, H, W] — real/imag channels of k-space

    Returns:
        x_hat: [B, 2, H, W] — real/imag image
    """
    k_complex = torch.view_as_complex(kspace_real.permute(0, 2, 3, 1).contiguous())
    img = ifft2c(k_complex)
    return torch.view_as_real(img).permute(0, 3, 1, 2).contiguous()


def get_uniform_w(n_lines: int, budget: float = None, device: str = "cpu") -> torch.Tensor:
    """
    Return w_m = 1 ∀m (full coverage NEX=1) or uniform budget allocation.

    Args:
        n_lines: N
        budget:  if None, returns all-ones; else returns B/N per line

    Returns:
        w: [N] float tensor
    """
    if budget is None:
        return torch.ones(n_lines, device=device)
    return torch.full((n_lines,), budget / n_lines, device=device)
