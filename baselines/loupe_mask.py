"""Baseline 5: LOUPE-style learned binary sampling mask at NEX = 1.

Implements the LOUPE (Learning-Based Optimization of the Under-Sampling Pattern)
approach (Bahadir et al., IPMI 2019 / IEEE TCI 2020) adapted to our framework:

  - Learn per-line log-odds θ_m.
  - At training time: Bernoulli-relax each line with a straight-through sigmoid:
      p_m = σ(slope · (θ_m - threshold))   where threshold is found to match budget B.
  - At inference: hard binary mask (top-B lines by probability).
  - Every acquired line gets exactly NEX = 1 (no averaging axis).

Key competitor: proves that the *averaging axis* in Proposed matters beyond just
learning WHICH lines to acquire.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LOUPEPolicy(nn.Module):
    """
    LOUPE-style learned binary sampling mask.

    Args:
        n_lines:  Number of phase-encode lines N.
        budget:   Number of lines to select B (Σ mask_m = B).
        slope:    Sigmoid slope for Bernoulli relaxation.
    """

    def __init__(
        self,
        n_lines: int,
        budget: float,
        slope: float = 5.0,
    ):
        super().__init__()
        self.n_lines = n_lines
        self.budget = budget
        self.slope = slope

        # Learnable log-odds per line (initialise to uniform 50% probability)
        self.logits = nn.Parameter(torch.zeros(n_lines))

    def _threshold_for_budget(self, probs: torch.Tensor) -> float:
        """Binary search for threshold τ such that Σ σ(slope·(θ-τ)) ≈ B."""
        # selected(τ) is monotonically DECREASING in τ (higher threshold → fewer lines).
        # To match the budget: if too many are selected, raise τ; if too few, lower τ.
        lo, hi = -10.0, 10.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            selected = torch.sigmoid(self.slope * (probs - mid)).sum().item()
            if selected > self.budget:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mask_soft: [N] soft Bernoulli relaxation (for training gradient)
            prob:      [N] raw sigmoid probabilities
        """
        probs = torch.sigmoid(self.logits)
        tau = self._threshold_for_budget(probs.detach())
        mask_soft = torch.sigmoid(self.slope * (probs - tau))
        return mask_soft, probs

    @torch.no_grad()
    def get_hard_mask(self) -> torch.Tensor:
        """
        Binary mask at inference: select top-B lines by probability.

        Returns:
            mask: [N] BoolTensor
        """
        probs = torch.sigmoid(self.logits)
        n_select = int(round(self.budget))
        topk = torch.topk(probs, n_select).indices
        mask = torch.zeros(self.n_lines, dtype=torch.bool, device=self.logits.device)
        mask[topk] = True
        return mask

    def get_w_from_mask(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert mask to w vector (1 for acquired, 0 for not)."""
        if mask is None:
            mask = self.get_hard_mask()
        return mask.float()


def loupe_measurement(
    x_real: torch.Tensor,
    mask_soft: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply LOUPE soft mask to k-space (straight-through for gradient).

    Args:
        x_real:    [B, 2, H, W] image
        mask_soft: [N] soft mask in [0, 1]
        sigma:     [N] per-line noise std

    Returns:
        y_masked:  [B, 2, H, W] masked k-space
        rho:       [B, N] precision (using soft mask as effective w)
    """
    from acq.measurement import fft2c, ifft2c

    B = x_real.shape[0]
    x_complex = torch.view_as_complex(x_real.permute(0, 2, 3, 1).contiguous())
    kspace = fft2c(x_complex)  # [B, H, W]

    # Apply mask (broadcast over batch and readout)
    mask_2d = mask_soft.unsqueeze(0).unsqueeze(0)   # [1, 1, W]
    kspace_masked = kspace * mask_2d

    # Add noise at NEX=1 (w_m = mask_soft for soft gradient, 1/σ for acquired).
    # 1/sqrt(2) normalises the complex noise to E|η|^2 = 1 so the injected power is
    # σ_m^2/w_m, consistent with simulate_measurement and the σ calibration.
    noise_std = sigma / torch.sqrt(mask_soft + epsilon)
    noise_real = torch.randn_like(kspace.real)
    noise_imag = torch.randn_like(kspace.imag)
    noise = torch.complex(noise_real, noise_imag) * 0.7071067811865476
    noise_2d = noise_std.unsqueeze(0).unsqueeze(0)
    kspace_noisy = kspace_masked + noise_2d * noise * mask_2d

    y_real = torch.view_as_real(kspace_noisy).permute(0, 3, 1, 2).contiguous()

    # Precision: ρ_m = mask_soft_m / σ_m² (soft during training)
    rho = mask_soft / (sigma ** 2 + epsilon)
    rho = rho.unsqueeze(0).expand(B, -1)

    return y_real, rho
