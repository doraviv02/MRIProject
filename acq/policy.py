"""Differentiable, budget-feasible averaging-allocation policy.

Implements Section 4.3 of the spec:
    w = B · softmax(a / τ)   →   w_m ≥ 0,  Σ_m w_m = B

Includes:
  - AcquisitionPolicy: learnable logits → allocation w
  - Temperature annealing
  - Entropy / sparsity regularisers
  - Integer rounding (largest-remainder method) for inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AcquisitionPolicy(nn.Module):
    """
    Learnable per-line averaging allocation under a total scan-time budget.

    Args:
        n_lines:          Number of phase-encode lines (N).
        budget:           Total averaging budget B (Σ_m w_m = B).
        temperature_init: Initial softmax temperature τ_0.
        entropy_reg:      Weight β_H for −H[softmax(a)] entropy penalty (promotes spreading).
        sparsity_reg:     Weight β_L1 for L1 on softmax(a) (promotes concentration).
        logit_init:       'zeros' | 'uniform_rand' | 'center_biased'.
        epsilon:          Small ε for numerical stability.
    """

    def __init__(
        self,
        n_lines: int,
        budget: float,
        temperature_init: float = 1.0,
        entropy_reg: float = 0.0,
        sparsity_reg: float = 0.0,
        logit_init: str = "zeros",
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.n_lines = n_lines
        self.budget = budget
        self.epsilon = epsilon
        self.entropy_reg = entropy_reg
        self.sparsity_reg = sparsity_reg

        # τ is an external scalar; store as a buffer (not a parameter)
        self.register_buffer("temperature", torch.tensor(temperature_init))

        # Learnable logits a ∈ R^N
        a_init = self._init_logits(logit_init, n_lines)
        self.logits = nn.Parameter(a_init)

    @staticmethod
    def _init_logits(mode: str, n: int) -> torch.Tensor:
        if mode == "zeros":
            return torch.zeros(n)
        elif mode == "uniform_rand":
            return 0.01 * torch.randn(n)
        elif mode == "center_biased":
            # Gaussian bias toward k-space center
            idx = torch.arange(n, dtype=torch.float32) - n / 2
            return -0.5 * (idx / (n / 8)) ** 2
        else:
            raise ValueError(f"Unknown logit_init: {mode}")

    def set_temperature(self, tau: float) -> None:
        """Update annealing temperature (called by trainer)."""
        self.temperature.fill_(tau)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            w: [N] allocation vector, non-negative, sums exactly to B.
        """
        w = self.budget * F.softmax(self.logits / self.temperature, dim=0)
        return w

    def regularization_loss(self) -> torch.Tensor:
        """Entropy/sparsity regularisation on the softmax distribution."""
        p = F.softmax(self.logits / self.temperature, dim=0)
        loss = torch.tensor(0.0, device=self.logits.device)
        if self.entropy_reg != 0.0:
            # Negative entropy (minimise → concentrate budget; positive reg → spread)
            entropy = -(p * (p + 1e-12).log()).sum()
            # Minimising negative entropy = concentrating; add positive weight to penalise concentration
            loss = loss - self.entropy_reg * entropy
        if self.sparsity_reg != 0.0:
            loss = loss + self.sparsity_reg * p.abs().sum()
        return loss

    @torch.no_grad()
    def get_integer_allocation(self, max_reps: Optional[int] = None) -> torch.Tensor:
        """
        Round w* to nonneg integers summing to round(B) via largest-remainder method.

        Args:
            max_reps: if given, cap each line at max_reps and redistribute the surplus,
                      so the exported protocol matches what realize_reps can physically
                      realise (only max_reps repetitions exist in M4Raw).

        Returns:
            w_int: [N] LongTensor
        """
        w = self.forward().cpu().numpy()
        return largest_remainder(w, max_val=max_reps)


def largest_remainder(w: "np.ndarray", max_val: Optional[int] = None) -> torch.Tensor:
    """Round a nonneg float vector to integers with the same sum via largest-remainder.

    If max_val is given, no entry exceeds max_val; surplus budget is redistributed to
    the next-highest-remainder entries still under the cap (preserves the total when
    feasible, i.e. when round(Σw) ≤ N·max_val)."""
    import numpy as np
    total = int(round(float(w.sum())))
    base = np.floor(w).astype(int)
    if max_val is not None:
        base = np.minimum(base, int(max_val))
    remainders = w - np.floor(w)
    order = np.argsort(remainders)[::-1]          # high fractional remainder first
    deficit = total - int(base.sum())
    n = len(base)
    i, guard = 0, 0
    while deficit > 0 and guard < 100 * max(n, 1):
        idx = order[i % n]
        if max_val is None or base[idx] < int(max_val):
            base[idx] += 1
            deficit -= 1
        i += 1
        guard += 1
    return torch.from_numpy(base)


def anneal_temperature(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    tau_init: float,
    tau_final: float,
    anneal_epochs: int,
) -> float:
    """Cosine annealing of temperature τ after warmup."""
    if epoch < warmup_epochs:
        return tau_init
    progress = min(epoch - warmup_epochs, anneal_epochs) / anneal_epochs
    tau = tau_final + 0.5 * (tau_init - tau_final) * (1 + math.cos(math.pi * progress))
    return max(tau, tau_final)


def compute_precision(w: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute per-line precision ρ_m = w_m / σ_m².

    Args:
        w:     [N] allocation (float, may be non-integer during training)
        sigma: [N] per-line noise std (from noise calibration)
    Returns:
        rho: [N]
    """
    return w / (sigma ** 2 + epsilon)
