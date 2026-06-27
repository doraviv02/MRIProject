"""Baseline 3: CS-VDA — Compressed Sensing with Variable-Density Averaging.

Implements the hand-designed center-weighted averaging allocation from:
  Schoormans, Strijkers, Hansen, Nederveen, Coolen, "Compressed Sensing MRI with
  Variable-Density Averaging (CS-VDA) outperforms full sampling at low SNR."
  arXiv:1909.01672 / Phys. Med. Biol. 2020.

The allocation:
  w_m ∝ (1 + α · |m - N/2| / (N/2))^{-β}   (center-heavy polynomial decay)

with parameters α (steepness) and β (decay exponent) tuned so Σ w_m = B.

Note: the original CS-VDA both undersamples k-space AND varies the number of
averages, and reconstructs with classical compressed sensing. Here we keep only
the center-weighted *averaging allocation* and pair it with the shared unrolled
recon, so this baseline isolates the allocation profile rather than literally
reproducing Schoormans et al.
"""

import numpy as np
import torch
from typing import Tuple


def cs_vda_allocation(
    n_lines: int,
    budget: float,
    alpha: float = 2.0,
    beta: float = 2.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate the CS-VDA hand-designed variable-density averaging allocation.

    Weight of line m is proportional to a center-biased polynomial:
        u_m = (1 + alpha * |m - center| / center)^{-beta}
        w_m = B * u_m / Σ u_m

    Args:
        n_lines: N
        budget:  B (total averages budget; Σ w_m = B)
        alpha:   controls how fast averaging falls off from center
        beta:    exponent of the polynomial decay
        device:  torch device

    Returns:
        w: [N] float tensor, sums to B
    """
    center = n_lines / 2.0
    m = np.arange(n_lines, dtype=np.float32)
    dist = np.abs(m - center) / center
    u = (1.0 + alpha * dist) ** (-beta)
    u = u / u.sum() * budget
    return torch.from_numpy(u).to(device)


def poisson_disc_mask(
    n_lines: int,
    budget: float,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Variable-density Poisson-disc 1D sampling mask (for Baseline 4 as well).

    Uses a quadratic density profile that concentrates lines near the center.
    Returns a binary mask (0/1) with approximately budget lines selected.
    Lines with mask=1 get one average each (NEX=1 sampling strategy).

    Args:
        n_lines: N
        budget:  approximate number of lines to acquire (≤ N)
        device:  torch device

    Returns:
        mask: [N] int tensor with values in {0, 1}
    """
    rng = np.random.default_rng(42)
    center = n_lines // 2
    # Quadratic density: probability of selecting line m ∝ Gaussian around center
    m = np.arange(n_lines, dtype=np.float32) - center
    sigma_pd = n_lines / 4.0
    prob = np.exp(-0.5 * (m / sigma_pd) ** 2)
    prob = prob / prob.sum()

    n_select = min(int(round(budget)), n_lines)
    selected = rng.choice(n_lines, size=n_select, replace=False, p=prob)
    mask = np.zeros(n_lines, dtype=np.int32)
    mask[selected] = 1
    return torch.from_numpy(mask).to(device)


def cs_vda_rho(
    w: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute precision from CS-VDA allocation."""
    return w / (sigma ** 2 + epsilon)


def fista_tv_reconstruct(
    y_complex: "torch.Tensor",
    rho: "torch.Tensor",
    mask: "torch.Tensor",
    n_iter: int = 50,
    lambda_tv: float = 1e-3,
    step_size: float = 0.5,
) -> "torch.Tensor":
    """
    FISTA with Total Variation regularisation + precision-weighted DC.

    Minimises: Σ_m ρ_m ||(Fx)_m - ỹ_m||² + λ_TV · ||∇x||₁

    Args:
        y_complex: [H, W] complex k-space measurement
        rho:       [W] per-line precision
        mask:      [W] binary mask (acquired lines)
        n_iter:    number of FISTA iterations
        lambda_tv: TV regularisation weight
        step_size: gradient step η

    Returns:
        x_hat: [H, W] complex reconstructed image
    """
    from acq.measurement import fft2c, ifft2c

    device = y_complex.device
    H, W = y_complex.shape

    x = torch.zeros(H, W, dtype=torch.complex64, device=device)
    z = x.clone()
    t = 1.0

    rho_2d = rho.unsqueeze(0).expand(H, W)  # [H, W]

    for _ in range(n_iter):
        # Gradient of data term
        Fz = fft2c(z)
        residual = mask.unsqueeze(0) * rho_2d * (Fz - y_complex)
        grad = ifft2c(residual)

        x_new = z - step_size * grad

        # TV proximal step (anisotropic, gradient descent approximation)
        x_new = _tv_prox(x_new, lambda_tv * step_size)

        t_new = (1.0 + (1.0 + 4.0 * t ** 2) ** 0.5) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return x


def _tv_prox(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Approximate TV proximal via soft-thresholding of image gradients."""
    # Simple gradient magnitude soft-threshold (not true TV prox, but fast)
    xr = x.real
    xi = x.imag
    for _ in range(5):  # inner iterations
        dx_r = torch.roll(xr, -1, 0) - xr
        dy_r = torch.roll(xr, -1, 1) - xr
        dx_i = torch.roll(xi, -1, 0) - xi
        dy_i = torch.roll(xi, -1, 1) - xi
        mag = (dx_r ** 2 + dy_r ** 2 + dx_i ** 2 + dy_i ** 2).sqrt() + 1e-8
        shrink = torch.clamp(1.0 - lam / mag, min=0.0)
        # Divergence
        gdx_r = dx_r * shrink
        gdy_r = dy_r * shrink
        gdx_i = dx_i * shrink
        gdy_i = dy_i * shrink
        div_r = gdx_r - torch.roll(gdx_r, 1, 0) + gdy_r - torch.roll(gdy_r, 1, 1)
        div_i = gdx_i - torch.roll(gdx_i, 1, 0) + gdy_i - torch.roll(gdy_i, 1, 1)
        xr = xr + lam * div_r
        xi = xi + lam * div_i
    return torch.complex(xr, xi)
