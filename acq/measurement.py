"""Differentiable acquisition measurement model (Section 4.2).

Implements:
    ỹ_m = (F x)_m  +  (σ_m / sqrt(w_m + ε)) · η_m,    η_m ~ CN(0, I)

This single equation unifies:
  - Sampling: w_m → 0 means the line is not acquired (noise → ∞)
  - Averaging: w_m > 1 means the line is averaged multiple times (noise ↓)

The gradient flows through w (via the 1/sqrt(w_m + ε) term) into the policy logits.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

SQRT1_2 = 0.7071067811865476  # 1/sqrt(2): normalises complex CN(0,1) noise to E|z|^2 = 1


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT (image-centered ↔ k-centered). x: [..., H, W] complex.

    Full centered pair: ifftshift on input, fftshift on output — matches the data
    loader (data/m4raw.py::_fft2c) and noise calibration so that image and k-space
    are in the same centered frame (no half-pixel/half-image roll)."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT, inverse of fft2c. k: [..., H, W] complex."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """[..., 2, H, W] → [..., H, W] complex."""
    return torch.view_as_complex(x.permute(*range(x.ndim - 3), -2, -1, -3).contiguous())


def complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """[..., H, W] complex → [..., 2, H, W] real."""
    r = torch.view_as_real(z)  # [..., H, W, 2]
    return r.permute(*range(z.ndim - 2), -1, -3, -2).contiguous()


def simulate_measurement(
    x_real: torch.Tensor,
    w: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the differentiable measurement model to a batch of images.

    Args:
        x_real: [B, 2, H, W] image (real and imaginary channels)
        w:      [N] or [B, N] per-line averaging weights (N = number of phase-encode lines = W)
        sigma:  [N] or [B, N] per-line noise std
        epsilon: numerical stability constant

    Returns:
        y_tilde: [B, 2, H, W] measured k-space (real + imag)
        rho:     [B, N] per-line precision ρ_m = w_m / σ_m²
    """
    B, _, H, W = x_real.shape

    # Convert to complex and go to k-space
    x_complex = torch.view_as_complex(x_real.permute(0, 2, 3, 1).contiguous())  # [B, H, W]
    kspace = fft2c(x_complex)  # [B, H, W] complex

    # Expand w and sigma to batch if needed
    if w.ndim == 1:
        w = w.unsqueeze(0).expand(B, -1)     # [B, W]
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(0).expand(B, -1)  # [B, W]

    # Effective noise std per line: σ_m / sqrt(w_m + ε)
    noise_std = sigma / torch.sqrt(w + epsilon)  # [B, W]

    # Draw unit-variance complex Gaussian noise η ~ CN(0, 1) for each readout-phase
    # location. The 1/sqrt(2) makes E|η|^2 = 1 (variance split evenly over re/imag),
    # so the injected per-line noise power is exactly σ_m^2/w_m — matching the
    # calibration (σ = total inter-rep complex std) and making ρ = w/σ^2 the true
    # inverse-variance. Without it the simulated noise power was 2σ^2/w (√2 too high).
    noise_real = torch.randn_like(kspace.real)     # [B, H, W]
    noise_imag = torch.randn_like(kspace.imag)
    noise = torch.complex(noise_real, noise_imag) * SQRT1_2  # [B, H, W], E|noise|^2 = 1

    # Broadcast noise std over readout dimension: [B, 1, W] × [B, H, W]
    noise_std_2d = noise_std.unsqueeze(1)          # [B, 1, W]
    y_tilde = kspace + noise_std_2d * noise        # [B, H, W] complex

    # Per-line precision
    rho = w / (sigma ** 2 + epsilon)              # [B, W]

    # Convert back to real [B, 2, H, W]
    y_real = torch.view_as_real(y_tilde).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]
    return y_real, rho


def apply_measurement_mask(
    kspace: torch.Tensor,
    w_int: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply integer allocation mask: lines with w=0 are zeroed (unacquired).
    Lines with w≥1 are averaged w times using actual noise.

    Args:
        kspace: [B, 2, H, W] clean k-space
        w_int:  [N] integer allocation (LongTensor)
        sigma:  [N] per-line noise std

    Returns:
        y_masked: [B, 2, H, W] masked+averaged k-space
        rho:      [B, N] precision
    """
    w_float = w_int.float()
    return simulate_measurement(kspace, w_float, sigma, epsilon)


def dc_gradient_step(
    x_complex: torch.Tensor,
    y_tilde: torch.Tensor,
    rho: torch.Tensor,
    step_size: float = 0.1,
) -> torch.Tensor:
    """
    Precision-weighted data-consistency gradient step.

    Minimises: Σ_m ρ_m ||(Fx)_m - ỹ_m||²
    Gradient w.r.t. x: F^H [ ρ ⊙ (Fx - ỹ) ]

    Args:
        x_complex: [B, H, W] complex current estimate
        y_tilde:   [B, H, W] complex measurement (in k-space)
        rho:       [B, W] per-line precision
        step_size: scalar η

    Returns:
        x_updated: [B, H, W] complex
    """
    Fx = fft2c(x_complex)
    residual = Fx - y_tilde
    # Weight residual: ρ [B, W] broadcasts over readout dim H → [B, H, W]
    weighted_res = rho.unsqueeze(1) * residual
    grad = ifft2c(weighted_res)  # single call; result is complex [B, H, W]
    return x_complex - step_size * grad
