"""Precision-weighted unrolled reconstruction network (Section 4.4).

Implements K unrolled iterations of:
    x_{k+1} = Denoiser_θ_k( x_k - η_k · ∇_x [Σ_m ρ_m ||(Fx)_m - ỹ_m||²] )

The denoiser is conditioned on the precision map ρ (concatenated as a spatial channel)
so it knows where in k-space the data is reliable.

References:
  - E2E-VarNet (Sriram et al., MICCAI 2020) for the unrolled structure.
  - Precision-weighted DC follows the CS-VDA noise-optimal weighted-ℓ2 formulation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from acq.measurement import fft2c, ifft2c
from models.denoiser import build_denoiser


class DCLayer(nn.Module):
    """
    Precision-weighted data-consistency gradient step.

    x ← x - η · F^H [ ρ ⊙ (Fx - ỹ) ]
    """

    def __init__(self, step_size: float = 0.1, learnable: bool = True):
        super().__init__()
        if learnable:
            self.eta = nn.Parameter(torch.tensor(step_size))
        else:
            self.register_buffer("eta", torch.tensor(step_size))

    def forward(
        self,
        x: torch.Tensor,
        y_tilde: torch.Tensor,
        rho: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, H, W] complex current estimate
            y_tilde: [B, H, W] complex measured k-space
            rho:     [B, W] per-line precision
            mask:    [B, W] optional binary mask (1 = acquired line)

        Returns:
            x_updated: [B, H, W] complex
        """
        Fx = fft2c(x)                             # [B, H, W] complex
        residual = Fx - y_tilde                    # [B, H, W]
        rho_2d = rho.unsqueeze(1)                 # [B, 1, W]

        if mask is not None:
            rho_2d = rho_2d * mask.unsqueeze(1)

        # Scale-robust data consistency: normalise the precision to mean 1 per sample so
        # the gradient-step magnitude depends only on the RELATIVE per-line precision, not
        # the absolute noise level. Without this, ρ = w/σ² makes the step scale ∝ 1/σ², so
        # the learnable η can't serve a wide SNR range (training diverges/collapses under
        # noise augmentation, and eval needs σ-clamps). The relative weighting — which
        # encodes the averaging policy — is preserved; absolute noise is conveyed implicitly
        # by the residual/input and the (normalised) precision-conditioning channel.
        rho_2d = rho_2d / (rho_2d.mean(dim=(1, 2), keepdim=True) + 1e-8)

        weighted = rho_2d * residual               # [B, H, W]
        grad = ifft2c(weighted)                    # [B, H, W] complex
        return x - self.eta * grad


class UnrolledRecon(nn.Module):
    """
    K-step unrolled reconstruction network conditioned on per-line precision.

    Each step:
      1. DC gradient step (precision-weighted)
      2. CNN denoiser conditioned on precision map ρ

    Args:
        cfg: recon config dict (from recon.yaml)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.K = cfg.get("num_unrolled_steps", 6)
        self.precision_conditioning = cfg.get("precision_conditioning", True)
        learn_dc = cfg.get("learn_dc_step", True)
        step_size = cfg.get("dc_step_size", 0.1)

        self.dc_layers = nn.ModuleList([
            DCLayer(step_size, learnable=learn_dc)
            for _ in range(self.K)
        ])
        # Build K shared-or-unshared denoisers
        # Unshared by default for maximum expressivity
        self.denoisers = nn.ModuleList([
            build_denoiser(cfg) for _ in range(self.K)
        ])

    def forward(
        self,
        y_tilde: torch.Tensor,
        rho: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_tilde: [B, 2, H, W] measured k-space (real/imag channels)
            rho:     [B, W] per-line precision
            mask:    [B, W] optional binary mask (1 = acquired)

        Returns:
            x_hat: [B, 2, H, W] reconstructed image (real/imag)
        """
        B, _, H, W = y_tilde.shape

        # Convert measured k-space to complex
        y_complex = torch.view_as_complex(y_tilde.permute(0, 2, 3, 1).contiguous())  # [B, H, W]

        # Zero-fill unacquired lines (ρ_m == 0) before the IFFT init so their
        # (effectively infinite-variance) simulated noise does not leak into x_0.
        # No-op when every line is acquired, e.g. budget B = N where all ρ_m > 0;
        # the DC step already ignores these lines via the ρ weighting.
        acq = (rho > 0).to(y_complex.real.dtype)            # [B, W]
        y_complex = y_complex * acq.unsqueeze(1)            # broadcast over readout H

        # Initialise x with zero-filled IFFT
        x = ifft2c(y_complex)  # [B, H, W] complex

        # Precision-conditioning map. Per-sample max-normalise ρ to [0, 1] before
        # feeding the denoiser: raw ρ = w/σ² is ~10³–10⁴ while the image channels are
        # ~[0,1], which otherwise lets the precision channel dominate the first conv.
        # Normalisation preserves the *relative* reliability pattern (what encodes the
        # policy) while matching the image scale. The DC step still uses the raw ρ.
        rho_norm = rho / (rho.amax(dim=1, keepdim=True) + 1e-8)        # [B, W] in [0,1]
        rho_spatial = rho_norm.unsqueeze(1).unsqueeze(2).repeat(1, 1, H, 1)  # [B, 1, H, W]

        for dc, denoiser in zip(self.dc_layers, self.denoisers):
            # Data-consistency step
            x = dc(x, y_complex, rho, mask)

            # Convert to real for denoiser: [B, 2, H, W]
            x_real = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()

            # Condition denoiser on precision map
            if self.precision_conditioning:
                x_in = torch.cat([x_real, rho_spatial], dim=1)  # [B, 3, H, W]
            else:
                x_in = x_real  # [B, 2, H, W]

            x_real = denoiser(x_in)  # [B, 2, H, W]
            x = torch.view_as_complex(x_real.permute(0, 2, 3, 1).contiguous())

        # Return as [B, 2, H, W]
        return torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()

    def reconstruct_from_kspace(
        self,
        kspace_real: torch.Tensor,
        w: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Convenience method: reconstruct from real-valued k-space + allocation.

        Args:
            kspace_real: [B, 2, H, W]
            w:           [B, W] or [W] allocation
            sigma:       [B, W] or [W] noise std
        Returns:
            x_hat: [B, 2, H, W]
        """
        if w.ndim == 1:
            w = w.unsqueeze(0).expand(kspace_real.shape[0], -1)
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(0).expand(kspace_real.shape[0], -1)
        rho = w / (sigma ** 2 + epsilon)
        mask = (w > 0).float()
        return self.forward(kspace_real, rho, mask)


def build_recon(cfg: dict) -> UnrolledRecon:
    """Build the reconstruction network from recon config."""
    return UnrolledRecon(cfg)
