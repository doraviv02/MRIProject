"""Shared inference-time stabiliser for the unrolled reconstruction.

The unrolled DC step can diverge on samples where the precision ρ is over-confident
relative to the actual noise (e.g. the cs_vda allocation on a few hard slices, or
real-rep scans noisier than the global calibration). This picks the most confident ρ
that still yields a finite, bounded reconstruction: try ρ as-is, then progressively
reduce confidence (ρ·f, equivalent to inflating σ) until the output is sane.

Methods that don't diverge are unaffected — the first (f=1.0) attempt passes.
"""

import torch

# f multiplies ρ; equals 1/k² for σ→k·σ. Ladder mirrors realize_reps: k = 1, 1.5, 2, 3.
_FACTORS = (1.0, 1.0 / 2.25, 1.0 / 4.0, 1.0 / 9.0)


def stable_recon(recon, y_tilde, rho, target_mag, max_ratio: float = 5.0):
    """Return recon(y_tilde, ρ) using the most confident ρ that stays bounded.

    Args:
        recon:      unrolled recon module (callable (y_tilde, rho) -> [B,2,H,W]).
        y_tilde:    [B,2,H,W] measured k-space.
        rho:        [B,N] base precision.
        target_mag: target magnitude tensor (for the sanity bound).
        max_ratio:  reject a reconstruction whose max magnitude exceeds this × target max.
    """
    tgt_max = float(target_mag.max()) + 1e-6
    x_hat = recon(y_tilde, rho)  # default attempt
    for f in _FACTORS:
        if f != 1.0:
            x_hat = recon(y_tilde, rho * f)
        mag = (x_hat[:, 0] ** 2 + x_hat[:, 1] ** 2).sqrt()
        if torch.isfinite(mag).all() and float(mag.max()) <= max_ratio * tgt_max:
            return x_hat
    return x_hat  # last (most conservative) attempt
