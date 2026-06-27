"""Image quality metrics: PSNR, SSIM, NMSE, LPIPS (fastMRI definitions).

All functions accept [B, H, W] or [H, W] float tensors in [0, 1].
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


Tensor = torch.Tensor


def _data_range(target: Tensor, data_range) -> Tensor:
    """Per-image data range (fastMRI convention: gt.max()). Returns [B] tensor.
    If data_range is given explicitly, broadcast it to [B]."""
    flat = target.reshape(target.shape[0], -1) if target.ndim >= 3 else target.reshape(1, -1)
    if data_range is None:
        dr = flat.amax(dim=-1)
    else:
        dr = torch.as_tensor(float(data_range), device=target.device,
                             dtype=target.dtype).expand(flat.shape[0]).clone()
    return dr.clamp_min(1e-12)


def compute_psnr(pred: Tensor, target: Tensor, data_range=None) -> float:
    """Peak signal-to-noise ratio (dB), per-image data range (gt.max). Averaged over batch."""
    dr = _data_range(target, data_range)                 # [B]
    mse = ((pred - target) ** 2).reshape(dr.shape[0], -1).mean(dim=-1)  # [B]
    psnr = 10 * torch.log10(dr ** 2 / (mse + 1e-12))
    return psnr.mean().item()


def compute_ssim(pred: Tensor, target: Tensor, data_range=None) -> float:
    """SSIM using an 11×11 Gaussian window, per-image data range, averaged over batch."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    dr = _data_range(target, data_range).view(-1, 1, 1, 1)  # [B,1,1,1]
    C1 = (0.01 * dr) ** 2
    C2 = (0.03 * dr) ** 2
    win = _gaussian_window(11, 1.5).to(device=pred.device, dtype=pred.dtype)

    pred_4d = pred.unsqueeze(1)       # [B, 1, H, W]
    tgt_4d = target.unsqueeze(1)

    mu1 = F.conv2d(pred_4d, win, padding=5, groups=1)
    mu2 = F.conv2d(tgt_4d, win, padding=5, groups=1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    sig1 = F.conv2d(pred_4d ** 2, win, padding=5, groups=1) - mu1_sq
    sig2 = F.conv2d(tgt_4d ** 2, win, padding=5, groups=1) - mu2_sq
    sig12 = F.conv2d(pred_4d * tgt_4d, win, padding=5, groups=1) - mu1 * mu2
    lum = (2 * mu1 * mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2 * sig12 + C2) / (sig1 + sig2 + C2)
    return (lum * cs).mean().item()


def compute_nmse(pred: Tensor, target: Tensor) -> float:
    """Normalised mean squared error (fastMRI def): ||pred-target||²/||target||²."""
    num = ((pred - target) ** 2).sum(dim=(-2, -1))
    den = (target ** 2).sum(dim=(-2, -1)) + 1e-12
    return (num / den).mean().item()


def compute_lpips(pred: Tensor, target: Tensor) -> float:
    """LPIPS perceptual metric (requires lpips package)."""
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(pred.device)
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        p3 = pred.unsqueeze(1).expand(-1, 3, -1, -1) * 2 - 1
        t3 = target.unsqueeze(1).expand(-1, 3, -1, -1) * 2 - 1
        with torch.no_grad():
            return lpips_fn(p3, t3).mean().item()
    except ImportError:
        return float("nan")


def _gaussian_window(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.outer(g)).unsqueeze(0).unsqueeze(0)


def evaluate_all(
    pred: Tensor,
    target: Tensor,
    data_range=None,
    compute_lpips_flag: bool = True,
) -> dict:
    """Compute all metrics and return a dict."""
    return {
        "psnr": compute_psnr(pred, target, data_range),
        "ssim": compute_ssim(pred, target, data_range),
        "nmse": compute_nmse(pred, target),
        "lpips": compute_lpips(pred, target) if compute_lpips_flag else float("nan"),
    }


@torch.no_grad()
def evaluate_loader(
    recon_fn,
    loader,
    device: str = "cpu",
    compute_lpips_flag: bool = False,
) -> dict:
    """
    Run recon_fn on all batches and aggregate metrics.

    recon_fn(batch) → (pred [B,H,W], target [B,H,W]) both float, normalised.
    """
    metrics = {"psnr": [], "ssim": [], "nmse": [], "lpips": []}
    for batch in loader:
        pred, target = recon_fn(batch)
        pred = pred.to(device)
        target = target.to(device)
        m = evaluate_all(pred, target, compute_lpips_flag=compute_lpips_flag)
        for k in metrics:
            metrics[k].append(m[k])

    return {k: float(np.mean([v for v in vs if not np.isnan(v)])) for k, vs in metrics.items()}
