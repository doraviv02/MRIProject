"""Image reconstruction loss: L1 + (1 - SSIM) + optional perceptual."""

import torch
import torch.nn.functional as F
from typing import Optional


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Differentiable SSIM approximation (1 - SSIM)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = _gaussian_window(11, 1.5).to(device=pred.device, dtype=pred.dtype)

    # Ensure [B, 1, H, W] for conv2d
    pred_u = pred.unsqueeze(1) if pred.ndim == 3 else pred
    tgt_u = target.unsqueeze(1) if target.ndim == 3 else target

    mu1 = F.conv2d(pred_u, window, padding=5, groups=1)
    mu2 = F.conv2d(tgt_u, window, padding=5, groups=1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sig1 = F.conv2d(pred_u ** 2, window, padding=5, groups=1) - mu1_sq
    sig2 = F.conv2d(tgt_u ** 2, window, padding=5, groups=1) - mu2_sq
    sig12 = F.conv2d(pred_u * tgt_u, window, padding=5, groups=1) - mu1_mu2
    lum = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2 * sig12 + C2) / (sig1 + sig2 + C2)
    ssim_map = lum * cs
    return 1.0 - ssim_map.mean()


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return (g.outer(g)).unsqueeze(0).unsqueeze(0)


def image_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Combined image reconstruction loss.

    Args:
        pred:   [B, 2, H, W] predicted (real/imag or [B, 1, H, W] magnitude)
        target: [B, 2, H, W] or [B, 1, H, W] target
        cfg:    loss config dict with l1_weight, ssim_weight, perceptual_weight

    Returns:
        scalar loss
    """
    l1_w = cfg.get("l1_weight", 0.84)
    ssim_w = cfg.get("ssim_weight", 0.16)
    perceptual_w = cfg.get("perceptual_weight", 0.0)

    # Work on magnitude images for loss computation
    if pred.shape[1] == 2:
        mag_pred = (pred[:, 0] ** 2 + pred[:, 1] ** 2).sqrt()   # [B, H, W]
    else:
        mag_pred = pred[:, 0]

    if target.shape[1] == 2:
        mag_tgt = (target[:, 0] ** 2 + target[:, 1] ** 2).sqrt()
    elif target.shape[1] == 1:
        mag_tgt = target[:, 0]
    else:
        mag_tgt = target[:, 0]

    loss = l1_w * F.l1_loss(mag_pred, mag_tgt)
    if ssim_w > 0:
        loss = loss + ssim_w * ssim_loss(mag_pred, mag_tgt)

    if perceptual_w > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(pred.device)
            # LPIPS expects [B, 3, H, W] in [-1, 1]
            p3 = mag_pred.unsqueeze(1).expand(-1, 3, -1, -1) * 2 - 1
            t3 = mag_tgt.unsqueeze(1).expand(-1, 3, -1, -1) * 2 - 1
            loss = loss + perceptual_w * lpips_fn(p3, t3).mean()
        except ImportError:
            pass  # lpips not installed; skip silently

    return loss
