"""Train a fixed-policy baseline (frozen allocation, learnable reconstruction only).

Used for baselines 2–5 where the policy is not jointly optimised:
  - uniform_avg    : w_m = B/N ∀m
  - cs_vda         : centre-weighted polynomial allocation
  - fixed_poisson  : binary Poisson-disc mask at NEX=1
  - loupe          : LOUPE learned binary mask at NEX=1 (learns policy too, but NEX=1)

Usage:
    python -m train.train_baseline \
        --baseline uniform_avg \
        --data configs/data.yaml \
        --noise configs/noise.yaml \
        --policy configs/policy.yaml \
        --recon configs/recon.yaml \
        --train configs/train.yaml
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from acq.measurement import simulate_measurement, ifft2c
from models.unrolled_recon import build_recon
from baselines.loupe_mask import LOUPEPolicy, loupe_measurement
from eval.metrics import compute_psnr, compute_ssim, compute_nmse
from train._loss import image_loss
from train._data_loader import get_data_loaders
from data.noise_calib import load_or_compute_sigma, sample_noise_scale


BASELINE_CHOICES = ["full_nex1", "uniform_avg", "cs_vda", "fixed_poisson", "loupe"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="uniform_avg", choices=BASELINE_CHOICES)
    p.add_argument("--data",   default="configs/data.yaml")
    p.add_argument("--noise",  default="configs/noise.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--recon",  default="configs/recon.yaml")
    p.add_argument("--train",  default="configs/train.yaml")
    return p.parse_args()


def load_configs(args) -> dict:
    cfg = {}
    for k in ["data", "noise", "policy", "recon", "train"]:
        with open(getattr(args, k)) as f:
            cfg[k] = yaml.safe_load(f)
    cfg["baseline"] = args.baseline
    return cfg


def get_fixed_w(baseline: str, n_lines: int, budget: float, sigma: torch.Tensor,
                device: torch.device) -> torch.Tensor:
    """Return the fixed allocation vector for a given baseline."""
    if baseline == "full_nex1":
        return torch.ones(n_lines, device=device)
    elif baseline == "uniform_avg":
        from baselines.uniform_avg import get_uniform_w
        return get_uniform_w(n_lines, budget, device=str(device))
    elif baseline == "cs_vda":
        from baselines.cs_vda import cs_vda_allocation
        return cs_vda_allocation(n_lines, budget, device=str(device))
    elif baseline == "fixed_poisson":
        from baselines.fixed_poisson_cs import get_poisson_allocation
        return get_poisson_allocation(n_lines, budget, device=str(device))
    else:
        raise ValueError(f"Unsupported fixed baseline: {baseline}")


def train_baseline(cfg: dict):
    cfg_data = cfg["data"]
    cfg_noise = cfg["noise"]
    cfg_policy = cfg["policy"]
    cfg_recon = cfg["recon"]
    cfg_train = cfg["train"]
    baseline = cfg["baseline"]

    torch.manual_seed(cfg_train.get("seed", 42))
    device_str = cfg_train.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU"); device_str = "cpu"
    elif device_str == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("[WARNING] MPS not available, falling back to CPU"); device_str = "cpu"
    device = torch.device(device_str)

    train_loader, val_loader, _ = get_data_loaders(cfg_data, cfg_train)
    n_lines = _infer_n_lines(cfg_data)
    budget = cfg_policy.get("budget_factor", 1.0) * n_lines

    cfg_merged = {**cfg_data, "noise": cfg_noise}
    sigma = load_or_compute_sigma(cfg_merged).to(device)

    recon = build_recon(cfg_recon).to(device)

    # LOUPE has its own learnable policy
    loupe_policy = None
    if baseline == "loupe":
        loupe_slope = cfg_policy.get("loupe_slope", 5.0)
        loupe_policy = LOUPEPolicy(n_lines, budget, slope=loupe_slope).to(device)

    opt_cfg = cfg_train.get("optimizer", {})
    params = list(recon.parameters())
    if loupe_policy is not None:
        params += list(loupe_policy.parameters())
    optimizer = optim.Adam(
        params,
        lr=opt_cfg.get("lr_recon", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_train.get("epochs", 50),
        eta_min=cfg_train.get("scheduler", {}).get("min_lr", 1e-5),
    )

    ckpt_dir = cfg_train.get("checkpoint_dir", "checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)
    loss_cfg = cfg_recon.get("loss", {})
    clip = cfg_train.get("grad_clip_norm", 1.0)
    best_ssim = -1.0

    for epoch in range(cfg_train.get("epochs", 50)):
        recon.train()
        total_loss, n_b = 0.0, 0

        for batch in tqdm(train_loader, leave=False, desc=f"[{baseline}] ep{epoch+1}"):
            if isinstance(batch, (list, tuple)):
                x_clean, target = batch
            else:
                x_clean = target = batch
            x_clean = x_clean.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)
            if x_clean.ndim == 5:  # M4Raw: [B, reps, 2, H, W] k-space → image
                kspace_avg = x_clean.mean(dim=1)
                ks_c = torch.view_as_complex(kspace_avg.permute(0, 2, 3, 1).contiguous())
                x_clean = torch.view_as_real(ifft2c(ks_c)).permute(0, 3, 1, 2).contiguous()

            # Per-batch σ scale (1.0 unless SNR augmentation enabled in cfg_noise).
            sigma_b = sigma * sample_noise_scale(cfg_noise)
            if baseline == "loupe":
                mask_soft, _ = loupe_policy()
                y_tilde, rho = loupe_measurement(x_clean, mask_soft, sigma_b)
            else:
                w = get_fixed_w(baseline, n_lines, budget, sigma_b, device)
                y_tilde, rho = simulate_measurement(x_clean, w, sigma_b)

            x_hat = recon(y_tilde, rho)
            loss = image_loss(x_hat, target, loss_cfg)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        sched.step()

        # Validation
        recon.eval()
        val_psnr, val_ssim, n_v = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x_clean, target = batch
                else:
                    x_clean = target = batch
                x_clean = x_clean.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                if x_clean.ndim == 5:  # M4Raw k-space → image
                    kspace_avg = x_clean.mean(dim=1)
                    ks_c = torch.view_as_complex(kspace_avg.permute(0, 2, 3, 1).contiguous())
                    x_clean = torch.view_as_real(ifft2c(ks_c)).permute(0, 3, 1, 2).contiguous()

                if baseline == "loupe":
                    mask_soft, _ = loupe_policy()
                    y_tilde, rho = loupe_measurement(x_clean, mask_soft, sigma)
                else:
                    w = get_fixed_w(baseline, n_lines, budget, sigma, device)
                    y_tilde, rho = simulate_measurement(x_clean, w, sigma)

                x_hat = recon(y_tilde, rho)
                mag_hat = (x_hat[:, 0] ** 2 + x_hat[:, 1] ** 2).sqrt()
                mag_tgt = (target[:, 0] ** 2 + target[:, 1] ** 2).sqrt() if target.shape[1] >= 2 else target[:, 0]
                val_psnr += compute_psnr(mag_hat, mag_tgt)
                val_ssim += compute_ssim(mag_hat, mag_tgt)
                n_v += 1

        n_v = max(n_v, 1)
        print(
            f"[{baseline}] Epoch {epoch+1:03d} | loss={total_loss/max(n_b,1):.4f} | "
            f"val PSNR={val_psnr/n_v:.2f} SSIM={val_ssim/n_v:.4f}"
        )

        if val_ssim / n_v > best_ssim:
            best_ssim = val_ssim / n_v
            ckpt = {"epoch": epoch, "recon": recon.state_dict()}
            if loupe_policy is not None:
                ckpt["loupe"] = loupe_policy.state_dict()
            torch.save(ckpt, os.path.join(ckpt_dir, f"best_{baseline}.pt"))

    return recon, loupe_policy


def _infer_n_lines(cfg_data: dict) -> int:
    dataset = cfg_data.get("dataset", "synthetic")
    key = {"synthetic": "synthetic", "m4raw": "m4raw", "fastmri": "fastmri"}.get(dataset, "synthetic")
    size = cfg_data.get(key, {}).get("image_size", [128, 128])
    return size[-1] if isinstance(size, list) else size


if __name__ == "__main__":
    args = parse_args()
    cfg = load_configs(args)
    train_baseline(cfg)
