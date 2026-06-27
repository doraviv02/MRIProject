"""Joint co-design training: learn policy logits a and reconstruction parameters θ together.

Training loop (Section 4.5):
  min_{a,θ} E_{x,η} [ L_img(R_θ(ỹ(w,x,η), ρ), x) ] + β · Reg(a)
  s.t.  w = B · softmax(a/τ)

Milestones:
  1. Warmup phase (epochs 0..warmup_epochs-1): train recon under fixed UNIFORM allocation.
  2. Co-design phase (epochs warmup_epochs..): unfreeze policy logits and train jointly.

Usage:
    python -m train.train_codesign \
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

from acq.policy import AcquisitionPolicy, anneal_temperature, compute_precision
from acq.measurement import simulate_measurement, ifft2c
from models.unrolled_recon import build_recon
from eval.metrics import compute_psnr, compute_ssim, compute_nmse
from train._loss import image_loss
from train._data_loader import get_data_loaders
from data.noise_calib import load_or_compute_sigma, sample_noise_scale


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="configs/data.yaml")
    p.add_argument("--noise",  default="configs/noise.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--recon",  default="configs/recon.yaml")
    p.add_argument("--train",  default="configs/train.yaml")
    return p.parse_args()


def load_configs(args) -> dict:
    cfg = {}
    for k, path in vars(args).items():
        with open(path) as f:
            cfg[k] = yaml.safe_load(f)
    return cfg


def build_optimizers(policy: AcquisitionPolicy, recon: nn.Module, cfg: dict):
    opt_cfg = cfg.get("optimizer", {})
    lr_r = opt_cfg.get("lr_recon", 1e-3)
    lr_p = opt_cfg.get("lr_policy", 1e-2)
    wd = opt_cfg.get("weight_decay", 1e-5)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    opt_recon = optim.Adam(recon.parameters(), lr=lr_r, weight_decay=wd, betas=betas)
    opt_policy = optim.Adam([policy.logits], lr=lr_p, weight_decay=0.0, betas=betas)
    return opt_recon, opt_policy


def train_epoch(
    recon: nn.Module,
    policy: AcquisitionPolicy,
    loader: DataLoader,
    opt_recon: optim.Optimizer,
    opt_policy: optim.Optimizer,
    sigma: torch.Tensor,
    cfg_recon: dict,
    cfg_train: dict,
    device: torch.device,
    policy_frozen: bool,
    cfg_noise: dict = None,
) -> dict:
    recon.train()
    policy.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0

    clip = cfg_train.get("grad_clip_norm", 1.0)
    loss_cfg = cfg_recon.get("loss", {})

    for batch in tqdm(loader, leave=False, desc="train"):
        # batch is (kspace [B,2,H,W] or (kspace,target) depending on dataset)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x_clean, target = batch
        else:
            x_clean = batch
            target = x_clean
        x_clean = x_clean.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        # For synthetic/fastmri datasets, x_clean is already [B,2,H,W] image.
        # For M4Raw, x_clean is [B, reps, 2, H, W] k-space; IFFT averaged k-space → image.
        if x_clean.ndim == 5:
            kspace_avg = x_clean.mean(dim=1)  # [B, 2, H, W] averaged k-space
            ks_complex = torch.view_as_complex(kspace_avg.permute(0, 2, 3, 1).contiguous())
            img_complex = ifft2c(ks_complex)  # [B, H, W]
            x_clean = torch.view_as_real(img_complex).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        # Forward through policy → measurement → reconstruction.
        # Per-batch σ scale (1.0 unless SNR augmentation is enabled in cfg_noise).
        w = policy()  # [N]
        sigma_dev = sigma.to(device) * sample_noise_scale(cfg_noise)

        y_tilde, rho = simulate_measurement(x_clean, w, sigma_dev)  # noisy k-space
        x_hat = recon(y_tilde, rho)  # [B, 2, H, W]

        # Image loss
        loss_img = image_loss(x_hat, target, loss_cfg)
        reg_loss = policy.regularization_loss()
        loss = loss_img + reg_loss

        opt_recon.zero_grad()
        if not policy_frozen:
            opt_policy.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(recon.parameters(), clip)
        if not policy_frozen:
            nn.utils.clip_grad_norm_([policy.logits], clip)

        opt_recon.step()
        if not policy_frozen:
            opt_policy.step()

        with torch.no_grad():
            # PSNR on magnitude image
            mag_hat = (x_hat[:, 0] ** 2 + x_hat[:, 1] ** 2).sqrt()
            mag_tgt = (target[:, 0] ** 2 + target[:, 1] ** 2).sqrt() if target.shape[1] >= 2 else target[:, 0]
            psnr_val = compute_psnr(mag_hat, mag_tgt)

        total_loss += loss.item()
        total_psnr += psnr_val
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "psnr": total_psnr / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(
    recon: nn.Module,
    policy: AcquisitionPolicy,
    loader: DataLoader,
    sigma: torch.Tensor,
    cfg_recon: dict,
    device: torch.device,
) -> dict:
    recon.eval()
    policy.eval()
    total_psnr, total_ssim, total_nmse, n = 0.0, 0.0, 0.0, 0

    loss_cfg = cfg_recon.get("loss", {})

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x_clean, target = batch
        else:
            x_clean = batch
            target = x_clean
        x_clean = x_clean.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        if x_clean.ndim == 5:
            kspace_avg = x_clean.mean(dim=1)
            ks_complex = torch.view_as_complex(kspace_avg.permute(0, 2, 3, 1).contiguous())
            img_complex = ifft2c(ks_complex)
            x_clean = torch.view_as_real(img_complex).permute(0, 3, 1, 2).contiguous()

        w = policy()
        sigma_dev = sigma.to(device)
        y_tilde, rho = simulate_measurement(x_clean, w, sigma_dev)
        x_hat = recon(y_tilde, rho)

        mag_hat = (x_hat[:, 0] ** 2 + x_hat[:, 1] ** 2).sqrt()
        mag_tgt = (target[:, 0] ** 2 + target[:, 1] ** 2).sqrt() if target.shape[1] >= 2 else target[:, 0]

        total_psnr += compute_psnr(mag_hat, mag_tgt)
        total_ssim += compute_ssim(mag_hat, mag_tgt)
        total_nmse += compute_nmse(mag_hat, mag_tgt)
        n += 1

    n = max(n, 1)
    return {"psnr": total_psnr / n, "ssim": total_ssim / n, "nmse": total_nmse / n}


def train(cfg: dict):
    cfg_data = cfg["data"]
    cfg_noise = cfg["noise"]
    cfg_policy = cfg["policy"]
    cfg_recon = cfg["recon"]
    cfg_train = cfg["train"]

    torch.manual_seed(cfg_train.get("seed", 42))
    device_str = cfg_train.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU"); device_str = "cpu"
    elif device_str == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("[WARNING] MPS not available, falling back to CPU"); device_str = "cpu"
    device = torch.device(device_str)

    # Data
    train_loader, val_loader, _ = get_data_loaders(cfg_data, cfg_train)
    n_lines = _infer_n_lines(train_loader, cfg_data)

    # Noise calibration
    cfg_merged = {**cfg_data, "noise": cfg_noise}
    sigma = load_or_compute_sigma(cfg_merged)

    # Budget
    budget = cfg_policy.get("budget_factor", 1.0) * n_lines

    # Policy
    policy = AcquisitionPolicy(
        n_lines=n_lines,
        budget=budget,
        temperature_init=cfg_policy.get("temperature_init", 1.0),
        entropy_reg=cfg_policy.get("entropy_reg_weight", 0.0),
        sparsity_reg=cfg_policy.get("sparsity_reg_weight", 0.0),
        logit_init=cfg_policy.get("logit_init", "zeros"),
    ).to(device)

    # Reconstruction network
    recon = build_recon(cfg_recon).to(device)

    # Optimisers
    opt_recon, opt_policy = build_optimizers(policy, recon, cfg_train)
    sch_cfg = cfg_train.get("scheduler", {})
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt_recon,
        T_max=cfg_train.get("epochs", 50),
        eta_min=sch_cfg.get("min_lr", 1e-5),
    )

    ckpt_dir = cfg_train.get("checkpoint_dir", "checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)

    warmup = cfg_train.get("warmup_epochs", 5)
    total_epochs = cfg_train.get("epochs", 50)
    tau_init = cfg_policy.get("temperature_init", 1.0)
    tau_final = cfg_policy.get("temperature_final", 0.1)
    tau_anneal = cfg_policy.get("temperature_anneal_epochs", 30)

    best_ssim = -1.0
    for epoch in range(total_epochs):
        policy_frozen = (epoch < warmup)
        if policy_frozen:
            # During warmup, force uniform allocation
            with torch.no_grad():
                policy.logits.fill_(0.0)

        # Temperature annealing
        tau = anneal_temperature(epoch, warmup, total_epochs, tau_init, tau_final, tau_anneal)
        policy.set_temperature(tau)

        train_stats = train_epoch(
            recon, policy, train_loader, opt_recon, opt_policy,
            sigma, cfg_recon, cfg_train, device, policy_frozen, cfg_noise=cfg_noise,
        )
        val_stats = val_epoch(recon, policy, val_loader, sigma, cfg_recon, device)
        sched.step()

        print(
            f"Epoch {epoch+1:03d}/{total_epochs} | τ={tau:.3f} | "
            f"loss={train_stats['loss']:.4f} | "
            f"val PSNR={val_stats['psnr']:.2f} SSIM={val_stats['ssim']:.4f} "
            f"NMSE={val_stats['nmse']:.4f} | frozen={policy_frozen}"
        )

        if val_stats["ssim"] > best_ssim:
            best_ssim = val_stats["ssim"]
            torch.save(
                {
                    "epoch": epoch,
                    "recon": recon.state_dict(),
                    "policy": policy.state_dict(),
                    "val_stats": val_stats,
                },
                os.path.join(ckpt_dir, "best_codesign.pt"),
            )

        if (epoch + 1) % cfg_train.get("save_every_n_epochs", 5) == 0:
            torch.save(
                {"epoch": epoch, "recon": recon.state_dict(), "policy": policy.state_dict()},
                os.path.join(ckpt_dir, f"codesign_ep{epoch+1:03d}.pt"),
            )

    print(f"Training complete. Best val SSIM: {best_ssim:.4f}")
    return recon, policy


def _infer_n_lines(loader: DataLoader, cfg_data: dict) -> int:
    """Infer N (number of phase-encode lines) from the first batch."""
    dataset = cfg_data.get("dataset", "synthetic")
    if dataset == "synthetic":
        size = cfg_data.get("synthetic", {}).get("image_size", [128, 128])
        return size[-1] if isinstance(size, list) else size
    elif dataset == "m4raw":
        size = cfg_data.get("m4raw", {}).get("image_size", [256, 256])
        return size[-1] if isinstance(size, list) else size
    else:
        size = cfg_data.get("fastmri", {}).get("image_size", [320, 320])
        return size[-1] if isinstance(size, list) else size


if __name__ == "__main__":
    args = parse_args()
    cfg = load_configs(args)
    train(cfg)
