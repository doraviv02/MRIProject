"""SNR sweep experiment (E2): metrics vs σ scale for all methods.

Sweeps sigma_scale in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0] — higher means lower SNR.
For each scale, runs all methods on the test set and records PSNR, SSIM, NMSE.

Usage:
    python -m eval.snr_sweep \
        --checkpoints_dir checkpoints/ \
        --data configs/data.yaml \
        --noise configs/noise.yaml \
        --policy configs/policy.yaml \
        --recon configs/recon.yaml \
        --eval configs/eval.yaml \
        --out results/snr_sweep.csv
"""

import argparse
import csv
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from eval.metrics import evaluate_all
from eval.recon_stable import stable_recon
from data.noise_calib import load_or_compute_sigma
from acq.measurement import ifft2c
from acq.measurement import simulate_measurement
from models.unrolled_recon import build_recon
from acq.policy import AcquisitionPolicy
from baselines.cs_vda import cs_vda_allocation
from baselines.uniform_avg import get_uniform_w
from baselines.fixed_poisson_cs import get_poisson_allocation
from baselines.loupe_mask import LOUPEPolicy, loupe_measurement
from train._data_loader import get_data_loaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", default="checkpoints/")
    p.add_argument("--data",   default="configs/data.yaml")
    p.add_argument("--noise",  default="configs/noise.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--recon",  default="configs/recon.yaml")
    p.add_argument("--eval",   default="configs/eval.yaml")
    p.add_argument("--out",    default="results/snr_sweep.csv")
    return p.parse_args()


def load_method(name: str, ckpt_dir: str, cfg_recon: dict, n_lines: int,
                budget: float, device: torch.device):
    """Load a trained method (recon + optional policy)."""
    recon = build_recon(cfg_recon).to(device)
    policy = None
    ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Checkpoint not found: {ckpt_path}. Using random weights.")
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        recon.load_state_dict(ckpt["recon"])
        if name == "codesign":
            policy = AcquisitionPolicy(n_lines, budget).to(device)
            policy.load_state_dict(ckpt.get("policy", {}))
        elif name == "loupe":
            policy = LOUPEPolicy(n_lines, budget).to(device)
            if "loupe" in ckpt:
                policy.load_state_dict(ckpt["loupe"])
    recon.eval()
    return recon, policy


def run_sweep(args):
    configs = {}
    for k in ["data", "noise", "policy", "recon", "eval"]:
        with open(getattr(args, k)) as f:
            configs[k] = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cfg_data, cfg_noise, cfg_policy, cfg_recon, cfg_eval = (
        configs["data"], configs["noise"], configs["policy"], configs["recon"], configs["eval"]
    )

    _, _, test_loader = get_data_loaders(cfg_data, {"batch_size": 1, "num_workers": 2,
                                                      "pin_memory": False})
    n_lines = _infer_n_lines(cfg_data)
    budget = cfg_policy.get("budget_factor", 1.0) * n_lines
    sigma_scales = cfg_eval.get("snr_sweep", {}).get("sigma_scales", [0.25, 0.5, 1.0, 2.0, 4.0])

    # Reference sigma at the TRAINING scale. ρ is intentionally evaluated at this scale
    # rather than the per-sweep test scale, for two reasons:
    #   (1) Correctness: the per-line *relative* precision ρ_m ∝ w_m/σ_m² is invariant to
    #       the global sweep factor (it cancels), so ρ_ref carries the same reliability
    #       pattern the recon needs — it is the faithful way to evaluate the *same* trained
    #       model across noise levels (its learnable DC step η is calibrated to this scale).
    #   (2) Stability: feeding raw test-scale ρ (≈ ρ_ref / scale²) would blow up the DC
    #       gradient step at scale ≪ 1. The noisy measurement y_tilde DOES use the test
    #       scale, so the SNR sweep still measures true degradation vs noise.
    train_scale = cfg_noise.get("sigma_scale", 1.0)
    cfg_noise_ref = dict(cfg_noise, sigma_scale=train_scale)
    sigma_ref = load_or_compute_sigma({**cfg_data, "noise": cfg_noise_ref}).to(device)

    methods = ["full_nex1", "uniform_avg", "cs_vda", "fixed_poisson", "loupe", "codesign"]
    results = []

    for scale in tqdm(sigma_scales, desc="sigma_scale"):
        # Actual sigma at this scale (used for noise simulation only)
        cfg_noise_s = dict(cfg_noise, sigma_scale=scale)
        cfg_merged = {**cfg_data, "noise": cfg_noise_s}
        sigma = load_or_compute_sigma(cfg_merged).to(device)

        # Effective sigma for the PRECISION ρ fed to the recon: the true test-scale σ,
        # but floored at the training scale (σ_eff = max(σ_test, σ_ref)). Rationale:
        #   - high noise (scale>1): use the true (lower) precision so the data-consistency
        #     step doesn't over-trust noisy data and diverge;
        #   - low noise (scale<1): cap confidence at the trained level, else ρ≈ρ_ref/scale²
        #     blows up the DC gradient step (η was calibrated to the training scale).
        # The relative per-line precision pattern is identical under any global scale, so
        # this faithfully evaluates the same trained model across the whole SNR range.
        sigma_eff = torch.maximum(sigma, sigma_ref)

        for method in methods:
            # full_nex1 uses zero-filled IFFT; skip checkpoint loading
            if method == "full_nex1":
                recon, policy = None, None
            else:
                recon, policy = load_method(
                    method, args.checkpoints_dir, cfg_recon, n_lines, budget, device,
                )

            # Determine allocation
            if method == "full_nex1":
                w = torch.ones(n_lines, device=device)
            elif method == "uniform_avg":
                w = get_uniform_w(n_lines, budget, device=str(device))
            elif method == "cs_vda":
                w = cs_vda_allocation(n_lines, budget, device=str(device))
            elif method == "fixed_poisson":
                w = get_poisson_allocation(n_lines, budget, device=str(device))
            elif method == "loupe":
                w = None  # handled separately
            elif method == "codesign":
                w = policy() if policy else get_uniform_w(n_lines, budget, device=str(device))
            else:
                w = None

            psnrs, ssims, nmses = [], [], []
            with torch.no_grad():
                for batch in test_loader:
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

                    mag_tgt = (target[:, 0] ** 2 + target[:, 1] ** 2).sqrt() if target.shape[1] == 2 else target[:, 0]

                    if method == "full_nex1":
                        from baselines.full_nex1 import full_nex1_reconstruct
                        y_tilde, rho = simulate_measurement(x_clean, w, sigma)
                        x_hat = full_nex1_reconstruct(y_tilde)
                    elif method == "loupe" and policy is not None:
                        mask_soft, _ = policy()
                        # Noise at the test scale; precision from sigma_eff (see header note).
                        y_tilde, _ = loupe_measurement(x_clean, mask_soft, sigma)
                        _, rho = loupe_measurement(x_clean, mask_soft, sigma_eff)
                        x_hat = stable_recon(recon, y_tilde, rho, mag_tgt)
                    else:
                        # Noise injected at the test scale (sigma); precision from sigma_eff =
                        # max(sigma_test, sigma_ref) — true precision floored at training scale,
                        # which keeps the DC step stable across the whole SNR range (header note).
                        y_tilde, _ = simulate_measurement(x_clean, w, sigma)
                        _, rho = simulate_measurement(x_clean, w, sigma_eff)
                        x_hat = stable_recon(recon, y_tilde, rho, mag_tgt)

                    mag_hat = (x_hat[:, 0] ** 2 + x_hat[:, 1] ** 2).sqrt()
                    m = evaluate_all(mag_hat.squeeze(), mag_tgt.squeeze(), compute_lpips_flag=False)
                    psnrs.append(m["psnr"])
                    ssims.append(m["ssim"])
                    nmses.append(m["nmse"])

            results.append({
                "sigma_scale": scale,
                "method": method,
                "psnr": np.mean(psnrs),
                "ssim": np.mean(ssims),
                "nmse": np.mean(nmses),
            })
            print(f"  σ_scale={scale:.2f} | {method:15s} PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f}")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma_scale", "method", "psnr", "ssim", "nmse"])
        writer.writeheader()
        writer.writerows(results)
    print(f"SNR sweep results saved to {args.out}")
    return results


def _infer_n_lines(cfg_data: dict) -> int:
    dataset = cfg_data.get("dataset", "synthetic")
    key = {"synthetic": "synthetic", "m4raw": "m4raw", "fastmri": "fastmri"}.get(dataset, "synthetic")
    size = cfg_data.get(key, {}).get("image_size", [128, 128])
    return size[-1] if isinstance(size, list) else size


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
