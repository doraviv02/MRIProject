"""Generate evaluation tables for all methods on the test set (E1).

Runs all trained models on the M4Raw (or synthetic) test set and collects
per-sample metrics, then produces:
  - results/tables/main_table.csv (mean ± std per method × metric)
  - results/per_sample_metrics.csv (one row per sample; needed for stats.py)

Usage:
    python scripts/make_tables.py \
        --checkpoints_dir checkpoints/ \
        --data configs/data.yaml \
        --noise configs/noise.yaml \
        --policy configs/policy.yaml \
        --recon configs/recon.yaml \
        --eval configs/eval.yaml \
        --out_dir results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import yaml
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List

from acq.measurement import simulate_measurement, ifft2c
from models.unrolled_recon import build_recon
from acq.policy import AcquisitionPolicy
from baselines.loupe_mask import LOUPEPolicy, loupe_measurement
from baselines.cs_vda import cs_vda_allocation
from baselines.uniform_avg import get_uniform_w
from baselines.fixed_poisson_cs import get_poisson_allocation
from baselines.full_nex1 import full_nex1_reconstruct
from eval.metrics import evaluate_all
from eval.recon_stable import stable_recon
from data.noise_calib import load_or_compute_sigma
from train._data_loader import get_data_loaders

METHODS = ["full_nex1", "uniform_avg", "cs_vda", "fixed_poisson", "loupe", "codesign"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", default="checkpoints/")
    p.add_argument("--data",   default="configs/data.yaml")
    p.add_argument("--noise",  default="configs/noise.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--recon",  default="configs/recon.yaml")
    p.add_argument("--eval",   default="configs/eval.yaml")
    p.add_argument("--out_dir", default="results/")
    return p.parse_args()


def load_method(name: str, ckpt_dir: str, cfg_recon: dict, n_lines: int,
                budget: float, device: torch.device):
    recon = build_recon(cfg_recon).to(device)
    policy = None
    ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "best_codesign.pt") if name == "codesign" else ckpt_path
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        recon.load_state_dict(ckpt["recon"])
        if name == "codesign" and "policy" in ckpt:
            policy = AcquisitionPolicy(n_lines, budget).to(device)
            policy.load_state_dict(ckpt["policy"])
        elif name == "loupe" and "loupe" in ckpt:
            policy = LOUPEPolicy(n_lines, budget).to(device)
            policy.load_state_dict(ckpt["loupe"])
    recon.eval()
    return recon, policy


def _infer_n_lines(cfg_data: dict) -> int:
    dataset = cfg_data.get("dataset", "synthetic")
    key = {"synthetic": "synthetic", "m4raw": "m4raw", "fastmri": "fastmri"}.get(dataset, "synthetic")
    size = cfg_data.get(key, {}).get("image_size", [128, 128])
    return size[-1] if isinstance(size, list) else size


def main(args):
    configs = {}
    for k in ["data", "noise", "policy", "recon", "eval"]:
        with open(getattr(args, k)) as f:
            configs[k] = yaml.safe_load(f)

    cfg_data  = configs["data"]
    cfg_noise = configs["noise"]
    cfg_policy= configs["policy"]
    cfg_recon = configs["recon"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _, _, test_loader = get_data_loaders(cfg_data, {"batch_size": 1, "num_workers": 2, "pin_memory": False})

    n_lines = _infer_n_lines(cfg_data)
    budget = cfg_policy.get("budget_factor", 1.0) * n_lines

    cfg_merged = {**cfg_data, "noise": cfg_noise}
    sigma = load_or_compute_sigma(cfg_merged).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)

    all_rows: List[dict] = []
    summary: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for method in METHODS:
        print(f"\n--- Evaluating: {method} ---")
        if method == "full_nex1":
            recon, policy = None, None
        else:
            recon, policy = load_method(method, args.checkpoints_dir, cfg_recon, n_lines, budget, device)

        with torch.no_grad():
            # Compute fixed w for non-loupe, non-codesign methods
            if method == "full_nex1":
                w = torch.ones(n_lines, device=device)
            elif method == "uniform_avg":
                w = get_uniform_w(n_lines, budget, device=str(device))
            elif method == "cs_vda":
                w = cs_vda_allocation(n_lines, budget, device=str(device))
            elif method == "fixed_poisson":
                w = get_poisson_allocation(n_lines, budget, device=str(device))
            elif method == "codesign":
                w = policy() if policy else get_uniform_w(n_lines, budget, device=str(device))
            else:
                w = None  # LOUPE handled per-batch

        sample_id = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=method):
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

                if method == "loupe" and policy is not None:
                    mask_soft, _ = policy()
                    y_tilde, rho = loupe_measurement(x_clean, mask_soft, sigma)
                else:
                    y_tilde, rho = simulate_measurement(x_clean, w, sigma)

                if target.shape[1] == 1:
                    mag_tgt = target[:, 0]
                else:
                    mag_tgt = (target[:, 0] ** 2 + target[:, 1] ** 2).sqrt()

                if method == "full_nex1":
                    x_hat = full_nex1_reconstruct(y_tilde)
                else:
                    x_hat = stable_recon(recon, y_tilde, rho, mag_tgt)

                mag_hat = (x_hat[:, 0] ** 2 + x_hat[:, 1] ** 2).sqrt()

                m = evaluate_all(mag_hat.squeeze(), mag_tgt.squeeze(), compute_lpips_flag=False)
                row = {"method": method, "sample": sample_id, **m}
                all_rows.append(row)
                for k, v in m.items():
                    summary[method][k].append(v)
                sample_id += 1

        mean_psnr = np.mean(summary[method]["psnr"])
        mean_ssim = np.mean(summary[method]["ssim"])
        mean_nmse = np.mean(summary[method]["nmse"])
        print(f"  PSNR={mean_psnr:.2f}  SSIM={mean_ssim:.4f}  NMSE={mean_nmse:.4f}")

    # Save per-sample
    per_sample_path = os.path.join(args.out_dir, "per_sample_metrics.csv")
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sample", "psnr", "ssim", "nmse", "lpips"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-sample metrics: {per_sample_path}")

    # Save summary table
    main_table_path = os.path.join(args.out_dir, "tables", "main_table.csv")
    with open(main_table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "PSNR mean", "PSNR std", "SSIM mean", "SSIM std",
                         "NMSE mean", "NMSE std"])
        for method in METHODS:
            if method not in summary:
                continue
            writer.writerow([
                method,
                f"{np.mean(summary[method]['psnr']):.3f}",
                f"{np.std(summary[method]['psnr']):.3f}",
                f"{np.mean(summary[method]['ssim']):.4f}",
                f"{np.std(summary[method]['ssim']):.4f}",
                f"{np.mean(summary[method]['nmse']):.4f}",
                f"{np.std(summary[method]['nmse']):.4f}",
            ])
    print(f"Main table: {main_table_path}")


if __name__ == "__main__":
    main(parse_args())
