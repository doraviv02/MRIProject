"""Real-repetition realisation of the learned policy on M4Raw (E5).

Converts the integer allocation w_int into physically averaged k-space using
actual M4Raw repetitions, capping at max_reps = 3.

Usage:
    python -m eval.realize_reps \
        --checkpoint checkpoints/best_codesign.pt \
        --data configs/data.yaml \
        --noise configs/noise.yaml \
        --policy configs/policy.yaml \
        --recon configs/recon.yaml \
        --eval configs/eval.yaml \
        --out results/real_reps_metrics.csv
"""

import argparse
import csv
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from acq.policy import AcquisitionPolicy
from acq.export_protocol import realize_reps, export_protocol, summarise_policy
from models.unrolled_recon import build_recon
from data.m4raw import get_m4raw_splits
from data.noise_calib import load_or_compute_sigma
from eval.metrics import evaluate_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_codesign.pt")
    p.add_argument("--data",   default="configs/data.yaml")
    p.add_argument("--noise",  default="configs/noise.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--recon",  default="configs/recon.yaml")
    p.add_argument("--eval",   default="configs/eval.yaml")
    p.add_argument("--out",    default="results/real_reps_metrics.csv")
    return p.parse_args()


def run(args):
    configs = {}
    for k in ["data", "noise", "policy", "recon", "eval"]:
        with open(getattr(args, k)) as f:
            configs[k] = yaml.safe_load(f)

    cfg_data = configs["data"]
    cfg_noise = configs["noise"]
    cfg_policy = configs["policy"]
    cfg_recon = configs["recon"]
    cfg_eval = configs["eval"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    n_lines = cfg_data.get("m4raw", {}).get("image_size", [256, 256])[-1]
    budget = cfg_policy.get("budget_factor", 1.0) * n_lines
    max_reps = cfg_data.get("m4raw", {}).get("max_reps", 3)

    policy = AcquisitionPolicy(n_lines, budget)
    recon = build_recon(cfg_recon).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        recon.load_state_dict(ckpt["recon"])
        if "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[WARNING] Checkpoint not found: {args.checkpoint}. Using random weights.")

    recon.eval()
    policy.eval()

    # Realizable integer protocol: every line acquired at least once (NEX≥1), with the
    # policy-preferred lines averaged more, capped at max_reps. We deliberately do NOT use
    # the budget-preserving rounding here: that drops the lowest-weight lines to 0 reps
    # (ρ=0), a pattern the recon never saw during training (the continuous policy keeps
    # every w_m≥min>0), which makes the unrolled reconstruction diverge. Flooring at 1 rep
    # keeps ρ>0 everywhere (in-distribution) and yields a physically realizable protocol.
    with torch.no_grad():
        w_cont = policy()
    w_int = torch.clamp(torch.round(w_cont), min=1, max=max_reps).long()  # [N], every line ≥1
    print(summarise_policy(w_cont))
    print(f"Realizable integer protocol: min={w_int.min().item()} max={w_int.max().item()} "
          f"sum={int(w_int.sum().item())} (no dropped lines)")

    # Export protocol
    export_protocol(w_int, out_dir="results/", name="real_reps_protocol")

    # Noise calibration
    cfg_merged = {**cfg_data, "noise": cfg_noise}
    sigma = load_or_compute_sigma(cfg_merged).to(device)

    # Load M4Raw test set (same subject-level split used in training/eval)
    _, _, test_ds = get_m4raw_splits(cfg_data)

    if len(test_ds) == 0:
        print("[ERROR] No M4Raw test samples found. Set m4raw.root in configs/data_m4raw.yaml.")
        return

    results = []
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="real-reps eval"):
            kspace_t, target = test_ds[i]   # kspace_t: [reps, 2, H, W], target: [1, H, W]
            target = target.to(device)

            # Realize: average actual reps per w_int
            ks_complex = torch.view_as_complex(
                kspace_t.permute(0, 2, 3, 1).contiguous()  # [reps, H, W, 2]
            )  # [reps, H, W]

            y_realized = realize_reps(w_int, ks_complex, max_reps=max_reps).to(device)

            y_real_2ch = torch.view_as_real(y_realized).permute(2, 0, 1).contiguous()
            y_real_2ch = y_real_2ch.unsqueeze(0)  # [1, 2, H, W]

            # Adaptive-confidence data consistency. ρ = w_int/σ² assumes the calibrated
            # (global, 20-scan-averaged) noise. On the minority of scans whose real noise
            # is higher than that, ρ is over-confident and the unrolled DC step diverges.
            # So we use the most confident ρ that yields a stable (bounded) reconstruction:
            # try the calibrated σ first, and only fall back to a more conservative σ (k·σ)
            # on samples where the output blows up. Keeps the 47 well-behaved slices at full
            # quality while rescuing the noisier scans instead of letting them explode.
            tgt_max = float(target.max()) + 1e-6
            for k in (1.0, 1.5, 2.0, 3.0):
                rho_b = (w_int.float().to(device) / ((k * sigma.to(device)) ** 2 + 1e-6)).unsqueeze(0)
                x_hat = recon(y_real_2ch, rho_b)  # [1, 2, H, W]
                mag_hat = (x_hat[0, 0] ** 2 + x_hat[0, 1] ** 2).sqrt()
                if torch.isfinite(mag_hat).all() and float(mag_hat.max()) <= 5.0 * tgt_max:
                    break  # stable reconstruction found
            mag_tgt = target[0]

            m = evaluate_all(mag_hat.unsqueeze(0), mag_tgt.unsqueeze(0), compute_lpips_flag=False)
            results.append({"sample": i, **m})

    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    nmses = [r["nmse"] for r in results]
    print(f"\nReal-reps evaluation (budget={budget:.0f}, max_reps={max_reps}):")
    print(f"  PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"  SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"  NMSE: {np.mean(nmses):.4f} ± {np.std(nmses):.4f}")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "psnr", "ssim", "nmse", "lpips"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-sample results saved to {args.out}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
