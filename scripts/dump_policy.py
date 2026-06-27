"""Dump the learned averaging allocation w* from a trained codesign checkpoint.

Saves results/w_star.npy (the real policy), used by make_figures.py fig (c).

Usage:
    python scripts/dump_policy.py \
        --checkpoint checkpoints_m4raw/best_codesign.pt \
        --data configs/data_m4raw.yaml --policy configs/policy.yaml \
        --out results/w_star.npy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch

from acq.policy import AcquisitionPolicy


def _infer_n_lines(cfg_data: dict) -> int:
    dataset = cfg_data.get("dataset", "synthetic")
    key = {"synthetic": "synthetic", "m4raw": "m4raw", "fastmri": "fastmri"}.get(dataset, "synthetic")
    size = cfg_data.get(key, {}).get("image_size", [128, 128])
    return size[-1] if isinstance(size, list) else size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints_m4raw/best_codesign.pt")
    p.add_argument("--data", default="configs/data_m4raw.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--out", default="results/w_star.npy")
    args = p.parse_args()

    cfg_data = yaml.safe_load(open(args.data))
    cfg_policy = yaml.safe_load(open(args.policy))
    n_lines = _infer_n_lines(cfg_data)
    budget = cfg_policy.get("budget_factor", 1.0) * n_lines

    ck = torch.load(args.checkpoint, map_location="cpu")
    if "policy" not in ck:
        raise KeyError(f"No 'policy' in checkpoint {args.checkpoint}")
    policy = AcquisitionPolicy(n_lines, budget)
    policy.load_state_dict(ck["policy"])
    with torch.no_grad():
        w = policy().numpy()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, w)
    print(f"Saved w* ({len(w)} lines, B={w.sum():.1f}) → {args.out}")
    print(f"  min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f} std={w.std():.3f}")


if __name__ == "__main__":
    main()
