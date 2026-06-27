"""Train all baselines sequentially.

Usage:
    python scripts/run_baselines.py \
        --data configs/data.yaml \
        --noise configs/noise.yaml \
        --policy configs/policy.yaml \
        --recon configs/recon.yaml \
        --train configs/train.yaml
"""

import argparse
import subprocess
import sys


# full_nex1 uses zero-filled IFFT — no recon network to train; skip here.
BASELINES = ["uniform_avg", "cs_vda", "fixed_poisson", "loupe"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="configs/data.yaml")
    p.add_argument("--noise",  default="configs/noise.yaml")
    p.add_argument("--policy", default="configs/policy.yaml")
    p.add_argument("--recon",  default="configs/recon.yaml")
    p.add_argument("--train",  default="configs/train.yaml")
    return p.parse_args()


def run_baseline(baseline: str, args) -> int:
    cmd = [
        sys.executable, "-m", "train.train_baseline",
        "--baseline", baseline,
        "--data",   args.data,
        "--noise",  args.noise,
        "--policy", args.policy,
        "--recon",  args.recon,
        "--train",  args.train,
    ]
    print(f"\n{'='*60}")
    print(f"Training baseline: {baseline}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    return result.returncode


def main(args):
    failed = []
    for bl in BASELINES:
        rc = run_baseline(bl, args)
        if rc != 0:
            print(f"[WARNING] Baseline {bl} exited with code {rc}.")
            failed.append(bl)

    if failed:
        print(f"\n[SUMMARY] Failed baselines: {', '.join(failed)}")
    else:
        print("\n[SUMMARY] All baselines completed successfully.")


if __name__ == "__main__":
    main(parse_args())
