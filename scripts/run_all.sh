#!/usr/bin/env bash
# run_all.sh — End-to-end pipeline: smoke-test → train → eval → figures → tables
# Usage: bash scripts/run_all.sh [--dataset synthetic|m4raw] [--gpu 0]
#
# Steps:
#   1. Smoke test on synthetic phantom (no data required)
#   2. (Optional) Noise calibration from M4Raw
#   3. Train all baselines
#   4. Train proposed co-design model
#   5. Evaluate all methods → per-sample metrics + main table
#   6. SNR sweep
#   7. Real-reps evaluation (M4Raw only)
#   8. Generate all figures
#   9. Statistical significance tests

set -e

DATASET="${1:-synthetic}"
GPU="${2:-0}"
CONFIGS_PREFIX="configs"

echo "============================================"
echo " Learned Acquisition Allocation — Low-Field MRI"
echo " Dataset: ${DATASET}  |  GPU: ${GPU}"
echo "============================================"

export CUDA_VISIBLE_DEVICES=${GPU}

# ── 1. Smoke test ──────────────────────────────────────────────────────────────
echo ""
echo "Step 1: Smoke test (synthetic phantom, forward pass only)"
python -c "
import torch
from data.synthetic_phantom import get_phantom_splits
from acq.policy import AcquisitionPolicy
from acq.measurement import simulate_measurement
from models.unrolled_recon import build_recon
from eval.metrics import compute_psnr, compute_ssim

import yaml
with open('${CONFIGS_PREFIX}/recon.yaml') as f:
    cfg = yaml.safe_load(f)

n = 128; B = n; sigma = torch.full((n,), 0.05)
policy = AcquisitionPolicy(n, B)
recon = build_recon(cfg)

x = torch.zeros(2, 2, n, n); x[:, 0, 30:90, 30:90] = 1.0
w = policy()
y, rho = simulate_measurement(x, w, sigma)
xhat = recon(y, rho)
mag = (xhat[:, 0]**2 + xhat[:, 1]**2).sqrt()
tgt = (x[:, 0]**2 + x[:, 1]**2).sqrt()
print(f'Smoke test PSNR: {compute_psnr(mag, tgt):.2f} dB  SSIM: {compute_ssim(mag, tgt):.4f}')
print('Smoke test PASSED.')
"

# ── 2. Noise calibration (M4Raw only) ─────────────────────────────────────────
if [ "${DATASET}" = "m4raw" ]; then
    echo ""
    echo "Step 2: Noise calibration from M4Raw"
    python -m data.noise_calib \
        --config ${CONFIGS_PREFIX}/data.yaml \
        --noise_config ${CONFIGS_PREFIX}/noise.yaml \
        --out results/sigma_profile.npy
fi

# ── 3. Train baselines ─────────────────────────────────────────────────────────
echo ""
echo "Step 3: Train all baselines"
python scripts/run_baselines.py \
    --data   ${CONFIGS_PREFIX}/data.yaml \
    --noise  ${CONFIGS_PREFIX}/noise.yaml \
    --policy ${CONFIGS_PREFIX}/policy.yaml \
    --recon  ${CONFIGS_PREFIX}/recon.yaml \
    --train  ${CONFIGS_PREFIX}/train.yaml

# ── 4. Train proposed co-design ───────────────────────────────────────────────
echo ""
echo "Step 4: Train proposed co-design model"
python -m train.train_codesign \
    --data   ${CONFIGS_PREFIX}/data.yaml \
    --noise  ${CONFIGS_PREFIX}/noise.yaml \
    --policy ${CONFIGS_PREFIX}/policy.yaml \
    --recon  ${CONFIGS_PREFIX}/recon.yaml \
    --train  ${CONFIGS_PREFIX}/train.yaml

# ── 5. Main evaluation table ──────────────────────────────────────────────────
echo ""
echo "Step 5: Evaluate all methods on test set"
python scripts/make_tables.py \
    --checkpoints_dir checkpoints/ \
    --data   ${CONFIGS_PREFIX}/data.yaml \
    --noise  ${CONFIGS_PREFIX}/noise.yaml \
    --policy ${CONFIGS_PREFIX}/policy.yaml \
    --recon  ${CONFIGS_PREFIX}/recon.yaml \
    --eval   ${CONFIGS_PREFIX}/eval.yaml \
    --out_dir results/

# ── 6. SNR sweep ──────────────────────────────────────────────────────────────
echo ""
echo "Step 6: SNR sweep experiment (E2)"
python -m eval.snr_sweep \
    --checkpoints_dir checkpoints/ \
    --data   ${CONFIGS_PREFIX}/data.yaml \
    --noise  ${CONFIGS_PREFIX}/noise.yaml \
    --policy ${CONFIGS_PREFIX}/policy.yaml \
    --recon  ${CONFIGS_PREFIX}/recon.yaml \
    --eval   ${CONFIGS_PREFIX}/eval.yaml \
    --out results/snr_sweep.csv

# ── 7. Real-reps evaluation (M4Raw only) ──────────────────────────────────────
if [ "${DATASET}" = "m4raw" ]; then
    echo ""
    echo "Step 7: Real-repetition evaluation (E5)"
    python -m eval.realize_reps \
        --checkpoint checkpoints/best_codesign.pt \
        --data   ${CONFIGS_PREFIX}/data.yaml \
        --noise  ${CONFIGS_PREFIX}/noise.yaml \
        --policy ${CONFIGS_PREFIX}/policy.yaml \
        --recon  ${CONFIGS_PREFIX}/recon.yaml \
        --eval   ${CONFIGS_PREFIX}/eval.yaml \
        --out results/real_reps_metrics.csv
fi

# ── 8. Figures ────────────────────────────────────────────────────────────────
echo ""
echo "Step 8: Generate all figures"
python scripts/make_figures.py \
    --results_dir results/ \
    --out_dir results/figures/

# ── 9. Statistical tests ──────────────────────────────────────────────────────
echo ""
echo "Step 9: Paired Wilcoxon significance tests"
python -m eval.stats \
    --results_dir results/ \
    --metric ssim

echo ""
echo "============================================"
echo " Pipeline complete."
echo " Results in: results/"
echo " Figures in: results/figures/"
echo " Tables  in: results/tables/"
echo "============================================"
