#!/usr/bin/env bash
# Re-run evaluation/aggregation on the FINAL M4Raw checkpoints (checkpoints_m4raw/).
# The original run_all.sh hardcodes checkpoints/ (synthetic) + recon.yaml, which is why
# results/ went stale. This points eval at the real models with the M4Raw config set.
set -e
cd "$(dirname "$0")/.."

PY="${PY:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python3}}"; PY="${PY:-python3}"   # uses active conda env if any, else PATH python3
CKPT=checkpoints_m4raw/
DATA=configs/data_m4raw.yaml
NOISE=configs/noise.yaml
POLICY=configs/policy.yaml
RECON=configs/recon_m4raw.yaml
EVAL=configs/eval.yaml

echo "=== Step 1/5: main table + per-sample metrics (E1) ==="
"$PY" scripts/make_tables.py --checkpoints_dir "$CKPT" --data "$DATA" --noise "$NOISE" \
    --policy "$POLICY" --recon "$RECON" --eval "$EVAL" --out_dir results/

echo "=== Step 2/5: SNR sweep (E2) ==="
"$PY" -m eval.snr_sweep --checkpoints_dir "$CKPT" --data "$DATA" --noise "$NOISE" \
    --policy "$POLICY" --recon "$RECON" --eval "$EVAL" --out results/snr_sweep.csv

echo "=== Step 3/5: real-reps eval (E5) ==="
"$PY" -m eval.realize_reps --checkpoint "$CKPT/best_codesign.pt" --data "$DATA" --noise "$NOISE" \
    --policy "$POLICY" --recon "$RECON" --eval "$EVAL" --out results/real_reps_metrics.csv || \
    echo "[WARN] real-reps step failed; continuing"

echo "=== Step 4/5: figures ==="
"$PY" scripts/make_figures.py --results_dir results/ --out_dir results/figures/

echo "=== Step 5/5: stats (paired Wilcoxon) ==="
"$PY" -m eval.stats --results_dir results/ --metric ssim || echo "[WARN] stats step failed"

echo "=== DONE ==="
