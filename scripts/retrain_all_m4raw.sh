#!/usr/bin/env bash
# Full retrain + re-eval under the CORRECTED code (√2 noise fix + per-image metrics).
# All 5 trainable models are rebuilt so they are consistent with the new noise model,
# then the whole results/ set is regenerated.
#
# Run it awake & plugged in (~5h on MPS):
#   caffeinate -i bash scripts/retrain_all_m4raw.sh 2>&1 | tee results/retrain_all.log
set -e
cd "$(dirname "$0")/.."

PY="${PY:-/Users/doraviv/miniconda3/bin/python3}"   # override with PY=... ; old MRI-Project-1 venv was deleted
export PYTHONPATH="$PWD"
D=configs/data_m4raw.yaml
N=configs/noise.yaml
P=configs/policy.yaml
R=configs/recon_m4raw.yaml
T=configs/train_m4raw.yaml

echo "=== [1/3] Train 4 baselines (uniform_avg, cs_vda, fixed_poisson, loupe) ==="
"$PY" scripts/run_baselines.py --data "$D" --noise "$N" --policy "$P" --recon "$R" --train "$T"

echo "=== [2/3] Train proposed co-design ==="
"$PY" -m train.train_codesign --data "$D" --noise "$N" --policy "$P" --recon "$R" --train "$T"

echo "=== [3/3] Re-run eval + policy dump + figures ==="
bash scripts/rerun_eval_m4raw.sh
"$PY" scripts/dump_policy.py --checkpoint checkpoints_m4raw/best_codesign.pt \
    --data "$D" --policy "$P" --out results/w_star.npy
"$PY" scripts/make_figures.py --results_dir results/ --out_dir results/figures/

echo "=== DONE — corrected results in results/ ==="
