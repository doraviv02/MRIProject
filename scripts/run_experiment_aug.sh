#!/usr/bin/env bash
# SNR-AUGMENTED experiment (tests the "largest gains at low SNR" claim properly).
# Trains all 5 models with per-batch log-uniform σ scaling (configs/noise_aug.yaml) into a
# SEPARATE checkpoint dir, then evaluates them with the standard fixed/swept noise so the
# fixed-noise results in results/ are untouched. Resumable per-model (.done markers).
#
# Run it in a real terminal (the harness reaps background jobs after ~30 min):
#   caffeinate -i bash scripts/run_experiment_aug.sh 2>&1 | tee -a results_aug/run.log
set -e
cd "$(dirname "$0")/.."

PY="${PY:-/Users/doraviv/miniconda3/bin/python3}"
export PYTHONPATH="$PWD"
D=configs/data_m4raw.yaml; P=configs/policy.yaml; R=configs/recon_m4raw.yaml
N_AUG=configs/noise_aug.yaml      # augmented noise — TRAINING only
N_EVAL=configs/noise.yaml         # standard noise — EVAL (sweep injects its own scales)
T=configs/train_m4raw_aug.yaml
M=checkpoints_m4raw_aug; mkdir -p "$M" results_aug/figures

train_one () {
  local name="$1"
  if [ -f "$M/$name.done" ]; then echo "== skip $name (done) =="; return; fi
  echo "== training $name (SNR-augmented) =="
  if [ "$name" = "codesign" ]; then
    "$PY" -m train.train_codesign --data "$D" --noise "$N_AUG" --policy "$P" --recon "$R" --train "$T"
  else
    "$PY" -m train.train_baseline --baseline "$name" --data "$D" --noise "$N_AUG" --policy "$P" --recon "$R" --train "$T"
  fi
  touch "$M/$name.done"; echo "== done $name =="
}

for m in uniform_avg cs_vda fixed_poisson loupe codesign; do train_one "$m"; done

echo "== eval augmented models (standard noise) → results_aug/ =="
"$PY" scripts/make_tables.py --checkpoints_dir "$M" --data "$D" --noise "$N_EVAL" --policy "$P" --recon "$R" --eval configs/eval.yaml --out_dir results_aug/
"$PY" -m eval.snr_sweep    --checkpoints_dir "$M" --data "$D" --noise "$N_EVAL" --policy "$P" --recon "$R" --eval configs/eval.yaml --out results_aug/snr_sweep.csv
"$PY" -m eval.realize_reps --checkpoint "$M/best_codesign.pt" --data "$D" --noise "$N_EVAL" --policy "$P" --recon "$R" --eval configs/eval.yaml --out results_aug/real_reps_metrics.csv || echo "[WARN] realize_reps failed"
"$PY" scripts/dump_policy.py --checkpoint "$M/best_codesign.pt" --data "$D" --policy "$P" --out results_aug/w_star.npy
"$PY" scripts/make_figures.py --results_dir results_aug/ --out_dir results_aug/figures/
echo "== AUGMENTED EXPERIMENT COMPLETE (compare results_aug/snr_sweep.csv vs results/snr_sweep.csv) =="
