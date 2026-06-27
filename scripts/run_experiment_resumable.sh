#!/usr/bin/env bash
# Resumable full experiment. Trains each model only if not already marked done,
# so a sleep/kill costs at most the in-progress model — just re-launch to continue.
#   caffeinate -dimsu bash scripts/run_experiment_resumable.sh 2>&1 | tee -a results/full_run.log
# (run plugged in + lid OPEN; caffeinate can't stop clamshell sleep)
set -e
cd "$(dirname "$0")/.."

PY="${PY:-/Users/doraviv/miniconda3/bin/python3}"
export PYTHONPATH="$PWD"
D=configs/data_m4raw.yaml; N=configs/noise.yaml; P=configs/policy.yaml
R=configs/recon_m4raw.yaml; T=configs/train_m4raw.yaml
M=checkpoints_m4raw; mkdir -p "$M"

train_one () {  # $1 = model name; trains unless $M/<name>.done exists
  local name="$1"
  if [ -f "$M/$name.done" ]; then echo "== skip $name (already done) =="; return; fi
  echo "== training $name =="
  if [ "$name" = "codesign" ]; then
    "$PY" -m train.train_codesign --data "$D" --noise "$N" --policy "$P" --recon "$R" --train "$T"
  else
    "$PY" -m train.train_baseline --baseline "$name" --data "$D" --noise "$N" --policy "$P" --recon "$R" --train "$T"
  fi
  touch "$M/$name.done"   # only on success (set -e aborts before this on failure)
  echo "== done $name =="
}

for m in uniform_avg cs_vda fixed_poisson loupe codesign; do train_one "$m"; done

echo "== all models trained; running eval =="
bash scripts/rerun_eval_m4raw.sh
"$PY" scripts/dump_policy.py --checkpoint "$M/best_codesign.pt" --data "$D" --policy "$P" --out results/w_star.npy
"$PY" scripts/make_figures.py --results_dir results/ --out_dir results/figures/
echo "== EXPERIMENT COMPLETE =="
