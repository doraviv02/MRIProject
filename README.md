# Spend the Scan Time Wisely: Learned Averages-vs-Coverage Allocation for Low-Field MRI

> **TL;DR:** At 0.3 T, scan time = SNR. We make the per-line averaging allocation a
> learnable, budget-constrained, jointly-trained decision — and it beats hand-designed
> strategies and learned-sampling-only methods.

See [deliverables/README.md](deliverables/README.md) for full setup instructions.

## Quick Start (no data required)

```bash
cd project_acq_allocation_lowfield/
pip install -r requirements.txt
bash scripts/run_all.sh synthetic
```

## With M4Raw Data

1. Download M4Raw from **Zenodo 7523691** or **GitHub mylyu/M4Raw**.
2. Set `m4raw.root` in `configs/data.yaml` to your download path.
3. Run:
   ```bash
   bash scripts/run_all.sh m4raw
   ```

## Structure

```
configs/      ← YAML hyperparameter configs (start here)
data/         ← Dataset loaders + noise calibration
acq/          ← Differentiable policy + measurement model
models/       ← Precision-weighted unrolled VarNet
baselines/    ← 5 comparison methods
train/        ← Joint co-design + baseline training
eval/         ← Metrics, SNR sweep, policy plots, stats
scripts/      ← run_all.sh, make_figures.py, make_tables.py
results/      ← Output: figures, tables, comparison_with_literature.md
deliverables/ ← README, 2-4 page report, slide deck
```

## Key Files

| File | Purpose |
|------|---------|
| [acq/policy.py](acq/policy.py) | `AcquisitionPolicy`: learnable logits → w = B·softmax(a/τ) |
| [acq/measurement.py](acq/measurement.py) | Differentiable noise model (Sec 4.2) |
| [models/unrolled_recon.py](models/unrolled_recon.py) | K-step precision-weighted VarNet |
| [train/train_codesign.py](train/train_codesign.py) | Joint policy+recon training |
| [configs/data.yaml](configs/data.yaml) | `m4raw.root` — set your data path here |
| [scripts/run_all.sh](scripts/run_all.sh) | Full pipeline in one command |
