# Spend the Scan Time Wisely: Learned Averages-vs-Coverage Allocation for Low-Field MRI

**Course:** Advanced Topics in Deep Learning for Medical Imaging

---

## Overview

This project jointly learns *where* in k-space to invest scan time — as **averaging counts per phase-encode line** — together with a reconstruction network, for low-field MRI where SNR is the scarce resource.

At low field, the classic tradeoff is:
- **More averages** on a line → lower noise variance on that line (SNR ∝ √averages).
- **More k-space coverage** → fewer aliasing artifacts and higher spatial resolution.

Our contribution: a **differentiable, budget-constrained, jointly-learned averaging-allocation policy** that generalises both pure undersampling (LOUPE-style) and fixed averaging strategies (CS-VDA), validated on real multi-repetition low-field M4Raw data.

---

## Dataset

**Primary: M4Raw**
- Reference: Lyu et al., *Scientific Data* 2023; 10:264
- Download: **Zenodo 7523691** or **GitHub mylyu/M4Raw**
- 0.3 T, 4-channel coil, 183 subjects, T1w / T2w / FLAIR, **2–3 repetitions each**
- Format: **fastMRI-format `.h5` files** (k-space arrays, shape `[reps, coils, slices, readout, phase_encode]`)
- **Config key for data path:** `configs/data.yaml` → `m4raw.root`

**Optional: fastMRI** (high-SNR control, E7)
- Register: fastmri.med.nyu.edu
- Config key: `configs/data.yaml` → `fastmri.root`

**No-download fallback:** Set `configs/data.yaml` → `dataset: synthetic` to use the built-in Shepp-Logan phantom.

---

## File Types Read

| Source   | Format | Key      |
|----------|--------|----------|
| M4Raw    | `.h5` (HDF5, fastMRI-style) | `m4raw.root` |
| fastMRI  | `.h5` (HDF5, fastMRI-style) | `fastmri.root` |
| Synthetic| generated in-memory | — |

---

## Installation

```bash
pip install -r requirements.txt
```

Optional (multi-coil CS):
```bash
pip install sigpy
```

---

## Reproducing Every Figure / Table

### Quick smoke test (no data required):
```bash
cd project_acq_allocation_lowfield/
bash scripts/run_all.sh synthetic
```

### Full M4Raw pipeline:
1. Download M4Raw to `data/m4raw/` (set `m4raw.root` in `configs/data.yaml`).
2. ```bash
   bash scripts/run_all.sh m4raw
   ```

### Individual steps:
```bash
# Noise calibration
python -m data.noise_calib --config configs/data.yaml --out results/sigma_profile.npy

# Train proposed model
python -m train.train_codesign

# Train a specific baseline
python -m train.train_baseline --baseline loupe

# Train all baselines
python scripts/run_baselines.py

# Main evaluation table
python scripts/make_tables.py

# SNR sweep (E2)
python -m eval.snr_sweep

# Real-reps evaluation (E5, M4Raw only)
python -m eval.realize_reps

# All figures
python scripts/make_figures.py

# Statistical tests
python -m eval.stats
```

---

## Repository Structure

```
project_acq_allocation_lowfield/
  configs/             # All YAML hyperparameter configs
  data/                # Dataset loaders + noise calibration
  acq/                 # Policy layer, measurement model, protocol export
  models/              # Unrolled reconstructor + CNN denoiser
  baselines/           # 5 baselines (full_nex1, uniform_avg, cs_vda, fixed_poisson, loupe)
  train/               # Co-design + baseline trainers
  eval/                # Metrics, SNR sweep, policy plots, stats, real-reps
  scripts/             # run_all.sh, run_baselines.py, make_figures.py, make_tables.py
  results/
    figures/           # All paper figures (PDF)
    tables/            # CSV tables
    comparison_with_literature.md
  deliverables/
    README.md          # This file
    report/
      report.md        # 2–4 page course report
    slides/            # Presentation slides
  requirements.txt
```

---

## Key Config Keys

| File | Key | Effect |
|------|-----|--------|
| `configs/data.yaml` | `dataset` | `synthetic` / `m4raw` / `fastmri` |
| `configs/data.yaml` | `m4raw.root` | Path to downloaded M4Raw `.h5` files |
| `configs/policy.yaml` | `budget_factor` | B/N (1.0 = iso-time with full NEX=1) |
| `configs/noise.yaml` | `sigma_scale` | Scale factor for noise sweep (E2) |
| `configs/train.yaml` | `warmup_epochs` | Recon-only pretraining before policy unfreezes |
| `configs/recon.yaml` | `num_unrolled_steps` | K (unrolled iterations) |

---

## Methods Compared

| # | Method | Key novelty compared |
|---|--------|---------------------|
| 1 | Full NEX=1 IFFT | Conventional reference |
| 2 | Uniform averaging + VarNet | Fixed allocation, same recon |
| 3 | CS-VDA (hand-designed) | Fixed center-weighted allocation |
| 4 | Fixed Poisson-disc + VarNet | Fixed undersampling strategy |
| 5 | LOUPE (learned mask, NEX=1) | Learned sampling, **no averaging axis** |
| — | **Proposed** | Learned allocation + reconstruction jointly; **averaging axis** |

---

## Citation

If you use this code, please cite:

```
Lyu M, Mei Y, Huang Z, et al. M4Raw: a multi-contrast, multi-repetition, multi-channel MRI
k-space dataset for low-field MRI research. Scientific Data 2023;10:264.

Lau KS, Xiao L, Zhao T, et al. Pushing the limits of low-cost ultra-low-field MRI by
dual-acquisition deep learning 3D superresolution. Magn Reson Med 2023;90(2):400–416.
```
