"""Per-line noise calibration from M4Raw inter-repetition variance.

Estimates σ_m (single-average complex-Gaussian noise std per phase-encode line m)
from the variance across repetitions in M4Raw k-space data.

Usage:
    python -m data.noise_calib --config configs/data.yaml --out results/sigma_profile.npy
"""

import argparse
import math
import os
import random
import h5py
import numpy as np
import torch
from typing import List, Optional


def sample_noise_scale(cfg_noise: Optional[dict]) -> float:
    """Per-batch random σ multiplier for SNR-augmented training.

    Returns 1.0 unless cfg_noise['augment']['enabled'] is true, in which case it
    draws a log-uniform scale in [scale_min, scale_max]. Training with this teaches
    the recon to handle a range of SNRs (so the E2 sweep can fairly test whether the
    learned allocation's gains are largest at low SNR) instead of overfitting one level.
    """
    aug = (cfg_noise or {}).get("augment", {})
    if not aug.get("enabled", False):
        return 1.0
    lo = float(aug.get("scale_min", 0.25))
    hi = float(aug.get("scale_max", 4.0))
    return float(math.exp(random.uniform(math.log(lo), math.log(hi))))


def _fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2-D FFT (orthonormal) matching the data loader convention."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def estimate_sigma_from_reps(kspace_reps: np.ndarray) -> np.ndarray:
    """
    Estimate per-line noise std σ_m from inter-repetition variance.

    Args:
        kspace_reps: [reps, H, W] complex k-space; H = readout, W = phase-encode

    Returns:
        sigma: [W] real, one value per phase-encode line
    """
    # Variance across repetitions per (readout, phase-encode) point
    var = kspace_reps.var(axis=0)  # [H, W] real
    # Average variance over readout dimension → per phase-encode line variance
    sigma_sq = var.mean(axis=0)    # [W]
    # σ_m is std of the complex noise in a single readout point of line m;
    # since complex variance = var_real + var_imag, and var we compute is already
    # the scalar variance of complex numbers, we take sqrt directly.
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-12))
    return sigma.astype(np.float32)


def estimate_sigma_from_background(kspace: np.ndarray, roi: int = 10) -> np.ndarray:
    """
    Estimate per-line σ_m from background (air) region of a single k-space.

    Args:
        kspace: [H, W] complex k-space
        roi: number of rows in background ROI (corner of image space)

    Returns:
        sigma: [W] estimated per-line std (uniform across lines as fallback)
    """
    # Transform to image space, take corner ROI
    img = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)))
    corner = img[:roi, :roi]
    sigma_global = float(np.abs(corner).std())
    # Uniform profile as fallback
    sigma = np.full(kspace.shape[-1], sigma_global, dtype=np.float32)
    return sigma


def calibrate_from_scan_group(
    rep_files: List[str],
    image_size: int = 256,
    n_slices: int = 12,
) -> Optional[np.ndarray]:
    """
    Calibrate σ_m from one scan group (list of per-repetition .h5 files).

    Uses reconstruction_rss (already coil-combined per rep) → FFT → inter-rep
    variance, so sigma is in the SAME units as the k-space used in training
    (which is FFT of the RSS image).

    Returns:
        sigma: [image_size] float32, or None if <2 reps available
    """
    if len(rep_files) < 2:
        return None

    total_slices = 18
    lo = (total_slices - n_slices) // 2
    slice_indices = list(range(lo, lo + n_slices))

    profiles = []
    for sl in slice_indices:
        ks_reps = []
        # Load the first rep to get the per-scan scale
        with h5py.File(rep_files[0], "r") as hf:
            rss0 = hf["reconstruction_rss"][sl].astype(np.float32)
        scale = rss0.max() + 1e-8

        for fpath in rep_files:
            with h5py.File(fpath, "r") as hf:
                rss = hf["reconstruction_rss"][sl].astype(np.float32)
            rss = rss / scale  # same normalisation as the data loader
            # Center-crop if needed
            H, W = rss.shape
            if H != image_size or W != image_size:
                sh = (H - image_size) // 2
                sw = (W - image_size) // 2
                rss = rss[sh: sh + image_size, sw: sw + image_size]
            # Single-coil k-space = FFT(RSS image) — matches training representation
            ks = _fft2c(rss)  # [H, W] complex
            ks_reps.append(ks.astype(np.complex64))

        ks_stack = np.stack(ks_reps, axis=0)  # [reps, H, W]
        sigma = estimate_sigma_from_reps(ks_stack)
        profiles.append(sigma)

    return np.stack(profiles, axis=0).mean(axis=0)  # [image_size]


def calibrate_from_m4raw_dir(
    root: str,
    contrast: str = "T2w",
    image_size: int = 256,
    max_scans: int = 20,
) -> np.ndarray:
    """
    Estimate σ_m profile from the actual M4Raw flat directory structure.
    Groups files by (subject, contrast), estimates sigma from inter-rep variance.
    """
    import re
    from collections import defaultdict
    _CONTRAST_TAG = {"T1w": "T1", "T2w": "T2", "FLAIR": "FLAIR", "T1": "T1", "T2": "T2"}
    _REP_PATTERN = re.compile(r"^(\d+)_(T1|T2|FLAIR)(\d{2})$")
    tag = _CONTRAST_TAG.get(contrast, contrast)

    groups = defaultdict(list)
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".h5"):
            continue
        stem = fname[:-3]
        m = _REP_PATTERN.match(stem)
        if m is None or m.group(2) != tag:
            continue
        scan_key = f"{m.group(1)}_{m.group(2)}"
        groups[scan_key].append(os.path.join(root, fname))
    groups = {k: sorted(v) for k, v in sorted(groups.items())}

    all_profiles = []
    for i, (key, rep_files) in enumerate(groups.items()):
        if i >= max_scans:
            break
        prof = calibrate_from_scan_group(rep_files, image_size=image_size)
        if prof is not None:
            all_profiles.append(prof)

    if not all_profiles:
        return flat_sigma_profile(image_size, sigma_val=0.05)
    return np.stack(all_profiles, axis=0).mean(axis=0).astype(np.float32)


def flat_sigma_profile(n: int, sigma_val: float = 0.05) -> np.ndarray:
    """Uniform σ_m profile (synthetic fallback)."""
    return np.full(n, sigma_val, dtype=np.float32)


def load_or_compute_sigma(cfg: dict, h5_files: Optional[List[str]] = None) -> torch.Tensor:
    """
    Load precomputed σ profile or compute from h5_files.

    Returns:
        sigma: torch.FloatTensor [N]
    """
    noise_cfg = cfg.get("noise", {})
    save_path = noise_cfg.get("save_profile", "results/sigma_profile.npy")
    precomputed = noise_cfg.get("sigma_profile", None)
    scale = float(noise_cfg.get("sigma_scale", 1.0))

    if precomputed and os.path.isfile(precomputed):
        sigma = np.load(precomputed).astype(np.float32)

    elif os.path.isfile(save_path) and cfg.get("dataset") == "m4raw":
        sigma = np.load(save_path).astype(np.float32)

    elif cfg.get("dataset") == "m4raw":
        m4raw_cfg = cfg.get("m4raw", {})
        root = m4raw_cfg.get("root", "data/m4raw")
        contrast = m4raw_cfg.get("contrast", "T2w")
        image_size = m4raw_cfg.get("image_size", [256, 256])
        if isinstance(image_size, list):
            image_size = image_size[-1]
        print(f"[noise_calib] Estimating σ from M4Raw ({contrast}) ...")
        sigma = calibrate_from_m4raw_dir(root, contrast=contrast, image_size=image_size)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, sigma)
        print(f"[noise_calib] σ: min={sigma.min():.4f} max={sigma.max():.4f} "
              f"mean={sigma.mean():.4f} → saved to {save_path}")

    elif h5_files:
        data_cfg = cfg.get("m4raw", cfg.get("data", {}))
        image_size = data_cfg.get("image_size", [256, 256])
        if isinstance(image_size, list):
            image_size = image_size[-1]
        sigma = calibrate_from_m4raw_dir(
            os.path.dirname(h5_files[0]), image_size=image_size
        )
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, sigma)

    else:
        # Fallback: flat profile — infer N from the active dataset config
        dataset = cfg.get("dataset", "synthetic")
        data_cfg = cfg.get(dataset, cfg.get("synthetic", {}))
        N = data_cfg.get("image_size", [128, 128])
        if isinstance(N, list):
            N = N[-1]
        sigma = flat_sigma_profile(N, sigma_val=0.05)

    sigma = sigma * scale
    return torch.from_numpy(sigma)


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--noise_config", default="configs/noise.yaml")
    parser.add_argument("--out", default="results/sigma_profile.npy")
    args = parser.parse_args()

    with open(args.config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.noise_config) as f:
        noise_cfg = yaml.safe_load(f)

    m4raw_root = data_cfg["m4raw"]["root"]
    contrast = data_cfg["m4raw"]["contrast"]
    train_dir = os.path.join(m4raw_root, contrast, "multicoil_train")
    h5_files = sorted([
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith(".h5")
    ])[:20]  # use first 20 files for speed

    sigma = calibrate_from_files(
        h5_files,
        max_reps=data_cfg["m4raw"]["max_reps"],
        image_size=data_cfg["m4raw"]["image_size"][-1],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, sigma)
    print(f"Saved σ profile [{sigma.shape}] to {args.out}")
    print(f"  σ_m: min={sigma.min():.4f}, max={sigma.max():.4f}, mean={sigma.mean():.4f}")
