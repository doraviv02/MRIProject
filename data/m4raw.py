"""M4Raw dataset loader — actual on-disk format.

M4Raw (Lyu et al., Scientific Data 2023; Zenodo 7523691) stores data as one
.h5 file per repetition per (subject, contrast):

    {subject}_{contrast_tag}{rep:02d}.h5
    e.g.  2022061203_T201.h5  →  subject 2022061203, T2, repetition 01

Each .h5 contains:
    kspace              : (slices=18, coils=4, 256, 256) complex64
    reconstruction_rss  : (slices=18, 256, 256) float32  (per-rep RSS image)

All files live in a single flat directory (cfg['m4raw']['root']).  We group
by (subject, contrast), split subjects into train/val/test, and return per-
slice samples.  The single-coil k-space per rep is F{ reconstruction_rss },
which avoids having to do coil-combination in the training loop.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ──────────────────────────── helpers ──────────────────────────────────────

_CONTRAST_TAG: Dict[str, str] = {
    "T1w": "T1", "T2w": "T2", "FLAIR": "FLAIR",
    "T1":  "T1", "T2":  "T2",
}

_REP_PATTERN = re.compile(r"^(\d+)_(T1|T2|FLAIR)(\d{2})$")


def _parse_filename(path: str):
    """Return (subject, contrast_tag, rep_int) or None if not matched."""
    stem = os.path.basename(path)
    if stem.endswith(".h5"):
        stem = stem[:-3]
    m = _REP_PATTERN.match(stem)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2-D FFT of a real or complex array (orthonormal)."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def _group_files(root: str, contrast: str) -> Dict[str, List[str]]:
    """
    Scan root for .h5 files matching the contrast, group by scan key.

    Returns:
        dict mapping scan_key -> sorted list of rep file paths
        e.g. '2022061203_T2' -> ['.../2022061203_T201.h5', '.../2022061203_T202.h5', ...]
    """
    tag = _CONTRAST_TAG.get(contrast, contrast)
    groups: Dict[str, List[str]] = defaultdict(list)
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".h5"):
            continue
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        subject, ctag, _rep = parsed
        if ctag != tag:
            continue
        scan_key = f"{subject}_{ctag}"
        groups[scan_key].append(os.path.join(root, fname))
    return {k: sorted(v) for k, v in sorted(groups.items())}


def _split_by_subject(
    groups: Dict[str, List[str]],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """Split scan groups by subject so no subject leaks across splits."""
    subjects = sorted({k.split("_")[0] for k in groups})
    rng = np.random.default_rng(seed)
    subjects_arr = np.array(subjects)
    rng.shuffle(subjects_arr)
    n = len(subjects_arr)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    train_subj = set(subjects_arr[:n_train])
    val_subj   = set(subjects_arr[n_train: n_train + n_val])

    train_g, val_g, test_g = {}, {}, {}
    for k, v in groups.items():
        s = k.split("_")[0]
        if s in train_subj:
            train_g[k] = v
        elif s in val_subj:
            val_g[k] = v
        else:
            test_g[k] = v
    return train_g, val_g, test_g


# ──────────────────────────── dataset ──────────────────────────────────────

class M4RawSliceDataset(Dataset):
    """
    One sample = one slice from one (subject, contrast) scan.

    Returns:
        kspace_t  : [reps, 2, H, W]  – single-coil k-space per rep (real/imag)
        target_t  : [1, H, W]         – magnitude target (avg RSS across reps)
    """

    def __init__(
        self,
        groups: Dict[str, List[str]],
        image_size: Tuple[int, int] = (256, 256),
        max_reps: int = 3,
        num_slices: int = 12,
    ):
        self.image_size = image_size
        self.max_reps = max_reps
        # Build index: list of (scan_key, rep_files, slice_idx)
        self._index: List[Tuple[List[str], int]] = []
        total_slices = 18  # M4Raw always has 18 slices per scan
        lo = (total_slices - num_slices) // 2
        slice_indices = list(range(lo, lo + num_slices))
        for _key, rep_files in sorted(groups.items()):
            reps = rep_files[:max_reps]
            for sl in slice_indices:
                self._index.append((reps, sl))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rep_files, sl = self._index[idx]
        H, W = self.image_size

        rss_reps = []
        for fpath in rep_files:
            with h5py.File(fpath, "r") as hf:
                rss = hf["reconstruction_rss"][sl]  # [H, W] float32
            rss_reps.append(rss.astype(np.float32))

        # Normalise by max of first rep so all reps share the same scale
        scale = rss_reps[0].max() + 1e-8
        rss_reps = [r / scale for r in rss_reps]

        # Target: average RSS across all available repetitions
        target = np.mean(rss_reps, axis=0)  # [H, W]

        # Single-coil k-space per rep: F{ rss_rep }
        kspace_list = []
        for rss in rss_reps:
            ks = _fft2c(rss)  # [H, W] complex
            ks_r = torch.from_numpy(ks.real.astype(np.float32))
            ks_i = torch.from_numpy(ks.imag.astype(np.float32))
            kspace_list.append(torch.stack([ks_r, ks_i], dim=0))  # [2, H, W]

        kspace_t = torch.stack(kspace_list, dim=0)   # [reps, 2, H, W]
        target_t = torch.from_numpy(target).unsqueeze(0)  # [1, H, W]
        return kspace_t, target_t


# ──────────────────────────── public API ───────────────────────────────────

def get_m4raw_splits(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Return (train, val, test) M4RawSliceDataset from config dict."""
    m = cfg.get("m4raw", {})
    root       = m.get("root", "data/m4raw")
    contrast   = m.get("contrast", "T2w")
    image_size = tuple(m.get("image_size", [256, 256]))
    max_reps   = m.get("max_reps", 3)
    num_slices = m.get("num_slices", 12)
    train_frac = m.get("train_frac", 0.70)
    val_frac   = m.get("val_frac",   0.15)
    seed       = cfg.get("seed", 42)

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"M4Raw root '{root}' not found.\n"
            "Set m4raw.root in configs/data_m4raw.yaml to the directory containing .h5 files."
        )

    groups = _group_files(root, contrast)
    if not groups:
        raise FileNotFoundError(
            f"No M4Raw .h5 files for contrast '{contrast}' found in '{root}'."
        )

    train_g, val_g, test_g = _split_by_subject(groups, train_frac, val_frac, seed)

    kwargs = dict(image_size=image_size, max_reps=max_reps, num_slices=num_slices)
    return (
        M4RawSliceDataset(train_g, **kwargs),
        M4RawSliceDataset(val_g,   **kwargs),
        M4RawSliceDataset(test_g,  **kwargs),
    )
