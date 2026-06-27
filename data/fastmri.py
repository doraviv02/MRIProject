"""FastMRI dataset loader for high-SNR control experiments.

Uses the same fastMRI-format .h5 files as M4Raw.  Adds Gaussian noise in k-space
to simulate varying SNR levels and verify the physics sanity check (E7):
  at high SNR the learned policy should collapse toward full coverage at NEX=1.

Download: fastmri.med.nyu.edu (requires registration).
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


class FastMRIDataset(Dataset):
    """
    Single-coil RSS fastMRI slices.

    Returns (image [2, H, W], target [1, H, W]) — the first element is the clean
    real/imag IMAGE (not k-space), matching the synthetic / M4Raw-post-IFFT
    convention; the training pipeline applies the forward FFT itself inside
    simulate_measurement. ``added_noise_sigma`` optionally degrades the *input*
    image to emulate lower SNR while the target stays clean.
    """

    def __init__(
        self,
        h5_files: List[str],
        image_size: Tuple[int, int] = (320, 320),
        added_noise_sigma: float = 0.0,
        contrast_key: Optional[str] = None,
        seed: int = 42,
        rotate_k: int = 2,
    ):
        self.image_size = image_size
        self.added_noise_sigma = added_noise_sigma
        self.contrast_key = contrast_key
        self.rotate_k = int(rotate_k) % 4   # 90°-CCW rotations to upright fastMRI orientation
        self.rng = np.random.default_rng(seed)
        self._index: List[Tuple[str, int]] = []
        for f in h5_files:
            with h5py.File(f, "r") as hf:
                n_slices = hf["kspace"].shape[0]
                for sl in range(n_slices):
                    self._index.append((f, sl))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fpath, sl = self._index[idx]
        with h5py.File(fpath, "r") as hf:
            kspace = hf["kspace"][sl]  # [coils, H, W] or [H, W]

        kspace = kspace.astype(np.complex64)
        if kspace.ndim == 2:
            kspace = kspace[np.newaxis]  # [1, H, W]

        H, W = self.image_size
        cy, cx = kspace.shape[-2] // 2, kspace.shape[-1] // 2
        kspace = kspace[..., cy - H // 2: cy + H // 2, cx - W // 2: cx + W // 2]

        # Clean RSS magnitude image — the ground-truth signal the pipeline measures.
        images = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1))
        rss = np.sqrt((np.abs(images) ** 2).sum(axis=0))  # [H, W]
        rss = (rss / (rss.max() + 1e-8)).astype(np.float32)

        # fastMRI reconstructions come out rotated; rotate to upright so all downstream
        # figures/comparisons are correctly oriented. Applied once to the RSS so the input
        # image and target stay consistent. Adjust rotate_k (×90° CCW) if the data differ.
        if self.rotate_k:
            rss = np.ascontiguousarray(np.rot90(rss, self.rotate_k))

        # Return the IMAGE (real/imag), not its FFT: the training pipeline treats a
        # 4-D sample as an image and applies the forward FFT in simulate_measurement.
        img_real = rss.copy()
        img_imag = np.zeros_like(rss)

        # Optional extra input degradation to emulate lower SNR; target stays clean.
        # 1/sqrt(2) normalises the complex noise to E|eta|^2 = 1, matching
        # simulate_measurement / loupe_measurement. Leave at 0 to let the pipeline's
        # measurement model be the sole (calibrated) noise source.
        if self.added_noise_sigma > 0:
            s = self.added_noise_sigma * 0.7071067811865476
            img_real = img_real + (s * self.rng.standard_normal(rss.shape)).astype(np.float32)
            img_imag = img_imag + (s * self.rng.standard_normal(rss.shape)).astype(np.float32)

        image_t = torch.stack([
            torch.from_numpy(img_real),
            torch.from_numpy(img_imag),
        ], dim=0)  # [2, H, W] real/imag IMAGE

        target_t = torch.from_numpy(rss).unsqueeze(0)  # [1, H, W] clean magnitude
        return image_t, target_t


def get_fastmri_splits(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Return train/val/test FastMRI datasets from config."""
    fm = cfg.get("fastmri", {})
    root = fm.get("root", "data/fastmri")
    image_size = tuple(fm.get("image_size", [320, 320]))
    sigma = fm.get("added_noise_sigma", 0.05)
    rotate_k = fm.get("rotate_k", 2)   # ×90° CCW to upright fastMRI orientation
    seed = cfg.get("seed", 42)

    def _files(split: str) -> List[str]:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            return []
        return sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".h5")])

    train_f, val_f, test_f = _files("train"), _files("val"), _files("test")
    if not train_f:
        raise FileNotFoundError(
            f"No fastMRI .h5 files found at {root}. "
            "Register and download from fastmri.med.nyu.edu."
        )
    kwargs = dict(image_size=image_size, added_noise_sigma=sigma, seed=seed, rotate_k=rotate_k)
    return (
        FastMRIDataset(train_f, **kwargs),
        FastMRIDataset(val_f, **kwargs),
        FastMRIDataset(test_f, **kwargs),
    )
