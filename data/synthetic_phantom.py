"""Synthetic phantom dataset for smoke-testing the full pipeline with no downloads."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def _shepp_logan_phantom(size: int) -> np.ndarray:
    """Generate a normalised Shepp-Logan phantom of given square size."""
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    img = shepp_logan_phantom()
    img = resize(img, (size, size), anti_aliasing=True)
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)


def _circles_phantom(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random filled circles on black background."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = np.meshgrid(np.arange(size), np.arange(size))
    n_circles = rng.integers(3, 8)
    for _ in range(n_circles):
        r = rng.integers(5, size // 4)
        x0 = rng.integers(r, size - r)
        y0 = rng.integers(r, size - r)
        val = float(rng.uniform(0.3, 1.0))
        mask = (cx - x0) ** 2 + (cy - y0) ** 2 < r ** 2
        img[mask] = val
    return img


class SyntheticPhantomDataset(Dataset):
    """
    Synthetic phantom dataset that returns ground-truth images.

    Each sample is a complex-valued image x ∈ C^{H×W} represented as a
    real-valued tensor of shape [2, H, W] (real, imag channels).
    The imaginary part is zero for phantom images (magnitude-only).
    """

    def __init__(
        self,
        size: int = 128,
        num_samples: int = 800,
        phantom_type: str = "shepp_logan",
        seed: int = 42,
    ):
        self.size = size
        self.num_samples = num_samples
        self.phantom_type = phantom_type
        self.seed = seed
        self._images = self._build(seed)

    def _build(self, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        imgs = []
        base = _shepp_logan_phantom(self.size)
        for i in range(self.num_samples):
            if self.phantom_type == "shepp_logan":
                img = base + 0.02 * rng.standard_normal(base.shape).astype(np.float32)
                img = np.clip(img, 0, 1)
            elif self.phantom_type == "circles":
                img = _circles_phantom(self.size, rng)
            else:  # mixed
                img = base if i % 2 == 0 else _circles_phantom(self.size, rng)
                img = np.clip(img, 0, 1).astype(np.float32)
            imgs.append(img)
        arr = np.stack(imgs, axis=0)  # [N, H, W]
        # Return as [N, 2, H, W]: real and zero imaginary
        real = torch.from_numpy(arr).unsqueeze(1)
        imag = torch.zeros_like(real)
        return torch.cat([real, imag], dim=1)  # [N, 2, H, W]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._images[idx]  # [2, H, W]
        return img, img  # (measurement placeholder, target)


def get_phantom_splits(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Return train/val/test phantom datasets from config."""
    s = cfg.get("synthetic", {})
    size = s.get("image_size", [128, 128])[0]
    phantom_type = s.get("phantom_type", "shepp_logan")
    seed = cfg.get("seed", 42)
    n_train = s.get("num_train", 800)
    n_val = s.get("num_val", 100)
    n_test = s.get("num_test", 100)
    train_ds = SyntheticPhantomDataset(size, n_train, phantom_type, seed=seed)
    val_ds = SyntheticPhantomDataset(size, n_val, phantom_type, seed=seed + 1)
    test_ds = SyntheticPhantomDataset(size, n_test, phantom_type, seed=seed + 2)
    return train_ds, val_ds, test_ds
