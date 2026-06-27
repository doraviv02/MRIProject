"""Data loader factory: returns (train, val, test) DataLoaders from config."""

import torch
from torch.utils.data import DataLoader
from typing import Tuple


def get_data_loaders(cfg_data: dict, cfg_train: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders based on cfg_data['dataset']."""
    dataset_name = cfg_data.get("dataset", "synthetic")
    bs = cfg_train.get("batch_size", 4)
    nw = cfg_train.get("num_workers", 4)
    pin = cfg_train.get("pin_memory", True)

    if dataset_name == "synthetic":
        from data.synthetic_phantom import get_phantom_splits
        train_ds, val_ds, test_ds = get_phantom_splits(cfg_data)
    elif dataset_name == "m4raw":
        from data.m4raw import get_m4raw_splits
        train_ds, val_ds, test_ds = get_m4raw_splits(cfg_data)
    elif dataset_name == "fastmri":
        from data.fastmri import get_fastmri_splits
        train_ds, val_ds, test_ds = get_fastmri_splits(cfg_data)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, test_loader
