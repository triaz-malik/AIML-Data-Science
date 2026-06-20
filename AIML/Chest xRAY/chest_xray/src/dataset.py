"""Datasets, augmentation, and dataloaders (manifest-driven).

All splits are read from outputs/manifests/*.csv produced by data_prep.py.
Run `python -m src.data_prep` first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config as C


# --------------------------------------------------------------------------- #
# Transforms  (augmentation per project spec: rotation, zoom, flip, brightness,
# small shifts). Validation/test use a deterministic resize only.
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # X-ray -> 3ch for pretrained
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=15,                 # rotation +/-15 deg
                translate=(0.05, 0.05),     # small shifts
                scale=(0.9, 1.1),           # zoom ~0.1
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15),  # brightness 0.1-0.2
            transforms.ToTensor(),
            transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, manifest_csv: Path, img_size: int, train: bool):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.tf = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            img = self.tf(im.convert("RGB"))
        label = int(row["label_idx"])
        return img, label


def compute_class_weights(manifest_csv: Path) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss, normalised to mean 1."""
    df = pd.read_csv(manifest_csv)
    counts = df["label_idx"].value_counts().sort_index()
    freqs = counts.values.astype(np.float64)
    weights = freqs.sum() / (len(freqs) * freqs)   # inverse frequency
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(cfg) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from manifests."""
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=True, persistent_workers=cfg.num_workers > 0)
    loaders = {}
    for split, is_train in [("train", True), ("val", False), ("test", False)]:
        ds = ChestXrayDataset(C.MANIFEST_DIR / f"{split}.csv",
                              cfg.img_size, train=is_train)
        loaders[split] = DataLoader(ds, shuffle=is_train, drop_last=is_train,
                                    **common)
    return loaders
