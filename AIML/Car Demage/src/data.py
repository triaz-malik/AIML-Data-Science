"""Datasets, transforms and dataloaders for Vehicle Damage Detection."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_WORKERS,
    TRAIN_DIR,
    VAL_DIR,
)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_images(root: Path) -> list[tuple[Path, int]]:
    """Return [(path, label_idx), ...] for every readable image under `root`.

    Labels follow CLASS_TO_IDX (damage=1, whole=0).
    """
    samples: list[tuple[Path, int]] = []
    for class_name, idx in CLASS_TO_IDX.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTENSIONS:
                samples.append((p, idx))
    return samples


class CarDamageDataset(Dataset):
    """Image-folder dataset that is robust to occasional corrupt files."""

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms (Phase 3 — Feature Engineering / Augmentation)
# ---------------------------------------------------------------------------
def build_transforms(img_size: int = IMG_SIZE, train: bool = True):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_loaders(
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train loader (from training/) and test loader (from validation/)."""
    train_samples = list_images(TRAIN_DIR)
    test_samples = list_images(VAL_DIR)

    train_ds = CarDamageDataset(train_samples, build_transforms(img_size, train=augment))
    test_ds = CarDamageDataset(test_samples, build_transforms(img_size, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization. Accepts CxHxW tensor."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1)
