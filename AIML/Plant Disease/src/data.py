"""Dataset scanning, leak-free splitting, augmentation and DataLoaders.

Key design point
----------------
The training images are partly *pre-augmented*: a single source leaf appears
as several files that share a GUID prefix before ``___`` (e.g.
``<guid>___FREC_Scab 3003.JPG`` and ``<guid>___FREC_Scab 3003_270deg.JPG``).
If we split files randomly, rotated/flipped copies of the same leaf leak across
train/val/test and accuracy is wildly optimistic. We therefore split on the
GUID *group*, stratified by class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config as C


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _guid(filename: str) -> str:
    """Group key = the source-leaf id (everything before the first ``___``)."""
    return filename.split("___")[0]


def scan_dataset(train_dir: Path = C.TRAIN_DIR) -> pd.DataFrame:
    """Walk train/<Crop>/<Disease>/* into a tidy DataFrame."""
    rows = []
    for crop_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        crop = crop_dir.name
        for disease_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            disease = disease_dir.name
            for f in disease_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    rows.append(
                        {
                            "path": str(f),
                            "crop": crop,
                            "disease": disease,
                            "label": f"{crop}___{disease}".replace(" ", "_"),
                            "is_healthy": disease.strip().lower() == "healthy",
                            "guid": _guid(f.name),
                        }
                    )
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No images found under {train_dir}")
    return df


def make_split(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Add a ``split`` column (train/val/test), group-aware + stratified.

    Two nested StratifiedGroupKFold passes: first carve out test, then carve
    val out of the remainder. Groups (GUIDs) never cross a split boundary.
    """
    df = df.copy()
    labels = df["label"].to_numpy()
    groups = df["guid"].to_numpy()

    # ---- pass 1: hold out TEST_FRAC ----
    n_splits_test = max(2, round(1 / C.TEST_FRAC))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf.split(df, labels, groups))

    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    # ---- pass 2: carve VAL out of the train_val remainder ----
    tv = df.iloc[train_val_idx]
    rel_val = C.VAL_FRAC / (1 - C.TEST_FRAC)
    n_splits_val = max(2, round(1 / rel_val))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    tr_local, val_local = next(sgkf2.split(tv, tv["label"], tv["guid"]))
    val_global = tv.index.to_numpy()[val_local]
    df.loc[val_global, "split"] = "val"

    # sanity: zero GUID overlap between splits
    g = {s: set(df.loc[df.split == s, "guid"]) for s in ("train", "val", "test")}
    assert not (g["train"] & g["val"]), "GUID leak train/val"
    assert not (g["train"] & g["test"]), "GUID leak train/test"
    assert not (g["val"] & g["test"]), "GUID leak val/test"
    return df


# --------------------------------------------------------------------------- #
# Label encoding
# --------------------------------------------------------------------------- #
def build_label_encoder(df: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(sorted(df["label"].unique()))
    return le


# --------------------------------------------------------------------------- #
# Transforms (Phase 3 — on-the-fly augmentation)
# --------------------------------------------------------------------------- #
def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(25),
            transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(C.MEAN, C.STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(C.MEAN, C.STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Dataset / loaders
# --------------------------------------------------------------------------- #
@dataclass
class PlantDataset(Dataset):
    df: pd.DataFrame
    le: object
    transform: transforms.Compose

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)
        y = int(self.le.transform([row["label"]])[0])
        return x, y


def make_loaders(df: pd.DataFrame, le, batch_size: int = C.BATCH_SIZE):
    parts = {}
    for split, tf in (
        ("train", train_transform()),
        ("val", eval_transform()),
        ("test", eval_transform()),
    ):
        sub = df[df.split == split].reset_index(drop=True)
        ds = PlantDataset(sub, le, tf)
        parts[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=C.NUM_WORKERS,
            pin_memory=(C.DEVICE == "cuda"),
            drop_last=(split == "train"),
            persistent_workers=(C.NUM_WORKERS > 0),
            prefetch_factor=(4 if C.NUM_WORKERS > 0 else None),
        )
    return parts["train"], parts["val"], parts["test"]


def class_weights(df: pd.DataFrame, le) -> np.ndarray:
    """Inverse-frequency weights on the TRAIN split for imbalance handling."""
    tr = df[df.split == "train"]
    counts = tr["label"].map(lambda l: l).value_counts()
    classes = le.classes_
    freq = np.array([counts.get(c, 0) for c in classes], dtype=np.float64)
    freq[freq == 0] = 1
    w = freq.sum() / (len(classes) * freq)
    return w.astype(np.float32)
