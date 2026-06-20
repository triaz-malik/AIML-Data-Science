"""PyTorch datasets and dataframe prep for classification and segmentation.

``build_image_df`` collapses the per-defect ``train.csv`` into one row per image
with: list of defect classes, multi-label one-hot, and a single classification
label (0 = no defect, else the dominant/first defect class).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .rle import build_multiclass_mask


def load_annotations(csv_path=config.TRAIN_CSV) -> pd.DataFrame:
    """Load the raw per-defect annotation csv."""
    df = pd.read_csv(csv_path)
    df["ClassId"] = df["ClassId"].astype(int)
    return df


def build_image_df(ann: pd.DataFrame, img_dir=config.TRAIN_IMG_DIR) -> pd.DataFrame:
    """One row per image in ``img_dir`` with defect labels joined from ``ann``.

    Columns: ImageId, defects (list[int]), has_defect (0/1),
    label (int 0-4, multi-label uses first class), c1..c4 (one-hot), num_defects.
    """
    import os

    all_images = sorted(os.listdir(img_dir))
    grouped = ann.groupby("ImageId")["ClassId"].apply(list).to_dict()

    rows = []
    for img in all_images:
        defects = sorted(grouped.get(img, []))
        onehot = [1 if c in defects else 0 for c in config.DEFECT_CLASSES]
        rows.append({
            "ImageId": img,
            "defects": defects,
            "has_defect": int(len(defects) > 0),
            "label": defects[0] if defects else 0,
            "c1": onehot[0], "c2": onehot[1], "c3": onehot[2], "c4": onehot[3],
            "num_defects": len(defects),
        })
    return pd.DataFrame(rows)


def _read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SteelClassificationDataset(Dataset):
    """Yields ``(image_tensor, label)`` for 5-class classification."""

    def __init__(self, df: pd.DataFrame, img_dir=config.TRAIN_IMG_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.img_dir / row["ImageId"])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, int(row["label"])


class SteelSegmentationDataset(Dataset):
    """Yields ``(image_tensor, mask_tensor)`` with a ``(4, H, W)`` mask."""

    def __init__(self, df: pd.DataFrame, ann: pd.DataFrame, img_dir=config.TRAIN_IMG_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.ann = ann
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.img_dir / row["ImageId"])
        mask = build_multiclass_mask(row["ImageId"], self.ann)  # (H, W, 4)
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask)
            mask = mask.permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return img, mask
