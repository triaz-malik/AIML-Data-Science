"""Data pipeline for multi-label classification.

Target is a 4-dim multi-hot vector ``[c1, c2, c3, c4]``; an all-zero target means
*no defect*. We reuse ``src.dataset.build_image_df`` for the per-image labels.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src import config, utils
from src.dataset import build_image_df, load_annotations

LABEL_COLS = ["c1", "c2", "c3", "c4"]


def make_splits(val_size: float = 0.15, seed: int = config.SEED):
    """Return (train_df, val_df, ann) stratified on the single-label `label` col."""
    ann = load_annotations()
    img_df = build_image_df(ann)
    train_df, val_df = train_test_split(
        img_df, test_size=val_size, random_state=seed, stratify=img_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), ann


def build_transforms(train: bool):
    """torchvision v2 transform applied to a uint8 CHW tensor."""
    tfs = [v2.ToDtype(torch.float32, scale=True)]
    if train:
        tfs += [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    tfs.append(v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD))
    return v2.Compose(tfs)


def pos_weights(train_df) -> torch.Tensor:
    """Per-class BCE pos_weight = (#neg / #pos) to counter class imbalance."""
    pos = train_df[LABEL_COLS].sum().values.astype(np.float32)
    neg = len(train_df) - pos
    return torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)


class ClsDataset(Dataset):
    def __init__(self, df, img_dir=config.TRAIN_IMG_DIR, img_size=(256, 512),
                 transform=None, clahe=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.h, self.w = img_size
        self.transform = transform
        self.clahe = clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.img_dir / row["ImageId"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.clahe:
            img = utils.apply_clahe(img)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 CHW
        if self.transform is not None:
            t = self.transform(t)
        target = torch.tensor(row[LABEL_COLS].values.astype(np.float32))
        return t, target
