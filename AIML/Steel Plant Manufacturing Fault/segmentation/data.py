"""Segmentation data pipeline: image + 4-channel mask, resized consistently.

We keep augmentation mask-aware (flips applied to both image and mask) and avoid
albumentations to stay dependency-light.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src import config
from src.dataset import build_image_df, load_annotations
from src.rle import build_multiclass_mask


def make_splits(val_size: float = 0.15, seed: int = config.SEED, defective_only: bool = False):
    """(train_df, val_df, ann). Optionally keep only defective images for seg training."""
    ann = load_annotations()
    img_df = build_image_df(ann)
    if defective_only:
        img_df = img_df[img_df.has_defect == 1].reset_index(drop=True)
    train_df, val_df = train_test_split(
        img_df, test_size=val_size, random_state=seed, stratify=img_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), ann


class SegDataset(Dataset):
    def __init__(self, df, ann, img_dir=config.TRAIN_IMG_DIR, img_size=(256, 512), train=False):
        self.df = df.reset_index(drop=True)
        self.ann = ann
        self.img_dir = img_dir
        self.h, self.w = img_size
        self.train = train
        self.mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(config.IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / row["ImageId"])), cv2.COLOR_BGR2RGB)
        mask = build_multiclass_mask(row["ImageId"], self.ann)          # (H, W, 4) uint8

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 2:  # cv2 collapses singleton channel dims defensively
            mask = mask[:, :, None]

        if self.train:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[:, ::-1, :].copy()   # h-flip
            if np.random.rand() < 0.5:
                img, mask = img[::-1, :, :].copy(), mask[::-1, :, :].copy()   # v-flip

        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(mask).permute(2, 0, 1).contiguous().float()
        return img_t, mask_t
