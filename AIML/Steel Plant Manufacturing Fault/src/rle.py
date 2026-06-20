"""Run-length encoding (RLE) utilities for Severstal masks.

Severstal uses **column-major (Fortran) order**, 1-indexed pixel positions.
Each ``EncodedPixels`` string is a sequence of ``start length`` pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def rle_decode(rle: str | float, shape: tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH)) -> np.ndarray:
    """Decode a single RLE string into a binary ``(H, W)`` uint8 mask.

    A NaN / empty value (no defect) returns an all-zero mask.
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, float) or rle is None or (isinstance(rle, str) and rle.strip() == ""):
        return mask.reshape(shape, order="F")

    parts = np.asarray(rle.split(), dtype=int)
    starts, lengths = parts[0::2] - 1, parts[1::2]  # 1-indexed -> 0-indexed
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary ``(H, W)`` mask back into a Severstal RLE string."""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


def build_multiclass_mask(
    image_id: str,
    df: pd.DataFrame,
    shape: tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
) -> np.ndarray:
    """Build a ``(H, W, 4)`` one-hot mask for the 4 defect classes of one image.

    ``df`` must have columns ``ImageId``, ``ClassId``, ``EncodedPixels``.
    """
    mask = np.zeros((shape[0], shape[1], config.NUM_DEFECT_CLASSES), dtype=np.uint8)
    rows = df[df["ImageId"] == image_id]
    for _, row in rows.iterrows():
        cls = int(row["ClassId"])
        mask[:, :, cls - 1] = rle_decode(row["EncodedPixels"], shape)
    return mask


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Render a ``(H, W, 4)`` one-hot mask as an ``(H, W, 3)`` RGB overlay."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in config.DEFECT_CLASSES:
        out[mask[:, :, cls - 1] == 1] = config.CLASS_COLORS[cls]
    return out
