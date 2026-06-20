"""Shared helpers: reproducibility, device, image enhancement, metrics."""
from __future__ import annotations

import os
import random

import cv2
import numpy as np

from . import config


def seed_everything(seed: int = config.SEED) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device, falling back to CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Phase 2: image enhancement --------------------------------------------
def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def enhance(img: np.ndarray, clahe: bool = True, blur: bool = False) -> np.ndarray:
    """Standard enhancement pipeline: optional CLAHE then optional Gaussian blur."""
    out = apply_clahe(img) if clahe else img
    if blur:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


# --- Metrics ----------------------------------------------------------------
def dice_coef(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Dice coefficient between two binary masks."""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Intersection-over-Union between two binary masks."""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))


# --- Phase 7: severity ------------------------------------------------------
def defect_area_pct(mask: np.ndarray) -> float:
    """Percentage of pixels flagged as any defect in a ``(H, W)`` or ``(H, W, C)`` mask."""
    flat = mask.reshape(mask.shape[0], mask.shape[1], -1).max(axis=-1) if mask.ndim == 3 else mask
    return float((flat > 0).sum() / flat.size * 100.0)


def severity_from_area(area_pct: float) -> str:
    """Map a defect-area percentage to a severity label."""
    for label, (lo, hi) in config.SEVERITY_THRESHOLDS.items():
        if lo <= area_pct < hi:
            return label
    return "Critical"


def decision_from_severity(severity: str) -> str:
    """Map a severity label to a quality decision (Accept/Rework/Reject)."""
    return config.SEVERITY_DECISION.get(severity, "Reject")
