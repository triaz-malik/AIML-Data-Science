"""Shared helpers: reproducibility, device selection, image listing."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Keep cudnn deterministic-ish without killing throughput.
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_device():
    """Return the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_images(folder: Path) -> list[Path]:
    """All image files directly inside `folder` (non-recursive)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def count_params(model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
