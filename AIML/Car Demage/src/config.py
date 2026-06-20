"""Central configuration for the Vehicle Damage Detection project."""
from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "training"
VAL_DIR = ROOT / "validation"          # used as the held-out TEST set
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GRADCAM_DIR = ROOT / "gradcam"

for _d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, GRADCAM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes  (folder name -> index).  Index 1 == "damage" (the positive class).
# Folders: 00-damage, 01-whole.  We want "damage" to be the positive class for
# precision/recall/ROC, so we map damage -> 1, whole -> 0.
# ---------------------------------------------------------------------------
CLASS_TO_IDX = {"01-whole": 0, "00-damage": 1}
IDX_TO_CLASS = {0: "whole", 1: "damage"}
CLASS_NAMES = ["whole", "damage"]      # index order
POSITIVE_CLASS = "damage"

# ---------------------------------------------------------------------------
# Image / training hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Make runs reproducible."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
