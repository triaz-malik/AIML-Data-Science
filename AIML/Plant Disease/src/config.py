"""Central configuration for the Plant Disease project.

Everything path-, model-, and training-related lives here so the rest of the
code stays declarative. Paths are resolved relative to the project root so the
scripts work regardless of the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"          # train/<Crop>/<Disease>/*.jpg
TEST_DIR = ROOT / "test"            # loose, unlabeled inference images

OUTPUTS = ROOT / "outputs"
PLOTS_DIR = OUTPUTS / "plots"
MODELS_DIR = OUTPUTS / "models"
REPORTS_DIR = OUTPUTS / "reports"
SPLITS_DIR = OUTPUTS / "splits"

for _d in (OUTPUTS, PLOTS_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
IMG_SIZE = 224                       # pretrained backbones expect 224x224
# ImageNet normalisation (used by torchvision pretrained weights)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Split fractions (Phase 4). Splitting is GROUP-AWARE on the source-image GUID
# so augmented copies of the same leaf never straddle train/val/test.
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
# Workers are safe here because every entry point runs under `if __name__ ==
# "__main__"`, so Windows spawn can re-import cleanly. Override with PD_WORKERS.
NUM_WORKERS = int(os.environ.get("PD_WORKERS", 6))
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = SEED) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
