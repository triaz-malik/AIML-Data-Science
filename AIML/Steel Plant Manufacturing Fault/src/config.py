"""Central configuration for the Steel Defect Detection project.

All paths are resolved relative to the project root so the code works the same
whether run from a notebook in ``notebooks/`` or a script anywhere in the repo.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "train.csv"
SAMPLE_SUBMISSION = PROJECT_ROOT / "sample_submission.csv"
TRAIN_IMG_DIR = PROJECT_ROOT / "train_images"
TEST_IMG_DIR = PROJECT_ROOT / "test_images"

EDA_DIR = PROJECT_ROOT / "eda"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRADCAM_DIR = PROJECT_ROOT / "gradcam"
SEG_DIR = PROJECT_ROOT / "segmentation"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# --- Dataset constants ------------------------------------------------------
# Severstal images are 1600 (W) x 256 (H), RGB.
IMG_HEIGHT = 256
IMG_WIDTH = 1600
IMG_CHANNELS = 3

# Defect classes. 0 is the implicit "no defect" class used for classification.
DEFECT_CLASSES = [1, 2, 3, 4]
NUM_DEFECT_CLASSES = len(DEFECT_CLASSES)          # 4 (segmentation channels)
NUM_CLF_CLASSES = NUM_DEFECT_CLASSES + 1          # 5 (incl. NoDefect) for classification

CLASS_NAMES = {
    0: "NoDefect",
    1: "Class1",
    2: "Class2",
    3: "Class3",
    4: "Class4",
}

# Distinct colors for overlaying the 4 defect masks (RGB, 0-255).
CLASS_COLORS = {
    1: (255, 0, 0),     # red
    2: (0, 255, 0),     # green
    3: (0, 0, 255),     # blue
    4: (255, 255, 0),   # yellow
}

# --- Training defaults (overridable per phase / by Optuna) ------------------
SEED = 42
DEVICE = "cuda"  # falls back to cpu in utils.get_device()

# ImageNet normalization stats (for transfer-learning backbones).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Severity thresholds on defect-area percentage of the sheet.
SEVERITY_THRESHOLDS = {
    "Minor": (0.0, 2.0),       # accept
    "Moderate": (2.0, 5.0),    # rework
    "Critical": (5.0, 100.0),  # reject
}
SEVERITY_DECISION = {
    "Minor": "Accept",
    "Moderate": "Rework",
    "Critical": "Reject",
}


def ensure_dirs() -> None:
    """Create all output directories if they do not yet exist."""
    for d in (EDA_DIR, MODELS_DIR, REPORTS_DIR, GRADCAM_DIR, SEG_DIR, DASHBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)
