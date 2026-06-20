"""Central configuration for the Pneumonia Detection project.

All paths are resolved relative to the project root (the parent of this file's
`src/` directory) so the scripts work regardless of where you launch them from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT                      # contains train/ val/ test/
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"             # EDA + result plots
MANIFEST_DIR = OUTPUT_DIR / "manifests"      # CSV train/val/test splits
CKPT_DIR = OUTPUT_DIR / "checkpoints"        # saved model weights
REPORT_DIR = OUTPUT_DIR / "reports"          # metrics, Grad-CAM, json

for _d in (OUTPUT_DIR, FIG_DIR, MANIFEST_DIR, CKPT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
# Index ordering matters: PNEUMONIA is the positive class (index 1) because the
# hospital KPI is recall on pneumonia (minimise false negatives).
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
POSITIVE_CLASS = "PNEUMONIA"
POSITIVE_IDX = CLASS_TO_IDX[POSITIVE_CLASS]

# --------------------------------------------------------------------------- #
# Image / training hyperparameters
# --------------------------------------------------------------------------- #
IMG_SIZE = 224                               # 224 for ResNet/EfficientNetB0
# ImageNet normalisation (used by pretrained backbones). X-rays are grayscale
# but we replicate to 3 channels so we can leverage ImageNet pretrained weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42
VAL_FRACTION = 0.15                          # carved out of (train + tiny val)


@dataclass
class TrainConfig:
    """Default training hyperparameters (override on the CLI)."""
    model: str = "efficientnet_b0"           # custom_cnn | resnet50 | efficientnet_b0
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"                  # adam | rmsprop | sgd
    dropout: float = 0.3
    img_size: int = IMG_SIZE
    num_workers: int = 4
    # Two-phase transfer learning: train head first, then unfreeze backbone.
    freeze_epochs: int = 3                    # epochs with frozen backbone
    finetune_lr: float = 1e-4                # lr after unfreezing
    early_stop_patience: int = 6             # epochs w/o val-recall improvement
    use_class_weights: bool = True           # weight loss by inverse frequency
    label_smoothing: float = 0.0
    monitor: str = "recall"                  # checkpoint selection metric

    extra: dict = field(default_factory=dict)
