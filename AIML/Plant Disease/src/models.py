"""Model zoo: baseline CNN, ResNet50, EfficientNetB0.

All builders return an ``nn.Module`` whose final layer outputs ``n_classes``
logits, so the training loop is identical across architectures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------------------------------- #
# Phase 5 — baseline CNN built from scratch
# --------------------------------------------------------------------------- #
class BaselineCNN(nn.Module):
    """Conv-Pool x3 -> Dense -> Softmax (logits)."""

    def __init__(self, n_classes: int, dropout: float = 0.4, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Phase 6 — transfer-learning backbones
# --------------------------------------------------------------------------- #
def _resnet50(n_classes: int, dropout: float = 0.3, pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_classes))
    return m


def _efficientnet_b0(n_classes: int, dropout: float = 0.3, pretrained: bool = True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_classes))
    return m


BUILDERS = {
    "baseline_cnn": BaselineCNN,
    "resnet50": _resnet50,
    "efficientnet_b0": _efficientnet_b0,
}


def build_model(name: str, n_classes: int, dropout: float | None = None, **kw):
    name = name.lower()
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(BUILDERS)}")
    if name == "baseline_cnn":
        return BaselineCNN(n_classes, dropout=dropout if dropout is not None else 0.4)
    return BUILDERS[name](n_classes, dropout=dropout if dropout is not None else 0.3, **kw)


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Return the last conv layer for Grad-CAM, per architecture."""
    name = name.lower()
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    if name == "baseline_cnn":
        return model.features[-3]  # last Conv2d block
    raise ValueError(name)
