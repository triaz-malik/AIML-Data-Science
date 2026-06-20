"""Phases 4-6 - Model architectures: baseline CNN, ResNet50, EfficientNetB0."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Phase 4 — Baseline custom CNN
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Simple Conv→Pool×3 + Dense classifier. Establishes a benchmark."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 224 -> 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 112 -> 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 56 -> 28
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                    # -> 128 x 1 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Phases 5-6 — Transfer-learning models
# ---------------------------------------------------------------------------
def build_resnet50(num_classes: int = 2, dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    in_f = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return net


def build_efficientnet_b0(num_classes: int = 2, dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_f = net.classifier[1].in_features
    net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return net


def build_model(name: str, num_classes: int = 2, dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in ("cnn", "baseline", "baseline_cnn"):
        return BaselineCNN(num_classes, dropout=max(dropout, 0.5))
    if name in ("resnet50", "resnet"):
        return build_resnet50(num_classes, dropout, pretrained)
    if name in ("efficientnet", "efficientnet_b0", "effnet", "efficientnetb0"):
        return build_efficientnet_b0(num_classes, dropout, pretrained)
    raise ValueError(f"Unknown model name: {name!r}")


def gradcam_target_layer(model: nn.Module, name: str):
    """Return the conv layer to hook for Grad-CAM (Phase 9)."""
    name = name.lower()
    if isinstance(model, BaselineCNN):
        return model.features[-2]                 # last Conv before AdaptiveAvgPool
    if name.startswith("resnet"):
        return model.layer4[-1]
    if "efficient" in name or "effnet" in name:
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer defined for {name!r}")


MODEL_NAMES = ["cnn", "resnet50", "efficientnet_b0"]
