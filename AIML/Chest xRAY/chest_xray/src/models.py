"""Model zoo: custom CNN baseline + ResNet50 / EfficientNetB0 transfer learning.

All models output 2 logits (NORMAL, PNEUMONIA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from . import config as C

NUM_CLASSES = len(C.CLASSES)


# --------------------------------------------------------------------------- #
# Model 1 — custom CNN baseline (Conv/Pool x3 -> FC), per project spec.
# --------------------------------------------------------------------------- #
class CustomCNN(nn.Module):
    def __init__(self, dropout: float = 0.3, img_size: int = C.IMG_SIZE):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128),
            nn.AdaptiveAvgPool2d(1),       # -> (N,128,1,1), size-agnostic
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# Transfer-learning backbones
# --------------------------------------------------------------------------- #
def _build_resnet50(dropout: float, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    in_feat = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    return net


def _build_efficientnet_b0(dropout: float, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_feat = net.classifier[1].in_features
    net.classifier = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(in_feat, NUM_CLASSES))
    return net


def build_model(name: str, dropout: float = 0.3,
                pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in ("custom_cnn", "cnn", "custom"):
        return CustomCNN(dropout=dropout)
    if name == "resnet50":
        return _build_resnet50(dropout, pretrained)
    if name in ("efficientnet_b0", "efficientnet", "effnet"):
        return _build_efficientnet_b0(dropout, pretrained)
    raise ValueError(f"Unknown model '{name}'. "
                     "Use: custom_cnn | resnet50 | efficientnet_b0")


# --------------------------------------------------------------------------- #
# Freeze / unfreeze helpers for two-phase transfer learning
# --------------------------------------------------------------------------- #
def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Freeze/unfreeze the feature extractor, always leaving the head trainable."""
    name = name.lower()
    if name == "resnet50":
        head_params = set(model.fc.parameters())
    elif name in ("efficientnet_b0", "efficientnet", "effnet"):
        head_params = set(model.classifier.parameters())
    else:  # custom CNN — nothing pretrained to freeze
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_params:                 # head is always trainable
        p.requires_grad = True


def target_layer_for_gradcam(model: nn.Module, name: str):
    """Return the conv layer Grad-CAM should hook (last spatial feature map)."""
    name = name.lower()
    if name == "resnet50":
        return model.layer4[-1]
    if name in ("efficientnet_b0", "efficientnet", "effnet"):
        return model.features[-1]
    return model.features[2]              # custom CNN: last conv block
