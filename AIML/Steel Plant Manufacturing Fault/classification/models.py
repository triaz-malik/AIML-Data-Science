"""Model factory: Custom CNN, ResNet50, EfficientNetB0 with multi-label heads."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """A compact 4-block VGG-style CNN baseline (no pretraining)."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(name: str, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes, dropout)

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m

    raise ValueError(f"Unknown model '{name}'")
