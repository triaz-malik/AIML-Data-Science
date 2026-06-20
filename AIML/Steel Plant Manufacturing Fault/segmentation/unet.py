"""A clean, dependency-free U-Net for multi-label defect segmentation.

Output has 4 channels (one per defect class); trained with BCE+Dice so channels
are independent (a pixel can belong to multiple classes). Optional ResNet-style
encoder is avoided to keep the model self-contained and easy to read.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 4, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _cat(up, skip):
        # pad if odd spatial sizes cause a 1px mismatch
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))
        x = self.u4(self._cat(self.up4(bn), c4))
        x = self.u3(self._cat(self.up3(x), c3))
        x = self.u2(self._cat(self.up2(x), c2))
        x = self.u1(self._cat(self.up1(x), c1))
        return self.head(x)  # logits (B, 4, H, W)


# --- losses & metrics -------------------------------------------------------
class DiceBCELoss(nn.Module):
    """Sum of BCE-with-logits and soft Dice over all defect channels."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * target).sum(dims)
        union = probs.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


@torch.no_grad()
def dice_per_class(logits, target, thr: float = 0.5, smooth: float = 1.0):
    """Return a (4,) tensor of Dice scores, one per defect channel."""
    probs = (torch.sigmoid(logits) >= thr).float()
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    return (2 * inter + smooth) / (union + smooth)
