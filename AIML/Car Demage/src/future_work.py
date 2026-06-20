"""Phases 10-11 - Damage severity & repair-cost estimation (SCAFFOLD ONLY).

IMPORTANT: the Car Damage Detection dataset used here only has BINARY labels
(damage vs whole). It contains no severity grades and no repair-cost figures, so
these phases cannot be trained on the current data without additional labels.

This file documents the intended design and provides ready-to-use model heads so
the project can be extended the moment labelled data is available.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------------------------
# Phase 10 — Damage severity (multi-class): No / Minor / Moderate / Severe
# ---------------------------------------------------------------------------
SEVERITY_CLASSES = ["no_damage", "minor", "moderate", "severe"]


def build_severity_model(num_classes: int = 4, dropout: float = 0.4) -> nn.Module:
    """Same backbone as Phase 6, but a 4-way head. Train once severity labels exist."""
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = net.classifier[1].in_features
    net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return net


# ---------------------------------------------------------------------------
# Phase 11 — Repair-cost regression
# ---------------------------------------------------------------------------
# Input features (once available): predicted damage_type (one-hot),
# severity (ordinal), vehicle_type (one-hot), damage_area_ratio (float).
# Output: estimated repair cost (USD).
class RepairCostRegressor(nn.Module):
    """Small MLP mapping structured damage features -> estimated cost (USD)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),                      # cost in USD
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# A transparent rule-of-thumb baseline so the app can show *something* today.
# These are illustrative placeholders, NOT learned values.
_SEVERITY_BASE_COST = {"no_damage": 0.0, "minor": 250.0, "moderate": 900.0, "severe": 2200.0}
_VEHICLE_MULTIPLIER = {"sedan": 1.0, "suv": 1.25, "truck": 1.3, "luxury": 1.8, "unknown": 1.1}


def heuristic_repair_cost(severity: str, vehicle_type: str = "unknown") -> float:
    """Illustrative cost estimate until a real regression model is trained."""
    base = _SEVERITY_BASE_COST.get(severity, 0.0)
    return round(base * _VEHICLE_MULTIPLIER.get(vehicle_type, 1.1), 2)


if __name__ == "__main__":
    print("Phases 10-11 are scaffolds — they require severity/cost labels not present "
          "in this dataset.")
    for s in SEVERITY_CLASSES:
        print(f"  heuristic cost [{s:10s}] sedan=${heuristic_repair_cost(s, 'sedan'):.0f}  "
              f"suv=${heuristic_repair_cost(s, 'suv'):.0f}")
