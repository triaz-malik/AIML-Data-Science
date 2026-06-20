"""Data loading utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"

for d in (PROCESSED_DIR, MODELS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        sys.exit(f"Missing train.csv or test.csv in {RAW_DIR}")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test
