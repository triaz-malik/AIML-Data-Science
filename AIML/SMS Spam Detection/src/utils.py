"""Shared utilities: deterministic train/test split used by ALL models."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "sms_features.csv"
CLASSES = ["Normal", "Promotion", "Spam", "Phishing", "Fraud"]
SEED = 42


def load_split(test_size: float = 0.2):
    """Return (train_df, test_df) with an identical stratified split everywhere."""
    df = pd.read_csv(FEATURES)
    df["text"] = df["text"].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["label5"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
