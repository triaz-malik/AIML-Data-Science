"""Shared helpers: data loading, cleaning, and figure saving."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend so the pipeline runs without a display
import matplotlib.pyplot as plt
import pandas as pd

import config


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes coming from the raw CSV."""
    df = df.copy()
    # Churn / plan columns arrive as strings ("True"/"False", "Yes"/"No")
    if df[config.TARGET].dtype == object:
        df[config.TARGET] = (
            df[config.TARGET].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
        )
    else:
        df[config.TARGET] = df[config.TARGET].astype(int)

    for col in config.CATEGORICAL_RAW:
        df[col] = (
            df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
        )
    return df


def load_raw(split: str = "all") -> pd.DataFrame:
    """Load the raw dataset.

    split: "train" (80%), "test" (20%), or "all" (both concatenated).
    """
    train = _clean(pd.read_csv(config.RAW_TRAIN_CSV))
    test = _clean(pd.read_csv(config.RAW_TEST_CSV))
    if split == "train":
        return train.reset_index(drop=True)
    if split == "test":
        return test.reset_index(drop=True)
    return pd.concat([train, test], ignore_index=True)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the provided 80/20 split (used as train / holdout test)."""
    return (
        _clean(pd.read_csv(config.RAW_TRAIN_CSV)).reset_index(drop=True),
        _clean(pd.read_csv(config.RAW_TEST_CSV)).reset_index(drop=True),
    )


def savefig(fig: plt.Figure, name: str) -> None:
    """Save a figure to the figures folder as PNG and close it."""
    path = config.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.relative_to(config.PROJECT_ROOT)}")


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")
