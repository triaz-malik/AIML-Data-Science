"""Data loading, cleaning, the preprocessing transformer, and the train/test split.

All scripts import from here so the split and feature handling stay identical
everywhere (no leakage, reproducible results via a fixed random_state).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C


def load_train() -> pd.DataFrame:
    """Load the labelled training CSV."""
    return pd.read_csv(C.TRAIN_CSV)


def load_unlabelled_test() -> pd.DataFrame:
    """Load the unlabelled Test_data.csv (used only for final predictions)."""
    return pd.read_csv(C.TEST_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and constant columns. Reports what it removed."""
    df = df.copy()
    n_missing = int(df.isnull().sum().sum())
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    dropped = [c for c in C.CONSTANT_CANDIDATES if c in df.columns]
    # also catch any other accidentally-constant feature columns
    for c in df.columns:
        if c != C.TARGET and df[c].nunique(dropna=False) <= 1 and c not in dropped:
            dropped.append(c)
    df = df.drop(columns=dropped)

    print(f"[clean] missing values: {n_missing} | duplicate rows removed: {n_dupes} "
          f"| constant columns dropped: {dropped}")
    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) where y is 1 for the positive/attack class, else 0."""
    X = df.drop(columns=[C.TARGET])
    y = (df[C.TARGET] == C.POSITIVE_LABEL).astype(int)
    return X, y


def feature_columns(X: pd.DataFrame):
    """Split feature names into (categorical, numeric) for the transformer."""
    cats = [c for c in C.CATEGORICAL if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    return cats, nums


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals (ignore unseen), standard-scale numerics."""
    cats, nums = feature_columns(X)
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("num", StandardScaler(), nums),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_split():
    """Full path: load -> clean -> split features/target -> stratified train/test.

    Returns X_train, X_test, y_train, y_test (raw, unscaled feature frames).
    """
    df = clean(load_train())
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=C.TEST_SIZE,
        stratify=y,
        random_state=C.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test
