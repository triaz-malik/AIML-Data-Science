"""
Reusable data loading + feature engineering for the Sparkov fraud dataset.

Used by the baseline and later modeling notebooks so train/test are processed
identically. Deliberately excludes:
  - PII / identifiers: first, last, street, trans_num, cc_num (raw), dob (raw)
  - leakage: unix_time (duplicates the timestamp)
  - geo-distance: profiling showed it is NOT predictive in this simulated data
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "Credit Card Train Data.csv"
TEST_PATH = ROOT / "Credit Card Test Data.csv"

# Numeric features fed to the model.
NUMERIC = [
    "log_amt", "amt", "hour", "is_night", "day_of_week",
    "is_weekend", "age", "log_city_pop", "gender_M",
]
# Low-cardinality categorical -> one-hot.
CATEGORICAL = ["category"]
TARGET = "is_fraud"


def load_raw(path):
    """Load a raw CSV (col 0 is an unnamed index)."""
    return pd.read_csv(path, index_col=0)


def engineer_features(df):
    """Add engineered columns. Pure / row-wise, so no train->test leakage."""
    df = df.copy()
    ts = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = ts.dt.hour
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 3)).astype(int)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    dob = pd.to_datetime(df["dob"])
    df["age"] = ((ts - dob).dt.days // 365).astype(int)

    df["log_amt"] = np.log1p(df["amt"])
    df["log_city_pop"] = np.log1p(df["city_pop"])
    df["gender_M"] = (df["gender"] == "M").astype(int)
    return df


def build_xy(df, dummy_columns=None):
    """
    Build the model matrix X and target y.

    dummy_columns: if given (the train columns), align one-hot columns to it so
    train and test always share the exact same feature space.
    """
    feat = engineer_features(df)
    x_num = feat[NUMERIC].reset_index(drop=True)
    x_cat = pd.get_dummies(feat[CATEGORICAL], prefix="cat").reset_index(drop=True)
    X = pd.concat([x_num, x_cat], axis=1)

    if dummy_columns is not None:
        X = X.reindex(columns=dummy_columns, fill_value=0)

    y = feat[TARGET].reset_index(drop=True)
    return X, y


def load_train_test():
    """Convenience: load both files and return aligned (X_train, y_train, X_test, y_test)."""
    X_train, y_train = build_xy(load_raw(TRAIN_PATH))
    X_test, y_test = build_xy(load_raw(TEST_PATH), dummy_columns=X_train.columns)
    return X_train, y_train, X_test, y_test
