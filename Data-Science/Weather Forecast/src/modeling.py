"""Shared modeling utilities: chronological splits, metrics, and sequence
windowing for the deep-learning models. Used across Phases 4-8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# Chronological split fractions (time series => no shuffling).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> test

# Window of past hours fed to the sequence models.
WINDOW = 24


def metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, MAPE, R^2."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # MAPE guarded against near-zero temperatures (°C crosses 0).
    denom = np.where(np.abs(y_true) < 1.0, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 3), "R2": round(r2, 4)}


def chrono_split_idx(n: int):
    """Return (train_end, val_end) indices for a chronological split."""
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return train_end, val_end


def load_feature_table() -> pd.DataFrame:
    return pd.read_parquet(f"{C.TABLE_DIR}/features.parquet")


# --------------------------------------------------------------------------- #
# Tabular split (for tree / linear baselines)
# --------------------------------------------------------------------------- #
def tabular_splits(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c != "target"]
    X = df[feat_cols].values
    y = df["target"].values
    tr, va = chrono_split_idx(len(df))
    return {
        "feat_cols": feat_cols,
        "X_train": X[:tr], "y_train": y[:tr],
        "X_val": X[tr:va], "y_val": y[tr:va],
        "X_test": X[va:], "y_test": y[va:],
        "index": df.index,
        "test_index": df.index[va:],
    }


# --------------------------------------------------------------------------- #
# Sequence windowing (for LSTM / GRU / Bi-LSTM)
# --------------------------------------------------------------------------- #
def make_windows(values: np.ndarray, target: np.ndarray, window: int,
                 horizon: int = 1, multi: bool = False):
    """Build sliding windows.

    values : (T, F) feature matrix
    target : (T,)   target series (already aligned to "next step" semantics
             for single-step; for multi-step we read horizon steps ahead).
    Returns X:(N, window, F), y:(N,) or (N, horizon).
    """
    X, y = [], []
    T = len(values)
    if multi:
        # predict target[t .. t+horizon-1] from window ending at t-1
        for t in range(window, T - horizon + 1):
            X.append(values[t - window:t])
            y.append(target[t:t + horizon])
    else:
        for t in range(window, T):
            X.append(values[t - window:t])
            y.append(target[t])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def sequence_splits(df: pd.DataFrame, window: int = WINDOW,
                    horizon: int = 1, multi: bool = False):
    """Chronological train/val/test sequence tensors with train-fit scaling.

    Scaling (standardization) is fit on the training portion only to prevent
    leakage. The target here is the *current* temperature column shifted by the
    windowing, so we use the raw `temperature` and engineered features.
    """
    # Use a focused, non-redundant feature set for sequences.
    seq_features = [
        "temperature", "pressure", "humidity", "temp_dew", "vp_act",
        "air_density", "wind_speed", "hour_sin", "hour_cos",
        "doy_sin", "doy_cos",
    ]
    seq_features = [c for c in seq_features if c in df.columns]

    data = df[seq_features].values.astype("float32")
    # Target is temperature; for single step we want temp[t] from window<t.
    temp = df["temperature"].values.astype("float32")

    tr, va = chrono_split_idx(len(df))

    # Fit scaler on train rows only.
    mu = data[:tr].mean(axis=0)
    sd = data[:tr].std(axis=0)
    sd[sd == 0] = 1.0
    data_s = (data - mu) / sd

    # Target scaling (temperature) — store to invert later.
    t_mu, t_sd = float(temp[:tr].mean()), float(temp[:tr].std())
    temp_s = (temp - t_mu) / t_sd

    X, y = make_windows(data_s, temp_s, window, horizon, multi)
    # Recompute split indices in window space.
    n = len(X)
    # Each window i corresponds to original time (window + i). Map fractions.
    tr_w = int(n * TRAIN_FRAC)
    va_w = int(n * (TRAIN_FRAC + VAL_FRAC))

    out = {
        "features": seq_features,
        "X_train": X[:tr_w], "y_train": y[:tr_w],
        "X_val": X[tr_w:va_w], "y_val": y[tr_w:va_w],
        "X_test": X[va_w:], "y_test": y[va_w:],
        "t_mu": t_mu, "t_sd": t_sd,
        "n_features": X.shape[2],
        "window": window, "horizon": horizon,
    }
    return out


def inv_target(y_scaled, t_mu, t_sd):
    return np.asarray(y_scaled) * t_sd + t_mu
