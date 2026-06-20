"""Phase 3 - Feature Engineering.

Lag features (t-1, t-6, t-24, t-48), rolling statistics (7h, 24h, 48h),
calendar features (hour, day, month, season, weekend) and cyclic encodings
of periodic signals (hour-of-day, day-of-year, wind direction).

Produces a model-ready feature table saved to outputs/tables/features.parquet
with the regression target `target = temperature[t+1]` (next hour).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# Base meteorological features carried into the models.
BASE_FEATURES = [
    "pressure", "temperature", "temp_dew", "humidity",
    "vp_act", "vp_def", "specific_humidity", "air_density",
    "wind_speed", "wind_speed_max",
]

LAGS = [1, 6, 24, 48]
ROLL_WINDOWS = [7, 24, 48]


def build_features(h: pd.DataFrame) -> pd.DataFrame:
    df = h.copy()

    # --- Calendar features ---
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["dayofyear"] = df.index.dayofyear

    # --- Cyclic encodings (avoid artificial discontinuity at wrap-around) ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["wd_sin"] = np.sin(np.deg2rad(df["wind_dir"]))
    df["wd_cos"] = np.cos(np.deg2rad(df["wind_dir"]))

    # --- Lag features on temperature ---
    for lag in LAGS:
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

    # --- Rolling statistics on temperature (causal: no leakage) ---
    for w in ROLL_WINDOWS:
        roll = df["temperature"].rolling(w)
        df[f"temp_roll_mean_{w}"] = roll.mean()
        df[f"temp_roll_std_{w}"] = roll.std()

    # --- Lag on a couple of strong covariates ---
    df["humidity_lag_1"] = df["humidity"].shift(1)
    df["pressure_lag_1"] = df["pressure"].shift(1)

    # --- Target: next-hour temperature ---
    df["target"] = df["temperature"].shift(-1)

    df = df.dropna()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"target", "season"}
    return [c for c in df.columns if c not in drop and df[c].dtype != "O"]


def main():
    print("[Phase 3] Building features ...")
    h = C.load_hourly()
    feats = build_features(h)
    cols = feature_columns(feats)
    out = feats[cols + ["target"]]
    path = f"{C.TABLE_DIR}/features.parquet"
    out.to_parquet(path)
    print(f"[Phase 3] Feature table: {out.shape[0]} rows x {len(cols)} features")
    print(f"[Phase 3] Features: {cols}")
    print(f"[Phase 3] Saved -> {path}")


if __name__ == "__main__":
    main()
