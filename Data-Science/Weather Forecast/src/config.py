"""Shared configuration and helpers for the Weather Forecasting project.

Multi-Step Weather Forecasting using LSTM, GRU and Bi-LSTM
Dataset: Jena Climate 2009-2016 (10-minute resolution, 14 features).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODEL_DIR = os.path.join(OUT_DIR, "models")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

for _d in (FIG_DIR, MODEL_DIR, TABLE_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

RAW_CSV = os.path.join(DATA_DIR, "jena_climate_2009_2016.csv")

# --------------------------------------------------------------------------- #
# Column naming: map the dataset's raw headers to clean, code-friendly names.
# --------------------------------------------------------------------------- #
COLUMN_MAP = {
    "Date Time": "datetime",
    "p (mbar)": "pressure",
    "T (degC)": "temperature",
    "Tpot (K)": "temp_pot",
    "Tdew (degC)": "temp_dew",
    "rh (%)": "humidity",
    "VPmax (mbar)": "vp_max",
    "VPact (mbar)": "vp_act",
    "VPdef (mbar)": "vp_def",
    "sh (g/kg)": "specific_humidity",
    "H2OC (mmol/mol)": "h2o_conc",
    "rho (g/m**3)": "air_density",
    "wv (m/s)": "wind_speed",
    "max. wv (m/s)": "wind_speed_max",
    "wd (deg)": "wind_dir",
}

TARGET = "temperature"

# Resampling: the raw data is at 10-minute steps. For modeling we use hourly
# data (every 6th record) — this is standard practice for the Jena dataset and
# keeps CPU training tractable while preserving the forecasting signal.
SAMPLE_STEP = 6  # 6 * 10min = 1 hour


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(parse_dates: bool = True) -> pd.DataFrame:
    """Load the full 10-minute-resolution dataset with clean column names."""
    df = pd.read_csv(RAW_CSV)
    df = df.rename(columns=COLUMN_MAP)
    if parse_dates:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M:%S")
    return df


def load_hourly() -> pd.DataFrame:
    """Load the dataset resampled to hourly resolution, datetime-indexed.

    Known data issues handled here:
      * A handful of physically impossible wind values (-9999) are treated as
        missing and forward-filled.
      * Duplicate timestamps are dropped (keep first).
    """
    df = load_raw(parse_dates=True)

    # Fix sentinel bad wind readings (-9999) before any aggregation.
    for col in ("wind_speed", "wind_speed_max"):
        df.loc[df[col] < 0, col] = np.nan

    df = df.drop_duplicates(subset="datetime", keep="first")
    df = df.set_index("datetime").sort_index()

    # Forward-fill the few NaNs we just introduced.
    df = df.ffill().bfill()

    # Downsample to hourly.
    hourly = df.iloc[::SAMPLE_STEP].copy()
    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from a datetime index."""
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["day"] = idx.day
    out["month"] = idx.month
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["season"] = ((idx.month % 12) // 3).map(
        {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    )
    return out


if __name__ == "__main__":
    d = load_hourly()
    print(d.shape)
    print(d.head())
