"""
Phase 2 - Feature Engineering.

Creates:
  * Total Usage Minutes
  * Total Charges
  * International Usage Ratio
  * Customer Value Score
  * Revenue Segment   (Low / Medium / High)
  * Usage Segment     (Light / Medium / Heavy)

Writes the enriched table to outputs/data/telecom_features.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from utils import load_raw, section


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Total Usage Minutes"] = df[config.MINUTE_COLS].sum(axis=1)
    df["Total Charges"] = df[config.CHARGE_COLS].sum(axis=1)

    # share of usage that is international (avoid divide-by-zero)
    df["International Usage Ratio"] = (
        df["Total intl minutes"] / df["Total Usage Minutes"].replace(0, np.nan)
    ).fillna(0)

    df["Total Calls"] = (
        df["Total day calls"]
        + df["Total eve calls"]
        + df["Total night calls"]
        + df["Total intl calls"]
    )

    # ---- Customer Value Score --------------------------------------------
    # Weighted blend of spend, tenure and engagement, scaled to 0-100.
    scaler = MinMaxScaler()
    comps = scaler.fit_transform(
        df[["Total Charges", "Account length", "Total Calls", "Number vmail messages"]]
    )
    value = (
        0.55 * comps[:, 0]   # spend dominates value
        + 0.20 * comps[:, 1]  # tenure / loyalty
        + 0.20 * comps[:, 2]  # call engagement
        + 0.05 * comps[:, 3]  # voicemail engagement
    )
    df["Customer Value Score"] = (value * 100).round(2)

    # ---- Categorical segments (tertiles) ---------------------------------
    df["Revenue Segment"] = pd.qcut(
        df["Total Charges"], q=3, labels=["Low", "Medium", "High"]
    )
    df["Usage Segment"] = pd.qcut(
        df["Total Usage Minutes"], q=3, labels=["Light", "Medium", "Heavy"]
    )

    return df


def run() -> pd.DataFrame:
    section("PHASE 2 - FEATURE ENGINEERING")
    df = load_raw("all")
    df = add_features(df)

    new_cols = [
        "Total Usage Minutes",
        "Total Charges",
        "International Usage Ratio",
        "Customer Value Score",
        "Revenue Segment",
        "Usage Segment",
    ]
    print("Engineered features:")
    print(df[new_cols].describe(include="all").T.to_string())

    df.to_csv(config.FEATURES_CSV, index=False)
    print(f"\n[data] {config.FEATURES_CSV.relative_to(config.PROJECT_ROOT)} "
          f"({df.shape[0]} rows x {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    run()
