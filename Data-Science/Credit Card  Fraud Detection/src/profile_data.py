"""
Quick data-profiling pass for the Sparkov credit-card fraud dataset.

Purpose: understand dtypes, missing values, cardinality, target balance, and a
few key fraud signals BEFORE building the EDA notebook or any model. Runs on the
full train file but only computes cheap aggregates (no plotting).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "Credit Card Train Data.csv"

# Column 0 is an unnamed index in the raw file.
df = pd.read_csv(TRAIN, index_col=0)

print("=" * 70)
print(f"SHAPE: {df.shape[0]:,} rows x {df.shape[1]} cols")
print("=" * 70)

print("\n--- DTYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES (cols with any) ---")
na = df.isna().sum()
print(na[na > 0] if na.any() else "None")

print("\n--- TARGET BALANCE (is_fraud) ---")
vc = df["is_fraud"].value_counts()
print(vc)
print(f"fraud rate: {df['is_fraud'].mean() * 100:.3f}%")

print("\n--- CARDINALITY (categorical-ish cols) ---")
for c in ["merchant", "category", "city", "state", "job", "gender", "cc_num"]:
    print(f"{c:12s}: {df[c].nunique():,} unique")

print("\n--- AMOUNT (amt) by class ---")
print(df.groupby("is_fraud")["amt"].describe()[["mean", "50%", "max"]])

print("\n--- FRAUD RATE BY CATEGORY (top 10) ---")
g = df.groupby("category")["is_fraud"].agg(["mean", "sum", "count"])
g["mean_pct"] = (g["mean"] * 100).round(3)
print(g.sort_values("mean", ascending=False)[["mean_pct", "sum", "count"]].head(10))

# Geo-distance signal: haversine between cardholder and merchant.
print("\n--- HOME->MERCHANT DISTANCE (km) by class ---")
R = 6371.0
lat1, lon1, lat2, lon2 = map(np.radians, [df["lat"], df["long"], df["merch_lat"], df["merch_long"]])
dlat, dlon = lat2 - lat1, lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
df["dist_km"] = R * 2 * np.arcsin(np.sqrt(a))
print(df.groupby("is_fraud")["dist_km"].describe()[["mean", "50%", "max"]])

# Temporal signal: fraud rate by hour of day.
print("\n--- FRAUD RATE BY HOUR (from trans_date_trans_time) ---")
ts = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = ts.dt.hour
h = df.groupby("hour")["is_fraud"].mean().mul(100).round(3)
print(h.to_string())
