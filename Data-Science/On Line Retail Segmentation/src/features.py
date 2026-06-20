"""
Phase 4 - Feature Engineering (RFM + customer intelligence).

Transforms the cleaned line-item table into one row per customer. The core is
the classic RFM trio, enriched with behavioural features that downstream models
(segmentation, CLV) and the business narrative rely on:

  Recency        - days since last purchase (lower = more engaged)
  Frequency      - number of distinct orders
  Monetary       - total revenue
  AvgBasketValue - mean revenue per order
  ProductsBought - total units
  ProductDiversity - distinct products purchased (breadth of interest)
  Tenure         - days between first and last purchase (lifecycle length)
  AvgInterpurchase - mean days between orders (cadence)
  Country        - primary market

Also assigns the standard RFM 1-5 scores and an RFM segment label using quantile
binning, which gives an interpretable baseline alongside the ML clusters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load() -> pd.DataFrame:
    return pd.read_parquet(config.CLEAN_PARQUET)


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=config.SNAPSHOT_OFFSET_DAYS)

    # Order-level revenue to derive basket statistics.
    orders = (df.groupby(["CustomerID", "Invoice"])
                .agg(order_revenue=("Revenue", "sum"),
                     order_date=("InvoiceDate", "max"))
                .reset_index())

    feats = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        ProductsBought=("Quantity", "sum"),
        ProductDiversity=("StockCode", "nunique"),
    )

    feats["Recency"] = (snapshot - feats["last_purchase"]).dt.days
    feats["Tenure"] = (feats["last_purchase"] - feats["first_purchase"]).dt.days
    feats["AvgBasketValue"] = (feats["Monetary"] / feats["Frequency"]).round(2)

    # Average days between consecutive orders (cadence); 0 for single-order custs.
    cadence = (orders.sort_values("order_date")
                     .groupby("CustomerID")["order_date"]
                     .apply(lambda s: s.diff().dt.days.mean()))
    feats["AvgInterpurchase"] = cadence.fillna(0).round(1)

    # Primary country (mode).
    primary_country = (df.groupby("CustomerID")["Country"]
                         .agg(lambda s: s.mode().iat[0]))
    feats["Country"] = primary_country

    feats = feats.drop(columns=["last_purchase", "first_purchase"])
    feats = feats.reset_index()
    return feats


def add_rfm_scores(feats: pd.DataFrame) -> pd.DataFrame:
    """Quantile 1-5 RFM scores + an interpretable segment label."""
    f = feats.copy()

    # Recency: lower is better -> reverse the labels.
    f["R_score"] = pd.qcut(f["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency & Monetary: higher is better. rank(method="first") breaks ties so
    # qcut never fails on the heavily-tied frequency distribution.
    f["F_score"] = pd.qcut(f["Frequency"].rank(method="first"), 5,
                           labels=[1, 2, 3, 4, 5]).astype(int)
    f["M_score"] = pd.qcut(f["Monetary"].rank(method="first"), 5,
                           labels=[1, 2, 3, 4, 5]).astype(int)
    f["RFM_sum"] = f[["R_score", "F_score", "M_score"]].sum(axis=1)

    f["RFM_segment"] = f.apply(_rfm_label, axis=1)
    return f


def _rfm_label(row: pd.Series) -> str:
    r, fr, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and fr >= 4 and m >= 4:
        return "Champions"
    if fr >= 4 and m >= 4:
        return "Loyal"
    if r >= 4 and fr <= 2:
        return "New / Promising"
    if r <= 2 and fr >= 3 and m >= 3:
        return "At-Risk"
    if r <= 2 and fr <= 2:
        return "Lost"
    return "Regular"


def run() -> pd.DataFrame:
    df = load()
    feats = build_customer_features(df)
    feats = add_rfm_scores(feats)
    feats.to_parquet(config.CUSTOMER_PARQUET, index=False)

    print(f"Built features for {len(feats):,} customers "
          f"({feats.shape[1]} columns)")
    print("\nRFM segment distribution:")
    dist = feats["RFM_segment"].value_counts()
    for seg, n in dist.items():
        rev = feats.loc[feats["RFM_segment"] == seg, "Monetary"].sum()
        print(f"  {seg:<18} {n:>6,} customers   "
              f"{rev/feats['Monetary'].sum():>6.1%} of revenue")
    print(f"\nSaved -> {config.CUSTOMER_PARQUET}")
    return feats


if __name__ == "__main__":
    run()
