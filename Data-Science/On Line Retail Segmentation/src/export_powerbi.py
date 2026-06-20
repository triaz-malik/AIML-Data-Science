"""
Phase 10 - Power BI star-schema export.

Power BI is happiest with a star schema: one fact table surrounded by
conformed dimension tables. This module writes analysis-ready CSVs to
data/powerbi/ that a .pbix can load directly and relate on the natural keys:

  fact_sales         (grain: line item)   -> CustomerID, StockCode, DateKey
  dim_customer       (grain: customer)    RFM, segment, CLV score
  dim_product        (grain: product)     units, revenue, recommendation flag
  dim_date           (grain: day)
  fact_recommendations (customer x recommended product)
  association_rules  (product -> product, support/confidence/lift)

The CLV model (trained in Phase 8) is applied to every customer's full-history
features to attach a forward-looking value score to dim_customer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def export() -> None:
    tx = pd.read_parquet(config.CLEAN_PARQUET)
    cust = pd.read_parquet(config.CUSTOMER_PARQUET)

    # ----- fact_sales ----------------------------------------------------- #
    fact = tx[["Invoice", "StockCode", "CustomerID", "InvoiceDate",
               "Quantity", "Price", "Revenue", "Country"]].copy()
    fact["DateKey"] = fact["InvoiceDate"].dt.strftime("%Y%m%d").astype(int)
    fact.to_csv(config.POWERBI_DIR / "fact_sales.csv", index=False)

    # ----- dim_customer (+ CLV score) ------------------------------------- #
    bundle = joblib.load(config.MODEL_DIR / "clv_model.joblib")
    model, feats = bundle["model"], bundle["features"]
    cust = cust.copy()
    cust["PredictedCLV_6m"] = np.clip(
        np.expm1(model.predict(cust[feats])), 0, None).round(2)
    dim_customer = cust[[
        "CustomerID", "Country", "Recency", "Frequency", "Monetary",
        "AvgBasketValue", "ProductDiversity", "Tenure", "AvgInterpurchase",
        "R_score", "F_score", "M_score", "RFM_sum", "RFM_segment",
        "segment", "PredictedCLV_6m",
    ]].copy()
    dim_customer.to_csv(config.POWERBI_DIR / "dim_customer.csv", index=False)

    # ----- dim_product ---------------------------------------------------- #
    dim_product = (tx.groupby(["StockCode", "Description"])
                     .agg(units=("Quantity", "sum"),
                          revenue=("Revenue", "sum"),
                          orders=("Invoice", "nunique"),
                          customers=("CustomerID", "nunique"))
                     .reset_index())
    rec_path = config.PROCESSED_DIR / "recommendations.parquet"
    if rec_path.exists():
        recommended = set(pd.read_parquet(rec_path)["StockCode"].unique())
        dim_product["is_recommended"] = dim_product["StockCode"].isin(recommended)
    dim_product.to_csv(config.POWERBI_DIR / "dim_product.csv", index=False)

    # ----- dim_date ------------------------------------------------------- #
    dates = pd.date_range(tx["InvoiceDate"].min().normalize(),
                          tx["InvoiceDate"].max().normalize(), freq="D")
    dim_date = pd.DataFrame({"Date": dates})
    dim_date["DateKey"] = dim_date["Date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Month"] = dim_date["Date"].dt.month
    dim_date["MonthName"] = dim_date["Date"].dt.month_name()
    dim_date["Quarter"] = dim_date["Date"].dt.quarter
    dim_date["Weekday"] = dim_date["Date"].dt.day_name()
    dim_date.to_csv(config.POWERBI_DIR / "dim_date.csv", index=False)

    # ----- fact_recommendations ------------------------------------------ #
    if rec_path.exists():
        pd.read_parquet(rec_path).to_csv(
            config.POWERBI_DIR / "fact_recommendations.csv", index=False)

    # ----- association_rules (copy from reports) -------------------------- #
    ar = config.REPORT_DIR / "association_rules.csv"
    if ar.exists():
        pd.read_csv(ar).to_csv(config.POWERBI_DIR / "association_rules.csv",
                               index=False)

    print("Power BI star-schema exports written to data/powerbi/:")
    for f in sorted(config.POWERBI_DIR.glob("*.csv")):
        print(f"  {f.name:<28} {f.stat().st_size/1e6:6.2f} MB")


if __name__ == "__main__":
    export()
