"""
Phase 2 - Data Cleaning.

Business rationale
------------------
Recommendations and segments are only as trustworthy as the data behind them.
We turn ~1M raw line items into an analysis-ready transaction table by removing
the records that would otherwise corrupt customer intelligence:

  * missing Customer ID      -> cannot attribute behaviour to a customer
  * cancellations / returns  -> negative quantity, handled separately
  * non-product stock codes  -> postage, fees, manual adjustments, etc.
  * non-positive price        -> adjustments / data-entry errors
  * exact duplicate rows     -> double-counted revenue

The function prints a transparent "data quality funnel" so every dropped row is
accounted for, and writes a cleaned Parquet table for the rest of the pipeline.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Stock codes that are not sellable products (case-insensitive, exact or prefix).
NON_PRODUCT_CODES = {
    "POST", "DOT", "D", "M", "C2", "BANK CHARGES", "BANK_CHARGE",
    "AMAZONFEE", "CRUK", "S", "GIFT", "PADS", "TEST001", "TEST002",
    "ADJUST", "ADJUST2",
}
# Pattern for clearly-not-a-product codes (pure letters like "POST", "DCGS...").
_LETTER_ONLY = re.compile(r"^[A-Za-z]+$")


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV with correct dtypes."""
    df = pd.read_csv(
        path,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Description": "string",
            "Quantity": "int64",
            "Price": "float64",
            "Customer ID": "float64",
            "Country": "string",
        },
        parse_dates=["InvoiceDate"],
    )
    df = df.rename(columns={"Customer ID": "CustomerID"})
    return df


def _is_non_product(code: str) -> bool:
    if code is None:
        return True
    c = code.strip().upper()
    if c in NON_PRODUCT_CODES:
        return True
    # Pure-letter codes (no digits) are administrative, real products carry digits.
    if _LETTER_ONLY.match(c) and not any(ch.isdigit() for ch in c):
        return True
    return False


def clean(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (clean_sales, stats). Clean = valid positive-revenue purchases."""
    stats: dict[str, int] = {}
    n0 = len(df)
    stats["raw_rows"] = n0

    # Normalise text
    df = df.copy()
    df["Invoice"] = df["Invoice"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["Description"] = df["Description"].str.strip()

    # Flag cancellations (Invoice prefixed with 'C') before we drop anything.
    df["IsCancellation"] = df["Invoice"].str.startswith("C", na=False)
    stats["cancellation_rows"] = int(df["IsCancellation"].sum())

    # 1) Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    stats["dropped_duplicates"] = before - len(df)

    # 2) Drop missing Customer ID
    before = len(df)
    df = df[df["CustomerID"].notna()]
    stats["dropped_missing_customer"] = before - len(df)

    # 3) Drop missing / blank description
    before = len(df)
    df = df[df["Description"].notna() & (df["Description"].str.len() > 0)]
    stats["dropped_missing_description"] = before - len(df)

    # 4) Drop non-product stock codes
    before = len(df)
    mask_product = ~df["StockCode"].map(_is_non_product)
    df = df[mask_product]
    stats["dropped_non_product"] = before - len(df)

    # 5) Separate returns (Quantity <= 0) from sales
    returns = df[(df["Quantity"] <= 0) | df["IsCancellation"]].copy()
    df = df[(df["Quantity"] > 0) & (~df["IsCancellation"])]
    stats["return_rows"] = len(returns)

    # 6) Drop non-positive price
    before = len(df)
    df = df[df["Price"] > 0]
    stats["dropped_nonpositive_price"] = before - len(df)

    # Final typing + derived columns
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
    df = df.reset_index(drop=True)

    stats["clean_rows"] = len(df)
    stats["clean_customers"] = df["CustomerID"].nunique()
    stats["clean_products"] = df["StockCode"].nunique()
    stats["clean_revenue"] = float(df["Revenue"].sum())
    stats["date_min"] = str(df["InvoiceDate"].min())
    stats["date_max"] = str(df["InvoiceDate"].max())

    if verbose:
        _print_funnel(stats)

    return df, stats


def _print_funnel(s: dict) -> None:
    print("\n" + "=" * 60)
    print("DATA QUALITY FUNNEL")
    print("=" * 60)
    print(f"Raw rows .......................... {s['raw_rows']:>10,}")
    print(f"  - duplicates .................... {s['dropped_duplicates']:>10,}")
    print(f"  - missing Customer ID ........... {s['dropped_missing_customer']:>10,}")
    print(f"  - missing description ........... {s['dropped_missing_description']:>10,}")
    print(f"  - non-product stock codes ....... {s['dropped_non_product']:>10,}")
    print(f"  - returns / cancellations ....... {s['return_rows']:>10,}")
    print(f"  - non-positive price ............ {s['dropped_nonpositive_price']:>10,}")
    print("-" * 60)
    print(f"Clean sale rows ................... {s['clean_rows']:>10,}")
    print(f"Unique customers .................. {s['clean_customers']:>10,}")
    print(f"Unique products ................... {s['clean_products']:>10,}")
    print(f"Total revenue ..................... {s['clean_revenue']:>13,.2f}")
    print(f"Date range ........................ {s['date_min']} -> {s['date_max']}")
    print("=" * 60 + "\n")


def run() -> pd.DataFrame:
    print("Loading raw data ...")
    raw = load_raw()
    clean_df, stats = clean(raw)
    clean_df.to_parquet(config.CLEAN_PARQUET, index=False)
    pd.Series(stats).to_csv(config.REPORT_DIR / "cleaning_funnel.csv")
    print(f"Saved cleaned transactions -> {config.CLEAN_PARQUET}")
    return clean_df


if __name__ == "__main__":
    run()
