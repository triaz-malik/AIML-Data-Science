"""
Phase 3 - Exploratory Data Analysis.

Produces business-framed charts (saved to outputs/figures) and a text summary
answering the questions a retail stakeholder would ask:

  Customers : Who spends the most? Who buys frequently? Where are they?
  Products  : Which products drive sales? Which barely move?
  Time      : When are the peak periods? Is there seasonality?
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sns.set_theme(style="whitegrid", palette=config.PALETTE)


def _save(fig, name: str) -> None:
    path = config.FIGURE_DIR / name
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path.name}")


def load() -> pd.DataFrame:
    return pd.read_parquet(config.CLEAN_PARQUET)


def customer_analysis(df: pd.DataFrame) -> None:
    cust = df.groupby("CustomerID").agg(
        revenue=("Revenue", "sum"),
        orders=("Invoice", "nunique"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(cust["revenue"].clip(upper=cust["revenue"].quantile(0.99)),
                 bins=50, ax=axes[0], color="#3b528b")
    axes[0].set_title("Customer spend distribution (99th pct clip)")
    axes[0].set_xlabel("Total revenue per customer")
    sns.histplot(cust["orders"].clip(upper=cust["orders"].quantile(0.99)),
                 bins=50, ax=axes[1], color="#21918c")
    axes[1].set_title("Purchase frequency distribution")
    axes[1].set_xlabel("Number of orders per customer")
    _save(fig, "01_customer_distributions.png")

    # Pareto: share of revenue from top customers
    rev_sorted = cust["revenue"].sort_values(ascending=False).reset_index(drop=True)
    cum_share = rev_sorted.cumsum() / rev_sorted.sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(cum_share) + 1), cum_share.values, color="#440154")
    top20 = int(len(cum_share) * 0.2)
    ax.axvline(top20, ls="--", color="grey")
    ax.axhline(cum_share.iloc[top20 - 1], ls="--", color="grey")
    ax.set_title(f"Revenue Pareto — top 20% of customers = "
                 f"{cum_share.iloc[top20 - 1]:.0%} of revenue")
    ax.set_xlabel("Customers ranked by spend")
    ax.set_ylabel("Cumulative revenue share")
    _save(fig, "02_customer_pareto.png")


def country_analysis(df: pd.DataFrame) -> None:
    by_country = (df.groupby("Country")
                    .agg(revenue=("Revenue", "sum"),
                         customers=("CustomerID", "nunique"))
                    .sort_values("revenue", ascending=False))
    # Exclude the UK so the rest of the world is legible.
    row = by_country.drop(index="United Kingdom", errors="ignore").head(12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=by_country["revenue"].head(10).values,
                y=by_country["revenue"].head(10).index, ax=axes[0])
    axes[0].set_title("Revenue by country (incl. UK)")
    axes[0].set_xlabel("Revenue")
    sns.barplot(x=row["revenue"].values, y=row["revenue"].index, ax=axes[1])
    axes[1].set_title("Revenue by country (excl. UK)")
    axes[1].set_xlabel("Revenue")
    _save(fig, "03_country_revenue.png")


def product_analysis(df: pd.DataFrame) -> None:
    prod = df.groupby("Description").agg(
        units=("Quantity", "sum"),
        revenue=("Revenue", "sum"),
    )
    top_units = prod.sort_values("units", ascending=False).head(15)
    top_rev = prod.sort_values("revenue", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=top_units["units"].values, y=top_units.index, ax=axes[0])
    axes[0].set_title("Top 15 products by units sold")
    axes[0].set_xlabel("Units")
    sns.barplot(x=top_rev["revenue"].values, y=top_rev.index, ax=axes[1])
    axes[1].set_title("Top 15 products by revenue")
    axes[1].set_xlabel("Revenue")
    _save(fig, "04_top_products.png")


def time_analysis(df: pd.DataFrame) -> None:
    d = df.copy()
    d["YearMonth"] = d["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    monthly = d.groupby("YearMonth")["Revenue"].sum()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#3b528b")
    ax.set_title("Monthly revenue trend")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "05_monthly_revenue.png")

    # Weekday x hour heatmap of order volume
    d["weekday"] = d["InvoiceDate"].dt.day_name()
    d["hour"] = d["InvoiceDate"].dt.hour
    order_keys = d.drop_duplicates("Invoice")
    pivot = (order_keys.pivot_table(index="weekday", columns="hour",
                                    values="Invoice", aggfunc="count")
             .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                       "Friday", "Saturday", "Sunday"]))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(pivot, cmap="viridis", ax=ax, cbar_kws={"label": "orders"})
    ax.set_title("Order volume by weekday and hour")
    _save(fig, "06_weekday_hour_heatmap.png")


def write_summary(df: pd.DataFrame) -> None:
    cust = df.groupby("CustomerID").agg(revenue=("Revenue", "sum"),
                                        orders=("Invoice", "nunique"))
    rev_sorted = cust["revenue"].sort_values(ascending=False)
    top20 = int(len(rev_sorted) * 0.2)
    by_country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    top_prod = (df.groupby("Description")["Revenue"].sum()
                  .sort_values(ascending=False).head(5))

    lines = [
        "EDA SUMMARY",
        "=" * 50,
        f"Customers ................ {df['CustomerID'].nunique():,}",
        f"Products ................. {df['StockCode'].nunique():,}",
        f"Invoices ................. {df['Invoice'].nunique():,}",
        f"Total revenue ............ {df['Revenue'].sum():,.2f}",
        f"Avg revenue / customer ... {cust['revenue'].mean():,.2f}",
        f"Median revenue / customer  {cust['revenue'].median():,.2f}",
        f"Avg orders / customer .... {cust['orders'].mean():.2f}",
        "",
        f"Top 20% of customers drive {rev_sorted.head(top20).sum() / rev_sorted.sum():.1%} of revenue.",
        f"UK share of revenue ...... {by_country['United Kingdom'] / by_country.sum():.1%}",
        "",
        "Top 5 products by revenue:",
    ]
    for name, rev in top_prod.items():
        lines.append(f"  {rev:>12,.0f}  {name}")
    text = "\n".join(lines)
    (config.REPORT_DIR / "eda_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text + "\n")


def run() -> None:
    df = load()
    print("Generating EDA figures ...")
    customer_analysis(df)
    country_analysis(df)
    product_analysis(df)
    time_analysis(df)
    write_summary(df)
    print(f"All EDA figures saved to {config.FIGURE_DIR}")


if __name__ == "__main__":
    run()
