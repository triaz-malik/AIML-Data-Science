"""
Phase 1 - Exploratory Data Analysis.

Generates the required plots and answers the business questions:
  * Who are heavy users?
  * Who pays the most?
  * Who calls customer care frequently?
  * What drives churn?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from utils import load_raw, savefig, section

sns.set_theme(style="whitegrid", palette="deep")


def run() -> pd.DataFrame:
    section("PHASE 1 - EXPLORATORY DATA ANALYSIS")
    df = load_raw("all")
    print(f"Loaded {len(df):,} customers, {df.shape[1]} columns.")
    churn_rate = df[config.TARGET].mean()
    print(f"Overall churn rate: {churn_rate:.1%}")

    # --- helper engineered columns just for plotting -----------------------
    df["_total_minutes"] = df[config.MINUTE_COLS].sum(axis=1)
    df["_total_charges"] = df[config.CHARGE_COLS].sum(axis=1)
    churn_label = df[config.TARGET].map({0: "Retained", 1: "Churned"})

    # 1. Churn distribution -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[config.TARGET].map({0: "Retained", 1: "Churned"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v}\n({v/len(df):.1%})", ha="center")
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Customers")
    savefig(fig, "01_churn_distribution")

    # 2. Day minutes distribution ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Total day minutes"], kde=True, bins=40, ax=ax)
    ax.axvline(df["Total day minutes"].mean(), color="red", ls="--", label="mean")
    ax.set_title("Total Day Minutes Distribution")
    ax.legend()
    savefig(fig, "02_day_minutes_distribution")

    # 3. International usage distribution -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Total intl minutes"], kde=True, bins=30, ax=axes[0])
    axes[0].set_title("International Minutes Distribution")
    sns.countplot(x="International plan", data=df, ax=axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["No Plan", "Has Plan"])
    axes[1].set_title("International Plan Adoption")
    savefig(fig, "03_international_usage_distribution")

    # 4. Charges distribution ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["_total_charges"], kde=True, bins=40, color="seagreen", ax=ax)
    ax.axvline(df["_total_charges"].mean(), color="red", ls="--", label="mean")
    ax.set_title("Total Charges Distribution")
    ax.set_xlabel("Total Charges ($)")
    ax.legend()
    savefig(fig, "04_charges_distribution")

    # 5. Customer service calls distribution -------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="Customer service calls", hue=churn_label, data=df, ax=ax)
    ax.set_title("Customer Service Calls vs Churn")
    ax.legend(title="")
    savefig(fig, "05_customer_service_calls_distribution")

    # 6. Correlation heatmap -----------------------------------------------
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["_total_minutes", "_total_charges"])
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    savefig(fig, "06_correlation_heatmap")

    # 7. Boxplots churned vs non-churned -----------------------------------
    box_features = [
        "Total day minutes",
        "_total_charges",
        "Customer service calls",
        "Total intl minutes",
        "Account length",
        "Number vmail messages",
    ]
    titles = [
        "Day Minutes",
        "Total Charges",
        "Customer Service Calls",
        "Intl Minutes",
        "Account Length",
        "Voicemail Messages",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat, title in zip(axes.ravel(), box_features, titles):
        sns.boxplot(x=churn_label, y=df[feat], ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Churned vs Retained - Feature Distributions", fontsize=14)
    savefig(fig, "07_boxplots_churn_vs_retained")

    # --- answer the business questions ------------------------------------
    section("EDA INSIGHTS")
    insights = []

    heavy_cut = df["_total_minutes"].quantile(0.9)
    heavy = df[df["_total_minutes"] >= heavy_cut]
    insights.append(
        f"Heavy users (top 10% by total minutes, >= {heavy_cut:.0f} min): "
        f"{len(heavy)} customers, avg charge ${heavy['_total_charges'].mean():.2f}, "
        f"churn {heavy[config.TARGET].mean():.1%} vs {churn_rate:.1%} overall."
    )

    pay_cut = df["_total_charges"].quantile(0.9)
    top_pay = df[df["_total_charges"] >= pay_cut]
    insights.append(
        f"Top payers (top 10% by charges, >= ${pay_cut:.2f}): churn "
        f"{top_pay[config.TARGET].mean():.1%}; day-minutes drive most of the bill "
        f"(day charge corr with total charge = "
        f"{df['Total day charge'].corr(df['_total_charges']):.2f})."
    )

    freq_care = df[df["Customer service calls"] >= 4]
    insights.append(
        f"Frequent care callers (>=4 calls): {len(freq_care)} customers, churn "
        f"{freq_care[config.TARGET].mean():.1%} - a major red flag vs {churn_rate:.1%} overall."
    )

    intl_churn = df.groupby("International plan")[config.TARGET].mean()
    insights.append(
        "Churn drivers: international-plan holders churn "
        f"{intl_churn.get(1, float('nan')):.1%} vs {intl_churn.get(0, float('nan')):.1%} without; "
        "high day usage and >=4 service calls are the strongest churn signals."
    )

    for line in insights:
        print(" - " + line)

    # persist insights to a markdown report
    report = config.REPORT_DIR / "phase1_eda_insights.md"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# Phase 1 - EDA Insights\n\n")
        fh.write(f"- Customers analysed: **{len(df):,}**\n")
        fh.write(f"- Overall churn rate: **{churn_rate:.1%}**\n\n")
        fh.write("## Business questions\n\n")
        for line in insights:
            fh.write(f"- {line}\n")
    print(f"\n[report] {report.relative_to(config.PROJECT_ROOT)}")
    return df


if __name__ == "__main__":
    run()
