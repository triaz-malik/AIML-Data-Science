"""Exploratory Data Analysis — saves figures to outputs/figures/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/CI
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid")


def _save(fig, name: str):
    path = C.FIGURE_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.ROOT_DIR)}")


def run_eda(train: pd.DataFrame):
    """Generate the full EDA figure set from the raw (pre-clean) train frame."""
    print("Running EDA...")

    # 1. Target distribution: raw vs log1p ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(train[C.TARGET], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"SalePrice (skew={train[C.TARGET].skew():.2f})")
    logged = np.log1p(train[C.TARGET])
    sns.histplot(logged, kde=True, ax=axes[1], color="seagreen")
    axes[1].set_title(f"log1p(SalePrice) (skew={logged.skew():.2f})")
    _save(fig, "01_target_distribution.png")

    # 2. Correlation heatmap (top features vs SalePrice) -------------------
    num = train.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)
    top = corr[C.TARGET].abs().sort_values(ascending=False).head(12).index
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num[top].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                square=True, ax=ax)
    ax.set_title("Top-12 feature correlations")
    _save(fig, "02_correlation_heatmap.png")

    # 3. SalePrice vs GrLivArea (outliers visible) -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=train, x="GrLivArea", y=C.TARGET, alpha=0.5, ax=ax)
    ax.set_title("SalePrice vs GrLivArea (note bottom-right outliers)")
    _save(fig, "03_saleprice_vs_grlivarea.png")

    # 4. Neighbourhood boxplot ---------------------------------------------
    order = (train.groupby("Neighborhood")[C.TARGET].median()
             .sort_values().index)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=train, x="Neighborhood", y=C.TARGET, order=order, ax=ax)
    ax.set_title("SalePrice by Neighborhood (sorted by median)")
    ax.tick_params(axis="x", rotation=90)
    _save(fig, "04_neighborhood_boxplot.png")

    # 5. House age vs price -------------------------------------------------
    tmp = train.copy()
    tmp["HouseAge"] = tmp["YrSold"] - tmp["YearBuilt"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=tmp, x="HouseAge", y=C.TARGET, alpha=0.4, ax=ax)
    sns.regplot(data=tmp, x="HouseAge", y=C.TARGET, scatter=False,
                color="red", ax=ax)
    ax.set_title("SalePrice vs HouseAge (newer homes cost more)")
    _save(fig, "05_house_age.png")

    # 6. Missing-value bar chart -------------------------------------------
    miss = (train.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(miss))))
        sns.barplot(x=miss.values, y=miss.index, ax=ax, color="indianred")
        ax.set_xlabel("% missing")
        ax.set_title("Missing values by column")
        _save(fig, "06_missing_values.png")

    print("EDA complete.\n")
