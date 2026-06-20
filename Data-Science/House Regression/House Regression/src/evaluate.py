"""Evaluation: cross-validation metrics and figure generation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from src.data_loader import FIG_DIR, ROOT

SEED = 42
PALETTE = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC", "axes.grid": True, "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def rmsle_cv(model, X, y, n_folds: int = 5) -> np.ndarray:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error", cv=kf))


# --------------------------------------------------------------------------- #
# EDA figures
# --------------------------------------------------------------------------- #
def eda_target(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SalePrice Distribution Analysis", fontsize=16, fontweight="bold")
    axes[0].hist(train["SalePrice"], bins=50, color=PALETTE[0], edgecolor="white", alpha=0.9)
    axes[0].set_title("Raw SalePrice")
    axes[0].axvline(train["SalePrice"].mean(), color=PALETTE[1], lw=2, ls="--",
                    label=f"Mean: ${train['SalePrice'].mean():,.0f}")
    axes[0].axvline(train["SalePrice"].median(), color=PALETTE[2], lw=2, ls="--",
                    label=f"Median: ${train['SalePrice'].median():,.0f}")
    axes[0].legend()
    log_price = np.log1p(train["SalePrice"])
    axes[1].hist(log_price, bins=50, color=PALETTE[2], edgecolor="white", alpha=0.9)
    axes[1].set_title("log(1 + SalePrice)")
    (osm, osr), (slope, intercept, r) = stats.probplot(log_price, dist="norm")
    axes[2].scatter(osm, osr, alpha=0.4, s=10, color=PALETTE[3])
    axes[2].plot(osm, slope * np.array(osm) + intercept, color=PALETTE[1], lw=2)
    axes[2].set_title(f"QQ Plot — R^2={r**2:.4f}")
    plt.tight_layout()
    save(fig, "01_target_distribution.png")
    print(f"Skewness: raw={train['SalePrice'].skew():.3f} -> log={log_price.skew():.3f}")


def eda_missing(train: pd.DataFrame, test: pd.DataFrame) -> None:
    def s(df):
        m = df.isnull().sum()
        m = m[m > 0].sort_values(ascending=False)
        return pd.DataFrame({"Missing": m, "Pct": (m / len(df) * 100).round(2)})

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, ms, label, color in zip(axes, [s(train), s(test)],
                                    ["Train", "Test"], [PALETTE[0], PALETTE[2]]):
        top = ms.head(20)
        ax.barh(range(len(top)), top["Pct"], color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index); ax.invert_yaxis()
        ax.set_title(f"{label} — Top Missing Features")
        for i, (val, pct) in enumerate(zip(top["Missing"], top["Pct"])):
            ax.text(pct + 0.3, i, f"{val} ({pct}%)", va="center", fontsize=8)
    plt.tight_layout()
    save(fig, "02_missing.png")


def eda_correlation(train: pd.DataFrame) -> None:
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    corr = train[num_cols].corrwith(train["SalePrice"]).abs().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    top_corr = corr.head(20).drop("SalePrice", errors="ignore")
    colors_c = [PALETTE[1] if train[c].corr(train["SalePrice"]) > 0 else PALETTE[0]
                for c in top_corr.index]
    axes[0].barh(range(len(top_corr)), top_corr.values, color=colors_c,
                 alpha=0.85, edgecolor="white")
    axes[0].set_yticks(range(len(top_corr))); axes[0].set_yticklabels(top_corr.index)
    axes[0].invert_yaxis(); axes[0].set_title("Feature Correlations with SalePrice")
    top15 = corr.head(16).index.tolist()
    heat = train[top15].corr()
    mask = np.triu(np.ones_like(heat, dtype=bool))
    sns.heatmap(heat, ax=axes[1], mask=mask, cmap="RdYlGn", annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, square=True, vmin=-1, vmax=1)
    axes[1].set_title("Pairwise Correlation — Top 15")
    plt.tight_layout()
    save(fig, "03_correlation.png")
    print("Top 5 correlators:")
    for feat, val in corr.drop("SalePrice", errors="ignore").head(5).items():
        print(f"  {feat:20s}: {val:.3f}")


def eda_key_features(train: pd.DataFrame) -> None:
    keys = {"OverallQual": "Overall Quality (1-10)",
            "GrLivArea": "Above-Grade Living Area (sqft)",
            "GarageCars": "Garage Capacity (cars)",
            "TotalBsmtSF": "Total Basement Area (sqft)",
            "1stFlrSF": "1st Floor Area (sqft)",
            "YearBuilt": "Year Built"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for i, (col, title) in enumerate(keys.items()):
        ax = axes[i]; x = train[col]
        ax.scatter(x, train["SalePrice"] / 1000, alpha=0.3, s=15,
                   c=train["OverallQual"], cmap="RdYlGn", vmin=1, vmax=10)
        m_v = x.notna()
        m, b = np.polyfit(x[m_v], train["SalePrice"][m_v] / 1000, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, color=PALETTE[1], lw=2.5, alpha=0.9)
        r_val = np.corrcoef(x[m_v], train["SalePrice"][m_v])[0, 1]
        ax.set_xlabel(title); ax.set_ylabel("SalePrice ($K)")
        ax.set_title(f"{title.split('(')[0].strip()} | r = {r_val:.3f}")
    plt.suptitle("Key Features vs SalePrice", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "04_key_features.png")


def eda_neighborhood(train: pd.DataFrame) -> None:
    nbhd = (train.groupby("Neighborhood")["SalePrice"]
                 .agg(["median", "mean", "count"]).sort_values("median", ascending=False))
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_colors = [PALETTE[0] if m / 1000 < 140 else PALETTE[2] if m / 1000 < 200
                  else PALETTE[1] for m in nbhd["median"]]
    ax.bar(range(len(nbhd)), nbhd["median"] / 1000, color=bar_colors,
           alpha=0.85, edgecolor="white")
    for i, (med, cnt) in enumerate(zip(nbhd["median"], nbhd["count"])):
        ax.text(i, med / 1000 + 2, f"n={cnt}", ha="center", va="bottom",
                fontsize=7, rotation=90)
    ax.set_xticks(range(len(nbhd)))
    ax.set_xticklabels(nbhd.index, rotation=45, ha="right")
    ax.set_ylabel("Median SalePrice ($K)")
    ax.set_title("Neighborhood Median Prices", fontsize=14)
    ax.axhline(train["SalePrice"].median() / 1000, color=PALETTE[1], lw=2, ls="--",
               label=f"Overall Median: ${train['SalePrice'].median()/1000:.0f}K")
    ax.legend()
    plt.tight_layout()
    save(fig, "05_neighborhood.png")


# --------------------------------------------------------------------------- #
# Model & diagnostics figures
# --------------------------------------------------------------------------- #
def plot_model_comparison(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    names, means = list(results.keys()), [v.mean() for v in results.values()]
    stds = [v.std() for v in results.values()]
    best = means.index(min(means))
    colors = [PALETTE[1] if i == best else PALETTE[0] for i in range(len(means))]
    bars = axes[0].bar(names, means, color=colors, alpha=0.85, edgecolor="white",
                       yerr=stds, capsize=5, error_kw={"elinewidth": 2, "ecolor": "#555"})
    for bar, mean in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{mean:.5f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("RMSLE (lower is better)")
    axes[0].set_title("5-Fold CV RMSLE")
    axes[0].set_ylim(0, max(means) * 1.2)
    axes[0].tick_params(axis="x", rotation=20)
    baseline = results["Ridge"].mean()
    improvements = [(baseline - v.mean()) / baseline * 100 for v in results.values()]
    colors2 = [PALETTE[2] if i > 0 else PALETTE[1] for i in improvements]
    axes[1].bar(names, improvements, color=colors2, alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="#888", lw=1.5, ls="--")
    axes[1].set_ylabel("% Improvement over Ridge")
    axes[1].set_title("Gain vs Ridge Baseline")
    axes[1].tick_params(axis="x", rotation=20)
    for bar, val in zip(axes[1].patches, improvements):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    save(fig, "06_model_comparison.png")


def plot_feature_importance(xgb_model, lgb_model, X_train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, model, name, color in zip(axes, [xgb_model, lgb_model],
                                      ["XGBoost", "LightGBM"], [PALETTE[0], PALETTE[2]]):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        top = importances.nlargest(20)
        engineered = ["TotalSF" in f or "Qual" in f or "Remod" in f or "Age" in f
                      or "Bath" in f or "Cluster" in f for f in top.index]
        bar_colors = [PALETTE[1] if e else color for e in engineered]
        ax.barh(range(len(top)), top.values, color=bar_colors,
                alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
        ax.invert_yaxis()
        ax.set_title(f"{name} — Top 20 (red = engineered)")
    plt.suptitle("Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "07_feature_importance.png")


def plot_diagnostics(y, oof, ens_rmsle: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    residuals = y.values - oof
    axes[0].scatter(oof, residuals, alpha=0.3, s=10, color=PALETTE[0])
    axes[0].axhline(0, color=PALETTE[1], lw=2, ls="--")
    axes[0].set_xlabel("Predicted log(Price)"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot")
    axes[1].scatter(y.values, oof, alpha=0.3, s=10, color=PALETTE[2])
    lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    axes[1].plot(lims, lims, color=PALETTE[1], lw=2, ls="--", label="Perfect")
    axes[1].set_xlabel("Actual log(Price)"); axes[1].set_ylabel("Predicted log(Price)")
    axes[1].set_title("Predicted vs Actual (OOF)"); axes[1].legend()
    plt.suptitle(f"Ridge Stacker Diagnostics — OOF RMSLE: {ens_rmsle:.5f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "08_diagnostics.png")


def plot_submission_check(train: pd.DataFrame, submission: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(train["SalePrice"] / 1000, bins=50, alpha=0.6,
                 color=PALETTE[0], label="Train actual", density=True)
    axes[0].hist(submission["SalePrice"] / 1000, bins=50, alpha=0.6,
                 color=PALETTE[2], label="Test predicted", density=True)
    axes[0].set_xlabel("SalePrice ($K)")
    axes[0].set_title("Train vs Predicted Distribution"); axes[0].legend()
    axes[1].scatter(range(len(submission)), submission["SalePrice"] / 1000,
                    alpha=0.4, s=8, color=PALETTE[0])
    axes[1].set_xlabel("Test sample index")
    axes[1].set_ylabel("Predicted SalePrice ($K)")
    axes[1].set_title("Predicted Test Prices")
    plt.tight_layout()
    save(fig, "09_submission_check.png")


def plot_shap(xgb_model, X_train: pd.DataFrame) -> pd.Series:
    sample_size = min(500, len(X_train))
    Xs = X_train.sample(sample_size, random_state=SEED)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(Xs)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, Xs, max_display=20, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Summary — Tuned XGBoost", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "10_shap_summary.png")

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0),
                         index=X_train.columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(mean_abs)), mean_abs.values, color=PALETTE[2],
            alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(mean_abs))); ax.set_yticklabels(mean_abs.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features by SHAP Magnitude")
    plt.tight_layout()
    save(fig, "11_shap_top20.png")
    return mean_abs


def cv_rmsle(model, X, y, n_folds: int = 5) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []
    for tri, vali in kf.split(X):
        Xt, Xv = X.iloc[tri], X.iloc[vali]
        yt, yv = y.iloc[tri], y.iloc[vali]
        model.fit(Xt, yt)
        scores.append(np.sqrt(mean_squared_error(yv, model.predict(Xv))))
    return float(np.mean(scores))
