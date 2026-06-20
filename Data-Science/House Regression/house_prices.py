"""
House Prices: Advanced Regression Techniques — full pipeline.

Stages:
  Act I    EDA — target skew, missingness, correlations, key features, neighborhood
  Act II   Preprocessing — outlier removal, semantic NaN imputation
  Act III  Feature engineering — totals, ages, flags, KMeans neighborhood clusters
  Act IV   Models — Lasso/Ridge/ElasticNet baselines + tuned XGBoost/LightGBM/CatBoost
  Act V    Ridge meta-stacker on 5-fold OOF predictions
  Act VI   SHAP interpretability + Streamlit artifact export

Run:
    python house_prices.py                 # full pipeline (default 30 Optuna trials/model)
    python house_prices.py --trials 100    # spec-faithful 100 trials per model
    python house_prices.py --quick         # skip Optuna, use reference hyperparameters
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import optuna
import shap

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
FIG_DIR = ROOT / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
FIG_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

PALETTE = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC", "axes.grid": True, "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def banner(t: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{t}\n{bar}")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_data():
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        sys.exit(f"Missing train.csv or test.csv in {ROOT}")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test


# --------------------------------------------------------------------------- #
# Act I — EDA
# --------------------------------------------------------------------------- #
def eda_target(train):
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


def eda_missing(train, test):
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


def eda_correlation(train):
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


def eda_key_features(train):
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


def eda_neighborhood(train):
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
# Act II — Preprocessing
# --------------------------------------------------------------------------- #
NONE_COLS = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
             "GarageType", "GarageFinish", "GarageQual", "GarageCond",
             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
             "BsmtFinType2", "MasVnrType", "MSSubClass"]
ZERO_COLS = ["GarageYrBlt", "GarageArea", "GarageCars",
             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
             "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]
MODE_COLS = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st",
             "Exterior2nd", "SaleType", "Functional", "Utilities"]

QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0, "NA": 0}
ORDINAL_MAPS = {
    "Functional":   {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                     "Min2": 6, "Min1": 7, "Typ": 8},
    "BsmtFinType1": {"NA": 0, "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3,
                     "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"NA": 0, "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3,
                     "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtExposure": {"NA": 0, "None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "GarageFinish": {"NA": 0, "None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
}
QUALITY_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]


def preprocess(train, test):
    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].copy()
    y = np.log1p(train["SalePrice"])
    print(f"After outlier removal: {train.shape}")

    ntrain = train.shape[0]
    test_id = test["Id"]
    all_data = pd.concat([
        train.drop(["Id", "SalePrice"], axis=1),
        test.drop("Id", axis=1)], axis=0).reset_index(drop=True)

    for c in NONE_COLS: all_data[c] = all_data[c].fillna("None")
    for c in ZERO_COLS: all_data[c] = all_data[c].fillna(0)
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for c in MODE_COLS: all_data[c] = all_data[c].fillna(all_data[c].mode()[0])
    print(f"Remaining missing: {all_data.isnull().sum().sum()}")
    return train, y, all_data, ntrain, test_id


# --------------------------------------------------------------------------- #
# Act III — Feature engineering (with KMeans neighborhood clusters)
# --------------------------------------------------------------------------- #
def engineer(all_data, train, n_clusters=5):
    all_data["TotalSF"] = (all_data["TotalBsmtSF"] + all_data["1stFlrSF"]
                           + all_data["2ndFlrSF"])
    all_data["TotalBathrooms"] = (all_data["FullBath"] + 0.5 * all_data["HalfBath"]
                                  + all_data["BsmtFullBath"]
                                  + 0.5 * all_data["BsmtHalfBath"])
    all_data["TotalPorchSF"] = (all_data["OpenPorchSF"] + all_data["EnclosedPorch"]
                                + all_data["3SsnPorch"] + all_data["ScreenPorch"])
    all_data["AllFloorsSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
    all_data["YearsSinceRemod"] = all_data["YrSold"] - all_data["YearRemodAdd"]
    all_data["IsRemodeled"] = (all_data["YearRemodAdd"] != all_data["YearBuilt"]).astype(int)
    all_data["IsNewHouse"] = (all_data["YrSold"] == all_data["YearBuilt"]).astype(int)
    all_data["HasPool"] = (all_data["PoolArea"] > 0).astype(int)
    all_data["HasGarage"] = (all_data["GarageArea"] > 0).astype(int)
    all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] > 0).astype(int)
    all_data["HasBsmt"] = (all_data["TotalBsmtSF"] > 0).astype(int)
    all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)
    all_data["QualArea"] = all_data["OverallQual"] * all_data["GrLivArea"]
    all_data["QualTotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]

    # Target encoding — fit on TRAIN ONLY, then map
    nbhd_med = train.groupby("Neighborhood")["SalePrice"].median()
    all_data["NeighborhoodPrice"] = (all_data["Neighborhood"]
                                     .map(nbhd_med).fillna(nbhd_med.median()))

    # KMeans on neighborhood-level aggregated stats (fit on train only -> no leakage)
    nbhd_stats = train.groupby("Neighborhood").agg(
        nbhd_med_price=("SalePrice", "median"),
        nbhd_med_sqft=("GrLivArea", "median"),
        nbhd_med_qual=("OverallQual", "median"),
        nbhd_count=("SalePrice", "count")).reset_index()
    scaler = StandardScaler()
    X_nbhd = scaler.fit_transform(nbhd_stats[["nbhd_med_price", "nbhd_med_sqft",
                                              "nbhd_med_qual", "nbhd_count"]])
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    nbhd_stats["NeighborhoodCluster"] = km.fit_predict(X_nbhd)
    cluster_map = dict(zip(nbhd_stats["Neighborhood"], nbhd_stats["NeighborhoodCluster"]))
    all_data["NeighborhoodCluster"] = all_data["Neighborhood"].map(cluster_map).fillna(-1)
    print(f"KMeans neighborhood clusters: {n_clusters} groups assigned")

    # Ordinal encodings
    for c in QUALITY_COLS:
        all_data[c] = all_data[c].map(QUALITY_MAP).fillna(0).astype(int)
    for c, mp in ORDINAL_MAPS.items():
        all_data[c] = all_data[c].map(mp).fillna(0)

    # Skew correction
    numeric_feats = all_data.select_dtypes(include=[np.number]).columns.tolist()
    skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed = skewed[abs(skewed) > 0.75]
    print(f"Features with |skew| > 0.75: {len(skewed)}")
    for f in skewed.index:
        all_data[f] = np.log1p(all_data[f].clip(lower=0))

    all_data = pd.get_dummies(all_data)
    print(f"After one-hot: {all_data.shape}")

    # Save preprocessing artifacts for Streamlit
    artifacts = {
        "nbhd_med": nbhd_med.to_dict(),
        "nbhd_med_default": float(nbhd_med.median()),
        "cluster_map": cluster_map,
        "skewed_features": skewed.index.tolist(),
        "feature_columns": all_data.columns.tolist(),
    }
    return all_data, artifacts


# --------------------------------------------------------------------------- #
# Act IV — Models, baselines, Optuna tuning
# --------------------------------------------------------------------------- #
def rmsle_cv(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error", cv=kf))


def baseline_models():
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=12, random_state=SEED))
    lasso = make_pipeline(RobustScaler(),
                          Lasso(alpha=0.0005, max_iter=10000, random_state=SEED))
    enet = make_pipeline(RobustScaler(),
                         ElasticNet(alpha=0.0005, l1_ratio=0.9,
                                    max_iter=10000, random_state=SEED))
    gbm = GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4,
        max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
        loss="huber", random_state=SEED)
    return {"Ridge": ridge, "Lasso": lasso, "ElasticNet": enet, "GBM": gbm}


# Optuna objectives -------------------------------------------------------- #
def _cv_rmsle(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []
    for tri, vali in kf.split(X):
        Xt, Xv = X.iloc[tri], X.iloc[vali]
        yt, yv = y.iloc[tri], y.iloc[vali]
        model.fit(Xt, yt)
        scores.append(np.sqrt(mean_squared_error(yv, model.predict(Xv))))
    return float(np.mean(scores))


def tune_xgb(X, y, n_trials):
    def obj(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 5.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            n_estimators=1500, random_state=SEED, n_jobs=-1, verbosity=0,
        )
        return _cv_rmsle(xgb.XGBRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGB best RMSLE: {study.best_value:.5f}")
    return xgb.XGBRegressor(**study.best_params, n_estimators=1500,
                            random_state=SEED, n_jobs=-1, verbosity=0)


def tune_lgb(X, y, n_trials):
    def obj(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 5, 64),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            n_estimators=1500, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        return _cv_rmsle(lgb.LGBMRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  LGB best RMSLE: {study.best_value:.5f}")
    return lgb.LGBMRegressor(**study.best_params, n_estimators=1500,
                             random_state=SEED, n_jobs=-1, verbose=-1)


def tune_cat(X, y, n_trials):
    def obj(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            iterations=1500, random_seed=SEED, verbose=0,
            allow_writing_files=False, bootstrap_type="Bernoulli",
        )
        return _cv_rmsle(CatBoostRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  CAT best RMSLE: {study.best_value:.5f}")
    return CatBoostRegressor(**study.best_params, iterations=1500,
                             random_seed=SEED, verbose=0,
                             allow_writing_files=False, bootstrap_type="Bernoulli")


def reference_boosters():
    """Reference hyperparameters for --quick mode."""
    xgb_m = xgb.XGBRegressor(
        colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
        max_depth=3, min_child_weight=1.7817, n_estimators=2200,
        reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213,
        random_state=SEED, n_jobs=-1, verbosity=0)
    lgb_m = lgb.LGBMRegressor(
        objective="regression", num_leaves=5, learning_rate=0.05,
        n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5,
        feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9,
        min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
        random_state=SEED, verbose=-1)
    cat_m = CatBoostRegressor(
        iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        random_seed=SEED, verbose=0, allow_writing_files=False)
    return xgb_m, lgb_m, cat_m


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_model_comparison(results):
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


def plot_feature_importance(xgb_model, lgb_model, X_train):
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


def plot_diagnostics(y, oof, ens_rmsle, model_names):
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


def plot_submission_check(train, submission):
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


def plot_shap(xgb_model, X_train):
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


# --------------------------------------------------------------------------- #
# Act V — Ridge meta-stacker on OOF predictions
# --------------------------------------------------------------------------- #
def stacked_ridge(base_models, X_tr, y_tr, X_te, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n_models = len(base_models)
    oof_preds = np.zeros((len(X_tr), n_models))
    test_preds = np.zeros((len(X_te), n_models))

    for m_idx, (name, model) in enumerate(base_models.items()):
        print(f"  Training {name} ...", end=" ", flush=True)
        test_fp = np.zeros((len(X_te), n_folds))
        for fold, (tri, vali) in enumerate(kf.split(X_tr, y_tr)):
            Xt, Xv = X_tr.iloc[tri], X_tr.iloc[vali]
            yt = y_tr.iloc[tri]
            model.fit(Xt, yt)
            oof_preds[vali, m_idx] = model.predict(Xv)
            test_fp[:, fold] = model.predict(X_te)
        test_preds[:, m_idx] = test_fp.mean(axis=1)
        print("done")

    meta = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 25.0])
    meta.fit(oof_preds, y_tr)
    print(f"  RidgeCV alpha: {meta.alpha_:.2f} | weights: "
          + ", ".join(f"{n}={w:.2f}" for n, w in zip(base_models.keys(), meta.coef_)))

    oof_blend = meta.predict(oof_preds)
    test_blend = meta.predict(test_preds)
    rmsle = np.sqrt(mean_squared_error(y_tr, oof_blend))
    return oof_blend, test_blend, rmsle, meta, oof_preds, test_preds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="House Prices regression pipeline")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per booster (default 30, spec 100)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip Optuna; use reference hyperparameters")
    args = parser.parse_args()

    banner("LOAD")
    train_raw, test_raw = load_data()

    banner("ACT I — EDA")
    eda_target(train_raw)
    eda_missing(train_raw, test_raw)
    eda_correlation(train_raw)
    eda_key_features(train_raw)
    eda_neighborhood(train_raw)

    banner("ACT II — PREPROCESS")
    train, y, all_data, ntrain, test_id = preprocess(train_raw, test_raw)

    banner("ACT III — FEATURE ENGINEERING")
    all_data, fe_artifacts = engineer(all_data, train, n_clusters=5)
    X_train = all_data[:ntrain].copy()
    X_test = all_data[ntrain:].copy().reindex(columns=X_train.columns, fill_value=0)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    banner("ACT IV — BASELINES (5-fold CV)")
    results = {}
    for name, model in baseline_models().items():
        sc = rmsle_cv(model, X_train, y); results[name] = sc
        print(f"  {name:11s}: {sc.mean():.5f} +/- {sc.std():.5f}")

    banner("ACT IV — BOOSTER MODELS")
    if args.quick:
        print("  --quick mode: using reference hyperparameters (no Optuna)")
        xgb_model, lgb_model, cat_model = reference_boosters()
    else:
        print(f"  Running Optuna ({args.trials} trials per model)...")
        print("  [XGBoost]")
        xgb_model = tune_xgb(X_train, y, args.trials)
        print("  [LightGBM]")
        lgb_model = tune_lgb(X_train, y, args.trials)
        print("  [CatBoost]")
        cat_model = tune_cat(X_train, y, args.trials)

    for name, m in [("XGBoost", xgb_model), ("LightGBM", lgb_model), ("CatBoost", cat_model)]:
        sc = rmsle_cv(m, X_train, y); results[name] = sc
        print(f"  {name:11s}: {sc.mean():.5f} +/- {sc.std():.5f}")

    plot_model_comparison(results)

    # Fit XGB/LGB on full data for feature importance + SHAP
    xgb_model.fit(X_train, y); lgb_model.fit(X_train, y)
    plot_feature_importance(xgb_model, lgb_model, X_train)

    banner("ACT V — RIDGE META-STACKER")
    base = baseline_models()
    base.update({"XGBoost": xgb_model, "LightGBM": lgb_model, "CatBoost": cat_model})
    oof, test_blend, ens_rmsle, meta, oof_preds, test_preds = stacked_ridge(
        base, X_train, y, X_test)
    print(f"  Stacker OOF RMSLE: {ens_rmsle:.5f}")
    plot_diagnostics(y, oof, ens_rmsle, list(base.keys()))

    banner("ACT VI — SHAP INTERPRETABILITY")
    shap_top = plot_shap(xgb_model, X_train)
    print("Top 5 by SHAP magnitude:")
    for f, v in shap_top.head(5).items():
        print(f"  {f:25s}: {v:.5f}")

    banner("SUBMISSION")
    final = np.clip(np.expm1(test_blend), 0, None)
    submission = pd.DataFrame({"Id": test_id.values, "SalePrice": final})
    sub_csv = ROOT / "submission.csv"
    submission.to_csv(sub_csv, index=False)
    with zipfile.ZipFile(ROOT / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_csv, arcname="submission.csv")
    print(f"submission.csv: {len(submission)} rows")
    print(f"  range: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
    print(f"  mean : ${submission['SalePrice'].mean():,.0f}")
    plot_submission_check(train_raw, submission)

    banner("ARTIFACTS — for Streamlit app")
    train_defaults = train_raw.median(numeric_only=True).to_dict()
    train_modes = {c: train_raw[c].mode()[0]
                   for c in train_raw.select_dtypes(include="object").columns}
    artifacts = {
        "meta_model": meta,
        "base_models": base,
        "fe": fe_artifacts,
        "train_medians": train_defaults,
        "train_modes": train_modes,
        "train_min_year": int(train_raw["YearBuilt"].min()),
        "train_max_year": int(train_raw["YrSold"].max()),
        "neighborhoods": sorted(train_raw["Neighborhood"].unique().tolist()),
    }
    art_path = ARTIFACT_DIR / "model.pkl"
    joblib.dump(artifacts, art_path)
    print(f"  saved {art_path.relative_to(ROOT)}  ({art_path.stat().st_size/1e6:.1f} MB)")

    banner("SCORECARD")
    all_results = {**{k: v.mean() for k, v in results.items()}, "Stacker": ens_rmsle}
    for name, sc in sorted(all_results.items(), key=lambda kv: kv[1]):
        bar = "#" * int((0.15 - sc) * 1000)
        print(f"  {name:12s}: {sc:.5f}  {bar}")


if __name__ == "__main__":
    main()
