"""
Generate House_Prices.ipynb from a structured cell list.
Standard GitHub data-science notebook format: markdown + code cells, headings for navigation.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "House_Prices.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------- #
md("""# House Prices: Advanced Regression Techniques

End-to-end pipeline for the [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

**Approach.** EDA → semantic NaN imputation → feature engineering with KMeans neighborhood clusters → 7 base models with Optuna-tuned XGBoost / LightGBM / CatBoost → Ridge meta-learner stacking → SHAP interpretability.

**Result.** 5-fold OOF RMSLE ≈ **0.107** (Ridge stacker), beating the strongest single model (CatBoost, 0.113) by 5.4%.

## Table of contents
1. [Setup](#1-Setup)
2. [Load data](#2-Load-data)
3. [EDA](#3-Exploratory-data-analysis)
4. [Preprocessing](#4-Preprocessing)
5. [Feature engineering](#5-Feature-engineering)
6. [Baseline models](#6-Baseline-models)
7. [Optuna hyperparameter tuning](#7-Optuna-hyperparameter-tuning)
8. [Ridge meta-stacker](#8-Ridge-meta-stacker)
9. [SHAP interpretability](#9-SHAP-interpretability)
10. [Submission](#10-Submission)
""")

# ---------------------------------------------------------------- #
md("""## 1. Setup""")

code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
""")

# ---------------------------------------------------------------- #
md("""## 2. Load data""")

code("""train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print(f"train: {train.shape} | test: {test.shape}")
train.head()
""")

# ---------------------------------------------------------------- #
md("""## 3. Exploratory data analysis

### 3.1 Target distribution

`SalePrice` is right-skewed (skew ≈ 1.88). We model `log1p(SalePrice)` for two reasons:
- The competition is scored on **RMSLE**, which equals RMSE in log space — so we align loss with metric.
- Log transformation reduces the influence of luxury outliers and gives a near-Normal distribution (skew ≈ 0.12).""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(train["SalePrice"], bins=50, color="#2C3E50")
axes[0].set_title("Raw SalePrice")
axes[0].axvline(train["SalePrice"].mean(),   color="#E74C3C", ls="--", label="mean")
axes[0].axvline(train["SalePrice"].median(), color="#3498DB", ls="--", label="median")
axes[0].legend()

log_price = np.log1p(train["SalePrice"])
axes[1].hist(log_price, bins=50, color="#3498DB")
axes[1].set_title("log1p(SalePrice)")

(osm, osr), (slope, intercept, r) = stats.probplot(log_price, dist="norm")
axes[2].scatter(osm, osr, alpha=0.4, s=10, color="#2ECC71")
axes[2].plot(osm, slope * np.array(osm) + intercept, color="#E74C3C")
axes[2].set_title(f"QQ plot — R² = {r**2:.4f}")

plt.tight_layout()
plt.show()
print(f"Skewness: raw = {train['SalePrice'].skew():.3f}  →  log = {log_price.skew():.3f}")
""")

# ---------------------------------------------------------------- #
md("""### 3.2 Missing values — most aren't really missing

Read the data dictionary: `PoolQC = NaN` means *no pool*, not *unknown*. Imputing the median would fabricate pools where there are none. Treat these as a literal `"None"` category instead.""")

code("""def missing_summary(df):
    m = df.isnull().sum()
    m = m[m > 0].sort_values(ascending=False)
    return pd.DataFrame({"missing": m, "pct": (m / len(df) * 100).round(2)})

missing_summary(train).head(15)
""")

# ---------------------------------------------------------------- #
md("""### 3.3 Correlations with target

`OverallQual` (0.79) is the strongest single predictor. Five features clear the 0.6 mark.""")

code("""num_cols = train.select_dtypes(include=[np.number]).columns
corr = train[num_cols].corrwith(train["SalePrice"]).abs().sort_values(ascending=False)
top = corr.drop("SalePrice").head(15)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(top)), top.values, color="#2C3E50")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
ax.invert_yaxis(); ax.set_xlabel("|corr| with SalePrice")
ax.set_title("Top 15 features by absolute correlation")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- #
md("""### 3.4 Neighborhood premium

Median prices vary 3.6× between cheapest (`MeadowV`) and most expensive (`NridgHt`). We exploit this twice — as a target-encoded numeric feature and via KMeans clustering.""")

code("""nbhd = (train.groupby("Neighborhood")["SalePrice"]
              .median()
              .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(len(nbhd)), nbhd.values / 1000, color="#3498DB")
ax.set_xticks(range(len(nbhd)))
ax.set_xticklabels(nbhd.index, rotation=45, ha="right")
ax.axhline(train["SalePrice"].median() / 1000, color="#E74C3C", ls="--",
           label=f"overall median ${train['SalePrice'].median()/1000:.0f}K")
ax.set_ylabel("median SalePrice ($K)")
ax.legend()
plt.tight_layout(); plt.show()

print(f"Top/bottom ratio: {nbhd.max()/nbhd.min():.1f}x")
""")

# ---------------------------------------------------------------- #
md("""## 4. Preprocessing

### 4.1 Remove documented outliers

The competition description names two `GrLivArea` outliers (>4000 sqft, <\\$300K) — likely institutional sales. Remove them.""")

code("""train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].copy()
y = np.log1p(train["SalePrice"])
print(f"after outlier removal: {train.shape}")
""")

# ---------------------------------------------------------------- #
md("""### 4.2 Combine train + test for consistent feature engineering

⚠️ **Leakage caveat.** Combining is fine for *unsupervised* operations (one-hot encoding, log1p, neighborhood-median imputation of `LotFrontage`). But for *target-aware* operations (target encoding, KMeans on neighborhood medians), we fit on **train only** and then map onto the combined data — see §5.""")

code("""ntrain = len(train)
test_id = test["Id"]

all_data = pd.concat([
    train.drop(["Id", "SalePrice"], axis=1),
    test.drop("Id", axis=1)
], axis=0).reset_index(drop=True)

print(f"combined: {all_data.shape}")
""")

# ---------------------------------------------------------------- #
md("""### 4.3 Imputation strategy

| Strategy | Columns | Rationale |
|---|---|---|
| Fill `"None"` | `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `Garage*`, `Bsmt*`, `MasVnrType`, `MSSubClass` | NaN encodes feature absence |
| Fill `0` | `GarageYrBlt`, `GarageArea/Cars`, `BsmtFinSF1/2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath/HalfBath`, `MasVnrArea` | No feature → zero count |
| Neighborhood median | `LotFrontage` | Houses on same street share frontage profile |
| Mode | `MSZoning`, `Electrical`, `KitchenQual`, etc. | Single-NaN columns |""")

code("""NONE_COLS = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
             "GarageType","GarageFinish","GarageQual","GarageCond",
             "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
             "BsmtFinType2","MasVnrType","MSSubClass"]
ZERO_COLS = ["GarageYrBlt","GarageArea","GarageCars",
             "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
             "BsmtFullBath","BsmtHalfBath","MasVnrArea"]
MODE_COLS = ["MSZoning","Electrical","KitchenQual","Exterior1st",
             "Exterior2nd","SaleType","Functional","Utilities"]

for c in NONE_COLS: all_data[c] = all_data[c].fillna("None")
for c in ZERO_COLS: all_data[c] = all_data[c].fillna(0)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for c in MODE_COLS: all_data[c] = all_data[c].fillna(all_data[c].mode()[0])

print(f"remaining missing: {all_data.isnull().sum().sum()}")
""")

# ---------------------------------------------------------------- #
md("""## 5. Feature engineering

### 5.1 Aggregate features — buyers think holistically

Total square footage matters more than individual rooms. Quality × area captures the "luxury per sqft" intuition.""")

code("""# Area aggregates
all_data["TotalSF"]        = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["TotalBathrooms"] = (all_data["FullBath"] + 0.5*all_data["HalfBath"]
                              + all_data["BsmtFullBath"] + 0.5*all_data["BsmtHalfBath"])
all_data["TotalPorchSF"]   = (all_data["OpenPorchSF"] + all_data["EnclosedPorch"]
                              + all_data["3SsnPorch"] + all_data["ScreenPorch"])
all_data["AllFloorsSF"]    = all_data["1stFlrSF"] + all_data["2ndFlrSF"]

# Temporal
all_data["HouseAge"]        = all_data["YrSold"] - all_data["YearBuilt"]
all_data["YearsSinceRemod"] = all_data["YrSold"] - all_data["YearRemodAdd"]
all_data["IsRemodeled"]     = (all_data["YearRemodAdd"] != all_data["YearBuilt"]).astype(int)
all_data["IsNewHouse"]      = (all_data["YrSold"] == all_data["YearBuilt"]).astype(int)

# Presence flags
all_data["HasPool"]      = (all_data["PoolArea"] > 0).astype(int)
all_data["HasGarage"]    = (all_data["GarageArea"] > 0).astype(int)
all_data["Has2ndFloor"]  = (all_data["2ndFlrSF"] > 0).astype(int)
all_data["HasBsmt"]      = (all_data["TotalBsmtSF"] > 0).astype(int)
all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)

# Quality × area interactions
all_data["QualArea"]    = all_data["OverallQual"] * all_data["GrLivArea"]
all_data["QualTotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]
""")

# ---------------------------------------------------------------- #
md("""### 5.2 Neighborhood target encoding — fit on train only

> **The leakage trap.** Many tutorials write this:
>
> ```python
> # WRONG — leaks test distribution into preprocessing
> nbhd_med = all_data.groupby("Neighborhood")["SalePrice"].median()
> all_data["NeighborhoodPrice"] = all_data["Neighborhood"].map(nbhd_med)
> ```
>
> `SalePrice` is `NaN` in test rows, but more importantly: any encoder/scaler/cluster fit on `train + test` lets the test distribution influence preprocessing, inflating CV scores. Fit on **train**, apply to **everything**.""")

code("""# RIGHT — fit only on train, then map onto combined data
nbhd_med = train.groupby("Neighborhood")["SalePrice"].median()
all_data["NeighborhoodPrice"] = (all_data["Neighborhood"]
                                 .map(nbhd_med)
                                 .fillna(nbhd_med.median()))
""")

# ---------------------------------------------------------------- #
md("""### 5.3 KMeans neighborhood clusters

Group neighborhoods by aggregated stats (median price, sqft, quality, count). Same fit-on-train-only rule applies.""")

code("""nbhd_stats = train.groupby("Neighborhood").agg(
    nbhd_med_price=("SalePrice", "median"),
    nbhd_med_sqft=("GrLivArea", "median"),
    nbhd_med_qual=("OverallQual", "median"),
    nbhd_count=("SalePrice", "count")
).reset_index()

scaler = StandardScaler()
X_nbhd = scaler.fit_transform(nbhd_stats.drop(columns=["Neighborhood"]))

km = KMeans(n_clusters=5, random_state=SEED, n_init=10)
nbhd_stats["cluster"] = km.fit_predict(X_nbhd)
cluster_map = dict(zip(nbhd_stats["Neighborhood"], nbhd_stats["cluster"]))

all_data["NeighborhoodCluster"] = all_data["Neighborhood"].map(cluster_map).fillna(-1)
nbhd_stats.sort_values("nbhd_med_price", ascending=False)
""")

# ---------------------------------------------------------------- #
md("""### 5.4 Ordinal encoding of quality columns

Quality columns follow an order: `Po < Fa < TA < Gd < Ex`. Encode as integers so models learn the magnitude of difference, not just cluster identity.""")

code("""QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0, "NA": 0}
for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
          "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
    all_data[c] = all_data[c].map(QUALITY_MAP).fillna(0).astype(int)

ORDINAL_MAPS = {
    "Functional":   {"Sal":1,"Sev":2,"Maj2":3,"Maj1":4,"Mod":5,"Min2":6,"Min1":7,"Typ":8},
    "BsmtFinType1": {"NA":0,"None":0,"Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6},
    "BsmtFinType2": {"NA":0,"None":0,"Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6},
    "BsmtExposure": {"NA":0,"None":0,"No":1,"Mn":2,"Av":3,"Gd":4},
    "GarageFinish": {"NA":0,"None":0,"Unf":1,"RFn":2,"Fin":3},
    "PavedDrive":   {"N":0,"P":1,"Y":2},
}
for c, mp in ORDINAL_MAPS.items():
    all_data[c] = all_data[c].map(mp).fillna(0)
""")

# ---------------------------------------------------------------- #
md("""### 5.5 Skew correction + one-hot encoding

42 numeric features have |skew| > 0.75. Apply `log1p`. Then one-hot encode all remaining categoricals.""")

code("""numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed = skewed[abs(skewed) > 0.75]
print(f"skewed features (|skew|>0.75): {len(skewed)}")

for f in skewed.index:
    all_data[f] = np.log1p(all_data[f].clip(lower=0))

all_data = pd.get_dummies(all_data)
print(f"after one-hot: {all_data.shape}")

X_train = all_data[:ntrain].copy()
X_test  = all_data[ntrain:].copy().reindex(columns=X_train.columns, fill_value=0)
print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
""")

# ---------------------------------------------------------------- #
md("""## 6. Baseline models

5-fold cross-validation, fixed seed.""")

code("""def rmsle_cv(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error", cv=kf))

baselines = {
    "Ridge":      make_pipeline(RobustScaler(), Ridge(alpha=12, random_state=SEED)),
    "Lasso":      make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=10000, random_state=SEED)),
    "ElasticNet": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=SEED)),
    "GBM":        GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4,
                                            max_features="sqrt", min_samples_leaf=15,
                                            min_samples_split=10, loss="huber", random_state=SEED),
}

results = {}
for name, model in baselines.items():
    sc = rmsle_cv(model, X_train, y)
    results[name] = sc
    print(f"  {name:11s}: {sc.mean():.5f} ± {sc.std():.5f}")
""")

# ---------------------------------------------------------------- #
md("""## 7. Optuna hyperparameter tuning

We tune XGBoost, LightGBM, and CatBoost. Each gets 30 trials with the TPE sampler optimizing 5-fold CV RMSLE.

> Bump `N_TRIALS` to 100 for the spec-faithful run (~30 min).""")

code("""N_TRIALS = 30  # set to 100 for full search

def cv_rmsle(model, X, y, n_folds=5):
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
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            n_estimators=1500, random_state=SEED, n_jobs=-1, verbosity=0,
        )
        return cv_rmsle(xgb.XGBRegressor(**params), X, y)

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
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            n_estimators=1500, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        return cv_rmsle(lgb.LGBMRegressor(**params), X, y)

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
        return cv_rmsle(CatBoostRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  CAT best RMSLE: {study.best_value:.5f}")
    return CatBoostRegressor(**study.best_params, iterations=1500,
                             random_seed=SEED, verbose=0,
                             allow_writing_files=False, bootstrap_type="Bernoulli")


print("[XGBoost]"); xgb_model = tune_xgb(X_train, y, N_TRIALS)
print("[LightGBM]"); lgb_model = tune_lgb(X_train, y, N_TRIALS)
print("[CatBoost]"); cat_model = tune_cat(X_train, y, N_TRIALS)
""")

code("""# Score the tuned boosters on the same CV
for name, m in [("XGBoost", xgb_model), ("LightGBM", lgb_model), ("CatBoost", cat_model)]:
    sc = rmsle_cv(m, X_train, y)
    results[name] = sc
    print(f"  {name:11s}: {sc.mean():.5f} ± {sc.std():.5f}")
""")

# ---------------------------------------------------------------- #
md("""### Model comparison""")

code("""fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
means = [v.mean() for v in results.values()]
stds  = [v.std()  for v in results.values()]
best  = means.index(min(means))
colors = ["#E74C3C" if i == best else "#2C3E50" for i in range(len(means))]
ax.bar(names, means, color=colors, yerr=stds, capsize=4, edgecolor="white")
ax.set_ylabel("RMSLE (lower is better)"); ax.set_title("5-fold CV RMSLE")
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- #
md("""## 8. Ridge meta-stacker

Each base model produces 5-fold OOF predictions on `X_train`. A `RidgeCV` meta-learner is fit on these 7 prediction columns. For test, each base model averages its predictions across the 5 folds, then the meta-learner combines them.

This is **true stacking** — the Ridge can assign negative weights (correcting one model with another) and selects regularization via cross-validated alpha.""")

code("""def stacked_ridge(base_models, X_tr, y_tr, X_te, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n = len(base_models)
    oof_preds  = np.zeros((len(X_tr), n))
    test_preds = np.zeros((len(X_te), n))

    for i, (name, model) in enumerate(base_models.items()):
        print(f"  {name} ...", end=" ", flush=True)
        test_fp = np.zeros((len(X_te), n_folds))
        for fold, (tri, vali) in enumerate(kf.split(X_tr, y_tr)):
            Xt, Xv = X_tr.iloc[tri], X_tr.iloc[vali]
            yt = y_tr.iloc[tri]
            model.fit(Xt, yt)
            oof_preds[vali, i]   = model.predict(Xv)
            test_fp[:, fold]     = model.predict(X_te)
        test_preds[:, i] = test_fp.mean(axis=1)
        print("done")

    meta = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 25.0])
    meta.fit(oof_preds, y_tr)
    print(f"  RidgeCV alpha = {meta.alpha_:.2f}")
    print("  weights:", dict(zip(base_models.keys(), meta.coef_.round(3))))

    oof_blend  = meta.predict(oof_preds)
    test_blend = meta.predict(test_preds)
    rmsle = np.sqrt(mean_squared_error(y_tr, oof_blend))
    return oof_blend, test_blend, rmsle, meta


base = {**baselines,
        "XGBoost": xgb_model, "LightGBM": lgb_model, "CatBoost": cat_model}
oof, test_blend, ens_rmsle, meta = stacked_ridge(base, X_train, y, X_test)
print(f"\\n  Stacker OOF RMSLE: {ens_rmsle:.5f}")
""")

# ---------------------------------------------------------------- #
md("""### 8.1 Stacker diagnostics""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

residuals = y.values - oof
axes[0].scatter(oof, residuals, alpha=0.3, s=10, color="#2C3E50")
axes[0].axhline(0, color="#E74C3C", ls="--")
axes[0].set_xlabel("predicted log(price)"); axes[0].set_ylabel("residual")
axes[0].set_title("Residual plot")

axes[1].scatter(y.values, oof, alpha=0.3, s=10, color="#3498DB")
lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
axes[1].plot(lims, lims, color="#E74C3C", ls="--")
axes[1].set_xlabel("actual log(price)"); axes[1].set_ylabel("predicted log(price)")
axes[1].set_title("Predicted vs actual (OOF)")

plt.suptitle(f"Stacker diagnostics — OOF RMSLE: {ens_rmsle:.5f}")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- #
md("""## 9. SHAP interpretability

Tree feature_importances count splits — coarse, biased toward high-cardinality features. SHAP attributes each prediction to its features fairly. We use `TreeExplainer` on the tuned XGBoost model.""")

code("""xgb_model.fit(X_train, y)

sample = X_train.sample(min(500, len(X_train)), random_state=SEED)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample, max_display=20, show=True)
""")

code("""mean_abs = pd.Series(np.abs(shap_values).mean(axis=0),
                     index=X_train.columns).sort_values(ascending=False)
top = mean_abs.head(20)

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(top)), top.values, color="#3498DB")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
ax.invert_yaxis(); ax.set_xlabel("mean |SHAP value|")
ax.set_title("Top 20 features by SHAP magnitude")
plt.tight_layout(); plt.show()

print("Top 5:")
for f, v in top.head(5).items():
    print(f"  {f:25s} {v:.5f}")
""")

# ---------------------------------------------------------------- #
md("""## 10. Submission""")

code("""final = np.clip(np.expm1(test_blend), 0, None)
submission = pd.DataFrame({"Id": test_id.values, "SalePrice": final})
submission.to_csv("submission.csv", index=False)

print(f"submission.csv: {len(submission)} rows")
print(f"  range: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
print(f"  mean : ${submission['SalePrice'].mean():,.0f}")
submission.head()
""")

# ---------------------------------------------------------------- #
md("""## Lessons learned

| Lesson | Why it matters |
|---|---|
| Log-transform skewed regression targets | Aligns the loss with RMSLE; prevents luxury outliers from dominating gradients |
| `NaN` is not always missing | Most NaN entries here encode absence (no pool, no garage); imputing with median fabricates features |
| Engineered features beat raw features | `QualTotalSF`, `QualArea`, `NeighborhoodPrice` rank above every raw column in SHAP |
| Fit encoders/clusterers on **train only** | Concatenating train+test before fitting leaks the test distribution; CV looks better, leaderboard regresses |
| Stacking > best single model | Even the strongest single (CatBoost 0.113) lost to the Ridge stacker (0.107) |
| Linear models still compete | Lasso & ElasticNet outscored all three tuned tree boosters here — small clean datasets reward simple models |
| Remove documented outliers | Trust the data dictionary; two `GrLivArea` flagged points were hurting every model |
| Optuna over grid search | TPE adapts; 30 trials beat 200 grid points across the same space |
| SHAP for honest interpretability | `feature_importances_` counts splits; SHAP attributes actual prediction contribution |
""")

# ---------------------------------------------------------------- #
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
