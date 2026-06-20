"""Preprocessing + feature engineering for the House Prices dataset.

Contains all transformations needed by both training (`src.train`) and
inference (`src.predict`, the Streamlit app):
- semantic NaN handling (None / 0 / neighborhood-median / mode)
- outlier removal (documented GrLivArea outliers)
- engineered features (totals, ages, presence flags, quality interactions)
- target encoding + KMeans neighborhood clusters (fit on train only)
- ordinal encodings for quality / type columns
- log1p skew correction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEED = 42

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


# --------------------------------------------------------------------------- #
# Stage 1 — outlier removal + imputation
# --------------------------------------------------------------------------- #
def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].copy()
    y = np.log1p(train["SalePrice"])
    print(f"After outlier removal: {train.shape}")

    ntrain = train.shape[0]
    test_id = test["Id"]
    all_data = pd.concat([
        train.drop(["Id", "SalePrice"], axis=1),
        test.drop("Id", axis=1)], axis=0).reset_index(drop=True)

    for c in NONE_COLS:
        all_data[c] = all_data[c].fillna("None")
    for c in ZERO_COLS:
        all_data[c] = all_data[c].fillna(0)
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for c in MODE_COLS:
        all_data[c] = all_data[c].fillna(all_data[c].mode()[0])
    print(f"Remaining missing: {all_data.isnull().sum().sum()}")
    return train, y, all_data, ntrain, test_id


# --------------------------------------------------------------------------- #
# Stage 2 — feature engineering (with KMeans neighborhood clusters)
# --------------------------------------------------------------------------- #
def engineer(all_data: pd.DataFrame, train: pd.DataFrame, n_clusters: int = 5):
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

    # Target encoding — fit on TRAIN ONLY, then map (no leakage)
    nbhd_med = train.groupby("Neighborhood")["SalePrice"].median()
    all_data["NeighborhoodPrice"] = (all_data["Neighborhood"]
                                     .map(nbhd_med).fillna(nbhd_med.median()))

    # KMeans on neighborhood-level aggregated stats (train only)
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

    for c in QUALITY_COLS:
        all_data[c] = all_data[c].map(QUALITY_MAP).fillna(0).astype(int)
    for c, mp in ORDINAL_MAPS.items():
        all_data[c] = all_data[c].map(mp).fillna(0)

    numeric_feats = all_data.select_dtypes(include=[np.number]).columns.tolist()
    skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed = skewed[abs(skewed) > 0.75]
    print(f"Features with |skew| > 0.75: {len(skewed)}")
    for f in skewed.index:
        all_data[f] = np.log1p(all_data[f].clip(lower=0))

    all_data = pd.get_dummies(all_data)
    print(f"After one-hot: {all_data.shape}")

    artifacts = {
        "nbhd_med": nbhd_med.to_dict(),
        "nbhd_med_default": float(nbhd_med.median()),
        "cluster_map": cluster_map,
        "skewed_features": skewed.index.tolist(),
        "feature_columns": all_data.columns.tolist(),
    }
    return all_data, artifacts


# --------------------------------------------------------------------------- #
# Stage 3 — single-row build for inference (Streamlit / predict)
# --------------------------------------------------------------------------- #
def build_row(user_inputs: dict, art: dict) -> pd.DataFrame:
    """Build a single-row dataframe from user input merged with training defaults,
    applying the same transformations as the training pipeline."""
    medians = art["train_medians"]
    modes = art["train_modes"]
    fe = art["fe"]

    row = {**modes, **medians}
    row.update(user_inputs)

    for c in NONE_COLS:
        if c not in row or pd.isna(row.get(c)):
            row[c] = "None"
    for c in ZERO_COLS:
        if c not in row or pd.isna(row.get(c)):
            row[c] = 0

    df = pd.DataFrame([row])

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["AllFloorsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNewHouse"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
    df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["NeighborhoodPrice"] = df["Neighborhood"].map(fe["nbhd_med"]).fillna(
        fe["nbhd_med_default"])
    df["NeighborhoodCluster"] = df["Neighborhood"].map(fe["cluster_map"]).fillna(-1)

    for c in QUALITY_COLS:
        df[c] = df[c].map(QUALITY_MAP).fillna(0).astype(int)
    for c, mp in ORDINAL_MAPS.items():
        df[c] = df[c].map(mp).fillna(0)

    for f in fe["skewed_features"]:
        if f in df.columns:
            df[f] = np.log1p(df[f].clip(lower=0))

    df = pd.get_dummies(df)
    df = df.reindex(columns=fe["feature_columns"], fill_value=0)
    return df
