"""Feature engineering and encoding."""
import numpy as np
import pandas as pd

from . import config as C


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features that capture age, size and quality."""
    df = df.copy()

    # --- Age features -------------------------------------------------------
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    # Garages built in year 0 (i.e. no garage) produce nonsense ages -> 0.
    df.loc[df["GarageYrBlt"] == 0, "GarageAge"] = 0
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    # Guard against negative ages from data-entry quirks.
    for col in ["HouseAge", "RemodAge", "GarageAge"]:
        df[col] = df[col].clip(lower=0)

    # --- Size / area features ----------------------------------------------
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
        + df["ScreenPorch"] + df["WoodDeckSF"]
    )

    # --- Quality features ---------------------------------------------------
    df["QualScore"] = df["OverallQual"] * df["OverallCond"]

    # --- Boolean "has-feature" flags ---------------------------------------
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    return df


def encode_and_align(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode categoricals and align train/test to the same columns.

    Returns (X_train, X_test) with identical column sets.
    """
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, drop_first=True)

    X_train = combined.iloc[:n_train].reset_index(drop=True)
    X_test = combined.iloc[n_train:].reset_index(drop=True)
    return X_train, X_test
