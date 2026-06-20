"""Data loading, outlier removal, and missing-value imputation."""
import numpy as np
import pandas as pd

from . import config as C


def load_data():
    """Load the raw train and test CSVs."""
    train = pd.read_csv(C.TRAIN_CSV)
    test = pd.read_csv(C.TEST_CSV)
    return train, test


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop the two well-known GrLivArea outliers (huge homes, low price).

    These points are documented by the dataset author (De Cock, 2011) as
    partial sales / anomalies and consistently hurt model fit.
    """
    mask = (train["GrLivArea"] > 4000) & (train[C.TARGET] < 300_000)
    removed = int(mask.sum())
    train = train.loc[~mask].reset_index(drop=True)
    print(f"  removed {removed} outliers (GrLivArea > 4000 & SalePrice < 300k)")
    return train


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using domain-aware rules.

    - "None"  for categoricals where NA == feature absent
    - 0       for numerics where NA == feature absent
    - neighbourhood-median for LotFrontage
    - mode/median for everything that remains
    """
    df = df.copy()

    for col in C.NONE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in C.ZERO_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage is strongly tied to the neighbourhood block layout.
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Functional: data description says NA means "Typ".
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna("Typ")

    # Anything still missing: median for numerics, mode for everything else.
    # NB: pandas >=3.0 uses a dedicated string dtype, so test numeric-ness
    # explicitly rather than comparing against "object".
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def cast_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric-looking category codes (e.g. MSSubClass) to strings."""
    df = df.copy()
    for col in C.NUMERIC_AS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df
