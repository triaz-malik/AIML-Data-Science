"""Smoke tests for preprocessing + feature engineering."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    NONE_COLS, ZERO_COLS, QUALITY_MAP,
    engineer, preprocess,
)


@pytest.fixture
def tiny_train():
    """A minimal train-shaped frame covering the columns the pipeline relies on."""
    return pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "MSSubClass": [60, 20, 70, 60, 50],
        "MSZoning": ["RL", "RL", "RL", "RL", "RM"],
        "LotFrontage": [65, 80, np.nan, 70, 85],
        "LotArea": [8450, 9600, 11250, 9550, 14260],
        "Alley": [np.nan, np.nan, np.nan, np.nan, "Pave"],
        "OverallQual": [7, 6, 7, 7, 8],
        "OverallCond": [5, 8, 5, 5, 5],
        "YearBuilt": [2003, 1976, 2001, 1915, 2000],
        "YearRemodAdd": [2003, 1976, 2002, 1970, 2000],
        "MasVnrType": ["BrkFace", "None", "BrkFace", "None", "BrkFace"],
        "MasVnrArea": [196.0, 0.0, 162.0, 0.0, 350.0],
        "ExterQual": ["Gd", "TA", "Gd", "TA", "Gd"],
        "ExterCond": ["TA", "TA", "TA", "TA", "TA"],
        "BsmtQual": ["Gd", "Gd", "Gd", "TA", "Gd"],
        "BsmtCond": ["TA", "TA", "TA", "Gd", "TA"],
        "BsmtExposure": ["No", "Gd", "Mn", "No", "Av"],
        "BsmtFinType1": ["GLQ", "ALQ", "GLQ", "ALQ", "GLQ"],
        "BsmtFinSF1": [706, 978, 486, 216, 655],
        "BsmtFinType2": ["Unf", "Unf", "Unf", "Unf", "Unf"],
        "BsmtFinSF2": [0, 0, 0, 0, 0],
        "BsmtUnfSF": [150, 284, 434, 540, 490],
        "TotalBsmtSF": [856, 1262, 920, 756, 1145],
        "HeatingQC": ["Ex", "Ex", "Ex", "Gd", "Ex"],
        "Electrical": ["SBrkr", "SBrkr", "SBrkr", "SBrkr", "SBrkr"],
        "1stFlrSF": [856, 1262, 920, 961, 1145],
        "2ndFlrSF": [854, 0, 866, 756, 1053],
        "GrLivArea": [1710, 1262, 1786, 1717, 2198],
        "BsmtFullBath": [1, 0, 1, 1, 1],
        "BsmtHalfBath": [0, 1, 0, 0, 0],
        "FullBath": [2, 2, 2, 1, 2],
        "HalfBath": [1, 0, 1, 0, 1],
        "BedroomAbvGr": [3, 3, 3, 3, 4],
        "KitchenAbvGr": [1, 1, 1, 1, 1],
        "KitchenQual": ["Gd", "TA", "Gd", "Gd", "Gd"],
        "TotRmsAbvGrd": [8, 6, 6, 7, 9],
        "Functional": ["Typ", "Typ", "Typ", "Typ", "Typ"],
        "Fireplaces": [0, 1, 1, 1, 1],
        "FireplaceQu": [np.nan, "TA", "TA", "Gd", "TA"],
        "GarageType": ["Attchd", "Attchd", "Attchd", "Detchd", "Attchd"],
        "GarageYrBlt": [2003, 1976, 2001, 1998, 2000],
        "GarageFinish": ["RFn", "RFn", "RFn", "Unf", "RFn"],
        "GarageCars": [2, 2, 2, 3, 3],
        "GarageArea": [548, 460, 608, 642, 836],
        "GarageQual": ["TA", "TA", "TA", "TA", "TA"],
        "GarageCond": ["TA", "TA", "TA", "TA", "TA"],
        "PavedDrive": ["Y", "Y", "Y", "Y", "Y"],
        "WoodDeckSF": [0, 298, 0, 0, 192],
        "OpenPorchSF": [61, 0, 42, 35, 84],
        "EnclosedPorch": [0, 0, 0, 272, 0],
        "3SsnPorch": [0, 0, 0, 0, 0],
        "ScreenPorch": [0, 0, 0, 0, 0],
        "PoolArea": [0, 0, 0, 0, 0],
        "PoolQC": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "Fence": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "MiscFeature": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "MiscVal": [0, 0, 0, 0, 0],
        "MoSold": [2, 5, 9, 2, 12],
        "YrSold": [2008, 2007, 2008, 2006, 2008],
        "SaleType": ["WD", "WD", "WD", "WD", "WD"],
        "SaleCondition": ["Normal", "Normal", "Normal", "Abnorml", "Normal"],
        "Neighborhood": ["CollgCr", "Veenker", "CollgCr", "Crawfor", "NoRidge"],
        "Condition1": ["Norm", "Feedr", "Norm", "Norm", "Norm"],
        "Condition2": ["Norm", "Norm", "Norm", "Norm", "Norm"],
        "BldgType": ["1Fam", "1Fam", "1Fam", "1Fam", "1Fam"],
        "HouseStyle": ["2Story", "1Story", "2Story", "2Story", "2Story"],
        "RoofStyle": ["Gable", "Gable", "Gable", "Gable", "Gable"],
        "RoofMatl": ["CompShg"] * 5,
        "Exterior1st": ["VinylSd", "MetalSd", "VinylSd", "Wd Sdng", "VinylSd"],
        "Exterior2nd": ["VinylSd", "MetalSd", "VinylSd", "Wd Shng", "VinylSd"],
        "Foundation": ["PConc", "CBlock", "PConc", "BrkTil", "PConc"],
        "Heating": ["GasA"] * 5,
        "CentralAir": ["Y"] * 5,
        "LotShape": ["Reg", "Reg", "IR1", "IR1", "IR1"],
        "LandContour": ["Lvl"] * 5,
        "Utilities": ["AllPub"] * 5,
        "LotConfig": ["Inside", "FR2", "Inside", "Corner", "FR2"],
        "LandSlope": ["Gtl"] * 5,
        "Street": ["Pave"] * 5,
        "LowQualFinSF": [0] * 5,
        "SalePrice": [208500, 181500, 223500, 140000, 250000],
    })


@pytest.fixture
def tiny_test(tiny_train):
    test = tiny_train.drop(columns=["SalePrice"]).copy()
    test["Id"] = test["Id"] + 1000
    return test


def test_quality_map_covers_strings():
    for k in ["Po", "Fa", "TA", "Gd", "Ex", "None", "NA"]:
        assert k in QUALITY_MAP
    assert QUALITY_MAP["Ex"] > QUALITY_MAP["TA"] > QUALITY_MAP["Po"]


def test_preprocess_target_is_log(tiny_train, tiny_test):
    train, y, all_data, ntrain, test_id = preprocess(tiny_train, tiny_test)
    expected = np.log1p(tiny_train["SalePrice"]).values
    assert np.allclose(y.values, expected)
    assert ntrain == len(tiny_train)
    assert all_data.isnull().sum().sum() == 0
    assert len(test_id) == len(tiny_test)


def test_preprocess_imputes_absences(tiny_train, tiny_test):
    _, _, all_data, _, _ = preprocess(tiny_train, tiny_test)
    for col in NONE_COLS:
        if col in all_data.columns:
            assert not all_data[col].isnull().any()
    for col in ZERO_COLS:
        if col in all_data.columns:
            assert not all_data[col].isnull().any()


def test_engineer_adds_expected_features(tiny_train, tiny_test):
    train, y, all_data, ntrain, _ = preprocess(tiny_train, tiny_test)
    all_data, fe = engineer(all_data, train, n_clusters=2)
    for f in ["TotalSF", "TotalBathrooms", "HouseAge", "QualArea", "QualTotalSF",
              "NeighborhoodPrice", "NeighborhoodCluster"]:
        cols = [c for c in all_data.columns if c == f or c.startswith(f"{f}_")]
        assert cols, f"engineered feature missing: {f}"
    assert "feature_columns" in fe and "cluster_map" in fe
