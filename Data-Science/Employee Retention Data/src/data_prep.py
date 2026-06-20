"""
Data loading, cleaning, and feature engineering.

Two public entry points:
    load_raw()              -> the raw dataframe, target still "Yes"/"No"
    build_features(df)      -> df with engineered columns + binary target
"""
from __future__ import annotations

import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Read the raw IBM HR CSV."""
    df = pd.read_csv(config.RAW_CSV)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant / identifier columns that carry no predictive signal.

    EmployeeCount (always 1), StandardHours (always 80) and Over18 (always 'Y')
    are constants; EmployeeNumber is just an ID.
    """
    df = df.copy()
    drop = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Attrition Yes/No -> 1/0."""
    df = df.copy()
    df[config.TARGET] = (df[config.TARGET].str.strip().str.lower() == "yes").astype(int)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the business-driven features described in the project brief.

    Every feature encodes a hypothesis HR can act on:
      * SalaryBand / SalaryPercentile  -> internal pay fairness
      * ExperienceGroup                -> attrition behaviour differs by seniority
      * PromotionDelayFlag             -> career stagnation
      * LongCommuteFlag / CommuteCategory -> travel fatigue / work-life balance
      * OverTimeFlag                   -> burnout indicator
      * EarlyCareerFlag                -> weak onboarding / early churn
      * TrainingGapFlag                -> lack of development opportunity
    """
    df = df.copy()

    # --- Salary band (terciles) + percentile -----------------------------
    df["SalaryBand"] = pd.qcut(
        df["MonthlyIncome"], q=3, labels=["Low", "Medium", "High"]
    ).astype(str)
    df["SalaryPercentile"] = df["MonthlyIncome"].rank(pct=True).round(3)

    # --- Experience group -------------------------------------------------
    def _exp_group(years: float) -> str:
        if years < 3:
            return "Fresh"
        if years <= 10:
            return "Mid"
        return "Senior"

    df["ExperienceGroup"] = df["TotalWorkingYears"].apply(_exp_group)

    # --- Career-stagnation flag ------------------------------------------
    df["PromotionDelayFlag"] = (df["YearsSinceLastPromotion"] > 4).astype(int)

    # --- Commute ----------------------------------------------------------
    df["LongCommuteFlag"] = (df["DistanceFromHome"] > 20).astype(int)
    df["CommuteCategory"] = pd.cut(
        df["DistanceFromHome"],
        bins=[-1, 5, 15, 100],
        labels=["Near", "Moderate", "Far"],
    ).astype(str)

    # --- Burnout ----------------------------------------------------------
    df["OverTimeFlag"] = (df["OverTime"].str.strip().str.lower() == "yes").astype(int)

    # --- Early-career / onboarding risk ----------------------------------
    df["EarlyCareerFlag"] = (df["YearsAtCompany"] < 2).astype(int)

    # --- Development gap --------------------------------------------------
    df["TrainingGapFlag"] = (df["TrainingTimesLastYear"] == 0).astype(int)

    # --- Income relative to job level (internal fairness) ----------------
    level_median = df.groupby("JobLevel")["MonthlyIncome"].transform("median")
    df["IncomeVsLevelRatio"] = (df["MonthlyIncome"] / level_median).round(3)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full prep: clean -> encode target -> engineer features."""
    df = basic_clean(df)
    df = encode_target(df)
    df = engineer_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) excluding the target."""
    categorical = [c for c in config.CATEGORICAL_COLS if c in df.columns]
    numeric = [
        c
        for c in df.columns
        if c != config.TARGET and c not in categorical
    ]
    return numeric, categorical


if __name__ == "__main__":
    raw = load_raw()
    feats = build_features(raw)
    num, cat = get_feature_columns(feats)
    print(f"Rows: {len(feats)}  |  numeric: {len(num)}  categorical: {len(cat)}")
    print(f"Attrition rate: {feats[config.TARGET].mean():.1%}")
