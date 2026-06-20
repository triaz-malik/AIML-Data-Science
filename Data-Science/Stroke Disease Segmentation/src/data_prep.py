"""
Phase 1-2 & 4: data understanding, cleaning, and feature engineering.

Single source of truth for turning the raw CSV into a clean, feature-rich
DataFrame. Every downstream script (EDA, modeling, SHAP, Power BI export)
imports `load_clean()` so they all see identical data.
"""

import numpy as np
import pandas as pd

from config import DATA_FILE


# --------------------------------------------------------------------------- #
# Cleaning  (Phase 2)
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Read the CSV exactly as shipped."""
    return pd.read_csv(DATA_FILE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 'N/A' string in bmi -> real NaN -> median imputation.
    - Drop the id column (identifier, no signal).
    - Drop the single 'Other' gender row (singleton category, can't model).
    - Remove duplicate rows.
    - Standardize categorical text (strip/lower-cased keys kept readable).
    - Clip extreme BMI/glucose outliers to the 0.5/99.5 percentile (winsorize)
      so a handful of data-entry-style extremes don't dominate scaling.
    """
    df = df.copy()

    # bmi: 'N/A' -> NaN -> numeric -> median impute
    df["bmi"] = pd.to_numeric(df["bmi"].replace("N/A", np.nan), errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # drop identifier
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # drop singleton gender
    df = df[df["gender"] != "Other"].copy()

    # duplicate removal
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[clean] removed {dropped} duplicate rows")

    # category standardization (consistent casing/whitespace)
    for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        df[col] = df[col].astype(str).str.strip()

    # winsorize the continuous clinical measures
    for col in ["bmi", "avg_glucose_level"]:
        lo, hi = df[col].quantile([0.005, 0.995])
        df[col] = df[col].clip(lo, hi)

    return df


# --------------------------------------------------------------------------- #
# Feature engineering  (Phase 4)
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically meaningful engineered features."""
    df = df.copy()

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-0.1, 18, 45, 65, 200],
        labels=["Child", "Young", "Adult", "Senior"],
    ).astype(str)

    # BMI categories (WHO bands)
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[-0.1, 18.5, 25, 30, 200],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    ).astype(str)

    # Glucose risk categories (clinical-style thresholds for avg glucose)
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[-0.1, 114, 140, 1000],
        labels=["Normal", "Moderate", "High"],
    ).astype(str)

    # Composite Health Risk Score (0-8): age + BMI + hypertension + heart disease.
    # Simple additive clinical heuristic — interpretable, not learned.
    age_pts = np.select(
        [df["age"] >= 65, df["age"] >= 45], [2, 1], default=0
    )
    bmi_pts = np.select(
        [df["bmi"] >= 30, df["bmi"] >= 25], [2, 1], default=0
    )
    df["health_risk_score"] = (
        age_pts + bmi_pts + 2 * df["hypertension"] + 2 * df["heart_disease"]
    ).astype(int)

    return df


def load_clean(with_features: bool = True) -> pd.DataFrame:
    """Convenience: raw -> clean -> (optionally) engineered, in one call."""
    df = clean(load_raw())
    if with_features:
        df = add_features(df)
    return df


if __name__ == "__main__":
    d = load_clean()
    print(f"shape: {d.shape}")
    print(f"columns: {list(d.columns)}")
    print(d[["age_group", "bmi_category", "glucose_category", "health_risk_score"]].head())
    print("\nhealth_risk_score distribution:")
    print(d["health_risk_score"].value_counts().sort_index())
