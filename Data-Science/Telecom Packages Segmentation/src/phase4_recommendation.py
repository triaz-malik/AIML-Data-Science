"""
Phase 4 - KNN Package Recommendation Engine.

For every subscriber we find the 5 most similar customers (by usage,
charges, international calls, account length and service calls) and
recommend the package most common among those neighbours:

    "Customers similar to you are using Package X."

A rule-based package catalogue assigns each customer a current package;
the recommender then surfaces the dominant neighbour package, flagging
cross-sell / upsell opportunities where it differs.

Writes outputs/data/recommendations.csv
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config
from phase3_segmentation import run as run_phase3
from utils import section

PACKAGES = [
    "Basic Saver",
    "Day Talker Pro",
    "Evening & Night Saver",
    "Global Connect",
    "Voicemail Plus",
    "Premium Unlimited",
]


def assign_package(row: pd.Series, q: dict) -> str:
    """Rule-based 'current package' from a customer's usage profile."""
    if row["Customer Value Score"] >= q["value_high"] and row["Total Charges"] >= q["charge_high"]:
        return "Premium Unlimited"
    if row["International Usage Ratio"] >= q["intl_high"] or row["Total intl minutes"] >= q["intl_min_high"]:
        return "Global Connect"
    if row["Number vmail messages"] >= q["vmail_high"]:
        return "Voicemail Plus"
    if row["Total day minutes"] >= q["day_high"]:
        return "Day Talker Pro"
    if (row["Total eve minutes"] + row["Total night minutes"]) >= q["evenight_high"]:
        return "Evening & Night Saver"
    return "Basic Saver"


def build_package_catalog(df: pd.DataFrame) -> pd.Series:
    q = {
        "value_high": df["Customer Value Score"].quantile(0.75),
        "charge_high": df["Total Charges"].quantile(0.70),
        "intl_high": df["International Usage Ratio"].quantile(0.85),
        "intl_min_high": df["Total intl minutes"].quantile(0.85),
        "vmail_high": 20,
        "day_high": df["Total day minutes"].quantile(0.75),
        "evenight_high": (df["Total eve minutes"] + df["Total night minutes"]).quantile(0.75),
    }
    return df.apply(lambda r: assign_package(r, q), axis=1)


def run() -> pd.DataFrame:
    section("PHASE 4 - KNN PACKAGE RECOMMENDATION ENGINE")

    if config.SEGMENTED_CSV.exists():
        df = pd.read_csv(config.SEGMENTED_CSV)
    else:
        df = run_phase3()

    # current package per customer
    df["Current Package"] = build_package_catalog(df)
    print("Current package distribution:")
    print(df["Current Package"].value_counts().to_string())

    # ---- fit nearest-neighbours model -----------------------------------
    feats = [c for c in config.RECO_FEATURES if c in df.columns]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])

    k = 6  # 5 neighbours + self
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    pkgs = df["Current Package"].to_numpy()
    recommended, n_diff_pkg, examples = [], [], []
    for i in range(len(df)):
        neigh = idx[i][1:]  # drop self
        neigh_pkgs = pkgs[neigh]
        vals, counts = np.unique(neigh_pkgs, return_counts=True)
        rec = vals[int(np.argmax(counts))]
        recommended.append(rec)

    df["Recommended Package"] = recommended
    df["Upsell Opportunity"] = df["Recommended Package"] != df["Current Package"]
    df["Recommendation Text"] = df["Recommended Package"].map(
        lambda p: f"Customers similar to you are using {p}."
    )

    n_up = int(df["Upsell Opportunity"].sum())
    print(f"\nNeighbours used per customer: 5")
    print(f"Cross-sell / upsell opportunities flagged: {n_up:,} ({n_up/len(df):.1%})")
    print("\nRecommended package distribution:")
    print(df["Recommended Package"].value_counts().to_string())

    # show a few example recommendations
    print("\nSample recommendations:")
    cols = ["Total Charges", "Total intl minutes", "Current Package",
            "Recommended Package", "Recommendation Text"]
    print(df[cols].head(8).to_string(index=False))

    # ---- persist --------------------------------------------------------
    out_cols = [
        "Account length", "Total Usage Minutes", "Total Charges",
        "Total intl minutes", "Customer service calls", "Customer Value Score",
        "Segment", "Current Package", "Recommended Package",
        "Upsell Opportunity", "Recommendation Text",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(config.RECOMMENDATIONS_CSV, index=False)
    print(f"\n[data] {config.RECOMMENDATIONS_CSV.relative_to(config.PROJECT_ROOT)}")

    joblib.dump(
        {"model": nn, "scaler": scaler, "features": feats, "packages": pkgs},
        config.MODEL_DIR / "knn_recommender.joblib",
    )
    # carry package columns forward
    df.to_csv(config.SEGMENTED_CSV, index=False)
    return df


if __name__ == "__main__":
    run()
