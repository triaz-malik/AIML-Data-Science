"""
Phase 4 - Feature Engineering
Reads data/processed/sms_labeled.csv, adds engineered numeric features,
writes data/processed/sms_features.csv and a feature-vs-class figure.
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_features, FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sms_labeled.csv"
OUT = ROOT / "data" / "processed" / "sms_features.csv"
FIG = ROOT / "outputs" / "figures"
CLASSES = ["Normal", "Promotion", "Spam", "Phishing", "Fraud"]


def main():
    df = pd.read_csv(DATA)
    df = add_features(df, "text")
    df.to_csv(OUT, index=False, encoding="utf-8")

    means = df.groupby("label5")[FEATURE_COLS].mean().reindex(CLASSES)
    print("Saved", OUT, "\n")
    print("=== Mean engineered features by class ===")
    print(means.round(2).to_string())

    # heatmap of normalised feature means by class
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(norm, annot=means.round(1), fmt="g", cmap="rocket_r",
                cbar_kws={"label": "min-max normalised"}, ax=ax)
    ax.set_title("Engineered Feature Profile by Class (annotated with raw means)",
                 fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG / "05_feature_profile.png", dpi=130)
    plt.close(fig)
    print("\nFigure -> outputs/figures/05_feature_profile.png")

    # correlation of features with 'is malicious'
    df["is_mal"] = (df["label5"] != "Normal").astype(int)
    corr = df[FEATURE_COLS + ["is_mal"]].corr()["is_mal"].drop("is_mal").sort_values(ascending=False)
    print("\n=== Correlation of features with 'is malicious' ===")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
