"""Exploratory Data Analysis -> saves figures to reports/figures and prints findings."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from preprocessing import load_train, clean

sns.set_theme(style="whitegrid")


def _save(fig, name: str):
    path = C.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved {path.name}")


def run():
    df = clean(load_train())
    print(f"[eda] dataset shape after cleaning: {df.shape}")

    # 1) Target distribution -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, y=C.TARGET, ax=ax, palette="Set2", hue=C.TARGET, legend=False)
    ax.set_title("Target Distribution (normal vs anomaly)")
    _save(fig, "01_target_distribution.png")
    vc = df[C.TARGET].value_counts()
    attack_ratio = vc.get("anomaly", 0) / len(df)
    print(f"[eda] target counts: {vc.to_dict()} | attack ratio: {attack_ratio:.2%}")

    # 2) Protocol distribution (by class) -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="protocol_type", hue=C.TARGET, ax=ax, palette="Set2")
    ax.set_title("Protocol Type by Class")
    _save(fig, "02_protocol_distribution.png")
    proto_attack = (
        df[df[C.TARGET] == "anomaly"]["protocol_type"].value_counts(normalize=True)
    )
    print(f"[eda] attack share by protocol: {proto_attack.round(3).to_dict()}")

    # 3) Top 20 services -----------------------------------------------------
    top_services = df["service"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_services.values, y=top_services.index, ax=ax,
                palette="viridis", hue=top_services.index, legend=False)
    ax.set_title("Top 20 Services")
    ax.set_xlabel("count")
    _save(fig, "03_top_services.png")
    print(f"[eda] top services: {list(top_services.head(8).index)}")

    # 4) Connection 'flag' distribution by class ----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="flag", hue=C.TARGET, ax=ax, palette="Set2")
    ax.set_title("Connection Flag by Class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "04_flag_distribution.png")

    # 5) Correlation heatmap (numeric features) -----------------------------
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=False)
    ax.set_title("Correlation Heatmap (numeric features)")
    _save(fig, "05_correlation_heatmap.png")
    # report highly-correlated pairs (|r| > 0.9)
    high = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.9:
                high.append((cols[i], cols[j], round(float(r), 3)))
    print(f"[eda] highly-correlated pairs |r|>0.9: {high}")

    # 6) Feature distributions (key numeric features) -----------------------
    feats = [f for f in ["src_bytes", "dst_bytes", "count", "srv_count"] if f in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, f in zip(axes.ravel(), feats):
        # log1p to tame heavy right-skew for visualisation
        sns.histplot(data=df, x=f, hue=C.TARGET, bins=50, ax=ax,
                     log_scale=(False, True), palette="Set2", element="step")
        ax.set_title(f"{f} (y log-scaled)")
    fig.suptitle("Key Feature Distributions", y=1.02)
    _save(fig, "06_feature_distributions.png")
    print("[eda] skew (key features):",
          {f: round(float(df[f].skew()), 2) for f in feats})

    print("[eda] done. Figures in", C.FIGURES_DIR)


if __name__ == "__main__":
    run()
