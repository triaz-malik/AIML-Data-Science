"""
Phase 5 - Customer Segmentation.

Two complementary unsupervised views of the customer base:

  KMeans  - partitions every customer into K business segments. We search K with
            the elbow (inertia) and silhouette score, then fit the final model
            and translate cluster centroids into human labels
            (VIP / Loyal / Occasional / At-Risk) by ranking on R, F, M.

  DBSCAN  - density-based; surfaces a "noise" group of unusual customers
            (e.g. wholesale-scale buyers) that don't fit any dense segment.

RFM is right-skewed, so we log-transform before standardizing — otherwise a few
whale customers dominate the Euclidean geometry and KMeans degenerates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sns.set_theme(style="whitegrid", palette=config.PALETTE)

CLUSTER_FEATURES = ["Recency", "Frequency", "Monetary",
                    "AvgBasketValue", "ProductDiversity"]


def load() -> pd.DataFrame:
    return pd.read_parquet(config.CUSTOMER_PARQUET)


def build_matrix(feats: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = feats[CLUSTER_FEATURES].copy()
    # log1p tames the heavy right tail of R/F/M before scaling.
    for col in CLUSTER_FEATURES:
        X[col] = np.log1p(X[col].clip(lower=0))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def choose_k(Xs: np.ndarray) -> int:
    inertias, silhouettes, ks = [], [], list(config.KMEANS_K_RANGE)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_title("Elbow — inertia vs K")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, silhouettes, marker="o", color="#21918c")
    axes[1].set_title("Silhouette vs K")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")
    best_k = ks[int(np.argmax(silhouettes))]
    axes[1].axvline(best_k, ls="--", color="grey")
    fig.savefig(config.FIGURE_DIR / "07_kmeans_k_selection.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  best silhouette at K={best_k}; "
          f"using configured K={config.KMEANS_N_SEGMENTS} for business segments")
    return config.KMEANS_N_SEGMENTS


def label_clusters(feats: pd.DataFrame, km: KMeans) -> dict[int, str]:
    """Rank clusters by a value score and map to business names."""
    profile = feats.groupby("cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    )
    # Higher F & M and lower R => more valuable. Compose a simple rank score.
    score = (profile["Monetary"].rank() + profile["Frequency"].rank()
             - profile["Recency"].rank())
    order = score.sort_values(ascending=False).index.tolist()
    names = ["VIP", "Loyal", "Occasional", "At-Risk", "Dormant",
             "Segment-6", "Segment-7"]
    return {cl: names[i] for i, cl in enumerate(order)}


def plot_pca(Xs: np.ndarray, labels: np.ndarray, feats: pd.DataFrame) -> None:
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    for seg in sorted(feats["segment"].unique()):
        m = feats["segment"].values == seg
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.5, label=seg)
    ax.set_title(f"Customer segments in PCA space "
                 f"({pca.explained_variance_ratio_.sum():.0%} variance)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(markerscale=2)
    fig.savefig(config.FIGURE_DIR / "08_segments_pca.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def run() -> pd.DataFrame:
    feats = load()
    Xs, scaler = build_matrix(feats)

    k = choose_k(Xs)
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    feats["cluster"] = km.fit_predict(Xs)
    mapping = label_clusters(feats, km)
    feats["segment"] = feats["cluster"].map(mapping)

    # DBSCAN anomaly view on the same standardized matrix.
    db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES)
    feats["dbscan"] = db.fit_predict(Xs)
    n_noise = int((feats["dbscan"] == -1).sum())

    plot_pca(Xs, feats["cluster"].values, feats)

    # Persist artefacts
    joblib.dump({"kmeans": km, "scaler": scaler,
                 "features": CLUSTER_FEATURES, "mapping": mapping},
                config.MODEL_DIR / "segmentation.joblib")
    feats.to_parquet(config.CUSTOMER_PARQUET, index=False)

    # Segment profile report
    profile = (feats.groupby("segment")
               .agg(customers=("CustomerID", "count"),
                    avg_recency=("Recency", "mean"),
                    avg_frequency=("Frequency", "mean"),
                    avg_monetary=("Monetary", "mean"),
                    total_revenue=("Monetary", "sum"))
               .sort_values("total_revenue", ascending=False))
    profile["revenue_share"] = profile["total_revenue"] / profile["total_revenue"].sum()
    profile.to_csv(config.REPORT_DIR / "segment_profile.csv")

    print(f"\nKMeans segments (K={k}):")
    print(profile.round(2).to_string())
    print(f"\nDBSCAN flagged {n_noise:,} unusual customers (noise = -1)")
    print(f"Saved segmentation model -> {config.MODEL_DIR / 'segmentation.joblib'}")
    return feats


if __name__ == "__main__":
    run()
