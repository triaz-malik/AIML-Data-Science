"""
Phase 3 - Customer Segmentation (K-Means + DBSCAN).

Clusters customers on usage / value features, picks k by silhouette,
profiles each cluster and maps it to a business segment name:
  Premium Users, Voice Heavy Users, International Users,
  Low Revenue Users, High Risk Users.

Writes outputs/data/telecom_segmented.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from phase2_features import add_features
from utils import load_raw, savefig, section

sns.set_theme(style="whitegrid")


def _load_features() -> pd.DataFrame:
    if config.FEATURES_CSV.exists():
        return pd.read_csv(config.FEATURES_CSV)
    return add_features(load_raw("all"))


def _choose_k(X: np.ndarray, k_range=range(2, 9)) -> tuple[int, list, list]:
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
    best_k = list(k_range)[int(np.argmax(sils))]
    return best_k, inertias, sils


# Number of clusters for the business segmentation. The data is fairly
# homogeneous so the silhouette-optimal k is small (2); we deliberately
# segment into the 5 business-relevant groups the project requires.
SEGMENT_K = 5


def _label_clusters(cluster_profiles: pd.DataFrame) -> dict:
    """Greedily assign each cluster a unique business segment name based on
    which dimension it is most extreme in relative to the other clusters."""
    remaining = set(cluster_profiles.index)
    mapping: dict = {}

    # composite risk: actual churn dominates, service-call intensity is a
    # light tie-breaker (all clusters have similar service-call means).
    sc = cluster_profiles["Customer service calls"]
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
    risk = cluster_profiles["churn_rate"] + 0.1 * sc_norm

    # (label, ranking series) in priority order
    priorities = [
        ("High Risk Users", risk),
        ("International Users", cluster_profiles["International Usage Ratio"]),
        ("Voice Heavy Users", cluster_profiles["Number vmail messages"]),
        ("Premium Users", cluster_profiles["Customer Value Score"]),
    ]
    for label, series in priorities:
        cand = series[list(remaining)].sort_values(ascending=False)
        if len(cand):
            chosen = cand.index[0]
            mapping[chosen] = label
            remaining.discard(chosen)

    # whatever is left (lowest value) -> Low Revenue Users
    for cl in sorted(remaining, key=lambda c: cluster_profiles.loc[c, "Customer Value Score"]):
        mapping[cl] = "Low Revenue Users"
    return mapping


def run() -> pd.DataFrame:
    section("PHASE 3 - CUSTOMER SEGMENTATION")
    df = _load_features()

    feats = [c for c in config.SEGMENTATION_FEATURES if c in df.columns]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])

    # ---- choose k --------------------------------------------------------
    k_range = range(2, 9)
    best_k, inertias, sils = _choose_k(X, k_range)
    seg_k = SEGMENT_K
    print(f"Silhouette-optimal k = {best_k} (silhouette={max(sils):.3f}); "
          f"segmenting into k={seg_k} for business granularity.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-")
    axes[0].axvline(seg_k, color="red", ls="--", label=f"chosen k={seg_k}")
    axes[0].set(title="Elbow (Inertia)", xlabel="k", ylabel="Inertia")
    axes[0].legend()
    axes[1].plot(list(k_range), sils, "o-", color="darkorange")
    axes[1].axvline(best_k, color="green", ls="--", label=f"best silhouette k={best_k}")
    axes[1].axvline(seg_k, color="red", ls="--", label=f"chosen k={seg_k}")
    axes[1].set(title="Silhouette Score", xlabel="k", ylabel="silhouette")
    axes[1].legend()
    savefig(fig, "08_kmeans_k_selection")

    # ---- K-Means ---------------------------------------------------------
    km = KMeans(n_clusters=seg_k, random_state=config.RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X)

    # ---- profile each cluster -------------------------------------------
    df["churn_rate"] = df[config.TARGET]
    prof = df.groupby("cluster").agg(
        size=("cluster", "size"),
        churn_rate=(config.TARGET, "mean"),
        avg_charges=("Total Charges", "mean"),
        avg_minutes=("Total Usage Minutes", "mean"),
        intl_ratio=("International Usage Ratio", "mean"),
        vmail=("Number vmail messages", "mean"),
        service_calls=("Customer service calls", "mean"),
        value_score=("Customer Value Score", "mean"),
    )
    print("\nCluster profiles:")
    print(prof.round(2).to_string())

    # map cluster -> unique business label
    cluster_profiles = df.groupby("cluster").agg(
        {
            "Customer service calls": "mean",
            "International Usage Ratio": "mean",
            "Number vmail messages": "mean",
            "Customer Value Score": "mean",
            config.TARGET: "mean",
        }
    ).rename(columns={config.TARGET: "churn_rate"})

    mapping = _label_clusters(cluster_profiles)
    df["Segment"] = df["cluster"].map(mapping)
    print("\nCluster -> Segment mapping:")
    for cl, name in mapping.items():
        print(f"  cluster {cl} -> {name}")

    seg_summary = df.groupby("Segment").agg(
        customers=("Segment", "size"),
        churn_rate=(config.TARGET, "mean"),
        avg_charges=("Total Charges", "mean"),
        avg_value=("Customer Value Score", "mean"),
    ).round(2)
    print("\nSegment summary:")
    print(seg_summary.to_string())

    # ---- PCA scatter of segments ----------------------------------------
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X)
    df["_pc1"], df["_pc2"] = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(x="_pc1", y="_pc2", hue="Segment", data=df, s=18, alpha=0.7, ax=ax)
    ax.set_title(f"Customer Segments (K-Means, k={best_k}) - PCA projection")
    ax.set(xlabel="PC1", ylabel="PC2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    savefig(fig, "09_segments_pca_scatter")

    # segment size bar
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df["Segment"].value_counts().index
    sns.countplot(y="Segment", data=df, order=order, ax=ax)
    ax.set_title("Segment Sizes")
    savefig(fig, "10_segment_sizes")

    # ---- DBSCAN comparison ----------------------------------------------
    db = DBSCAN(eps=2.0, min_samples=10)
    db_labels = db.fit_predict(X)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int((db_labels == -1).sum())
    df["dbscan_cluster"] = db_labels
    print(f"\nDBSCAN (eps=2.0, min_samples=10): {n_clusters} clusters, "
          f"{n_noise} noise points ({n_noise/len(df):.1%}).")
    if n_clusters >= 2:
        mask = db_labels != -1
        print(f"  DBSCAN silhouette (excl. noise): "
              f"{silhouette_score(X[mask], db_labels[mask]):.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        x=df["_pc1"], y=df["_pc2"],
        hue=df["dbscan_cluster"].astype(str), s=18, alpha=0.7, ax=ax, legend="brief"
    )
    ax.set_title("DBSCAN Clusters (-1 = noise) - PCA projection")
    ax.set(xlabel="PC1", ylabel="PC2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="cluster")
    savefig(fig, "11_dbscan_pca_scatter")

    # ---- persist ---------------------------------------------------------
    df = df.drop(columns=["churn_rate", "_pc1", "_pc2"])
    df.to_csv(config.SEGMENTED_CSV, index=False)
    print(f"\n[data] {config.SEGMENTED_CSV.relative_to(config.PROJECT_ROOT)}")

    seg_summary.to_csv(config.REPORT_DIR / "phase3_segment_summary.csv")
    return df


if __name__ == "__main__":
    run()
