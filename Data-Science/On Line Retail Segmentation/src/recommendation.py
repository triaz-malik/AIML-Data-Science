"""
Phase 6 - KNN Recommendation Engine (the core deliverable).

Two recommenders, both built on a sparse customer x product purchase matrix:

  User-based collaborative filtering (KNN)
      "Customers similar to you also bought ..."
      For a target customer we find the K nearest customers (cosine/euclidean/
      manhattan), pool the products they bought, score by similarity-weighted
      votes, drop what the customer already owns, and return the top-N.

  Item-based similarity
      "People who bought X also bought Y"
      Nearest-neighbour search over the *transposed* matrix gives product
      similarities for cross-sell / bundling.

Evaluation uses a leave-out protocol: for a sample of customers we hide a random
20% of their products, recommend from the rest, and measure how many hidden
products are recovered — Precision@N, Recall@N, Hit Rate and MAP@N. We tune K
and the distance metric against these metrics.
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
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sns.set_theme(style="whitegrid", palette=config.PALETTE)
RNG = np.random.default_rng(config.RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Matrix construction
# --------------------------------------------------------------------------- #
def build_user_item(df: pd.DataFrame):
    """Sparse binary customer x product matrix + id<->index maps + descriptions."""
    customers = df["CustomerID"].unique()
    products = df["StockCode"].unique()
    cust_idx = {c: i for i, c in enumerate(customers)}
    prod_idx = {p: i for i, p in enumerate(products)}

    rows = df["CustomerID"].map(cust_idx).to_numpy()
    cols = df["StockCode"].map(prod_idx).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    mat = sparse.csr_matrix((data, (rows, cols)),
                            shape=(len(customers), len(products)))
    mat.data[:] = 1.0  # binary "purchased" signal

    desc = (df.drop_duplicates("StockCode").set_index("StockCode")["Description"]
              .reindex(products).to_dict())
    return mat, customers, products, cust_idx, prod_idx, desc


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
class UserBasedKNN:
    def __init__(self, metric: str = config.KNN_METRIC):
        self.metric = metric
        self.nn = NearestNeighbors(metric=metric, algorithm="brute")

    def fit(self, mat: sparse.csr_matrix):
        self.mat = mat
        self.nn.fit(mat)
        return self

    def _score_items(self, user_vec, k: int, exclude_idx: int | None = None):
        # Pull a couple of extra neighbours so we can drop self / the excluded
        # row (critical for honest evaluation — otherwise a customer's own row
        # leaks the held-out items straight back).
        pad = 3
        dist, idx = self.nn.kneighbors(user_vec, n_neighbors=k + pad)
        dist, idx = dist[0], idx[0]
        keep = idx != exclude_idx if exclude_idx is not None else np.ones_like(idx, bool)
        dist, idx = dist[keep][:k], idx[keep][:k]
        sims = 1.0 - dist if self.metric == "cosine" else 1.0 / (1.0 + dist)
        neighbour_mat = self.mat[idx]                       # k x items
        scores = np.asarray(neighbour_mat.T.dot(sims)).ravel()
        return scores

    def recommend(self, user_vec, k: int, n: int, exclude_idx: int | None = None):
        scores = self._score_items(user_vec, k, exclude_idx=exclude_idx)
        owned = user_vec.indices
        scores[owned] = -np.inf                              # exclude owned items
        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(scores[top])[::-1]]
        return top[scores[top] > -np.inf]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _metrics_for_user(recommended: np.ndarray, held_out: set, n: int):
    if not held_out:
        return None
    hits = [i for i in recommended if i in held_out]
    precision = len(hits) / n
    recall = len(hits) / len(held_out)
    hit_rate = 1.0 if hits else 0.0
    # MAP@N
    ap, correct = 0.0, 0
    for rank, item in enumerate(recommended, start=1):
        if item in held_out:
            correct += 1
            ap += correct / rank
    ap = ap / min(len(held_out), n)
    return precision, recall, hit_rate, ap


def evaluate(mat, model: UserBasedKNN, k: int, n: int,
             sample: int = 600, min_items: int = 5):
    eligible = np.where(np.diff(mat.indptr) >= min_items)[0]
    if len(eligible) > sample:
        eligible = RNG.choice(eligible, sample, replace=False)

    P, R, H, M = [], [], [], []
    for u in eligible:
        items = mat[u].indices
        n_hide = max(1, int(len(items) * 0.2))
        hidden = set(RNG.choice(items, n_hide, replace=False).tolist())
        visible = np.array([i for i in items if i not in hidden])
        if len(visible) == 0:
            continue
        vis_vec = sparse.csr_matrix(
            (np.ones(len(visible), np.float32),
             (np.zeros(len(visible), int), visible)),
            shape=(1, mat.shape[1]))
        # exclude_idx=u removes the customer's own (full) row from the neighbour
        # pool so the held-out items cannot leak back through it.
        recs = model.recommend(vis_vec, k=k, n=n, exclude_idx=u)
        res = _metrics_for_user(recs, hidden, n)
        if res:
            P.append(res[0]); R.append(res[1]); H.append(res[2]); M.append(res[3])
    return {"precision@N": np.mean(P), "recall@N": np.mean(R),
            "hit_rate": np.mean(H), "map@N": np.mean(M), "n_eval": len(P)}


def tune(mat, n: int = config.TOP_N_RECOMMENDATIONS):
    metrics = ["cosine", "euclidean", "manhattan"]
    ks = [3, 5, 7, 9, 11, 15]
    rows = []
    for metric in metrics:
        model = UserBasedKNN(metric=metric).fit(mat)
        for k in ks:
            res = evaluate(mat, model, k=k, n=n)
            rows.append({"metric": metric, "k": k, **res})
            print(f"  {metric:<10} K={k:<3} "
                  f"P@{n}={res['precision@N']:.3f} R@{n}={res['recall@N']:.3f} "
                  f"HitRate={res['hit_rate']:.3f} MAP={res['map@N']:.3f}")
    results = pd.DataFrame(rows)
    results.to_csv(config.REPORT_DIR / "knn_tuning.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metrics:
        sub = results[results["metric"] == metric]
        ax.plot(sub["k"], sub["hit_rate"], marker="o", label=metric)
    ax.set_title(f"KNN Hit Rate@{n} vs K by distance metric")
    ax.set_xlabel("K (neighbours)"); ax.set_ylabel(f"Hit Rate@{n}")
    ax.legend()
    fig.savefig(config.FIGURE_DIR / "09_knn_tuning.png",
                dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    best = results.sort_values("hit_rate", ascending=False).iloc[0]
    print(f"\nBest config: metric={best['metric']} K={int(best['k'])} "
          f"(Hit Rate={best['hit_rate']:.3f})")
    return best


def item_similarity_examples(mat, products, desc, n_examples: int = 10, topn: int = 5):
    """'People who bought X also bought Y' for the most popular products."""
    item_mat = mat.T.tocsr()
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(item_mat)
    popularity = np.asarray(mat.sum(axis=0)).ravel()
    top_items = np.argsort(popularity)[::-1][:n_examples]

    rows = []
    for i in top_items:
        dist, idx = nn.kneighbors(item_mat[i], n_neighbors=topn + 1)
        for d, j in zip(dist[0], idx[0]):
            if j == i:
                continue
            rows.append({"product": desc.get(products[i], products[i]),
                         "also_bought": desc.get(products[j], products[j]),
                         "similarity": round(1 - d, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORT_DIR / "item_similarity_examples.csv", index=False)
    return out


def generate_all_recommendations(mat, model, customers, products, desc,
                                 k: int, n: int):
    """Top-N recommendations for every customer -> Power BI / CRM feed."""
    rows = []
    for u in range(mat.shape[0]):
        recs = model.recommend(mat[u], k=k, n=n)
        for rank, item in enumerate(recs, start=1):
            rows.append({"CustomerID": int(customers[u]),
                         "rank": rank,
                         "StockCode": products[item],
                         "Description": desc.get(products[item], "")})
    out = pd.DataFrame(rows)
    out.to_parquet(config.PROCESSED_DIR / "recommendations.parquet", index=False)
    return out


def run() -> None:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    mat, customers, products, cust_idx, prod_idx, desc = build_user_item(df)
    print(f"User-item matrix: {mat.shape[0]:,} customers x {mat.shape[1]:,} "
          f"products, density={mat.nnz / np.prod(mat.shape):.4%}")

    print("\nTuning KNN (metric x K) ...")
    best = tune(mat)
    best_metric, best_k = best["metric"], int(best["k"])

    print("\nFitting final recommender + generating outputs ...")
    model = UserBasedKNN(metric=best_metric).fit(mat)

    sim = item_similarity_examples(mat, products, desc)
    print("\nSample 'also bought' rules:")
    print(sim.head(8).to_string(index=False))

    recs = generate_all_recommendations(mat, model, customers, products, desc,
                                        k=best_k, n=config.TOP_N_RECOMMENDATIONS)
    print(f"\nGenerated {len(recs):,} recommendations "
          f"for {recs['CustomerID'].nunique():,} customers")

    joblib.dump({"metric": best_metric, "k": best_k,
                 "products": products, "customers": customers, "desc": desc},
                config.MODEL_DIR / "recommender.joblib")
    print(f"Saved recommender metadata -> {config.MODEL_DIR / 'recommender.joblib'}")


if __name__ == "__main__":
    run()
