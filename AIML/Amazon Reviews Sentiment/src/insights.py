"""
Phase 11 — Business insights + model comparison.

Aggregates the cleaned data and model metrics into an executive-friendly summary:
  - headline KPIs (% positive/negative, mean rating, review count)
  - worst categories by % negative / lowest mean rating
  - products (parent_asin) with the highest negative concentration (min volume gate)
  - top complaint themes (negative bigrams) from the EDA stats
  - model comparison table (baseline vs DistilBERT vs BERT)
  - written findings + recommendations

Outputs: outputs/metrics/business_insights.json,
         outputs/figures/12_model_comparison.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
DATA = os.path.join(BASE, "data", "clean.parquet")
METRICS = os.path.join(BASE, "outputs", "metrics")
FIG = os.path.join(BASE, "outputs", "figures")
MIN_PRODUCT_REVIEWS = 25


def load_json(name):
    p = os.path.join(METRICS, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def model_comparison():
    rows = []
    base = load_json("baseline_metrics.json")
    if base:
        rows.append({"model": "TF-IDF + LogReg",
                     "accuracy": base["test_accuracy"],
                     "f1_macro": base["test_f1_macro"],
                     "roc_auc": base["test_roc_auc_ovr_macro"]})
    tr = load_json("transformer_metrics.json")
    if tr:
        for name in ["distilbert", "bert"]:
            if name in tr:
                m = tr[name]
                rows.append({"model": name,
                             "accuracy": m["test_accuracy"],
                             "f1_macro": m["test_f1_macro"],
                             "roc_auc": m["test_roc_auc_ovr_macro"]})
    return rows


def plot_comparison(rows):
    if not rows:
        return
    df = pd.DataFrame(rows).melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="metric", y="score", hue="model")
    plt.ylim(0, 1.0); plt.title("Model comparison")
    plt.xlabel(""); plt.ylabel("Score"); plt.legend(title="")
    for c in plt.gca().containers:
        plt.gca().bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "12_model_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_parquet(DATA, columns=["overall", "sentiment", "category",
                                        "parent_asin", "helpful"])
    n = len(df)
    sent_pct = (df["sentiment"].value_counts(normalize=True) * 100).round(2)

    cat = df.groupby("category").agg(
        reviews=("overall", "size"),
        mean_rating=("overall", "mean"),
        pct_negative=("sentiment", lambda s: round((s == "negative").mean() * 100, 2)),
    ).round(3).sort_values("pct_negative", ascending=False)

    # products with highest negative concentration (volume-gated)
    prod = df.groupby("parent_asin").agg(
        reviews=("overall", "size"),
        pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
        mean_rating=("overall", "mean"),
    )
    prod = prod[prod["reviews"] >= MIN_PRODUCT_REVIEWS]
    worst_products = (prod.sort_values("pct_negative", ascending=False).head(10)
                      .round(2).reset_index().to_dict(orient="records"))

    eda = load_json("eda_stats.json") or {}
    neg_bigrams = eda.get("top_bigrams_negative", [])[:10]

    models = model_comparison()
    plot_comparison(models)
    best = max(models, key=lambda r: r["f1_macro"]) if models else None

    worst_cat = cat.index[0]
    findings = [
        f"{sent_pct.get('positive', 0):.1f}% of reviews are positive, "
        f"{sent_pct.get('negative', 0):.1f}% negative, {sent_pct.get('neutral', 0):.1f}% neutral.",
        f"'{worst_cat}' has the highest negative rate ({cat.iloc[0]['pct_negative']:.1f}%) "
        f"and a mean rating of {cat.iloc[0]['mean_rating']:.2f} — prioritise quality review here.",
        f"Top complaint themes (negative bigrams): "
        f"{', '.join(b['ngram'] for b in neg_bigrams[:6])}.",
    ]
    if best:
        findings.append(
            f"Best model: {best['model']} (macro-F1 {best['f1_macro']:.3f}, "
            f"accuracy {best['accuracy']:.3f}) — accurate enough to automate sentiment monitoring.")

    recommendations = [
        f"Stand up automated sentiment monitoring on incoming reviews using the {best['model'] if best else 'best'} model; "
        "route any review scored negative to the product/support queue in near-real-time.",
        f"Run a focused quality investigation on the '{worst_cat}' category and the worst-performing products listed below.",
        "Mine recurring negative bigrams/trigrams monthly to detect emerging defect themes (e.g. battery, sizing, breakage).",
        "Track % negative as a product-health KPI and alert when a SKU crosses a threshold.",
    ]

    out = {
        "headline_kpis": {
            "total_reviews": int(n),
            "pct_positive": float(sent_pct.get("positive", 0)),
            "pct_negative": float(sent_pct.get("negative", 0)),
            "pct_neutral": float(sent_pct.get("neutral", 0)),
            "mean_rating": round(float(df["overall"].mean()), 3),
        },
        "category_breakdown": cat.reset_index().to_dict(orient="records"),
        "worst_products_by_negative_rate": worst_products,
        "top_negative_themes": neg_bigrams,
        "model_comparison": models,
        "best_model": best,
        "findings": findings,
        "recommendations": recommendations,
    }
    with open(os.path.join(METRICS, "business_insights.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"headline_kpis": out["headline_kpis"],
                      "findings": findings,
                      "model_comparison": models}, indent=2))
    print("\nWrote business_insights.json + 12_model_comparison.png")


if __name__ == "__main__":
    main()
