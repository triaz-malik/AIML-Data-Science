"""
Assemble report/REPORT.md from the JSON metrics + figures produced by the pipeline.
Run last (after insights.py). Pure formatting — no modeling.
"""
import json
import os

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
METRICS = os.path.join(BASE, "outputs", "metrics")
REPORT = os.path.join(BASE, "report", "REPORT.md")
FIGREL = "../outputs/figures"


def j(name):
    p = os.path.join(METRICS, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    ds = j("data_summary.json")
    eda = j("eda_stats.json")
    base = j("baseline_metrics.json")
    tr = j("transformer_metrics.json")
    ins = j("business_insights.json")
    err = j("error_analysis.json")

    L = []
    L.append("# Amazon Customer Review Sentiment Analysis — Results Report\n")
    L.append("AI-powered sentiment classification (negative / neutral / positive) on "
             f"**{ds.get('rows_final', 'N/A'):,} Amazon reviews** across "
             f"{ds.get('n_categories', 'N/A')} product categories.\n")

    # KPIs
    k = ins.get("headline_kpis", {})
    if k:
        L.append("## Headline KPIs\n")
        L.append(f"- **Total reviews:** {k.get('total_reviews', 0):,}")
        L.append(f"- **Positive:** {k.get('pct_positive', 0):.1f}%  |  "
                 f"**Negative:** {k.get('pct_negative', 0):.1f}%  |  "
                 f"**Neutral:** {k.get('pct_neutral', 0):.1f}%")
        L.append(f"- **Mean rating:** {k.get('mean_rating', 0):.2f} / 5\n")

    # Data understanding
    L.append("## Phase 1 — Data\n")
    L.append(f"- Raw rows: {ds.get('rows_total_raw', 0):,}; after dedup/clean: "
             f"{ds.get('rows_final', 0):,} ({ds.get('duplicate_reviews_removed', 0):,} duplicates removed).")
    L.append(f"- Date range: {ds.get('date_min')} → {ds.get('date_max')}; "
             f"{ds.get('pct_verified', 0)}% verified purchases.\n")
    L.append(f"![Rating distribution]({FIGREL}/01_rating_distribution.png)\n")

    # EDA
    L.append("## Phase 2 — EDA highlights\n")
    if ins.get("category_breakdown"):
        L.append(table(ins["category_breakdown"],
                       ["category", "reviews", "mean_rating", "pct_negative"],
                       ["Category", "Reviews", "Mean rating", "% negative"]))
        L.append("")
    if eda.get("top_bigrams_negative"):
        themes = ", ".join(f"`{b['ngram']}`" for b in eda["top_bigrams_negative"][:8])
        L.append(f"**Top negative bigrams:** {themes}\n")
    L.append(f"![Sentiment by category]({FIGREL}/04_sentiment_by_category.png)\n")
    L.append(f"![Word clouds]({FIGREL}/05_wordclouds.png)\n")

    # Models
    L.append("## Phases 5–8 — Model comparison\n")
    rows = []
    if base:
        rows.append({"Model": "TF-IDF + LogReg", "Accuracy": base.get("test_accuracy"),
                     "Macro-F1": base.get("test_f1_macro"),
                     "ROC-AUC": base.get("test_roc_auc_ovr_macro")})
    for name in ["distilbert", "bert"]:
        if name in tr:
            m = tr[name]
            rows.append({"Model": name, "Accuracy": m.get("test_accuracy"),
                         "Macro-F1": m.get("test_f1_macro"),
                         "ROC-AUC": m.get("test_roc_auc_ovr_macro")})
    L.append(table(rows, ["Model", "Accuracy", "Macro-F1", "ROC-AUC"],
                   ["Model", "Accuracy", "Macro-F1", "ROC-AUC"]))
    L.append(f"\n![Model comparison]({FIGREL}/12_model_comparison.png)\n")
    L.append("All models evaluated on the **same** stratified held-out test set "
             f"(n={base.get('n_test', 'N/A'):,}).\n")

    # Explainability + errors
    L.append("## Phases 9–10 — Explainability & error analysis\n")
    L.append(f"![Token importance]({FIGREL}/11_token_importance.png)\n")
    if err:
        L.append(f"- Test error rate: {err.get('error_rate', 0):.1%} "
                 f"({err.get('n_errors', 0):,} / {err.get('test_size', 0):,}).")
        L.append("- Common failure modes: mixed-sentiment reviews (praise + complaint "
                 "in one text) and ambiguous neutral/3-star reviews.\n")

    # Insights
    L.append("## Phase 11 — Business findings & recommendations\n")
    for f in ins.get("findings", []):
        L.append(f"- {f}")
    L.append("\n**Recommendations:**")
    for r in ins.get("recommendations", []):
        L.append(f"- {r}")
    L.append("")

    if ins.get("worst_products_by_negative_rate"):
        L.append("### Worst products by negative rate (min 25 reviews)\n")
        L.append(table(ins["worst_products_by_negative_rate"][:10],
                       ["parent_asin", "reviews", "pct_negative", "mean_rating"],
                       ["Product (ASIN)", "Reviews", "% negative", "Mean rating"]))
        L.append("")

    L.append("## Phase 12 — Dashboard\n")
    L.append("Interactive Streamlit dashboard: `streamlit run src/dashboard.py` "
             "(Executive / Products / NLP-Models views).\n")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
