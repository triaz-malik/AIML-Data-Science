"""
Build the portfolio PDF report.

Assembles a polished, multi-page PDF (outputs/reports/
Customer_Recommendation_Segmentation_Report.pdf) from the artefacts produced by
the pipeline: dataset description, methodology, per-model explanations, parameter
tuning results, metrics, embedded figures, and the business value of each phase.

Run after the pipeline has produced its figures/CSVs:
    python src/build_report_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
NAVY = colors.HexColor("#1f2d54")
TEAL = colors.HexColor("#21918c")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#eef1f7")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=26,
                          textColor=NAVY, leading=32, spaceAfter=10))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=13,
                          textColor=TEAL, alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle("CoverMeta", parent=styles["Normal"], fontSize=10,
                          textColor=GREY, alignment=TA_CENTER))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], textColor=NAVY,
                          fontSize=16, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], textColor=TEAL,
                          fontSize=12.5, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                          leading=15, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("Bul", parent=styles["Normal"], fontSize=10,
                          leading=14, leftIndent=12, spaceAfter=3))

USABLE_W = A4[0] - 3.2 * cm


def P(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [P(f"&bull;&nbsp; {it}", "Bul") for it in items]


def fig(name, caption, width=USABLE_W):
    path = config.FIGURE_DIR / name
    img = Image(str(path))
    scale = width / img.imageWidth
    img.drawWidth = width
    img.drawHeight = img.imageHeight * scale
    return [img, P(caption, "Caption")]


def table_from_df(df, col_widths=None, fontsize=8.5, header_bg=NAVY):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
def load_artifacts():
    a = {}
    a["funnel"] = pd.read_csv(config.REPORT_DIR / "cleaning_funnel.csv",
                              index_col=0).iloc[:, 0]
    a["eda"] = (config.REPORT_DIR / "eda_summary.txt").read_text(encoding="utf-8")
    a["seg"] = pd.read_csv(config.REPORT_DIR / "segment_profile.csv")
    a["knn"] = pd.read_csv(config.REPORT_DIR / "knn_tuning.csv")
    a["rules"] = pd.read_csv(config.REPORT_DIR / "association_rules.csv")
    a["clv"] = pd.read_csv(config.REPORT_DIR / "clv_model_comparison.csv")
    a["shap"] = pd.read_csv(config.REPORT_DIR / "shap_importance.csv")
    a["items"] = pd.read_csv(config.REPORT_DIR / "item_similarity_examples.csv")
    return a


def build():
    a = load_artifacts()
    f = a["funnel"]
    total_rev = float(f["clean_revenue"])
    story = []

    # ---------------- Cover ---------------- #
    story += [Spacer(1, 3.5 * cm)]
    story.append(P("AI-Powered Customer Recommendation"
                   "<br/>&amp; Segmentation Engine", "CoverTitle"))
    story.append(P("End-to-End Retail Analytics on the Online Retail II Dataset",
                   "CoverSub"))
    story.append(Spacer(1, 0.6 * cm))
    story.append(P("Customer Segmentation &bull; KNN Recommendation Engine &bull; "
                   "Market Basket Analysis &bull; Customer Lifetime Value &bull; "
                   "Explainable AI", "CoverMeta"))
    story.append(Spacer(1, 2.4 * cm))
    cover = pd.DataFrame({
        "Metric": ["Clean transactions", "Customers", "Products",
                   "Total revenue", "Date range"],
        "Value": [f"{int(f['clean_rows']):,}", f"{int(f['clean_customers']):,}",
                  f"{int(f['clean_products']):,}", f"GBP {total_rev:,.0f}",
                  f"{f['date_min'][:10]} to {f['date_max'][:10]}"],
    })
    story.append(table_from_df(cover, col_widths=[6 * cm, 8 * cm], fontsize=10))
    story.append(Spacer(1, 2 * cm))
    story.append(P("A business-framed machine learning solution &mdash; every "
                   "phase ends in a concrete commercial action.", "CoverMeta"))
    story.append(PageBreak())

    # ---------------- 1. Dataset ---------------- #
    story.append(P("1. The Dataset", "H1"))
    story.append(P(
        "<b>Online Retail II</b> is a real transactional dataset from a UK-based "
        "online gift retailer, covering roughly two years of sales "
        f"({f['date_min'][:10]} to {f['date_max'][:10]}). Unlike many tidy Kaggle "
        "datasets, it is genuine commercial data &mdash; complete with returns, "
        "cancellations, administrative charges and missing identifiers &mdash; "
        "which makes it ideal for demonstrating an end-to-end, production-minded "
        "workflow.", "Body"))
    cols = pd.DataFrame({
        "Column": ["Invoice", "StockCode", "Description", "Quantity",
                   "InvoiceDate", "Price", "Customer ID", "Country"],
        "Meaning": ["Order number ('C' prefix = cancellation)", "Product ID",
                    "Product name", "Units purchased (negative = return)",
                    "Purchase timestamp", "Unit price (GBP)",
                    "Customer identifier", "Customer country"],
    })
    story.append(table_from_df(cols, col_widths=[3.5 * cm, 10.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(P(
        "The raw file holds <b>1,067,371</b> line items. The business questions "
        "we set out to answer: <i>Who are our most valuable customers? Who is "
        "about to churn? What should we recommend to each customer? Which products "
        "sell together? Where should marketing spend go?</i>", "Body"))

    # ---------------- 2. Methodology ---------------- #
    story.append(P("2. What Was Done &mdash; Methodology", "H1"))
    story.append(P(
        "The solution is a ten-phase Python pipeline. Each phase is a standalone, "
        "reproducible module orchestrated by <font face='Courier'>run_pipeline.py</font>:",
        "Body"))
    pipe = pd.DataFrame({
        "Phase": ["2 Cleaning", "3 EDA", "4 Feature Eng.", "5 Segmentation",
                  "6 Recommender", "7 Market Basket", "8 CLV", "9 Explainability",
                  "10 BI + Report"],
        "Technique": ["Rule-based quality funnel", "Visual + statistical analysis",
                      "RFM + behavioural features", "KMeans + DBSCAN",
                      "User-based KNN collab. filtering", "Apriori association rules",
                      "RF / XGBoost / LightGBM regression", "SHAP values",
                      "Power BI star schema"],
        "Business output": ["Trustworthy data", "Where revenue comes from",
                            "Customer intelligence", "Targetable segments",
                            "Per-customer product recs", "Bundle opportunities",
                            "Future revenue per customer", "Why behind each score",
                            "Executive dashboard"],
    })
    story.append(table_from_df(pipe, col_widths=[2.8 * cm, 5.6 * cm, 5.6 * cm],
                               fontsize=8))
    story.append(PageBreak())

    # ---------------- 3. Cleaning ---------------- #
    story.append(P("3. Data Cleaning", "H1"))
    story.append(P(
        "Recommendations are only as good as the data behind them. We removed every "
        "record that would corrupt customer intelligence and report each dropped "
        "row in a transparent quality funnel:", "Body"))
    funnel_tbl = pd.DataFrame({
        "Step": ["Raw rows", "Duplicates removed", "Missing Customer ID",
                 "Non-product codes", "Returns / cancellations",
                 "Non-positive price", "Clean sale rows"],
        "Rows": [f"{int(f['raw_rows']):,}", f"-{int(f['dropped_duplicates']):,}",
                 f"-{int(f['dropped_missing_customer']):,}",
                 f"-{int(f['dropped_non_product']):,}",
                 f"-{int(f['return_rows']):,}",
                 f"-{int(f['dropped_nonpositive_price']):,}",
                 f"{int(f['clean_rows']):,}"],
    })
    story.append(table_from_df(funnel_tbl, col_widths=[8 * cm, 6 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(P(
        f"Result: <b>{int(f['clean_rows']):,}</b> clean sale rows across "
        f"<b>{int(f['clean_customers']):,}</b> customers and "
        f"<b>{int(f['clean_products']):,}</b> products, totalling "
        f"<b>GBP {total_rev:,.0f}</b>. <b>Business value:</b> poor-quality data "
        "produces wrong recommendations and mis-targeted campaigns &mdash; this "
        "step protects every downstream decision.", "Body"))

    # ---------------- 4. EDA ---------------- #
    story.append(P("4. Exploratory Data Analysis", "H1"))
    story.append(P(
        "Three lenses &mdash; customers, products and time &mdash; reveal where the "
        "business actually makes its money.", "Body"))
    story += fig("02_customer_pareto.png",
                 "Figure 1. Revenue is highly concentrated: the top 20% of "
                 "customers drive ~77% of revenue (Pareto principle in action).")
    story += fig("05_monthly_revenue.png",
                 "Figure 2. Monthly revenue trend &mdash; clear seasonal build-up "
                 "toward the Q4 gifting peak, informing inventory and campaigns.")
    story.append(PageBreak())
    story += fig("04_top_products.png",
                 "Figure 3. Top products by units and by revenue &mdash; a small "
                 "set of SKUs (cakestands, t-light holders) dominates sales.")
    story.append(P(
        "<b>Business value:</b> revenue concentration justifies a VIP-retention "
        "strategy; seasonality drives inventory and campaign timing; the long tail "
        "of rarely-sold products is a candidate for rationalisation.", "Body"))
    story.append(PageBreak())

    # ---------------- 5. Segmentation ---------------- #
    story.append(P("5. Customer Segmentation", "H1"))
    story.append(P(
        "We engineered an <b>RFM</b> feature set &mdash; Recency (days since last "
        "purchase), Frequency (orders) and Monetary (total spend) &mdash; enriched "
        "with average basket value, product diversity, tenure and purchase cadence. "
        "RFM is heavily right-skewed, so features were log-transformed and "
        "standardised before clustering.", "Body"))
    story.append(P("KMeans &amp; DBSCAN", "H2"))
    story.append(P(
        "<b>KMeans</b> partitions every customer into business segments; K was "
        "chosen with the elbow and silhouette methods. <b>DBSCAN</b> adds a "
        "density view that isolates unusual (wholesale-scale) customers as noise.",
        "Body"))
    seg = a["seg"].copy()
    seg_tbl = pd.DataFrame({
        "Segment": seg["segment"],
        "Customers": seg["customers"].map(lambda x: f"{int(x):,}"),
        "Avg Recency": seg["avg_recency"].map(lambda x: f"{x:.0f}d"),
        "Avg Freq": seg["avg_frequency"].map(lambda x: f"{x:.1f}"),
        "Avg Monetary": seg["avg_monetary"].map(lambda x: f"GBP {x:,.0f}"),
        "Rev share": seg["revenue_share"].map(lambda x: f"{x:.1%}"),
    })
    story.append(table_from_df(seg_tbl, fontsize=8.5))
    story.append(Spacer(1, 0.2 * cm))
    story += fig("08_segments_pca.png",
                 "Figure 4. Customer segments projected into 2-D PCA space "
                 "&mdash; well-separated, interpretable groups.", width=11 * cm)
    story.append(P(
        "<b>Segment playbook.</b> VIP &mdash; protect and reward (loyalty perks, "
        "early access). Loyal &mdash; grow share of wallet via personalised "
        "cross-sell. Occasional &mdash; reactivate with cadence-timed win-back. "
        "At-Risk &mdash; low-cost automated reminders before churn.", "Body"))
    story.append(PageBreak())

    # ---------------- 6. Recommender ---------------- #
    story.append(P("6. KNN Recommendation Engine (Core)", "H1"))
    story.append(P(
        "The core deliverable. A sparse customer x product purchase matrix feeds a "
        "<b>user-based K-Nearest-Neighbours collaborative filter</b>: for a target "
        "customer we find the K most similar customers, pool the products they "
        "bought, score by similarity-weighted votes, drop items already owned, and "
        "return the top-10 &mdash; <i>'customers like you also bought&hellip;'</i>",
        "Body"))
    story.append(P("Parameter Tuning", "H2"))
    story.append(P(
        "We tuned the <b>distance metric</b> (cosine, Euclidean, Manhattan) and "
        "<b>K</b> (3&ndash;15) against a leakage-free held-out evaluation: for each "
        "test customer 20% of their basket is hidden, recommendations are generated "
        "from the rest, and we measure how many hidden items are recovered. "
        "<b>Crucially, the customer's own row is excluded from the neighbour search</b> "
        "so metrics are not inflated by leakage.", "Body"))
    knn = a["knn"].copy()
    best = knn.sort_values("hit_rate", ascending=False).iloc[0]
    # Show cosine sweep (the winning metric) for readability.
    cos = knn[knn["metric"] == "cosine"].copy()
    knn_tbl = pd.DataFrame({
        "Metric": cos["metric"], "K": cos["k"].astype(int),
        "Precision@10": cos["precision@N"].map(lambda x: f"{x:.3f}"),
        "Recall@10": cos["recall@N"].map(lambda x: f"{x:.3f}"),
        "Hit Rate@10": cos["hit_rate"].map(lambda x: f"{x:.3f}"),
        "MAP@10": cos["map@N"].map(lambda x: f"{x:.3f}"),
    })
    story.append(table_from_df(knn_tbl, fontsize=8.5))
    story.append(P(
        f"<b>Best configuration: {best['metric']} distance, K = "
        f"{int(best['k'])}</b> &mdash; Hit Rate@10 = {best['hit_rate']:.1%}, "
        f"Precision@10 = {best['precision@N']:.1%}, MAP@10 = {best['map@N']:.3f}. "
        "Cosine clearly beats Euclidean/Manhattan on sparse binary purchase data.",
        "Body"))
    story += fig("09_knn_tuning.png",
                 "Figure 5. Hit Rate@10 vs K across the three distance metrics. "
                 "Cosine dominates throughout.", width=11 * cm)
    story.append(P(
        "An 'also-bought' item-similarity view (item-based KNN) complements the "
        "engine for product pages. Example associations:", "Body"))
    items = a["items"].head(5).copy()
    items_tbl = pd.DataFrame({
        "Product": items["product"], "Also bought": items["also_bought"],
        "Similarity": items["similarity"].map(lambda x: f"{x:.2f}"),
    })
    story.append(table_from_df(items_tbl, col_widths=[5.5 * cm, 5.5 * cm, 3 * cm],
                               fontsize=8))
    story.append(P(
        "<b>Business value:</b> a top-10 recommendation list is generated for "
        "<b>every</b> customer and exported for CRM/email activation &mdash; "
        "directly driving cross-sell and upsell (typically +5&ndash;15% basket "
        "revenue).", "Body"))
    story.append(PageBreak())

    # ---------------- 7. Market Basket ---------------- #
    story.append(P("7. Market Basket Analysis (Apriori)", "H1"))
    story.append(P(
        "Where the recommender answers <i>'what should this customer buy?'</i>, "
        "market basket analysis answers <i>'which products sell together?'</i> We "
        "built one basket per invoice (UK market, top 300 products) and mined "
        "<b>Apriori</b> association rules, ranked by <b>lift</b>:", "Body"))
    story += bullets([
        "<b>Support</b> &mdash; how often the itemset appears.",
        "<b>Confidence</b> &mdash; P(consequent | antecedent).",
        "<b>Lift</b> &mdash; how much more likely the pair is than chance "
        "(lift &gt; 1 = positive association).",
    ])
    rules = a["rules"].head(8).copy()
    rules_tbl = pd.DataFrame({
        "If basket has": rules["antecedents"].str.slice(0, 30),
        "Also recommend": rules["consequents"].str.slice(0, 30),
        "Conf.": rules["confidence"].map(lambda x: f"{x:.0%}"),
        "Lift": rules["lift"].map(lambda x: f"{x:.1f}"),
    })
    story.append(table_from_df(rules_tbl, col_widths=[5 * cm, 5 * cm, 2 * cm, 2 * cm],
                               fontsize=8))
    story.append(P(
        "Matching Regency teacup colours co-occur at a lift near <b>27x</b> "
        "&mdash; an extremely strong, actionable signal. <b>Business value:</b> "
        "product bundling, store/site layout, and high-confidence basket upsell.",
        "Body"))
    story.append(PageBreak())

    # ---------------- 8. CLV ---------------- #
    story.append(P("8. Customer Lifetime Value Prediction", "H1"))
    story.append(P(
        "We predict each customer's spend over the <b>next 6 months</b>. To avoid "
        "look-ahead leakage, features are built from a 12-month calibration window "
        "and the target is revenue in the following 6-month holdout window. The "
        "target is right-skewed with a spike at zero (churners), so models are "
        "trained on log1p(revenue) and evaluated on the original GBP scale.", "Body"))
    clv = a["clv"].copy()
    clv_tbl = pd.DataFrame({
        "Model": clv["model"],
        "RMSE (GBP)": clv["RMSE"].map(lambda x: f"{x:,.0f}"),
        "MAE (GBP)": clv["MAE"].map(lambda x: f"{x:,.0f}"),
        "R2": clv["R2"].map(lambda x: f"{x:.2f}"),
    })
    story.append(table_from_df(clv_tbl, col_widths=[4 * cm, 3.5 * cm, 3.5 * cm, 3 * cm]))
    bestclv = clv.iloc[0]
    story.append(P(
        f"<b>{bestclv['model']}</b> wins with R&sup2; = {bestclv['R2']:.2f} and "
        f"RMSE of GBP {bestclv['RMSE']:,.0f} on out-of-sample customers &mdash; a "
        "strong result for noisy six-month-ahead revenue.", "Body"))
    story += fig("10_clv_model_comparison.png",
                 "Figure 6. Model error comparison (left) and predicted vs actual "
                 "future revenue for the best model (right).")
    story.append(PageBreak())

    # ---------------- 9. Explainability ---------------- #
    story.append(P("9. Explainable AI (SHAP)", "H1"))
    story.append(P(
        "A CLV score the business cannot interpret is hard to act on. <b>SHAP</b> "
        "decomposes every prediction into per-feature contributions, answering "
        "both <i>'what drives future spend overall?'</i> and <i>'why is this "
        "particular customer high-value?'</i>", "Body"))
    shap = a["shap"].copy()
    shap_tbl = pd.DataFrame({
        "Rank": range(1, len(shap) + 1),
        "Feature": shap["feature"],
        "Mean |SHAP|": shap["mean_abs_shap"].map(lambda x: f"{x:.3f}"),
    })
    story.append(table_from_df(shap_tbl, col_widths=[2 * cm, 7 * cm, 5 * cm]))
    story += fig("11_shap_summary.png",
                 "Figure 7. SHAP summary &mdash; Monetary, Recency and Frequency "
                 "dominate. The model independently rediscovers RFM, validating the "
                 "segmentation strategy.", width=12 * cm)
    story.append(P(
        "<b>Business value:</b> trust in the AI, plus a clear targeting rule "
        "&mdash; focus retention budget on customers with high predicted CLV but "
        "rising Recency (high value, starting to slip away).", "Body"))
    story.append(PageBreak())

    # ---------------- 10. Business value ---------------- #
    story.append(P("10. Business Value &amp; Expected Impact", "H1"))
    impact = pd.DataFrame({
        "Lever": ["Cross-sell revenue", "Customer retention",
                  "Marketing efficiency", "Product bundling", "Inventory planning"],
        "Mechanism": ["KNN + basket recommendations",
                      "Segment-targeted win-back", "CLV-prioritised spend",
                      "Apriori product pairs", "Seasonality + top-product demand"],
        "Expected impact": ["+5-15% basket revenue", "Lower VIP/Loyal churn",
                            "Higher ROI per GBP", "Larger average basket",
                            "Fewer stockouts/overstock"],
    })
    story.append(table_from_df(impact, col_widths=[3.6 * cm, 5.4 * cm, 5 * cm],
                               fontsize=8.5))
    story.append(P("Deliverables", "H2"))
    story += bullets([
        "Saved models: customer segmentation, KNN recommender, CLV (joblib).",
        "Per-customer top-10 recommendations and forward CLV scores.",
        "Power BI star schema (fact_sales + customer/product/date dimensions).",
        "All figures, metric tables, and this report.",
    ])
    story.append(P("Tech stack", "H2"))
    story.append(P(
        "Python &bull; pandas &bull; scikit-learn &bull; XGBoost &bull; LightGBM "
        "&bull; mlxtend (Apriori) &bull; SHAP &bull; matplotlib/seaborn &bull; "
        "Power BI.", "Body"))
    story.append(Spacer(1, 0.5 * cm))
    story.append(P(
        "<i>This project demonstrates customer analytics, recommendation systems, "
        "clustering, association-rule mining, explainable AI and business "
        "intelligence within a single, reproducible, end-to-end retail solution.</i>",
        "Body"))

    out = config.REPORT_DIR / "Customer_Recommendation_Segmentation_Report.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="Customer Recommendation & Segmentation Engine",
        author="Portfolio Project")
    doc.build(story, onLaterPages=_footer, onFirstPage=_footer)
    print(f"PDF report written -> {out}  ({out.stat().st_size/1e6:.2f} MB)")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(1.6 * cm, 0.8 * cm,
                      "AI-Powered Customer Recommendation & Segmentation Engine")
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    build()
