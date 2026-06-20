"""Generate the business report PDF: what was done, results, business value."""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
OUT = ROOT / "Telecom_Segmentation_Report.pdf"

NAVY = colors.HexColor("#1a4d7a")
ACCENT = colors.HexColor("#0b7285")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#eef3f8")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, textColor=NAVY,
                    spaceBefore=10, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, textColor=ACCENT,
                    spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY)
CAP = ParagraphStyle("cap", parent=styles["Normal"], fontSize=8.5, textColor=GREY,
                     alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
BULLET = ParagraphStyle("bullet", parent=BODY, alignment=TA_JUSTIFY)


def fig(name: str, width: float, caption: str) -> list:
    path = FIG / f"{name}.png"
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    img.hAlign = "CENTER"
    return [img, Paragraph(caption, CAP)]


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=10) for t in items],
        bulletType="bullet", start="•", leftIndent=14, spaceBefore=2, spaceAfter=6,
    )


def styled_table(data, col_widths, header_bg=NAVY, highlight_row=None):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight_row is not None:
        cmds.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row),
                     colors.HexColor("#d3f0e0")))
        cmds.append(("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"))
    t.setStyle(TableStyle(cmds))
    return t


def main() -> None:
    seg = pd.read_csv(REP / "phase3_segment_summary.csv")
    comp = pd.read_csv(REP / "phase5_model_comparison.csv", index_col=0)
    shap = pd.read_csv(REP / "phase6_shap_importance.csv", index_col=0)

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.6 * inch,
        title="Telecom Customer Segmentation - Business Report",
        author="triaz.malik")
    s = []

    # ---------------- Title ----------------
    s.append(Spacer(1, 1.6 * inch))
    s.append(Paragraph("Telecom Customer Segmentation &amp;<br/>Package Recommendation System",
                       ParagraphStyle("t", parent=styles["Title"], fontSize=24, leading=30,
                                      textColor=NAVY)))
    s.append(Spacer(1, 10))
    s.append(Paragraph("Business Report — Methodology, Results &amp; Value",
                       ParagraphStyle("st", parent=styles["Normal"], alignment=TA_CENTER,
                                      fontSize=13, textColor=GREY)))
    s.append(Spacer(1, 30))
    s.append(Paragraph(
        "Dataset: Orange / BigML Telecom Churn — 3,333 subscribers &bull; "
        "Overall churn rate 14.5%",
        ParagraphStyle("st2", parent=styles["Normal"], alignment=TA_CENTER,
                       fontSize=10.5, textColor=colors.black)))

    # ---------------- Executive summary ----------------
    s.append(PageBreak())
    s.append(Paragraph("1. Executive Summary", H1))
    s.append(Paragraph(
        "This project turns a raw telecom subscriber table into an actionable "
        "customer-intelligence system. We profiled 3,333 customers, engineered "
        "value and usage features, grouped customers into five business segments, "
        "built a recommendation engine that suggests a better package for each "
        "subscriber, and trained a churn-prediction model that flags at-risk "
        "customers before they leave — with an explainability layer that shows "
        "<i>why</i> each prediction is made.", BODY))
    s.append(Spacer(1, 4))
    s.append(Paragraph("Headline results", H2))
    s.append(bullets([
        "<b>Churn prediction:</b> XGBoost achieves <b>0.924 ROC-AUC</b> and "
        "<b>87% recall</b> on churners (holdout test) — most leavers are caught.",
        "<b>Segmentation:</b> five clear segments; the <b>High Risk</b> segment "
        "(728 customers) churns at <b>32%</b> while paying the <b>highest bills</b> "
        "($72.80 avg) — the priority retention target.",
        "<b>Recommendations:</b> a cross-sell / upsell opportunity is flagged for "
        "<b>~40%</b> of subscribers.",
        "<b>Top churn drivers</b> (SHAP): total charges, voicemail plan, "
        "international calls, and customer-service calls.",
    ]))

    # ---------------- What was done ----------------
    s.append(Paragraph("2. What Was Done", H1))
    s.append(Paragraph(
        "The work is organised as a reproducible 7-phase pipeline:", BODY))
    phase_data = [
        ["Phase", "Activity", "Output"],
        ["1. EDA", "Distributions, correlations, churn boxplots",
         "7 charts + insights"],
        ["2. Feature Engineering", "Total usage/charges, intl ratio, value score, segments",
         "Enriched dataset"],
        ["3. Segmentation", "K-Means + DBSCAN, profiled & named clusters",
         "5 business segments"],
        ["4. Recommendation", "KNN — 5 most-similar customers per subscriber",
         "Package per customer"],
        ["5. Churn Prediction", "KNN / RF / XGBoost + SMOTE + tuned CV",
         "Best model (XGBoost)"],
        ["6. Explainability", "SHAP on the tuned model",
         "Ranked churn drivers"],
        ["7. Business Value", "Translate analytics into operator actions",
         "This report"],
    ]
    s.append(styled_table(
        [[Paragraph(c, ParagraphStyle("c", parent=BODY, fontSize=8.5,
          textColor=colors.white if r == 0 else colors.black,
          fontName="Helvetica-Bold" if r == 0 else "Helvetica")) for c in row]
         for r, row in enumerate(phase_data)],
        col_widths=[1.3 * inch, 3.0 * inch, 2.0 * inch]))

    # ---------------- EDA results ----------------
    s.append(PageBreak())
    s.append(Paragraph("3. Key Findings (EDA)", H1))
    s.append(bullets([
        "<b>Heavy users</b> (top 10% by minutes) churn <b>47%</b> vs 14.5% overall.",
        "<b>Top payers</b> (top 10% by charges) churn <b>63%</b> — the most valuable "
        "customers are the most likely to leave.",
        "Customers with <b>≥4 service calls</b> churn <b>52%</b> — a clear distress signal.",
        "<b>International-plan</b> holders churn <b>42%</b> vs 11.5% without.",
    ]))
    row = fig("01_churn_distribution", 2.7 * inch, "Churn distribution (14.5% churn)")
    row2 = fig("05_customer_service_calls_distribution", 3.1 * inch,
               "Churn rises sharply with service calls")
    side = Table([[row[0], row2[0]]], colWidths=[3.0 * inch, 3.4 * inch])
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    s.append(side)
    s.append(Paragraph("Churn distribution &nbsp;&nbsp;|&nbsp;&nbsp; "
                       "Service calls vs churn", CAP))
    s.extend(fig("07_boxplots_churn_vs_retained", 5.6 * inch,
                 "Churned vs retained customers across key features"))

    # ---------------- Segmentation results ----------------
    s.append(PageBreak())
    s.append(Paragraph("4. Customer Segments", H1))
    s.append(Paragraph(
        "K-Means (with a DBSCAN cross-check) groups customers into five segments, "
        "each mapped to a business profile:", BODY))
    seg_order = ["High Risk Users", "Premium Users", "Voice Heavy Users",
                 "International Users", "Low Revenue Users"]
    seg = seg.set_index("Segment").loc[seg_order].reset_index()
    seg_tbl = [["Segment", "Customers", "Churn", "Avg Bill", "Value Score"]]
    for _, r in seg.iterrows():
        seg_tbl.append([r["Segment"], f"{int(r['customers'])}",
                        f"{r['churn_rate']*100:.0f}%", f"${r['avg_charges']:.2f}",
                        f"{r['avg_value']:.1f}"])
    s.append(styled_table(seg_tbl,
             col_widths=[1.8 * inch, 1.0 * inch, 0.8 * inch, 1.0 * inch, 1.0 * inch],
             highlight_row=1))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>High Risk Users</b> (highlighted) are the standout concern: they pay the "
        "most yet churn at 32% — protecting them protects the most revenue.", BODY))
    s.extend(fig("09_segments_pca_scatter", 4.7 * inch,
                 "Customer segments (K-Means) — PCA projection"))

    # ---------------- Recommendation ----------------
    s.append(PageBreak())
    s.append(Paragraph("5. Package Recommendation Engine", H1))
    s.append(Paragraph(
        "For every subscriber the engine finds the 5 most similar customers — by "
        "usage, charges, international calls, tenure and service calls — and "
        "recommends the package most common among them: "
        "<i>“Customers similar to you are using Package X.”</i> "
        "This surfaces a cross-sell / upsell opportunity for roughly "
        "<b>40% of the base</b>, and gives front-line staff a concrete next-best "
        "offer for each customer.", BODY))

    # ---------------- Churn model results ----------------
    s.append(Paragraph("6. Churn Prediction Results", H1))
    s.append(Paragraph(
        "Three models were tuned with SMOTE (to handle the 14.5% imbalance) and "
        "Stratified 5-fold cross-validation, then scored on the untouched 20% "
        "holdout split.", BODY))
    comp_tbl = [["Model", "CV ROC-AUC", "Test ROC-AUC", "Test F1"]]
    best_idx = comp["Test ROC-AUC"].astype(float).idxmax()
    hl = None
    for i, (name, r) in enumerate(comp.iterrows(), 1):
        comp_tbl.append([name, f"{r['CV ROC-AUC']:.3f}", f"{r['Test ROC-AUC']:.3f}",
                         f"{r['Test F1']:.3f}"])
        if name == best_idx:
            hl = i
    s.append(styled_table(comp_tbl,
             col_widths=[1.6 * inch, 1.3 * inch, 1.3 * inch, 1.1 * inch],
             highlight_row=hl))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>XGBoost</b> is the best model: <b>0.924 ROC-AUC</b>, 95% accuracy, and "
        "<b>87% recall on churners</b> — it catches the large majority of customers "
        "who are about to leave.", BODY))
    row = fig("12_roc_curves", 3.0 * inch, "ROC curves")
    row2 = fig("13_confusion_matrix", 2.7 * inch, "Confusion matrix (XGBoost)")
    side = Table([[row[0], row2[0]]], colWidths=[3.2 * inch, 3.0 * inch])
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    s.append(side)
    s.append(Paragraph("ROC curves &nbsp;&nbsp;|&nbsp;&nbsp; "
                       "Confusion matrix (best model)", CAP))

    # ---------------- Explainability ----------------
    s.append(PageBreak())
    s.append(Paragraph("7. Why Customers Churn (SHAP)", H1))
    s.append(Paragraph(
        "SHAP explains the model's predictions. The strongest churn drivers are:", BODY))
    top = shap.sort_values("mean_abs_shap", ascending=False).head(6)
    shap_tbl = [["Rank", "Driver", "Impact (mean |SHAP|)"]]
    for i, (name, r) in enumerate(top.iterrows(), 1):
        shap_tbl.append([str(i), name, f"{r['mean_abs_shap']:.3f}"])
    s.append(styled_table(shap_tbl,
             col_widths=[0.7 * inch, 2.8 * inch, 1.8 * inch]))
    s.extend(fig("15_shap_importance_bar", 5.2 * inch,
                 "Global feature importance (SHAP)"))

    # ---------------- Business value ----------------
    s.append(PageBreak())
    s.append(Paragraph("8. Business Value", H1))
    s.append(Paragraph(
        "The pipeline converts raw subscriber data into five concrete operator "
        "capabilities:", BODY))
    s.append(bullets([
        "<b>Identify high-value customers</b> — the Customer Value Score and "
        "segments rank the base by worth.",
        "<b>Recommend better packages</b> — the KNN engine gives every subscriber "
        "a next-best-offer, with upsell flags.",
        "<b>Reduce churn</b> — every subscriber gets a churn probability and risk "
        "band, so retention teams act <i>before</i> customers leave.",
        "<b>Increase ARPU</b> — move Low-Revenue users onto fitting paid packages "
        "and upsell Premium / International users.",
        "<b>Sharper retention campaigns</b> — target the <b>High-Risk &amp; "
        "High-Value</b> overlap, where saved revenue per customer is greatest.",
    ]))
    s.append(Spacer(1, 6))
    s.append(Paragraph("Recommended actions", H2))
    s.append(bullets([
        "Launch a proactive save-desk for High-Risk customers with ≥3 service calls.",
        "Audit the international plan — its holders churn ~4× the base; revisit "
        "pricing / quality.",
        "Feed the scored dataset (<font face='Courier'>telecom_scored.csv</font>) "
        "into a Power BI dashboard for ongoing monitoring.",
    ]))
    s.append(Spacer(1, 10))
    s.append(Paragraph(
        "<i>All figures in this report are produced by the reproducible pipeline; "
        "the scored dataset is Power BI-ready.</i>",
        ParagraphStyle("foot", parent=BODY, fontSize=9, textColor=GREY)))

    doc.build(s)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
