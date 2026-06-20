"""Generate a professional multi-page PDF report for the Credit Card Fraud
Detection project.

Reads the real metrics produced by the notebooks
(reports/model_comparison.csv|json, reports/threshold.json) and embeds the
existing figures under reports/figures/. NO metric is fabricated: every number
comes from a CSV/JSON artifact or the README/docx narrative, and every figure
embed is guarded with an existence check.

Run:  python make_report.py

Output:
  reports/Credit_Card_Fraud_Detection_Report.pdf
  Credit_Card_Fraud_Detection_Report.pdf   (copy at project root)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

# --- Paths (derived from this script's location) ---------------------------
BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"
REPORT_PATH = REPORTS / "Credit_Card_Fraud_Detection_Report.pdf"
ROOT_COPY = BASE / "Credit_Card_Fraud_Detection_Report.pdf"

# --- Brand palette ---------------------------------------------------------
NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#2e6da4")
LIGHT = colors.HexColor("#eaf1f8")
GREY = colors.HexColor("#555555")
GREEN = colors.HexColor("#d4edda")

# --- Styles ----------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=30,
                          textColor=NAVY, leading=36, spaceAfter=12))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=14,
                          textColor=GREY, alignment=TA_CENTER, leading=20))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17,
                          textColor=NAVY, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                          textColor=BLUE, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5,
                          leading=15, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle("Bul", parent=styles["Body"], leftIndent=14,
                          bulletIndent=4, spaceAfter=2))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle("KPI", parent=styles["Normal"], fontSize=22,
                          textColor=NAVY, alignment=TA_CENTER, leading=24))
styles.add(ParagraphStyle("KPILabel", parent=styles["Normal"], fontSize=8.5,
                          textColor=GREY, alignment=TA_CENTER, leading=11))


def p(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", styles["Bul"]) for t in items]


def fig(name, width=15 * cm, caption=None):
    """Return flowables for a figure if it exists, else an empty list."""
    path = FIGURES / name
    if not path.exists():
        return []
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    out = [Spacer(1, 4), img]
    if caption:
        out.append(p(caption, "Caption"))
    else:
        out.append(Spacer(1, 8))
    return out


def hr():
    return HRFlowable(width="100%", thickness=1, color=LIGHT,
                      spaceBefore=6, spaceAfter=10)


# --- Artifact loading ------------------------------------------------------
def load_json(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_comparison():
    """Return the model-comparison DataFrame (index = model name) or None."""
    path = REPORTS / "model_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index.name = "Model"
    return df


# --- KPI cards -------------------------------------------------------------
def kpi_card(value, label):
    inner = Table([[p(value, "KPI")], [p(label, "KPILabel")]],
                  colWidths=[5.2 * cm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return inner


def kpi_row(cmp_df, thr):
    """Headline KPI cards from real artifacts (omit any value we don't have)."""
    cards = []

    # Best PR-AUC across compared models.
    if cmp_df is not None and "PR_AUC" in cmp_df.columns:
        best_name = cmp_df["PR_AUC"].idxmax()
        best_pr = cmp_df["PR_AUC"].max()
        cards.append((f"{best_pr:.3f}", f"Best PR-AUC<br/>({best_name})"))

    # Recall at the chosen operating threshold.
    if thr is not None and "recall" in thr:
        cards.append((f"{thr['recall'] * 100:.0f}%",
                      "Recall at operating threshold"))

    # Precision at that same threshold.
    if thr is not None and "precision" in thr:
        cards.append((f"{thr['precision'] * 100:.1f}%",
                      "Precision at operating threshold"))

    if not cards:
        return None

    cells = [kpi_card(v, l) for v, l in cards]
    outer = Table([cells], colWidths=[5.6 * cm] * len(cells))
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return outer


# --- Model comparison table ------------------------------------------------
def comparison_table(cmp_df):
    header = ["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]
    data = [header]
    cols = ["PR_AUC", "ROC_AUC", "precision", "recall", "F1"]
    best_pr_name = cmp_df["PR_AUC"].idxmax() if "PR_AUC" in cmp_df.columns else None
    winner_row = None
    for i, (name, r) in enumerate(cmp_df.iterrows(), start=1):
        row = [str(name)]
        for c in cols:
            row.append(f"{r[c]:.3f}" if c in cmp_df.columns and pd.notna(r[c]) else "—")
        data.append(row)
        if name == best_pr_name:
            winner_row = i

    t = Table(data, colWidths=[5.2 * cm, 2 * cm, 2 * cm, 2.2 * cm, 2 * cm, 1.8 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    if winner_row is not None:
        style.append(("BACKGROUND", (0, winner_row), (-1, winner_row), GREEN))
        style.append(("FONTNAME", (0, winner_row), (-1, winner_row), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def simple_table(data, col_widths, header_color=BLUE):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build():
    cmp_df = load_comparison()
    thr = load_json(REPORTS / "threshold.json")

    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Credit Card Fraud Detection Report",
        author="Muhammad Tahir Riaz",
    )
    S = []

    # ---------------- Cover ------------------------------------------------
    S.append(Spacer(1, 3.2 * cm))
    S.append(p("Credit Card Fraud Detection", "CoverTitle"))
    S.append(p("Detecting Fraudulent Card Transactions on a Highly "
               "Imbalanced Dataset", "CoverSub"))
    S.append(Spacer(1, 0.4 * cm))
    S.append(hr())
    S.append(Spacer(1, 0.2 * cm))
    S.append(p("EDA &rarr; Feature Engineering &rarr; Model Comparison &rarr; "
               "Tuning &rarr; SHAP &rarr; Business Threshold", "CoverSub"))
    S.append(Spacer(1, 1.3 * cm))
    kr = kpi_row(cmp_df, thr)
    if kr is not None:
        S.append(kr)
    S.append(Spacer(1, 1.2 * cm))
    S.append(p("Muhammad Tahir Riaz", "CoverSub"))
    S.append(p("github.com/triaz-malik/Machine-Learning", "Caption"))
    S.append(PageBreak())

    # ---------------- Executive Summary -----------------------------------
    S.append(p("Executive Summary", "H1"))
    S.append(hr())
    S.append(p("We built a fraud-detection system on roughly 1.85 million card "
               "transactions where only <b>0.58%</b> are fraudulent. Because the "
               "data is so imbalanced, accuracy is meaningless &mdash; a model "
               "that always predicts &ldquo;not fraud&rdquo; scores 99.4% "
               "accuracy while catching zero fraud. We therefore optimise "
               "<b>recall</b>, <b>precision</b>, and <b>PR-AUC</b> instead."))
    if cmp_df is not None and "PR_AUC" in cmp_df.columns:
        base_pr = cmp_df["PR_AUC"].min()
        best_pr = cmp_df["PR_AUC"].max()
        best_name = cmp_df["PR_AUC"].idxmax()
        txt = (f"Moving from a linear baseline to a tuned tree model lifted "
               f"PR-AUC from <b>{base_pr:.2f}</b> to <b>{best_pr:.2f}</b> "
               f"({best_name}).")
        if thr is not None and "recall" in thr and "precision" in thr:
            txt += (f" At a business-chosen operating point the model catches "
                    f"<b>{thr['recall'] * 100:.0f}% of fraud</b> at "
                    f"<b>{thr['precision'] * 100:.1f}% precision</b> &mdash; "
                    f"flagging only ~0.35% of legitimate transactions for "
                    f"review. Per 10,000 fraud cases this turns roughly 5,840 "
                    f"caught frauds (baseline) into ~9,000.")
        S.append(p(txt))

    # ---------------- 1. Business Problem ---------------------------------
    S.append(p("1. Business Problem", "H1"))
    S.append(hr())
    S.append(p("Banks must decide, in real time, whether to approve or block "
               "each card transaction. Fraud is rare but expensive, and the "
               "cost structure is <b>asymmetric</b>:"))
    S.extend(bullets([
        "<b>Missed fraud (false negative)</b> &mdash; direct financial loss and "
        "chargebacks.",
        "<b>False alarm (false positive)</b> &mdash; manual-review cost and "
        "customer friction.",
    ]))
    S.append(p("The objective is to maximise the share of fraud caught "
               "(<b>recall</b>) while keeping false alarms low enough that the "
               "review team can cope (<b>precision</b>)."))
    S.append(p("Why PR-AUC, not accuracy", "H2"))
    S.append(p("With a 0.58% positive rate, accuracy and even ROC-AUC are "
               "misleadingly optimistic &mdash; both are dominated by the huge "
               "negative class. The <b>precision&ndash;recall curve</b> focuses "
               "entirely on the rare fraud class, so <b>PR-AUC</b> is the "
               "honest headline metric for this problem and the one we tune and "
               "rank models on."))

    # ---------------- 2. Dataset & Class Imbalance ------------------------
    S.append(p("2. Dataset &amp; Class Imbalance", "H1"))
    S.append(hr())
    S.append(p("<b>Dataset:</b> simulated credit-card transactions (Sparkov), "
               "pre-split into train and test files. Fraud is a tiny fraction "
               "of all activity:"))
    S.append(simple_table([
        ["Split", "Transactions", "Fraud (rate)"],
        ["Train", "1,296,675", "7,506  (0.579%)"],
        ["Test (held-out)", "555,719", "2,145  (0.386%)"],
    ], [5 * cm, 5 * cm, 5 * cm]))
    S.append(Spacer(1, 6))
    S.extend(fig("01_class_imbalance.png", width=9 * cm, caption=
                 "Fig 1. Extreme class imbalance &mdash; fraud is 0.58% of all "
                 "transactions (log scale)."))
    S.append(PageBreak())

    # ---------------- 3. EDA ----------------------------------------------
    S.append(p("3. Exploratory Data Analysis", "H1"))
    S.append(hr())
    S.append(p("Profiling the full training set surfaced three dominant fraud "
               "signals, each later confirmed by the model:"))
    S.append(p("3.1 Amount", "H2"))
    S.append(p("Fraudulent transactions skew to far higher amounts &mdash; mean "
               "<b>$531</b> vs <b>$68</b> for legitimate transactions. Amount "
               "(and its log) is the single strongest signal."))
    S.extend(fig("02_amount_by_class.png", width=13 * cm, caption=
                 "Fig 2. Transaction amount by class &mdash; fraud skews high."))
    S.append(p("3.2 Time of day", "H2"))
    S.append(p("Fraud is a night-time phenomenon: rates spike between "
               "<b>22:00 and 03:59</b>, up to ~25&times; the daytime rate."))
    S.extend(fig("03_fraud_by_hour.png", width=14 * cm, caption=
                 "Fig 3. Fraud rate by hour of day."))
    S.append(PageBreak())
    S.append(p("3.3 Merchant category", "H2"))
    S.append(p("Online categories (<b>shopping_net</b>, <b>misc_net</b>) carry "
               "the highest fraud rates &mdash; in-person categories are far "
               "safer."))
    S.extend(fig("04_fraud_by_category.png", width=13 * cm, caption=
                 "Fig 4. Fraud rate by merchant category."))
    S.append(p("3.4 A signal that was <i>not</i> there", "H2"))
    S.append(p("Home&ndash;merchant geographic distance is a common fraud "
               "heuristic, but profiling showed both classes average ~76 km "
               "from home. It carries <b>no signal</b> on this simulated data, "
               "so it was dropped &mdash; verified, not assumed."))
    S.append(PageBreak())

    # ---------------- 4. Feature Engineering ------------------------------
    S.append(p("4. Feature Engineering", "H1"))
    S.append(hr())
    S.append(p("From the raw columns we engineered a leak-safe feature set "
               "(<font name='Courier'>src/features.py</font>). All transforms "
               "are pure / row-wise so there is no train&rarr;test leakage. PII "
               "and identifiers were dropped so the model cannot memorise "
               "individuals; leakage and non-predictive columns were removed."))
    S.append(simple_table([
        ["Feature group", "Details / rationale"],
        ["Amount", "amt and log_amt — the single strongest signal."],
        ["Time of day", "hour and is_night (22:00–03:59), where fraud concentrates."],
        ["Calendar", "day_of_week, is_weekend."],
        ["Demographic", "age (from dob), gender_M, log city population."],
        ["Merchant category", "category — one-hot encoded."],
        ["Dropped — PII", "first, last, street, trans_num, raw cc_num, raw dob."],
        ["Dropped — leakage", "unix_time (duplicates the timestamp)."],
        ["Dropped — no signal", "home–merchant distance (both classes ~76 km)."],
    ], [4.2 * cm, 10.8 * cm]))
    S.append(Spacer(1, 6))
    S.append(p("Class imbalance itself was handled at training time with "
               "<b>class weights</b> (Logistic Regression, Random Forest) and "
               "<b>scale_pos_weight</b> (XGBoost), not by discarding data."))
    S.append(PageBreak())

    # ---------------- 5. Modeling & Comparison ----------------------------
    S.append(p("5. Modeling &amp; Comparison", "H1"))
    S.append(hr())
    S.append(p("We trained an explainable baseline, then progressively stronger "
               "models &mdash; each evaluated on the <b>same held-out test set</b> "
               "with the same metrics."))
    S.append(p("5.1 Baseline (Logistic Regression)", "H2"))
    S.append(p("The linear baseline establishes a floor and remains fully "
               "interpretable via its coefficients, but cannot represent the "
               "feature interactions that define fraud."))
    S.extend(fig("05_baseline_pr_confusion.png", width=15 * cm, caption=
                 "Fig 5. Baseline precision-recall curve and confusion matrix."))
    S.extend(fig("06_baseline_coefficients.png", width=9.5 * cm, caption=
                 "Fig 6. Baseline logistic-regression coefficients."))
    S.append(PageBreak())
    S.append(p("5.2 Model comparison", "H2"))
    if cmp_df is not None:
        S.append(comparison_table(cmp_df))
        S.append(Spacer(1, 8))
        if "PR_AUC" in cmp_df.columns:
            best_name = cmp_df["PR_AUC"].idxmax()
            best_pr = cmp_df["PR_AUC"].max()
            base_pr = cmp_df["PR_AUC"].min()
            S.append(p(f"<b>PR-AUC is the right metric for imbalanced fraud.</b> "
                       f"It leaps from <b>{base_pr:.3f}</b> (linear baseline) to "
                       f"<b>{best_pr:.3f}</b> ({best_name}) once we move to tree "
                       f"models. Note ROC-AUC is already ~0.96+ for every model "
                       f"&mdash; a vivid reminder that ROC-AUC flatters models on "
                       f"imbalanced data while PR-AUC reveals the real gap."))
    else:
        S.append(p("model_comparison.csv not found — run the comparison "
                   "notebook to regenerate it.", "Body"))
    S.append(p("Why trees win: fraud lives in <b>interactions</b> &mdash; high "
               "amount <i>and</i> night <i>and</i> an online category. A linear "
               "model cannot represent these; gradient-boosted trees can.", "Body"))
    S.extend(fig("07_model_comparison_pr.png", width=12 * cm, caption=
                 "Fig 7. Precision-recall curves: tree models dominate across "
                 "the whole range."))
    S.extend(fig("08_pr_auc_bars.png", width=11 * cm, caption=
                 "Fig 8. PR-AUC by model."))
    S.append(p("<b>Tuning:</b> a PR-AUC-scored RandomizedSearchCV for XGBoost "
               "selected max_depth=6, learning_rate=0.05, n_estimators=600, "
               "subsample=0.7, gamma=1 (CV PR-AUC 0.90).", "Body"))
    S.append(PageBreak())

    # ---------------- 6. Threshold Selection ------------------------------
    S.append(p("6. Threshold Selection", "H1"))
    S.append(hr())
    S.append(p("Probabilities still need an operating threshold. The bank "
               "policy is simple: <b>catch at least 90% of fraud</b>. We pick "
               "the highest-precision probability threshold that meets that "
               "recall and read off the operational trade-off on the held-out "
               "test set."))
    if thr is not None:
        rows = [["Operating point", "Value"]]
        if "threshold" in thr:
            rows.append(["Decision threshold", f"{thr['threshold']:.4f}"])
        if "target_recall" in thr:
            rows.append(["Target recall (policy)", f"{thr['target_recall'] * 100:.0f}%"])
        if "recall" in thr:
            rows.append(["Achieved recall", f"{thr['recall'] * 100:.1f}%"])
        if "precision" in thr:
            rows.append(["Precision at threshold", f"{thr['precision'] * 100:.1f}%"])
        if "pr_auc" in thr:
            rows.append(["PR-AUC (tuned model)", f"{thr['pr_auc']:.3f}"])
        S.append(simple_table(rows, [8 * cm, 7 * cm], header_color=NAVY))
        S.append(Spacer(1, 8))
        S.append(p("The precision&ndash;recall trade-off is explicit: pushing "
                   "recall to 90% drops precision to ~50%, meaning roughly half "
                   "of all flags are genuine fraud. On 553,574 legitimate test "
                   "transactions that is only ~0.35% sent for review &mdash; a "
                   "volume the review team can absorb."))
    else:
        S.append(p("threshold.json not found — run the tuning notebook to "
                   "regenerate it.", "Body"))
    S.append(p("Outcome at the chosen threshold", "H2"))
    S.append(simple_table([
        ["Outcome at the 90%-recall threshold", "Result"],
        ["Frauds caught (true positives)", "1,931 of 2,145  —  90.0%"],
        ["Frauds missed (false negatives)", "214  —  10.0%"],
        ["False alarms sent to review", "1,910 of 553,574 legit  —  0.35%"],
        ["Precision (flags that are real fraud)", "50.3%"],
    ], [9.5 * cm, 5.5 * cm], header_color=NAVY))
    S.append(PageBreak())

    # ---------------- 7. Explainability (SHAP) ----------------------------
    S.append(p("7. Explainability (SHAP)", "H1"))
    S.append(hr())
    S.append(p("The model is <b>auditable, not a black box</b>. SHAP confirms "
               "the EDA story: amount and night-time dominate, followed by hour "
               "and online categories."))
    S.extend(fig("09_shap_beeswarm.png", width=12 * cm, caption=
                 "Fig 9. Global SHAP (beeswarm) &mdash; log_amt and is_night are "
                 "the top drivers of fraud probability."))
    S.extend(fig("10_shap_waterfall.png", width=12 * cm, caption=
                 "Fig 10. Local SHAP (waterfall) &mdash; why one specific "
                 "transaction was flagged as fraud."))
    S.append(PageBreak())

    # ---------------- 8. Business Impact & Future Work --------------------
    S.append(p("8. Business Impact &amp; Future Work", "H1"))
    S.append(hr())
    S.append(p("Translated to scale &mdash; per 10,000 fraud cases:"))
    S.extend(bullets([
        "<b>Baseline Logistic Regression (58% recall):</b> catches ~5,840, "
        "misses ~4,160.",
        "<b>Tuned XGBoost (90% recall):</b> catches ~9,000, misses ~1,000.",
    ]))
    S.append(p("<b>Net impact:</b> missed fraud falls by ~75%, while only 0.35% "
               "of legitimate customers experience a review &mdash; a strong "
               "recall gain at a manageable false-alarm cost."))
    S.append(p("Key findings", "H2"))
    S.extend(bullets([
        "<b>Amount</b> is the strongest signal — fraud averages $531 vs $68.",
        "<b>Time of day</b> matters enormously — fraud spikes 22:00–03:59.",
        "<b>Online categories</b> (shopping_net, misc_net) carry the highest "
        "fraud rates.",
        "<b>Geo-distance is not predictive</b> here — verified on the data, not "
        "assumed.",
        "<b>SHAP confirms the EDA</b> — log_amt and is_night are the top model "
        "drivers.",
    ]))
    S.append(p("Future work", "H2"))
    S.extend(bullets([
        "<b>Cost-sensitive threshold</b> driven by real chargeback vs review "
        "costs, not a fixed 90%-recall policy.",
        "<b>Velocity / sequence features</b> (transactions per card per hour) "
        "to catch bursty fraud rings.",
        "<b>Probability calibration</b> so flagged scores map to real fraud "
        "likelihoods for triage.",
        "<b>Real-time deployment</b> behind a scoring API with monitoring for "
        "drift and concept change.",
    ]))
    S.append(Spacer(1, 0.5 * cm))
    S.append(hr())
    S.append(p("Generated from project artifacts in <font name='Courier'>"
               "reports/</font>. Metrics sourced from model_comparison.csv and "
               "threshold.json; figures from reports/figures/.", "Caption"))

    doc.build(S)
    shutil.copyfile(REPORT_PATH, ROOT_COPY)
    print(f"Saved report -> {REPORT_PATH}")
    print(f"Copied to    -> {ROOT_COPY}")


if __name__ == "__main__":
    build()
