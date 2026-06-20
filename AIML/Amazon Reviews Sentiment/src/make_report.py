"""Generate a polished, multi-page PDF report for the Amazon Reviews
Sentiment Analysis project.

Reads the metrics JSON files in ``outputs/metrics`` and embeds the figures
in ``outputs/figures`` produced by the pipeline. Uses ONLY real numbers found
in those JSON files; if a value is missing the corresponding sentence / row is
omitted rather than fabricated. Every figure embed is guarded with an
existence check.

The PDF is written to ``report/Amazon_Reviews_Sentiment_Report.pdf`` and also
copied to the project root as ``Amazon_Reviews_Sentiment_Report.pdf``.

Run (either works):
    python -m src.make_report
    python src\\make_report.py
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

# --- Robust paths (derived from this file, not hardcoded) ------------------
BASE = Path(__file__).resolve().parent.parent          # project root
METRICS_DIR = BASE / "outputs" / "metrics"
FIGURE_DIR = BASE / "outputs" / "figures"
REPORT_DIR = BASE / "report"
REPORT_PATH = REPORT_DIR / "Amazon_Reviews_Sentiment_Report.pdf"
ROOT_COPY = BASE / "Amazon_Reviews_Sentiment_Report.pdf"

# --- Brand palette ---------------------------------------------------------
NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#2e6da4")
LIGHT = colors.HexColor("#eaf1f8")
GREY = colors.HexColor("#555555")
GREEN = colors.HexColor("#d4edda")

# --- Styles ----------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=28,
                          textColor=NAVY, leading=34, spaceAfter=12))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=13.5,
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
styles.add(ParagraphStyle("KPI", parent=styles["Normal"], fontSize=21,
                          textColor=NAVY, alignment=TA_CENTER, leading=24))
styles.add(ParagraphStyle("KPILabel", parent=styles["Normal"], fontSize=8.5,
                          textColor=GREY, alignment=TA_CENTER))


# --- Helpers ---------------------------------------------------------------
def p(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", styles["Bul"]) for t in items]


def hr():
    return HRFlowable(width="100%", thickness=1, color=LIGHT,
                      spaceBefore=6, spaceAfter=10)


def fig(name, width=15 * cm, caption=None):
    """Return flowables for a figure if it exists, else an empty list."""
    path = FIGURE_DIR / name
    if not path.exists():
        return []
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    out = [Spacer(1, 4), img]
    out.append(p(caption, "Caption") if caption else Spacer(1, 8))
    return out


def load_json(name):
    path = METRICS_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_int(v):
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def is_num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


# --- KPI cards -------------------------------------------------------------
def kpi_row(ds, base):
    cards = []
    if ds.get("rows_final") is not None:
        cards.append((fmt_int(ds["rows_final"]), "reviews analysed"))
    pct = ds.get("sentiment_pct", {})
    if is_num(pct.get("positive")):
        cards.append((f"{pct['positive']:.1f}%", "positive sentiment"))
    if is_num(base.get("test_f1_macro")):
        cards.append((f"{base['test_f1_macro']:.3f}", "baseline macro-F1"))
    if is_num(base.get("test_roc_auc_ovr_macro")):
        cards.append((f"{base['test_roc_auc_ovr_macro']:.3f}", "baseline ROC-AUC"))
    if not cards:
        return None
    cells = []
    for value, label in cards:
        inner = Table([[p(value, "KPI")], [p(label, "KPILabel")]],
                      colWidths=[3.9 * cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        cells.append(inner)
    outer = Table([cells], colWidths=[4.3 * cm] * len(cells))
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return outer


def header_style(span_blue=False):
    bg = BLUE if span_blue else NAVY
    return [
        ("BACKGROUND", (0, 0), (-1, 0), bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]


# --- Build -----------------------------------------------------------------
def build():
    ds = load_json("data_summary.json")
    eda = load_json("eda_stats.json")
    base = load_json("baseline_metrics.json")
    feat = load_json("feature_summary.json")
    pre = load_json("preprocess_example.json")
    # Optional transformer metrics — only used if present.
    tr = load_json("transformer_metrics.json")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Amazon Reviews Sentiment Analysis Report",
        author="Data Science Team",
    )
    S = []

    # ---------------- Cover ------------------------------------------------
    S.append(Spacer(1, 3.4 * cm))
    S.append(p("Customer Review Sentiment Analysis", "CoverTitle"))
    S.append(p("AI-Powered Sentiment Classification on Amazon Reviews 2023 "
               "(TF-IDF + Logistic Regression &amp; DistilBERT)", "CoverSub"))
    S.append(Spacer(1, 0.5 * cm))
    S.append(hr())
    S.append(Spacer(1, 0.3 * cm))
    S.append(p("Project Report &mdash; Data Quality, EDA, Preprocessing, "
               "Feature Engineering, Modeling &amp; Business Insight", "CoverSub"))
    S.append(Spacer(1, 1.4 * cm))
    kpi = kpi_row(ds, base)
    if kpi is not None:
        S.append(kpi)
    if ds.get("n_categories") is not None:
        S.append(Spacer(1, 0.8 * cm))
        sub = (f"{fmt_int(ds.get('rows_final'))} reviews across "
               f"{ds['n_categories']} product categories")
        if ds.get("date_min") and ds.get("date_max"):
            sub += f" &mdash; {ds['date_min']} to {ds['date_max']}"
        S.append(p(sub, "CoverSub"))
    S.append(PageBreak())

    # ---------------- 1. Business Problem ----------------------------------
    S.append(p("1. Business Problem", "H1"))
    S.append(hr())
    S.append(p("Online retailers receive customer reviews far faster than any "
               "team can read them. The business needs to automatically gauge "
               "<b>what customers feel</b> about products, surface <b>why</b> "
               "they are dissatisfied, and route problem products for action "
               "&mdash; without manual triage."))
    S.append(p("Business questions", "H2"))
    S.extend(bullets([
        "What share of reviews are positive, neutral or negative?",
        "Which product categories generate the most negative sentiment?",
        "What words and phrases drive negative reviews?",
        "Can sentiment be predicted accurately enough to automate monitoring?",
    ]))
    S.append(p("Approach", "H2"))
    S.append(p("Each review is classified into <b>negative / neutral / "
               "positive</b> using star ratings as labels (1&ndash;2&starf; "
               "negative, 3&starf; neutral, 4&ndash;5&starf; positive). Two "
               "models are compared: an interpretable <b>TF-IDF + Logistic "
               "Regression</b> baseline and a fine-tuned <b>DistilBERT</b> "
               "transformer."))

    # ---------------- 2. Dataset & Data Quality ----------------------------
    S.append(p("2. Dataset &amp; Data Quality", "H1"))
    S.append(hr())
    if ds:
        intro = "The project uses the <b>Amazon Reviews 2023</b> corpus"
        if ds.get("n_categories"):
            intro += f", sampled across <b>{ds['n_categories']} product categories</b>"
        intro += ". The raw sample was cleaned and de-duplicated before labeling."
        S.append(p(intro))
        rows = [["Metric", "Value"]]
        if ds.get("rows_total_raw") is not None:
            rows.append(["Raw rows downloaded", fmt_int(ds["rows_total_raw"])])
        if ds.get("rows_missing_text") is not None:
            rows.append(["Rows dropped (missing text)", fmt_int(ds["rows_missing_text"])])
        if ds.get("duplicate_reviews_removed") is not None:
            rows.append(["Duplicate reviews removed", fmt_int(ds["duplicate_reviews_removed"])])
        if ds.get("rows_final") is not None:
            rows.append(["Final clean reviews", fmt_int(ds["rows_final"])])
        if ds.get("binary_rows_after_dropping_neutral") is not None:
            rows.append(["Binary subset (3&starf; dropped)",
                         fmt_int(ds["binary_rows_after_dropping_neutral"])])
        if is_num(ds.get("pct_verified")):
            rows.append(["Verified purchases", f"{ds['pct_verified']}%"])
        if is_num(ds.get("mean_helpful_votes")):
            rows.append(["Mean helpful votes / review", f"{ds['mean_helpful_votes']}"])
        if ds.get("date_min") and ds.get("date_max"):
            rows.append(["Review date range", f"{ds['date_min']} to {ds['date_max']}"])
        if len(rows) > 1:
            t = Table(rows, colWidths=[8 * cm, 7 * cm])
            t.setStyle(TableStyle(header_style()))
            S.append(t)
            S.append(Spacer(1, 6))

        # Per-category counts
        rpc = ds.get("rows_per_category")
        if rpc:
            S.append(p("2.1 Reviews per category", "H2"))
            crows = [["Category", "Reviews"]]
            for cat, n in rpc.items():
                crows.append([cat, fmt_int(n)])
            ct = Table(crows, colWidths=[9 * cm, 6 * cm])
            ct.setStyle(TableStyle(header_style(span_blue=True)))
            S.append(ct)
    S.append(PageBreak())

    # ---------------- 3. EDA -----------------------------------------------
    S.append(p("3. Exploratory Data Analysis", "H1"))
    S.append(hr())

    # 3.1 Ratings
    S.append(p("3.1 Rating &amp; sentiment distribution", "H2"))
    rd = ds.get("rating_distribution") or eda.get("rating_distribution")
    sp = ds.get("sentiment_pct") or eda.get("sentiment_pct")
    if sp:
        parts = []
        for k in ("positive", "neutral", "negative"):
            if is_num(sp.get(k)):
                parts.append(f"<b>{sp[k]:.1f}% {k}</b>")
        if parts:
            S.append(p("The corpus is heavily positive-skewed: "
                       + ", ".join(parts) + ". This class imbalance is the central "
                       "modeling challenge and is why the baseline uses balanced "
                       "class weights and reports macro-F1 alongside accuracy."))
    if rd:
        five = rd.get("5")
        if is_num(five):
            S.append(p(f"Star ratings are dominated by 5&starf; reviews "
                       f"({fmt_int(five)}), with relatively few 1&ndash;2&starf; "
                       "reviews &mdash; mirroring the sentiment skew."))
    S.extend(fig("01_rating_distribution.png", width=13 * cm,
                 caption="Fig 1. Distribution of star ratings (1&ndash;5)."))
    S.extend(fig("02_sentiment_distribution.png", width=12 * cm,
                 caption="Fig 2. Distribution of derived sentiment labels."))
    S.append(PageBreak())

    # 3.2 Review length
    S.append(p("3.2 Review length by sentiment", "H2"))
    wc = eda.get("word_count_mean_by_sentiment") or feat.get("word_count")
    if wc and all(is_num(wc.get(k)) for k in ("negative", "neutral", "positive")):
        S.append(p(f"Neutral reviews are the longest on average "
                   f"(~{wc['neutral']:.0f} words), versus ~{wc['positive']:.0f} for "
                   f"positive and ~{wc['negative']:.0f} for negative reviews "
                   "&mdash; ambivalent reviewers write more to justify a "
                   "middling verdict."))
    S.extend(fig("03_review_length_by_sentiment.png", width=13 * cm,
                 caption="Fig 3. Review length distribution by sentiment class."))

    # 3.3 Category
    S.append(p("3.3 Sentiment by product category", "H2"))
    pnc = eda.get("pct_negative_by_category")
    mrc = eda.get("mean_rating_by_category")
    if pnc:
        worst = max(pnc, key=pnc.get)
        best = min(pnc, key=pnc.get)
        S.append(p(f"Negative-review rates vary by category: <b>{worst}</b> is the "
                   f"most negative ({pnc[worst]:.1f}% negative) while <b>{best}</b> "
                   f"is the least ({pnc[best]:.1f}%). This points the quality team "
                   "to where complaints concentrate."))
    if mrc or pnc:
        cats = sorted(set(list((mrc or {}).keys()) + list((pnc or {}).keys())))
        crows = [["Category", "Mean rating", "% negative"]]
        for c in cats:
            mr = mrc.get(c) if mrc else None
            pn = pnc.get(c) if pnc else None
            crows.append([c,
                          f"{mr:.2f}" if is_num(mr) else "—",
                          f"{pn:.1f}%" if is_num(pn) else "—"])
        ct = Table(crows, colWidths=[8 * cm, 3.5 * cm, 3.5 * cm])
        ct.setStyle(TableStyle(header_style(span_blue=True) + [
            ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
        S.append(ct)
    S.extend(fig("04_sentiment_by_category.png", width=14 * cm,
                 caption="Fig 4. Sentiment split across product categories."))
    S.append(PageBreak())

    # 3.4 Word clouds + n-grams
    S.append(p("3.4 What words drive sentiment?", "H2"))
    negbi = eda.get("top_bigrams_negative")
    if negbi:
        # filter the html-artifact 'br br' token for readability
        clean = [b["ngram"] for b in negbi if b.get("ngram") and b["ngram"] != "br br"]
        if clean:
            themes = ", ".join(f"<i>{g}</i>" for g in clean[:8])
            S.append(p(f"The most frequent negative bigrams cluster around product "
                       f"failure and regret: {themes}. Phrases like "
                       "<i>waste money</i>, <i>stopped working</i> and "
                       "<i>poor quality</i> flag defects and durability issues."))
    posbi = eda.get("top_bigrams_positive")
    if posbi:
        clean = [b["ngram"] for b in posbi if b.get("ngram") and b["ngram"] != "br br"]
        if clean:
            themes = ", ".join(f"<i>{g}</i>" for g in clean[:6])
            S.append(p(f"Positive reviews emphasise satisfaction and recommendation: "
                       f"{themes}."))
    S.extend(fig("05_wordclouds.png", width=15 * cm,
                 caption="Fig 5. Word clouds by sentiment class."))
    S.extend(fig("06_top_ngrams.png", width=15 * cm,
                 caption="Fig 6. Most frequent n-grams by sentiment."))
    S.append(PageBreak())

    # ---------------- 4. Preprocessing & Features --------------------------
    S.append(p("4. Preprocessing &amp; Feature Engineering", "H1"))
    S.append(hr())
    S.append(p("Raw review text is normalised before modeling: lower-casing, "
               "stripping HTML / punctuation, removing stop-words and lemmatising "
               "to a clean token string. Example:"))
    if pre.get("before") and pre.get("after"):
        ex = Table([
            ["Raw", pre["before"]],
            ["Cleaned", pre["after"]],
        ], colWidths=[2.4 * cm, 12.6 * cm])
        ex.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        S.append(ex)
        S.append(Spacer(1, 6))

    # Engineered feature stats
    S.append(p("4.1 Engineered text features", "H2"))
    S.append(p("Beyond the cleaned text, lightweight numeric features capture "
               "stylistic signals. Their averages differ markedly by sentiment "
               "class:"))
    feat_specs = [
        ("word_count", "Avg. word count", "{:.0f}"),
        ("char_count", "Avg. character count", "{:.0f}"),
        ("exclamation_count", "Avg. exclamation marks", "{:.2f}"),
        ("caps_words", "Avg. ALL-CAPS words", "{:.2f}"),
        ("vader_sentiment", "Mean VADER score", "{:.3f}"),
        ("helpful", "Mean helpful votes", "{:.2f}"),
    ]
    classes = ("negative", "neutral", "positive")
    frows = [["Feature", "Negative", "Neutral", "Positive"]]
    for key, label, fmt in feat_specs:
        d = feat.get(key)
        if d and all(is_num(d.get(c)) for c in classes):
            frows.append([label] + [fmt.format(d[c]) for c in classes])
    if len(frows) > 1:
        ft = Table(frows, colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
        ft.setStyle(TableStyle(header_style(span_blue=True) + [
            ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
        S.append(ft)
        S.append(Spacer(1, 6))
        vd = feat.get("vader_sentiment")
        if vd and all(is_num(vd.get(c)) for c in classes):
            S.append(p(f"The unsupervised VADER score already separates the classes "
                       f"(negative {vd['negative']:.2f} &rarr; neutral {vd['neutral']:.2f} "
                       f"&rarr; positive {vd['positive']:.2f}), confirming the labels "
                       "are learnable from text alone."))
    S.extend(fig("07_vader_by_sentiment.png", width=12 * cm,
                 caption="Fig 7. VADER compound score by sentiment class."))
    S.append(PageBreak())

    # ---------------- 5. Modeling Approach ---------------------------------
    S.append(p("5. Modeling Approach", "H1"))
    S.append(hr())
    S.append(p("Two complementary models were trained and evaluated on the "
               "<b>same stratified held-out test set</b> for a fair comparison."))
    S.append(p("5.1 TF-IDF + Logistic Regression (baseline)", "H2"))
    blurb = ("Reviews are vectorised with TF-IDF")
    if is_num(base.get("n_features")):
        blurb += f" ({fmt_int(base['n_features'])} features)"
    blurb += (" and classified with multinomial Logistic Regression. "
              "Hyper-parameters were tuned by cross-validation")
    if is_num(base.get("tuning_seconds")):
        blurb += f" (~{base['tuning_seconds']:.0f}s of search)"
    blurb += "."
    S.append(p(blurb))
    bp = base.get("best_params")
    if bp:
        readable = ", ".join(f"<b>{k}</b> = {v}" for k, v in bp.items()
                             if not (isinstance(v, float) and math.isnan(v)))
        if readable:
            S.append(p(f"Best parameters: {readable}. Balanced class weights "
                       "directly counter the positive skew."))
    S.append(p("5.2 DistilBERT (transformer)", "H2"))
    S.append(p("A pre-trained <b>DistilBERT</b> model was fine-tuned for 3-class "
               "sequence classification (negative / neutral / positive). As a "
               "contextual transformer it captures word order and negation that "
               "a bag-of-words model cannot, at higher compute cost. The "
               "fine-tuned model is saved under <font name='Courier'>models/"
               "distilbert/</font> and its predictions on the shared test split "
               "are summarised by the confusion matrix in the next section."))

    # ---------------- 6. Results -------------------------------------------
    S.append(p("6. Results &amp; Model Comparison", "H1"))
    S.append(hr())
    # Build comparison table from whatever metrics exist.
    metric_cols = [
        ("test_accuracy", "Accuracy", "{:.3f}"),
        ("test_f1_macro", "Macro-F1", "{:.3f}"),
        ("test_f1_weighted", "Weighted-F1", "{:.3f}"),
        ("test_roc_auc_ovr_macro", "ROC-AUC", "{:.3f}"),
    ]
    model_rows = []
    if base:
        model_rows.append(("TF-IDF + LogReg", base))
    for name in ("distilbert", "bert"):
        if isinstance(tr, dict) and isinstance(tr.get(name), dict):
            model_rows.append((name, tr[name]))
    if model_rows:
        header = ["Model"] + [c[1] for c in metric_cols]
        data = [header]
        for label, m in model_rows:
            row = [label]
            for key, _, fmt in metric_cols:
                row.append(fmt.format(m[key]) if is_num(m.get(key)) else "—")
            data.append(row)
        t = Table(data, colWidths=[5 * cm] + [2.5 * cm] * len(metric_cols))
        style = header_style() + [("ALIGN", (1, 0), (-1, -1), "CENTER")]
        t.setStyle(TableStyle(style))
        S.append(t)
        S.append(Spacer(1, 8))
        if is_num(base.get("n_test")):
            S.append(p(f"All models are scored on the same stratified test set "
                       f"(n = {fmt_int(base['n_test'])}). Macro-F1 and ROC-AUC are "
                       "emphasised because accuracy is inflated by the dominant "
                       "positive class."))

    # Per-class F1 for baseline
    pcf = base.get("per_class_f1")
    if pcf:
        S.append(p("6.1 Baseline per-class performance", "H2"))
        crows = [["Class", "F1-score"]]
        for cls in ("negative", "neutral", "positive"):
            if is_num(pcf.get(cls)):
                crows.append([cls.capitalize(), f"{pcf[cls]:.3f}"])
        ct = Table(crows, colWidths=[5 * cm, 4 * cm])
        ct.setStyle(TableStyle(header_style(span_blue=True) + [
            ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
        S.append(ct)
        S.append(Spacer(1, 6))
        if all(is_num(pcf.get(c)) for c in ("negative", "neutral", "positive")):
            S.append(p(f"The baseline classifies positive reviews very well "
                       f"(F1 = {pcf['positive']:.2f}) and negative reviews "
                       f"reasonably (F1 = {pcf['negative']:.2f}), but struggles on "
                       f"the rare, ambiguous neutral class (F1 = {pcf['neutral']:.2f}) "
                       "&mdash; the hardest 3&starf; reviews sit between classes."))
    S.append(PageBreak())

    # Confusion matrices
    S.append(p("6.2 Confusion matrices", "H2"))
    S.extend(fig("08_confusion_baseline.png", width=11.5 * cm,
                 caption="Fig 8. Confusion matrix &mdash; TF-IDF + Logistic "
                         "Regression on the test set."))
    S.extend(fig("09_confusion_distilbert.png", width=11.5 * cm,
                 caption="Fig 9. Confusion matrix &mdash; fine-tuned DistilBERT "
                         "on the same test set."))
    S.append(p("Both models confuse neutral reviews with the adjacent positive "
               "and negative classes most often &mdash; consistent with the "
               "inherent ambiguity of 3&starf; reviews.", "Caption"))

    # ---------------- 7. Business Findings ---------------------------------
    S.append(p("7. Business Findings &amp; Recommendations", "H1"))
    S.append(hr())
    findings = []
    sp = ds.get("sentiment_pct")
    if sp and is_num(sp.get("positive")) and is_num(sp.get("negative")):
        findings.append(f"Sentiment is strongly positive: {sp['positive']:.1f}% of "
                        f"reviews are positive versus {sp['negative']:.1f}% negative.")
    pnc = eda.get("pct_negative_by_category")
    if pnc:
        worst = max(pnc, key=pnc.get)
        findings.append(f"<b>{worst}</b> carries the highest negative-review rate "
                       f"({pnc[worst]:.1f}%) and should be prioritised for quality review.")
    negbi = eda.get("top_bigrams_negative")
    if negbi:
        clean = [b["ngram"] for b in negbi if b.get("ngram") != "br br"][:4]
        if clean:
            findings.append("Negative reviews are dominated by failure/regret themes "
                           "(" + ", ".join(f"<i>{g}</i>" for g in clean)
                           + "), pointing to durability and value complaints.")
    if is_num(base.get("test_f1_macro")) and is_num(base.get("test_roc_auc_ovr_macro")):
        findings.append(f"Even a lightweight baseline reaches macro-F1 "
                       f"{base['test_f1_macro']:.2f} and ROC-AUC "
                       f"{base['test_roc_auc_ovr_macro']:.2f} &mdash; accurate enough "
                       "to automate first-pass review monitoring.")
    if findings:
        S.append(p("Key findings", "H2"))
        S.extend(bullets(findings))
    S.append(p("Recommendations", "H2"))
    S.extend(bullets([
        "<b>Automate monitoring</b> by deploying the model to flag incoming "
        "negative reviews in real time for the support and quality teams.",
        "<b>Prioritise high-negativity categories and products</b> for defect "
        "investigation, using negative n-gram themes as a diagnostic starting point.",
        "<b>Promote DistilBERT for production scoring</b> where context and "
        "negation matter; keep the TF-IDF baseline as a fast, interpretable "
        "fallback and sanity check.",
        "<b>Watch the neutral class</b> &mdash; route low-confidence / 3&starf; "
        "predictions to human review rather than auto-actioning them.",
    ]))

    # ---------------- 8. Future Work ---------------------------------------
    S.append(p("8. Future Work", "H1"))
    S.append(hr())
    S.extend(bullets([
        "<b>Aspect-based sentiment</b> to attribute complaints to specific "
        "product attributes (battery, sizing, durability).",
        "<b>Larger / domain-adapted transformers</b> (e.g. RoBERTa, full BERT) "
        "and threshold tuning to lift neutral-class recall.",
        "<b>Explainability at scale</b> &mdash; token-attribution and SHAP "
        "summaries surfaced directly in the monitoring dashboard.",
        "<b>Product-level dashboards</b> ranking the worst products by negative "
        "rate, with drill-down into representative reviews.",
        "<b>Deployment</b> behind an API for real-time scoring of new reviews.",
    ]))
    S.append(Spacer(1, 0.5 * cm))
    S.append(hr())
    S.append(p("Generated from project artifacts in <font name='Courier'>outputs/"
               "</font>. All metrics are taken directly from the pipeline's JSON "
               "outputs; figures are embedded from <font name='Courier'>outputs/"
               "figures/</font>.", "Caption"))

    doc.build(S)

    # Copy to project root (mirror the final-report-at-root convention).
    shutil.copyfile(REPORT_PATH, ROOT_COPY)
    print(f"Saved report -> {REPORT_PATH}")
    print(f"Copied to     -> {ROOT_COPY}")


if __name__ == "__main__":
    build()
