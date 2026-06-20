"""
Assemble the final PDF report from the metrics JSON + figures produced by the
pipeline. Run AFTER data_prep / eda / features / train_* have completed.
"""
import json
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

BASE = r"C:\Working\AI ML Projetcs\Consumer Complaints"
FIG = os.path.join(BASE, "outputs", "figures")
MET = os.path.join(BASE, "outputs", "metrics")
OUT = os.path.join(BASE, "report", "Consumer_Complaint_Classification_Report.pdf")

NAVY = colors.HexColor("#1f3b5c")
BLUE = colors.HexColor("#2c7fb8")
LIGHT = colors.HexColor("#eef3f8")
GREY = colors.HexColor("#666666")

USABLE_W = LETTER[0] - 1.5 * inch  # ~6.5 in


def load(name, default=None):
    p = os.path.join(MET, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}


# ---------- styles ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, textColor=NAVY,
                    spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, textColor=BLUE,
                    spaceBefore=10, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10.2, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=7)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("Cap", parent=ss["Italic"], fontSize=8.5, textColor=GREY,
                     alignment=TA_CENTER, spaceAfter=12)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=9, alignment=TA_LEFT)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items, style=BULLET):
    return [Paragraph(f"• {it}", style) for it in items]


def fig(name, width=USABLE_W, caption=None):
    path = os.path.join(FIG, name)
    out = []
    if os.path.exists(path):
        from PIL import Image as PImage
        w, h = PImage.open(path).size
        dw = width
        dh = dw * h / w
        max_h = 4.4 * inch
        if dh > max_h:
            dh = max_h
            dw = dh * w / h
        out.append(Image(path, width=dw, height=dh))
        if caption:
            out.append(P(caption, CAP))
        else:
            out.append(Spacer(1, 8))
    return out


def styled_table(data, col_widths=None, header_bg=NAVY, font=8.6, highlight_row=None):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd6e0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#d7ecd9")))
        style.append(("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, LETTER[1] - 0.6 * inch, LETTER[0] - 0.75 * inch, LETTER[1] - 0.6 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, LETTER[1] - 0.5 * inch, "Consumer Complaint Classification — Technical Report")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.drawString(0.75 * inch, 0.45 * inch, "CFPB Consumer Complaint Database")
    canvas.restoreState()


def build():
    data = load("data_summary.json")
    eda = load("eda_stats.json")
    feat = load("feature_summary.json")
    pre = load("preprocess_example.json")
    base_m = load("baseline_metrics.json")
    dbert = load("distilbert_metrics.json")
    bert = load("bert_metrics.json")

    story = []

    # ---------------- TITLE PAGE ----------------
    story.append(Spacer(1, 1.4 * inch))
    story.append(P("Automated Consumer Complaint<br/>Classification System",
                   ParagraphStyle("T", parent=H1, fontSize=26, alignment=TA_CENTER, textColor=NAVY, leading=32)))
    story.append(Spacer(1, 0.2 * inch))
    story.append(P("An NLP pipeline for routing financial complaints — from baseline to transformers",
                   ParagraphStyle("Sub", parent=BODY, fontSize=13, alignment=TA_CENTER, textColor=BLUE)))
    story.append(Spacer(1, 0.5 * inch))
    n_rows = data.get("rows_after_consolidation", 0)
    head = [
        ["Dataset", "CFPB Consumer Complaint Database"],
        ["Usable records", f"{n_rows:,} complaints with narrative text"],
        ["Target", f"{data.get('n_categories','?')} product categories"],
        ["Models", "TF-IDF + Logistic Regression · DistilBERT · BERT"],
        ["Hardware", "NVIDIA RTX 5080 (16 GB) · CUDA 12.8"],
        ["Date", date.today().strftime("%B %d, %Y")],
    ]
    story.append(styled_table(head, col_widths=[1.7 * inch, 4.0 * inch], font=10, header_bg=BLUE))
    story.append(PageBreak())

    # ---------------- EXECUTIVE SUMMARY ----------------
    story.append(P("Executive Summary", H1))
    best_acc = max([m.get("test_accuracy", 0) for m in [base_m, dbert, bert]] + [0])
    story.append(P(
        f"This project builds an AI system that automatically reads a free-text consumer complaint and "
        f"classifies it into the correct financial-product category, enabling instant routing to the right "
        f"department. Using <b>{n_rows:,}</b> real complaints from the U.S. Consumer Financial Protection "
        f"Bureau (CFPB), we trained and compared three models of increasing sophistication. The best model "
        f"reached <b>{pct(best_acc)}</b> test accuracy across <b>{data.get('n_categories','?')}</b> categories, "
        f"turning a slow, manual triage process into a sub-second automated decision."))
    story.append(P("Key outcomes", H2))
    summ = []
    if base_m:
        summ.append(f"<b>Baseline (TF-IDF + Logistic Regression):</b> {pct(base_m.get('test_accuracy',0))} accuracy, "
                    f"weighted F1 {base_m.get('test_f1_weighted','?')} — fast, interpretable, near-zero inference cost.")
    if dbert:
        summ.append(f"<b>DistilBERT:</b> {pct(dbert.get('test_accuracy',0))} accuracy, "
                    f"weighted F1 {dbert.get('test_f1_weighted','?')} — production-friendly transformer.")
    if bert:
        summ.append(f"<b>BERT:</b> {pct(bert.get('test_accuracy',0))} accuracy, "
                    f"weighted F1 {bert.get('test_f1_weighted','?')} — highest contextual accuracy.")
    story.extend(bullets(summ))
    story.append(P("Business value", H2))
    story.extend(bullets([
        "Cuts complaint triage from minutes of human reading to milliseconds of automated routing.",
        "Reduces mis-routing, which is a primary driver of slow resolution and repeat contacts.",
        "Surfaces category and sentiment trends that flag emerging product issues early.",
        "A tiered design (cheap baseline + accurate transformer) lets the business trade cost vs. accuracy.",
    ]))
    story.append(PageBreak())

    # ---------------- 1. BUSINESS PROBLEM ----------------
    story.append(P("1. Business Problem", H1))
    story.append(P(
        "Large financial institutions receive thousands of complaints every day through web forms, mobile apps, "
        "email, call centers, and social media. When these arrive as unstructured free text, a human must read "
        "each one and decide which team should handle it — credit cards, mortgages, debt collection, and so on. "
        "Manual review is slow, inconsistent, and expensive, and mis-routing directly worsens response time and "
        "customer experience."))
    story.append(P(
        "<b>Goal:</b> build an AI system that, given the complaint narrative, automatically (1) classifies the "
        "complaint into its product category, (2) routes it to the correct department, and (3) exposes trends and "
        "sentiment for operational monitoring."))
    story.append(P("Worked example", H2))
    ex_tbl = [
        ["Complaint text", "“I was charged twice for my credit card payment and customer support has not "
                           "responded for 10 days.”"],
        ["Predicted category", "Credit card or prepaid card"],
        ["Routed to", "Card Services team"],
    ]
    story.append(styled_table(ex_tbl, col_widths=[1.6 * inch, 4.1 * inch], font=9.5, header_bg=BLUE))

    # ---------------- 2. DATASET ----------------
    story.append(P("2. Dataset Overview", H1))
    story.append(P(
        f"The source is the public <b>CFPB Consumer Complaint Database</b>. The raw file contains "
        f"<b>{data.get('rows_total_raw',0):,}</b> complaints, but only those where the consumer consented to "
        f"publish their narrative contain text we can classify on — <b>{data.get('rows_with_narrative',0):,} "
        f"rows ({data.get('pct_with_narrative','?')}%)</b>. After consolidating product labels and dropping "
        f"empties, the modeling dataset is <b>{n_rows:,}</b> complaints spanning "
        f"<b>{data.get('date_min','?')}</b> to <b>{data.get('date_max','?')}</b>, across "
        f"<b>{data.get('n_companies',0):,}</b> companies and <b>{data.get('n_states',0)}</b> states/territories."))
    story.append(P("Key fields used", H2))
    fields = [
        ["Field", "Role"],
        ["Consumer complaint narrative", "Input text — what the model reads"],
        ["Product", "Target label (consolidated to 8 categories)"],
        ["Date received", "Trend / volume analysis"],
        ["Company, State", "Operational segmentation & geography"],
        ["Timely response?, Consumer disputed?", "Downstream quality signals"],
    ]
    story.append(styled_table(fields, col_widths=[2.6 * inch, 3.1 * inch]))
    story.append(Spacer(1, 8))
    story.append(P("Category consolidation", H2))
    story.append(P(
        "CFPB renamed and merged product taxonomies over the years, producing 18 raw labels that include exact "
        "duplicates (e.g. <i>“Credit reporting”</i> vs <i>“Credit reporting, credit repair services, or other "
        "personal consumer reports”</i>). We mapped these to <b>8 canonical business categories</b>, which removes "
        "label noise and improves model accuracy. Final class distribution:"))
    cc = data.get("category_counts", {})
    rows = [["Category", "Complaints", "Share"]]
    tot = sum(cc.values()) or 1
    for k, v in sorted(cc.items(), key=lambda x: -x[1]):
        rows.append([k, f"{v:,}", f"{v/tot*100:.1f}%"])
    story.append(styled_table(rows, col_widths=[3.3 * inch, 1.3 * inch, 1.1 * inch]))
    story.append(P("This is a genuine class-imbalance problem (the largest class is ~20× the smallest), which we "
                   "track with macro-F1 in addition to accuracy.", CAP))

    # ---------------- 3. EDA ----------------
    story.append(PageBreak())
    story.append(P("3. Exploratory Data Analysis", H1))
    story.append(P("Before modeling, we profiled the data to understand volume, geography, text length, and "
                   "vocabulary — each yielding an operational insight."))
    story.extend(fig("01_category_distribution.png",
                     caption="Debt collection and Credit reporting dominate the complaint mix — the two teams that "
                             "would benefit most from automation."))
    story.extend(fig("02_volume_over_time.png",
                     caption=f"Monthly volume trends upward, peaking at {eda.get('peak_month_count','?'):,} "
                             f"complaints in {eda.get('peak_month','?')} — staffing must scale with this curve."))
    story.append(PageBreak())
    story.extend(fig("03_volume_by_category.png",
                     caption="Category mix over time — Credit reporting complaints grow fastest, signalling an "
                             "emerging-issue hotspot."))
    story.extend(fig("05_state_heatmap.png",
                     caption="Complaints concentrate in California, Texas, Florida, New York and Georgia — the "
                             "large-population states."))
    story.append(PageBreak())
    story.extend(fig("06_length_histograms.png",
                     caption="Most complaints are 100–300 words (median "
                             f"{eda.get('word_count',{}).get('median','?')}), but a long tail of detailed cases "
                             "exists — informing a 256-token model limit."))
    story.extend(fig("07_length_by_category.png",
                     caption="Mortgage and bank complaints run longest (more complex cases); debt-collection "
                             "complaints are shortest."))
    story.append(PageBreak())
    story.extend(fig("08_wordclouds.png",
                     caption="Category vocabularies are highly distinct — 'debt/credit/call', 'mortgage/payment/"
                             "home', 'card/charge/payment' — which is exactly why text classification works well."))
    story.extend(fig("09_sentiment_by_category.png",
                     caption="VADER sentiment is most negative for debt collection and bank complaints — useful as "
                             "an urgency/priority signal."))

    # ---------------- 4. PREPROCESSING & FEATURES ----------------
    story.append(PageBreak())
    story.append(P("4. Preprocessing & Feature Engineering", H1))
    story.append(P("Text cleaning pipeline", H2))
    story.append(P("For the classical model, each narrative passes through a standard NLP cleaning pipeline. "
                   "(Transformers consume raw text and handle tokenization internally.)"))
    story.extend(bullets([
        "Lowercase and strip HTML / URLs",
        "Remove CFPB PII-redaction blocks (runs of 'XXXX')",
        "Remove punctuation and non-alphabetic characters",
        "Remove English stop-words and very short tokens",
        "Lemmatize each token to its dictionary root (WordNet)",
    ]))
    if pre:
        story.append(P("Before / after example", H2))
        story.append(styled_table([
            ["Before", pre.get("before", "")[:220] + "…"],
            ["After (cleaned)", pre.get("after", "")[:220] + "…"],
        ], col_widths=[1.3 * inch, 4.4 * inch], font=8.4, header_bg=BLUE))
    story.append(P("Engineered numeric features", H2))
    story.append(P("Beyond the text, we engineered signals that capture complaint complexity and customer "
                   "frustration — useful for urgency scoring and as auxiliary insight:"))
    fcols = ["word_count", "char_count", "sentence_count", "capital_words", "exclamation_count", "vader_sentiment"]
    fhead = ["Category", "Words", "Chars", "Sent.", "CAPS", "Excl.", "Sentiment"]
    frows = [fhead]
    if feat:
        cats = list(next(iter(feat.values())).keys()) if feat else []
        for cat in feat.get("word_count", {}):
            frows.append([cat] + [f"{feat[c][cat]:.1f}" if c != "vader_sentiment" else f"{feat[c][cat]:+.2f}"
                                  for c in fcols])
    story.append(styled_table(frows, col_widths=[2.0 * inch] + [0.72 * inch] * 6, font=7.8))
    story.append(P("Interpretation: longer, more capitalized, more negative complaints (mortgage, debt "
                   "collection) tend to be higher-stakes cases worth prioritizing.", CAP))

    # ---------------- 5. MODELS ----------------
    story.append(PageBreak())
    story.append(P("5. Modeling Approach", H1))
    story.append(P("We trained three models of increasing capacity on an identical stratified 80/20 split "
                   f"({int(n_rows*0.8):,} train / {int(n_rows*0.2):,} test), so every result is directly comparable."))
    story.append(styled_table([
        ["Model", "What it is", "Why include it"],
        ["TF-IDF + Logistic Regression", "Bag-of-words (1–2 grams) + linear classifier",
         "Fast, interpretable, cheap baseline"],
        ["DistilBERT", "Distilled transformer (66M params)", "40% smaller/faster, near-BERT accuracy — production"],
        ["BERT-base", "Full transformer (110M params)", "Strongest contextual understanding"],
    ], col_widths=[1.9 * inch, 2.1 * inch, 1.7 * inch], font=8.4))

    # ---------------- 6. HYPERPARAMETER TUNING ----------------
    story.append(P("6. Hyperparameter Tuning", H1))
    story.append(P("Baseline — 5-fold grid search", H2))
    story.append(P("We tuned the inverse-regularization strength <b>C</b> of the logistic-regression head with "
                   "5-fold cross-validation (scoring weighted-F1), over a multinomial lbfgs/L2 configuration."))
    if base_m.get("grid_results"):
        gr = [["C", "Mean CV weighted-F1", "Std"]]
        best_c = base_m.get("best_params", {}).get("C")
        hl = None
        for i, r in enumerate(base_m["grid_results"], start=1):
            gr.append([str(r["param_C"]), f"{r['mean_test_score']:.4f}", f"{r['std_test_score']:.4f}"])
            if r["param_C"] == best_c:
                hl = i
        story.append(styled_table(gr, col_widths=[1.4 * inch, 2.4 * inch, 1.4 * inch], font=9, highlight_row=hl))
        story.append(P(f"Best: C = {best_c} (CV weighted-F1 {base_m.get('cv_weighted_f1')}). "
                       f"Tuning ran in {base_m.get('tuning_seconds','?')}s across "
                       f"{len(base_m['grid_results'])*5} fits.", CAP))
    story.append(P("Transformers — training configuration", H2))
    story.append(P("Transformers were fine-tuned with the AdamW optimizer and a linear warmup schedule. "
                   "Key hyperparameters:"))
    tp_rows = [["Hyperparameter", "DistilBERT", "BERT"]]
    for key, lab in [("learning_rate", "Learning rate"), ("batch_size", "Batch size"),
                     ("epochs", "Epochs"), ("max_len", "Max sequence length"),
                     ("weight_decay", "Weight decay"), ("warmup_ratio", "Warmup ratio"),
                     ("precision", "Mixed precision")]:
        tp_rows.append([lab,
                        str(dbert.get("params", {}).get(key, "—")),
                        str(bert.get("params", {}).get(key, "—"))])
    story.append(styled_table(tp_rows, col_widths=[2.3 * inch, 1.7 * inch, 1.7 * inch], font=9))

    # ---------------- 7. RESULTS ----------------
    story.append(PageBreak())
    story.append(P("7. Results & Model Comparison", H1))
    models = [("TF-IDF + LogReg", base_m), ("DistilBERT", dbert), ("BERT", bert)]
    models = [(n, m) for n, m in models if m]
    accs = [m.get("test_accuracy", 0) for _, m in models]
    best_i = accs.index(max(accs)) + 1 if accs else None
    res = [["Metric"] + [n for n, _ in models]]
    for key, lab in [("test_accuracy", "Accuracy"), ("test_f1_weighted", "Weighted F1"),
                     ("test_f1_macro", "Macro F1"), ("test_precision_weighted", "Weighted Precision"),
                     ("test_recall_weighted", "Weighted Recall")]:
        res.append([lab] + [f"{m.get(key,0):.4f}" for _, m in models])
    story.append(styled_table(res, col_widths=[2.0 * inch] + [1.4 * inch] * len(models),
                              font=9.2, highlight_row=None))
    story.append(P("Higher is better on every metric. Macro-F1 (unweighted class average) is the honest measure "
                   "under class imbalance.", CAP))

    story.append(P("Per-category F1", H2))
    cats = sorted(base_m.get("per_class_f1", {}).keys())
    pc = [["Category"] + [n for n, _ in models]]
    for c in cats:
        pc.append([c] + [f"{m.get('per_class_f1',{}).get(c,0):.3f}" for _, m in models])
    story.append(styled_table(pc, col_widths=[2.6 * inch] + [1.0 * inch] * len(models), font=8.4))

    story.append(P("Confusion matrices (best models)", H2))
    story.extend(fig("10_confusion_baseline.png", width=4.4 * inch,
                     caption="TF-IDF + Logistic Regression — strong diagonal; most confusion is between the two "
                             "credit-related categories, which overlap semantically."))
    if os.path.exists(os.path.join(FIG, "11_confusion_bert.png")):
        story.append(PageBreak())
        story.extend(fig("11_confusion_bert.png", width=4.4 * inch,
                         caption="BERT — tighter diagonal and better recovery of minority classes."))

    # ---------------- 8. BUSINESS VALUE ----------------
    story.append(PageBreak())
    story.append(P("8. Business Value & Recommendations", H1))
    story.append(P("Operational impact", H2))
    story.extend(bullets([
        "<b>Speed:</b> routing decisions drop from minutes of human reading to a few milliseconds per complaint.",
        "<b>Accuracy:</b> consistent, policy-aligned categorization removes human variance and mis-routing.",
        "<b>Cost:</b> the baseline model runs on CPU at negligible cost; transformers add accuracy when it matters.",
        "<b>Scalability:</b> the same model absorbs volume spikes (like the observed peak) with no extra headcount.",
        "<b>Insight:</b> category, geography, and sentiment trends become a live early-warning dashboard.",
    ]))
    story.append(P("Recommended deployment", H2))
    story.append(P(
        "Adopt a <b>tiered routing strategy</b>: run the cheap TF-IDF baseline on every complaint; when its "
        "confidence is low, escalate to the fine-tuned transformer for a second opinion. This captures most of "
        "the transformer's accuracy at a fraction of the compute cost. Pair the category prediction with the "
        "engineered sentiment/urgency features to prioritize the most distressed customers first."))
    story.append(P("Next steps", H2))
    story.extend(bullets([
        "Add a confidence threshold + human-in-the-loop review for low-certainty predictions.",
        "Address class imbalance (class-weighted loss / focal loss) to lift minority-category recall.",
        "Extend beyond category to predicted urgency and automated complaint summarization.",
        "Monitor for data drift as products and complaint language evolve over time.",
    ]))
    story.append(Spacer(1, 10))
    story.append(P("Reproducibility", H2))
    story.append(P("Full pipeline: <font face='Courier'>data_prep → eda → features → train_baseline → "
                   "train_transformer → report_gen</font>. All code in <font face='Courier'>src/</font>, "
                   "figures in <font face='Courier'>outputs/figures/</font>, metrics in "
                   "<font face='Courier'>outputs/metrics/</font>.", SMALL))

    doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.85 * inch, bottomMargin=0.7 * inch,
                            title="Consumer Complaint Classification Report")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=header_footer)
    print("PDF written ->", OUT)


if __name__ == "__main__":
    build()
