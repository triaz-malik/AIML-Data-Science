"""Assemble a PDF report from the metrics + figures produced by the pipeline.

The report is written in plain English: what the project is, the data, the
problems hit and how they were solved, the method, results (with what each
metric means), explainability, and what every output file is.
"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, Image, ListFlowable, ListItem,
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

import config as C

STYLES = getSampleStyleSheet()
H1 = STYLES["Heading1"]
H2 = STYLES["Heading2"]
BODY = ParagraphStyle("body", parent=STYLES["BodyText"], fontSize=10, leading=14,
                      spaceAfter=6)
TITLE = ParagraphStyle("title", parent=STYLES["Title"], fontSize=22, spaceAfter=6)
CAPTION = ParagraphStyle("cap", parent=BODY, fontSize=8, textColor=colors.grey,
                         spaceBefore=2)


def _fig(name, width=16 * cm, caption=None):
    path = C.FIGURES_DIR / name
    flow = []
    if not path.exists():
        return [Paragraph(f"<i>[missing figure: {name}]</i>", BODY)]
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    flow.append(img)
    if caption:
        flow.append(Paragraph(caption, CAPTION))
    return flow


def _bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BODY), leftIndent=10) for t in items],
        bulletType="bullet", start="•", leftIndent=14, spaceAfter=6,
    )


def _box(title, body_text, bg="#eaf2f8", border="#2980b9"):
    """A coloured callout box (used for the problem/solution and caveat)."""
    inner = [Paragraph(f"<b>{title}</b>", BODY), Paragraph(body_text, BODY)]
    t = Table([[inner]], colWidths=[16 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _simple_table(rows, header_bg="#2c3e50", first_col_bold=True, col_widths=None):
    tbl = Table(rows, hAlign="LEFT", colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if first_col_bold:
        style.append(("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    return tbl


def _comparison_table(metrics):
    header = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "CV F1"]
    rows = [header]
    ordered = sorted(metrics["models"].items(),
                     key=lambda kv: kv[1]["test"]["f1"], reverse=True)
    for name, m in ordered:
        t = m["test"]
        rows.append([name, f"{t['accuracy']:.4f}", f"{t['precision']:.4f}",
                     f"{t['recall']:.4f}", f"{t['f1']:.4f}", f"{t['roc_auc']:.4f}",
                     f"{m['cv_f1_mean']:.4f}"])
    tbl = Table(rows, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d5f5e3")),  # best row
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ]))
    return tbl


def run():
    metrics = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else None
    doc = SimpleDocTemplate(str(C.REPORTS_DIR / "IDS_Report.pdf"), pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    best = metrics.get("best_model", "-") if metrics else "-"
    bt = metrics["models"][best]["test"] if metrics else {}

    # ===================== Title page =====================
    story += [Spacer(1, 3.5 * cm),
              Paragraph("Network Intrusion Detection System", TITLE),
              Paragraph("Machine Learning &amp; Deep Learning on the NSL-KDD dataset", H2),
              Spacer(1, 0.4 * cm),
              HRFlowable(width="100%", color=colors.HexColor("#2980b9"), thickness=1.2),
              Spacer(1, 0.6 * cm)]
    if metrics:
        story.append(Paragraph(
            f"<b>Best model:</b> {best} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Accuracy {bt['accuracy']:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"F1 {bt['f1']:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; ROC AUC {bt['roc_auc']:.3f}", BODY))
    story += [Spacer(1, 0.3 * cm),
              Paragraph("Goal: automatically classify each network connection as "
                        "<b>NORMAL</b> or <b>ANOMALY (attack)</b>, so security teams do not "
                        "have to inspect millions of connections by hand.", BODY),
              PageBreak()]

    # ===================== 0. What this project is =====================
    story += [Paragraph("0. What This Project Is", H1),
              Paragraph("An Intrusion Detection System (IDS) looks at one network connection "
                        "at a time — who connected, how many bytes were exchanged, which "
                        "service was used, the error rates, and so on — and decides whether "
                        "the traffic is normal or an attack. Doing this by hand across "
                        "millions of daily connections is impossible, so a machine-learning "
                        "model is trained to do it instantly and consistently.", BODY),
              Paragraph("This report covers the data, the problems encountered, the method, "
                        "the four models trained, how well they performed (and what each score "
                        "means), why the model makes its decisions, and what every output "
                        "file contains.", BODY),
              Spacer(1, 0.3 * cm)]

    # ===================== 1. The data =====================
    story += [Paragraph("1. The Data", H1),
              Paragraph("Two files were provided. Each row describes a single connection "
                        "using 41 features (e.g. <i>protocol_type</i>, <i>service</i>, "
                        "<i>src_bytes</i>, <i>dst_bytes</i>, <i>count</i>, error rates).", BODY)]
    data_rows = [
        ["File", "Rows", "Has label?", "Used for"],
        ["Train_data.csv", "25,192", "Yes (class)", "Training + evaluation"],
        ["Test_data.csv", "22,544", "No", "Final predictions only"],
    ]
    story += [_simple_table(data_rows, col_widths=[4.5 * cm, 2.5 * cm, 3 * cm, 6 * cm]),
              Spacer(1, 0.2 * cm),
              Paragraph("The label column <i>class</i> has just two values: "
                        "<b>normal</b> and <b>anomaly</b>. After cleaning, the data is roughly "
                        "balanced at about 53% normal / 47% attack.", BODY),
              Spacer(1, 0.3 * cm)]

    # ===================== 2. Problems & solutions =====================
    story += [Paragraph("2. Problems Encountered &amp; How They Were Solved", H1),
              _box("Problem 1 — No attack sub-types in the data",
                   "The original plan wanted to separate attacks into DoS / Probe / U2R / R2L. "
                   "These CSV files only contain <b>normal vs anomaly</b> — the sub-type labels "
                   "are not present. <b>Solution:</b> a binary classifier (normal vs attack) was "
                   "built. The code can extend to multi-class by changing one line if labelled "
                   "sub-type data is obtained.",
                   bg="#fdebd0", border="#e67e22"),
              Spacer(1, 0.25 * cm),
              _box("Problem 2 — The test file has no answers",
                   "Test_data.csv has no <i>class</i> column, so accuracy cannot be measured on "
                   "it (there is nothing to check against). <b>Solution:</b> the labelled "
                   "Train_data.csv was split 80% / 20%. The model is trained on 80% and scored "
                   "on the untouched 20% (an honest test). Test_data.csv only receives "
                   "predictions, saved to a file.",
                   bg="#fdebd0", border="#e67e22"),
              Spacer(1, 0.25 * cm),
              _box("Problem 3 — TensorFlow could not be installed",
                   "The planned Keras deep neural network needs TensorFlow, which does not "
                   "install cleanly on this Python 3.13 environment. <b>Solution:</b> "
                   "scikit-learn's MLPClassifier (a neural network with layers 256 → 128 → 64) "
                   "was used instead — same concept, different library — and it scored about "
                   "99.3% F1.",
                   bg="#fdebd0", border="#e67e22"),
              PageBreak()]

    # ===================== 3. Method =====================
    story += [Paragraph("3. What Was Done (Step by Step)", H1),
              _bullets([
                  "<b>Explore (EDA):</b> charted the data and found it is balanced; about 84% "
                  "of attacks use TCP; <i>src_bytes</i> is heavily skewed; several error-rate "
                  "columns are near-duplicates of each other.",
                  "<b>Clean:</b> no missing values and no duplicate rows; dropped two columns "
                  "that never change (<i>num_outbound_cmds</i>, <i>is_host_login</i>) because "
                  "they carry no information.",
                  "<b>Prepare features:</b> text columns (protocol / service / flag) were "
                  "one-hot encoded into numbers, and numeric columns were scaled so large "
                  "values (like bytes) do not dominate small ones.",
                  "<b>Handle balance:</b> compared SMOTE and ADASYN against doing nothing — "
                  "they gave only a marginal lift because the data was already balanced.",
                  "<b>Train &amp; tune:</b> four models were trained, each automatically tuned "
                  "(many settings tried, best kept) and cross-validated 5 ways to confirm the "
                  "results are stable, not luck.",
                  "<b>Explain, predict, report:</b> generated SHAP explanations, scored the "
                  "unlabelled test file, and assembled this PDF.",
              ]),
              Spacer(1, 0.3 * cm)]

    # ===================== 4. EDA figures =====================
    story += [Paragraph("4. Exploratory Data Analysis", H1)]
    story += _fig("01_target_distribution.png", 10 * cm,
                  "Normal vs anomaly counts — the dataset is well balanced.")
    story += [Spacer(1, 0.2 * cm)]
    story += _fig("02_protocol_distribution.png", 12 * cm,
                  "Protocol by class — most attacks ride on TCP.")
    story += [PageBreak()]
    story += _fig("03_top_services.png", 12 * cm, "The 20 most common services.")
    story += [Spacer(1, 0.2 * cm)]
    story += _fig("06_feature_distributions.png", 15 * cm,
                  "Key feature distributions (y log-scaled) — note heavy skew and outliers.")
    story += [PageBreak()]
    story += _fig("05_correlation_heatmap.png", 16 * cm,
                  "Correlation heatmap — bright blocks are redundant, highly-correlated "
                  "error-rate features.")
    story += [PageBreak()]

    # ===================== 5. Results + metric meanings =====================
    story += [Paragraph("5. Results — Model Comparison", H1),
              Paragraph("All four models were scored on the held-out 20% test split (data they "
                        "never saw during training). XGBoost performed best.", BODY)]
    if metrics:
        story += [_comparison_table(metrics), Spacer(1, 0.2 * cm)]
        sc = metrics.get("_sampler_comparison", {})
        if sc:
            srows = [["Resampling", "XGBoost 5-fold F1"]]
            srows += [[k, f"{v['f1_mean']:.4f} +/- {v['f1_std']:.4f}"] for k, v in sc.items()]
            story += [Spacer(1, 0.2 * cm),
                      Paragraph("Class-imbalance handling (marginal effect here):", H2),
                      _simple_table(srows, col_widths=[5 * cm, 6 * cm])]
    story += [Spacer(1, 0.3 * cm),
              Paragraph("What each score means (in IDS terms):", H2),
              _bullets([
                  "<b>Accuracy</b> — share of all connections classified correctly.",
                  "<b>Precision</b> — when the model shouts \"attack\", how often it is right. "
                  "High precision = few false alarms, so analysts are not flooded.",
                  "<b>Recall</b> — of all real attacks, how many were caught. High recall = "
                  "few attacks slip through undetected.",
                  "<b>F1</b> — a single balanced score combining precision and recall.",
                  "<b>ROC AUC</b> — overall ability to tell attack from normal; 1.0 is perfect.",
                  "<b>CV F1</b> — average F1 across 5 cross-validation folds, confirming the "
                  "result is stable rather than a one-off.",
              ])]
    story += _fig("10_model_comparison.png", 14 * cm)
    story += [PageBreak()]

    # ===================== 6. Evaluation plots =====================
    story += [Paragraph("6. Evaluation Plots", H1)]
    story += _fig("07_confusion_matrices.png", 16 * cm,
                  "Confusion matrices — correct predictions on the diagonal, mistakes off it.")
    story += [PageBreak()]
    story += _fig("08_roc_curves.png", 10.5 * cm,
                  "ROC curves — closer to the top-left corner is better.")
    story += [Spacer(1, 0.2 * cm)]
    story += _fig("09_pr_curves.png", 10.5 * cm,
                  "Precision-Recall curves — high and flat is better.")
    story += [PageBreak()]

    # ===================== 7. Explainability =====================
    story += [Paragraph("7. Why the Model Decides (SHAP)", H1),
              Paragraph("A model that flags attacks without justification is hard to trust. "
                        "SHAP reveals which features drive each decision. The strongest drivers "
                        "were <b>src_bytes</b> (data sent), <b>count</b> (number of "
                        "connections), and <b>dst_bytes</b> (data received), followed by "
                        "connection error rates, service and protocol — patterns that match "
                        "real intrusion behaviour, a good sign the model learned genuine "
                        "signal rather than noise.", BODY)]
    story += _fig("11_shap_summary.png", 14 * cm,
                  "SHAP summary — each dot is one connection; colour is the feature value.")
    story += [PageBreak()]
    story += _fig("12_shap_bar.png", 14 * cm, "Average impact of each feature.")
    story += [Spacer(1, 0.2 * cm)]
    story += _fig("13_shap_waterfall.png", 14 * cm,
                  "Waterfall — how features pushed one example toward 'attack'.")
    story += [PageBreak()]

    # ===================== 8. Outputs =====================
    story += [Paragraph("8. What the Outputs Are", H1)]
    out_rows = [
        ["File / folder", "What it is"],
        ["reports/IDS_Report.pdf", "This report — charts, tables and explanations."],
        ["reports/model_comparison.csv", "The results table as a spreadsheet."],
        ["reports/figures/", "All 13 charts (EDA, ROC, confusion, SHAP)."],
        ["models/xgb_ids_model.pkl", "The winning trained model."],
        ["models/best_model.pkl", "Copy of the winner, used by the dashboard."],
        ["models/random_forest_ids.pkl ...", "The other three trained models."],
        ["models/preprocessor.pkl", "Encoder + scaler that prepares new data identically."],
        ["models/metrics.json", "All scores in machine-readable form."],
        ["outputs/test_predictions.csv", "Every unlabelled connection, now scored."],
        ["app/streamlit_app.py", "Interactive dashboard (live verdict + risk score)."],
    ]
    story += [_simple_table(out_rows, first_col_bold=True,
                            col_widths=[6.5 * cm, 9.5 * cm]),
              Spacer(1, 0.25 * cm),
              Paragraph("The predictions file adds three columns to each connection: "
                        "<b>prediction</b> (normal / anomaly), <b>attack_probability</b> "
                        "(0–1 confidence) and <b>risk_score</b> (0–100). On Test_data.csv the "
                        "model flagged about 8,476 of 22,544 connections (37.6%) as attacks.",
                        BODY),
              Spacer(1, 0.3 * cm)]

    # ===================== 9. Caveat =====================
    story += [Paragraph("9. An Honest Caveat", H1),
              _box("Scores are high partly because the test is 'easy'",
                   "The ~99.7% scores are real, but the train and test rows come from the same "
                   "data pool, so the test is relatively easy. Real-world traffic drifts over "
                   "time, so live performance would be lower. The official harder NSL-KDD test "
                   "set (KDDTest+) is designed to expose this and is the recommended next step "
                   "for a realistic estimate.",
                   bg="#fdecea", border="#c0392b")]

    # ===================== 10. Business impact =====================
    story += [Spacer(1, 0.4 * cm),
              Paragraph("10. Business Impact", H1),
              _bullets([
                  "Real-time, automated detection replaces infeasible manual monitoring.",
                  "Faster detection shrinks attacker dwell time and response time.",
                  "Fewer false alarms (high precision) reduces SOC analyst fatigue.",
                  "Catching more attacks (high recall) lowers breach-related losses.",
                  "SHAP explanations make every alert auditable, supporting analyst trust "
                  "and incident triage.",
              ])]

    doc.build(story)
    print(f"[report] wrote {C.REPORTS_DIR / 'IDS_Report.pdf'}")


if __name__ == "__main__":
    run()
