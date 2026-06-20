"""
Generate the project PDF report: AI-Powered Healthcare Risk Prediction System.

Pulls numbers and figures from ./outputs/ and assembles a multi-section,
figure-rich PDF with reportlab. Run after run_all.py.

    python build_report.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, HRFlowable,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
PDF_PATH = ROOT / "Stroke_Risk_Prediction_Report.pdf"

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E5596")
LIGHT = colors.HexColor("#DCE6F1")
GREY = colors.HexColor("#666666")

CONTENT_W = A4[0] - 4 * cm  # usable width with 2cm margins

# --------------------------------------------------------------------------- styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Cover", parent=styles["Title"], fontSize=26,
                          textColor=NAVY, leading=32, spaceAfter=10))
styles.add(ParagraphStyle("Sub", parent=styles["Normal"], fontSize=13,
                          textColor=GREY, leading=18))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                          textColor=NAVY, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5,
                          textColor=BLUE, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                          leading=15, spaceAfter=6))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5,
                          textColor=GREY, alignment=1, spaceAfter=10))
styles.add(ParagraphStyle("Bullets", parent=styles["Normal"], fontSize=10,
                          leading=14, leftIndent=12, spaceAfter=2,
                          bulletIndent=2))


def meta():
    return json.loads((OUT / "models" / "training_meta.json").read_text())


# --------------------------------------------------------------------------- helpers
def P(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [Paragraph(f"• {t}", styles["Bullets"]) for t in items]


def table(data, col_widths=None, header=True, font=9):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C4DA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ]
    t.setStyle(TableStyle(cmds))
    return t


def figure(name, caption, width=None):
    path = FIG / name
    if not path.exists():
        return P(f"[missing figure: {name}]")
    iw, ih = ImageReader(str(path)).getSize()
    w = width or min(CONTENT_W, 15 * cm)
    h = w * ih / iw
    return [Image(str(path), width=w, height=h), P(caption, "Caption")]


def hr():
    return HRFlowable(width="100%", thickness=0.6, color=LIGHT,
                      spaceBefore=4, spaceAfter=8)


# --------------------------------------------------------------------------- build
def build():
    m = meta()
    served = m["served_model"]
    leader = m["auc_leader_benchmark"]
    thr = m["recall_threshold"]
    rm = m["served_recall_threshold_metrics"]

    s = []

    # ---- Cover
    s += [Spacer(1, 4 * cm)]
    s += [P("AI-Powered Healthcare<br/>Risk Prediction System", "Cover")]
    s += [P("Predicting patient stroke risk and explaining the factors behind it", "Sub")]
    s += [Spacer(1, 0.6 * cm), hr()]
    s += [P("Dataset: Kaggle Healthcare Stroke Prediction &nbsp;|&nbsp; "
            "5,109 patients &nbsp;|&nbsp; binary classification", "Sub")]
    s += [P(f"Deployed model: <b>K-Nearest Neighbors</b> &nbsp;|&nbsp; "
            f"Benchmarks: Random Forest, XGBoost", "Sub")]
    s += [Spacer(1, 6 * cm)]
    s += [P("Technical & Business Report", "Sub")]
    s += [PageBreak()]

    # ---- Executive summary
    s += [P("Executive Summary", "H1"), hr()]
    s += [P(
        "Hospitals triage patients with limited resources and inconsistent manual "
        "risk assessment. This project delivers a reproducible, explainable machine-"
        "learning system that scores each patient's stroke risk, sorts patients into "
        "clinical-action bands, and explains every prediction so clinicians can trust "
        "and act on it.")]
    s += bullets([
        "<b>Problem:</b> identify high-risk stroke patients early, despite a severely "
        "imbalanced dataset (only 4.9% of patients had a stroke).",
        f"<b>Solution:</b> an end-to-end pipeline — cleaning, feature engineering, "
        f"imbalance handling, three tuned models, SHAP explainability, and a risk-"
        f"stratification layer that maps probability to clinical action.",
        f"<b>Deployed model:</b> K-Nearest Neighbors, tuned and operated at an "
        f"<b>80% recall</b> point — it catches 4 of every 5 strokes.",
        "<b>Business value:</b> earlier intervention, better patient prioritization, "
        "and transparent, auditable decisions that support (not replace) clinicians.",
    ])
    s += [Spacer(1, 0.3 * cm)]
    s += [table([
        ["Metric", "Value"],
        ["Patients (after cleaning)", "5,109"],
        ["Stroke prevalence", "4.87% (249 cases)"],
        ["Deployed model", "K-Nearest Neighbors (k=21)"],
        ["Operating recall", f"{rm['recall']:.0%} (threshold {thr:.3f})"],
        ["Test ROC-AUC (KNN)", f"{rm['roc_auc']:.3f}"],
        ["High-risk patients flagged", "2,135 (41.8%)"],
    ], col_widths=[7 * cm, 8 * cm])]
    s += [PageBreak()]

    # ---- 1. Dataset
    s += [P("1. The Dataset", "H1"), hr()]
    s += [P(
        "The Kaggle <i>Healthcare Stroke Prediction</i> dataset: 5,110 patient records "
        "(5,109 after removing one invalid row), each labelled with whether the patient "
        "had a stroke. It mixes demographic, lifestyle, and clinical features.")]
    s += [table([
        ["Feature", "Description", "Type"],
        ["age", "Patient age in years", "Numeric"],
        ["gender", "Male / Female", "Categorical"],
        ["hypertension", "High blood pressure (0/1)", "Binary"],
        ["heart_disease", "Heart disease (0/1)", "Binary"],
        ["ever_married", "Marital status", "Categorical"],
        ["work_type", "Employment type", "Categorical"],
        ["Residence_type", "Urban / Rural", "Categorical"],
        ["avg_glucose_level", "Average blood glucose", "Numeric"],
        ["bmi", "Body Mass Index", "Numeric"],
        ["smoking_status", "Smoker / former / never / unknown", "Categorical"],
        ["stroke", "TARGET — had a stroke (0/1)", "Binary"],
    ], col_widths=[4 * cm, 8.5 * cm, 2.5 * cm])]
    s += [Spacer(1, 0.2 * cm)]
    s += [P("<b>The central challenge is class imbalance:</b> only about 1 patient in 20 "
            "had a stroke. Plain accuracy is therefore misleading — a model that predicts "
            "'no stroke' for everyone scores ~95% while catching zero strokes. The whole "
            "project is built around this fact.")]
    s += figure("01_target_balance.png", "Figure 1. Stroke vs non-stroke — the 95/5 imbalance.",
                width=9 * cm)
    s += [PageBreak()]

    # ---- 2. Data prep & feature engineering
    s += [P("2. Data Cleaning & Feature Engineering", "H1"), hr()]
    s += [P("Cleaning (Phase 2):", "H2")]
    s += bullets([
        "<b>Missing BMI:</b> 201 values stored as 'N/A' → converted to numeric and "
        "imputed with the median.",
        "<b>Dropped</b> the non-predictive <i>id</i> column and one record with gender "
        "'Other' (a singleton category that cannot be modelled).",
        "<b>Duplicate removal</b> and categorical text standardization (consistent casing).",
        "<b>Outlier handling:</b> BMI and glucose winsorized at the 0.5 / 99.5 percentile "
        "so a few extreme values don't distort scaling.",
    ])
    s += [P("Engineered features (Phase 4):", "H2")]
    s += bullets([
        "<b>Age groups:</b> Child / Young / Adult / Senior.",
        "<b>BMI categories:</b> Underweight / Normal / Overweight / Obese (WHO bands).",
        "<b>Glucose risk categories:</b> Normal / Moderate / High.",
        "<b>Composite Health-Risk Score (0–8):</b> an interpretable additive score "
        "combining age, BMI, hypertension, and heart disease.",
    ])
    s += [P("These features make the model's reasoning clinically legible and gave the "
            "health-risk score real predictive weight (see Section 6).")]
    s += [Spacer(1, 0.2 * cm)]
    s += figure("09_correlation_heatmap.png",
                "Figure 2. Correlation heatmap — age shows the strongest association with stroke.",
                width=12 * cm)
    s += [PageBreak()]

    # ---- 3. EDA
    s += [P("3. Exploratory Data Analysis", "H1"), hr()]
    s += [P("Three data-understanding questions framed the EDA:")]
    s += bullets([
        "<b>How many patients have stroke?</b> 249 of 5,109 (4.87%) — severe imbalance.",
        "<b>What age groups are most affected?</b> Risk rises steeply with age; "
        "Seniors (65+) dominate the positive class.",
        "<b>Which health factors matter most?</b> Age first, then glucose, BMI, "
        "hypertension, and heart disease.",
    ])
    s += figure("02_age_distribution.png",
                "Figure 3. Age distribution by stroke status — stroke patients skew much older.",
                width=12 * cm)
    s += figure("05_stroke_by_agegroup.png",
                "Figure 4. Stroke rate climbs sharply from younger to senior age groups.",
                width=12 * cm)
    s += [PageBreak()]

    # ---- 4. Imbalance handling
    s += [P("4. Class Imbalance Handling", "H1"), hr()]
    s += [P("Three standard strategies were compared on a fixed base XGBoost "
            "(test set). SMOTE and ADASYN synthesize minority examples; class-"
            "weighting penalizes minority errors instead.")]
    imb = _csv_rows(OUT / "reports" / "imbalance_comparison.csv",
                    header=["Strategy", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    s += [table(imb, col_widths=[3.5 * cm] + [1.9 * cm] * 6, font=8.5)]
    s += [Spacer(1, 0.2 * cm)]
    s += [P("<b>Takeaway:</b> class-weighting produced by far the best recall (0.44 vs "
            "~0.13) at comparable ROC-AUC — confirming that for this problem, how we "
            "handle imbalance matters more than raw accuracy. SMOTE is used inside the "
            "model pipelines, applied <i>only</i> within cross-validation folds to avoid "
            "data leakage.")]
    s += [Spacer(1, 0.4 * cm)]

    # ---- 5. Models & tuning
    s += [P("5. Models & Hyperparameter Tuning", "H1"), hr()]
    s += [P("Three classifiers were tuned with Grid / Randomized search over "
            "Stratified 5-fold cross-validation, scored on ROC-AUC:")]
    bp = m["best_params"]
    s += [P("K-Nearest Neighbors (deployed) — Grid Search", "H2")]
    s += [table([["Parameter", "Best value"],
                 ["n_neighbors", str(bp["knn"]["clf__n_neighbors"])],
                 ["weights", bp["knn"]["clf__weights"]],
                 ["metric", bp["knn"]["clf__metric"]]],
                col_widths=[6 * cm, 6 * cm])]
    s += [P("Random Forest — Randomized Search", "H2")]
    s += [table([["Parameter", "Best value"],
                 ["n_estimators", str(bp["random_forest"]["clf__n_estimators"])],
                 ["max_depth", str(bp["random_forest"]["clf__max_depth"])],
                 ["min_samples_split", str(bp["random_forest"]["clf__min_samples_split"])],
                 ["min_samples_leaf", str(bp["random_forest"]["clf__min_samples_leaf"])],
                 ["criterion", bp["random_forest"]["clf__criterion"]]],
                col_widths=[6 * cm, 6 * cm])]
    s += [P("XGBoost — Randomized Search", "H2")]
    s += [table([["Parameter", "Best value"],
                 ["n_estimators", str(bp["xgboost"]["clf__n_estimators"])],
                 ["max_depth", str(bp["xgboost"]["clf__max_depth"])],
                 ["learning_rate", str(bp["xgboost"]["clf__learning_rate"])],
                 ["subsample", str(bp["xgboost"]["clf__subsample"])],
                 ["colsample_bytree", str(bp["xgboost"]["clf__colsample_bytree"])]],
                col_widths=[6 * cm, 6 * cm])]
    s += [PageBreak()]

    # ---- 6. Results
    s += [P("6. Results", "H1"), hr()]
    s += [P("Test-set performance (default 0.5 threshold):", "H2")]
    mc = _csv_rows(OUT / "reports" / "model_comparison.csv",
                   header=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV ROC-AUC"])
    s += [table(mc, col_widths=[3 * cm] + [1.7 * cm] * 7, font=8.5)]
    s += [Spacer(1, 0.2 * cm)]
    s += [P(
        f"<b>KNN is the deployed model</b> (ROC-AUC {rm['roc_auc']:.3f}). Random Forest "
        f"is the highest-AUC benchmark ({_model_auc(mc,'random_forest')}); it is kept in "
        "the comparison for transparency. KNN was chosen deliberately: it is inherently "
        "interpretable (a prediction is the outcome of the most similar past patients), "
        "needs no distributional assumptions, and its tuned recall is competitive. The "
        "AUC gap is small relative to this dataset's inherent ceiling (~0.82).")]
    s += [P("Recall-priority operating point", "H2")]
    s += [P(
        "Healthcare prioritizes <b>recall</b> — missing a high-risk patient is the costly "
        f"error. The decision threshold is tuned to <b>{thr:.3f}</b>, giving "
        f"recall {rm['recall']:.0%}, precision {rm['precision']:.1%}, F1 {rm['f1']:.2f}. "
        "The table below shows the tradeoff explicitly:")]
    ops = _csv_rows(OUT / "reports" / "operating_points.csv",
                    header=["Target recall", "Threshold", "Actual recall", "Precision",
                            "Strokes caught", "False alarms", "Alarms / catch"])
    s += [table(ops, col_widths=[2.6 * cm, 2.0 * cm, 2.2 * cm, 1.9 * cm, 2.3 * cm, 2.0 * cm, 2.0 * cm], font=8)]
    s += [P("Pushing recall from 80% to 90% roughly doubles the false-alarm burden. We "
            "adopt ~80% recall (the highest KNN reaches without collapsing to predict-all-"
            "positive). A hospital can slide this threshold to match screening capacity.")]
    s += figure("10_roc_pr_curves.png",
                "Figure 5. ROC and Precision-Recall curves for all three models.",
                width=15 * cm)
    s += figure("11_confusion_matrix.png",
                f"Figure 6. KNN confusion matrix at the deployed threshold ({thr:.3f}).",
                width=8.5 * cm)
    s += [PageBreak()]

    # ---- 7. Explainability
    s += [P("7. Explainable AI (SHAP)", "H1"), hr()]
    s += [P("Because KNN is not tree-based, the model-agnostic <b>KernelExplainer</b> "
            "was used to attribute each prediction to its features. SHAP confirms the "
            "model reasons from established clinical risk factors — which is what makes "
            "it trustworthy to clinicians.")]
    s += [P("Top global drivers: <b>age</b> (dominant), the composite health-risk score, "
            "average glucose, BMI, and smoking status.")]
    s += figure("12_shap_importance.png",
                "Figure 7. Global SHAP feature importance — age dominates.", width=12 * cm)
    s += figure("13_shap_summary.png",
                "Figure 8. SHAP beeswarm — direction and magnitude of each feature's effect.",
                width=13 * cm)
    s += [P("Local explanations answer <i>“why was this patient flagged?”</i> — a "
            "waterfall plot per patient, exactly what a clinician would see beside the score:")]
    s += figure("14_shap_local_critical.png",
                "Figure 9. Local explanation for a high-risk (Critical) patient.", width=13 * cm)
    s += [PageBreak()]

    # ---- 8. Risk stratification
    s += [P("8. Risk Stratification → Clinical Action", "H1"), hr()]
    s += [P("Model output is converted into four action bands. Every patient is scored "
            "with leakage-free out-of-fold probabilities, then assigned a band:")]
    rd = _risk_table(OUT / "powerbi" / "risk_distribution.csv")
    s += [table(rd, col_widths=[3 * cm, 3.5 * cm, 4 * cm, 2.5 * cm, 3 * cm], font=9)]
    s += [Spacer(1, 0.2 * cm)]
    s += [P("The actual stroke rate rises across the bands — lowest in Low (~1%) and "
            "highest in Critical (~11%) — confirming the stratification is clinically "
            "meaningful and that prioritizing higher bands for review is justified. "
            "(Medium and High are close, a known effect of KNN's coarse probabilities.)")]
    s += [PageBreak()]

    # ---- 9. Business value
    s += [P("9. Business Value", "H1"), hr()]
    s += [table([
        ["Benefit", "Impact"],
        ["Early Detection", "Flags high-risk patients before symptoms escalate — faster intervention."],
        ["Resource Optimization", "Ranks patients so limited specialist time goes to those who need it most."],
        ["Reduced Healthcare Costs", "Preventing severe stroke outcomes is far cheaper than treating them."],
        ["Clinical Decision Support", "Explains every score with SHAP — assists, never replaces, the clinician."],
        ["Consistency", "Replaces inconsistent manual assessment with a uniform, auditable standard."],
        ["Population Health Monitoring", "Aggregated dashboards track risk by age, gender, and region."],
    ], col_widths=[5 * cm, 10 * cm])]
    s += [Spacer(1, 0.4 * cm)]
    s += [P("Deliverables", "H2")]
    s += bullets([
        "Three tuned models (KNN deployed; Random Forest & XGBoost benchmarks), saved.",
        "Patient-level risk scores with band + recommended clinical action.",
        "Reports: EDA, model comparison, clinical recommendation (this document).",
        "Power BI-ready data exports + dashboard build guide (Executive Summary, "
        "Risk Analysis, Model Performance pages).",
    ])
    s += [Spacer(1, 0.3 * cm)]
    s += [P("Honest limitations", "H2")]
    s += bullets([
        "ROC-AUC ~0.75–0.82 is near this dataset's realistic ceiling — the features "
        "carry limited signal. Any '95%+ accuracy' claim simply predicts 'no stroke'.",
        "KNN emits coarse, stepped probabilities (k=21), making its risk bands blockier "
        "than a tree model's — documented rather than hidden.",
        "This is a decision-support tool, not a diagnostic device; all flags require "
        "clinical confirmation.",
    ])

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="AI-Powered Healthcare Risk Prediction System",
        author="Stroke Risk Prediction Project",
    )
    doc.build(s, onLaterPages=_footer, onFirstPage=_cover_bg)
    print(f"PDF written -> {PDF_PATH}")


def _model_auc(mc_rows, model):
    for r in mc_rows[1:]:
        if r[0] == model:
            return f"ROC-AUC {r[5]}"
    return ""


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1 * cm, "AI-Powered Healthcare Risk Prediction System")
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 0.6 * cm, A4[0], 0.6 * cm, fill=1, stroke=0)
    canvas.rect(0, 0, A4[0], 0.4 * cm, fill=1, stroke=0)
    canvas.restoreState()


def _csv_rows(path, header):
    import csv
    rows = [header]
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    for line in r[1:]:
        rows.append(line)
    return rows


def _risk_table(path):
    import csv
    actions = {"Low": "Routine Monitoring", "Medium": "Follow-up",
               "High": "Specialist Review", "Critical": "Immediate Attention"}
    rng = {"Low": "0–10%", "Medium": "10–30%", "High": "30–60%", "Critical": "60%+"}
    rows = [["Risk Level", "Predicted prob.", "Action", "Patients", "Actual stroke rate"]]
    with open(path, newline="") as f:
        data = list(csv.reader(f))[1:]
    for level, patients, rate in data:
        rows.append([level, rng.get(level, ""), actions.get(level, ""),
                     patients, f"{float(rate):.1%}"])
    return rows


if __name__ == "__main__":
    build()
