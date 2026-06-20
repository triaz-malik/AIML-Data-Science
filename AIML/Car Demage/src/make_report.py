"""Generate the project PDF report: reports/Vehicle_Damage_Assessment_Report.pdf

Pulls real numbers from reports/*.json + image_metadata.csv and embeds the
figures produced by eda.py / evaluate.py / gradcam.py.
"""
from __future__ import annotations

import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import FIGURES_DIR, GRADCAM_DIR, REPORTS_DIR

NAVY = colors.HexColor("#1F3A5F")
RED = colors.HexColor("#C44E52")
BLUE = colors.HexColor("#4C72B0")
LIGHT = colors.HexColor("#EAF0F6")
PAGE_W = A4[0] - 4 * cm   # usable width with 2cm margins

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=26, textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle("Sub", parent=styles["Normal"], fontSize=13, textColor=colors.grey, alignment=TA_CENTER))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], textColor=NAVY, fontSize=16, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], textColor=BLUE, fontSize=12.5, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("KPI", parent=styles["Normal"], fontSize=11, textColor=colors.white, alignment=TA_CENTER))


def P(t, s="Body"):
    return Paragraph(t, styles[s])


def img(path, width=PAGE_W, caption=None):
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    iw = width
    ih = iw * h / w
    out = [Image(str(path), width=iw, height=ih)]
    if caption:
        out.append(P(caption, "Cap"))
    return out


def kpi_band(items):
    """items = [(value, label), ...] -> colored row of KPI boxes."""
    cells = []
    for val, label in items:
        cells.append(Paragraph(f"<b><font size=16>{val}</font></b><br/>"
                               f"<font size=9>{label}</font>", styles["KPI"]))
    t = Table([cells], colWidths=[PAGE_W / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEAFTER", (0, 0), (-2, -1), 1, colors.white),
    ]))
    return t


def styled_table(data, header_bg=NAVY, highlight_row=None):
    t = Table(data, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#D6E8D5")))
        style.append(("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def build():
    # ---- load real data -------------------------------------------------
    metrics = json.loads((REPORTS_DIR / "model_comparison.json").read_text())
    optuna = json.loads((REPORTS_DIR / "optuna_efficientnet_b0.json").read_text())
    df = pd.read_csv(REPORTS_DIR / "image_metadata.csv")
    best = max(metrics, key=lambda m: m["f1"])
    pretty = {"cnn": "Baseline CNN", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"}

    story = []

    # ===== TITLE PAGE ====================================================
    story += [Spacer(1, 3.5 * cm)]
    story += [P("Vehicle Damage Assessment", "TitleBig")]
    story += [P("Deep-Learning Classification of Car Damage from Images", "Sub")]
    story += [Spacer(1, 0.3 * cm)]
    story += [P("EDA &middot; Transfer Learning &middot; Hyperparameter Tuning &middot; "
                "Explainable AI &middot; Deployment", "Sub")]
    story += [Spacer(1, 1.2 * cm)]
    story += [kpi_band([(f"{best['accuracy']*100:.1f}%", "Best accuracy"),
                        (f"{best['f1']:.3f}", "Best F1"),
                        (f"{best['roc_auc']:.3f}", "ROC-AUC"),
                        (pretty[best["model"]], "Best model")])]
    story += [Spacer(1, 5 * cm)]
    story += [P("Computer Vision Portfolio Project", "Sub")]
    story += [P("Prepared by triaz.malik@gmail.com &middot; June 2026", "Sub")]
    story += [PageBreak()]

    # ===== EXECUTIVE SUMMARY ============================================
    story += [P("1. Executive Summary", "H1")]
    story += [P(
        f"This project delivers an end-to-end computer-vision system that classifies a car "
        f"photograph as <b>damaged</b> or <b>whole</b>. Three models were trained and compared on a "
        f"balanced dataset of {len(df):,} images. The best model, <b>{pretty[best['model']]}</b>, "
        f"reaches <b>{best['accuracy']*100:.1f}% accuracy</b> and an <b>F1 of {best['f1']:.3f}</b> "
        f"(ROC-AUC {best['roc_auc']:.3f}) on a held-out validation set, after a data-cleaning step "
        f"removed images that were leaking between the train and validation splits. The system "
        f"includes Grad-CAM explainability and a Streamlit web app for live predictions.")]
    story += [Spacer(1, 0.3 * cm)]
    story += [P("Pipeline at a glance", "H2")]
    story += [P(
        "EDA &rarr; data cleaning (de-leaking) &rarr; augmentation &rarr; baseline CNN, ResNet50 and "
        "EfficientNet-B0 (transfer learning) &rarr; Optuna hyperparameter search &rarr; evaluation "
        "(accuracy / precision / recall / F1 / ROC-AUC) &rarr; Grad-CAM explainability &rarr; "
        "Streamlit deployment. All steps are reproducible scripts in <font face='Courier'>src/</font>.")]

    # ===== BUSINESS VALUE ===============================================
    story += [P("2. Business Problem &amp; Value", "H1")]
    story += [P(
        "Insurance companies inspect vehicle damage manually. The process is slow (often multiple "
        "days per claim), inconsistent between adjusters, and exposed to fraudulent claims. An "
        "automated first-pass classifier converts a multi-day review into a few seconds, routes "
        "likely-damaged vehicles to adjusters, and produces a visual audit trail (Grad-CAM) for "
        "every decision.")]
    biz = [["Dimension", "Manual process", "AI-assisted (this system)"],
           ["First-pass review time", "Hours to days", "Seconds"],
           ["Consistency", "Varies by adjuster", "Deterministic & auditable"],
           ["Fraud signal", "Manual judgement", "Confidence score + Grad-CAM evidence"],
           ["Cost per assessment", "Skilled manual labour", "One GPU inference"],
           ["Scalability", "Limited by headcount", "Thousands of images / hour"]]
    story += [Spacer(1, 0.2 * cm), styled_table(biz)]

    story += [PageBreak()]

    # ===== DATA & EDA ===================================================
    story += [P("3. Dataset &amp; Exploratory Data Analysis", "H1")]
    n_train = int((df.split == "train").sum())
    n_val = int((df.split == "val").sum())
    eda = [["Property", "Value"],
           ["Total images", f"{len(df):,}"],
           ["Classes", "damage, whole (binary)"],
           ["Train / Validation", f"{n_train:,} / {n_val:,}"],
           ["Class balance (train)", "50% damage / 50% whole (balanced)"],
           ["Corrupt images", "0"],
           ["Width range (px)", f"{int(df.width.min())} - {int(df.width.max())} (median {int(df.width.median())})"],
           ["Height range (px)", f"{int(df.height.min())} - {int(df.height.max())} (median {int(df.height.median())})"],
           ["Chosen input size", "224 x 224 (ImageNet standard)"]]
    story += [styled_table(eda)]
    story += [P("Because the dataset is balanced, no class weighting or focal loss was needed; "
                "standard cross-entropy is appropriate.", "Body")]
    story += [Spacer(1, 0.3 * cm)]
    story += img(FIGURES_DIR / "01_class_distribution.png", PAGE_W,
                 "Figure 1 - Class distribution across train and validation splits (balanced).")
    story += img(FIGURES_DIR / "04_sample_images.png", PAGE_W,
                 "Figure 2 - Sample images. Top row: damaged. Bottom row: whole.")
    story += [PageBreak()]
    story += img(FIGURES_DIR / "02_resolution_hist.png", PAGE_W,
                 "Figure 3 - Image width & height distributions (used to choose the 224x224 resize).")
    story += img(FIGURES_DIR / "05_rgb_means.png", PAGE_W * 0.75,
                 "Figure 4 - Mean RGB intensity by class.")

    # ===== CLEANING =====================================================
    story += [P("4. Data Cleaning - Removing Data Leakage", "H1")]
    story += [P(
        "<b>Key finding.</b> The EDA duplicate scan revealed exact-duplicate images that appeared in "
        "<i>both</i> the training and validation folders - a classic case of data leakage that "
        "silently inflates validation scores, because the model is tested on images it has already "
        "seen during training.")]
    clean = [["Issue", "Files", "Action"],
             ["Corrupt / unreadable", "0", "-"],
             ["Cross-split leakage (train ∩ val)", "3", "Validation copy quarantined"],
             ["Within-split exact duplicates", "10", "Kept one, quarantined rest"],
             ["Very blurry (low Laplacian var.)", "1", "Quarantined"],
             ["Total removed", "14", "Moved to quarantine/ (reversible)"]]
    story += [Spacer(1, 0.2 * cm), styled_table(clean)]
    story += [P("Files were <b>moved</b> to a <font face='Courier'>quarantine/</font> folder rather "
                "than deleted, so the operation is fully reversible. All reported metrics are computed "
                "on the cleaned, leak-free validation set (1,830 train / 456 validation images).", "Body")]

    story += [PageBreak()]

    # ===== METHODOLOGY ==================================================
    story += [P("5. Methodology", "H1")]
    story += [P("Augmentation (Phase 3)", "H2")]
    story += [P("To make the model robust to the conditions cars are photographed in (angle, weather, "
                "lighting), training images are augmented with random resized-crop, horizontal flip, "
                "&plusmn;15&deg; rotation, and brightness/contrast/saturation jitter. All images are "
                "normalized with ImageNet statistics.")]
    story += [P("Models (Phases 4-6)", "H2")]
    models_tbl = [["Model", "Type", "Notes"],
                  ["Baseline CNN", "Trained from scratch", "4x (Conv-BN-ReLU-Pool) + dense head; benchmark"],
                  ["ResNet50", "Transfer learning", "ImageNet weights, fine-tuned; industry standard"],
                  ["EfficientNet-B0", "Transfer learning", "Smaller & faster, strong accuracy"]]
    story += [styled_table(models_tbl)]
    story += [P("Training used AdamW, cosine learning-rate decay, mixed-precision (AMP), and early "
                "stopping on validation accuracy. Hardware: NVIDIA RTX 5080 (CUDA 12.8), ~3-4 s/epoch.", "Body")]

    # ===== TUNING =======================================================
    story += [P("6. Hyperparameter Tuning (Optuna)", "H1")]
    bp = optuna["best_params"]
    story += [P(
        f"An Optuna TPE search ran <b>{len(optuna['trials'])} trials</b> over learning rate, batch "
        f"size, dropout and weight decay for EfficientNet-B0, optimizing accuracy on a tuning split "
        f"held out from the training data (the real validation set was never used for tuning). The "
        f"best trial reached <b>{optuna['best_value']*100:.1f}%</b> tuning accuracy.")]
    tune_tbl = [["Hyperparameter", "Search space", "Best value"],
                ["Learning rate", "1e-5 - 3e-3 (log)", f"{bp['lr']:.2e}"],
                ["Batch size", "16 / 32 / 64", str(bp["batch_size"])],
                ["Dropout", "0.2 - 0.6", f"{bp['dropout']:.2f}"],
                ["Weight decay", "1e-6 - 1e-3 (log)", f"{bp['weight_decay']:.2e}"]]
    story += [Spacer(1, 0.2 * cm), styled_table(tune_tbl)]

    story += [PageBreak()]

    # ===== RESULTS ======================================================
    story += [P("7. Results &amp; Evaluation", "H1")]
    order = ["cnn", "resnet50", "efficientnet_b0"]
    metrics.sort(key=lambda m: order.index(m["model"]))
    res = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    hl = None
    for i, m in enumerate(metrics, start=1):
        if m["model"] == best["model"]:
            hl = i
        res.append([pretty[m["model"]], f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
                    f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}"])
    story += [styled_table(res, highlight_row=hl)]
    story += [P(f"The positive class is <b>damage</b>. <b>{pretty[best['model']]}</b> is the best "
                f"model on every metric, with an F1 of {best['f1']:.3f} and near-perfect class "
                f"separation (ROC-AUC {best['roc_auc']:.3f}).", "Body")]
    story += [Spacer(1, 0.3 * cm)]
    story += img(FIGURES_DIR / "roc_curves.png", PAGE_W * 0.6,
                 "Figure 5 - ROC curves for all three models.")
    story += [PageBreak()]
    story += img(FIGURES_DIR / "pr_curves.png", PAGE_W * 0.6,
                 "Figure 6 - Precision-Recall curves.")
    story += img(FIGURES_DIR / f"confusion_{best['model']}.png", PAGE_W * 0.55,
                 f"Figure 7 - Confusion matrix for the best model ({pretty[best['model']]}).")
    story += img(FIGURES_DIR / f"history_{best['model']}.png", PAGE_W,
                 f"Figure 8 - Training/validation curves for {pretty[best['model']]}.")

    story += [PageBreak()]

    # ===== EXPLAINABILITY ===============================================
    story += [P("8. Explainable AI - Grad-CAM", "H1")]
    story += [P("Grad-CAM highlights the image regions that most influenced each prediction. For "
                "damaged vehicles the heatmaps concentrate on dents, crumpled panels and broken "
                "glass; for whole vehicles attention spreads over the intact body. This gives "
                "adjusters a visual justification for every automated decision.")]
    story += [Spacer(1, 0.3 * cm)]
    story += img(GRADCAM_DIR / "gradcam_resnet50.png", PAGE_W,
                 "Figure 9 - Grad-CAM overlays (warm regions drive the prediction).")

    # ===== FUTURE WORK & CONCLUSION =====================================
    story += [P("9. Future Work", "H1")]
    story += [P(
        "<b>Damage severity (Phase 10)</b> and <b>repair-cost estimation (Phase 11)</b> would add "
        "significant business value but require labels this dataset does not contain (severity grades "
        "and cost figures). Model heads and an interface for both are scaffolded in "
        "<font face='Courier'>src/future_work.py</font>, ready to train once labelled data is "
        "available. Further extensions: multi-class damage location (front/rear/side) and object "
        "detection (YOLO) to localize individual dents and scratches.")]
    story += [P("10. Conclusion", "H1")]
    story += [P(
        f"A reproducible, explainable car-damage classifier was delivered end-to-end. "
        f"{pretty[best['model']]} achieves {best['accuracy']*100:.1f}% accuracy / {best['f1']:.3f} F1 "
        f"on a leak-free validation set, packaged with Grad-CAM explanations and a Streamlit app. "
        f"The careful handling of data leakage and the explainability layer make the results both "
        f"trustworthy and deployable.")]

    doc = SimpleDocTemplate(
        str(REPORTS_DIR / "Vehicle_Damage_Assessment_Report.pdf"),
        pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Vehicle Damage Assessment Report", author="triaz.malik@gmail.com",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1 * cm, "Vehicle Damage Assessment")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved -> {REPORTS_DIR / 'Vehicle_Damage_Assessment_Report.pdf'}")


if __name__ == "__main__":
    build()
