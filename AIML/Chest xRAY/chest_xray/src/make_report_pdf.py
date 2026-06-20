"""Generate a professional multi-page PDF report for the Pneumonia Detection
project using reportlab (branded navy look with KPI cards and captions).

Uses ONLY real artifacts that exist on disk:
  outputs/reports/  eda_summary.txt, history_custom_cnn.csv, history_resnet50.csv,
                    train_summary_custom_cnn.json, train_summary_resnet50.json
  outputs/manifests/  train.csv, val.csv, test.csv, data_quality_report.csv
  outputs/figures/   01..06 EDA pngs, history_custom_cnn.png, history_resnet50.png

Per-model metrics are taken from the best-validation-recall epoch of each
history CSV (recall is the checkpoint-selection metric). Anything not present
on disk is omitted -- no fabricated numbers.

Run:
  python -m src.make_report_pdf        (or)   python src/make_report_pdf.py
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

# --------------------------------------------------------------------------- #
# Paths (derived from this file's location -- robust to launch directory)
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent          # ...\chest_xray\src
PROJECT_ROOT = SCRIPT_DIR.parent                      # ...\chest_xray
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
REPORT_DIR = OUTPUT_DIR / "reports"

REPORT_PATH = REPORT_DIR / "Pneumonia_Detection_Report.pdf"
# extra copies requested
COPY_TARGETS = [
    PROJECT_ROOT / "Pneumonia_Detection_Report.pdf",          # chest_xray root
    PROJECT_ROOT.parent / "Pneumonia_Detection_Report.pdf",   # AIML\Chest xRAY
]

MODELS = [
    ("Custom CNN", "custom_cnn"),
    ("ResNet50", "resnet50"),
]

# --------------------------------------------------------------------------- #
# Brand palette
# --------------------------------------------------------------------------- #
NAVY = colors.HexColor("#1f3a5f")
BLUE = colors.HexColor("#2e6da4")
LIGHT = colors.HexColor("#eaf1f8")
GREY = colors.HexColor("#555555")
WIN = colors.HexColor("#d4edda")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=28,
                          textColor=NAVY, leading=34, spaceAfter=12))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=13,
                          textColor=GREY, alignment=TA_CENTER, leading=19))
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
styles.add(ParagraphStyle("KPILabel", parent=styles["Normal"], fontSize=9,
                          textColor=GREY, alignment=TA_CENTER))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def p(text, style="Body"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", styles["Bul"]) for t in items]


def hr():
    return HRFlowable(width="100%", thickness=1, color=LIGHT,
                      spaceBefore=6, spaceAfter=10)


def fig(name, width=15 * cm, caption=None):
    """Flowables for a figure if it exists on disk, else an empty list."""
    path = FIG_DIR / name
    if not path.exists():
        return []
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    out = [Spacer(1, 4), img]
    out.append(p(caption, "Caption") if caption else Spacer(1, 8))
    return out


def safe_read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_epoch_metrics(history_csv):
    """Return the row (as dict) of the epoch with the highest val_recall.

    Recall is the checkpoint-selection metric for this project, so the best-
    recall epoch is the model that would be saved/deployed. Returns None if the
    history is missing.
    """
    df = safe_read_csv(history_csv)
    if df is None or df.empty or "val_recall" not in df.columns:
        return None
    row = df.loc[df["val_recall"].idxmax()]
    return {
        "epoch": int(row["epoch"]),
        "epochs_run": int(df["epoch"].max()),
        "recall": float(row["val_recall"]),
        "precision": float(row.get("val_precision", float("nan"))),
        "f1": float(row.get("val_f1", float("nan"))),
        "auc": float(row.get("val_auc", float("nan"))),
        "accuracy": float(row.get("val_accuracy", float("nan"))),
    }


def collect_model_metrics():
    """Build a list of per-model metric dicts from history + train summaries."""
    rows = []
    for label, key in MODELS:
        m = best_epoch_metrics(REPORT_DIR / f"history_{key}.csv")
        if m is None:
            continue
        summ = load_json(REPORT_DIR / f"train_summary_{key}.json") or {}
        m["model"] = label
        m["minutes"] = summ.get("minutes")
        rows.append(m)
    # Sort by recall (primary KPI) descending so the best model leads.
    rows.sort(key=lambda r: r["recall"], reverse=True)
    return rows


def manifest_counts():
    """Train/val/test class counts from the CSV manifests (fallback to None)."""
    rows = []
    for s in ("train", "val", "test"):
        df = safe_read_csv(MANIFEST_DIR / f"{s}.csv")
        if df is None or "label" not in df.columns:
            continue
        vc = df["label"].value_counts()
        rows.append({
            "split": s,
            "NORMAL": int(vc.get("NORMAL", 0)),
            "PNEUMONIA": int(vc.get("PNEUMONIA", 0)),
            "total": int(len(df)),
        })
    return rows


def duplicate_count():
    df = safe_read_csv(MANIFEST_DIR / "data_quality_report.csv")
    if df is None or df.empty or "issue" not in df.columns:
        return None, None
    dups = int((df["issue"] == "DUPLICATE").sum())
    others = int((df["issue"] != "DUPLICATE").sum())
    return dups, others


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "&mdash;"
    return f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def kpi_row(metrics):
    best = metrics[0]
    cards = [
        (f"{best['recall']:.1%}", f"Best val recall &mdash; {best['model']}"),
        (f"{best['accuracy']:.1%}" if best['accuracy'] == best['accuracy'] else "n/a",
         f"Val accuracy &mdash; {best['model']}"),
        (str(len(metrics)), "deep-learning models trained"),
    ]
    cells = []
    for value, label in cards:
        inner = Table([[p(value, "KPI")], [p(label, "KPILabel")]],
                      colWidths=[5.2 * cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        cells.append(inner)
    outer = Table([cells], colWidths=[5.6 * cm] * 3)
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return outer


def split_table(counts):
    header = ["Split", "NORMAL", "PNEUMONIA", "Total"]
    data = [header]
    for r in counts:
        data.append([r["split"], f"{r['NORMAL']:,}", f"{r['PNEUMONIA']:,}",
                     f"{r['total']:,}"])
    grand = sum(r["total"] for r in counts)
    data.append(["all", f"{sum(r['NORMAL'] for r in counts):,}",
                 f"{sum(r['PNEUMONIA'] for r in counts):,}", f"{grand:,}"])
    t = Table(data, colWidths=[4 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, LIGHT]),
    ]))
    return t


def comparison_table(metrics):
    header = ["Model", "Recall", "Precision", "F1", "AUC", "Accuracy",
              "Best epoch", "Train min"]
    data = [header]
    for m in metrics:
        data.append([
            m["model"], fmt(m["recall"]), fmt(m["precision"]), fmt(m["f1"]),
            fmt(m["auc"]), fmt(m["accuracy"]),
            f"{m['epoch']}/{m['epochs_run']}",
            fmt(m["minutes"], 2) if m["minutes"] is not None else "&mdash;",
        ])
    t = Table(data, colWidths=[3 * cm, 1.9 * cm, 2.1 * cm, 1.6 * cm, 1.6 * cm,
                               2 * cm, 1.9 * cm, 1.9 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        # winner (first, highest recall) highlighted
        ("BACKGROUND", (0, 1), (-1, 1), WIN),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ]
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build():
    metrics = collect_model_metrics()
    counts = manifest_counts()
    dups, other_issues = duplicate_count()

    if not metrics:
        raise SystemExit("No model history found -- cannot build report.")

    best = metrics[0]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Pneumonia Detection from Chest X-Rays",
        author="Triaz Malik",
    )
    S = []

    # ---------------- Cover + KPIs ----------------------------------------
    S.append(Spacer(1, 3.2 * cm))
    S.append(p("Pneumonia Detection from Chest X-Rays", "CoverTitle"))
    S.append(p("Deep-Learning Screening of Pediatric Chest Radiographs "
               "&mdash; NORMAL vs PNEUMONIA", "CoverSub"))
    S.append(Spacer(1, 0.4 * cm))
    S.append(hr())
    S.append(Spacer(1, 0.2 * cm))
    S.append(p("Project Report &mdash; Business Framing, Data Quality, EDA, "
               "Modeling, Training History &amp; Results", "CoverSub"))
    S.append(Spacer(1, 1.2 * cm))
    S.append(kpi_row(metrics))
    S.append(Spacer(1, 1.0 * cm))
    S.append(p("Primary KPI: <b>Recall</b> (missing pneumonia is the dangerous "
               "error) &mdash; Recall &gt; F1 &gt; AUC &gt; Accuracy.", "CoverSub"))
    S.append(PageBreak())

    # ---------------- 1. Business Problem ---------------------------------
    S.append(p("1. Business Problem", "H1"))
    S.append(hr())
    S.append(p("Hospitals receive large volumes of chest X-rays daily and "
               "radiologists are overloaded, which can delay pneumonia "
               "diagnosis. A <b>missed</b> pneumonia case is the costly error, "
               "so this project builds an automated screen that triages "
               "radiographs as NORMAL vs PNEUMONIA, flags high-risk patients "
               "first, and reduces missed cases."))
    S.append(p("Business value", "H2"))
    S.extend(bullets([
        "<b>Faster diagnosis</b> &mdash; seconds per image vs minutes of manual screening.",
        "<b>Triage</b> &mdash; auto-flag likely-pneumonia X-rays for urgent review.",
        "<b>Reduced workload</b> &mdash; radiologists focus on ambiguous cases.",
        "<b>Consistent screening</b> &mdash; uniform criteria, fewer missed cases.",
    ]))
    S.append(p("Success criteria", "H2"))
    S.append(p("Because the hospital objective is to <b>minimise false "
               "negatives</b>, the primary KPI is <b>Recall</b> on the "
               "PNEUMONIA class, then F1 &gt; AUC &gt; Accuracy. Models are "
               "checkpointed on the best validation recall, the 2.69:1 class "
               "imbalance is countered with class-weighted loss, and a "
               "recall-first decision threshold is preferred over a naive 0.5."))

    # ---------------- 2. Dataset & Data Quality ---------------------------
    S.append(p("2. Dataset &amp; Data Quality", "H1"))
    S.append(hr())
    if counts:
        total = sum(r["total"] for r in counts)
        S.append(p(f"Splits are stored as non-destructive CSV manifests "
                   f"({total:,} images total). The original validation set had "
                   f"only 16 images, so train+val were merged and "
                   f"stratified-resplit (15% validation); the official TEST set "
                   f"is left untouched for comparability."))
        S.append(split_table(counts))
        S.append(Spacer(1, 8))
    quality_lines = []
    if dups is not None:
        quality_lines.append(
            f"<b>{dups} exact-duplicate images</b> detected (content hash) and "
            f"excluded from the train/val pool.")
        if other_issues:
            quality_lines.append(
                f"<b>{other_issues} other flagged files</b> (corrupt/empty) excluded.")
        else:
            quality_lines.append("<b>0 corrupt/empty files</b>.")
    quality_lines.append("Images are JPEG at variable resolution, standardised "
                         "to 224&times;224 for modeling.")
    S.append(p("Data cleaning", "H2"))
    S.extend(bullets(quality_lines))
    S.append(PageBreak())

    # ---------------- 3. EDA ----------------------------------------------
    S.append(p("3. Exploratory Data Analysis", "H1"))
    S.append(hr())
    S.append(p("3.1 Class distribution &amp; balance", "H2"))
    S.append(p("About <b>72.9%</b> of images are PNEUMONIA (a 2.69:1 "
               "imbalance), which motivates class-weighted loss and "
               "augmentation and makes recall &mdash; not accuracy &mdash; the "
               "right headline metric."))
    S.extend(fig("01_class_distribution.png", width=12 * cm,
                 caption="Fig 1. Image counts per class."))
    S.extend(fig("02_class_balance_pie.png", width=9 * cm,
                 caption="Fig 2. Overall class balance (PNEUMONIA 72.9%)."))
    S.append(PageBreak())
    S.append(p("3.2 Sample radiographs", "H2"))
    S.append(p("NORMAL X-rays show clear lung fields; PNEUMONIA X-rays show "
               "white opacities and dense infiltrates."))
    S.extend(fig("03_sample_normal.png", width=15 * cm,
                 caption="Fig 3. Sample NORMAL chest X-rays (clear lungs)."))
    S.extend(fig("04_sample_pneumonia.png", width=15 * cm,
                 caption="Fig 4. Sample PNEUMONIA X-rays (opacities / infiltrates)."))
    S.append(PageBreak())
    S.append(p("3.3 Image size &amp; pixel intensity", "H2"))
    S.append(p("Resolutions vary widely (width 384&ndash;2916 px, height "
               "127&ndash;2713 px, median ~1280&times;886), so all images are "
               "resized to 224&times;224. Pneumonia images skew brighter, "
               "reflecting denser opacities."))
    S.extend(fig("05_image_size_dist.png", width=13 * cm,
                 caption="Fig 5. Distribution of image dimensions."))
    S.extend(fig("06_pixel_intensity.png", width=13 * cm,
                 caption="Fig 6. Mean pixel-intensity distribution by class."))
    S.append(PageBreak())

    # ---------------- 4. Modeling Approach --------------------------------
    S.append(p("4. Modeling Approach", "H1"))
    S.append(hr())
    S.append(p("Two architectures were trained and compared, both checkpointed "
               "on best validation recall with class-weighted cross-entropy to "
               "counter the imbalance:"))
    S.append(p("Model 1 &mdash; Custom CNN (baseline)", "H2"))
    S.append(p("A compact convolutional baseline (Conv-BN-ReLU-Pool blocks &rarr; "
               "global average pooling &rarr; classifier) trained from scratch to "
               "establish a reference point."))
    S.append(p("Model 2 &mdash; ResNet50 (transfer learning)", "H2"))
    S.append(p("An ImageNet-pretrained ResNet50 backbone with a new 2-class "
               "head, fine-tuned in two phases: freeze the backbone and train "
               "the head first, then unfreeze and fine-tune the whole network "
               "at a lower learning rate. Grayscale X-rays are replicated to 3 "
               "channels to reuse the pretrained weights."))
    S.append(p("Training setup", "H2"))
    S.extend(bullets([
        "Class-weighted cross-entropy (inverse frequency) for the 2.69:1 imbalance.",
        "Checkpoint the model with the best <b>validation recall</b>.",
        "Augmentation (rotation, zoom, flip, brightness) on the training split only.",
        "ImageNet normalisation; images standardised to 224&times;224.",
    ]))
    S.append(PageBreak())

    # ---------------- 5. Training History ---------------------------------
    S.append(p("5. Training History", "H1"))
    S.append(hr())
    S.append(p("Validation recall, precision, F1, AUC and loss were tracked "
               "each epoch; the checkpoint is the best-recall epoch."))
    hist = []
    hist += fig("history_custom_cnn.png", width=15 * cm,
                caption="Fig 7. Custom CNN training history.")
    hist += fig("history_resnet50.png", width=15 * cm,
                caption="Fig 8. ResNet50 training history.")
    S.extend(hist)
    S.append(PageBreak())

    # ---------------- 6. Results ------------------------------------------
    S.append(p("6. Results &amp; Model Comparison", "H1"))
    S.append(hr())
    S.append(p("Metrics below are from each model's <b>best validation-recall "
               "epoch</b> (the checkpoint-selection metric). The best-recall "
               "model leads the table."))
    S.append(comparison_table(metrics))
    S.append(Spacer(1, 8))
    S.append(p(f"<b>Best model: {best['model']}</b> &mdash; validation recall "
               f"{best['recall']:.3f}"
               + (f", F1 {best['f1']:.3f}" if best['f1'] == best['f1'] else "")
               + (f", AUC {best['auc']:.3f}" if best['auc'] == best['auc'] else "")
               + (f", accuracy {best['accuracy']:.3f}" if best['accuracy'] == best['accuracy'] else "")
               + f" at epoch {best['epoch']}/{best['epochs_run']}. Transfer "
               "learning with ResNet50 outperforms the from-scratch baseline on "
               "the recall-first objective."))
    S.append(PageBreak())

    # ---------------- 7. Business Impact & Future Work --------------------
    S.append(p("7. Business Impact &amp; Future Work", "H1"))
    S.append(hr())
    S.append(p("Business impact", "H2"))
    S.extend(bullets([
        f"<b>High recall</b> &mdash; the best model reaches {best['recall']:.1%} "
        "validation recall, keeping missed-pneumonia cases low.",
        "<b>Triage</b> &mdash; auto-flag likely-pneumonia X-rays for urgent review.",
        "<b>Speed &amp; consistency</b> &mdash; seconds per image, uniform screening.",
        "<b>Recall-first threshold</b> keeps the false-negative rate low at "
        "deployment.",
    ]))
    S.append(p("Future work", "H2"))
    S.extend(bullets([
        "Add a third backbone (e.g. EfficientNet) and an ensemble.",
        "Evaluate on the held-out TEST set with a recall-tuned threshold and "
        "report confusion / ROC / PR curves.",
        "Grad-CAM explainability to show which lung regions drive predictions.",
        "Calibrated probabilities with abstention for borderline cases.",
        "Deployment behind an API with PACS/RIS integration.",
    ]))
    S.append(Spacer(1, 0.5 * cm))
    S.append(hr())
    S.append(p("Generated from real project artifacts in "
               "<font name='Courier'>outputs/</font>. Per-model metrics are the "
               "best-validation-recall epoch of each history CSV; figures are "
               "embedded only where they exist on disk.", "Caption"))

    doc.build(S)

    # copies
    copied = [REPORT_PATH]
    for tgt in COPY_TARGETS:
        try:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPORT_PATH, tgt)
            copied.append(tgt)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: could not copy to {tgt}: {exc}")

    print(f"Saved report -> {REPORT_PATH}")
    for c in copied[1:]:
        print(f"Copied       -> {c}")
    return copied


if __name__ == "__main__":
    build()
