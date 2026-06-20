"""Assemble a multi-page PDF report from all generated artifacts.

Zero extra dependencies: uses matplotlib's PdfPages to lay out text pages and
embed the existing PNG figures, metrics JSON, and comparison CSV.

Run (after train -> evaluate -> compare -> gradcam [-> tune]):
  python -m src.make_report
Output:
  outputs/reports/Pneumonia_Detection_Report.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import config as C

PAGE = (8.27, 11.69)        # A4 portrait, inches
TODAY = "2026-06-07"


# --------------------------------------------------------------------------- #
# page builders
# --------------------------------------------------------------------------- #
def _new_page(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.06)
    return fig


def text_page(pdf, title, body, *, subtitle=None, footer=None):
    fig = _new_page(pdf)
    fig.text(0.08, 0.95, title, fontsize=20, fontweight="bold",
             color="#1a3d6d", va="top")
    if subtitle:
        fig.text(0.08, 0.905, subtitle, fontsize=11, color="#555", va="top")
    fig.text(0.08, 0.86, body, fontsize=10.5, va="top", family="monospace",
             wrap=True)
    if footer:
        fig.text(0.08, 0.04, footer, fontsize=8, color="#888")
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, image_paths, *, captions=None, ncols=1, note=None):
    paths = [p for p in image_paths if Path(p).exists()]
    if not paths:
        return
    fig = _new_page(fig_pdf := pdf) if False else _new_page(pdf)
    fig.text(0.08, 0.96, title, fontsize=16, fontweight="bold",
             color="#1a3d6d", va="top")
    n = len(paths)
    nrows = (n + ncols - 1) // ncols
    start_top = 0.90
    avail_h = start_top - 0.06
    for i, p in enumerate(paths):
        r, c = divmod(i, ncols)
        cell_h = avail_h / nrows
        cell_w = 0.84 / ncols
        x0 = 0.08 + c * cell_w
        y0 = start_top - (r + 1) * cell_h
        ax = fig.add_axes([x0, y0 + 0.01, cell_w * 0.96, cell_h * 0.88])
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
        if captions and i < len(captions):
            ax.set_title(captions[i], fontsize=9)
    if note:
        fig.text(0.08, 0.035, note, fontsize=8.5, color="#444", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, df: pd.DataFrame, *, note=None, fmt="%.4f"):
    fig = _new_page(pdf)
    fig.text(0.08, 0.96, title, fontsize=16, fontweight="bold",
             color="#1a3d6d", va="top")
    ax = fig.add_axes([0.06, 0.45, 0.88, 0.4]); ax.axis("off")
    disp = df.copy()
    for col in disp.select_dtypes("float").columns:
        disp[col] = disp[col].map(lambda v: fmt % v if pd.notna(v) else "—")
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a3d6d"); cell.set_text_props(color="white",
                                                               fontweight="bold")
    if note:
        fig.text(0.08, 0.4, note, fontsize=9.5, va="top", color="#333", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _safe_read_csv(p):
    return pd.read_csv(p) if Path(p).exists() else None


def _manifest_counts():
    rows = []
    for s in ("train", "val", "test"):
        df = _safe_read_csv(C.MANIFEST_DIR / f"{s}.csv")
        if df is not None:
            vc = df["label"].value_counts()
            rows.append({"split": s, "NORMAL": int(vc.get("NORMAL", 0)),
                         "PNEUMONIA": int(vc.get("PNEUMONIA", 0)),
                         "total": len(df)})
    return pd.DataFrame(rows)


def build():
    out = C.REPORT_DIR / "Pneumonia_Detection_Report.pdf"
    cmp_df = _safe_read_csv(C.REPORT_DIR / "model_comparison.csv")
    counts = _manifest_counts()

    # best model summary for exec summary / business page
    best_line = "(train + evaluate + compare first)"
    if cmp_df is not None and len(cmp_df):
        b = cmp_df.iloc[0]
        best_line = (f"{b['model']} — recall {b['recall']:.3f}, F1 {b['f1']:.3f}, "
                     f"AUC {b['auc']:.3f}, accuracy {b['accuracy']:.3f}")

    with PdfPages(out) as pdf:
        # ---- title / executive summary ----
        text_page(
            pdf,
            "Pneumonia Detection from Chest X-Rays",
            (
                "EXECUTIVE SUMMARY\n"
                "--------------------------------------------------------------\n"
                "Goal: automatically screen pediatric chest X-rays as NORMAL vs\n"
                "PNEUMONIA to speed diagnosis, prioritise critical patients, and\n"
                "reduce radiologist workload.\n\n"
                "Because a MISSED pneumonia case is the dangerous error, the\n"
                "primary KPI is RECALL (Recall > F1 > AUC > Accuracy).\n\n"
                f"Best model: {best_line}\n\n"
                "Approach: clean + stratified re-split -> EDA -> augmentation ->\n"
                "three models (Custom CNN, ResNet50, EfficientNetB0) with two-\n"
                "phase transfer learning and class-weighted loss -> hyperparameter\n"
                "sweep -> recall-first evaluation -> Grad-CAM explainability.\n\n"
                "Built in PyTorch (CUDA 12.8) on an NVIDIA RTX 5080."
            ),
            subtitle=f"Healthcare CNN project report   |   {TODAY}   |   triaz.malik@gmail.com",
            footer="Generated by src/make_report.py",
        )

        # ---- business problem ----
        text_page(
            pdf, "1. Business Problem & Success Criteria",
            (
                "PROBLEM\n"
                "Hospitals receive thousands of chest X-rays daily. Radiologists\n"
                "are overloaded; pneumonia diagnosis can be delayed; missed cases\n"
                "lead to severe complications.\n\n"
                "BUSINESS VALUE\n"
                "  - Faster diagnosis & earlier treatment\n"
                "  - Prioritise critical patients (triage)\n"
                "  - Reduce radiologist workload\n"
                "  - More consistent screening\n\n"
                "SUCCESS CRITERIA (hospital objective: minimise false negatives)\n"
                "  Primary KPI : Recall  (missing pneumonia is dangerous)\n"
                "  Then        : F1 > AUC > Accuracy\n\n"
                "Design choices that follow from this:\n"
                "  - Checkpoint the model with the best VALIDATION RECALL\n"
                "  - Class-weighted loss for the 2.69:1 imbalance\n"
                "  - Report a recall-tuned decision threshold, not just 0.5"
            ),
        )

        # ---- dataset + data quality ----
        if not counts.empty:
            table_page(
                pdf, "2. Dataset & Data Quality", counts, fmt="%d",
                note=(
                    "Splits are stored as non-destructive CSV manifests. The\n"
                    "original validation set had only 16 images, so train+val were\n"
                    "merged and stratified re-split (15% val); the official TEST set\n"
                    "is untouched for comparability.\n\n"
                    "Data cleaning: 26 exact-duplicate images detected (md5) and\n"
                    "excluded from the train/val pool; 0 corrupt/empty files.\n"
                    "Images are JPEG, variable resolution, standardised to 224x224."
                ),
            )

        # ---- EDA ----
        image_page(pdf, "3. EDA — Class Distribution & Balance",
                   [C.FIG_DIR / "01_class_distribution.png",
                    C.FIG_DIR / "02_class_balance_pie.png"], ncols=1,
                   note="~72.9% of images are PNEUMONIA (2.69:1 imbalance) -> "
                        "handled with class-weighted loss + augmentation.")
        image_page(pdf, "3. EDA — Sample X-rays",
                   [C.FIG_DIR / "03_sample_normal.png",
                    C.FIG_DIR / "04_sample_pneumonia.png"], ncols=1,
                   note="NORMAL: clear lungs. PNEUMONIA: white opacities / dense "
                        "infiltrates.")
        image_page(pdf, "3. EDA — Image Size & Pixel Intensity",
                   [C.FIG_DIR / "05_image_size_dist.png",
                    C.FIG_DIR / "06_pixel_intensity.png"], ncols=1,
                   note="Variable resolution -> resize to 224x224. Pneumonia images "
                        "skew brighter (denser opacities).")

        # ---- data prep / augmentation ----
        text_page(
            pdf, "4. Data Preparation & Augmentation",
            (
                "CLEANING\n"
                "  - Remove corrupted / empty files (PIL verify + size check)\n"
                "  - Remove exact duplicates (md5 content hash)\n"
                "  - Resize to 224 x 224\n\n"
                "AUGMENTATION (training only)\n"
                "  - Rotation +/- 15 deg\n"
                "  - Zoom (scale 0.9-1.1)\n"
                "  - Horizontal flip (p=0.5)\n"
                "  - Brightness jitter (0.15)\n"
                "  - Small translations (5%)\n\n"
                "NORMALISATION\n"
                "  - Grayscale X-ray replicated to 3 channels\n"
                "  - ImageNet mean/std (to reuse pretrained backbones)\n\n"
                "BENEFIT: more robust model, less overfitting on the minority\n"
                "(NORMAL) class."
            ),
        )

        # ---- model architectures ----
        text_page(
            pdf, "5. Modeling Approach",
            (
                "MODEL 1 — Custom CNN (baseline)\n"
                "  Conv-BN-ReLU-Pool x3 -> GAP -> FC -> 2 logits (~0.1M params)\n\n"
                "MODEL 2 — ResNet50 (transfer learning, ~23.5M params)\n"
                "  ImageNet-pretrained residual backbone, new 2-class head\n\n"
                "MODEL 3 — EfficientNetB0 (transfer learning, ~4.0M params)\n"
                "  Compound-scaled backbone; strong accuracy at low param count\n\n"
                "TWO-PHASE TRANSFER LEARNING\n"
                "  Phase 1: freeze backbone, train the head (lr=1e-3)\n"
                "  Phase 2: unfreeze, fine-tune whole net (lr=1e-4)\n\n"
                "TRAINING\n"
                "  - Class-weighted CrossEntropy (inverse frequency)\n"
                "  - Adam + ReduceLROnPlateau (on val recall)\n"
                "  - Mixed precision (AMP) on the RTX 5080\n"
                "  - Early stopping + checkpoint best val recall"
            ),
        )

        # ---- hyperparameter tuning ----
        tune_df = _safe_read_csv(C.REPORT_DIR / "tuning_results.csv")
        if tune_df is not None:
            table_page(pdf, "6. Hyperparameter Tuning", tune_df,
                       note="Grid over learning rate x optimizer x dropout on the "
                            "strongest architecture; selected by validation recall.")
            image_page(pdf, "6. Hyperparameter Tuning — Sweep",
                       [C.FIG_DIR / "tuning_results.png"])
        else:
            text_page(pdf, "6. Hyperparameter Tuning",
                      ("Search space (LR x optimizer x dropout) is implemented in\n"
                       "src/tune.py. Run `python -m src.tune` to populate this\n"
                       "section. The main training also adapts LR via\n"
                       "ReduceLROnPlateau and uses early stopping."))

        # ---- training history ----
        image_page(pdf, "7. Training History",
                   [C.FIG_DIR / "history_custom_cnn.png",
                    C.FIG_DIR / "history_resnet50.png",
                    C.FIG_DIR / "history_efficientnet_b0.png"], ncols=1)

        # ---- results ----
        if cmp_df is not None:
            show = cmp_df[["model", "recall", "precision", "f1", "auc",
                           "accuracy", "recall_tuned", "fn_default"]].copy()
            table_page(pdf, "8. Results — Model Comparison (TEST set)", show,
                       note="recall_tuned = recall at the recall-optimised threshold "
                            "(precision floor 0.80). fn_default = false negatives at "
                            "threshold 0.5 (the costly errors).")
        image_page(pdf, "8. Results — Comparison Chart",
                   [C.FIG_DIR / "model_comparison.png"])

        # ---- confusion matrices ----
        image_page(pdf, "9. Confusion Matrices (TEST, threshold 0.5)",
                   [C.FIG_DIR / "cm_custom_cnn_default.png",
                    C.FIG_DIR / "cm_resnet50_default.png",
                    C.FIG_DIR / "cm_efficientnet_b0_default.png"], ncols=2)

        # ---- ROC / PR ----
        image_page(pdf, "10. ROC & Precision-Recall Curves",
                   [C.FIG_DIR / "roc_efficientnet_b0.png",
                    C.FIG_DIR / "pr_efficientnet_b0.png",
                    C.FIG_DIR / "roc_resnet50.png",
                    C.FIG_DIR / "pr_resnet50.png"], ncols=2)

        # ---- grad-cam ----
        image_page(pdf, "11. Explainability — Grad-CAM",
                   [C.FIG_DIR / "gradcam_efficientnet_b0.png"],
                   note="Heatmaps show which lung regions drove the PNEUMONIA "
                        "prediction (original | heatmap | overlay). Builds clinician "
                        "trust and supports error analysis.")

        # ---- business impact + future ----
        # pull headline numbers for the business page
        biz_extra = ""
        if cmp_df is not None and len(cmp_df):
            b = cmp_df.iloc[0]
            fnr = b["fn_default"]
            biz_extra = (f"\nBest model ({b['model']}) on the test set:\n"
                         f"  Recall {b['recall']:.1%} | FN (missed pneumonia) = "
                         f"{int(fnr)} of test positives at thr=0.5\n")
        text_page(
            pdf, "12. Business Impact & Future Work",
            (
                "BUSINESS IMPACT\n"
                "  - Triage: auto-flag likely-pneumonia X-rays for urgent review\n"
                "  - Speed: seconds per image vs minutes of manual screening\n"
                "  - Consistency: uniform screening, fewer missed cases\n"
                "  - Recall-first threshold keeps the false-negative rate low\n"
                f"{biz_extra}\n"
                "EXAMPLE DASHBOARD METRICS\n"
                "  Daily X-rays, predicted pneumonia, high-risk count,\n"
                "  live Recall and False-Negative-Rate.\n\n"
                "FUTURE ENHANCEMENTS\n"
                "  - Multi-class (Normal / Pneumonia / COVID / TB)\n"
                "  - Vision Transformers (ViT), ensembles\n"
                "  - Calibrated probabilities + abstention for borderline cases\n"
                "  - Deployment on Azure ML / AWS SageMaker, PACS/RIS integration"
            ),
            footer="End of report.",
        )

    print(f"Report written -> {out}")
    return out


if __name__ == "__main__":
    build()
