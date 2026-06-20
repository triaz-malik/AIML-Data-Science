"""Generate a single-page executive summary: reports/Executive_Summary.pdf"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import FIGURES_DIR, REPORTS_DIR

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#4C72B0")
LIGHT = colors.HexColor("#EAF0F6")
GREEN = colors.HexColor("#D6E8D5")
PAGE_W = A4[0] - 3 * cm

st = getSampleStyleSheet()
st.add(ParagraphStyle("T", parent=st["Title"], fontSize=20, textColor=NAVY, spaceAfter=2))
st.add(ParagraphStyle("Sub", parent=st["Normal"], fontSize=10, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=6))
st.add(ParagraphStyle("H", parent=st["Heading2"], textColor=BLUE, fontSize=11, spaceBefore=6, spaceAfter=3))
st.add(ParagraphStyle("B", parent=st["Normal"], fontSize=9, leading=12.5, alignment=TA_JUSTIFY))
st.add(ParagraphStyle("KPI", parent=st["Normal"], alignment=TA_CENTER, textColor=colors.white))


def P(t, s="B"):
    return Paragraph(t, st[s])


def kpis(items):
    cells = [Paragraph(f"<b><font size=15>{v}</font></b><br/><font size=8>{l}</font>", st["KPI"])
             for v, l in items]
    t = Table([cells], colWidths=[PAGE_W / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEAFTER", (0, 0), (-2, -1), 1, colors.white)]))
    return t


def tbl(data, highlight=None):
    t = Table(data, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 6)]
    if highlight is not None:
        style += [("BACKGROUND", (0, highlight), (-1, highlight), GREEN),
                  ("FONTNAME", (0, highlight), (-1, highlight), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build():
    metrics = json.loads((REPORTS_DIR / "model_comparison.json").read_text())
    best = max(metrics, key=lambda m: m["f1"])
    pretty = {"cnn": "Baseline CNN", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"}
    order = ["cnn", "resnet50", "efficientnet_b0"]
    metrics.sort(key=lambda m: order.index(m["model"]))

    s = [P("Vehicle Damage Assessment", "T"),
         P("Executive Summary &middot; Deep-Learning Car Damage Classification", "Sub"),
         kpis([(f"{best['accuracy']*100:.1f}%", "Accuracy"), (f"{best['f1']:.3f}", "F1"),
               (f"{best['roc_auc']:.3f}", "ROC-AUC"), (pretty[best["model"]], "Best model")]),
         Spacer(1, 0.25 * cm)]

    s += [P("Objective", "H"),
          P("Automatically classify a car photo as <b>damaged</b> or <b>whole</b> to replace slow, "
            "inconsistent manual insurance inspections with an instant, auditable decision.")]

    s += [P("Approach", "H"),
          P("End-to-end pipeline on 2,300 balanced images: EDA &rarr; data cleaning &rarr; augmentation "
            "&rarr; three models (baseline CNN, ResNet50, EfficientNet-B0 transfer learning) &rarr; "
            "Optuna tuning &rarr; evaluation &rarr; Grad-CAM explainability &rarr; Streamlit app. "
            "Trained on an RTX 5080 (PyTorch, mixed precision).")]

    s += [P("Results (held-out validation, leak-free)", "H")]
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    hl = None
    for i, m in enumerate(metrics, 1):
        if m["model"] == best["model"]:
            hl = i
        rows.append([pretty[m["model"]], f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
                     f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}"])
    s += [tbl(rows, highlight=hl), Spacer(1, 0.2 * cm)]

    s += [P("Key insight", "H"),
          P("EDA caught <b>data leakage</b> &mdash; identical images present in both the training and "
            "validation folders. Removing them keeps the reported metrics honest, a common pitfall "
            "that otherwise inflates accuracy.")]

    # two-column: business value table + ROC figure
    biz = tbl([["", "Manual", "AI"],
               ["Review time", "Days", "Seconds"],
               ["Consistency", "Varies", "Deterministic"],
               ["Fraud signal", "Manual", "Score + Grad-CAM"],
               ["Scale", "Headcount", "1000s/hour"]])
    from PIL import Image as PILImage

    roc = FIGURES_DIR / "roc_curves.png"
    w, h = PILImage.open(roc).size
    iw = PAGE_W * 0.46
    roc_img = Image(str(roc), width=iw, height=iw * h / w)
    two = Table([[biz, roc_img]], colWidths=[PAGE_W * 0.5, PAGE_W * 0.5])
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s += [P("Business value", "H"), two]

    s += [Spacer(1, 0.15 * cm),
          P("<b>Recommendation:</b> deploy ResNet50 as a first-pass triage classifier with adjuster "
            "review on low-confidence cases. Next steps: damage-severity grading and repair-cost "
            "estimation (require additional labels).")]

    doc = SimpleDocTemplate(str(REPORTS_DIR / "Executive_Summary.pdf"), pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                            title="Vehicle Damage Assessment - Executive Summary")
    doc.build(s)
    print(f"Saved -> {REPORTS_DIR / 'Executive_Summary.pdf'}")


if __name__ == "__main__":
    build()
