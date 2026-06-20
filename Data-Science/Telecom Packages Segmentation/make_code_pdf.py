"""Generate a PDF containing the full project source code."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Telecom_Segmentation_Code.pdf"

# Files to include, in logical reading order
FILES = [
    "README.md",
    "requirements.txt",
    "src/config.py",
    "src/utils.py",
    "src/phase1_eda.py",
    "src/phase2_features.py",
    "src/phase3_segmentation.py",
    "src/phase4_recommendation.py",
    "src/phase5_churn.py",
    "src/phase6_explainability.py",
    "src/run_all.py",
    "make_code_pdf.py",
]

WRAP = 100  # soft-wrap long lines so nothing overflows the page


def softwrap(line: str, width: int = WRAP) -> list[str]:
    if len(line) <= width:
        return [line]
    out, cur = [], line
    while len(cur) > width:
        out.append(cur[:width])
        cur = "    " + cur[width:]  # indent continuation
    out.append(cur)
    return out


def main() -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=22, leading=26)
    sub_style = ParagraphStyle(
        "sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=12,
        textColor=colors.HexColor("#555555"), spaceBefore=6)
    file_style = ParagraphStyle(
        "file", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#1a4d7a"), spaceBefore=4, spaceAfter=6)
    code_style = ParagraphStyle(
        "code", parent=styles["Code"], fontName="Courier", fontSize=7.2,
        leading=8.6, textColor=colors.HexColor("#1a1a1a"))
    toc_style = ParagraphStyle(
        "toc", parent=styles["Normal"], fontName="Courier", fontSize=9,
        leading=14)

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.7 * inch, bottomMargin=0.6 * inch,
        title="Telecom Customer Segmentation - Source Code",
        author="triaz.malik")

    story = []

    # ---- title page ----
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph(
        "Telecom Customer Segmentation<br/>&amp; Package Recommendation System",
        title_style))
    story.append(Paragraph("Complete Source Code", sub_style))
    story.append(Paragraph(
        "EDA &bull; Feature Engineering &bull; K-Means / DBSCAN Segmentation "
        "&bull; KNN Recommender &bull; Churn Prediction (KNN / RF / XGBoost) "
        "&bull; SHAP Explainability", sub_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"{len(FILES)} files", sub_style))

    # ---- table of contents ----
    story.append(PageBreak())
    story.append(Paragraph("Contents", file_style))
    story.append(Spacer(1, 6))
    for i, rel in enumerate(FILES, 1):
        story.append(Paragraph(f"{i:>2}.  {rel}", toc_style))

    # ---- each file ----
    for rel in FILES:
        path = ROOT / rel
        story.append(PageBreak())
        story.append(Paragraph(rel, file_style))
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            story.append(Paragraph(f"[could not read: {exc}]", styles["Normal"]))
            continue
        numbered = []
        for n, raw in enumerate(text.splitlines(), 1):
            for j, piece in enumerate(softwrap(raw.rstrip("\n"))):
                prefix = f"{n:>4} | " if j == 0 else "     | "
                numbered.append(prefix + piece)
        story.append(Preformatted("\n".join(numbered), code_style))

    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
