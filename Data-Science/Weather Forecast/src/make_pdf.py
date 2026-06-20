"""Render outputs/reports/final_report.md to a polished PDF (reportlab).

Handles the subset of Markdown the report uses: H1/H2/H3 headings, paragraphs,
**bold** / `code` inline, bullet lists, GitHub-style tables, image embeds
(![alt](path)), italic captions (*text*) and horizontal rules. No external
binaries or network required.
"""
from __future__ import annotations

import os
import re
import html

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem,
)
from PIL import Image as PILImage

import config as C

REPORT_MD = f"{C.REPORT_DIR}/final_report.md"
OUT_PDF = f"{C.REPORT_DIR}/final_report.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1x", parent=ss["Heading1"], fontSize=20,
                          spaceBefore=6, spaceAfter=10, textColor=colors.HexColor("#1a2433")))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], fontSize=14,
                          spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#c0392b")))
    ss.add(ParagraphStyle("H3x", parent=ss["Heading3"], fontSize=11.5,
                          spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2c3e50")))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5,
                          leading=14, spaceAfter=6))
    ss.add(ParagraphStyle("Cap", parent=ss["BodyText"], fontSize=8,
                          leading=10, textColor=colors.grey, alignment=1,
                          spaceAfter=10))
    ss.add(ParagraphStyle("BulletX", parent=ss["BodyText"], fontSize=9.5,
                          leading=13, leftIndent=10))
    ss.add(ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8, leading=10))
    ss.add(ParagraphStyle("CellH", parent=ss["BodyText"], fontSize=8,
                          leading=10, textColor=colors.white))
    return ss


def inline(text: str) -> str:
    """Convert a subset of inline Markdown to reportlab mini-HTML."""
    text = html.escape(text)
    # `code`
    text = re.sub(r"`([^`]+)`",
                  r'<font face="Courier" size="8.5">\1</font>', text)
    # **bold**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def make_table(rows, ss):
    # rows: list of list[str] (markdown cells already split)
    header = [Paragraph(inline(c.strip()), ss["CellH"]) for c in rows[0]]
    body = [[Paragraph(inline(c.strip()), ss["Cell"]) for c in r] for r in rows[1:]]
    data = [header] + body
    ncols = len(rows[0])
    tbl = Table(data, colWidths=[CONTENT_W / ncols] * ncols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dbdb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def scaled_image(path: str):
    if not os.path.exists(path):
        return None
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = h / w
    draw_w = min(CONTENT_W, 16 * cm)
    draw_h = draw_w * ratio
    max_h = 9.5 * cm
    if draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h / ratio
    return Image(path, width=draw_w, height=draw_h)


def parse(md: str, ss):
    flow = []
    lines = md.splitlines()
    i = 0
    para_buf = []

    def flush_para():
        if para_buf:
            flow.append(Paragraph(inline(" ".join(para_buf).strip()), ss["Body"]))
            para_buf.clear()

    bullets = []

    def flush_bullets():
        if bullets:
            items = [ListItem(Paragraph(inline(b), ss["BulletX"]), leftIndent=12)
                     for b in bullets]
            flow.append(ListFlowable(items, bulletType="bullet", start="•",
                                     leftIndent=10))
            flow.append(Spacer(1, 4))
            bullets.clear()

    while i < len(lines):
        ln = lines[i].rstrip()

        # Table block
        if ln.strip().startswith("|") and "|" in ln:
            flush_para(); flush_bullets()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for r in block:
                if re.match(r"^\|[\s:|-]+\|?$", r):  # separator row
                    continue
                cells = [c for c in r.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                flow.append(make_table(rows, ss))
                flow.append(Spacer(1, 8))
            continue

        # Image
        m = re.match(r"!\[(.*?)\]\((.*?)\)", ln.strip())
        if m:
            flush_para(); flush_bullets()
            rel = m.group(2)
            abspath = os.path.normpath(os.path.join(C.REPORT_DIR, rel))
            img = scaled_image(abspath)
            if img:
                flow.append(img)
            i += 1
            continue

        if not ln.strip():
            flush_para(); flush_bullets()
            i += 1
            continue

        if ln.startswith("### "):
            flush_para(); flush_bullets()
            flow.append(Paragraph(inline(ln[4:]), ss["H3x"]))
        elif ln.startswith("## "):
            flush_para(); flush_bullets()
            flow.append(Paragraph(inline(ln[3:]), ss["H2x"]))
        elif ln.startswith("# "):
            flush_para(); flush_bullets()
            flow.append(Paragraph(inline(ln[2:]), ss["H1x"]))
        elif ln.strip() == "---":
            flush_para(); flush_bullets()
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.HexColor("#bbbbbb")))
            flow.append(Spacer(1, 4))
        elif ln.lstrip().startswith("- "):
            flush_para()
            bullets.append(ln.lstrip()[2:])
        elif ln.startswith("*") and ln.endswith("*") and not ln.startswith("**"):
            flush_para(); flush_bullets()
            flow.append(Paragraph(inline(ln.strip("*")), ss["Cap"]))
        else:
            flush_bullets()
            para_buf.append(ln.strip())
        i += 1

    flush_para(); flush_bullets()
    return flow


def main():
    with open(REPORT_MD, encoding="utf-8") as f:
        md = f.read()
    ss = styles()
    flow = parse(md, ss)
    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Multi-Step Weather Forecasting — Final Report",
        author="Weather Forecasting Project",
    )
    doc.build(flow)
    size_kb = os.path.getsize(OUT_PDF) / 1024
    print(f"[PDF] Written -> {OUT_PDF}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
