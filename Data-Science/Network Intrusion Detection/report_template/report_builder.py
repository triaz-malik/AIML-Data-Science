"""Reusable, project-agnostic PDF report builder (same look & feel everywhere).

Drop this single file into any project. Describe the report as a list of "blocks"
(a Python dict or a JSON file) and call build_report(). You get the same branded
format: title page, headings, paragraphs, bullet lists, coloured callout boxes,
tables (with optional highlighted best row), and captioned figures.

USAGE
-----
1) From Python:
       from report_builder import build_report
       build_report(spec_dict, "out.pdf", base_dir="reports/figures")

2) From the command line (spec in a JSON file):
       python report_builder.py spec.json out.pdf [base_dir_for_figures]

SPEC FORMAT
-----------
spec = {
    "title":    "My Project",                 # title-page title
    "subtitle": "One-line subtitle",          # optional
    "highlight":"Best model: X | F1 0.99",    # optional bold line on title page
    "intro":    "One paragraph of context.",  # optional
    "blocks":   [ <block>, <block>, ... ]     # the body, in order
}

Each <block> is a dict with a "type". Supported types:
  {"type":"heading", "level":1, "text":"..."}          # level 1 or 2
  {"type":"paragraph", "text":"...<b>html</b>..."}      # supports <b> <i> tags
  {"type":"bullets", "items":["...", "..."]}
  {"type":"box", "title":"...", "text":"...",
                 "style":"info|warning|danger|success"} # coloured callout
  {"type":"table", "header":["A","B"], "rows":[["1","2"]],
                   "col_widths_cm":[5,6],               # optional
                   "highlight_row":1,                    # optional (1-based body row)
                   "first_col_bold":true}                # optional
  {"type":"figure", "path":"chart.png", "width_cm":14, "caption":"..."}
  {"type":"pagebreak"}
  {"type":"spacer", "height_cm":0.3}
  {"type":"hr"}

Any key beginning with "_" is ignored (use for comments in JSON).
Figure "path" is resolved relative to base_dir if not absolute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, Image, ListFlowable, ListItem,
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

# --- Theme (edit these to re-brand every report at once) ---------------------
BRAND_PRIMARY = "#2980b9"   # accent rule + info boxes
HEADER_BG = "#2c3e50"       # table header background
BAND_BEST = "#d5f5e3"       # highlighted "best" table row
BOX_STYLES = {              # callout box: (background, border)
    "info":    ("#eaf2f8", "#2980b9"),
    "warning": ("#fdebd0", "#e67e22"),
    "danger":  ("#fdecea", "#c0392b"),
    "success": ("#eafaf1", "#27ae60"),
}

_S = getSampleStyleSheet()
TITLE = ParagraphStyle("t_title", parent=_S["Title"], fontSize=22, spaceAfter=6)
H1 = _S["Heading1"]
H2 = _S["Heading2"]
BODY = ParagraphStyle("t_body", parent=_S["BodyText"], fontSize=10, leading=14,
                      spaceAfter=6)
CAPTION = ParagraphStyle("t_cap", parent=BODY, fontSize=8, textColor=colors.grey,
                         spaceBefore=2)


# --- Block renderers ---------------------------------------------------------
def _fig(block, base_dir):
    path = Path(block["path"])
    if not path.is_absolute():
        path = base_dir / path
    out = []
    if not path.exists():
        out.append(Paragraph(f"<i>[missing figure: {path.name}]</i>", BODY))
        return out
    img = Image(str(path))
    width = block.get("width_cm", 15) * cm
    img.drawWidth = width
    img.drawHeight = width * (img.imageHeight / img.imageWidth)
    out.append(img)
    if block.get("caption"):
        out.append(Paragraph(block["caption"], CAPTION))
    return out


def _bullets(block):
    return ListFlowable(
        [ListItem(Paragraph(t, BODY), leftIndent=10) for t in block["items"]],
        bulletType="bullet", start="•", leftIndent=14, spaceAfter=6,
    )


def _box(block):
    bg, border = BOX_STYLES.get(block.get("style", "info"), BOX_STYLES["info"])
    inner = []
    if block.get("title"):
        inner.append(Paragraph(f"<b>{block['title']}</b>", BODY))
    inner.append(Paragraph(block.get("text", ""), BODY))
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


def _table(block):
    rows = [block["header"]] + block["rows"]
    widths = [w * cm for w in block["col_widths_cm"]] if block.get("col_widths_cm") else None
    tbl = Table(rows, hAlign="LEFT", colWidths=widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEADER_BG)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if block.get("first_col_bold", True):
        style.append(("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"))
    hr = block.get("highlight_row")  # 1-based index into body rows
    if hr:
        style += [("BACKGROUND", (0, hr), (-1, hr), colors.HexColor(BAND_BEST)),
                  ("FONTNAME", (0, hr), (-1, hr), "Helvetica-Bold")]
    tbl.setStyle(TableStyle(style))
    return tbl


def _render(block, base_dir):
    """Return a list of flowables for one block."""
    t = block["type"]
    if t == "heading":
        return [Paragraph(block["text"], H1 if block.get("level", 1) == 1 else H2)]
    if t == "paragraph":
        return [Paragraph(block["text"], BODY)]
    if t == "bullets":
        return [_bullets(block)]
    if t == "box":
        return [_box(block)]
    if t == "table":
        return [_table(block)]
    if t == "figure":
        return _fig(block, base_dir)
    if t == "pagebreak":
        return [PageBreak()]
    if t == "spacer":
        return [Spacer(1, block.get("height_cm", 0.3) * cm)]
    if t == "hr":
        return [HRFlowable(width="100%", color=colors.HexColor(BRAND_PRIMARY), thickness=1.0)]
    raise ValueError(f"Unknown block type: {t!r}")


# --- Public API --------------------------------------------------------------
def build_report(spec, out_path, base_dir="."):
    """Build a PDF from a spec dict. base_dir resolves relative figure paths."""
    base_dir = Path(base_dir)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    # Title page
    if spec.get("title"):
        story += [Spacer(1, 3.5 * cm), Paragraph(spec["title"], TITLE)]
        if spec.get("subtitle"):
            story.append(Paragraph(spec["subtitle"], H2))
        story += [Spacer(1, 0.4 * cm),
                  HRFlowable(width="100%", color=colors.HexColor(BRAND_PRIMARY), thickness=1.2),
                  Spacer(1, 0.6 * cm)]
        if spec.get("highlight"):
            story.append(Paragraph(f"<b>{spec['highlight']}</b>", BODY))
        if spec.get("intro"):
            story += [Spacer(1, 0.3 * cm), Paragraph(spec["intro"], BODY)]
        story.append(PageBreak())

    # Body
    for block in spec.get("blocks", []):
        if not isinstance(block, dict) or "type" not in block:
            continue
        story += _render(block, base_dir)

    doc.build(story)
    print(f"[report_builder] wrote {out_path}")
    return out_path


def build_from_json(spec_path, out_path, base_dir="."):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    return build_report(spec, out_path, base_dir)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python report_builder.py <spec.json> <out.pdf> [figures_base_dir]")
        raise SystemExit(1)
    spec_file, out_pdf = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else "."
    build_from_json(spec_file, out_pdf, base)
