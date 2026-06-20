# Reusable PDF Report Template

A drop-in, project-agnostic PDF report generator that produces the **same branded
format** every time: title page, headings, paragraphs, bullet lists, coloured
callout boxes, tables (with an optional highlighted "best" row), and captioned
figures.

Use it on any project — you only write a small **spec** (content + figure paths);
the formatting is handled for you.

## Files
| File | Purpose |
|------|---------|
| `report_builder.py` | The engine. Copy this into any project. Don't edit per-project. |
| `report_spec_template.json` | A ready-to-fill skeleton. Copy → rename → edit. |
| `example_report.pdf` | A rendered sample (generated from the template). |

## Requirements
```bash
pip install reportlab
```

## Quick start (3 ways)

**A) Command line + JSON spec**
```bash
python report_builder.py my_spec.json My_Report.pdf path/to/figures
```
(The 3rd argument is the folder that relative figure paths resolve against.)

**B) From Python with a JSON spec**
```python
from report_builder import build_from_json
build_from_json("my_spec.json", "My_Report.pdf", base_dir="reports/figures")
```

**C) From Python with a dict (build the spec programmatically)**
```python
from report_builder import build_report

spec = {
    "title": "Churn Prediction",
    "subtitle": "ML on customer data",
    "highlight": "Best model: XGBoost | F1 0.91",
    "intro": "Predicts which customers will churn.",
    "blocks": [
        {"type": "heading", "level": 1, "text": "1. Results"},
        {"type": "table",
         "header": ["Model", "F1"],
         "rows": [["XGBoost", "0.91"], ["LogReg", "0.83"]],
         "highlight_row": 1},
        {"type": "figure", "path": "roc.png", "width_cm": 14, "caption": "ROC curves"},
    ],
}
build_report(spec, "My_Report.pdf", base_dir="reports/figures")
```

## Spec format

Top-level keys: `title`, `subtitle`, `highlight`, `intro` (all optional but
recommended), and `blocks` (the body, rendered in order).

Each block is a dict with a `type`:

| `type` | Required keys | Optional keys |
|--------|---------------|---------------|
| `heading` | `text` | `level` (1 or 2; default 1) |
| `paragraph` | `text` | — |
| `bullets` | `items` (list) | — |
| `box` | `text` | `title`, `style` = `info`/`warning`/`danger`/`success` |
| `table` | `header`, `rows` | `col_widths_cm`, `highlight_row` (1-based), `first_col_bold` |
| `figure` | `path` | `width_cm` (default 15), `caption` |
| `pagebreak` | — | — |
| `spacer` | — | `height_cm` (default 0.3) |
| `hr` | — | — |

Notes:
- Text supports `<b>bold</b>` and `<i>italic</i>`.
- Any key starting with `_` is ignored — use it for comments inside JSON.
- A missing figure renders a visible `[missing figure: name]` placeholder instead
  of crashing.
- `highlight_row` shades that body row green and bolds it — use it for the best model.

## Re-branding (colours / fonts) in one place
Open `report_builder.py` and edit the theme block near the top:
```python
BRAND_PRIMARY = "#2980b9"   # accent rule + info boxes
HEADER_BG     = "#2c3e50"   # table header background
BAND_BEST     = "#d5f5e3"   # highlighted best row
BOX_STYLES    = {...}        # callout colours
```
Every report built afterwards picks up the new look automatically.

## Recommended section flow (the "house style")
0. What the project is → 1. Data → 2. Problems & solutions →
3. What was done → 4. EDA → 5. Results (+ what metrics mean) →
6. Evaluation plots → 7. Explainability → 8. Outputs → 9. Caveat →
10. Business impact.

`report_spec_template.json` already follows this flow — start there.
