"""Phase 7-8: defect severity score and quality decision engine.

Severity is derived from the **defect area as a percentage of the sheet**:

    Minor    0-2%   -> Accept
    Moderate 2-5%   -> Rework
    Critical 5%+    -> Reject

Thresholds and the decision map live in ``src/config.py`` so they are tunable in
one place. The area can come from either ground-truth RLE (analysis / KPIs) or a
predicted segmentation mask (production), via ``area_pct`` injection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, utils
from src.dataset import build_image_df, load_annotations

SHEET_PX = config.IMG_HEIGHT * config.IMG_WIDTH


def rle_area_px(rle) -> int:
    """Pixel count of a single RLE string (sum of run lengths)."""
    if pd.isna(rle) or not isinstance(rle, str):
        return 0
    return int(np.asarray(rle.split(), dtype=int)[1::2].sum())


def classify_severity(area_pct: float, has_defect: bool) -> str:
    """Severity label for a sheet given its total defect-area percentage."""
    if not has_defect or area_pct <= 0:
        return "None"
    return utils.severity_from_area(area_pct)


def decide(severity: str) -> str:
    """Quality decision for a severity label."""
    if severity == "None":
        return "Accept"
    return utils.decision_from_severity(severity)


def build_quality_table(ann: pd.DataFrame | None = None,
                        img_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-image table: defect area %, severity, and Accept/Rework/Reject decision.

    Uses ground-truth RLE areas. Defect classes rarely overlap, so summing per-class
    pixel areas is a close approximation of the union area (capped at 100%).
    """
    if ann is None:
        ann = load_annotations()
    if img_df is None:
        img_df = build_image_df(ann)

    areas = ann.assign(area_px=ann["EncodedPixels"].map(rle_area_px)) \
               .groupby("ImageId")["area_px"].sum()

    df = img_df[["ImageId", "has_defect", "defects", "num_defects"]].copy()
    df["defect_area_px"] = df["ImageId"].map(areas).fillna(0).astype(int)
    df["defect_area_pct"] = (df["defect_area_px"] / SHEET_PX * 100).clip(upper=100).round(4)
    df["severity"] = [classify_severity(p, bool(h))
                      for p, h in zip(df["defect_area_pct"], df["has_defect"])]
    df["decision"] = df["severity"].map(decide)
    return df


def kpi_summary(qt: pd.DataFrame) -> dict:
    """Headline KPIs for the dashboard."""
    n = len(qt)
    defective = int((qt["has_defect"] == 1).sum())
    rejected = int((qt["decision"] == "Reject").sum())
    rework = int((qt["decision"] == "Rework").sum())
    return {
        "total_sheets": n,
        "defective_sheets": defective,
        "defect_pct": round(100 * defective / n, 2),
        "accepted": int((qt["decision"] == "Accept").sum()),
        "rework": rework,
        "rejected": rejected,
        "reject_pct": round(100 * rejected / n, 2),
        "critical_sheets": int((qt["severity"] == "Critical").sum()),
        "mean_defect_area_pct_defective": round(
            qt.loc[qt.has_defect == 1, "defect_area_pct"].mean(), 3),
    }


if __name__ == "__main__":
    config.ensure_dirs()
    qt = build_quality_table()
    out = config.REPORTS_DIR / "quality_decisions.csv"
    qt.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(qt):,} rows)")
    print("\nSeverity distribution:")
    print(qt["severity"].value_counts())
    print("\nDecision distribution:")
    print(qt["decision"].value_counts())
    print("\nKPIs:")
    for k, v in kpi_summary(qt).items():
        print(f"  {k:35s} {v}")
