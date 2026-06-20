"""
Phase 11: Power BI data exports.

Reads the scored patient table and produces clean, dashboard-ready CSVs plus a
build guide. Power BI (.pbix) is assembled in Power BI Desktop from these tables
— this script generates the data model that feeds every dashboard page.
"""

import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

from config import PRED_DIR, PBI_DIR, REPORT_DIR, MODEL_DIR, TARGET


def export():
    src = PRED_DIR / "patient_scores.csv"
    df = pd.read_csv(src)

    # 1. Fact table — one row per patient with score + band
    shutil.copy(src, PBI_DIR / "fact_patient_scores.csv")

    # 2. Executive KPI summary
    total = len(df)
    high = int(df["risk_level"].isin(["High", "Critical"]).sum())
    kpi = pd.DataFrame([
        {"metric": "Total Patients", "value": total},
        {"metric": "High-Risk Patients (High+Critical)", "value": high},
        {"metric": "High-Risk Share", "value": round(high / total, 4)},
        {"metric": "Actual Stroke Cases", "value": int(df[TARGET].sum())},
        {"metric": "Stroke Rate", "value": round(df[TARGET].mean(), 4)},
        {"metric": "Mean Predicted Risk", "value": round(df["stroke_probability"].mean(), 4)},
    ])
    kpi.to_csv(PBI_DIR / "kpi_summary.csv", index=False)

    # 3. Risk breakdowns (dimension tables for the Risk Analysis page)
    _breakdown(df, "age_group", PBI_DIR / "risk_by_age.csv",
               order=["Child", "Young", "Adult", "Senior"])
    _breakdown(df, "gender", PBI_DIR / "risk_by_gender.csv")
    _breakdown(df, "Residence_type", PBI_DIR / "risk_by_region.csv")
    _breakdown(df, "smoking_status", PBI_DIR / "risk_by_smoking.csv")

    # 4. Risk-level distribution (for donut/funnel)
    dist = (df.groupby("risk_level")
              .agg(patients=("patient_id", "count"),
                   actual_stroke_rate=(TARGET, "mean"))
              .reindex(["Low", "Medium", "High", "Critical"]).reset_index())
    dist["actual_stroke_rate"] = dist["actual_stroke_rate"].round(4)
    dist.to_csv(PBI_DIR / "risk_distribution.csv", index=False)

    # 5. Model performance — metrics table
    mc = REPORT_DIR / "model_comparison.csv"
    if mc.exists():
        pd.read_csv(mc, index_col=0).to_csv(PBI_DIR / "model_metrics.csv")

    # 6. ROC curve points + confusion matrix for the best model (from OOF scores)
    fpr, tpr, thr = roc_curve(df[TARGET], df["stroke_probability"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        PBI_DIR / "roc_curve_points.csv", index=False)

    best = (MODEL_DIR / "best_model_name.txt").read_text(encoding="utf-8").strip()
    pred = (df["stroke_probability"] >= 0.30).astype(int)  # operating point
    cm = confusion_matrix(df[TARGET], pred)
    cm_df = pd.DataFrame(
        [["Actual No Stroke", cm[0, 0], cm[0, 1]],
         ["Actual Stroke", cm[1, 0], cm[1, 1]]],
        columns=["", "Pred No Stroke", "Pred Stroke"])
    cm_df.to_csv(PBI_DIR / "confusion_matrix.csv", index=False)

    # 7. SHAP importance table
    shap_fig_note = REPORT_DIR / "clinical_recommendation_report.md"  # ref only
    _shap_table()

    _write_guide(best)
    print(f"[powerbi] exports -> {PBI_DIR}")
    for p in sorted(PBI_DIR.glob("*.csv")):
        print("   ", p.name)


def _breakdown(df, col, path, order=None):
    g = (df.groupby(col)
           .agg(patients=("patient_id", "count"),
                high_risk=("risk_level", lambda s: s.isin(["High", "Critical"]).sum()),
                mean_risk=("stroke_probability", "mean"),
                actual_stroke_rate=(TARGET, "mean"))
           .reset_index())
    g["high_risk_share"] = (g["high_risk"] / g["patients"]).round(4)
    g["mean_risk"] = g["mean_risk"].round(4)
    g["actual_stroke_rate"] = g["actual_stroke_rate"].round(4)
    if order:
        g = g.set_index(col).reindex(order).reset_index()
    g.to_csv(path, index=False)


def _shap_table():
    """Re-derive a tidy SHAP importance CSV from the clinical report if present,
    else skip. Kept simple: importance is recomputed in explain_risk; here we
    just pull the figure-backing numbers if a cached csv exists."""
    cached = REPORT_DIR / "shap_importance.csv"
    if cached.exists():
        shutil.copy(cached, PBI_DIR / "shap_importance.csv")


def _write_guide(best):
    lines = [
        "# Power BI Dashboard — Build Guide",
        "",
        "All tables live in `outputs/powerbi/`. In Power BI Desktop:",
        "**Home → Get Data → Text/CSV**, load each file, then build the pages below.",
        "",
        "## Data model",
        "- `fact_patient_scores.csv` — fact table (one row per patient: features,",
        "  `stroke_probability`, `risk_level`, `clinical_action`).",
        "- `kpi_summary.csv`, `risk_distribution.csv`, `risk_by_*.csv` — pre-aggregated",
        "  dimension tables (can also be reproduced as DAX measures on the fact table).",
        "- `model_metrics.csv`, `roc_curve_points.csv`, `confusion_matrix.csv` — model page.",
        "",
        "## Page 1 — Executive Summary",
        "- **Cards:** Total Patients, High-Risk Patients, Stroke Rate, Mean Predicted Risk",
        "  (from `kpi_summary.csv`).",
        "- **Donut:** patients by `risk_level` (`risk_distribution.csv`).",
        "",
        "## Page 2 — Risk Analysis",
        "- **Bar:** high-risk share by age group (`risk_by_age.csv`).",
        "- **Bar:** mean risk by gender (`risk_by_gender.csv`).",
        "- **Map / bar:** risk by region (`risk_by_region.csv`).",
        "- **Slicer:** `risk_level` to cross-filter all visuals.",
        "",
        "## Page 3 — Model Performance",
        f"- **Table:** model metrics (`model_metrics.csv`) — best model: **{best}**.",
        "- **Line:** ROC curve, `tpr` vs `fpr` (`roc_curve_points.csv`); add a y=x reference.",
        "- **Matrix:** confusion matrix (`confusion_matrix.csv`).",
        "- **Image:** SHAP plots from `outputs/figures/12_shap_importance.png` and",
        "  `13_shap_summary.png` (insert as images).",
        "",
        "## Suggested DAX measures (optional)",
        "```",
        "High Risk Patients = CALCULATE(COUNTROWS(fact_patient_scores),",
        "    fact_patient_scores[risk_level] IN {\"High\",\"Critical\"})",
        "Stroke Rate = AVERAGE(fact_patient_scores[stroke])",
        "Avg Predicted Risk = AVERAGE(fact_patient_scores[stroke_probability])",
        "```",
    ]
    (PBI_DIR / "POWERBI_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    export()
