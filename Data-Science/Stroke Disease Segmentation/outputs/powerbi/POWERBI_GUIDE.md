# Power BI Dashboard — Build Guide

All tables live in `outputs/powerbi/`. In Power BI Desktop:
**Home → Get Data → Text/CSV**, load each file, then build the pages below.

## Data model
- `fact_patient_scores.csv` — fact table (one row per patient: features,
  `stroke_probability`, `risk_level`, `clinical_action`).
- `kpi_summary.csv`, `risk_distribution.csv`, `risk_by_*.csv` — pre-aggregated
  dimension tables (can also be reproduced as DAX measures on the fact table).
- `model_metrics.csv`, `roc_curve_points.csv`, `confusion_matrix.csv` — model page.

## Page 1 — Executive Summary
- **Cards:** Total Patients, High-Risk Patients, Stroke Rate, Mean Predicted Risk
  (from `kpi_summary.csv`).
- **Donut:** patients by `risk_level` (`risk_distribution.csv`).

## Page 2 — Risk Analysis
- **Bar:** high-risk share by age group (`risk_by_age.csv`).
- **Bar:** mean risk by gender (`risk_by_gender.csv`).
- **Map / bar:** risk by region (`risk_by_region.csv`).
- **Slicer:** `risk_level` to cross-filter all visuals.

## Page 3 — Model Performance
- **Table:** model metrics (`model_metrics.csv`) — best model: **knn**.
- **Line:** ROC curve, `tpr` vs `fpr` (`roc_curve_points.csv`); add a y=x reference.
- **Matrix:** confusion matrix (`confusion_matrix.csv`).
- **Image:** SHAP plots from `outputs/figures/12_shap_importance.png` and
  `13_shap_summary.png` (insert as images).

## Suggested DAX measures (optional)
```
High Risk Patients = CALCULATE(COUNTROWS(fact_patient_scores),
    fact_patient_scores[risk_level] IN {"High","Critical"})
Stroke Rate = AVERAGE(fact_patient_scores[stroke])
Avg Predicted Risk = AVERAGE(fact_patient_scores[stroke_probability])
```