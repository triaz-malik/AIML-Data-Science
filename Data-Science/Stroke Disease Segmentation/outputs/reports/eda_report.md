# EDA Report — Stroke Risk Prediction

## Phase 1 — Data Understanding

- **Total patients:** 5,109
- **Stroke cases:** 249 (4.87%) — **severe class imbalance**.
- **Non-stroke:** 4,860 (95.13%).

### How many patients have stroke?
249 of 5109 patients (4.87%). Roughly 1 in 20. Accuracy is a
misleading metric here — a model predicting 'no stroke' for everyone scores ~95%.

### What age groups are most affected?

| Age group | Stroke rate | Stroke cases |
|---|---|---|
| Child | 0.22% | 2 |
| Young | 0.65% | 11 |
| Adult | 5.24% | 80 |
| Senior | 16.17% | 156 |

Stroke risk rises steeply with age — **Seniors (65+) dominate the positive class**.

### Which health factors matter most?

Correlation with the stroke target (numeric view):

| Factor | Correlation with stroke |
|---|---|
| age | +0.245 |
| health_risk_score | +0.225 |
| heart_disease | +0.135 |
| avg_glucose_level | +0.132 |
| hypertension | +0.128 |
| bmi | +0.038 |

**Age** is by far the strongest single signal, followed by heart disease,
hypertension, average glucose level, and the composite health-risk score.

## Phase 2 — Data Cleaning (applied in `data_prep.py`)
- 201 missing BMI values (`N/A`) → median imputation.
- Dropped `id` column and the single `Other`-gender row.
- Removed duplicate rows; standardized categorical text.
- Winsorized BMI and glucose at the 0.5/99.5 percentile.

## Phase 3 — EDA figures
See `outputs/figures/`:
- `01_target_balance.png` — stroke vs non-stroke
- `02-04` — age / BMI / glucose distributions by stroke
- `05-08` — stroke rate by age group / hypertension / heart disease / smoking
- `09_correlation_heatmap.png` — correlation heatmap

### Key takeaways
- **Age strongly influences stroke** — the dominant predictor.
- **Glucose level is significant** — high-glucose patients show elevated stroke rates.
- **Smoking** shows a moderate association (former/current smokers higher than never).
- Hypertension and heart disease roughly **double-to-triple** the stroke rate.