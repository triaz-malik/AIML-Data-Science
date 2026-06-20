# AI-Powered Healthcare Risk Prediction System

Predict whether a patient is at high risk of **stroke** and identify the key
factors driving that risk — turning an inconsistent, manual triage process into
a reproducible, explainable, and actionable ML pipeline.

> Dataset: Kaggle *Healthcare Stroke Prediction* (`healthcare-dataset-stroke-data.csv`,
> 5,109 patients after cleaning, ~4.9% stroke prevalence — **severely imbalanced**).

## Business problem
Hospitals triage patients with limited resources and inconsistent manual risk
assessment. This system predicts per-patient stroke risk, stratifies patients
into clinical-action bands, and explains every prediction so clinicians can trust
and act on it.

## Pipeline (11 phases)

| Phase | What it does | Code |
|---|---|---|
| 1 Data Understanding | size, balance, age effects, key factors | `src/eda.py` |
| 2 Data Cleaning | missing BMI, duplicates, outlier winsorizing, standardization | `src/data_prep.py` |
| 3 EDA | distribution / relationship / correlation plots | `src/eda.py` |
| 4 Feature Engineering | age groups, BMI & glucose categories, composite health-risk score | `src/data_prep.py` |
| 5 Class Imbalance | SMOTE vs ADASYN vs class-weighting comparison | `src/modeling.py` |
| 6 Modeling | KNN, Random Forest, XGBoost | `src/modeling.py` |
| 7 Hyperparameter Tuning | Grid / Random search + Stratified K-Fold | `src/modeling.py` |
| 8 Evaluation | Accuracy/Precision/Recall/F1/ROC-AUC (**recall-priority**) | `src/modeling.py` |
| 9 Explainable AI | SHAP global + local explanations | `src/explain_risk.py` |
| 10 Risk Stratification | Low / Medium / High / Critical + clinical action | `src/explain_risk.py` |
| 11 Dashboard | Power BI-ready exports + build guide | `src/powerbi_export.py` |

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # runs every phase, writes ./outputs/
```

Run an individual phase:
```bash
cd src
python eda.py
python modeling.py
python explain_risk.py
python powerbi_export.py
```

## Outputs (`outputs/`)
- `figures/` — 13 plots (EDA, ROC/PR, confusion matrix, SHAP).
- `models/` — tuned `knn/random_forest/xgboost.joblib`, `best_model.joblib`,
  `training_meta.json` (best params, thresholds, metrics).
- `reports/` — `eda_report.md`, `model_comparison_report.md`,
  `clinical_recommendation_report.md` + supporting CSVs.
- `predictions/patient_scores.csv` — every patient scored (leakage-free
  out-of-fold probability) with risk band + clinical action.
- `powerbi/` — dashboard fact/dimension tables + `POWERBI_GUIDE.md`.

## Key design decisions
- **KNN is the deployed model**, with Random Forest and XGBoost kept as honest
  benchmarks in the comparison table. KNN is interpretable (a prediction is the
  outcome of the most similar past patients) and its recall after threshold
  tuning is competitive; RF has a modestly higher ROC-AUC (0.82 vs 0.75), and
  the report states this openly. Set `SERVED_MODEL` in `src/config.py` to switch.
- **Recall is prioritized.** Missing a stroke costs far more than a false alarm,
  so the served model ships with a recall-oriented decision threshold (~0.85 recall),
  not the default 0.5. Accuracy is deliberately *not* the headline metric — a
  trivial "always no-stroke" model scores ~95%. `operating_points.csv` quantifies
  the recall/false-alarm tradeoff so the threshold is a defensible business choice.
- **No leakage.** SMOTE runs *inside* cross-validation folds; patient-level risk
  scores use out-of-fold predictions.
- **Tuning on ROC-AUC, operating on recall.** Tuning directly on recall is
  degenerate; we tune on ROC-AUC then pick the operating threshold for recall.
- **Explainability is first-class.** Model-agnostic SHAP (KernelExplainer for KNN)
  gives global importance, a beeswarm, and per-patient *waterfall* explanations
  (`14_shap_local_*.png`) — confirming the model reasons from established risk
  factors (age, glucose, BMI, health-risk score, hypertension, heart disease).

> **Note on KNN's probabilities:** with k=21 neighbors, KNN emits coarse,
> stepped probabilities (~0.048 increments). This makes its risk bands blockier
> than a tree model's and is why the 90%-recall operating point jumps to ~100%.
> It's a genuine tradeoff of choosing KNN — documented rather than hidden.

## Risk stratification → clinical action
| Risk level | Predicted prob. | Action |
|---|---|---|
| Low | 0–10% | Routine monitoring |
| Medium | 10–30% | Follow-up |
| High | 30–60% | Specialist review |
| Critical | 60%+ | Immediate attention |

## Project layout
```
.
├── healthcare-dataset-stroke-data.csv
├── requirements.txt
├── run_all.py                 # orchestrator
├── src/
│   ├── config.py              # paths, constants, risk bands
│   ├── data_prep.py           # clean + feature engineering
│   ├── eda.py                 # Phase 1-3
│   ├── modeling.py            # Phase 5-8
│   ├── explain_risk.py        # Phase 9-10
│   └── powerbi_export.py      # Phase 11
└── outputs/                   # generated artifacts
```
