# Employee Attrition Prediction

End-to-end, interview-grade HR analytics project on the **IBM HR Analytics
Employee Attrition** dataset. It predicts which employees are likely to resign
so HR can intervene proactively — and quantifies the result as a multi-million
dollar ROI.

> **Headline:** a tuned, class-balanced **Logistic Regression** reaches
> **ROC-AUC 0.83** with **81% recall** on held-out data, translating to a
> projected **~$4.5M / year** in avoided attrition cost.

---

## The business problem

| | |
|---|---|
| Headcount | 10,000 |
| Annual attrition | 15% → 1,500 people |
| Cost per leaver | \$15,000 (\$5k recruitment + \$3k training + \$7k productivity) |
| **Annual loss** | **\$22.5M** |

**Goal:** flag likely leavers 3–6 months early so HR can act before the
resignation, not after.

---

## What's inside

```
.
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # raw data (1,470 employees, 35 cols)
├── main.py                                 # run the whole pipeline
├── requirements.txt
├── src/
│   ├── config.py            # paths + ALL business assumptions (auditable)
│   ├── data_prep.py         # cleaning + hypothesis-driven feature engineering
│   ├── eda.py               # 8 diagnostic plots + 6 business-question answers
│   ├── modeling.py          # LogReg / Random Forest / XGBoost, tuned + evaluated
│   ├── explain.py           # SHAP global + per-employee explanations
│   ├── business_value.py    # ROI from MEASURED recall
│   └── report.py            # builds the PDF
├── outputs/
│   ├── figures/             # all PNGs
│   ├── models/              # serialized models + preprocessor
│   └── metrics/             # JSON findings (single source of truth for the report)
└── reports/
    └── Employee_Attrition_Report.pdf       # the deliverable
```

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

Runs in ~15s on a laptop and regenerates every figure, metric, model and the
PDF report. Individual stages can also be run standalone:

```bash
python -m src.eda
python -m src.modeling
python -m src.explain
python -m src.report
```

---

## Pipeline at a glance

1. **EDA** — answers 6 HR questions directly from the data (overtime staff leave
   ~3× more; leavers earn ~38% less; churn peaks in years 1–3; Sales is the
   highest-attrition department).
2. **Feature engineering** — every feature encodes a testable HR hypothesis:
   `OverTimeFlag`, `SalaryBand`, `ExperienceGroup`, `PromotionDelayFlag`,
   `EarlyCareerFlag`, `LongCommuteFlag`, `TrainingGapFlag`, `IncomeVsLevelRatio`.
3. **Modeling** — Logistic Regression, Random Forest, XGBoost with class-imbalance
   handling (`class_weight` / `scale_pos_weight`) and 5-fold randomised search.
   Selected on ROC-AUC (threshold-independent), evaluated on precision/recall/F1.
4. **Explainability** — SHAP gives a global driver ranking **and** a per-employee
   waterfall so HR knows *why* each person is flagged.
5. **Business value** — ROI computed from the model's *measured* recall, not a
   guess. Every assumption lives in `src/config.py`.

---

## Model results (held-out 20% test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.74 | 0.31 | **0.81** | 0.45 | **0.83** |
| Random Forest | 0.80 | 0.41 | 0.64 | 0.50 | 0.79 |
| XGBoost | 0.83 | 0.48 | 0.34 | 0.40 | 0.79 |

> **An honest result worth discussing in an interview:** on this small
> (~1,500-row), largely linear dataset, the simplest model won. Complexity is not
> automatically better — model choice should be evidenced, not assumed. We
> optimise for **recall** because a missed leaver costs \$15k while a false alarm
> only costs a retention conversation, so the decision threshold is lowered to
> 0.35.

---

## Top attrition drivers (SHAP)

`OverTime` · `StockOptionLevel` · `Age` · `NumCompaniesWorked` ·
`DistanceFromHome` · `EnvironmentSatisfaction` · `JobSatisfaction` ·
`MonthlyIncome`

---

## Business value

`1,500 leavers × 81% detected × 25% retained = ~303 employees saved/yr`
`× $15,000 = ` **`~$4.5M annual savings`**

All numbers are reproduced by the code; assumptions are centralised and
auditable in [`src/config.py`](src/config.py).

---

## Notes & honest caveats

- The IBM dataset is small (1,470 rows). It's ideal for demonstrating the full
  workflow (EDA → FE → tuning → SHAP → ROI) but a larger HR dataset would give
  more stable estimates — see ideas below.
- The ROI uses planning assumptions (25% of flagged employees successfully
  retained) that should be validated with an A/B retention pilot in production.

### To make it even stronger
Add features like *manager-change count*, *absenteeism*, *overtime hours*, and
*training gap*, and validate on a larger HR dataset that includes hiring,
performance and termination history.
