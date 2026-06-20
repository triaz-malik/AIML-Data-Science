# Telecom Customer Segmentation & Package Recommendation System

End-to-end data-science project on the **Orange / BigML Telecom Churn** dataset
(3,333 subscribers, provided as an 80/20 split). It combines **EDA, feature
engineering, unsupervised segmentation, a KNN recommendation engine, churn
classification, and explainable AI (SHAP)** into a single reproducible pipeline.

> Why this is stronger than a standard churn project: it pairs **classification**
> with **clustering**, a **recommendation system**, telecom **domain features**,
> **explainable AI**, and a **Power BI-ready** scored dataset.

---

## Dataset

Orange Telecom Churn dataset (Kaggle / BigML). Key fields: account length,
day/evening/night/international minutes·calls·charges, voicemail & international
plan flags, customer-service calls, and the `Churn` target.

| File | Rows | Role |
|------|------|------|
| `churn-bigml-80.csv` | 2,666 | training split |
| `churn-bigml-20.csv` | 667 | holdout test split |

Overall churn rate: **14.5%** (imbalanced).

---

## Quick start

```bash
pip install -r requirements.txt
python src/run_all.py          # runs Phase 1 -> 7, ~30s
```

Run a single phase, e.g.:

```bash
python src/phase3_segmentation.py
```

All artifacts are written under `outputs/` (`figures/`, `data/`, `models/`, `reports/`).

---

## Pipeline

### Phase 1 — EDA  (`phase1_eda.py`)
Churn / day-minutes / international / charges / service-call distributions,
correlation heatmap, and churned-vs-retained boxplots.

**Key findings**
- Heavy users (top 10% by minutes) churn **47%** vs 14.5% overall.
- Top payers (top 10% by charges) churn **63%** — high-value customers are the most at-risk.
- Customers with **≥4 service calls** churn **52%**.
- International-plan holders churn **42%** vs 11.5% without.

### Phase 2 — Feature Engineering  (`phase2_features.py`)
`Total Usage Minutes`, `Total Charges`, `International Usage Ratio`,
`Customer Value Score` (0–100 weighted blend of spend, tenure, engagement),
plus `Revenue Segment` (Low/Med/High) and `Usage Segment` (Light/Med/Heavy) tertiles.

### Phase 3 — Customer Segmentation  (`phase3_segmentation.py`)
K-Means (k chosen for business granularity; silhouette + elbow plotted) with a
DBSCAN comparison. Clusters are mapped to **5 business segments** via a greedy,
distinctiveness-based labeller:

| Segment | Customers | Churn | Avg charges | Profile |
|---------|-----------|-------|-------------|---------|
| **High Risk Users** | 728 | **32%** | $72.80 | high-spend, high-churn → retention priority |
| **Premium Users** | 836 | 9% | $60.10 | high value, stable |
| **Voice Heavy Users** | 509 | 10% | $60.27 | heavy voicemail usage |
| **International Users** | 568 | 13% | $44.97 | high intl-usage ratio |
| **Low Revenue Users** | 692 | 7% | $55.91 | low value, low churn |

### Phase 4 — KNN Recommendation Engine  (`phase4_recommendation.py`)
A rule-based catalogue (Basic Saver, Day Talker Pro, Evening & Night Saver,
Global Connect, Voicemail Plus, Premium Unlimited) assigns each customer a
*current* package. `NearestNeighbors` then finds each subscriber's **5 most
similar customers** (usage, charges, intl calls, account length, service calls)
and recommends the dominant neighbour package:

> *"Customers similar to you are using Package X."*

**~40% of subscribers** are flagged with a cross-sell / upsell opportunity.

### Phase 5 — Churn Prediction  (`phase5_churn.py`)
Three models — **KNN, Random Forest, XGBoost** — each in an imbalanced-learn
pipeline with **SMOTE**, tuned with **GridSearchCV** over **Stratified 5-Fold**
CV, evaluated on the holdout split.

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|-------|-----------|--------------|---------|
| KNN | 0.880 | 0.904 | 0.560 |
| Random Forest | 0.915 | 0.920 | 0.876 |
| **XGBoost** ⭐ | **0.925** | **0.924** | 0.843 |

Best model (XGBoost) on the holdout: **churn recall 0.87**, precision 0.81,
overall accuracy 0.95. Saved to `outputs/models/best_churn_model.joblib`.

### Phase 6 — Explainability  (`phase6_explainability.py`)
SHAP `TreeExplainer` on the tuned model — beeswarm summary + global importance.
Top churn drivers: **Total Charges (revenue), Voicemail plan, Voicemail
messages, International calls, Customer service calls, International plan** —
matching domain expectations.

### Phase 7 — Business Value
The operator can **identify high-value customers**, **recommend better
packages**, **reduce churn** (per-subscriber probability + risk band),
**increase ARPU**, and **target retention campaigns** at the High-Risk +
High-Value overlap.

---

## Outputs

| Path | Description |
|------|-------------|
| `outputs/figures/01..15_*.png` | All EDA, segmentation, model & SHAP plots |
| `outputs/data/telecom_features.csv` | Engineered features (Phase 2) |
| `outputs/data/telecom_segmented.csv` | + segment & package labels (Phase 3–4) |
| `outputs/data/recommendations.csv` | Per-customer package recommendations |
| `outputs/data/telecom_scored.csv` | **Power BI-ready**: features + segments + churn probability + risk band |
| `outputs/models/*.joblib` | Saved recommender & best churn model |
| `outputs/reports/*` | Insight & metric summaries |

### Power BI dashboard
Load `outputs/data/telecom_scored.csv`. Suggested visuals: churn rate by
`Segment`, `Churn Risk Band` distribution, ARPU (`Total Charges`) by segment,
`Current` vs `Recommended Package` flow, and a high-risk/high-value target list.

---

## Project structure

```
.
├── churn-bigml-80.csv / churn-bigml-20.csv   # raw data
├── requirements.txt
├── README.md
├── src/
│   ├── config.py            # paths, feature groups, constants
│   ├── utils.py             # loading, cleaning, figure helpers
│   ├── phase1_eda.py
│   ├── phase2_features.py
│   ├── phase3_segmentation.py
│   ├── phase4_recommendation.py
│   ├── phase5_churn.py
│   ├── phase6_explainability.py
│   └── run_all.py           # orchestrator (Phase 1 -> 7)
└── outputs/                 # generated: figures / data / models / reports
```
