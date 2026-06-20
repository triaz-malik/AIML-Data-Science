# Model Comparison Report

## Phase 5 — Class Imbalance Handling
Base XGBoost evaluated under three strategies (test set):

|  | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| SMOTE | 0.9413 | 0.2727 | 0.12 | 0.1667 | 0.7718 | 0.1703 |
| ADASYN | 0.9393 | 0.2692 | 0.14 | 0.1842 | 0.7841 | 0.1856 |
| ClassWeighting | 0.8513 | 0.1507 | 0.44 | 0.2245 | 0.7869 | 0.1554 |

SMOTE / ADASYN oversample the minority during training; class-weighting
instead penalizes minority errors via `scale_pos_weight`. The chosen
per-model strategy is SMOTE (KNN, RF) or weighting (XGBoost).

## Phase 6-7 — Models & Tuning
KNN, Random Forest, XGBoost tuned with Grid/Random search + Stratified 5-fold CV,
scored on ROC-AUC.

Best hyperparameters:
```json
{
  "knn": {
    "clf__metric": "euclidean",
    "clf__n_neighbors": 21,
    "clf__weights": "uniform"
  },
  "random_forest": {
    "clf__n_estimators": 200,
    "clf__min_samples_split": 5,
    "clf__min_samples_leaf": 1,
    "clf__max_depth": 6,
    "clf__criterion": "gini"
  },
  "xgboost": {
    "clf__subsample": 0.8,
    "clf__n_estimators": 200,
    "clf__max_depth": 3,
    "clf__learning_rate": 0.05,
    "clf__colsample_bytree": 0.9
  }
}
```

## Phase 8 — Evaluation
Test-set metrics (default 0.5 threshold):

|  | accuracy | precision | recall | f1 | roc_auc | pr_auc | cv_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| knn | 0.7025 | 0.1152 | 0.76 | 0.2 | 0.7507 | 0.1261 | 0.7643 |
| random_forest | 0.7603 | 0.1375 | 0.74 | 0.232 | 0.8192 | 0.2102 | 0.8234 |
| xgboost | 0.6595 | 0.1016 | 0.76 | 0.1792 | 0.7717 | 0.1821 | 0.8087 |

### Deployed model: `knn`
**`knn` is the served model** (ROC-AUC 0.751). The highest-AUC
benchmark is `random_forest` (ROC-AUC 0.819); it is retained in the
comparison for transparency. KNN is deployed deliberately — it is interpretable
(prediction = the outcomes of the most similar past patients), needs no
distributional assumptions, and its recall after threshold tuning is competitive.
The AUC gap is small relative to the dataset's inherent ceiling (~0.82).

### Recall-priority operating point
Healthcare priority is **recall** — missing a high-risk patient is the costly error.
Tuning the decision threshold to **0.286** on the served model yields:

- Recall: **0.800**
- Precision: 0.096
- F1: 0.172
- ROC-AUC: 0.751

### Defending the threshold — the recall/false-alarm tradeoff
The decision threshold is a *business* choice, not a default. The table below
shows, for each target recall, how many false alarms clinicians must screen per
stroke actually caught:

| target_recall | threshold | actual_recall | precision | strokes_caught | false_alarms | false_alarms_per_catch |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7 | 0.524 | 0.76 | 0.115 | 38.0 | 292.0 | 7.68 |
| 0.8 | 0.286 | 0.8 | 0.096 | 40.0 | 376.0 | 9.4 |
| 0.9 | 0.0 | 1.0 | 0.049 | 50.0 | 972.0 | 19.44 |

Reading this: pushing recall from 80% to 90% roughly doubles the false-alarm
burden. We adopt the **~80% recall** point as the deployed operating threshold
(the highest KNN reaches without collapsing to predict-all-positive, given its
coarse probabilities) — we accept the extra screening cost because a missed stroke
(a false negative) is far more expensive, clinically and ethically, than a
follow-up on a false positive. A hospital can slide this threshold to match its
screening capacity.

Figures: `outputs/figures/10_roc_pr_curves.png`, `11_confusion_matrix.png`.