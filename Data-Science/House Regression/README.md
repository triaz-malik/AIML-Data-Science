# House Prices: Advanced Regression Techniques

End-to-end pipeline for the [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. EDA → preprocessing → feature engineering (incl. KMeans neighborhood clusters) → 7 models → Optuna-tuned XGBoost/LightGBM/CatBoost → Ridge meta-stacker → SHAP interpretability → Streamlit estimator.

**OOF RMSLE: ~0.108** with Ridge stacking over 7 base models.

## Quick start

```bash
pip install -r requirements.txt

# Train (full pipeline, ~10–15 min with default 30 Optuna trials per booster)
python house_prices.py

# Spec-faithful 100 trials (slower, slight gain)
python house_prices.py --trials 100

# Skip Optuna for fast iteration
python house_prices.py --quick

# Launch the price estimator UI
streamlit run streamlit_app.py
```

## Pipeline overview

| Stage | What happens |
|---|---|
| **EDA** | Target skew + log transform, missing patterns, correlations, neighborhood premium |
| **Preprocess** | Drop documented `GrLivArea` outliers; `NaN` → `"None"` for absent features, `0` for absent areas, neighborhood-median for `LotFrontage`, mode for the rest |
| **Feature engineering** | `TotalSF`, `TotalBathrooms`, `HouseAge`, presence flags, `QualArea`, neighborhood **target encoding** + **KMeans clusters** (5 groups on aggregated stats), ordinal encoding, log1p on skewed features, one-hot |
| **Baseline models** | Ridge, Lasso, ElasticNet, sklearn GBM (huber loss) |
| **Tuned boosters** | XGBoost, LightGBM, CatBoost — Optuna 30/100 trials each, optimizing 5-fold RMSLE on `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `reg_alpha`, `reg_lambda`, etc. |
| **Stacking** | **Ridge meta-learner** (RidgeCV) fit on 5-fold OOF predictions of all 7 base models — true stacking, not blending |
| **Interpretability** | SHAP TreeExplainer summary + top-20 by mean |SHAP\| magnitude |
| **Production** | Trained artifacts saved to `artifacts/model.pkl`; Streamlit form (`streamlit_app.py`) for interactive price estimates |

## Why log-transform `SalePrice`?

The raw target is right-skewed (skew ≈ 1.88) — a few luxury homes dominate. The competition is scored on **RMSLE**, which is mathematically equivalent to RMSE on `log(SalePrice)`. So predicting `log1p(SalePrice)` directly aligns the loss with the metric. After `log1p`, skew drops to ~0.12 and the QQ plot is nearly linear — every model improves.

## The leakage trap (senior signal)

> Most write-ups use `OrdinalEncoder().fit(pd.concat([train, test]))`. **That's leakage.**

Encoders, scalers, target statistics, and clustering models must be **fit on training data only**, then applied to test. If you fit on `train + test`, the test distribution leaks into how you transform train, your CV score gets optimistic, and your leaderboard score regresses.

```python
# WRONG — leaks test distribution into train preprocessing
all_data = pd.concat([train, test])
all_data["Neighborhood_enc"] = OrdinalEncoder().fit_transform(all_data[["Neighborhood"]])

# WRONG — leaks test target statistics
nbhd_med = all_data.groupby("Neighborhood")["SalePrice"].median()  # SalePrice is NaN in test
all_data["NeighborhoodPrice"] = all_data["Neighborhood"].map(nbhd_med)

# RIGHT — fit only on train, then map
nbhd_med = train.groupby("Neighborhood")["SalePrice"].median()
all_data["NeighborhoodPrice"] = all_data["Neighborhood"].map(nbhd_med).fillna(nbhd_med.median())
```

This pipeline does the right thing for both **target encoding** and **KMeans clustering** — both are fit on train only, then mapped onto test (see `engineer()` in `house_prices.py`).

## Project layout

```
house_prices.py          # full pipeline (EDA → ensemble → SHAP → artifacts)
streamlit_app.py         # interactive price estimator
train.csv / test.csv     # competition data
data_description.txt     # field reference
figures/                 # 11 generated PNGs
artifacts/model.pkl      # serialized models + preprocessing artifacts
submission.csv           # generated Kaggle submission
requirements.txt
```

## Reproducibility

Fixed `SEED=42`; 5-fold KFold with the same random state for both CV and stacking; Optuna `TPESampler(seed=42)`. Re-runs on identical hardware produce identical metrics.

## Hyperparameter search space (Optuna)

| Booster | Tuned parameters |
|---|---|
| XGBoost | `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| LightGBM | `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| CatBoost | `learning_rate`, `depth`, `l2_leaf_reg`, `subsample` |

All optimized for 5-fold CV RMSLE with `TPESampler`. Default `--trials 30` (≈10 min total); spec calls for `--trials 100` (≈30 min total) for a small additional gain.
