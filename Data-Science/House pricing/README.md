# 🏠 House Price Prediction

End-to-end regression project on the classic **Ames, Iowa "House Prices: Advanced
Regression Techniques"** dataset (1,460 houses × 80 features). The goal is to
estimate a property's `SalePrice` before listing, so a real-estate business can
price faster and more accurately.

This repo demonstrates a complete ML workflow: **data cleaning → EDA → feature
engineering → baseline & advanced models → hyperparameter tuning →
cross-validation → explainability (SHAP) → Kaggle submission.**

---

## 📊 Results

Models are compared by **5-fold cross-validated RMSE on `log1p(SalePrice)`** —
the official Kaggle metric. Lower RMSE is better.

| Model              | R² (holdout) | CV RMSE (log) |
|--------------------|:------------:|:-------------:|
| **XGBoost**        | **0.906**    | **0.120**     |
| LightGBM           | 0.898        | 0.125         |
| Linear Regression  | 0.884        | 0.136         |
| Random Forest      | 0.874        | 0.134         |

> Numbers above are from the default tuned run. The exact, reproducible
> leaderboard is written to [`outputs/model_comparison.csv`](outputs/model_comparison.csv)
> every time you run the pipeline. Gradient boosting (XGBoost / LightGBM) wins,
> as expected for tabular data.

**Top price drivers (from SHAP):** overall quality, total living area,
neighborhood, total square footage, garage capacity, and house age.

---

## 🗂️ Project Structure

```
House pricing/
├── data/                       # raw dataset (train/test/sample + description)
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── src/
│   ├── config.py               # paths, constants, feature groups
│   ├── data_prep.py            # load, outlier removal, missing-value imputation
│   ├── feature_engineering.py  # engineered features + one-hot encoding
│   ├── eda.py                  # 6 EDA figures
│   ├── models.py               # model zoo, search spaces, CV utilities
│   └── explain.py              # predicted-vs-actual + SHAP plots
├── outputs/
│   ├── figures/                # all generated PNGs
│   ├── models/                 # serialized best model (.joblib)
│   ├── model_comparison.csv    # leaderboard
│   └── submission.csv          # Kaggle-ready predictions
├── main.py                     # runs the whole pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (EDA + tuning + SHAP + submission)
python main.py

# Faster variants
python main.py --fast      # skip hyperparameter tuning (baseline models)
python main.py --no-eda    # skip figure generation
```

Outputs (figures, leaderboard, trained model, `submission.csv`) land in
`outputs/`.

---

## 🔬 Methodology

### 1. Data Cleaning ([`src/data_prep.py`](src/data_prep.py))
- **Outliers:** drop the two documented `GrLivArea > 4000` partial sales.
- **Domain-aware imputation:** `"None"` for categoricals where missing means the
  feature is absent (pool, garage, basement…), `0` for the matching numeric
  fields, **neighborhood-median** for `LotFrontage`, and mode/median for the rest.

### 2. EDA ([`src/eda.py`](src/eda.py))
- Target is heavily right-skewed → modeled on `log1p(SalePrice)`.
- Quality (`OverallQual`) and size (`GrLivArea`) correlate most with price.
- Neighborhood drives a 3–4× price spread; newer homes command a premium.

### 3. Feature Engineering ([`src/feature_engineering.py`](src/feature_engineering.py))
`HouseAge`, `RemodAge`, `GarageAge`, `TotalSF`, `TotalArea`, `TotalBath`,
`TotalPorchSF`, `QualScore`, plus `Has*` flags (pool, garage, basement, 2nd
floor, fireplace). Categoricals are one-hot encoded with aligned train/test
columns.

### 4. Modeling ([`src/models.py`](src/models.py))
Linear Regression (baseline) → Random Forest → XGBoost → LightGBM, each tuned
with `RandomizedSearchCV` and validated with 5-fold CV.

### 5. Explainability ([`src/explain.py`](src/explain.py))
SHAP `TreeExplainer` on the best model plus a predicted-vs-actual diagnostic.

---

## 🖼️ Selected Figures

| | |
|---|---|
| ![Target distribution](outputs/figures/01_target_distribution.png) | ![Correlation heatmap](outputs/figures/02_correlation_heatmap.png) |
| ![Neighborhood](outputs/figures/04_neighborhood_boxplot.png) | ![SHAP importance](outputs/figures/09_shap_importance.png) |

---

## 🔭 Possible Extensions
- **Stacked ensemble** of XGBoost + LightGBM + Random Forest (typically +1–3%).
- **External signals:** school ratings, crime rate, distance to city center.
- **Deep learning** tabular baseline for comparison against boosting.
- **BI dashboard** (Power BI / Tableau) for price drivers and neighborhood ranking.

---

## 📚 Dataset
Dean De Cock, *"Ames, Iowa: Alternative to the Boston Housing Data Set"* (2011) —
distributed via the Kaggle competition *House Prices: Advanced Regression
Techniques*. See [`data/data_description.txt`](data/data_description.txt) for the
full feature dictionary.
