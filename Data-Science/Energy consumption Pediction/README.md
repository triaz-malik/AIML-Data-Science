# Energy Consumption Prediction — Appliances Energy (Wh)

Predict household appliance energy consumption and explain what drives it, to
support load shifting, HVAC optimisation, and demand‑peak forecasting.

**Dataset:** UCI/Kaggle *Appliances Energy Prediction* — 19,735 records @ 10‑min
cadence (Jan–May 2016), 9 zones of indoor temperature/humidity, outdoor
weather‑station data. Target: `Appliances` (Wh).

## Results (held‑out **future** period — chronological 80/20 split)

| Model | Test R² | RMSE (Wh) | MAE (Wh) |
|---|---|---|---|
| Linear Regression | 0.40 | 70.1 | 28.1 |
| Random Forest | 0.56 | 60.2 | 27.9 |
| XGBoost | 0.56 | 60.1 | 23.6 |
| **LightGBM (best)** | **0.57** | **59.3** | 23.7 |

> The ~0.57 R² matches the published benchmark (Candanedo et al., 2017). Many
> notebooks report 0.80–0.90 on this data, but that comes from a *random* split
> that leaks the autocorrelated past into the future. This project uses a
> **chronological split** and `.shift()`‑based lag features, so the numbers are
> honest and reproducible on truly unseen future data.

## Method highlights
- **Leakage‑aware:** chronological train/test split; lag/rolling features only
  ever see the past; `TimeSeriesSplit` for all hyperparameter tuning.
- **Right‑skew handled:** model `log1p(Appliances)`, invert for Wh metrics;
  high‑consumption spikes kept (they are the demand‑peak events of interest).
- **Models:** Linear Regression → Random Forest → XGBoost → LightGBM, tuned with
  `RandomizedSearchCV` and **Optuna**.
- **Explainability:** SHAP (importance + beeswarm) on the best model.

## Files
| Path | What it is |
|---|---|
| `Energy_Consumption_Prediction.ipynb` | Executed notebook — full narrative, plots, models, SHAP |
| `Energy_Consumption_Prediction.html` | Static report (open in any browser) |
| `energy_pipeline.py` | Same pipeline as a runnable script (jupytext percent format) |
| `outputs/plots/` | All 12 EDA / results / SHAP figures (PNG) |
| `outputs/recommendations.md` | Auto‑generated business insights |
| `outputs/data/predictions_for_dashboard.csv` | Forecast vs actual + residuals (Power BI / Tableau ready) |
| `outputs/data/model_comparison.csv` | Metrics table |
| `outputs/models/best_model.joblib` | Saved best model + features + scaler |

## Reproduce
```bash
python energy_pipeline.py                      # run the whole pipeline (~3 min)
# or regenerate the executed notebook:
jupytext --to notebook energy_pipeline.py -o Energy_Consumption_Prediction.ipynb
jupyter nbconvert --to notebook --execute --inplace Energy_Consumption_Prediction.ipynb
```

Requires: pandas, numpy, scikit‑learn, matplotlib, seaborn, xgboost, lightgbm,
optuna, shap, joblib, jupytext.

## Next steps
Time‑series models (Prophet / LSTM / GRU), weather‑forecast integration,
Isolation‑Forest anomaly detection on residuals, an RL appliance scheduler, and
a Power BI dashboard built on `predictions_for_dashboard.csv`.
