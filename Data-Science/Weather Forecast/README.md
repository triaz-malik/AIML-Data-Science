# Multi-Step Weather Forecasting — LSTM · GRU · Bi-LSTM

End-to-end time-series project that forecasts air temperature from the
**Jena Climate 2009–2016** dataset (Max Planck Institute), comparing classical
ML baselines against deep recurrent networks and extending the best models to
multi-step (24h / 48h / 7-day) forecasting.

## Business problem
Accurate temperature forecasting drives decisions in **airlines, agriculture,
energy, logistics and smart cities** — energy demand planning, frost/heat
early-warning, and transport scheduling. Goal: predict future temperature from
historical multivariate weather conditions.

## Dataset
- 420,551 records at 10-minute resolution, 14 meteorological features.
- Resampled to **hourly** (~70k rows) for tractable, leakage-free modeling.
- Place `jena_climate_2009_2016.csv` in `data/` (already there).

## Project structure
```
data/                      raw dataset
src/
  config.py                paths, loading, cleaning, time features
  modeling.py              splits, metrics, sequence windowing
  phase1_eda.py            EDA + data quality + 11 figures
  phase2_timeseries.py     seasonality, trend, decomposition, ACF/PACF
  phase3_features.py       lag / rolling / calendar / cyclic features
  phase4_baselines.py      Persistence, LinearRegression, RandomForest, XGBoost
  phase5_deep_learning.py  LSTM, GRU, Bi-LSTM (single-step)
  phase6_tuning.py         hyperparameter search (window/units/dropout/lr/batch)
  phase7_multistep.py      24h / 48h / 168h direct multi-output forecasting
  phase8_evaluation.py     master comparison table + figures
  phase9_explainability.py SHAP on XGBoost (key drivers)
  phase10_report.py        assembles outputs/reports/final_report.md
  run_all.py               run the whole pipeline
dashboard/app.py           Streamlit forecasting dashboard
outputs/                   figures, models, tables, reports (generated)
```

## Quick start
```bash
pip install -r requirements.txt

# Full pipeline (CPU-friendly; tuning is the slow part)
python src/run_all.py            # or: python src/run_all.py --fast

# Or run phases individually, e.g.
python src/phase1_eda.py

# Dashboard
streamlit run dashboard/app.py
```
Run scripts from the project root (they import `config`/`modeling` from `src/`,
which works because each script's directory is on `sys.path`). If you invoke
from elsewhere, run `python -m` from within `src/` or add `src/` to `PYTHONPATH`.

## Results (single-step, hourly, chronological test set)
See `outputs/tables/master_comparison.csv` and
`outputs/reports/final_report.md` (generated). XGBoost and Bi-LSTM lead on RMSE;
all models comfortably beat the persistence baseline.

## Notes
- **No GPU required.** Models are sized for CPU training with early stopping.
- All scaling is fit on the training split only (no leakage); splits are
  chronological (no shuffling) as required for time series.
