# Multi-Step Weather Forecasting — Final Report
### LSTM · GRU · Bi-LSTM on the Jena Climate Dataset

## 1. Executive Summary

We built and compared classical and deep-learning models to forecast air temperature from historical weather observations. The best single-step model was **XGBoost** (RMSE **0.640 °C**, MAE **0.450 °C**, R² **0.9932**). Deep recurrent models were extended to multi-step forecasting (24h / 48h / 7-day) for realistic operational use, and SHAP analysis identified the dominant physical drivers.

## 2. Business Problem

Accurate temperature forecasting drives decisions across **airlines, agriculture, energy, logistics, and smart cities** — from energy demand planning to crop protection and transport scheduling. The goal: predict future temperature from historical multivariate weather conditions.

## 3. Data Understanding

- **Source:** Jena Climate 2009–2016 (Max Planck Institute).
- **Raw records:** 420,551 at 10-minute resolution (2009-01-01 00:10:00 → 2017-01-01 00:00:00).
- **Features:** 14 meteorological variables (temperature, pressure, humidity, dew point, vapor pressure, air density, wind speed/direction).
- **Resampling:** hourly (≈70k rows) for tractable, leakage-free modeling.
- **Data quality:** missing values = none; duplicate timestamps = 327 (dropped); impossible wind readings = 38 (treated as missing, filled).

## 4. EDA Findings

![Daily mean temperature 2009–2016](../figures/01_temperature_trend.png)

*Daily mean temperature 2009–2016*

![Feature correlation heatmap](../figures/07_correlation_heatmap.png)

*Feature correlation heatmap*

![Temperature distribution by season](../figures/06_seasonal_analysis.png)

*Temperature distribution by season*

**Key observations:** strong annual seasonality (summer–winter swing of ~17 °C); temperature is highly correlated with dew point, vapor pressure and (negatively) air density; humidity is inversely related to temperature; wind speed is right-skewed and mildly seasonal.

## 5. Time-Series Analysis

![Additive seasonal decomposition](../figures/15_seasonal_decomposition.png)

*Additive seasonal decomposition*

![ACF / PACF of hourly temperature](../figures/16_acf_pacf.png)

*ACF / PACF of hourly temperature*

- Long-term trend: **0.321 °C/year**.
- Summer mean **18.04 °C** vs winter **0.78 °C**.
- Autocorrelation: lag-1 = **0.993**, lag-24 = **0.918** — strong daily memory, which motivates sequence models.

## 6. Modeling

**Feature engineering:** lag features (t-1, t-6, t-24, t-48), rolling mean/std (7h, 24h, 48h), calendar features and cyclic (sin/cos) encodings of hour, day-of-year and wind direction.

**Models:** Persistence baseline → Linear Regression → Random Forest → XGBoost → LSTM → GRU → Bi-LSTM. All evaluated on the same chronological 70/15/15 split with train-only scaling.

### Single-Step Leaderboard (test set)

|                  |    MAE |   RMSE |   MAPE |     R2 |
|:-----------------|-------:|-------:|-------:|-------:|
| XGBoost          | 0.4502 | 0.6405 |  7.138 | 0.9932 |
| BiLSTM           | 0.4697 | 0.6609 |  7.502 | 0.9928 |
| RandomForest     | 0.4591 | 0.662  |  7.219 | 0.9928 |
| LinearRegression | 0.4824 | 0.6806 |  7.826 | 0.9924 |
| LSTM             | 0.5779 | 0.7655 | 10.757 | 0.9903 |
| GRU              | 0.5644 | 0.7789 |  8.278 | 0.99   |
| Persistence      | 0.6788 | 0.9607 | 10.006 | 0.9848 |

![RMSE / MAE by model](../figures/17_model_comparison.png)

*RMSE / MAE by model*

![Actual vs predicted (best DL model)](../figures/18_actual_vs_predicted.png)

*Actual vs predicted (best DL model)*

## 7. Hyperparameter Tuning

Random search over window size, hidden units, dropout, learning rate and batch size. **Best configuration:** `{"window": 24, "units": 128, "dropout": 0.1, "lr": 0.0005, "batch": 128, "arch": "BiLSTM"}` (val RMSE 0.6961).

## 8. Multi-Step Forecasting

|      |    MAE |   RMSE |   MAPE |     R2 |   per_step_rmse_first |   per_step_rmse_last |
|:-----|-------:|-------:|-------:|-------:|----------------------:|---------------------:|
| 24h  | 1.7743 | 2.3174 | 31.082 | 0.9112 |                0.8507 |               2.8597 |
| 48h  | 2.2636 | 2.949  | 37.352 | 0.8559 |                1.0137 |               3.7043 |
| 168h | 3.0843 | 3.954  | 49.719 | 0.7403 |                1.5697 |               4.4093 |

![Error growth with lead time](../figures/20_multistep_error_growth.png)

*Error growth with lead time*

![Example 7-day forecast](../figures/21_example_7day_forecast.png)

*Example 7-day forecast*

As expected, error grows with lead time; the 24h horizon is the most accurate and the 7-day horizon the hardest.

## 9. Explainability (SHAP)

![Global SHAP feature importance](../figures/22_shap_importance_bar.png)

*Global SHAP feature importance*

![SHAP beeswarm (direction & magnitude)](../figures/23_shap_beeswarm.png)

*SHAP beeswarm (direction & magnitude)*

Key physical drivers of the next-hour prediction (mean |SHAP|):
- `temp_lag_1`: 0.1465
- `pressure_lag_1`: 0.0557
- `humidity`: 0.0544
- `temp_roll_mean_24`: 0.0493
- `humidity_lag_1`: 0.0350
- `wind_speed`: 0.0254
- `pressure`: 0.0151

Recent-temperature lags dominate (thermal inertia), with humidity and pressure providing the meteorological context that the model uses to adjust the short-term trajectory.

## 10. Business Impact

- **Energy:** day-ahead temperature forecasts sharpen heating/cooling demand prediction and grid load balancing.
- **Agriculture:** frost and heat-stress early warning for crop protection and irrigation scheduling.
- **Transport & logistics:** proactive scheduling around adverse conditions, reducing delays.
- **Smart cities:** automated HVAC and public-service planning.

---
*Generated from the project's saved artifacts in `outputs/`.*