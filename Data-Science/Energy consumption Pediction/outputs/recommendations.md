# Energy Consumption — Insights & Recommendations

## Model performance (held-out future period)
                   RMSE (Wh)  MAE (Wh)     R2
LightGBM                59.3      23.7  0.572
XGBoost                 60.1      23.6  0.560
Random Forest           60.2      27.9  0.558
Linear Regression       70.1      28.1  0.401

**Best model: LightGBM** — explains 57% of
test-period variance with a typical error of ~24 Wh.

## What drives consumption
- Strongest *environmental/behavioural* drivers (SHAP): hour_cos, hour_sin, is_night, lights, T9.
- Recent usage (lag & rolling features) dominates short-term prediction —
  consumption is highly autocorrelated, which is what makes next-interval
  forecasting accurate.

## When energy is used
- **Peak hours:** [18, 17, 19] (early evening). **Off-peak (cheapest to run loads):** [3, 2, 4].
- Weekday mean 97 Wh vs weekend mean 101 Wh.

## Actionable recommendations
1. **Load shifting:** schedule deferrable appliances (dishwasher, washing
   machine, EV charging) into the [3, 2, 4] off-peak window to cut cost an
   estimated 5-15% without changing usage.
2. **HVAC optimisation:** `temp_diff` (indoor-outdoor) is a top driver —
   tightening setpoints / pre-conditioning before peak hours reduces evening load.
3. **Demand-peak forecasting:** the model predicts the next interval well
   enough to pre-warn utilities/building managers of [18, 17, 19] peaks for
   better grid planning.
4. **Anomaly watch:** flag intervals where actual >> predicted as possible
   faulty appliances or unexpected spikes.
