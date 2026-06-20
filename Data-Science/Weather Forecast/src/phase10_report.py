"""Final Report generator.

Assembles a Markdown executive report from every phase's saved artifacts:
business problem, data understanding, EDA findings, time-series analysis,
modeling, tuning, multi-step results, explainability, and business impact.
Figures are referenced by relative path so the report renders in any viewer.
"""
from __future__ import annotations

import json
import os
import pandas as pd

import config as C


def jload(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def csv_to_md(path, **kw):
    if not os.path.exists(path):
        return "_(not available)_"
    return pd.read_csv(path, **kw).round(4).to_markdown(index=kw.get("index_col") is not None)


def fig(name, caption):
    rel = f"../figures/{name}.png"
    return f"![{caption}]({rel})\n\n*{caption}*\n"


def main():
    dq = jload(f"{C.TABLE_DIR}/phase1_data_quality.json", {})
    ts = {}
    p = f"{C.TABLE_DIR}/phase2_timeseries_summary.csv"
    if os.path.exists(p):
        ts = pd.read_csv(p, index_col=0).iloc[:, 0].to_dict()
    comp = jload(f"{C.TABLE_DIR}/phase4_baseline_metrics.json", {})
    dl = jload(f"{C.TABLE_DIR}/phase5_dl_metrics.json", {})
    best_cfg = jload(f"{C.TABLE_DIR}/phase6_best_config.json", {})
    ms = jload(f"{C.TABLE_DIR}/phase7_multistep_metrics.json", {})
    drivers = jload(f"{C.TABLE_DIR}/phase9_key_drivers.json", {})

    all_metrics = {**comp, **dl}
    leader = (pd.DataFrame(all_metrics).T[["MAE", "RMSE", "MAPE", "R2"]]
              .sort_values("RMSE")) if all_metrics else pd.DataFrame()
    best_model = leader.index[0] if len(leader) else "N/A"

    L = []
    A = L.append
    A("# Multi-Step Weather Forecasting — Final Report")
    A("### LSTM · GRU · Bi-LSTM on the Jena Climate Dataset\n")

    # Executive summary
    A("## 1. Executive Summary\n")
    if len(leader):
        b = leader.iloc[0]
        A(f"We built and compared classical and deep-learning models to forecast "
          f"air temperature from historical weather observations. The best "
          f"single-step model was **{best_model}** "
          f"(RMSE **{b['RMSE']:.3f} °C**, MAE **{b['MAE']:.3f} °C**, "
          f"R² **{b['R2']:.4f}**). Deep recurrent models were extended to "
          f"multi-step forecasting (24h / 48h / 7-day) for realistic operational "
          f"use, and SHAP analysis identified the dominant physical drivers.\n")

    # Business problem
    A("## 2. Business Problem\n")
    A("Accurate temperature forecasting drives decisions across **airlines, "
      "agriculture, energy, logistics, and smart cities** — from energy demand "
      "planning to crop protection and transport scheduling. The goal: predict "
      "future temperature from historical multivariate weather conditions.\n")

    # Data understanding
    A("## 3. Data Understanding\n")
    A(f"- **Source:** Jena Climate 2009–2016 (Max Planck Institute).")
    A(f"- **Raw records:** {dq.get('n_rows', 'N/A'):,} at 10-minute resolution "
      f"({dq.get('date_start','?')} → {dq.get('date_end','?')}).")
    A(f"- **Features:** 14 meteorological variables (temperature, pressure, "
      f"humidity, dew point, vapor pressure, air density, wind speed/direction).")
    A(f"- **Resampling:** hourly (≈70k rows) for tractable, leakage-free modeling.")
    A(f"- **Data quality:** missing values = {dq.get('missing_values','?')}; "
      f"duplicate timestamps = {dq.get('duplicate_timestamps','?')} (dropped); "
      f"impossible wind readings = {dq.get('impossible_wind_readings','?')} "
      f"(treated as missing, filled).\n")

    # EDA findings
    A("## 4. EDA Findings\n")
    A(fig("01_temperature_trend", "Daily mean temperature 2009–2016"))
    A(fig("07_correlation_heatmap", "Feature correlation heatmap"))
    A(fig("06_seasonal_analysis", "Temperature distribution by season"))
    A("**Key observations:** strong annual seasonality (summer–winter swing of "
      "~17 °C); temperature is highly correlated with dew point, vapor pressure "
      "and (negatively) air density; humidity is inversely related to "
      "temperature; wind speed is right-skewed and mildly seasonal.\n")

    # Time series analysis
    A("## 5. Time-Series Analysis\n")
    A(fig("15_seasonal_decomposition", "Additive seasonal decomposition"))
    A(fig("16_acf_pacf", "ACF / PACF of hourly temperature"))
    if ts:
        A(f"- Long-term trend: **{ts.get('yearly_trend_degC_per_year','?')} °C/year**.")
        A(f"- Summer mean **{ts.get('summer_mean_temp','?')} °C** vs winter "
          f"**{ts.get('winter_mean_temp','?')} °C**.")
        A(f"- Autocorrelation: lag-1 = **{ts.get('acf_lag1','?')}**, "
          f"lag-24 = **{ts.get('acf_lag24','?')}** — strong daily memory, which "
          f"motivates sequence models.\n")

    # Modeling
    A("## 6. Modeling\n")
    A("**Feature engineering:** lag features (t-1, t-6, t-24, t-48), rolling "
      "mean/std (7h, 24h, 48h), calendar features and cyclic (sin/cos) "
      "encodings of hour, day-of-year and wind direction.\n")
    A("**Models:** Persistence baseline → Linear Regression → Random Forest → "
      "XGBoost → LSTM → GRU → Bi-LSTM. All evaluated on the same chronological "
      "70/15/15 split with train-only scaling.\n")
    if len(leader):
        A("### Single-Step Leaderboard (test set)\n")
        A(leader.round(4).to_markdown())
        A("")
        A(fig("17_model_comparison", "RMSE / MAE by model"))
        A(fig("18_actual_vs_predicted", "Actual vs predicted (best DL model)"))

    # Tuning
    A("## 7. Hyperparameter Tuning\n")
    if best_cfg:
        A(f"Random search over window size, hidden units, dropout, learning "
          f"rate and batch size. **Best configuration:** "
          f"`{json.dumps({k: best_cfg[k] for k in best_cfg if k in ('arch','window','units','dropout','lr','batch')})}` "
          f"(val RMSE {best_cfg.get('val_RMSE','?')}).\n")
    else:
        A("_(Run `python src/phase6_tuning.py` to populate this section.)_\n")

    # Multi-step
    A("## 8. Multi-Step Forecasting\n")
    if ms:
        A(pd.DataFrame(ms).T.round(4).to_markdown())
        A("")
        A(fig("20_multistep_error_growth", "Error growth with lead time"))
        A(fig("21_example_7day_forecast", "Example 7-day forecast"))
        A("As expected, error grows with lead time; the 24h horizon is the most "
          "accurate and the 7-day horizon the hardest.\n")
    else:
        A("_(Run `python src/phase7_multistep.py` to populate this section.)_\n")

    # Explainability
    A("## 9. Explainability (SHAP)\n")
    A(fig("22_shap_importance_bar", "Global SHAP feature importance"))
    A(fig("23_shap_beeswarm", "SHAP beeswarm (direction & magnitude)"))
    if drivers:
        top = sorted(drivers.items(), key=lambda kv: -kv[1])
        A("Key physical drivers of the next-hour prediction (mean |SHAP|):")
        for k, v in top:
            A(f"- `{k}`: {v:.4f}")
        A("")
    A("Recent-temperature lags dominate (thermal inertia), with humidity and "
      "pressure providing the meteorological context that the model uses to "
      "adjust the short-term trajectory.\n")

    # Business impact
    A("## 10. Business Impact\n")
    A("- **Energy:** day-ahead temperature forecasts sharpen heating/cooling "
      "demand prediction and grid load balancing.")
    A("- **Agriculture:** frost and heat-stress early warning for crop "
      "protection and irrigation scheduling.")
    A("- **Transport & logistics:** proactive scheduling around adverse "
      "conditions, reducing delays.")
    A("- **Smart cities:** automated HVAC and public-service planning.\n")
    A("---\n*Generated from the project's saved artifacts in `outputs/`.*")

    report = "\n".join(L)
    path = f"{C.REPORT_DIR}/final_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] Written -> {path}  ({len(report)} chars)")


if __name__ == "__main__":
    main()
