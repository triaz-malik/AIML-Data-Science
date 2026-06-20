"""Phase 2 - Time Series Analysis.

Seasonality (summer vs winter), long-term trend, cyclic/daily patterns,
seasonal decomposition, and autocorrelation (ACF / PACF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import config as C

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name):
    fig.savefig(f"{C.FIG_DIR}/{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


def main():
    print("[Phase 2] Loading hourly data ...")
    h = C.add_time_features(C.load_hourly())
    temp = h["temperature"]

    # 1. Seasonality: summer vs winter monthly profile
    fig, ax = plt.subplots(figsize=(9, 4))
    for season, color in [("Summer", "#e74c3c"), ("Winter", "#3498db")]:
        sub = h[h["season"] == season].groupby("hour")["temperature"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=season, color=color)
    ax.set_title("Diurnal Temperature Cycle: Summer vs Winter")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean Temp (°C)")
    ax.legend()
    save(fig, "12_seasonality_summer_winter")

    # 2. Long-term trend via yearly means + linear fit
    yearly = temp.resample("YE").mean()
    years = np.arange(len(yearly))
    slope, intercept = np.polyfit(years, yearly.values, 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(yearly.index, yearly.values, "o-", label="Yearly mean")
    ax.plot(yearly.index, intercept + slope * years, "--",
            label=f"Trend: {slope:+.3f} °C/yr", color="black")
    ax.set_title("Long-Term Temperature Trend")
    ax.set_ylabel("Mean Temp (°C)")
    ax.legend()
    save(fig, "13_longterm_trend")

    # 3. Cyclic / daily pattern: month x hour heatmap
    pivot = h.pivot_table(index="month", columns="hour",
                          values="temperature", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="RdBu_r", center=pivot.values.mean(), ax=ax,
                cbar_kws={"label": "Mean Temp (°C)"})
    ax.set_title("Temperature by Month and Hour (cyclic behavior)")
    save(fig, "14_month_hour_cycle")

    # 4. Seasonal decomposition (use daily series, yearly period=365)
    daily = temp.resample("D").mean().interpolate()
    dec = seasonal_decompose(daily, model="additive", period=365)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    dec.observed.plot(ax=axes[0], color="#2c3e50"); axes[0].set_ylabel("Observed")
    dec.trend.plot(ax=axes[1], color="#c0392b"); axes[1].set_ylabel("Trend")
    dec.seasonal.plot(ax=axes[2], color="#27ae60"); axes[2].set_ylabel("Seasonal")
    dec.resid.plot(ax=axes[3], color="#7f8c8d"); axes[3].set_ylabel("Residual")
    axes[0].set_title("Seasonal Decomposition (additive, period=365 days)")
    save(fig, "15_seasonal_decomposition")

    # 5. ACF / PACF on hourly temperature (up to 72 lags = 3 days)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    plot_acf(temp, lags=72, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF) — hourly temperature, 72 lags")
    plot_pacf(temp, lags=72, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation (PACF) — 72 lags")
    save(fig, "16_acf_pacf")

    # Numeric summary
    summary = {
        "yearly_trend_degC_per_year": round(float(slope), 4),
        "overall_mean_temp": round(float(temp.mean()), 2),
        "summer_mean_temp": round(float(h[h.season == "Summer"].temperature.mean()), 2),
        "winter_mean_temp": round(float(h[h.season == "Winter"].temperature.mean()), 2),
        "acf_lag1": round(float(temp.autocorr(1)), 3),
        "acf_lag24": round(float(temp.autocorr(24)), 3),
    }
    pd.Series(summary).to_csv(f"{C.TABLE_DIR}/phase2_timeseries_summary.csv")
    print("[Phase 2] Summary:", summary)
    print("[Phase 2] Done.")


if __name__ == "__main__":
    main()
