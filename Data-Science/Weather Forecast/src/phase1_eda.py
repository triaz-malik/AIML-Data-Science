"""Phase 1 - Exploratory Data Analysis.

Data quality (missing values, duplicates, outliers) + visualizations:
temperature/humidity/pressure trends, distributions, monthly & seasonal
patterns, correlation heatmap, wind analysis, boxplots, rolling averages.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as C

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name):
    path = f"{C.FIG_DIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}.png")


def data_quality(raw: pd.DataFrame) -> dict:
    """Compute and report data-quality metrics on the full-resolution data."""
    report = {}
    report["n_rows"] = int(len(raw))
    report["n_cols"] = int(raw.shape[1])
    report["date_start"] = str(raw["datetime"].min())
    report["date_end"] = str(raw["datetime"].max())

    # Missing values
    miss = raw.isna().sum()
    report["missing_values"] = {k: int(v) for k, v in miss.items() if v > 0} or "none"

    # Duplicate timestamps
    report["duplicate_timestamps"] = int(raw["datetime"].duplicated().sum())

    # Sentinel / impossible wind readings (-9999 in the raw data)
    bad_wind = int((raw["wind_speed"] < 0).sum() + (raw["wind_speed_max"] < 0).sum())
    report["impossible_wind_readings"] = bad_wind

    # Outliers via IQR on key numeric columns
    num = raw.select_dtypes(include=[np.number])
    outliers = {}
    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((num[col] < lo) | (num[col] > hi)).sum())
        outliers[col] = {"count": n, "pct": round(100 * n / len(num), 2)}
    report["outliers_iqr"] = outliers

    return report


def main():
    print("[Phase 1] Loading data ...")
    raw = C.load_raw(parse_dates=True)
    # Clean copy for quality scan (keep raw bad values visible in the report)
    print("[Phase 1] Computing data-quality report ...")
    dq = data_quality(raw)
    with open(f"{C.TABLE_DIR}/phase1_data_quality.json", "w") as f:
        json.dump(dq, f, indent=2)
    print(json.dumps({k: dq[k] for k in list(dq)[:6]}, indent=2))

    # Hourly, cleaned data for visualization
    h = C.load_hourly()
    h = C.add_time_features(h)

    # Summary statistics table
    h.select_dtypes(include=[np.number]).describe().T.to_csv(
        f"{C.TABLE_DIR}/phase1_summary_stats.csv"
    )

    print("[Phase 1] Generating figures ...")

    # 1. Temperature trend (daily-resampled for readability)
    daily = h["temperature"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily.index, daily.values, lw=0.7, color="#c0392b")
    ax.set_title("Daily Mean Temperature (2009-2016)")
    ax.set_ylabel("Temperature (°C)")
    save(fig, "01_temperature_trend")

    # 2. Humidity trend
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(h["humidity"].resample("D").mean(), lw=0.7, color="#2980b9")
    ax.set_title("Daily Mean Relative Humidity")
    ax.set_ylabel("Humidity (%)")
    save(fig, "02_humidity_trend")

    # 3. Pressure trend
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(h["pressure"].resample("D").mean(), lw=0.7, color="#27ae60")
    ax.set_title("Daily Mean Pressure")
    ax.set_ylabel("Pressure (mbar)")
    save(fig, "03_pressure_trend")

    # 4. Temperature distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(h["temperature"], bins=60, kde=True, color="#c0392b", ax=ax)
    ax.set_title("Temperature Distribution")
    ax.set_xlabel("Temperature (°C)")
    save(fig, "04_temperature_distribution")

    # 5. Monthly average temperature
    monthly = h.groupby("month")["temperature"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=monthly.index, y=monthly.values, ax=ax, color="#e67e22")
    ax.set_title("Average Temperature by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Temp (°C)")
    save(fig, "05_monthly_avg_temperature")

    # 6. Seasonal analysis (boxplot by season)
    order = ["Winter", "Spring", "Summer", "Autumn"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=h, x="season", y="temperature", order=order, ax=ax)
    ax.set_title("Temperature by Season")
    ax.set_ylabel("Temperature (°C)")
    save(fig, "06_seasonal_analysis")

    # 7. Correlation heatmap
    corr = h.select_dtypes(include=[np.number]).drop(
        columns=["hour", "day", "month", "dayofweek", "is_weekend"], errors="ignore"
    ).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save(fig, "07_correlation_heatmap")

    # 8. Wind speed analysis: distribution + by month
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(h["wind_speed"], bins=50, color="#8e44ad", ax=axes[0])
    axes[0].set_title("Wind Speed Distribution")
    axes[0].set_xlabel("Wind speed (m/s)")
    sns.lineplot(data=h.groupby("month")["wind_speed"].mean(), ax=axes[1],
                 marker="o", color="#8e44ad")
    axes[1].set_title("Mean Wind Speed by Month")
    axes[1].set_xlabel("Month")
    save(fig, "08_wind_speed_analysis")

    # 9. Boxplots of key features (standardized for comparability)
    feats = ["temperature", "humidity", "pressure", "wind_speed",
             "vp_act", "air_density"]
    z = (h[feats] - h[feats].mean()) / h[feats].std()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=z, ax=ax)
    ax.set_title("Standardized Feature Boxplots (outlier view)")
    ax.set_ylabel("z-score")
    save(fig, "09_boxplots")

    # 10. Rolling averages of temperature (7d, 30d)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily, lw=0.4, alpha=0.4, label="Daily", color="grey")
    ax.plot(daily.rolling(7).mean(), lw=1.2, label="7-day", color="#2980b9")
    ax.plot(daily.rolling(30).mean(), lw=1.8, label="30-day", color="#c0392b")
    ax.set_title("Temperature Rolling Averages")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    save(fig, "10_rolling_averages")

    # 11. Hourly (diurnal) temperature pattern
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=h.groupby("hour")["temperature"].mean(), marker="o",
                 ax=ax, color="#c0392b")
    ax.set_title("Average Temperature by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean Temp (°C)")
    save(fig, "11_diurnal_pattern")

    print("[Phase 1] Done. Figures + tables written to outputs/.")


if __name__ == "__main__":
    main()
