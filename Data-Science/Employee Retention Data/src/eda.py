"""
Exploratory Data Analysis.

Generates the eight diagnostic plots from the project brief and answers the
six initial business questions. All figures are written to outputs/figures and
a machine-readable summary of the findings to outputs/metrics/eda_findings.json
(consumed by the PDF report generator).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless backend -- no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .data_prep import build_features, load_raw

sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}  # stay = blue, leave = red


def _save(fig, name: str) -> str:
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    """Attrition rate (%) grouped by a column, sorted descending."""
    return (df.groupby(col)[config.TARGET].mean() * 100).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Individual plots
# --------------------------------------------------------------------------- #
def plot_attrition_distribution(df: pd.DataFrame) -> str:
    counts = df[config.TARGET].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Stay (0)", "Leave (1)"], counts.values,
                  color=[PALETTE[0], PALETTE[1]])
    for b, p in zip(bars, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                f"{p:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Plot 1 — Attrition Distribution (class imbalance)")
    ax.set_ylabel("Employees")
    return _save(fig, "01_attrition_distribution.png")


def plot_department(df: pd.DataFrame) -> str:
    rate = _rate_by(df, "Department")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(rate.index, rate.values, color="#C44E52")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{b.get_height():.1f}%", ha="center", fontweight="bold")
    ax.set_title("Plot 2 — Attrition Rate by Department")
    ax.set_ylabel("Attrition %")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "02_department_attrition.png")


def plot_income_box(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x=config.TARGET, y="MonthlyIncome", hue=config.TARGET,
                palette=PALETTE, legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stay", "Leave"])
    ax.set_title("Plot 3 — Monthly Income vs Attrition")
    ax.set_xlabel("")
    return _save(fig, "03_income_boxplot.png")


def plot_overtime(df: pd.DataFrame) -> str:
    rate = _rate_by(df, "OverTime")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(rate.index, rate.values, color=["#C44E52", "#4C72B0"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{b.get_height():.1f}%", ha="center", fontweight="bold")
    ax.set_title("Plot 4 — Attrition Rate by Overtime")
    ax.set_ylabel("Attrition %")
    ax.set_xlabel("Works Overtime")
    return _save(fig, "04_overtime_attrition.png")


def plot_tenure(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    stay = df[df[config.TARGET] == 0]["YearsAtCompany"]
    leave = df[df[config.TARGET] == 1]["YearsAtCompany"]
    bins = np.arange(0, df["YearsAtCompany"].max() + 2)
    ax.hist([stay, leave], bins=bins, stacked=True,
            color=[PALETTE[0], PALETTE[1]], label=["Stay", "Leave"])
    ax.set_title("Plot 5 — Years At Company (when do people leave?)")
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Employees")
    ax.legend()
    return _save(fig, "05_years_at_company.png")


def plot_promotion(df: pd.DataFrame) -> str:
    rate = (df.groupby("YearsSinceLastPromotion")[config.TARGET].mean() * 100)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rate.index, rate.values, marker="o", color="#C44E52")
    ax.axvline(4, ls="--", color="grey", alpha=0.7)
    ax.text(4.1, ax.get_ylim()[1] * 0.9, "stagnation\nthreshold (>4y)", color="grey")
    ax.set_title("Plot 6 — Promotion Delay vs Attrition")
    ax.set_xlabel("Years since last promotion")
    ax.set_ylabel("Attrition %")
    return _save(fig, "06_promotion_delay.png")


def plot_job_satisfaction(df: pd.DataFrame) -> str:
    rate = (df.groupby("JobSatisfaction")[config.TARGET].mean() * 100)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(rate.index.astype(str), rate.values, color="#C44E52")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{b.get_height():.1f}%", ha="center", fontweight="bold")
    ax.set_title("Plot 7 — Job Satisfaction vs Attrition")
    ax.set_xlabel("Job satisfaction (1 = low … 4 = high)")
    ax.set_ylabel("Attrition %")
    return _save(fig, "07_job_satisfaction.png")


def plot_correlation(df: pd.DataFrame) -> str:
    cols = ["MonthlyIncome", "JobLevel", "TotalWorkingYears",
            "YearsAtCompany", "YearsSinceLastPromotion", "Age",
            "JobSatisfaction", "EnvironmentSatisfaction", config.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Plot 8 — Correlation Heatmap")
    return _save(fig, "08_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Business questions
# --------------------------------------------------------------------------- #
def answer_business_questions(df: pd.DataFrame) -> dict:
    dept = _rate_by(df, "Department")
    ot = _rate_by(df, "OverTime")
    income_stay = df[df[config.TARGET] == 0]["MonthlyIncome"].median()
    income_leave = df[df[config.TARGET] == 1]["MonthlyIncome"].median()
    early = df[df["YearsAtCompany"] < 2][config.TARGET].mean() * 100
    tenured = df[df["YearsAtCompany"] >= 2][config.TARGET].mean() * 100
    promo_delayed = df[df["YearsSinceLastPromotion"] > 4][config.TARGET].mean() * 100
    promo_recent = df[df["YearsSinceLastPromotion"] <= 4][config.TARGET].mean() * 100
    sat = (df.groupby("JobSatisfaction")[config.TARGET].mean() * 100)

    return {
        "Q1_department": {
            "question": "Which departments lose most employees?",
            "answer": f"{dept.index[0]} has the highest attrition at "
                      f"{dept.iloc[0]:.1f}%, vs {dept.index[-1]} at {dept.iloc[-1]:.1f}%.",
            "data": {k: round(v, 1) for k, v in dept.items()},
        },
        "Q2_overtime": {
            "question": "Does overtime cause resignations?",
            "answer": f"Overtime employees leave at {ot.get('Yes', float('nan')):.1f}% "
                      f"vs {ot.get('No', float('nan')):.1f}% without overtime "
                      f"({ot.get('Yes') / ot.get('No'):.1f}x higher).",
            "data": {k: round(v, 1) for k, v in ot.items()},
        },
        "Q3_salary": {
            "question": "Do low salaries increase churn?",
            "answer": f"Median income of leavers is ${income_leave:,.0f} vs "
                      f"${income_stay:,.0f} for stayers "
                      f"({(1 - income_leave / income_stay) * 100:.0f}% lower).",
            "data": {"median_income_stay": float(income_stay),
                     "median_income_leave": float(income_leave)},
        },
        "Q4_new_employees": {
            "question": "Are new employees leaving quickly?",
            "answer": f"Employees with <2 years tenure leave at {early:.1f}% "
                      f"vs {tenured:.1f}% for more tenured staff.",
            "data": {"early_career_pct": round(early, 1),
                     "tenured_pct": round(tenured, 1)},
        },
        "Q5_promotion": {
            "question": "Does promotion delay impact attrition?",
            "answer": f"Employees not promoted in >4 years leave at {promo_delayed:.1f}% "
                      f"vs {promo_recent:.1f}% for recently promoted staff.",
            "data": {"promo_delayed_pct": round(promo_delayed, 1),
                     "promo_recent_pct": round(promo_recent, 1)},
        },
        "Q6_risky_profile": {
            "question": "Which employee profiles are most risky?",
            "answer": "Highest risk profile: works overtime, in the lowest salary band, "
                      "early career (<2y), with delayed promotion and low job "
                      f"satisfaction (satisfaction=1 leaves at {sat.get(1, float('nan')):.1f}%).",
            "data": {f"job_satisfaction_{int(k)}": round(v, 1) for k, v in sat.items()},
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    df = build_features(load_raw())

    figures = {
        "attrition_distribution": plot_attrition_distribution(df),
        "department": plot_department(df),
        "income_box": plot_income_box(df),
        "overtime": plot_overtime(df),
        "tenure": plot_tenure(df),
        "promotion": plot_promotion(df),
        "job_satisfaction": plot_job_satisfaction(df),
        "correlation": plot_correlation(df),
    }
    findings = answer_business_questions(df)

    summary = {
        "n_rows": int(len(df)),
        "attrition_rate_pct": round(df[config.TARGET].mean() * 100, 1),
        "figures": figures,
        "business_questions": findings,
    }
    out = config.METRIC_DIR / "eda_findings.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"EDA complete -> {len(figures)} figures, findings saved to {out.name}")
    return summary


if __name__ == "__main__":
    run()
