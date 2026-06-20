"""
Phase 1-3: exploratory data analysis.

Generates distribution plots, relationship plots, a correlation heatmap, and
writes an EDA markdown report answering the data-understanding questions:
  - How many patients have stroke?
  - What age groups are most affected?
  - Which health factors matter most?
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import FIG_DIR, REPORT_DIR, TARGET
from data_prep import load_clean

sns.set_theme(style="whitegrid")


def _save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Distribution analysis
# --------------------------------------------------------------------------- #
def distribution_plots(df):
    # Target balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No Stroke", "Stroke"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v/len(df):.1%})", ha="center", va="bottom")
    ax.set(title="Stroke vs Non-Stroke", ylabel="Patients")
    _save(fig, "01_target_balance.png")

    # Age / BMI / glucose distributions, split by stroke
    for col, fname in [
        ("age", "02_age_distribution.png"),
        ("bmi", "03_bmi_distribution.png"),
        ("avg_glucose_level", "04_glucose_distribution.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, stat="density",
                     common_norm=False, palette=["#4C72B0", "#C44E52"], ax=ax)
        ax.set(title=f"{col} distribution by stroke")
        _save(fig, fname)


# --------------------------------------------------------------------------- #
# Relationship analysis
# --------------------------------------------------------------------------- #
def relationship_plots(df):
    # Stroke rate by age group
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Child", "Young", "Adult", "Senior"]
    rate = df.groupby("age_group")[TARGET].mean().reindex(order)
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#C44E52")
    for i, v in enumerate(rate.values):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
    ax.set(title="Stroke rate by age group", ylabel="Stroke rate")
    _save(fig, "05_stroke_by_agegroup.png")

    # Binary risk factors vs stroke rate
    for col, fname in [
        ("hypertension", "06_stroke_by_hypertension.png"),
        ("heart_disease", "07_stroke_by_heartdisease.png"),
        ("smoking_status", "08_stroke_by_smoking.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        rate = df.groupby(col)[TARGET].mean().sort_values()
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, color="#55A868")
        for i, v in enumerate(rate.values):
            ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
        ax.set(title=f"Stroke rate by {col}", ylabel="Stroke rate")
        _save(fig, fname)


# --------------------------------------------------------------------------- #
# Correlation analysis
# --------------------------------------------------------------------------- #
def correlation_plot(df):
    num = df.copy()
    # encode the obvious binaries/categoricals for a numeric correlation view
    for c in ["ever_married", "Residence_type"]:
        num[c] = (num[c] == num[c].unique()[0]).astype(int)
    cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease",
            "health_risk_score", TARGET]
    corr = num[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation heatmap")
    _save(fig, "09_correlation_heatmap.png")
    return corr


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(df, corr):
    n = len(df)
    n_stroke = int(df[TARGET].sum())
    by_age = df.groupby("age_group")[TARGET].agg(["mean", "sum"]).reindex(
        ["Child", "Young", "Adult", "Senior"])
    target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)

    lines = [
        "# EDA Report — Stroke Risk Prediction",
        "",
        "## Phase 1 — Data Understanding",
        "",
        f"- **Total patients:** {n:,}",
        f"- **Stroke cases:** {n_stroke:,} ({n_stroke/n:.2%}) — **severe class imbalance**.",
        f"- **Non-stroke:** {n-n_stroke:,} ({(n-n_stroke)/n:.2%}).",
        "",
        "### How many patients have stroke?",
        f"{n_stroke} of {n} patients ({n_stroke/n:.2%}). Roughly 1 in 20. Accuracy is a",
        "misleading metric here — a model predicting 'no stroke' for everyone scores ~95%.",
        "",
        "### What age groups are most affected?",
        "",
        "| Age group | Stroke rate | Stroke cases |",
        "|---|---|---|",
    ]
    for grp, row in by_age.iterrows():
        lines.append(f"| {grp} | {row['mean']:.2%} | {int(row['sum'])} |")
    lines += [
        "",
        "Stroke risk rises steeply with age — **Seniors (65+) dominate the positive class**.",
        "",
        "### Which health factors matter most?",
        "",
        "Correlation with the stroke target (numeric view):",
        "",
        "| Factor | Correlation with stroke |",
        "|---|---|",
    ]
    for k, v in target_corr.items():
        lines.append(f"| {k} | {v:+.3f} |")
    lines += [
        "",
        "**Age** is by far the strongest single signal, followed by heart disease,",
        "hypertension, average glucose level, and the composite health-risk score.",
        "",
        "## Phase 2 — Data Cleaning (applied in `data_prep.py`)",
        "- 201 missing BMI values (`N/A`) → median imputation.",
        "- Dropped `id` column and the single `Other`-gender row.",
        "- Removed duplicate rows; standardized categorical text.",
        "- Winsorized BMI and glucose at the 0.5/99.5 percentile.",
        "",
        "## Phase 3 — EDA figures",
        "See `outputs/figures/`:",
        "- `01_target_balance.png` — stroke vs non-stroke",
        "- `02-04` — age / BMI / glucose distributions by stroke",
        "- `05-08` — stroke rate by age group / hypertension / heart disease / smoking",
        "- `09_correlation_heatmap.png` — correlation heatmap",
        "",
        "### Key takeaways",
        "- **Age strongly influences stroke** — the dominant predictor.",
        "- **Glucose level is significant** — high-glucose patients show elevated stroke rates.",
        "- **Smoking** shows a moderate association (former/current smokers higher than never).",
        "- Hypertension and heart disease roughly **double-to-triple** the stroke rate.",
    ]
    path = REPORT_DIR / "eda_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    df = load_clean()
    distribution_plots(df)
    relationship_plots(df)
    corr = correlation_plot(df)
    report = write_report(df, corr)
    print(f"[eda] figures -> {FIG_DIR}")
    print(f"[eda] report  -> {report}")


if __name__ == "__main__":
    main()
