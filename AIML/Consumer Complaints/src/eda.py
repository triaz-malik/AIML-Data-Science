"""
Exploratory Data Analysis for the CFPB complaint dataset.
Generates publication-quality figures saved to outputs/figures and
summary statistics to outputs/metrics/eda_stats.json.
"""
import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

BASE = r"C:\Working\AI ML Projetcs\Consumer Complaints"
DATA = os.path.join(BASE, "data", "clean.parquet")
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")

sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("viridis", 8)


def savefig(name):
    path = os.path.join(FIG, name)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print("saved", name)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_parquet(DATA)
    stats = {}

    # ---- 1. Category distribution ----
    counts = df["category"].value_counts()
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    for i, v in enumerate(counts.values):
        ax.text(v + 300, i, f"{v:,}", va="center", fontsize=9)
    plt.xlabel("Number of complaints")
    plt.ylabel("")
    plt.title("Complaint volume by category (with consumer narrative)")
    savefig("01_category_distribution.png")
    stats["category_counts"] = counts.to_dict()

    # ---- 2. Volume over time (monthly) ----
    ts = df.set_index("date").resample("ME").size()
    plt.figure(figsize=(11, 4.5))
    plt.plot(ts.index, ts.values, marker="o", ms=3, color="#2c7fb8")
    plt.fill_between(ts.index, ts.values, alpha=0.15, color="#2c7fb8")
    plt.title("Monthly complaint volume over time")
    plt.ylabel("Complaints / month")
    plt.xlabel("")
    savefig("02_volume_over_time.png")
    stats["peak_month"] = str(ts.idxmax().date())
    stats["peak_month_count"] = int(ts.max())

    # ---- 2b. Stacked category trend ----
    monthly_cat = df.groupby([pd.Grouper(key="date", freq="ME"), "category"]).size().unstack(fill_value=0)
    plt.figure(figsize=(11, 5))
    monthly_cat.plot.area(ax=plt.gca(), colormap="viridis", alpha=0.85, linewidth=0)
    plt.title("Monthly complaint volume by category")
    plt.ylabel("Complaints / month")
    plt.xlabel("")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    savefig("03_volume_by_category.png")

    # ---- 3. State-wise complaints (top 15) ----
    st = df["state"].value_counts().head(15)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=st.values, y=st.index, palette="rocket")
    for i, v in enumerate(st.values):
        ax.text(v + 100, i, f"{v:,}", va="center", fontsize=9)
    plt.title("Top 15 states by complaint volume")
    plt.xlabel("Number of complaints")
    savefig("04_top_states.png")
    stats["top_states"] = st.head(10).to_dict()

    # ---- 3b. US choropleth (state heatmap) ----
    try:
        import plotly.express as px
        sc = df["state"].value_counts().reset_index()
        sc.columns = ["state", "count"]
        sc = sc[sc["state"].str.len() == 2]
        fig = px.choropleth(sc, locations="state", locationmode="USA-states",
                            color="count", scope="usa", color_continuous_scale="Reds",
                            title="Complaint volume by U.S. state")
        fig.write_image(os.path.join(FIG, "05_state_heatmap.png"), width=1000, height=600, scale=2)
        print("saved 05_state_heatmap.png")
    except Exception as e:
        print("choropleth skipped:", e)

    # ---- 4. Complaint length analysis ----
    df["word_count"] = df["narrative"].str.split().str.len()
    df["char_count"] = df["narrative"].str.len()
    stats["word_count"] = {
        "mean": round(float(df["word_count"].mean()), 1),
        "median": int(df["word_count"].median()),
        "p95": int(df["word_count"].quantile(0.95)),
        "max": int(df["word_count"].max()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(df["word_count"].clip(upper=1000), bins=60, color="#41b6c4")
    axes[0].set_title("Complaint length (words)")
    axes[0].set_xlabel("Word count (clipped at 1000)")
    axes[1].hist(df["char_count"].clip(upper=6000), bins=60, color="#225ea8")
    axes[1].set_title("Complaint length (characters)")
    axes[1].set_xlabel("Char count (clipped at 6000)")
    savefig("06_length_histograms.png")

    # ---- 4b. Length by category (boxplot) ----
    order = counts.index.tolist()
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="word_count", y="category", order=order,
                showfliers=False, palette="viridis")
    plt.xlim(0, df["word_count"].quantile(0.95))
    plt.title("Complaint length by category (words, outliers hidden)")
    plt.xlabel("Word count")
    plt.ylabel("")
    savefig("07_length_by_category.png")

    # ---- 5. Word clouds for top 4 categories ----
    # CFPB redacts PII with strings of X's -> add to stopwords.
    sw = set(STOPWORDS) | {"xxxx", "xx", "xxxxxxxx", "xxxxx", "account", "company", "said", "told", "will"}
    top4 = counts.index[:4].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, cat in zip(axes.ravel(), top4):
        text = " ".join(df[df["category"] == cat]["narrative"].sample(
            min(4000, (df["category"] == cat).sum()), random_state=42).tolist())
        text = re.sub(r"x{2,}", " ", text.lower())
        wc = WordCloud(width=600, height=400, background_color="white",
                       stopwords=sw, colormap="viridis", max_words=80).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cat, fontsize=12, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Most frequent terms by complaint category", fontsize=14, y=1.01)
    savefig("08_wordclouds.png")

    with open(os.path.join(METRICS, "eda_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("\nEDA stats:", json.dumps(stats, indent=2)[:1200])


if __name__ == "__main__":
    main()
