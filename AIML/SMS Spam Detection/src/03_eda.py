"""
Phase 3 - Exploratory Data Analysis
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

Reads data/processed/sms_labeled.csv and writes figures to outputs/figures/
plus a text summary to reports/eda_summary.md

Figures:
  01_class_distribution.png      - 5-class + binary bars
  02_message_length.png          - length histogram + boxplot by class
  03_wordclouds.png              - per-class word clouds
  04_top_ngrams.png              - top unigrams/bigrams/trigrams for malicious msgs
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sms_labeled.csv"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Promotion", "Spam", "Phishing", "Fraud"]
PALETTE = {"Normal": "#2ca02c", "Promotion": "#1f77b4", "Spam": "#ff7f0e",
           "Phishing": "#d62728", "Fraud": "#9467bd"}
sns.set_theme(style="whitegrid")


def fig_class_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vc = df["label5"].value_counts().reindex(CLASSES)
    sns.barplot(x=vc.index, y=vc.values, hue=vc.index, legend=False,
                palette=[PALETTE[c] for c in CLASSES], ax=ax[0])
    ax[0].set_title("5-Class Distribution", fontweight="bold")
    ax[0].set_ylabel("Messages")
    for i, v in enumerate(vc.values):
        ax[0].text(i, v + 30, f"{v}\n{v/len(df)*100:.1f}%", ha="center", fontsize=9)

    bvc = df["binary_label"].value_counts()
    sns.barplot(x=bvc.index, y=bvc.values, hue=bvc.index, legend=False,
                palette=["#2ca02c", "#d62728"], ax=ax[1])
    ax[1].set_title("Binary Distribution (ground truth)", fontweight="bold")
    ax[1].set_ylabel("Messages")
    for i, v in enumerate(bvc.values):
        ax[1].text(i, v + 30, f"{v}\n{v/len(df)*100:.1f}%", ha="center", fontsize=9)
    fig.suptitle("Fraud / Phishing / Spam Landscape", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "01_class_distribution.png", dpi=130)
    plt.close(fig)


def fig_message_length(df):
    df = df.copy()
    df["len"] = df["text"].astype(str).str.len()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c in CLASSES:
        sub = df[df["label5"] == c]["len"].clip(upper=320)
        ax[0].hist(sub, bins=40, alpha=0.5, label=c, color=PALETTE[c])
    ax[0].set_title("Message Length Distribution by Class", fontweight="bold")
    ax[0].set_xlabel("Characters"); ax[0].set_ylabel("Count"); ax[0].legend()

    sns.boxplot(data=df, x="label5", y="len", order=CLASSES, hue="label5",
                legend=False, palette=PALETTE, ax=ax[1])
    ax[1].set_ylim(0, 320)
    ax[1].set_title("Length Spread by Class", fontweight="bold")
    ax[1].set_xlabel(""); ax[1].set_ylabel("Characters")
    fig.tight_layout()
    fig.savefig(FIG / "02_message_length.png", dpi=130)
    plt.close(fig)
    return df.groupby("label5")["len"].median().reindex(CLASSES)


def fig_wordclouds(df):
    sw = set(STOPWORDS) | {"u", "ur", "im", "2", "4", "n", "s", "call", "ll", "now",
                           "will", "go", "got", "get", "one", "txt", "text"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for i, c in enumerate(CLASSES):
        text = " ".join(df[df["label5"] == c]["text"].astype(str)).lower()
        if text.strip():
            wc = WordCloud(width=700, height=450, background_color="white",
                           stopwords=sw, colormap="viridis",
                           collocations=False).generate(text)
            axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(c, fontweight="bold", color=PALETTE[c], fontsize=13)
        axes[i].axis("off")
    axes[-1].axis("off")
    fig.suptitle("Word Clouds by Message Class", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "03_wordclouds.png", dpi=130)
    plt.close(fig)


def _top_ngrams(corpus, ngram, k=12):
    vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words="english", min_df=2)
    X = vec.fit_transform(corpus)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    idx = freqs.argsort()[::-1][:k]
    return terms[idx][::-1], freqs[idx][::-1]


def fig_top_ngrams(df):
    mal = df[df["label5"].isin(["Spam", "Phishing", "Fraud", "Promotion"])]["text"].astype(str)
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    for j, (n, name) in enumerate([(1, "Unigrams"), (2, "Bigrams"), (3, "Trigrams")]):
        terms, freqs = _top_ngrams(mal, n)
        ax[j].barh(terms, freqs, color=sns.color_palette("rocket", len(terms)))
        ax[j].set_title(f"Top {name} (malicious msgs)", fontweight="bold")
        ax[j].set_xlabel("Frequency")
    fig.tight_layout()
    fig.savefig(FIG / "04_top_ngrams.png", dpi=130)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA)
    fig_class_distribution(df)
    med_len = fig_message_length(df)
    fig_wordclouds(df)
    fig_top_ngrams(df)

    lines = ["# EDA Summary\n",
             f"Total messages: **{len(df)}**\n",
             "## Class distribution\n",
             "```\n" + df["label5"].value_counts().reindex(CLASSES).to_frame("count").to_string() + "\n```",
             "\n## Median message length (chars) by class\n",
             "```\n" + med_len.round(0).astype(int).to_frame("median_len").to_string() + "\n```",
             "\n## Key business questions answered\n",
             f"- Fraud share of all messages: **{(df['label5']=='Fraud').mean()*100:.1f}%**",
             f"- Phishing share: **{(df['label5']=='Phishing').mean()*100:.1f}%**",
             f"- Promotion share: **{(df['label5']=='Promotion').mean()*100:.1f}%**",
             f"- Any malicious/unwanted (non-Normal): **{(df['label5']!='Normal').mean()*100:.1f}%**",
             ]
    (REP / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("EDA figures written to outputs/figures/:")
    for p in sorted(FIG.glob("*.png")):
        print("  •", p.name)
    print("\nMedian length by class:\n", med_len.round(0).astype(int).to_string())
    print("\nSummary -> reports/eda_summary.md")


if __name__ == "__main__":
    main()
