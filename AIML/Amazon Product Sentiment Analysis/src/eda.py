"""Exploratory data analysis: distributions, review-length, word clouds.

Every function saves a figure to reports/figures/ and returns the path, so the
EDA notebook stays thin and the same analysis can run headless as a script:

    python -m src.eda --sample 50000
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks override with %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from . import config, data, features, preprocess

sns.set_theme(style="whitegrid")


def _save(fig, name: str):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label"].map(config.LABEL_NAMES).value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                palette={"negative": "#d9534f", "positive": "#5cb85c"}, legend=False)
    ax.set(title="Review sentiment distribution", xlabel="", ylabel="count")
    return _save(fig, "01_class_balance.png")


def plot_review_length(df):
    df = features.add_text_features(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [(config.NEG, "#d9534f"), (config.POS, "#5cb85c")]:
        sub = df[df["label"] == label]["word_count"].clip(upper=200)
        sns.kdeplot(sub, ax=ax, fill=True, alpha=0.4, color=color,
                    label=config.LABEL_NAMES[label])
    ax.set(title="Review length by sentiment", xlabel="word count (clipped@200)")
    ax.legend()
    return _save(fig, "02_review_length.png")


def plot_feature_box(df):
    df = features.add_text_features(df)
    df["sentiment"] = df["label"].map(config.LABEL_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pal = {"negative": "#d9534f", "positive": "#5cb85c"}
    sns.boxplot(data=df, x="sentiment", y="n_exclaim", ax=axes[0], showfliers=False,
                hue="sentiment", palette=pal, legend=False)
    axes[0].set(title="Exclamation marks", xlabel="", ylabel="count per review")
    sns.boxplot(data=df, x="sentiment", y="upper_ratio", ax=axes[1], showfliers=False,
                hue="sentiment", palette=pal, legend=False)
    axes[1].set(title="Uppercase ratio", xlabel="", ylabel="fraction of chars")
    fig.tight_layout()
    return _save(fig, "03_feature_boxplots.png")


def _wordcloud_for(df, label, name, color):
    preprocess.ensure_nltk()
    text = " ".join(preprocess.clean_series(df[df["label"] == label]["text"].tolist()))
    wc = WordCloud(width=900, height=450, background_color="white",
                   colormap=color, max_words=120).generate(text)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Most frequent words — {config.LABEL_NAMES[label]} reviews")
    return _save(fig, name)


def plot_wordclouds(df):
    return [
        _wordcloud_for(df, config.POS, "04_wordcloud_positive.png", "Greens"),
        _wordcloud_for(df, config.NEG, "05_wordcloud_negative.png", "Reds"),
    ]


def top_words(df, label, n=20):
    preprocess.ensure_nltk()
    tokens = " ".join(preprocess.clean_series(df[df["label"] == label]["text"].tolist())).split()
    return Counter(tokens).most_common(n)


def run_all(sample_size: int):
    df = data.load_split("train", sample_size=sample_size)
    paths = [plot_class_balance(df), plot_review_length(df), plot_feature_box(df)]
    paths += plot_wordclouds(df)
    print("Saved figures:")
    for p in paths:
        print(f"  {p}")
    print("\nTop positive words:", top_words(df, config.POS, 15))
    print("Top negative words:", top_words(df, config.NEG, 15))


def main():
    ap = argparse.ArgumentParser(description="Generate EDA figures.")
    ap.add_argument("--sample", type=int, default=50_000)
    args = ap.parse_args()
    run_all(args.sample)


if __name__ == "__main__":
    main()
