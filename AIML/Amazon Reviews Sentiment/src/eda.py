"""
Phase 2 — Exploratory Data Analysis.

Reads data/clean.parquet and produces:
  01_rating_distribution.png      bar chart of 1-5 stars
  02_sentiment_distribution.png   positive / neutral / negative counts
  03_review_length_by_sentiment.png  word-count histograms (neg vs pos)
  04_sentiment_by_category.png    mean rating + %negative per category
  05_wordclouds.png               positive vs negative word clouds
  06_top_ngrams.png               top bigrams/trigrams in negative reviews
Plus outputs/metrics/eda_stats.json (length stats + top n-gram tables).

n-gram + word-cloud work runs on a capped sample for speed; full data drives the
count/length stats.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS, WordCloud

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
DATA = os.path.join(BASE, "data", "clean.parquet")
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")

SAMPLE = 60_000          # cap for wordcloud / n-gram vectorising
SEED = 42
SENT_ORDER = ["negative", "neutral", "positive"]
SENT_PAL = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
TOKEN = re.compile(r"[^a-z\s]")


def top_ngrams(texts, n, k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english",
                          min_df=5, max_features=40000)
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    order = sums.argsort()[::-1][:k]
    return [(vocab[i], int(sums[i])) for i in order]


def main():
    sns.set_theme(style="whitegrid")
    df = pd.read_parquet(DATA, columns=["reviewText", "overall", "sentiment", "category"])
    print(f"Rows: {len(df):,}")

    df["word_count"] = df["reviewText"].str.split().str.len()

    # --- 01 rating distribution ---
    plt.figure(figsize=(7, 4.5))
    rc = df["overall"].value_counts().sort_index()
    sns.barplot(x=rc.index, y=rc.values, hue=rc.index, palette="viridis", legend=False)
    plt.title("Rating distribution (stars)")
    plt.xlabel("Stars"); plt.ylabel("Reviews")
    for i, v in enumerate(rc.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_rating_distribution.png"), dpi=130); plt.close()

    # --- 02 sentiment distribution ---
    plt.figure(figsize=(6.5, 4.5))
    sc = df["sentiment"].value_counts().reindex(SENT_ORDER)
    sns.barplot(x=sc.index, y=sc.values, hue=sc.index, palette=SENT_PAL, legend=False)
    plt.title("Sentiment distribution")
    plt.xlabel(""); plt.ylabel("Reviews")
    total = len(df)
    for i, v in enumerate(sc.values):
        plt.text(i, v, f"{v/total*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "02_sentiment_distribution.png"), dpi=130); plt.close()

    # --- 03 review length by sentiment (neg vs pos) ---
    plt.figure(figsize=(8, 4.5))
    for s in ["negative", "positive"]:
        sub = df.loc[df["sentiment"] == s, "word_count"].clip(upper=300)
        sns.histplot(sub, bins=50, stat="density", element="step",
                     color=SENT_PAL[s], alpha=0.4, label=s)
    plt.title("Review length by sentiment (words, clipped at 300)")
    plt.xlabel("Word count"); plt.ylabel("Density"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "03_review_length_by_sentiment.png"), dpi=130); plt.close()

    # --- 04 sentiment by category ---
    cat = df.groupby("category").agg(
        mean_rating=("overall", "mean"),
        pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
    ).sort_values("mean_rating")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(x=cat["mean_rating"], y=cat.index, hue=cat.index,
                palette="RdYlGn", legend=False, ax=ax[0])
    ax[0].set_title("Mean rating by category"); ax[0].set_xlabel("Mean stars"); ax[0].set_ylabel("")
    sns.barplot(x=cat["pct_negative"], y=cat.index, hue=cat.index,
                palette="Reds", legend=False, ax=ax[1])
    ax[1].set_title("% negative reviews by category"); ax[1].set_xlabel("% negative"); ax[1].set_ylabel("")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "04_sentiment_by_category.png"), dpi=130); plt.close()

    # --- sample for text-heavy plots ---
    samp = df.sample(min(SAMPLE, len(df)), random_state=SEED)
    samp_clean = samp["reviewText"].str.lower().map(lambda t: TOKEN.sub(" ", t))
    pos_text = " ".join(samp_clean[samp["sentiment"] == "positive"])
    neg_text = " ".join(samp_clean[samp["sentiment"] == "negative"])

    # --- 05 word clouds ---
    sw = set(STOPWORDS) | {"book", "product", "one", "will", "im", "ive", "br"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, txt, title, cmap in [(ax[0], pos_text, "Positive reviews", "Greens"),
                                (ax[1], neg_text, "Negative reviews", "Reds")]:
        wc = WordCloud(width=700, height=500, background_color="white",
                       stopwords=sw, colormap=cmap, max_words=120).generate(txt)
        a.imshow(wc); a.axis("off"); a.set_title(title, fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "05_wordclouds.png"), dpi=130); plt.close()

    # --- 06 top n-grams in negative reviews ---
    neg_series = samp_clean[samp["sentiment"] == "negative"]
    bigrams = top_ngrams(neg_series, 2, 15)
    trigrams = top_ngrams(neg_series, 3, 15)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, grams, title in [(ax[0], bigrams, "Top bigrams — negative"),
                            (ax[1], trigrams, "Top trigrams — negative")]:
        labels, vals = zip(*grams)
        sns.barplot(x=list(vals), y=list(labels), hue=list(labels),
                    palette="rocket", legend=False, ax=a)
        a.set_title(title); a.set_xlabel("Count"); a.set_ylabel("")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "06_top_ngrams.png"), dpi=130); plt.close()

    # --- stats json ---
    stats = {
        "rating_distribution": df["overall"].value_counts().sort_index().to_dict(),
        "sentiment_pct": (df["sentiment"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "word_count_mean_by_sentiment": df.groupby("sentiment")["word_count"].mean().round(1).to_dict(),
        "word_count_median_by_sentiment": df.groupby("sentiment")["word_count"].median().to_dict(),
        "mean_rating_by_category": cat["mean_rating"].round(3).to_dict(),
        "pct_negative_by_category": cat["pct_negative"].round(2).to_dict(),
        "top_bigrams_negative": [{"ngram": g, "count": c} for g, c in bigrams],
        "top_trigrams_negative": [{"ngram": g, "count": c} for g, c in trigrams],
        "top_bigrams_positive": [{"ngram": g, "count": c}
                                 for g, c in top_ngrams(samp_clean[samp["sentiment"] == "positive"], 2, 15)],
    }
    with open(os.path.join(METRICS, "eda_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("EDA complete. Figures + eda_stats.json written.")
    print(json.dumps({k: stats[k] for k in ["sentiment_pct", "word_count_mean_by_sentiment",
                                             "mean_rating_by_category"]}, indent=2))


if __name__ == "__main__":
    main()
