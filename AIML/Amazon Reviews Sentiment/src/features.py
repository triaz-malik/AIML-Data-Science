"""
Phase 4 — Feature engineering.

Reads data/clean.parquet and produces data/features.parquet with:
  clean_text       : lowercased, URL/HTML/punct stripped, stopwords removed, lemmatized
  word_count       : tokens in the raw review
  char_count       : characters in the raw review
  exclamation_count: '!' count (emotion indicator)
  caps_words       : ALL-CAPS words (shouting indicator)
  vader_sentiment  : VADER compound polarity of the raw review
  helpful          : carried through from prep (review importance)
Also writes outputs/metrics/feature_summary.json and a VADER-by-sentiment figure.
"""
import json
import os
import re
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords as nltk_stop
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
DATA = os.path.join(BASE, "data", "clean.parquet")
OUT = os.path.join(BASE, "data", "features.parquet")
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP = set(nltk_stop.words("english"))
LEM = WordNetLemmatizer()
URL = re.compile(r"http\S+|www\.\S+")
HTML = re.compile(r"<[^>]+>")
NONALPHA = re.compile(r"[^a-z\s]")
CAPWORD = re.compile(r"\b[A-Z]{2,}\b")


@lru_cache(maxsize=200_000)
def _lem(tok):
    return LEM.lemmatize(tok)


def clean(text):
    t = text.lower()
    t = URL.sub(" ", t)
    t = HTML.sub(" ", t)
    t = NONALPHA.sub(" ", t)
    toks = [_lem(w) for w in t.split() if w not in STOP and len(w) > 2]
    return " ".join(toks)


def main():
    df = pd.read_parquet(DATA)
    print(f"Rows: {len(df):,}")
    txt = df["reviewText"]

    # --- numeric features on RAW text ---
    df["word_count"] = txt.str.split().str.len()
    df["char_count"] = txt.str.len()
    df["exclamation_count"] = txt.str.count("!")
    df["caps_words"] = txt.apply(lambda s: len(CAPWORD.findall(s)))

    print("Scoring VADER sentiment...")
    sia = SentimentIntensityAnalyzer()
    df["vader_sentiment"] = txt.apply(lambda s: sia.polarity_scores(s[:2000])["compound"])

    print("Cleaning + lemmatizing text (this takes a few minutes)...")
    df["clean_text"] = txt.apply(clean)

    df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}")

    # --- feature summary by sentiment ---
    feat_cols = ["word_count", "char_count", "exclamation_count",
                 "caps_words", "vader_sentiment", "helpful"]
    summary = df.groupby("sentiment")[feat_cols].mean().round(3)
    summary.to_json(os.path.join(METRICS, "feature_summary.json"), indent=2)
    print(summary.to_string())

    # --- VADER vs label sanity figure ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    order = ["negative", "neutral", "positive"]
    sns.boxplot(data=df, x="sentiment", y="vader_sentiment", order=order,
                hue="sentiment", palette={"negative": "#d62728", "neutral": "#7f7f7f",
                                          "positive": "#2ca02c"}, legend=False, showfliers=False)
    plt.title("VADER compound score by star-derived sentiment label")
    plt.xlabel(""); plt.ylabel("VADER compound")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "07_vader_by_sentiment.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("saved 07_vader_by_sentiment.png")

    ex = {"before": txt.iloc[0][:300], "after": df["clean_text"].iloc[0][:300]}
    with open(os.path.join(METRICS, "preprocess_example.json"), "w") as f:
        json.dump(ex, f, indent=2)


if __name__ == "__main__":
    main()
