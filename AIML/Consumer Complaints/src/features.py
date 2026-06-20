"""
Text preprocessing + feature engineering.

Produces data/features.parquet containing:
  - clean_text : lowercased, URL/HTML/punct stripped, stopwords removed, lemmatized
  - engineered numeric features: word_count, char_count, sentence_count,
    capital_words, exclamation_count, vader_sentiment
Also saves a figure of sentiment-by-category and a feature summary table.
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

BASE = r"C:\Working\AI ML Projetcs\Consumer Complaints"
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
XREDACT = re.compile(r"x{2,}")
CAPWORD = re.compile(r"\b[A-Z]{2,}\b")


@lru_cache(maxsize=200_000)
def _lem(tok):
    return LEM.lemmatize(tok)


def clean(text):
    t = text.lower()
    t = URL.sub(" ", t)
    t = HTML.sub(" ", t)
    t = XREDACT.sub(" ", t)          # drop CFPB PII redaction blocks (xxxx)
    t = NONALPHA.sub(" ", t)
    toks = [_lem(w) for w in t.split() if w not in STOP and len(w) > 2]
    return " ".join(toks)


def main():
    df = pd.read_parquet(DATA)
    print(f"Rows: {len(df):,}")

    nar = df["narrative"]
    # --- engineered numeric features (computed on RAW text) ---
    df["word_count"] = nar.str.split().str.len()
    df["char_count"] = nar.str.len()
    df["sentence_count"] = nar.str.count(r"[.!?]+").clip(lower=1)
    df["capital_words"] = nar.apply(lambda s: len(CAPWORD.findall(s)))
    df["exclamation_count"] = nar.str.count("!")

    print("Scoring VADER sentiment...")
    sia = SentimentIntensityAnalyzer()
    df["vader_sentiment"] = nar.apply(lambda s: sia.polarity_scores(s[:2000])["compound"])

    print("Cleaning + lemmatizing text (this takes a few minutes)...")
    df["clean_text"] = nar.apply(clean)

    df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}")

    # --- feature summary by category ---
    feat_cols = ["word_count", "char_count", "sentence_count",
                 "capital_words", "exclamation_count", "vader_sentiment"]
    summary = df.groupby("category")[feat_cols].mean().round(2)
    summary.to_json(os.path.join(METRICS, "feature_summary.json"), indent=2)
    print(summary.to_string())

    # --- sentiment by category figure ---
    sns.set_theme(style="whitegrid")
    order = df.groupby("category")["vader_sentiment"].mean().sort_values().index
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="vader_sentiment", y="category", order=order,
                hue="category", palette="coolwarm_r", legend=False, errorbar=None)
    plt.title("Average VADER sentiment by category (more negative = more distressed)")
    plt.xlabel("Mean compound sentiment")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "09_sentiment_by_category.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("saved 09_sentiment_by_category.png")

    # before/after preprocessing example
    ex = {"before": nar.iloc[0][:300], "after": df["clean_text"].iloc[0][:300]}
    with open(os.path.join(METRICS, "preprocess_example.json"), "w") as f:
        json.dump(ex, f, indent=2)


if __name__ == "__main__":
    main()
