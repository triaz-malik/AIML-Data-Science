"""
Phase 1 (Data Understanding) + Phase 3 (Cleaning) — labeling & deduplication.

- Loads data/raw.parquet (from download_data.py).
- Casts types, parses timestamps, drops missing/empty review text.
- Removes duplicate reviews.
- Builds sentiment labels from the star rating:
    3-class:  1-2 -> negative, 3 -> neutral, 4-5 -> positive   (column `sentiment`)
    binary :  1-2 -> negative, 4-5 -> positive, 3-star dropped (column `sentiment_bin`)
- Saves data/clean.parquet + outputs/metrics/data_summary.json.

Heavy text normalisation (lemmatised clean_text) happens in features.py, mirroring
the project's existing split between prep and feature engineering.
"""
import json
import os

import pandas as pd

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
RAW = os.path.join(BASE, "data", "raw.parquet")
DATA_DIR = os.path.join(BASE, "data")
METRICS_DIR = os.path.join(BASE, "outputs", "metrics")


def rating_to_sentiment(stars):
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    print("Loading raw parquet...")
    df = pd.read_parquet(RAW)
    n_total = len(df)

    # --- types ---
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["helpful"] = pd.to_numeric(df["helpful"], errors="coerce").fillna(0).astype(int)
    df["reviewTime_ms"] = pd.to_numeric(df["reviewTime_ms"], errors="coerce")
    df["date"] = pd.to_datetime(df["reviewTime_ms"], unit="ms", errors="coerce")
    df["verified_purchase"] = df["verified_purchase"].astype(str).str.lower().eq("true")

    # --- missing / empty text ---
    df["reviewText"] = df["reviewText"].fillna("").astype(str).str.strip()
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()
    n_missing_text = int((df["reviewText"].str.len() == 0).sum())
    df = df[df["reviewText"].str.len() > 0]
    df = df[df["overall"].between(1, 5)]

    # --- duplicates (same body text) ---
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["reviewText"]).reset_index(drop=True)
    n_dupes = n_before_dedup - len(df)

    # --- labels ---
    df["overall"] = df["overall"].round().astype(int)
    df["sentiment"] = df["overall"].apply(rating_to_sentiment)
    df["sentiment_bin"] = df["sentiment"].where(df["sentiment"] != "neutral")  # NaN on 3-star

    keep = ["reviewText", "summary", "overall", "sentiment", "sentiment_bin",
            "category", "helpful", "verified_purchase", "date",
            "asin", "parent_asin"]
    df = df[keep]

    out = os.path.join(DATA_DIR, "clean.parquet")
    df.to_parquet(out, index=False)

    summary = {
        "rows_total_raw": int(n_total),
        "rows_missing_text": n_missing_text,
        "duplicate_reviews_removed": int(n_dupes),
        "rows_final": int(len(df)),
        "n_categories": int(df["category"].nunique()),
        "rows_per_category": df["category"].value_counts().to_dict(),
        "rating_distribution": df["overall"].value_counts().sort_index().to_dict(),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "sentiment_pct": (df["sentiment"].value_counts(normalize=True) * 100)
            .round(2).to_dict(),
        "binary_rows_after_dropping_neutral": int(df["sentiment_bin"].notna().sum()),
        "pct_verified": round(df["verified_purchase"].mean() * 100, 2),
        "date_min": str(df["date"].min().date()) if df["date"].notna().any() else None,
        "date_max": str(df["date"].max().date()) if df["date"].notna().any() else None,
        "mean_helpful_votes": round(float(df["helpful"].mean()), 3),
    }
    with open(os.path.join(METRICS_DIR, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved cleaned dataset -> {out}")


if __name__ == "__main__":
    main()
