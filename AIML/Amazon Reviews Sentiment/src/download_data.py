"""
Download a multi-category slice of the Amazon Reviews 2023 dataset.

Source: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace), the `raw_review_*` configs.
We stream each category (so we never download the full multi-GB file), shuffle a
buffer for a representative rating mix, take a capped number of rows, and stack
them into a single data/raw.parquet with a `category` column.

Fields kept (schema mirrors the project plan):
    rating (1-5)  ->  overall
    title         ->  summary
    text          ->  reviewText
    helpful_vote  ->  helpful
    timestamp(ms) ->  reviewTime
    plus: asin, parent_asin, verified_purchase, category
"""
import os
import time

import pandas as pd
from datasets import load_dataset

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
DATA_DIR = os.path.join(BASE, "data")
OUT = os.path.join(DATA_DIR, "raw.parquet")

# The dataset ships a loading script (unsupported in datasets>=4), so we stream
# the raw per-category JSONL files straight off the Hub with the generic json loader.
HF_BASE = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
           "resolve/main/raw/review_categories")

# 5 diverse categories -> lets us compare sentiment across product lines.
CATEGORIES = [
    "Electronics",
    "Books",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Toys_and_Games",
]
PER_CATEGORY = 50_000
SHUFFLE_BUFFER = 20_000
SEED = 42

KEEP = ["rating", "title", "text", "helpful_vote", "timestamp",
        "asin", "parent_asin", "verified_purchase"]


def pull_category(cat):
    """Stream one category's JSONL, shuffle a buffer, take PER_CATEGORY rows."""
    url = f"{HF_BASE}/{cat}.jsonl"
    ds = load_dataset("json", data_files=url, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    rows = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i >= PER_CATEGORY:
            break
        rows.append({k: ex.get(k) for k in KEEP})
    df = pd.DataFrame(rows)
    df["category"] = cat.replace("_", " ")
    print(f"  {cat}: {len(df):,} rows in {time.time() - t0:.0f}s")
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    parts = []
    for cat in CATEGORIES:
        print(f"Streaming {cat} ...")
        parts.append(pull_category(cat))

    df = pd.concat(parts, ignore_index=True)

    # Rename to the plan's schema names.
    df = df.rename(columns={
        "rating": "overall",
        "title": "summary",
        "text": "reviewText",
        "helpful_vote": "helpful",
        "timestamp": "reviewTime_ms",
    })

    df.to_parquet(OUT, index=False)
    print(f"\nSaved {len(df):,} rows x {df.shape[1]} cols -> {OUT}")
    print("\nRows per category:")
    print(df["category"].value_counts().to_string())
    print("\nRating distribution:")
    print(df["overall"].value_counts().sort_index().to_string())
    print("\nSample row:")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
