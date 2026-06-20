"""Load the Amazon Reviews Polarity (fastText) dataset into tidy DataFrames.

Each raw line looks like:

    __label__2 Great product: It arrived on time and works perfectly...

i.e. ``__label__{1|2} <title>: <body>``. label 1 = negative, 2 = positive.
The title and body are separated by the first ``": "``; we keep the full
review text (title + body) for modeling and also expose them separately.

Loading is parquet-cached so the (slow) bz2 parse happens once per config.
"""
from __future__ import annotations

import bz2
import re

import pandas as pd

from . import config

_LABEL_RE = re.compile(r"^__label__([12])\s+(.*)$", re.DOTALL)


def _parse_line(line: str) -> tuple[int, str, str] | None:
    """Return (label0/1, title, body) for one raw line, or None if malformed."""
    m = _LABEL_RE.match(line.rstrip("\n"))
    if not m:
        return None
    label = int(m.group(1)) - 1  # 1->0 (neg), 2->1 (pos)
    text = m.group(2)
    # Title and body are joined by the first ": ". Some reviews have no colon;
    # treat the whole thing as body in that case.
    if ": " in text:
        title, body = text.split(": ", 1)
    else:
        title, body = "", text
    return label, title, body


def _read_archive(path, limit: int | None = None) -> pd.DataFrame:
    rows: list[tuple[int, str, str]] = []
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parsed = _parse_line(line)
            if parsed is None:
                continue
            rows.append(parsed)
            if limit is not None and len(rows) >= limit:
                break
    df = pd.DataFrame(rows, columns=["label", "title", "body"])
    df["text"] = (df["title"] + ". " + df["body"]).str.strip(". ").str.strip()
    return df


def _balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Return up to `n` rows with a 50/50 label balance, shuffled."""
    per_class = n // 2
    parts = [
        g.sample(n=min(per_class, len(g)), random_state=seed)
        for _, g in df.groupby("label")
    ]
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def load_split(
    split: str = "train",
    sample_size: int | None = config.DEFAULT_SAMPLE_SIZE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load a split as a DataFrame with columns: label, title, body, text.

    Parameters
    ----------
    split : "train" or "test".
    sample_size : balanced subsample size; None loads the full split.
    use_cache : read/write a parquet cache keyed by (split, sample_size).
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    tag = "full" if sample_size is None else str(sample_size)
    cache = config.PROCESSED_DIR / f"{split}_{tag}.parquet"

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    archive = config.train_archive() if split == "train" else config.test_archive()

    if sample_size is None:
        df = _read_archive(archive)
    else:
        # Read a generous superset (rarely perfectly balanced upstream) then
        # downsample to an exact 50/50 split.
        df = _read_archive(archive, limit=sample_size * 3)
        df = _balanced_sample(df, sample_size, config.SEED)

    if use_cache:
        df.to_parquet(cache, index=False)
    return df


if __name__ == "__main__":
    # Smoke test: print shape + class balance for a tiny sample.
    d = load_split("train", sample_size=2000, use_cache=False)
    print(d.shape)
    print(d["label"].value_counts())
    print(d.iloc[0][["label", "title", "body"]].to_dict())
