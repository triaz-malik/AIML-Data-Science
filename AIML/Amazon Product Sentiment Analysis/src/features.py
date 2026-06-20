"""Hand-crafted, interpretable text features.

These complement the TF-IDF representation and are useful for EDA (e.g. "do
negative reviews use more exclamation marks?"). They depend only on the raw
review text — no external metadata required.
"""
from __future__ import annotations

import re

import pandas as pd

_WORD_RE = re.compile(r"\b\w+\b")


def add_text_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of `df` with engineered columns appended."""
    out = df.copy()
    s = out[text_col].fillna("")

    out["char_count"] = s.str.len()
    out["word_count"] = s.apply(lambda t: len(_WORD_RE.findall(t)))
    out["avg_word_len"] = (out["char_count"] / out["word_count"].clip(lower=1)).round(2)
    out["n_exclaim"] = s.str.count("!")
    out["n_question"] = s.str.count(r"\?")
    out["n_upper_words"] = s.apply(
        lambda t: sum(1 for w in t.split() if len(w) > 1 and w.isupper())
    )
    out["upper_ratio"] = (
        s.apply(lambda t: sum(1 for c in t if c.isupper())) / out["char_count"].clip(lower=1)
    ).round(3)
    return out


FEATURE_COLS = [
    "char_count",
    "word_count",
    "avg_word_len",
    "n_exclaim",
    "n_question",
    "n_upper_words",
    "upper_ratio",
]
