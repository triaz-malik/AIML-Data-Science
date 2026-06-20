"""
Shared feature-engineering utilities.
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

Hand-crafted signals known to separate scam/promo messages from normal SMS:
  msg_length, char_count, word_count, n_urls, n_digits, digit_ratio,
  n_currency, n_uppercase, uppercase_ratio, n_exclaim, n_special, has_phone
"""
from __future__ import annotations
import re
import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|info|ng|uk|co|ly|link|click|xyz)\b)", re.I)
CURRENCY_RE = re.compile(r"[£$€₦]")
PHONE_RE = re.compile(r"\b\d{4,}\b")
SPECIAL_RE = re.compile(r"[^a-zA-Z0-9\s]")

FEATURE_COLS = [
    "msg_length", "char_count", "word_count", "n_urls", "n_digits",
    "digit_ratio", "n_currency", "n_uppercase", "uppercase_ratio",
    "n_exclaim", "n_special", "has_phone",
]


def engineer(text: str) -> dict:
    t = str(text)
    n = max(len(t), 1)
    letters = sum(c.isalpha() for c in t)
    return {
        "msg_length": len(t),
        "char_count": len(t.replace(" ", "")),
        "word_count": len(t.split()),
        "n_urls": len(URL_RE.findall(t)),
        "n_digits": sum(c.isdigit() for c in t),
        "digit_ratio": sum(c.isdigit() for c in t) / n,
        "n_currency": len(CURRENCY_RE.findall(t)),
        "n_uppercase": sum(c.isupper() for c in t),
        "uppercase_ratio": (sum(c.isupper() for c in t) / max(letters, 1)),
        "n_exclaim": t.count("!"),
        "n_special": len(SPECIAL_RE.findall(t)),
        "has_phone": int(bool(PHONE_RE.search(t))),
    }


def add_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    feats = pd.DataFrame([engineer(t) for t in df[text_col]], index=df.index)
    return pd.concat([df, feats], axis=1)
