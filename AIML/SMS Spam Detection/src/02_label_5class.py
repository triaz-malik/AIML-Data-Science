"""
Phase 2 - Weak Supervision: derive 5 fine-grained classes
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

Input : data/raw/sms_raw.csv  [text, binary_label, source, smish_seed]
Output: data/processed/sms_labeled.csv  (+ label5 column)

Classes: Normal | Promotion | Spam | Phishing | Fraud

Approach (transparent & reproducible):
  - ham  -> Normal (true personal messages)
  - spam/smishing -> classified by PRIORITY rule matching, severe first:
        Fraud  (winnings / money scams)
        Phishing (credential & account theft, esp. with URLs)
        Promotion (legitimate-style marketing)
        Spam   (everything else spammy = fallback)
  - smish_seed==1 messages that fall through default to Phishing
    (they came from a curated smishing corpus).

These fine labels are HEURISTIC, not human-annotated. The binary ham/spam
label remains the true ground truth and is preserved alongside.
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sms_raw.csv"
OUT = ROOT / "data" / "processed" / "sms_labeled.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

URL_RE = re.compile(r"(https?://|www\.|\b\w+\.(com|net|org|info|ng|uk|co|ly|link|click|xyz)\b)", re.I)

# Keyword sets (lower-cased, matched as substrings on a normalised message).
FRAUD = [
    "won", "winner", "win a", "you win", "prize", "award", "awarded", "lottery",
    "lotto", "jackpot", "congratulations", "congrats", "claim your", "cash prize",
    "guaranteed", "reward", "£1000", "£500", "$1000", "1 million", "million",
    "investment", "bitcoin", "crypto", "inheritance", "beneficiary", "selected to receive",
    "you have been selected", "claim now", "100% free", "free entry", "draw",
]
PHISH = [
    "verify", "verification", "confirm your", "your account", "account is", "suspend",
    "suspended", "log in", "login", "password", "otp", "one-time", "one time code",
    "pin code", "your pin", "kyc", "validate", "update your", "unusual activity",
    "locked", "unauthorized", "unauthorised", "secure your", "card blocked", "card has been",
    "bank", "debit", "credit alert", "click the link", "click here to", "dear customer",
    "dear subscriber", "reactivate", "restore your",
]
PROMO = [
    "offer", "sale", "discount", "% off", "voucher", "coupon", "free delivery",
    "shop now", "buy now", "subscribe", "unsubscribe", "reply stop", "txt stop",
    "text stop", "ringtone", "new arrival", "limited time", "sign up", "deal",
    "save up to", "promo", "upgrade", "latest", "exclusive", "tones", "to opt out",
]


def _contains(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def classify(row) -> str:
    if row["binary_label"] == "ham":
        return "Normal"

    t = str(row["text"]).lower()
    has_url = bool(URL_RE.search(t))

    # priority: most harmful first
    if _contains(t, FRAUD):
        return "Fraud"
    if _contains(t, PHISH) or (has_url and row.get("smish_seed", 0) == 1):
        return "Phishing"
    if _contains(t, PROMO):
        return "Promotion"
    # smishing-corpus fallthrough -> treat as Phishing prior
    if row.get("smish_seed", 0) == 1:
        return "Phishing"
    return "Spam"


def main() -> None:
    df = pd.read_csv(RAW)
    if "smish_seed" not in df:
        df["smish_seed"] = 0
    df["label5"] = df.apply(classify, axis=1)

    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {OUT}\n")
    print("=== 5-class distribution ===")
    print(df["label5"].value_counts())
    print("\n=== binary vs 5-class crosstab ===")
    print(pd.crosstab(df["binary_label"], df["label5"]))
    print("\n=== sample per class ===")
    for c in ["Normal", "Promotion", "Spam", "Phishing", "Fraud"]:
        print(f"\n[{c}]")
        for m in df[df["label5"] == c]["text"].head(3):
            print("  •", str(m)[:95])


if __name__ == "__main__":
    main()
