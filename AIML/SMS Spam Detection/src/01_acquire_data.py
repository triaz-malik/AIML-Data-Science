"""
Phase 1 - Data Acquisition
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

Builds a unified raw corpus of SMS messages from multiple public sources:
  1. UCI SMS Spam Collection (5,574 msgs, ham/spam)         -> primary
  2. SMS Smishing / phishing corpus (banking/OTP/delivery scams) -> fraud coverage

Output: data/raw/sms_raw.csv  with columns [text, binary_label, source]
        binary_label in {ham, spam}  (true ground truth)

The fine-grained 5-class labels (Normal/Promotion/Spam/Phishing/Fraud) are
derived later in Phase 2 via weak supervision, NOT here.
"""
from __future__ import annotations
import io
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (research; sms-spam-detection)"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------
# Source 1: UCI SMS Spam Collection
# ---------------------------------------------------------------------------
def load_uci() -> pd.DataFrame:
    # Try HuggingFace datasets first (most reliable, cached)
    try:
        from datasets import load_dataset
        ds = load_dataset("ucirvine/sms_spam", split="train")
        df = ds.to_pandas()
        # columns: sms, label (0=ham,1=spam)
        df = df.rename(columns={"sms": "text"})
        df["binary_label"] = df["label"].map({0: "ham", 1: "spam"})
        df = df[["text", "binary_label"]].copy()
        df["source"] = "uci_sms"
        df["smish_seed"] = 0
        print(f"[uci] loaded {len(df)} via HuggingFace")
        return df
    except Exception as e:
        print(f"[uci] HF load failed ({e}); falling back to direct UCI zip")

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    blob = _get(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if "SMSSpamCollection" in n][0]
        raw = z.read(name).decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        if "\t" in line:
            lab, txt = line.split("\t", 1)
            rows.append((txt, "ham" if lab == "ham" else "spam"))
    df = pd.DataFrame(rows, columns=["text", "binary_label"])
    df["source"] = "uci_sms"
    df["smish_seed"] = 0
    print(f"[uci] loaded {len(df)} via direct download")
    return df


# ---------------------------------------------------------------------------
# Source 2: Smishing / phishing SMS corpus
# Tries a few public mirrors; if none reachable, returns empty (non-fatal).
# ---------------------------------------------------------------------------
def load_smishing(cap: int = 3000) -> pd.DataFrame:
    """Mishra & Soni combined smishing corpus (Nigerian-telecom heavy).
    Noisy: we keep only smishing-labelled, de-fragmented messages as a
    phishing/fraud SEED, capped so it does not drown the clean UCI data.
    """
    url = "https://raw.githubusercontent.com/shaghayegh-hp/Smishing_Dataset/main/Combined-Labeled-Dataset.csv"
    try:
        df = pd.read_csv(io.BytesIO(_get(url, timeout=90)))
    except Exception as e:
        print(f"[smish] source unreachable ({e}); continuing with UCI only")
        return pd.DataFrame(columns=["text", "binary_label", "source", "smish_seed"])

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"message": "text"})
    df["smishing label"] = pd.to_numeric(df.get("smishing label"), errors="coerce")
    df["text"] = df["text"].astype(str).str.strip()

    # keep smishing seeds only; drop fragments / OCR noise
    m = (
        (df["smishing label"] == 1)
        & (df["text"].str.len().between(30, 300))
        & (~df["text"].str.contains(r"[�]", regex=True))  # drop replacement-char rows
    )
    out = df.loc[m, ["text"]].drop_duplicates().copy()
    # deterministic cap (sorted, no RNG) to keep runs reproducible
    out = out.sort_values("text").head(cap).reset_index(drop=True)
    out["binary_label"] = "spam"
    out["source"] = "smishing"
    out["smish_seed"] = 1
    print(f"[smish] kept {len(out)} smishing seeds (capped at {cap})")
    return out


def main() -> None:
    parts = [load_uci(), load_smishing()]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)

    # clean
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[clean] {before} -> {len(df)} after dedupe")

    if "smish_seed" not in df:
        df["smish_seed"] = 0
    df["smish_seed"] = df["smish_seed"].fillna(0).astype(int)

    out = RAW / "sms_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {out}")
    print(df["binary_label"].value_counts())
    print(df["source"].value_counts())


if __name__ == "__main__":
    sys.exit(main())
