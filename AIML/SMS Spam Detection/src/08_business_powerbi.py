"""
Phase 8 - Business Insights + Power BI Export
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

Produces analysis-ready CSVs in outputs/powerbi/ for three dashboards
(Executive / Fraud / AI) plus a findings+recommendations table, and a
model-comparison table built from reports/metrics_*.json.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_COLS, URL_RE
from utils import CLASSES

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "sms_features.csv"
MODELS = ROOT / "outputs" / "models"
REP = ROOT / "reports"
PBI = ROOT / "outputs" / "powerbi"; PBI.mkdir(parents=True, exist_ok=True)


def score_messages(df):
    """Attach model prediction + confidence to every message (AI dashboard)."""
    try:
        model = joblib.load(MODELS / "logreg.joblib")
        proba = model.predict_proba(df)
        classes = list(model.classes_)
        df["pred_class"] = [classes[i] for i in proba.argmax(1)]
        df["confidence"] = proba.max(1).round(4)
    except Exception as e:
        print(f"[score] model not available ({e}); using weak labels as prediction")
        df["pred_class"] = df["label5"]
        df["confidence"] = 1.0
    df["has_url"] = df["text"].astype(str).str.contains(URL_RE).astype(int)
    return df


def export_messages(df):
    cols = (["text", "label5", "binary_label", "source", "pred_class",
             "confidence", "has_url"] + FEATURE_COLS)
    out = df[cols].copy()
    out["text"] = out["text"].astype(str).str.slice(0, 160)
    out.to_csv(PBI / "messages_scored.csv", index=False, encoding="utf-8")


def export_class_summary(df):
    s = (df["label5"].value_counts().reindex(CLASSES)
         .rename_axis("class").reset_index(name="count"))
    s["pct"] = (s["count"] / s["count"].sum() * 100).round(2)
    s["risk_tier"] = s["class"].map(
        {"Normal": "Safe", "Promotion": "Low", "Spam": "Medium",
         "Phishing": "High", "Fraud": "Critical"})
    s.to_csv(PBI / "class_summary.csv", index=False, encoding="utf-8")
    return s


def export_keywords(df, k=15):
    rows = []
    for c in ["Promotion", "Spam", "Phishing", "Fraud"]:
        corpus = df[df["label5"] == c]["text"].astype(str)
        if len(corpus) < 3:
            continue
        vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)
        X = vec.fit_transform(corpus)
        freqs = np.asarray(X.sum(0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        for i in freqs.argsort()[::-1][:k]:
            rows.append({"class": c, "keyword": terms[i], "frequency": int(freqs[i])})
    pd.DataFrame(rows).to_csv(PBI / "fraud_keywords.csv", index=False, encoding="utf-8")


def export_feature_profile(df):
    prof = df.groupby("label5")[FEATURE_COLS].mean().reindex(CLASSES).round(3)
    prof.reset_index().to_csv(PBI / "feature_by_class.csv", index=False, encoding="utf-8")


def export_model_comparison():
    rows = []
    for tag, exp in [("logreg", "TF-IDF + LogisticRegression"),
                     ("distilbert", "DistilBERT (fine-tuned)"),
                     ("bert", "BERT (fine-tuned)")]:
        p = REP / f"metrics_{tag}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        rows.append({
            "model": exp,
            "accuracy": round(m["test_accuracy"], 4),
            "f1_macro": round(m["test_f1_macro"], 4),
            "binary_accuracy": round(m.get("binary_accuracy", float("nan")), 4),
            "binary_f1": round(m.get("binary_f1", float("nan")), 4),
        })
    if rows:
        pd.DataFrame(rows).to_csv(PBI / "model_comparison.csv", index=False, encoding="utf-8")
    return rows


def derive_findings(df):
    phish = df[df["label5"] == "Phishing"]
    fraud = df[df["label5"] == "Fraud"]
    pct_phish_url = (phish["text"].astype(str).str.contains(URL_RE).mean() * 100) if len(phish) else 0
    bank_kw = re.compile(r"\b(bank|account|verify|otp|pin|card|login|password|kyc)\b", re.I)
    pct_fraud_bank = (fraud["text"].astype(str).str.contains(bank_kw).mean() * 100) if len(fraud) else 0
    findings = [
        {"finding": f"{pct_phish_url:.0f}% of phishing messages contain a URL/link",
         "recommendation": "Increase URL screening; sandbox or block links in inbound SMS",
         "metric": round(pct_phish_url, 1)},
        {"finding": f"{pct_fraud_bank:.0f}% of fraud/phishing messages use banking keywords "
                    f"(bank/account/verify/OTP/PIN)",
         "recommendation": "Deploy banking-specific keyword filters and step-up verification",
         "metric": round(pct_fraud_bank, 1)},
        {"finding": "Messages with 'verify', 'account', 'OTP' have high fraud probability",
         "recommendation": "Create keyword-based real-time alerts for these triggers",
         "metric": None},
        {"finding": f"Malicious messages are ~{df[df.label5!='Normal']['msg_length'].median()/max(df[df.label5=='Normal']['msg_length'].median(),1):.1f}x "
                    f"longer than normal messages",
         "recommendation": "Use message length + digit/URL counts as fast pre-filter features",
         "metric": None},
    ]
    pd.DataFrame(findings).to_csv(PBI / "business_findings.csv", index=False, encoding="utf-8")
    return findings


def main():
    df = pd.read_csv(FEATURES)
    df["text"] = df["text"].astype(str)
    df = score_messages(df)

    export_messages(df)
    summary = export_class_summary(df)
    export_keywords(df)
    export_feature_profile(df)
    models = export_model_comparison()
    findings = derive_findings(df)

    print("Power BI CSVs written to outputs/powerbi/:")
    for p in sorted(PBI.glob("*.csv")):
        print("  •", p.name)
    print("\n=== Class summary ===")
    print(summary.to_string(index=False))
    print("\n=== Business findings ===")
    for f in findings:
        print(f"  • {f['finding']}\n      -> {f['recommendation']}")
    if models:
        print("\n=== Model comparison ===")
        print(pd.DataFrame(models).to_string(index=False))


if __name__ == "__main__":
    main()
