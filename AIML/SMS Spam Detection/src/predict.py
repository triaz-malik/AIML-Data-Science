"""
Inference CLI — score new SMS messages with the trained baseline model.
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

    python src/predict.py "Your account is suspended. Verify at http://bit.ly/x"
    python src/predict.py            # runs a built-in demo set

Outputs predicted class, confidence, full class probabilities, and a
recommended action (Allow / Review / Warn / Block).
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_features

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs" / "models" / "logreg.joblib"

ACTION = {"Normal": "ALLOW", "Promotion": "REVIEW", "Spam": "WARN",
          "Phishing": "BLOCK", "Fraud": "BLOCK"}

DEMO = [
    "Hey, are we still on for lunch tomorrow?",
    "URGENT! You have WON a £1000 cash prize! Call 09061701461 to claim now.",
    "Your bank account has been suspended. Verify your details at http://secure-bank.ng/login",
    "50% OFF everything this weekend! Reply STOP to unsubscribe.",
    "Your OTP is 482910. Do not share it. Dear customer, confirm to reactivate your card.",
]


def predict(messages):
    model = joblib.load(MODEL)
    df = pd.DataFrame({"text": [str(m) for m in messages]})
    df = add_features(df, "text")
    proba = model.predict_proba(df)
    classes = list(model.classes_)
    out = []
    for i, msg in enumerate(messages):
        p = proba[i]
        cls = classes[int(p.argmax())]
        out.append({
            "text": msg, "pred": cls, "confidence": float(p.max()),
            "action": ACTION.get(cls, "REVIEW"),
            "proba": {c: round(float(v), 3) for c, v in zip(classes, p)},
        })
    return out


def main():
    msgs = sys.argv[1:] or DEMO
    for r in predict(msgs):
        print("\n" + "-" * 70)
        print(f"MESSAGE : {r['text'][:90]}")
        print(f"PREDICT : {r['pred']}  (confidence {r['confidence']:.2f})  ->  {r['action']}")
        probs = sorted(r["proba"].items(), key=lambda kv: -kv[1])
        print("PROBS   : " + "  ".join(f"{c}={v:.2f}" for c, v in probs))


if __name__ == "__main__":
    main()
