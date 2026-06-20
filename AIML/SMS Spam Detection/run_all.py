"""
End-to-end pipeline runner.
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

    python run_all.py            # full pipeline (incl. BERT/DistilBERT)
    python run_all.py --fast     # skip transformer fine-tuning
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
FAST = "--fast" in sys.argv

STEPS = [
    ("Phase 1  Acquire data",          ["src/01_acquire_data.py"]),
    ("Phase 2  Weak labelling",        ["src/02_label_5class.py"]),
    ("Phase 3  EDA",                   ["src/03_eda.py"]),
    ("Phase 4  Feature engineering",   ["src/04_feature_engineering.py"]),
    ("Phase 5  LogReg baseline",       ["src/05_model_logreg.py"]),
]
if not FAST:
    STEPS += [
        ("Phase 6a DistilBERT", ["src/06_model_transformers.py", "distilbert-base-uncased", "distilbert", "3", "2e-5"]),
        ("Phase 6b BERT",       ["src/06_model_transformers.py", "bert-base-uncased", "bert", "3", "2e-5"]),
    ]
STEPS += [
    ("Phase 7  Explainability + errors", ["src/07_explain_errors.py"]),
    ("Phase 8  Business + Power BI",     ["src/08_business_powerbi.py"]),
]


def main():
    for title, cmd in STEPS:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        r = subprocess.run([PY, *cmd], cwd=ROOT)
        if r.returncode != 0:
            print(f"!! step failed: {title}")
            sys.exit(r.returncode)
    print("\nPipeline complete. See outputs/ and reports/.")


if __name__ == "__main__":
    main()
