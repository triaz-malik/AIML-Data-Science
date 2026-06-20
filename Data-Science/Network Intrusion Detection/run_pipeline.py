"""Run the full IDS pipeline end-to-end: EDA -> train -> evaluate -> SHAP -> predict -> report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import eda, train_models, evaluate, explain_shap, predict_test, make_report  # noqa: E402

STEPS = [
    ("EDA", eda.run),
    ("Train models", train_models.main),
    ("Evaluate", evaluate.run),
    ("SHAP explainability", explain_shap.run),
    ("Predict unlabelled test", predict_test.run),
    ("PDF report", make_report.run),
]

if __name__ == "__main__":
    for i, (name, fn) in enumerate(STEPS, 1):
        print(f"\n{'='*70}\n[{i}/{len(STEPS)}] {name}\n{'='*70}")
        fn()
    print("\n✅ Pipeline complete. See models/, reports/ and outputs/.")
