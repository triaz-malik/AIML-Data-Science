"""Generate predictions for the unlabelled Test_data.csv using the best model."""
from __future__ import annotations

import joblib
import pandas as pd

import config as C
from preprocessing import load_unlabelled_test


def run():
    if not C.BEST_MODEL_PKL.exists():
        raise SystemExit("best_model.pkl not found. Run train_models.py first.")
    model = joblib.load(C.BEST_MODEL_PKL)

    df = load_unlabelled_test()
    # align columns: drop any constant cols the model didn't see is handled by
    # the ColumnTransformer (remainder='drop'); it selects by name.
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["attack_probability"] = proba.round(4)
    out["prediction"] = pd.Series(pred).map({0: "normal", 1: "anomaly"}).values
    out["risk_score"] = (proba * 100).round(1)

    path = C.OUTPUTS_DIR / "test_predictions.csv"
    out.to_csv(path, index=False)
    n_attack = int(pred.sum())
    print(f"[predict] wrote {path} | {len(out)} rows | "
          f"flagged {n_attack} ({n_attack/len(out):.1%}) as anomaly")


if __name__ == "__main__":
    run()
