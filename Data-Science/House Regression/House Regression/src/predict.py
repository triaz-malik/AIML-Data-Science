"""Inference: load trained artifacts and score new rows.

Used by `app/streamlit_app.py` and as a CLI for batch scoring:
    python -m src.predict --csv path/to/houses.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import MODELS_DIR
from src.preprocessing import build_row, engineer, preprocess

ARTIFACT_PATH = MODELS_DIR / "model.pkl"


def load_artifacts(path: Path = ARTIFACT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifacts not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_row(user_inputs: dict, art: dict) -> float:
    """Predict price for a single row described as a {field: value} dict."""
    df = build_row(user_inputs, art)
    base = art["base_models"]
    meta = art["meta_model"]
    base_preds = np.column_stack([m.predict(df) for m in base.values()])
    log_pred = meta.predict(base_preds)[0]
    return float(np.expm1(log_pred))


def predict_batch(df: pd.DataFrame, art: dict) -> np.ndarray:
    """Predict prices for a dataframe shaped like the raw test set
    (must contain the same columns as competition test.csv minus SalePrice)."""
    df = df.copy()
    if "Id" not in df.columns:
        df["Id"] = np.arange(len(df))
    train_csv = MODELS_DIR.parent / "data" / "raw" / "train.csv"
    train = pd.read_csv(train_csv)
    _, _, all_data, ntrain, _ = preprocess(train, df.assign(SalePrice=np.nan)
                                           if "SalePrice" not in df else df.drop(columns=["SalePrice"]))
    all_data, _ = engineer(all_data, train)
    X_test = all_data[ntrain:].reindex(columns=art["fe"]["feature_columns"], fill_value=0)
    base = art["base_models"]
    meta = art["meta_model"]
    base_preds = np.column_stack([m.predict(X_test) for m in base.values()])
    log_pred = meta.predict(base_preds)
    return np.clip(np.expm1(log_pred), 0, None)


def main():
    ap = argparse.ArgumentParser(description="Batch prediction from a CSV.")
    ap.add_argument("--csv", required=True, help="Input CSV (test-shaped).")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV path.")
    args = ap.parse_args()

    art = load_artifacts()
    df = pd.read_csv(args.csv)
    preds = predict_batch(df, art)
    out = pd.DataFrame({"Id": df.get("Id", np.arange(len(df))), "SalePrice": preds})
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()
