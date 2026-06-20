"""End-to-end House Price Prediction pipeline.

Run:  python main.py            (full pipeline incl. hyperparameter tuning)
      python main.py --fast     (skip tuning, use baseline models)
      python main.py --no-eda   (skip figure generation)
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C
from src import data_prep, feature_engineering as fe, eda, models, explain

warnings.filterwarnings("ignore")


def build_features():
    """Load, clean, engineer and encode features. Returns X, y, X_test, ids."""
    train_raw, test_raw = data_prep.load_data()
    print(f"Loaded train={train_raw.shape}, test={test_raw.shape}")

    train_raw = data_prep.remove_outliers(train_raw)

    y = np.log1p(train_raw[C.TARGET])
    test_ids = test_raw[C.ID_COL]

    train_df = train_raw.drop(columns=[C.TARGET, C.ID_COL])
    test_df = test_raw.drop(columns=[C.ID_COL])

    # Clean + engineer each frame with identical logic.
    frames = []
    for df in (train_df, test_df):
        df = data_prep.impute_missing(df)
        df = data_prep.cast_categorical_codes(df)
        df = fe.add_engineered_features(df)
        frames.append(df)
    train_df, test_df = frames

    X, X_test = fe.encode_and_align(train_df, test_df)
    print(f"Feature matrix: X={X.shape}, X_test={X_test.shape}\n")
    return X, y, X_test, test_ids


def run_models(X, y, fast: bool):
    """Train/evaluate every model on a hold-out split + 5-fold CV."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE
    )

    results = []
    fitted = {}
    for name, model in models.get_models().items():
        print(f"[{name}]")
        if not fast and name != "LinearRegression":
            print("  tuning (RandomizedSearchCV)...")
            model = models.tune(name, model, X_tr, y_tr)

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        metrics = models.evaluate(y_val, val_pred)
        cv_mean, cv_std = models.cv_rmse(model, X, y)

        results.append({
            "Model": name,
            "R2": round(metrics["R2"], 4),
            "RMSE_holdout": round(metrics["RMSE"], 5),
            "MAE": round(metrics["MAE"], 5),
            "CV_RMSE": round(cv_mean, 5),
            "CV_std": round(cv_std, 5),
        })
        fitted[name] = model
        print(f"  holdout R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.5f}  "
              f"CV_RMSE={cv_mean:.5f}+/-{cv_std:.5f}\n")

    leaderboard = (pd.DataFrame(results)
                   .sort_values("CV_RMSE")
                   .reset_index(drop=True))
    return leaderboard, fitted, (X_val, y_val)


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument("--fast", action="store_true",
                        help="skip hyperparameter tuning")
    parser.add_argument("--no-eda", action="store_true",
                        help="skip EDA figure generation")
    args = parser.parse_args()

    if not args.no_eda:
        train_raw, _ = data_prep.load_data()
        eda.run_eda(train_raw)

    X, y, X_test, test_ids = build_features()
    leaderboard, fitted, (X_val, y_val) = run_models(X, y, fast=args.fast)

    print("=" * 60)
    print("MODEL LEADERBOARD (sorted by CV RMSE on log target)")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    leaderboard.to_csv(C.OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved leaderboard -> outputs/model_comparison.csv")

    # Best model = lowest CV RMSE.
    best_name = leaderboard.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Diagnostics + explainability on the best model.
    explain.plot_predicted_vs_actual(y_val, best_model.predict(X_val), best_name)
    if best_name in ("RandomForest", "XGBoost", "LightGBM"):
        sample = X_val.sample(min(300, len(X_val)), random_state=C.RANDOM_STATE)
        explain.shap_summary(best_model, sample, best_name)

    # Refit best model on ALL training data, then predict the test set.
    print(f"\nRefitting {best_name} on full training data...")
    best_model.fit(X, y)
    joblib.dump(best_model, C.MODEL_DIR / f"{best_name}.joblib")
    print(f"Saved model -> outputs/models/{best_name}.joblib")

    test_pred = np.expm1(best_model.predict(X_test))
    submission = pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_pred})
    submission.to_csv(C.SUBMISSION_CSV, index=False)
    print(f"Saved submission -> outputs/submission.csv  ({len(submission)} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
