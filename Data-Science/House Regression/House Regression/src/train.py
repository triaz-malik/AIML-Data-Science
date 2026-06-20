"""Full training pipeline: EDA -> preprocess -> FE -> 7 base models -> Ridge stack -> SHAP.

Run with:
    python -m src.train                 # full pipeline (default 30 Optuna trials/model)
    python -m src.train --trials 100    # spec-faithful 100 trials per model
    python -m src.train --quick         # skip Optuna, use reference hyperparameters
"""
from __future__ import annotations

import argparse
import warnings
import zipfile

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.data_loader import MODELS_DIR, ROOT, load_data
from src.evaluate import (
    cv_rmsle,
    eda_correlation,
    eda_key_features,
    eda_missing,
    eda_neighborhood,
    eda_target,
    plot_diagnostics,
    plot_feature_importance,
    plot_model_comparison,
    plot_shap,
    plot_submission_check,
    rmsle_cv,
)
from src.preprocessing import engineer, preprocess

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)


def banner(t: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{t}\n{bar}")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def baseline_models() -> dict:
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=12, random_state=SEED))
    lasso = make_pipeline(RobustScaler(),
                          Lasso(alpha=0.0005, max_iter=10000, random_state=SEED))
    enet = make_pipeline(RobustScaler(),
                         ElasticNet(alpha=0.0005, l1_ratio=0.9,
                                    max_iter=10000, random_state=SEED))
    gbm = GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4,
        max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
        loss="huber", random_state=SEED)
    return {"Ridge": ridge, "Lasso": lasso, "ElasticNet": enet, "GBM": gbm}


def tune_xgb(X, y, n_trials):
    def obj(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 5.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            n_estimators=1500, random_state=SEED, n_jobs=-1, verbosity=0,
        )
        return cv_rmsle(xgb.XGBRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGB best RMSLE: {study.best_value:.5f}")
    return xgb.XGBRegressor(**study.best_params, n_estimators=1500,
                            random_state=SEED, n_jobs=-1, verbosity=0)


def tune_lgb(X, y, n_trials):
    def obj(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 5, 64),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            n_estimators=1500, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        return cv_rmsle(lgb.LGBMRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  LGB best RMSLE: {study.best_value:.5f}")
    return lgb.LGBMRegressor(**study.best_params, n_estimators=1500,
                             random_state=SEED, n_jobs=-1, verbose=-1)


def tune_cat(X, y, n_trials):
    def obj(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            iterations=1500, random_seed=SEED, verbose=0,
            allow_writing_files=False, bootstrap_type="Bernoulli",
        )
        return cv_rmsle(CatBoostRegressor(**params), X, y)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  CAT best RMSLE: {study.best_value:.5f}")
    return CatBoostRegressor(**study.best_params, iterations=1500,
                             random_seed=SEED, verbose=0,
                             allow_writing_files=False, bootstrap_type="Bernoulli")


def reference_boosters():
    """Reference hyperparameters for --quick mode."""
    xgb_m = xgb.XGBRegressor(
        colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
        max_depth=3, min_child_weight=1.7817, n_estimators=2200,
        reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213,
        random_state=SEED, n_jobs=-1, verbosity=0)
    lgb_m = lgb.LGBMRegressor(
        objective="regression", num_leaves=5, learning_rate=0.05,
        n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5,
        feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9,
        min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
        random_state=SEED, verbose=-1)
    cat_m = CatBoostRegressor(
        iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        random_seed=SEED, verbose=0, allow_writing_files=False)
    return xgb_m, lgb_m, cat_m


# --------------------------------------------------------------------------- #
# Stacking
# --------------------------------------------------------------------------- #
def stacked_ridge(base_models: dict, X_tr, y_tr, X_te, n_folds: int = 5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n_models = len(base_models)
    oof_preds = np.zeros((len(X_tr), n_models))
    test_preds = np.zeros((len(X_te), n_models))

    for m_idx, (name, model) in enumerate(base_models.items()):
        print(f"  Training {name} ...", end=" ", flush=True)
        test_fp = np.zeros((len(X_te), n_folds))
        for fold, (tri, vali) in enumerate(kf.split(X_tr, y_tr)):
            Xt, Xv = X_tr.iloc[tri], X_tr.iloc[vali]
            yt = y_tr.iloc[tri]
            model.fit(Xt, yt)
            oof_preds[vali, m_idx] = model.predict(Xv)
            test_fp[:, fold] = model.predict(X_te)
        test_preds[:, m_idx] = test_fp.mean(axis=1)
        print("done")

    meta = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 25.0])
    meta.fit(oof_preds, y_tr)
    print(f"  RidgeCV alpha: {meta.alpha_:.2f} | weights: "
          + ", ".join(f"{n}={w:.2f}" for n, w in zip(base_models.keys(), meta.coef_)))

    oof_blend = meta.predict(oof_preds)
    test_blend = meta.predict(test_preds)
    rmsle = np.sqrt(mean_squared_error(y_tr, oof_blend))
    return oof_blend, test_blend, rmsle, meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="House Prices regression pipeline")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per booster (default 30, spec 100)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip Optuna; use reference hyperparameters")
    args = parser.parse_args()

    banner("LOAD")
    train_raw, test_raw = load_data()

    banner("ACT I — EDA")
    eda_target(train_raw)
    eda_missing(train_raw, test_raw)
    eda_correlation(train_raw)
    eda_key_features(train_raw)
    eda_neighborhood(train_raw)

    banner("ACT II — PREPROCESS")
    train, y, all_data, ntrain, test_id = preprocess(train_raw, test_raw)

    banner("ACT III — FEATURE ENGINEERING")
    all_data, fe_artifacts = engineer(all_data, train, n_clusters=5)
    X_train = all_data[:ntrain].copy()
    X_test = all_data[ntrain:].copy().reindex(columns=X_train.columns, fill_value=0)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    banner("ACT IV — BASELINES (5-fold CV)")
    results = {}
    for name, model in baseline_models().items():
        sc = rmsle_cv(model, X_train, y); results[name] = sc
        print(f"  {name:11s}: {sc.mean():.5f} +/- {sc.std():.5f}")

    banner("ACT IV — BOOSTER MODELS")
    if args.quick:
        print("  --quick mode: using reference hyperparameters (no Optuna)")
        xgb_model, lgb_model, cat_model = reference_boosters()
    else:
        print(f"  Running Optuna ({args.trials} trials per model)...")
        print("  [XGBoost]"); xgb_model = tune_xgb(X_train, y, args.trials)
        print("  [LightGBM]"); lgb_model = tune_lgb(X_train, y, args.trials)
        print("  [CatBoost]"); cat_model = tune_cat(X_train, y, args.trials)

    for name, m in [("XGBoost", xgb_model), ("LightGBM", lgb_model), ("CatBoost", cat_model)]:
        sc = rmsle_cv(m, X_train, y); results[name] = sc
        print(f"  {name:11s}: {sc.mean():.5f} +/- {sc.std():.5f}")

    plot_model_comparison(results)

    xgb_model.fit(X_train, y); lgb_model.fit(X_train, y)
    plot_feature_importance(xgb_model, lgb_model, X_train)

    banner("ACT V — RIDGE META-STACKER")
    base = baseline_models()
    base.update({"XGBoost": xgb_model, "LightGBM": lgb_model, "CatBoost": cat_model})
    oof, test_blend, ens_rmsle, meta = stacked_ridge(base, X_train, y, X_test)
    print(f"  Stacker OOF RMSLE: {ens_rmsle:.5f}")
    plot_diagnostics(y, oof, ens_rmsle)

    banner("ACT VI — SHAP INTERPRETABILITY")
    shap_top = plot_shap(xgb_model, X_train)
    print("Top 5 by SHAP magnitude:")
    for f, v in shap_top.head(5).items():
        print(f"  {f:25s}: {v:.5f}")

    banner("SUBMISSION")
    final = np.clip(np.expm1(test_blend), 0, None)
    submission = pd.DataFrame({"Id": test_id.values, "SalePrice": final})
    sub_csv = ROOT / "submission.csv"
    submission.to_csv(sub_csv, index=False)
    with zipfile.ZipFile(ROOT / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_csv, arcname="submission.csv")
    print(f"submission.csv: {len(submission)} rows")
    print(f"  range: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
    print(f"  mean : ${submission['SalePrice'].mean():,.0f}")
    plot_submission_check(train_raw, submission)

    banner("ARTIFACTS — for Streamlit app")
    train_defaults = train_raw.median(numeric_only=True).to_dict()
    train_modes = {c: train_raw[c].mode()[0]
                   for c in train_raw.select_dtypes(include="object").columns}
    artifacts = {
        "meta_model": meta,
        "base_models": base,
        "fe": fe_artifacts,
        "train_medians": train_defaults,
        "train_modes": train_modes,
        "train_min_year": int(train_raw["YearBuilt"].min()),
        "train_max_year": int(train_raw["YrSold"].max()),
        "neighborhoods": sorted(train_raw["Neighborhood"].unique().tolist()),
    }
    art_path = MODELS_DIR / "model.pkl"
    joblib.dump(artifacts, art_path)
    print(f"  saved {art_path.relative_to(ROOT)}  ({art_path.stat().st_size/1e6:.1f} MB)")

    banner("SCORECARD")
    all_results = {**{k: v.mean() for k, v in results.items()}, "Stacker": ens_rmsle}
    for name, sc in sorted(all_results.items(), key=lambda kv: kv[1]):
        bar = "#" * int((0.15 - sc) * 1000)
        print(f"  {name:12s}: {sc:.5f}  {bar}")


if __name__ == "__main__":
    main()
