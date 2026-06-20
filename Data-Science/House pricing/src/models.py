"""Model definitions, hyperparameter search spaces, and CV evaluation."""
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from . import config as C


def get_models():
    """Return baseline estimators keyed by name.

    Linear regression is wrapped in a scaler; tree models are scale-invariant.
    Predictions/targets are on the log1p scale throughout.
    """
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=C.RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            random_state=C.RANDOM_STATE, n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
    }


def get_search_spaces():
    """RandomizedSearchCV parameter distributions per tunable model."""
    return {
        "RandomForest": {
            "n_estimators": randint(200, 800),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 6),
            "max_features": uniform(0.3, 0.6),
        },
        "XGBoost": {
            "n_estimators": randint(300, 1200),
            "learning_rate": uniform(0.01, 0.1),
            "max_depth": randint(2, 6),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0.0, 0.5),
            "reg_lambda": uniform(0.0, 2.0),
        },
        "LightGBM": {
            "n_estimators": randint(300, 1200),
            "learning_rate": uniform(0.01, 0.1),
            "num_leaves": randint(15, 60),
            "max_depth": randint(3, 12),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_samples": randint(5, 40),
        },
    }


def _set_n_jobs(estimator, n):
    """Best-effort set inner parallelism on an estimator or pipeline.

    Crucial for avoiding CPU oversubscription: when an outer
    RandomizedSearchCV / cross_val_score runs with n_jobs=-1, the *inner*
    estimators must stay single-threaded, otherwise dozens of processes
    fight over the same cores and everything crawls.
    """
    targets = {k: n for k in estimator.get_params()
               if k == "n_jobs" or k.endswith("__n_jobs")}
    if targets:
        estimator.set_params(**targets)
    return estimator


def rmse(y_true, y_pred) -> float:
    """Root-mean-squared error (operates on whatever scale it is given)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    """Compute R2, MAE and RMSE on the log scale (the Kaggle metric)."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }


def cv_rmse(model, X, y) -> tuple[float, float]:
    """5-fold CV RMSE (mean, std) on the log target."""
    kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    # Clone + single-thread the inner model so only the CV loop parallelizes.
    inner = _set_n_jobs(clone(model), 1)
    scores = cross_val_score(
        inner, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1
    )
    scores = -scores
    return float(scores.mean()), float(scores.std())


def tune(name, base_model, X, y):
    """Run RandomizedSearchCV for a tunable model; return the best estimator."""
    space = get_search_spaces().get(name)
    if space is None:
        return base_model  # nothing to tune (e.g. LinearRegression)

    kf = KFold(n_splits=3, shuffle=True, random_state=C.RANDOM_STATE)
    # Inner model single-threaded; the search itself parallelizes (no nesting).
    inner = _set_n_jobs(clone(base_model), 1)
    search = RandomizedSearchCV(
        inner, space, n_iter=C.N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error", cv=kf,
        n_jobs=-1, random_state=C.RANDOM_STATE, verbose=0,
    )
    search.fit(X, y)
    print(f"    best CV RMSE={-search.best_score_:.5f}")
    # Restore full parallelism for the later single-model fits on full data.
    return _set_n_jobs(search.best_estimator_, -1)
