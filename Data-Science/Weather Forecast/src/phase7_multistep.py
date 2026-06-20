"""Phase 7 - Multi-Step Forecasting.

Direct multi-output forecasting of the next H hours of temperature for
H in {24, 48, 168} (1 day, 2 days, 7 days). The model emits the whole horizon
in one shot (a Dense(H) head), which avoids error compounding from recursive
one-step rollouts. Uses the best architecture/config from Phases 5-6.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import config as C
import modeling as M

tf.random.set_seed(42)
np.random.seed(42)

HORIZONS = [24, 48, 168]
DEFAULTS = {"arch": "GRU", "window": 48, "units": 64, "dropout": 0.2,
            "lr": 1e-3, "batch": 128}


def load_best():
    cfg = dict(DEFAULTS)
    p = f"{C.TABLE_DIR}/phase6_best_config.json"
    if os.path.exists(p):
        with open(p) as f:
            b = json.load(f)
        for k in ("arch", "window", "units", "dropout", "lr", "batch"):
            if k in b and b[k] is not None:
                cfg[k] = type(DEFAULTS[k])(b[k])
    return cfg


def build(window, n_features, horizon, cfg):
    inp = layers.Input(shape=(window, n_features))
    arch = cfg["arch"]
    if arch == "LSTM":
        x = layers.LSTM(cfg["units"])(inp)
    elif arch == "BiLSTM":
        x = layers.Bidirectional(layers.LSTM(cfg["units"]))(inp)
    else:
        x = layers.GRU(cfg["units"])(inp)
    x = layers.Dropout(cfg["dropout"])(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(horizon)(x)          # multi-output head
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss="mse",
              metrics=["mae"])
    return m


def main():
    cfg = load_best()
    print(f"[Phase 7] Multi-step config: {cfg}")
    df = M.load_feature_table()

    results = {}
    sample_store = {}

    for H in HORIZONS:
        print(f"\n[Phase 7] Horizon = {H}h ...")
        s = M.sequence_splits(df, window=cfg["window"], horizon=H, multi=True)
        model = build(cfg["window"], s["n_features"], H, cfg)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                     restore_best_weights=True)
        rl = callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
        model.fit(s["X_train"], s["y_train"],
                  validation_data=(s["X_val"], s["y_val"]),
                  epochs=30, batch_size=cfg["batch"], callbacks=[es, rl],
                  verbose=2)

        pred = M.inv_target(model.predict(s["X_test"], verbose=0),
                            s["t_mu"], s["t_sd"])
        true = M.inv_target(s["y_test"], s["t_mu"], s["t_sd"])

        # Overall (flattened) metrics across the whole horizon.
        overall = M.metrics(true, pred)
        # Per-step RMSE to show error growth with lead time.
        per_step_rmse = np.sqrt(np.mean((true - pred) ** 2, axis=0)).round(4)
        results[f"{H}h"] = {
            **overall,
            "per_step_rmse_first": float(per_step_rmse[0]),
            "per_step_rmse_last": float(per_step_rmse[-1]),
        }
        model.save(f"{C.MODEL_DIR}/multistep_{H}h.keras")

        # Store one example forecast (last test window) + per-step rmse.
        sample_store[f"{H}h_true"] = true[-1]
        sample_store[f"{H}h_pred"] = pred[-1]
        sample_store[f"{H}h_perstep_rmse"] = per_step_rmse
        print(f"  {H}h:", results[f"{H}h"])

    with open(f"{C.TABLE_DIR}/phase7_multistep_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(f"{C.TABLE_DIR}/phase7_samples.npz", **sample_store)

    print("\n[Phase 7] Multi-step results:")
    print(pd.DataFrame(results).T)
    print("[Phase 7] Done.")


if __name__ == "__main__":
    main()
