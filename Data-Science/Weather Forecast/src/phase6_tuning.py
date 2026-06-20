"""Phase 6 - Hyperparameter Tuning.

Lightweight, reproducible grid/random search over the key knobs (window size,
hidden units, dropout, learning rate, batch size) for the best single-step
architecture. Runs on CPU, so the search space is intentionally compact and
each candidate trains with early stopping. Selection is by validation RMSE.
"""
from __future__ import annotations

import json
import itertools
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

# Compact search space (CPU-friendly). Window is handled by rebuilding windows.
SEARCH_SPACE = {
    "window": [12, 24, 48],
    "units": [32, 64, 128],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 5e-4],
    "batch": [128, 256],
}
# Cap the number of random candidates evaluated to keep runtime bounded.
N_CANDIDATES = 10
ARCH = "GRU"  # filled in dynamically from Phase 5 result if available


def build(window, n_features, units, dropout, lr, arch):
    inp = layers.Input(shape=(window, n_features))
    if arch == "LSTM":
        x = layers.LSTM(units)(inp)
    elif arch == "BiLSTM":
        x = layers.Bidirectional(layers.LSTM(units))(inp)
    else:
        x = layers.GRU(units)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return m


def pick_best_arch():
    """Use the winning single-step architecture from Phase 5 if available."""
    path = f"{C.TABLE_DIR}/phase5_dl_metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
        best = min(res, key=lambda k: res[k]["RMSE"])
        return {"LSTM": "LSTM", "GRU": "GRU", "BiLSTM": "BiLSTM"}[best]
    return ARCH


def main():
    arch = pick_best_arch()
    print(f"[Phase 6] Tuning architecture: {arch}")
    df = M.load_feature_table()

    # Build a reproducible random sample of candidates.
    keys = list(SEARCH_SPACE)
    all_combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_combos), size=min(N_CANDIDATES, len(all_combos)),
                     replace=False)
    candidates = [dict(zip(keys, all_combos[i])) for i in idx]

    trials = []
    # Cache sequence splits per window to avoid rebuilding repeatedly.
    seq_cache = {}

    for i, cfg in enumerate(candidates, 1):
        w = cfg["window"]
        if w not in seq_cache:
            seq_cache[w] = M.sequence_splits(df, window=w, horizon=1, multi=False)
        s = seq_cache[w]
        print(f"\n[Phase 6] Candidate {i}/{len(candidates)}: {cfg}")
        model = build(w, s["n_features"], cfg["units"], cfg["dropout"],
                      cfg["lr"], arch)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                     restore_best_weights=True)
        model.fit(s["X_train"], s["y_train"],
                  validation_data=(s["X_val"], s["y_val"]),
                  epochs=20, batch_size=cfg["batch"], callbacks=[es], verbose=2)
        val_pred = M.inv_target(model.predict(s["X_val"], verbose=0),
                                s["t_mu"], s["t_sd"])
        val_true = M.inv_target(s["y_val"], s["t_mu"], s["t_sd"])
        vm = M.metrics(val_true, val_pred)
        trials.append({**cfg, "arch": arch, "val_RMSE": vm["RMSE"],
                       "val_MAE": vm["MAE"]})
        print(f"  -> val RMSE={vm['RMSE']} MAE={vm['MAE']}")

    trials_df = pd.DataFrame(trials).sort_values("val_RMSE").reset_index(drop=True)
    trials_df.to_csv(f"{C.TABLE_DIR}/phase6_tuning_trials.csv", index=False)
    best = trials_df.iloc[0].to_dict()
    with open(f"{C.TABLE_DIR}/phase6_best_config.json", "w") as f:
        json.dump(best, f, indent=2, default=float)

    print("\n[Phase 6] Tuning trials (sorted by val RMSE):")
    print(trials_df.to_string(index=False))
    print(f"\n[Phase 6] Best config: {best}")
    print("[Phase 6] Done.")


if __name__ == "__main__":
    main()
