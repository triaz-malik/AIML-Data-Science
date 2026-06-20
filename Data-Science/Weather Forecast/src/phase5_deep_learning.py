"""Phase 5 - Deep Learning Models: LSTM, GRU, Bi-LSTM (single-step).

Each model consumes a 24-hour multivariate window and predicts the next hour's
temperature. Trained on CPU with early stopping. Metrics are reported on the
held-out chronological test set (same split as the baselines).
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

EPOCHS = 30
BATCH = 128
UNITS = 64


def build_model(kind: str, window: int, n_features: int):
    inp = layers.Input(shape=(window, n_features))
    if kind == "LSTM":
        x = layers.LSTM(UNITS)(inp)
    elif kind == "GRU":
        x = layers.GRU(UNITS)(inp)
    elif kind == "BiLSTM":
        x = layers.Bidirectional(layers.LSTM(UNITS))(inp)
    else:
        raise ValueError(kind)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=["mae"])
    return model


def main():
    print("[Phase 5] Preparing sequence windows ...")
    df = M.load_feature_table()
    s = M.sequence_splits(df, window=M.WINDOW, horizon=1, multi=False)
    print(f"  train={s['X_train'].shape}  val={s['X_val'].shape}  test={s['X_test'].shape}")

    results, histories, preds_store = {}, {}, {}
    t_mu, t_sd = s["t_mu"], s["t_sd"]
    y_test_real = M.inv_target(s["y_test"], t_mu, t_sd)

    for kind in ["LSTM", "GRU", "BiLSTM"]:
        print(f"\n[Phase 5] Training {kind} ...")
        model = build_model(kind, s["window"], s["n_features"])
        es = callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                     restore_best_weights=True)
        rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=3, min_lr=1e-5)
        hist = model.fit(
            s["X_train"], s["y_train"],
            validation_data=(s["X_val"], s["y_val"]),
            epochs=EPOCHS, batch_size=BATCH, callbacks=[es, rl], verbose=2,
        )
        pred_real = M.inv_target(model.predict(s["X_test"], verbose=0), t_mu, t_sd)
        results[kind] = M.metrics(y_test_real, pred_real)
        preds_store[kind] = pred_real.ravel()
        histories[kind] = {"loss": hist.history["loss"],
                           "val_loss": hist.history["val_loss"]}
        model.save(f"{C.MODEL_DIR}/{kind.lower()}_singlestep.keras")
        print(f"  {kind}:", results[kind])

    with open(f"{C.TABLE_DIR}/phase5_dl_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{C.TABLE_DIR}/phase5_histories.json", "w") as f:
        json.dump(histories, f, indent=2)
    np.savez(f"{C.TABLE_DIR}/phase5_test_predictions.npz",
             y_test=y_test_real.ravel(), **preds_store)

    print("\n[Phase 5] Deep learning results:")
    print(pd.DataFrame(results).T[["MAE", "RMSE", "MAPE", "R2"]])
    print("[Phase 5] Done.")


if __name__ == "__main__":
    main()
