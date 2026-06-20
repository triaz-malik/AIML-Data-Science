"""Phase 10 - Streamlit Dashboard.

Interactive multi-step temperature forecasting. Pick a point in the test
period; the app feeds the preceding window to the trained multi-step models
and shows 24h / 48h / 7-day forecasts against the actual values, plus the
running error.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Make src importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C          # noqa: E402
import modeling as M        # noqa: E402

st.set_page_config(page_title="Weather Forecasting", layout="wide")


@st.cache_resource
def load_models():
    from tensorflow.keras.models import load_model
    models = {}
    for H in [24, 48, 168]:
        p = f"{C.MODEL_DIR}/multistep_{H}h.keras"
        if os.path.exists(p):
            models[H] = load_model(p, compile=False)
    return models


@st.cache_data
def load_data():
    df = M.load_feature_table()
    cfg = {"window": 48}
    p = f"{C.TABLE_DIR}/phase6_best_config.json"
    if os.path.exists(p):
        with open(p) as f:
            b = json.load(f)
        cfg["window"] = int(b.get("window", 48))
    return df, cfg


@st.cache_data
def load_comparison():
    p = f"{C.TABLE_DIR}/master_comparison.csv"
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0)
    return None


def main():
    st.title("🌡️ Multi-Step Weather Forecasting")
    st.caption("LSTM · GRU · Bi-LSTM — Jena Climate dataset")

    models = load_models()
    df, cfg = load_data()
    window = cfg["window"]

    if not models:
        st.warning("No trained multi-step models found. Run `python src/phase7_multistep.py` first.")
        st.stop()

    # Rebuild the same scaled sequence splits to get test windows.
    s = M.sequence_splits(df, window=window, horizon=24, multi=True)
    Xte = s["X_test"]
    t_mu, t_sd = s["t_mu"], s["t_sd"]

    # The test windows align to the tail of the dataframe index.
    n_test = len(Xte)
    test_index = df.index[-n_test:]

    with st.sidebar:
        st.header("Controls")
        horizon = st.selectbox("Forecast horizon", [24, 48, 168],
                               format_func=lambda h: f"{h} h ({h//24 if h>=24 else h} day"
                               + ("s" if h >= 48 else "") + ")" if h >= 24 else f"{h} h")
        # index into available windows (leave room for the largest horizon)
        max_h = max(models)
        max_start = n_test - max_h - 1
        pos = st.slider("Forecast start (test-set position)", 0,
                        max(0, max_start), value=min(100, max(0, max_start)))
        ts = test_index[pos]
        st.metric("Forecast origin", str(ts))

    model = models.get(horizon)
    if model is None:
        st.error(f"No model for horizon {horizon}h.")
        st.stop()

    # Build the input window for this position from horizon-specific split.
    s_h = M.sequence_splits(df, window=window, horizon=horizon, multi=True)
    x = s_h["X_test"][pos:pos + 1]
    y_true = M.inv_target(s_h["y_test"][pos], t_mu, t_sd)
    y_pred = M.inv_target(model.predict(x, verbose=0)[0], t_mu, t_sd)

    # --- Top metrics ---
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Horizon", f"{horizon} h")
    c2.metric("Forecast MAE", f"{mae:.2f} °C")
    c3.metric("Forecast RMSE", f"{rmse:.2f} °C")

    # --- Forecast vs actual ---
    st.subheader("Forecast vs Actual")
    hours = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hours, y_true, label="Actual", color="#2c3e50", lw=2)
    ax.plot(hours, y_pred, label="Forecast", color="#e74c3c", lw=2, ls="--")
    ax.fill_between(hours, y_true, y_pred, color="#e74c3c", alpha=0.12)
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # --- Error trend ---
    st.subheader("Error Trend (per lead time)")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.bar(hours, np.abs(y_true - y_pred), color="#8e44ad", alpha=0.7)
    ax2.set_xlabel("Hours ahead")
    ax2.set_ylabel("|Error| (°C)")
    st.pyplot(fig2)

    # --- Model comparison table ---
    comp = load_comparison()
    if comp is not None:
        st.subheader("Single-Step Model Leaderboard (test set)")
        st.dataframe(comp.style.format("{:.4f}").highlight_min(
            subset=["RMSE", "MAE"], color="#d4efdf"))


if __name__ == "__main__":
    main()
