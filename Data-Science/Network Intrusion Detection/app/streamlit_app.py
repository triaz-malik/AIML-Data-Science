"""Streamlit IDS dashboard: enter connection features -> Normal/Attack + risk score.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from preprocessing import clean, load_train  # noqa: E402

st.set_page_config(page_title="Network IDS", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    if not C.BEST_MODEL_PKL.exists():
        return None
    return joblib.load(C.BEST_MODEL_PKL)


@st.cache_data
def feature_template():
    """A representative (median/mode) row + value options, from training data."""
    df = clean(load_train()).drop(columns=[C.TARGET])
    template, options = {}, {}
    for col in df.columns:
        if df[col].dtype == object or col in C.CATEGORICAL:
            template[col] = df[col].mode()[0]
            options[col] = sorted(df[col].unique().tolist())
        else:
            template[col] = float(df[col].median())
    return template, options, list(df.columns)


model = load_model()
template, options, columns = feature_template()

st.title("🛡️ Network Intrusion Detection System")
st.caption("NSL-KDD · binary classifier (normal vs anomaly) with risk scoring")

if model is None:
    st.error("No trained model found. Run `python src/train_models.py` first.")
    st.stop()

tab_single, tab_batch = st.tabs(["🔎 Single connection", "📦 Batch CSV"])

# --- Single connection -------------------------------------------------------
with tab_single:
    st.subheader("Enter connection features")
    st.caption("Defaults are dataset medians/modes — change the ones you care about.")
    KEY = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "count", "srv_count", "logged_in", "dst_host_count",
           "dst_host_srv_count", "serror_rate", "same_srv_rate"]
    values = dict(template)
    cols = st.columns(3)
    for i, feat in enumerate([f for f in KEY if f in columns]):
        with cols[i % 3]:
            if feat in options:
                values[feat] = st.selectbox(feat, options[feat],
                                            index=options[feat].index(template[feat]))
            else:
                values[feat] = st.number_input(feat, value=float(template[feat]))

    with st.expander("Advanced — all remaining features"):
        rest = [f for f in columns if f not in KEY]
        rcols = st.columns(3)
        for i, feat in enumerate(rest):
            with rcols[i % 3]:
                if feat in options:
                    values[feat] = st.selectbox(feat, options[feat],
                                                index=options[feat].index(template[feat]),
                                                key=f"adv_{feat}")
                else:
                    values[feat] = st.number_input(feat, value=float(template[feat]),
                                                   key=f"adv_{feat}")

    if st.button("Analyze connection", type="primary"):
        row = pd.DataFrame([values])[columns]
        proba = float(model.predict_proba(row)[:, 1][0])
        risk = round(proba * 100, 1)
        c1, c2, c3 = st.columns(3)
        if proba >= 0.5:
            c1.error("### ⚠️ ATTACK")
        else:
            c1.success("### ✅ NORMAL")
        c2.metric("Attack probability", f"{proba:.1%}")
        c3.metric("Risk score", f"{risk}/100")
        st.progress(min(proba, 1.0))
        level = ("CRITICAL" if risk >= 80 else "HIGH" if risk >= 50
                 else "MEDIUM" if risk >= 20 else "LOW")
        st.write(f"**Risk level:** {level}")

# --- Batch CSV ---------------------------------------------------------------
with tab_batch:
    st.subheader("Score a CSV of connections")
    st.caption("Upload a CSV with the NSL-KDD feature columns (no label needed).")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        proba = model.predict_proba(data)[:, 1]
        out = data.copy()
        out["attack_probability"] = proba.round(4)
        out["prediction"] = pd.Series((proba >= 0.5).astype(int)).map(
            {0: "normal", 1: "anomaly"}).values
        out["risk_score"] = (proba * 100).round(1)
        n_attack = int((proba >= 0.5).sum())
        st.metric("Flagged as anomaly", f"{n_attack} / {len(out)}")
        st.dataframe(out.head(200), use_container_width=True)
        st.download_button("Download results", out.to_csv(index=False),
                           "ids_predictions.csv", "text/csv")
