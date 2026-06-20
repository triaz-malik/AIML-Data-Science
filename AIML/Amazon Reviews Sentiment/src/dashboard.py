"""
Phase 12 — Interactive dashboard.

Run with:  streamlit run src/dashboard.py

Three views:
  Executive  — headline KPIs, sentiment mix, rating distribution
  Products   — sentiment by category, worst products, category table
  NLP/Models — negative word cloud, top complaint themes, model comparison
Reads data/clean.parquet + the JSON metrics produced by the pipeline.
"""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
DATA = os.path.join(BASE, "data", "clean.parquet")
METRICS = os.path.join(BASE, "outputs", "metrics")
FIG = os.path.join(BASE, "outputs", "figures")
SENT_COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


@st.cache_data
def load_data():
    return pd.read_parquet(DATA, columns=["overall", "sentiment", "category",
                                          "parent_asin", "helpful", "date"])


def load_json(name):
    p = os.path.join(METRICS, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


st.set_page_config(page_title="Amazon Review Sentiment", layout="wide")
df = load_data()
insights = load_json("business_insights.json") or {}

st.sidebar.title("Amazon Review Sentiment")
view = st.sidebar.radio("View", ["Executive", "Products", "NLP / Models"])
cats = sorted(df["category"].unique())
chosen = st.sidebar.multiselect("Categories", cats, default=cats)
d = df[df["category"].isin(chosen)] if chosen else df

# ---------------- Executive ----------------
if view == "Executive":
    st.title("Executive dashboard")
    pos = (d["sentiment"] == "positive").mean() * 100
    neg = (d["sentiment"] == "negative").mean() * 100
    c = st.columns(4)
    c[0].metric("Total reviews", f"{len(d):,}")
    c[1].metric("Positive %", f"{pos:.1f}%")
    c[2].metric("Negative %", f"{neg:.1f}%")
    c[3].metric("Avg rating", f"{d['overall'].mean():.2f}")

    a, b = st.columns(2)
    with a:
        sc = d["sentiment"].value_counts().reindex(["negative", "neutral", "positive"])
        fig = px.bar(sc, color=sc.index, color_discrete_map=SENT_COLORS,
                     labels={"value": "Reviews", "index": ""}, title="Sentiment mix")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        rc = d["overall"].value_counts().sort_index()
        fig = px.bar(rc, labels={"value": "Reviews", "index": "Stars"},
                     title="Rating distribution")
        st.plotly_chart(fig, use_container_width=True)

    if insights.get("findings"):
        st.subheader("Key findings")
        for f in insights["findings"]:
            st.markdown(f"- {f}")

# ---------------- Products ----------------
elif view == "Products":
    st.title("Product & category view")
    cat = d.groupby("category").agg(
        reviews=("overall", "size"),
        mean_rating=("overall", "mean"),
        pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
    ).round(2).sort_values("pct_negative", ascending=False)
    fig = px.bar(cat.reset_index(), x="pct_negative", y="category", orientation="h",
                 color="pct_negative", color_continuous_scale="Reds",
                 title="% negative reviews by category")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cat, use_container_width=True)

    st.subheader("Worst products by negative rate (min 25 reviews)")
    wp = insights.get("worst_products_by_negative_rate", [])
    if wp:
        st.dataframe(pd.DataFrame(wp), use_container_width=True)

# ---------------- NLP / Models ----------------
else:
    st.title("NLP themes & model performance")
    wc = os.path.join(FIG, "05_wordclouds.png")
    if os.path.exists(wc):
        st.image(wc, caption="Positive vs negative word clouds", use_container_width=True)

    themes = insights.get("top_negative_themes", [])
    if themes:
        st.subheader("Top complaint themes (negative bigrams)")
        st.dataframe(pd.DataFrame(themes), use_container_width=True)

    models = insights.get("model_comparison", [])
    if models:
        st.subheader("Model comparison")
        mdf = pd.DataFrame(models)
        st.dataframe(mdf, use_container_width=True)
        fig = px.bar(mdf.melt(id_vars="model", var_name="metric", value_name="score"),
                     x="metric", y="score", color="model", barmode="group",
                     range_y=[0, 1], title="Accuracy / F1 / ROC-AUC by model")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run train_baseline.py / train_transformer.py / insights.py to populate model metrics.")
