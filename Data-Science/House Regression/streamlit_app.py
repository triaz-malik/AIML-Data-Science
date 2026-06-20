"""
House Price Estimator — Streamlit form.

Run after training:
    python house_prices.py
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from house_prices import (
    NONE_COLS, ZERO_COLS, MODE_COLS,
    QUALITY_MAP, QUALITY_COLS, ORDINAL_MAPS,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "model.pkl"

st.set_page_config(page_title="House Price Estimator", page_icon=":house:",
                   layout="centered")


@st.cache_resource
def load_artifacts():
    if not ART.exists():
        st.error(f"Artifacts not found at {ART}. "
                 "Run `python house_prices.py` first to train models.")
        st.stop()
    return joblib.load(ART)


def build_row(user_inputs: dict, art: dict) -> pd.DataFrame:
    """Construct a single-row dataframe by merging user inputs with training defaults."""
    medians = art["train_medians"]
    modes = art["train_modes"]
    fe = art["fe"]

    row = {**modes, **medians}  # categoricals + numerics from training
    row.update(user_inputs)

    # Apply same NaN -> None / 0 / median treatment as preprocess()
    for c in NONE_COLS:
        if c not in row or pd.isna(row.get(c)):
            row[c] = "None"
    for c in ZERO_COLS:
        if c not in row or pd.isna(row.get(c)):
            row[c] = 0

    df = pd.DataFrame([row])

    # Engineering (mirrors house_prices.engineer)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["AllFloorsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNewHouse"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
    df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["NeighborhoodPrice"] = df["Neighborhood"].map(fe["nbhd_med"]).fillna(
        fe["nbhd_med_default"])
    df["NeighborhoodCluster"] = df["Neighborhood"].map(fe["cluster_map"]).fillna(-1)

    # Ordinal encoding
    for c in QUALITY_COLS:
        df[c] = df[c].map(QUALITY_MAP).fillna(0).astype(int)
    for c, mp in ORDINAL_MAPS.items():
        df[c] = df[c].map(mp).fillna(0)

    # log1p for skewed features
    for f in fe["skewed_features"]:
        if f in df.columns:
            df[f] = np.log1p(df[f].clip(lower=0))

    df = pd.get_dummies(df)
    df = df.reindex(columns=fe["feature_columns"], fill_value=0)
    return df


def predict(df: pd.DataFrame, art: dict) -> float:
    base = art["base_models"]
    meta = art["meta_model"]
    base_preds = np.column_stack([m.predict(df) for m in base.values()])
    log_pred = meta.predict(base_preds)[0]
    return float(np.expm1(log_pred))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title(":house: House Price Estimator")
st.caption("Stacked Ridge meta-learner over Lasso/Ridge/ElasticNet/GBM/XGBoost/LightGBM/CatBoost")

art = load_artifacts()

with st.form("estimator"):
    st.subheader("Property details")
    c1, c2 = st.columns(2)
    with c1:
        neighborhood = st.selectbox("Neighborhood",
                                    art["neighborhoods"],
                                    index=art["neighborhoods"].index("CollgCr")
                                    if "CollgCr" in art["neighborhoods"] else 0)
        overall_qual = st.slider("Overall quality (1-10)", 1, 10, 7)
        gr_liv_area = st.number_input("Above-grade living area (sqft)",
                                      300, 6000, 1500, step=50)
        garage_cars = st.slider("Garage capacity (cars)", 0, 4, 2)
    with c2:
        total_bsmt_sf = st.number_input("Total basement area (sqft)",
                                        0, 4000, 900, step=50)
        first_flr_sf = st.number_input("1st floor area (sqft)",
                                       300, 4000, 1200, step=50)
        full_bath = st.slider("Full bathrooms", 0, 4, 2)
        bedrooms = st.slider("Bedrooms above grade", 0, 8, 3)

    c3, c4 = st.columns(2)
    with c3:
        year_built = st.slider("Year built",
                               art["train_min_year"], art["train_max_year"], 2000)
    with c4:
        year_sold = st.slider("Year sold",
                              art["train_min_year"], art["train_max_year"],
                              art["train_max_year"])

    submitted = st.form_submit_button("Estimate price", use_container_width=True)

if submitted:
    user_inputs = {
        "Neighborhood": neighborhood,
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "YrSold": year_sold,
        "GarageArea": garage_cars * 250,  # rough proxy
    }
    df = build_row(user_inputs, art)
    price = predict(df, art)

    nbhd_price = art["fe"]["nbhd_med"].get(neighborhood,
                                           art["fe"]["nbhd_med_default"])
    delta = price - nbhd_price

    st.markdown("---")
    st.metric("Estimated SalePrice", f"${price:,.0f}",
              delta=f"{delta:+,.0f} vs neighborhood median")
    st.caption(f"Neighborhood {neighborhood} median: ${nbhd_price:,.0f}")

    with st.expander("Model details"):
        st.write(f"**Stacker:** Ridge meta-learner (alpha={art['meta_model'].alpha_:.2f})")
        st.write("**Base models:**")
        for name, w in zip(art["base_models"].keys(), art["meta_model"].coef_):
            st.write(f"- {name}: weight {w:+.3f}")
        st.write(f"**Engineered features used:** {len(art['fe']['feature_columns'])}")

st.markdown("---")
st.caption("Defaults for unspecified fields use training-set medians/modes. "
           "For full per-feature control, edit the form directly in `streamlit_app.py`.")
