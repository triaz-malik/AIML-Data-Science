"""House Price Estimator — Streamlit form.

Run after training:
    python -m src.train
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, predict_row  # noqa: E402

st.set_page_config(page_title="House Price Estimator", page_icon=":house:",
                   layout="centered")


@st.cache_resource
def _load():
    try:
        return load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


st.title(":house: House Price Estimator")
st.caption("Stacked Ridge meta-learner over Lasso/Ridge/ElasticNet/GBM/XGBoost/LightGBM/CatBoost")

art = _load()

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
        "GarageArea": garage_cars * 250,
    }
    price = predict_row(user_inputs, art)

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
           "For full per-feature control, edit the form directly in `app/streamlit_app.py`.")
