"""Simple Streamlit dashboard for Sri Lankan used vehicle price prediction."""

from html import escape
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_vehicle_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
PAGES = [
    "Overview",
    "Dataset Exploration",
    "Ensemble Model Comparison",
    "Model Performance",
    "Vehicle Price Prediction",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_json(filename: str) -> dict:
    with (ROOT / "artifacts" / filename).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def navigate(page: str):
    st.session_state["navigation"] = page


st.set_page_config(
    page_title="Sri Lankan Vehicle Price Dashboard",
    page_icon="car",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(13, 130, 120, 0.07), transparent 34%),
            #f5f7fa;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbfc 0%, #f1f5f7 100%);
        border-right: 1px solid #e2e8ee;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #172b45;
        letter-spacing: -0.025em;
    }
    h1 { font-weight: 750; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        transition: background-color 140ms ease, color 140ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #e7f2f1;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #deefed;
        color: #0b625c;
        font-weight: 650;
    }
    .overview-hero {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.3fr) minmax(180px, 0.7fr);
        align-items: center;
        gap: 1.5rem;
        overflow: hidden;
        min-height: 250px;
        margin: 0.5rem 0 1.2rem;
        padding: clamp(1.5rem, 4vw, 3rem);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 24px;
        background:
            radial-gradient(circle at 78% 24%, rgba(116, 225, 203, 0.22), transparent 30%),
            linear-gradient(118deg, #12334d 0%, #145c61 62%, #0d8278 100%);
        box-shadow: 0 18px 42px rgba(18, 51, 77, 0.16);
        color: #ffffff;
    }
    .hero-kicker {
        margin-bottom: 0.8rem;
        color: #a7e8d9;
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .overview-hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 760;
        letter-spacing: -0.045em;
        line-height: 1.08;
    }
    .overview-hero p {
        max-width: 39rem;
        margin: 0.9rem 0 0;
        color: rgba(255, 255, 255, 0.78);
        font-size: 1.03rem;
        line-height: 1.6;
    }
    .hero-art {
        width: 100%;
        max-width: 360px;
        justify-self: end;
        filter: drop-shadow(0 16px 16px rgba(4, 25, 38, 0.22));
    }
    .hero-art svg { display: block; width: 100%; height: auto; }
    .hero-badge {
        position: absolute;
        right: 1.5rem;
        bottom: 1.25rem;
        padding: 0.55rem 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 999px;
        background: rgba(7, 37, 51, 0.36);
        color: #e7fbf5;
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f9fbfc 100%);
        border: 1px solid #e3eaf0;
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(23, 43, 69, 0.045);
        padding: 16px;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 9px 22px rgba(23, 43, 69, 0.08);
    }
    [data-testid="stMetricLabel"] {
        color: #65758a;
        font-weight: 600;
        line-height: 1.25;
        white-space: normal;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 2.1vw, 1.8rem);
    }
    [data-testid="stButton"] button {
        border-radius: 11px;
        font-weight: 650;
        transition: transform 140ms ease, box-shadow 140ms ease;
    }
    [data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(13, 130, 120, 0.16);
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
    }
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
    }
    .estimate-value {
        margin: 0.5rem 0;
        padding: 1.25rem 1.4rem;
        border: 1px solid #cce5e1;
        border-radius: 16px;
        background: linear-gradient(135deg, #eaf7f4 0%, #f8fcfb 100%);
        box-shadow: 0 8px 24px rgba(13, 130, 120, 0.08);
        color: #08766d;
        font-size: clamp(1.7rem, 3vw, 2.35rem);
        font-weight: 750;
        line-height: 1.15;
    }
    @media (max-width: 720px) {
        .block-container { padding-top: 1.25rem; }
        .overview-hero {
            grid-template-columns: 1fr;
            min-height: 0;
        }
        .hero-art {
            width: min(100%, 300px);
            margin: -0.5rem auto 0;
        }
        .hero-badge { right: 1rem; bottom: 1rem; }
    }
    @media (max-width: 1100px) {
        div[data-testid="stMetric"] { padding: 11px; }
        [data-testid="stMetricLabel"] { font-size: 0.72rem; }
        [data-testid="stMetricValue"] { font-size: 1rem; }
        .overview-hero h1 { font-size: clamp(1.65rem, 3vw, 2.15rem); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

data = load_data()
categories = load_json("input_categories.json")
metadata = load_json("model_metadata.json")
results = load_csv("artifacts/model_results.csv")
test_metrics = metadata["test_metrics"]
listing_year = int(pd.to_datetime(data["Date"]).dt.year.max())

if st.session_state.get("navigation") not in PAGES:
    st.session_state["navigation"] = "Overview"

with st.sidebar:
    st.title("Vehicle Price")
    st.caption("Sri Lanka")
    page = st.radio(
        "Navigation",
        PAGES,
        key="navigation",
        label_visibility="collapsed",
    )


if page == "Overview":
    st.markdown(
        f"""
        <section class="overview-hero">
            <div>
                <div class="hero-kicker">Sri Lanka · Vehicle market insights</div>
                <h1>Know the value.<br>Drive smarter.</h1>
                <p>Explore real local listings and get a data-driven estimate for your next vehicle decision.</p>
            </div>
            <div class="hero-art" aria-hidden="true">
                <svg viewBox="0 0 420 210" role="img">
                    <ellipse cx="211" cy="177" rx="164" ry="15" fill="#062938" opacity=".38"/>
                    <path d="M43 139 67 111l66-13 39-48q8-10 23-10h79q17 0 28 14l37 45 42 13q13 4 15 18l-3 23H42Z" fill="#65d7c2"/>
                    <path d="m151 96 36-45h43v45Zm91-45h31q10 0 18 10l28 35h-77Z" fill="#173c50"/>
                    <path d="M239 52v41h36V52" fill="none" stroke="#7fb5b0" stroke-width="3" opacity=".55"/>
                    <path d="M44 140h360v18H44z" fill="#167f79"/>
                    <path d="M51 119 75 113l-3 15-24 4Zm334-7 20 7 3 12h-25Z" fill="#eafff8"/>
                    <circle cx="123" cy="157" r="31" fill="#102c3b"/>
                    <circle cx="123" cy="157" r="17" fill="#a9c5c6"/>
                    <circle cx="123" cy="157" r="8" fill="#355562"/>
                    <circle cx="342" cy="157" r="31" fill="#102c3b"/>
                    <circle cx="342" cy="157" r="17" fill="#a9c5c6"/>
                    <circle cx="342" cy="157" r="8" fill="#355562"/>
                </svg>
            </div>
            <div class="hero-badge">Historical listings · {listing_year}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "Estimate a vehicle price",
        type="primary",
        on_click=navigate,
        args=("Vehicle Price Prediction",),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean listings", f"{len(data):,}")
    c2.metric("Vehicle brands", str(data["Brand"].nunique()))
    c3.metric("Models", str(len(results)))
    c4.metric("Selected model", metadata["selected_model"])

    st.subheader("Data overview")
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            data,
            x="Price",
            nbins=50,
            title="Distribution of vehicle prices",
            labels={"Price": "Price (Lakhs)"},
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    with right:
        by_year = data.groupby("YOM", as_index=False)["Price"].median()
        fig = px.line(
            by_year,
            x="YOM",
            y="Price",
            title="Median price by manufacturing year",
            labels={"YOM": "Manufacturing year", "Price": "Price (Lakhs)"},
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Selected model performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", f'{test_metrics["MAE_Lakhs"]:.2f} Lakhs')
    m2.metric("Test RMSE", f'{test_metrics["RMSE_Lakhs"]:.2f} Lakhs')
    m3.metric("Test R-squared", f'{test_metrics["R2"]:.3f}')

elif page == "Dataset Exploration":
    st.title("Dataset Exploration")

    brands = st.multiselect(
        "Filter by brand",
        sorted(data["Brand"].dropna().unique()),
    )
    filtered = data[data["Brand"].isin(brands)].copy() if brands else data.copy()

    page_size = 100
    total_rows = len(filtered)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    filter_key = tuple(brands)

    if st.session_state.get("dataset_filter_key") != filter_key:
        st.session_state["dataset_filter_key"] = filter_key
        st.session_state["dataset_page"] = 1
    if "dataset_page" not in st.session_state:
        st.session_state["dataset_page"] = 1

    current_page = min(st.session_state["dataset_page"], total_pages)
    start = (current_page - 1) * page_size
    table_page = filtered.iloc[start:start + page_size]

    st.dataframe(table_page, width="stretch", height=500)

    previous, page_label, next_page = st.columns([1, 4, 1])
    with previous:
        if st.button("Previous", disabled=current_page <= 1):
            st.session_state["dataset_page"] = current_page - 1
            st.rerun()
    with page_label:
        st.caption(
            f"Showing rows {start + 1}-{min(start + page_size, total_rows)} "
            f"of {total_rows:,} | Page {current_page} of {total_pages}"
        )
    with next_page:
        if st.button("Next", disabled=current_page >= total_pages):
            st.session_state["dataset_page"] = current_page + 1
            st.rerun()

    col1, col2 = st.columns(2)
    fig = px.histogram(filtered, x="Price", nbins=50, title="Vehicle Price Distribution")
    fig.update_layout(xaxis_title="Price (Lakhs)", yaxis_title="Listings")
    col1.plotly_chart(fig, width="stretch")

    brand_median = filtered.groupby("Brand", as_index=False)["Price"].median()
    brand_median = brand_median.sort_values("Price", ascending=False).head(20)
    fig = px.bar(brand_median, x="Price", y="Brand", orientation="h",
                 title="Top 20 Brands by Median Price")
    fig.update_layout(xaxis_title="Median Price (Lakhs)", yaxis_title="Brand")
    col2.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    fig = px.scatter(filtered, x="Millage(KM)", y="Price", color="Fuel Type",
                     title="Mileage versus Price", opacity=0.55)
    fig.update_layout(xaxis_title="Mileage (KM)", yaxis_title="Price (Lakhs)")
    col1.plotly_chart(fig, width="stretch")


elif page == "Ensemble Model Comparison":
    st.title("Ensemble Model Comparison")
    ranked = results.sort_values("MAE_Lakhs")

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", metadata["selected_model"])
    c2.metric("Cross-validation MAE", f'{metadata["CV MAE Lakhs"]:.2f} Lakhs')
    c3.metric("Models evaluated", str(len(results)))

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            ranked,
            x="MAE_Lakhs",
            y="Model",
            orientation="h",
            title="Test MAE - lower is better",
            labels={"MAE_Lakhs": "MAE (Lakhs)", "Model": "Model"},
        )
        fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.bar(
            ranked,
            x="R2",
            y="Model",
            orientation="h",
            title="Test R-squared - higher is better",
            labels={"R2": "R-squared", "Model": "Model"},
        )
        fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Evaluation results")
    st.dataframe(ranked, width="stretch", hide_index=True)


elif page == "Model Performance":
    st.title("Model Performance")
    predictions = load_csv("artifacts/test_predictions.csv")
    importance = load_csv("artifacts/feature_importance.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f'{test_metrics["MAE_Lakhs"]:.2f} Lakhs')
    c2.metric("Test RMSE", f'{test_metrics["RMSE_Lakhs"]:.2f} Lakhs')
    c3.metric("Test R-squared", f'{test_metrics["R2"]:.3f}')

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            predictions,
            x="Actual Lakhs",
            y="Predicted Lakhs",
            title="Actual vs predicted price",
            opacity=0.5,
        )
        limit = float(
            max(
                predictions["Actual Lakhs"].max(),
                predictions["Predicted Lakhs"].max(),
            )
        )
        fig.add_scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            name="Ideal prediction",
            line={"dash": "dash"},
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.histogram(
            predictions,
            x="Residual Lakhs",
            nbins=50,
            title="Prediction residuals",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Feature importance")
    top_features = importance.nlargest(15, "Importance").copy()
    top_features["Feature"] = (
        top_features["Feature"]
        .str.replace(r"^(categorical|numeric)__", "", regex=True)
        .str.replace("_", " ")
    )
    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 features",
    )
    fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")


elif page == "Vehicle Price Prediction":
    st.title("Vehicle Price Prediction")

    form_col, result_col = st.columns([1.5, 1])
    with form_col:
        c1, c2 = st.columns(2)
        brand = c1.selectbox("Brand", categories["Brand"])
        matching_models = sorted(
            data.loc[data["Brand"] == brand, "Model"].dropna().astype(str).unique()
        )
        model_name = c2.selectbox(
            "Model",
            matching_models or ["Unknown"],
            key=f"model_{brand}",
        )

        c1, c2, c3 = st.columns(3)
        year = c1.number_input(
            "Manufacture year",
            min_value=1956,
            max_value=listing_year,
            value=min(2018, listing_year),
        )
        engine = c2.number_input(
            "Engine (cc)",
            min_value=100,
            max_value=10000,
            value=1500,
            step=100,
        )
        mileage = c3.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=2000000,
            value=80000,
            step=1000,
        )

        c1, c2 = st.columns(2)
        gear = c1.selectbox("Transmission", categories["Gear"])
        fuel = c2.selectbox("Fuel type", categories["Fuel Type"])
        town = c1.selectbox("Town", categories["Town"])
        condition = c2.selectbox("Condition", categories["Condition"])
        leasing = c1.selectbox("Leasing status", categories["Leasing"])

        c1, c2 = st.columns(2)
        air = c1.selectbox("Air conditioning", categories["AIR CONDITION"])
        steering = c2.selectbox("Power steering", categories["POWER STEERING"])
        mirror = c1.selectbox("Power mirrors", categories["POWER MIRROR"])
        window = c2.selectbox("Power windows", categories["POWER WINDOW"])

        predict = st.button(
            "Estimate vehicle price",
            type="primary",
            width="stretch",
        )

    vehicle_input = {
        "Brand": brand,
        "Model": model_name,
        "YOM": int(year),
        "Engine (cc)": float(engine),
        "Gear": gear,
        "Fuel Type": fuel,
        "Millage(KM)": float(mileage),
        "Town": town,
        "Leasing": leasing,
        "Condition": condition,
        "AIR CONDITION": air,
        "POWER STEERING": steering,
        "POWER MIRROR": mirror,
        "POWER WINDOW": window,
        "Vehicle Age": max(0, listing_year - int(year)),
    }

    if predict:
        prediction_lakhs = float(
            load_model().predict(pd.DataFrame([vehicle_input]))[0]
        )
        st.session_state["estimate"] = {
            "input": vehicle_input.copy(),
            "lakhs": prediction_lakhs,
        }

    with result_col:
        estimate = st.session_state.get("estimate")
        current_estimate = estimate and estimate["input"] == vehicle_input
        if current_estimate:
            value = float(estimate["lakhs"])
            st.markdown(
                f'<div class="estimate-value">LKR {value * 100000:,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.write(f"{value:.2f} Lakhs")
            st.write(f"{escape(brand)} {escape(model_name)}")
            st.write(f"{year} | {engine:,} cc | {mileage:,} km")
        else:
            st.info("Enter vehicle details, then select Estimate vehicle price.")
            if estimate:
                st.info(
                    "Vehicle details changed. Run the estimate again to update your result."
                )
