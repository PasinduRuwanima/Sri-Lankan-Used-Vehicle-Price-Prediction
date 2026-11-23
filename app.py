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
    [data-testid="stSidebar"] { background-color: #f7f8fa; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    h1, h2, h3 { color: #172033; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    "Overview",
    "Dataset Exploration",
    "Ensemble Model Comparison",
    "Model Performance",
    "Vehicle Price Prediction",
]
page = st.sidebar.radio("Navigation", pages)

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
    st.title("Sri Lankan Used Vehicle Price Prediction")
    st.write(
        "Compare ensemble models and estimate used vehicle prices from historical listing data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean listings", f"{len(data):,}")
    c2.metric("Vehicle brands", str(data["Brand"].nunique()))
    c3.metric("Models compared", str(len(results)))
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

    st.button(
        "Estimate a vehicle price",
        type="primary",
        on_click=navigate,
        args=("Vehicle Price Prediction",),
    )


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

    st.dataframe(table_page, use_container_width=True, height=500)

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
    col1.plotly_chart(fig, use_container_width=True)

    brand_median = filtered.groupby("Brand", as_index=False)["Price"].median()
    brand_median = brand_median.sort_values("Price", ascending=False).head(20)
    fig = px.bar(brand_median, x="Price", y="Brand", orientation="h",
                 title="Top 20 Brands by Median Price")
    fig.update_layout(xaxis_title="Median Price (Lakhs)", yaxis_title="Brand")
    col2.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    fig = px.scatter(filtered, x="Millage(KM)", y="Price", color="Fuel Type",
                     title="Mileage versus Price", opacity=0.55)
    fig.update_layout(xaxis_title="Mileage (KM)", yaxis_title="Price (Lakhs)")
    col1.plotly_chart(fig, use_container_width=True)


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
