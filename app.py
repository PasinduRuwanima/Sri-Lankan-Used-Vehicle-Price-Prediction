"""Streamlit dashboard for Sri Lankan used vehicle price prediction."""

from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_vehicle_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_categories() -> dict[str, list[str]]:
    with (ROOT / "artifacts" / "input_categories.json").open() as file:
        return json.load(file)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


st.set_page_config(
    page_title="Sri Lankan Vehicle Price Ensemble",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color:#F7F9FC; }
    [data-testid="stMetric"] {
        background-color:white;
        border:1px solid #E6EBF1;
        border-radius:14px;
        padding:16px;
    }
    h1,h2,h3 { color:#16324F; }
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
categories = load_categories()


def price_kpi(value: float, label: str):
    st.metric(label, f"{value:.2f} Lakhs")


if page == "Overview":
    st.title("Sri Lankan Used Vehicle Price Prediction")
    st.caption("Ensemble learning regression dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Records", f"{len(data):,}")
    col2.metric("Model Features", "15")
    col3.metric("Models Evaluated", "11")
    col4.metric("Best Model", "Extra Trees")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test MAE", "5.15 Lakhs")
    col2.metric("Test RMSE", "15.08 Lakhs")
    col3.metric("Test R²", "0.894")

    st.subheader("About the ensemble approach")
    st.write(
        "Ensemble learning combines multiple regression models to reduce error and improve "
        "generalization. Bagging trains many independent trees, boosting improves weak "
        "learners sequentially, and stacking combines diverse predictions using a meta-model."
    )

elif page == "Dataset Exploration":
    st.title("Dataset Exploration")
    st.caption("Price unit: Lakhs of Sri Lankan Rupees")

    brands = st.multiselect("Filter brands", sorted(data["Brand"].unique()), default=[])
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

    fig = px.box(filtered, x="Gear", y="Price", color="Gear",
                 title="Price by Transmission")
    fig.update_layout(xaxis_title="Gear", yaxis_title="Price (Lakhs)", showlegend=False)
    col2.plotly_chart(fig, use_container_width=True)

elif page == "Ensemble Model Comparison":
    st.title("Ensemble Model Comparison")
    results = load_csv("artifacts/model_results.csv")

    ranked = results.sort_values("MAE_Lakhs")
    st.dataframe(ranked, use_container_width=True)

    col1, col2 = st.columns(2)
    fig = px.bar(ranked, x="Model", y="MAE_Lakhs", color="Model",
                 title="Mean Absolute Error Comparison")
    fig.update_layout(xaxis_title="Model", yaxis_title="MAE (Lakhs)", showlegend=False)
    col1.plotly_chart(fig, use_container_width=True)

    fig = px.bar(ranked, x="Model", y="R2", color="Model",
                 title="R-squared Comparison")
    fig.update_layout(xaxis_title="Model", yaxis_title="R-squared", showlegend=False)
    col2.plotly_chart(fig, use_container_width=True)

    st.info(
        "Extra Trees was selected using 3-fold cross-validation MAE. "
        "The untouched test set is reported only for final evaluation."
    )

elif page == "Model Performance":
    st.title("Model Performance Analysis")
    predictions = load_csv("artifacts/test_predictions.csv")
    importance = load_csv("artifacts/feature_importance.csv")

    col1, col2 = st.columns(2)
    fig = px.scatter(predictions, x="Actual Lakhs", y="Predicted Lakhs",
                     title="Actual versus Predicted Price", opacity=0.55)
    limit = max(predictions["Actual Lakhs"].max(), predictions["Predicted Lakhs"].max())
    fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines",
                             name="Ideal prediction", line=dict(dash="dash", color="red")))
    fig.update_layout(xaxis_title="Actual Price (Lakhs)", yaxis_title="Predicted Price (Lakhs)")
    col1.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(predictions, x="Residual Lakhs", nbins=50,
                       title="Residual Distribution")
    fig.update_layout(xaxis_title="Residual (Lakhs)", yaxis_title="Count")
    col2.plotly_chart(fig, use_container_width=True)

    fig = px.bar(importance.head(15), x="Importance", y="Feature", orientation="h",
                 title="Extra Trees Feature Importance")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

elif page == "Vehicle Price Prediction":
    st.title("Vehicle Price Prediction")
    st.caption("Enter the vehicle details to estimate its historical listing value")

    col1, col2 = st.columns(2)
    brand = col1.selectbox("Brand", categories["Brand"])
    matching_models = sorted(
        data.loc[data["Brand"] == brand, "Model"].dropna().astype(str).unique()
    )
    model_name = col2.selectbox("Model", matching_models or ["Unknown"])

    col1, col2, col3 = st.columns(3)
    year = col1.number_input("Year of Manufacture", min_value=1956, max_value=2025, value=2018)
    engine = col2.number_input("Engine Capacity (cc)", min_value=100, max_value=10000, value=1500)
    mileage = col3.number_input("Mileage (KM)", min_value=0, max_value=2000000, value=80000)

    col1, col2, col3 = st.columns(3)
    gear = col1.selectbox("Gear", categories["Gear"])
    fuel = col2.selectbox("Fuel Type", categories["Fuel Type"])
    town = col3.selectbox("Town", categories["Town"])

    col1, col2 = st.columns(2)
    leasing = col1.selectbox("Leasing", categories["Leasing"])
    condition = col2.selectbox("Condition", categories["Condition"])

    col1, col2, col3, col4 = st.columns(4)
    air = col1.selectbox("Air Conditioning", categories["AIR CONDITION"])
    steering = col2.selectbox("Power Steering", categories["POWER STEERING"])
    mirror = col3.selectbox("Power Mirror", categories["POWER MIRROR"])
    window = col4.selectbox("Power Window", categories["POWER WINDOW"])

    if st.button("Predict Vehicle Price", type="primary", use_container_width=True):
        listing_year = 2025
        vehicle_age = max(0, listing_year - int(year))
        input_frame = pd.DataFrame([{
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
            "Vehicle Age": vehicle_age,
        }])

        prediction_lakhs = float(load_model().predict(input_frame)[0])
        lkr = prediction_lakhs * 100000
        st.success("Estimated Vehicle Price")
        st.metric("Predicted Price", f"LKR {lkr:,.0f}")
        st.caption(f"Equivalent: {prediction_lakhs:.2f} Lakhs")
        st.warning(
            "This estimate uses historical vehicle listing data and is not a guaranteed "
            "transaction price."
        )
