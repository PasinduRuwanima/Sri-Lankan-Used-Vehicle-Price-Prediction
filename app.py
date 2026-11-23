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

    c1, c2, c3 = st.columns(3)
    c1.metric("Listings", f"{len(filtered):,}")
    c2.metric("Median price", f'{filtered["Price"].median():.2f} Lakhs')
    c3.metric("Median mileage", f'{filtered["Millage(KM)"].median():,.0f} km')

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            filtered,
            x="Price",
            nbins=50,
            title="Price distribution",
            labels={"Price": "Price (Lakhs)"},
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    with right:
        medians = (
            filtered.groupby("Brand", as_index=False)["Price"]
            .median()
            .nlargest(12, "Price")
        )
        fig = px.bar(
            medians,
            x="Price",
            y="Brand",
            orientation="h",
            title="Top brands by median price",
            labels={"Price": "Median price (Lakhs)", "Brand": "Brand"},
        )
        fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Browse listings")
    filter_key = tuple(brands)
    if st.session_state.get("dataset_filter_key") != filter_key:
        st.session_state["dataset_filter_key"] = filter_key
        st.session_state["dataset_page"] = 1
    total_pages = max(1, (len(filtered) + 99) // 100)
    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        step=1,
        key="dataset_page",
    )
    start = (current_page - 1) * 100
    st.dataframe(
        filtered.iloc[start:start + 100],
        width="stretch",
        hide_index=True,
        height=420,
    )
    st.caption(
        f"Showing {start + 1:,}-{min(start + 100, len(filtered)):,} "
        f"of {len(filtered):,} listings"
    )


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
                st.info("Vehicle details changed. Run the estimate again to update your result.")

        st.caption(
            "This is a historical listing estimate, not a guaranteed transaction price."
        )
