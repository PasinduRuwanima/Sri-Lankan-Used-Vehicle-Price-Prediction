"""Streamlit dashboard for Sri Lankan used vehicle price prediction."""

from html import escape
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
COLORS = ["#0d9488", "#304f78", "#60b8b0", "#e6ad62", "#8197b2", "#8b7bb1"]
PAGES = [
    "Overview", "Dataset Exploration", "Ensemble Model Comparison",
    "Model Performance", "Vehicle Price Prediction",
]
NAV_LABELS = {
    "Overview": "◈   Overview",
    "Dataset Exploration": "▦   Explore dataset",
    "Ensemble Model Comparison": "▤   Compare models",
    "Model Performance": "◎   Model performance",
    "Vehicle Price Prediction": "◇   Price estimator",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_json(filename: str) -> dict:
    with (ROOT / "artifacts" / filename).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def html(content: str):
    st.markdown(content, unsafe_allow_html=True)


def heading(eyebrow: str, title: str, description: str):
    html(f'<div class="page-heading"><div class="eyebrow">{eyebrow}</div>'
         f'<h1>{title}</h1><p>{description}</p></div>')


def metric(label: str, value: str, detail: str, accent: str = "teal"):
    html(f'<div class="stat-card {accent}"><div class="stat-label">{escape(label)}</div>'
         f'<div class="stat-value">{escape(value)}</div>'
         f'<div class="stat-detail">{escape(detail)}</div></div>')


def chart(fig, height: int = 320):
    fig.update_layout(
        template="plotly_white", height=height,
        colorway=COLORS, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#66778d"),
        title=dict(font=dict(size=16, color="#182e48"), x=0),
        margin=dict(l=12, r=16, t=50, b=20),
        legend=dict(orientation="h", y=-0.25, title_text=""),
        hoverlabel=dict(bgcolor="#102b40", font_color="white"),
    )
    fig.update_xaxes(gridcolor="#edf1f5", zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f5", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def navigate(page: str):
    st.session_state["navigation"] = page


st.set_page_config(
    page_title="AutoValue | Sri Lankan Vehicle Insights", page_icon="🚘",
    layout="wide", initial_sidebar_state="expanded",
)
html(f'<style>{(ROOT / "assets" / "dashboard.css").read_text(encoding="utf-8")}</style>')

data = load_data()
categories = load_json("input_categories.json")
metadata = load_json("model_metadata.json")
results = load_csv("artifacts/model_results.csv")
test_metrics = metadata["test_metrics"]
listing_year = int(pd.to_datetime(data["Date"]).dt.year.max())

if st.session_state.get("navigation") not in PAGES:
    st.session_state["navigation"] = "Overview"

with st.sidebar:
    html('<div class="brand"><div class="brand-icon">AV<span>↗</span></div>'
         '<div><div class="brand-name">AutoValue<span>LK</span></div>'
         '<div class="brand-tagline">VEHICLE INTELLIGENCE</div></div></div>')
    html('<div class="nav-caption">WORKSPACE</div>')
    page = st.radio("Navigation", PAGES, key="navigation",
                    format_func=NAV_LABELS.get, label_visibility="collapsed")

html('<div class="topbar"><span>WORKSPACE <span class="breadcrumb">/ '
     f'{escape(page)}</span></span><span class="market-badge">'
     '<span class="status-dot"></span>Sri Lanka · LKR</span></div>')

if page == "Overview":
    heading("YOUR MARKET, AT A GLANCE", "Vehicle insights. Better decisions.",
            "A clearer view of Sri Lanka’s used vehicle market, powered by ensemble learning.")
    with st.container(key="hero"):
        hero_text, hero_art = st.columns([1.5, 1])
        with hero_text:
            html('<div class="hero-copy"><span class="hero-pill">DATA-DRIVEN VEHICLE VALUATION</span>'
                 '<h2>Know the value.<br><span>Move with confidence.</span></h2>'
                 '<p>Turn vehicle details into a useful price estimate.<br>'
                 'Built on real Sri Lankan listings and a carefully evaluated model.</p></div>')
            st.button("Estimate a vehicle price  →", type="primary", on_click=navigate,
                      args=("Vehicle Price Prediction",))
            html('<div class="hero-footnote">Historical listing estimates · Prices in Sri Lankan Rupees</div>')
        with hero_art:
            html('''<div class="hero-art"><div class="art-orbit"></div>
<svg viewBox="0 0 520 260" role="img" aria-label="Illustration of a teal vehicle">
<defs><linearGradient id="body" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#acf0de"/><stop offset="1" stop-color="#35a69b"/></linearGradient></defs>
<ellipse cx="270" cy="221" rx="205" ry="16" fill="#051e2a" opacity=".5"/>
<path d="M53 169L83 130 161 116 204 67Q216 54 240 54H323Q343 54 358 72L398 119 459 139Q478 144 481 166L477 191H49Z" fill="url(#body)"/>
<path d="M178 116L218 70H258V116Z" fill="#153d51"/><path d="M270 70H319Q330 70 339 81L369 116H270Z" fill="#153d51"/>
<path d="M225 76L193 111H211L242 76ZM280 76V109H300L300 76Z" fill="#5f9f9d" opacity=".4"/>
<path d="M55 173H474" stroke="#ddfff2" stroke-width="2" opacity=".6"/>
<path d="M271 125V179M166 128L153 162M389 127L400 162" fill="none" stroke="#24897f" stroke-width="2"/>
<rect x="284" y="130" width="21" height="5" rx="2.5" fill="#1b5c5d"/>
<path d="M60 151L88 144 85 157 54 162Z" fill="#e7fff6"/><path d="M452 144L471 151 474 161H450Z" fill="#f4ca86"/>
<path d="M48 183H477V195H48Z" fill="#1c5d64"/>
<circle cx="132" cy="191" r="37" fill="#102c3b"/><circle cx="132" cy="191" r="22" fill="#9bb7be"/><circle cx="132" cy="191" r="12" fill="#355562"/>
<circle cx="404" cy="191" r="37" fill="#102c3b"/><circle cx="404" cy="191" r="22" fill="#9bb7be"/><circle cx="404" cy="191" r="12" fill="#355562"/>
</svg><div class="art-caption"><span class="status-dot"></span> LOCAL DATA. SMARTER INSIGHTS.</div></div>''')
    cols = st.columns(4)
    with cols[0]: metric("VEHICLE LISTINGS", f"{len(data):,}", "Cleaned and ready to explore")
    with cols[1]: metric("VEHICLE BRANDS", str(data["Brand"].nunique()), "Across the local dataset", "blue")
    with cols[2]: metric("MODELS COMPARED", str(len(results)), "Saved evaluation results", "gold")
    with cols[3]: metric("SELECTED MODEL", metadata["selected_model"], "Lowest cross-validation MAE", "blue")
    left, right = st.columns([1.55, 1])
    with left, st.container(border=True, key='panel-market-snapshot'):
        st.subheader("A snapshot of the market")
        st.caption("Median listing price by manufacturing year · Lakhs")
        by_year = data.groupby("YOM", as_index=False)["Price"].median()
        fig = px.area(by_year, x="YOM", y="Price", color_discrete_sequence=COLORS)
        fig.update_traces(line_width=3, fillcolor="rgba(13,148,136,0.10)",
                          hovertemplate="Year %{x}<br>%{y:.2f} Lakhs<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title="Median price (Lakhs)")
        chart(fig, 280)
    with right, st.container(border=True, key='panel-performance-summary'):
        st.subheader("How the model performs")
        st.caption("Extra Trees · Held-out test set")
        for label, value, note in [
            ("Mean absolute error", f'{test_metrics["MAE_Lakhs"]:.2f}', "Lakhs · average absolute prediction error"),
            ("Root mean squared error", f'{test_metrics["RMSE_Lakhs"]:.2f}', "Lakhs · more sensitive to large errors"),
            ("R-squared score", f'{test_metrics["R2"]:.3f}', "Proportion of price variance explained"),
        ]:
            html(f'<div class="performance-row"><div><strong>{label}</strong><small>{note}</small>'
                 f'</div><b>{value}</b></div>')
        st.button("Explore model performance  ↗", on_click=navigate, args=("Model Performance",))
    st.subheader("From listings to insight")
    for col, number, title, text in zip(st.columns(3), ["01", "02", "03"],
        ["Explore the data", "Compare the models", "Find an estimated value"],
        ["Filter real listings and discover patterns in price, mileage and brands.",
         "See how bagging, boosting and stacking perform on the same task.",
         "Add your vehicle details and get an estimate in LKR and Lakhs."]):
        with col:
            html(f'<div class="step-card"><span>{number}</span><h4>{title}</h4><p>{text}</p></div>')

elif page == "Dataset Exploration":
    heading("LOOK CLOSER", "Explore the dataset", "Discover the listings, brands and patterns behind every estimate.")
    with st.container(border=True, key='panel-brand-filter'):
        brands = st.multiselect("Filter by brand", sorted(data["Brand"].unique()),
                                placeholder="All brands · choose one or more")
    filtered = data[data["Brand"].isin(brands)].copy() if brands else data.copy()
    cols = st.columns(3)
    with cols[0]: metric("MATCHING LISTINGS", f"{len(filtered):,}", "Based on your brand selection")
    with cols[1]: metric("MEDIAN PRICE", f'{filtered["Price"].median():.2f} Lakhs', "Middle value of matching listings", "blue")
    with cols[2]: metric("MEDIAN MILEAGE", f'{filtered["Millage(KM)"].median():,.0f} km', "Across matching listings", "gold")
    distributions, listings = st.tabs(["Market patterns", "Browse listings"])
    with distributions:
        col1, col2 = st.columns(2)
        with col1, st.container(border=True, key='panel-price-distribution'):
            fig = px.histogram(filtered, x="Price", nbins=50, title="Price distribution", color_discrete_sequence=COLORS)
            fig.update_layout(xaxis_title="Price (Lakhs)", yaxis_title="Listings", bargap=.08)
            chart(fig)
        with col2, st.container(border=True, key='panel-brand-prices'):
            medians = filtered.groupby("Brand", as_index=False)["Price"].median().nlargest(12, "Price")
            fig = px.bar(medians, x="Price", y="Brand", orientation="h", title="Top brands by median price", color_discrete_sequence=COLORS)
            fig.update_layout(xaxis_title="Median price (Lakhs)", yaxis_title=None, yaxis_categoryorder="total ascending")
            chart(fig)
        col1, col2 = st.columns(2)
        with col1, st.container(border=True, key='panel-mileage-chart'):
            fig = px.scatter(filtered, x="Millage(KM)", y="Price", color="Fuel Type", title="Mileage & listing price", opacity=.55, color_discrete_sequence=COLORS)
            fig.update_layout(xaxis_title="Mileage (km)", yaxis_title="Price (Lakhs)")
            chart(fig, 370)
        with col2, st.container(border=True, key='panel-transmission-chart'):
            fig = px.box(filtered, x="Gear", y="Price", color="Gear", title="Price by transmission", color_discrete_sequence=COLORS)
            fig.update_layout(xaxis_title=None, yaxis_title="Price (Lakhs)", showlegend=False)
            chart(fig, 370)
    with listings, st.container(border=True, key='panel-listing-library'):
        st.subheader("The listing library")
        st.caption("Price is shown in Lakhs · 1 Lakh = LKR 100,000")
        filter_key = tuple(brands)
        if st.session_state.get("dataset_filter_key") != filter_key:
            st.session_state["dataset_filter_key"] = filter_key
            st.session_state["dataset_page"] = 1
        total_pages = max(1, (len(filtered) + 99) // 100)
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, key="dataset_page")
        start = (current_page - 1) * 100
        st.dataframe(filtered.iloc[start:start + 100], width="stretch", hide_index=True, height=420)
        st.caption(f"Showing {start + 1:,}–{min(start + 100, len(filtered)):,} of {len(filtered):,} listings · {total_pages} pages")

elif page == "Ensemble Model Comparison":
    heading("THE EVIDENCE BEHIND THE ESTIMATE", "Compare the models", "Explore the trade-offs between prediction error and explanatory power.")
    cols = st.columns(3)
    with cols[0]: metric("SELECTED MODEL", metadata["selected_model"], "Selected using cross-validation")
    with cols[1]: metric("CROSS-VALIDATION MAE", f'{metadata["CV MAE Lakhs"]:.2f} Lakhs', "3-fold cross-validation", "blue")
    with cols[2]: metric("EVALUATED MODELS", str(len(results)), "Saved model evaluation rows", "gold")
    st.info("Extra Trees was selected using 3-fold cross-validation MAE. The untouched test set is used only for final evaluation.", icon="ℹ️")
    ranked = results.sort_values("MAE_Lakhs")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True, key='panel-mae-comparison'):
        fig = px.bar(ranked, x="MAE_Lakhs", y="Model", orientation="h", title="Test mean absolute error · lower is better", color_discrete_sequence=COLORS)
        fig.update_layout(xaxis_title="MAE (Lakhs)", yaxis_title=None, yaxis_autorange="reversed")
        chart(fig, 380)
    with col2, st.container(border=True, key='panel-r2-comparison'):
        fig = px.bar(ranked, x="R2", y="Model", orientation="h", title="Test R-squared · higher is better", color_discrete_sequence=[COLORS[1]])
        fig.update_layout(xaxis_title="R-squared", yaxis_title=None, yaxis_autorange="reversed")
        chart(fig, 380)
    with st.container(border=True, key='panel-evaluation-table'):
        st.subheader("Evaluation details")
        st.dataframe(ranked, width="stretch", hide_index=True,
                     column_config={"MAE_Lakhs": st.column_config.NumberColumn("Test MAE (Lakhs)", format="%.3f"),
                                    "RMSE_Lakhs": st.column_config.NumberColumn("Test RMSE (Lakhs)", format="%.3f"),
                                    "R2": st.column_config.NumberColumn("Test R²", format="%.3f")})

elif page == "Model Performance":
    heading("UNDERSTAND THE ACCURACY", "Model performance", "A closer look at predictions, errors and the features that shape vehicle prices.")
    predictions = load_csv("artifacts/test_predictions.csv")
    importance = load_csv("artifacts/feature_importance.csv")
    cols = st.columns(3)
    with cols[0]: metric("TEST MAE", f'{test_metrics["MAE_Lakhs"]:.2f} Lakhs', "Mean absolute error")
    with cols[1]: metric("TEST RMSE", f'{test_metrics["RMSE_Lakhs"]:.2f} Lakhs', "Root mean squared error", "blue")
    with cols[2]: metric("TEST R-SQUARED", f'{test_metrics["R2"]:.3f}', f"Evaluated on {len(predictions):,} held-out listings", "gold")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True, key='panel-prediction-chart'):
        fig = px.scatter(predictions, x="Actual Lakhs", y="Predicted Lakhs", title="Actual vs. predicted price", opacity=.5, color_discrete_sequence=COLORS)
        limit = max(predictions["Actual Lakhs"].max(), predictions["Predicted Lakhs"].max())
        fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="Ideal prediction", line=dict(dash="dash", color="#d99c52")))
        fig.update_layout(xaxis_title="Actual (Lakhs)", yaxis_title="Predicted (Lakhs)")
        chart(fig, 370)
    with col2, st.container(border=True, key='panel-residual-chart'):
        fig = px.histogram(predictions, x="Residual Lakhs", nbins=50, title="Where prediction errors fall", color_discrete_sequence=COLORS)
        fig.update_layout(xaxis_title="Residual (Lakhs)", yaxis_title="Listings", bargap=.08)
        chart(fig, 370)
    with st.container(border=True, key='panel-feature-chart'):
        st.subheader("What influences the estimate?")
        st.caption("Top 15 encoded features · Extra Trees feature importance")
        top_features = importance.nlargest(15, "Importance").copy()
        top_features["Feature"] = top_features["Feature"].str.replace(r"^(categorical|numeric)__", "", regex=True).str.replace("_", " ")
        fig = px.bar(top_features, x="Importance", y="Feature", orientation="h", color_discrete_sequence=COLORS)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), yaxis_title=None, xaxis_title="Relative importance")
        chart(fig, 430)

elif page == "Vehicle Price Prediction":
    heading("MAKE YOUR NEXT MOVE INFORMED", "What’s your vehicle worth?", "Tell us about the vehicle. We’ll turn the details into a historical listing price estimate.")
    form_col, result_col = st.columns([1.65, 1], gap="large")
    with form_col:
        with st.container(border=True, key='panel-vehicle-essentials'):
            st.subheader("01 / The essentials")
            st.caption("Start with the make, model and specifications.")
            col1, col2 = st.columns(2)
            brand = col1.selectbox("Brand", categories["Brand"])
            matching_models = sorted(data.loc[data["Brand"] == brand, "Model"].dropna().astype(str).unique())
            model_name = col2.selectbox("Model", matching_models or ["Unknown"], key=f"model_{brand}")
            col1, col2, col3 = st.columns(3)
            year = col1.number_input("Manufacture year", min_value=1956, max_value=listing_year, value=min(2018, listing_year), help=f"Historical estimates use a {listing_year} listing year.")
            engine = col2.number_input("Engine (cc)", min_value=100, max_value=10000, value=1500, step=100)
            mileage = col3.number_input("Mileage (km)", min_value=0, max_value=2000000, value=80000, step=1000)
        with st.container(border=True, key='panel-vehicle-details'):
            st.subheader("02 / A few more details")
            col1, col2 = st.columns(2)
            gear = col1.selectbox("Transmission", categories["Gear"])
            fuel = col2.selectbox("Fuel type", categories["Fuel Type"])
            town = col1.selectbox("Town", categories["Town"])
            condition = col2.selectbox("Condition", categories["Condition"])
            leasing = st.selectbox("Leasing status", categories["Leasing"])
        with st.container(border=True, key='panel-equipment'):
            st.subheader("03 / Comfort & equipment")
            col1, col2 = st.columns(2)
            air = col1.selectbox("Air conditioning", categories["AIR CONDITION"])
            steering = col2.selectbox("Power steering", categories["POWER STEERING"])
            mirror = col1.selectbox("Power mirrors", categories["POWER MIRROR"])
            window = col2.selectbox("Power windows", categories["POWER WINDOW"])
        predict = st.button("Estimate vehicle price  →", type="primary", width="stretch")
    vehicle_input = {
        "Brand": brand, "Model": model_name, "YOM": int(year), "Engine (cc)": float(engine),
        "Gear": gear, "Fuel Type": fuel, "Millage(KM)": float(mileage), "Town": town,
        "Leasing": leasing, "Condition": condition, "AIR CONDITION": air,
        "POWER STEERING": steering, "POWER MIRROR": mirror, "POWER WINDOW": window,
        "Vehicle Age": max(0, listing_year - int(year)),
    }
    if predict:
        with st.spinner("Finding your vehicle’s estimated value…"):
            prediction_lakhs = float(load_model().predict(pd.DataFrame([vehicle_input]))[0])
            st.session_state["estimate"] = {"input": vehicle_input.copy(), "lakhs": prediction_lakhs}
    with result_col:
        estimate = st.session_state.get("estimate")
        current_estimate = estimate and estimate["input"] == vehicle_input
        if current_estimate:
            value = estimate["lakhs"]
            html(f'<div class="estimate-card"><div class="eyebrow">YOUR ESTIMATED LISTING VALUE</div>'
                 f'<div class="estimate-currency">LKR</div><div class="estimate-value">{value * 100000:,.0f}</div>'
                 f'<div class="estimate-lakhs">{value:.2f} Lakhs</div><hr>'
                 f'<h3>{escape(brand)} {escape(model_name)}</h3>'
                 f'<p>{year} · {engine:,} cc · {mileage:,} km</p>'
                 f'<div class="estimate-tag">{escape(metadata["selected_model"])} ensemble estimate</div></div>')
        else:
            html('<div class="estimate-card empty"><div class="eyebrow">YOUR VEHICLE VALUATION</div>'
                 '<div class="estimate-symbol">↗</div><h2>A clearer picture<br>of your vehicle’s value.</h2>'
                 '<p>Complete the details and select <strong>Estimate vehicle price</strong>'
                 ' to see the result here.</p><div class="estimate-tag">LKR + Lakhs · One simple estimate</div></div>')
            if estimate:
                st.info("Vehicle details changed. Run the estimate again to update your result.")
        with st.container(border=True, key='panel-estimate-notes'):
            st.subheader("Good to know")
            st.write(f"This model uses historical listings up to {listing_year}. Prices are estimates of listing values, not guaranteed transaction prices.")
            st.caption("Condition, negotiation and changes in the market can affect the final selling price.")
            st.divider()
            st.caption(f'Average test error: {test_metrics["MAE_Lakhs"]:.2f} Lakhs. This is a dataset-level metric, not a confidence interval for your vehicle.')

html('<div class="app-footer"><span>AutoValue LK <span class="footer-dot">·</span> Vehicle intelligence, made clear.</span>'
     '<span>Historical listing data · Estimates in LKR</span></div>')
