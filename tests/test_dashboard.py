"""Exercise the dashboard's navigation and stateful user flows."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_all_dashboard_pages_and_overview_shortcut():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45)
    app.session_state["navigation"] = "About the Project"
    app.run()
    assert not app.exception
    assert len(app.radio) == 1
    assert app.radio[0].value == "Overview"
    app.button[0].click().run()
    assert app.radio[0].value == "Vehicle Price Prediction"
    for page in [
        "Dataset Exploration", "Ensemble Model Comparison",
        "Model Performance",
    ]:
        app.radio[0].set_value(page).run()
        assert not app.exception


def test_estimate_tracks_vehicle_details():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    app.radio[0].set_value("Vehicle Price Prediction").run()
    app.selectbox[0].set_value("TOYOTA").run()
    data = pd.read_csv("data/processed/cleaned_vehicle_data.csv")
    assert app.selectbox[1].value in set(data.loc[data["Brand"] == "TOYOTA", "Model"])
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["estimate"]["lakhs"] > 0
    assert any('class="estimate-value"' in item.value for item in app.markdown)
    app.number_input[2].set_value(90000).run()
    assert any("details changed" in item.value for item in app.info)
    assert not any('class="estimate-value"' in item.value for item in app.markdown)
    app.button[0].click().run()
    assert app.session_state["estimate"]["input"]["Millage(KM)"] == 90000
    assert not app.exception


def test_dataset_filter_resets_pagination():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    app.radio[0].set_value("Dataset Exploration").run()
    data = pd.read_csv("data/processed/cleaned_vehicle_data.csv")
    app.button[1].click().run()
    assert app.dataframe[0].value.iloc[0]["Model"] == data.iloc[100]["Model"]
    app.multiselect[0].set_value(["AUDI"]).run()
    assert app.session_state["dataset_page"] == 1
    assert set(app.dataframe[0].value["Brand"]) == {"AUDI"}
    assert not app.exception
