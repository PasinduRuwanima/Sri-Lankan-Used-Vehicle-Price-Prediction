import joblib
import pandas as pd


def test_saved_model_loads_and_predicts():
    model = joblib.load("models/best_model.joblib")
    data = pd.read_csv("data/processed/cleaned_vehicle_data.csv")
    sample = data.drop(columns=["Price", "Date"]).iloc[[0]]

    prediction = float(model.predict(sample)[0])

    assert prediction > 0
    assert prediction * 100000 > 0
