import pandas as pd

from src.data_processing import clean_vehicle_data, load_raw_data
from src.train import build_preprocessor, feature_columns, split_dataset


def test_cleaned_data_removes_identifier_and_has_valid_target():
    raw = load_raw_data("data/raw/car_price_dataset.csv")
    cleaned, audit = clean_vehicle_data(raw)

    assert "Unnamed: 0" not in cleaned.columns
    assert "Price" in cleaned.columns
    assert (cleaned["Price"] > 0).all()
    assert audit["clean_rows"] == len(cleaned)


def test_split_is_reproducible_and_excludes_date():
    data = pd.read_csv("data/processed/cleaned_vehicle_data.csv")
    X_train, X_test, y_train, y_test = split_dataset(data)

    assert len(X_train) == 7628
    assert len(X_test) == 1908
    assert "Date" not in X_train.columns
    assert "Price" not in feature_columns(data)
    assert y_train.name == "Price"
    assert y_test.name == "Price"


def test_preprocessor_transforms_training_and_test_data():
    data = pd.read_csv("data/processed/cleaned_vehicle_data.csv")
    X_train, X_test, _, _ = split_dataset(data)
    preprocessor = build_preprocessor(X_train)

    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    assert transformed_train.shape[0] == 7628
    assert transformed_test.shape[0] == 1908
