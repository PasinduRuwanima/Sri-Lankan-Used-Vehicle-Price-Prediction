"""Data loading and cleaning for vehicle price prediction."""

from pathlib import Path
import pandas as pd


IDENTIFIER_COLUMNS: tuple[str, ...] = ("Unnamed: 0",)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the source CSV and validate that required columns exist."""
    data = pd.read_csv(path)
    required = {
        "Brand",
        "Model",
        "YOM",
        "Engine (cc)",
        "Gear",
        "Fuel Type",
        "Millage(KM)",
        "Town",
        "Date",
        "Leasing",
        "Condition",
        "AIR CONDITION",
        "POWER STEERING",
        "POWER MIRROR",
        "POWER WINDOW",
        "Price",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def normalize_text(value: object) -> object:
    """Normalize whitespace and underscore values for categorical fields."""
    if pd.isna(value):
        return value
    text = " ".join(str(value).strip().split())
    return text.replace("_", " ")


def clean_vehicle_data(
    data: pd.DataFrame,
    listing_max_year: int = 2025,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Clean the vehicle listing data and return an audit summary."""
    cleaned = data.copy()

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(normalize_text)

    cleaned = cleaned.drop(columns=list(IDENTIFIER_COLUMNS), errors="ignore")
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")

    numeric_columns = ["YOM", "Engine (cc)", "Millage(KM)", "Price"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    before = len(cleaned)
    valid_yom = cleaned["YOM"].between(1900, listing_max_year)
    valid_engine = cleaned["Engine (cc)"].between(100, 10000)
    valid_mileage = cleaned["Millage(KM)"].between(0, 2_000_000)
    valid_price = cleaned["Price"].gt(0)
    valid_date = cleaned["Date"].notna()

    valid_rows = valid_yom & valid_engine & valid_mileage & valid_price & valid_date
    cleaned = cleaned.loc[valid_rows].copy()
    invalid_rows_removed = before - len(cleaned)

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].fillna("Unknown")

    cleaned["Vehicle Age"] = cleaned["Date"].dt.year - cleaned["YOM"]
    valid_age = cleaned["Vehicle Age"].between(0, 100)
    cleaned = cleaned.loc[valid_age].copy()
    invalid_age_removed = valid_rows.sum() - len(cleaned)

    near_duplicate_columns = [
        column
        for column in cleaned.columns
        if column not in {"Date", "Vehicle Age"}
    ]
    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=near_duplicate_columns, keep="first")
    near_duplicates_removed = before_deduplication - len(cleaned)

    cleaned = cleaned.reset_index(drop=True)
    audit = {
        "raw_rows": int(before),
        "invalid_rows_removed": int(invalid_rows_removed),
        "invalid_age_removed": int(invalid_age_removed),
        "near_duplicates_removed": int(near_duplicates_removed),
        "clean_rows": int(len(cleaned)),
        "price_unit": "Lakhs",
    }
    return cleaned, audit
