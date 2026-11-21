"""Training utilities for preprocessing and model preparation."""

from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


TARGET_COLUMN = "Price"
EXCLUDED_FEATURES = {TARGET_COLUMN, "Date"}


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Return model features while excluding target and direct identifiers."""
    return [column for column in data.columns if column not in EXCLUDED_FEATURES]


def split_dataset(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible random train/test split."""
    features = feature_columns(data)
    X = data[features].copy()
    y = data[TARGET_COLUMN].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build training-only-fitted numeric and categorical transformations."""
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def baseline_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Create Dummy and Ridge baseline pipelines."""
    return {
        "Dummy Mean": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }


def ensemble_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline | StackingRegressor | VotingRegressor]:
    """Create untuned ensemble model pipelines."""
    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    gradient_boosting = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        random_state=42,
    )
    xgboost = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    extra_trees = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    ridge = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", random_forest),
        ]
    )
    gbr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gradient_boosting),
        ]
    )
    xgb = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgboost),
        ]
    )
    extra = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", extra_trees),
        ]
    )

    voting = VotingRegressor(
        estimators=[
            ("ridge", ridge),
            ("random_forest", rf),
            ("gradient_boosting", gbr),
        ],
        n_jobs=-1,
    )
    stacking = StackingRegressor(
        estimators=[
            ("ridge", ridge),
            ("random_forest", rf),
            ("gradient_boosting", gbr),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    return {
        "Random Forest": rf,
        "Gradient Boosting": gbr,
        "XGBoost": xgb,
        "Extra Trees": extra,
        "Voting": voting,
        "Stacking": stacking,
    }
