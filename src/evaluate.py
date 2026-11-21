"""Model evaluation helpers."""

from time import perf_counter

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regressor(
    model: RegressorMixin,
    X_test,
    y_test,
) -> dict[str, float]:
    """Calculate regression metrics in the original Lakhs target unit."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {
        "MAE_Lakhs": float(mae),
        "RMSE_Lakhs": float(rmse),
        "R2": float(r2),
    }


def timed_fit(model: RegressorMixin, X_train, y_train) -> tuple[RegressorMixin, float]:
    """Fit a model and report elapsed training seconds."""
    start = perf_counter()
    model.fit(X_train, y_train)
    elapsed = perf_counter() - start
    return model, float(elapsed)
