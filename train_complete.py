"""Train and compare the complete ensemble learning suite."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.data_processing import clean_vehicle_data, load_raw_data
from src.evaluate import evaluate_regressor, timed_fit
from src.train import build_preprocessor, ensemble_models, split_dataset


ROOT = Path(__file__).resolve().parent


def main() -> None:
    raw_path = ROOT / "data" / "raw" / "car_price_dataset.csv"
    processed_path = ROOT / "data" / "processed"
    artifacts_path = ROOT / "artifacts"
    models_path = ROOT / "models"
    processed_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(raw_path)
    data, audit = clean_vehicle_data(raw)
    data.to_csv(processed_path / "cleaned_vehicle_data.csv", index=False)
    (artifacts_path / "data_cleaning_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    X_train, X_test, y_train, y_test = split_dataset(data)
    preprocessor = build_preprocessor(X_train)
    models = ensemble_models(preprocessor)

    results = []
    predictions_by_model = {}
    cv_results = []
    for name, model in models.items():
        print(f"Training {name}...")
        fitted, elapsed = timed_fit(model, X_train, y_train)
        metrics = evaluate_regressor(fitted, X_test, y_test)
        cv_mae = cross_val_score(
            fitted,
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=3,
            n_jobs=-1,
        )
        cv_mae_lakhs = float((-cv_mae).mean())
        row = {
            "Model": name,
            "Training Time (s)": elapsed,
            "CV MAE Lakhs": cv_mae_lakhs,
            **metrics,
        }
        results.append(row)
        predictions_by_model[name] = fitted.predict(X_test)
        cv_results.append({
            "Model": name,
            "CV MAE Lakhs": cv_mae_lakhs,
            "CV MAE Std Lakhs": float((-cv_mae).std()),
        })

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(artifacts_path / "ensemble_results.csv", index=False)
    result_frame.rename(columns={
        "CV MAE Lakhs": "CV MAE Lakhs",
        "MAE_Lakhs": "MAE_Lakhs",
        "RMSE_Lakhs": "RMSE_Lakhs",
        "R2": "R2",
    }).to_csv(artifacts_path / "model_results.csv", index=False)
    pd.DataFrame(cv_results).to_csv(
        artifacts_path / "cross_validation_results.csv",
        index=False,
    )

    selected_name = result_frame.sort_values(
        ["CV MAE Lakhs", "MAE_Lakhs"]
    ).iloc[0]["Model"]
    selected = models[selected_name]
    selected.fit(X_train, y_train)
    joblib.dump(selected, models_path / "best_model.joblib")

    if hasattr(selected, "named_steps") and hasattr(
        selected.named_steps["model"],
        "feature_importances_",
    ):
        encoded_names = selected.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        importance_frame = pd.DataFrame({
            "Feature": encoded_names,
            "Importance": selected.named_steps[
                "model"
            ].feature_importances_,
        }).sort_values("Importance", ascending=False)
        importance_frame.to_csv(
            artifacts_path / "feature_importance.csv",
            index=False,
        )

    predictions = pd.DataFrame({
        "Actual Lakhs": y_test.to_numpy(),
        "Predicted Lakhs": predictions_by_model[selected_name],
    })
    predictions["Residual Lakhs"] = (
        predictions["Actual Lakhs"] - predictions["Predicted Lakhs"]
    )
    predictions.to_csv(
        artifacts_path / "test_predictions.csv",
        index=False,
    )

    metadata = {
        "selected_model": selected_name,
        "selection_metric": "3-fold cross-validation MAE",
        "CV MAE Lakhs": float(
            result_frame.loc[
                result_frame["Model"] == selected_name,
                "CV MAE Lakhs",
            ].iloc[0]
        ),
        "test_metrics": {
            key: float(
                result_frame.loc[
                    result_frame["Model"] == selected_name,
                    key,
                ].iloc[0]
            )
            for key in ("MAE_Lakhs", "RMSE_Lakhs", "R2")
        },
        "price_unit": "Lakhs",
        "price_lkr_multiplier": 100000,
        "random_state": 42,
        "feature_names": X_train.columns.tolist(),
    }
    (artifacts_path / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (artifacts_path / "input_categories.json").write_text(
        json.dumps(
            {
                column: sorted(data[column].dropna().astype(str).unique().tolist())
                for column in [
                    "Brand", "Gear", "Fuel Type", "Town", "Leasing",
                    "Condition", "AIR CONDITION", "POWER STEERING",
                    "POWER MIRROR", "POWER WINDOW",
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result_frame.to_string(index=False))
    print(f"Selected deployment model: {selected_name}")


if __name__ == "__main__":
    main()
