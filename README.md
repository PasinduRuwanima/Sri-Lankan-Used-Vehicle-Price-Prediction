# Sri Lankan Used Vehicle Price Prediction Using Ensemble Learning

An undergraduate machine learning project that predicts used vehicle prices in Sri Lankan Rupees using tabular regression and ensemble methods.

## Price Unit

The target `Price` is stored in **Lakhs**:

- 1 Lakh = LKR 100,000
- 75.50 Lakhs = LKR 7,550,000

Training uses the original Lakhs values. MAE and RMSE are reported in Lakhs; the dashboard converts predictions to LKR by multiplying by 100,000.

## Dataset

The source file is `data/raw/car_price_dataset.csv`. It contains 9,788 listings and 17 columns including brand, model, manufacturing year, engine capacity, transmission, fuel type, mileage, town, listing date, leasing status, condition, feature availability, and price.

## Cleaning

- Removed the `Unnamed: 0` identifier column.
- Normalized text whitespace and underscore formatting.
- Parsed listing dates.
- Validated manufacturing year, engine capacity, mileage, price, and listing dates.
- Added `Vehicle Age` using each listing date year minus manufacturing year.
- Removed 252 near-duplicate listings using the listing fields excluding `Date`.
- Result: 9,536 clean rows.

## Methodology

- 80/20 random train/test split with `random_state=42`.
- Training rows: 7,628.
- Test rows: 1,908.
- Numeric median imputation and standardization.
- Categorical most-frequent imputation and one-hot encoding with unseen-category handling.
- Models: Dummy Mean, Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, Voting, and Stacking.
- Boosting coverage also includes untuned LightGBM and CatBoost candidates.
- Hybrid coverage includes an `Random Forest + ANN` stacking model with Ridge as the meta-learner.

## Real Results

Primary model-selection metric: 3-fold cross-validation MAE in Lakhs.

| Model | CV MAE (Lakhs) | Test MAE (Lakhs) | Test RMSE (Lakhs) | Test R-squared |
|---|---:|---:|---:|---:|
| Extra Trees | 5.304 | 5.150 | 15.079 | 0.894 |
| Random Forest | 5.495 | 5.356 | 14.983 | 0.896 |
| Stacking | 5.799 | 5.815 | 14.678 | 0.900 |
| RF + ANN Hybrid | 5.505 | 5.597 | 14.268 | 0.906 |
| XGBoost | 6.526 | 6.463 | 14.286 | 0.905 |
| LightGBM | 6.904 | 6.550 | 13.648 | 0.914 |
| Voting | 6.479 | 6.604 | 15.815 | 0.884 |
| Gradient Boosting | 8.275 | 8.533 | 16.795 | 0.869 |
| CatBoost | 8.144 | 8.462 | 16.706 | 0.870 |

The Extra Trees model is selected for deployment because it has the lowest 3-fold cross-validation MAE. LightGBM has the lowest held-out RMSE and highest held-out R-squared in this run, but model selection is based on the cross-validation MAE to avoid selecting from test-set performance.

The selected deployment model is Extra Trees because it had the lowest cross-validation MAE among the cross-validated candidates.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Run Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

## Notebooks

The notebooks are:

- `notebooks/ensemble_comparison.ipynb`: executed complete ensemble comparison, tuning results, held-out metrics, charts, dataset analysis, and interpretation.

To execute the complete notebook again:

```powershell
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute notebooks\ensemble_comparison.ipynb --inplace --ExecutePreprocessor.timeout=600
```

## Streamlit Cloud Deployment

1. Push this project to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select the repository and set the main file to `app.py`.
4. Set Python dependencies to `requirements.txt`.
5. Deploy only after reviewing the repository and results.

No deployment has been performed automatically.

## Limitations

Predictions reflect historical listing data and may not capture negotiation, defects, market shocks, or future price changes. They are estimates, not guaranteed transaction prices.

## AI Usage

GitHub Copilot was used as a development aid for project code and dashboard UI work. Final implementation, model execution, results, and project documentation were reviewed locally.
