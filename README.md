# Appliances Energy Consumption Predictor

A machine learning web app that predicts household appliance energy consumption (in Wh), based on indoor sensor readings, outdoor weather conditions, and time of day.

## Problem Statement

Energy management in homes depends on understanding what drives appliance usage. This project predicts appliance energy consumption from sensor data available at any given moment — indoor temperature/humidity, outdoor weather, and time of day — so usage patterns can be anticipated instead of only measured after the fact.

## Dataset

[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) — ~19,735 readings taken at 10-minute intervals over ~4.5 months from a low-energy house in Belgium, combining indoor room sensors with an outdoor weather station.

## Approach

- Checked for missing values and duplicates — found none, so no imputation or deduplication was needed
- Identified `rv1` and `rv2` as synthetic random noise columns (identical statistics, near-zero correlation with the target) and dropped them
- Removed outliers from the target variable using the IQR method, to reduce the influence of extreme energy-usage spikes on model training
- Converted the raw `date` column into `hour`, `day_of_week`, `is_weekend`, and `month`, since appliance use follows a strong daily/weekly rhythm that a raw timestamp can't capture on its own
- Noted that no individual feature had a strong linear correlation with the target (all under 0.2), which pointed toward tree-based models that can capture non-linear interactions
- Trained and compared three models — Decision Tree, Random Forest, and XGBoost — and selected Random Forest based on the best R², RMSE, and MAE on the test set
- Reduced the final model's tree count to 100 estimators to keep the file size deployable on GitHub, after confirming it performs nearly identically to larger configurations tested during tuning
- Built and ran an interactive Streamlit app for real-time predictions

## Model Comparison

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Random Forest** | **0.692** | **16.06** | **10.60** |
| XGBoost | 0.632 | 17.57 | 11.86 |
| Decision Tree | 0.551 | 19.39 | 12.72 |

Random Forest outperformed the other models across every metric and was selected as the final model. Its bagging approach (averaging many trees) handles this dataset's noise better than a single Decision Tree, and XGBoost was run with default, untuned hyperparameters — it may perform better with proper tuning, but that wasn't the focus of this comparison.

## Results (Final Model)

| Metric | Score |
|---|---|
| R² (test) | 0.69 |
| RMSE (test) | 16.06 Wh |
| MAE (test) | 10.60 Wh |

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- Streamlit (UI + deployment)
- Matplotlib, Seaborn (EDA)
