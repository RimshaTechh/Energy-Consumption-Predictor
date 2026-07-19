# Appliances Energy Consumption Predictor
A machine learning web app that predicts household appliance energy consumption (in Wh), based on indoor sensor readings, outdoor weather conditions, and time of day.


## Problem Statement
Energy management in homes depends on understanding what drives appliance usage. This project predicts appliance energy consumption from sensor data available at any given moment — indoor temperature/humidity, outdoor weather, and time of day — so usage patterns can be anticipated instead of only measured after the fact.

## Dataset
[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) — ~19,735 readings taken at 10-minute intervals over ~4.5 months from a low-energy house in Belgium, combining indoor room sensors with an outdoor weather station.

## Approach
- Checked for missing values and duplicates — found none, so no imputation or deduplication was needed
- Identified `rv1` and `rv2` as synthetic random noise columns (identical statistics, near-zero correlation with the target) and dropped them
- Converted the raw `date` column into `hour`, `day_of_week`, `is_weekend`, and `month`, since appliance use follows a strong daily/weekly rhythm that a raw timestamp can't capture on its own
- Noted that no individual feature had a strong linear correlation with the target (all under 0.2), which ruled out linear models as a strong fit and pointed toward a tree-based approach that can capture non-linear interactions
- Trained a baseline Random Forest, found a large train/test gap (R² 0.94 vs 0.56), and ran a manual, one-variable-at-a-time tuning sweep (trees, depth, leaf size, feature sampling) instead of a blind grid search
- Found that `n_estimators=100` with unrestricted depth and `max_features='sqrt'` outperformed every other configuration tested, including much larger tree counts
- Built and ran an interactive Streamlit app for real-time predictions

## Results
| Metric | Score |
|---|---|
| R² (test) | 0.69 |
| RMSE (test) | 16.0 Wh |
| MAE (test) | 10.5 Wh |

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- Streamlit (UI + deployment)
- Matplotlib, Seaborn (EDA)



