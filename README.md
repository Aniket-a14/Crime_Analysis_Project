# Crime Analysis & Prediction Project

## Overview
This project performs a comprehensive analysis of monthly crime datasets (January to September 2025) to identify trends, classify crime severity, predict hotspots, forecast future crime loads, and assess public safety risks. The goal is to provide actionable insights for law enforcement and public safety planning.

## Objectives
1.  **Data Integration & EDA**: Merge monthly datasets and explore crime distributions.
2.  **Rising Trend Detection**: Identify crimes that are increasing rapidly.
3.  **Severity Classification**: Classify crimes as High, Medium, or Low severity.
4.  **Hotspot Prediction**: Predict which crime types will be high-frequency hotspots.
5.  **Crime Forecasting**: Forecast crime counts for the upcoming month.
6.  **Risk Scoring**: Calculate a composite risk score for each crime type.

## Project Structure
- `datasets/`: Folder containing monthly CSV files (Jan-Sep 2025).
- `01_data_integration_eda_cleaning.py`: Merges data, performs EDA, and cleans the dataset.
- `02_rising_trend_analysis.py`: Detects rising crime trends using Linear Regression.
- `03_severity_classification.py`: Classifies crime severity using Decision Trees and TF-IDF.
- `04_hotspot_prediction.py`: Predicts high-crime hotspots using Logistic Regression.
- `05_crime_forecasting.py`: Forecasts future crime counts using Polynomial Regression.
- `06_public_safety_risk_model.py`: Calculates public safety risk scores using a composite index.
- `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`: The final cleaned and merged dataset.
- `requirements.txt`: List of Python dependencies.

## How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Scripts in Order**:
    ```bash
    python 01_data_integration_eda_cleaning.py
    python 02_rising_trend_analysis.py
    python 03_severity_classification.py
    python 04_hotspot_prediction.py
    python 05_crime_forecasting.py
    python 06_public_safety_risk_model.py
    ```

## Outputs
-   **Visualizations**: Plots are displayed during script execution (e.g., Trend Lines, Confusion Matrices, ROC Curves).
-   **CSV Reports**:
    -   `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`
    -   `rising_crime_trends.csv`
    -   `public_safety_risk_scores.csv`
