# Code Documentation

This document explains the logic, inputs, and outputs of each Python script in the project.

## 1. `01_data_integration_eda_cleaning.py`
**Purpose**: Integrates monthly crime datasets, performs Exploratory Data Analysis (EDA), and cleans the data for downstream tasks.
-   **Inputs**: 9 CSV files in the `datasets/` folder (Jan-Sep 2025).
-   **Logic**:
    -   Loads each CSV, removes empty columns, and standardizes column names.
    -   Adds `MONTH_NAME` and `MONTH_INDEX` columns.
    -   Merges all dataframes into one.
    -   **EDA**: Calculates missing values, descriptive statistics, and visualizes the top 15 crime types and monthly trends.
    -   **Cleaning**: Fills missing values ('Unknown' for text, 0 for numbers) and removes duplicates.
    -   **Feature Engineering**: Adds a `SEVERITY` column based on keywords (Murder=High, Theft=Medium, etc.) if not present.
-   **Outputs**: `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv` (Cleaned Dataset).

## 2. `02_rising_trend_analysis.py`
**Purpose**: Identifies crime types that are showing a statistically significant rising trend over the 9-month period.
-   **Inputs**: `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`.
-   **Logic**:
    -   Aggregates crime counts by `CRIME_TYPE` and `MONTH_INDEX`.
    -   Filters out crimes with very low total occurrence (<10) to avoid noise.
    -   Applies **Simple Linear Regression** for each crime type (X=Month, y=Count).
    -   Calculates the **Slope** (rate of change). A positive slope indicates an increase.
-   **Outputs**:
    -   `rising_crime_trends.csv`: List of crimes sorted by slope.
    -   **Plot**: Top 5 rising crime trends.
    -   **Console Alerts**: Warnings for crimes with a slope > 0.5.

## 3. `03_severity_classification.py`
**Purpose**: Classifies crimes into severity levels (High, Medium, Low) based on their description.
-   **Inputs**: `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`.
-   **Logic**:
    -   Uses **TF-IDF Vectorization** to convert crime descriptions (`CRIME_TYPE`) into numerical features.
    -   Trains a **Decision Tree Classifier** to map these text features to the `SEVERITY` label (generated in step 1).
    -   Evaluates the model using Accuracy, Precision, Recall, and F1-Score.
-   **Outputs**:
    -   **Plot**: Confusion Matrix showing classification performance.
    -   **Console**: Top important words driving the classification (e.g., "Murder", "Theft").
    -   **Sample Predictions**: Demonstrates the model on new, unseen crime descriptions.

## 4. `04_hotspot_prediction.py`
**Purpose**: Predicts whether a specific crime type will be a "Hotspot" (High Frequency) in a given month.
-   **Inputs**: `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`.
-   **Logic**:
    -   Defines "High Crime" as a count above the median monthly count.
    -   Features: **TF-IDF** of Crime Type + **Month Index**.
    -   Trains **Logistic Regression** and **Naive Bayes** models.
    -   Compares models using ROC-AUC.
-   **Outputs**:
    -   **Plot**: ROC Curve comparing Logistic Regression and Naive Bayes.
    -   **Console**: Top factors contributing to high crime (Logistic Regression coefficients).
    -   **Predictions**: List of predicted high-risk crime types for the next month (October).

## 5. `05_crime_forecasting.py`
**Purpose**: Forecasts the exact number of cases for top crime types in the upcoming month.
-   **Inputs**: `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`.
-   **Logic**:
    -   Focuses on the Top 20 most frequent crime types.
    -   Uses **Multiple Linear Regression** (OLS) with **Polynomial Features** (Month^2) to capture non-linear trends.
    -   Features: One-Hot Encoded Crime Type + Month + Month^2.
-   **Outputs**:
    -   **Console**: OLS Regression Summary and Forecasted counts for Month 10 (October).
    -   **Plot**: Actual vs. Forecasted values for the top 3 crimes.

## 6. `06_public_safety_risk_model.py`
**Purpose**: Develops a comprehensive Risk Score to identify the most dangerous public safety threats.
-   **Inputs**: `CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv`.
-   **Logic**:
    -   Calculates three factors for each crime type:
        1.  **Frequency**: Total count.
        2.  **Severity**: Rule-based score (3=High, 2=Med, 1=Low).
        3.  **Trend**: Slope from linear regression.
    -   **Composite Risk Score**: Weighted sum (40% Frequency + 40% Severity + 20% Trend).
    -   Classifies crimes into **High/Medium/Low Risk** based on score quantiles.
    -   Trains **SVM** and **KNN** models to predict these risk levels.
-   **Outputs**:
    -   `public_safety_risk_scores.csv`: Detailed risk report.
    -   **Plot**: Scatter plot of Frequency vs. Risk Score.
    -   **Console**: List of Top 15 Highest Risk Crimes.
