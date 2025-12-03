# Topics & Concepts Used

This document outlines the Machine Learning and Data Science concepts applied in this project.

## 1. Data Preprocessing & EDA
-   **Data Integration**: Merging multiple datasets (monthly CSVs) into a unified dataframe using `pandas.concat`.
-   **Cleaning**: Handling missing values (imputation) and removing duplicates.
-   **Exploratory Data Analysis (EDA)**: Using statistical summaries (`describe`) and visualizations (`seaborn`, `matplotlib`) to understand data distributions and trends.

## 2. Natural Language Processing (NLP)
-   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used in `03_severity_classification.py` and `04_hotspot_prediction.py`.
    -   Converts text data (Crime Descriptions) into numerical vectors.
    -   Highlights words that are unique to specific crime types while downweighting common words.

## 3. Regression Analysis
-   **Simple Linear Regression**: Used in `02_rising_trend_analysis.py`.
    -   Models the relationship between a single independent variable (Time/Month) and a dependent variable (Crime Count) to find the **Slope** (Trend).
-   **Multiple Linear Regression (OLS)**: Used in `05_crime_forecasting.py`.
    -   Predicts crime counts based on multiple variables (Crime Type, Month).
-   **Polynomial Regression**: Used in `05_crime_forecasting.py`.
    -   Adds polynomial terms (Month^2) to the regression model to capture non-linear patterns (e.g., seasonal spikes).

## 4. Classification Algorithms
-   **Decision Tree Classifier**: Used in `03_severity_classification.py`.
    -   A tree-structured model that splits data based on feature values to predict a target class (Severity). Selected for its interpretability.
-   **Logistic Regression**: Used in `04_hotspot_prediction.py`.
    -   A statistical model for binary classification (High Crime vs. Not High Crime). It estimates the probability of an event occurring.
-   **Naive Bayes (Multinomial)**: Used in `04_hotspot_prediction.py`.
    -   A probabilistic classifier based on Bayes' theorem. Effective for text classification tasks.
-   **Support Vector Machine (SVM)**: Used in `06_public_safety_risk_model.py`.
    -   Finds a hyperplane that best separates different classes (Risk Levels) in high-dimensional space.
-   **K-Nearest Neighbors (KNN)**: Used in `06_public_safety_risk_model.py`.
    -   Classifies a data point based on the majority class of its 'k' nearest neighbors.

## 5. Model Evaluation Metrics
-   **Accuracy**: The proportion of correct predictions.
-   **Precision & Recall**: Used to evaluate classification performance, especially for imbalanced classes.
-   **F1-Score**: The harmonic mean of precision and recall.
-   **MAE / MSE / RMSE**: Error metrics used to evaluate regression models (Forecasting).
-   **R-Squared (R²)**: Measures how well the regression model explains the variance in the data.
-   **ROC-AUC**: Evaluates the performance of binary classifiers (Hotspot Prediction) across different thresholds.
-   **Confusion Matrix**: Visualizes the performance of a classification model by showing True Positives, False Positives, etc.

## 6. Risk Scoring
-   **Composite Indexing**: Creating a new metric ("Risk Score") by combining multiple weighted factors (Frequency, Severity, Trend) to provide a holistic view of public safety threats.
