import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

DATA_FILE = "CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found.")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.upper()

count_col = 'DURING THE CURRENT MONTH'

print(f"Using '{count_col}' as crime count column.")

df['MAJOR HEADS'] = df['MAJOR HEADS'].fillna('Unknown')
df['MINOR HEADS'] = df['MINOR HEADS'].fillna('Unknown')

def create_crime_type(row):
    major = str(row['MAJOR HEADS']).strip()
    minor = str(row['MINOR HEADS']).strip()
    if minor == 'Unknown' or minor == '':
        return major
    return f"{major} - {minor}"

df['CRIME_TYPE'] = df.apply(create_crime_type, axis=1)

monthly_stats = df.groupby(['CRIME_TYPE', 'MONTH_INDEX'])[count_col].sum().reset_index()
total_counts = monthly_stats.groupby('CRIME_TYPE')[count_col].sum()
active_crimes = total_counts[total_counts >= 10].index
monthly_stats = monthly_stats[monthly_stats['CRIME_TYPE'].isin(active_crimes)]

results = []
print("\n--- Training Linear Regression Models for Trend Detection ---")

unique_crimes = monthly_stats['CRIME_TYPE'].unique()

for crime in unique_crimes:
    crime_data = monthly_stats[monthly_stats['CRIME_TYPE'] == crime]
    full_months = pd.DataFrame({'MONTH_INDEX': range(1, 10)})
    crime_data = pd.merge(full_months, crime_data, on='MONTH_INDEX', how='left').fillna(0)
    
    X = crime_data[['MONTH_INDEX']]
    y = crime_data[count_col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    results.append({
        'Crime_Type': crime,
        'Slope': model.coef_[0],
        'Intercept': model.intercept_,
        'MAE': mean_absolute_error(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'R2': r2_score(y, y_pred),
        'Total_Count': y.sum()
    })

results_df = pd.DataFrame(results)
rising_trends = results_df.sort_values(by='Slope', ascending=False)

print("\n--- Top 10 Rising Crime Trends (Highest Increase Rate) ---")
print(rising_trends[['Crime_Type', 'Slope', 'Total_Count', 'R2']].head(10))

rising_trends.to_csv("rising_crime_trends.csv", index=False)
print("\nSaved trend analysis to rising_crime_trends.csv")

top_5_rising = rising_trends.head(5)['Crime_Type'].tolist()

plt.figure(figsize=(14, 8))
for crime in top_5_rising:
    crime_data = monthly_stats[monthly_stats['CRIME_TYPE'] == crime]
    full_months = pd.DataFrame({'MONTH_INDEX': range(1, 10)})
    crime_data = pd.merge(full_months, crime_data, on='MONTH_INDEX', how='left').fillna(0)
    sns.lineplot(x=crime_data['MONTH_INDEX'], y=crime_data[count_col], label=crime, marker='o')

plt.title("Top 5 Rising Crime Trends (Jan-Sep 2025)")
plt.xlabel("Month")
plt.ylabel("Crime Count")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- 🚨 TREND ALERTS 🚨 ---")
for index, row in rising_trends.head(5).iterrows():
    if row['Slope'] > 0.5:
        print(f"ALERT: {row['Crime_Type']} is increasing at a rate of {row['Slope']:.2f} cases/month.")
