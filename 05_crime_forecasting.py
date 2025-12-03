import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

df['MAJOR HEADS'] = df['MAJOR HEADS'].fillna('Unknown')
df['MINOR HEADS'] = df['MINOR HEADS'].fillna('Unknown')

def create_crime_type(row):
    major = str(row['MAJOR HEADS']).strip()
    minor = str(row['MINOR HEADS']).strip()
    if minor == 'Unknown' or minor == '':
        return major
    return f"{major} - {minor}"

df['CRIME_TYPE'] = df.apply(create_crime_type, axis=1)

count_col = 'DURING THE CURRENT MONTH'

df[count_col] = pd.to_numeric(df[count_col], errors='coerce').fillna(0)
monthly_stats = df.groupby(['CRIME_TYPE', 'MONTH_INDEX'])[count_col].sum().reset_index()

top_crimes = monthly_stats.groupby('CRIME_TYPE')[count_col].sum().sort_values(ascending=False).head(20).index
print(f"Modeling Top 20 Crime Types: {list(top_crimes)}")

data_top = monthly_stats[monthly_stats['CRIME_TYPE'].isin(top_crimes)].copy()
data_encoded = pd.get_dummies(data_top, columns=['CRIME_TYPE'], drop_first=True)

X = data_encoded.drop([count_col], axis=1).astype(float)
y = data_encoded[count_col].astype(float)

poly = PolynomialFeatures(degree=2, include_bias=False)
month_poly = poly.fit_transform(X[['MONTH_INDEX']])
poly_cols = [f"MONTH_INDEX_^{i}" for i in range(1, 3)]
month_poly_df = pd.DataFrame(month_poly, columns=poly_cols, index=X.index)

X_final = pd.concat([month_poly_df, X.drop('MONTH_INDEX', axis=1)], axis=1)
X_final_const = sm.add_constant(X_final)

model = sm.OLS(y, X_final_const).fit()
print("\n--- OLS Regression Summary ---")
print(model.summary())

y_pred = model.predict(X_final_const)
print("\n--- Model Performance Metrics ---")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"R2 Score: {r2_score(y, y_pred):.4f}")

print("\n--- Crime Forecasting for Month 10 (October) ---")
forecasts = []

for crime in top_crimes:
    input_df = pd.DataFrame({'CRIME_TYPE': [crime], 'MONTH_INDEX': [10]})
    temp_df = pd.concat([data_top[['CRIME_TYPE', 'MONTH_INDEX']], input_df], ignore_index=True)
    temp_encoded = pd.get_dummies(temp_df, columns=['CRIME_TYPE'], drop_first=True)
    
    input_row = temp_encoded.iloc[[-1]].drop('MONTH_INDEX', axis=1).astype(float)
    month_val = temp_encoded.iloc[[-1]][['MONTH_INDEX']].astype(float)
    month_poly_val = poly.transform(month_val)
    month_poly_df_val = pd.DataFrame(month_poly_val, columns=poly_cols, index=input_row.index)
    
    input_final = pd.concat([month_poly_df_val, input_row], axis=1)
    input_final = input_final.reindex(columns=X_final.columns, fill_value=0)
    input_final_const = sm.add_constant(input_final, has_constant='add')
    
    pred = model.predict(input_final_const).iloc[0]
    forecasts.append({'Crime_Type': crime, 'Predicted_Count': max(0, pred)})

forecast_df = pd.DataFrame(forecasts).sort_values(by='Predicted_Count', ascending=False)
print(forecast_df)
print(f"\nTotal Predicted Crime Load for Top 20 Crimes in Month 10: {forecast_df['Predicted_Count'].sum():.2f}")

top_3_forecast = forecast_df.head(3)['Crime_Type'].tolist()
plt.figure(figsize=(14, 8))
for crime in top_3_forecast:
    actual_data = data_top[data_top['CRIME_TYPE'] == crime]
    plt.plot(actual_data['MONTH_INDEX'], actual_data[count_col], marker='o', label=f"Actual - {crime}")
    pred_val = forecast_df[forecast_df['Crime_Type'] == crime]['Predicted_Count'].values[0]
    plt.plot(10, pred_val, marker='x', markersize=10, linestyle='None', label=f"Forecast - {crime}")

plt.title("Crime Forecasting: Actual (Jan-Sep) vs Forecast (Oct)")
plt.xlabel("Month Index")
plt.ylabel("Crime Count")
plt.legend()
plt.grid(True)
plt.show()
