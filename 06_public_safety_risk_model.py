import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

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

frequency = df.groupby('CRIME_TYPE')[count_col].sum().reset_index()
frequency.rename(columns={count_col: 'Total_Count'}, inplace=True)

slopes = []
for crime in frequency['CRIME_TYPE']:
    crime_data = df[df['CRIME_TYPE'] == crime].groupby('MONTH_INDEX')[count_col].sum().reset_index()
    if len(crime_data) > 1:
        model = LinearRegression()
        model.fit(crime_data[['MONTH_INDEX']], crime_data[count_col])
        slope = model.coef_[0]
    else:
        slope = 0
    slopes.append(slope)

frequency['Trend_Slope'] = slopes

def get_severity_score(crime_name):
    crime_name = str(crime_name).upper()
    if any(x in crime_name for x in ['MURDER', 'RAPE', 'ROBBERY', 'DACOITY', 'KIDNAPPING', 'ARSON', 'DOWRY DEATH', 'ACID', 'TRAFFICKING', 'POCSO']):
        return 3
    elif any(x in crime_name for x in ['THEFT', 'BURGLARY', 'ASSAULT', 'SNATCHING', 'RIOT', 'HURT', 'CHEATING', 'FORGERY', 'CYBER']):
        return 2
    else:
        return 1

frequency['Severity_Score'] = frequency['CRIME_TYPE'].apply(get_severity_score)

scaler = MinMaxScaler()
features_to_scale = ['Total_Count', 'Trend_Slope', 'Severity_Score']
frequency_scaled = frequency.copy()
frequency_scaled[features_to_scale] = scaler.fit_transform(frequency[features_to_scale])

frequency['Risk_Score'] = (
    0.4 * frequency_scaled['Total_Count'] + 
    0.4 * frequency_scaled['Severity_Score'] + 
    0.2 * frequency_scaled['Trend_Slope']
)

q1 = frequency['Risk_Score'].quantile(0.25)
q3 = frequency['Risk_Score'].quantile(0.75)

def get_risk_level(score):
    if score >= q3:
        return 'High'
    elif score >= q1:
        return 'Medium'
    else:
        return 'Low'

frequency['Risk_Level'] = frequency['Risk_Score'].apply(get_risk_level)

print("\n--- Risk Level Distribution ---")
print(frequency['Risk_Level'].value_counts())

X = frequency[['Total_Count', 'Trend_Slope', 'Severity_Score']]
y = frequency['Risk_Level']
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "SVM": SVC(kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

print("\n--- Model Evaluation ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=['High', 'Medium', 'Low']))

print("\n--- Public Safety Risk Classification (Top 15 Highest Risk Crimes) ---")
top_risk_crimes = frequency.sort_values(by='Risk_Score', ascending=False).head(15)
print(top_risk_crimes[['CRIME_TYPE', 'Total_Count', 'Severity_Score', 'Trend_Slope', 'Risk_Level']])

frequency.to_csv("public_safety_risk_scores.csv", index=False)
print("\nSaved risk scores to public_safety_risk_scores.csv")

plt.figure(figsize=(12, 8))
sns.scatterplot(data=frequency, x='Total_Count', y='Risk_Score', hue='Risk_Level', palette={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
plt.title("Public Safety Risk Analysis: Frequency vs Risk Score")
plt.xlabel("Total Crime Count")
plt.ylabel("Calculated Risk Score")
plt.show()
