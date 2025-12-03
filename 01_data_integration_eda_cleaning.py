import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

DATA_DIR = r"c:\3rd Year\INT_234\Crime_Analysis_Project\datasets"
FILES = {
    "January": "CRIME_REVIEW_FOR_THE_MONTH_OF_JANUARY_2025.csv",
    "February": "CRIME_REVIEW_FOR_THE_MONTH_OF_FEBRUARY_2025_0.csv",
    "March": "CRIME_REVIEW_FOR_THE_MONTH_OF_MARCH_2025_0.csv",
    "April": "CRIME_REVIEW_FOR_THE_MONTH_OF_APRIL_2025.csv",
    "May": "CRIME_REVIEW_FOR_THE_MONTH_OF_MAY_2025_0.csv",
    "June": "CRIME_REVIEW_FOR_THE_MONTH_OF_JUNE_2025.csv",
    "July": "CRIME_REVIEW_FOR_THE_MONTH_OF_JULY_2025_0_0.csv",
    "August": "CRIME_REVIEW_FOR_THE_MONTH_OF_AUGUST_2025.csv",
    "September": "CRIME_REVIEW_FOR_THE_MONTH_OF_SEPTEMBER_2025_0.csv"
}

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9
}

dfs = []
print("Loading datasets...")
for month_name, filename in FILES.items():
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        continue
    
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.dropna(axis=1, how='all', inplace=True)
        df.columns = df.columns.str.strip().str.upper()
        df['MONTH_NAME'] = month_name
        df['MONTH_INDEX'] = MONTH_MAP[month_name]
        df['YEAR'] = 2025
        dfs.append(df)
        print(f"Loaded {month_name}: {df.shape}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if not dfs:
    raise ValueError("No datasets loaded!")

df_all = pd.concat(dfs, ignore_index=True)

print("\nMerged Dataset Info:")
print(f"Shape: {df_all.shape}")
print(df_all.info())

print("\n--- 2.1 Missing Values Analysis ---")
missing_percent = df_all.isnull().mean() * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

print("\n--- 2.2 Descriptive Statistics ---")
print(df_all.describe(include='all'))

print("\n--- 2.3 Crime Distribution ---")
crime_col = 'HEADS CRIME'

if crime_col:
    print(f"Using '{crime_col}' as crime type column.")
    plt.figure(figsize=(14, 8))
    top_crimes = df_all[crime_col].value_counts().head(15)
    sns.barplot(y=top_crimes.index, x=top_crimes.values, palette="viridis")
    plt.title("Top 15 Crime Types (Jan-Sep 2025)")
    plt.xlabel("Count")
    plt.ylabel("Crime Type")
    plt.tight_layout()
    plt.show()
else:
    print("Could not identify Crime Type column automatically.")

print("\n--- 2.4 Time-based Crime Trends ---")
monthly_counts = df_all.groupby('MONTH_INDEX').size()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o')
plt.title("Total Crimes per Month (Jan-Sep 2025)")
plt.xlabel("Month Index")
plt.ylabel("Total Crimes")
plt.xticks(range(1, 10))
plt.grid(True)
plt.show()

print("\n--- Data Cleaning ---")
df_clean = df_all.copy()

for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].fillna('Unknown')
    else:
        df_clean[col] = df_clean[col].fillna(0)

df_clean.drop_duplicates(inplace=True)

if 'SEVERITY' not in df_clean.columns and crime_col:
    def get_severity(crime_name):
        crime_name = str(crime_name).upper()
        if any(x in crime_name for x in ['MURDER', 'RAPE', 'ROBBERY', 'DOWRY DEATH']):
            return 'High'
        elif any(x in crime_name for x in ['THEFT', 'BURGLARY', 'ASSAULT', 'SNATCHING']):
            return 'Medium'
        else:
            return 'Low'
    
    df_clean['SEVERITY'] = df_clean[crime_col].apply(get_severity)

print("\nFinal Cleaned Dataset Info:")
print(df_clean.info())

output_file = "CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv"
df_clean.to_csv(output_file, index=False)
print(f"\nSuccessfully saved cleaned dataset to {output_file}")
