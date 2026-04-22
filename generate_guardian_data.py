import pandas as pd
import json
import os

# Paths
ROOT_CSV = 'CRIME_REVIEW_FOR_MONTHS_FROM_JAN_TO_SEP.csv'
TRENDS_CSV = 'dashboard/rising_crime_trends.csv'
RISK_CSV = 'dashboard/public_safety_risk_scores.csv'
OUTPUT_DIR = 'dashboard/public/outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Guardian Data Synthesis ---")

# 1. Global KPIs
try:
    df = pd.read_csv(ROOT_CSV)
    df.columns = df.columns.str.strip().str.upper()
    
    # Use standard names
    count_col = 'DURING THE CURRENT MONTH'
    
    stats = {
        'total_records': len(df),
        'total_crimes_detected': int(df[count_col].sum()),
        'unique_categories': int(df['MAJOR HEADS'].nunique()),
        'avg_monthly_load': float(df.groupby('MONTH_INDEX')[count_col].sum().mean()),
        'severity_breakdown': df['SEVERITY'].value_counts().to_dict() if 'SEVERITY' in df.columns else {},
        'monthly_volatility': df.groupby('MONTH_INDEX')[count_col].sum().pct_change().fillna(0).tolist()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data_01.json'), 'w') as f:
        json.dump(stats, f)
    print("Created data_01.json (Global KPIs)")
except Exception as e:
    print(f"Error in data_01: {e}")

# 2. Intelligence Ledger (Master Merge)
try:
    df_trends = pd.read_csv(TRENDS_CSV)
    df_risk = pd.read_csv(RISK_CSV)
    
    # Merge on Crime_Type
    # Trends cols: Crime_Type, Slope, Total_Count, R2, etc.
    # Risk cols: CRIME_TYPE, Total_Count, Severity_Score, Trend_Slope, Risk_Score, Risk_Level
    
    # Standardize column naming for merge
    df_trends.rename(columns={'Crime_Type': 'KEY'}, inplace=True)
    df_risk.rename(columns={'CRIME_TYPE': 'KEY'}, inplace=True)
    
    master_ledger = pd.merge(df_risk, df_trends[['KEY', 'Slope', 'R2']], on='KEY', how='left')
    
    # Final cleanup
    master_ledger = master_ledger.fillna(0)
    
    ledger_data = master_ledger.to_dict(orient='records')
    
    with open(os.path.join(OUTPUT_DIR, 'ledger.json'), 'w') as f:
        json.dump(ledger_data, f)
    print(f"Created ledger.json with {len(ledger_data)} intelligence entries.")
    
except Exception as e:
    print(f"Error in ledger: {e}")

print("--- Synthesis Complete ---")
