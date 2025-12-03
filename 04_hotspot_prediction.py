import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.sparse import hstack
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

monthly_stats = df.groupby(['CRIME_TYPE', 'MONTH_INDEX'])[count_col].sum().reset_index()
median_count = monthly_stats[count_col].median()
monthly_stats['High_Crime'] = (monthly_stats[count_col] > median_count).astype(int)

X_text = monthly_stats['CRIME_TYPE']
X_month = monthly_stats[['MONTH_INDEX']]
y = monthly_stats['High_Crime']

tfidf = TfidfVectorizer(stop_words='english', max_features=500)
X_text_tfidf = tfidf.fit_transform(X_text)
X = hstack([X_text_tfidf, X_month])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = {}
print("\n--- Model Evaluation ---")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "y_test": y_test,
        "y_prob": y_prob
    }
    
    print(f"\n{name}:")
    print(f"Accuracy: {results[name]['Accuracy']:.4f}")
    print(f"Precision: {results[name]['Precision']:.4f}")
    print(f"Recall: {results[name]['Recall']:.4f}")
    print(f"F1 Score: {results[name]['F1']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 8))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - High-Crime Prediction')
plt.legend(loc="lower right")
plt.show()

lr_model = models["Logistic Regression"]
feature_names = list(tfidf.get_feature_names_out()) + ['MONTH_INDEX']
coefs = lr_model.coef_[0]
top_indices = np.argsort(coefs)[::-1][:10]

print("\n--- Top 10 Factors Contributing to High Crime (Logistic Regression) ---")
for i in top_indices:
    print(f"{feature_names[i]}: {coefs[i]:.4f}")

unique_crimes = monthly_stats['CRIME_TYPE'].unique()
next_month = 10
next_month_data = pd.DataFrame({'CRIME_TYPE': unique_crimes, 'MONTH_INDEX': next_month})

X_next_text = tfidf.transform(next_month_data['CRIME_TYPE'])
X_next_month = next_month_data[['MONTH_INDEX']]
X_next = hstack([X_next_text, X_next_month])

next_month_data['Probability'] = lr_model.predict_proba(X_next)[:, 1]
high_risk_crimes = next_month_data.sort_values(by='Probability', ascending=False)

print(f"\n--- Predicted High-Crime Hotspots (Crime Types) for Month {next_month} ---")
print(high_risk_crimes[['CRIME_TYPE', 'Probability']].head(10))
