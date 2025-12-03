import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
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

if 'SEVERITY' not in df.columns:
    def get_severity(crime_name):
        crime_name = str(crime_name).upper()
        if any(x in crime_name for x in ['MURDER', 'RAPE', 'ROBBERY', 'DACOITY', 'KIDNAPPING', 'ARSON', 'DOWRY DEATH', 'ACID', 'TRAFFICKING', 'POCSO']):
            return 'High'
        elif any(x in crime_name for x in ['THEFT', 'BURGLARY', 'ASSAULT', 'SNATCHING', 'RIOT', 'HURT', 'CHEATING', 'FORGERY', 'CYBER']):
            return 'Medium'
        else:
            return 'Low'
    df['SEVERITY'] = df['CRIME_TYPE'].apply(get_severity)

X_text = df['CRIME_TYPE']
y = df['SEVERITY']

tfidf = TfidfVectorizer(stop_words='english', max_features=500)
X = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(random_state=42, max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- Model Performance: Decision Tree Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['High', 'Medium', 'Low'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High', 'Medium', 'Low'], yticklabels=['High', 'Medium', 'Low'])
plt.title("Confusion Matrix - Severity Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_names = tfidf.get_feature_names_out()
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- Top 10 Most Important Words for Classification ---")
for i in range(10):
    if i < len(indices):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

print("\n--- Severity Prioritization System (Sample Predictions) ---")
sample_crimes = [
    "Murder for Gain",
    "Theft of Bicycle",
    "Cyber Crime - Phishing",
    "Kidnapping of Child",
    "Public Nuisance"
]
sample_vec = tfidf.transform(sample_crimes)
sample_preds = clf.predict(sample_vec)

for crime, severity in zip(sample_crimes, sample_preds):
    priority = "URGENT ACTION" if severity == 'High' else ("IMMEDIATE REVIEW" if severity == 'Medium' else "ROUTINE CHECK")
    print(f"Crime: {crime:<30} | Severity: {severity:<10} | Priority: {priority}")
