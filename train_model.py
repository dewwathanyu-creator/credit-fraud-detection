"""
Run this script once to train the Random Forest model and save the pipeline.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib

print("Loading dataset...")
df = pd.read_csv("AIML Dataset.csv")
print(f"  {len(df):,} rows loaded")

# Feature engineering
df["balanceDiffOrig"]  = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"]  = df["newbalanceDest"] - df["oldbalanceDest"]
df["isAccountDrained"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)

df_model = df.drop(columns=["step", "nameOrig", "nameDest", "isFlaggedFraud"])

categorical = ["type"]
numeric = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balanceDiffOrig", "balanceDiffDest", "isAccountDrained"
]

X = df_model.drop("isFraud", axis=1)
y = df_model["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical)
], remainder="drop")

print("Training Random Forest...")
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ))
])
pipeline.fit(X_train, y_train)
print("  Done")

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Evaluation ---")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_proba):.4f}")

joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
print("\n✔ Saved → fraud_detection_pipeline.pkl")
