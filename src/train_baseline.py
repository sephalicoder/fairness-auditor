"""
train_baseline.py
Trains a baseline classifier on the Adult Income dataset, WITHOUT using sex/race as direct input features.
Saves the trained model + test split so later steps can reuse them without retraining.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_adult_data

MODEL_DIR = Path("data")
MODEL_DIR.mkdir(exist_ok=True)

def build_preprocessing_pipeline(X: pd.DataFrame):
    """Builds a ColumnTransformer that handles numeric + categorical columns."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def train_and_evaluate():
    # 1. Load the raw data arrays
    X_raw, y, sensitive = load_adult_data()
    
    # 2. Drop the sensitive features here to prevent leakage and clear the assertion
    X = X_raw.drop(columns=["sex", "race"])
    
    # Sanity check: make sure sex/race are NOT in the training features
    assert "sex" not in X.columns, "sex column leaked into features!"
    assert "race" not in X.columns, "race column leaked into features!"
    
    # Split data — keep sensitive attributes aligned with the same split via indices
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessing_pipeline(X)
    
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    
    print("\nTraining baseline Logistic Regression classifier...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Save everything needed for fairness analysis later
    joblib.dump(clf, MODEL_DIR / "baseline_model.joblib")
    X_test.to_pickle(MODEL_DIR / "X_test.pkl")
    y_test.to_pickle(MODEL_DIR / "y_test.pkl")
    sens_test.to_pickle(MODEL_DIR / "sens_test.pkl")
    joblib.dump(y_pred, MODEL_DIR / "y_pred_baseline.joblib")
    
    print(f"\nSaved model + test data to '{MODEL_DIR}/' for the fairness analysis step.")
    return clf, X_test, y_test, y_pred, sens_test

if __name__ == "__main__":
    train_and_evaluate()