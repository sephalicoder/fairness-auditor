"""
fairness_metrics.py
Loads the baseline model's saved test predictions and computes fairness metrics 
using Fairlearn, broken down by sex, race, and their intersection.
"""
print("Script started: Initializing fairness audit...")

import pandas as pd
import joblib
from pathlib import Path
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)
from sklearn.metrics import accuracy_score

DATA_DIR = Path("data")

def load_baseline_artifacts():
    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")
    sens_test = pd.read_pickle(DATA_DIR / "sens_test.pkl")
    y_pred = joblib.load(DATA_DIR / "y_pred_baseline.joblib")
    return y_test, y_pred, sens_test

def compute_fairness_report(y_test, y_pred, sensitive_features, group_col_name="sex"):
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features[group_col_name]
    )
    
    dp_diff = demographic_parity_difference(
        y_test, y_pred, sensitive_features=sensitive_features[group_col_name]
    )
    eo_diff = equalized_odds_difference(
        y_test, y_pred, sensitive_features=sensitive_features[group_col_name]
    )
    
    print(f"\n===== Fairness report by '{group_col_name}' =====")
    print("\nPer-group metrics:")
    print(mf.by_group)
    print(f"\nDemographic parity difference: {dp_diff:.4f}")
    print(f"Equalized odds difference: {eo_diff:.4f}")
    
    return mf, dp_diff, eo_diff

def compute_intersectional_report(y_test, y_pred, sensitive_features):
    intersect = sensitive_features["sex"].astype(str) + " & " + sensitive_features["race"].astype(str)
    
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=intersect
    )
    
    print("\n===== Intersectional report (sex & race) =====")
    print(mf.by_group)
    return mf

# Run the execution directly without the conditional block to force it to run
y_test, y_pred, sens_test = load_baseline_artifacts()
compute_fairness_report(y_test, y_pred, sens_test, group_col_name="sex")
compute_fairness_report(y_test, y_pred, sens_test, group_col_name="race")
compute_intersectional_report(y_test, y_pred, sens_test)