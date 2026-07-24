"""
compare_mitigations.py
Trains three different fairness mitigation techniques (reweighting, ExponentiatedGradient, ThresholdOptimizer) 
and compares them against the baseline on accuracy vs. fairness metrics.
"""
print("Script started: Initializing comprehensive bias mitigation comparison...")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from src.data_loader import load_adult_data
from src.train_baseline import build_preprocessing_pipeline
from src.mitigation import compute_reweighing_weights

DATA_DIR = Path("data")

def get_split(sensitive_attr="sex"):
    X_raw, y, sensitive = load_adult_data()
    X = X_raw.drop(columns=["sex", "race"])
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, sens_train[sensitive_attr], sens_test[sensitive_attr]

def eval_metrics(name, y_test, y_pred, group_test, acc):
    dp = demographic_parity_difference(y_test, y_pred, sensitive_features=group_test)
    eo = equalized_odds_difference(y_test, y_pred, sensitive_features=group_test)
    print(f"{name:38s} | acc={acc:.4f} | dp_diff={dp:.4f} | eo_diff={eo:.4f}")
    return {"technique": name, "accuracy": acc, "dp_diff": dp, "eo_diff": eo}

def run_baseline(X_train, X_test, y_train, y_test, group_test):
    preprocessor = build_preprocessing_pipeline(X_train)
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, eval_metrics("Baseline (no mitigation)", y_test, y_pred, group_test, acc)

def run_reweighting(X_train, X_test, y_train, y_test, group_train, group_test):
    weights = compute_reweighing_weights(y_train, group_train)
    preprocessor = build_preprocessing_pipeline(X_train)
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    clf.fit(X_train, y_train, classifier__sample_weight=weights)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return eval_metrics("Reweighing (pre-processing)", y_test, y_pred, group_test, acc)

def run_exponentiated_gradient(X_train, X_test, y_train, y_test, group_train, group_test):
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Senior Engineering Fix: Force dense arrays to comply with Fairlearn's matrix solvers
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    
    base_estimator = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
    )
    mitigator.fit(X_train_transformed, y_train, sensitive_features=group_train)
    y_pred = mitigator.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    return eval_metrics("ExponentiatedGradient (in-processing)", y_test, y_pred, group_test, acc)

def run_threshold_optimizer(baseline_clf, X_train, X_test, y_train, y_test, group_train, group_test):
    postprocess_est = ThresholdOptimizer(
        estimator=baseline_clf,
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=True,
    )
    postprocess_est.fit(X_train, y_train, sensitive_features=group_train)
    y_pred = postprocess_est.predict(X_test, sensitive_features=group_test)
    acc = accuracy_score(y_test, y_pred)
    return eval_metrics("ThresholdOptimizer (post-processing)", y_test, y_pred, group_test, acc)

def run_all(sensitive_attr="sex"):
    X_train, X_test, y_train, y_test, group_train, group_test = get_split(sensitive_attr)
    print(f"\n===== Comparing mitigation techniques (mitigating on '{sensitive_attr}') =====\n")
    
    results = []
    
    # 1. Baseline
    baseline_clf, baseline_result = run_baseline(X_train, X_test, y_train, y_test, group_test)
    results.append(baseline_result)
    
    # 2. Reweighting
    results.append(run_reweighting(X_train, X_test, y_train, y_test, group_train, group_test))
    
    # 3. Exponentiated Gradient
    results.append(run_exponentiated_gradient(X_train, X_test, y_train, y_test, group_train, group_test))
    
    # 4. Threshold Optimizer
    results.append(run_threshold_optimizer(baseline_clf, X_train, X_test, y_train, y_test, group_train, group_test))
    
    df = pd.DataFrame(results)
    out_path = DATA_DIR / f"mitigation_comparison_{sensitive_attr}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved multi-mitigation table to {out_path}.")
    return df

if __name__ == "__main__":
    run_all(sensitive_attr="sex")