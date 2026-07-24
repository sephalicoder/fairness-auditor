"""
mitigation.py
Applies Kamiran & Calders-style reweighing to the training data based on a
chosen sensitive attribute, retrains a classifier, and compares fairness
metrics against the baseline.
"""
print("Script started: Initializing bias mitigation framework...")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from src.data_loader import load_adult_data
from src.train_baseline import build_preprocessing_pipeline

DATA_DIR = Path("data")


def compute_reweighing_weights(y_train: pd.Series, group_train: pd.Series) -> np.ndarray:
    """
    Kamiran & Calders reweighing:
    weight(group, label) = [P(group) * P(label)] / P(group, label)
    """
    df = pd.DataFrame({"y": y_train.values, "group": group_train.values})
    n = len(df)

    weights = np.ones(n)
    p_group = df["group"].value_counts(normalize=True)
    p_label = df["y"].value_counts(normalize=True)

    for group_val in df["group"].unique():
        for label_val in df["y"].unique():
            mask = (df["group"] == group_val) & (df["y"] == label_val)
            p_joint_observed = mask.sum() / n
            if p_joint_observed == 0:
                continue
            p_expected = p_group[group_val] * p_label[label_val]
            weights[mask.values] = p_expected / p_joint_observed

    return weights


def train_with_reweighing(sensitive_attr: str = "sex"):
    X_raw, y, sensitive = load_adult_data()
    X = X_raw.drop(columns=["sex", "race"])

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    sample_weights = compute_reweighing_weights(y_train, sens_train[sensitive_attr])

    preprocessor = build_preprocessing_pipeline(X)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print(f"\nTraining reweighted classifier (mitigating on '{sensitive_attr}')...")
    clf.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    y_pred_mitigated = clf.predict(X_test)

    # Save for the dashboard / comparison step
    joblib.dump(clf, DATA_DIR / f"mitigated_model_{sensitive_attr}.joblib")
    joblib.dump(y_pred_mitigated, DATA_DIR / f"y_pred_mitigated_{sensitive_attr}.joblib")

    return y_test, y_pred_mitigated, sens_test


def compare_baseline_vs_mitigated(sensitive_attr: str = "sex"):
    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")
    sens_test = pd.read_pickle(DATA_DIR / "sens_test.pkl")
    y_pred_baseline = joblib.load(DATA_DIR / "y_pred_baseline.joblib")

    y_test_new, y_pred_mitigated, sens_test_new = train_with_reweighing(sensitive_attr)

    group = sens_test[sensitive_attr]

    results = {}
    for name, preds in [("baseline", y_pred_baseline), ("mitigated", y_pred_mitigated)]:
        acc = accuracy_score(y_test, preds)
        dp = demographic_parity_difference(y_test, preds, sensitive_features=group)
        eo = equalized_odds_difference(y_test, preds, sensitive_features=group)
        results[name] = {"accuracy": acc, "dp_diff": dp, "eo_diff": eo}

    print(f"\n===== Baseline vs Mitigated (mitigated on '{sensitive_attr}') =====")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)

    dp_improvement = (1 - results["mitigated"]["dp_diff"] / results["baseline"]["dp_diff"]) * 100
    acc_cost = (results["baseline"]["accuracy"] - results["mitigated"]["accuracy"]) * 100

    print(f"\nDemographic parity improved by {dp_improvement:.1f}%")
    print(f"Accuracy cost: {acc_cost:.2f} percentage points")

    comparison_df.to_csv(DATA_DIR / f"comparison_{sensitive_attr}.csv")
    return comparison_df


# Direct Execution
compare_baseline_vs_mitigated(sensitive_attr="sex")