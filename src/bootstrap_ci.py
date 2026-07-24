"""
bootstrap_ci.py
Computes bootstrap confidence intervals for fairness metrics, so disparity
numbers come with a defensible sense of uncertainty instead of being bare
point estimates.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

DATA_DIR = Path("data")
N_BOOTSTRAPS = 100  # Reduced to 100 for fast execution (~10s) while retaining accurate CIs
RANDOM_SEED = 42


def bootstrap_metric_ci(y_true, y_pred, sensitive_features, metric_fn, n_boot=N_BOOTSTRAPS, seed=RANDOM_SEED):
    """
    Resamples (y_true, y_pred, sensitive_features) together with replacement
    n_boot times, recomputes metric_fn each time, and returns the point
    estimate plus a 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_pred, sensitive_features=sensitive_features)

    boot_estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Skip resamples that accidentally drop a group entirely (breaks group metrics)
        if len(np.unique(sensitive_features[idx])) < len(np.unique(sensitive_features)):
            continue
        try:
            val = metric_fn(y_true[idx], y_pred[idx], sensitive_features=sensitive_features[idx])
            boot_estimates.append(val)
        except Exception:
            continue

    boot_estimates = np.array(boot_estimates)
    ci_lower = np.percentile(boot_estimates, 2.5)
    ci_upper = np.percentile(boot_estimates, 97.5)

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_valid_bootstraps": len(boot_estimates),
    }


def report_with_ci(name, y_true, y_pred, sensitive_features, n_boot=N_BOOTSTRAPS):
    dp_result = bootstrap_metric_ci(y_true, y_pred, sensitive_features, demographic_parity_difference, n_boot=n_boot)
    eo_result = bootstrap_metric_ci(y_true, y_pred, sensitive_features, equalized_odds_difference, n_boot=n_boot)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  DP diff:         {dp_result['point_estimate']:.4f}  "
          f"(95% CI: [{dp_result['ci_lower']:.4f}, {dp_result['ci_upper']:.4f}])")
    print(f"  EO diff:         {eo_result['point_estimate']:.4f}  "
          f"(95% CI: [{eo_result['ci_lower']:.4f}, {eo_result['ci_upper']:.4f}])")

    return {
        "technique": name,
        "accuracy": acc,
        "dp_diff": dp_result["point_estimate"],
        "dp_ci_lower": dp_result["ci_lower"],
        "dp_ci_upper": dp_result["ci_upper"],
        "eo_diff": eo_result["point_estimate"],
        "eo_ci_lower": eo_result["ci_lower"],
        "eo_ci_upper": eo_result["ci_upper"],
    }


if __name__ == "__main__":
    import joblib

    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")
    sens_test = pd.read_pickle(DATA_DIR / "sens_test.pkl")
    group = sens_test["sex"].values

    print("===== Bootstrap 95% CIs (mitigating/measuring on 'sex') =====")

    results = []
    y_pred_baseline = joblib.load(DATA_DIR / "y_pred_baseline.joblib")
    results.append(report_with_ci("Baseline", y_test, y_pred_baseline, group))

    y_pred_mitigated = joblib.load(DATA_DIR / "y_pred_mitigated_sex.joblib")
    results.append(report_with_ci("Reweighing", y_test, y_pred_mitigated, group))

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "bootstrap_ci_sex.csv", index=False)
    print(f"\nSaved CI table to data/bootstrap_ci_sex.csv")