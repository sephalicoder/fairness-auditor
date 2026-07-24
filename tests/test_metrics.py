"""
test_metrics.py
Unit tests for custom fairness logic.
"""

import pandas as pd
import numpy as np
import pytest

from src.mitigation import compute_reweighing_weights


def test_weights_are_positive():
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    group_train = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    weights = compute_reweighing_weights(y_train, group_train)
    assert (weights > 0).all(), "All reweighing weights must be strictly positive"


def test_weights_length_matches_input():
    y_train = pd.Series([1, 0, 1, 0, 1])
    group_train = pd.Series(["A", "A", "B", "B", "B"])
    weights = compute_reweighing_weights(y_train, group_train)
    assert len(weights) == len(y_train)


def test_balanced_data_gives_weights_near_one():
    y_train = pd.Series([1, 0, 1, 0] * 10)
    group_train = pd.Series((["A"] * 4 + ["B"] * 4) * 5)
    weights = compute_reweighing_weights(y_train, group_train)
    np.testing.assert_allclose(weights, 1.0, atol=0.05)


def test_underrepresented_combo_gets_upweighted():
    y_train = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    group_train = pd.Series(["A"] * 7 + ["B"] * 7)
    y_train.iloc[6] = 1

    weights = compute_reweighing_weights(y_train, group_train)
    df = pd.DataFrame({"y": y_train.values, "group": group_train.values, "weight": weights})

    rare_combo_weight = df[(df["group"] == "A") & (df["y"] == 1)]["weight"].iloc[0]
    common_combo_weight = df[(df["group"] == "A") & (df["y"] == 0)]["weight"].iloc[0]

    assert rare_combo_weight > common_combo_weight

