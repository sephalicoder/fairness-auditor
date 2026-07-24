# AI Fairness Auditor — Bias Detection & Mitigation for ML Classifiers

An end-to-end fairness audit of an income-prediction classifier: measuring demographic bias across sex and race using Fairlearn, applying three different mitigation techniques, and quantifying the accuracy-fairness tradeoff with bootstrap confidence intervals — all wrapped in an interactive Streamlit dashboard.

**Live findings, not a tutorial**: this project doesn't stop at "we improved fairness by X%." It shows that the *right* mitigation technique depends on which fairness definition you care about and which sensitive attribute you're auditing — and that these choices can genuinely conflict with each other.

## Key findings

**Mitigating on `sex`**: reweighing improved demographic parity by 52.7% (0.168 → 0.080, both 95% CIs non-overlapping — a statistically robust improvement) at a cost of 0.91 accuracy points. But equalized odds got *worse* (0.084 → 0.141) — a real, measured example of the fairness impossibility theorem: you generally cannot satisfy demographic parity and equalized odds simultaneously.

**Mitigating on `race`**: the story is different. `ExponentiatedGradient` improved *both* demographic parity (0.164 → 0.036) and equalized odds (0.240 → 0.082) simultaneously, at a steeper accuracy cost (0.854 → 0.822). Meanwhile `ThresholdOptimizer` looked attractive on accuracy (0.847, barely below baseline) but quietly made equalized odds substantially worse (0.240 → 0.362).

**Takeaway**: no single mitigation technique is "the fix." The best choice depends on (a) which fairness definition matters for your specific decision context, and (b) which sensitive attribute you're protecting against — and these can pull in different directions.

## Why demographic parity vs. equalized odds actually matters

These are genuinely different, sometimes-conflicting definitions of "fair":
- **Demographic parity**: each group should receive positive predictions (`>50K`) at the same rate.
- **Equalized odds**: each group should have the same true positive *and* false positive rate.

They optimize for different things. For a loan decision, you might care more about equalized odds (don't want to disproportionately deny qualified people in one group). For an ad-targeting decision, demographic parity might matter more (don't want to under-show opportunities to one group regardless of individual qualification). Picking the wrong definition for the context is a documented real-world failure mode, not a hypothetical.

## Pipeline

1. **Data**: Adult Income dataset (OpenML) — sex and race held out as sensitive attributes, never used as model features.
2. **Baseline model**: Logistic Regression, standard preprocessing pipeline (imputation, scaling, one-hot encoding).
3. **Fairness measurement**: Fairlearn `MetricFrame`, per-group and intersectional (sex × race) breakdowns.
4. **Mitigation** (3 techniques compared):
   - *Reweighing* (pre-processing) — Kamiran & Calders sample reweighting
   - *ExponentiatedGradient* (in-processing) — constrained optimization during training
   - *ThresholdOptimizer* (post-processing) — per-group decision threshold adjustment
5. **Statistical validation**: 1,000-resample bootstrap confidence intervals on every disparity metric.
6. **Dashboard**: interactive Streamlit app for exploring all of the above.

## Limitations

- Results are specific to Logistic Regression + the Adult Income dataset; they don't necessarily generalize to other models or domains.
- The impossibility theorem result here is empirical (observed across our specific experiments), not a formal proof — for the mathematical guarantee, see Kleinberg, Mullainathan & Raghavan (2016), "Inherent Trade-Offs in the Fair Determination of Risk Scores."
- Intersectional subgroups (e.g. specific race×sex combinations) have small sample sizes in places; treat those specific numbers as noisier than the top-line results.
- We optimized for one sensitive attribute at a time; mitigating for `sex` and `race` simultaneously is a harder, unsolved extension not covered here.

## Tech stack

| Tool | Purpose |
|---|---|
| Python, scikit-learn, pandas, numpy | Data + modeling |
| Fairlearn | Fairness metrics + mitigation techniques |
| Streamlit + Plotly | Interactive dashboard |
| pytest | Unit tests on custom reweighing logic |

100% free/open-source, runs entirely locally — no API keys, no cloud costs.

## Running it locally

```bash
pip install -r requirements.txt

# Generate all data (run once, in order)
python -m src.train_baseline
python -m src.fairness_metrics
python -m src.compare_mitigations
python -m src.bootstrap_ci
python -m src.pareto_chart

# Launch the dashboard
streamlit run app/dashboard.py
```

## Tests

```bash
pytest tests/test_metrics.py -v
```