"""
dashboard.py
Interactive Streamlit dashboard for the AI Fairness Auditor.
Reads precomputed artifacts from data/ (models, predictions, comparison
tables, bootstrap CIs) — does NOT retrain anything live, so the app loads
instantly.
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path("data")

st.set_page_config(
    page_title="AI Fairness Auditor",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system — CSS injection (IBM Plex fonts, audit-report palette)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #1C1E21;
    --paper: #FAFAF8;
    --line: #E4E2DD;
    --signal-fail: #B3441E;
    --signal-pass: #3D6B4F;
    --accent-data: #2B4570;
}

.stApp {
    background-color: var(--paper) !important;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-data);
    margin-bottom: 0.25rem;
}

.metric-card {
    background-color: var(--paper);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
}

.metric-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--ink) !important;
}

.metric-label {
    font-size: 0.85rem;
    color: #5A5D63;
    margin-bottom: 0.35rem;
}

.pass-tag {
    color: var(--signal-pass);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}

.fail-tag {
    color: var(--signal-fail);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}

hr {
    border-color: var(--line);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">FAIRNESS AUDIT · ADULT INCOME MODEL</div>', unsafe_allow_html=True)
st.title("Demographic Parity & Equalized Odds Audit")
st.write(
    "This dashboard compares a baseline income-prediction classifier against "
    "three bias mitigation techniques, measured across sex and race, with "
    "bootstrap confidence intervals on every disparity figure."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Audit settings")
sensitive_attr = st.sidebar.selectbox("Sensitive attribute", ["sex", "race"])
technique = st.sidebar.selectbox(
    "Mitigation technique to inspect",
    ["Reweighing", "ExponentiatedGradient", "ThresholdOptimizer"],
)

# ---------------------------------------------------------------------------
# Load comparison data
# ---------------------------------------------------------------------------
@st.cache_data
def load_comparison(attr):
    path = DATA_DIR / f"mitigation_comparison_{attr}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


comparison_df = load_comparison(sensitive_attr)

if comparison_df is None:
    st.warning(
        f"No comparison data found for '{sensitive_attr}'. "
        f"Run `python -m src.compare_mitigations` with sensitive_attr='{sensitive_attr}' first."
    )
    st.stop()

baseline_row = comparison_df[comparison_df["technique"].str.contains("Baseline")].iloc[0]
technique_row = comparison_df[comparison_df["technique"].str.contains(technique)].iloc[0]

# ---------------------------------------------------------------------------
# Metric cards — baseline vs selected technique
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    delta_acc = (technique_row["accuracy"] - baseline_row["accuracy"]) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Accuracy (baseline → {technique})</div>
        <div class="metric-number">{baseline_row['accuracy']:.3f} → {technique_row['accuracy']:.3f}</div>
        <div class="{'fail-tag' if delta_acc < 0 else 'pass-tag'}">{delta_acc:+.2f} pp</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    dp_improvement = (1 - technique_row["dp_diff"] / baseline_row["dp_diff"]) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Demographic Parity Diff</div>
        <div class="metric-number">{baseline_row['dp_diff']:.3f} → {technique_row['dp_diff']:.3f}</div>
        <div class="pass-tag">{dp_improvement:+.1f}% change</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    eo_change = (1 - technique_row["eo_diff"] / baseline_row["eo_diff"]) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Equalized Odds Diff</div>
        <div class="metric-number">{baseline_row['eo_diff']:.3f} → {technique_row['eo_diff']:.3f}</div>
        <div class="{'pass-tag' if eo_change > 0 else 'fail-tag'}">{eo_change:+.1f}% change</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Pareto frontier — interactive plotly version
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">SIGNATURE CHART</div>', unsafe_allow_html=True)
st.subheader("Accuracy vs. Fairness Tradeoff")

metric_choice = st.radio(
    "Fairness metric",
    ["Demographic Parity Difference", "Equalized Odds Difference"],
    horizontal=True,
)
metric_col = "dp_diff" if metric_choice == "Demographic Parity Difference" else "eo_diff"

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=comparison_df[metric_col],
    y=comparison_df["accuracy"],
    mode="markers+text",
    text=comparison_df["technique"],
    textposition="top center",
    marker=dict(size=16, color="#2B4570", line=dict(width=1, color="#1C1E21")),
))
fig.update_layout(
    xaxis_title=f"{metric_choice} (Lower = Fairer) →",
    yaxis_title="Accuracy",
    xaxis=dict(autorange="reversed", gridcolor="#E4E2DD"),
    yaxis=dict(gridcolor="#E4E2DD"),
    plot_bgcolor="#FAFAF8",
    paper_bgcolor="#FAFAF8",
    font=dict(family="IBM Plex Sans", color="#1C1E21"),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Full comparison table
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">FULL COMPARISON TABLE</div>', unsafe_allow_html=True)
st.dataframe(
    comparison_df.style.format({"accuracy": "{:.4f}", "dp_diff": "{:.4f}", "eo_diff": "{:.4f}"}),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Bootstrap CI section (if available)
# ---------------------------------------------------------------------------
ci_path = DATA_DIR / f"bootstrap_ci_{sensitive_attr}.csv"
if ci_path.exists():
    st.divider()
    st.markdown('<div class="eyebrow">STATISTICAL CONFIDENCE</div>', unsafe_allow_html=True)
    st.subheader("Bootstrap 95% Confidence Intervals")
    ci_df = pd.read_csv(ci_path)
    st.dataframe(ci_df, use_container_width=True)
    st.caption(
        "Confidence intervals computed via 1,000 bootstrap resamples. "
        "Non-overlapping intervals between baseline and mitigated indicate "
        "a statistically robust difference, not sampling noise."
    )