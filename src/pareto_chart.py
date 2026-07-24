"""
pareto_chart.py
Plots the Pareto frontiers comparing fairness mitigation strategies on 
Predictive Accuracy vs. Disparity Gaps (both Demographic Parity and Equalized Odds).
"""
print("Script started: Initializing visualization engine...")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
COMP_CSV = DATA_DIR / "mitigation_comparison_sex.csv"

def plot_pareto_matplotlib(metric_col, metric_label, save_path):
    if not COMP_CSV.exists():
        print(f"Error: Comparison data not found at {COMP_CSV}. Please run `python -m src.compare_mitigations` first.")
        return

    # Load multi-mitigation results
    df = pd.read_csv(COMP_CSV)

    plt.figure(figsize=(10, 6), dpi=100)
    
    # Senior UI Style Constants
    colors = {
        "Baseline (no mitigation)": "#EF4444",         # Red
        "Reweighing (pre-processing)": "#3B82F6",      # Blue
        "ExponentiatedGradient (in-processing)": "#F59E0B", # Amber
        "ThresholdOptimizer (post-processing)": "#10B981"  # Emerald
    }

    # Plot each profile marker
    for idx, row in df.iterrows():
        name = row["technique"]
        color = colors.get(name, "#6B7280")
        
        plt.scatter(
            row[metric_col], 
            row["accuracy"], 
            color=color, 
            s=180, 
            label=name, 
            edgecolors="black", 
            zorder=3
        )
        
        # Add clean annotations next to points
        plt.annotate(
            f" {name.split(' (')[0]}", 
            (row[metric_col], row["accuracy"]),
            fontsize=10, 
            fontweight="bold",
            va="center",
            ha="left"
        )

    # Invert x-axis: Higher fairness (lower disparity gap) moves left-to-right
    plt.gca().invert_xaxis()
    
    # Structural styling dynamically shifting titles based on the target metric
    metric_title = "Demographic Parity" if metric_col == "dp_diff" else "Equalized Odds"
    plt.title(f"Pareto Frontier: Model Accuracy vs. {metric_title} Disparity Gap", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(metric_label, fontsize=11, labelpad=10)
    plt.ylabel("Out-of-Sample Accuracy", fontsize=11, labelpad=10)
    
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.tight_layout()
    
    # Save chart artifact for documentation / dashboard inclusion
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Chart saved successfully to {save_path}!")
    plt.close()  # Close plot window to allow the execution to continue cleanly

if __name__ == "__main__":
    print("\n===== Generating Fairness Pareto Charts =====")
    plot_pareto_matplotlib(
        metric_col="dp_diff",
        metric_label="Demographic Parity Disparity Gap (Lower = Fairer) →",
        save_path=DATA_DIR / "pareto_frontier_dp.png",
    )
    plot_pareto_matplotlib(
        metric_col="eo_diff",
        metric_label="Equalized Odds Disparity Gap (Lower = Fairer) →",
        save_path=DATA_DIR / "pareto_frontier_eo.png",
    )