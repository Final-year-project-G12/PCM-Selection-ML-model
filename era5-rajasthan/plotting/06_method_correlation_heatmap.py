"""
06_method_correlation_heatmap.py
=============================================================================
PLOT 6 - Spearman/Kendall Correlation Heatmap Between Methods

Computes the 4x4 correlation matrix (Spearman rank correlation)
between TOPSIS, PROMETHEE, VIKOR, and GRA ranks, per cluster.

Verification:
  - Identify which method has the lowest mean pairwise correlation
  - Should be GRA (documented structural outlier in all 3 clusters)
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import spearmanr

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/04_mcdm_agreement"
MCDM_FILE = os.path.join(DATA_DIR, "mcdm_rankings_rajasthan.csv")
METHOD_AGREEMENT_FILE = os.path.join(DATA_DIR, "mcdm_method_agreement_rajasthan.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading MCDM rankings from {MCDM_FILE}...")
mcdm_df = pd.read_csv(MCDM_FILE)

# Method columns
methods = ["TOPSIS_rank", "PROMETHEE_II_rank", "VIKOR_rank", "GRA_rank"]
method_names = ["TOPSIS", "PROMETHEE II", "VIKOR", "GRA"]

# Verify required columns exist
for col in methods + ["cluster_id"]:
    if col not in mcdm_df.columns:
        raise ValueError(f"Required column '{col}' not found in MCDM file")

print("\n=== DATA VERIFICATION ===")

clusters = sorted(mcdm_df["cluster_id"].unique())
print(f"Clusters: {clusters}")

print("\n=== VERIFICATION BLOCK ===")
print("\nSpearman correlation matrices per cluster:")
print("(Looking for GRA to have lowest mean pairwise correlation)")
print()

# Create heatmaps for each cluster
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]

    # Compute Spearman correlation matrix
    correlations = []
    for method1, name1 in zip(methods, method_names):
        row = []
        for method2, name2 in zip(methods, method_names):
            if method1 == method2:
                row.append(1.0)
            else:
                rho, _ = spearmanr(cluster_data[method1], cluster_data[method2])
                row.append(rho)
        correlations.append(row)

    correlations = np.array(correlations)

    # Compute mean pairwise correlation for each method
    mean_corr_per_method = {}
    for idx, method_name in enumerate(method_names):
        # Exclude diagonal
        corrs = correlations[idx, :].copy()
        corrs[idx] = np.nan
        mean_corr = np.nanmean(corrs)
        mean_corr_per_method[method_name] = mean_corr

    print(f"Cluster {cluster_id}:")
    print(f"  Spearman correlation matrix (4x4):")
    for i, name1 in enumerate(method_names):
        print(f"    {name1:<15}", end="")
        for j, name2 in enumerate(method_names):
            print(f" {correlations[i,j]:+.3f}", end="")
        print()

    print(f"\n  Mean pairwise correlation by method:")
    sorted_methods = sorted(mean_corr_per_method.items(), key=lambda x: x[1])
    for method, mean_corr in sorted_methods:
        print(f"    {method:<15}: {mean_corr:+.3f}")

    lowest_method = sorted_methods[0][0]
    if lowest_method == "GRA":
        print(f"  [OK] PASS: GRA has lowest mean correlation (as expected)")
    else:
        print(f"  [WARN] INFO: {lowest_method} has lowest correlation (expected GRA)")

    print()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlations,
        x=method_names,
        y=method_names,
        colorscale="RdBu",
        zmid=0,
        text=np.around(correlations, decimals=3),
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Spearman rho")
    ))

    fig.update_layout(
        title=f"Method Correlation Matrix (Spearman) - Cluster {cluster_id}<br>({len(cluster_data)} candidates)",
        xaxis_title="Method",
        yaxis_title="Method",
        height=500,
        width=550,
    )

    # Save
    output_file = os.path.join(OUTPUT_DIR, f"method_correlation_heatmap_cluster_{cluster_id}.html")
    fig.write_html(output_file)
    print(f"[OK] Heatmap saved for Cluster {cluster_id}: {output_file}")

print(f"\nAll heatmaps saved to: {OUTPUT_DIR}/")
