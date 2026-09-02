"""
05_bump_chart.py
=============================================================================
PLOT 5 - Bump Chart: Rank per Method + Consensus (MCDM Agreement)

Shows each candidate's rank under TOPSIS, PROMETHEE II, VIKOR, GRA,
and Borda-consensus as connected lines across 5 x-positions.

One chart per cluster (3 total).

Verification:
  - Compute Spearman rho between VIKOR and TOPSIS ranks per cluster
  - WARN if rho < -0.5 (signature of VIKOR sign-inversion bug)
  - Expected: positive or near-zero correlation, not strongly negative
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import spearmanr

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/04_mcdm_agreement"
MCDM_FILE = os.path.join(DATA_DIR, "mcdm_rankings_rajasthan.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading MCDM rankings from {MCDM_FILE}...")
mcdm_df = pd.read_csv(MCDM_FILE)

print("\n=== DATA VERIFICATION ===")

# Method columns
methods = ["TOPSIS_rank", "PROMETHEE_II_rank", "VIKOR_rank", "GRA_rank"]
consensus = "borda_score"

# Verify required columns exist
for col in methods + [consensus, "cluster_id", "pcm_id"]:
    if col not in mcdm_df.columns:
        raise ValueError(f"Required column '{col}' not found in MCDM file")

print(f"Loaded {len(mcdm_df)} candidates from {len(mcdm_df['cluster_id'].unique())} clusters")

# Get cluster list
clusters = sorted(mcdm_df["cluster_id"].unique())
print(f"Clusters: {clusters}")

print("\n=== VERIFICATION BLOCK ===")

# Check for VIKOR sign-inversion bug
print("\nChecking for VIKOR sign-inversion bug (Spearman rho between VIKOR and TOPSIS):")

for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]

    # Compute Borda rank from borda_score (lower score = better rank)
    cluster_data = cluster_data.copy()
    cluster_data["borda_rank"] = cluster_data["borda_score"].rank()

    # Compute Spearman rho between VIKOR and TOPSIS
    if len(cluster_data) > 2:
        rho, pval = spearmanr(cluster_data["VIKOR_rank"], cluster_data["TOPSIS_rank"])
    else:
        rho = np.nan
        pval = np.nan

    print(f"  Cluster {cluster_id}: rho(VIKOR, TOPSIS) = {rho:+.3f} (p={pval:.3f})")

    if not np.isnan(rho):
        if rho < -0.5:
            print(f"    [WARN] WARN: Strong negative correlation suggests VIKOR sign-inversion bug!")
        elif rho > 0.3:
            print(f"    [OK] PASS: Positive correlation, no sign-inversion detected")
        else:
            print(f"    [INFO] INFO: Weak/near-zero correlation (acceptable)")

print("\n" + "=" * 50)

# Create bump chart for each cluster
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id].copy()

    # Create ranks for each method
    for method in methods:
        # Rank is already in the data
        pass

    # Compute borda rank from borda_score
    cluster_data["borda_rank"] = cluster_data["borda_score"].rank()

    # Sort by borda rank for better visualization
    cluster_data = cluster_data.sort_values("borda_rank")

    # Create figure
    fig = go.Figure()

    # X-axis positions for methods
    x_pos = list(range(len(methods))) + [len(methods)]  # +1 for borda consensus
    x_labels = methods + ["Consensus (Borda)"]

    # Add lines for each candidate
    colors = {}
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    for idx, (_, row) in enumerate(cluster_data.iterrows()):
        pcm_name = row["pcm_id"]
        color = color_palette[idx % len(color_palette)]
        colors[pcm_name] = color

        # Gather ranks
        ranks = [row[method] for method in methods] + [row["borda_rank"]]

        fig.add_trace(go.Scatter(
            x=x_pos,
            y=ranks,
            mode="lines+markers",
            name=pcm_name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f"{pcm_name}<br>%{{x}}: rank %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"MCDM Method Rank Agreement - Cluster {cluster_id}<br>({len(cluster_data)} candidates)",
        xaxis=dict(
            tickvals=x_pos,
            ticktext=x_labels,
            tickangle=-45
        ),
        yaxis=dict(
            title="Rank (lower is better)",
            autorange="reversed",
        ),
        height=600,
        width=900,
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    # Save
    output_file = os.path.join(OUTPUT_DIR, f"bump_chart_cluster_{cluster_id}.html")
    fig.write_html(output_file)
    print(f"[OK] Bump chart saved for Cluster {cluster_id}: {output_file}")

print(f"\nAll bump charts saved to: {OUTPUT_DIR}/")
