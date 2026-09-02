"""
08_rank_reversal_frequency.py
=============================================================================
PLOT 8 - Rank-Reversal Frequency Across Monte Carlo Draws (Violin/Bar)

Shows the per-candidate rank-reversal frequency (how often any two
candidates swap order across the 1000 Monte Carlo draws).

Verification:
  - Cluster 0 (Kendall's W=0.388, weakest cross-method agreement)
    should show HIGHER rank-reversal frequency
  - Clusters 1/2 (W=0.634-0.635) should show LOWER frequency
  - This is a specific, testable prediction from the project's numbers
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/05_montecarlo"
MCDM_FILE = os.path.join(DATA_DIR, "mcdm_rankings_rajasthan.csv")
CLUSTER_PROFILE_FILE = os.path.join(DATA_DIR, "cluster_profiles_rajasthan.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading MCDM rankings from {MCDM_FILE}...")
mcdm_df = pd.read_csv(MCDM_FILE)

print(f"Loading cluster profiles from {CLUSTER_PROFILE_FILE}...")
profiles_df = pd.read_csv(CLUSTER_PROFILE_FILE)

print("\n=== DATA VERIFICATION ===")

# Check for rank-reversal frequency column
if "mc_rank_reversal_freq_cluster" not in mcdm_df.columns:
    raise ValueError("'mc_rank_reversal_freq_cluster' column not found in MCDM file")

print(f"Loaded {len(mcdm_df)} candidates")

# Get Kendall's W values per cluster from MCDM data
kendalls_w_by_cluster = mcdm_df.groupby("cluster_id")["kendalls_w_cluster"].first().to_dict()
clusters = sorted(mcdm_df["cluster_id"].unique())

print(f"Clusters: {clusters}")
print(f"\nKendall's W values:")
for cid in clusters:
    if cid in kendalls_w_by_cluster:
        print(f"  Cluster {cid}: W = {kendalls_w_by_cluster[cid]:.4f}")

print("\n=== VERIFICATION BLOCK ===")

# Compute mean rank-reversal frequency per cluster
print("\nMean rank-reversal frequency per cluster:")
print("(Note: N_DRAWS = 1000 in this pipeline)")
print()

cluster_freq_data = {}
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]

    # All rows in a cluster should have the same mc_rank_reversal_freq_cluster value
    freq_values = cluster_data["mc_rank_reversal_freq_cluster"].unique()
    if len(freq_values) == 1:
        freq = freq_values[0]
    else:
        freq = cluster_data["mc_rank_reversal_freq_cluster"].mean()

    cluster_freq_data[cluster_id] = freq

    w = kendalls_w_by_cluster.get(cluster_id, np.nan)
    print(f"  Cluster {cluster_id}: freq = {freq:.3f} (Kendall's W = {w:.4f})")

# Check prediction: Cluster 0 should have higher freq than Clusters 1/2
if cluster_freq_data[0] > cluster_freq_data[1] and cluster_freq_data[0] > cluster_freq_data[2]:
    print(f"\n[OK] PASS: Cluster 0 has higher rank-reversal frequency than Clusters 1/2")
    print(f"         This matches the prediction based on Kendall's W values")
else:
    print(f"\n[WARN] WARN: Cluster 0 does NOT have higher rank-reversal frequency")
    print(f"         This contradicts the Kendall's W-based prediction")

print("\n" + "=" * 50)

# Create violin plot for each cluster
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"Cluster {c}" for c in clusters],
    specs=[[{"secondary_y": False} for _ in clusters]]
)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for col_idx, cluster_id in enumerate(clusters, 1):
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]

    # Get per-candidate rank-reversal frequencies
    # Note: They may vary per candidate within a cluster
    freqs = cluster_data["mc_rank_reversal_freq_cluster"].values

    # Create violin plot
    fig.add_trace(
        go.Violin(
            y=freqs,
            name=f"Cluster {cluster_id}",
            side="negative",
            points="all",
            pointpos=-0.1,
            jitter=0.05,
            marker=dict(size=6),
            meanline_visible=True,
            line_color=colors[col_idx - 1],
            fillcolor=colors[col_idx - 1],
            opacity=0.6,
            hovertemplate="Freq: %{y:.3f}<extra></extra>"
        ),
        row=1, col=col_idx
    )

    # Add mean line
    mean_freq = np.mean(freqs)
    fig.add_hline(y=mean_freq, line_dash="dash", line_color="red",
                  row=1, col=col_idx)

    fig.update_yaxes(
        title_text="Rank-Reversal Frequency",
        row=1, col=col_idx
    )

fig.update_layout(
    title="Rank-Reversal Frequency Across Monte Carlo Draws (N_DRAWS=1000)<br>" +
          "Per cluster and per candidate",
    height=500,
    width=1200,
    showlegend=False,
    hovermode="closest"
)

# Save
output_file = os.path.join(OUTPUT_DIR, "rank_reversal_frequency_rajasthan.html")
fig.write_html(output_file)
print(f"\n[OK] Rank-reversal plot saved to: {output_file}")

# Also create a simple bar chart showing cluster-level summary
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=[f"Cluster {c}" for c in clusters],
    y=[cluster_freq_data[c] for c in clusters],
    marker_color=colors,
    text=[f"{cluster_freq_data[c]:.3f}" for c in clusters],
    textposition="auto",
))

fig2.update_layout(
    title="Mean Rank-Reversal Frequency by Cluster",
    yaxis_title="Mean Frequency",
    height=400,
    showlegend=False,
)

output_file2 = os.path.join(OUTPUT_DIR, "rank_reversal_frequency_summary.html")
fig2.write_html(output_file2)
print(f"[OK] Summary bar chart saved to: {output_file2}")
