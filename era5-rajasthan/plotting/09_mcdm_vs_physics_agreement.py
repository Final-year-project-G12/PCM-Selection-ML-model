"""
09_mcdm_vs_physics_agreement.py
=============================================================================
PLOT 9 - MCDM Consensus Rank vs. Simulated Performance (Physics Validation)

This is Phase 7's headline result visualized: does a higher Borda/Copeland
MCDM rank actually deliver better simulated annual solar fraction?

Scatter plot per cluster with trend line and Spearman rho computation.

Verification:
  - Compute Spearman rho per cluster
  - Compare against audit-documented values (pre-correction):
    * Cluster 0: -0.385 (downward trend)
    * Cluster 1: +0.125 (weak upward trend, best of three)
    * Cluster 2: -0.097 (flat/weak downward)
  - If phases have been re-run post-correction, compare against
    spearman_rho_by_cluster_rajasthan.csv instead
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/06_physics_validation"
MCDM_FILE = os.path.join(DATA_DIR, "mcdm_rankings_rajasthan.csv")
PHYSICS_FILE = os.path.join(DATA_DIR, "physics_validation_rajasthan.csv")
RHO_FILE = os.path.join(DATA_DIR, "spearman_rho_by_cluster_rajasthan.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading MCDM rankings from {MCDM_FILE}...")
mcdm_df = pd.read_csv(MCDM_FILE)

print(f"Loading physics validation from {PHYSICS_FILE}...")
physics_df = pd.read_csv(PHYSICS_FILE)

print(f"Loading pre-computed rho values from {RHO_FILE}...")
rho_df = pd.read_csv(RHO_FILE)

print("\n=== DATA VERIFICATION ===")

# Verify required columns
for col in ["cluster_id", "pcm_id", "borda_score"]:
    if col not in mcdm_df.columns:
        raise ValueError(f"MCDM file missing '{col}'")

for col in ["cluster_id", "pcm_id", "annual_solar_fraction"]:
    if col not in physics_df.columns:
        raise ValueError(f"Physics file missing '{col}'")

print(f"MCDM file: {len(mcdm_df)} rows")
print(f"Physics file: {len(physics_df)} rows")

# Join on cluster_id and pcm_id
print("\nJoining MCDM rankings with physics validation...")
joined = physics_df.merge(
    mcdm_df[["cluster_id", "pcm_id", "borda_score"]],
    on=["cluster_id", "pcm_id"],
    how="inner",
    suffixes=("", "_from_mcdm")
)

print(f"Joined dataset: {len(joined)} rows")

clusters = sorted(joined["cluster_id"].unique())
print(f"Clusters in joined data: {clusters}")

print("\n=== VERIFICATION BLOCK ===")

# Compute Spearman rho per cluster and compare to reference
print("\nSpearman rho (MCDM Borda rank vs. Simulated annual solar fraction):")
print("(Higher MCDM rank [lower score] should correlate with higher solar fraction)")
print()

audit_rho_values = {
    0: -0.385,
    1: +0.125,
    2: -0.097,
}

for cluster_id in clusters:
    cluster_data = joined[joined["cluster_id"] == cluster_id]

    # Compute borda rank from borda_score (lower score = better rank)
    borda_rank = cluster_data["borda_score"].rank()

    # Compute Spearman rho
    rho, pval = spearmanr(borda_rank, cluster_data["annual_solar_fraction"])

    print(f"Cluster {cluster_id}:")
    print(f"  rho = {rho:+.3f} (p = {pval:.3f}), n = {len(cluster_data)}")

    # Compare to audit values
    if cluster_id in audit_rho_values:
        audit_val = audit_rho_values[cluster_id]
        print(f"  Audit value (pre-correction): rho = {audit_val:+.3f}")
        if abs(rho - audit_val) < 0.01:
            print(f"  [OK] PASS: Matches audit value closely")
        else:
            print(f"  [INFO] INFO: Differs from audit (may indicate re-run post-correction)")

    # Check if rho value is in the current rho_df
    rho_row = rho_df[rho_df["cluster_id"] == cluster_id]
    if not rho_row.empty:
        rho_borda = rho_row["spearman_rho_vs_borda"].values[0]
        if not pd.isna(rho_borda):
            print(f"  File value (spearman_rho_vs_borda): rho = {rho_borda:+.3f}")

    print()

print("Interpretation:")
print("  Cluster 0: negative rho means higher MCDM rank delivers WORSE performance")
print("             (weak disagreement between MCDM and physics)")
print("  Cluster 1: positive rho (weak) means higher MCDM rank delivers BETTER performance")
print("             (this is the best agreement of the three)")
print("  Cluster 2: near-zero rho means essentially no correlation")
print()

print("=" * 50)

# Create subplots (one per cluster)
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"Cluster {c}" for c in clusters],
    specs=[[{"secondary_y": False} for _ in clusters]]
)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for col_idx, cluster_id in enumerate(clusters, 1):
    cluster_data = joined[joined["cluster_id"] == cluster_id].copy()

    # Compute borda rank from borda_score (lower score = better rank)
    borda_rank = cluster_data["borda_score"].rank()
    cluster_data["borda_rank"] = borda_rank

    # Sort for trend line
    cluster_data = cluster_data.sort_values("borda_rank")

    # Compute Spearman rho for title
    rho, pval = spearmanr(borda_rank, cluster_data["annual_solar_fraction"])

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=cluster_data["borda_rank"],
            y=cluster_data["annual_solar_fraction"],
            mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(size=8, color=colors[col_idx - 1], opacity=0.7),
            text=cluster_data["pcm_id"],
            customdata=np.column_stack((
                cluster_data["pcm_id"],
                cluster_data["borda_score"].values,
                cluster_data["annual_solar_fraction"].values
            )),
            hovertemplate="%{customdata[0]}<br>Borda rank: %{x:.0f}<br>Solar frac: %{y:.3f}<extra></extra>"
        ),
        row=1, col=col_idx
    )

    # Add trend line
    if len(cluster_data) > 1:
        z = np.polyfit(cluster_data["borda_rank"], cluster_data["annual_solar_fraction"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(cluster_data["borda_rank"].min(), cluster_data["borda_rank"].max(), 100)
        y_trend = p(x_trend)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name=f"Trend rho={rho:.3f}",
                line=dict(color=colors[col_idx - 1], dash="dash", width=2),
                hoverinfo="skip"
            ),
            row=1, col=col_idx
        )

    # Update axes
    fig.update_xaxes(title_text="Borda Rank", row=1, col=col_idx)
    fig.update_yaxes(title_text="Annual Solar Fraction", row=1, col=col_idx)

fig.update_layout(
    title="MCDM Consensus Rank vs. Simulated Performance (Physics Validation)<br>" +
          "Does higher MCDM ranking predict better thermal performance?",
    height=500,
    width=1200,
    hovermode="closest",
    showlegend=False
)

# Save
output_file = os.path.join(OUTPUT_DIR, "mcdm_vs_physics_agreement_rajasthan.html")
fig.write_html(output_file)
print(f"\n[OK] Agreement plot saved to: {output_file}")
