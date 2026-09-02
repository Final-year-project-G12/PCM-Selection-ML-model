"""
03_pcm_feasibility_scatter.py
=============================================================================
PLOT 3 - Melting Point vs. Latent Heat Scatter (Feasible Candidates Highlighted)

This plot shows:
  - All PCM candidates as a scatter (Tm vs. latent heat)
  - Survivors colored by cluster_id
  - Non-survivors in light grey
  - Vertical band for 42-70degC target Tm range
  - Horizontal line for L_required (read from cluster_profiles)

Verification:
  - Print survivor count per cluster
  - Compare against pcm_database_status in the CSV
  - Warn if 0 survivors found at nominal kappa (stale pre-correction file)
  - Check fingerprint against cluster_profiles_rajasthan.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Add parent directory to path to import provenance_lib
sys.path.insert(0, ".")
try:
    from provenance_lib import file_fingerprint, fingerprint_id
except ImportError:
    print("[WARN] WARNING: provenance_lib not found, fingerprint check skipped")
    file_fingerprint = None
    fingerprint_id = None

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/03_feasibility"
CLUSTER_PROFILE_FILE = os.path.join(DATA_DIR, "cluster_profiles_rajasthan.csv")
SURVIVORS_FILE = os.path.join(DATA_DIR, "feasibility_survivors_rajasthan_kappa_calibrated.csv")
PCM_DB_FILE = os.path.join("../../PCM_data/data", "PCM_Properties_cleaned_mice_pmm_detailed.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading cluster profiles from {CLUSTER_PROFILE_FILE}...")
profiles_df = pd.read_csv(CLUSTER_PROFILE_FILE)

print(f"Loading feasibility survivors from {SURVIVORS_FILE}...")
survivors_df = pd.read_csv(SURVIVORS_FILE)

print(f"Loading PCM database from {PCM_DB_FILE}...")
pcm_db_df = pd.read_csv(PCM_DB_FILE)

print("\n=== DATA VERIFICATION ===")

# Check fingerprints if provenance_lib is available
if file_fingerprint and fingerprint_id:
    current_fp = file_fingerprint(CLUSTER_PROFILE_FILE)
    current_fp_id = fingerprint_id(current_fp)

    stamped_fps = survivors_df["upstream_cluster_profile_fingerprint"].unique()
    if len(stamped_fps) == 1:
        stamped_fp_id = stamped_fps[0]
        if stamped_fp_id == current_fp_id:
            print(f"[OK] Fingerprint matches: survivors file is current")
        else:
            print(f"[WARN] WARNING: Fingerprint mismatch!")
            print(f"  Survivors built from: {stamped_fp_id}")
            print(f"  Current cluster_profiles: {current_fp_id}")
            print(f"  DATA IS STALE - will add watermark to plot")
    else:
        print(f"[WARN] WARNING: Multiple fingerprints in survivors file (data corruption)")

# Read L_required values from cluster profiles
l_required = profiles_df.set_index("cluster_id")["L_required_kJ_per_kg"].to_dict()
l_required_mean = np.mean(list(l_required.values()))

print(f"\nL_required by cluster:")
for cid, lreq in l_required.items():
    print(f"  Cluster {cid}: {lreq:.1f} kJ/kg")
print(f"  Mean L_required: {l_required_mean:.1f} kJ/kg")

# PCM database status info (if available)
if "pcm_database_status" in survivors_df.columns:
    db_status = survivors_df["pcm_database_status"].iloc[0]
    print(f"\nPCM Database Status: {db_status}")

print("\n=== VERIFICATION BLOCK ===")

# Create all_candidates dataset (survivors + non-survivors)
# Rename PCM database columns to match survivors file naming
pcm_db_df_renamed = pcm_db_df.copy()
pcm_db_df_renamed["pcm_id"] = pcm_db_df_renamed["product"]
pcm_db_df_renamed["Tm_C"] = pcm_db_df_renamed["Tm_melting"]
pcm_db_df_renamed["latent_heat_kJ_kg"] = pcm_db_df_renamed["latent_heat_melting"]

survivors_set = set(survivors_df["pcm_id"].values)
all_candidates = pcm_db_df_renamed.copy()
all_candidates["is_survivor"] = all_candidates["pcm_id"].isin(survivors_set)

# For survivors, add cluster assignment
survivor_clusters = survivors_df.groupby("pcm_id")["cluster_id"].first().to_dict()
all_candidates["cluster_id"] = all_candidates["pcm_id"].map(survivor_clusters)

# Count survivors per cluster
print("Survivor count per cluster:")
for cluster_id in sorted(profiles_df["cluster_id"].unique()):
    count = len(survivors_df[survivors_df["cluster_id"] == cluster_id])
    print(f"  Cluster {cluster_id}: {count} survivors")

total_survivors = len(survivors_df)
print(f"Total survivors: {total_survivors}")

if total_survivors == 0:
    print(f"[WARN] WARN: Zero survivors found! This indicates STALE pre-correction file.")
    print(f"         (Post-correction should have 39 total across all clusters)")
else:
    if total_survivors == 39:
        print(f"[OK] PASS: Total survivor count matches post-correction baseline (39)")
    else:
        print(f"[INFO] INFO: Total survivors = {total_survivors} (post-correction baseline = 39)")

print("\n" + "=" * 50)

# Prepare plotting data
colors = {
    0: "#1f77b4",  # blue
    1: "#ff7f0e",  # orange
    2: "#2ca02c",  # green
}

# Create scatter plot
fig = go.Figure()

# Add non-survivors first (so they appear behind)
non_survivors = all_candidates[~all_candidates["is_survivor"]]
fig.add_trace(go.Scatter(
    x=non_survivors["Tm_C"],
    y=non_survivors["latent_heat_kJ_kg"],
    mode="markers",
    name="Non-survivors",
    marker=dict(
        size=6,
        color="lightgray",
        opacity=0.5
    ),
    text=non_survivors["pcm_id"],
    hovertemplate="%{text}<br>Tm: %{x:.1f}degC<br>L: %{y:.1f} kJ/kg<extra></extra>"
))

# Add survivors, grouped by cluster
for cluster_id in sorted(profiles_df["cluster_id"].unique()):
    cluster_survivors = all_candidates[
        (all_candidates["is_survivor"]) &
        (all_candidates["cluster_id"] == cluster_id)
    ]
    fig.add_trace(go.Scatter(
        x=cluster_survivors["Tm_C"],
        y=cluster_survivors["latent_heat_kJ_kg"],
        mode="markers",
        name=f"Cluster {cluster_id} ({len(cluster_survivors)} survivors)",
        marker=dict(
            size=8,
            color=colors.get(cluster_id, "gray"),
            opacity=0.7,
            line=dict(width=1, color="white")
        ),
        text=cluster_survivors["pcm_id"],
        hovertemplate="%{text}<br>Tm: %{x:.1f}degC<br>L: %{y:.1f} kJ/kg<extra></extra>"
    ))

# Add vertical band for 42-70degC target range
fig.add_vrect(
    x0=42, x1=70,
    fillcolor="lightblue",
    opacity=0.2,
    layer="below",
    line_width=0,
    name="42-70degC target"
)

# Add horizontal line for mean L_required
fig.add_hline(
    y=l_required_mean,
    line_dash="dash",
    line_color="red",
    opacity=0.5,
    name=f"L_required (mean={l_required_mean:.1f} kJ/kg)"
)

fig.update_layout(
    title=f"PCM Feasibility Screening: Tm vs. Latent Heat<br>(L_required = {l_required_mean:.1f} kJ/kg mean across clusters)",
    xaxis_title="Melting Point Tm (degC)",
    yaxis_title="Latent Heat (kJ/kg)",
    hovermode="closest",
    height=600,
    width=900,
    showlegend=True
)

# Save as PNG
output_png = os.path.join(OUTPUT_DIR, "pcm_feasibility_scatter.png")
try:
    fig.write_image(output_png, width=900, height=600)
    print(f"\n[OK] PNG plot saved to: {output_png}")
except Exception as e:
    print(f"\n[WARN] Could not save PNG: {e}")
    print(f"  (kaleido package may not be installed)")

# Also save as HTML for interactive viewing
output_html = os.path.join(OUTPUT_DIR, "pcm_feasibility_scatter.html")
fig.write_html(output_html)
print(f"[OK] HTML plot saved to: {output_html}")
