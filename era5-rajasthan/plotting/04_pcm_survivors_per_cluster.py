"""
04_pcm_survivors_per_cluster.py
=============================================================================
PLOT 4 - Number of Feasible PCM Candidates per Climate Regime

Grouped bar chart showing:
  - Primary run (kappa=0.7 fixed) survivor count
  - Kappa-calibrated survivor count
  Per cluster

Verification:
  - Print totals and compare against audit-documented numbers
  - If fingerprint matches known run: should be 39 total (post-correction)
  - Annotate each cluster's calibrated kappa value on its bar
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Add parent directory to path to import provenance_lib
sys.path.insert(0, ".")
try:
    from provenance_lib import file_fingerprint, fingerprint_id
except ImportError:
    print("[WARN] WARNING: provenance_lib not found")
    file_fingerprint = None

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/03_feasibility"
CLUSTER_PROFILE_FILE = os.path.join(DATA_DIR, "cluster_profiles_rajasthan.csv")
SURVIVORS_PRIMARY = os.path.join(DATA_DIR, "feasibility_survivors_rajasthan.csv")
SURVIVORS_CALIBRATED = os.path.join(DATA_DIR, "feasibility_survivors_rajasthan_kappa_calibrated.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading cluster profiles from {CLUSTER_PROFILE_FILE}...")
profiles_df = pd.read_csv(CLUSTER_PROFILE_FILE)

print(f"Loading primary survivors from {SURVIVORS_PRIMARY}...")
survivors_primary = pd.read_csv(SURVIVORS_PRIMARY)

print(f"Loading calibrated survivors from {SURVIVORS_CALIBRATED}...")
survivors_cal = pd.read_csv(SURVIVORS_CALIBRATED)

print("\n=== DATA LOADING ===")
print(f"Primary survivors total: {len(survivors_primary)}")
print(f"Calibrated survivors total: {len(survivors_cal)}")

# Check fingerprint staleness
if file_fingerprint and fingerprint_id:
    current_fp = file_fingerprint(CLUSTER_PROFILE_FILE)
    current_fp_id = fingerprint_id(current_fp)

    for fname, df in [("primary", survivors_primary), ("calibrated", survivors_cal)]:
        stamped_fps = df["upstream_cluster_profile_fingerprint"].unique()
        if len(stamped_fps) == 1:
            stamped_fp_id = stamped_fps[0]
            if stamped_fp_id == current_fp_id:
                print(f"[OK] {fname} survivors fingerprint matches current cluster_profiles")
            else:
                print(f"[WARN] {fname} survivors STALE (fingerprint mismatch)")

print("\n=== VERIFICATION BLOCK ===")

# Count survivors per cluster for both runs
clusters = sorted(profiles_df["cluster_id"].unique())
primary_counts = []
calibrated_counts = []
calibrated_kappas = []

print("\nSurvivor counts by cluster:")
print(f"{'Cluster':<10} {'Primary (κ=0.7)':<20} {'Calibrated':<20} {'κ_calibrated':<15}")
print("-" * 65)

for cluster_id in clusters:
    primary_count = len(survivors_primary[survivors_primary["cluster_id"] == cluster_id])
    calibrated_count = len(survivors_cal[survivors_cal["cluster_id"] == cluster_id])

    # Get calibrated kappa value from this cluster's survivors
    kappas_in_cluster = survivors_cal[survivors_cal["cluster_id"] == cluster_id]["calibrated_kappa"].unique()
    if len(kappas_in_cluster) > 0:
        kappa_mean = kappas_in_cluster.mean()
    else:
        kappa_mean = 0.0

    primary_counts.append(primary_count)
    calibrated_counts.append(calibrated_count)
    calibrated_kappas.append(kappa_mean)

    print(f"{cluster_id:<10} {primary_count:<20} {calibrated_count:<20} {kappa_mean:.4f}")

total_primary = sum(primary_counts)
total_calibrated = sum(calibrated_counts)

print("-" * 65)
print(f"{'TOTAL':<10} {total_primary:<20} {total_calibrated:<20}")

# Verification against known baselines
print("\nComparison to audit-documented baselines:")
print(f"  Post-correction (2026-08-31): 39 total calibrated survivors")
print(f"  Pre-correction (stale): 20 total calibrated survivors")
print(f"\nFound: {total_calibrated} calibrated survivors")

if total_calibrated == 39:
    print(f"  [OK] PASS: Matches post-correction baseline")
elif total_calibrated == 20:
    print(f"  [WARN] WARN: Matches pre-correction (STALE) baseline")
else:
    print(f"  [INFO] INFO: Different from both known baselines (new run?)")

print("\n" + "=" * 50)

# Create grouped bar chart
fig = go.Figure()

x_labels = [f"Cluster {c}" for c in clusters]
x = list(range(len(clusters)))
width = 0.35

# Primary run bars
fig.add_trace(go.Bar(
    x=x,
    y=primary_counts,
    name="Primary (κ=0.7 fixed)",
    marker_color="lightblue",
    text=primary_counts,
    textposition="auto",
))

# Calibrated run bars with kappa annotation
calibrated_text = [
    f"{count}<br>κ={kappa:.3f}"
    for count, kappa in zip(calibrated_counts, calibrated_kappas)
]

fig.add_trace(go.Bar(
    x=x,
    y=calibrated_counts,
    name="Calibrated (κ_optimized)",
    marker_color="darkblue",
    text=calibrated_text,
    textposition="outside",
))

fig.update_layout(
    title=f"PCM Feasibility Survivors per Cluster<br>Primary vs. Calibrated (Total: {total_primary} vs. {total_calibrated})",
    xaxis=dict(
        tickvals=x,
        ticktext=x_labels
    ),
    yaxis_title="Number of Surviving Candidates",
    barmode="group",
    height=500,
    hovermode="x unified",
)

# Save as PNG
output_png = os.path.join(OUTPUT_DIR, "pcm_survivors_per_cluster.png")
try:
    fig.write_image(output_png, width=800, height=500)
    print(f"\n[OK] PNG plot saved to: {output_png}")
except Exception as e:
    print(f"\n[WARN] Could not save PNG: {e}")

# Also save as HTML
output_html = os.path.join(OUTPUT_DIR, "pcm_survivors_per_cluster.html")
fig.write_html(output_html)
print(f"[OK] HTML plot saved to: {output_html}")
