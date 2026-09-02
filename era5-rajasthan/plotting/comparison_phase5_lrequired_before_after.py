"""
comparison_phase5_lrequired_before_after.py
=============================================================================
COMPARISON PLOT - Phase 5: L_required Before/After Correction (2026-08-31)

This plot visualizes the single most important before/after in the pipeline:
  - Pre-correction (SHARE_PCM=1.0): L_required ~608-641 kJ/kg, 0 survivors
  - Post-correction (SHARE_PCM=0.5): L_required ~285-344 kJ/kg, 39 survivors

Creates a bar chart comparing survivor counts per cluster, pre vs. post.

GRACEFUL HANDLING: If only post-correction data exists on disk, prints
a message and skips this comparison rather than erroring.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/comparison_plots/phase5_lrequired_before_after"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to find both pre and post correction files
SURVIVORS_POST = os.path.join(DATA_DIR, "feasibility_survivors_rajasthan_kappa_calibrated.csv")
SURVIVORS_PRE_CANDIDATES = [
    os.path.join(DATA_DIR, "feasibility_survivors_rajasthan_precorrection.csv"),
    os.path.join(DATA_DIR, "feasibility_survivors_rajasthan_preL_required_correction.csv"),
    # Could also check backups or archive folders
]

print("Phase 5: L_required Before/After Comparison")
print("-" * 60)

# Check for post-correction file
if not os.path.exists(SURVIVORS_POST):
    print(f"[WARN] Post-correction file not found: {SURVIVORS_POST}")
    print("  Cannot generate this comparison.")
    exit(1)

print(f"[OK] Post-correction file found: {os.path.basename(SURVIVORS_POST)}")

# Try to find pre-correction file
survivors_pre_file = None
for candidate_file in SURVIVORS_PRE_CANDIDATES:
    if os.path.exists(candidate_file):
        survivors_pre_file = candidate_file
        break

if survivors_pre_file is None:
    print(f"\n[WARN] Pre-correction file not found in any expected location")
    print(f"  Searched for:")
    for candidate_file in SURVIVORS_PRE_CANDIDATES:
        print(f"    - {candidate_file}")
    print(f"\n  This comparison requires a backup/archive of the pre-correction output.")
    print(f"  Without it, the before/after visualization cannot be generated.")
    print(f"\n  NOTE: This is expected if Phase 5 was only run ONCE with the corrected")
    print(f"        L_required methodology. The old output was not retained.")
    print(f"\n[OK] Gracefully skipping this comparison.")
    exit(0)

print(f"[OK] Pre-correction file found: {os.path.basename(survivors_pre_file)}")

# Load both files
print("\nLoading data...")
survivors_pre = pd.read_csv(survivors_pre_file)
survivors_post = pd.read_csv(SURVIVORS_POST)

print(f"  Pre-correction: {len(survivors_pre)} rows")
print(f"  Post-correction: {len(survivors_post)} rows")

# Read L_required from cluster profiles (current = post-correction)
cluster_profile_file = os.path.join(DATA_DIR, "cluster_profiles_rajasthan.csv")
if os.path.exists(cluster_profile_file):
    profiles = pd.read_csv(cluster_profile_file)
    l_required_post = profiles.set_index("cluster_id")["L_required_kJ_per_kg"].to_dict()
else:
    l_required_post = {}

print("\n=== VERIFICATION BLOCK ===")
print("\nL_required values (post-correction, from cluster_profiles):")
for cid, lreq in sorted(l_required_post.items()):
    print(f"  Cluster {cid}: {lreq:.1f} kJ/kg")

print("\nSurvivor counts per cluster:")
print(f"{'Cluster':<10} {'Pre-correction':<20} {'Post-correction':<20} {'Change':<10}")
print("-" * 60)

clusters = sorted(set(survivors_pre["cluster_id"].unique()) | set(survivors_post["cluster_id"].unique()))

pre_counts = []
post_counts = []

for cluster_id in clusters:
    pre_count = len(survivors_pre[survivors_pre["cluster_id"] == cluster_id])
    post_count = len(survivors_post[survivors_post["cluster_id"] == cluster_id])
    change = post_count - pre_count

    pre_counts.append(pre_count)
    post_counts.append(post_count)

    change_str = f"+{change}" if change >= 0 else str(change)
    print(f"{cluster_id:<10} {pre_count:<20} {post_count:<20} {change_str:<10}")

print("-" * 60)
print(f"{'TOTAL':<10} {sum(pre_counts):<20} {sum(post_counts):<20}")

print("\n[OK] Audit baselines for comparison:")
print(f"  Pre-correction should show: 0 survivors (L_required ~608-641 kJ/kg too high)")
print(f"  Post-correction should show: 39 total survivors (L_required ~285-344 kJ/kg)")

total_pre = sum(pre_counts)
total_post = sum(post_counts)

if total_pre == 0 and total_post == 39:
    print(f"\n[OK] PASS: Data matches both audit baselines exactly")
else:
    print(f"\n[WARN] INFO: Pre={total_pre}, Post={total_post} (baselines: 0 → 39)")

print("\n" + "=" * 60)

# Create bar chart
fig = go.Figure()

x_labels = [f"Cluster {c}" for c in clusters]
x = list(range(len(clusters)))

fig.add_trace(go.Bar(
    x=x,
    y=pre_counts,
    name="Pre-correction (L_req~600 kJ/kg)",
    marker_color="lightcoral",
    text=pre_counts,
    textposition="auto",
))

fig.add_trace(go.Bar(
    x=x,
    y=post_counts,
    name="Post-correction (L_req~300 kJ/kg)",
    marker_color="darkgreen",
    text=post_counts,
    textposition="auto",
))

fig.update_layout(
    title="Phase 5 Impact: L_required Methodology Correction (2026-08-31)<br>" +
          f"Pre: {total_pre} survivors → Post: {total_post} survivors",
    xaxis=dict(
        tickvals=x,
        ticktext=x_labels
    ),
    yaxis_title="Number of Feasible PCM Candidates",
    barmode="group",
    height=500,
    hovermode="x unified",
)

# Save
output_file = os.path.join(OUTPUT_DIR, "lrequired_before_after_rajasthan.html")
fig.write_html(output_file)
print(f"\n[OK] Comparison plot saved to: {output_file}")
