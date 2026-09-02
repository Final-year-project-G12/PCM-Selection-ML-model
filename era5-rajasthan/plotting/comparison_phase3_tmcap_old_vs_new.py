"""
comparison_phase3_tmcap_old_vs_new.py
=============================================================================
COMPARISON PLOT - Phase 3: Tm_target_capped Methodology Revision (2026-08-11)

Visualizes why the 2026-08-11 methodology revision was needed:
  - Old basis (p05-day): single worst day's capacity (40.8-49.5degC, implausibly low)
  - New basis (worst-month): 30-day worst-month capacity (51.1-55.2degC, realistic)

Both columns are already present in climate_signature_rajasthan.csv per the
Phase 3 audit (retained for audit trail).

Shows scatter with y=x reference line to visualize the gap.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/comparison_plots/phase3_tmcap_old_vs_new"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load climate signature file
sig_file = os.path.join(DATA_DIR, "climate_signature_rajasthan.csv")

print("Phase 3: Tm_target_capped Methodology Revision (2026-08-11)")
print("-" * 60)

if not os.path.exists(sig_file):
    print(f"[WARN] Climate signature file not found: {sig_file}")
    print("  Cannot generate this comparison.")
    exit(1)

print(f"Loading climate signature from {sig_file}...")
sig_df = pd.read_csv(sig_file)

print(f"Loaded {len(sig_df)} climate signature records")

# Check for required columns
old_col = "Tm_target_capped_C_p05day"
new_col = "Tm_target_capped_C"

if old_col not in sig_df.columns:
    print(f"\n[WARN] Old basis column '{old_col}' not found")
    print(f"  Available columns: {list(sig_df.columns)}")
    print(f"  This comparison requires the old column to be retained (audit trail).")
    exit(1)

if new_col not in sig_df.columns:
    print(f"\n[WARN] New basis column '{new_col}' not found")
    print(f"  Available columns: {list(sig_df.columns)}")
    exit(1)

print(f"[OK] Both columns found: '{old_col}' and '{new_col}'")

# Extract data
old_vals = sig_df[old_col].dropna()
new_vals = sig_df[new_col].dropna()

print(f"\n=== VERIFICATION BLOCK ===")
print(f"\nOld basis (p05-day, single worst day):")
print(f"  Mean: {old_vals.mean():.2f}degC")
print(f"  Range: {old_vals.min():.2f}-{old_vals.max():.2f}degC")
print(f"  Audit baseline: 40.8-49.5degC (implausibly low)")

print(f"\nNew basis (worst-month, 30-day integrated):")
print(f"  Mean: {new_vals.mean():.2f}degC")
print(f"  Range: {new_vals.min():.2f}-{new_vals.max():.2f}degC")
print(f"  Audit baseline: 51.1-55.2degC (realistic)")

# Compute gap
gap = new_vals - old_vals
print(f"\nMethodology gap (new - old):")
print(f"  Mean: +{gap.mean():.2f}degC")
print(f"  Range: {gap.min():.2f}-{gap.max():.2f}degC")
print(f"  All positive: {(gap > 0).all()} (should be True)")

if gap.min() > 0:
    print(f"  [OK] PASS: New methodology always produces higher (more realistic) targets")
else:
    print(f"  [WARN] WARN: Some negative gaps detected (data anomaly?)")

print("\n" + "=" * 60)

# Create scatter plot with y=x reference line
fig = go.Figure()

# Add scatter
fig.add_trace(go.Scatter(
    x=old_vals,
    y=new_vals,
    mode="markers",
    name="Data points",
    marker=dict(size=8, color="blue", opacity=0.6),
    text=sig_df["point_id"],
    hovertemplate="Point: %{text}<br>Old: %{x:.2f}degC<br>New: %{y:.2f}degC<extra></extra>"
))

# Add y=x reference line
min_val = min(old_vals.min(), new_vals.min()) - 1
max_val = max(old_vals.max(), new_vals.max()) + 1
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    name="y=x (no change)",
    line=dict(color="red", dash="dash", width=2),
    hoverinfo="skip"
))

fig.update_layout(
    title="Phase 3 Methodology Revision: Tm_target_capped<br>" +
          "Old basis (p05-day) vs. New basis (worst-month), 2026-08-11",
    xaxis_title="Old Basis: Tm_target_capped_C_p05day (degC)",
    yaxis_title="New Basis: Tm_target_capped_C (degC)",
    hovermode="closest",
    height=600,
    width=700,
    showlegend=True,
)

# Equal aspect ratio
fig.update_yaxes(scaleanchor="x", scaleratio=1)

# Save
output_file = os.path.join(OUTPUT_DIR, "tmcap_methodology_comparison_rajasthan.html")
fig.write_html(output_file)
print(f"\n[OK] Comparison plot saved to: {output_file}")
