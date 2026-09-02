"""
01_raw_vs_preprocessed.py
=============================================================================
PLOT 1 - Raw vs. Preprocessed Radiation (Phase 2 → Phase 2.5 Audit)

This script plots overlaid distributions of raw vs. cleaned climate data,
specifically focusing on GHI (which should remain nearly unchanged, being
deliberately excluded from Hampel filtering) vs. T_amb (which should show
visible tail-trimming after cleaning).

Verification: Compute KS statistics for GHI (should be small, <0.05)
and T_amb (should be larger, >0.1), and print both.

Output: outputs/objective1_plots_rajasthan/01_raw_vs_preprocessed/ghi_tamb_distributions.html
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/01_raw_vs_preprocessed"
RAW_FILE = os.path.join(DATA_DIR, "climate_rajasthan_points.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "climate_rajasthan_points_clean.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"Loading raw data from {RAW_FILE}...")
raw_df = pd.read_csv(RAW_FILE)
print(f"Loaded {len(raw_df)} raw records")

print(f"Loading cleaned data from {CLEAN_FILE}...")
clean_df = pd.read_csv(CLEAN_FILE)
print(f"Loaded {len(clean_df)} cleaned records")

# Verify columns exist
required_cols = ["era5_GHI", "era5_T_amb"]
for col in required_cols:
    if col not in raw_df.columns or col not in clean_df.columns:
        raise ValueError(f"Required column '{col}' not found in data files")

# Extract variables and handle missing values
ghi_raw = raw_df["era5_GHI"].dropna()
ghi_clean = clean_df["era5_GHI"].dropna()
tamb_raw = raw_df["era5_T_amb"].dropna()
tamb_clean = clean_df["era5_T_amb"].dropna()

print(f"\n=== VERIFICATION BLOCK ===")

# Compute KS statistics
ks_ghi, ks_ghi_pval = ks_2samp(ghi_raw, ghi_clean)
ks_tamb, ks_tamb_pval = ks_2samp(tamb_raw, tamb_clean)

print(f"KS statistic for GHI (raw vs clean): {ks_ghi:.4f} (p={ks_ghi_pval:.4e})")
print(f"  Expected: small (<0.05) because GHI is deliberately excluded from Hampel filtering")
if ks_ghi < 0.05:
    print(f"  [OK] PASS: GHI distributions remain nearly unchanged")
else:
    print(f"  [WARN] GHI KS statistic unusually large, possible over-filtering")

print(f"\nKS statistic for T_amb (raw vs clean): {ks_tamb:.4f} (p={ks_tamb_pval:.4e})")
print(f"  Expected: larger (>0.1) because T_amb IS Hampel-filtered")
if ks_tamb > 0.1:
    print(f"  [OK] PASS: T_amb shows visible tail-trimming after cleaning")
else:
    print(f"  [WARN] T_amb KS statistic smaller than expected, check Hampel filter application")

print("\n" + "=" * 50)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("GHI Distribution (raw vs. clean)",
                   "T_amb Distribution (raw vs. clean)"),
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

# Plot 1: GHI (should be nearly identical)
fig.add_trace(
    go.Histogram(
        x=ghi_raw,
        name="GHI Raw",
        nbinsx=50,
        opacity=0.6,
        marker_color="lightgray",
        legendgroup="ghi"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(
        x=ghi_clean,
        name="GHI Clean",
        nbinsx=50,
        opacity=0.6,
        marker_color="blue",
        legendgroup="ghi"
    ),
    row=1, col=1
)

# Plot 2: T_amb (should show visible tail-trimming)
fig.add_trace(
    go.Histogram(
        x=tamb_raw,
        name="T_amb Raw",
        nbinsx=50,
        opacity=0.6,
        marker_color="lightgray",
        legendgroup="tamb"
    ),
    row=1, col=2
)

fig.add_trace(
    go.Histogram(
        x=tamb_clean,
        name="T_amb Clean",
        nbinsx=50,
        opacity=0.6,
        marker_color="orange",
        legendgroup="tamb"
    ),
    row=1, col=2
)

# Update layout
fig.update_xaxes(title_text="GHI (W/m²)", row=1, col=1)
fig.update_xaxes(title_text="T_amb (degC)", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.update_layout(
    title="Phase 2 → 2.5 Data Cleaning Verification: GHI (excluded) vs. T_amb (Hampel-filtered)",
    height=500,
    hovermode="x unified",
    showlegend=True
)

# Save
output_file = os.path.join(OUTPUT_DIR, "ghi_tamb_distributions.html")
fig.write_html(output_file)
print(f"\nPlot saved to: {output_file}")
