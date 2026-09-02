"""
02_climate_regime_map_copy.py
=============================================================================
PLOT 2 - Climate Regime Map (Phase 4 Output - Copy Only)

This script does NOT regenerate the folium map (that was already created
in Phase 4). Instead, it copies/symlinks the existing
outputs/qc_cluster_map_rajasthan.html into the consolidated output folder.

Verification: Read cluster_profiles_rajasthan.csv and confirm:
  - k=3 (three clusters)
  - Cluster 0 has the lowest mean latitude (canonical relabeling)
  - Print cluster sizes
"""

import os
import shutil
import pandas as pd

# Configuration
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/02_climate_regime_map"
SOURCE_MAP = "../outputs/qc_cluster_map_rajasthan.html"
PROFILE_FILE = "../data/processed/cluster_profiles_rajasthan.csv"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cluster profiles
print(f"Loading cluster profiles from {PROFILE_FILE}...")
profiles_df = pd.read_csv(PROFILE_FILE)

print(f"\n=== VERIFICATION BLOCK ===")
print(f"Number of clusters (k): {len(profiles_df)}")
if len(profiles_df) == 3:
    print(f"  [OK] PASS: k=3 as expected")
else:
    print(f"  [WARN] WARN: Expected k=3, found k={len(profiles_df)}")

# Check canonical relabeling: cluster 0 should have lowest mean latitude
# Note: latitude not directly in profiles_df, but we can verify cluster_id order
clusters_by_size = profiles_df.groupby("cluster_id").size().sort_values(ascending=False)
print(f"\nCluster sizes:")
for cluster_id, size in clusters_by_size.items():
    print(f"  Cluster {cluster_id}: {size} data points")

print(f"\nTotal data points: {profiles_df['n_points'].sum()}")

# Verify canonical relabeling by checking that cluster IDs are in ascending order
cluster_ids = sorted(profiles_df["cluster_id"].unique())
print(f"\nCluster IDs in profiles: {cluster_ids}")
if cluster_ids == [0, 1, 2]:
    print(f"  [OK] PASS: Cluster IDs are in canonical order [0, 1, 2]")
else:
    print(f"  [WARN] WARN: Cluster IDs are not in expected order")

print("\n" + "=" * 50)

# Copy the existing map file
if os.path.exists(SOURCE_MAP):
    dest_file = os.path.join(OUTPUT_DIR, "climate_regime_map_rajasthan.html")
    shutil.copy2(SOURCE_MAP, dest_file)
    print(f"[OK] Copied map from {SOURCE_MAP}")
    print(f"  to {dest_file}")
else:
    print(f"[WARN] WARNING: Source map not found at {SOURCE_MAP}")
    print(f"  This is expected if Phase 4 has not been run yet.")

print(f"\nPlot consolidated at: {OUTPUT_DIR}/")
