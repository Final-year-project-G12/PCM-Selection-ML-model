"""
STEP 5: SPATIAL JOINS (NEAREST NEIGHBOR)
=========================================
Extract climate data for target locations (e.g., Coimbatore, Jaisalmer, Chennai SWH prototypes).

For each target point, find nearest ERA5 grid cell using xarray.sel(method='nearest').
Optionally use IDW for smoother interpolation.

Target locations (customizable):
  - Coimbatore_SWH: 11.0°N, 76.96°E (prototype 1)
  - Jaisalmer_SWH: 26.92°N, 70.9°E (reference/validation)
  - Chennai_SWH: 13.08°N, 80.27°E (alternative location)

Output: Regional CSVs (~8,784 rows × all features each)

python .\step_5_spatial.py

Input: step_4_cleaned.csv
Output: climate_{location}.csv for each target
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSING_DIR = os.path.join(BASE_DIR, 'processing')

INPUT_CSV = os.path.join(PROCESSING_DIR, 'step_4_cleaned.csv')
LOG_FILE = os.path.join(PROCESSING_DIR, 'step_5_spatial_report.txt')

# Target locations for spatial extraction
TARGET_LOCATIONS = [
    {'name': 'Coimbatore_SWH', 'lat': 11.0, 'lon': 76.96},
    {'name': 'Jaisalmer_SWH', 'lat': 26.92, 'lon': 70.9},
    {'name': 'Chennai_SWH', 'lat': 13.08, 'lon': 80.27},
]

def find_nearest_grid_point(lat_target, lon_target, lat_grid, lon_grid):
    """
    Find nearest grid point to target location.
    Returns: (nearest_lat, nearest_lon, distance_km)
    """
    # Euclidean distance in lat-lon (approximate for small regions)
    distances = np.sqrt((lat_grid - lat_target)**2 + (lon_grid - lon_target)**2)
    nearest_idx = np.argmin(distances)
    nearest_lat = lat_grid[nearest_idx]
    nearest_lon = lon_grid[nearest_idx]
    distance = distances[nearest_idx] * 111  # Rough conversion to km
    return nearest_lat, nearest_lon, distance

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 5: SPATIAL JOINS (NEAREST NEIGHBOR)")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load data
    print(f"\n--- LOADING DATA ---")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} cols")
    
    # Get unique grid points
    lat_grid = np.array(df['lat'].unique())
    lon_grid = np.array(df['lon'].unique())
    print(f"Grid points available: {len(lat_grid)} unique lats × {len(lon_grid)} unique lons")
    
    print(f"\n--- EXTRACTING DATA FOR TARGET LOCATIONS ---")
    
    log_lines = []
    
    for target in TARGET_LOCATIONS:
        name = target['name']
        lat = target['lat']
        lon = target['lon']
        
        print(f"\n{name} (target: {lat:.2f}°N, {lon:.2f}°E)")
        
        # Find nearest grid point
        # Get all available lat, lon pairs
        unique_points = df[['lat', 'lon']].drop_duplicates().values
        unique_lats = unique_points[:, 0]
        unique_lons = unique_points[:, 1]
        
        distances = np.sqrt((unique_lats - lat)**2 + (unique_lons - lon)**2)
        nearest_idx = np.argmin(distances)
        nearest_lat = unique_lats[nearest_idx]
        nearest_lon = unique_lons[nearest_idx]
        distance_km = distances[nearest_idx] * 111
        
        print(f"  → Nearest grid: {nearest_lat:.2f}°N, {nearest_lon:.2f}°E")
        print(f"  → Distance: {distance_km:.1f} km")
        
        # Extract data for this location
        mask = (df['lat'] == nearest_lat) & (df['lon'] == nearest_lon)
        df_location = df[mask].copy()
        
        if len(df_location) == 0:
            print(f"  ⚠️  No data found for this location!")
            continue
        
        print(f"  → Rows: {len(df_location):,}")
        print(f"  → Columns: {len(df_location.columns)}")
        
        # Verify temporal coverage
        date_range = f"{df_location['timestamp'].min()} to {df_location['timestamp'].max()}"
        print(f"  → Temporal range: {date_range}")
        
        # Save to CSV (in data folder for permanent storage)
        output_path = os.path.join(DATA_DIR, f'climate_{name}.csv')
        df_location.to_csv(output_path, index=False)
        print(f"  [OK] Saved: climate_{name}.csv ({len(df_location):,} rows)")
        
        log_lines.append(f"{name}: {len(df_location):,} rows, nearest grid {nearest_lat:.2f}°N {nearest_lon:.2f}°E")
    
    print(f"\n--- SPATIAL EXTRACTION SUMMARY ---")
    print(f"Target locations processed: {len(TARGET_LOCATIONS)}")
    for target in TARGET_LOCATIONS:
        output_path = os.path.join(DATA_DIR, f"climate_{target['name']}.csv")
        if os.path.exists(output_path):
            df_check = pd.read_csv(output_path)
            print(f"[OK] {target['name']}: {len(df_check):,} rows")
        else:
            print(f"[WARNING] {target['name']}: file not created")
    
    # Log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STEP 5: SPATIAL JOINS (NEAREST NEIGHBOR)\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nTarget locations processed:\n")
        for line in log_lines:
            f.write(f"  {line}\n")
        f.write(f"\nOutput files: climate_{{LOCATION}}.csv\n")
        f.write(f"STATUS: ✅ COMPLETE\n")
    
    print(f"\n✓ STEP 5 COMPLETE: Regional CSVs extracted")
    print(f"  Use these for prototype environment setup")
    print(f"  Next: Step 6 (normalization & scaling)")

if __name__ == '__main__':
    main()
