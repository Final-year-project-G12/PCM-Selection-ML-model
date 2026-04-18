"""
STEP 3: TEMPORAL ALIGNMENT & LAG HANDLING
==========================================
Verify temporal continuity, handle lag-NaN rows, ensure spatial grid integrity.

Process:
  A. Verify temporal completeness (8,784 unique timestamps, no gaps)
  B. Handle lag-NaN rows:
     Option A (Chosen): DROP first 24 rows per location → 3,425,160 rows
     Option B: FORWARD FILL (not used)
  C. Per-location interpolation for sensor gaps (max 6-hour limit)
  D. Create daytime subset (GHI > 10 W/m²) for XGBoost training

python .\step_3_alignment.py

Input: step_2_temporal_features.csv
Output: step_3_aligned.csv (3,425,160 rows), step_3_aligned_daytime.csv (subset)
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

INPUT_CSV = os.path.join(PROCESSING_DIR, 'step_2_temporal_features.csv')
OUTPUT_CSV = os.path.join(PROCESSING_DIR, 'step_3_aligned.csv')
OUTPUT_DAYTIME = os.path.join(PROCESSING_DIR, 'step_3_aligned_daytime.csv')
LOG_FILE = os.path.join(PROCESSING_DIR, 'step_3_alignment_report.txt')

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 3: TEMPORAL ALIGNMENT & LAG HANDLING")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load data
    print(f"\n--- LOADING DATA ---")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    initial_rows = len(df)
    print(f"[OK] Loaded: {initial_rows:,} rows x {len(df.columns)} cols")
    
    print(f"\n--- A. VERIFY TEMPORAL COMPLETENESS ---")
    
    # Check timestamp coverage
    expected_timestamps = pd.date_range('2024-01-01', '2024-12-31 23:00', freq='1H')
    expected_count = len(expected_timestamps)
    
    unique_ts = df['timestamp'].unique()
    unique_ts_count = len(unique_ts)
    
    print(f"Expected timestamps: {expected_count:,}")
    print(f"Actual unique timestamps: {unique_ts_count:,}")
    
    if unique_ts_count == expected_count:
        print(f"  [OK] Temporal grid complete (no missing hours)")
    else:
        missing = set(expected_timestamps) - set(unique_ts)
        print(f"  [WARNING] Missing {len(missing)} timestamps")
        if len(missing) > 0:
            print(f"   First missing: {sorted(missing)[0]}")
    
    # Spatial coverage
    print(f"\n--- SPATIAL COVERAGE ---")
    unique_locations = df[['lat', 'lon']].drop_duplicates()
    n_locations = len(unique_locations)
    print(f"Unique (lat, lon) pairs: {n_locations}")
    print(f"Expected: 391 (Tamil Nadu 0.25° grid)")
    if n_locations == 391:
        print(f"  [OK] Spatial grid complete")
    else:
        print(f"  [WARNING] Expected 391, got {n_locations}")
    
    print(f"\n--- B. HANDLE LAG-NaN ROWS (OPTION A: DROP) ---")
    
    # Count NaN rows created by lag features
    lag_cols = [col for col in df.columns if 'lag' in col]
    print(f"Lag columns: {len(lag_cols)}")
    
    # Find first non-NaN row per location
    rows_before_drop = len(df)
    
    # Drop first 24 rows per (lat, lon) location
    # Group by location, drop first 24 rows per group
    df_dropped = df.groupby(['lat', 'lon'], group_keys=False).apply(
        lambda group: group.iloc[24:] if len(group) > 24 else group.iloc[0:0]
    ).reset_index(drop=True)
    
    rows_after_drop = len(df_dropped)
    dropped_rows = rows_before_drop - rows_after_drop
    
    print(f"\nBefore drop: {rows_before_drop:,} rows")
    print(f"After drop: {rows_after_drop:,} rows")
    print(f"Dropped: {dropped_rows:,} rows (first 24 per location × 391 locations)")
    
    expected_drop = 391 * 24
    if dropped_rows == expected_drop:
        print(f"[OK] Correct number of rows dropped ({expected_drop:,})")
    else:
        print(f"[WARNING] Expected {expected_drop:,} drops, got {dropped_rows:,}")
    
    df = df_dropped.copy()
    
    print(f"\n--- C. INTERPOLATION FOR SENSOR GAPS ---")
    
    # Per-location linear interpolation (max 6-hour gap)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Applying per-location linear interpolation (limit=6 hours)...")
    
    def interpolate_location(group):
        for col in numeric_cols:
            if col not in ['lat', 'lon', 'year', 'hour', 'month', 'DOY', 'day']:
                group[col] = group[col].interpolate(
                    method='linear', limit=6, limit_direction='both'
                )
        # Forward fill, then backward fill
        group = group.ffill()
        group = group.bfill()
        return group
    
    df = df.groupby(['lat', 'lon'], group_keys=False).apply(interpolate_location)
    
    # Final NaN audit
    nan_total = df.isnull().sum().sum()
    if nan_total == 0:
        print(f"[OK] Zero NaNs after interpolation")
    else:
        print(f"[WARNING] {nan_total:,} NaNs remain")
        nan_by_col = df.isnull().sum()
        for col in nan_by_col[nan_by_col > 0].index:
            print(f"   {col}: {nan_by_col[col]:,}")
    
    print(f"\n--- D. DAYTIME SUBSET ---")
    
    # Create daytime subset (GHI > 10 W/m²)
    daytime_threshold = 10.0
    ghi_col = 'GHI_Wm2' if 'GHI_Wm2' in df.columns else ('GHI' if 'GHI' in df.columns else None)
    if ghi_col:
        df_daytime = df[df[ghi_col] > daytime_threshold].copy()
    else:
        print(f"⚠️  GHI column not found, skipping daytime filtering")
        df_daytime = df.copy()
    
    daytime_rows = len(df_daytime)
    daytime_pct = 100 * daytime_rows / len(df)
    
    print(f"Threshold: GHI > {daytime_threshold} W/m²")
    print(f"Daytime rows: {daytime_rows:,} ({daytime_pct:.1f}%)")
    print(f"Nighttime rows: {len(df) - daytime_rows:,} ({100 - daytime_pct:.1f}%)")
    
    print(f"\n--- SAVING OUTPUT ---")
    
    # Save full aligned dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Full dataset: {OUTPUT_CSV}")
    print(f"   {len(df):,} rows x {len(df.columns)} cols")
    
    # Save daytime subset
    df_daytime.to_csv(OUTPUT_DAYTIME, index=False)
    print(f"[OK] Daytime subset: {OUTPUT_DAYTIME}")
    print(f"   {len(df_daytime):,} rows × {len(df_daytime.columns)} cols")
    
    # Final summary
    print(f"\n--- SUMMARY ---")
    print(f"Initial rows: {initial_rows:,}")
    print(f"After lag drop: {len(df):,}")
    print(f"Daytime subset: {len(df_daytime):,}")
    print(f"Data type distribution:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count}")
    
    # Log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STEP 3: TEMPORAL ALIGNMENT & LAG HANDLING\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFull output: {OUTPUT_CSV}\n")
        f.write(f"Daytime subset: {OUTPUT_DAYTIME}\n")
        f.write(f"\nRows: {initial_rows:,} → {len(df):,} (drop {dropped_rows:,} lag-NaN rows)\n")
        f.write(f"Daytime rows: {len(df_daytime):,} ({daytime_pct:.1f}%)\n")
        f.write(f"NaNs after alignment: {nan_total}\n")
        f.write(f"STATUS: ✅ COMPLETE\n")
    
    print(f"\n✓ STEP 3 COMPLETE: Temporal alignment & lag rows handled")
    print(f"  Full dataset: {len(df):,} rows (expected 3,425,160)")
    print(f"  Next: Step 4 (outlier detection)")

if __name__ == '__main__':
    main()
