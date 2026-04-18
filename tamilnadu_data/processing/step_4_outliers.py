"""
STEP 4: OUTLIER DETECTION & ANOMALY FLAGGING
==============================================
Apply physical bounds validation and Isolation Forest anomaly detection.
Flag outliers as NaN, then interpolate.

Bounds applied to: GHI, DNI, DHI, LW_down, T_amb, T_dew, RH, Wind, Precip, Cloud, P_atm, CSI, SZA, ETR, GHI_clearsky

Optional: Isolation Forest on (GHI, DNI, DHI) triplet to catch physically impossible combinations.

python .\step_4_outliers.py


Input: step_3_aligned.csv
Output: step_4_cleaned.csv (with outliers flagged & interpolated)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSING_DIR = os.path.join(BASE_DIR, 'processing')

INPUT_CSV = os.path.join(PROCESSING_DIR, 'step_3_aligned.csv')
OUTPUT_CSV = os.path.join(PROCESSING_DIR, 'step_4_cleaned.csv')
LOG_FILE = os.path.join(PROCESSING_DIR, 'step_4_outlier_report.txt')

# Physical bounds (from Process.txt)
PHYSICAL_BOUNDS = {
    'GHI': (0, 1400),
    'DNI': (0, 1200),
    'DHI': (0, 800),
    'LW_down': (100, 600),
    'T_amb': (-10, 55),
    'T_dew': (-20, 40),
    'RH_percent': (0, 100),
    'Wind_speed_ms': (0, 30),
    'Wind_dir_deg': (0, 360),
    'Precip_mm': (0, 200),
    'cloud_cover': (0, 1),
    'P_atm_hPa': (800, 1100),
    'CSI': (0, 1.5),
    'SZA': (0, 180),
    'ETR': (0, 1500),
    'GHI_clearsky': (0, 1400),
}

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 4: OUTLIER DETECTION & ANOMALY FLAGGING")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load data
    print(f"\n--- LOADING DATA ---")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"[OK] Loaded: {len(df):,} rows x {len(df.columns)} cols")
    
    print(f"\n--- A. PHYSICAL BOUNDS VALIDATION ---")
    
    outlier_log = []
    total_outliers = 0
    
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            # Skip columns that don't exist (like RRTDHS if not computed)
            continue
        
        mask = (df[col] < lo) | (df[col] > hi)
        outlier_count = mask.sum()
        
        if outlier_count > 0:
            df.loc[mask, col] = np.nan
            total_outliers += outlier_count
            pct = 100 * outlier_count / len(df)
            msg = f"  {col}: {outlier_count:,} outliers ({pct:.3f}%) → NaN"
            print(msg)
            outlier_log.append(msg)
    
    if total_outliers == 0:
        print(f"[OK] No outliers detected -- all values within physical bounds")
    else:
        print(f"[WARNING] Total outliers flagged: {total_outliers:,}")
    
    print(f"\n--- B. ISOLATION FOREST ANOMALY DETECTION (OPTIONAL) ---")
    
    # Check for physically impossible solar triplets (GHI-DNI-DHI)
    solar_cols = ['GHI', 'DNI', 'DHI']
    if all(col in df.columns for col in solar_cols):
        print(f"Applying Isolation Forest on (GHI, DNI, DHI)...")
        
        # Fill NaN for anomaly detection
        solar_data = df[solar_cols].fillna(0)
        
        iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        anomaly_labels = iso.fit_predict(solar_data)
        
        # Add anomaly flag
        df['solar_anomaly_flag'] = (anomaly_labels == -1).astype(int)
        
        anomaly_count = (anomaly_labels == -1).sum()
        pct = 100 * anomaly_count / len(df)
        print(f"[OK] Detected {anomaly_count:,} anomalies ({pct:.2f}%)")
        print(f"   Kept as feature (not removed) for XGBoost to learn")
    else:
        # Flag not available, create dummy
        df['solar_anomaly_flag'] = 0
        print(f"[WARNING] Solar triplet not available, skipping Isolation Forest")
    
    print(f"\n--- C. INTERPOLATION OF FLAGGED NaNs ---")
    
    nan_before = df.isnull().sum().sum()
    print(f"NaNs before interpolation: {nan_before:,}")
    
    # Per-location linear interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def interpolate_location(group):
        for col in numeric_cols:
            if col not in ['lat', 'lon', 'year', 'hour', 'month', 'DOY']:
                group[col] = group[col].interpolate(
                    method='linear', limit=6, limit_direction='both'
                )
        group = group.ffill()
        group = group.bfill()
        return group
    
    df = df.groupby(['lat', 'lon'], group_keys=False).apply(interpolate_location)
    
    nan_after = df.isnull().sum().sum()
    print(f"NaNs after interpolation: {nan_after:,}")
    
    if nan_before > nan_after:
        print(f"[OK] {nan_before - nan_after:,} NaNs filled via interpolation")
    
    if nan_after == 0:
        print(f"[OK] Zero NaNs -- dataset complete!")
    else:
        print(f"[WARNING] {nan_after:,} NaNs remain (unfillable gaps)")
    
    print(f"\n--- D. DATA QUALITY SUMMARY ---")
    
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Completeness: {100 * (1 - nan_after / (len(df) * len(df.columns))):.2f}%")
    
    # Range check after cleaning
    print(f"\nRange check (post-cleaning):")
    for col in ['GHI', 'T_amb', 'RH_percent', 'Wind_speed_ms']:
        if col in df.columns:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    
    print(f"\n--- SAVING OUTPUT ---")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved: {OUTPUT_CSV}")
    print(f"   {len(df):,} rows × {len(df.columns)} cols")
    
    # Log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STEP 4: OUTLIER DETECTION & ANOMALY FLAGGING\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nOutput: {OUTPUT_CSV}\n")
        f.write(f"\nOUTLIERS DETECTED:\n")
        for line in outlier_log:
            f.write(line + "\n")
        f.write(f"\nTotal outliers flagged: {total_outliers:,}\n")
        f.write(f"Isolation Forest anomalies: {anomaly_count:,} (flagged, not removed)\n")
        f.write(f"\nNaNs before: {nan_before:,}\n")
        f.write(f"NaNs after: {nan_after:,}\n")
        f.write(f"STATUS: ✅ COMPLETE\n")
    
    print(f"\n✓ STEP 4 COMPLETE: Outliers detected & cleaned")
    print(f"  Next: Step 5 (spatial joins)")

if __name__ == '__main__':
    main()
