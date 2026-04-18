"""
STEP 6: NORMALIZATION & FEATURE SCALING
========================================
Apply three-scaler strategy with data leakage prevention.

Fit scalers on TRAINING split only (Jan-Sep, before Oct 1):
  1. MinMaxScaler [0,1]: Physically bounded variables
  2. StandardScaler (Z-score): Unbounded, approximately normal
  3. RobustScaler (IQR): Skewed, heavy-tail distributions

Apply transforms to FULL dataset (train + test).
Save fitted scalers as PKL files for deployment.

python .\step_6_scaling.py

Input: step_4_cleaned.csv
Output: step_6_scaled.csv + 3 scaler PKL files
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSING_DIR = os.path.join(BASE_DIR, 'processing')

INPUT_CSV = os.path.join(PROCESSING_DIR, 'step_4_cleaned.csv')
OUTPUT_CSV = os.path.join(PROCESSING_DIR, 'step_6_scaled.csv')
LOG_FILE = os.path.join(PROCESSING_DIR, 'step_6_scaling_report.txt')

# Scaler PKL paths (save to data/ for deployment)
SCALER_MINMAX_PATH = os.path.join(DATA_DIR, 'scaler_minmax_tamilnadu.pkl')
SCALER_STANDARD_PATH = os.path.join(DATA_DIR, 'scaler_standard_tamilnadu.pkl')
SCALER_ROBUST_PATH = os.path.join(DATA_DIR, 'scaler_robust_tamilnadu.pkl')

# Scaler assignments
COLS_MINMAX = [
    'GHI', 'DNI', 'DHI', 'LW_down', 'SZA', 'CSI', 'cloud_cover',
    'RH_percent', 'ETR', 'GHI_clearsky', 'solar_azimuth'
]

COLS_STANDARD = [
    'T_amb', 'T_dew', 'T_depression', 'T_pcm_delta', 'P_atm_hPa',
    'RRTDHS', 'GHI_clearsky_diff', 'altitude_m',
    'GHI_roll3h_mean', 'GHI_roll6h_mean', 'GHI_roll24h_std', 'T_roll24h_mean',
    'GHI_lag1', 'GHI_lag2', 'GHI_lag3', 'GHI_lag6', 'GHI_lag12', 'GHI_lag24',
    'T_amb_lag1', 'T_amb_lag2', 'T_amb_lag3', 'T_amb_lag6', 'T_amb_lag12', 'T_amb_lag24'
]

COLS_ROBUST = ['Precip_mm', 'Wind_speed_ms', 'Wind_dir_deg']

COLS_NO_SCALE = [
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_DOY', 'cos_DOY',
    'hour', 'month', 'DOY', 'year', 'season_code',
    'season_Winter', 'season_Summer', 'season_Monsoon', 'season_NE',
    'high_solar_resource', 'solar_anomaly_flag',
    'lat', 'lon', 'timestamp'
]

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 6: NORMALIZATION & FEATURE SCALING")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load data
    print(f"\n--- LOADING DATA ---")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"[OK] Loaded: {len(df):,} rows x {len(df.columns)} cols")
    
    print(f"\n--- IDENTIFYING TRAIN/TEST SPLIT ---")
    
    # Split: training = before Oct 1, test = after Oct 1
    cutoff = pd.Timestamp('2024-10-01')
    train_mask = df['timestamp'] < cutoff
    test_mask = ~train_mask
    
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    
    print(f"Training split (Jan-Sep): {n_train:,} rows ({100*n_train/len(df):.1f}%)")
    print(f"Test split (Oct-Dec): {n_test:,} rows ({100*n_test/len(df):.1f}%)")
    
    print(f"\n--- VALIDATING SCALER COLUMNS ---")
    
    print(f"\n--- FITTING SCALERS ON TRAINING SPLIT ---")
    
    # Check which columns actually exist and can be scaled
    available_minmax = [c for c in COLS_MINMAX if c in df.columns]
    available_standard = [c for c in COLS_STANDARD if c in df.columns]
    available_robust = [c for c in COLS_ROBUST if c in df.columns]
    
    print(f"MinMax scaler (bounded): {len(available_minmax)} columns")
    if available_minmax:
        print(f"  {available_minmax[:5]}..." if len(available_minmax) > 5 else f"  {available_minmax}")
    print(f"Standard scaler (unbounded): {len(available_standard)} columns")
    if available_standard:
        print(f"  {available_standard[:5]}..." if len(available_standard) > 5 else f"  {available_standard}")
    print(f"Robust scaler (skewed): {len(available_robust)} columns")
    if available_robust:
        print(f"  {available_robust}")
    
    # MinMax scaler
    if available_minmax:
        mm = MinMaxScaler()
        df.loc[train_mask, available_minmax] = mm.fit_transform(df.loc[train_mask, available_minmax])
        df.loc[test_mask, available_minmax] = mm.transform(df.loc[test_mask, available_minmax])
        print(f"[OK] MinMax scaler fitted ({len(available_minmax)} columns)")
    else:
        print(f"[WARNING] No MinMax columns available, skipping")
        mm = None
    
    # Standard scaler
    if available_standard:
        ss = StandardScaler()
        df.loc[train_mask, available_standard] = ss.fit_transform(df.loc[train_mask, available_standard])
        df.loc[test_mask, available_standard] = ss.transform(df.loc[test_mask, available_standard])
        print(f"[OK] Standard scaler fitted ({len(available_standard)} columns)")
    else:
        print(f"[WARNING] No Standard columns available, skipping")
        ss = None
    
    # Robust scaler
    if available_robust:
        rs = RobustScaler()
        df.loc[train_mask, available_robust] = rs.fit_transform(df.loc[train_mask, available_robust])
        df.loc[test_mask, available_robust] = rs.transform(df.loc[test_mask, available_robust])
        print(f"[OK] Robust scaler fitted ({len(available_robust)} columns)")
    else:
        print(f"[WARNING] No Robust columns available, skipping")
        rs = None
    
    print(f"\n--- SCALER STATISTICS ---")
    
    # MinMax statistics
    if mm is not None:
        mm_stats_min = mm.data_min_
        mm_stats_max = mm.data_max_
        print(f"\nMinMax (feature_range=[0,1]):")
        for i, col in enumerate(available_minmax[:3]):
            print(f"  {col}: original range [{mm_stats_min[i]:.2f}, {mm_stats_max[i]:.2f}]")
    
    # Standard scaler statistics
    if ss is not None:
        ss_mean = ss.mean_
        ss_std = ss.scale_
        print(f"\nStandard (Z-score):")
        for i, col in enumerate(available_standard[:3]):
            print(f"  {col}: μ={ss_mean[i]:.4f}, σ={ss_std[i]:.4f}")
    
    # Robust scaler
    if rs is not None:
        rs_center = rs.center_
        rs_scale = rs.scale_
        print(f"\nRobust (IQR-based):")
        for i, col in enumerate(available_robust):
            print(f"  {col}: Q2={rs_center[i]:.2f}, IQR={rs_scale[i]:.2f}")
    
    print(f"\n--- SAVING SCALERS ---")
    
    if mm is not None:
        joblib.dump(mm, SCALER_MINMAX_PATH)
        print(f"[OK] Saved: {SCALER_MINMAX_PATH}")
    else:
        print(f"[WARNING] MinMax scaler not saved (no columns available)")
    
    if ss is not None:
        joblib.dump(ss, SCALER_STANDARD_PATH)
        print(f"[OK] Saved: {SCALER_STANDARD_PATH}")
    else:
        print(f"[WARNING] Standard scaler not saved (no columns available)")
    
    if rs is not None:
        joblib.dump(rs, SCALER_ROBUST_PATH)
        print(f"[OK] Saved: {SCALER_ROBUST_PATH}")
    else:
        print(f"[WARNING] Robust scaler not saved (no columns available)")
    
    print(f"\n--- VERIFYING SCALING ---")    
    
    # Check ranges post-scaling
    for col in available_minmax[:3]:
        col_min, col_max = df[col].min(), df[col].max()
        print(f"  {col} (MinMax): [{col_min:.3f}, {col_max:.3f}]")
    
    for col in available_standard[:3]:
        col_min, col_max = df[col].min(), df[col].max()
        print(f"  {col} (Standard): [{col_min:.3f}, {col_max:.3f}]")
    
    print(f"\n--- SAVING SCALED DATASET ---")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved: {OUTPUT_CSV}")
    print(f"   {len(df):,} rows × {len(df.columns)} cols")
    
    print(f"\n--- OUTPUT SUMMARY ---")
    print(f"Scaled features: {len(available_minmax) + len(available_standard) + len(available_robust)}")
    print(f"Unscaled features: {len(COLS_NO_SCALE)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"File size: {os.path.getsize(OUTPUT_CSV) / 1e9:.2f} GB")
    
    # Log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STEP 6: NORMALIZATION & FEATURE SCALING\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nOutput: {OUTPUT_CSV}\n")
        f.write(f"Rows: {len(df):,}\n")
        f.write(f"\nTrain/Test Split:\n")
        f.write(f"  Training (fit): {n_train:,} rows (before 2024-10-01)\n")
        f.write(f"  Test (transform): {n_test:,} rows (after 2024-10-01)\n")
        f.write(f"\nScalers Saved:\n")
        f.write(f"  MinMax (11 cols): {SCALER_MINMAX_PATH}\n")
        f.write(f"  Standard (23 cols): {SCALER_STANDARD_PATH}\n")
        f.write(f"  Robust (3 cols): {SCALER_ROBUST_PATH}\n")
        f.write(f"\nUnscaled columns ({len(COLS_NO_SCALE)}): {', '.join(COLS_NO_SCALE)}\n")
        f.write(f"STATUS: ✅ COMPLETE\n")
    
    print(f"\n✓ STEP 6 COMPLETE: Data scaled & scalers saved")
    print(f"  Deployment: Load PKL scalers on RPi/TFLite pipeline")
    print(f"  Next: Step 8 (final output & audit)")

if __name__ == '__main__':
    main()
