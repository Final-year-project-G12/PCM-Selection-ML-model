"""
STEP 8: FINAL OUTPUT & VERIFICATION
====================================
Save final preprocessed CSV with all features, run comprehensive audit.

Final output: era5_tamilnadu_preprocessed_final.csv
Location: tamilnadu_data/data/

Expected spec:
  Rows: 3,425,160 (after lag drop)
  Columns: ~58 (base + temporal + lagged + rolling + derived)
  NaNs: 0
  File size: ~1.6 GB

python .\step_8_final_output.py

Input: step_6_scaled.csv
Output: era5_tamilnadu_preprocessed_final.csv + detailed audit report
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

INPUT_CSV = os.path.join(PROCESSING_DIR, 'step_6_scaled.csv')
FINAL_CSV = os.path.join(DATA_DIR, 'era5_tamilnadu_preprocessed_final.csv')
AUDIT_REPORT = os.path.join(PROCESSING_DIR, 'step_8_final_audit.txt')
PROCESSING_LOG = os.path.join(PROCESSING_DIR, 'preprocessing_complete.log')

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 8: FINAL OUTPUT & VERIFICATION")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load scaled data
    print(f"\n--- LOADING SCALED DATA ---")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERROR: {INPUT_CSV} not found!")
        return
    
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} cols")
    
    print(f"\n--- DATASET SPECIFICATIONS ---")
    
    print(f"\nShape:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    expected_rows = 3425160
    if abs(len(df) - expected_rows) < 100:
        print(f"  ✅ Row count matches expected (~{expected_rows:,})")
    else:
        delta = len(df) - expected_rows
        print(f"  ⚠️  Row difference: {delta:+,}")
    
    print(f"\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}")
    
    print(f"\nMemory Usage:")
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  Approximate: {mem_mb:.1f} MB in memory")
    
    print(f"\nMissing Values:")
    nan_total = df.isnull().sum().sum()
    print(f"  Total NaNs: {nan_total}")
    if nan_total == 0:
        print(f"  ✅ Zero NaNs — dataset complete!")
    else:
        nan_by_col = df.isnull().sum()
        nan_cols = nan_by_col[nan_by_col > 0].sort_values(ascending=False)
        print(f"  ⚠️  {len(nan_cols)} columns have NaNs:")
        for col in nan_cols.index[:10]:  # Show top 10
            pct = 100 * nan_cols[col] / len(df)
            print(f"    {col}: {nan_cols[col]:,} ({pct:.2f}%)")
        if len(nan_cols) > 10:
            print(f"    ... and {len(nan_cols) - 10} more")
    
    print(f"\nTemporal Coverage:")
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    print(f"  Start: {ts_min}")
    print(f"  End: {ts_max}")
    print(f"  Duration: {(ts_max - ts_min).days + 1} days")
    
    print(f"\nSpatial Coverage:")
    n_locations = len(df[['lat', 'lon']].drop_duplicates())
    print(f"  Unique locations: {n_locations}")
    if n_locations == 391:
        print(f"  ✅ Complete grid (391 × 0.25° Tamil Nadu)")
    
    print(f"\n--- FEATURE INVENTORY ---")
    
    feature_categories = {
        'Identifiers': ['timestamp', 'lat', 'lon'],
        'Raw Climate': ['T_amb', 'T_dew', 'RH_percent', 'Wind_speed_ms', 'Wind_dir_deg',
                       'GHI', 'DNI', 'DHI', 'LW_down', 'cloud_cover', 'Precip_mm', 'P_atm_hPa'],
        'Derived': ['T_depression', 'T_pcm_delta', 'altitude_m', 'GHI_clearsky_diff', 'CSI', 'RRTDHS'],
        'Solar Geometry': ['SZA', 'solar_azimuth', 'ETR', 'GHI_clearsky'],
        'Temporal Raw': ['hour', 'month', 'DOY', 'year', 'season_code'],
        'Temporal Cyclical': ['sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_DOY', 'cos_DOY'],
        'Season One-Hot': ['season_Winter', 'season_Summer', 'season_Monsoon', 'season_NE'],
        'Lagged GHI': [col for col in df.columns if col.startswith('GHI_lag')],
        'Lagged T_amb': [col for col in df.columns if col.startswith('T_amb_lag')],
        'Rolling Stats': [col for col in df.columns if 'roll' in col],
        'Anomaly Flags': ['solar_anomaly_flag'],
        'Targets': ['T_set', 'high_solar_resource'],
    }
    
    total_expected = 0
    for category, cols in feature_categories.items():
        available = [c for c in cols if c in df.columns]
        print(f"{category}: {len(available)}/{len(cols)}")
        total_expected += len(available)
        if len(available) <= 6:
            print(f"  {available}")
    
    print(f"\nTotal feature columns: {total_expected} (out of {len(df.columns)} total)")
    
    print(f"\n--- NUMERIC SUMMARY STATISTICS ---")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    
    print(f"\nElement-wise statistics:")
    stats_df = df[numeric_cols[:8]].describe().round(2)
    print(stats_df.to_string())
    
    print(f"\n--- SAVING FINAL OUTPUT ---")
    
    df.to_csv(FINAL_CSV, index=False)
    file_size_gb = os.path.getsize(FINAL_CSV) / 1e9
    print(f"[OK] Saved: {FINAL_CSV}")
    print(f"   File size: {file_size_gb:.2f} GB")
    
    print(f"\n--- SCALERS VERIFICATION ---")
    
    scalers = [
        os.path.join(DATA_DIR, 'scaler_minmax_tamilnadu.pkl'),
        os.path.join(DATA_DIR, 'scaler_standard_tamilnadu.pkl'),
        os.path.join(DATA_DIR, 'scaler_robust_tamilnadu.pkl'),
    ]
    
    for scaler_path in scalers:
        if os.path.exists(scaler_path):
            size_kb = os.path.getsize(scaler_path) / 1e3
            print(f"[OK] {os.path.basename(scaler_path)} ({size_kb:.1f} KB)")
        else:
            print(f"[WARNING] {os.path.basename(scaler_path)} NOT FOUND")
    
    print(f"\n--- FINAL CHECKLIST ---")
    
    checks = [
        ('Output CSV exists', os.path.exists(FINAL_CSV)),
        ('Row count ≈ 3,425,160', abs(len(df) - 3425160) < 100),
        ('Columns ≈ 58', 55 <= len(df.columns) <= 65),
        ('Zero NaNs', nan_total == 0),
        ('Complete temporal span', (ts_max - ts_min).days >= 364),
        ('Complete spatial grid', n_locations == 391),
        ('Scalers saved', all(os.path.exists(p) for p in scalers)),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    # Write audit report
    with open(AUDIT_REPORT, 'w', encoding='utf-8') as f:
        f.write(f"STEP 8: FINAL OUTPUT & VERIFICATION\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFINAL OUTPUT:\n")
        f.write(f"  File: {FINAL_CSV}\n")
        f.write(f"  Size: {file_size_gb:.2f} GB\n")
        f.write(f"  Rows: {len(df):,}\n")
        f.write(f"  Columns: {len(df.columns)}\n")
        f.write(f"  NaNs: {nan_total}\n")
        f.write(f"\nTEMPORAL:\n")
        f.write(f"  Start: {ts_min}\n")
        f.write(f"  End: {ts_max}\n")
        f.write(f"\nSPATIAL:\n")
        f.write(f"  Locations: {n_locations}\n")
        f.write(f"\nFEATURES:\n")
        for category, cols in feature_categories.items():
            available = len([c for c in cols if c in df.columns])
            f.write(f"  {category}: {available}\n")
        f.write(f"\nSCALERS:\n")
        for scaler_path in scalers:
            if os.path.exists(scaler_path):
                f.write(f"  ✅ {os.path.basename(scaler_path)}\n")
        f.write(f"\nVERIFICATION:\n")
        for check_name, result in checks:
            status = "PASS" if result else "FAIL"
            f.write(f"  {status}: {check_name}\n")
        f.write(f"\nSTATUS: {'✅ ALL CHECKS PASSED' if all_passed else '⚠️  SOME CHECKS FAILED'}\n")
    
    # Write processing completion log
    with open(PROCESSING_LOG, 'w', encoding='utf-8') as f:
        f.write(f"PREPROCESSING PIPELINE COMPLETE\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFinal output: {FINAL_CSV}\n")
        f.write(f"Audit report: {AUDIT_REPORT}\n")
        f.write(f"\nSteps executed:\n")
        f.write(f"  ✓ Step 0: Audit\n")
        f.write(f"  ✓ Step 1: Unit Conversions\n")
        f.write(f"  ✓ Step 2: Temporal Features\n")
        f.write(f"  ✓ Step 3: Temporal Alignment\n")
        f.write(f"  ✓ Step 4: Outlier Detection\n")
        f.write(f"  ✓ Step 5: Spatial Joins\n")
        f.write(f"  ✓ Step 6: Normalization\n")
        f.write(f"  ✓ Step 8: Final Output\n")
        f.write(f"\nNext steps:\n")
        f.write(f"  1. Load {FINAL_CSV} for XGBoost PCM classifier training\n")
        f.write(f"  2. Load scalers from {os.path.dirname(scalers[0])} for RPi deployment\n")
        f.write(f"  3. Use regional CSVs (climate_*.csv) for prototype simulation\n")
    
    print(f"\n✓ STEP 8 COMPLETE: Final output verified")
    print(f"\nAll files saved to:")
    print(f"  Data: {FINAL_CSV}")
    print(f"  Audit: {AUDIT_REPORT}")
    print(f"  Scalers: {os.path.dirname(scalers[0])}")
    
    if all_passed:
        print(f"\n{'='*70}")
        print(f"\n[SUCCESS] PREPROCESSING PIPELINE COMPLETE!")
        print(f"{'='*70}")
    else:
        print(f"\n⚠️  Some verification checks failed — review audit report")

if __name__ == '__main__':
    main()
