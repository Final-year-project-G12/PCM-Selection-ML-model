"""
STEP 2: TEMPORAL FEATURE ENGINEERING
=====================================
Add cyclical time encodings, lagged features, rolling statistics, and season one-hot encoding.

Features:
  A. Cyclical time (6 features):
     - sin_hour, cos_hour, sin_month, cos_month, sin_DOY, cos_DOY
  
  B. Lagged features (12 features):
     - GHI_lag1, GHI_lag2, GHI_lag3, GHI_lag6, GHI_lag12, GHI_lag24
     - T_amb_lag1, T_amb_lag2, T_amb_lag3, T_amb_lag6, T_amb_lag12, T_amb_lag24
     ⚠️  CRITICAL: Group by (lat, lon), NOT by city
  
  C. Rolling statistics (4 features):
     - GHI_roll3h_mean, GHI_roll6h_mean, GHI_roll24h_std, T_roll24h_mean
  
  D. Season one-hot encoding (4 features):
     - season_Winter, season_Summer, season_Monsoon, season_NE


python .\step_2_temporal_features.py
Input: era5_climate_tamilnadu_2024_features.csv (has timestamp, hour, month, DOY, GHI, T_amb)
Output: Features with lagged + rolling columns; first 24 rows per location = NaN
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

INPUT_CSV = os.path.join(PROCESSING_DIR, 'step_1_converted.csv')  # Input from Step 1
OUTPUT_CSV = os.path.join(PROCESSING_DIR, 'step_2_temporal_features.csv')
LOG_FILE = os.path.join(PROCESSING_DIR, 'step_2_temporal_report.txt')

def encode_cyclical(values, period):
    """Encode circular feature as sin/cos pair."""
    sin_vals = np.sin(2 * np.pi * values / period)
    cos_vals = np.cos(2 * np.pi * values / period)
    return sin_vals, cos_vals

def map_season(month):
    """
    Map month (1-12) to season.
    Returns: 'Winter', 'Summer', 'Monsoon', 'NE-Monsoon'
    
    Winter: Jan, Feb, Dec (1, 2, 12)
    Summer: Mar, Apr, May (3, 4, 5)
    Monsoon: Jun, Jul, Aug, Sep (6, 7, 8, 9)
    NE-Monsoon: Oct, Nov (10, 11)
    """
    if month in [1, 2, 12]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'NE-Monsoon'
    else:
        return None

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 2: TEMPORAL FEATURE ENGINEERING")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load Step 1 output (converted and derived features)
    print(f"\n--- LOADING DATA ---")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"[OK] Loaded: {len(df):,} rows x {len(df.columns)} cols")
    
    # Extract temporal components if not already present
    if 'hour' not in df.columns:
        print(f"Extracting temporal components from timestamp...")
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        df['DOY'] = df['timestamp'].dt.dayofyear
        df['year'] = df['timestamp'].dt.year
    
    print(f"\n--- A. CYCLICAL TIME ENCODING (6 features) ---")
    
    # Cyclical encoding for hour
    sin_hour, cos_hour = encode_cyclical(df['hour'], period=24)
    df['sin_hour'] = sin_hour
    df['cos_hour'] = cos_hour
    print(f"[OK] sin_hour, cos_hour (hour of day)")
    
    # Cyclical encoding for month
    sin_month, cos_month = encode_cyclical(df['month'], period=12)
    df['sin_month'] = sin_month
    df['cos_month'] = cos_month
    print(f"[OK] sin_month, cos_month (month of year)")
    
    # Cyclical encoding for DOY
    sin_doy, cos_doy = encode_cyclical(df['DOY'], period=365)
    df['sin_DOY'] = sin_doy
    df['cos_DOY'] = cos_doy
    print(f"[OK] sin_DOY, cos_DOY (day of year)")
    
    print(f"\n--- B. LAGGED FEATURES (12 features) ---")
    print(f"[WARNING] Grouping by (lat, lon) to preserve spatial independence...")
    
    lag_windows = [1, 2, 3, 6, 12, 24]
    
    # Lagged GHI (from Step 1, column is GHI_Wm2)
    ghi_col = 'GHI_Wm2' if 'GHI_Wm2' in df.columns else 'GHI'
    for lag in lag_windows:
        df[f'GHI_lag{lag}'] = df.groupby(['lat', 'lon'])[ghi_col].shift(lag)
        if lag == 1:
            print(f"[OK] GHI_lag{lag} through GHI_lag{max(lag_windows)}")
    
    # Lagged T_amb (from Step 1, column is T_amb_C)
    temp_col = 'T_amb_C' if 'T_amb_C' in df.columns else 'T_amb'
    for lag in lag_windows:
        df[f'T_amb_lag{lag}'] = df.groupby(['lat', 'lon'])[temp_col].shift(lag)
        if lag == 1:
            print(f"[OK] T_amb_lag{lag} through T_amb_lag{max(lag_windows)}")
    
    lag_nans_per_location = 24  # First 24 rows per (lat, lon)
    total_lag_nans = len(df[['lat', 'lon']].drop_duplicates()) * lag_nans_per_location
    print(f"\n[WARNING] Lag NaNs created: {lag_nans_per_location} rows per location")
    print(f"   Total NaN rows: {total_lag_nans:,} (will be handled in Step 3)")
    
    print(f"\n--- C. ROLLING STATISTICS (4 features) ---")
    
    # GHI rolling statistics (use GHI_Wm2 if available, else GHI)
    ghi_col = 'GHI_Wm2' if 'GHI_Wm2' in df.columns else 'GHI'
    temp_col = 'T_amb_C' if 'T_amb_C' in df.columns else 'T_amb'
    
    df['GHI_roll3h_mean'] = df.groupby(['lat', 'lon'])[ghi_col].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    print(f"[OK] GHI_roll3h_mean")
    
    df['GHI_roll6h_mean'] = df.groupby(['lat', 'lon'])[ghi_col].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    print(f"[OK] GHI_roll6h_mean")
    
    df['GHI_roll24h_std'] = df.groupby(['lat', 'lon'])[ghi_col].transform(
        lambda x: x.rolling(24, min_periods=1).std()
    )
    print(f"[OK] GHI_roll24h_std")
    
    # T_amb rolling statistic
    df['T_roll24h_mean'] = df.groupby(['lat', 'lon'])[temp_col].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )
    print(f"[OK] T_roll24h_mean")
    
    print(f"\n--- D. SEASON ENCODING (5 features) ---")
    
    # Ensure season column exists
    if 'season' not in df.columns:
        df['season'] = df['month'].apply(map_season)
    
    # One-hot encode season
    season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=False)
    # Ensure all 4 seasons are present (with hyphenated names)
    for season in ['Winter', 'Summer', 'Monsoon', 'NE-Monsoon']:
        col_name = f'season_{season}'
        if col_name not in season_dummies.columns:
            season_dummies[col_name] = 0
        df[col_name] = season_dummies[col_name].astype(int)
    
    print(f"[OK] season_Winter, season_Summer, season_Monsoon, season_NE-Monsoon")
    
    # Also ensure season_code exists
    season_to_code = {'Winter': 1, 'Summer': 2, 'Monsoon': 3, 'NE-Monsoon': 4}
    df['season_code'] = df['season'].map(season_to_code).astype(int)
    print(f"[OK] season_code (1-4)")
    
    print(f"\n--- NaN AUDIT AFTER FEATURE ENGINEERING ---")
    nan_by_col = df.isnull().sum()
    nan_cols = nan_by_col[nan_by_col > 0].sort_values(ascending=False)
    print(f"Columns with NaNs:")
    for col, count in nan_cols.head(10).items():
        print(f"  {col}: {count:,} NaNs ({100*count/len(df):.2f}%)")
    
    print(f"\n--- SAVING OUTPUT ---")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved: {OUTPUT_CSV}")
    print(f"   Shape: {len(df):,} rows × {len(df.columns)} cols")
    
    # Summary stats
    print(f"\n--- NEW FEATURES SUMMARY ---")
    print(f"Cyclical time features: 6")
    print(f"Lagged features: 12")
    print(f"Rolling statistics: 4")
    print(f"Season encoding: 5 (one-hot 4 + code 1)")
    print(f"Total new features: 27")
    print(f"Total columns in output: {len(df.columns)}")
    
    # Log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STEP 2: TEMPORAL FEATURE ENGINEERING\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nOutput: {OUTPUT_CSV}\n")
        f.write(f"Rows: {len(df):,}, Columns: {len(df.columns)}\n")
        f.write(f"\nNew features added:\n")
        f.write(f"  Cyclical: 6\n")
        f.write(f"  Lagged: 12\n")
        f.write(f"  Rolling: 4\n")
        f.write(f"  Season: 5\n")
        f.write(f"\nLag NaNs: {total_lag_nans:,} rows per location (first 24 hours)\n")
        f.write(f"STATUS: ✅ COMPLETE\n")
    
    print(f"\n✓ STEP 2 COMPLETE: Temporal features engineered")
    print(f"  Next: Step 3 (temporal alignment & lag handling)")

if __name__ == '__main__':
    main()
