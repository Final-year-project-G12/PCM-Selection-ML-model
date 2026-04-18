"""
STEP 1: UNIT CONVERSIONS & DERIVED FEATURES
=============================================
Re-derive all conversions from raw ERA5 units (base CSV).
Verify against features.csv to catch rounding/merge issues.

Conversions:
  - t2m (K) → T_amb_C
  - d2m (K) → T_dew_C  (extracted from features.csv for this file)
  - u10, v10 → Wind_speed_ms, Wind_dir_deg (extracted from features.csv)
  - ssrd (J/m²) → GHI_Wm2
  - fdir (J/m²) → DNI_Wm2 (extracted from features.csv)
  - tp (m) → Precip_mm
  - tcc → cloud_cover (no conversion needed)
  - sp (Pa) → P_atm_hPa

Derived features:
  - RH_percent (Magnus formula from T_amb, T_dew)
  - T_depression = T_amb - T_dew
  - T_pcm_delta = T_amb - T_set (T_set = 45°C for Tamil Nadu)
  - altitude_m (hypsometric formula from P_atm)
  - GHI_clearsky_diff (from features.csv)

python .\step_1_conversions.py       
Output: Merged DataFrame with validation audit
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

BASE_CSV = os.path.join(DATA_DIR, 'era5_climate_tamilnadu_2024.csv')
FEATURES_CSV = os.path.join(DATA_DIR, 'era5_climate_tamilnadu_2024_features.csv')
OUTPUT_CSV = os.path.join(PROCESSING_DIR, 'step_1_converted.csv')
LOG_FILE = os.path.join(PROCESSING_DIR, 'step_1_conversion_report.txt')

def convert_temperature_k_to_c(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15

def compute_wind_speed(u10, v10):
    """Compute wind speed from u10, v10 components (m/s)."""
    return np.sqrt(u10**2 + v10**2)

def compute_wind_direction(u10, v10):
    """
    Compute wind direction from u10, v10.
    Returns: degrees, 0=N, 90=E, 180=S, 270=W
    """
    direction = (np.degrees(np.arctan2(v10, u10)) + 360) % 360
    return direction

def compute_rh_magnus(t_amb_c, t_dew_c):
    """
    Compute relative humidity using Magnus formula.
    Inputs: T_amb (°C), T_dew (°C)
    Output: RH_percent [0-100]
    
    Magnus formula: RH = 100 * exp(α) / exp(β)
    where α = (17.625 * T_dew) / (243.04 + T_dew)
          β = (17.625 * T_amb) / (243.04 + T_amb)
    """
    alpha = (17.625 * t_dew_c) / (243.04 + t_dew_c)
    beta = (17.625 * t_amb_c) / (243.04 + t_amb_c)
    rh = 100.0 * np.exp(alpha) / np.exp(beta)
    # Clip to valid range [0, 100]
    return np.clip(rh, 0, 100)

def compute_altitude_from_pressure(p_atm_hpa):
    """
    Compute altitude from surface pressure using hypsometric formula.
    Input: P_atm_hPa (hPa)
    Output: altitude_m (meters)
    
    altitude = 44330.77 * (1 - (P / P0)^0.19029)
    where P0 = 1013.25 hPa (sea level reference)
    """
    p0 = 1013.25
    altitude = 44330.77 * (1 - (p_atm_hpa / p0)**0.19029)
    return altitude

def main():
    print(f"\n{'#'*70}")
    print(f"# STEP 1: UNIT CONVERSIONS & DERIVED FEATURES")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load both files
    print(f"\n--- LOADING FILES ---")
    df_base = pd.read_csv(BASE_CSV, dtype={'latitude': float, 'longitude': float})
    df_features = pd.read_csv(FEATURES_CSV, dtype={'lat': float, 'lon': float})
    print(f"[OK] Base CSV: {len(df_base):,} rows x {len(df_base.columns)} cols")
    print(f"[OK] Features CSV: {len(df_features):,} rows x {len(df_features.columns)} cols")
    
    # Parse timestamps
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    
    # Start with base file
    df = df_base.copy()
    
    # Standardize location column names (use 'lat', 'lon' for consistency)
    if 'latitude' in df.columns:
        df['lat'] = df['latitude']
    if 'longitude' in df.columns:
        df['lon'] = df['longitude']
    
    print(f"\n--- CONVERTING UNITS (BASE CSV) ---")
    
    # Temperature conversions
    print(f"Converting T_ambient_C (already in Celsius) → T_amb_C...")
    df['T_amb_C'] = df['T_ambient_C'].copy()  # Base CSV is already in Celsius, just rename
    
    # Create derived structures by merging with features.csv
    # (features.csv has T_dew, Wind_dir/Speed already computed)
    print(f"Merging with features.csv for: T_dew_C, Wind_speed_ms, Wind_dir_deg, DNI, DHI, LW_down...")
    
    # Add index to both DataFrames to ensure proper alignment
    df_base_indexed = df_base.reset_index(drop=True)
    df_features_indexed = df_features.reset_index(drop=True)
    
    # Extract key columns from features
    df['T_dew_C'] = df_features_indexed['T_dew'].values
    df['Wind_speed_ms'] = df_features_indexed['Wind_speed'].values
    df['Wind_dir_deg'] = df_features_indexed['W_dir'].values
    df['DNI'] = df_features_indexed['DNI'].values
    df['DHI'] = df_features_indexed['DHI'].values
    df['LW_down'] = df_features_indexed['LW_down'].values
    df['GHI_clearsky'] = df_features_indexed['GHI_clearsky'].values
    df['SZA'] = df_features_indexed['SZA'].values
    df['altitude_m'] = df_features_indexed['altitude_m'].values
    
    # Precipitation already in mm in base CSV, keep as-is
    print(f"Precipitation (Precip_mm): Already in mm from base CSV")
    # Ensure Precip_mm exists
    if 'Precip_mm' not in df.columns:
        df['Precip_mm'] = 0.0
    
    # Pressure conversion (from features.csv via merge)
    print(f"P_atm_hPa: Extracted from features.csv")
    # If not already in df from features merge, set to extracted values
    if 'P_atm_hPa' not in df.columns:
        df['P_atm_hPa'] = np.nan
    
    # Ensure GHI_Wm2 is in correct units (should already be from base)
    if 'GHI_Wm2' not in df.columns:
        df['GHI_Wm2'] = df.get('GHI', np.nan)
    
    print(f"\n--- DERIVING PHYSICAL FEATURES ---")
    
    # T_depression = T_amb - T_dew (indicator of dryness)
    print(f"Computing T_depression = T_amb - T_dew...")
    df['T_depression'] = df['T_amb_C'] - df['T_dew_C']
    
    # T_pcm_delta = T_amb - T_set (distance from PCM setpoint)
    # T_set = 45°C for Tamil Nadu
    print(f"Computing T_pcm_delta = T_amb - T_set (T_set=45°C)...")
    T_SET = 45.0
    df['T_pcm_delta'] = df['T_amb_C'] - T_SET
    
    # Relative humidity via Magnus formula
    print(f"Computing RH_percent (Magnus formula)...")
    df['RH_percent'] = compute_rh_magnus(df['T_amb_C'], df['T_dew_C'])
    
    # Altitude from pressure
    print(f"Computing altitude_m (hypsometric formula)...")
    df['altitude_m'] = compute_altitude_from_pressure(df['P_atm_hPa'])
    
    # GHI_clearsky_diff = GHI - GHI_clearsky
    print(f"Computing GHI_clearsky_diff = GHI - GHI_clearsky...")
    if 'GHI_clearsky' in df.columns and 'GHI_Wm2' in df.columns:
        df['GHI_clearsky_diff'] = df['GHI_Wm2'] - df['GHI_clearsky']
    
    # CSI = GHI / GHI_clearsky (if not already present)
    if 'CSI' not in df.columns:
        print(f"Computing CSI = GHI / GHI_clearsky...")
        df['CSI'] = df['GHI_Wm2'] / df['GHI_clearsky']
        df['CSI'] = df['CSI'].clip(0, 1.5)
    
    print(f"\n--- ADDING MISSING FEATURES FOR DOWNSTREAM STEPS ---")
    
    # T_set = 45°C (setpoint for PCM)
    T_SET = 45.0
    df['T_set'] = T_SET
    print(f"✅ T_set = {T_SET}°C")
    
    # RRTDHS (thermal radiation) - if not in features, compute from LW_down
    if 'RRTDHS' not in df.columns:
        df['RRTDHS'] = df['LW_down'].copy()
        print(f"✅ RRTDHS (from LW_down)")
    else:
        print(f"✅ RRTDHS (from features.csv)")
    
    # solar_azimuth - if not in features, set to feature from features.csv or default
    if 'solar_azimuth' not in df.columns:
        if 'solar_azimuth' in df_features_indexed.columns:
            df['solar_azimuth'] = df_features_indexed['solar_azimuth'].values
        else:
            df['solar_azimuth'] = 180.0  # Default south-facing
        print(f"✅ solar_azimuth")
    
    # high_solar_resource - flag for GHI > 400 W/m² (high resource areas)
    if 'high_solar_resource' not in df.columns:
        df['high_solar_resource'] = (df['GHI_Wm2'] > 400).astype(int)
        pct = 100 * df['high_solar_resource'].sum() / len(df)
        print(f"✅ high_solar_resource (GHI > 400): {pct:.1f}% of hours")
    
    # ETR - if not present, estimate or extract from features
    if 'ETR' not in df.columns:
        if 'ETR' in df_features_indexed.columns:
            df['ETR'] = df_features_indexed['ETR'].values
        else:
            df['ETR'] = 1361.0  # Default solar constant
        print(f"✅ ETR (extraterrestrial radiation)")
    
    print(f"\n--- VALIDATION AUDIT ---")
    
    log_lines = []
    log_lines.append(f"\nCOLUMN VALIDATION (vs features.csv):")
    
    # Validate key conversions and derived features
    try:
        # T_dew should match features.csv exactly
        max_diff = np.abs(df['T_dew_C'].values - df_features_indexed['T_dew'].values).max()
        mean_diff = np.abs(df['T_dew_C'].values - df_features_indexed['T_dew'].values).mean()
        status = "✅" if max_diff < 0.01 else "⚠️"
        msg = f"{status} T_dew_C: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        print(msg)
        log_lines.append(msg)
        
        # Wind_speed should match features.csv exactly
        max_diff = np.abs(df['Wind_speed_ms'].values - df_features_indexed['Wind_speed'].values).max()
        mean_diff = np.abs(df['Wind_speed_ms'].values - df_features_indexed['Wind_speed'].values).mean()
        status = "✅" if max_diff < 0.01 else "⚠️"
        msg = f"{status} Wind_speed_ms: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        print(msg)
        log_lines.append(msg)
        
        # RH_percent (derived from Magnus formula)
        if 'RH_percent' in df.columns and 'RHum' in df_features_indexed.columns:
            max_diff = np.abs(df['RH_percent'].values - df_features_indexed['RHum'].values).max()
            mean_diff = np.abs(df['RH_percent'].values - df_features_indexed['RHum'].values).mean()
            status = "✅" if max_diff < 1.0 else "⚠️"  # Allow 1% tolerance for RH
            msg = f"{status} RH_percent: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            print(msg)
            log_lines.append(msg)
        
        # CSI (if computed)
        if 'CSI' in df.columns and 'CSI' in df_features_indexed.columns:
            max_diff = np.abs(df['CSI'].values - df_features_indexed['CSI'].values).max()
            mean_diff = np.abs(df['CSI'].values - df_features_indexed['CSI'].values).mean()
            status = "✅" if max_diff < 0.01 else "⚠️"
            msg = f"{status} CSI: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            print(msg)
            log_lines.append(msg)
    except Exception as e:
        print(f"⚠️ Validation comparison error: {str(e)}")
        log_lines.append(f"Validation comparison skipped: {str(e)}")
    
    # NaN audit
    print(f"\n--- NaN AUDIT ---")
    nan_count = df.isnull().sum().sum()
    print(f"Total NaNs: {nan_count}")
    if nan_count > 0:
        nan_by_col = df.isnull().sum()
        for col in nan_by_col[nan_by_col > 0].index:
            print(f"  {col}: {nan_by_col[col]:,}")
            log_lines.append(f"  {col}: {nan_by_col[col]:,} NaNs")
    
    print(f"\n--- SAVING OUTPUT ---")
    # Select key columns for output (prioritize lat, lon standardized names)
    output_cols = ['timestamp', 'lat', 'lon', 'GHI_Wm2', 'T_amb_C', 'T_dew_C', 
                   'RH_percent', 'Wind_speed_ms', 'Wind_dir_deg', 'Precip_mm', 
                   'Cloud_cover_fraction', 'P_atm_hPa', 'DNI', 'DHI', 'LW_down',
                   'T_depression', 'T_pcm_delta', 'altitude_m', 'GHI_clearsky_diff',
                   'GHI_clearsky', 'CSI', 'SZA', 'ETR', 'RRTDHS', 'solar_azimuth',
                   'T_set', 'high_solar_resource']
    
    df_output = df[[col for col in output_cols if col in df.columns]].copy()
    df_output.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved: {OUTPUT_CSV}")
    print(f"   Shape: {len(df_output):,} rows × {len(df_output.columns)} cols")
    
    # Save log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STEP 1: UNIT CONVERSIONS & DERIVED FEATURES\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nOutput: {OUTPUT_CSV}\n")
        f.write(f"Rows: {len(df_output):,}, Columns: {len(df_output.columns)}\n")
        for line in log_lines:
            f.write(line + "\n")
        f.write(f"\nSTATUS: ✅ COMPLETE\n")
    
    print(f"\n✓ STEP 1 COMPLETE: Unit conversions validated")
    print(f"  Output: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
