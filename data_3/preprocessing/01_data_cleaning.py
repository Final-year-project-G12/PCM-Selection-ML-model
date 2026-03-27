import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Define paths relative to the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_all_cities.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_cleaned.csv")
PROC_DIR = os.path.dirname(OUTPUT_FILE)

PHYSICAL_BOUNDS = {
    "GHI":            (0.0,    1400.0),
    "DNI":            (0.0,    1200.0),
    "DHI":            (0.0,     800.0),
    "avg_sdirswrf":   (0.0,    1200.0),
    "LW_down":        (100.0,   600.0),
    "T_amb":          (-10.0,   55.0),
    "T_dew":          (-20.0,   40.0),
    "RHum":           (0.0,    100.0),
    "W_spd":          (0.0,     30.0),
    "W_dir":          (0.0,    360.0),
    "precipitation":  (0.0,    200.0),
    "cloud_cover":    (0.0,      1.0),
    "P_atm":          (800.0, 1100.0),
    "CSI":            (0.0,     1.5),
    "SZA":            (0.0,    180.0),
    "ETR":            (0.0,   1500.0),
    "GHI_clearsky":   (0.0,   1400.0),
}

def step_A_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[A] DATA CLEANING")
    print("-" * 40)
    original_rows = len(df)
    
    # 1. Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Rows loaded: {original_rows:,}")

    # 2. Remove duplicates
    dupes = df.duplicated(subset=["timestamp", "city"]).sum()
    df = df[~df.duplicated(subset=["timestamp", "city"], keep="first")]
    print(f"  Duplicate timestamps removed: {dupes}")

    # 3. Physical bounds → NaN
    outlier_counts = {}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            outlier_counts[col] = n
            
    if outlier_counts:
        print(f"  Outliers set to NaN: {outlier_counts}")
    else:
        print("  No physical bound outliers detected.")

    # 4. Per-city time-series interpolation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cities = df["city"].unique()
    dfs_clean = []
    for city in cities:
        sub = df[df["city"] == city].copy()
        sub[numeric_cols] = sub[numeric_cols].interpolate(
            method="linear", limit=6, limit_direction="both"
        )
        sub[numeric_cols] = sub[numeric_cols].ffill().bfill()
        dfs_clean.append(sub)
    
    df = pd.concat(dfs_clean, ignore_index=True)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # 5. Final missing count
    remaining_nan = df[numeric_cols].isnull().sum().sum()
    print(f"  Remaining NaN after interpolation: {remaining_nan}")
    print(f"  Final rows: {len(df):,}")
    return df

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 02_combine.py first?")
        exit(1)
        
    os.makedirs(PROC_DIR, exist_ok=True)
    df_raw = pd.read_csv(INPUT_FILE)
    df_clean = step_A_clean(df_raw)
    
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
