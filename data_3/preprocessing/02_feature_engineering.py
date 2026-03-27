import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_cleaned.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_features.csv")

def step_A2_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[A2] FEATURE ENGINEERING")
    print("-" * 40)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Cyclical
    if "hour" in df.columns and "sin_hour" not in df.columns:
        df["sin_hour"]  = np.sin(2 * np.pi * df["hour"]  / 24)
        df["cos_hour"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    if "month" in df.columns and "sin_month" not in df.columns:
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    if "DOY" in df.columns and "sin_DOY" not in df.columns:
        df["sin_DOY"]   = np.sin(2 * np.pi * df["DOY"]   / 365)
        df["cos_DOY"]   = np.cos(2 * np.pi * df["DOY"]   / 365)
        
    # Lags
    for lag in [1, 2, 3, 6, 12, 24]:
        if f"GHI_lag{lag}" not in df.columns and "GHI" in df.columns:
            df[f"GHI_lag{lag}"]   = df.groupby("city")["GHI"].shift(lag)
        if f"T_amb_lag{lag}" not in df.columns and "T_amb" in df.columns:
            df[f"T_amb_lag{lag}"] = df.groupby("city")["T_amb"].shift(lag)
            
    # Rolling
    if "GHI_roll3h_mean" not in df.columns and "GHI" in df.columns:
        df["GHI_roll3h_mean"]  = df.groupby("city")["GHI"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["GHI_roll6h_mean"]  = df.groupby("city")["GHI"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df["GHI_roll24h_std"]  = df.groupby("city")["GHI"].transform(lambda x: x.rolling(24, min_periods=1).std().fillna(0))
        
    # Derived
    if "T_depression" not in df.columns and "T_amb" in df.columns and "T_dew" in df.columns:
        df["T_depression"]       = df["T_amb"] - df["T_dew"]
    if "GHI_clearsky_diff" not in df.columns and "GHI" in df.columns and "GHI_clearsky" in df.columns:
        df["GHI_clearsky_diff"]  = df["GHI"]   - df["GHI_clearsky"]
    if "T_pcm_delta" not in df.columns and "T_amb" in df.columns and "T_set" in df.columns:
        df["T_pcm_delta"]        = df["T_amb"] - df["T_set"]
        
    print(f"  Added engineered features (lags, rolling, cyclical).")
    return df

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 01_data_cleaning.py first.")
        exit(1)
        
    df = pd.read_csv(INPUT_FILE)
    df_features = step_A2_feature_engineering(df)
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
