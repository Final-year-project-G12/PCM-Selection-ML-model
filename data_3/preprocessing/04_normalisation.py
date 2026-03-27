import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_features.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_scaled.csv")
PROC_DIR = os.path.join(BASE_DIR, "../data_2")

def step_C_normalise(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[C] NORMALISATION")
    print("-" * 40)
    
    df_scaled = df.copy()

    # MinMax: bounded physical quantities
    minmax_cols = [c for c in [
        "GHI", "DNI", "DHI", "avg_sdirswrf", "LW_down",
        "GHI_clearsky", "CSI", "ETR", "SZA", "solar_azimuth",
        "cloud_cover", "RHum",
    ] if c in df.columns]

    # Standard: unbounded / normally distributed
    standard_cols = [c for c in [
        "T_amb", "T_dew", "T_depression", "T_pcm_delta",
        "P_atm", "RRTDHS", "GHI_clearsky_diff",
        "GHI_roll3h_mean", "GHI_roll6h_mean", "GHI_roll24h_std",
        "W_spd",
    ] if c in df.columns]
    
    # Include all lag columns
    standard_cols += [c for c in df.columns if "lag" in c]

    # Robust: skewed / outlier-prone
    robust_cols = [c for c in ["precipitation", "W_dir"] if c in df.columns]

    # Fit and transform
    if minmax_cols:
        mm = MinMaxScaler()
        df_scaled[minmax_cols] = mm.fit_transform(df[minmax_cols])
        joblib.dump(mm, os.path.join(PROC_DIR, "scaler_minmax.pkl"))
        print(f"  MinMax scaled {len(minmax_cols)} columns (e.g., {minmax_cols[:4]}...)")

    if standard_cols:
        ss = StandardScaler()
        df_scaled[standard_cols] = ss.fit_transform(df[standard_cols])
        joblib.dump(ss, os.path.join(PROC_DIR, "scaler_standard.pkl"))
        print(f"  Standard scaled {len(standard_cols)} columns (e.g., {standard_cols[:4]}...)")

    if robust_cols:
        rs = RobustScaler()
        df_scaled[robust_cols] = rs.fit_transform(df[robust_cols])
        joblib.dump(rs, os.path.join(PROC_DIR, "scaler_robust.pkl"))
        print(f"  Robust scaled {len(robust_cols)} columns: {robust_cols}")

    no_scale = ["sin_hour", "cos_hour", "sin_month", "cos_month",
                "sin_DOY", "cos_DOY", "high_solar_resource",
                "hour", "month", "DOY", "year", "season_code",
                "sunrise_hour", "sunset_hour"]
    print(f"  Not scaled (cyclical/binary/time): {[c for c in no_scale if c in df.columns][:6]}...")

    return df_scaled

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 02_feature_engineering.py first.")
        exit(1)
        
    os.makedirs(PROC_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    df_scaled = step_C_normalise(df)
    
    df_scaled.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
