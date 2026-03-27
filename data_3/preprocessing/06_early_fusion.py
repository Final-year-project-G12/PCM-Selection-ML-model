import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_scaled.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_all_cities_preprocessed.csv")

FEATURE_GROUPS = {
    "solar_sensor":  [
        "GHI", "DNI", "DHI", "avg_sdirswrf", "LW_down",
        "GHI_clearsky", "CSI", "ETR", "SZA", "solar_azimuth",
        "sunrise_hour", "sunset_hour",
    ],
    "weather_nwp":   [
        "T_amb", "T_dew", "RHum", "W_spd", "W_dir",
        "cloud_cover", "precipitation", "P_atm", "RRTDHS",
    ],
    "pcm_thermal":   [
        "T_set", "T_pcm_delta", "T_depression",
    ],
    "time_features": [
        "hour", "month", "DOY", "year", "season_code",
        "sin_hour", "cos_hour", "sin_month", "cos_month",
        "sin_DOY", "cos_DOY", "high_solar_resource",
    ],
    "lag_features":  [],
    "rolling_features": [],
}

def step_E_early_fusion(df_scaled: pd.DataFrame) -> pd.DataFrame:
    print("\n[E] EARLY FUSION — assembling feature matrix")
    print("-" * 40)
    
    FEATURE_GROUPS["lag_features"]     = [c for c in df_scaled.columns if "lag" in c]
    FEATURE_GROUPS["rolling_features"] = [c for c in df_scaled.columns if "roll" in c]

    all_feature_cols = []
    for group, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df_scaled.columns]
        FEATURE_GROUPS[group] = present
        all_feature_cols.extend(present)
        print(f"  {group:20s}: {len(present):3d} features")

    # Remove duplicates
    seen = set()
    all_feature_cols = [c for c in all_feature_cols if not (c in seen or seen.add(c))]

    # Non-feature metadata to keep
    meta_cols = [c for c in ["timestamp", "city", "lat", "lon",
                              "altitude_m", "climate_zone", "season"]
                 if c in df_scaled.columns]

    X = df_scaled[meta_cols + all_feature_cols].copy()
    
    # Drop rows that have NaNs due to initial lag calculations
    n_before = len(X)
    X = X.dropna(subset=all_feature_cols)
    
    print(f"\n  Rows before dropna: {n_before:,}")
    print(f"  Rows after dropna:  {len(X):,}  (dropped {n_before-len(X):,} from lag NaN)")
    print(f"  Final feature matrix X: {X.shape}")
    print(f"  Total features: {len(all_feature_cols)}")

    return X

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 04_normalisation.py first.")
        exit(1)
        
    df_scaled = pd.read_csv(INPUT_FILE)
    X_final = step_E_early_fusion(df_scaled)
    
    X_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*60}")
    print(f"✅ Final preprocessed dataset saved → {OUTPUT_FILE}")
    print(f"   Shape: {X_final.shape}")
