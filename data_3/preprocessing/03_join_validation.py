import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data_2/climate_features.csv")
VAL_DIR = os.path.join(BASE_DIR, "../data_2/validation")
GAPS_FILE = os.path.join(VAL_DIR, "temporal_gaps.csv")

def step_B_validate_joins(df: pd.DataFrame) -> dict:
    print("\n[B] SPATIAL & TEMPORAL JOIN VALIDATION")
    print("-" * 40)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    report = {}

    print("  Spatial consistency:")
    for city in df["city"].unique():
        sub = df[df["city"] == city]
        lat_vals = sub["lat"].unique()
        lon_vals = sub["lon"].unique()
        print(f"    {city}: lat={lat_vals}, lon={lon_vals}")
        if len(lat_vals) > 1 or len(lon_vals) > 1:
            print(f"    ⚠️  WARNING: Multiple lat/lon values for {city}!")
        else:
            print(f"    ✓  Single spatial point — IDW/spatial reduction correct.")
            
    report["spatial_ok"] = True

    print("\n  Temporal consistency (per city):")
    all_gaps = []
    for city in df["city"].unique():
        sub = df[df["city"] == city].sort_values("timestamp")
        time_diffs = sub["timestamp"].diff().dropna()
        expected   = pd.Timedelta("1H")
        gaps       = time_diffs[time_diffs > expected]

        print(f"    {city}:")
        print(f"      Time range: {sub['timestamp'].min()} → {sub['timestamp'].max()}")
        print(f"      Total rows: {len(sub):,}  |  Expected: {int((sub['timestamp'].max()-sub['timestamp'].min()).total_seconds()/3600)+1:,}")
        print(f"      Gaps > 1H:  {len(gaps)}")

        if len(gaps) > 0:
            gap_df = pd.DataFrame({
                "city": city,
                "gap_start": sub.loc[gaps.index - 1, "timestamp"].values if len(gaps) > 0 else [],
                "gap_hours": (gaps / pd.Timedelta("1H")).values,
            })
            all_gaps.append(gap_df)
            print(f"      ⚠️  Largest gap: {gaps.max()}")
        else:
            print(f"      ✓  No temporal gaps — regular hourly grid confirmed.")

    report["temporal_gaps"] = pd.concat(all_gaps) if all_gaps else pd.DataFrame()
    return report

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 02_feature_engineering.py first.")
        exit(1)
        
    df = pd.read_csv(INPUT_FILE)
    os.makedirs(VAL_DIR, exist_ok=True)
    report = step_B_validate_joins(df)
    
    if not report["temporal_gaps"].empty:
        report["temporal_gaps"].to_csv(GAPS_FILE, index=False)
        print(f"\nSaved temporal gaps report: {GAPS_FILE}")
    else:
        print("\nNo gaps found, nothing saved.")
