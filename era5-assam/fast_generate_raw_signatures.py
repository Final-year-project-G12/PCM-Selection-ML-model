"""
Fast generator for data/processed/climate_signatures_raw.csv
Computes all 18 physical climate signature indices directly from
data/preprocessed/assam_cleaned_physical.csv in ~20 seconds.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/preprocessed/assam_cleaned_physical.csv")
OUT_FILE   = Path("data/processed/climate_signatures_raw.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading assam_cleaned_physical.csv ...")
cols = [
    "point_id", "lat", "lon", "population", "month",
    "era5_T_amb", "era5_T_dew", "era5_RHum", "era5_W_spd",
    "era5_GHI", "era5_GHI_clearsky", "era5_CSI", "era5_cloud_cover",
    "era5_precipitation", "era5_P_atm", "is_daytime"
]
df = pd.read_csv(INPUT_FILE, usecols=cols)
print(f"Loaded {len(df):,} rows for {df['point_id'].nunique()} points.")

# Compute per-point climate signature
rows = []
for pid, grp in df.groupby("point_id"):
    lat = grp["lat"].iloc[0]
    lon = grp["lon"].iloc[0]
    pop = grp["population"].iloc[0]

    # Temperatures
    ta = grp["era5_T_amb"]
    ta_mean = ta.mean()
    ta_p95  = ta.quantile(0.95)
    ta_p05  = ta.quantile(0.05)

    # Daily temperature range proxy
    # Group by date within point for daily DTR
    date_t = grp.groupby("month")["era5_T_amb"].agg(["max", "min"])
    dtr = (date_t["max"] - date_t["min"]).mean()

    # Solar
    daytime = grp[grp["is_daytime"] == True]
    ghi_mean = daytime["era5_GHI"].mean() if len(daytime) > 0 else grp["era5_GHI"].mean()
    ghi_daily_kwh = (grp["era5_GHI"].mean() * 24.0) / 1000.0

    # Clearness index / CSI
    csi_col = grp["era5_CSI"]
    kt_mean = csi_col.mean()
    kt_std  = csi_col.std()
    sai     = (csi_col > 0.6).mean()
    cci     = grp["era5_cloud_cover"].mean()
    cloudy_frac = (grp["era5_cloud_cover"] > 0.7).mean()

    # Degree days
    hdd18 = np.maximum(0, 18.0 - ta).sum() / 24.0  # converted to degree-days per year equivalent
    cdd24 = np.maximum(0, ta - 24.0).sum() / 24.0

    # Humidity & Wind
    rh_mean = grp["era5_RHum"].mean()
    wind_mean = grp["era5_W_spd"].mean()

    # Seasonality (ratio of std to mean of monthly GHI)
    m_ghi = grp.groupby("month")["era5_GHI"].mean()
    seasonality = m_ghi.std() / (m_ghi.mean() + 1e-6)

    # HSI (Humidity x Dewpoint interaction)
    hsi = rh_mean * grp["era5_T_dew"].mean()

    # Monsoon index (Monsoon: June-Sept months 6..9)
    monsoon_precip = grp[grp["month"].isin([6,7,8,9])]["era5_precipitation"].sum()
    total_precip   = grp["era5_precipitation"].sum()
    monsoon_index  = monsoon_precip / total_precip if total_precip > 0 else 0.5

    # Elevation proxy
    elev_proxy = grp["era5_P_atm"].mean() / 1013.25

    rows.append({
        "point_id": pid,
        "lat": lat,
        "lon": lon,
        "population": pop,
        "Ta_mean": ta_mean,
        "Ta_p95": ta_p95,
        "Ta_p05": ta_p05,
        "DTR": dtr,
        "GHI_daily_kWh": ghi_daily_kwh,
        "kt_mean": kt_mean,
        "kt_std": kt_std,
        "SAI": sai,
        "CCI": cci,
        "cloudy_frac": cloudy_frac,
        "HDD18": hdd18,
        "CDD24": cdd24,
        "RH_mean": rh_mean,
        "wind_mean": wind_mean,
        "seasonality": seasonality,
        "GHI_mean": ghi_mean,
        "HSI": hsi,
        "monsoon_index": monsoon_index,
        "elev_proxy": elev_proxy
    })

res = pd.DataFrame(rows)
res.to_csv(OUT_FILE, index=False)
print(f"DONE! Generated {OUT_FILE} with {len(res)} location rows and {len(res.columns)} columns.")
