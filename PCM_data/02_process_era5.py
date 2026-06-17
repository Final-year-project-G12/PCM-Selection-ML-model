"""
STEP 2 — ERA5 PROCESSING & FEATURE ENGINEERING
================================================
Converts raw ERA5 NetCDF → clean hourly CSV with all derived features.

Derived features computed here:
  - T_amb (°C)           from Kelvin
  - RHum (%)             from T_amb + T_dew using Magnus formula
  - W_spd (m/s)          from u + v wind components
  - GHI (W/m²)           from accumulated J/m² → divide by 3600
  - DNI (W/m²)           from mean direct flux (already W/m²)
  - DHI (W/m²)           GHI - DNI*cos(SZA)  [approximate decomposition]
  - CSI                  GHI / GHI_clearsky  (Clear Sky Index)
  - SZA (degrees)        Solar Zenith Angle via pvlib
  - ETR (W/m²)           Extraterrestrial radiation via pvlib
  - RRTDHS               Kou 2025 climate index (monthly aggregate)
  - Season               from Month
  - Sunrise/Sunset hour  from pvlib

Sources:
  - Ghodusinejad 2026: all standard met inputs, CSI, SZA, ETR
  - Barqawi 2025: GHI, DNI, DHI, Tamb, Wspd, RHum, Hour, Month
  - Kou 2025: RRTDHS = Q_sol_ave / (T_set - T_out_ave)
  - Odoi-Yorke 2025: solar irradiance, temp, cloud cover, demand
  - Chen 2025: wind speed, ambient temp for flat-plate collector model

Requirements:
    pip install xarray netCDF4 pvlib pandas numpy scipy
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pvlib
from pvlib import clearsky, atmosphere, solarposition

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CITIES = {
    "Coimbatore": {
        "lat": 11.0, "lon": 77.0,
        "altitude_m": 411,
        "climate_zone": "hot-humid",
        "T_set": 45.0,   # target water temp for SWH (°C) — Singh 2025: 40–70°C range
    },
    "Jaisalmer": {
        "lat": 26.9, "lon": 70.9,
        "altitude_m": 225,
        "climate_zone": "hot-arid",
        "T_set": 55.0,   # higher setpoint for arid high-irradiance — Kou 2025
    },
}

# Kou 2025 threshold: if RRTDHS > 5.7 → high solar resource → higher optimal Tm
RRTDHS_THRESHOLD = 5.7

INPUT_DIR  = "data/raw/era5"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def kelvin_to_celsius(T_K):
    return T_K - 273.15


def magnus_rh(T_amb_C, T_dew_C):
    """
    Compute relative humidity (%) from ambient and dew point temperatures.
    Magnus approximation — standard meteorological formula.
    """
    a, b = 17.625, 243.04
    gamma_dew = (a * T_dew_C) / (b + T_dew_C)
    gamma_amb = (a * T_amb_C) / (b + T_amb_C)
    rh = 100.0 * np.exp(gamma_dew - gamma_amb)
    return np.clip(rh, 0, 100)


def wind_speed(u, v):
    """Compute wind speed magnitude from u and v components."""
    return np.sqrt(u**2 + v**2)


def accumulated_to_flux(da, freq_hours=1):
    """
    Convert ERA5 accumulated radiation (J/m²) to mean flux (W/m²).
    ERA5 accumulates from 00:00 UTC each day, so we difference per hour.
    For hourly data: divide by 3600 seconds.
    """
    return np.maximum(da / 3600.0, 0)   # W/m², clamp negatives to 0


def get_season(month_series):
    """
    Map month to Indian climate season.
    Pre-monsoon: Mar-May | Monsoon: Jun-Sep | Post-monsoon: Oct-Nov | Winter: Dec-Feb
    """
    def _map(m):
        if m in [12, 1, 2]:   return "winter"
        elif m in [3, 4, 5]:  return "pre-monsoon"
        elif m in [6, 7, 8, 9]: return "monsoon"
        else:                 return "post-monsoon"
    return month_series.map(_map)


def compute_solar_features(df, lat, lon, altitude_m):
    """
    Compute SZA, ETR, GHI_clearsky, CSI using pvlib.
    Source: Ghodusinejad 2026 — SZA and ETR listed as key derived inputs.
    """
    times = pd.DatetimeIndex(df["timestamp"])
    location = pvlib.location.Location(
        latitude=lat, longitude=lon,
        altitude=altitude_m, tz="Asia/Kolkata"
    )

    # Solar position
    solpos = location.get_solarposition(times)
    df["SZA"] = solpos["apparent_zenith"].values        # degrees
    df["solar_azimuth"] = solpos["azimuth"].values

    # Extraterrestrial radiation (W/m²)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    df["ETR"] = dni_extra.values

    # Clear-sky GHI using Ineichen model (standard for India)
    airmass = location.get_airmass(solar_position=solpos)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
        times, lat, lon
    )
    cs = location.get_clearsky(times, model="ineichen",
                                linke_turbidity=linke_turbidity)
    df["GHI_clearsky"] = cs["ghi"].values

    # Clear Sky Index — Ghodusinejad 2026
    df["CSI"] = np.where(
        df["GHI_clearsky"] > 10,
        np.clip(df["GHI"] / df["GHI_clearsky"], 0, 1.5),
        0.0
    )

    return df


def compute_dhi(ghi, dni, sza_deg):
    """
    DHI = GHI - DNI * cos(SZA)
    Standard decomposition — Barqawi 2025, Ghodusinejad 2026.
    """
    sza_rad = np.radians(sza_deg)
    cos_sza = np.cos(sza_rad)
    dhi = ghi - dni * cos_sza
    return np.maximum(dhi, 0)


def compute_rrtdhs(df, T_set):
    """
    RRTDHS = Q_sol_ave / (T_set - T_out_ave)
    Kou 2025 — key climate index for PCM optimal Tm selection.
    Computed monthly (aggregate over each calendar month).
    Returns a column with monthly RRTDHS value assigned to each row.
    """
    df["year_month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("year_month").agg(
        Q_sol_ave=("GHI", "mean"),
        T_out_ave=("T_amb", "mean"),
    ).reset_index()
    monthly["RRTDHS"] = monthly["Q_sol_ave"] / (T_set - monthly["T_out_ave"])
    monthly["RRTDHS"] = monthly["RRTDHS"].clip(lower=0)
    df = df.merge(monthly[["year_month", "RRTDHS"]], on="year_month", how="left")
    df = df.drop(columns=["year_month"])
    return df


def compute_sunrise_sunset(df, lat, lon, altitude_m):
    """Add sunrise and sunset hours for each day."""
    location = pvlib.location.Location(latitude=lat, longitude=lon,
                                       altitude=altitude_m, tz="Asia/Kolkata")
    dates = pd.DatetimeIndex(df["timestamp"].dt.normalize().unique())
    sun = location.get_sun_rise_set_transit(dates, method="spa")
    sun["date"] = sun.index.date
    sun["sunrise_hour"] = sun["sunrise"].dt.hour + sun["sunrise"].dt.minute / 60
    sun["sunset_hour"]  = sun["sunset"].dt.hour  + sun["sunset"].dt.minute  / 60

    df["date"] = df["timestamp"].dt.date
    df = df.merge(sun[["date", "sunrise_hour", "sunset_hour"]], on="date", how="left")
    df = df.drop(columns=["date"])
    return df


# ─────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────

def process_city(city_name: str, config: dict) -> pd.DataFrame:
    nc_path = os.path.join(INPUT_DIR, f"era5_{city_name.lower()}.nc")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(
            f"ERA5 file not found: {nc_path}\n"
            f"Run 01_download_era5.py first."
        )

    print(f"\n[PROCESSING] {city_name}...")
    ds = xr.open_dataset(nc_path)

    # Spatial mean over the bounding box
    ds_mean = ds.mean(dim=["latitude", "longitude"])
    df = ds_mean.to_dataframe().reset_index()
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Temperature ──────────────────────────
    # ERA5 variable: 2m_temperature (K)
    t2m_col = [c for c in df.columns if "2m_temperature" in c or "t2m" in c.lower()]
    dew_col  = [c for c in df.columns if "dewpoint" in c or "d2m" in c.lower()]

    if t2m_col:
        df["T_amb"] = kelvin_to_celsius(df[t2m_col[0]])
    if dew_col:
        df["T_dew"] = kelvin_to_celsius(df[dew_col[0]])
        df["RHum"]  = magnus_rh(df["T_amb"], df["T_dew"])

    # ── Wind ─────────────────────────────────
    u_col = [c for c in df.columns if "10m_u" in c or "u10" in c.lower()]
    v_col = [c for c in df.columns if "10m_v" in c or "v10" in c.lower()]
    if u_col and v_col:
        df["W_spd"] = wind_speed(df[u_col[0]], df[v_col[0]])
        df["W_dir"] = np.degrees(np.arctan2(df[u_col[0]], df[v_col[0]])) % 360

    # ── Solar Radiation ───────────────────────
    ssrd_col = [c for c in df.columns if "ssrd" in c.lower() or "surface_solar_radiation" in c]
    fdir_col = [c for c in df.columns if "fdir" in c.lower() or "direct_short" in c]
    strd_col = [c for c in df.columns if "strd" in c.lower() or "thermal_radiation" in c]

    if ssrd_col:
        df["GHI"] = accumulated_to_flux(df[ssrd_col[0]])
    if fdir_col:
        df["DNI"] = df[fdir_col[0]]   # already W/m² for mean flux fields
    if strd_col:
        df["LW_down"] = accumulated_to_flux(df[strd_col[0]])   # longwave

    # ── Cloud & Precipitation ─────────────────
    cc_col   = [c for c in df.columns if "cloud" in c.lower() or "tcc" in c.lower()]
    rain_col = [c for c in df.columns if "precip" in c.lower() or "tp" in c.lower()]
    pres_col = [c for c in df.columns if "pressure" in c.lower() or "sp" in c.lower()]

    if cc_col:
        df["cloud_cover"] = df[cc_col[0]].clip(0, 1)
    if rain_col:
        df["precipitation"] = df[rain_col[0]] * 1000   # m → mm
    if pres_col:
        df["P_atm"] = df[pres_col[0]] / 100   # Pa → hPa

    # ── Time Features ─────────────────────────
    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["DOY"]   = df["timestamp"].dt.dayofyear
    df["year"]  = df["timestamp"].dt.year
    df["season"] = get_season(df["month"])
    # Encode season as integer for ML (Barqawi 2025 uses Hour, Month as features)
    season_map = {"winter": 0, "pre-monsoon": 1, "monsoon": 2, "post-monsoon": 3}
    df["season_code"] = df["season"].map(season_map)

    # ── Solar Geometry & Clear Sky ─────────────
    df = compute_solar_features(df, config["lat"], config["lon"], config["altitude_m"])

    # ── DHI from GHI and DNI ──────────────────
    if "GHI" in df.columns and "DNI" in df.columns:
        df["DHI"] = compute_dhi(df["GHI"], df["DNI"], df["SZA"])

    # ── RRTDHS Index (Kou 2025) ───────────────
    df = compute_rrtdhs(df, config["T_set"])

    # ── Sunrise / Sunset ──────────────────────
    df = compute_sunrise_sunset(df, config["lat"], config["lon"], config["altitude_m"])

    # ── Location Metadata ─────────────────────
    df["city"]         = city_name
    df["lat"]          = config["lat"]
    df["lon"]          = config["lon"]
    df["altitude_m"]   = config["altitude_m"]
    df["climate_zone"] = config["climate_zone"]
    df["T_set"]        = config["T_set"]

    # ── RRTDHS-based solar resource flag ──────
    # Kou 2025: RRTDHS > 5.7 → high solar resource
    df["high_solar_resource"] = (df["RRTDHS"] > RRTDHS_THRESHOLD).astype(int)

    # ── Drop raw ERA5 variable columns ────────
    raw_cols = [c for c in df.columns if any(
        x in c for x in ["ssrd", "fdir", "strd", "tp", "tcc", "u10", "v10",
                          "t2m", "d2m", "sp", "expver", "number"]
    )]
    df = df.drop(columns=raw_cols, errors="ignore")

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   GHI stats: mean={df['GHI'].mean():.1f}, max={df['GHI'].max():.1f} W/m²")
    print(f"   T_amb stats: mean={df['T_amb'].mean():.1f}°C")
    print(f"   RRTDHS (monthly): {df['RRTDHS'].describe().to_dict()}")

    return df


if __name__ == "__main__":
    all_dfs = []
    for city, config in CITIES.items():
        df = process_city(city, config)
        out_path = os.path.join(OUTPUT_DIR, f"climate_{city.lower()}.csv")
        df.to_csv(out_path, index=False)
        print(f"   → Saved: {out_path}")
        all_dfs.append(df)

    # Combined dataset
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "climate_all_cities.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n✅ Combined climate dataset saved → {combined_path}")
    print(f"   Total rows: {len(combined)}")
    print("\nNext step: run 03_process_pcm.py")
