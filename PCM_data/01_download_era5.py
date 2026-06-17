"""
STEP 1 — ERA5 DATA DOWNLOAD
============================
Downloads ERA5 reanalysis data for your two Indian validation cities:
  - Coimbatore (Tamil Nadu) — hot-humid coastal climate
  - Jaisalmer (Rajasthan)  — hot-arid desert climate

Requirements:
    pip install cdsapi netCDF4 xarray

Setup (one-time):
    1. Register at https://cds.climate.copernicus.eu/
    2. Accept the ERA5 licence
    3. Create ~/.cdsapirc with your API key:
         url: https://cds.climate.copernicus.eu/api/v2
         key: YOUR-UID:YOUR-API-KEY

Sources:
    - Ghodusinejad et al. 2026 (Solar Compass) — recommended ERA5/ECMWF reanalysis
    - Barqawi 2025 (Muthanna) — feature vector [GHI, DNI, DHI, Tamb, Wspd, RHum, Hour, Month]
    - Odoi-Yorke 2025 — solar irradiance, ambient temp, flow rate, demand profile
    - Kou 2025 — RRTDHS index requires Q_sol_ave and T_out_ave
"""

import cdsapi
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CITIES = {
    "Coimbatore": {
        "lat_min": 10.5, "lat_max": 11.5,
        "lon_min": 76.5, "lon_max": 77.5,
        "climate_zone": "hot-humid",
        "altitude_m": 411,
    },
    "Jaisalmer": {
        "lat_min": 26.5, "lat_max": 27.5,
        "lon_min": 70.5, "lon_max": 71.5,
        "climate_zone": "hot-arid",
        "altitude_m": 225,
    },
}

# Download years — 3 years gives solid seasonal coverage
YEARS  = ["2020", "2021", "2022"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
HOURS  = [f"{h:02d}:00" for h in range(0, 24)]

OUTPUT_DIR = "data/raw/era5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ERA5 VARIABLES — sourced from all papers
# ─────────────────────────────────────────────
#
# Variable mapping:
#   2m_temperature              → T_amb (°C after conversion from K)
#   2m_dewpoint_temperature     → T_dew → used to derive RHum
#   surface_solar_radiation_downwards → GHI proxy (J/m² accumulated, convert to W/m²)
#   mean_surface_direct_short_wave_radiation_flux → DNI equivalent
#   surface_thermal_radiation_downwards → DHI proxy
#   10m_u_component_of_wind     → wind_u
#   10m_v_component_of_wind     → wind_v  → W_spd = sqrt(u² + v²)
#   total_precipitation         → Rain (mm)
#   total_cloud_cover           → CC (0–1)
#   surface_pressure            → P_atm (Pa → hPa)

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "mean_surface_direct_short_wave_radiation_flux",
    "surface_thermal_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "total_cloud_cover",
    "surface_pressure",
]


def download_city(city_name: str, config: dict):
    """Download ERA5 data for one city and save as NetCDF."""
    out_file = os.path.join(OUTPUT_DIR, f"era5_{city_name.lower()}.nc")

    if os.path.exists(out_file):
        print(f"[SKIP] {city_name} already downloaded → {out_file}")
        return out_file

    print(f"[DOWNLOAD] Starting ERA5 download for {city_name}...")

    area = [
        config["lat_max"],   # North
        config["lon_min"],   # West
        config["lat_min"],   # South
        config["lon_max"],   # East
    ]

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": YEARS,
            "month": MONTHS,
            "day": DAYS,
            "time": HOURS,
            "area": area,
            "format": "netcdf",
        },
        out_file,
    )

    print(f"[DONE] Saved → {out_file}")
    return out_file


if __name__ == "__main__":
    for city, config in CITIES.items():
        download_city(city, config)
    print("\n✅ ERA5 download complete for all cities.")
    print("   Next step: run 02_process_era5.py")
