import cdsapi
import os

CITIES = {
    "Coimbatore": {
        "lat_min": 10.5, "lat_max": 11.5,
        "lon_min": 76.5, "lon_max": 77.5,
    },
    "Jaisalmer": {
        "lat_min": 26.5, "lat_max": 27.5,
        "lon_min": 70.5, "lon_max": 71.5,
    },
}

#YEARS  = ["2020", "2021", "2022"]
YEARS = ["2024"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
HOURS  = [f"{h:02d}:00" for h in range(0, 24)]

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

OUTPUT_DIR = "data/raw/era5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

c = cdsapi.Client()

for city, cfg in CITIES.items():
    area = [cfg["lat_max"], cfg["lon_min"], cfg["lat_min"], cfg["lon_max"]]

    for year in YEARS:
        for month in MONTHS:

            file_name = f"{OUTPUT_DIR}/era5_{city}_{year}_{month}.nc"

            if os.path.exists(file_name):
                print("Skipping:", file_name)
                continue

            print("Downloading:", city, year, month)

            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ERA5_VARIABLES,
                    "year": year,
                    "month": month,
                    "day": DAYS,
                    "time": HOURS,
                    "area": area,
                    "format": "netcdf",
                },
                file_name,
            )

print("All downloads completed.")