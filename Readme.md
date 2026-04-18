# ERA5 Tamil Nadu Climate Data Pipeline
### Complete hourly climate & solar resource dataset for all of Tamil Nadu (2024–2025)

---

## What This Project Does — Big Picture

This pipeline downloads **real atmospheric data from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 reanalysis dataset** and transforms it into clean, analysis-ready CSV files covering **every major city, town, taluk and district in Tamil Nadu**, plus every ERA5 grid point across the state.

The output is a dataset of **hourly climate and solar radiation values** for 2 full years (2024 + 2025) across 222 named locations and 575 spatial grid points — roughly **14 million rows of data total**.

This data can be used to:
- Train machine learning models for **solar irradiance forecasting**
- Perform **site selection analysis** for solar panel installation
- Study **spatial variation of solar energy** across Tamil Nadu
- Analyze **climate zone differences** between coastal, inland, semi-arid, and hill station regions
- Build **solar energy yield prediction** tools for any location in Tamil Nadu

---

## The Three Scripts

```
00_unzip_accum.py          ← Run FIRST  (one time only)
01_download_era5_tamilnadu.py  ← Run SECOND (downloads raw data from ECMWF)
02_combine_tamilnadu.py    ← Run THIRD  (processes raw data into CSVs)
```

---

## Script 00 — `00_unzip_accum.py`

### What problem does it solve?
The CDS (Copernicus Data Store) API sometimes delivers downloaded files as **ZIP archives disguised with a `.nc` extension**. The file looks like `era5_TN_grid_2024_01_accum.nc` but is actually a ZIP containing the real NetCDF file inside. This causes every subsequent script to fail with `Unknown file format`.

### What it does
1. Scans every `*_accum.nc` file in `data/raw/era5/grid/`
2. Checks the first 2 bytes of each file — ZIP files always start with `PK` (the ZIP magic bytes)
3. If it finds a disguised ZIP, it extracts the real `.nc` NetCDF file from inside
4. Replaces the fake file with the real one in-place
5. Skips files that are already valid NetCDF

### When to run it
Only once, right after `01_download_era5_tamilnadu.py` finishes downloading the accum files. If you re-download, run it again.

### Output
No new files — it fixes the existing accum `.nc` files in place.

---

## Script 01 — `01_download_era5_tamilnadu.py`

### What is ERA5?
ERA5 is the world's most widely used atmospheric reanalysis dataset, produced by ECMWF. It provides **hourly estimates of atmospheric variables** (temperature, wind, solar radiation, precipitation etc.) on a global 0.25° × 0.25° grid (~25 km resolution) from 1940 to present. It is derived by combining millions of weather observations (satellites, weather stations, radiosondes, ships) with a numerical weather model. It is not raw sensor data — it is a physically consistent reconstruction of the atmosphere.

### The core insight — download a grid, not individual cities
Previous approaches downloaded data separately for each city — that meant 76 cities × 12 months × 2 years = **1,824 API calls**, taking days. This script downloads the **entire Tamil Nadu state as a single bounding box** — only **48 API calls total** (2 years × 12 months × 2 variable types). All 222 cities are then extracted from this single grid in script 02.

### The bounding box
```
North: 13.75°N    (above Chennai/Tiruvallur)
South:  7.75°N    (below Kanyakumari)
West:  75.75°E    (beyond Nilgiris/Gudalur)
East:  81.25°E    (beyond Chennai coast)
```
This box covers the entire state of Tamil Nadu with a small buffer on all sides.

### Why two separate API calls per month?
ERA5 has two fundamentally different variable types that **cannot be mixed in one request**:

| Type | ERA5 name | Variables downloaded | Physical meaning |
|------|-----------|---------------------|------------------|
| **Instant (AN)** | Analysis | Temperature, dewpoint, wind U, wind V, cloud cover, pressure | Snapshot of the atmosphere at each hour |
| **Accumulated (FC)** | Forecast | Solar radiation (GHI), thermal radiation, precipitation, mean DNI | Energy/mass that has accumulated since the start of a forecast run |

Mixing them in one request causes MARS (the ECMWF retrieval system) to reject the query. So each month requires exactly 2 API calls.

### Variables downloaded

**Instant variables** (saved as `era5_TN_grid_{year}_{month}_instant.nc`):

| ERA5 name | Physical variable | Unit | Used for |
|-----------|------------------|------|----------|
| `t2m` | 2m air temperature | K | T_amb (°C) |
| `d2m` | 2m dewpoint temperature | K | T_dew → relative humidity |
| `u10` | 10m eastward wind | m/s | Wind speed + direction |
| `v10` | 10m northward wind | m/s | Wind speed + direction |
| `tcc` | Total cloud cover | 0–1 | Cloud fraction |
| `sp` | Surface pressure | Pa | P_atm (hPa) |

**Accumulated variables** (saved as `era5_TN_grid_{year}_{month}_accum.nc`):

| ERA5 name | Physical variable | Unit | Used for |
|-----------|------------------|------|----------|
| `ssrd` | Surface solar radiation downwards | J/m² | GHI (W/m²) |
| `msdwswrf` | Mean surface direct short-wave radiation flux | W/m² | DNI (W/m²) |
| `strd` | Surface thermal radiation downwards | J/m² | LW_down (W/m²) |
| `tp` | Total precipitation | m | precipitation (mm) |

**Note:** `surface_diffuse_solar_radiation_downwards` does NOT exist in ERA5. DHI (Diffuse Horizontal Irradiance) is computed in script 02 as: `DHI = GHI - DNI × cos(SZA)`.

### File naming
```
data/raw/era5/grid/
  era5_TN_grid_2024_01_instant.nc   (Jan 2024, temperature/wind/pressure)
  era5_TN_grid_2024_01_accum.nc     (Jan 2024, solar/rain)
  era5_TN_grid_2024_02_instant.nc
  era5_TN_grid_2024_02_accum.nc
  ... (48 files total: 2 years × 12 months × 2 types)
```

### Smart resume
Every completed download is logged in `data/raw/era5/download_status.csv`. If the script is interrupted, re-running it skips already-downloaded files automatically.

---

## Script 02 — `02_combine_tamilnadu.py`

This is the main processing script. It reads the 48 raw NetCDF files and produces clean CSVs. It does two completely separate things:

### Part 1 — Extract 222 named locations

For each of the 222 cities/towns/taluks in `TN_LOCATIONS`, the script:

**Step 1: Find the nearest ERA5 grid point**
Each location has an exact latitude/longitude (e.g. Ettimadai: 10.9282°N, 76.8780°E). ERA5's grid is at 0.25° intervals (10.75°, 11.00°, 11.25° etc.). The script finds the closest grid point using `argmin(|lat_ERA5 - lat_location|)`. The output CSV stores both the exact location coordinates (`lat`, `lon`) and the ERA5 grid point used (`grid_lat`, `grid_lon`).

**Step 2: De-accumulate solar radiation**
ERA5 stores solar radiation as a **running total** that resets every 12 hours (at 00 UTC and 12 UTC forecast starts). To get the hourly value you need to take the difference between consecutive hours. At reset hours (hour 1 and hour 13 UTC), the raw value itself is the hourly increment. The `deaccumulate()` function handles this correctly and converts from J/m² to W/m² by dividing by 3600.

**Step 3: Compute derived variables**
Using `pvlib` (a solar energy library):
- **SZA** — Solar Zenith Angle (degrees from vertical)
- **solar_azimuth** — compass direction of the sun
- **ETR** — Extraterrestrial Radiation (solar power above atmosphere)
- **GHI_clearsky** — what GHI would be on a perfectly clear day (Ineichen model)
- **CSI** — Clear Sky Index = GHI / GHI_clearsky (1.0 = perfect clear sky)
- **DNI** — Direct Normal Irradiance (from msdwswrf, or derived from GHI/cos(SZA))
- **DHI** — Diffuse Horizontal Irradiance = GHI - DNI × cos(SZA)
- **sunrise_hour / sunset_hour** — daily sunrise and sunset times in UTC

**Step 4: Compute RRTDHS**
RRTDHS (Relative Resource-Temperature-Duration-High-Solar) is a monthly composite score that measures solar resource quality accounting for:
- Mean GHI of the month
- How close the ambient temperature is to the panel's optimal operating temperature
- Fraction of daylight hours with significant solar radiation

**Step 5: Add metadata**
Each row gets: `city`, `district`, `climate_zone`, `altitude_m`, `T_set`, `high_solar_resource` flag.

**Step 6: Save**
- Individual CSV: `data/processed/by_location/climate_{city}.csv`
- All locations combined: written location by location to avoid memory overflow

### Part 2 — Full ERA5 grid CSV

Extracts every single ERA5 grid point inside the bounding box — **25 latitude × 23 longitude = 575 grid points**. For each point it runs the same unit conversions and solar computations (except sunrise/sunset, which is too slow for 575 × 730 days). This produces a spatially complete dataset where every 0.25° cell across Tamil Nadu has hourly data.

---

## Complete Output Structure

```
data/
├── raw/
│   └── era5/
│       └── grid/
│           ├── era5_TN_grid_2024_01_instant.nc   ← raw ERA5 (48 files)
│           ├── era5_TN_grid_2024_01_accum.nc
│           └── ...
│
└── processed/
    ├── by_location/
    │   ├── climate_chennai.csv              ← 17,544 rows
    │   ├── climate_coimbatore.csv           ← 17,544 rows
    │   ├── climate_ettimadai.csv            ← 17,544 rows
    │   └── ... (222 files, one per location)
    │
    ├── grid/
    │   └── era5_TN_grid_all.csv             ← 10,087,800 rows (575 grid points)
    │
    └── climate_tamilnadu_all.csv            ← 3,894,768 rows (222 locations combined)
```

---

## Columns in Every Output CSV

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp` | UTC datetime | Hourly timestamp |
| `GHI` | W/m² | Global Horizontal Irradiance (total solar on flat surface) |
| `DNI` | W/m² | Direct Normal Irradiance (beam radiation) |
| `DHI` | W/m² | Diffuse Horizontal Irradiance (scattered sky radiation) |
| `avg_sdirswrf` | W/m² | ERA5 mean direct shortwave flux (source of DNI) |
| `LW_down` | W/m² | Downward longwave (thermal) radiation |
| `T_amb` | °C | Air temperature at 2m |
| `T_dew` | °C | Dewpoint temperature at 2m |
| `RHum` | % | Relative humidity (derived from T_amb and T_dew) |
| `W_spd` | m/s | Wind speed at 10m |
| `W_dir` | degrees | Wind direction (meteorological convention) |
| `P_atm` | hPa | Atmospheric pressure at surface |
| `cloud_cover` | 0–1 | Total cloud cover fraction |
| `precipitation` | mm | Hourly precipitation |
| `SZA` | degrees | Solar Zenith Angle |
| `solar_azimuth` | degrees | Solar azimuth angle |
| `ETR` | W/m² | Extraterrestrial radiation |
| `GHI_clearsky` | W/m² | Clear-sky GHI (Ineichen model) |
| `CSI` | 0–1.5 | Clear Sky Index |
| `RRTDHS` | score | Monthly solar resource quality score |
| `sunrise_hour` | UTC hour | Daily sunrise time |
| `sunset_hour` | UTC hour | Daily sunset time |
| `hour` | 0–23 | Hour of day (UTC) |
| `month` | 1–12 | Month |
| `DOY` | 1–366 | Day of year |
| `year` | 2024/2025 | Year |
| `season` | text | Winter/Summer/Monsoon/Retreat |
| `season_code` | 1–4 | Numeric season code |
| `city` | text | Location name |
| `lat` | degrees | Exact location latitude |
| `lon` | degrees | Exact location longitude |
| `grid_lat` | degrees | Nearest ERA5 grid point latitude |
| `grid_lon` | degrees | Nearest ERA5 grid point longitude |
| `altitude_m` | metres | Elevation above sea level |
| `district` | text | Tamil Nadu district |
| `climate_zone` | text | Climate classification |
| `T_set` | °C | Panel optimal temperature for RRTDHS |
| `high_solar_resource` | 0/1 | 1 if RRTDHS > 0.5 |

---

## Coverage — All 222 Named Locations

Locations span all **38 districts** of Tamil Nadu across **8 climate zones**:

| Climate Zone | Description | Example locations |
|---|---|---|
| `hot-humid-coastal` | Hot, humid, sea breeze | Chennai, Nagapattinam, Kanyakumari |
| `hot-humid` | Hot and humid, inland | Coimbatore, Thanjavur, Tenkasi |
| `hot-semi-arid` | Hot, moderate rainfall | Madurai, Virudhunagar, Sivakasi |
| `semi-arid` | Moderate temp, low rain | Salem, Vellore, Tiruchirappalli |
| `hot-arid-coastal` | Dry coastal | Rameswaram, Thoothukudi, Ramanathapuram |
| `hot-arid` | Dry inland | Paramakudi |
| `cool-hilly` | Western Ghats highlands | Ooty, Kodaikanal, Valparai, Yercaud |
| `semi-arid-elevated` | High plateau | Hosur, Denkanikottai |
| `hot-humid-elevated` | Humid foothills | Ettimadai |

---

## Data Scale Summary

| Dataset | Locations | Rows | Time span |
|---------|-----------|------|-----------|
| Per-location CSVs | 222 | 17,544 each | Jan 2024 – Dec 2025 |
| `climate_tamilnadu_all.csv` | 222 combined | 3,894,768 | Jan 2024 – Dec 2025 |
| `era5_TN_grid_all.csv` | 575 grid points | 10,087,800 | Jan 2024 – Dec 2025 |

---

## How to Run (Full Pipeline)

```powershell
# Step 1: Download raw ERA5 data (takes ~4–6 hours, 48 API calls)
python 01_download_era5_tamilnadu.py 2>&1 | Tee-Object -FilePath download_log.txt

# Step 2: Fix the accum files (CDS API delivers them as hidden ZIPs)
python 00_unzip_accum.py

# Step 3: Process everything into CSVs (takes ~30–90 minutes)
python 02_combine_tamilnadu.py
```

If step 1 is interrupted, just re-run it — completed files are skipped automatically.

---

## Requirements

```
pip install cdsapi xarray netCDF4 pvlib pandas numpy scipy
```

You also need a free CDS account and API key from https://cds.climate.copernicus.eu

---

## Key Design Decisions

**Why one grid download instead of per-city downloads?**
Downloading the entire TN bounding box takes 48 API calls. Downloading each city separately would take 1,800+ calls and days of queuing time on the CDS servers. The grid approach is 40× faster and produces identical data.

**Why does ERA5 resolution matter?**
ERA5's 0.25° grid means each grid cell is roughly 25 km × 25 km. Two cities within the same 25 km cell (e.g. Coimbatore city and Sulur) will share the same ERA5 values — the ERA5 cannot distinguish them. The `grid_lat`/`grid_lon` columns in the CSV tell you exactly which ERA5 cell was used for each location.

**Why are the accum files smaller than instant files?**
The accumulated variables (GHI, precipitation) compress better than temperature/wind fields because large portions of the night have zero solar radiation, making the data very compressible.

**Why does the accum de-accumulation matter?**
If you use the raw `ssrd` values directly, you get a continuously increasing number that resets twice a day. After de-accumulation and dividing by 3600, you get the correct W/m² for each hour. Getting this wrong produces wildly incorrect GHI values.