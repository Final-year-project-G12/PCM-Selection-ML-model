# data_3 — ERA5 Climate Data Pipeline for PCM Selection

This folder contains the **complete climate data pipeline** used to build the ML-ready dataset for PCM (Phase Change Material) selection. It downloads raw ERA5 reanalysis data, engineers all derived climate features, and produces a clean, scaled feature matrix ready for model training.

---

## 📁 Directory Structure

```
data_3/
├── 01_download_era5.py          ← Step 1: Download raw ERA5 NetCDF files
├── 02_combine.py                ← Step 2: Combine NetCDF → clean hourly CSV + feature engineering
├── 03_preprocess_validate.py    ← Step 3: Full preprocessing, validation & early fusion (monolithic)
│
├── preprocessing/               ← Modular version of Step 3 (one file per stage)
│   ├── 01_data_cleaning.py      ← Step A: Outlier removal & interpolation
│   ├── 02_feature_engineering.py← Step A2: Cyclical encodings, lags, rolling stats
│   ├── 03_join_validation.py    ← Step B: Spatial & temporal join validation
│   ├── 04_normalisation.py      ← Step C: MinMax / Standard / Robust scaling
│   ├── 05_data_validation_report.py ← Step D: QC report + plots
│   └── 06_early_fusion.py       ← Step E: Assemble final feature matrix X
│
└── data/
    ├── raw/era5/                ← Raw NetCDF files downloaded by Step 1
    ├── processed/               ← Intermediate and final CSVs + scaler .pkl files
    └── validation/              ← QC report CSV + PNG plots
```

---

## 🔄 Pipeline Overview

```
ERA5 API
   │
   ▼
01_download_era5.py  →  data/raw/era5/*.nc
   │
   ▼
02_combine.py        →  data/processed/climate_<city>.csv
                         data/processed/climate_all_cities.csv
   │
   ▼
03_preprocess_validate.py  →  data/processed/climate_all_cities_preprocessed.csv
                               data/processed/scaler_minmax.pkl
                               data/processed/scaler_standard.pkl
                               data/validation/qc_report.csv
                               data/validation/correlation_matrix.png
                               data/validation/distributions.png
                               data/validation/temporal_heatmap_*.png
```

> **Run order:** `01_download_era5.py` → `02_combine.py` → `03_preprocess_validate.py`  
> Or use the modular `preprocessing/` scripts for finer control (see below).

---

## 🧭 Cities Covered

| City | Latitude | Longitude | Altitude | Climate Zone | T_set (°C) |
|------|----------|-----------|----------|--------------|-----------|
| Coimbatore | 11.0 | 77.0 | 411 m | hot-humid | 45 |
| Jaisalmer | 26.9 | 70.9 | 225 m | hot-arid | 55 |

`T_set` is the target water temperature for the solar water heater (SWH). A higher value is used for Jaisalmer because of its high irradiance (Kou 2025).

---

## 📄 Script-by-Script Explanation

---

### `01_download_era5.py` — ERA5 Data Download

**Purpose:** Downloads hourly ERA5 reanalysis data for each city and each month, saving one `.nc` (NetCDF) file per city-year-month combination.

**How it works:**
1. Defines a bounding box (`area`) for each city: `[lat_max, lon_min, lat_min, lon_max]`.
2. Loops over cities × years × months.
3. Skips already-downloaded files to allow resuming interrupted downloads.
4. Calls the **Copernicus CDS API** (`cdsapi`) to retrieve the `reanalysis-era5-single-levels` product.

**ERA5 variables downloaded:**

| Variable | Description |
|---|---|
| `2m_temperature` | Ambient temperature at 2 m height (K) |
| `2m_dewpoint_temperature` | Dew-point temperature at 2 m (K) |
| `surface_solar_radiation_downwards` | Cumulative GHI (J/m²) |
| `mean_surface_direct_short_wave_radiation_flux` | Direct Normal Irradiance mean flux (W/m²) |
| `surface_thermal_radiation_downwards` | Downwelling longwave radiation (J/m²) |
| `10m_u_component_of_wind` | East-West wind component at 10 m (m/s) |
| `10m_v_component_of_wind` | North-South wind component at 10 m (m/s) |
| `total_precipitation` | Total precipitation (m) |
| `total_cloud_cover` | Fractional cloud cover (0–1) |
| `surface_pressure` | Atmospheric pressure at surface (Pa) |

**Output:** `data/raw/era5/era5_{City}_{year}_{month}.nc`

**Requirements:** `pip install cdsapi`  
**Note:** A valid `~/.cdsapirc` credentials file is required (Copernicus account).

---

### `02_combine.py` — NetCDF → CSV + Feature Engineering

**Purpose:** Opens every raw `.nc` file for each city, spatially averages across the ERA5 grid cells inside the bounding box (reducing to a single point), concatenates months, and computes all derived climate features.

#### Helper functions

| Function | What it does |
|---|---|
| `kelvin_to_celsius(T_K)` | Subtracts 273.15 to convert Kelvin → °C |
| `magnus_rh(T_amb_C, T_dew_C)` | Magnus formula: computes relative humidity (%) from ambient and dew-point temperatures |
| `wind_speed(u, v)` | Pythagorean magnitude: `√(u² + v²)` |
| `accumulated_to_flux(da)` | Divides accumulated J/m² by 3600 → W/m² (ERA5 accumulates radiation hourly from midnight) |
| `get_season(month_series)` | Maps month → Indian season: Winter (Dec–Feb), Pre-monsoon (Mar–May), Monsoon (Jun–Sep), Post-monsoon (Oct–Nov) |
| `compute_solar_features(df, lat, lon, altitude_m)` | Uses **pvlib** to compute Solar Zenith Angle (SZA), Extraterrestrial Radiation (ETR), clear-sky GHI (Ineichen model with Linke turbidity), and Clear Sky Index (CSI = GHI / GHI_clearsky) |
| `compute_dhi(ghi, dni, sza_deg)` | Diffuse Horizontal Irradiance: `DHI = GHI − DNI × cos(SZA)` |
| `compute_rrtdhs(df, T_set)` | RRTDHS = Q_sol_ave / (T_set − T_out_ave), computed monthly. This climate index (Kou 2025) drives PCM optimal melting temperature selection. Values > 5.7 indicate a high solar resource. |
| `compute_sunrise_sunset(df, lat, lon, altitude_m)` | Uses pvlib's SPA algorithm to compute daily sunrise/sunset hours. |
| `unzip_if_needed(filepath)` | Handles cases where CDS returns a ZIP-wrapped NetCDF. Extracts the inner `.nc` files and renames them. |
| `process_city(city_name, config)` | Main per-city processing function. Opens all `.nc` files, spatially averages, merges time-steps, detects ERA5 column names dynamically (handles `t2m`/`2m_temperature` variants), applies all feature computations, drops raw ERA5 columns, and returns a clean DataFrame. |

#### Derived features produced

| Feature | Formula / Source |
|---|---|
| `T_amb` | 2 m temperature − 273.15 (°C) |
| `T_dew` | Dew-point temperature − 273.15 (°C) |
| `RHum` | Magnus formula (%) |
| `W_spd` | √(u10² + v10²) (m/s) |
| `W_dir` | `arctan2(u, v)` × 180/π mod 360 (degrees from North) |
| `GHI` | ssrd / 3600 (W/m²) |
| `DNI` | fdir (already W/m² for mean flux) |
| `LW_down` | strd / 3600 (W/m²) |
| `cloud_cover` | tcc clipped to [0, 1] |
| `precipitation` | tp × 1000 (m → mm) |
| `P_atm` | sp / 100 (Pa → hPa) |
| `SZA` | Solar zenith angle (degrees) |
| `solar_azimuth` | Solar azimuth angle (degrees) |
| `ETR` | Extraterrestrial radiation (W/m²) |
| `GHI_clearsky` | Ineichen clear-sky GHI (W/m²) |
| `CSI` | GHI / GHI_clearsky, clipped to [0, 1.5] |
| `DHI` | GHI − DNI × cos(SZA) (W/m²) |
| `RRTDHS` | Monthly solar resource intensity index (Kou 2025) |
| `season` | Indian season name (string) |
| `season_code` | 0=winter, 1=pre-monsoon, 2=monsoon, 3=post-monsoon |
| `hour`, `month`, `DOY`, `year` | Time components |
| `sunrise_hour`, `sunset_hour` | Fractional hours (local time) |
| `high_solar_resource` | Binary flag: 1 if RRTDHS > 5.7 |

**Output:**
- `data/processed/climate_<city>.csv` — per-city hourly data
- `data/processed/climate_all_cities.csv` — combined dataset

**Requirements:** `pip install xarray netCDF4 pvlib pandas numpy scipy`

---

### `03_preprocess_validate.py` — Full Preprocessing & Validation (Monolithic)

**Purpose:** Reads the combined CSV from Step 2 and runs the complete preprocessing pipeline in a single script. This is the "all-in-one" version; the `preprocessing/` folder contains the same logic broken into smaller files.

The script is divided into five labelled stages:

#### Stage A — `step_A_clean()` — Data Cleaning

1. **Parse & sort timestamps** — ensures the time series is in chronological order.
2. **Remove duplicate timestamps** — ERA5 monthly files can overlap at boundaries; duplicates are identified per `(timestamp, city)` and dropped (keeping first).
3. **Physical bounds → NaN** — values outside the `PHYSICAL_BOUNDS` dictionary (e.g., GHI > 1400 W/m²) are treated as sensor artifacts and replaced with `NaN`.
4. **Per-city linear interpolation** — fills NaN values using `interpolate(method='linear', limit=6)`, meaning up to 6 consecutive missing hours are filled. Larger gaps remain NaN.
5. **Forward/backward fill** — `ffill()` then `bfill()` handles any remaining NaN at series edges.

#### Stage A2 — `step_A2_feature_engineering()` — Engineered Features

Generates additional features if not already present:

| Feature type | Features created |
|---|---|
| **Cyclical encodings** | `sin_hour`, `cos_hour`, `sin_month`, `cos_month`, `sin_DOY`, `cos_DOY` — maps circular time features onto a unit circle so ML models understand period boundaries (e.g., hour 23 ≈ hour 0) |
| **Lag features** | `GHI_lag1`, `GHI_lag2`, `GHI_lag3`, `GHI_lag6`, `GHI_lag12`, `GHI_lag24` and same for `T_amb` — previous values in time series |
| **Rolling statistics** | `GHI_roll3h_mean`, `GHI_roll6h_mean`, `GHI_roll24h_std` — short-term trend and variability |
| **Derived thermal** | `T_depression = T_amb − T_dew` (dryness index), `GHI_clearsky_diff = GHI − GHI_clearsky`, `T_pcm_delta = T_amb − T_set` (how far ambient is from PCM setpoint) |

#### Stage B — `step_B_validate_joins()` — Spatial & Temporal Join Validation

**Spatial validation:** Checks that every city has exactly one unique (lat, lon) pair — confirming that the IDW/spatial averaging in Step 2 worked correctly and there is no mixing of grid cells.

**Temporal validation:** Computes the time difference between consecutive rows per city. Any gap > 1 hour is flagged and catalogued. Reports:
- Date range covered
- Total rows vs. expected rows (if perfectly regular)
- Number and location of gaps

#### Stage C — `step_C_normalise()` — Normalisation

Three scaling strategies are applied depending on the nature of each feature:

| Scaler | Formula | Applied to | Reason |
|---|---|---|---|
| **MinMaxScaler** | `(X − min) / (max − min)` → [0, 1] | GHI, DNI, DHI, CSI, ETR, SZA, RHum, cloud_cover, etc. | Bounded physical quantities; neural networks prefer [0, 1] inputs |
| **StandardScaler** | `(X − μ) / σ` → ~N(0, 1) | T_amb, T_dew, T_depression, T_pcm_delta, P_atm, RRTDHS, lag features, rolling stats | Unbounded or near-Gaussian variables; required for gradient-based models |
| **RobustScaler** | `(X − median) / IQR` | precipitation, W_dir | Highly skewed / outlier-prone; median is more robust than mean |

**Not scaled:** Cyclical features (`sin_*`, `cos_*`), binary flags (`high_solar_resource`), integer time indices (`hour`, `month`, `DOY`, `year`, `season_code`), and fractional hour features (`sunrise_hour`, `sunset_hour`).

Fitted scaler objects are saved as `scaler_minmax.pkl`, `scaler_standard.pkl`, and `scaler_robust.pkl` so that predictions can be inverse-transformed later.

#### Stage D — `step_D_validate()` — Data Validation Report

Generates a full QC (Quality Control) report including:

| Sub-step | Output |
|---|---|
| D1 | Missing data % per column (printed to console) |
| D2 | Physical range statistics — min, max, mean, std, outlier count per feature |
| D3 | Temporal gap details (if any) |
| D4 | **Correlation matrix heatmap** → `validation/correlation_matrix.png` |
| D5 | **Feature distribution histograms** with mean/median lines → `validation/distributions.png` |
| D6 | **GHI temporal coverage heatmap** (hours × days) → `validation/temporal_heatmap_{city}.png` |
| D7 | **QC summary CSV** → `validation/qc_report.csv` |

The QC report CSV contains one row per city with completeness, range, spatial and temporal metadata (Mansouri 2025 §VII.B benchmark protocol).

#### Stage E — `step_E_early_fusion()` — Final Feature Matrix

Assembles all feature groups into a single feature matrix **X** (early fusion, Paper §IV.A):

| Group | Features included |
|---|---|
| `solar_sensor` | GHI, DNI, DHI, LW_down, GHI_clearsky, CSI, ETR, SZA, solar_azimuth, sunrise_hour, sunset_hour |
| `weather_nwp` | T_amb, T_dew, RHum, W_spd, W_dir, cloud_cover, precipitation, P_atm, RRTDHS |
| `pcm_thermal` | T_set, T_pcm_delta, T_depression |
| `time_features` | hour, month, DOY, year, season_code, sin/cos hour/month/DOY, high_solar_resource |
| `lag_features` | All `*_lag*` columns (auto-detected) |
| `rolling_features` | All `*_roll*` columns (auto-detected) |

Rows with NaN in any feature column (created at the start of the series by lag shifting) are dropped. Metadata columns (`timestamp`, `city`, `lat`, `lon`, `altitude_m`, `climate_zone`, `season`) are kept alongside X for traceability.

**Final output:** `data/processed/climate_all_cities_preprocessed.csv`

---

## 📁 `preprocessing/` — Modular Scripts

The `preprocessing/` folder contains the exact same logic as `03_preprocess_validate.py`, split into six standalone scripts. Each reads from and writes to `../data_2/` (note the path difference). Use these for debugging individual stages.

| Script | Stage | Input | Output |
|---|---|---|---|
| `01_data_cleaning.py` | A | `climate_all_cities.csv` | `climate_cleaned.csv` |
| `02_feature_engineering.py` | A2 | `climate_cleaned.csv` | `climate_features.csv` |
| `03_join_validation.py` | B | `climate_features.csv` | `validation/temporal_gaps.csv` |
| `04_normalisation.py` | C | `climate_features.csv` | `climate_scaled.csv` + scaler `.pkl` files |
| `05_data_validation_report.py` | D | `climate_features.csv` + `climate_scaled.csv` | `validation/correlation_matrix.png`, `distributions.png`, `temporal_heatmap_*.png`, `qc_report.csv` |
| `06_early_fusion.py` | E | `climate_scaled.csv` | `climate_all_cities_preprocessed.csv` |

**Run order:**
```bash
cd preprocessing/
python 01_data_cleaning.py
python 02_feature_engineering.py
python 03_join_validation.py
python 04_normalisation.py
python 05_data_validation_report.py
python 06_early_fusion.py
```

---

## ⚙️ Requirements

```bash
pip install cdsapi xarray netCDF4 pvlib pandas numpy scipy \
            scikit-learn matplotlib seaborn joblib
```

A Copernicus CDS API key is required for `01_download_era5.py`. Set up `~/.cdsapirc` following the [official guide](https://cds.climate.copernicus.eu/api-how-to).

---

## 📐 Physical Bounds Reference

Used by cleaning and validation stages to flag outliers:

| Variable | Min | Max | Unit |
|---|---|---|---|
| GHI | 0 | 1400 | W/m² |
| DNI | 0 | 1200 | W/m² |
| DHI | 0 | 800 | W/m² |
| LW_down | 100 | 600 | W/m² |
| T_amb | −10 | 55 | °C |
| T_dew | −20 | 40 | °C |
| RHum | 0 | 100 | % |
| W_spd | 0 | 30 | m/s |
| W_dir | 0 | 360 | ° |
| precipitation | 0 | 200 | mm |
| cloud_cover | 0 | 1 | fraction |
| P_atm | 800 | 1100 | hPa |
| CSI | 0 | 1.5 | — |
| SZA | 0 | 180 | ° |
| ETR | 0 | 1500 | W/m² |
| GHI_clearsky | 0 | 1400 | W/m² |

---

## 📚 Paper References

| Code symbol | Reference |
|---|---|
| CSI, SZA, ETR | Ghodusinejad 2026 |
| GHI, DNI, DHI, RHum, W_spd | Barqawi 2025 |
| RRTDHS, T_set | Kou 2025 |
| Cloud cover, demand | Odoi-Yorke 2025 |
| Flat-plate collector model | Chen 2025 |
| QC benchmark protocol | Mansouri 2025 §VII.B |
