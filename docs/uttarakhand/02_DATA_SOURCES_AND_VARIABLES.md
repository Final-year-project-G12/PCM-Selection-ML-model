# 02 — Data Sources and Variables

All entries below are taken from the Uttarakhand scripts themselves
(`01_download_era5_uttarakhand.py`, `01b_download_nasapower.py`, `00a_build_population_grid.py`,
`02_combine_uttarakhand.py`, `02b_build_daily_aggregates.py`, `04_preprocess_uttarakhand.py`,
`04b_climate_signature.py`, `06_build_pcm_database.py`) or from the committed data artefacts.

## Primary data sources

### ERA5 (ECMWF Reanalysis v5)

- **Provider**: Copernicus Climate Data Store (CDS), accessed via `cdsapi`
- **Product**: `reanalysis-era5-single-levels`, hourly, `product_type: ["reanalysis"]`
- **Format requested**: `data_format: "netcdf"`, `download_format: "unarchived"`
- **Spatial extent**: the bounding envelope of `population_grid_points.csv`, padded **0.5°**
  (`load_points_bbox(pad=0.5)`) — *not* the whole state
- **Temporal coverage**: 2016–2025 (10 full calendar years), all days
- **Hours requested**: computed dynamically from `suntimes.csv`, not fixed clock hours — three
  circular (mod-24) windows around sunrise / solar noon / sunset, each padded `HOUR_MARGIN = 1`
- **Call structure**: 10 years × 12 months × 2 variable types = **240 API calls**
- **Output naming**: `data/raw/era5/points/era5_UK_points_{year}_{month}_{instant|accum}.nc`
- **Status tracking**: `data/raw/era5/download_status_points.csv` (fields: timestamp, year, month,
  var_type, status, filepath, size_mb, note). Retry policy `MAX_RETRIES = 3`, `RETRY_WAIT = 60 s`.
  A file under 50,000 bytes is treated as a corrupt download and removed.
- **Separation from an older archive**: `config.py` keeps `RAW_POINTS_DIR` /
  `POINTS_DOWNLOAD_STATUS_FILE` distinct from `RAW_GRID_DIR` / `DOWNLOAD_STATUS_FILE`; the
  docstring states the new pipeline "**does not touch** the old `data/raw/era5/grid/` archive".

**ERA5 variables — INSTANT group** (`INSTANT_VARS`, analysis fields):

| CDS variable | Short name | Unit as delivered | Converted to |
|---|---|---|---|
| `2m_temperature` | `t2m` | K | `era5_T_amb` (°C) |
| `2m_dewpoint_temperature` | `d2m` | K | `era5_T_dew` (°C), and `era5_RHum` via Magnus |
| `10m_u_component_of_wind` | `u10` | m/s | `era5_W_spd`, `era5_W_dir` |
| `10m_v_component_of_wind` | `v10` | m/s | `era5_W_spd`, `era5_W_dir` |
| `total_cloud_cover` | `tcc` | 0–1 fraction | `era5_cloud_cover` |
| `surface_pressure` | `sp` | Pa | `era5_P_atm` (hPa) |

**ERA5 variables — ACCUM group** (`ACCUM_VARS`, forecast fields):

| CDS variable | Short name | Unit as delivered | Converted to |
|---|---|---|---|
| `surface_solar_radiation_downwards` | `ssrd` | J/m² (accumulated) | `era5_GHI` (W/m²) |
| `mean_surface_direct_short_wave_radiation_flux` | `msdwswrf` | W/m² (mean rate) | `era5_DNI` (W/m²) |
| `surface_thermal_radiation_downwards` | `strd` | J/m² (accumulated) | `era5_LW_down` (W/m²) |
| `total_precipitation` | `tp` | m (accumulated) | `era5_precipitation` (mm) |

The accum request additionally downloads every target hour's **immediate predecessor**:
`ACCUM_HOURS = INSTANT_HOURS union {(h - 1) mod 24 for h in INSTANT_HOURS}`, because
`deaccumulate()` in `02_combine_uttarakhand.py` recovers hourly flux by `diff()`.

### NASA POWER

- **Provider**: NASA Langley Research Center, Prediction Of Worldwide Energy Resources
- **Endpoint**: `https://power.larc.nasa.gov/api/temporal/hourly/point` — **hourly** point data,
  no API key required
- **Community**: `RE`
- **Time standard requested**: `UTC`
- **Parameters** (`POWER_PARAMETERS`): `ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M`
- **Coverage**: 2016–2025, one JSON per point per year -> 45 × 10 = **450 point-year caches**
- **Output naming**: `data/raw/nasapower/power_{point_id}_{year}.json`
- **Status tracking**: `data/raw/nasapower/download_status_power.csv`. `MAX_RETRIES = 3`,
  `RETRY_WAIT = 20 s`, `REQUEST_SLEEP = 1.0 s` between successful calls, `REQUEST_TIMEOUT = 60 s`.
  Files under 1,000 bytes are treated as corrupt.
- **Fill value handling**: `-999` is replaced with `NaN` in both `02_combine_uttarakhand.py` and
  `02b_build_daily_aggregates.py`.
- **Dual role**: only 3 of the ~8,760 hours/year are consumed by `02`'s sun-event merge; the full
  cache is re-read by `02b_build_daily_aggregates.py` to build true daily integrals.

> **Stated limitation, in `01b`/`02b`/`04b` docstrings and `README.md`:** `POWER_PARAMETERS` does
> **not** include precipitation (`PRECTOTCORR`). `monsoon_index` therefore remains an ERA5
> 3×/day proxy and never receives a Tier-2 "true" version.

### Population raster

- **Source**: WorldPop unconstrained global mosaic, India, UN-adjusted, 100 m, **2020**
- **URL**: `https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_UNadj.tif`
- **Size**: ~1.5–2 GB, one-time, cached in `data/raw/population/`. Download auto-retries up to 5
  attempts and resumes via HTTP `Range` requests.
- **Stated assumption** (`00a` docstring): "WorldPop doesn't publish a distinct India raster per
  year at this resolution, so this pipeline uses a single static 2020 snapshot to weight sampling
  locations across the whole 2016-2025 study period. That's a standard simplifying assumption …
  not something this script tries to correct for."
- **Nodata handling**: `rio_mask(..., nodata=0, filled=True)` then `band[band < 0] = 0.0` —
  WorldPop's negative nodata sentinels are zeroed.

### State boundary

- **Source**: GADM v4.1, India administrative level 1, GeoJSON
- **URL**: `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json`
- **Filter**: `NAME_1 == "Uttarakhand"`, `uk.geometry.iloc[0]` (first matching geometry)
- **Failure mode**: raises with the full list of available `NAME_1` values if not found

### PCM property data

- **Raw input**: `PCM_data/PCM_data/data/PCM_Properties_55records_42_70C_dense.csv` (55 records)
- **Cleaned output** consumed by the pipeline:
  `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` (55 rows × 59 columns)
- **Cleaning method**: MICE + Random-Forest + Predictive Mean Matching
  (`PCM_data/PCM_data/01_preprocess.py`, `N_ITER = 8`, `N_DONORS = 3`, `RANDOM_STATE = 42`)
- See `07_PHASE_5_AUDIT.md` for the full composition and imputation audit.

## Sampling grid

| Parameter | Value | Source |
|---|---|---|
| Grid resolution | `GRID_RES = 0.25°` | `00a_build_population_grid.py` |
| Grid origin | `ERA5_ORIGIN_LAT = 90.0`, `ERA5_ORIGIN_LON = -180.0` | `00a` |
| Coverage target | `COVERAGE_TARGET = 0.875` | `00a` |
| Points selected | **45** | observed: 45 markers/popups in `data/plots/comprehensive/maps/A2_population_map.html` |
| Point-ID format | `UKP_0001` … `UKP_0045` (contiguous, no gaps) | observed |
| Latitude range | 28.875° N – 30.625° N | observed from map marker coordinates |
| Longitude range | 77.875° E – 80.125° E | observed |
| Population covered | **10,475,711** (sum of the 45 cells' WorldPop 2020 values) | observed from map popups |
| Largest cell | `UKP_0001` = 1,061,041 | observed |
| Smallest cell | `UKP_0045` = 85,265 | observed |

The columns written are `point_id, lat, lon, population, weight`, with `weight` renormalised over
the selected subset only.

## Derived variables computed in `02_combine_uttarakhand.py`

| Output column | Derivation | Notes |
|---|---|---|
| `era5_T_amb` | `t2m - 273.15` | values `< -5` or `> 60` set to `NaN` in this script |
| `era5_T_dew` | `d2m - 273.15` | |
| `era5_RHum` | Magnus: `100·exp(17.625·Td/(243.04+Td)) / exp(17.625·T/(243.04+T))`, clipped 0–100 | |
| `era5_W_spd` | `sqrt(u10² + v10²)` | m/s |
| `era5_W_dir` | `(degrees(atan2(u10, v10)) + 360) mod 360` | degrees |
| `era5_P_atm` | `sp / 100` | hPa |
| `era5_cloud_cover` | `tcc` passed through | 0–1 |
| `era5_GHI` | `deaccumulate(ssrd) / 3600`, clipped >= 0 | `< 0 -> 0`; `> 1400 -> NaN` |
| `era5_LW_down` | `deaccumulate(strd) / 3600`, clipped >= 0 | W/m² |
| `era5_precipitation` | `deaccumulate(tp) × 1000`, clipped >= 0 | mm |
| `era5_SZA` | pvlib `get_solarposition().zenith` | degrees |
| `era5_solar_azimuth` | pvlib `get_solarposition().azimuth` | degrees |
| `era5_GHI_clearsky` | pvlib `get_clearsky(model="ineichen").ghi` | W/m² |
| `era5_CSI` | `GHI / GHI_clearsky` clipped [0, 1.5]; forced 0 where `GHI_clearsky <= 10` | |
| `era5_DNI` | `msdwswrf` clipped [0, 1400] (primary); else `GHI / cos(SZA)` fallback | see `04_PHASE_2_AUDIT.md` Part A.7 |
| `era5_DHI` | `(GHI - DNI·cos(SZA)).clip(0)` | closure residual, not measured |

`ETR` (extraterrestrial radiation) is computed by `compute_solar()` but is **not** in
`ERA5_OUTPUT_VARS`, so it is not written to the combined CSV.

## NASA POWER columns carried into the combined CSV

`POWER_VARS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]`, written with a
`power_` prefix: `power_ALLSKY_SFC_SW_DWN`, `power_CLRSKY_SFC_SW_DWN`, `power_T2M`, `power_RH2M`,
`power_WS10M`.

## Point metadata and calendar columns

Written per row by `process_point()`: `point_id`, `lat`, `lon`, `population`, `weight`, `date`,
`event`, `time_utc`, `grid_lat`, `grid_lon`, `month`, `DOY`, `year`, `season`, `season_code`.

## Season classification (`SEASON_MAP` in `02_combine_uttarakhand.py`)

| Months | Season | Code |
|---|---|---|
| Dec, Jan, Feb | **Winter** | 1 |
| Mar, Apr, May | **Summer** | 2 |
| Jun, Jul, Aug | **Monsoon** | 3 |
| Sep, Oct, Nov | **Retreat** | 4 |

Monsoon is **3 months (JJA)** in the season column. Note the inconsistency documented in
`03_PHASE_1_AUDIT.md`: `04b_climate_signature.py` computes `monsoon_index` over
**JJAS (Jun–Sep)**, which does not match `SEASON_MAP`'s JJA definition.

## Climate signature variables (`04b_climate_signature.py`)

### The 18 named indices

`INDEX_COLS` in `04b`, used for the correlation heatmap and distribution plots:

```
Ta_mean, Ta_p95, Ta_p05, DTR, GHI_daily_kWh, kt_mean, kt_std, SAI, CCI,
cloudy_frac, HDD18, CDD24, RH_mean, HSI, wind_mean, seasonality,
monsoon_index, elev_proxy
```

### Tier 1 — sun-event-only indices (computed in `build_signature_tier1`)

| Index | Derivation from the 3 sun-events/day |
|---|---|
| `Ta_mean_proxy` | mean of the daily mean of (sunrise, noon, sunset) `era5_T_amb` |
| `Ta_p95_proxy` / `Ta_p05_proxy` | 95th / 5th percentile of that daily mean |
| `DTR_proxy` | mean of `era5_T_amb_noon - era5_T_amb_sunrise` — **explicitly a proxy, not Tmax-Tmin** |
| `GHI_mean` | mean noon `era5_GHI` (W/m²) |
| `GHI_daily_kWh_proxy` | half-sine approximation `(2/pi) · GHI_noon(kW) · daylength_hours` |
| `kt_mean_proxy` / `kt_std_proxy` | mean / std of noon `era5_CSI` |
| `SAI_proxy` | `sum(era5_GHI) / sum(era5_GHI_clearsky)` over all rows |
| `cloudy_frac_proxy` | fraction of days with noon `CSI < KT_CLOUDY_THRESHOLD (0.35)` |
| `CCI_proxy` | longest consecutive run of cloudy days |
| `HDD18_proxy` / `CDD24_proxy` | `sum(max(0, 18 - Ta_daily))` / `sum(max(0, Ta_daily - 24))` |
| `RH_mean` | mean `era5_RHum` over all rows (**no Tier-2 override**) |
| `HSI` | `RH_mean × fraction of rows with (T_amb - T_dew) < 3 K` (**no Tier-2 override**) |
| `wind_mean` | mean `era5_W_spd` (**no Tier-2 override**) |
| `seasonality_proxy` | `std / mean` of monthly-mean noon `era5_GHI` |
| `monsoon_index` | JJAS `era5_precipitation` sum / total precipitation sum — **proxy only, permanently** |
| `elev_proxy` | `mean(era5_P_atm) / 1013.25` |

### Tier 2 — true daily-integral indices (from `02b_build_daily_aggregates.py`)

Written to `tier2_signature_uttarakhand.csv`, one row per `point_id`:

| Column | Derivation from the full NASA POWER hourly cache |
|---|---|
| `n_days_used` | days with >= `MIN_HOURS_PER_DAY = 20` of 24 hours present |
| `GHI_daily_kWh_mean` | mean of daily `sum(ALLSKY_SFC_SW_DWN) / 1000` |
| `kt_daily_mean` / `kt_daily_std` | daily `GHI/GHIcs` clipped [0, 1.5], guarded at `GHIcs > 0.05` |
| `SAI_true` | `sum(GHI_daily) / sum(GHIcs_daily)` |
| `cloudy_frac_true` | fraction of days with `kt_daily < 0.35` |
| `CCI_true` | longest consecutive cloudy-day run |
| `DTR_true_mean` | mean of daily `max(T2M) - min(T2M)` — **true diurnal range** |
| `Ta_mean_true`, `Ta_p95_true`, `Ta_p05_true` | mean / q95 / q05 of daily-mean `T2M` |
| `HDD18_true` / `CDD24_true` | degree-days from the true daily mean |
| `RH_mean_true`, `wind_mean_true` | mean of daily-mean `RH2M` / `WS10M` |
| `seasonality_true` | `std/mean` of monthly-mean daily GHI |

### Canonical-column rule (`CANON_MAP` in `04b`)

For each of `GHI_daily_kWh, DTR, kt_mean, kt_std, SAI, cloudy_frac, CCI, HDD18, CDD24, Ta_mean,
Ta_p95, Ta_p05, seasonality` the canonical column takes the **true Tier-2 value where present**
and falls back to the Tier-1 proxy otherwise. Both are kept side by side (`_proxy` / `_true`
suffixes) and both are **excluded from the clustering matrix** so only the canonical version
clusters.

`RH_mean`, `HSI`, `wind_mean`, `monsoon_index` and `elev_proxy` have **no** Tier-2 counterpart in
`CANON_MAP` and remain sun-event-derived. `wind_mean_true` and `RH_mean_true` are computed by
`02b` but are not mapped, so they are dropped from the clustering matrix by the `_true` suffix
rule. `GHI_mean` has no `_proxy` suffix and no `CANON_MAP` entry, so it enters the clustering
matrix directly as an ERA5 quantity.

### Derived PCM-facing quantities

```python
T_DELIVERY_C  = 50.0
DT_APPROACH_C =  7.0
TM_TARGET_C   = T_DELIVERY_C + DT_APPROACH_C          # 57 °C, constant for every point

DRAW_RATE_KG_PER_S  = 60.0 / 1000 / 60                # = 0.001 kg/s
CP_WATER            = 4.186                           # kJ/kg·K
ASSUMED_PCM_MASS_KG = 50.0

sig["T_mains_est_C"]        = sig["Ta_mean"] - 2.0
q_night_kw                  = DRAW_RATE_KG_PER_S * CP_WATER * (T_DELIVERY_C - T_mains_est_C)
sig["L_required_kJ_per_kg"] = (q_night_kw * 3600 * 7) / ASSUMED_PCM_MASS_KG
```

Notes carried forward as caveats (see `05_PHASE_3_AUDIT.md`):
- The `- 2.0` K mains-temperature offset is **unsourced in-code**.
- There is **no `SHARE_PCM` factor** in this formula — the Uttarakhand `04b` sizes `L_required`
  from a 7-hour draw at 0.001 kg/s against the full 50 kg PCM mass.
- `Tsoil_proxy_C = Ta_mean - 3.0` is defined only to feed the `int_wind_x_TaMinusTsoil`
  interaction term and is dropped from the clustering matrix.

### 5 interaction terms

`int_GHI_x_ktstd`, `int_DTR_x_cloudyfrac`, `int_RH_x_TaMinusTm`, `int_wind_x_TaMinusTsoil`,
`int_CCI_x_1minusSAI`.

### PCA block

`PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]`,
`StandardScaler` then `PCA(n_components=0.95, random_state=42)`. Loadings written to
`pca_loadings.csv`. The number of retained components for the Uttarakhand run is **not available
in the source files** — `pca_loadings.csv` is under the git-ignored `data/processed/` tree.

## Physical bounds table (`BOUNDS` in `04_preprocess_uttarakhand.py`)

| Column | Lower | Upper |
|---|---|---|
| `era5_GHI` | 0 | 1400 W/m² |
| `era5_DNI` | 0 | 1400 W/m² |
| `era5_DHI` | 0 | 900 W/m² |
| `era5_GHI_clearsky` | 0 | 1400 W/m² |
| `era5_CSI` | 0 | 1.5 |
| `era5_LW_down` | **50** | **600** W/m² |
| `era5_T_amb` | -30 | 55 °C |
| `era5_T_dew` | -30 | 40 °C |
| `era5_RHum` | 0 | 100 % |
| `era5_W_spd` | 0 | 50 m/s |
| `era5_P_atm` | **850** | **1060** hPa |
| `era5_cloud_cover` | 0 | 1 |
| `era5_precipitation` | 0 | 200 mm |
| `era5_SZA` | 0 | 180 ° |
| `power_ALLSKY_SFC_SW_DWN` | 0 | 1400 W/m² |
| `power_CLRSKY_SFC_SW_DWN` | 0 | 1400 W/m² |
| `power_T2M` | -30 | 55 °C |
| `power_RH2M` | 0 | 100 % |
| `power_WS10M` | 0 | 50 m/s |

Out-of-range values become `NaN` (never silently clipped) and are then imputed by step 4.
The `era5_P_atm >= 850 hPa` and `era5_LW_down >= 50 W/m²` bounds are the two that bite hardest for
Uttarakhand — see `04_PHASE_2_AUDIT.md` Part B.
