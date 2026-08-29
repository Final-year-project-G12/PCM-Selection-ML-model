# 02 — Data Sources and Variables

## External data sources (exact, as requested in code)

| Source | Product | Access | Used by |
|---|---|---|---|
| Copernicus Climate Data Store (CDS) | `reanalysis-era5-single-levels`, `product_type=reanalysis` | `cdsapi.Client`, requires `.cdsapirc` credentials | `00c_attach_elevation.py`, `01_download_era5_rajasthan.py` |
| NASA POWER | Hourly point API, `community=RE` (Renewable Energy) | `https://power.larc.nasa.gov/api/temporal/hourly/point`, no API key | `01b_download_nasapower.py` |
| GADM v4.1 | India admin level 1 boundary (GeoJSON) | `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json` | `00a_build_population_grid.py` |
| WorldPop | India 2020 unconstrained population, UN-adjusted, 100 m | `https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_UNadj.tif` (~1.5–2 GB) | `00a_build_population_grid.py` |

## ERA5 variables requested (exact CDS short names, from `01_download_era5_rajasthan.py`)

**Instant (analysis, TYPE=AN)** — snapshot values, no deaccumulation needed:
```
2m_temperature                → t2m      → T_amb (K → °C)
2m_dewpoint_temperature       → d2m      → T_dew → RHum (Magnus formula)
10m_u_component_of_wind       → u10      → W_spd, W_dir (m/s, °)
10m_v_component_of_wind       → v10      → (combined with u10)
total_cloud_cover             → tcc      → cloud_cover (0–1, unconverted)
surface_pressure               → sp       → P_atm (Pa → hPa)
```

**Accumulated (forecast, TYPE=FC)** — see `13_SOLAR_DERIVED_VARIABLES.md` for why "accumulated" is
in scare quotes for this pipeline's actual download:
```
surface_solar_radiation_downwards              → ssrd     → GHI (J/m² per downloaded hour → W/m²)
mean_surface_direct_short_wave_radiation_flux  → msdwswrf → avg_sdirswrf → DNI (already W/m²)
surface_thermal_radiation_downwards             → strd     → LW_down (J/m² → W/m²)
total_precipitation                             → tp       → precipitation (m → mm)
```
`00c_attach_elevation.py` additionally requests **`geopotential`** (single time-invariant field, one
API call, `2020-01-01T00:00`) → `elevation_m = z / 9.80665`.

## NASA POWER parameters (exact, from `01b_download_nasapower.py`)

```
ALLSKY_SFC_SW_DWN   — all-sky surface shortwave downward irradiance (≈ GHI equivalent)
CLRSKY_SFC_SW_DWN   — clear-sky surface shortwave downward irradiance
T2M                 — 2 m temperature
RH2M                — 2 m relative humidity
WS10M               — 10 m wind speed
```
Fill value `-999` is replaced with `NaN` on ingest (`02_combine_rajasthan.py`, blanket, no
column-specific bound check). **`PRECTOTCORR` (precipitation) was never requested** — confirmed by
direct code inspection — which is why `monsoon_index` (Tier 2) is always a GHI-fraction proxy in
this pipeline, never a true precipitation-derived index (see `16_CLIMATE_SIGNATURE.md`).

## Full variable transformation table

| Variable | ERA5/POWER name | Original unit | Stored unit | Transformation | Validation |
|---|---|---|---|---|---|
| Air temperature | `t2m` / `T2M` | K | °C | `−273.15` | Range check [−5, 60]°C |
| Dew point | `d2m` | K | °C | `−273.15` | none dedicated |
| Relative humidity | derived from `t2m`,`d2m` / `RH2M` | — | % | Magnus-Tetens (Alduchov & Eskridge 1996, a=17.625, b=243.04) | clip [0,100] |
| Wind speed | `u10`,`v10` / `WS10M` | m/s | m/s | `√(u²+v²)` | Range check [0,40] m/s |
| Wind direction | `u10`,`v10` | — | ° | `(degrees(atan2(u,v))+360) mod 360` | none |
| Surface pressure | `sp` | Pa | hPa | `/100` | Range check [800,1050] hPa |
| Cloud cover | `tcc` | fraction | fraction | none | Range check [0,1] |
| GHI | `ssrd` | J/m² (per downloaded hour) | W/m² | `accum_to_flux(x)/3600`, clip≥0 | Range check [0,1400] W/m² |
| DNI (primary) | `msdwswrf`/`fdir`/`msdrswrf` | already W/m² (assumed) | W/m² | `clip(0,1400)` only, **no /3600** | Range check [0,1400] |
| DNI (fallback) | derived from GHI, SZA | — | W/m² | `GHI/cos(SZA)` where `cosZ>0.05`, clip[0,1400] | same |
| DHI | derived | — | W/m² | `(GHI − DNI·cosZ)`, clip≥0 (residual, not modeled) | Range check [0,1400] |
| Clear-sky GHI | pvlib Ineichen model | — | W/m² | model output | Range check [0,1400] |
| Clearness index (CSI) | `GHI/GHI_clearsky` | — | dimensionless | forced 0 if `GHI_clearsky≤10`, else clip[0,1.5] | QC bound [0,2] — looser than pipeline clip, dead check |
| Longwave down | `strd` | J/m² | W/m² | `accum_to_flux(x)/3600`, clip≥0 | Range check [0,700] |
| Precipitation | `tp` | m | mm | `accum_to_flux(x)×1000`, clip≥0 | Range check [0,200] |
| Solar zenith angle | pvlib `get_solarposition` | — | ° | direct | Range check [0,180] |
| Solar azimuth | pvlib `get_solarposition` | — | ° | direct | Range check [0,360] |
| Elevation | ERA5 `z` (geopotential) | m²/s² | m | `/9.80665` (standard gravity) | Outlier flag [−420, 8850] m (Dead Sea..Everest), not clipped |
| ETR (extraterrestrial) | pvlib `get_extra_radiation` | — | W/m² | computed | **computed but never written to output CSV** |

See `13_SOLAR_DERIVED_VARIABLES.md` for the DNI/DHI derivation logic in full, and
`09_ERA5_DATA_PIPELINE.md` for the deaccumulation story that motivates the "already W/m²" caveat on
GHI/LW/precip above.

## Column-name ambiguity worth flagging

`avg_sdirswrf` is populated from whichever of `msdwswrf`, `fdir`, or `msdrswrf` matches first in the
downloaded NetCDF (`next((c for c in df.columns if c in (...)), None)`). These are **not the same
physical quantity** in ERA5's variable catalogue: `fdir` is an accumulated direct-radiation field
(needs the same J/m²→W/m² treatment as `ssrd`); `msdwswrf`/`msdrswrf` are mean-rate fields (already
W/m², no conversion needed). The code applies identical treatment (clip only, no `/3600`) regardless
of which one actually matched — see `20_IMPLEMENTATION_ISSUES.md` item 8 for the audit consequence.

## Output variable list (`ERA5_OUTPUT_VARS`, exact, from `02_combine_rajasthan.py`)

```
T_amb, T_dew, RHum, W_spd, W_dir, GHI, DNI, DHI, LW_down, cloud_cover,
precipitation, P_atm, SZA, solar_azimuth, GHI_clearsky, CSI
```
Prefixed `era5_` in the combined CSV; the five NASA POWER variables are prefixed `power_`.
