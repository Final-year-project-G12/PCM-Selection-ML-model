# 02 — Data Sources and Variables

## External data sources (exact, as coded)

| Source | Product | Access | Used by |
|---|---|---|---|
| Copernicus CDS | `reanalysis-era5-single-levels`, `product_type=reanalysis` | `cdsapi.Client`, `.cdsapirc` | `01_download_era5_tamilnadu.py` |
| NASA POWER | Hourly point API, `community=RE` | `https://power.larc.nasa.gov/api/temporal/hourly/point`, no key | `01b_download_nasapower.py` |
| GADM v4.1 | India admin level 1, filtered `NAME_1 ∈ {"TamilNadu"[, "Puducherry"]}` | GeoJSON, `INCLUDE_PUDUCHERRY=False` by default | `00a_build_population_grid.py` |
| WorldPop | India 2020 UN-adjusted, 100 m | same URL pattern as Rajasthan | `00a_build_population_grid.py` |

Puducherry's enclaves are deliberately excluded by default (`INCLUDE_PUDUCHERRY=False`) — "a distinct
administrative unit," matching Rajasthan's own precedent of excluding non-target-state territory.

## ERA5 variables (identical variable list to Rajasthan)

```
Instant:  2m_temperature, 2m_dewpoint_temperature, 10m_u_component_of_wind,
          10m_v_component_of_wind, total_cloud_cover, surface_pressure
Accum:    surface_solar_radiation_downwards, mean_surface_direct_short_wave_radiation_flux,
          surface_thermal_radiation_downwards, total_precipitation
```
`msdwswrf` is explicitly commented in-code as a **mean-rate** field (already W/m²), distinct from
`ssrd`/`strd`/`tp`'s accumulated (J/m² or m) convention — and confirmed in the combine script to
receive different treatment (`.clip(0)` only, no deaccumulation) — see `04_PHASE_2_AUDIT.md`.

## NASA POWER parameters (identical to Rajasthan)

```
ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M
```
`PRECTOTCORR` (precipitation) is **not requested**, identical omission to Rajasthan — `monsoon_index`
is therefore a proxy (ERA5's sparser 3×/day precipitation sampling) in both pipelines, self-documented
in both.

## Variable transformation table

| Variable | ERA5/POWER name | Original unit | Stored unit | Transformation | Notes |
|---|---|---|---|---|---|
| Air temperature | `t2m`/`T2M` | K | °C | `−273.15` | bound `[-30,55]` in QC step (wider than Rajasthan's `[-5,60]`) |
| Relative humidity | derived / `RH2M` | — | % | Magnus-Tetens, a=17.625, b=243.04 | clip [0,100] |
| Wind speed | `u10,v10`/`WS10M` | m/s | m/s | `√(u²+v²)` | bound `[0,50]` (Rajasthan: `[0,40]`) |
| Surface pressure | `sp` | Pa | hPa | `/100` | bound `[850,1060]` |
| GHI | `ssrd` | J/m² | W/m² | `deaccumulate(x)/3600` | **true diff-based deaccumulation, confirmed correct** — see `04_PHASE_2_AUDIT.md` |
| DNI (primary) | `msdwswrf` | already W/m² | W/m² | `.clip(0,1400)` only | |
| DNI (fallback) | derived | — | W/m² | `GHI/cos(SZA)` where `cos(SZA)>0.05` | identical closure-equation approach to Rajasthan |
| DHI | derived | — | W/m² | `GHI − DNI·cos(SZA)`, clip≥0 | residual, not independently modeled — same as Rajasthan |
| LW down | `strd` | J/m² | W/m² | `deaccumulate(x)/3600` | |
| Precipitation | `tp` | m | mm | `deaccumulate(x)×1000` | |
| Clear-sky GHI | pvlib Ineichen | — | W/m² | model output | no explicit `linke_turbidity` override, same as Rajasthan |
| CSI | `GHI/GHI_clearsky` | — | dimensionless | forced 0 below 10 W/m² clearsky, clip[0,1.5] | identical logic to Rajasthan |
| Elevation (population points) | none real | — | m | **flat `DEFAULT_ALT_M=150`** | no ERA5-geopotential attachment step exists for TN (unlike Rajasthan's `00c_attach_elevation.py`) |
| Elevation (signature-level pseudo-proxy) | ERA5 `sp` | hPa | dimensionless ratio | `mean(P_atm)/1013.25` | a **separate**, pressure-ratio-based pseudo-elevation used only inside `04b_climate_signature.py`'s PCA block — distinct mechanism from the flat 150 m population-grid proxy; the two elevation concepts are not reconciled with each other anywhere in the codebase |

## Two elevation concepts, not one — worth flagging explicitly

Unlike Rajasthan (one real per-point elevation, attached once, used consistently), Tamil Nadu carries
**two independent, never-reconciled elevation stand-ins**: (1) a flat `150 m` constant fed to pvlib's
solar-geometry calls in the combine script (`DEFAULT_ALT_M = 150`), and (2) an entirely separate
`elev_proxy = mean(era5_P_atm)/1013.25` pressure-ratio pseudo-elevation computed inside the climate
signature script's PCA block. Neither is real per-point elevation; the two exist for different
purposes (solar geometry vs. PCA feature) and neither one is validated against the other. Both are
self-documented as acceptable for Tamil Nadu's comparatively gentle terrain (coastal plain + interior
plateau, with the Western Ghats/Nilgiris hills as the one region where this matters more) — but this
project has no scripted way to check that the two ever imply the same physical picture.

## Output schema differences from Rajasthan

`climate_tamilnadu_points.csv` matches Rajasthan's `climate_rajasthan_points.csv` schema closely
(same `era5_*`/`power_*` column set, same metadata columns), but with **no `elevation_m` column** —
Rajasthan's combine script reads a real per-point elevation from `population_grid_points.csv`;
Tamil Nadu's has no such column to read (no `00c`-equivalent script exists), so every point uses the
flat 150 m default unconditionally, not as a fallback for missing values.
