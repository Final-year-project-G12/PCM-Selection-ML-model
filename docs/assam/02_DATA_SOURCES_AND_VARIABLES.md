# 02 — Data Sources and Variables

## Primary Data Sources

### ERA5 (ECMWF Reanalysis v5)
- **Provider**: Copernicus Climate Data Store (CDS), accessed via `cdsapi`
- **Product**: ERA5 hourly data on single levels
- **Spatial resolution**: 0.25° × 0.25° native grid
- **Temporal coverage**: 2016–2025 (10 years)
- **Sampling strategy**: Population-weighted grid sampling (**129 active points**, covering 87.8% of Assam's population)
- **Download format**: NetCDF, sun-event-aligned hours (sunrise, solar noon, sunset) and full hourly series for medoids
- **ERA5 variables downloaded**:

| Variable | ERA5 Parameter | Native Unit | Notes / Role |
|---|---|---|---|
| Solar radiation (GHI proxy) | `ssrd` (surface solar radiation downwards) | J/m² | Accumulated; converted to flux via duration-overlap |
| Thermal radiation | `strd` (surface thermal radiation downwards) | J/m² | Accumulated; atmospheric longwave exchange |
| 2m temperature | `t2m` | K → °C | Instantaneous dry-bulb temperature |
| 2m dewpoint | `d2m` | K → °C | Used for Magnus-formula RH derivation |
| 10m U-wind | `u10` | m/s | Zonal surface wind component |
| 10m V-wind | `v10` | m/s | Meridional surface wind component |
| Surface pressure | `sp` (`surface_pressure`) | Pa → hPa | Surface atmospheric pressure (code requests `surface_pressure`, not mean sea level pressure `msl`) |
| Total cloud cover | `tcc` | 0–1 fraction | Fractional cloud cover |
| Total precipitation | `tp` | m → mm | Accumulated precipitation |
| Mean surface direct solar | `avg_sdirswrf` (matches `msdwswrf`/`fdir`/`msdrswrf`) | W/m² | Direct radiation component |
| Clear-sky solar radiation | `ssrdc` (`surface_solar_radiation_downward_clear_sky`) | J/m² → W/m² | ERA5's own clear-sky GHI; preferred source for `GHI_clearsky` in `02_combine_assam.py`, with pvlib Ineichen as fallback (`clearsky_source` column records which was used) |

### NASA POWER
- **Provider**: NASA Langley Research Center Prediction of Worldwide Energy Resources
- **Product**: **Hourly** point API only (`https://power.larc.nasa.gov/api/temporal/hourly/point`), parameters `ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M` (`01b_download_nasapower.py`). There is no daily-product download, no `T2M_MAX`/`T2M_MIN` request (daily max/min are instead derived in `02b` from the 24 hourly `T2M` values), wind is 10 m (`WS10M`) not 2 m, and **`PRECTOTCORR` is not downloaded** — `monsoon_index` is therefore computed from ERA5's 3×/day precipitation samples, not a true POWER daily precipitation integral.
- **Temporal coverage**: 2016–2025 (10 years, matching ERA5)
- **Role**: Independent cross-source validation against ERA5; generation of daily integrals for Tier 2 signature indices
- **Point IDs**: `ASP_0001` through `ASP_0129` (all 129 points active)
- **Authoritative record count**: `daily_aggregates_assam.csv` contains **467,367 daily rows** (reflecting valid days where $\ge 20$ hours met strict retrieval criteria; incomplete days dropped)

### Population Raster
- **Source**: WorldPop unconstrained global mosaic, India, UN-adjusted, 100m, 2020
- **Use**: Aggregated to 0.25° ERA5 grid cells; highest-density cells selected sequentially to achieve 87.8% population coverage of Assam

### State Boundary
- **Source**: GADM v4.1, India administrative level 1 (`NAME_1 == "Assam"`)
- **Filter**: All candidate grid centroids clipped strictly within administrative boundaries

---

## Cross-Source Validation: ERA5 vs. NASA POWER

Cross-source agreement analysis (`03b_agreement_analysis_assam.py`) quantitatively evaluated daytime GHI:
- Mean Bias Error (MBE) between ERA5 and NASA POWER was **1.1%** (well within the $\le 10\%$ tolerance threshold).
- Generated authoritative decision: **`BACKBONE`** (`bias_decision_assam.txt`).
- Consequence: ERA5 data flows into downstream clustering and physics simulation unmodified, without empirical quantile mapping.

---

## Derived Climate Variables (`02_combine_assam.py`)

| Variable | Derivation / Formula | Physical Meaning |
|---|---|---|
| `RHum` | Magnus formula from $T_{\text{amb}}$ and $T_{\text{dew}}$ | Relative humidity (%) |
| `W_spd` | $\sqrt{u_{10}^2 + v_{10}^2}$ | Scalar wind speed (m/s) |
| `W_dir` | $\text{atan2}(u_{10}, v_{10})$ | Wind direction (degrees) |
| `DNI` | Primary: ERA5 direct-radiation field (`avg_sdirswrf`), clipped [0,1400]. Fallback (rare): `GHI / cos(SZA)` when the field is absent — a crude closure, not a decomposition model | Direct Normal Irradiance |
| `CSI` | $GHI / GHI_{\text{clearsky}}$, clipped to [0, 1.2] (recomputed in `04_preprocess_assam.py`; `GHI_clearsky` prefers ERA5 `ssrdc`, falling back to pvlib Ineichen) | Clear-sky index |
| `elevation_m` | Default 100m (Assam valley baseline) | Atmospheric pressure adjustment |

---

## Season Classification (`02_combine_assam.py`)

| Month | Season Name | Code | Climatological Character |
|---|---|---|---|
| Dec, Jan, Feb | Winter | 1 | Cool, dry, clear skies, lowest ambient temperatures |
| Mar, Apr, May | Pre-Monsoon | 2 | Rising temperatures, convective activity, increasing humidity |
| Jun, Jul, Aug, Sep | Monsoon | 3 | Peak precipitation (>2500 mm/yr), persistent cloud cover, high RH |
| Oct, Nov | Post-Monsoon | 4 | Retreating monsoon, transitional temperatures |

---

## Climate Signature Structure (19 Indices, `04b_climate_signature.py`)

**Correction (verified against the current script):** `04b_climate_signature.py` computes every
index directly from `data/preprocessed/assam_cleaned_physical.csv` (the event-sampled — sunrise/
noon/sunset — physical dataset). It does **not** read `daily_aggregates_assam.csv` or
`tier2_signature_assam.csv` at all, even though `02b_build_daily_aggregates_assam.py` produces
both. There is currently no Tier-1/Tier-2 split inside the signature script itself — that is a
stale description of an earlier architecture. All 19 indices below are Tier-1 (event-sampled)
quantities:

- `Ta_mean`, `Ta_p95`, `Ta_p05`: mean / 95th / 5th percentile of `era5_T_amb` across **all three
  daily events** (not noon-only, and not a true daily mean)
- `DTR`: mean of (noon `T_amb` − min(sunrise, sunset `T_amb`)) per day; falls back to
  `(Ta_p95 − Ta_p05) / 2` if no valid day has all three events
- `HDD18` / `CDD24`: `mean(max(0, 18 − Ta)) × 365.25` / `mean(max(0, Ta − 24)) × 365.25`, using
  all-event `Ta` (an event-sampled annualisation, not a true daily-integral degree-day sum)
- `GHI_mean`: mean of `era5_GHI` over all events with `GHI > 0`
- `GHI_daily_kWh_est`: **proxy** estimate — noon-event mean GHI × 6.5 equivalent solar hours ÷
  1000; explicitly documented in the script as not a true 24-hour integral
- `kt_mean`, `kt_std`: mean / std of `era5_CSI` over events with `CSI > 0` (defaults 0.60 / 0.15
  if the CSI column is absent)
- `SAI`: fraction of days whose noon-based daily GHI estimate ≥ 2.0 kWh/m²/day
- `cloudy_frac`: fraction of events with `GHI / max(10, GHI_clearsky) < 0.35`
- `CCI`: `1 − std(daily cloudy-event fraction)` — a day-to-day cloudiness-persistence index, not
  a "Cloud Cover Index" and not a longest-run count
- `RH_mean`: mean `era5_RHum` across all events
- `HSI`: `RH_mean × mean(fraction of events with (Ta − Td) < 3 K)` — a dew-point-proximity index,
  **not** `RH_mean × GHI_daily`
- `wind_mean`: mean `era5_W_spd`
- `elev_proxy`: mean `era5_P_atm` ÷ 1013.25
- `monsoon_index`: Σ ERA5 event-sampled precipitation in months 6–9 ÷ Σ annual ERA5 event-sampled
  precipitation (POWER's `PRECTOTCORR` is not downloaded, so this is not a true daily integral)
- `seasonality`: std ÷ mean of the 12 monthly-mean `GHI_mean` values

`precipitation_annual`, `Ta_min_true`, and `Ta_max_true` are **not** computed anywhere in the
signature script and do not appear in `climate_signatures_raw.csv`.

Five interaction terms are also computed and carried into the standardised clustering matrix
(`climate_signatures_matrix.csv`), but are not listed in earlier drafts of this document:
`ix_GHI_x_kt_std`, `ix_DTR_x_cloudy`, `ix_RH_x_Ta`, `ix_wind_x_Ta`, `ix_CCI_x_1mSAI`.

*Note on Soil Temperature*: soil temperature was never downloaded; no soil-temperature term
appears in the current signature or interaction terms.

---

## PCM Property Database (`pcm_database_final.csv`)

### Final Locked Database (58 PCMs)
- **Dataset**: `data/processed/pcm/pcm_database_final.csv`
- **Scope**: **58 deduplicated PCM records** spanning commercial paraffins (Rubitherm RT), bio-based organics (PLUSS savE), fatty acids, and eutectics.
- **Properties**: 41 columns capturing thermodynamic, physical, safety, and operational parameters.
- **Strict Provenance**:
  - `source_type`: Explicit attribution (Manufacturer datasheet, Literature primary source).
  - `value_status`: Cell-level flags (`Reported`, `Imputed`, `Missing`).
- **Strict Specific Heat Capacity Policy**:
  - $C_{p,\text{avg}} = 0.5 \times (C_{p,\text{solid}} + C_{p,\text{liquid}})$ is computed **only** when both phase-specific values are reported.
  - The model **never** silently falls back from a missing phase to a single reported phase.

### Historical Prototype Database (`pcm_database_assam.csv`)
- An early 25-row prototype (`pcm_database_assam.csv`) used during initial $K=4$ pipeline exploration is retained as a **locked historical artifact** and must not be confused with the final 58-row production database.
