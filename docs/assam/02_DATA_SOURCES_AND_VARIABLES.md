# 02 — Data Sources and Variables (Assam)

## Primary Data Sources

### ERA5 (ECMWF Reanalysis v5)

- **Provider**: Copernicus Climate Data Store (CDS), accessed via `cdsapi`.
- **Product**: ERA5 hourly data on single levels.
- **Native spatial resolution**: 0.25° × 0.25°.
- **Temporal coverage**: 2016–2025 (10 years).
- **Current Assam sampling**: 80 active population-weighted grid points, IDs `ASP_0001`–`ASP_0080`.
- **Population coverage**: 87.8% of Assam's population.
- **Download format**: NetCDF. The pipeline uses sun-event-aligned samples (sunrise, solar noon, sunset) and full hourly series for selected medoid locations.

#### ERA5 variables used

| Variable | ERA5 parameter / source field | Native unit | Role / transformation |
|---|---|---|---|
| Global horizontal solar radiation (GHI proxy) | `ssrd` | J/m² | Accumulated radiation; converted to flux using the pipeline's duration-overlap handling |
| Thermal radiation | `strd` | J/m² | Downward atmospheric longwave radiation |
| 2 m air temperature | `t2m` | K | Converted to °C |
| 2 m dewpoint temperature | `d2m` | K | Converted to °C; used to derive relative humidity |
| 10 m U-wind | `u10` | m/s | Zonal wind component |
| 10 m V-wind | `v10` | m/s | Meridional wind component |
| Surface pressure | `sp` / `surface_pressure` | Pa | Converted to hPa; this is surface pressure, not mean sea-level pressure |
| Total cloud cover | `tcc` | 0–1 fraction | Fractional cloud cover |
| Total precipitation | `tp` | m | Converted to mm |
| Direct solar radiation field | `avg_sdirswrf` | W/m² or source-dependent | Used as the primary direct-radiation input; see the field-name/unit caveat below |
| Clear-sky solar radiation | `ssrdc` / `surface_solar_radiation_downward_clear_sky` | J/m² | Converted to W/m²; preferred for `GHI_clearsky`, with pvlib Ineichen as fallback |

**Direct-radiation field caveat:** the pipeline's `avg_sdirswrf` may be populated from one of `msdwswrf`, `fdir`, or `msdrswrf`. These fields are not physically interchangeable: `fdir` is accumulated radiation, whereas `msdwswrf`/`msdrswrf` are mean-rate fields. The code's treatment must be checked against the actual matched field and its units before treating every record as equivalent. The Assam derivation description identifies `avg_sdirswrf` as the primary DNI source and a GHI/SZA closure as a fallback when the field is absent; this fallback is not a full irradiance decomposition model.

### NASA POWER

- **Provider**: NASA Langley Research Center, Prediction of Worldwide Energy Resources.
- **Product / endpoint**: Hourly point API: `https://power.larc.nasa.gov/api/temporal/hourly/point`.
- **Parameters requested**: `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, and `WS10M`.
- **Temporal coverage**: 2016–2025, matching ERA5.
- **Role**: Independent cross-source validation against ERA5 and input to the daily-aggregation workflow.
- **Wind height**: `WS10M` is 10 m wind speed, not 2 m wind speed.
- **Precipitation limitation**: `PRECTOTCORR` is not downloaded. Therefore, the current `monsoon_index` is based on ERA5 event-sampled precipitation, not a NASA POWER daily precipitation integral.
- **Daily aggregation output**: the latest audited Assam pipeline contains **292,160 rows** in `daily_aggregates_assam.csv` (80 points × 3,652 days), covering **2016-01-02 to 2025-12-31**. This replaces the stale 467,367-row figure, which belongs to the earlier 129-point dataset.

The available audit establishes the daily aggregate row count and date range. It does not establish that every row is a complete 24-hour POWER day; do not describe the output as a complete daily integral without checking the aggregation and missing-hour rules in the current script.

### Population Raster

- **Source**: WorldPop unconstrained global mosaic, India, UN-adjusted, 100 m, 2020.
- **Use**: Population values are aggregated to the 0.25° ERA5 grid; candidate cells are selected by population density.
- **Current Assam result**: 80 active grid points, with 87.8% population coverage.

### State Boundary

- **Source**: GADM v4.1, India administrative level 1.
- **Filter**: Assam boundary selected using `NAME_1 == "Assam"`; candidate grid centroids are clipped to the state boundary.

---

## Cross-Source Validation: ERA5 vs. NASA POWER

The cross-source agreement analysis is implemented in `03b_agreement_analysis_assam.py`.

- **Daytime GHI mean bias error (MBE)**: 1.1%.
- **Decision**: `BACKBONE`, recorded in `bias_decision_assam.txt`.
- **Pipeline consequence**: ERA5 is retained as the downstream climate-data backbone without empirical quantile mapping.

The 1.1% value is the reported audited result. It should be rechecked if the input population grid, time alignment, or source preprocessing is changed.

---

## Derived Climate Variables (`02_combine_assam.py`)

| Variable | Derivation / formula | Physical meaning |
|---|---|---|
| `RHum` | Magnus formula using ambient temperature and dewpoint | Relative humidity (%) |
| `W_spd` | \(\sqrt{u_{10}^{2}+v_{10}^{2}}\) | Scalar wind speed (m/s) |
| `W_dir` | `atan2(u10, v10)` with the implementation's angle convention | Wind direction (degrees) |
| `DNI` | Primary: direct-radiation field `avg_sdirswrf`, clipped to [0, 1400]. Fallback: `GHI / cos(SZA)` when the field is absent; this is a crude closure | Direct normal irradiance |
| `GHI_clearsky` | Prefer ERA5 `ssrdc`; use pvlib Ineichen if the ERA5 clear-sky field is unavailable | Clear-sky global horizontal irradiance |
| `CSI` | `GHI / GHI_clearsky`, clipped to [0, 1.2] in the preprocessing path described by the Assam audit | Clear-sky index |
| `elevation_m` | Default value 100 m in the Assam combination description | Elevation / atmospheric-pressure context |

---

## Season Classification (`02_combine_assam.py`)

| Months | Season | Code | Climatological character |
|---|---|---:|---|
| Dec, Jan, Feb | Winter | 1 | Cooler and generally drier period |
| Mar, Apr, May | Pre-monsoon | 2 | Rising temperatures and convective activity |
| Jun, Jul, Aug, Sep | Monsoon | 3 | High precipitation, cloud cover, and humidity |
| Oct, Nov | Post-monsoon | 4 | Retreating-monsoon transition |

---

## Climate Signature Structure (19 Indices, `04b_climate_signature.py`)

**Implementation note verified against the current Assam script:** `04b_climate_signature.py` computes all 19 climate indices directly from `data/preprocessed/assam_cleaned_physical.csv`, the event-sampled sunrise/noon/sunset physical dataset. It does **not** read `daily_aggregates_assam.csv` or `tier2_signature_assam.csv`, even though `02b_build_daily_aggregates_assam.py` produces those files. There is currently no Tier-1/Tier-2 split inside the signature script. Consequently, the 19 indices below should be described as event-sampled quantities, not as a mixture of event-sampled and daily-integral features.

- `Ta_mean`, `Ta_p95`, `Ta_p05`: mean, 95th percentile, and 5th percentile of `era5_T_amb` across all three daily events. These are event-sample statistics, not a true hourly or daily mean.
- `DTR`: mean of (noon `T_amb` − minimum of sunrise and sunset `T_amb`) per day; fallback is `(Ta_p95 − Ta_p05) / 2` if no valid day has all three events.
- `HDD18` / `CDD24`: `mean(max(0, 18 − Ta)) × 365.25` / `mean(max(0, Ta − 24)) × 365.25`, using all event samples. These are event-sampled annualisations, not true daily-integral degree-day sums.
- `GHI_mean`: mean `era5_GHI` over events with `GHI > 0`.
- `GHI_daily_kWh_est`: proxy estimate based on noon-event mean GHI × 6.5 equivalent solar hours ÷ 1000; it is not a true 24-hour radiation integral.
- `kt_mean`, `kt_std`: mean and standard deviation of `era5_CSI` over events with `CSI > 0`; defaults are 0.60 and 0.15 if the CSI column is absent.
- `SAI`: fraction of days whose noon-based daily GHI estimate is at least 2.0 kWh/m²/day.
- `cloudy_frac`: fraction of events for which `GHI / max(10, GHI_clearsky) < 0.35`.
- `CCI`: `1 − std(daily cloudy-event fraction)`. This is a day-to-day cloudiness-persistence index, not a longest-run count.
- `RH_mean`: mean `era5_RHum` across all events.
- `HSI`: `RH_mean × mean(fraction of events with (Ta − Td) < 3 K)`. This is a dewpoint-proximity index, not `RH_mean × GHI_daily`.
- `wind_mean`: mean `era5_W_spd`.
- `elev_proxy`: mean `era5_P_atm` ÷ 1013.25.
- `monsoon_index`: sum of ERA5 event-sampled precipitation in June–September divided by total annual ERA5 event-sampled precipitation. Because POWER `PRECTOTCORR` is not downloaded, this is not a true daily precipitation integral.
- `seasonality`: standard deviation divided by mean of the 12 monthly mean `GHI_mean` values.

The signature script does **not** compute `precipitation_annual`, `Ta_min_true`, or `Ta_max_true`; these do not appear in `climate_signatures_raw.csv`.

Five interaction terms are also carried into the standardised clustering matrix (`climate_signatures_matrix.csv`):

- `ix_GHI_x_kt_std`
- `ix_DTR_x_cloudy`
- `ix_RH_x_Ta`
- `ix_wind_x_Ta`
- `ix_CCI_x_1mSAI`

**Soil temperature:** soil temperature was not downloaded and no soil-temperature term appears in the current signature or interaction terms.

---

## PCM Property Database (`pcm_database_final.csv`)

### Final Locked Database

- **Dataset**: `data/processed/pcm/pcm_database_final.csv`.
- **Current database size**: 58 deduplicated PCM records.
- **Property count**: 41 columns covering thermodynamic, physical, safety, and operational parameters.
- **Scope**: commercial paraffins (Rubitherm RT), bio-based organics (PLUSS savE), fatty acids, and eutectics.
- **Provenance fields**:
  - `source_type`: explicit attribution, such as manufacturer datasheet or primary literature source.
  - `value_status`: cell-level status (`Reported`, `Imputed`, `Missing`).
- **Specific heat capacity policy**:
  - \(C_{p,\mathrm{avg}} = 0.5 \times (C_{p,\mathrm{solid}} + C_{p,\mathrm{liquid}})\) is computed only when both phase-specific values are reported.
  - The model does not silently substitute a single reported phase when the other phase is missing.

### Historical Prototype Database (`pcm_database_assam.csv`)

The early 25-row prototype used during initial K=4 pipeline exploration is a historical artifact. It must not be confused with the final 58-record production database.

---

## Current Assam Dataset and Version-Control Notes

| Item | Current audited value |
|---|---|
| Active population-weighted points | 80 (`ASP_0001`–`ASP_0080`) |
| Population coverage | 87.8% |
| Daily aggregate rows | 292,160 |
| Daily aggregate date range | 2016-01-02 to 2025-12-31 |
| Final climate regimes | K = 3 |
| Cluster sizes | 27 / 28 / 25 |
| Medoid point IDs | `ASP_0003`, `ASP_0036`, `ASP_0080` |

**Do not carry forward the earlier 129-point / 467,367-row descriptions as current Assam values.** Those counts refer to an earlier pipeline version. When updating this document after future reruns, verify the grid, daily aggregation, and signature outputs together so that point counts and row counts remain consistent.

**Known implementation gap:** daily aggregates and `tier2_signature_assam.csv` are generated, but the current `04b_climate_signature.py` does not consume them. Keep this caveat visible until the implementation is changed and re-audited.
