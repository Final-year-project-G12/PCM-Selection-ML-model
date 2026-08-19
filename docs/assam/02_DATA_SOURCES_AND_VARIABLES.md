# 02 — Data Sources and Variables

## Primary data sources

### ERA5 (ECMWF Reanalysis v5)
- **Provider**: Copernicus Climate Data Store (CDS), accessed via `cdsapi`
- **Product**: ERA5 hourly data on single levels
- **Spatial resolution**: 0.25° × 0.25° native grid
- **Temporal coverage**: 2016–2025 (10 years)
- **Sampling strategy**: Population-weighted points (128 points, 87.5% Assam coverage)
- **Download format**: NetCDF, sun-event-aligned hours (sunrise, solar noon, sunset)
- **ERA5 variables downloaded**:

| Variable | ERA5 parameter | Unit | Notes |
|---|---|---|---|
| Solar radiation (GHI proxy) | `ssrd` (surface solar radiation downwards) | J/m² | Accumulated; per-hour flux from CDS |
| Thermal radiation | `strd` (surface thermal radiation downwards) | J/m² | Accumulated |
| 2m temperature | `t2m` | K → °C | Instantaneous |
| 2m dewpoint | `d2m` | K → °C | Used for RH derivation |
| 10m U-wind | `u10` | m/s | |
| 10m V-wind | `v10` | m/s | Combined with U for speed/direction |
| Mean sea level pressure | `msl` | Pa → hPa | |
| Total cloud cover | `tcc` | 0–1 fraction | |
| Total precipitation | `tp` | m → mm | |
| Mean surface direct shortwave radiation | `avg_sdirswrf` | W/m² | Direct-radiation surrogate |

### NASA POWER
- **Provider**: NASA Langley Research Center Prediction of Worldwide Energy Resources
- **Product**: Daily data (ALLSKY_SFC_SW_DWN, T2M_MAX, T2M_MIN, RH2M, WS2M, PRECTOTCORR)
- **Temporal coverage**: 2016–2025 (10 years, same as ERA5)
- **Role**: Independent cross-validation against ERA5 solar radiation; daily aggregates for
  Tier 2 signature indices (CCI, SAI, kt, etc.)
- **Point IDs**: `ASP_0001` through `ASP_0129` (128 active points; IDs up to 0129 due to
  boundary rejection of some grid cells)

### Population raster
- **Source**: WorldPop unconstrained global mosaic, India, UN-adjusted, 100m, 2020
- **URL**: `https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_UNadj.tif`
- **Use**: Aggregated to 0.25° ERA5 grid cells; top cells by population selected to reach 87.5%
  coverage of Assam's total estimated population

### State boundary
- **Source**: GADM v4.1, India administrative level 1
- **URL**: `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json`
- **Filter**: `NAME_1 == "Assam"`

## Derived variables (computed in `02_combine_assam.py`)

| Variable | Derivation | Physical meaning |
|---|---|---|
| `RHum` | Magnus formula from T2m + Td | Relative humidity (%) |
| `W_spd` | √(u10² + v10²) | Wind speed (m/s) |
| `W_dir` | atan2(u10, v10) | Wind direction (degrees) |
| `DNI` | pvlib decomposition from GHI, DHI | Direct Normal Irradiance |
| `CSI` | GHI / GHI_clearsky (pvlib Ineichen) | Clear-sky index (0–1) |
| `elevation_m` | Default 100m (Assam valley baseline) | Used in P_atm approximation |

## Season classification (`02_combine_assam.py`)

| Month | Season | Code |
|---|---|---|
| Dec, Jan, Feb | Winter | 1 |
| Mar, Apr, May | Pre-Monsoon | 2 |
| Jun, Jul, Aug, Sep | Monsoon | 3 |
| Oct, Nov | Post-Monsoon | 4 |

Note: Monsoon spans 4 months (Jun–Sep) for Assam, reflecting the longer monsoon season
relative to Rajasthan's 3-month definition (Jun–Aug).

## Climate signature variables (18 indices, `04b_climate_signature.py`)

### Tier 1 — Sun-event statistics (from ERA5 hourly, event-aligned)

| Index | Description |
|---|---|
| `Ta_mean` | Annual mean ambient temperature |
| `Ta_p95` | 95th percentile ambient temperature (hot design day) |
| `Ta_p05` | 5th percentile ambient temperature (cold design day) |
| `HDD18` | Heating degree days (base 18°C) |
| `CDD24` | Cooling degree days (base 24°C) |
| `RH_mean` | Annual mean relative humidity |
| `GHI_daily_kWh` | Mean daily GHI (kWh/m²/day) |
| `DTR` | Diurnal Temperature Range (Ta_p95_noon − Ta_p05_night) |
| `HSI` | Humidity-Solar Interaction index (RH_mean × GHI_daily) — **load-bearing for corrosion veto** |

### Tier 2 — Daily-integral indices (from NASA POWER daily data, `02b` output)

| Index | Description |
|---|---|
| `kt_mean` | Annual mean clearness index |
| `cloudy_frac` | Fraction of days with kt < 0.4 |
| `monsoon_index` | Fraction of annual rainfall in Jun–Sep |
| `CCI` | Cloud Cover Index |
| `SAI` | Solar Availability Index |
| `precipitation_annual` | Annual accumulated precipitation (mm/yr) |
| `Ta_min_true` | True annual minimum temperature |
| `Ta_max_true` | True annual maximum temperature |
| `elev_proxy` | Elevation proxy (atmospheric pressure-derived) |

### Tsoil_mean — stated approximation

Soil temperature was not downloaded for Assam. `Tsoil_mean` is approximated as `Ta_mean`
(standard fallback for shallow soil temperature in the absence of measured data). This is
explicitly documented in `04b_climate_signature.py`'s docstring and was user-approved.
Any downstream quantity that depends on this (mains-temperature estimate) carries this
approximation as an inherited caveat.

## PCM property database (`06_build_pcm_database.py`)

- **Base**: MICE+RF+PMM cleaned manufacturer dataset
  (`PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv`) — 18 manufacturer rows
- **Literature additions**: 7 rows from Singh2025 (Table 2 — fatty acids, eutectics, paraffins
  in the 42–70°C band)
- **Total**: 25 rows in `pcm_database_assam.csv`
- **Target band**: 42–70°C (Tm absolute bounds in feasibility filter)
- **Derived properties**: `rho_H_MJ_m3` (volumetric latent heat), `supercooling_K`, `TC_W_mK`
  (average of liquid and solid conductivity), `cycles_confidence` (log-scaled)
