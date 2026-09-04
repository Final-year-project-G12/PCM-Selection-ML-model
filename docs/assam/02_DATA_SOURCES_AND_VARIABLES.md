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
| Mean sea level pressure | `msl` | Pa → hPa | Surface atmospheric pressure |
| Total cloud cover | `tcc` | 0–1 fraction | Fractional cloud cover |
| Total precipitation | `tp` | m → mm | Accumulated precipitation |
| Mean surface direct solar | `avg_sdirswrf` | W/m² | Direct radiation component |

### NASA POWER
- **Provider**: NASA Langley Research Center Prediction of Worldwide Energy Resources
- **Product**: Hourly and daily aggregates (`ALLSKY_SFC_SW_DWN`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PRECTOTCORR`)
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
| `DNI` | pvlib decomposition from GHI, DHI | Direct Normal Irradiance |
| `CSI` | $GHI / GHI_{\text{clearsky}}$ (pvlib Ineichen) | Clear-sky index (0–1) |
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

## Climate Signature Structure (18 Indices, `04b_climate_signature.py`)

### Tier 1 — Sun-Event Statistics (ERA5 Event-Aligned)
- `Ta_mean`: Annual mean ambient temperature (°C)
- `Ta_p95`: 95th percentile ambient temperature (°C)
- `Ta_p05`: 5th percentile ambient temperature (°C)
- `HDD18`: Heating degree days (base 18°C)
- `CDD24`: Cooling degree days (base 24°C)
- `RH_mean`: Annual mean relative humidity (%)
- `GHI_daily_kWh`: Mean daily GHI (kWh/m²/day)
- `DTR`: Diurnal Temperature Range ($Ta_{\text{noon}} - Ta_{\text{sunrise}}$)
- `HSI`: Humidity-Solar Interaction index ($RH_{\text{mean}} \times GHI_{\text{daily}}$)

### Tier 2 — Daily-Integral Indices (NASA POWER Hourly/Daily)
- `kt_mean`: Annual mean clearness index
- `cloudy_frac`: Fraction of days with $k_t < 0.4$
- `monsoon_index`: Fraction of annual rainfall occurring in Jun–Sep
- `CCI`: Cloud Cover Index
- `SAI`: Solar Availability Index
- `precipitation_annual`: Annual accumulated precipitation (mm/yr)
- `Ta_min_true`: True annual minimum daily temperature (°C)
- `Ta_max_true`: True annual maximum daily temperature (°C)
- `elev_proxy`: Atmospheric pressure proxy for elevation

*Note on Soil Temperature*: In the absence of measured shallow soil temperatures, $T_{\text{soil,mean}} \approx T_{a,\text{mean}}$ was adopted as a documented physical fallback.

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
