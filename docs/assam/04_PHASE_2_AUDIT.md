# 04 — Phase 2 Audit: Preprocessing & Cross-Source Validation

**Script(s)**: `02_combine_assam.py`, `02b_build_daily_aggregates_assam.py`, `03b_agreement_analysis_assam.py`

**Status**: COMPLETE (Authoritative Final)

---

## `02_combine_assam.py` — Climate Merge & Solar Geometry

### Methodology
Aligns the hourly ERA5 NetCDF reanalysis series with the NASA POWER dataset across all 129 points, performing physical unit conversions, solar position calculations, and derived variable generation.

### Solar Geometry & Solar Radiation
- **Stateless Flux Extraction**: Applies `accum_to_flux()` to convert accumulated shortwave radiation (`ssrd`) into physically valid hourly flux without numerical degradation.
- **`pvlib` Integration**: Computes astronomical solar zenith and azimuth angles per event, evaluating clear-sky irradiance via the Ineichen model to establish the Clear-Sky Index ($CSI = GHI / GHI_{\text{clearsky}}$).
- **Direct Beam Decomposition**: Direct Normal Irradiance (DNI) is derived via standard geometric decomposition.

### Climatological Season Mapping
Reflecting the agro-climatic reality of Northeast India, the Assam pipeline defines four distinct seasons:
- **Winter (Code 1)**: Dec, Jan, Feb
- **Pre-Monsoon (Code 2)**: Mar, Apr, May
- **Monsoon (Code 3)**: Jun, Jul, Aug, Sep (4-month duration, reflecting prolonged monsoon rainfall)
- **Post-Monsoon (Code 4)**: Oct, Nov

---

## `02b_build_daily_aggregates_assam.py` — Daily Integrals

### Methodology
Processes the full NASA POWER series to calculate un-aliased daily integral statistics for each coordinate:
- True daily global horizontal irradiation energy ($GHI_{\text{daily,kWh}}$).
- True Diurnal Temperature Range ($DTR = T_{\max} - T_{\min}$).
- Daily clearness index ($k_t$), cloud fraction, and precipitation totals.

### Authoritative Output
- **Dataset**: `data/processed/daily_aggregates_assam.csv`
- **Audited Row Count**: **467,367 daily rows** across the 129 spatial coordinates over 2016–2025.
  *(Note: Days with $<20$ valid hourly observations from NASA POWER were dropped to eliminate gap-induced distortion; 467,367 is the exact verified count).*
- **Tier 2 Output**: `data/processed/tier2_signature_assam.csv` (129 rows, one per site, containing aggregated Tier-2 climatological indices).

---

## `03b_agreement_analysis_assam.py` — Cross-Source Validation

### Implementation & Decision
A dedicated cross-source agreement script (`03b_agreement_analysis_assam.py`) was executed to compare ERA5 daytime GHI against NASA POWER:
- **Mean Bias Error (MBE)**: The mean bias across Assam's daytime solar radiation was determined to be **1.1%**.
- **Decision Rule**: The framework specifies that if $|\text{MBE}| \le 10\%$, reanalysis data is accepted directly as the structural backbone without artificial empirical distortion; if $>10\%$, empirical quantile mapping is triggered.
- **Authoritative Decision**: **`BACKBONE`** (documented in `bias_decision_assam.txt`).
- **Downstream Impact — correction**: `04_preprocess_assam.py` never reads `bias_decision_assam.txt`
  and has no conditional on it. Its own per-(point, season) empirical quantile-mapping step runs
  **unconditionally** and always overwrites `era5_GHI`, regardless of the BACKBONE decision. See
  `14_ERA5_POWER_VALIDATION.md` for detail. The BACKBONE decision is a valid, favorable finding on
  its own terms (1.1% MBE), but it is not currently wired into whether the correction is applied.
