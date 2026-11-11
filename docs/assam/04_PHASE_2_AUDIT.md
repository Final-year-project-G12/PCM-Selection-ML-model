# 04 — Phase 2 Audit: Preprocessing & Cross-Source Validation (Assam)

**Script(s)**: `02_combine_assam.py`, `02b_build_daily_aggregates_assam.py`, `03b_agreement_analysis_assam.py`

**Status**: COMPLETE for the current audited 80-point dataset, with a documented preprocessing-control gap.

---

## `02_combine_assam.py` — Climate Merge & Solar Geometry

### Methodology

Aligns the ERA5 reanalysis data with NASA POWER data for the current Assam population-weighted grid, performs physical unit conversions, calculates solar geometry, and generates derived climate variables.

- **Current spatial dataset**: 80 active points, `ASP_0001`–`ASP_0080`.
- **Population coverage**: 87.8%.
- The earlier 129-point description is a superseded dataset version and must not be used to describe current Assam outputs.

### Solar Geometry & Solar Radiation

- **Accumulated-to-flux conversion**: `accum_to_flux()` is used to convert accumulated shortwave radiation (`ssrd`) into a flux representation. The conversion must respect the source accumulation interval and time-boundary handling.
- **`pvlib` integration**: Solar zenith and azimuth angles are calculated for the sampled events. Clear-sky irradiance is obtained using the available ERA5 clear-sky field where present, with the documented pvlib Ineichen fallback.
- **Direct-beam input**: The current Assam data-source description identifies `avg_sdirswrf` as the primary direct-radiation input. Its source field may be `msdwswrf`, `fdir`, or `msdrswrf`; these fields have different physical meanings and units. Verify the actual selected source field and unit conversion before describing this as a uniform DNI derivation.
- **Fallback limitation**: When the direct-radiation field is absent, the documented `GHI / cos(SZA)` fallback is a simplified geometric closure, not a complete physical irradiance decomposition model.

### Climatological Season Mapping

The Assam pipeline defines four seasons:

| Season | Months | Code |
|---|---|---:|
| Winter | Dec, Jan, Feb | 1 |
| Pre-Monsoon | Mar, Apr, May | 2 |
| Monsoon | Jun, Jul, Aug, Sep | 3 |
| Post-Monsoon | Oct, Nov | 4 |

---

## `02b_build_daily_aggregates_assam.py` — Daily Aggregates

### Methodology

The script builds daily aggregate features from the available NASA POWER time series. The intended variables include:

- Daily global horizontal irradiation energy (`GHI_daily_kWh`).
- Diurnal temperature range (`DTR`).
- Daily clearness index (`kt`).
- Cloud fraction.
- Precipitation totals, where the required source variable is available.

**Interpretation caution:** the presence of a daily aggregate output does not by itself prove that every retained date has a complete 24-hour record. Describe these as daily aggregates unless the current script's valid-hour threshold, missing-hour handling, and integration logic have been checked. Do not call every field a complete daily integral without that verification.

### Current Audited Output

- **Dataset**: `data/processed/daily_aggregates_assam.csv`
- **Current audited row count**: **292,160 rows**.
- **Spatial coverage**: 80 points (`ASP_0001`–`ASP_0080`).
- **Date range**: **2016-01-02 to 2025-12-31**.
- **Row-count reconciliation**: 80 points × 3,652 dates = 292,160 rows.
- The earlier **467,367-row** count belongs to the superseded 129-point dataset and must not be reported as the current Assam output.

The pipeline also produces `data/processed/tier2_signature_assam.csv`. Its existence is documented, but do not carry forward the old 129-row count. Confirm the current row count from the actual file before documenting it.

---

## `03b_agreement_analysis_assam.py` — Cross-Source Validation

### Implementation & Decision

The cross-source agreement analysis compares ERA5 daytime GHI against NASA POWER.

- **Reported mean bias error (MBE)**: **1.1%** for daytime GHI.
- **Decision threshold**: if \(|\mathrm{MBE}| \leq 10\%\), ERA5 is accepted as the structural climate-data backbone without requiring empirical quantile mapping on the basis of this agreement test. If the threshold is exceeded, the framework's decision rule calls for considering empirical quantile mapping.
- **Recorded decision**: **`BACKBONE`**, documented in `bias_decision_assam.txt`.

The 1.1% result is the reported audited value. Recheck it if the point set, temporal alignment, or source preprocessing changes.

### Important Downstream Implementation Gap

The `BACKBONE` decision is **not currently controlling the preprocessing correction**:

- `04_preprocess_assam.py` does not read `bias_decision_assam.txt`.
- The script has no conditional branch that skips quantile mapping when the decision is `BACKBONE`.
- Its per-point, per-season empirical quantile-mapping step runs unconditionally and overwrites `era5_GHI`.

Therefore, the 1.1% MBE and `BACKBONE` decision are valid reported findings from the cross-source agreement analysis, but they do **not** mean the downstream preprocessing pipeline currently leaves ERA5 GHI uncorrected. This control-flow issue is documented in `14_ERA5_POWER_VALIDATION.md` and should be resolved or explicitly accepted before describing the decision as operationally enforced.

---

## Current Dataset Reconciliation

| Item | Current audited Assam value |
|---|---:|
| Active population-weighted points | 80 |
| Point ID range | `ASP_0001`–`ASP_0080` |
| Population coverage | 87.8% |
| Daily aggregate rows | 292,160 |
| Daily aggregate date range | 2016-01-02 to 2025-12-31 |
| Reported daytime GHI MBE | 1.1% |
| Recorded agreement decision | `BACKBONE` |

**Historical values to retire from current-state descriptions:** 129 spatial points, 467,367 daily aggregate rows, and 129 Tier-2 rows. These refer to an earlier dataset version. The current Tier-2 row count has not been asserted here because it should be read from the current output file.

**Required follow-up:** make the `BACKBONE` decision operational in `04_preprocess_assam.py` (or document why unconditional quantile mapping is intentionally retained), then rerun and audit the affected preprocessing outputs. Until then, report the agreement result and the downstream implementation gap separately.
