# 05 — Phase 2.5 / 3 Audit: Quality Control & Climate Signature (Assam)

**Scripts:** `04_preprocess_assam.py`, `04b_climate_signature.py`

**Dataset basis:** Current audited Assam Objective 1 run — 80 population-weighted points (`ASP_0001`–`ASP_0080`), with 87.8% population coverage.

**Status:** Phase 2.5 / Phase 3 documentation updated to the current 80-point run. Some implementation and output-file details remain to be verified from the current run artifacts; these are marked explicitly below.

---

## Phase 2.5 — Quality Control (`04_preprocess_assam.py`)

### Processing and bounds checking

The preprocessing stage reads the combined Assam physical-climate dataset and applies physical-bound checks and multivariate outlier flagging before climate-signature construction.

| Parameter | Lower bound | Upper bound | Enforcement policy |
|---|---:|---:|---|
| `era5_GHI` | 0 W/m² | 1400 W/m² | Out-of-bounds values flagged; valid extremes should be retained |
| `era5_T_amb` | −30 °C | 55 °C | Bounds verification |
| `era5_RHum` | 0% | 100% | Bounds verification |
| `era5_T_dew` | −30 °C | 40 °C | Bounds verification |
| `era5_W_spd` | 0 m/s | 50 m/s | Bounds verification |
| `era5_P_atm` | 850 hPa | 1060 hPa | Bounds verification |
| `era5_cloud_cover` | 0.0 | 1.0 | Fractional cloud-cover bound |
| `era5_precipitation` | 0 mm | 200 mm | Precipitation bound |

### Multivariate outlier detection

- **Algorithm documented in the Assam audit:** scikit-learn `IsolationForest`.
- **Retention policy documented in the Assam audit:** outliers are flagged rather than deleted; the `is_outlier` flag is carried forward.
- **Current point coverage:** the preprocessing run must correspond to the current 80-point population grid, not the historical 129-point grid.
- **Output partition count and names:** **NOT VERIFIED** against the current output directory. The old claim of exactly 129 parquet files (`ASP_0001.parquet`–`ASP_0129.parquet`) is obsolete and must not be retained.

The bounds and IsolationForest settings above are carried forward from the supplied Assam audit. Their current implementation and output counts should be checked against the present `04_preprocess_assam.py` and generated artifacts before final publication.

---

## Phase 3 — Climate Signature Construction (`04b_climate_signature.py`)

### Purpose

Reduce each location's multi-year climate record to a compact set of climate descriptors for climate-regime clustering and PCM-facing target derivation.

### Current Assam implementation

- The current `04b_climate_signature.py` computes **19 climate-signature indices directly from the event-sampled physical dataset**.
- It does **not** read `daily_aggregates_assam.csv` or `tier2_signature_assam.csv`.
- Therefore, the current Assam implementation does **not** implement a working Tier-1/Tier-2 join inside `04b_climate_signature.py`. Daily aggregates are produced separately by Phase 2, but their presence does not mean they are consumed by the current signature-construction script.
- This is an implementation gap relative to a two-tier signature design and should be described accurately in the methodology.

### Feature groups recorded in the earlier Assam audit

The supplied Assam audit lists the following groups:

- **Thermodynamic block (7):** `Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `RH_mean`, `elev_proxy`.
- **Solar / variability block (12):** `GHI_mean`, `GHI_daily_kWh_est`, `kt_mean`, `kt_std`, `SAI`, `cloudy_frac`, `CCI`, `HSI`, `wind_mean`, `monsoon_index`, `seasonality`, `DTR`.
- **Derived PCM-facing quantities:** `Tm_target`, `T_mains_est`, and `L_required_kWh` are also listed separately in the earlier audit as derived targets, not clustering features.

**Schema-count clarification:** the earlier document says “19 physical indices” but then lists 7 + 12 indices and separately lists 3 derived targets. Keep the 19-index count for the climate-signature feature groups; do not describe the three derived targets as additional members of those 19 indices unless the current CSV header confirms that interpretation.

The earlier audit also states that `precipitation_annual`, `Ta_min_true`, and `Ta_max_true` are absent from the current signature output. Recheck the current output header before reintroducing these variables into the documented schema.

### Standardization and PCA

The earlier Assam audit describes:

- A standardized climate-signature matrix, with normalization performed after temporal aggregation.
- PCA applied to the thermodynamic feature block to inspect covariance without compressing the solar-radiation signal.
- A final five-feature GMM input:
  - `GHI_mean`
  - `Ta_mean`
  - `DTR`
  - `RH_mean`
  - `wind_mean`

The current audited Assam clustering result is:

| Metric | Current audited value |
|---|---:|
| Number of climate regimes | 3 |
| GMM covariance type | Diagonal |
| BIC at K=3 | 1030.97 |
| Bootstrap mean ARI | 0.6939 |
| Bootstrap median ARI | 0.7255 |
| Cluster sizes | 27 / 28 / 25 |
| Medoid point IDs | `ASP_0003`, `ASP_0036`, `ASP_0080` |

These current values replace the historical K=3 result in the old Assam document (BIC 1574.94; bootstrap ARI 0.6289), which was produced using the earlier 129-point dataset / earlier clustering configuration. Do not mix the old and current metrics.

**Feature-configuration verification:** the current GMM result above is the audited result. The exact current feature list, scaling artifact, PCA component count, and PCA loading output should be confirmed against the current `04b_climate_signature.py`, clustering script, and generated CSVs before claiming that every earlier implementation detail remains unchanged.

---

## PCM-Facing Target Derivations

The old Assam audit records a uniform `Tm_target` of 44.0°C, based on a 50°C delivery target and a 6 K heat-exchanger approach. That is a historical design configuration and is not the current audited Objective 1 target basis.

For the current Assam Objective 1 run:

- **Nominal PCM melting target:** 56.0°C.
- **Regime-specific melting-temperature caps:**
  - Cluster 0: 49.659°C
  - Cluster 1: 49.395°C
  - Cluster 2: 49.437°C

These are the current audited values. Do not retain the old 44.0°C uniform target as the current Assam result.

The earlier Assam document also records two different mains-temperature approximations:
- `04b_climate_signature.py`: `max(5.0, Ta_mean − 6.0 K)`.
- `10_physics_validation.py`: `Ta_mean − 2.0 K`.

The inconsistency was not reconciled in the supplied audit. **Current implementation status: NOT VERIFIED.** Do not claim these formulas have been reconciled without checking the current scripts.

---

## Current Dataset Reconciliation

| Item | Current audited Assam basis |
|---|---:|
| Population-weighted grid points | 80 |
| Point ID range | `ASP_0001`–`ASP_0080` |
| Population coverage | 87.8% |
| Climate regimes | 3 |
| Cluster sizes | 27 / 28 / 25 |
| GMM BIC at K=3 | 1030.97 |
| Bootstrap mean ARI | 0.6939 |
| Bootstrap median ARI | 0.7255 |
| Current nominal PCM melting target | 56.0°C |
| Current regime caps | 49.659 / 49.395 / 49.437°C |

### Historical values to retire

The following values in the original Assam file belong to the earlier 129-point run and must not be presented as current:

- 129 spatial coordinates and 129 parquet partitions.
- 129 rows in the raw climate-signature output and 129 rows in the standardized matrix.
- GMM BIC = 1574.94 and bootstrap ARI = 0.6289.
- Uniform `Tm_target` = 44.0°C.

### Open verification items

1. Confirm the current preprocessing parquet partition count and inspect whether all 80 point IDs are represented.
2. Confirm the current raw-signature CSV row count and exact column header; reconcile the 19-index versus derived-target count wording.
3. Confirm the current standardization and PCA configuration from the active code and outputs.
4. Confirm whether the five-feature GMM input list remains unchanged in the current run.
5. Reconcile the mains-temperature formula between signature construction and physics validation.
6. Decide whether the separate daily aggregates / Tier-2 signature should be joined into the climate-signature pipeline. Until implemented, describe the current Assam signature as event-sampled and explicitly note that it does not consume the daily aggregate output.
