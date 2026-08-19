# 05 — Phase 2.5 / 3 Audit: Quality Control & Climate Signature

## Phase 2.5 — Quality Control (`04_preprocess_assam.py`)

**Status**: COMPLETE

### What it does
Reads `climate_assam_points.csv`, applies per-point QC, and writes one parquet file per point
to `preprocessed/parquet/{point_id}.parquet`.

### Physical bounds checking (Table 9 of plan doc)

| Variable | Lower | Upper |
|---|---|---|
| `era5_GHI` | 0 | 1400 W/m² |
| `era5_T_amb` | -30 | 55 °C |
| `era5_RHum` | 0 | 100 % |
| `era5_T_dew` | -30 | 40 °C |
| `era5_W_spd` | 0 | 50 m/s |
| `era5_P_atm` | 850 | 1060 hPa |
| `era5_cloud_cover` | 0 | 1 |
| `era5_precipitation` | 0 | 200 mm |

Out-of-bounds values are flagged but not deleted (they receive a flag column).

### Outlier detection
- **Algorithm**: IsolationForest (scikit-learn) — an ensemble tree-based method suitable for
  multivariate outlier detection without assuming Gaussianity, appropriate for Assam's heavy-tailed
  monsoon distributions
- **Contrast with Rajasthan**: Rajasthan used Hampel filter (univariate, per-column) and later
  corrected it by excluding GHI/CSI. Assam uses IsolationForest multivariate, which avoids
  that specific failure mode.
- **Policy**: Outliers are **flagged but never deleted** — the flag column is carried through to
  downstream phases but does not remove any row.

### Missing data imputation
- Applied after flagging; imputed values receive an `_imputed` boolean flag column
- Strategy: forward-fill / interpolation within point (specific method in script)

### Output: `preprocessed/parquet/{point_id}.parquet`
- One file per point, 128 files total
- Physical units, QC-passed, outlier-flagged, imputed, no scaling applied

### Known differences from Rajasthan QC

Rajasthan's Phase 2.5 had three sequential corrections (Hampel filter over-corrected GHI/CSI,
fixed by excluding those columns). Assam's IsolationForest approach sidesteps that specific issue,
but does not produce the same set of QC plots (no `qc_clean_*.html`, no `qc_raw_*.html` equivalents
documented).

---

## Phase 3 — Climate Signature Construction (`04b_climate_signature.py`)

**Status**: COMPLETE

### Inputs
- `preprocessed/parquet/{point_id}.parquet` (per-point event data)
- `daily_aggregates_assam.csv` (Tier 2 daily integrals from `02b`)
- `tier2_signature_assam.csv` (aggregated Tier 2 per site from `02b`)
- `population_grid_points.csv` (lat, lon, elevation proxy)

### Outputs
- `climate_signatures_raw.csv` — 128 rows × 18 indices (physical units)
- `climate_signatures_matrix.csv` — PCA-reduced + standardised, ready for clustering
- `pca_loadings.csv` — PCA component loadings for thesis methodology section
- `preprocessed/climate_signature_report.txt` — diagnostic summary

### 18-index signature design

The plan doc requires every index to answer "which PCM property does this constrain, and by what
physical mechanism?" — all 18 satisfy this:

**Thermodynamic block (7 indices → PCA applied):**
`Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `RH_mean`, `elev_proxy`

PCA is applied **only** to this correlated block. The solar and variability indices are kept out
of PCA to preserve interpretability of the dominant signal for PCM selection (solar resource).

**Solar block (4 indices → no PCA):**
`GHI_daily_kWh`, `kt_mean`, `SAI`, `CCI`

**Variability / climate character (5 indices → no PCA):**
`DTR`, `cloudy_frac`, `monsoon_index`, `HSI`, `precipitation_annual`

**Derived targets (2 per site → not in clustering matrix):**
`Tm_target`, `L_required`

### Key design decisions

**Tm_target = 44°C (uniform across all clusters):**
- Derivation: T_delivery = 50°C (Indian domestic standard), ΔT_approach = 6°C → Tm_target = 44°C
- This is uniform for all 128 Assam sites (same as plan §8 for Indian domestic SWH)
- Unlike Rajasthan, there is no per-cluster capping or `Tm_target_capped_C` correction —
  44°C is already well within the 42–70°C feasibility band

**Tsoil_mean ≈ Ta_mean:**
- Soil temperature not downloaded for Assam
- Standard fallback: shallow soil temperature ≈ annual mean surface temperature
- Explicitly stated in `04b_climate_signature.py` docstring; user-approved

**Normalisation timing:**
Applied to the final clustering matrix (zero mean, unit variance across the 128 points), NOT to
the hourly data. This avoids Plan §5.2 Trap 1 (normalising before aggregation).

### Known issues

1. **No separate QC-before vs QC-after comparison**: Rajasthan had `before_phase_3` folder checks.
   Assam proceeds directly from preprocessed parquets to signature construction.

2. **L_required_kJ_per_kg basis**: Uses `T_mains_est_C = Ta_mean − 2.0` (same unsourced offset
   as Rajasthan). This drives `L_required` per site, which determines the feasibility filter's
   latent-heat floor. The −2.0 K offset has no cited source and is a documented caveat inherited
   across all four states.
