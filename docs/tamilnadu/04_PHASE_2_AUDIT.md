# 04 — Phase 2 Audit: Preprocessing and Cross-Source Validation

Scripts: `02_combine_tamilnadu.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`, `03b_interactive_raw_qa.py`, `04_preprocess_tamilnadu.py`, `04c_postprocess_plots.py`, `04c_interactive_postprocess_qc.py`.

## Purpose
Combine ERA5 and NASA POWER weather variables at the sun-event instants, compute true daily averages/integrals, perform quality control, and impute missing values.

## Processing Details
1. **Combine Script (`02_combine_tamilnadu.py`)**:
   - Snaps coordinates to the nearest ERA5 grid node, concatenates NetCDFs, applies deaccumulation (with `deaccumulate()`), computes solar geometry (SZA, azimuth, clearsky GHI) via `pvlib`, and merges with NASA POWER hourly data within a 3-hour match window.
2. **Daily Aggregates (`02b_build_daily_aggregates.py`)**:
   - Reads the full hourly NASA POWER series. Integrates GHI trapezoidally to get daily GHI (kWh/m²/day), and calculates true DTR (Tmax - Tmin), HDD18, CDD24, cloudy fraction, and CCI (cloudy runs).
3. **13-Step Preprocessing (`04_preprocess_tamilnadu.py`)**:
   - *Step 1*: Dataset inspection.
   - *Step 2*: Physical validation (replaces out-of-range with NaN; night-masks solar variables to 0.0 when SZA >= 90).
   - *Step 3*: Hampel filter (MAD-based outliers flagged to NaN).
   - *Step 3b*: Yeo-Johnson skew diagnostic (visual check).
   - *Step 4*: Imputation (linear interpolation → ffill/bfill → point median → impute_zone spatial median → global median → MICE fallback).
   - *Step 5*: Temporal validation.
   - *Step 6*: Feature engineering (wind sine/cosine vectors, cloud opacity, temperature depression, daytime flag, solar hour angle).
   - *Step 7*: Lag features (1d, 7d, 30d shift).
   - *Step 8*: Rolling statistics (7d, 30d trailing mean/std).
   - *Step 9*: Delta features (1d difference).
   - *Step 9c*: Drop lag-warmup rows (drops first 30 occurrences).
   - *Step 9b*: Savitzky-Golay smoothing diagnostic (visual check).
   - *Step 10*: Correlation analysis.
   - *Step 11*: Collinearity & VIF diagnostics.
   - *Step 12*: MinMax scaling (train-only fit on first 70% of chronological rows).
   - *Step 13*: Pass/fail QC report gate.

## Critical Audit Findings
1. **Uncorrected Deaccumulation Bug**:
   Unlike Rajasthan (where `deaccumulate()` was replaced with a stateless `accum_to_flux()`), the Tamil Nadu script `02_combine_tamilnadu.py` STILL uses the diff-based `deaccumulate()`. Since the raw NetCDF data contains hourly fluxes, diffing consecutive hours subtracts the fluxes, producing near-zero GHI. This is confirmed by:
   - raw noon GHI Pearson correlation with NASA POWER: **r = 0.3963**
   - raw noon GHI MBE: **-231.89 W/m²** (massive underestimation)
2. **Missing Quantile-Mapping Correction**:
   In Rajasthan, `03b_agreement_analysis.py` ran quantile mapping to correct GHI bias. In Tamil Nadu, no such script exists, and `04_preprocess_tamilnadu.py` does not perform any quantile mapping. The near-zero ERA5 GHI values are propagated directly into downstream steps (like `GHI_mean` in GMM clustering).

## Status
**NEEDS CORRECTION** (Due to the active deaccumulation bug and missing bias correction).
