# 04 — Phase 2 Audit: Preprocessing and Cross-Source Validation

Scripts: `02_combine_tamilnadu.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`, `03b_agreement_analysis.py`, `03b_interactive_raw_qa.py`, `04_preprocess_tamilnadu.py`, `04c_postprocess_plots.py`, `04c_interactive_postprocess_qc.py`.

## Purpose
Combine ERA5 and NASA POWER weather variables at the sun-event instants, compute true daily averages/integrals, perform quality control, and impute missing values.

## Processing Details
1. **Combine Script (`02_combine_tamilnadu.py`)** — **v3.1 corrected**:
   - Snaps coordinates to the nearest ERA5 grid node, concatenates NetCDFs, applies `accum_to_flux()` (stateless clip — NOT diff-based deaccumulation), computes solar geometry via `pvlib`, and merges with NASA POWER within a 3-hour match window.
2. **Daily Aggregates (`02b_build_daily_aggregates.py`)**:
   - Reads full hourly NASA POWER series. Integrates GHI trapezoidally to daily kWh/m²/day; calculates DTR, HDD18, CDD24, cloudy fraction, CCI.
3. **Cross-Source Agreement (`03b_agreement_analysis.py`)** — **NEW v3.1**:
   - Stratified MBE/RMSE/Pearson-r table; decision gate (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW); GHI scatter by season.
4. **13-Step Preprocessing (`04_preprocess_tamilnadu.py`)** — **v3.1 corrected**:
   - Steps 1–13 unchanged (inspection, physical validation, Hampel, imputation, features, lags, scaling, QC gate).
   - **Step 2b (NEW)**: Per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution; saves `ghi_quantile_mapping_report.csv`.

## Corrected Audit Findings (v3.1)
1. **Deaccumulation Bug — FIXED**:
   - `02_combine_tamilnadu.py` now uses `accum_to_flux(s) = s.clip(lower=0)`.
   - Pre-fix stats (for reference): noon GHI r = 0.3963, MBE = −231.89 W/m². Post-fix expected: r > 0.80 (Rajasthan reference: r = 0.8102).
2. **Quantile-Mapping — FIXED**:
   - Step 2b in `04_preprocess_tamilnadu.py` applies per-season QM after physical validation.
   - `03b_agreement_analysis.py` documents the cross-source decision branch.

## Status
**COMPLETE (v3.1 fixes applied — re-run `02_combine` → `04_preprocess` for updated outputs)**

## Literature Support
| Method | Reference | Source |
|---|---|---|
| ERA5 vs satellite GHI validation | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Quantile mapping / bias correction | Mansouri et al. (2025) | `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
| Hampel MAD outlier detection | Standard QC practice | `15_QUALITY_CONTROL.md` |
| MICE imputation | Rubin (1987); sklearn IterativeImputer | `15_QUALITY_CONTROL.md` |
