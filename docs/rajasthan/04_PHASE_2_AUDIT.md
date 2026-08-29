# 04 — Phase 2 Audit: Preprocessing and Cross-Source Validation

Scripts: `02_combine_rajasthan.py`, `02b_build_daily_aggregates.py`, `03_verify_climate_csv.py`,
`03_qc_plots.py`, `03b_agreement_analysis.py`, `03c_plots_raw_rajasthan.py` (added 2026-08-11, raw
QC plots). **This is the most scientifically consequential phase in the pipeline** — see
`14_ERA5_POWER_VALIDATION.md` for the full validation story and `09_ERA5_DATA_PIPELINE.md` for the
deaccumulation deep-dive; this file gives the phase-level audit.

**Phase 2.5 note**: `03b_quality_check_rajasthan.py` and `03b_validate_quality_fix_rajasthan.py`
(Hampel-filter outlier winsorizing + missing-data imputation, producing `climate_rajasthan_points_
clean.csv`) sit BETWEEN this phase and Phase 3 — see `15_QUALITY_CONTROL.md` Part 2 for the full
audit. This phase's own output (`climate_rajasthan_points.csv`) is Phase 2.5's INPUT, not Phase 3's
input directly (see the corrected Dependencies section below).

## Purpose

Convert raw NetCDF/JSON into physical-unit, quality-controlled, cross-source-validated tabular data,
and — critically — **decide whether ERA5 alone is defensible as the climate backbone**, before any
downstream index construction touches the physical values.

## Inputs

`data/raw/era5/points/*.nc`, `data/raw/nasapower/*.json`, `population_grid_points.csv`,
`suntimes.csv` (all from Phase 1).

## Processing

### `02_combine_rajasthan.py` — the merge/physics script
1. Nearest-grid-cell snap (two independent 1-D `argmin`s on lat/lon — correct for a regular grid,
   would not generalize to a curvilinear one) — once per point, not per event.
2. Concatenate each point's full hourly series across all years, apply `accum_to_flux()` (see
   `09_ERA5_DATA_PIPELINE.md`) to the accumulated fields, apply unit conversions.
3. Compute solar geometry via `pvlib.location.Location(...).get_solarposition()` and
   `.get_clearsky(model="ineichen")` — see `12_SOLAR_GEOMETRY.md`.
4. Derive GHI/DNI/DHI/CSI — see `13_SOLAR_DERIVED_VARIABLES.md`.
5. For each `(point_id, date, event)` row in `suntimes.csv`, nearest-in-time match against both the
   ERA5 series and the NASA POWER series independently, each rejected if farther than
   `MAX_MATCH_HOURS = 3` from the true event time.
6. Apply physical-plausibility bounds (GHI>1400→NaN, T_amb<−5 or >60→NaN, RH clip[0,100], etc.)

### `02b_build_daily_aggregates.py` — Tier-2 daily integrals (NASA POWER only)
`climate_rajasthan_points.csv` has only 3 samples/day — insufficient for true daily energy
integrals. This script reads the already-cached full-hourly NASA POWER JSON directly (no
re-download) and trapezoidally integrates GHI/clear-sky GHI over UTC hour-of-day
(`numpy.trapz`/`trapezoid`, requires ≥2 valid hourly points/day), producing `GHI_daily_kWh`, `SAI`
(confirmed identical to `kt_daily_mean`), `kt_daily_mean/std`, `cloudy_frac` (kt<0.3, an
undocumented-elsewhere threshold), `CCI` (Pearson r between daily GHI and daily clear-sky GHI, n≥3),
`HDD18`/`CDD24` (base 18°C/24°C degree-days), `DTR_true` (true daily max−min), `seasonality`
(coefficient of variation of monthly-mean GHI), `monsoon_index` (Jun–Sep GHI fraction — a **proxy**,
since `PRECTOTCORR` was never downloaded).

### `03_verify_climate_csv.py` and `03_qc_plots.py` — QA
Six ordered checks (schema, point coverage, row coverage, null rates, physical-sanity range checks,
cross-source correlation) — see `15_QUALITY_CONTROL.md` for the full threshold table. Eight
QC visualizations (spatial folium maps + distributional plotly charts) — see the QC section of
`15_QUALITY_CONTROL.md`.

### `03b_agreement_analysis.py` — the decision engine
Computes MBE/RMSE/Pearson r for GHI, T_amb, RHum, W_spd, stratified by season × sun-event (80 rows
total), applies a pre-registered three-branch decision rule at solar noon specifically (BACKBONE /
QUANTILE_MAP / MANUAL_REVIEW), and — because the actual data landed in QUANTILE_MAP — fits and
reports (but does not persist back into the dataset) an empirical 100-quantile mapping of ERA5 GHI
onto the POWER distribution, per season. Full numbers and decision text in `14_ERA5_POWER_VALIDATION.md`.

## Code mapping

```
02_combine_rajasthan.py
    ↓ accum_to_flux() + apply_unit_conversions()
    ↓ compute_solar() [pvlib]
    ↓ nearest_row() [±3h match]
    ↓
climate_rajasthan_points.csv  (34 columns, 1 row/point/date/event)
```
```
03b_agreement_analysis.py
    ↓ compute_stats() [MBE/RMSE/r]
    ↓ decide_branch() [BACKBONE|QUANTILE_MAP|MANUAL_REVIEW]
    ↓ apply_quantile_mapping() [only if QUANTILE_MAP]
    ↓
era5_power_agreement_rajasthan.csv, bias_decision_rajasthan.txt
```

## Mathematical operations

RH (Magnus-Tetens): `RH = 100·exp(a·Td/(b+Td)) / exp(a·T/(b+T))`, a=17.625, b=243.04.
MBE: `mean(ERA5 − POWER)` (positive = ERA5 overestimates). RMSE: `√mean((ERA5−POWER)²)`. Pearson r
via pandas `.corr()`. Quantile mapping: 101-point empirical quantile-to-quantile piecewise-linear
interpolation (`np.interp`), fit independently per season on daytime (`ERA5 GHI>0`) rows.

## Literature support

Alduchov & Eskridge (1996) for the Magnus-Tetens RH coefficients (a=17.625, b=243.04 — standard,
widely-cited values, consistent with the code's own implicit sourcing; not independently verified
against a Sources/ folder entry since this is a meteorological-constants citation, not a
project-domain paper). The framework doc's own §5.1–5.2 directly prescribes the MBE/RMSE/Pearson-r,
season×event stratification, and three-branch decision rule as implemented — the code matches the
spec closely (see `14_ERA5_POWER_VALIDATION.md` for the one-to-one correspondence check).

## Validation

This phase *is itself* a validation step (that is its purpose) — its own output is validated by the
n≥30-paired-rows gate on quantile-mapping fits (with a printed WARN below that, not a hard stop) and
by `03_verify_climate_csv.py`'s independent cross-source correlation check (Check 6), which is
WARN-only and can never fail the whole QA script on cross-source disagreement alone.

## Outputs

`climate_rajasthan_points.csv`, `daily_aggregates_rajasthan{,_summary}.csv`,
`era5_power_agreement_rajasthan.csv`, `outputs/qc_era5_power_scatter_rajasthan.html`,
`outputs/bias_decision_rajasthan.txt`, 8 QC HTML files.

## Dependencies

Requires Phase 1's complete point/time/NetCDF/JSON set. **Corrected 2026-08-11 — this section
previously said "Everything from Phase 3 onward reads `climate_rajasthan_points.csv` directly,"
which is now factually wrong.** Phase 2.5 (`03b_quality_check_rajasthan.py`) reads
`climate_rajasthan_points.csv` and produces `climate_rajasthan_points_clean.csv`; Phase 3
(`04_climate_signature_rajasthan.py`) reads the CLEAN file, not this phase's raw output directly —
see `15_QUALITY_CONTROL.md` Part 2. `daily_aggregates_rajasthan_summary.csv` (from `02b`, not
touched by the quality-check step) is still read directly by Phase 3. This file
(`climate_rajasthan_points.csv`) remains the single most-depended-upon RAW output in the pipeline,
but it is no longer the most-depended-upon FINAL input to Phase 3 — that is now the Phase 2.5 clean
file.

## Problems / risks

- **The deaccumulation bug** (fixed) is the headline finding of this entire audit — see
  `09_ERA5_DATA_PIPELINE.md` and `20_IMPLEMENTATION_ISSUES.md` item 1.
- **Quantile-mapped GHI is never persisted** — `03b`'s correction is reported (before/after
  diagnostic table) but not written back into `climate_rajasthan_points.csv` or any other dataset
  that Phase 3 reads. **This means Phase 3 onward currently consumes the *uncorrected* (though
  already deaccumulation-fixed) ERA5 GHI values**, not the bias-corrected ones — the quantile-mapping
  result exists only as a methodology-section number, not as an applied correction. This is worth an
  explicit decision: either apply the correction upstream before Phase 3, or explicitly document in
  the methodology write-up that Phase 3+ intentionally uses raw (not bias-corrected) ERA5 GHI and why
  that is still defensible (e.g., because the correction is small relative to the signal at the
  daily/seasonal aggregation level Phase 3 actually uses).
- **The "documented 2016-01-01 edge case"** is referenced in three places (`02`'s conceptual framing,
  `03_verify`'s docstring, `03b`'s docstring) but **no code in `02_combine_rajasthan.py` actually
  special-cases it** — the mechanism is implicit (pandas `diff()`-free `accum_to_flux()` has no
  predecessor-hour dependency at all anymore, so the originally-cited edge case may be a stale
  reference from before the deaccumulation fix, when `deaccumulate()` genuinely did need a
  predecessor hour). Worth reconciling this comment against current code before citing it in a
  methodology write-up.
- **Monsoon-month definition mismatch** between `02` (Jun–Aug) and `02b` (Jun–Sep) — see
  `20_IMPLEMENTATION_ISSUES.md` item 7.
- **No matched-timestamp columns are ever written** (`era5_matched_time_utc`/`power_matched_time_utc`),
  which structurally disables `03_qc_plots.py`'s rejection-window diagnostic and forces `03b`'s
  MANUAL_REVIEW-branch diagnostics to use an SZA-based proxy instead of a direct time-offset
  measurement — low-cost to fix (two extra output columns) if the rejection-window QC is ever needed.

## Status

**COMPLETE — with the deaccumulation fix as a documented, verified correction, and one open
methodological decision** (whether/how to apply the quantile-mapping correction upstream) that
should be resolved and stated explicitly before this phase is cited as final in a methodology write-up.
