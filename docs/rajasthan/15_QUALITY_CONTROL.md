# 15 — Quality Control Audit

**Updated 2026-08-11 — this file previously covered only `03_verify_climate_csv.py` and
`03_qc_plots.py`. A second, more consequential quality-control stage
(`03b_quality_check_rajasthan.py`, Phase 2.5) already existed on disk before this update but was
never documented anywhere in `docs/rajasthan/`, despite `04_climate_signature_rajasthan.py`'s own
docstring stating (since 2026-08-11) that Phase 3 now reads that stage's CLEAN output, not
`climate_rajasthan_points.csv` directly. This was the single most factually-wrong gap in the doc set
prior to this update — see `04_PHASE_2_AUDIT.md`'s corrected Dependencies section too.**

## Part 1 — `03_verify_climate_csv.py` (Phase 2, raw-output sanity gate)

Script: `03_verify_climate_csv.py`, six ordered checks, read-only against
`climate_rajasthan_points.csv`, safe to run at any point including mid-download.

## Check 1 — Schema

Verifies presence of all 30 expected columns (`METADATA_COLS + ERA5_COLS + POWER_COLS`). Missing →
**FAIL**. Unexpected extra columns → **WARN**.

## Check 2 — Point coverage

Every `point_id` from `population_grid_points.csv` should appear in the output. Missing points →
**WARN** (correctly reasoned: expected mid-run, not a defect). Extra, unrecognized point_ids →
**FAIL**.

## Check 3 — Row coverage

| Rule | Reason | Threshold | Action | False-positive risk |
|---|---|---|---|---|
| Duplicate `(point_id, date, event)` | data-integrity | any duplicate | **FAIL** | none — duplicates are always a bug |
| Row count per point vs. `suntimes.csv`-implied count | detects partial writes | any mismatch | **WARN** | low — correctly attributed to partial writes, MAX_MATCH_HOURS gaps, or the (possibly stale) 2016-01-01 case |
| Event value outside `{sunrise, noon, sunset}` | schema integrity | any | **FAIL** | none |
| Date outside `[2016-01-01, 2025-12-31]` | scope integrity | any | **WARN** | none |

## Check 4 — Null rates

```
NULL_WARN_THRESHOLD = 0.05   (5%)
NULL_FAIL_THRESHOLD = 0.30   (30%)
```
Per `era5_*`/`power_*` column, null fraction ≥30% → **FAIL**; ≥5% → **WARN**; else OK. Reasonable,
round-number thresholds — not independently derived from a statistical power calculation, but
defensible engineering judgment for a QA gate (the framework doc does not specify these numbers, so
they are implementation-defined, correctly not over-cited).

## Check 5 — Physical sanity (`RANGE_CHECKS`)

| Column | Min | Max | Basis |
|---|---|---|---|
| `era5_T_amb`, `power_T2M` | −5 | 60 °C | matches the pipeline's own clip bound |
| `era5_T_dew` | −30 | 40 °C | QC-only, no upstream clip |
| `era5_RHum`, `power_RH2M` | 0 | 100 % | physical bound |
| `era5_W_spd`, `power_WS10M` | 0 | 40 m/s | QC-only |
| `era5_GHI/DNI/DHI/GHI_clearsky`, `power_ALLSKY/CLRSKY_SFC_SW_DWN` | 0 | 1400 W/m² | matches pipeline clip |
| `era5_LW_down` | 0 | 700 W/m² | QC-only |
| `era5_cloud_cover` | 0 | 1 | physical bound |
| `era5_precipitation` | 0 | 200 mm | QC-only |
| `era5_P_atm` | 800 | 1050 hPa | physical bound for surface pressure |
| `era5_SZA` | 0 | 180 ° | physical bound |
| `era5_solar_azimuth` | 0 | 360 ° | physical bound |
| `era5_CSI` | 0 | **2** | **looser than the pipeline's own [0,1.5] clip — structurally dead check, can never fire** |

Violation severity: >1% of values out-of-range → **FAIL**; else → **WARN**; fully compliant → **OK**.

**The `era5_CSI` bound is worth fixing** (tighten to `[0,1.5]` to match the actual clip, or document
explicitly that it's intentionally loose as a defense-in-depth margin) — as written it can never
contribute a finding, which could mask a real future regression if the upstream clip were ever
accidentally removed.

## Check 6 — Cross-source agreement

```
pairs = [(era5_GHI, power_ALLSKY_SFC_SW_DWN, "GHI"), (era5_T_amb, power_T2M, "temperature")]
```
Requires ≥30 paired non-null rows, else **WARN** ("too few paired rows"). Computes Pearson r and
mean absolute difference. `r < 0.5` → **WARN**; else **OK**. **This check has no FAIL branch at all**
— cross-source disagreement, however severe, can only ever WARN in this script, never fail the whole
QA run. (The actual, more rigorous cross-source decision logic lives in `03b_agreement_analysis.py`,
which does have a MANUAL_REVIEW escalation path — this check in `03_verify_climate_csv.py` is a
lighter-weight, always-safe-to-run sanity gate, not the primary validation mechanism.)

## Exit behavior

`sys.exit(1)` only if any check produced a **FAIL**; WARN-only runs exit 0 with a
"PASSED WITH WARNINGS" message. Cross-source disagreement alone can never fail the script (only
Checks 1–5 have a FAIL path).

## QC visualizations (`03_qc_plots.py`)

Eight self-contained HTML outputs, every one independently skippable (prints `[SKIP]` rather than
crashing if its required input doesn't yet exist — genuinely safe to run mid-download).

| Output | Content | Interpretive logic |
|---|---|---|
| `qc_population_map.html` | Points sized by population, colored by weight, Rajasthan boundary overlay | `YlOrRd_09` colormap |
| `qc_elevation_map.html` | Points colored by elevation, 5-stop green→red diverging scale | NaN elevation flagged specially (black outline); values merely "close to 300m" are **not** flagged, since the code correctly reasons the 300m fallback is never written back to the population CSV |
| `qc_download_status_map.html` | 3-color green/yellow/red completion bucket | ERA5 completion is pipeline-wide (one bbox download), so every point shares the identical ERA5 status; only POWER completion varies per point |
| `qc_population_weight_scatter.html`, `qc_population_histogram.html` | Distributional sanity | 40-bin histograms |
| `qc_elevation_histogram.html`, `qc_elevation_boxplot.html` | Distributional sanity | single-group (Rajasthan-only) boxplot, explicitly noted as not cross-state-comparable without external combination |
| `qc_suntimes_line.html` | UTC decimal-hour of sunrise/noon/sunset, 5 representative longitude-spanning points, 2016–2025 | detects and annotates cross-midnight wraparound as expected behavior, not a bug |
| `qc_download_status_by_year.html` | Completion bar chart per year per source | green/orange/red; "partial" specifically means "not yet attempted" here, a distinct semantic from a true partial-completion state |
| `qc_rejection_window.html` | Requested-vs-matched time offset vs. the 3h threshold | **always skipped** — the combined CSV never carries matched-timestamp columns (see `10_TEMPORAL_PROCESSING.md`) |

## Part 1 overall assessment

The QC layer is well-designed for its stated purpose (data-acquisition sanity-checking, not final
scientific validation) — every threshold is either physically grounded, matched to an upstream clip,
or explicitly acknowledged as implementation-defined rather than falsely presented as
literature-derived. The two genuine, low-cost improvements identified are: (1) tighten or document
the `era5_CSI` dead-check bound, and (2) add matched-timestamp output columns so the already-written
rejection-window QC plot can actually run.

## Part 2 — `03b_quality_check_rajasthan.py` (Phase 2.5, outlier detection + imputation)

**This is a real preprocessing stage, not a QC report — its output (`climate_rajasthan_points_
clean.csv`) is what Phase 3 onward actually consumes.** Two mechanisms:

**1. Hampel filter (median ± n_sigma × MAD × 1.4826), per (point_id, event) series sorted by date,
window measured in occurrences (not hours) — matching this pipeline's 3-events/day schema.**
`HAMPEL_WINDOW_EACH_SIDE=15` occurrences, `HAMPEL_N_SIGMA=3.0`. Applied ONLY to
`era5_T_amb`/`era5_RHum`/`era5_W_spd` — **`era5_GHI`/`era5_CSI` are deliberately excluded**, per a
three-step empirical investigation documented in that script's own module docstring:
  - First pass (±3-occurrence window): 6-7% flag rate, but re-running Phase 3 afterward showed a
    uniform, same-direction shift in `GHI_noon_mean`/`kt_noon_std` across all 320 points — a red flag
    caught by the companion validation script's signature-diff check (see below), not assumed benign.
  - Inspecting actual flagged values showed the filter was winsorizing genuine cloud-driven GHI
    variability (e.g. a real cloudy-day reading of 304.6 W/m² "corrected" to 779.6 W/m²) — not noise.
  - Widening the window to ±15 (matching a Tamil Nadu precedent) made the uniform shift WORSE, not
    better — confirming this was not a window-size problem: a MAD-based outlier filter is
    structurally unsuited to a variable whose legitimate distribution has a heavy, meaningful low
    tail that downstream indices (`cloudy_frac`, `CCI`, `kt_daily_std`, `monsoon_index`) are
    specifically designed to measure.
  - **Final fix: exclude GHI/CSI from Hampel detection entirely**, keeping them only in the
    missingness/imputation pass below. Validated result: GHI/CSI show exactly 0.000000 mean/std
    change; T_amb/RHum/W_spd outlier rates settled at 1.56%/2.20%/2.54% (down from 7%+ in the first,
    over-aggressive pass); maximum resulting signature-level shift across all 320 points dropped from
    1.9 standard deviations (broken run) to 0.37 (final).

**2. Missing-data imputation** (all 5 quality variables): 3-step fallback chain — linear
interpolation for gaps ≤3 occurrences → point-seasonal mean → point-event fallback mean. Rajasthan's
actual missingness was 0.000% across all 5 variables and all 320 points, so this step is present for
robustness/reuse-across-states but did no imputation work on the current data.

**Outputs**: `climate_rajasthan_points_clean.csv` (same schema as the Phase 2 input, plus
`*_outlier_flag` audit columns for the 3 Hampel-checked variables), `quality_report_rajasthan.md`,
`quality_report_rajasthan.json`. Each row/point is fingerprinted (`file_fingerprint()`, mtime+size+
row_count) for reproducibility tracking, matching the convention `provenance_lib.py` later
generalized for the Phase 5→6→7→8 chain (see `21_REPRODUCIBILITY.md`).

**`03b_validate_quality_fix_rajasthan.py`** — independent re-verification, not a rubber stamp: checks
row counts match, re-prints before/after GHI/T_amb stats from the JSON report, backs up the pre-fix
`climate_signature_rajasthan.csv`, re-runs `04_climate_signature_rajasthan.py` via subprocess, and
diffs old vs. new signature values by a std-normalized threshold (`SIGNATURE_DIFF_THRESHOLD_STD=0.05`)
— this diff is exactly what caught the first Hampel over-correction pass described above.

## Part 3 — New QC plot scripts (2026-08-11)

`03c_plots_raw_rajasthan.py` (raw, pre-QC, `outputs/qc_raw_*.html`: point map colored by mean noon
GHI, sun-event profile with error bars, missing-data heatmap, seasonal boxplots, multi-year trend)
and `03b_quality_check_plots_rajasthan.py` (post-QC, `outputs/qc_clean_*.html`: post-clean missing
heatmap, raw-vs-winsorized distribution histograms restricted to the 3 Hampel-checked variables, an
outlier-flag-percentage bar chart with the systematic-issue threshold drawn as a reference line, one
sample point's annual time series with flagged values marked, and a post-clean correlation heatmap)
— both read-only, both explicitly noted in their own docstrings as visualization-only additions that
perform no new computation, reading what `03b_quality_check_rajasthan.py` already produced.

## Combined overall assessment

The now-complete QC/quality-check picture (Part 1 sanity gate → Part 2 outlier/imputation stage →
Part 3 visual QC) is a genuinely strong methodology story for a write-up: Part 2's three-round Hampel
correction (over-aggressive → worse-with-a-wider-window → correctly-scoped-by-excluding-two-
variables) is the same "diagnose → verify empirically → fix → re-verify" shape as the ERA5
deaccumulation bug, arrived at via the same discipline of not trusting a plausible-looking fix without
independently checking its downstream effect (here, via the signature-diff validation script).
