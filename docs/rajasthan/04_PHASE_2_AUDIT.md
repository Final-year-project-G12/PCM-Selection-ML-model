# 04 — Phase 2 Audit: Preprocessing, Cross-Source Validation, and Quality Control

**Scope of this file:** Phase 2 (raw combine + cross-source validation) and Phase 2.5 (quality
control + cleaning), combined into a single audit because Phase 2.5 sits directly between Phase 2
and Phase 3 and cannot be understood in isolation from Phase 2's output.

Scripts covered:
- **Phase 2:** `02_combine_rajasthan.py`, `02b_build_daily_aggregates.py`,
  `03_verify_climate_csv.py`, `03_qc_plots.py`, `03b_agreement_analysis.py`,
  `03c_plots_raw_rajasthan.py` (added 2026-08-11, raw QC plots)
- **Phase 2.5:** `03b_quality_check_rajasthan.py`, `03b_validate_quality_fix_rajasthan.py`,
  `03c_plots_raw_rajasthan.py`, `03b_quality_check_plots_rajasthan.py`
- **Preprocessing (alternate/Tamil-Nadu-aligned path):** `04_preprocess_rajasthan.py` — audit stub in §B.11
- **Visualization / interactive QC (not in the core chain):** `03b_coverage_viz_rajasthan.py`,
  `03b_qmap_before_after_viz_rajasthan.py`, `03d_interactive_raw_qa.py`, `04d_postprocess_plots.py`,
  `04e_interactive_postprocess_qc.py` — audit stubs in §B.12

**Cross-references:** `00_MASTER_OVERVIEW.md` ("Current known issues", "Phase 1–8 status at a
glance"). All supporting details now embedded in this file.

**Critical context (documentation history):** Phase 2.5 was implemented on disk (code exists,
script runs, outputs produced) but was entirely undocumented in the `docs/rajasthan/` folder until
2026-08-11, despite Phase 3 (`04b_climate_signature.py`) having explicitly read its CLEAN
output since that same date. **Phase 3 does not read Phase 2's raw output directly** — it reads a
cleaned Phase 2.5 file.

**Updated 2026-09-08 — Phase 2.5 converged onto the Tamil Nadu `04_preprocess` contract.**
`04_preprocess_rajasthan.py` (previously present but unwired — old §B.11 "audit stub") is now
Rajasthan's Phase 2.5 stage: `BOUNDS` physical screen + `SZA ≥ 90°` night-mask + per-season
ERA5→NASA-POWER quantile map **persisted** + Hampel filter + 4-stage/`IterativeImputer` (MICE)
imputation, writing `data/preprocessed/rajasthan_cleaned_physical.csv`. Phase 3
(`04b_climate_signature.py`) now reads that file (its `PHYSICAL_FILE`), pulling the same 8 base
columns via `usecols`. `03b_quality_check_rajasthan.py` (the leaner script Part B below describes)
is retained as a **standalone diagnostic** — it still writes `climate_rajasthan_points_clean.csv` +
`quality_report_rajasthan.{md,json}`, but nothing in `run_all_rajasthan.py`'s core chain reads
them. `03b_validate_quality_fix_rajasthan.py` is **deprecated** (its clean-vs-raw signature diff and
its shell-out to a nonexistent `04_climate_signature_rajasthan.py` no longer apply). Parts B.4 and
B.11 below are kept for history and annotated inline; the "quantile-map persistence" open item in
Parts C/D is now **resolved** (the map is applied in Phase 2.5).

**Pipeline order at a glance (current):**

```
Phase 1 (raw NetCDF/JSON, points, suntimes)
    ↓
Phase 2   — 02_combine_rajasthan.py, 02b_build_daily_aggregates.py,
            03_verify_climate_csv.py, 03_qc_plots.py, 03b_agreement_analysis.py
    ↓  climate_rajasthan_points.csv (RAW, 34 cols)
Phase 2.5 — 04_preprocess_rajasthan.py   (BOUNDS + SZA night-mask + per-season
            quantile map persisted + Hampel + 4-stage/MICE imputation)
    ↓  data/preprocessed/rajasthan_cleaned_physical.csv
Phase 3   — 04b_climate_signature.py     (reads rajasthan_cleaned_physical.csv)

    (diagnostic only, not in the core chain:
     03b_quality_check_rajasthan.py → climate_rajasthan_points_clean.csv)
```

---

# PART A — Phase 2: Preprocessing and Cross-Source Validation

**This is the most scientifically consequential phase in the pipeline.** See §A.8 (this file) for the
full cross-source validation story and §A.3 (this file) for the deaccumulation deep-dive.

## A.1 Purpose

Convert raw NetCDF/JSON into physical-unit, quality-controlled, cross-source-validated tabular data,
and — critically — **decide whether ERA5 alone is defensible as the climate backbone**, before any
downstream index construction touches the physical values.

## A.2 Inputs

`data/raw/era5/points/*.nc`, `data/raw/nasapower/*.json`, `population_grid_points.csv`,
`suntimes.csv` (all from Phase 1).

## A.3 Processing

### ERA5 Accumulated Fields & Deaccumulation — The Critical Bug Fix

**Mandatory audit checkpoint:** The deaccumulation story. This single fix determined whether all
downstream analysis (Phases 3–6) was built on physically valid GHI data.

**What was originally assumed:** ERA5's accumulated fields (`ssrd`, `strd`, `tp`) follow the classic
MARS convention: cumulative since last forecast reset (00Z or 12Z), requiring `diff()` against the
previous hour to recover hourly flux, with special case at post-reset hours (1 and 13). An earlier
function `deaccumulate()` implemented exactly this, with `01_download_era5_rajasthan.py` deliberately
downloading each target hour's predecessor to feed the diff.

**What was actually found:** `03b_agreement_analysis.py` flagged ERA5-vs-POWER GHI as physically
implausible (median ERA5 ~2 W/m² vs POWER ~37 W/m² at same instants, noon Pearson r≈0.01).
Tracing to raw NetCDF showed **34–44% of consecutive-hour raw values were *lower* than their
predecessor within the same accumulation cycle** — impossible for genuine cumulative-since-reset
(which can only increase monotonically until reset). **Conclusion: each hour for this pipeline's CDS
request is already its own ~1-hour accumulated value, not a running total.**

**The fix — `accum_to_flux()`, simple and correct:**
```python
def accum_to_flux(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    return s.clip(lower=0)
```
**No diffing at all.** Stateless clip-to-nonnegative. The function was renamed from `deaccumulate()`
specifically so a future edit would not casually reintroduce a diff step. **Post-fix verification:**
Physics-correct GHI with seasonal peaks (~900 W/m² pre-monsoon, ~700 W/m² monsoon, ~650 W/m²
winter). Solar-noon ERA5-vs-POWER: **MBE=10.95 W/m², RMSE=113.8 W/m², Pearson r=0.810**
(n=1,168,960) — categorical improvement from pre-fix r≈0.01.

**Unit conversion correctness:** `GHI = accum_to_flux(ssrd)/3600` (J/m² → W/m², correct given the
"already per-hour" premise). `LW_down` identical treatment. `precipitation = accum_to_flux(tp)×1000`
(m → mm).

**One unresolved inconsistency:** `avg_sdirswrf` (DNI surrogate) receives `.clip(0)` regardless of
which ERA5 field matched (`msdwswrf`/`fdir`/`msdrswrf`). Only correct if matched field is always a
mean-rate variant — **not independently verified against actual NetCDF variable names.** This is a
plausible unit-error risk and should be checked before DNI is presented as fully validated (see
§A.7 "Unit-consistency caveat" below and `00_MASTER_OVERVIEW.md` known issue 8).

### `02_combine_rajasthan.py` — the merge/physics script

1. Nearest-grid-cell snap (two independent 1-D `argmin`s on lat/lon — correct for a regular grid,
   would not generalize to a curvilinear one) — once per point, not per event.
2. Concatenate each point's full hourly series across all years, apply `accum_to_flux()` (stateless
   clip, **no diffing**) to the accumulated fields, apply unit conversions.
3. Compute solar geometry via `pvlib.location.Location(...).get_solarposition()` and
   `.get_clearsky(model="ineichen")` — see §A.6 below.
4. Derive GHI/DNI/DHI/CSI — see §A.7 below.
5. For each `(point_id, date, event)` row in `suntimes.csv`, nearest-in-time match against both the
   ERA5 series and the NASA POWER series independently, each rejected if farther than
   `MAX_MATCH_HOURS = 3` from the true event time.
6. Apply physical-plausibility bounds (GHI>1400→NaN, T_amb<−5 or >60→NaN, RH clip[0,100], etc.)

### `02b_build_daily_aggregates.py` — Tier-2 daily integrals (NASA POWER only)

`climate_rajasthan_points.csv` has only 3 samples/day — insufficient for true daily energy
integrals. This script reads the already-cached full-hourly NASA POWER JSON directly (no
re-download) and trapezoidally integrates GHI/clear-sky GHI over UTC hour-of-day
(`numpy.trapz`/`trapezoid`, requires ≥2 valid hourly points/day), producing:

- `GHI_daily_kWh`
- `SAI` (confirmed identical to `kt_daily_mean`)
- `kt_daily_mean` / `kt_daily_std`
- `cloudy_frac` (kt < 0.3, an undocumented-elsewhere threshold)
- `CCI` (Pearson r between daily GHI and daily clear-sky GHI, n ≥ 3)
- `HDD18` / `CDD24` (base 18 °C / 24 °C degree-days)
- `DTR_true` (true daily max−min)
- `seasonality` (coefficient of variation of monthly-mean GHI)
- `monsoon_index` (Jun–Sep GHI fraction — a **proxy**, since `PRECTOTCORR` was never downloaded)

### `03_verify_climate_csv.py` and `03_qc_plots.py` — QA

Six ordered checks (schema, point coverage, row coverage, null rates, physical-sanity range checks,
cross-source correlation) — see §B.2 (Part 1: Sanity Checks) below for the full threshold table.
Eight QC visualizations (spatial folium maps + distributional plotly charts) — see §B.3.

### `03b_agreement_analysis.py` — the decision engine

Computes MBE/RMSE/Pearson r for GHI, T_amb, RHum, W_spd, stratified by season × sun-event (80 rows
total), applies a pre-registered three-branch decision rule at solar noon specifically (BACKBONE /
QUANTILE_MAP / MANUAL_REVIEW), and — because the actual data landed in QUANTILE_MAP — fits and
reports (but does not persist back into the dataset) an empirical 100-quantile mapping of ERA5 GHI
onto the POWER distribution, per season. Full numbers and decision text in §A.8 below.

## A.4 Code mapping

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

## A.5 Temporal Processing in the Merge

**Nearest-in-time matching:** For each `(point_id, date, event)` row in `suntimes.csv`, the merge
in `02_combine_rajasthan.py` independently matches ERA5 and POWER timestamps using
`nearest_row(series_df, target_time, max_hours=3)`. A match farther than 3 hours from the true
sun-event instant is rejected, turning missing/sparse readings into `NaN` rather than wrong
pairings. Importantly, ERA5 and POWER can use different actual matched timestamps (e.g., ERA5 up to
3h before the event, POWER up to 3h after) — there is no requirement for cross-source temporal
alignment.

**Gap: unrecorded matched timestamps.** The actual matched time is never persisted; only the
requested `time_utc` appears in `climate_rajasthan_points.csv`. This is a genuine, acknowledged
limitation — adding `era5_matched_time_utc`/`power_matched_time_utc` output columns would both
enable already-written QC diagnostics and let reviewers verify how often sources are paired from
meaningfully different instants. Currently low-cost to fix; currently not fixed. **This gap
propagates into Phase 2.5**, where it structurally disables `03_qc_plots.py`'s rejection-window
diagnostic and forces `03b`'s MANUAL_REVIEW-branch diagnostics onto an SZA-based proxy instead of a
direct time-offset measurement (see §B.3).

**Missing/duplicated timestamp handling:** Duplicated `(point_id, date, event)` combinations are
flagged as hard FAIL in `03_verify_climate_csv.py` Check 3. Missing timestamps become `NaN` rows
(via the 3-hour rejection window). No special-case handling exists for the documented "2016-01-01
edge case" — the referenced mechanism (predecessor-hour dependency in deaccumulation) was fixed
upstream (`accum_to_flux()` is stateless), so this comment is likely a stale reference worth
reconciling against current code before methodology write-up.

## A.6 Solar Geometry (why it's computed this way)

**Solar position algorithm (`get_solarposition`):** Called without explicit `method=` argument,
relying on pvlib's default (likely NREL SPA in current versions). Recommendation: pin method
explicitly before final write-up for reproducibility, or record installed pvlib version.
Sunrise/sunset computation (in Phase 1) *does* explicitly pin `method="spa"`, so that path is
reproducible; solar-position computation should match.

**Clear-sky model (Ineichen):** `get_clearsky(times, model="ineichen")` with default Linke-turbidity
climatology lookup. Standard, defensible choice for this project's scope — a location-specific
measured turbidity record would be excessive burden. Note: Rajasthan's actual aerosol loading (dust
storms) may deviate from climatological default on specific days, affecting `GHI_clearsky` and thus
`CSI` — worth a caveat in methodology, not a required fix.

**Altitude usage:** `alt_m = point_row.elevation_m` if present, else 300 m fallback. Feeds
atmospheric-pressure/airmass assumptions in Ineichen model and a small refraction correction in
solar-position computation. Since Phase 1 now populates real elevation for all 320 points, the
fallback is defensive only.

**Nighttime handling (division-by-zero protection):** `CSI` (clearness index) forced to exactly `0`
(not `NaN`) where `GHI_clearsky ≤ 10` W/m² (nighttime and near-sunrise/sunset where ratio is
numerically unstable). This suppresses an "undefined" ratio into a defined zero — defensible
practical choice (keeps column always numeric), but `CSI=0` in output could mean either "genuinely
clear-sky-free" or "ratio was unstable and suppressed" — not distinguishable from output alone.

## A.7 Solar-Derived Variables (construction & assumptions)

**GHI (Global Horizontal Irradiance):** `GHI = accum_to_flux(ssrd)/3600`, clipped ≥0. This is the
pipeline's most consequential derived variable and the one that surfaced the deaccumulation bug (see
§A.3 above).

**DNI (Direct Normal Irradiance) — two-branch derivation, neither a true decomposition model:**

- Branch 1 (primary): DNI taken directly from ERA5's direct-radiation field
  (`msdwswrf`/`fdir`/`msdrswrf`), not decomposed from GHI. Correctness depends on field unit
  convention matching code assumption.
- Branch 2 (fallback): `DNI = GHI / cos(SZA)` where `cos(SZA) > 0.05`, else `0`. Crude algebraic
  closure (how much direct beam is needed at this sun angle to account for all GHI, if zero
  diffuse) — **not** a genuine decomposition model like DISC/Erbs/DIRINT. Branch 1 likely used
  essentially always (direct radiation field requested unconditionally), so Branch 2 rarely
  exercised, though this was not independently confirmed.

**DHI (Diffuse Horizontal Irradiance) — a closure residual, not independently modeled:**
`DHI = (GHI − DNI·cos(SZA)).clip(0)`. By construction, always exactly satisfies
`GHI = DHI + DNI·cos(SZA)` — it is never independently modeled or observed. Any error in GHI or DNI
propagates entirely into DHI; DHI cannot be used as an independent cross-check on the other two
variables.

**Clearness Index (CSI):** `CSI = GHI/GHI_clearsky`, clipped `[0, 1.5]` in pipeline and forced `0`
below 10 W/m² threshold (see nighttime handling above). Note: the plausibility check in
`03_verify_climate_csv.py` allows `[0, 2]`, which is looser than the actual `[0, 1.5]` clip — makes
that QC check structurally redundant (can never fire). **See §B.2, Check 5, for the identical issue
restated in Phase 2.5's own QC layer.**

**Unit-consistency caveat (open):** `avg_sdirswrf` column-matching logic treats three ERA5 field
names as interchangeable with identical treatment, regardless of field type. `fdir` is accumulated
(would need `/3600` conversion); `msdwswrf`/`msdrswrf` are already mean-rate W/m² (correctly need no
conversion). Audit did not independently verify which name is actually present in downloaded NetCDF
files — `01_download_era5_rajasthan.py` requests `msdwswrf` specifically (already correct), so
practice is likely always hitting the correct path, but code's generality represents latent risk if
variable list changes. Recommend verifying directly before final write-up.

## A.8 Cross-Source Validation Decision (why QUANTILE_MAP was chosen)

**Variable pairs compared:** ERA5 GHI ↔ NASA POWER ALLSKY_SFC_SW_DWN, plus T_amb, RHum, W_spd.

**Matching:** Reuses Phase 2's row-level merge — same point, same `(date, event)`, each source
independently nearest-in-time-matched within 3 hours of true event instant. Note: ERA5 and POWER
can in principle match to different actual instants within that window (matched timestamps never
persisted).

**Decision rule thresholds (evaluated at solar noon only):**

- `BACKBONE` (no correction): r ≥ 0.90 AND |MBE|/mean(POWER GHI) ≤ 5% AND max–min season MBE spread
  ≤ 5%
- `QUANTILE_MAP` (empirical correction): r ≥ 0.70 but stricter conditions fail
- `MANUAL_REVIEW`: r < 0.70 or undefined
- Fixed-weight blending explicitly rejected by design ("no principled derivation for fixed weight
  between independent reanalysis/satellite-derived products")

**Rajasthan result — actual numbers for write-up:**

- Overall: MBE = 6.94 W/m², RMSE = 83.34 W/m², r = 0.9727
- **Solar noon (decision-driving row): MBE = 10.95 W/m², RMSE = 113.79 W/m², r = 0.8102**
- Per-season noon MBE spread: 73.88 W/m² (10% of mean daytime GHI, exceeds 5% gate)
- **Decision: QUANTILE_MAP** — r_noon = 0.8102 fails BACKBONE's ≥0.90 gate but clears the ≥0.70
  floor.
- Quantile mapping fit independently per season on daytime rows; RMSE improved 4/4 seasons, r
  improved 3/4.

**Critical caveat — RESOLVED 2026-09-08.** `03b_agreement_analysis.py` itself still never persists
the correction (it stays read-only/advisory), but Phase 2.5's `04_preprocess_rajasthan.py` Step 2b
now re-fits the same per-season empirical quantile map on daytime rows and writes the corrected
`era5_GHI` + recomputed `era5_CSI` into `rajasthan_cleaned_physical.csv`, which Phase 3 reads. The
map is applied **unconditionally per season** (not gated on the `03b` branch), matching Tamil Nadu.
Phase 3 onward therefore now consumes bias-corrected ERA5 GHI/CSI. Note Tier-2
(`daily_aggregates_*` from `02b`) is NASA-POWER-derived and unchanged — both tiers are now
POWER-aligned rather than one raw-ERA5 and one POWER.

## A.9 Mathematical operations

- RH (Magnus-Tetens): `RH = 100·exp(a·Td/(b+Td)) / exp(a·T/(b+T))`, a = 17.625, b = 243.04.
- MBE: `mean(ERA5 − POWER)` (positive = ERA5 overestimates).
- RMSE: `√mean((ERA5−POWER)²)`.
- Pearson r via pandas `.corr()`.
- Quantile mapping: 101-point empirical quantile-to-quantile piecewise-linear interpolation
  (`np.interp`), fit independently per season on daytime (`ERA5 GHI>0`) rows.

## A.10 Literature support

Alduchov & Eskridge (1996) for the Magnus-Tetens RH coefficients (a=17.625, b=243.04 — standard,
widely-cited values, consistent with the code's own implicit sourcing; not independently verified
against a `sources/` folder entry since this is a meteorological-constants citation, not a
project-domain paper). The framework doc's own §5.1–5.2 directly prescribes the MBE/RMSE/Pearson-r,
season×event stratification, and three-branch decision rule as implemented — the code matches the
spec closely (see §A.8 above for the one-to-one correspondence check).

## A.11 Validation

This phase *is itself* a validation step (that is its purpose) — its own output is validated by the
n≥30-paired-rows gate on quantile-mapping fits (with a printed WARN below that, not a hard stop) and
by `03_verify_climate_csv.py`'s independent cross-source correlation check (Check 6), which is
WARN-only and can never fail the whole QA script on cross-source disagreement alone. **This same
Check 6 recurs, essentially unchanged, as part of Phase 2.5's Part 1 sanity layer — see §B.2.**

## A.12 Outputs

`climate_rajasthan_points.csv`, `daily_aggregates_rajasthan{,_summary}.csv`,
`era5_power_agreement_rajasthan.csv`, `outputs/qc_era5_power_scatter_rajasthan.html`,
`outputs/bias_decision_rajasthan.txt`, 8 QC HTML files.

## A.13 Dependencies

Requires Phase 1's complete point/time/NetCDF/JSON set. Phase 3 does **not** read
`climate_rajasthan_points.csv` directly. **Updated 2026-09-08:** Phase 2.5
(`04_preprocess_rajasthan.py`) reads `climate_rajasthan_points.csv` and produces
`data/preprocessed/rajasthan_cleaned_physical.csv`; Phase 3 (`04b_climate_signature.py`) reads that
file. `daily_aggregates_rajasthan_summary.csv` (from `02b`, untouched by Phase 2.5) is still read
directly by Phase 3. `climate_rajasthan_points.csv` remains the single most-depended-upon RAW
output; the most-depended-upon FINAL input to Phase 3 is now `rajasthan_cleaned_physical.csv`
(previously `climate_rajasthan_points_clean.csv`, from the now-diagnostic
`03b_quality_check_rajasthan.py`).

---

# PART B — Phase 2.5: Quality Control & Data Cleaning

## B.1 Purpose

> **Superseded 2026-09-08.** Part B documents `03b_quality_check_rajasthan.py`, which was Rajasthan's
> Phase 2.5 until the pipeline converged onto the Tamil Nadu `04_preprocess` contract. Phase 2.5 is
> now `04_preprocess_rajasthan.py` (see the top-of-file update box and §B.11), and Phase 3 reads
> `data/preprocessed/rajasthan_cleaned_physical.csv`. `03b_quality_check_rajasthan.py` still runs as
> a standalone diagnostic and still writes `climate_rajasthan_points_clean.csv` +
> `quality_report_rajasthan.{md,json}`, but nothing in the core chain reads them. The rest of Part B
> is kept for history; read it as "what the retired diagnostic does", not "what feeds Phase 3".

Gate Phase 2's output (`climate_rajasthan_points.csv`) through a two-layer quality-check pipeline:
first a read-only sanity check that never modifies data (Part 1), then an actual data-cleaning step
with explicit outlier detection and imputation (Part 2).

**Why this phase exists at all:** Phase 2's cross-source validation (§A) caught the deaccumulation
bug and established which data source to use for GHI. But even valid data can contain rare outliers
(sensor glitches, data-transmission errors, edge cases in interpolation logic). A quality-check
phase *between* raw collection and downstream signature construction ensures Phase 3's climate
indices are built on data that passes both (1) schema/coverage sanity and (2) statistical
plausibility checks. This is standard practice in climate-data pipelines and essential before
deriving anything downstream.

## B.2 Part 1: Sanity Checks (`03_verify_climate_csv.py`)

Six ordered read-only checks against `climate_rajasthan_points.csv`. Safe to run at any time,
including mid-download.

### Check 1 — Schema
Verifies presence of all 30 expected columns. Missing → **FAIL**. Unexpected extras → **WARN**.

### Check 2 — Point coverage
Every `point_id` from `population_grid_points.csv` should appear. Missing → **WARN** (expected
mid-run). Extra/unrecognized → **FAIL**.

### Check 3 — Row coverage

| Rule | Threshold | Action |
|---|---|---|
| Duplicate `(point_id, date, event)` | any duplicate | **FAIL** |
| Row count per point mismatch vs `suntimes.csv` | any mismatch | **WARN** |
| Event value outside `{sunrise, noon, sunset}` | any | **FAIL** |
| Date outside `[2016-01-01, 2025-12-31]` | any | **WARN** |

### Check 4 — Null rates

**Thresholds:** ≥30% → **FAIL**, ≥5% → **WARN**, else OK. Per-column, applied to all `era5_*` and
`power_*` columns. Round-number thresholds (not independently derived from statistical power
calculation), but defensible engineering judgment for a QA gate.

### Check 5 — Physical sanity range checks

| Column | Min | Max | Source |
|---|---|---|---|
| T_amb | −5 | 60 °C | matches pipeline clip |
| T_dew | −30 | 40 °C | QC-only, no upstream clip |
| RHum | 0 | 100 % | physical bound |
| W_spd | 0 | 40 m/s | QC-only |
| GHI/DNI/DHI/GHI_clearsky | 0 | 1400 W/m² | matches pipeline clip |
| LW_down | 0 | 700 W/m² | QC-only |
| cloud_cover | 0 | 1 | physical bound |
| precipitation | 0 | 200 mm | QC-only |
| P_atm | 800 | 1050 hPa | physical bound |
| SZA | 0 | 180 ° | physical bound |
| solar_azimuth | 0 | 360 ° | physical bound |
| CSI | 0 | **2** | **dead check** — looser than pipeline's [0,1.5] clip, can never fire |

**Violation severity:** >1% out-of-range → **FAIL**, else → **WARN**, fully compliant → **OK**.

**Issue flagged:** the CSI check bound should be tightened to [0,1.5] to match the actual clip, or
documented as an intentional defense-in-depth margin — this is the same redundancy noted for CSI in
§A.7, restated here as a concrete QC-check-level finding.

### Check 6 — Cross-source agreement

Pairs: `(era5_GHI, power_ALLSKY_SFC_SW_DWN)`, `(era5_T_amb, power_T2M)`. Requires ≥30 paired
non-null rows; fewer → **WARN**. Computes Pearson r; r < 0.5 → **WARN**, else → **OK**. **No FAIL
branch** — cross-source disagreement can only WARN here (the more rigorous decision logic lives in
Phase 2's `03b_agreement_analysis.py`, §A.8).

## B.3 Part 1b: Visual QC

`03_qc_plots.py` generates 8 interactive HTML visualizations (spatial folium maps + distributional
plotly charts) showing spatial coverage, elevation distribution, data-coverage heatmaps,
distributional histograms per variable and season, and summary statistics. The rejection-window
diagnostic is permanently skipped (with an in-code message) because matched timestamps are never
persisted — see §A.5 for the upstream root cause.

## B.4 Part 2: Actual Data Cleaning (`03b_quality_check_rajasthan.py`)

**Critical design choice:** Only T_amb, RHum, W_spd are outlier-filtered. **GHI and CSI are
deliberately excluded** because they are weather-driven (clouds, clear skies are real, not errors).
A Hampel filter initially over-corrected genuine cloud-driven GHI/CSI variability; this was
identified 2026-08-11 and the solution was to exclude those two variables from outlier detection
entirely.

**Hampel filter:** identifies outliers as points where `|value − median| / (1.4826 * MAD)` exceeds
a threshold (default 3.5 for outlier, 2.5 for winsorizing candidate). Applied per variable, per
season, per point. Over-aggressive filtering detected on GHI/CSI → excluded; remaining application
is correct and defensible.

**Missing-data imputation (as actually coded in `03b_quality_check_rajasthan.py`, correcting an
earlier draft of this section):** *not* MICE / random-forest. Three named, counted operations per
variable, in order: (a) linear interpolation for interior gaps ≤ 3 occurrences within a
`(point_id, event)` series; (b) that **same point's** own `(event, season)` mean for longer gaps
(never a global or cross-point mean); (c) that same point's `(event)`-level mean as a final
fallback. Produces `climate_rajasthan_points_clean.csv` with `HAMPEL_VARS` outliers winsorized to
the local rolling median and gaps filled.

> **Contrast with the current Phase 2.5 (`04_preprocess_rajasthan.py`):** it Hampel-filters
> `era5_GHI` and `era5_cloud_cover` too (→ NaN), and imputes with the Tamil Nadu 4-stage cascade —
> interpolate → ffill/bfill → point/KMeans-zone/global median → `IterativeImputer` (MICE) — which
> **can** pull values across points via the zone median and the iterative model. The GHI/CSI Hampel
> exclusion documented in §B.4/§B.7 and §C is therefore **not** in force under the current stage;
> whether that re-introduces the uniform low-clearness-tail erosion those corrections fixed should
> be checked against the regenerated `climate_signature_rajasthan.csv`.

## B.5 Part 2b: Validation of the Cleaning

`03b_validate_quality_fix_rajasthan.py` re-runs Phase 2.5's own sanity checks (§B.2) against the
cleaned output (`climate_rajasthan_points_clean.csv`), independently verifying that cleaning did
not introduce schema violations or new failures. Confirms the cleaning was safe to apply.

## B.6 Part 2c: Visual QC (Before/After)

`03c_plots_raw_rajasthan.py` and `03b_quality_check_plots_rajasthan.py` generate pre-cleaning and
post-cleaning distributional plots (histograms, box plots, spatial maps), showing what the Hampel
filter changed and justifying the exclusion of GHI/CSI.

## B.7 The weather-vs-error insight

The deliberate exclusion of GHI/CSI from outlier detection reflects a key insight: weather *is*
real and should not be smoothed away. Outliers in solar radiation are clouds; clouds are not
errors. Temperature outliers, by contrast, are likely sensor/transmission errors and *should* be
caught.

## B.8 Inputs

`climate_rajasthan_points.csv` (from Phase 2, §A), `population_grid_points.csv`,
`daily_aggregates_rajasthan_summary.csv` (for seasonal aggregation logic).

## B.9 Outputs

`climate_rajasthan_points_clean.csv` (for Phase 3), `quality_report_rajasthan.{md,json}`
(human-readable + structured report), `outputs/qc_raw_*.html` (8 pre-cleaning plots),
`outputs/qc_clean_*.html` (8 post-cleaning plots), validation confirmation stdout.

## B.10 Dependencies

Requires Phase 2's complete output. Phase 3 (Climate Signature) reads this phase's CLEAN output,
not Phase 2's raw output directly.

## B.11 — `04_preprocess_rajasthan.py` — now the active Phase 2.5

**Status (2026-09-08): this IS Rajasthan's Phase 2.5.** `run_all_rajasthan.py`'s core chain runs
`02b_build_daily_aggregates.py` → **`04_preprocess_rajasthan.py`** → `04b_climate_signature.py` →
`05_cluster_rajasthan.py` → … . `04b_climate_signature.py` reads this script's
`rajasthan_cleaned_physical.csv` (its `PHYSICAL_FILE` constant). The earlier "wiring not
established" status is resolved: it was resolved *by wiring it in*, replacing
`03b_quality_check_rajasthan.py` in the core chain (that script and
`03b_validate_quality_fix_rajasthan.py` now carry SUPERSEDED / DEPRECATED banners).

**What it is:** the Rajasthan port of the Tamil Nadu pipeline's `04_preprocess` step —
functionally near-identical to `era5-tamilnadu/04_preprocess_tamilnadu.py` (same `BOUNDS` dict,
same `SZA ≥ 90°` solar-column zeroing, same Step-2b per-season quantile map persisted, same
4-stage + `IterativeImputer` imputation, same MinMax scaling into a separate `_scaled` artifact),
plus a `SKLEARN_N_JOBS=1` / `LOKY_MAX_CPU_COUNT=1` Windows guard. It reads
`data/processed/climate_rajasthan_points.csv` (`02_combine` output) and writes to
`data/preprocessed/`:

- `rajasthan_cleaned_physical.csv` — physical-unit, QC-passed, imputed rows, **not** scaled (its
  docstring notes that the Phase-3 indices are non-linear functions of physical values, so scaling
  first would corrupt them — matching plan doc §5.2).
- `rajasthan_cleaned_scaled.csv` — same rows, MinMax-scaled feature columns, for later ML/DRL use.
- `scalers.pkl` — per-column `MinMaxScaler`, fit on the first 70 % of chronologically-sorted rows
  only (no leakage).
- Supporting reports observed alongside these on disk: `vif_report.csv`, `yeo_johnson_skew.csv`,
  `correlation_pearson.csv`, `correlation_spearman.csv`, `ghi_quantile_mapping_report.csv`.

**Consistency point — RESOLVED 2026-09-08.** The earlier contradiction (`04_preprocess_rajasthan.py`
docstring said "Phase 3 reads THIS", but `04b_climate_signature.py` read
`climate_rajasthan_points_clean.csv`, and `run_all_rajasthan.py` listed neither correctly) was
settled by making `04_preprocess_rajasthan.py` authoritative:

- `04b_climate_signature.py` now defines `PHYSICAL_FILE = PREPROCESSED_DIR /
  "rajasthan_cleaned_physical.csv"` and reads it (mirroring Tamil Nadu's `04b`).
- `run_all_rajasthan.py`'s `CORE_SCRIPTS` now lists `04_preprocess_rajasthan.py` and
  `04b_climate_signature.py` (the previous entry `04_climate_signature_rajasthan.py` named a file
  that does not exist, so `run_script()` was silently `[SKIP]`-ing Phase 3 in the runner — also
  fixed).
- `03b_quality_check_rajasthan.py` moved to `OPTIONAL_SCRIPTS`; `03b_validate_quality_fix_rajasthan.py`
  removed from the runner.

The `04d`/`04e` post-preprocessing QC scripts already targeted `rajasthan_cleaned_physical.csv`, so
they now sit correctly downstream of the active Phase 2.5 rather than of a side path.

## B.12 — Phase 2 / 2.5 visualization & interactive-QC scripts — audit stubs

None of these are in `run_all_rajasthan.py`'s core chain; all are read-only with respect to their
inputs and exist for visual sanity-checking.

- **`03b_coverage_viz_rajasthan.py`** — two standalone HTML views built from
  `03b_agreement_analysis.py`'s inputs/output: `outputs/spatial_coverage_map.html` (folium; grid
  points coloured by data-completeness % and by mean GHI MBE, ERA5−POWER, with points that would
  independently fail the BACKBONE/QUANTILE_MAP/MANUAL_REVIEW gate marked) and a temporal companion
  view. Never writes back to any input.
- **`03b_qmap_before_after_viz_rajasthan.py`** — re-fits the identical per-season empirical
  quantile mapping of daytime ERA5 GHI onto NASA POWER (`N_QUANTILES=100`, same functions as
  `03b_agreement_analysis.py`) and renders a before/after view (`outputs/
  era5_power_scatter_before_after.html`). Purely diagnostic — like `03b_agreement_analysis.py`
  itself, it changes nothing about the fact that the mapping is never persisted and Phase 3 consumes
  uncorrected ERA5 GHI (see Part C).
- **`03d_interactive_raw_qa.py`** — interactive (Plotly + Folium) raw-data QA, the 1:1 port of the
  Tamil Nadu pipeline's `03b_interactive_raw_qa.py` (same six checks A–F), reading
  `climate_rajasthan_points.csv` as zoomable/hoverable HTML. Sits alongside
  `03c_plots_raw_rajasthan.py` (which covers A–E in Plotly); kept as the structural twin of the
  Tamil Nadu script for cross-region review. Read-only.
- **`04d_postprocess_plots.py`** — static-PNG post-preprocessing QA, the port of Tamil Nadu's
  `04c_postprocess_plots.py`. Runs **after** `04_preprocess_rajasthan.py` on its output
  (`rajasthan_cleaned_physical.csv`): missing-data heatmap (should be ~all-zero), post-cleaning
  distribution sanity, etc. — the cleaned-data counterpart to `03c_plots_raw_rajasthan.py`.
- **`04e_interactive_postprocess_qc.py`** — interactive (Plotly) version of `04d_postprocess_plots.py`
  (checks A, B, D, E, F), reading `rajasthan_cleaned_physical.csv` with a `usecols` filter (the file
  is large). Output to `PLOTSV2/post_preprocess_interactive/*.html`.

---

# PART C — Combined Problems / Risks (both phases)

- **The deaccumulation bug (fixed).** Headline finding of the entire audit — see §A.3 above and
  `00_MASTER_OVERVIEW.md` known issue 1. (Phase 2)

- **Quantile-mapped GHI persistence — RESOLVED 2026-09-08.** Previously `03b_agreement_analysis.py`
  computed the per-season ERA5→POWER quantile map but never wrote it back, so Phase 3 consumed
  uncorrected (deaccumulation-fixed) ERA5 GHI. The new Phase 2.5 stage
  `04_preprocess_rajasthan.py` (Step 2b) re-fits the same map on daytime rows and persists the
  corrected `era5_GHI` + recomputed `era5_CSI` into `rajasthan_cleaned_physical.csv`, applied
  unconditionally per season (matching Tamil Nadu / Uttarakhand). `03b_agreement_analysis.py`
  itself stays read-only/advisory. Downstream numbers (climate signature, clusters, MCDM, physics
  validation, recommendation cards) must be regenerated; see the Phase 3+ audits for the diffed
  results. (Phase 2 → resolved in Phase 2.5.)

- **The "documented 2016-01-01 edge case"** is referenced in three places (`02`'s conceptual
  framing, `03_verify`'s docstring, `03b`'s docstring) but **no code in `02_combine_rajasthan.py`
  actually special-cases it** — the mechanism is implicit (pandas `diff()`-free `accum_to_flux()`
  has no predecessor-hour dependency at all anymore, so the originally-cited edge case may be a
  stale reference from before the deaccumulation fix, when `deaccumulate()` genuinely did need a
  predecessor hour). Worth reconciling this comment against current code before citing it in a
  methodology write-up. (Phase 2)

- **Monsoon-month definition mismatch** between `02_combine_rajasthan.py` (Jun–Aug) and
  `02b_build_daily_aggregates.py` (Jun–Sep) — see `00_MASTER_OVERVIEW.md` known issue 7. (Phase 2)

- **No matched-timestamp columns are ever written** (`era5_matched_time_utc` /
  `power_matched_time_utc`), which structurally disables `03_qc_plots.py`'s rejection-window
  diagnostic (both the Phase 2 and Phase 2.5 instances of this script) and forces `03b`'s
  MANUAL_REVIEW-branch diagnostics to use an SZA-based proxy instead of a direct time-offset
  measurement — low-cost to fix (two extra output columns) if the rejection-window QC is ever
  needed. (Phase 2, propagates into Phase 2.5)

- **CSI plausibility check is structurally redundant** in both its Phase 2 form
  (`03_verify_climate_csv.py`'s `[0,2]` bound vs. the pipeline's actual `[0,1.5]` clip) and its
  restatement in §B.2 Check 5 — same finding, same fix (tighten to `[0,1.5]` or document as
  intentional margin). (Phase 2 / Phase 2.5)

- **Initial Hampel over-correction on GHI/CSI (fixed in `03b_quality_check_rajasthan.py`, then
  RE-OPENED 2026-09-08 by the `04_preprocess` switch).** `03b_quality_check_rajasthan.py` excluded
  GHI/CSI from Hampel filtering after three documented empirical corrections (filtering them
  produced a uniform `GHI_noon_mean`↑ / `kt_noon_std`↓ shift across all 320 points — real
  cloud-driven signal). The current Phase 2.5, `04_preprocess_rajasthan.py`, does **not** carry
  that exclusion — it Hampel-filters `era5_GHI` and `era5_cloud_cover` (→ NaN → imputed), window
  15 occurrences each side. **Action:** after regenerating `climate_signature_rajasthan.csv`,
  diff it against the pre-switch version and check whether `GHI_noon_mean` / `kt_noon_std` /
  `cloudy_frac` / `monsoon_index` move uniformly in one direction across points; if so, decide
  whether to port the GHI/CSI Hampel exclusion into `04_preprocess_rajasthan.py`. (Phase 2.5)

- **MICE missing-data imputation is not perfect.** It reconstructs values based on learned patterns
  in the available data. If an entire season is missing for a point, imputation cannot know what
  the "right" value should be. Check imputation fractions per variable; if any variable has >5%
  imputed rows, investigate manually. (Phase 2.5)

- **No outlier detection on GHI means real sensor failures in GHI might pass through.** By design —
  this is a deliberate choice to preserve weather variability. If a specific point's GHI data is
  suspected to be systematically wrong (not just cloudy), investigate via the visualization outputs
  or manual inspection rather than hoping the QC step catches it. (Phase 2.5)

---

# PART D — Combined Status

**Phase 2 — COMPLETE**, with the deaccumulation fix as a documented, verified correction. The
quantile-mapping persistence decision that was previously open is now settled — see below.

**Phase 2.5 — COMPLETE, restructured 2026-09-08.** Rajasthan converged onto the Tamil Nadu
`04_preprocess` contract: `04_preprocess_rajasthan.py` is now the Phase 2.5 stage
(`BOUNDS` + `SZA ≥ 90°` night-mask + per-season quantile map **persisted** + Hampel +
4-stage/MICE imputation → `data/preprocessed/rajasthan_cleaned_physical.csv`), replacing
`03b_quality_check_rajasthan.py` (now a standalone diagnostic). Part B above documents the retired
diagnostic and is annotated inline.

**Combined open item — CLOSED.** The quantile-mapping correction is now applied in Phase 2.5
(unconditionally per season) and Phase 3 reads the corrected data.

**New follow-up (opened by the restructure):** `04_preprocess_rajasthan.py` Hampel-filters
`era5_GHI` / `era5_cloud_cover`, which `03b_quality_check_rajasthan.py` deliberately did not.
Regenerate `climate_signature_rajasthan.csv` and the Phase 4–8 outputs, diff against the
pre-switch versions, and confirm the GHI low-clearness tail is not being uniformly eroded (the
signature that flagged the original Hampel-on-GHI problem). See Part C.