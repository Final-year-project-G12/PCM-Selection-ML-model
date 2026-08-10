# 15 — Quality Control Audit

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

## Overall QC assessment

The QC layer is well-designed for its stated purpose (data-acquisition sanity-checking, not final
scientific validation) — every threshold is either physically grounded, matched to an upstream clip,
or explicitly acknowledged as implementation-defined rather than falsely presented as
literature-derived. The two genuine, low-cost improvements identified are: (1) tighten or document
the `era5_CSI` dead-check bound, and (2) add matched-timestamp output columns so the already-written
rejection-window QC plot can actually run.
