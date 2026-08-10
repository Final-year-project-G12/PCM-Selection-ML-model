# 03 — Phase 1 Audit: Data Acquisition

True scripts (disk names in parentheses): `00a_build_population_grid.py` (`00b_build_suntimes (3).py`),
`00b_build_suntimes.py` (`00_unzip_accum (3).py`), `01_download_era5_tamilnadu.py`
(`02b_build_daily_aggregates (3).py`), `01b_download_nasapower.py`
(`01_download_era5_tamilnadu (3).py`), `00_unzip_accum.py` (`01b_download_nasapower (3).py`).

## Status: code complete, never executed — no `data/` folder exists under `tamilnadu/`

## Population grid builder

Explicitly documented as method-identical to Rajasthan: same GADM v4.1 boundary source, same
WorldPop 2020 UN-adjusted 100 m raster, same 0.25° grid resolution anchored to ERA5's own origin
(`lat=90.0, lon=-180.0`), same `COVERAGE_TARGET=0.875` (87.5%). State filter:
`NAME_1 ∈ {"TamilNadu"}` (Puducherry excluded by default via `INCLUDE_PUDUCHERRY=False`). Point IDs:
`TNP_{0001..NNNN}` (vs. Rajasthan's `RJP_`). README states this yields **~133 points** — not
independently ground-truthed here since the pipeline has never run.

## Suntimes builder

Identical method to Rajasthan (`pvlib.location.Location.get_sun_rise_set_transit(dates,
method="spa")`, `altitude=0` hardcoded, 2016-01-01..2025-12-31 date range, same
`time_utc`/nominal-`date` distinction for events near UTC midnight). One TN-specific comment worth
noting: *"Tamil Nadu sits close enough to 80°E that sun events land mostly within a single UTC
calendar day (unlike Rajasthan's western edge, which can push an event across midnight UTC) — but
the exact-instant time_utc / nominal date split is kept identical to the Rajasthan script for
consistency and in case a point falls on an unusual edge case near the boundary."` This is a
correctly-reasoned generalization: the cross-midnight-safe `circular_hour_window()` logic is kept
even though it is expected to be exercised less often for Tamil Nadu's longitude range.

## ERA5 downloader

Same design as Rajasthan: sun-event-aligned narrow UTC hour windows (not fixed clock hours), same
`circular_hour_window()` mod-24 algorithm, same `ACCUM_HOURS = INSTANT_HOURS ∪ {predecessor hours}`
construction for the deaccumulation step, same two-call-per-month (instant/accum) CDS request
pattern, same `MAX_RETRIES=3`/`RETRY_WAIT=60s`, same `50_000`-byte file-size validity floor. 10 years
× 12 months × 2 var-types = 240 calls (identical count to Rajasthan; point count differs but doesn't
affect call count since ERA5 downloads are bbox-wide, not per-point).

One documentation imprecision worth noting: the module docstring describes the reset convention as
"00 UTC or 12 UTC," while the actual reset-detection logic implemented downstream (in the combine
script) keys on hours **1 and 13** specifically (the first fully-accumulated hour after each 00Z/12Z
forecast-cycle start) — internally consistent with how ERA5's forecast-type accumulated fields
actually behave, but the docstring's phrasing is imprecise relative to the code.

## NASA POWER downloader

Identical method to Rajasthan: same API endpoint, same 5 parameters, same `MAX_RETRIES=3`,
`RETRY_WAIT=20s`, `REQUEST_SLEEP=1.0s`, `REQUEST_TIMEOUT=60s`, same `1000`-byte validity floor, same
response-content validation (`if not params_out or not any(params_out.values()): raise
RuntimeError(...)`— a genuine data-integrity guard, not silent). Expected call count: 133 points ×
10 years = 1330 (vs. Rajasthan's 3200 for 320 points).

## ERA5 zip-fixer

Byte-identical design to Rajasthan's: magic-byte detection (`PK` for zip, `CDF`/`\x89HDF` for
NetCDF), scans both a legacy grid archive and the points archive, only checks `*_accum.nc` files,
extracts-and-replaces in a temp directory with cleanup in a `finally` block.

## Scientific reasoning

Population-weighted, sun-event-aligned sampling serves the same purpose here as in Rajasthan:
concentrating the climate signature and cluster analysis on where Tamil Nadu's population actually
lives, and sampling at the physically meaningful solar-thermal instants rather than arbitrary clock
hours.

## Literature support

Same as Rajasthan Phase 1 — Reda & Andreas (2004) SPA, Hersbach et al. (2020) ERA5, WorldPop/GADM as
data-source citations, NASA POWER project documentation.

## Validation

None yet possible — no execution has occurred, so no QC report exists to review. The 6 read-only QA
scripts that exist in this pipeline (`03_plots_raw.py`, `03b_interactive_raw_qa.py`, and their
post-clean counterparts) would validate this phase's output once run, mirroring Rajasthan's
`03_qc_plots.py`/`03_verify_climate_csv.py` role, but none have been exercised.

## Outputs (expected, not confirmed)

`population_grid_points.csv` (~133×5 cols — no `elevation_m`, unlike Rajasthan), `suntimes.csv`
(~133 × 3653 days × 3 events ≈ 1,457,547 rows expected by the same formula Rajasthan's row count
matched exactly), `data/raw/era5/points/*.nc` (240 files expected), `data/raw/nasapower/*.json`
(1330 files expected).

## Dependencies

Nothing upstream. Every later phase depends on this phase's point set and sun-event times.

## Problems / risks

- **No elevation attachment step exists** for Tamil Nadu (no `00c`-equivalent script) — every point
  uses a flat 150 m default unconditionally in the combine step, not as a fallback for missing
  per-point data the way Rajasthan's flat-300 m fallback works. This is a real methodological
  simplification, self-documented as acceptable for TN's gentle terrain.
- **Never executed** — every constant, retry policy, and file-size threshold above is a code-level
  design, not a verified-working behavior. The equivalent Rajasthan scripts are known to work
  correctly (240/240 and 3200/3200 files confirmed on disk); Tamil Nadu's analogous scripts have not
  been run even once, so latent bugs that only surface at runtime (e.g., an API parameter rejected by
  the live CDS service, a rate-limit interaction) cannot be ruled out by code reading alone.

## Status

**CODE COMPLETE, NEVER RUN.** Methodologically sound and closely mirrors Rajasthan's already-working
Phase 1 — the main risk is simply the absence of a first real execution to confirm it behaves as
designed against the live APIs.
