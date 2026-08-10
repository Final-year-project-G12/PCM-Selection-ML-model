# 03 — Phase 1 Audit: Data Collection

Scripts: `00a_build_population_grid.py`, `00b_build_suntimes.py`, `00c_attach_elevation.py`,
`01_download_era5_rajasthan.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`.

## Purpose

Establish *where* and *when* to sample climate data, then pull two independent sources (ERA5,
NASA POWER) for exactly those points/times. The "where/when" design choice — population-weighted
points sampled at astronomically-computed sun-event times, instead of a uniform grid on fixed clock
hours — is the pipeline's own stated departure from the more common uniform-grid approach, and it is
the reason every later phase samples 320 points × 3 events/day rather than a full spatial grid ×
24 hours/day.

## Inputs

None upstream — this is the first stage. External: GADM boundary, WorldPop raster, ERA5 CDS API,
NASA POWER API.

## Processing

### Population-weighted sampling grid (`00a_build_population_grid.py`)
1. Download GADM v4.1 India admin-1 boundary, filter to `NAME_1 == "Rajasthan"`.
2. Download WorldPop India 2020 UN-adjusted 100 m population raster, clip to the Rajasthan boundary.
3. Aggregate pixel population onto a **0.25° grid deliberately aligned to ERA5's own grid origin**
   (`lat=90.0, lon=-180.0`) — this is a load-bearing design choice: it guarantees each selected
   sampling point's cell center lands exactly on an ERA5 grid node, so the population→ERA5 mapping
   is 1:1 wherever cells are genuinely distinct, rather than two nearby population cells silently
   collapsing onto the same ERA5 node due to grid misalignment.
4. Rank cells by population descending, keep the minimal set whose cumulative population reaches
   `COVERAGE_TARGET = 0.875` (87.5%, middle of a stated 85–90% target band).
5. `weight = population / population.sum()` — **renormalized over the selected 320-point subset**,
   not the full state population.

Result: **320 points**, `point_id` format `RJP_{0001..0320}`.

### Sun-event times (`00b_build_suntimes.py`)
For every point × every date 2016-01-01..2025-12-31, computes sunrise/solar-noon/sunset via
`pvlib.location.Location.get_sun_rise_set_transit(dates, method="spa")` — Reda & Andreas (2004)
Solar Position Algorithm, no manual equation-of-time code. `altitude=0` is hardcoded for this call
(elevation isn't yet attached to points at this pipeline stage, and even the later-attached
elevation is never fed back into this specific computation — a minor, low-impact omission since
altitude's effect on sunrise/sunset timing itself is negligible, though it does matter for the solar
*position/irradiance* calculations done later in `02_combine_rajasthan.py`, which do use the real
elevation).

Ground-truthed row count: **3,506,880** = 320 points × 3653 days (2016–2025, including leap years
2016/2020/2024) × 3 events — matches the formula exactly.

### Elevation attachment (`00c_attach_elevation.py`)
Downloads ERA5's time-invariant geopotential field (`z`), one API call for a single date/time
(orography doesn't change), and attaches `elevation_m = z / 9.80665` per point via nearest-neighbor
lookup on the geopotential grid. Replaces a flat 300 m fallback that `02_combine_rajasthan.py`
otherwise uses. Sanity-checks outliers against `[−420, 8850]` m (Dead Sea to Everest) but does not
clip or drop them — only warns.

### ERA5 download (`01_download_era5_rajasthan.py`)
Downloads three narrow UTC hour windows per month (sunrise/noon/sunset ± margin) instead of fixed
clock hours, using a **circular (mod-24) window algorithm** to correctly handle sun events that
straddle the UTC midnight boundary (documented real case: an eastern point's summer sunrise can land
at 23:55 UTC of the *previous* calendar date). Two API calls per (year, month): instant variables
(analysis type) and accumulated variables (forecast type, with each instant hour's immediate
predecessor also requested — needed for the deaccumulation step, see `09_ERA5_DATA_PIPELINE.md`).
10 years × 12 months × 2 var-types = 240 calls.

### NASA POWER download (`01b_download_nasapower.py`)
Full hourly year, per point, for the 5 parameters listed in `02_DATA_SOURCES_AND_VARIABLES.md`.
320 points × 10 years = 3200 calls.

### Zip-quirk fix (`00_unzip_accum.py`)
CDS API v2 sometimes returns a ZIP archive even when `download_format: unarchived` is requested;
this detects (`PK` magic bytes) and fixes `*_accum.nc` files in place, scanning both the legacy
full-grid archive and the new points archive.

## Scientific reasoning

Population-weighting the sampling grid (rather than uniform spatial sampling) directly serves the
project's downstream deliverable: a climate signature and PCM recommendation that is meaningful for
*where people actually live*, not for empty desert cells that would otherwise dilute a uniform
average. Sun-event-aligned sampling (rather than fixed clock hours) is the correct choice for a
solar-thermal application specifically because the physically meaningful instants — when charging
starts (sunrise), peak charging (noon), and when discharge begins (sunset) — are what the downstream
Tier-1 climate signature and Tm_target/L_required derivations are actually built from.

## Literature support

Reda & Andreas (2004), "Solar position algorithm for solar radiation applications," *Solar Energy*
76(5) — cited by name in `00b`'s docstring as the algorithm pvlib's `method="spa"` implements.
Hersbach et al. (2020), "The ERA5 global reanalysis," *QJRMS* 146(730) — the ERA5 product's own
citation (per the framework doc's §15 reference list; not separately re-verified in this pass beyond
confirming the framework doc names it). WorldPop and GADM are cited as data-source products, not
peer-reviewed methodology claims.

## Validation

`03_verify_climate_csv.py` Check 2 (point coverage) and Check 3 (row coverage) validate this phase's
output indirectly, downstream, in Phase 2. No dedicated Phase-1-only validation script exists;
`03_qc_plots.py`'s population/elevation/download-status maps serve this role.

## Outputs

`population_grid_points.csv` (320×6 cols incl. `elevation_m`), `suntimes.csv` (3,506,880×4 cols),
`data/raw/era5/points/*.nc` (240 files, 816 MB), `data/raw/nasapower/*.json` (3200 files, 2.47 GB),
`download_status_points.csv`, `download_status_power.csv`.

## Dependencies

Nothing upstream. Every later phase depends on this phase's point set and sun-event times being
fixed — re-running `00a`/`00b` with different parameters would silently invalidate every downstream
file without an automatic re-trigger (no dependency-graph enforcement exists in this pipeline; it is
a linear script-order convention, not a build system).

## Problems / risks

- **No re-verification of stale outputs**: `00a`'s population-grid CSV is unconditionally
  recomputed and overwritten on every run (no skip logic), but nothing downstream detects if the
  point set changed since `suntimes.csv`/ERA5/POWER were built against an older version — a silent
  point-set/downstream-data mismatch is possible if `00a` is re-run without re-running the entire
  chain after it.
- **Single static 2020 population snapshot** used to weight sampling across the whole 2016–2025
  study period — explicitly documented as a standard, accepted simplifying assumption (population
  distribution shifts far slower than weather), not a bug.
- **`00b`'s altitude=0 hardcoding** for the SPA sunrise/sunset computation is a minor inconsistency
  with the elevation-aware solar geometry used later in `02_combine_rajasthan.py`, though its
  practical effect on sunrise/sunset clock time is negligible.
- **Ground-truth confirms full completion**: 240/240 ERA5 files, 3200/3200 (after 1 retry) POWER
  files — no incomplete-download risk currently outstanding for Rajasthan.

## Status

**COMPLETE.**
