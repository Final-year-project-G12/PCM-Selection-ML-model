# ERA5 Tamil Nadu Pipeline (population-weighted points, sun-event-aligned, 10-year)

Same method as the Rajasthan pipeline, applied to Tamil Nadu, so both
regions' outputs are directly comparable for the cross-region clustering
step (Objective 1). The pipeline now runs end-to-end through Phase 8
(recommendation cards) — see `docs/tamilnadu/00_MASTER_OVERVIEW.md`
for the full phase-by-phase status and `docs/tamilnadu/CHANGELOG.md`
for what changed and when, most recently the 2026-09-16 pass (elevation
integration, a corrected `TM_TARGET_C`, and a k=5→k=3 clustering revert —
see that entry before quoting any number from an older doc).

All other project documentation — the changelog, run guide, per-script
file guide, preprocessing walkthrough, cleanup audit trail, and the
Objective 1/2 design-spec docs — lives in `docs/tamilnadu/`. This
README is the only `.md` file kept at the repo root.

## Pipeline overview (Phase 1 — data collection)

```
00a_build_population_grid.py   →  data/processed/population_grid_points.csv
00b_build_suntimes.py          →  data/processed/suntimes.csv
00c_attach_elevation.py        →  population_grid_points.csv gains elevation_m
01_download_era5_tamilnadu.py  →  data/raw/era5/points/*.nc
01b_download_nasapower.py      →  data/raw/nasapower/*.json
00_unzip_accum.py              →  (fixes zip-disguised-as-.nc files in place)
02_combine_tamilnadu.py        →  data/processed/climate_tamilnadu_points.csv
```

Phase 1 is only the start of the pipeline — it continues through
preprocessing (Phase 2), climate signatures (Phase 3), GMM clustering
(Phase 4), PCM feasibility filtering (Phase 5), MCDM ranking (Phase 6),
physics validation (Phase 7), and recommendation cards (Phase 8). See
`docs/tamilnadu/00_MASTER_OVERVIEW.md` for the complete phase map and
`docs/tamilnadu/RUN_TAMILNADU_PIPELINE.md` for the full run guide.

## Run order

**Whole pipeline, one command** (recommended):
```powershell
python run_all_tamilnadu.py                 # core pipeline (Phases 2-8; raw downloads excluded by default)
python run_all_tamilnadu.py --include-setup  # also runs Phase 1's raw-data downloads first
python run_all_tamilnadu.py --with-optional  # also runs diagnostic/QC plotting scripts
```
See `docs/tamilnadu/RUN_TAMILNADU_PIPELINE.md` for the full flag
reference (`--from`, `--dry-run`, etc.) and estimated run times per stage.

**Manual, Phase 1 only** (if you just want the raw data collection step):
```
python 00a_build_population_grid.py
python 00b_build_suntimes.py
python 00c_attach_elevation.py
python 01_download_era5_tamilnadu.py
python 01b_download_nasapower.py
python 00_unzip_accum.py
python 02_combine_tamilnadu.py
```

Each script is resumable — safe to Ctrl-C and re-run.

## What's different from the old (222-city, fixed-hour, 2-year) TN pipeline

| | Old pipeline | This pipeline |
|---|---|---|
| Sampling locations | 260+ hand-picked named cities | population-weighted 0.25° grid cells covering ~87.5% of state population |
| Time sampling | fixed hourly, all 24h/day | 3 events/day: sunrise, solar noon, sunset (astronomically computed per point/date via pvlib) |
| Study period | 2024–2025 (2 years) | 2016–2025 (10 years) |
| Cross-check source | ERA5 only | ERA5 **and** NASA POWER, side by side per row |
| Output | `climate_tamilnadu_all.csv` | `climate_tamilnadu_points.csv` |

## Requirements

```
pip install geopandas rasterio requests pandas numpy xarray netCDF4 pvlib scipy cdsapi
```

`geopandas`/`rasterio` are only needed for `00a`.

## Notes

- **Puducherry**: GADM's "Tamil Nadu" polygon does not include the
  Puducherry union territory enclaves. Set `INCLUDE_PUDUCHERRY = True` in
  `00a_build_population_grid.py` if you want them folded in.
- **Elevation**: `00c_attach_elevation.py` (added 2026-09-16) attaches real
  per-point elevation from ERA5's time-invariant geopotential field before
  `02_combine_tamilnadu.py` runs (-0.0 m to 1,283.4 m across the 133 points,
  mean 282.4 m — the ERA5 ~28km grid smooths the true Nilgiris peak, an
  accepted limitation). `DEFAULT_ALT_M = 150` in `02_combine_tamilnadu.py`
  remains only as a fallback for a point missing `elevation_m`, which
  should not happen once `00c` has run.
- **First day of the dataset**: 2016-01-01 has no prior-day predecessor
  hour for deaccumulation if a sun event's window touches hour 0 UTC —
  affected `era5_GHI`/related columns for that one day come out `NaN`
  rather than a wrong value. Every other month boundary is bridged
  automatically.
- If you already ran the *old* 222-point/2-year pipeline in this same
  project folder, its files live under `data/raw/era5/grid/` and are
  untouched by any script here — the new pipeline uses entirely separate
  paths (`data/raw/era5/points/`).

## Current status (see `docs/tamilnadu/00_MASTER_OVERVIEW.md` for detail)

All 8 phases are implemented and have been run end-to-end as of 2026-09-16
against elevation-corrected data. Headline results: k=3 climate clusters
(GMM, auto-selected via `cluster_lib.suggest_k`'s 3-tier cascade — not a
hand-pick); a 9-criterion MCDM engine (added `thermal_margin`, see below);
consensus Top-1 PCM is `RT57HC` in all three clusters.

**MCDM-vs-physics agreement, investigation closed**: `09_PHASE_7_AUDIT.md`
diagnosed a negative Spearman ρ between MCDM consensus rank and Phase 7's
physics-simulated performance in Cluster 0, traced it to no criterion
measuring thermal margin below the achievability ceiling, and fixed it with
a 9th criterion (`thermal_margin`) — ρ went -0.595→+0.381 there. Clusters
1/2 did **not** respond to the same fix (proven, not assumed: the identical
PCM `CrodaTherm 60` scores 38%/79%/66% simulated performance across the
three clusters despite unchanged properties — the driver is a dynamic
climate-PCM interaction no static ranking criterion can capture). That
residual disagreement is kept as a documented finding rather than chased
with further criteria tuning, which would risk overfitting the ranking to
one physics simulation's specific assumptions. See `docs/tamilnadu/
09_PHASE_7_AUDIT.md` for the full investigation.

Note this repo copy (`new_obj/tamilnadu_pipeline/`) lives one directory
level deeper than the original `PCM-Selection-ML-model/era5-tamilnadu/`
layout some scripts' relative-path logic assumed; `config.py`,
`06_build_pcm_database.py` and `08_mcdm_ranking.py` all now search both
layouts (see `docs/tamilnadu/CHANGELOG.md`'s 2026-09-16 entry) rather
than hardcoding one.
