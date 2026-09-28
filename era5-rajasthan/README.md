# ERA5 Rajasthan Pipeline

Builds a solar/climate dataset for Rajasthan, sampled at **population-weighted
locations** and **astronomically computed sun-event times** (sunrise, solar
noon, sunset) rather than a uniform grid on fixed clock hours. Pulls both
ERA5 reanalysis and NASA POWER for the same points/times so the two
independent sources can be cross-checked against each other.

## Pipeline overview

```
00a_build_population_grid.py   →  data/processed/population_grid_points.csv
00b_build_suntimes.py          →  data/processed/suntimes.csv
00c_attach_elevation.py        →  population_grid_points.csv gains `elevation_m`
01_download_era5_rajasthan.py  →  data/raw/era5/points/*.nc
01b_download_nasapower.py      →  data/raw/nasapower/*.json
00_unzip_accum.py              →  (fixes zip-disguised-as-.nc files in place)
02_combine_rajasthan.py        →  data/processed/climate_rajasthan_points.csv
02b_build_daily_aggregates.py  →  data/processed/daily_aggregates_rajasthan.csv
                                   data/processed/daily_aggregates_rajasthan_summary.csv

── QA / QC — read-only, safe to run anytime, not part of the linear chain ──
03_verify_climate_csv.py       →  (stdout report on climate_rajasthan_points.csv)
00d_population_grid_viz.py     →  outputs/grid_plot.html  (Plotly, sampling-grid sanity check)
03_plots_raw.py                →  PLOTSV2/raw/*.png  (raw-data QA, checks A-F)
03b_interactive_raw_qa.py      →  PLOTSV2/raw_interactive/*.html  (same checks, interactive)
```

NOTE: no script named `03_qc_plots.py` exists in this pipeline — see the corrected section below
("00d_population_grid_viz.py") for what actually produces the Phase-1 sampling-grid QC output.

## Run Order

```
python 00a_build_population_grid.py   # downloads GADM boundary + WorldPop raster, builds population_grid_points.csv
python 00b_build_suntimes.py          # builds suntimes.csv (sunrise/noon/sunset UTC times, pvlib)
python 00c_attach_elevation.py        # attaches real per-point elevation_m from ERA5 geopotential
python 01_download_era5_rajasthan.py  # ERA5 download, sized to the population points + sun-event hours
python 01b_download_nasapower.py      # NASA POWER cross-check data, per point/year
python 00_unzip_accum.py              # fixes any CDS zip-disguised-as-.nc files (now scans both old + new ERA5 dirs)
python 02_combine_rajasthan.py        # merges everything into climate_rajasthan_points.csv
python 02b_build_daily_aggregates.py  # true daily integrals/indices from cached NASA POWER hourly data

python 03_verify_climate_csv.py       # optional: QA report on climate_rajasthan_points.csv
python 00d_population_grid_viz.py     # optional: sampling-grid sanity-check plot (population/weight/elevation)
python 03_plots_raw.py                # optional: raw-data QC plots (static PNG, checks A-F)
python 03b_interactive_raw_qa.py      # optional: same checks, interactive Plotly+Folium HTML
```

Each script is resumable — safe to Ctrl-C and re-run; already-completed work
is skipped automatically. The two `03_*` scripts are read-only QA tools: they
never write into the pipeline's data files and can be run at any point,
including mid-download, to sanity-check progress so far.

## What each script does

### `config.py`
Shared, path-anchored configuration used by every script (works regardless
of the current working directory). Defines every input/output path,
`ensure_data_dirs()` to create them, and `get_cdsapi_client()` /
`load_cds_credentials()` for the CDS (Copernicus) API. Not run directly.

### `00a_build_population_grid.py`
Picks the sampling locations. Downloads the Rajasthan boundary (GADM v4.1,
admin level 1) and the WorldPop India population raster (2020,
UN-adjusted, 100m — ~1.5-2GB, one-time download), clips the raster to
Rajasthan, aggregates population onto a 0.25°(27.8 km) grid **aligned to ERA5's own
grid origin** (so each point maps to a distinct ERA5 cell downstream), ranks
cells by population, and keeps the minimal set covering ~87.5% of the
state's total population.

- Output: `data/processed/population_grid_points.csv` —
  `point_id, lat, lon, population, weight` (`00c_attach_elevation.py` later
  adds an `elevation_m` column to this same file)
- Uses a single static 2020 population snapshot for the whole 2016-2025
  study period (WorldPop doesn't publish a distinct India raster per year at
  this resolution) — a standard simplifying assumption, not a bug.
- Large raw downloads cached in `data/raw/population/` and
  `data/raw/boundary/`.

### `00b_build_suntimes.py`
For every point and every date 2016-01-01..2025-12-31, computes the exact
UTC sunrise, solar noon, and sunset via `pvlib`'s SPA algorithm (no manual
equation-of-time code).
Why is SPA needed?

Suppose you want to know:

When does the Sun rise today in Jaipur?
What is the exact time of solar noon?
What is the solar elevation at 3:42 PM?
What is the solar azimuth?

You cannot simply use:

Sunrise = 6:00 AM
Sunset = 6:00 PM

because these depend on:

Latitude
Longitude
Date
Earth's axial tilt
Earth's elliptical orbit
Atmospheric refraction
Time zone / UTC
Leap years

SPA models all of these effects.
- Output: `data/processed/suntimes.csv` —
  `point_id, date, event (sunrise|noon|sunset), time_utc`
- Note: sun events near the Rajasthan/UTC boundary can genuinely fall on the
  *previous* UTC calendar date (e.g. an eastern point's summer sunrise can
  land at 23:55 UTC the day before) — `time_utc` is always the true instant;
  `date` is pvlib's nominal calendar-date assignment for that event.

### `00c_attach_elevation.py`
Downloads ERA5's time-invariant surface geopotential field (`z`) over the
same bounding envelope `01_download_era5_rajasthan.py` uses — one CDS
request for a single date/time, since orography doesn't change over time —
and attaches a per-point `elevation_m = z / 9.80665` column to
`population_grid_points.csv`. Replaces the flat 300m elevation assumption
`02_combine_rajasthan.py` used to fall back to for every point.

- Output: `data/raw/era5/invariant/era5_RJ_geopotential.nc` (raw cache);
  `elevation_m` column added to `population_grid_points.csv` in place.
- Does not touch the sun-event instant/accum cache under
  `data/raw/era5/points/` — entirely separate download.

### `01_download_era5_rajasthan.py`
Downloads ERA5 hourly reanalysis over the bounding envelope of the
population points (not the whole state), for three narrow UTC hour windows
computed from `suntimes.csv` — one around sunrise, one around solar noon,
one around sunset — each padded ~1hr and correctly handling the
cross-midnight wraparound case above. Keeps the original pipeline's
instant/accum variable split and deaccumulation-helper-hour logic
(generalized to the new dynamic hour set — see the script's docstring and
`deaccumulate()` in `02_combine_rajasthan.py` for why that still works
correctly).

- Output: `data/raw/era5/points/era5_RJ_points_{year}_{month}_{instant,accum}.nc`
- Status tracking: `data/raw/era5/download_status_points.csv`
- **Does not touch** the old `data/raw/era5/grid/` archive or
  `download_status.csv` from the previous uniform-grid/fixed-hour pipeline —
  entirely separate paths.
- Requires `.cdsapirc` (CDS/Copernicus API credentials) in this folder.

### `01b_download_nasapower.py`
For every point and every year 2016-2025, downloads NASA POWER hourly point
data (`ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS10M`) — an
independent cross-check source. No API key needed.

- Output: `data/raw/nasapower/power_{point_id}_{year}.json` (raw cache)
- Status tracking: `data/raw/nasapower/download_status_power.csv`

### `00_unzip_accum.py`
The CDS API sometimes returns accum files as a ZIP even when an unarchived
NetCDF was requested. This detects and fixes those in place. Scans **both**
`data/raw/era5/grid/` (old pipeline) and `data/raw/era5/points/` (new
pipeline). Safe to re-run — valid NetCDF files are left alone.

### `02_combine_rajasthan.py`
The merge step. For each point: nearest-neighbor-snaps to the ERA5 grid,
concatenates its full hourly series across all years, deaccumulates,
computes solar geometry (`pvlib`). For each `(point_id, date, event)` row in
`suntimes.csv`, picks the nearest-in-time ERA5 reading and the nearest-in-time
NASA POWER reading (both rejected if farther than 3 hours from the true
event time), and merges them into one row.

- Output: `data/processed/climate_rajasthan_points.csv` — one row per
  point/date/event, with `era5_*` and `power_*` columns side by side for
  cross-checking, plus point metadata (`lat`, `lon`, `population`, `weight`,
  `elevation_m`) and calendar features (`month`, `DOY`, `year`, `season`,
  `season_code`).
- Fully replaces the old pipeline's per-city (`RJ_LOCATIONS`) and opt-in
  full-grid output — nothing else in the repo reads those, so there's no
  need to keep both schemes running.
- Uses each point's real `elevation_m` (from `00c_attach_elevation.py`) for
  pvlib solar geometry / clear-sky irradiance if present, falling back to
  the flat 300m default only if that column is missing or NaN for a point.

### `02b_build_daily_aggregates.py`
`climate_rajasthan_points.csv` only has 3 instantaneous samples/day
(sunrise, noon, sunset), which can't produce daily energy integrals, true
diurnal temperature range, or degree-day counts. This script reads the full
hourly series already cached by `01b_download_nasapower.py`
(`data/raw/nasapower/power_{point_id}_{year}.json`) directly — no
re-download — and builds true per-day aggregates (trapezoidal GHI/clear-sky
integrals, true daily min/max/mean temperature, mean RH/wind) plus a
per-point Tier 2 summary (`GHI_daily_kWh`, `SAI`, `kt_daily_mean/std`,
`cloudy_frac`, `CCI`, `HDD18`, `CDD24`, `DTR_true`, `seasonality`,
`monsoon_index` — see the script's docstring for exact definitions).
NASA-POWER-only by design; an ERA5-based daily-integral version would need
a new CDS request for all 24 hours/day and is out of scope here.

- Output: `data/processed/daily_aggregates_rajasthan.csv` (one row per
  point/day) and `data/processed/daily_aggregates_rajasthan_summary.csv`
  (one row per point, the Tier 2 indices).
- Status tracking: `data/processed/daily_aggregates_status.csv`
- `cloudy_frac`'s clearness threshold (`kt < 0.3`) is not defined anywhere
  else in the repo — documented assumption, not a canonical spec value.

### `03_verify_climate_csv.py`
Read-only QA report on `climate_rajasthan_points.csv` — never modifies it,
safe to run at any time, including while `02_combine_rajasthan.py` is still
running (it just reports partial coverage accurately rather than failing).
Checks, in order: (1) schema — every expected `era5_*`/`power_*`/metadata
column present; (2) point coverage — every `point_id` from
`population_grid_points.csv` shows up, flags missing/extra ids; (3) row
coverage — each point has exactly the rows `suntimes.csv` implies, flags
partial/duplicate rows; (4) null rates per column against warn/fail
thresholds (5%/30%) — real gaps are expected (e.g. the documented
2016-01-01 edge case) but a mostly-empty column signals a bug; (5) physical
sanity — value-range checks per variable, mirroring the bounds
`02_combine_rajasthan.py` itself enforces plus a few this script owns
(pressure, POWER irradiance); (6) cross-source agreement — correlation
between `era5_GHI`/`power_ALLSKY_SFC_SW_DWN` and `era5_T_amb`/`power_T2M`,
since agreement between the two independent sources is the whole point of
pulling both.

- Output: stdout report only (`[OK]`/`[WARN]`/`[FAIL]` per check, exits
  non-zero if any `[FAIL]`).

HOW TO RUN: `python 03_verify_climate_csv.py`

### `00d_population_grid_viz.py`
**CORRECTED — this section previously described a `03_qc_plots.py` script (folium
population/elevation/download-status maps + several separate plotly QC charts under
`outputs/qc_*.html`) that does not exist anywhere in this pipeline; no such file, and none of the
`qc_population_*.html` / `qc_elevation_*.html` / `qc_download_status_*.html` / `qc_suntimes_line.html`
/ `qc_rejection_window.html` outputs it described, were ever found on disk.**

What actually exists for this sanity-check role is `00d_population_grid_viz.py`: a read-only Plotly
renderer for the population-weighted sampling grid, producing ONE self-contained
`outputs/grid_plot.html` (`scatter_mapbox`-style, tokenless carto-darkmatter basemap) with a dropdown
to switch which metric — `population` / `weight` / `elevation_m` — drives marker size and colour.
Reads `population_grid_points.csv` only; produces no data. Not in the core chain. A prior Folium
version of this map was dropped 2026-08-12 in favor of the current Plotly-only renderer.

The requested-vs-matched reading-time-offset diagnostic this section used to describe under
`qc_rejection_window.html` does not exist either, for the same underlying reason: it would need
`era5_matched_time_utc` / `power_matched_time_utc` columns that `02_combine_rajasthan.py` never
writes (see `docs/rajasthan/04_PHASE_2_AUDIT.md` §A.5 for that open gap).

For raw-data distributional/cross-source QC (the closest thing to what this section originally
described), see `03_plots_raw.py` (static PNG, checks A-F) and `03b_interactive_raw_qa.py`
(interactive Plotly+Folium HTML) in `docs/rajasthan/04_PHASE_2_AUDIT.md`.

HOW TO RUN: `python 00d_population_grid_viz.py`

---

## Phase 3–5: Climate Signature + PCM Feasibility Filtering

Once the raw climate data is acquired and processed (scripts 00–03 above), the
next stage builds climate signatures for each point and filters PCM candidates
against climate-specific design constraints. Located in `era5-rajasthan/`:

```
04b_climate_signature.py            →  data/processed/climate_signature_rajasthan.csv
05_cluster_rajasthan.py             →  data/processed/cluster_profiles_rajasthan.csv
07_feasibility_filter.py            →  data/processed/feasibility_survivors_by_cluster.csv
                                        data/processed/feasibility_survivors_by_cluster_kappa_calibrated.csv
                                        (canonical Tamil-Nadu naming, unified 2026-09-08)
```

### `04b_climate_signature.py` (PHASE 3)
(renamed from `04_climate_signature_rajasthan.py` — script on disk is `04b_climate_signature.py`)

Reduces each point's 10-year daily and sun-event records to a single
**climate-signature vector** — the summary that Phase 4 (clustering) actually
operates on. Builds:

- **Tier 1**: Per-event aggregates (sunrise, solar noon, sunset temperature,
  humidity, wind, irradiance at each event)
- **Tier 2**: Daily integrals and indices (GHI, clearness, cloudiness, HDD18,
  CDD24, diurnal temperature range, seasonal variation)
- **PCM-facing quantities**: Tm_target (the target storage temperature for
  SWH), Tm_target_capped (climate-adjusted cap), L_required (latent-heat
  requirement for PCM sizing)
- **Interactions**: Five terms combining daily/hourly features to capture
  cycling stress, condensation risk, convective loss, and autonomy demand
- **PCA**: Dimensionality reduction on correlated temperature/pressure block

**METHODOLOGY NOTE (Corrected 2026-08-31):**

L_required is computed as **PCM's literature-anchored fractional share** of
total night-discharge thermal delivery, not 100% of the load alone. Avargani
et al. (2021)'s own system delivers the 300 L benchmark via integrated
collector + PCM tank + sensible-heat tank; literature on combined
sensible-latent SWH reports PCM contributing 40–78% of total delivery (Zhao
2022, Huang 2020, Abdelsalam 2020, Koželj 2021). Formula:

```
L_required = (SHARE_PCM * Q_night) / m_PCM
```

with SHARE_PCM = 0.5 (central estimate; range 0.4–0.7). This shifts from an
all-latent, zero-candidate baseline to a combined sensible+latent model where
majority of candidates survive Phase 5 filtering. See `04b_climate_signature.py`'s
docstring (corrections #4–5) for full rationale, or CLAUDE.md §3.1 for the
complete methodology justification and Phase 5 guidance. NOTE: `T_DELIVERY_C` (feeding this same
`L_required` formula) was subsequently raised 50→60°C on 2026-09-13 to match Avargani et al.
(2021)'s actual validated delivery temperature — see CLAUDE.md §3.2 and
`docs/rajasthan/05_PHASE_3_AUDIT.md`; `L_required` now runs ~410–469 kJ/kg, not the ~285–344 kJ/kg
this section's SHARE_PCM=0.5 example predates.

**HOW TO RUN:** `python 04b_climate_signature.py`

### `05_cluster_rajasthan.py` (PHASE 4)

Clusters the signature points using Gaussian Mixture Models (GMM) and
Agglomerative Clustering to identify distinct **climate regimes**. Each regime
becomes a "Level A cluster" with:

- Representative climate profile (mean Tm_target, L_required, monsoon_index, etc.)
- Cluster-level ground-truth dataset used by Phase 7 (charging feasibility modeling)

**Output:** `cluster_profiles_rajasthan.csv` — one row per cluster with all
signature columns aggregated to cluster level, plus `Tm_target_capped_C` and
`L_required_kJ_per_kg` re-derived per cluster.

**HOW TO RUN:** `python 05_cluster_rajasthan.py`

### `07_feasibility_filter.py` (PHASE 5)
(renamed from `07_feasibility_filter_rajasthan.py` — script on disk is `07_feasibility_filter.py`)

Hard-filters the shared PCM candidate database against each cluster's 8
design constraints (melting window, absolute Tm band, latent-heat floor,
cycling endurance, supercooling, charging feasibility, corrosion veto,
safety flags). Produces two outputs:

1. **PRIMARY (`feasibility_survivors_by_cluster.csv`)**: Fixed κ=0.7 latent-heat
   floor, with full diagnostic audit trail (per-cluster, per-constraint results).
   Expected to show the baseline (κ=0.7 against old L_required was a near-zero-survivor
   case, demonstrating why calibration was needed).

2. **COMPANION (`feasibility_survivors_by_cluster_kappa_calibrated.csv`)**: Per-cluster
   calibrated κ, stepped down from 0.7 until 8–20 candidates survive. Includes
   `breakeven_kappa` column (actual threshold each candidate sits at) for ranking.

**VALIDATION:** After Phase 3's L_required correction, re-run this script and
verify calibrated κ lands in the 0.5–0.7 range (much higher than the prior
0.2–0.3), validating the SHARE_PCM=0.5 assumption. Report both outputs
together: broken assumption (old κ=0.7 → zero survivors) → diagnosis (L_required
ceiling) → correction (SHARE_PCM factorization) → verification (new κ resets
higher).

**HOW TO RUN:** `python 07_feasibility_filter.py`

---

## Requirements

```
pip install geopandas rasterio requests pandas numpy xarray netCDF4 pvlib scipy cdsapi folium branca plotly
```

`geopandas`/`rasterio` are only needed for `00a`; `folium`/`branca`/`plotly`
are only needed for the QC/visualization scripts (`00d_population_grid_viz.py`, `03_plots_raw.py`,
`03b_interactive_raw_qa.py`); the rest of the pipeline only needs the others.

### Other files in this folder
- `.cdsapirc` — your personal CDS/Copernicus API credentials (`url:` /
  `key:` lines), read by `config.py`'s `load_cds_credentials()`. Not
  committed — keep this file private; it's your own account's API key.
- `.gitignore` — excludes the generated `data/` and `outputs/` trees (raw
  downloads, processed CSVs, QC plots) from version control, so only the
  pipeline code itself is tracked.

## Notes / known limitations

- **First day of the dataset**: 2016-01-01 has no prior day to supply an
  accumulation-deaccumulation predecessor hour if a sun event's window
  touches hour 0 UTC — the affected `era5_GHI`/related columns for that one
  day come out as a natural `NaN` rather than a wrong value. Every other
  month boundary is bridged automatically (see `01_download_era5_rajasthan.py`'s
  docstring for why).
- **Elevation**: `00c_attach_elevation.py` attaches real per-point elevation
  from ERA5's invariant geopotential field; `02_combine_rajasthan.py` falls
  back to a flat 300m approximation only if that column is missing (same
  approximation the old pipeline's full-grid mode used).
- **Elevation is a grid-cell mean**: ERA5's native grid is ~0.25°(~28km), so
  its orography value for a point is the *mean* elevation of that whole
  grid cell, not the point's exact local elevation. This is fine where
  terrain is fairly flat (Rajasthan, Assam, coastal Tamil Nadu) but smooths
  out real relief in high-relief regions (e.g. Uttarakhand's 200m-7000m+
  range) — an accepted, documented caveat, not something this pipeline
  tries to fix further.
- **WorldPop download size**: ~1.5-2GB, one-time, cached in
  `data/raw/population/`. The download auto-retries (up to 5 attempts) and
  resumes from where it left off via HTTP Range requests if the connection
  drops mid-stream — no manual intervention needed on a flaky connection.
