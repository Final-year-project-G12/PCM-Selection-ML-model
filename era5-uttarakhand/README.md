# ERA5 Uttarakhand Pipeline

Builds a solar/climate dataset for Uttarakhand, sampled at **population-weighted
locations** and **astronomically computed sun-event times** (sunrise, solar
noon, sunset) rather than a uniform grid on fixed clock hours. Pulls both
ERA5 reanalysis and NASA POWER for the same points/times so the two
independent sources can be cross-checked against each other.

## Pipeline overview

```
00a_build_population_grid.py   →  data/processed/population_grid_points.csv
00b_build_suntimes.py          →  data/processed/suntimes.csv
01_download_era5_uttarakhand.py→  data/raw/era5/points/*.nc
01b_download_nasapower.py      →  data/raw/nasapower/*.json
00_unzip_accum.py              →  (fixes zip-disguised-as-.nc files in place)
02_combine_uttarakhand.py      →  data/processed/climate_uttarakhand_points.csv
```

## Run Order

```
python 00a_build_population_grid.py   # downloads GADM boundary + WorldPop raster, builds population_grid_points.csv
python 00b_build_suntimes.py          # builds suntimes.csv (sunrise/noon/sunset UTC times, pvlib)
python 01_download_era5_uttarakhand.py  # ERA5 download, sized to the population points + sun-event hours
python 01b_download_nasapower.py      # NASA POWER cross-check data, per point/year
python 00_unzip_accum.py              # fixes any CDS zip-disguised-as-.nc files (now scans both old + new ERA5 dirs)
python 02_combine_uttarakhand.py      # merges everything into climate_uttarakhand_points.csv
```

Each script is resumable — safe to Ctrl-C and re-run; already-completed work
is skipped automatically.

## What each script does

### `config.py`
Shared, path-anchored configuration used by every script (works regardless
of the current working directory). Defines every input/output path,
`ensure_data_dirs()` to create them, and `get_cdsapi_client()` /
`load_cds_credentials()` for the CDS (Copernicus) API. Not run directly.

### `00a_build_population_grid.py`
Picks the sampling locations. Downloads the Uttarakhand boundary (GADM v4.1,
admin level 1) and the WorldPop India population raster (2020,
UN-adjusted, 100m — ~1.5-2GB, one-time download), clips the raster to
Uttarakhand, aggregates population onto a 0.25° grid **aligned to ERA5's own
grid origin** (so each point maps to a distinct ERA5 cell downstream), ranks
cells by population, and keeps the minimal set covering ~87.5% of the
state's total population.

- Output: `data/processed/population_grid_points.csv` —
  `point_id, lat, lon, population, weight`
- Uses a single static 2020 population snapshot for the whole 2016-2025
  study period (WorldPop doesn't publish a distinct India raster per year at
  this resolution) — a standard simplifying assumption, not a bug.
- Large raw downloads cached in `data/raw/population/` and
  `data/raw/boundary/`.

### `00b_build_suntimes.py`
For every point and every date 2016-01-01..2025-12-31, computes the exact
UTC sunrise, solar noon, and sunset via `pvlib`'s SPA algorithm (no manual
equation-of-time code).

- Output: `data/processed/suntimes.csv` —
  `point_id, date, event (sunrise|noon|sunset), time_utc`
- Note: sun events near the Uttarakhand/UTC boundary can genuinely fall on the
  *previous* UTC calendar date (e.g. an eastern point's summer sunrise can
  land at 23:55 UTC the day before) — `time_utc` is always the true instant;
  `date` is pvlib's nominal calendar-date assignment for that event.

### `01_download_era5_uttarakhand.py`
Downloads ERA5 hourly reanalysis over the bounding envelope of the
population points (not the whole state), for three narrow UTC hour windows
computed from `suntimes.csv` — one around sunrise, one around solar noon,
one around sunset — each padded ~1hr and correctly handling the
cross-midnight wraparound case above. Keeps the original pipeline's
instant/accum variable split and deaccumulation-helper-hour logic
(generalized to the new dynamic hour set — see the script's docstring and
`deaccumulate()` in `02_combine_uttarakhand.py` for why that still works
correctly).

- Output: `data/raw/era5/points/era5_UK_points_{year}_{month}_{instant,accum}.nc`
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

### `02_combine_uttarakhand.py`
The merge step. For each point: nearest-neighbor-snaps to the ERA5 grid,
concatenates its full hourly series across all years, deaccumulates,
computes solar geometry (`pvlib`). For each `(point_id, date, event)` row in
`suntimes.csv`, picks the nearest-in-time ERA5 reading and the nearest-in-time
NASA POWER reading (both rejected if farther than 3 hours from the true
event time), and merges them into one row.

- Output: `data/processed/climate_uttarakhand_points.csv` — one row per
  point/date/event, with `era5_*` and `power_*` columns side by side for
  cross-checking, plus point metadata (`lat`, `lon`, `population`, `weight`)
  and calendar features (`month`, `DOY`, `year`, `season`, `season_code`).

## Requirements

```
pip install geopandas rasterio requests pandas numpy xarray netCDF4 pvlib scipy cdsapi
```

`geopandas`/`rasterio` are only needed for `00a`; the rest of the pipeline
only needs the others.

## Notes / known limitations

- **First day of the dataset**: 2016-01-01 has no prior day to supply an
  accumulation-deaccumulation predecessor hour if a sun event's window
  touches hour 0 UTC — the affected `era5_GHI`/related columns for that one
  day come out as a natural `NaN` rather than a wrong value. Every other
  month boundary is bridged automatically (see `01_download_era5_uttarakhand.py`'s
  docstring for why).
- **Elevation**: population points don't carry elevation data, so
  `02_combine_uttarakhand.py` uses a flat 1200m approximation for solar-geometry
  calculations (higher than Rajasthan's 300m default, reflecting Uttarakhand's
  mountainous terrain — populated zones range roughly 200-2000m).
- **WorldPop download size**: ~1.5-2GB, one-time, cached in
  `data/raw/population/`. The download auto-retries (up to 5 attempts) and
  resumes from where it left off via HTTP Range requests if the connection
  drops mid-stream — no manual intervention needed on a flaky connection.
