# 03 — Phase 1 Audit: Data Collection

**Script(s)**: `00a_build_population_grid.py`, `00b_build_suntimes.py`, `01_download_era5_assam.py`,
`01b_download_nasapower.py`, `00_unzip_accum.py`

**Status**: COMPLETE

## Population-weighted grid (`00a_build_population_grid.py`)

- **Source**: WorldPop 100m raster (India, UN-adjusted 2020) + GADM v4.1 boundary (Assam, NAME_1=="Assam")
- **Method**: Clip raster to Assam boundary → aggregate to 0.25° ERA5 grid → rank cells by descending
  population → keep minimal prefix covering ≥87.5% of state total
- **Result**: **128 points** covering 87.5% of Assam's population
- **Output**: `population_grid_points.csv` (columns: `point_id`, `lat`, `lon`, `population`, `weight`)
- **Point IDs**: `ASP_0001` through `ASP_0129` (some intermediate IDs skipped due to boundary rejection)
- **Elevation**: Default 100m (Assam valley/plains baseline) — unlike Rajasthan, no per-point elevation
  was attached from ERA5 geopotential (no `00c_attach_elevation.py` in the Assam pipeline)

## Sun-times (`00b_build_suntimes.py`)

- **Output**: `suntimes.csv` — sunrise, solar noon, sunset UTC timestamps for each point × each day
  across the 10-year download period
- **Library**: pvlib SPA (Solar Position Algorithm)
- **Purpose**: Defines the three "events" (sunrise/noon/sunset) at which ERA5 is downloaded, giving
  solar-geometry-aligned hourly samples rather than arbitrary fixed UTC hours

## ERA5 download (`01_download_era5_assam.py`)

- **CDS product**: `reanalysis-era5-single-levels`, hourly
- **Variables**: `ssrd`, `strd`, `t2m`, `d2m`, `u10`, `v10`, `msl`, `tcc`, `tp`, `avg_sdirswrf`
- **Strategy**: Per-point download, sun-event-aligned hours, 2016–2025
- **Idempotency**: `download_status_points.csv` tracks completion; partial downloads are resumable
- **Format**: NetCDF (`.nc`); the CDS delivers some files as `.zip` disguised as `.nc` —
  `00_unzip_accum.py` handles the unwrapping

## NASA POWER download (`01b_download_nasapower.py`)

- **Product**: Daily data — `ALLSKY_SFC_SW_DWN`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PRECTOTCORR`
- **Period**: 2016–2025, per-point JSON files
- **Point naming**: `power_{point_id}_{year}.json` → 128 × 10 = 1,280 files
- **Idempotency**: `download_status_power.csv`

## Known issues / deviations from Rajasthan

1. **No `00c_attach_elevation.py`**: Rajasthan had a dedicated script to attach per-point ERA5
   geopotential elevation. Assam uses a fixed 100m default for the valley/plains baseline. Hill
   district points (Karbi Anglong, Dima Hasao) are at higher elevation in reality but are assigned
   the same 100m default. This is a documented approximation, not an undiscovered error.

2. **No explicit download-count verification on disk**: Rajasthan documented 240/240 ERA5 files and
   3200/3200 POWER files explicitly. Assam's download completeness is tracked via the status CSV
   but the exact per-file counts were not independently verified against disk at the time of this audit.

3. **Point ID gaps**: The `ASP_0001–ASP_0129` naming has non-contiguous IDs (some cells rejected
   during boundary clipping) resulting in 128 active points from a 129-slot ID space.
