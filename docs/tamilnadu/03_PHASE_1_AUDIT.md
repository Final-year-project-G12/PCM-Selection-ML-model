# 03 — Phase 1 Audit: Data Collection

Scripts: `00a_build_population_grid.py`, `00b_build_suntimes.py`, `01_download_era5_tamilnadu.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`.

## Purpose
Determine the coordinates (where) and timestamps (when) to sample climate data, then retrieve ERA5 and NASA POWER historical records for Tamil Nadu.

## Inputs
- GADM boundary file (v4.1 India admin-1).
- WorldPop 2020 UN-adjusted 100 m population density raster for India.
- CDS API access credentials.
- NASA POWER API.

## Processing Details
1. **Population-Weighted Sampling (`00a_build_population_grid.py`)**:
   - Aggregates population onto a 0.25° grid aligned to ERA5's grid origin (`lat=90.0, lon=-180.0`). This guarantees a 1:1 spatial grid mapping between population cells and ERA5 grid nodes.
   - Keeps the minimal set of highest-population cells covering `COVERAGE_TARGET = 0.875` (87.5%) of the state's population.
   - **Tamil Nadu Results**: Produces **133 points** (`TNP_0001` to `TNP_0133`).
2. **Sun-Event Times (`00b_build_suntimes.py`)**:
   - For every point × every date in 2016–2025, computes the exact UTC sunrise, solar noon, and sunset using `pvlib`'s SPA algorithm.
   - **Row Count**: **1,457,547 rows** (133 points × 3653 days × 3 events). Alt=0 is assumed for sunrise/sunset times, which is a standard simplification.
3. **ERA5 Download (`01_download_era5_tamilnadu.py`)**:
   - Downloads three narrow UTC hour windows around sunrise, solar noon, and sunset, using circular mod-24 logic to handle day wraparound (important for westernmost points).
   - Downloads both instant and accumulated fields (240 files).
4. **NASA POWER Download (`01b_download_nasapower.py`)**:
   - Pulls full hourly weather parameters (87,660 hours per point) for all 133 points across the 10-year span (1,330 JSON files).
5. **CDS Zip-Quirk Fix (`00_unzip_accum.py`)**:
   - Scans and extracts netCDF files that the CDS API returned as disguised ZIPs.

## Differences from Rajasthan
- **Point count**: 133 points for Tamil Nadu vs 320 points for Rajasthan. This reflects Tamil Nadu's smaller geographic footprint.
- **Elevation**: Rajasthan has a dedicated `00c_attach_elevation.py` script that downloads and extracts real elevation (m) using ERA5 geopotential. **Tamil Nadu does not have an elevation attachment script**. Instead, `02_combine_tamilnadu.py` uses a flat elevation approximation of 150 m for solar calculations.

## Status
**COMPLETE**
