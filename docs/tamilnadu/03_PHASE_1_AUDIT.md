# 03 — Phase 1 Audit: Data Collection

Scripts: `00a_build_population_grid.py`, `00b_build_suntimes.py`, `01_download_era5_tamilnadu.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`.

## Purpose
Determine the coordinates (where) and timestamps (when) to sample climate data, then retrieve ERA5 and NASA POWER historical records for Tamil Nadu with full spatial and temporal rigor.

---

## Inputs
- **GADM Boundary**: Administrative Level 1 boundary file for Tamil Nadu (v4.1 India admin-1).
- **WorldPop Raster**: WorldPop 2020 UN-adjusted 100 m population density raster for India.
- **CDS API Credentials**: Access tokens for ECMWF Copernicus Data Store.
- **NASA POWER API**: Hourly cached satellite and model meteorological dataset.

---

## Processing Details & Methodological Justifications

### 1. Population-Weighted Sampling (`00a_build_population_grid.py`)
- **Grid Alignment**: Aggregates population density onto a 0.25° grid pre-aligned to ERA5's grid origin (`lat=90.0, lon=-180.0`). This guarantees a 1:1 spatial grid mapping between population cells and ERA5 grid nodes, preventing grid cell collapse (where multiple points snap to the same reanalysis cell).
- **87.5% Population Rule**: Sorts grid cells by population density descending and retains the minimal set of cells required to cover **`COVERAGE_TARGET = 0.875` (87.5%)** of Tamil Nadu's total population.
- **Tamil Nadu Spatial Grid**: Yields **133 points** (`TNP_0001` to `TNP_0133`).

### 2. Spatial Processing Justification (formerly `11_SPATIAL_PROCESSING.md`)
- **Why Population-Weighting**: Uniform geometric grids spend computation on sparsely populated forest or mountain zones. Population-weighting ensures that discovered climate regimes represent regions where actual domestic solar water heating demand exists.
- **Nearest-Neighbor Snapping**: Coordinates snap to nearest ERA5 grid centers using Euclidean distance. Pre-aligning grid cells guarantees 1:1 mapping without distortion.
- **Elevation Handling & Flat Terrain Caveat**:
  - Tamil Nadu currently uses a **flat default terrain assumption of 150 m** in `02_combine_tamilnadu.py` for atmospheric pressure and clear-sky solar calculations.
  - *Justification & Limitation*: Flat 150 m is a reasonable proxy for the coastal plains and interior tablelands where >90% of Tamil Nadu's population resides. However, it ignores high-relief montane zones in the Western Ghats (e.g., Nilgiris / Ooty at ~2,240 m). Unlike Rajasthan (which lacks montane extremes), multi-state extensions or montane-specific deployments will require explicit geopotential elevation extraction (`00c_attach_elevation.py`).

### 3. Sun-Event Times & Temporal Alignment (`00b_build_suntimes.py`)
- **UTC Time Base**: ERA5 reanalysis and NASA POWER satellite data are stored and retrieved in Coordinated Universal Time (UTC). Indian Standard Time (IST) is UTC + 5:30.
- **Solar Position Algorithm (SPA)**: For every point × every date in 2016–2025, computes exact UTC sunrise, solar noon, and sunset using `pvlib`'s implementation of the Reda & Andreas (2004) SPA algorithm.
- **Dataset Size**: **1,457,547 rows** (133 points × 3653 days × 3 events). Zero altitude (`alt = 0 m`) is assumed for sun-event calculations across the 150 m plain proxy.

### 4. ERA5 & NASA POWER Downloads (`01_download_era5_tamilnadu.py`, `01b_download_nasapower.py`)
- **Circular Window Sampling**: Downloads three narrow UTC hour windows around sunrise, solar noon, and sunset, using circular mod-24 logic to eliminate midnight UTC boundary wrap errors.
- **ERA5 Cache**: Downloads both instant and accumulated fields (240 NetCDF files).
- **NASA POWER Cache**: Pulls full hourly weather parameters (87,660 hours per point) for all 133 points across the 10-year span (1,330 JSON files).
- **CDS Zip-Quirk Fix (`00_unzip_accum.py`)**: Scans and extracts netCDF files that the CDS API returned as disguised ZIP archives.

---

## Differences from Rajasthan
- **Point Count**: 133 points for Tamil Nadu vs 320 points for Rajasthan, reflecting Tamil Nadu's smaller geographical footprint.
- **Elevation Script**: Rajasthan has a dedicated `00c_attach_elevation.py` script downloading ERA5 geopotential. Tamil Nadu uses the 150 m flat proxy (documented as a known limitation above).

---

## Status
**COMPLETE** — 133 points, 240 NetCDF files, 1330 NASA POWER JSON files.

---

## Literature Support

| Component | Reference / Method | Source File / Details |
|---|---|---|
| Population Grid | GADM v4.1 + WorldPop 2020 UN-adjusted 100m raster | Framework doc §1.3 (N6 Novelty) |
| Solar Geometry / SPA | Reda & Andreas (2004) Solar Position Algorithm | `pvlib.location.Location.get_solarposition()` |
| Reanalysis Grid Alignment | ECMWF ERA5 0.25° grid specification | `02_DATA_SOURCES_AND_VARIABLES.md` |
| NASA POWER Cache | NASA POWER API hourly cache | `13_LITERATURE_MAPPING.md` |
| Flat Elevation Assumption | 150m coastal/plain proxy caveat | Framework doc §3 |
