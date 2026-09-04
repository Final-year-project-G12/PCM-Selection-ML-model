# 03 — Phase 1 Audit: Spatial Grid & Data Collection

**Script(s)**: `00a_build_population_grid.py`, `00b_build_suntimes.py`, `01_download_era5_assam.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`

**Status**: COMPLETE (Authoritative Final)

---

## Population-Weighted Spatial Grid (`00a_build_population_grid.py`)

- **Source Data**: WorldPop 100m unconstrained resolution raster (India, UN-adjusted 2020) intersected with the GADM v4.1 administrative boundary for Assam (`NAME_1 == "Assam"`).
- **Sampling Methodology**:
  1. Raster population counts are aggregated to ERA5's native 0.25° × 0.25° grid cells.
  2. Grid cells whose centroids fall within the administrative polygon are ranked in descending order of population.
  3. The minimal prefix of cells required to meet or exceed the target coverage is retained.
- **Authoritative Grid Count**: Exactly **129 population-weighted points** (`ASP_0001` through `ASP_0129`), achieving **87.8% cumulative population coverage** across Assam.
  *(Note: Stale pre-audit documentation occasionally cited 128 points and 87.5% coverage; audit of `population_grid_points.csv` confirms all 129 slots are active, populated, and unique).*
- **Primary Dataset**: `data/processed/population_grid_points.csv`
  - Columns: `point_id`, `latitude`, `longitude`, `population`, `weight`.
- **Topographic Baseline**: Default baseline elevation of 100 m above sea level (representing the Brahmaputra alluvial plains baseline) is used across grid coordinates.

---

## Sun-Event Timestamps (`00b_build_suntimes.py`)

- **Dataset**: `data/processed/suntimes.csv`
- **Methodology**: Evaluates solar geometry via the `pvlib` Solar Position Algorithm (SPA) for every coordinate and calendar day from 2016 through 2025.
- **Output Metrics**: Precise UTC timestamps for astronomical sunrise, solar noon, and sunset.
- **Scientific Role**: Establishes physically grounded sampling times for event-aligned reanalysis data extraction rather than relying on arbitrary fixed UTC intervals.

---

## ERA5 Reanalysis Retrieval (`01_download_era5_assam.py`)

- **CDS Product**: `reanalysis-era5-single-levels` (hourly).
- **Parameters**: Surface solar radiation downwards (`ssrd`), surface thermal radiation downwards (`strd`), 2m temperature (`t2m`), 2m dewpoint (`d2m`), 10m wind components (`u10`, `v10`), mean sea level pressure (`msl`), total cloud cover (`tcc`), total precipitation (`tp`), and direct beam solar radiation (`avg_sdirswrf`).
- **Operational Tracking**: `download_status_points.csv` ensures idempotent, resumable retrieval.
- **Format Normalization**: `00_unzip_accum.py` detects and decompresses CDS zip containers delivered with `.nc` extensions.

---

## NASA POWER Retrieval (`01b_download_nasapower.py`)

- **Product**: NASA POWER hourly and daily meteorology (`ALLSKY_SFC_SW_DWN`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PRECTOTCORR`).
- **Coverage**: 2016–2025 (10 continuous years) for all 129 spatial coordinates.
- **Operational Storage**: Cached locally in `data/raw/nasapower/power_{point_id}_{year}.json`.
- **Authoritative Output**: Feeds into `daily_aggregates_assam.csv` (audited count: **467,367 daily rows** after dropping incomplete days with $<20$ valid hours).

---

## Audit Findings & Deviations

1. **Grid Completeness**: All 129 point IDs (`ASP_0001` to `ASP_0129`) are present, valid, and accounted for in all downstream climate and clustering artifacts.
2. **Topographic Elevation**: Fixed 100 m valley baseline was retained without per-point geopotential extraction. While mountainous borders (Karbi Anglong and Dima Hasao) have higher true relief, the 100 m baseline serves as an approved, consistent proxy across the state.
3. **Cross-Source Reliability**: The dual download of ERA5 and NASA POWER enabled the Phase 2 cross-source agreement audit (`03b_agreement_analysis_assam.py`), confirming that ERA5 GHI has only a 1.1% mean bias in Assam.
