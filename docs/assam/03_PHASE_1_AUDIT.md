# 03 — Phase 1 Audit: Spatial Grid & Data Collection

**Script(s)**: `00a_build_population_grid.py`, `00b_build_suntimes.py`, `01_download_era5_assam.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`

**Status**: COMPLETE for the current 80-point dataset; historical 129-point figures are superseded.

---

## Population-Weighted Spatial Grid (`00a_build_population_grid.py`)

- **Source Data**: WorldPop 100m unconstrained resolution raster (India, UN-adjusted 2020) intersected with the GADM v4.1 administrative boundary for Assam (`NAME_1 == "Assam"`).
- **Sampling Methodology**:
  1. Raster population counts are aggregated to ERA5's native 0.25° × 0.25° grid cells.
  2. Grid cells whose centroids fall within the administrative polygon are ranked in descending order of population.
  3. The minimal prefix of cells required to meet or exceed the target coverage is retained.
- **Current audited grid count**: Exactly **80 population-weighted points** (`ASP_0001` through `ASP_0080`), achieving **87.8% cumulative population coverage** across Assam.
  *(The earlier 129-point grid is a superseded historical version and must not be mixed with current downstream outputs.)*
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
- **Parameters**: Surface solar radiation downwards (`ssrd`), clear-sky surface solar radiation downwards (`ssrdc`), surface thermal radiation downwards (`strd`), 2m temperature (`t2m`), 2m dewpoint (`d2m`), 10m wind components (`u10`, `v10`), **surface pressure (`sp` / `surface_pressure`, not mean sea level pressure `msl`)**, total cloud cover (`tcc`), total precipitation (`tp`), and direct beam solar radiation (`avg_sdirswrf`, matching ERA5 fields `msdwswrf`/`fdir`/`msdrswrf`).
- **Operational Tracking**: `download_status_points.csv` ensures idempotent, resumable retrieval.
- **Format Normalization**: `00_unzip_accum.py` detects and decompresses CDS zip containers delivered with `.nc` extensions.

---

## NASA POWER Retrieval (`01b_download_nasapower.py`)

- **Product**: NASA POWER **hourly-only** point API (`ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS10M`). There is no daily-product request, no `T2M_MAX`/`T2M_MIN` (derived instead in `02b` from the 24 hourly `T2M` values), and **`PRECTOTCORR` is not downloaded** — `monsoon_index` is computed from ERA5's event-sampled precipitation instead.
- **Coverage**: 2016–2025 (10-year period) for the current 80 spatial coordinates, `ASP_0001` through `ASP_0080`.
- **Operational Storage**: Cached locally in `data/raw/nasapower/power_{point_id}_{year}.json`.
- **Current audited output**: `daily_aggregates_assam.csv` contains **292,160 rows** (80 points × 3,652 dates), covering **2016-01-02 to 2025-12-31**. The earlier 467,367-row count belongs to the superseded 129-point dataset. Do not describe every row as a complete 24-hour day unless the current aggregation and missing-hour rules have been independently verified.

---

## Current Dataset Reconciliation

The original Phase 1 report described a 129-point grid and 467,367 daily aggregate rows. Those values refer to an earlier dataset version. The latest audited Assam Objective 1 results supersede them:

| Item | Current audited value |
|---|---:|
| Active population-weighted points | 80 |
| Point ID range | `ASP_0001`–`ASP_0080` |
| Population coverage | 87.8% |
| Daily aggregate rows | 292,160 |
| Daily aggregate date range | 2016-01-02 to 2025-12-31 |

The current daily aggregate row count is 80 × 3,652 dates. If the sun-event dataset contains three events for every point on all 3,653 dates from 2016-01-01 through 2025-12-31, its expected row count would be 876,720. This is a mathematical expectation, **not a verified output count** in the available audit evidence.

The current clustering audit also uses K=3, with cluster sizes 27 / 28 / 25 and medoids `ASP_0003`, `ASP_0036`, and `ASP_0080`. These are downstream context, not Phase 1 download-completion evidence.

**Do not retain the 129-point / 467,367-row statements as current values.** The Rajasthan Phase 1 document is a separate state's pipeline and must not be used to fill Assam-specific implementation details or output counts.

---

## Audit Findings & Deviations

1. **Current Grid**: The latest audited Objective 1 grid contains 80 points (`ASP_0001` to `ASP_0080`), with 87.8% population coverage. The previous 129-point statement is historical and no longer describes current pipeline outputs.
2. **Topographic Elevation**: The Assam data-source description records a fixed 100 m valley baseline. Do not import Rajasthan's per-point geopotential workflow into Assam unless the Assam scripts and outputs are changed and verified.
3. **Cross-Source Reliability**: The reported Phase 2 cross-source agreement result is a 1.1% mean bias error for daytime GHI, with the decision `BACKBONE` recorded in `bias_decision_assam.txt`. Recheck it if the point set, time alignment, or preprocessing changes.
