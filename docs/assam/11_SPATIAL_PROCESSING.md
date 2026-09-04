# 11 — Spatial Processing Audit: Grid Sampling & Coverage (Assam)

## ERA5 Grid Resolution and Alignment

ERA5 single-levels reanalysis operates on a native 0.25° × 0.25° (~28 km) rectilinear grid. In `00a_build_population_grid.py`, sampling coordinates are built directly on the **same 0.25° resolution, aligned to ERA5's grid origin**. Each sampling point's coordinate coincides with an ERA5 grid node, removing interpolation and alignment distortion by design.

---

## Administrative Boundary & Population Sampling

- **State Boundary**: Extracted from GADM v4.1 administrative level 1 (`NAME_1 == "Assam"`).
- **Population Aggregation**: WorldPop 100m unconstrained resolution raster (India, UN-adjusted 2020) was intersected with the Assam polygon. Grid cells were ranked in descending order of population, and the minimal prefix required to satisfy coverage was extracted.
- **Authoritative Point Count**: Exactly **129 population-weighted points** (`ASP_0001` through `ASP_0129`), achieving **87.8% cumulative population coverage** of Assam.
  *(Historical Note: Stale working notes occasionally cited 128 points and 87.5% coverage; audit confirms all 129 slots `ASP_0001` to `ASP_0129` are fully populated and active).*
- **Primary Dataset**: `data/processed/population_grid_points.csv`.

---

## Spatial Distribution Across Final $K=3$ Climate Regimes

The 129 points partition into the final locked $K=3$ Gaussian Mixture Model regimes as follows:

| Regime ID | Spatial Points | Pct of Sites | Total Population Covered | Climatological Character | Medoid Station ID |
|---|---|---|---|---|---|
| **Cluster 0** | 33 | 25.6% | 4,757,890.5 | Brahmaputra valley high-insolation regime | **`ASP_0012`** |
| **Cluster 1** | 61 | 47.3% | 4,271,199.2 | Central & lower valley humid monsoon corridor | **`ASP_0092`** |
| **Cluster 2** | 35 | 27.1% | 2,466,324.4 | Highland & southern foothill transition regime | **`ASP_0028`** |
| **Total** | **129** | **100.0%** | **~11.50M (87.8% coverage)** | — | — |

*Spatial Reality*: The 129-point selection is **population-representative**, concentrating points along the densely settled Brahmaputra and Barak river basins. Sparsely populated hill tracts (parts of Karbi Anglong and Dima Hasao) have fewer sample points. Furthermore, these clusters represent **statistical climate regimes**, not strictly contiguous administrative or geographic zones.

---

## Topographic Baseline Elevation: Fixed 100 m

Assam uses a fixed baseline elevation of `DEFAULT_ALT_M = 100` across all 129 coordinates:
- This accurately captures the Brahmaputra alluvial plains where the vast majority of Assam's population resides.
- For hill transition points, this default represents an accepted, documented baseline proxy that eliminates artificial discontinuities in pressure-dependent solar geometry equations.

---

## Population Weighting Application

Population weighting is applied strictly at two well-defined stages:
1. **Sample Selection**: Defining the 129 grid coordinates required to reach 87.8% population coverage.
2. **Cluster Reporting**: Computing population-weighted averages for thesis tables and profiles.
3. **Clustering Invariance**: Population weights are **not** applied inside the GMM clustering algorithm itself, preventing distortion of the physical climate-feature covariance structure.
