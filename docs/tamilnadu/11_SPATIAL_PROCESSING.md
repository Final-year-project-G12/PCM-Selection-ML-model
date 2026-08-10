# 11 — Spatial Processing Audit

## Population Bounding Box
- The state boundary of Tamil Nadu is extracted from the GADM spatial database.
- A WorldPop 100 m raster is cropped to the boundary.
- Pixel populations are aggregated onto a 0.25° grid to align with the ERA5 grid resolution.
- Only cells required to cover **87.5%** of the total state population are kept, yielding **133 points** (`TNP_0001` to `TNP_0133`).
- This population-weighting ensures that the climate regimes discovered reflect regions where domestic heating demand is actually present.

## Nearest-Neighbor Snapping
- Coordinates snap to the nearest ERA5 grid cell center using argmin on distance.
- Since the grid was pre-aligned to ERA5 coordinates, the snap maps exactly 1:1 to grid centers, preventing grid cell collapse (two points snapping to the same cell).

## Elevation Approximation
- Tamil Nadu has a flat default terrain assumption of 150 m. This is a reasonable approximation for the coastal plains and interior tablelands where the majority of the population lives, but ignores the high elevation of the Western Ghats (e.g., Ooty, ~2,240 m). For a multi-state deployment (especially Uttarakhand), actual elevation extraction is mandatory.
