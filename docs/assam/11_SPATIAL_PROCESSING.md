# 14 — Spatial Processing Audit (Assam)

## ERA5 grid resolution and alignment

ERA5 single-levels at 0.25° (~28 km). The population-sampling grid in `00a_build_population_grid.py`
is built at the **same 0.25° resolution, aligned to ERA5's own grid origin** — each sampling point's
cell center coincides exactly with an ERA5 grid node. This removes population-to-ERA5-cell
misalignment error by design.

## Assam boundary handling

GADM v4.1 admin-level-1, `NAME_1 == "Assam"`, `rj.geometry.iloc[0]` (first matching geometry).
Assam's state boundary is a single polygon in GADM — no multi-part geometry issue expected.

## Population aggregation and grid-cell selection

WorldPop 100m raster clipped to Assam boundary, negative nodata sentinels zeroed, pixels binned
into 0.25° cells via per-raster-row `np.bincount`. Cells ranked by population descending; minimal
set covering `COVERAGE_TARGET=87.5%` of total Assam population retained — **128 points**, weights
renormalized over that selected subset.

**Point IDs**: `ASP_0001` through `ASP_0129` (128 active points from a 129-slot ID space; some
intermediate IDs rejected during boundary clipping).

## Nearest-neighbor snapping

Every ERA5 lookup uses **nearest-neighbor snapping** — two independent 1-D `argmin`s on the lat
and lon axes separately. Correct for a regular rectilinear lat/lon grid (ERA5's native grid). No
bilinear interpolation.

Two sampling points that fall in the same 0.25° ERA5 cell receive numerically identical ERA5
readings — expected and harmless given the sampling design alignment.

## Assam-specific spatial coverage

| Cluster | n_points | Coverage character |
|---|---|---|
| 0 | 24 | Northeast hill/transition zone (Karbi Anglong, Arunachal fringe) |
| 1 | 52 | Brahmaputra valley mainstream (Guwahati, Nagaon, Sivasagar) |
| 2 | 11 | Barak valley / southern Assam (Cachar, Hailakandi) |
| 3 | 41 | Western plains and char areas (Kamrup, Barpeta, Dhubri) |

The 128-point selection **is not a uniform state-wide survey** — it is population-representative.
Genuine sparsely-populated hill districts (parts of Karbi Anglong, Dima Hasao) are underrepresented
relative to their geographic area. State this in any write-up that shows a spatial map.

## Elevation: 100m fixed default (Assam-specific caveat)

Unlike Rajasthan (which had `00c_attach_elevation.py` attaching per-point ERA5 geopotential),
Assam uses `DEFAULT_ALT_M = 100` for all 128 points. This is the appropriate default for the
Brahmaputra plains and river valley (most of Assam's population), but:

- **Hill district points** (Cluster 0, Karbi Anglong / Dima Hasao / Arunachal fringe) may be at
  300–900m+ in reality — the 100m default underestimates their elevation.
- This affects: `era5_P_atm` approximation, pvlib Ineichen clear-sky model (altitude-dependent
  turbidity), and the `elev_proxy` signature index.
- This is a documented, accepted approximation, not an undiscovered error. It is less consequential
  here than it would be for Uttarakhand (the project's next state with true montane terrain).

## Population weighting — where it is and is not applied

Population weighting is applied at: (1) sample-selection (which 128 of the possible ~thousands of
0.25° cells to keep, via the 87.5%-coverage rule) and (2) cluster-profile reporting
(population-weighted means in `cluster_profiles_assam.csv`). It is explicitly **not** applied
a third time inside the GMM clustering fit itself — see `06_PHASE_4_AUDIT.md` for why
double-weighting was deliberately avoided.

## Literature support

GADM and WorldPop are data-source products. No additional spatial-methodology citation needed —
standard GIS practices for regular-grid reanalysis data.
