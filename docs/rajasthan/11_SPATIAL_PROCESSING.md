# 11 — Spatial Processing Audit

## ERA5 grid resolution

ERA5 single-levels reanalysis is natively ~0.25° (~28 km). The population-sampling grid in
`00a_build_population_grid.py` is deliberately built at the **same** 0.25° resolution and aligned to
ERA5's own grid origin (`lat=90.0, lon=-180.0`), specifically so each selected sampling point's cell
center coincides exactly with an ERA5 grid node — not an approximation choice made for convenience,
but a design decision that removes a whole class of population-to-ERA5-cell misalignment error.

## Rajasthan boundary handling

GADM v4.1 admin-level-1 boundary, filtered to `NAME_1 == "Rajasthan"`, first matching geometry used
(`rj.geometry.iloc[0]`) — correct for a state that should have exactly one admin-1 polygon record; no
dissolve/union safeguard exists if GADM ever returned multiple rows for the same state name, which
would silently use only the first and drop the rest (not observed to be a problem here, but a latent
risk if GADM's schema changes).

## Population aggregation and grid-cell selection

WorldPop 100 m raster clipped to the Rajasthan boundary, negative nodata sentinels zeroed, pixels
binned into 0.25° cells via a per-raster-row `np.bincount` (a deliberate memory/performance choice
over a full 2D meshgrid, justified in-code by WorldPop's pixel count at this resolution). Cells
ranked by population descending; minimal set covering `COVERAGE_TARGET=87.5%` of total state
population retained — **320 points**, weights renormalized over just that selected subset.

## Grid-cell selection vs. nearest-neighbor vs. interpolation

Every ERA5/geopotential lookup in this pipeline uses **nearest-neighbor snapping**, never
interpolation:
- `extract_nearest()` in `02_combine_rajasthan.py`: two **independent 1-D `argmin`s** on the lat and
  lon axes separately (`li = argmin(|lat_arr - lat|)`, `lo = argmin(|lon_arr - lon|)`), done once per
  point (not per event). This is mathematically correct for a regular rectilinear lat/lon grid (which
  ERA5 is) — it is *not* a true 2D nearest-neighbor search and would be wrong for a curvilinear or
  irregular grid, but that mismatch does not apply here.
- `attach_elevation()` in `00c_attach_elevation.py`: identical nearest-neighbor pattern against the
  geopotential grid.

No bilinear or other interpolation is used anywhere for extracting a point value from the ERA5 grid
— every point simply inherits its containing 0.25° cell's value exactly. This is a defensible choice
given the population-grid's own alignment to ERA5's grid (each sampling point is, by construction,
very close to its cell's own grid node), but it does mean two sampling points that happen to fall in
the same 0.25° ERA5 cell will receive numerically identical ERA5 readings — expected and harmless
given the sampling design, but worth stating explicitly (it is not an interpolation artifact; it is
the intended behavior of grid-cell-based climate sampling).

## Edge cells / missing cells

No explicit handling of ERA5 grid edge cells beyond the standard 0.5° bounding-box padding applied
before every ERA5/geopotential download (`load_points_bbox(pad=0.5)`, duplicated — not
shared/imported — identically in `00c_attach_elevation.py` and `01_download_era5_rajasthan.py`).
This padding is generous enough (roughly two ERA5 grid cells) that no point should sit at the exact
edge of the downloaded domain, but no automated check confirms this for every one of the 320 points.

## Area weighting

Not applicable in the geometric sense (no spatial-average-over-area computation exists in this
pipeline) — the relevant "weighting" concept here is **population weighting**, applied at two
distinct points: (1) sample-selection (which 320 of the possible ~thousands of 0.25° cells to keep,
via the 87.5%-coverage rule) and (2) cluster-profile reporting (population-weighted means in
`cluster_profiles_rajasthan.csv`). It is explicitly **not** applied a third time inside the GMM
clustering fit itself — see `06_PHASE_4_AUDIT.md` for why double-weighting was deliberately avoided.

## Are all Rajasthan cells treated equally?

No, and this is by design: only the minimal set of highest-population cells covering 87.5% of the
state's population is sampled — genuinely low-population cells (large stretches of the Thar Desert,
for instance) are systematically underrepresented or absent from the 320-point set. This is the
correct choice for a project whose deliverable is a *population-relevant* PCM recommendation, but it
means the climate signature and cluster regimes are **not** representative of Rajasthan's full
geographic climate variability — they are representative of where Rajasthan's people actually live.
State this distinction explicitly in any write-up that shows a spatial map, since a reader could
otherwise mistake the 320-point coverage for a uniform state-wide climate survey.

## Elevation — grid-cell-mean caveat

ERA5's native ~0.25° (~28 km) grid means its orography value for any point is the **mean elevation
of the entire grid cell**, not the point's exact local elevation. Explicitly documented as an
accepted, undressed limitation for Rajasthan specifically (whose terrain is comparatively flat,
mostly 200–500 m) — the same caveat text in `00c_attach_elevation.py`'s docstring notably references
Uttarakhand's 200 m–7,000 m+ range as the case where this matters far more, indicating the comment is
shared/reused across the project's regional pipeline variants rather than written fresh for
Rajasthan. Confirmed ground-truthed: Rajasthan elevations in `population_grid_points.csv` show no
outliers beyond the `[−420, 8850]` m plausibility check.

## Is the spatial method appropriate for a climate-adaptive PCM system?

Yes, for the stated purpose. Nearest-neighbor grid-cell sampling at ERA5's native resolution,
population-weighted point selection, and grid-cell-mean elevation are all internally consistent,
correctly-reasoned choices for a project whose downstream goal is *regime-level* PCM recommendations
(one recommendation per climate cluster covering many points), not point-exact microclimate
modeling. The main documented limitation (grid-cell-mean elevation smoothing out local relief) is
correctly flagged as more consequential for a future high-relief state (Uttarakhand) than for
Rajasthan itself.

## Literature support

GADM and WorldPop are cited as data-source products (not peer-reviewed methodology claims). No
additional spatial-methodology citation (e.g., for the grid-alignment or nearest-neighbor-snapping
choices specifically) was found or is strictly needed — these are standard, correctly-applied GIS
practices for regular-grid reanalysis data, not novel methodological claims requiring independent
literature support.
