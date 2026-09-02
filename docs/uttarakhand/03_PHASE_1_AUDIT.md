# 03 — Phase 1 Audit: Data Collection

**Scripts**: `config.py`, `00a_build_population_grid.py`, `00b_build_suntimes.py`,
`01_download_era5_uttarakhand.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`

**Status**: **RUN** — evidenced by every downstream artefact in `data/plots/`. The raw files
(`data/raw/`) and `population_grid_points.csv` / `suntimes.csv` are listed in
`era5-uttarakhand/.gitignore` and are therefore **not present in this repository**.

---

## Purpose

Define *where* and *when* the pipeline samples Uttarakhand, then pull two independent climate
products for exactly those places and instants.

Two deliberate departures from a naive design, both stated in `README.md`:

1. **Population-weighted locations instead of a uniform state grid** — so results are
   representative of where domestic hot-water demand actually is, and so the sampling design is
   defensible against "why these locations?"
2. **Astronomically computed sun-event times instead of fixed clock hours** — sunrise, solar noon
   and sunset, per point, per day, so every sample sits at a physically meaningful instant of the
   solar cycle rather than at an arbitrary UTC hour.

---

## Inputs

| Input | Source | Cached to |
|---|---|---|
| State boundary | GADM v4.1, India admin level 1, GeoJSON, `NAME_1 == "Uttarakhand"` | `data/raw/boundary/gadm41_IND_1.json` |
| Population raster | WorldPop unconstrained global mosaic, India, UN-adjusted, 100 m, 2020 | `data/raw/population/ind_ppp_2020_UNadj.tif` (~1.5–2 GB) |
| ERA5 reanalysis | Copernicus CDS, `reanalysis-era5-single-levels`, hourly | `data/raw/era5/points/*.nc` |
| NASA POWER | `power.larc.nasa.gov/api/temporal/hourly/point`, community `RE` | `data/raw/nasapower/*.json` |

Credentials: `.cdsapirc` in the pipeline folder, or `CDSAPI_URL` / `CDSAPI_KEY` environment
variables. NASA POWER needs no key.

---

## Processing

### `config.py` — shared path anchoring

Not run directly. Anchors every path to `BASE_DIR = Path(__file__).resolve().parent`, so scripts
work from any working directory:

```
RAW_GRID_DIR                = data/raw/era5/grid/          (old full-state grid — untouched)
RAW_POINTS_DIR              = data/raw/era5/points/        (this pipeline)
DOWNLOAD_STATUS_FILE        = data/raw/era5/download_status.csv          (old)
POINTS_DOWNLOAD_STATUS_FILE = data/raw/era5/download_status_points.csv   (this pipeline)
RAW_POPULATION_DIR          = data/raw/population/
RAW_BOUNDARY_DIR            = data/raw/boundary/
RAW_POWER_DIR               = data/raw/nasapower/
POPULATION_GRID_FILE        = data/processed/population_grid_points.csv
SUNTIMES_FILE               = data/processed/suntimes.csv
COMBINED_POINTS_FILE        = data/processed/climate_uttarakhand_points.csv
PREPROCESSED_DIR            = data/preprocessed/
PLOTS_DIR                   = data/plots/
```

`ensure_data_dirs()` creates nine directories. `load_cds_credentials()` prefers environment
variables and falls back to parsing `.cdsapirc` (read `utf-8-sig`, tolerating a BOM), with
explicit error messages for missing/empty/incomplete config.

`CLIMATE_COMBINED_FILE`, `PROCESSED_NAMED_DIR` and `PROCESSED_GRID_DIR` are declared but no current
script writes to them — leftovers from the pre-points pipeline.

### `00a_build_population_grid.py` — sampling design

Method, verbatim from the docstring:

1. Clip the WorldPop raster to the Uttarakhand boundary polygon.
2. Aggregate pixel population onto a 0.25° lat/lon grid — "deliberately the same resolution as
   ERA5's native grid (not finer), and deliberately anchored to ERA5's own grid origin (lat=90.0,
   lon=−180.0, multiples of 0.25°) so each selected cell's center lands exactly on an ERA5 grid
   node."
3. Rank cells by population descending, keep the minimal prefix covering ≥ `COVERAGE_TARGET`.
4. Write `population_grid_points.csv`: `point_id, lat, lon, population, weight`.

Implementation details worth recording:

- **Memory-conscious aggregation**: pixels are binned row-by-row with `np.bincount` rather than
  building a full `(row × col)` index mesh — "at WorldPop's 100m resolution, Uttarakhand is tens of
  millions of pixels, so a per-pixel meshgrid would be needlessly memory-heavy."
- **Nodata**: `rio_mask(..., nodata=0, filled=True)` then `band[band < 0] = 0.0` — WorldPop's
  negative nodata sentinels are zeroed.
- **Selection**: `cutoff = (cumulative / total >= 0.875).idxmax()`, keeping `df.iloc[:cutoff+1]`.
- **Weights**: renormalised over the **selected** subset, not the state total.
- **Resumability**: cached files above `min_size_bytes` are skipped; the WorldPop download retries
  up to 5 times and resumes with HTTP `Range` requests.
- **Population year**: a single static 2020 snapshot is applied to the whole 2016–2025 study
  period, because "WorldPop doesn't publish a distinct India raster per year at this resolution."
  Declared as "a standard simplifying assumption … not something this script tries to correct for."

### `00b_build_suntimes.py` — sun-event time table

```python
loc = pvlib.location.Location(latitude=row.lat, longitude=row.lon, altitude=0, tz="UTC")
result = loc.get_sun_rise_set_transit(dates, method="spa")
```

`method="spa"` is **pinned explicitly**. `noon` in the output is pvlib's `transit`, i.e. true solar
transit, not clock noon. Output columns: `point_id, date, event, time_utc`.

Resumable: skipped entirely if every current `point_id` already appears in `suntimes.csv`;
`--force` rebuilds.

### `01_download_era5_uttarakhand.py` — ERA5 download

Two CDS requests per month, by ERA5 convention:

| Group | Type | Variables | Hours |
|---|---|---|---|
| `instant` | analysis (AN) | `2m_temperature`, `2m_dewpoint_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_cloud_cover`, `surface_pressure` | `INSTANT_HOURS` |
| `accum` | forecast (FC) | `surface_solar_radiation_downwards`, `mean_surface_direct_short_wave_radiation_flux`, `surface_thermal_radiation_downwards`, `total_precipitation` | `ACCUM_HOURS` |

`ACCUM_HOURS = INSTANT_HOURS ∪ {(h − 1) mod 24 for h in INSTANT_HOURS}` — every target hour's
immediate predecessor is downloaded so `deaccumulate()` in Phase 2 has something to difference
against.

Bounding box: `load_points_bbox(pad=0.5)` — the envelope of the population points padded 0.5°, not
the full state boundary. With the observed point extents this is approximately
`[N 31.125, W 77.375, S 28.375, E 80.625]`.

Download mechanics:

| Item | Value |
|---|---|
| API calls | 240 (10 years × 12 months × 2 var types) |
| Retry | `MAX_RETRIES = 3`, `RETRY_WAIT = 60 s` |
| Corrupt-file threshold | `< 50,000 bytes` → removed and re-downloaded |
| Skip logic | `StatusTracker.is_done(year, month, var_type)` on `status == "OK"`, plus an on-disk size check |
| Status CSV | `timestamp, year, month, var_type, status, filepath, size_mb, note`, flushed after **every** entry |

### `01b_download_nasapower.py` — cross-check download

| Item | Value |
|---|---|
| Parameters | `ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M` |
| Community / time standard | `RE` / `UTC` |
| Calls | 45 points × 10 years = **450** |
| Output | `data/raw/nasapower/power_{point_id}_{year}.json` |
| Validation | rejects an empty `properties.parameter`; rejects files `< 1000 bytes` |
| Retry / pacing | `MAX_RETRIES = 3`, `RETRY_WAIT = 20 s`, `REQUEST_SLEEP = 1.0 s`, `REQUEST_TIMEOUT = 60 s` |

The **full** hourly cache is kept even though `02` reads only 3 hours/day from it — "only 3 of its
~8760 hours/year get used directly in `02`'s sun-event merge, but the rest isn't wasted."
`02b_build_daily_aggregates.py` re-reads it in full.

### `00_unzip_accum.py` — CDS ZIP-disguised-as-NetCDF fixer

"CDS API v2 sometimes downloads files as .zip even when `download_format: unarchived` is
requested." Detection is by magic bytes (`PK` for ZIP; `CDF` or `\x89HDF` for NetCDF); the fix
extracts the first `.nc` member, verifies it, and moves it over the original path. Scans **both**
`RAW_GRID_DIR` and `RAW_POINTS_DIR`. Idempotent — valid NetCDF reports `[OK]` and is left alone.

---

## Scientific reasoning

**Why population weighting?** The deliverable is a PCM recommendation for domestic solar water
heating. A uniform state grid would give equal weight to uninhabited high-altitude terrain and to
Dehradun. Selecting the minimal set of 0.25° cells covering ≥ 87.5 % of the state's population
makes every recommendation demand-weighted by construction.

**Why sun-event alignment?** Fixed clock hours drift relative to the solar cycle across a 2.25°
longitude span and across a year in which day length varies by roughly four hours. Sampling at
sunrise / transit / sunset guarantees that "noon" always means solar noon at that specific point on
that specific day. `03_plots_raw.py` check B exists to verify this held.

**Why 0.25° and not finer?** Anything finer risks multiple population points snapping to the same
ERA5 cell downstream, which would produce numerically identical readings for supposedly distinct
sampling locations.

---

## Spatial Processing Justification

### ERA5 grid alignment

```python
GRID_RES = 0.25;  ERA5_ORIGIN_LAT = 90.0;  ERA5_ORIGIN_LON = -180.0

lon_cell_idx = floor((x − (−180)) / 0.25)     cell_lon = −180 + (lon_i + 0.5) · 0.25
lat_cell_idx = floor((90 − y)     / 0.25)     cell_lat =   90 − (lat_i + 0.5) · 0.25
```

The stated intent: "This keeps the population→ERA5 mapping 1:1 wherever cells are genuinely
distinct, instead of two nearby population cells silently collapsing onto the same ERA5 node due to
grid misalignment."

**Verified from the artefacts**: every one of the 45 observed point coordinates falls on an
`x.125 / x.375 / x.625 / x.875` value in both axes — exactly the node lattice these formulas
produce. The alignment is real, not merely asserted.

### Boundary handling

`gdf[gdf["NAME_1"] == "Uttarakhand"].geometry.iloc[0]`. If the filter returns empty the script
raises with the full list of available `NAME_1` values — a good failure mode. `.iloc[0]` takes only
the first matching feature; GADM stores each state as a single (possibly multi-part) geometry, so
this is normally correct, but the code does not check whether more than one row matched.

### Selected point set (observed)

Recovered from the 45 marker coordinates and popups embedded in
`data/plots/comprehensive/maps/A2_population_map.html`:

| Metric | Value |
|---|---|
| Points | **45** |
| Point IDs | `UKP_0001` … `UKP_0045`, contiguous, no gaps |
| Latitude range | **28.875 – 30.625 °N** (8 distinct lattice latitudes) |
| Longitude range | **77.875 – 80.125 °E** (10 distinct lattice longitudes) |
| Population covered | **10,475,711** |
| Coverage target | 87.5 % → implied state raster total ≈ 11.97 M |
| Largest cell | `UKP_0001` = 1,061,041 |
| Smallest cell | `UKP_0045` = 85,265 |
| Top-3 share of covered population | 2,950,113 / 10,475,711 = **28.2 %** |

The bounding box is 1.75° × 2.25°, a maximum of 8 × 10 = 80 lattice cells, of which 45 carry enough
population to be selected. Sampling is therefore reasonably dense within the populated part of the
state. The population distribution is strongly top-heavy.

### Nearest-neighbour extraction (applied in Phase 2)

`extract_nearest()` in `02_combine_uttarakhand.py` uses two **independent 1-D `argmin`s** on the
latitude and longitude axes — correct for a regular rectilinear grid, which is ERA5's native
layout. **No bilinear or inverse-distance interpolation.** The chosen node is carried into every
output row as `grid_lat` / `grid_lon`, so the snap is auditable after the fact.

Because `00a` aligned the sampling lattice to the ERA5 lattice, each point should land on its own
distinct ERA5 node. This follows from the alignment but is **not verified anywhere in the
pipeline**; a `groupby(["grid_lat","grid_lon"]).ngroups == 45` check on the combined CSV would
confirm it in one line.

### Elevation handling — the pipeline's central spatial limitation

**No per-point elevation exists anywhere in the pipeline.** `00a` writes only
`point_id, lat, lon, population, weight`; there is no elevation-attachment script. Three different
altitude assumptions coexist:

| Where | Altitude | Effect |
|---|---|---|
| `00b_build_suntimes.py` | **0 m** | sunrise / transit / sunset times |
| `02_combine_uttarakhand.py` | **1200 m** (`DEFAULT_ALT_M`) | pvlib `Location(altitude=…)` → Ineichen clear-sky and solar position |
| `04b_climate_signature.py` | derived: `elev_proxy = mean(era5_P_atm) / 1013.25` | PCA block member → clustering matrix |

The `DEFAULT_ALT_M` comment states the reasoning: "Uttarakhand is mountainous; populated zones
range roughly 200-2000m. Use 1200m as a representative default." The altitude value is **not**
written to the output rows, so the assumption is invisible in the data and recoverable only from
source.

`README_PREPROCESSING.md` is explicit that this is not a footnote:

> **elevation note — this is a real limitation here, not a footnote:** `02_combine_uttarakhand.py`
> uses a flat **1200m** proxy for every point's solar-geometry calculations, not real per-point
> elevation. … Uttarakhand's populated terrain genuinely spans roughly 200m (Terai plains near
> Udham Singh Nagar/Haridwar) to 2000m (hill towns), and elevation drives both solar-geometry
> inputs (air mass, clear-sky irradiance) and the temperature-based indices (HDD18/CDD24, Ta_mean)
> directly. This is plan v3.0's "Repair 2," written with Uttarakhand specifically in mind.

`NEXT_STEPS.md` makes it one of only two "**Do**" items in an otherwise "don't do this now" list,
and suggests two concrete fixes: an SRTM tile lookup, or a lookup against the GADM/WorldPop rasters
`00a` already downloads.

The consequence compounds in Phase 2: `04`'s physical-bounds table sets `era5_P_atm ≥ 850 hPa`
(≈ 1,450 m in a standard atmosphere) and **37.1 % of pressure readings fell below it** and were
NaN'd then imputed — one-sidedly, in the exact column `elev_proxy` is built from. See
`04_PHASE_2_AUDIT.md` Part C.

### Population weighting — where it is and is not applied

| Stage | Weighted? |
|---|---|
| Sample selection (which 45 cells) | **Yes** — the ≥ 87.5 % cumulative rule |
| `weight` column in `population_grid_points.csv` | **Yes** — renormalised over the selected 45 |
| Download, merge, preprocessing, signature | No — carried as metadata only |
| GMM fit (`05`) | **No** — `X` is the `_z` columns; population is not a sample weight |
| Cluster profiles (`05`) | **Yes** — `np.average(g[col], weights=g["population"])` |
| Recommendation cards (`09`) | **Yes**, inherited from the profiles |

Applied exactly twice — at sample selection and at profile reporting — and deliberately not inside
the clustering fit. That avoids double-weighting, since the point set is already
population-representative by construction.

### Why this spatial approach is appropriate

The deliverable is a **per-regime** PCM recommendation, not a microclimate model. A 0.25° ERA5
cell (~28 km) is coarser than Himalayan valley-scale variation, but the recommendation granularity
is the cluster, not the cell. The limitation to state plainly is that **the 45-point set is
population-representative, not area-representative**: sparsely populated high-Himalaya terrain is
under-sampled relative to its land area, and only 3 of 45 points (3.2 % of covered population) form
the coldest regime.

---

## Temporal Processing Justification (Dates, Times, Sunrise/Sunset)

### Study period

`2016-01-01` through `2025-12-31` inclusive — **3,653 days** (10 × 365 + leap days 2016, 2020,
2024). Hard-coded consistently in `00b`, `01`, `01b`, `02` and `02b`.

Expected `suntimes.csv` rows: 45 × 3,653 × 3 = **493,155**. This matches the observed row count of
`climate_uttarakhand_points.csv` exactly (see `04_PHASE_2_AUDIT.md`).

### UTC as the sole time reference

Every timestamp is UTC: `00b` builds `pd.date_range(..., tz="UTC")`; `01` requests UTC hours; `01b`
sends `"time-standard": "UTC"`; `02`'s `decode_time()` returns tz-naive UTC and then
`tz_localize("UTC")` before comparison; POWER keys are parsed with `format="%Y%m%d%H", utc=True`.

**The only IST conversion in the entire pipeline** is `04_preprocess_uttarakhand.py` step 6:

```python
ist = df["time_utc"] + pd.Timedelta(hours=5, minutes=30)
df["ist_hour_decimal"] = ist.dt.hour + ist.dt.minute / 60 + ist.dt.second / 3600
df["solar_hour_angle"] = (df["ist_hour_decimal"] - 12) * 15
```

Uttarakhand spans ~77.9–80.1° E, so solar noon falls at roughly **06:40–06:50 UTC**. Any figure
shown to a general audience needs an explicit UTC→IST note at presentation time.

> `solar_hour_angle` is derived from **IST clock time**, not true solar time — IST's 82.5° E
> reference meridian is east of the whole state, so this column is a clock-hour-angle offset from
> the true solar hour angle by roughly 9–18 minutes of longitude plus the equation of time. It is
> used only as an engineered feature, never in a physics calculation.

### Sun-event times via pvlib SPA

`method="spa"` is pinned explicitly in `00b`. **Altitude 0 m** is used here, which differs from the
1200 m used for irradiance geometry in `02` — sunrise/sunset times are altitude-sensitive at the
minute scale, so the two assumptions are inconsistent, though the magnitude is small relative to
the ±1 h `HOUR_MARGIN` and the 3 h match tolerance.

(By contrast, `compute_solar()` in `02` calls `get_solarposition(times)` with **no** `method=`
argument — see `04_PHASE_2_AUDIT.md` Part A.6.)

### Cross-midnight UTC handling

Treated as a real case, not a hypothetical. From `00b`'s docstring:

> **IMPORTANT — cross-midnight UTC dates are real, not a hypothetical edge case:** Uttarakhand's
> sunrise can fall before 00:00 UTC (i.e. on the *previous* UTC calendar date) for eastern points
> in summer — `time_utc` always reflects the true instant of the event; `date` is the nominal
> (pvlib-assigned) calendar date the event belongs to.

`01`'s `circular_hour_window(hours_observed, margin=1)` handles the consequence: it finds the
**largest unobserved circular gap** in the sorted hour set and takes the complement, padded by
`HOUR_MARGIN = 1` with modulo-24 arithmetic. The docstring gives the failure it prevents: "a plain
numeric min/max across hours like {23, 0, 1, 2} would be nonsensical (min=0, max=23 spans the whole
day)."

**The resolved hour lists for the actual run are not available in the source files** — they are
computed at runtime from `suntimes.csv` and no log is committed.

### De-accumulation predecessor logic and the 2016-01-01 edge case

Because ERA5 accumulated fields need `value(h) − value(h−1)`, `ACCUM_HOURS` includes every target
hour's predecessor. One true edge case is documented: **2016-01-01 has no 2015-12-31 file** to
supply hour 23 as hour 0's predecessor, so that single day's affected `era5_GHI` / `era5_LW_down` /
`era5_precipitation` values come out as a natural `NaN`. Every other month boundary is bridged
because `02` concatenates all months into one continuous sorted series per point *before* calling
`deaccumulate()`.

`deaccumulate()`'s `reset_mask = s.index.hour.isin([1, 13])` is a **fixed** constant while the
downloaded hour set is **dynamic**. That is mathematically safe (hours 1 and 13 either appear in
`ACCUM_HOURS` or the mask selects nothing), and the docstring argues it correctly — but it is a
coupling between a static constant and a runtime-computed hour set that a reader should know about.

### Nearest-in-time matching (the 3-hour rejection window)

`MAX_MATCH_HOURS = 3`, applied independently to the ERA5 series and the NASA POWER series in
Phase 2. **Observed result: zero rows lost** — 493,155 rows written against a theoretical maximum
of 493,155. The tolerance never rejected a match in this run, which is a positive coverage result
but also means it provides no evidence about typical match quality. **The matched timestamp is not
persisted**, so per-row offsets are unauditable from the output.

### Sun-event-aligned vs fixed-clock-hour sampling — the downstream consequence

Because the schema is 3 rows/day rather than 24, every temporal feature in Phase 2 is redefined
over **event occurrences**, not hours. `04_preprocess_uttarakhand.py` leads with this:

> "lag7" = the same sun-event 7 days earlier, not 7 hours earlier.

| Feature family | Window | Real-world meaning |
|---|---|---|
| Hampel filter | ±15 occurrences, centred | ≈ ±15 days at the same sun event |
| Lags | shift 1, 7, 30 occurrences | 1 day / 1 week / 1 month earlier, same event |
| Rolling | trailing 7, 30 occurrences | trailing week / month, same event |
| Deltas | `diff(1)` occurrence | day-over-day change, same event |
| Interpolation | `limit=3` occurrences | gaps up to 3 days, same event |

This is also exactly why `02b_build_daily_aggregates.py` exists: `DTR_proxy = noon − sunrise` is a
lower bound on the true diurnal range because true `Tmax` typically lags solar noon by 1–3 h.

### Seasonal definitions — an internal inconsistency to reconcile

`SEASON_MAP` in `02_combine_uttarakhand.py`:

```
Dec, Jan, Feb  → Winter   (1)
Mar, Apr, May  → Summer   (2)
Jun, Jul, Aug  → Monsoon  (3)          ← JJA, three months
Sep, Oct, Nov  → Retreat  (4)
```

**But `04b_climate_signature.py` computes `monsoon_index` over JJAS — four months:**

```python
jjas = precip[precip.index.month.isin([6, 7, 8, 9])].sum()
row["monsoon_index"] = jjas / total
```

September is in the **Retreat** season by `SEASON_MAP` but is counted in `monsoon_index`, which is
a member of the clustering matrix. Both definitions are individually defensible; they do not match,
and neither is declared authoritative in any source file. A write-up must say which one it means
each time it uses the word "monsoon."

A second, unrelated documentation inconsistency: `03_plots_raw.py`'s docstring describes its
seasonal check as a sanity check against "hot dry Apr-Jun, **NE monsoon Oct-Dec**" — that is not
Uttarakhand's regime. `PREPROCESSING_STEPS.md` describes the correct one for the same plot: "hot
foothill/Terai summer Apr–Jun, **southwest monsoon Jun–Sep**, cold high-altitude winter Dec–Feb."
The plot itself groups by `SEASON_MAP` categories and is unaffected; only the interpretation
guidance in the docstring is wrong.

---

## Literature support

**None present in the source files for Phase 1.** `00b` names "pvlib's SPA algorithm — no manual
equation-of-time code" without a citation; GADM, WorldPop, ERA5 and NASA POWER are named as data
products with their URLs only. No temporal- or spatial-methodology reference appears anywhere in
`era5-uttarakhand/`. See `11_LITERATURE_MAPPING.md` for what must be added before submission.

---

## Validation

| Check | Where | Result |
|---|---|---|
| Boundary filter finds Uttarakhand | `00a`, raises with available `NAME_1` values otherwise | Passed (45 points produced) |
| Downloaded file not corrupt | `01` (`< 50 kB`), `01b` (`< 1 kB`), plus a `properties.parameter` non-empty check | Not independently verifiable — status CSVs are git-ignored |
| ZIP-disguised NetCDF repaired | `00_unzip_accum.py` magic-byte sniff | Not verifiable — no log committed |
| Sun events land at the right time of day | `03_plots_raw.py` check B, in Phase 2 QA | **PASSED** — noon peaks both GHI and T_amb |
| Full point/day/event coverage | implicit in the combined row count | **PASSED** — 493,155 = 45 × 3,653 × 3 exactly |
| POWER cache completeness | `02b`'s printed run summary, quoted in `README_PREPROCESSING.md` | **PASSED** — 45/45 points, 0 skipped, `usable_days = 3653` each, 164,385 point-days |

The last two are the strongest available evidence that both downloads completed: the combined
output reached its full theoretical row count, and `02b` found ≥ 20 of 24 NASA POWER hours on
essentially every day of the 10-year span for every point.

---

## Outputs

| File | Rows | Committed? |
|---|---|---|
| `data/processed/population_grid_points.csv` | 45 | No (git-ignored) |
| `data/processed/suntimes.csv` | 493,155 *(expected)* | No |
| `data/raw/era5/points/era5_UK_points_{yyyy}_{mm}_{instant,accum}.nc` | 240 files *(expected)* | No |
| `data/raw/nasapower/power_{point_id}_{year}.json` | 450 files *(expected)* | No |
| `data/raw/era5/download_status_points.csv` | — | No |
| `data/raw/nasapower/download_status_power.csv` | — | No |
| `data/raw/population/ind_ppp_2020_UNadj.tif`, `data/raw/boundary/gadm41_IND_1.json` | — | No |

---

## Dependencies

`geopandas`, `rasterio`, `requests` (`00a` only); `pvlib`, `pandas` (`00b`); `cdsapi` (`01`);
`requests` (`01b`); standard library only (`00_unzip_accum.py`).

---

## Problems / risks

1. **Two inconsistent altitude assumptions.** `00b` computes sun-event times at 0 m; `02` computes
   solar geometry at 1200 m. Neither file acknowledges the other.
2. **No per-point elevation.** There is no elevation-attachment script. Consequences propagate to
   the Ineichen clear-sky model, to `elev_proxy`, and to the 850 hPa physical bound that destroys
   37 % of the pressure column in Phase 2. This is the single most Uttarakhand-specific weakness in
   the pipeline.
3. **Download completeness is not independently verifiable from the repository.** Both status CSVs
   are git-ignored and no run log is committed; the 493,155-row combined output is strong indirect
   evidence but there is no committed per-file count.
4. **The 45-point set is population-representative, not area-representative** — say so on any
   spatial map.
5. **Static 2020 population** applied to a 2016–2025 period — documented, not corrected.
6. **`config.py` carries dead paths** (`CLIMATE_COMBINED_FILE`, `PROCESSED_NAMED_DIR`,
   `PROCESSED_GRID_DIR`) that no current script writes.
7. **The `monsoon_index` JJAS vs `SEASON_MAP` JJA mismatch** is unreconciled, and `monsoon_index`
   is in the clustering matrix.
8. **`03_plots_raw.py`'s docstring cites the wrong regional climatology** for its seasonal check.

---

## Status

**COMPLETE.** 45 population-weighted points covering 10,475,711 people (87.5 % target), 10 years of
ERA5 and NASA POWER at sun-event-aligned instants, with full point/day/event coverage confirmed
downstream. The design decisions (population weighting, ERA5-lattice alignment, sun-event
alignment, circular hour windows) are sound and well documented in-code. The open items are
elevation and the two altitude assumptions.
