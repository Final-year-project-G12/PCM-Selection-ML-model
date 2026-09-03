# ERA5 Tamil Nadu Pipeline (population-weighted points, sun-event-aligned, 10-year)

Same method as the Rajasthan pipeline your friend built, applied to Tamil
Nadu, so both regions' outputs are directly comparable for the cross-region
clustering step (Objective 1).

## Pipeline overview

```
00a_build_population_grid.py   →  data/processed/population_grid_points.csv
00b_build_suntimes.py          →  data/processed/suntimes.csv
01_download_era5_tamilnadu.py  →  data/raw/era5/points/*.nc
01b_download_nasapower.py      →  data/raw/nasapower/*.json
00_unzip_accum.py              →  (fixes zip-disguised-as-.nc files in place)
02_combine_tamilnadu.py        →  data/processed/climate_tamilnadu_points.csv
```

## Run order

```
python 00a_build_population_grid.py
python 00b_build_suntimes.py
python 01_download_era5_tamilnadu.py
python 01b_download_nasapower.py
python 00_unzip_accum.py
python 02_combine_tamilnadu.py
```

Each script is resumable — safe to Ctrl-C and re-run.

## What's different from the old (222-city, fixed-hour, 2-year) TN pipeline

| | Old pipeline | This pipeline |
|---|---|---|
| Sampling locations | 260+ hand-picked named cities | population-weighted 0.25° grid cells covering ~87.5% of state population |
| Time sampling | fixed hourly, all 24h/day | 3 events/day: sunrise, solar noon, sunset (astronomically computed per point/date via pvlib) |
| Study period | 2024–2025 (2 years) | 2016–2025 (10 years) |
| Cross-check source | ERA5 only | ERA5 **and** NASA POWER, side by side per row |
| Output | `climate_tamilnadu_all.csv` | `climate_tamilnadu_points.csv` |

## Requirements

```
pip install geopandas rasterio requests pandas numpy xarray netCDF4 pvlib scipy cdsapi
```

`geopandas`/`rasterio` are only needed for `00a`.

## Notes

- **Puducherry**: GADM's "Tamil Nadu" polygon does not include the
  Puducherry union territory enclaves. Set `INCLUDE_PUDUCHERRY = True` in
  `00a_build_population_grid.py` if you want them folded in.
- **Elevation**: population points don't carry elevation, so
  `02_combine_tamilnadu.py` uses a flat 150m approximation for solar-geometry
  calculations (Tamil Nadu's population sits mostly on the coastal plain
  and interior plateau — lower than Rajasthan's 300m approximation).
- **First day of the dataset**: 2016-01-01 has no prior-day predecessor
  hour for deaccumulation if a sun event's window touches hour 0 UTC —
  affected `era5_GHI`/related columns for that one day come out `NaN`
  rather than a wrong value. Every other month boundary is bridged
  automatically.
- If you already ran the *old* 222-point/2-year pipeline in this same
  project folder, its files live under `data/raw/era5/grid/` and are
  untouched by any script here — the new pipeline uses entirely separate
  paths (`data/raw/era5/points/`).

---

## Phase 3–5: Climate Signature + PCM Feasibility Filtering

Once the raw climate data is acquired and processed (scripts 00–02 above), the
next stage builds climate signatures for each point and filters PCM candidates
against climate-specific design constraints. Located in `tamilnadu_pipeline/`:

```
04b_climate_signature.py        →  data/processed/signatures/climate_signature_tamilnadu.csv
05b_cluster_tamilnadu.py        →  data/processed/cluster_profiles_tamilnadu.csv
07_feasibility_filter.py        →  data/processed/feasibility_survivors_tamilnadu.csv
                                    data/processed/feasibility_survivors_tamilnadu_kappa_calibrated.csv
```

### `04b_climate_signature.py` (PHASE 3)

Reduces each point's 10-year daily and sun-event records to a single
**climate-signature vector** — the summary that Phase 4 (clustering) actually
operates on. Builds:

- **Tier 1**: Per-event aggregates (sunrise, solar noon, sunset temperature,
  humidity, wind, irradiance at each event)
- **Tier 2**: Daily integrals and indices (GHI, clearness, cloudiness, HDD18,
  CDD24, diurnal temperature range, seasonal variation)
- **PCM-facing quantities**: Tm_target (the target storage temperature for
  SWH), Tm_target_capped (climate-adjusted cap), L_required (latent-heat
  requirement for PCM sizing)
- **Interactions**: Five terms combining daily/hourly features to capture
  cycling stress, condensation risk, convective loss, and autonomy demand
- **PCA**: Dimensionality reduction on correlated temperature/pressure block

**METHODOLOGY NOTE (Corrected 2026-08-31):**

L_required is computed as **PCM's literature-anchored fractional share** of
total night-discharge thermal delivery, not 100% of the load alone. Avargani
et al. (2021)'s own system delivers the 300 L benchmark via integrated
collector + PCM tank + sensible-heat tank; literature on combined
sensible-latent SWH reports PCM contributing 40–78% of total delivery (Zhao
2022, Huang 2020, Abdelsalam 2020, Koželj 2021). Formula:

```
L_required = (SHARE_PCM * Q_night) / m_PCM
```

with SHARE_PCM = 0.5 (central estimate; range 0.4–0.7). This shifts from an
all-latent, zero-candidate baseline to a combined sensible+latent model where
majority of candidates survive Phase 5 filtering. See `04b_climate_signature.py`'s
docstring for full rationale, or CLAUDE.md §3.1 for the complete methodology
justification and Phase 5 guidance.

**HOW TO RUN:** `python 04b_climate_signature.py`

### `05b_cluster_tamilnadu.py` (PHASE 4)

Clusters the signature points using Gaussian Mixture Models (GMM) and
Agglomerative Clustering to identify distinct **climate regimes**. Each regime
becomes a "Level A cluster" with:

- Representative climate profile (mean Tm_target, L_required, monsoon_index, etc.)
- Cluster-level ground-truth dataset used by Phase 7 (charging feasibility modeling)

**Output:** `cluster_profiles_tamilnadu.csv` — one row per cluster with all
signature columns aggregated to cluster level, plus `Tm_target_capped_C` and
`L_required_kJ_per_kg` re-derived per cluster.

**HOW TO RUN:** `python 05b_cluster_tamilnadu.py`

### `07_feasibility_filter.py` (PHASE 5)

Hard-filters the shared PCM candidate database against each cluster's 8
design constraints (melting window, absolute Tm band, latent-heat floor,
cycling endurance, supercooling, charging feasibility, corrosion veto,
safety flags). Produces two outputs:

1. **PRIMARY (`feasibility_survivors_tamilnadu.csv`)**: Fixed κ=0.7 latent-heat
   floor, with full diagnostic audit trail (per-cluster, per-constraint results).
   Expected to show the baseline (κ=0.7 against old L_required was a near-zero-survivor
   case, demonstrating why calibration was needed).

2. **COMPANION (`feasibility_survivors_tamilnadu_kappa_calibrated.csv`)**: Per-cluster
   calibrated κ, stepped down from 0.7 until 8–20 candidates survive. Includes
   `breakeven_kappa` column (actual threshold each candidate sits at) for ranking.

**VALIDATION:** After Phase 3's L_required correction, re-run this script and
verify calibrated κ lands in the 0.5–0.7 range (much higher than the prior
0.2–0.3), validating the SHARE_PCM=0.5 assumption. Report both outputs
together: broken assumption (old κ=0.7 → zero survivors) → diagnosis (L_required
ceiling) → correction (SHARE_PCM factorization) → verification (new κ resets
higher).

**HOW TO RUN:** `python 07_feasibility_filter.py`

---

## Next: cross-region clustering (Objective 1)

Once all four regions (Tamil Nadu, Rajasthan, + the other two) have both
climate signatures (`climate_{region}_points.csv`) and cluster profiles
(`cluster_profiles_{region}.csv`), run `05_cluster_regions.py` to combine them
into one climate-feature table, cluster it (Gaussian Mixture Models over
standardized signature features, weighted by population), and identify
representative climate regimes for the entire country. Then use Phase 5's
PCM feasibility outputs to rank Top-2/Top-3 candidates per regime via MCDM.
