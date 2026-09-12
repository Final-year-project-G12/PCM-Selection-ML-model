# ERA5 Uttarakhand Pipeline

Builds a solar/climate dataset for Uttarakhand, sampled at **population-weighted
locations** and **astronomically computed sun-event times** (sunrise, solar
noon, sunset) rather than a uniform grid on fixed clock hours. Pulls both
ERA5 reanalysis and NASA POWER for the same points/times so the two
independent sources can be cross-checked against each other — then cleans,
builds a per-point climate signature, clusters the state into climate
regimes, and screens/ranks PCM (phase-change-material) candidates for each
regime, ending in one recommendation card per regime.

## Pipeline overview

```
PHASE 0/1 — SAMPLING DESIGN + RAW DOWNLOAD
  00a_build_population_grid.py    →  data/processed/population_grid_points.csv
  00b_build_suntimes.py           →  data/processed/suntimes.csv
  00c_attach_elevation.py         →  population_grid_points.csv gains `elevation_m`
                                      (real per-point elevation from ERA5 geopotential —
                                      run before 02_combine_uttarakhand.py; see "Notes /
                                      known limitations" below)
  01_download_era5_uttarakhand.py →  data/raw/era5/points/*.nc
  01b_download_nasapower.py       →  data/raw/nasapower/*.json
  00_unzip_accum.py               →  (fixes zip-disguised-as-.nc files in place)

PHASE 2 — COMBINE + DAILY AGGREGATES
  02_combine_uttarakhand.py       →  data/processed/climate_uttarakhand_points.csv
  02b_build_daily_aggregates.py   →  data/processed/daily_aggregates_uttarakhand.csv
                                      data/processed/tier2_signature_uttarakhand.csv

PHASE 2 QA — RAW DATA CHECKS (before any cleaning)
  03_plots_raw.py                 →  data/plots/raw/*.png
  03b_interactive_raw_qa.py       →  data/plots/raw_interactive/*.html

PHASE 2 — PREPROCESSING / QUALITY CONTROL
  04_preprocess_uttarakhand.py    →  data/preprocessed/uttarakhand_cleaned_physical.csv
                                      data/preprocessed/uttarakhand_cleaned_scaled.csv
                                      data/preprocessed/scalers.pkl, qc_report.txt, ...

PHASE 2 QA — POST-CLEANING CHECKS
  04c_postprocess_plots.py            →  data/plots/post_preprocess/*.png
  04c_interactive_postprocess_qc.py   →  data/plots/post_preprocess_interactive/*.html

PHASE 3 — CLIMATE SIGNATURE (Tier 1 sun-event + Tier 2 true-daily-integral)
  04b_climate_signature.py        →  data/processed/signatures/climate_signature_uttarakhand.csv
  04d_signature_interactive.py    →  data/processed/signatures/interactive/*.html

PHASE 4 — CLIMATE REGIME CLUSTERING
  05_cluster_uttarakhand.py       →  data/processed/clustering/cluster_assignments_uttarakhand.csv
                                      data/processed/clustering/cluster_profiles_uttarakhand.csv
  05b_cluster_interactive.py      →  data/processed/clustering/interactive/*.html
  05_cluster_regions.py           →  (multi-state, NOT for now — see its section below)

PHASE 4 — EXTRA EXPLORATION (optional, either order relative to 05/05b)
  05c_explore_interactive.py      →  Streamlit app (raw vs. processed vs. comparison)
  05d_plots_comprehensive.py      →  data/plots/comprehensive/*.png + *.html (batch maps/stats)

PHASE 5 — PCM DATABASE + FEASIBILITY FILTERING
  PCM_data/01_preprocess.py       →  PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv
  06_build_pcm_database.py        →  data/processed/pcm/pcm_candidates.csv
  07b_charging_feasibility.py     →  (optional) regime-dependent Tm cap, run before 07
  07_feasibility_filter.py        →  data/processed/pcm/feasibility_survivors_by_cluster.csv

PHASE 6 — MULTI-CRITERIA RANKING
  08_mcdm_ranking.py              →  data/processed/pcm/mcdm_topk_by_cluster.csv

PHASE 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py        →  data/processed/pcm/physics_validation_results.csv
                                     data/processed/pcm/physics_validation_spearman.csv

PHASE 8 — FINAL OUTPUT
  09_recommendation_cards.py      →  data/processed/pcm/recommendation_cards.md
```

## Run Order

To run the whole core chain (Phase 2 onward) in one command instead of
typing each line below by hand, use `run_all_uttarakhand.py`:

```bash
python run_all_uttarakhand.py                 # core pipeline only
python run_all_uttarakhand.py --with-optional # core + diagnostic/plot scripts
python run_all_uttarakhand.py --dry-run       # print the resolved order, run nothing
python run_all_uttarakhand.py --include-setup # ALSO run the raw-data download scripts first
python run_all_uttarakhand.py --from 05_cluster_uttarakhand.py   # resume from a given stage
```

It does not run `02_combine_uttarakhand.py`, `05c_explore_interactive.py`
(a Streamlit app — launch that one yourself), or `05_cluster_regions.py`
(the multi-state version, on standby for later) — see its own docstring
for the full explanation. Otherwise it's exactly the same steps as below,
in the same order, via `subprocess`.

```bash
# ── Phase 0/1 — sampling design + raw download ──────────────────────────
python 00a_build_population_grid.py     # GADM boundary + WorldPop raster -> population_grid_points.csv
python 00b_build_suntimes.py            # sunrise/noon/sunset UTC times (pvlib) -> suntimes.csv
python 00c_attach_elevation.py          # real per-point elevation (ERA5 geopotential) -> population_grid_points.csv
python 01_download_era5_uttarakhand.py  # ERA5, sized to the population points + sun-event hours
python 01b_download_nasapower.py        # NASA POWER cross-check data, per point/year
python 00_unzip_accum.py                # fixes any CDS zip-disguised-as-.nc files

# ── Phase 2 — combine + repair the daily-integral gap ───────────────────
python 02_combine_uttarakhand.py        # merges ERA5 + POWER -> climate_uttarakhand_points.csv
python 02b_build_daily_aggregates.py    # re-reads the FULL NASA POWER hourly cache (already on
                                         # disk from 01b) to build true daily GHI/DTR/HDD/CDD
                                         # integrals -> daily_aggregates & tier2_signature CSVs

# ── Phase 2 QA — inspect the RAW merged data before cleaning it ─────────
python 03_plots_raw.py                  # static PNG checks (point map, event profile,
                                         # ERA5-vs-POWER agreement, missing data, seasonality, trend)
python 03b_interactive_raw_qa.py        # same 6 checks, as zoomable/hoverable HTML
python 03b_agreement_analysis.py        # decides if ERA5 alone is a defensible backbone or needs
                                         # bias correction vs NASA POWER -> bias_decision_uttarakhand.txt
streamlit run 03e_interactive_raw_plotly.py   # optional: live Plotly explorer on the raw points
streamlit run 03f_interactive_raw_folium.py   # optional: live Folium map on the raw points

# STOP AND LOOK at 03's output before continuing. In particular:
#   - check B: GHI/T_amb should peak at the "noon" event, not sunrise/sunset
#   - check C: large ERA5-vs-POWER MBE is expected and gets addressed in 04
#   - check F: no year-over-year step-change (would flag a download/unit bug)
#   - bias_decision_uttarakhand.txt: read this before trusting 04's output —
#     it says whether ERA5 needed a bias correction against NASA POWER

# ── Phase 2 — clean, QC, engineer features ───────────────────────────────
python 04_preprocess_uttarakhand.py     # 13-step QC pipeline: physical bounds, Hampel outliers,
                                         # hierarchical imputation + MICE, lag/rolling/delta
                                         # features, correlation/VIF, MinMax scaling (train-only fit)

# ── Phase 2 QA — inspect what cleaning actually did ──────────────────────
python 04c_postprocess_plots.py             # static PNG checks (post-clean missing %, distributions,
                                             # QC flag counts, lag sanity, one-point time series, corr)
python 04c_interactive_postprocess_qc.py    # same checks, as zoomable/hoverable HTML

# STOP AND LOOK: check A should show ~0% missing everywhere; check E's seasonal
# shape should look smooth, not flattened, before trusting this for Phase 3.

streamlit run 04e_interactive_preprocessed_plotly.py  # optional: same Plotly explorer, on cleaned data
streamlit run 04f_interactive_preprocessed_folium.py  # optional: same Folium map, on cleaned data

# ── Phase 3 — build the per-point climate signature ──────────────────────
python 04b_climate_signature.py         # merges Tier-1 (sun-event) + Tier-2 (true daily-integral,
                                         # from 02b) indices, adds PCM-facing quantities
                                         # (Tm_target, L_required), interaction terms, PCA
python 04d_signature_interactive.py     # interactive multi-layer map + correlation + scatter matrix

# ── Phase 4 — cluster into climate regimes ────────────────────────────────
python 05_cluster_uttarakhand.py        # Gaussian Mixture over the standardized signature matrix;
                                         # reports BIC/silhouette/DB/CH across K=2..10, fits the
                                         # final model at K_FINAL, saves soft membership + profiles
python 05b_cluster_interactive.py       # interactive cluster map (hoverable membership probs),
                                         # profile comparison, population-per-cluster, K-selection curves

# STOP AND LOOK at bic_selection_uttarakhand.csv, choose K_FINAL where silhouette
# lands in the 0.15-0.40 band, edit K_FINAL at the top of 05, re-run once before
# treating cluster_profiles_uttarakhand.csv as final input to Phase 5.

python 11_level_b_seasonal_analysis.py  # per existing cluster, recomputes L_required per season
                                         # (Winter/Summer/Monsoon/Retreat) and re-ranks TOPSIS ->
                                         # flags whether the Top-3 PCM changes by season
                                         # (checks whether e.g. the Terai plains near Udham Singh
                                         # Nagar/Haridwar need a different PCM than the Himalayan
                                         # belt around Chamoli/Pithoragarh in winter vs monsoon)

# ── Phase 4 — optional extra exploration (either order, not required) ────
python 05c_explore_interactive.py       # Streamlit app: streamlit run 05c_explore_interactive.py
python 05d_plots_comprehensive.py       # batch maps/timeseries/stats plots, static + interactive

# ── Phase 5 — PCM database + feasibility filtering ────────────────────────
python PCM_data/01_preprocess.py        # (only if you haven't already) cleans the raw PCM
                                         # manufacturer/literature data -> PCM_Properties_cleaned_mice_pmm_detailed.csv
python 06_build_pcm_database.py         # edit INPUT_CSV at the top if PCM_data isn't a sibling
                                         # folder of this pipeline -> pcm_candidates.csv
python 07b_charging_feasibility.py      # optional: regime-dependent Tm ceiling, run BEFORE 07
                                         # if you want it factored into the melting-window filter
python 07_feasibility_filter.py         # hard filters per cluster's Tm_target/L_required ->
                                         # feasibility_survivors_by_cluster.csv

# ── Phase 6 — multi-criteria ranking ───────────────────────────────────────
python 08_mcdm_ranking.py               # TOPSIS + GRA, entropy/AHP weights, Gaussian Tm fitness,
                                         # Borda consensus -> mcdm_topk_by_cluster.csv (headline table)

# ── Phase 7 — physics-based validation ───────────────────────────────────────
python 10_physics_validation.py         # grey-box lumped-enthalpy tank model, driven by each
                                         # cluster's medoid point's REAL daily climate data ->
                                         # physics_validation_results.csv + Spearman rho vs the
                                         # MCDM consensus rank. Runs BEFORE 09 (numbering is not
                                         # run order) since 09 includes 10's solar-fraction output
                                         # when present.

# ── Phase 8 — final output ─────────────────────────────────────────────────
python 09_recommendation_cards.py       # aggregates Phases 4/6/7 into recommendation_cards.md —
                                         # this is your results section
python 12_mcdm_interactive_plots.py     # optional: builds mcdm_final_results_complete.csv +
                                         # interactive Plotly/Folium presentation maps, run after 08
```

Each Phase 0/1 script is resumable — safe to Ctrl-C and re-run; already-
completed work is skipped automatically. The Phase 2+ scripts (`02b`
onward, including all of Phase 5/6/8) always overwrite their outputs fresh
on each run rather than resuming, since they're fast relative to the
downloads and correctness matters more than incremental speed there.

**Hard gates worth knowing about before you run past them:**
- `04b_climate_signature.py` refuses to run until `02b_build_daily_aggregates.py`
  has produced `tier2_signature_uttarakhand.csv` — it needs the true
  daily-integral indices, not just the 3-events/day proxies.
- `04_preprocess_uttarakhand.py`'s step 13 is a hard validation gate (zero
  NaN/Inf in the physical file, zero duplicate rows, all required columns
  present, train-portion scaling in [0,1]) — if it reports FAIL, fix that
  before moving on to `04b`; don't let a failed gate silently propagate
  into the climate signature.
- `07_feasibility_filter.py` reads `cluster_profiles_uttarakhand.csv`
  directly — if you re-run `05_cluster_uttarakhand.py` with a different
  `K_FINAL` after already running Phase 5/6, re-run `06`→`09` again too,
  or your PCM rankings will be filtered against a stale set of clusters.
- `09_recommendation_cards.py` reads four files at once (`05`'s profile +
  assignment CSVs, `08`'s top-k CSV, `07`'s survivors CSV) and exits
  early with a clear "run the earlier phase scripts first" message if any
  are missing — no partial output gets written.

## What each script does

### `config.py`
Shared, path-anchored configuration used by every script (works regardless
of the current working directory). Defines every input/output path,
`ensure_data_dirs()` to create them, and `get_cdsapi_client()` /
`load_cds_credentials()` for the CDS (Copernicus) API. Not run directly.

### `00a_build_population_grid.py`
Picks the sampling locations. Downloads the Uttarakhand boundary (GADM v4.1,
admin level 1) and the WorldPop India population raster (2020,
UN-adjusted, 100m — ~1.5-2GB, one-time download), clips the raster to
Uttarakhand, aggregates population onto a 0.25° grid **aligned to ERA5's own
grid origin** (so each point maps to a distinct ERA5 cell downstream), ranks
cells by population, and keeps the minimal set covering ~87.5% of the
state's total population.

- Output: `data/processed/population_grid_points.csv` —
  `point_id, lat, lon, population, weight`
- Uses a single static 2020 population snapshot for the whole 2016-2025
  study period (WorldPop doesn't publish a distinct India raster per year at
  this resolution) — a standard simplifying assumption, not a bug.
- Large raw downloads cached in `data/raw/population/` and
  `data/raw/boundary/`.

### `00b_build_suntimes.py`
For every point and every date 2016-01-01..2025-12-31, computes the exact
UTC sunrise, solar noon, and sunset via `pvlib`'s SPA algorithm (no manual
equation-of-time code).

- Output: `data/processed/suntimes.csv` —
  `point_id, date, event (sunrise|noon|sunset), time_utc`
- Note: sun events near the Uttarakhand/UTC boundary can genuinely fall on the
  *previous* UTC calendar date (e.g. an eastern point's summer sunrise can
  land at 23:55 UTC the day before) — `time_utc` is always the true instant;
  `date` is pvlib's nominal calendar-date assignment for that event.

### `00c_attach_elevation.py`
Attaches real per-point elevation, replacing the flat `DEFAULT_ALT_M`
`02_combine_uttarakhand.py` otherwise falls back to for solar-geometry
(air mass, clear-sky irradiance) calculations. Downloads ERA5's
time-invariant surface geopotential field (one CDS request — orography
doesn't change over time, so no per-year download) over the same bounding
envelope `01_download_era5_uttarakhand.py` uses, and converts to elevation
via `z / 9.80665` (WMO standard gravity).

- Output: `data/raw/era5/invariant/era5_UK_geopotential.nc` (raw cache);
  `population_grid_points.csv` gains an `elevation_m` column, updated in place.
- Must run before `02_combine_uttarakhand.py` for the real elevation to
  take effect — otherwise it silently falls back to the flat default.
- **Real limitation, not resolved by this script**: ERA5's native grid is
  ~0.25° (~28km), so its orography is a grid-cell *mean* elevation. In
  Uttarakhand's high-relief terrain (200m-7000m+) a single cell value
  smooths out real local relief — still far closer to the truth than one
  flat number for the whole state, but not a substitute for a real DEM.
- Requires `.cdsapirc` (same credentials as `01_download_era5_uttarakhand.py`).

### `01_download_era5_uttarakhand.py`
Downloads ERA5 hourly reanalysis over the bounding envelope of the
population points (not the whole state), for three narrow UTC hour windows
computed from `suntimes.csv` — one around sunrise, one around solar noon,
one around sunset — each padded ~1hr and correctly handling the
cross-midnight wraparound case above. Keeps the original pipeline's
instant/accum variable split; the extra predecessor-hour fetch
(`ACCUM_HOURS = INSTANT_HOURS ∪ {h-1}`) is now vestigial rather than
required — `deaccumulate()` in `02_combine_uttarakhand.py` no longer diffs
against it (see that function's docstring for why: the CDS/cfgrib pipeline
delivers `ssrd`/`strd`/`tp` already as per-step values, and the old
diff-based logic was silently deflating GHI ~10x — fixed 2026-09).

- Output: `data/raw/era5/points/era5_UK_points_{year}_{month}_{instant,accum}.nc`
- Status tracking: `data/raw/era5/download_status_points.csv`
- **Does not touch** the old `data/raw/era5/grid/` archive or
  `download_status.csv` from the previous uniform-grid/fixed-hour pipeline —
  entirely separate paths.
- Requires `.cdsapirc` (CDS/Copernicus API credentials) in this folder.

### `01b_download_nasapower.py`
For every point and every year 2016-2025, downloads NASA POWER hourly point
data (`ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS10M`) — an
independent cross-check source. No API key needed.

- Output: `data/raw/nasapower/power_{point_id}_{year}.json` (raw cache)
- Status tracking: `data/raw/nasapower/download_status_power.csv`
- This full hourly cache is read again, in full, by `02b_build_daily_aggregates.py`
  — only 3 of its ~8760 hours/year get used directly in `02`'s sun-event merge,
  but the rest isn't wasted.

### `00_unzip_accum.py`
The CDS API sometimes returns accum files as a ZIP even when an unarchived
NetCDF was requested. This detects and fixes those in place. Scans **both**
`data/raw/era5/grid/` (old pipeline) and `data/raw/era5/points/` (new
pipeline). Safe to re-run — valid NetCDF files are left alone.

### `02_combine_uttarakhand.py`
The merge step. For each point: nearest-neighbor-snaps to the ERA5 grid,
concatenates its full hourly series across all years, deaccumulates,
computes solar geometry (`pvlib`). For each `(point_id, date, event)` row in
`suntimes.csv`, picks the nearest-in-time ERA5 reading and the nearest-in-time
NASA POWER reading (both rejected if farther than 3 hours from the true
event time), and merges them into one row.

- Output: `data/processed/climate_uttarakhand_points.csv` — one row per
  point/date/event, with `era5_*` and `power_*` columns side by side for
  cross-checking, plus point metadata (`lat`, `lon`, `population`, `weight`)
  and calendar features (`month`, `DOY`, `year`, `season`, `season_code`).

### `02b_build_daily_aggregates.py`
`climate_uttarakhand_points.csv` only has 3 rows/day (sunrise, noon,
sunset) — some indices genuinely can't be computed from three instantaneous
samples: the true daily GHI energy integral, true diurnal temperature range
(Tmax-Tmin, not noon-minus-sunrise), heating/cooling degree-days from a
true daily mean, cloudy-day fraction, and the longest consecutive-cloudy-day
run. This script re-reads the FULL NASA POWER hourly cache already on disk
from `01b` (no new download, no CDS/NASA queue time) and builds those
integrals for every point/day that has ≥20 of its 24 hours present.

- Output: `data/processed/daily_aggregates_uttarakhand.csv` — one row per
  (point_id, date): true daily GHI/clearsky integrals, true Tmax/Tmin/DTR,
  true daily-mean T/RH/wind.
- Output: `data/processed/tier2_signature_uttarakhand.csv` — one row per
  point_id, the point-level Tier-2 indices `04b_climate_signature.py` merges
  onto the Tier-1 sun-event signature.
- **Limitation, stated plainly**: NASA POWER's downloaded parameters don't
  include precipitation, so `monsoon_index` (computed in `04b`) stays a
  3x/day ERA5 proxy, not a true Tier-2 index. If you want a real one, add
  `PRECTOTCORR` to `POWER_PARAMETERS` in `01b_download_nasapower.py` and
  re-run just that script.

### `03_plots_raw.py` / `03b_interactive_raw_qa.py`
Raw-data QA — run **before** any cleaning, directly on `02`'s merged output.
Six checks: (A) does the point map actually look population-weighted and
cover Uttarakhand; (B) does GHI/T_amb peak at the "noon" event (a timezone
sanity check); (C) how much do ERA5 and NASA POWER disagree, per variable
(MBE/RMSE/scatter — this is what `04`'s bias handling responds to); (D) a
missing-data heatmap per point x variable; (E) seasonal boxplots against
known Uttarakhand climatology; (F) a year-by-year trend check for
discontinuities that would suggest a download/unit bug in one specific year.
`03b` is the same six checks as zoomable/hoverable Plotly/Folium HTML
instead of static PNGs — nothing in either script writes back to the data.

- Output: `data/plots/raw/*.png` and `data/plots/raw_interactive/*.html`

### `03b_agreement_analysis.py`
Read-only cross-source validation, run before `04` touches the physical
values: decides whether ERA5 alone is a defensible backbone for
preprocessing, or needs bias correction against NASA POWER first. Never
writes back to `climate_uttarakhand_points.csv`.

- Output: `data/processed/era5_power_agreement_uttarakhand.csv`,
  `outputs/qc_era5_power_scatter_uttarakhand.html`,
  `outputs/bias_decision_uttarakhand.txt` (read this before trusting `04`)

### `03e_interactive_raw_plotly.py` / `03f_interactive_raw_folium.py`
Optional live Streamlit apps for exploring the raw combined-points data —
a Plotly variable explorer and a Folium point map, respectively. Launch
with `streamlit run <file>.py`, not plain `python`. `04e_interactive_
preprocessed_plotly.py` / `04f_interactive_preprocessed_folium.py` are the
same two apps pointed at the cleaned/preprocessed data instead (they load
`03e`/`03f` dynamically, so don't rename those two files).

### `04_preprocess_uttarakhand.py`
Phase 2 preprocessing and quality control — 13 steps: dataset inspection,
physical-bounds validation (out-of-range → NaN, not silently clipped),
Hampel/MAD outlier flagging (windowed over occurrences of the *same*
(point_id, event) series sorted by date — not hours, since there are only 3
rows/day here), a Yeo-Johnson skew diagnostic (report-only), hierarchical
imputation (interpolate → ffill/bfill → point/zone/global median → MICE),
temporal-coverage validation, feature engineering (wind decomposition,
cloud opacity, IST decimal hour, solar hour angle), lag features (1/7/30
*occurrences* = 1 day/1 week/1 month prior at the *same* sun event, not
hours), rolling stats, delta features, a Savitzky-Golay smoothing
diagnostic, Pearson/Spearman correlation, VIF, and a final 13-step
validation gate.

- Output: `data/preprocessed/uttarakhand_cleaned_physical.csv` — physical
  units, QC-passed, imputed, **not scaled** (Phase 3 indices are non-linear
  functions of physical values, so scaling first would corrupt them). This
  is what `04b_climate_signature.py` reads.
- Output: `data/preprocessed/uttarakhand_cleaned_scaled.csv` — same rows,
  MinMax-scaled feature columns (scaler fit on the first 70% of
  chronologically-sorted rows only — no leakage), for any later ML/DRL use.
- Also: `scalers.pkl`, `qc_report.txt`, `correlation_pearson.csv` /
  `_spearman.csv` / heatmaps, `vif_report.csv`, `yeo_johnson_skew.csv`,
  `savitzky_golay_diagnostic.png`.

### `04c_postprocess_plots.py` / `04c_interactive_postprocess_qc.py`
Post-cleaning QA — run **after** `04`, on `uttarakhand_cleaned_physical.csv`,
so you can see exactly what the 13 preprocessing steps did before trusting
them for Phase 3. Checks: (A) missing-data heatmap (should be ~0 everywhere
— if not, step 4's imputation missed something and step 13's gate should
already have failed); (B) distribution sanity, watching for imputation
spikes; (C) how many values physical-bounds vs. Hampel filtering each
flagged (parsed from `qc_report.txt`); (D) lag-feature sanity (GHI vs.
GHI 7-days-prior should correlate positively but well below 1.0); (E) one
point's cleaned time series with 7d/30d rolling means overlaid, to confirm
cleaning didn't flatten the seasonal shape; (F) a post-cleaning correlation
heatmap including the engineered features.

- Output: `data/plots/post_preprocess/*.png` and
  `data/plots/post_preprocess_interactive/*.html`

### `04b_climate_signature.py`
Phase 3 — builds one climate-signature row per point_id. Computes Tier-1
sun-event-only indices (mean/p95/p05 temperature, DTR proxy, GHI proxy via
a half-sine daylength approximation, clear-sky index, cloudy-day run
length, HDD18/CDD24, heat-stress index, monsoon index, etc.), then merges
in `02b`'s Tier-2 true-daily-integral indices wherever a point has POWER
coverage — the canonical column takes the true value when available and
falls back to the sun-event proxy otherwise (both are kept side by side,
suffixed `_true`/`_proxy`, so you can report how closely they agree). Adds
PCM-facing derived quantities (`Tm_target_C`, `L_required_kJ_per_kg`),
5 interaction terms, PCA on the correlated temperature/pressure block, and
a standardized (`_z`-suffixed) copy of the clustering-ready columns.

- **Requires** `02b_build_daily_aggregates.py` to have already produced
  `tier2_signature_uttarakhand.csv` — raises `FileNotFoundError` with that
  instruction if it hasn't.
- Output: `data/processed/signatures/climate_signature_uttarakhand.csv`,
  plus `pca_loadings.csv`, `signature_correlation_heatmap.png`,
  `signature_distributions.png`, `point_signature_map.png`.

### `04d_signature_interactive.py`
Interactive explorer for `04b`'s output: a Folium map with one toggleable
layer per signature index (GHI_daily_kWh, Ta_mean, DTR, kt_mean,
cloudy_frac, CCI, HDD18, CDD24, RH_mean, HSI, monsoon_index,
L_required_kJ_per_kg — edit the `MAP_LAYERS` list to add more), plus an
interactive correlation heatmap, index-distribution histograms, and a
scatter matrix of the key PCM-facing indices to eyeball the clustering
structure before `05` finds it formally.

- Output: `data/processed/signatures/interactive/*.html`

### `05_cluster_uttarakhand.py`
Phase 4 — climate regime clustering, Uttarakhand only (not combined with
any other state's Phase 3 output — nothing in the objective requires
cross-state regimes, and this script's output format is compatible with
the multi-state `05_cluster_regions.py`, described below). Uses
Gaussian Mixture rather than K-Means because climate is a continuous
gradient — the boundary between, say, high-hill and valley/plains
Uttarakhand isn't a hard line, and a point near it genuinely has partial
membership in both regimes; soft membership probabilities are kept for
exactly that reason. Reports BIC + silhouette + Davies-Bouldin +
Calinski-Harabasz across K=2..10 (also a reported-only K-Means comparison,
to answer "why not K-Means" with a number), then fits the final model at
`K_FINAL` (edit this constant at the top of the script after reviewing the
BIC/silhouette table, then re-run) and produces population-weighted
per-cluster profiles.

- Output: `data/processed/clustering/bic_selection_uttarakhand.csv`,
  `kmeans_comparison_uttarakhand.csv`,
  `cluster_assignments_uttarakhand.csv` (soft membership probabilities),
  `cluster_profiles_uttarakhand.csv` (population-weighted profile per
  regime — feed this into Phase 5 PCM feasibility filtering),
  `cluster_map_uttarakhand.png`.

### `05b_cluster_interactive.py`
Interactive explorer for `05`'s output: a Folium cluster map where each
point's popup shows its full soft-membership probability vector (boundary
points — max membership below 1.5/K — are drawn with a faint ring so
they're visually distinct from confidently-assigned points), a Plotly
grouped-bar comparison of population-weighted cluster profiles, a
population-share pie chart per regime, and BIC/silhouette K-selection
curves if `bic_selection_uttarakhand.csv` exists.

- Output: `data/processed/clustering/interactive/*.html`

### `05_cluster_regions.py`
Multi-state version of Phase 4 — **not for now**. Needs
`climate_signature_{region}.csv` from at least one other state's own
pipeline folder (e.g. `../era5-rajasthan/data/processed/signatures/...`)
before it does anything useful; its `REGION_FILES` dict at the top already
points at Uttarakhand's own signature file plus a placeholder for
Rajasthan. Nothing in the Objective 1 definition requires cross-state
regimes — `05_cluster_uttarakhand.py` alone is sufficient to finish the
objective on Uttarakhand. Leave this one alone until you actually add a
second state's Phase 3 output; its output format matches
`05_cluster_uttarakhand.py`'s exactly, so nothing downstream needs to
change if/when you do.

### `11_level_b_seasonal_analysis.py`
"Level B" from the plan — for each EXISTING Level-A cluster (from `05`),
recomputes the climate-dependent MCDM inputs (`L_required`, same
`Tm_target`) separately per season (Winter/Summer/Monsoon/Retreat) and
re-ranks with a single-method TOPSIS using the same weights already
computed for the annual case, then reports whether the Top-3 PCM changes.
Not a full independent seasonal GMM re-clustering (that's a bigger
addition); this is the cheaper "nearly free" version the plan permits as
a starting point. `L_required_season` is computed with the exact same
formula as `04b_climate_signature.py`'s annual `L_required`
(`DRAW_RATE_KG_PER_S` continuous overnight draw over 7 hours, no
Tamil-Nadu-style `SHARE_PCM` split), just re-evaluated on each season's
own mean temperature — so annual and seasonal values are on the same
basis within this pipeline. Worth watching closely for Uttarakhand given
how much elevation its population points span (~200–2000m, Terai plains
up to the Himalayan belt) — a seasonal PCM flip is plausible in a way it
might not be for a flatter state.

- Output: `data/processed/pcm/level_b_seasonal_topk.csv`,
  `data/processed/pcm/level_b_seasonal_summary.md`

### `05c_explore_interactive.py`
A Streamlit app (not a plain script — run with `streamlit run
05c_explore_interactive.py`, not `python`) for interactively browsing raw
vs. processed data side by side, per point and per variable, plus a
direct-comparison view. Useful for spot-checking specific points/dates
that a static plot wouldn't surface — e.g. "did cleaning change this one
point's July 2019 noon GHI in a way I'd expect."

### `05d_plots_comprehensive.py`
A batch plotting pass across maps, time series, and statistical summaries
— the point/event-schema equivalent of a full "make every plot I might
want for the paper" run, both static PNG and interactive HTML. Useful as
a one-shot figure-generation pass once Phase 4 is settled, rather than
running `03`/`03b`/`04c`/`04d` individually again.

- Output: `data/plots/comprehensive/` (static) and interactive equivalents

### `PCM_data/01_preprocess.py`
State-agnostic — this cleans the raw PCM manufacturer/literature property
data (melting point, latent heat, thermal conductivity, density, etc.)
using MICE + random-forest + predictive mean matching imputation, with
every imputed value donor-logged and traceable (not blindly zeroed). This
is general PCM materials research, not tied to Uttarakhand or any other
state — if you already ran this for another state's pipeline, you can
reuse that output directly instead of re-running it here.

- Output: `PCM_data/data/PCM_Properties_cleaned_mice_pmm.csv` and
  `PCM_Properties_cleaned_mice_pmm_detailed.csv`
- Diagnostics: `01_missingness_before_after.png`,
  `02_cross_series_donor_audit.png`, `03_imputed_vs_reported_sanity.png`,
  `04_correlation_heatmap.png`, `05_imputation_provenance.csv`

### `06_build_pcm_database.py`
Phase 5 prep. Builds the candidate PCM database this pipeline screens
against — sourced from `PCM_data`'s MICE+RF+PMM-cleaned CSV, which now
contains **55 rows** (31 manufacturer + 24 literature) spanning 6 brands
(Rubitherm, Pluss, PCM Products Ltd., PureTemp, CrodaTherm, and literature
n-alkanes/fatty acids/composites), all fully imputed, covering the 42-70 °C
melting band.

- **Expects `PCM_data/` as a sibling folder** of this pipeline
  (`INPUT_CSV = PROCESSED_DIR.parent.parent / "PCM_data" / "data" / ...`)
  — either place it one level above this folder, or edit `INPUT_CSV` at
  the top of the script to point wherever you put it.
- Output: `data/processed/pcm/pcm_database_uttarakhand.csv`
- The 55-row set meets the 40-60 candidate target from the plan doc.

### `07b_charging_feasibility.py`
Optional — run **before** `07_feasibility_filter.py` if you want its
output factored in. Estimates, per cluster, a realistic upper bound on
achievable charging temperature from a flat-plate solar collector under
that cluster's actual clear-sky conditions (not a single state-wide
constant) — the ~70°C ceiling this uses is a generic collector-physics
assumption (consistent with the cited literature's 25-100°C FPC operating
band), not an Uttarakhand-specific number. Uttarakhand's higher-altitude
clusters, if anything, see *less* reliable clear-sky access than the
plains (cloud/fog persistence), which is exactly what each cluster's own
`kt_mean`/`kt_std` already capture, rather than needing a state-specific
constant.

### `07_feasibility_filter.py`
Phase 5 — for each cluster's `Tm_target`/`L_required` (from
`cluster_profiles_uttarakhand.csv`), applies hard filters against
`06`'s candidate database: melting window `[Tm_target-5, Tm_target+8]`,
absolute 42-70°C band, latent heat ≥ 0.7× `L_required`, corrosion veto if
that cluster's HSI is above its own 75th percentile, supercooling veto
>8K, safety exclusion. Reports survivor counts per cluster.

- Output: `data/processed/pcm/feasibility_survivors_by_cluster.csv`
- **Known limitation, stated in its own docstring**: the corrosion veto
  and a 5th-percentile-day charging-feasibility check from the plan
  doc's Table 12 aren't fully applied yet — the database/cluster profiles
  don't carry the data those two specific filters need. Documented, not
  silently skipped.

### `08_mcdm_ranking.py`
Phase 6 — the headline deliverable. For each cluster's feasibility
survivors: a **Gaussian Tm fitness transform**
(`f_Tm = exp(-(Tm-Tm_target)^2 / (2*sigma^2))`, sigma≈4K — this has to
come before anything else touches melting temperature, since a raw
distance metric gets this wrong) feeds into **TOPSIS** and **GRA**
(Grey Relational Analysis) run independently, with **entropy weights**
computed per cluster from that cluster's own filtered matrix (blended
0.5/0.5 with AHP priors if supplied, entropy-only otherwise). Ranks are
aggregated to a **Borda-count consensus**, with **Kendall's W** reported
per cluster as an explicit agreement/disagreement signal — a low W is
treated as a genuine, reportable finding (that regime's PCM choice is
ambiguous), not hidden.

- Output: `data/processed/pcm/mcdm_topk_by_cluster.csv`

### `10_physics_validation.py`
Phase 7 — physics-based validation, the step that makes the MCDM ranking
falsifiable rather than a tautology. A grey-box lumped-enthalpy PCM tank
model (3-phase: pre-melt sensible, isothermal melting, post-melt
sensible), solved with backward Euler, driven by each cluster's medoid
point's REAL daily climate data (from `02b`'s output, not synthetic
weather) for one representative year. Simulates every feasibility
survivor per cluster, computes annual solar fraction, checks it against
the published 54–84% benchmark band, and reports Spearman's rho between
the MCDM consensus rank and simulated performance per cluster. Numbered
10 but runs **before** `09` — `09` includes this script's solar-fraction
output when present.

- Output: `data/processed/pcm/physics_validation_results.csv`,
  `data/processed/pcm/physics_validation_spearman.csv`

### `09_recommendation_cards.py`
Phase 8 — pure aggregation, computes nothing new. Turns Phases 4-6's
output into one markdown recommendation card per cluster: point count and
population covered, approximate medoid point, population-weighted climate
signature table, `Tm_target`/`L_required`, survivor count, Top-3 PCM
candidates with per-method scores and the Kendall's W agreement note, and
a caveats section (thermal conductivity/density/specific heat not
reported for the literature-added candidates; cycling/corrosion vetoes
only partially applied — see `07`'s docstring).

- Output: `data/processed/pcm/recommendation_cards.md` — this is your
  results section; reformat the tables to your target format (e.g. IEEE
  style) when you paste it in, the content is what this script gives you.
- Reads four files at once and exits early with a clear message if any
  are missing, rather than writing partial output.

### `12_mcdm_interactive_plots.py`
Optional presentation layer, run after `08_mcdm_ranking.py`. Reads the
complete MCDM score table and cluster assignments, retains every column
from `mcdm_full_scores_by_cluster.csv`, and adds cluster representative
coordinates and population summary for two interactive maps (hover text
exposes the complete row, not a reduced set of score columns).

- Output: `data/processed/pcm/mcdm_final_results_complete.csv`,
  `data/plots/mcdm/mcdm_clusters_plotly.html`,
  `data/plots/mcdm/mcdm_clusters_folium.html`

### `run_all_uttarakhand.py`
Not a pipeline stage — runs every CORE stage above via `subprocess`, in
the correct dependency order, in one invocation (stops at the first
required-stage failure). See "Run Order" at the top of this file, or run
`python run_all_uttarakhand.py --dry-run` to print the resolved order
without running anything.

## Requirements

```
pip install geopandas rasterio requests pandas numpy xarray netCDF4 pvlib scipy cdsapi \
            scikit-learn statsmodels matplotlib seaborn plotly folium branca streamlit
```

`geopandas`/`rasterio` are only needed for `00a`. `plotly`/`folium`/
`branca` are needed for the `*_interactive.py` scripts (`03b`,
`04c_interactive`, `04d`, `05b`, `05d`) and `05c` additionally needs
`streamlit`. `scikit-learn` and `statsmodels` are needed from
`04_preprocess_uttarakhand.py` onward (imputation, PCA, VIF, clustering,
and — via `PCM_data/01_preprocess.py` — the PCM database's own MICE/
random-forest imputation).

## Notes / known limitations

- **ERA5 GHI/LW/precipitation deaccumulation bug — RESOLVED (2026-09)**:
  `02_combine_uttarakhand.py`'s `deaccumulate()` used to diff consecutive
  hours on the assumption that `ssrd`/`strd`/`tp` accumulate since the last
  00Z/12Z forecast reset. The CDS/cfgrib pipeline actually delivers these
  already as per-step (hourly) values, so the diff computed a "delta of
  hourly totals" and silently deflated GHI ~10x (noon GHI averaged ~60
  W/m^2 against an independently-computed clear-sky GHI of ~894 W/m^2 for
  the same rows). Fixed to use the raw per-step value directly; ERA5-vs-
  NASA-POWER agreement went from MBE=-602 W/m^2, r=-0.03 to MBE=+20 W/m^2,
  r=0.76. See `README_PREPROCESSING.md` for the full writeup. The old
  "first day of the dataset has no predecessor hour" caveat no longer
  applies — there's no diffing against a predecessor hour anymore.
- **Elevation — RESOLVED (2026-09)**: population points didn't carry
  elevation data, so `02_combine_uttarakhand.py` used a flat 1200m
  approximation for solar-geometry calculations — a real limitation for
  Uttarakhand specifically (populated zones range ~200-2500m).
  `00c_attach_elevation.py` now attaches real per-point elevation from
  ERA5's time-invariant geopotential field; `elevation_m` carries a
  balanced ~0.37 PCA loading on PC1 in `04b`, not an outsized artifact
  (see `README_PREPROCESSING.md` for more).
- **WorldPop download size**: ~1.5-2GB, one-time, cached in
  `data/raw/population/`. The download auto-retries (up to 5 attempts) and
  resumes from where it left off via HTTP Range requests if the connection
  drops mid-stream — no manual intervention needed on a flaky connection.
- **monsoon_index stays proxy-only**: NASA POWER's cached parameters don't
  include precipitation, so this one index never gets a Tier-2 "true"
  version even after `02b` runs — see `02b`'s section above if you want to
  fix that.
- **Rows/day is 3, not 24**: every lag/rolling/delta concept in
  `04_preprocess_uttarakhand.py` is defined over *occurrences* of the same
  (point_id, event) pair, sorted by date, not over hours — "lag7" means
  "the same sun event, 7 days earlier," not "7 hours earlier." This is
  called out at each relevant step in that script's own log output so it's
  traceable in a methodology write-up.
- **PCM database coverage**: 55 rows (31 manufacturer + 24 literature)
  across 6 brands, meeting the 40-60 candidate target (see `06`'s section
  above). Corrosion veto and 5th-percentile-day charging feasibility aren't
  fully wired into `07` yet either (see `07`'s section above).
- **Phase 7 (physics-based validation) is implemented** in `10_physics_validation.py`. It runs a single-PCM grey-box lumped-enthalpy-tank simulation per cluster, comparing simulated annual solar fraction against published benchmarks (54–84%). 92% of simulated runs land within this benchmark band.

## Further reading in this repo

- `PREPROCESSING_STEPS.md` — a shorter, mechanics-only reference for
  what `03`/`04`/`04b`/`05` each actually do internally.
- `README_PREPROCESSING.md` — the longer version of the same, with
  confirmed-run details and the elevation limitation discussed at length.
- `NEXT_STEPS.md` — sprint-style status tracker and day-by-day plan
  covering Phases 1-8, including what's explicitly out of scope for now.