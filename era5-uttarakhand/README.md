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

PHASE 8 — FINAL OUTPUT
  09_recommendation_cards.py      →  data/processed/pcm/recommendation_cards.md
```

(Phase 7 — physics-based validation via a grey-box lumped enthalpy tank
model — has no script in this repo yet; see "What's genuinely still open"
at the bottom.)

## Run Order

```bash
# ── Phase 0/1 — sampling design + raw download ──────────────────────────
python 00a_build_population_grid.py     # GADM boundary + WorldPop raster -> population_grid_points.csv
python 00b_build_suntimes.py            # sunrise/noon/sunset UTC times (pvlib) -> suntimes.csv
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

# STOP AND LOOK at 03's output before continuing. In particular:
#   - check B: GHI/T_amb should peak at the "noon" event, not sunrise/sunset
#   - check C: large ERA5-vs-POWER MBE is expected and gets addressed in 04
#   - check F: no year-over-year step-change (would flag a download/unit bug)

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

# ── Phase 8 — final output ─────────────────────────────────────────────────
python 09_recommendation_cards.py       # aggregates Phases 4/6 into recommendation_cards.md —
                                         # this is your results section
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

### `01_download_era5_uttarakhand.py`
Downloads ERA5 hourly reanalysis over the bounding envelope of the
population points (not the whole state), for three narrow UTC hour windows
computed from `suntimes.csv` — one around sunrise, one around solar noon,
one around sunset — each padded ~1hr and correctly handling the
cross-midnight wraparound case above. Keeps the original pipeline's
instant/accum variable split and deaccumulation-helper-hour logic
(generalized to the new dynamic hour set — see the script's docstring and
`deaccumulate()` in `02_combine_uttarakhand.py` for why that still works
correctly).

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
against — sources from `PCM_data`'s cleaned manufacturer rows (18, fully
populated via traceable imputation, not blindly zeroed) plus ~7 literature-
added rows, ~25 candidates total in the corrected 42-70°C melting band.

- **Expects `PCM_data/` as a sibling folder** of this pipeline
  (`INPUT_CSV = PROCESSED_DIR.parent.parent / "PCM_data" / "data" / ...`)
  — either place it one level above this folder, or edit `INPUT_CSV` at
  the top of the script to point wherever you put it.
- Output: `data/processed/pcm/pcm_candidates.csv`
- **Known coverage gap, stated in its own docstring**: ~25 rows is short
  of the 40-60 target; specific missing rows (RT58/RT60/RT62HC, PLUSS
  OM55/OM65, a properly-sourced salt hydrate) are listed there. The
  pipeline is correct either way — this is a coverage question, not a
  correctness one.

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

- **First day of the dataset**: 2016-01-01 has no prior day to supply an
  accumulation-deaccumulation predecessor hour if a sun event's window
  touches hour 0 UTC — the affected `era5_GHI`/related columns for that one
  day come out as a natural `NaN` rather than a wrong value. Every other
  month boundary is bridged automatically (see `01_download_era5_uttarakhand.py`'s
  docstring for why).
- **Elevation**: population points don't carry elevation data, so
  `02_combine_uttarakhand.py` uses a flat 1200m approximation for solar-geometry
  calculations. This is a real limitation for Uttarakhand specifically —
  populated zones range roughly 200-2000m — worth checking whether
  `elev_proxy` carries real weight in `04b`'s PCA/correlation output
  before treating a first `05` clustering run as final (see
  `README_PREPROCESSING.md` for more).
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
- **PCM database coverage**: ~25 candidates against a 40-60 target (see
  `06`'s section above) — a coverage gap, not a correctness issue.
  Corrosion veto and 5th-percentile-day charging feasibility aren't fully
  wired into `07` yet either (see `07`'s section above).
- **Phase 7 (physics-based validation) has no script here.** A minimal
  single-PCM grey-box lumped-enthalpy-tank simulation per cluster,
  compared against published annual-solar-fraction benchmarks (54-84%),
  is enough to defensibly write "consistent with published benchmarks" —
  but it isn't required for Objective 1 to stand as a working framework,
  and is explicitly an accepted "future work" outcome if you don't get to
  it. See `NEXT_STEPS.md` for more on this.

## Further reading in this repo

- `PREPROCESSING_STEPS.md` — a shorter, mechanics-only reference for
  what `03`/`04`/`04b`/`05` each actually do internally.
- `README_PREPROCESSING.md` — the longer version of the same, with
  confirmed-run details and the elevation limitation discussed at length.
- `NEXT_STEPS.md` — sprint-style status tracker and day-by-day plan
  covering Phases 1-8, including what's explicitly out of scope for now.