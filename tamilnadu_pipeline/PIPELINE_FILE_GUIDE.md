# Pipeline File Guide — Objective 1, Tamil Nadu

Every script in this project, in the order they actually run, grouped by
phase. For the specific bugs/upgrades made this round, see `CHANGELOG.md`
instead — this file is "what does X do," not "what changed in X."

---

## Phase 0 — Shared configuration

### `config.py`
Not a pipeline step — every other script imports paths and settings from
here (`PROCESSED_DIR`, `PREPROCESSED_DIR`, `RAW_POWER_DIR`,
`POPULATION_GRID_FILE`, `SUNTIMES_FILE`, `COMBINED_POINTS_FILE`, the CDS
API credential loader). If you ever move the project folder, this is the
one file where paths need checking.

---

## Phase 1 — Data Collection

### `00a_build_population_grid.py`
Downloads the GADM Tamil Nadu boundary + WorldPop 2020 population raster,
aggregates population onto a 0.25° grid aligned to ERA5's own grid
origin, keeps the minimal set of highest-population cells covering
~87.5% of the state's population. **Output:** `population_grid_points.csv`
— 133 points, each with `point_id`, `lat`, `lon`, `population`, `weight`.

### `00b_build_suntimes.py`
For every point and every date 2016–2025, computes exact UTC sunrise,
solar noon, and sunset via pvlib's SPA algorithm. **Output:**
`suntimes.csv` — this drives which UTC hours get downloaded (`01`) and
which hourly readings get matched to each "event" (`02`), and is also
used directly by `10_physics_validation.py` for real daylength.

### `01_download_era5_tamilnadu.py`
Downloads ERA5 reanalysis for 3 narrow UTC hour windows per day (around
sunrise/noon/sunset, computed from `suntimes.csv`), for the whole
2016–2025 span, split into "instant" (temperature, wind, humidity,
pressure) and "accum" (solar radiation, precipitation — these are
cumulative fields that need deaccumulation later) API calls. Resumable —
tracks progress in a status CSV, safe to re-run.

### `01b_download_nasapower.py`
Downloads the FULL hourly NASA POWER series (not just sun-events) for
every point/year — `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`,
`RH2M`, `WS10M`. This full hourly cache is what `02b` reads later to
build the Tier-2 daily-integral indices — that's why it's downloaded in
full even though `02_combine` only uses 3 hours of it per day.

### `00_unzip_accum.py`
CDS occasionally returns a ZIP disguised as `.nc`. This detects and fixes
that in place. Run once before `02_combine`, safe to re-run (skips
already-valid files).

### `02_combine_tamilnadu.py`
The merge step. For each point: snaps to the nearest ERA5 grid cell,
concatenates+deaccumulates the ERA5 series, computes solar geometry
(SZA, clear-sky GHI, CSI) via pvlib, then for each `(point_id, date,
event)` row in `suntimes.csv` picks the nearest-in-time ERA5 reading AND
the nearest-in-time NASA POWER reading (rejecting either if >3h off).
**Output:** `climate_tamilnadu_points.csv` — ~1.46M rows (133 points ×
3653 days × 3 events), both `era5_*` and `power_*` columns side by side
for cross-checking. **This is the single input every later script
ultimately traces back to.**

---

## Phase 2 — Preprocessing & QC

### `02b_build_daily_aggregates.py`
Reads the FULL NASA POWER hourly cache from `01b` (not the 3-event
subset `02` used) and integrates it properly per calendar day: true daily
GHI energy integral, true Tmax/Tmin/DTR, true daily-mean T/RH/wind. This
recovers the "Tier 2" indices that literally cannot be computed from 3
instantaneous samples/day. **Output:** `daily_aggregates_tamilnadu.csv`
(one row per point-day — also what `10_physics_validation.py` uses to
drive the tank simulation with real weather) and
`tier2_signature_tamilnadu.csv` (one row per point, aggregated).

### `03_plots_raw.py` / `03b_interactive_raw_qa.py`
Read-only QA on `02`'s output, before any cleaning. Six checks: point map,
event-profile sanity (does noon actually peak GHI — catches timezone
bugs), ERA5-vs-POWER agreement (MBE/RMSE/correlation), missing-data
heatmap, seasonal boxplots, multi-year trend (catches a step-change in
one bad year). `03b` is the same six checks as interactive Plotly/Folium
HTML instead of static PNGs.

### `04_preprocess_tamilnadu.py`
The 13-step Phase 2 QC pipeline: dataset inspection → physical-range
validation (out-of-range → NaN, plus night-masking solar fields to 0) →
Hampel/MAD outlier flagging → hierarchical imputation (interpolate →
ffill/bfill → point/zone/global median → MICE for whatever's left) →
temporal validation → feature engineering (wind vectors, cloud opacity,
T-depression, solar hour angle) → lag features (1/7/30-day) → rolling
stats → delta features → lag-warmup row drop → correlation/VIF diagnostics
→ MinMax scaling (train-only fit, leakage-safe) → a hard PASS/FAIL gate
written to `qc_report.txt`. **Output:** `tamilnadu_cleaned_physical.csv`
(what every downstream script reads) and a separate
`tamilnadu_cleaned_scaled.csv` (for later ML/DRL use only — Phase 3 never
reads this one, since scaling would corrupt the signature indices).

### `04c_postprocess_plots.py` / `04c_interactive_postprocess_qc.py`
QA on `04`'s output — did cleaning actually work, not just "did it run."
Missing-data heatmap post-clean (should be ~0), distribution histograms
(watch for an imputation spike), a parse of `qc_report.txt` into a bar
chart of what each check flagged, GHI-vs-7-day-lag scatter (lag features
should carry real structure), one point's cleaned annual time series with
rolling mean overlaid (seasonal shape shouldn't look flattened),
correlation heatmap including the engineered features.

---

## Phase 3 — Climate Signature Construction

### `04b_climate_signature.py`
Collapses each point's entire 10-year, 3×-daily record into one ~18-index
vector — the object Phase 4 actually clusters. Merges Tier 1 (sun-event
indices, computed here from `04`'s output) with Tier 2 (`02b`'s true
daily integrals) — true value preferred, sun-event proxy as fallback,
both kept in the output so you can report agreement between them.
Computes `Tm_target` (constant 57°C, the corrected delivery-temperature-
anchored rule) and `L_required` (from a fixed assumed PCM mass/draw
schedule), 5 interaction terms, PCA on the correlated temperature/pressure
block. Explicitly excludes lat/lon from the clustering matrix (clustering
on geography would trivially just recover the map). **Output:**
`climate_signature_tamilnadu.csv` — one row per point, this is what `05`
clusters.

### `04d_signature_interactive.py`
Folium map with a layer toggle to flip between any signature index
(GHI, HDD18, L_required, etc.) per point, plus an interactive correlation
heatmap, distribution plots, and a scatter matrix of the specific indices
that feed the MCDM criteria later.

---

## Phase 4 — Climate Regime Clustering

### `05_cluster_tamilnadu.py`
Gaussian Mixture clustering of the 133 points' signature vectors (soft
membership, not K-Means — climate is a gradient, not hard-boundaried).
Fits K=2..10, reports BIC/silhouette/Davies-Bouldin/Calinski-Harabasz per
K, K-Means run alongside purely as a reported comparison. `K_FINAL` is a
manual decision you make after reading `bic_selection_tamilnadu.csv` —
not something the script picks for you. **Output:**
`cluster_assignments_tamilnadu.csv` (soft membership probabilities per
point) and `cluster_profiles_tamilnadu.csv` (population-weighted mean
signature per cluster — this is what `07` reads for each cluster's
`Tm_target`/`L_required`).

### `05b_cluster_interactive.py`
Folium cluster map (hover shows the full membership-probability vector,
not just the hard label — boundary points get a visual flag), a cluster
profile comparison bar chart, population-share pie chart, K-selection
BIC/silhouette curves.

### `05c` / `05d` (your own additions from `files (5).zip`)
Streamlit-based live exploration app + a comprehensive batch plot
generator for the climate data — complementary to `03b`/`04c`'s static
QA dashboards. Not on the critical path for Objective 1's ranking
pipeline; use them whenever you want a closer interactive look at the
raw or processed climate data.

### `05_cluster_regions.py` (not currently used)
The ORIGINAL 4-state design (Tamil Nadu + Rajasthan + Assam +
Uttarakhand combined). Untouched, still correct, still there for when/if
you extend beyond Tamil Nadu — not part of the current TN-only run.

### `11_level_b_seasonal_analysis.py`
"Level B" from the plan — checks whether the #1 PCM changes by SEASON
within each existing Level-A cluster (not a full independent seasonal
GMM re-clustering — the "nearly free" version the plan explicitly
permits). Recomputes `L_required` per season (temperature varies
seasonally; `Tm_target` doesn't, per the plan's rule) and re-ranks with
TOPSIS using the same weights as the annual case. **Output:**
`level_b_seasonal_topk.csv` + `level_b_seasonal_summary.md`.

---

## Phase 5 — PCM Property Database & Feasibility Filtering

### `PCM_data/01_preprocess.py` (separate mini-pipeline, not in this repo's numbering)
Cleans your 18 raw manufacturer PCM rows (10 Rubitherm RT-line + 7 PLUSS
OM + 1 PLUSS HS36) using MICE + Random Forest + Predictive Mean Matching
— trains each property's fill-in model only on rows that report it,
predicts across product lines, and logs which real donor PCM(s) justify
every filled value (`05_imputation_provenance.csv`). **Output:**
`PCM_Properties_cleaned_mice_pmm_detailed.csv` — every property complete
and traceable, not silently zeroed.

### `06_build_pcm_database.py`
Reads that MICE-PMM-cleaned manufacturer data, renames to the schema the
rest of the pipeline expects, and appends 7 literature PCMs (fatty
acids/eutectics/paraffins from `Singh2025`'s Table 2, already in your
Sources/ folder) with unknown properties left honestly as NaN rather than
guessed. **Output:** `pcm_database_tamilnadu.csv` — ~25 candidates
(target is 40-60; the script's own docstring lists exactly which
Rubitherm/PLUSS grades would close the gap).

### `07_feasibility_filter.py`
Hard-filters the PCM database against EACH cluster's `Tm_target`/
`L_required` before any MCDM ranking — prevents a compensatory ranking
method from rewarding a PCM with an unreachable melting point just
because it has great latent heat. Implements all 8 Table-12 filters:
melting window, absolute 42-70°C band, latent-heat floor, cycling
(flagged not excluded if unreported), supercooling veto, corrosion veto,
safety exclusion, plus auto-relaxation of the melting window if a
cluster's survivor count falls below 5. **Output:**
`feasibility_survivors_by_cluster.csv`.

### `07b_charging_feasibility.py` (optional)
A heuristic, clearly-labeled-as-not-rigorous per-cluster upper cap on
`Tm_target`, based on each cluster's `kt_mean`/`kt_std` (day-to-day sun
reliability) — the mechanism the plan names as what makes `Tm_target`
regime-dependent in principle. Run this BEFORE `07` if you want it to
take effect (it adds a `Tm_target_C_regime_capped` column that `07`
automatically prefers if present).

---

## Phase 6 — Multi-Criteria Ranking

### `08_mcdm_ranking.py`
The core ranking engine. Converts melting temperature to a Gaussian
fitness score first (the step every PCM-MCDM paper gets wrong if
skipped), then ranks the feasibility survivors in each cluster with
**four independent methods** — TOPSIS, GRA, PROMETHEE II, VIKOR — using
entropy+AHP-blended weights (λ=0.5). Aggregates via Borda count, cross-
checked against Copeland pairwise-majority; flags when they disagree.
Runs a 5,000-draw Monte Carlo (Dirichlet-perturbed weights + Gaussian-
perturbed PCM properties) reporting each candidate's Top-3 inclusion
probability and Top-1 retention rate. **Includes the
`USE_CLIMATE_RELATIVE_LATENT_HEAT` fix** — ranks on
`latent_heat / L_required` (margin over what THAT cluster/season actually
needs) instead of raw latent heat, which is what actually lets climate
influence which PCM wins, not just whether it's eligible at all.
**Output:** `mcdm_topk_by_cluster.csv`, `mcdm_full_scores_by_cluster.csv`,
`monte_carlo_stability.csv`.

---

## Phase 7 — Physics-Based Validation

### `10_physics_validation.py`
A grey-box lumped-enthalpy PCM tank model (3-phase: pre-melt sensible,
isothermal melting, post-melt sensible), solved with backward Euler,
driven by each cluster's medoid point's REAL 10-year daily climate data
(from `02b`'s output) for one representative year — not synthetic
weather. Simulates every feasibility survivor per cluster, computes
annual solar fraction, checks it against the published 54-84% benchmark
band, and computes Spearman ρ between the MCDM consensus rank and
simulated performance per cluster — the step that makes the MCDM ranking
falsifiable rather than a tautology. All tank/collector parameters are
stated, cited assumptions (see the script's own docstring) — not
measurements. **Output:** `physics_validation_results.csv`,
`physics_validation_spearman.csv`.

---

## Phase 8 — Explanation & Final Output

### `09_recommendation_cards.py`
Pure aggregation — no new computation. Turns Phases 4-7's output into one
markdown card per cluster: climate signature summary, `Tm_target`/
`L_required`, feasibility survivor count, Top-3 with all 4 methods'
scores + Monte Carlo stability + Kendall's W, and (if `10` has been run)
the simulated solar fraction and Spearman ρ. **Output:**
`recommendation_cards.md` — this is your results section, paste-ready.

---

## Documentation files (not code)

- **`README_PREPROCESSING.md`** — narrative walkthrough of every
  preprocessing step (Phases 1-4), written before Phases 5-8 existed.
- **`CHANGELOG.md`** — exactly what was fixed/added in the most recent
  round (bugs, new methods, new scripts) — read this if you want to know
  what changed rather than what each file does.
- **`NEXT_STEPS.md`** — current status table + what's left, updated each
  round.
- **This file** — the reference for "what does script X do."

---

## Quick "what do I run and in what order" recap

```
00a → 00b → 01 → 01b → 00_unzip_accum → 02_combine        (Phase 1)
02b → 03(/03b) → 04 → 04c(/04c_interactive)                (Phase 2)
04b → 04d_interactive                                       (Phase 3)
05 → 05b_interactive → 11_level_b_seasonal_analysis         (Phase 4)
06 → 07b(optional) → 07                                     (Phase 5)
08                                                            (Phase 6)
10                                                            (Phase 7)
09                                                            (Phase 8)
```
