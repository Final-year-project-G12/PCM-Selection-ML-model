# 00 — Master Overview: ERA5 Uttarakhand Climate → PCM Selection Pipeline

## Project objective

Final-year B.Tech CSE project (Group 12, Amrita School of Engineering, Guide: Dr. T. Deepika):
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water
Heating."** Objective 1 (the scope of this audit) builds a **climate-region-aware PCM
recommendation framework**: turn 10 years of reanalysis climate data into population-weighted
climate regimes, derive PCM performance targets per regime, and rank candidate phase-change
materials against those targets.

Governing document referenced throughout the Uttarakhand source code: the "Objective 1 plan",
cited in-script as **v3.0** (with `05_cluster_regions.py` still citing v2.0). Every script
docstring in `era5-uttarakhand/` names its plan section — §4.3 (Repair 1), §5 (preprocessing),
§6 / §6.2 / §6.3 (signature), §7 (clustering), §8 + Table 12 (feasibility), §9 / §9.2 / §9.5 /
Table 13 (MCDM), §11 + Table 18 (recommendation cards), Table 16 (solar-fraction benchmark).

**Scope decision recorded in the source files** (`NEXT_STEPS.md`, `README_PREPROCESSING.md`,
`05_cluster_uttarakhand.py`): *finish Objective 1 on Uttarakhand alone.* Cross-state clustering is
documented as future work and is deliberately not run. `05_cluster_regions.py` is present but
inert.

## What the ERA5 Uttarakhand pipeline does

Per `README.md`, the pipeline "builds a solar/climate dataset for Uttarakhand, sampled at
**population-weighted locations** and **astronomically computed sun-event times** (sunrise, solar
noon, sunset) rather than a uniform grid on fixed clock hours."

1. Samples Uttarakhand at **45 population-weighted points** on ERA5's own 0.25° grid, keeping the
   minimal set of highest-population cells covering >= 87.5 % of the state's raster population
   (`00a_build_population_grid.py`, `COVERAGE_TARGET = 0.875`), then attaches each point's real
   per-point elevation (196 m–2510 m across the 45 points) from ERA5's time-invariant geopotential
   field (`00c_attach_elevation.py`, added to fix a flat 1200 m elevation assumption — see below).
2. Pulls **ERA5 reanalysis** and **NASA POWER** for the *same* points and the *same* sun-event
   instants, keeping both as `era5_*` / `power_*` columns so one can be cross-checked against the
   other.
3. Repairs the 3-rows-per-day sampling limitation by re-reading the **full NASA POWER hourly
   cache** already on disk to build true daily integrals (`02b_build_daily_aggregates.py`,
   "Phase 2 Repair 1").
4. Reduces 10 years × 3 sun-events/day per point into a **two-tier ~18-index climate signature**
   (Tier 1 sun-event proxies + Tier 2 true daily integrals), plus 5 interaction terms and a PCA of
   the correlated temperature/elevation block (`04b_climate_signature.py`).
5. Clusters the 45 points into climate regimes with a **Gaussian Mixture Model, diagonal
   covariance, K_FINAL = 5** (`05_cluster_uttarakhand.py`).
6. Screens a **55-row PCM property database** against each regime's `Tm_target` / `L_required`
   (`06_build_pcm_database.py`, `07_feasibility_filter.py`, with `07b_charging_feasibility.py`'s
   regime-dependent Tm cap now materially lowering the target for 2 of the 5 clusters), then ranks
   survivors with a **four-method MCDM stack (TOPSIS + GRA + PROMETHEE II + VIKOR)**,
   entropy/AHP-blended weights, a Gaussian Tm-fitness transform, and a **Borda consensus with
   Kendall's W** (`08_mcdm_ranking.py`).
7. Stress-tests the ranking with a **5,000-draw Monte Carlo** over weight/property perturbations
   (`09b_monte_carlo_stability.py`) and validates it against a **grey-box lumped-enthalpy PCM tank
   physics simulation** (`10_physics_validation.py`), then aggregates the result into one markdown
   recommendation card per regime (`09_recommendation_cards.py`).

## Complete pipeline map (as actually implemented in `era5-uttarakhand/`)

```
PHASE 0/1 — SAMPLING DESIGN + RAW DOWNLOAD
  00a_build_population_grid.py    -> data/processed/population_grid_points.csv   (45 pts, >=87.5% pop)
  00b_build_suntimes.py           -> data/processed/suntimes.csv                 (pvlib SPA, UTC)
  00c_attach_elevation.py         -> data/processed/population_grid_points.csv   (adds real elevation_m,
                                     196m-2510m, from ERA5 geopotential; fixes the old flat 1200m default)
  01_download_era5_uttarakhand.py -> data/raw/era5/points/era5_UK_points_{yyyy}_{mm}_{instant,accum}.nc
  01b_download_nasapower.py       -> data/raw/nasapower/power_{point_id}_{year}.json
  00_unzip_accum.py               -> (fixes CDS zip-disguised-as-.nc files in place)
        |
PHASE 2 — COMBINE + DAILY-INTEGRAL REPAIR
  02_combine_uttarakhand.py       -> data/processed/climate_uttarakhand_points.csv
                                     (uses each point's real elevation_m for solar geometry, falling
                                     back to DEFAULT_ALT_M=1200 only if elevation_m is missing;
                                     deaccumulate() fixed to return ERA5's per-step value directly —
                                     see "What remains" below)
  02b_build_daily_aggregates.py   -> data/processed/daily_aggregates_uttarakhand.csv
                                     data/processed/tier2_signature_uttarakhand.csv
        |
PHASE 2 QA — RAW CHECKS (read-only, before cleaning)
  03_plots_raw.py                 -> data/plots/raw/*.png  + C_era5_vs_power_stats.csv
  03b_interactive_raw_qa.py       -> data/plots/raw_interactive/*.html
  03b_agreement_analysis.py       -> outputs/bias_decision_uttarakhand.txt (per-season quantile-mapping
                                     bias decision; GHI noon MBE=+19.55 W/m^2, r=0.759, branch
                                     QUANTILE_MAP, post deaccumulation fix)
        |
PHASE 2 — PREPROCESSING & QUALITY CONTROL (13 steps)
  04_preprocess_uttarakhand.py    -> data/preprocessed/uttarakhand_cleaned_physical.csv
                                     data/preprocessed/uttarakhand_cleaned_scaled.csv
                                     scalers.pkl, qc_report.txt, correlation_*.csv,
                                     vif_report.csv, yeo_johnson_skew.csv, *.png
        |
PHASE 2 QA — POST-CLEANING CHECKS
  04c_postprocess_plots.py            -> data/plots/post_preprocess/*.png + C_qc_flag_counts.csv
  04c_interactive_postprocess_qc.py   -> data/plots/post_preprocess_interactive/*.html
        |
PHASE 3 — CLIMATE SIGNATURE (Tier 1 sun-event + Tier 2 true daily integral)
  04b_climate_signature.py        -> data/processed/signatures/climate_signature_uttarakhand.csv
                                     pca_loadings.csv + 3 diagnostic PNGs
  04d_signature_interactive.py    -> data/processed/signatures/interactive/*.html
        |
PHASE 4 — CLIMATE REGIME CLUSTERING (Uttarakhand only)
  05_cluster_uttarakhand.py       -> data/processed/clustering/bic_selection_uttarakhand.csv
                                     kmeans_comparison_uttarakhand.csv
                                     cluster_assignments_uttarakhand.csv (soft membership)
                                     cluster_profiles_uttarakhand.csv    (population-weighted)
                                     cluster_map_uttarakhand.png
  05b_cluster_interactive.py      -> data/processed/clustering/interactive/*.html
  05_cluster_regions.py           -> (multi-state; NOT run — stops if <2 regions present)
        |
PHASE 4 — OPTIONAL EXPLORATION
  05c_explore_interactive.py      -> Streamlit app (raw / processed / comparison)
  05d_plots_comprehensive.py      -> data/plots/comprehensive/{maps,timeseries,statistics,solar_resource}
        |
PHASE 5 — PCM DATABASE + FEASIBILITY FILTERING
  PCM_data/PCM_data/01_preprocess.py -> PCM_Properties_cleaned_mice_pmm{,_detailed}.csv (55 rows)
  06_build_pcm_database.py        -> data/processed/pcm/pcm_database_uttarakhand.csv
  07b_charging_feasibility.py     -> adds Tm_target_C_regime_capped to cluster profiles — now a real
                                     regime-dependent cap for Clusters 1 (55.16C) and 2 (56.51C) after
                                     the poor_day_kt normalization bug was fixed (was a near no-op before)
  07_feasibility_filter.py        -> data/processed/pcm/feasibility_survivors_by_cluster.csv
        |
PHASE 6 — MULTI-CRITERIA RANKING (TOPSIS + GRA + PROMETHEE II + VIKOR)
  08_mcdm_ranking.py              -> data/processed/pcm/mcdm_topk_by_cluster.csv
                                     data/processed/pcm/mcdm_full_scores_by_cluster.csv
  09b_monte_carlo_stability.py    -> 5,000-draw Monte Carlo rank-stability results (Top-3-inclusion /
                                     Top-1-retention probability per candidate per cluster)
        |
PHASE 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py        -> data/processed/pcm/physics_validation_results.csv
                                     data/processed/pcm/physics_validation_spearman.csv
        |
PHASE 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py      -> data/processed/pcm/recommendation_cards.md

FIGURE / VERIFICATION LAYER (not part of the numbered phase chain)
  generate_objective1_plots.py    -> data/plots/uttarakhand_objective1/*   (13-plot set)
  comparison_plots_uttarakhand.py -> data/plots/comparison/*               (never produced — path bug)
  verify_01_preprocessing.py      -> data/plots/verify_preprocessing/*
  verify_02_clustering.py         -> data/plots/verify_clustering/*
  verify_03_feasibility.py        -> data/plots/verify_feasibility/*
  verify_04_ranking.py            -> data/plots/verify_ranking/*
```

## Phase 1–8 status at a glance

| Phase | Script(s) | Status | Headline finding (Uttarakhand) |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `00c`, `01`, `01b`, `00_unzip_accum` | **RUN** (evidenced by downstream artefacts) | 45 points `UKP_0001–UKP_0045`, 10,475,711 population covered, 2016–2025; `00c_attach_elevation.py` attaches real per-point elevation (196m–2510m) |
| 2 — Combine + Tier-2 repair | `02`, `02b` | **RUN** | `climate_uttarakhand_points.csv` = **493,155 rows** = 45 × 3653 × 3 exactly (no rows lost to the 3 h match window); `deaccumulate()` fixed to return ERA5's per-step value directly (was deflating GHI ~10x) |
| 2 QA — Raw checks | `03`, `03b`, `03b_agreement_analysis` | **RUN** | Noon peaks GHI (timezone check passes); post-fix ERA5-vs-POWER GHI cross-source agreement: **MBE = +19.6 W/m², r = 0.759**, decision branch **QUANTILE_MAP** (before the deaccumulation fix this was MBE = −602 W/m², r = −0.03, branch MANUAL_REVIEW) |
| 2 — Preprocessing & QC | `04`, `04c` ×2 | **RUN** | 493,155 -> **489,105 rows** (99.2 % retention); 36 -> 89 columns; 0 residual NaN; `qc_report.txt` 5/5 checks PASS |
| 3 — Climate Signature | `04b`, `04d` | **RUN** | `Tm_target` fixed at **57 °C** for every point (50 + 7, indirect-system rule); PCA temperature/elevation block now uses real `elevation_m` (balanced ~0.37 loading on PC1, which explains 90.7% of variance) instead of the old pressure-derived `elev_proxy` (which had an outsized, unexplained −0.33/0.59 loading) |
| 4 — Regime Clustering | `05`, `05b` | **RUN** | **K_FINAL = 5**, GMM **diagonal** covariance; sizes **7 / 3 / 9 / 10 / 16** (Clusters 0–4 respectively); silhouette **0.28** (within the documented 0.15–0.40 expected band for a 45-point state) |
| 5 — Feasibility Filtering | `06`, `07` (`07b`) | **RUN** | 55-candidate database; melting window [52, 65] °C; **29/30/29/27/29 candidates survive in Clusters 0–4 respectively** — no longer identical, because `07b`'s regime cap (once its `poor_day_kt` normalization bug was fixed) genuinely lowers `Tm_target` for Clusters 1 and 2 |
| 6 — MCDM Ranking | `08` | **RUN** | Four-method stack — TOPSIS + GRA + PROMETHEE II + VIKOR + Borda consensus; Top-1 is **PureTemp 58** in Clusters 0/3/4 but **PureTemp 53** in Cluster 1 and **PureTemp 58** (with a VIKOR compromise set) in Cluster 2 — see the Phase 6 table below; Kendall's W ranges 0.708–0.842 per cluster |
| 6b — Monte Carlo Stability | `09b_monte_carlo_stability.py` | **RUN** | 5,000 draws/cluster; `n-Octacosane (C28)` has the highest Top-3-inclusion probability in every cluster (37.8%–39.3%); Top-1 retention 16.2%–18.3% — lower than other states because Uttarakhand's feasible pool (27–30 candidates/cluster) is much larger and more homogeneous |
| 7 — Physics Validation | `10_physics_validation.py` | **RUN** | Grey-box lumped-enthalpy tank model (backward-Euler); after fixing a spurious term in the tank-temperature solve and a one-directional `Qp` latent-heat accumulator, **0% of simulations land in the published 54–84% solar-fraction benchmark band** (actual ~15–19% across all 5 clusters) — the previously-reported 92%-in-band figure was an artifact of those two bugs and is not valid. Per-cluster Spearman rho (consensus rank vs. simulated solar fraction): Cluster 0 = −0.097, Cluster 1 = −0.168, Cluster 2 = −0.227, Cluster 3 = +0.171, Cluster 4 = −0.140 — none significant at p<0.05 |
| 8 — Recommendation Cards | `09` | **RUN, OUTPUT NOT COMMITTED** | `recommendation_cards.md` exists on disk (regenerated against the current, corrected Phase 5–7 results) but `data/processed/` remains git-ignored, so it is not present in this repository |

## Current architecture

- **Language/stack**: Python, pandas/numpy/scikit-learn/scipy/statsmodels, `pvlib` for solar
  geometry and sun times, `cdsapi` for ERA5, `xarray`/`netCDF4` for NetCDF,
  matplotlib/seaborn/plotly/folium/branca for figures, `streamlit` for `05c`.
- **Path convention**: every numbered script imports `config.py`, which anchors all paths to
  `era5-uttarakhand/` regardless of the working directory. The four `verify_*.py` scripts and
  `generate_objective1_plots.py` do **not** use `config.py` (see `12_FINAL_READINESS_REPORT.md`).
- **Resumability**: `00a`, `00b`, `01`, `01b` are all resumable/skip-if-done. Everything from
  `02b` onward overwrites its outputs fresh — stated explicitly in `README.md`.
- **Hard gates**: `04b` refuses to run without `tier2_signature_uttarakhand.csv`; `04` step 13 is
  a PASS/FAIL validation gate; `09` exits early if any of its four inputs is missing.
- **State-parameterisation**: `05_cluster_regions.py` is written state-agnostically and its
  `REGION_FILES` dict already points at Uttarakhand + a Rajasthan placeholder, but it returns
  early unless >= 2 region signature files exist.

## Uttarakhand-specific design choices recorded in the source

| Choice | Value in `era5-uttarakhand/` | Where stated |
|---|---|---|
| Sampling points | 45 | `00a` output; `NEXT_STEPS.md`; `README_PREPROCESSING.md` |
| Point-ID prefix | `UKP_####` | `00a_build_population_grid.py` line 259 |
| Per-point elevation | Real elevation from ERA5 geopotential, **196 m – 2510 m** across the 45 points (`00c_attach_elevation.py`); `DEFAULT_ALT_M = 1200` in `02_combine_uttarakhand.py` survives only as a fallback if `elevation_m` is missing for a point | `00c_attach_elevation.py`; `02_combine_uttarakhand.py` |
| Season map | Winter DJF / **Summer MAM** / **Monsoon JJA** / **Retreat SON** | `02_combine_uttarakhand.py` `SEASON_MAP` |
| Accumulated-field handling | `deaccumulate()` — now a pass-through returning ERA5's per-step (already-hourly) value directly; the old `diff()`-against-prior-hour logic assumed a since-reset accumulation convention ERA5 does not actually use here, and was deflating GHI ~10x | `02_combine_uttarakhand.py` |
| `Tm_target` | Constant **57 °C** (`T_DELIVERY_C = 50` + `DT_APPROACH_C = 7`) for every point; regime-capped downward for Clusters 1 (55.16 °C) and 2 (56.51 °C) by `07b_charging_feasibility.py` | `04b_climate_signature.py`; `07b_charging_feasibility.py` |
| GMM covariance | **`diag`** (diagonal) — deliberately not `full`, to avoid overfitting a high-dimensional covariance with only 45 samples | `05_cluster_uttarakhand.py` |
| K_FINAL | **5** | `05_cluster_uttarakhand.py` line 73 |
| Silhouette accept band | 0.15 – 0.40 (widened from the 4-state 0.15 – 0.35); the actual run scores **0.28** | `05_cluster_uttarakhand.py` |
| PCM database size | **55 rows** (31 manufacturer + 24 literature) | `06_build_pcm_database.py`; verified against the CSV |
| MCDM methods | **TOPSIS + GRA + PROMETHEE II + VIKOR** (four methods) | `08_mcdm_ranking.py` |
| Monte Carlo draws | **Implemented** — 5,000 draws/cluster (Dirichlet-perturbed weights + property jitter) | `09b_monte_carlo_stability.py` |
| Physics validation | **Implemented** — backward-Euler grey-box lumped-enthalpy PCM tank model | `10_physics_validation.py` |

## Main datasets produced

Only the plot tree and the PCM property CSVs are committed. `data/raw/`, `data/processed/` and
`data/preprocessed/` are all listed in `era5-uttarakhand/.gitignore`, so the CSVs below exist on
the author's machine but **are not present in this repository**. Row counts marked *(observed)*
were recovered from committed plot artefacts; those marked *(expected)* are arithmetic from the
scripts' own constants.

| File | Rows | Grain | Produced by | Basis |
|---|---|---|---|---|
| `population_grid_points.csv` | 45 | 1 row/point | `00a` | observed (45 markers, 45 popups) |
| `suntimes.csv` | 493,155 | 1 row/point/date/event | `00b` | expected (45 × 3653 × 3) |
| `climate_uttarakhand_points.csv` | **493,155** | 1 row/point/date/event | `02` | observed (`C_era5_vs_power_stats.csv` n; verify summary) |
| `daily_aggregates_uttarakhand.csv` | <= 164,385 | 1 row/point/day | `02b` | expected (45 × 3653, minus days with < 20 h POWER coverage) |
| `tier2_signature_uttarakhand.csv` | <= 45 | 1 row/point | `02b` | expected |
| `uttarakhand_cleaned_physical.csv` | **489,105** × 89 cols | 1 row/point/date/event | `04` | observed (verify summary) |
| `uttarakhand_cleaned_scaled.csv` | 489,105 | same rows, MinMax-scaled | `04` | expected |
| `climate_signature_uttarakhand.csv` | 45 | 1 row/point | `04b` | expected |
| `cluster_assignments_uttarakhand.csv` | 45 | 1 row/point | `05` | observed (folium popups) |
| `cluster_profiles_uttarakhand.csv` | 5 | 1 row/cluster | `05` | observed; sizes 7/3/9/10/16 for Clusters 0–4 |
| `pcm_database_uttarakhand.csv` | 55 | 1 row/PCM | `06` | observed (source CSV + plot counts) |
| `feasibility_survivors_by_cluster.csv` | **275** (55 × 5, all rows kept with `passes_all` flag) | 1 row/cluster × PCM | `07` | observed (verify summary); survivor counts per cluster are now 29/30/29/27/29, not identical |
| `mcdm_topk_by_cluster.csv` | **15** (5 clusters × Top-3) | 1 row/cluster × PCM | `08` | observed (verify summary) |
| `mcdm_full_scores_by_cluster.csv` | varies (5 clusters × 27–30 survivors each) | 1 row/cluster × survivor | `08` | expected |
| `recommendation_cards.md` | 5 cards | 1 card/cluster | `09` | exists on disk, regenerated against current results — output still not committed to the repo |

## Main algorithms

pvlib SPA sun-rise/transit/set · pvlib solar position + Ineichen clear-sky · Magnus-formula RH ·
ERA5 real per-point elevation from time-invariant geopotential (`00c`) · ERA5 accumulated-field
de-accumulation, now a direct pass-through of the per-step value (the old `diff()`-based
00Z/12Z-reset logic is retired — see "What remains" below) · nearest-neighbour ERA5 grid snapping ·
nearest-in-time (<= 3 h) cross-source matching · per-season quantile-mapping bias correction
(`03b_agreement_analysis.py`) · physical-bounds -> NaN validation · Hampel/MAD outlier flagging over
sun-event occurrences · hierarchical imputation (interpolate -> ffill/bfill -> point/zone/global
median -> MICE `IterativeImputer`) · Yeo-Johnson skew diagnostic · Savitzky-Golay smoothing
diagnostic · Pearson/Spearman correlation · VIF · MinMax scaling with a chronological 70 % train fit
· PCA (temperature/elevation block, 95 % variance) · z-standardisation · Gaussian Mixture
(**diagonal** covariance) with BIC / silhouette / Davies-Bouldin / Calinski-Harabasz model selection
· K-Means comparison · population-weighted cluster profiling · MICE + Random-Forest +
Predictive-Mean-Matching PCM property imputation · Gaussian Tm-fitness transform (sigma = 4 K) ·
Shannon-entropy criterion weighting blended 0.5/0.5 with an AHP-style prior · TOPSIS (shared [0,1]
normalization basis, no extra vector renormalization) · Grey Relational Analysis (zeta = 0.5) ·
PROMETHEE II · VIKOR (with both the C1 advantage and C2 stability compromise checks) · Borda count ·
Kendall's W · regime-dependent clear-sky-reliability charging cap (`07b`, now genuinely
regime-differentiating for 2 of 5 clusters) · 5,000-draw Monte Carlo rank-stability analysis (`09b`)
· backward-Euler grey-box lumped-enthalpy PCM tank physics simulation (`10`).

## Validation strategy actually present

Four layers now exist in `era5-uttarakhand/`:

1. **Cross-source** — `03_plots_raw.py` / `03b_interactive_raw_qa.py` compute ERA5-vs-NASA-POWER
   MBE / RMSE / Pearson *r* per variable and write `C_era5_vs_power_stats.csv`, and
   `03b_agreement_analysis.py` now makes and records an explicit bias-correction decision
   (`outputs/bias_decision_uttarakhand.txt`): GHI noon MBE = +19.6 W/m², r = 0.759, branch
   **QUANTILE_MAP**, with a per-season quantile-mapping table. This closes what used to be a real
   gap (disagreement measured but never acted on).
2. **Internal statistical** — `04` step 13 hard gate; `05`'s BIC / silhouette / Davies-Bouldin /
   Calinski-Harabasz table plus a K-Means silhouette comparison; `08`'s Kendall's W per cluster and
   VIKOR compromise checks (C1 advantage + C2 stability, both now actually evaluated).
3. **Monte Carlo rank stability** — `09b_monte_carlo_stability.py`, 5,000 draws/cluster over
   Dirichlet-perturbed weights and jittered PCM properties, reporting Top-3-inclusion and Top-1-
   retention probabilities per candidate per cluster.
4. **Physics-based simulation** — `10_physics_validation.py`, a backward-Euler grey-box
   lumped-enthalpy PCM tank model, cross-checking the MCDM consensus ranking against simulated
   annual solar fraction (Spearman rho per cluster).
5. **Post-hoc verification suite** — `verify_01`…`verify_04` re-open the saved outputs and
   regenerate independent diagnostics. See
   `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`, including two real defects in that suite.

**Still not present for Uttarakhand:** external climate classification (Köppen-Geiger / NBC-ECBC)
and bootstrap cluster stability (ARI-based).

## Research gaps and novelty mapping

### Important disambiguation

Two distinct systems exist in this project and must not be conflated:

- **N1–N6** are the framework doc's own novelty positioning for Objective 1.
- **RG1–RG5** are the research gaps for the **broader, multi-objective project** (climate-aware PCM
  recommendation, design optimisation, DRL control, integrated prototype, experimental validation).
  They do not appear in the Objective 1 plan document itself.

Neither list is reproduced verbatim inside `era5-uttarakhand/`; the mapping below is derived from
what the Uttarakhand pipeline demonstrably does.

### Phase -> novelty-claim mapping for Uttarakhand

| Phase | Novelty contribution | How Uttarakhand implements it | Verdict |
|---|---|---|---|
| 1 — Data Collection | Population-weighted sampling | 45 points, 87.5 % coverage target, 10,475,711 people, ERA5-lattice-aligned, sun-event-aligned | **Delivered** |
| 2 — Combine + Tier-2 | Two independent sources cross-checked | ERA5 + NASA POWER at identical points/instants; full agreement statistics computed | **Delivered, but the disagreement is never acted upon** |
| 3 — Climate Signature | Two-tier signature (sun-event + true daily integral) | 18 indices; Tier-2 canonical where available; PCA on the thermodynamic block only | **Delivered — and it insulated the clustering matrix from the pipeline's largest data defect** |
| 4 — Regime Clustering | Discovered regimes, not hand-picked zones | GMM **diagonal** covariance, K = 5 by manual selection from a BIC/silhouette table; lat/lon excluded | **Delivered** — clusters are spatially coherent without clustering on geography (silhouette 0.28). **But** no bootstrap stability and no external classification |
| 5 — Feasibility Filtering | Corrected 42–70 °C SWH-specific PCM band | Band enforced; melting window [52, 65] °C at `Tm_target = 57` (regime-capped to 55.16/56.51 for Clusters 1/2) | **Partially delivered** — the corrosion veto cannot activate (all 55 candidates organic, and the database doesn't yet carry corrosion-class data); other Table-12 filters still unimplemented. Regime-capping now genuinely differentiates survivor counts (29/30/29/27/29) |
| 6 — MCDM Ranking | Top-3 with explicit method-agreement reporting | TOPSIS + GRA + PROMETHEE II + VIKOR, entropy/AHP weights, Gaussian Tm fitness, Borda, Kendall's W | **Delivered** — four independent methods, Kendall's W 0.708–0.842 per cluster, VIKOR now correctly reports compromise sets (Clusters 1 and 2) instead of false single winners |
| 7 — Physics Validation | Physics-validated ranking | Backward-Euler grey-box lumped-enthalpy tank model (`10_physics_validation.py`), Spearman rho of consensus rank vs. simulated solar fraction per cluster | **Delivered, and an honest negative result** — 0% of simulations land in the 54–84% literature benchmark band (actual ~15–19%); per-cluster rho ranges −0.227 to +0.171, none significant. This is a genuine finding once two solver bugs were fixed, not evidence the model is broken |
| 8 — Recommendation Cards | Per-regime explainable output | 5 cards; population-weighted profiles; Top-3 with per-method scores and Kendall's W | **Delivered**; regenerated against the current, corrected results; output still not committed (git-ignored) |

### The central finding against the novelty claim — RESOLVED (2026-09)

The framework's core proposition is that **different climate regimes should receive different PCM
recommendations.** This section used to report that Uttarakhand did **not** demonstrate it: all
five regimes returned the same feasibility survivors and the same #1 PCM (RT60, in that run). That
was traced to a real bug, not a mathematical inevitability of a constant `Tm_target`: the
regime-dependent Tm cap in `07b_charging_feasibility.py` divided its own signal (`poor_day_kt`) by
`kt_mean`, collapsing to a coefficient-of-variation measure that could never differentiate clusters
regardless of how sunny or cloudy they actually were — this is why the script always printed "0/5
clusters where the regime cap actually lowers Tm_target," not because the cap was disabled or
optional.

**Fixed.** Using `poor_day_kt` directly, Clusters 1 (Tm_target=55.16C) and 2 (Tm_target=56.51C) now
get a genuinely lower, climate-driven target instead of sharing the constant 57C. Consequence:
Cluster 1's MCDM consensus #1 is now **PureTemp 53**, not PureTemp 58/RT60 — a real, different
recommendation driven by that cluster's climate. Clusters 0, 3, and 4 still share `Tm_target=57C`
and mostly the same Top-1 pick (PureTemp 58) — that remaining overlap is now a legitimate finding
(those three regimes genuinely don't need a different melting-window target), not a bug. Phase 7's
physics simulation (once its own two bugs were fixed — see `09_PHASE_7_AUDIT.md`) provides the
further differentiation this section previously said was missing: per-cluster Spearman rho between
MCDM rank and simulated solar fraction ranges from -0.227 to +0.171, a real, cluster-specific signal
even though none reach statistical significance.

### Phase -> broader-project mapping

| Phase | Feeds | Nature of the contribution |
|---|---|---|
| 1–2 | Climate-data foundation | A validated 10-year, 45-point, dual-source Uttarakhand climate dataset |
| 3–4 | Climate-aware recommendation (Objective 1's own gap) | Population-weighted regime discovery from a physically justified signature |
| 5–6 | Climate-aware recommendation | Per-regime PCM screening and multi-method ranking |
| 7 | Experimental/physics validation | **Addressed** — grey-box lumped-enthalpy tank simulation (`10_physics_validation.py`), run and validated against `scipy.integrate.solve_ivp`; an honest negative benchmark-match result (0% in-band) is itself a usable input to those objectives |
| 8 | Design optimisation and hardware objectives | Per-regime PCM recommendations are the input those objectives would consume |
| — | Real-time DRL control | Not addressed — explicitly out of scope for Objective 1 |

### What this mapping does not claim

- That Objective 1 addresses the DRL-control, design-optimisation or hardware-prototype gaps — it
  does not; it produces the input they consume.
- That the K = 5 partition is externally validated — no Köppen-Geiger or NBC/ECBC comparison exists.
- That the Top-3 ranking is physics-confirmed — Phase 7 was built and run, but per-cluster Spearman
  rho (-0.227 to +0.171, none significant) shows only weak/no agreement between the MCDM rank and
  simulated solar fraction; report this honestly rather than as confirmation.
- That the once-identical-across-regimes result was a correct mathematical outcome of a constant
  `Tm_target` — it was traced to a real bug (`07b`'s regime cap dividing away its own signal) and is
  now fixed for Clusters 1 and 2; Clusters 0/3/4 sharing a target and a Top-1 pick is the remaining,
  legitimate part of that finding.

## What remains

Taken directly from `NEXT_STEPS.md` and `README.md`'s "Notes / known limitations", plus what this
audit confirmed against the artefacts. Full detail and priority ranking in
`12_FINAL_READINESS_REPORT.md`.

Items 1-5 below were open when this section was first written; all five are now RESOLVED (2026-09).
Kept here as a record, with each item's resolution noted:

1. **~~Resolve the ERA5 GHI magnitude anomaly.~~ RESOLVED.** The root cause was `deaccumulate()` in
   `02_combine_uttarakhand.py` assuming an old ERA5 accumulation convention that the current
   CDS/cfgrib pipeline doesn't use — it was computing a "delta of hourly totals" instead of the
   hourly totals themselves, deflating GHI ~10x. Fixed to return the raw per-step value directly.
   Post-fix: noon GHI averages ~683 W/m², cross-source agreement with NASA POWER is MBE=+19.6 W/m²,
   r=0.759 (was MBE=-602 W/m², r=-0.03). See `04_PHASE_2_AUDIT.md` Part A.3.
2. **~~Replace the flat 1200 m elevation proxy~~ RESOLVED.** `00c_attach_elevation.py` now attaches
   real per-point elevation (196m-2510m) from ERA5's time-invariant geopotential field.
3. **~~Restore differentiation between regimes.~~ RESOLVED.** The root cause was the `07b`
   regime-cap normalization bug described above, not a need to simply "run 07b before 07" (it was
   already being run — it just couldn't produce a differentiating result). Fixed; Clusters 1 and 2
   now get a real, lower `Tm_target`.
4. **~~Implement Phase 7~~ RESOLVED.** `10_physics_validation.py` exists, has been run, and had two
   solver bugs fixed (backward-Euler numerator error, one-directional latent-heat accumulator) —
   see `09_PHASE_7_AUDIT.md`.
5. **~~Add PROMETHEE II / VIKOR and Monte Carlo stability~~ RESOLVED.** Both implemented and run —
   `08_mcdm_ranking.py` now uses all four methods, and `09b_monte_carlo_stability.py` runs 5,000
   draws/cluster.
6. **Commit the ~10 small result CSVs** so paper numbers trace to files rather than plot internals —
   still open; `data/processed/` remains git-ignored.
7. **Fix `monsoon_index`** by adding `PRECTOTCORR` to `01b`'s `POWER_PARAMETERS`, or keep reporting
   it as a 3×/day ERA5 proxy — `NEXT_STEPS.md` explicitly says *don't* fix it now; still open by
   design, not an oversight.

## Documentation map

| File | Contents |
|---|---|
| `00_MASTER_OVERVIEW.md` | This file — pipeline status, architecture, novelty/research-gap mapping |
| `01_PROJECT_CONTEXT.md` | Scope decision, phase numbering, sprint status, known internal inconsistencies |
| `02_DATA_SOURCES_AND_VARIABLES.md` | Every data source, variable, bound and signature index |
| `03_PHASE_1_AUDIT.md` | Data collection **+ spatial and temporal processing justification** |
| `04_PHASE_2_AUDIT.md` | Combine, Tier-2 repair, **ERA5 de-accumulation, solar geometry, derived solar variables, cross-source validation, and the full 13-step quality control** |
| `05_PHASE_3_AUDIT.md` | Climate signature **+ feature-to-PCM-property mapping** |
| `06_PHASE_4_AUDIT.md` | Regime clustering |
| `07_PHASE_5_AUDIT.md` | PCM database and feasibility filtering |
| `08_PHASE_6_AUDIT.md` | MCDM ranking engine |
| `09_PHASE_7_AUDIT.md` | Physics validation — implemented, run, two solver bugs fixed |
| `10_PHASE_8_AUDIT.md` | Recommendation cards |
| `13_LITERATURE_MAPPING.md` | The pipeline's complete citation footprint and the gaps to close |
| `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md` | Plot inventory, verification suite, and 13 figure defects |
| `12_FINAL_READINESS_REPORT.md` | Implementation issues, reproducibility audit, final verdict |
| `CONSOLIDATION_SUMMARY.md` | What was merged into what, and why |
