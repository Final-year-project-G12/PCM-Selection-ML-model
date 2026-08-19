# 00 — Master Overview: ERA5 Assam Climate → PCM Selection Pipeline

## Project objective

Final-year B.Tech CSE project (Group 12, Amrita School of Engineering, Guide: Dr. T. Deepika):
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water
Heating."** Objective 1 (the scope of this audit) builds a **climate-region-aware PCM
recommendation framework**: turn 10 years of reanalysis climate data into population-weighted
climate regimes, derive PCM performance targets per regime, and rank candidate phase-change
materials against those targets with an auditable, multi-method, uncertainty-aware pipeline.

Governing document: `Objective1_PCM_Climate_Framework_Plan_v3.docx` ("the framework doc"),
version 3.0, which supersedes v2.0. It defines **Phase 1 through Phase 8**. This documentation
covers **Assam** — the third state in the four-state pipeline, processed after Rajasthan and
Tamil Nadu.

## What the ERA5 Assam pipeline is trying to achieve

Assam is climatically unique in the four-state comparison: it is the only state dominated by an
extreme monsoon regime (>2500 mm/yr in parts), with high humidity, moderate but variable solar
radiation, and a distinct climate gradient from the Brahmaputra valley to the hill districts. The
pipeline:

1. Samples Assam at **128 population-weighted points** (87.5% population coverage) on ERA5's 0.25°
   native grid, so results are defensible against "why these locations?"
2. Pulls two independent climate data sources (ERA5 reanalysis, NASA POWER satellite/model product)
   for the *same* points and instants, and cross-validates one against the other.
3. Reduces 10 years of hourly/daily data per point into an **18-index two-tier climate signature**
   (sun-event statistics + daily-integral indices, including a Humidity-Solar Interaction index
   that is climatologically important for Assam's monsoon character).
4. Clusters 128 points into **4 climate regimes** (Gaussian Mixture Model, full covariance — chosen
   over diagonal because Assam's monsoon-RH-solar correlations span elongated clusters) at spatial
   Level A.
5. Derives a **uniform PCM performance target** across all Assam clusters: Tm_target = 44.0°C
   (delivery 50°C − 6°C approach), with cluster-specific L_required derived from each regime's
   own solar resource.
6. Filters a 25-row PCM property database against 7 physical/safety/corrosion constraints
   (including a corrosion veto that is **load-bearing for Assam** due to the HSI > global-p75
   threshold in the humid clusters), then ranks survivors with **four independent MCDM methods**
   plus 5,000-draw Monte Carlo uncertainty propagation.
7. Independently validates the MCDM ranking against a physics-based lumped-enthalpy tank
   simulation (Phase 7), and packages the result as per-cluster recommendation cards (Phase 8).
   All four clusters return weak Spearman rho (0.167–0.286) — a genuine negative validation
   consistent with the undersized PCM database.

## Complete pipeline map (as actually implemented)

```
Phase 1 — DATA COLLECTION
  00a_build_population_grid.py   → population_grid_points.csv (128 pts, 87.5% pop coverage)
  00b_build_suntimes.py          → suntimes.csv (sunrise/noon/sunset per point per day)
  01_download_era5_assam.py      → data/raw/era5/points/*.nc  (sun-event-aligned hours)
  01b_download_nasapower.py      → data/raw/nasapower/*.json  (10 years × 128 pts)
  00_unzip_accum.py              → (fixes CDS zip-disguised-as-.nc quirk)
        ↓
Phase 2 — PREPROCESSING & CROSS-SOURCE VALIDATION
  02_combine_assam.py            → climate_assam_points.csv (unit conv., solar geometry,
                                   ERA5+POWER merge, seasons: Winter/Pre-Monsoon/Monsoon/Post-Monsoon)
  02b_build_daily_aggregates_assam.py → daily_aggregates_assam.csv + tier2_signature_assam.csv
        ↓
Phase 2.5 — QUALITY CONTROL
  04_preprocess_assam.py         → preprocessed/parquet/{point_id}.parquet
                                   (physical bounds check, IsolationForest outlier flagging,
                                    imputation; outliers FLAGGED but NEVER deleted)
        ↓
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
  04b_climate_signature.py       → climate_signatures_raw.csv (18 indices per site)
                                   climate_signatures_matrix.csv (PCA + standardised)
                                   pca_loadings.csv
                                   (Tm_target = 44°C; Tsoil_mean ≈ Ta_mean fallback — approved)
        ↓
Phase 4 — CLIMATE REGIME CLUSTERING
  05_cluster_assam.py            → clustering/cluster_assignments_assam.csv (4 clusters, 128 pts)
                                   clustering/cluster_profiles_assam.csv
                                   clustering/bic_selection_assam.csv  (k=4 by BIC/silhouette)
                                   clustering/bootstrap_stability_assam.csv
                                   (GMM full covariance; k-Means as robustness comparison;
                                    ARI_mean=0.716, ARI_std=0.139, stable=False at k=4)
        ↓
Phase 5 — PCM DATABASE + FEASIBILITY FILTERING
  06_build_pcm_database.py       → pcm/pcm_database_assam.csv (25 rows: manufacturer MICE-RF-PMM
                                   + Singh2025 literature PCMs)
  07_feasibility_filter.py       → pcm/feasibility_survivors_assam.csv
                                   (7-constraint filter: melting window, abs band, latent-heat
                                    floor κ=0.7, cycling ≥300, corrosion veto when HSI>p75,
                                    supercooling ≤8K, safety keyword veto; auto-relaxes window
                                    +2K/step if <5 survive)
        ↓
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking.py             → pcm/mcdm_topk_assam.csv
                                   pcm/mcdm_full_scores_assam.csv
                                   pcm/monte_carlo_stability_assam.csv
                                   (TOPSIS+GRA+PROMETHEE II+VIKOR; Borda+Copeland consensus;
                                    Kendall's W; 5,000-draw Monte Carlo — N_DRAWS=5000, matching
                                    the plan spec, unlike Rajasthan's 1000-draw run)
        ↓
Phase 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py       → pcm/physics_validation_results_assam.csv
                                   pcm/physics_validation_spearman_assam.csv
                                   (grey-box lumped-enthalpy tank, real 10-year daily data from
                                    each cluster's medoid point; backward Euler implicit solver;
                                    calibration against 54–84% solar-fraction literature band)
        ↓
Phase 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py     → pcm/recommendation_cards_assam.md
                                   (includes Analytical Criterion Contributions — percentage
                                    breakdown per criterion per PCM — a requirement that the
                                    Tamil Nadu Phase 8 script missed; added here)
```

## Phase 1–8 status at a glance

| Phase | Script(s) | Status | Headline finding |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 128 pts, 87.5% pop coverage, 10 yrs ERA5 + POWER |
| 2 — Preprocessing & Validation | `02`, `02b` | **COMPLETE** | ERA5+POWER merge; 4-season classification including Monsoon |
| 2.5 — Quality Control | `04` | **COMPLETE** | IsolationForest outlier flagging; imputation; parquet per-point output |
| 3 — Climate Signature | `04b` | **COMPLETE** | 18 indices; PCA on thermodynamic block; Tm_target=44°C fixed; Tsoil≈Ta_mean fallback |
| 4 — Regime Clustering | `05` | **COMPLETE** | k=4 (GMM full covariance); BIC minimum at k=9 but k=4 chosen for interpretability vs. BIC; bootstrap ARI=0.716 (borderline stable) |
| 5 — Feasibility Filtering | `06`, `07` | **COMPLETE** | 25-row database; corrosion veto load-bearing for humid clusters; 6 or 8 survivors per cluster after κ-relaxation |
| 6 — MCDM Ranking | `08` | **COMPLETE** | 5,000-draw MC (matches plan spec); RT44HC #1 in all clusters; strong Kendall's W (0.807–0.845) |
| 7 — Physics Validation | `10` | **COMPLETE — genuine NEGATIVE result** | Spearman rho = 0.257/0.257/0.286/0.167 across 4 clusters — all weak |
| 8 — Recommendation Cards | `09` | **COMPLETE** | Includes Criterion Contributions breakdown (explainability mandate) |

## Current architecture

- **Language/stack**: Python, pandas/numpy/scikit-learn/scipy, `pvlib` for solar geometry,
  `cdsapi` for ERA5, `xarray`/`netCDF4` for NetCDF, `joblib` for GMM/scaler persistence.
- **Path convention**: every script imports `config.py`, which anchors all paths to
  `era5-assam/` regardless of working directory.
- **Resumability**: download stages have idempotency mechanisms (status-CSV logging + file checks).
- **Reproducibility**: fitted StandardScaler and GMM saved as `scaler_assam.joblib` and
  `gmm_model_assam.joblib`; `sklearn_version` recorded in every cluster output CSV.
- **State-parameterization**: scripts are written state-agnostically, anticipating the combined
  4-state clustering run.

## Key Assam-specific design choices (vs Rajasthan)

| Design choice | Rajasthan | Assam | Reason |
|---|---|---|---|
| Grid points | 320 | 128 | Assam is smaller; 87.5% coverage achieved with 128 |
| GMM covariance | `diag` (fixed bug) | `full` | Assam's monsoon-RH-solar correlations span elongated clusters; full covariance justified |
| k (clusters) | 3 | 4 | BIC keeps falling; k=4 chosen for interpretability (valley / hill / Barak / char geography) |
| Tm_target | Regime-specific (57°C capped) | 44°C uniform | Assam's moderate SWH load — T_delivery=50°C, ΔT=6°C; same Tm_target for all regimes |
| Monte Carlo draws | 1,000 (documented deviation) | 5,000 | Matches plan spec §9.6 exactly |
| Corrosion veto | Present but not load-bearing | **Load-bearing** | HSI > global p75 in humid clusters triggers inorganic PCM exclusion |
| Criterion Contributions | Not implemented | **Implemented** | Explainability mandate from plan doc — missed in TN, corrected in Assam |

## Main datasets produced

| File | Rows | Grain | Produced by |
|---|---|---|---|
| `population_grid_points.csv` | 128 | 1 row/point | `00a` |
| `suntimes.csv` | ~140k | 1 row/point/date/event | `00b` |
| `climate_assam_points.csv` | ~1.4M | 1 row/point/date/event | `02` |
| `daily_aggregates_assam.csv` | ~467k (128×3653) | 1 row/point/day | `02b` |
| `climate_signatures_raw.csv` | 128 | 1 row/point, 18 indices | `04b` |
| `climate_signatures_matrix.csv` | 128 | 1 row/point, PCA+std | `04b` |
| `cluster_assignments_assam.csv` | 128 | 1 row/point | `05` |
| `cluster_profiles_assam.csv` | 4 | 1 row/cluster | `05` |
| `pcm_database_assam.csv` | 25 | 1 row/PCM | `06` |
| `feasibility_survivors_assam.csv` | 28 (4 clusters × 6–8 survivors) | 1 row/cluster×PCM | `07` |
| `mcdm_topk_assam.csv` | 12 (4 clusters × top-3) | 1 row/cluster×PCM | `08` |
| `monte_carlo_stability_assam.csv` | 28 | 1 row/cluster×PCM | `08` |
| `physics_validation_results_assam.csv` | 28 | 1 row/cluster×simulated PCM | `10` |
| `physics_validation_spearman_assam.csv` | 4 | 1 row/cluster | `10` |
| `recommendation_cards_assam.md` | 4 cards + summary | 1 card/cluster | `09` |

## Main algorithms

Solar geometry (pvlib SPA + Ineichen clear-sky) · Magnus-formula RH · Gaussian-mixture clustering
(full covariance) with 500-bootstrap ARI stability · PCA (thermodynamic block only, solar+variability
kept out) · IsolationForest outlier detection · MICE-style imputation · Shannon-entropy criterion
weighting · TOPSIS · PROMETHEE II · VIKOR · Grey Relational Analysis · Borda count · Copeland
pairwise · Kendall's W · Dirichlet/Gaussian Monte Carlo uncertainty propagation (5,000 draws) ·
Lumped-enthalpy backward-Euler PCM tank simulation.

## Validation strategy

Two validation layers: (1) **cross-source** — ERA5 vs NASA POWER agreement analysis, and
(2) **internal statistical** — GMM bootstrap-ARI stability, silhouette/BIC/Davies-Bouldin/
Calinski-Harabasz for cluster count, Monte Carlo inclusion-probability for MCDM rank stability,
Kendall's W for cross-method ranking agreement. A third layer — **physics-based simulation
validation** (Phase 7) — is implemented and run, returning a genuine NEGATIVE result (weak
Spearman rho across all 4 clusters). External climate classification (Köppen-Geiger, NBC/ECBC)
is **not wired in** for Assam — this is an open gap.

## What remains

All 8 phases are implemented and run. What remains is the same diagnostic path as Rajasthan:

1. **Expand the PCM property database** to 40–60 rows in the 42–70°C band — the single blocking
   item for trustworthy Phase 5/6/7/8 results.
2. **Investigate the negative Phase 7 result** — weak rho across all 4 clusters is consistent with
   the undersized PCM pool (n=6 or n=8 per cluster), not necessarily a MCDM methodology failure.
3. **Wire in external climate classification** (Köppen-Geiger) for Phase 4 external validation —
   currently absent for Assam.
4. **Decide the κ-relaxation policy** for the latent-heat constraint permanently.
