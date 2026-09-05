Uttarakhand
Objective 1 Audit Consolidation
Climate-Region-Aware PCM Recommendation Framework
Complete consolidated audit set — source-preserving compilation
Prepared from the uploaded Uttarakhand audit files

# Consolidated File Inventory
This document consolidates the complete Uttarakhand-specific audit set identified in the uploaded files. The source order is preserved from the master overview.
1. 00_MASTER_OVERVIEW(3).md
1. 01_PROJECT_CONTEXT(3).md
1. 02_DATA_SOURCES_AND_VARIABLES(3).md
1. 03_PHASE_1_AUDIT(3).md
1. 04_PHASE_2_AUDIT(3).md
1. 05_PHASE_3_AUDIT(3).md
1. 06_PHASE_4_AUDIT(3).md
1. 07_PHASE_5_AUDIT(3).md
1. 08_PHASE_6_AUDIT(3).md
1. 09_PHASE_7_AUDIT(1).md
1. 10_PHASE_8_AUDIT(1).md
1. 11_LITERATURE_MAPPING.md

# Source File 0: 00_MASTER_OVERVIEW(3).md
Source path: /mnt/data/00_MASTER_OVERVIEW(3).md
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
(`00a_build_population_grid.py`, `COVERAGE_TARGET = 0.875`).
2. Pulls **ERA5 reanalysis** and **NASA POWER** for the *same* points and the *same* sun-event
instants, keeping both as `era5_*` / `power_*` columns so one can be cross-checked against the
other.
3. Repairs the 3-rows-per-day sampling limitation by re-reading the **full NASA POWER hourly
cache** already on disk to build true daily integrals (`02b_build_daily_aggregates.py`,
"Phase 2 Repair 1").
4. Reduces 10 years × 3 sun-events/day per point into a **two-tier ~18-index climate signature**
(Tier 1 sun-event proxies + Tier 2 true daily integrals), plus 5 interaction terms and a PCA of
the correlated temperature/pressure block (`04b_climate_signature.py`).
5. Clusters the 45 points into climate regimes with a **Gaussian Mixture Model, full covariance,
K_FINAL = 5** (`05_cluster_uttarakhand.py`).
6. Screens a **55-row PCM property database** against each regime's `Tm_target` / `L_required`
(`06_build_pcm_database.py`, `07_feasibility_filter.py`), then ranks survivors with a
**two-method MCDM stack (TOPSIS + GRA)**, entropy/AHP-blended weights, a Gaussian Tm-fitness
transform, and a **Borda consensus with Kendall's W** (`08_mcdm_ranking.py`).
7. Aggregates the result into one markdown recommendation card per regime
(`09_recommendation_cards.py`).

## Complete pipeline map (as actually implemented in `era5-uttarakhand/`)

PHASE 0/1 — SAMPLING DESIGN + RAW DOWNLOAD
  00a_build_population_grid.py    -> data/processed/population_grid_points.csv   (45 pts, >=87.5% pop)
  00b_build_suntimes.py           -> data/processed/suntimes.csv                 (pvlib SPA, UTC)
  01_download_era5_uttarakhand.py -> data/raw/era5/points/era5_UK_points_{yyyy}_{mm}_{instant,accum}.nc
  01b_download_nasapower.py       -> data/raw/nasapower/power_{point_id}_{year}.json
  00_unzip_accum.py               -> (fixes CDS zip-disguised-as-.nc files in place)
        |
PHASE 2 — COMBINE + DAILY-INTEGRAL REPAIR
  02_combine_uttarakhand.py       -> data/processed/climate_uttarakhand_points.csv
  02b_build_daily_aggregates.py   -> data/processed/daily_aggregates_uttarakhand.csv
                                     data/processed/tier2_signature_uttarakhand.csv
        |
PHASE 2 QA — RAW CHECKS (read-only, before cleaning)
  03_plots_raw.py                 -> data/plots/raw/*.png  + C_era5_vs_power_stats.csv
  03b_interactive_raw_qa.py       -> data/plots/raw_interactive/*.html
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
  07b_charging_feasibility.py     -> (optional) adds Tm_target_C_regime_capped to cluster profiles
  07_feasibility_filter.py        -> data/processed/pcm/feasibility_survivors_by_cluster.csv
        |
PHASE 6 — MULTI-CRITERIA RANKING
  08_mcdm_ranking.py              -> data/processed/pcm/mcdm_topk_by_cluster.csv
                                     data/processed/pcm/mcdm_full_scores_by_cluster.csv
        |
PHASE 7 — PHYSICS-BASED VALIDATION
  (NO SCRIPT PRESENT — see 09_PHASE_7_AUDIT.md)
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

## Phase 1–8 status at a glance

| Phase | Script(s) | Status | Headline finding (Uttarakhand) |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` | **RUN** (evidenced by downstream artefacts) | 45 points `UKP_0001–UKP_0045`, 10,475,711 population covered, 2016–2025 |
| 2 — Combine + Tier-2 repair | `02`, `02b` | **RUN** | `climate_uttarakhand_points.csv` = **493,155 rows** = 45 × 3653 × 3 exactly (no rows lost to the 3 h match window) |
| 2 QA — Raw checks | `03`, `03b` | **RUN** | Noon peaks GHI (timezone check passes) but ERA5-vs-POWER GHI **MBE = −211.4 W/m², r = 0.432** |
| 2 — Preprocessing & QC | `04`, `04c` ×2 | **RUN** | 493,155 -> **489,105 rows** (99.2 % retention); 36 -> 89 columns; 0 residual NaN |
| 3 — Climate Signature | `04b`, `04d` | **RUN** | `Tm_target` fixed at **57 °C** for every point (50 + 7, indirect-system rule) |
| 4 — Regime Clustering | `05`, `05b` | **RUN** | **K_FINAL = 5**, GMM full covariance; sizes **12 / 9 / 3 / 7 / 14**; silhouette 0.279 |
| 5 — Feasibility Filtering | `06`, `07` (`07b` optional) | **RUN** | 55-candidate database; melting window [52, 65] °C; **29 candidates satisfy every implemented filter, identically in all 5 clusters** |
| 6 — MCDM Ranking | `08` | **RUN** | TOPSIS + GRA + Borda; **RT60 is consensus rank 1 in all 5 clusters**; pooled TOPSIS-vs-GRA Spearman **rho = −0.930** |
| 7 — Physics Validation | — | **NOT IMPLEMENTED** | No script exists; `NEXT_STEPS.md` records it as accepted future work |
| 8 — Recommendation Cards | `09` | **CODE PRESENT, OUTPUT NOT COMMITTED** | `recommendation_cards.md` is under the git-ignored `data/processed/` tree |

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
| Default altitude for solar geometry | **1200 m**, flat, for every point | `02_combine_uttarakhand.py` `DEFAULT_ALT_M = 1200` |
| Season map | Winter DJF / **Summer MAM** / **Monsoon JJA** / **Retreat SON** | `02_combine_uttarakhand.py` `SEASON_MAP` |
| Accumulated-field handling | `deaccumulate()` — `diff()` with hour-1/hour-13 reset special case | `02_combine_uttarakhand.py` |
| `Tm_target` | Constant **57 °C** (`T_DELIVERY_C = 50` + `DT_APPROACH_C = 7`) | `04b_climate_signature.py` |
| GMM covariance | `full` | `05_cluster_uttarakhand.py` |
| K_FINAL | **5** | `05_cluster_uttarakhand.py` line 73 |
| Silhouette accept band | 0.15 – 0.40 (widened from the 4-state 0.15 – 0.35) | `05_cluster_uttarakhand.py` |
| PCM database size | **55 rows** (31 manufacturer + 24 literature) | `06_build_pcm_database.py`; verified against the CSV |
| MCDM methods | **TOPSIS + GRA only** | `08_mcdm_ranking.py` |
| Monte Carlo draws | **not implemented** | `08_mcdm_ranking.py` closing docstring |
| Physics validation | **not implemented** | `README.md`, `NEXT_STEPS.md` |

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
| `cluster_profiles_uttarakhand.csv` | 5 | 1 row/cluster | `05` | observed |
| `pcm_database_uttarakhand.csv` | 55 | 1 row/PCM | `06` | observed (source CSV + plot counts) |
| `feasibility_survivors_by_cluster.csv` | **275** (55 × 5, all rows kept with `passes_all` flag) | 1 row/cluster × PCM | `07` | observed (verify summary) |
| `mcdm_topk_by_cluster.csv` | **15** (5 clusters × Top-3) | 1 row/cluster × PCM | `08` | observed (verify summary) |
| `mcdm_full_scores_by_cluster.csv` | approx. 145 (5 × 29 survivors) | 1 row/cluster × survivor | `08` | expected |
| `recommendation_cards.md` | 5 cards | 1 card/cluster | `09` | not observed — output not committed |

## Main algorithms

pvlib SPA sun-rise/transit/set · pvlib solar position + Ineichen clear-sky · Magnus-formula RH ·
ERA5 accumulated-field `diff()` de-accumulation with 00Z/12Z reset handling · nearest-neighbour
ERA5 grid snapping · nearest-in-time (<= 3 h) cross-source matching · physical-bounds -> NaN
validation · Hampel/MAD outlier flagging over sun-event occurrences · hierarchical imputation
(interpolate -> ffill/bfill -> point/zone/global median -> MICE `IterativeImputer`) · Yeo-Johnson
skew diagnostic · Savitzky-Golay smoothing diagnostic · Pearson/Spearman correlation · VIF ·
MinMax scaling with a chronological 70 % train fit · PCA (temperature/pressure block only, 95 %
variance) · z-standardisation · Gaussian Mixture (full covariance) with BIC / silhouette /
Davies-Bouldin / Calinski-Harabasz model selection · K-Means comparison · population-weighted
cluster profiling · MICE + Random-Forest + Predictive-Mean-Matching PCM property imputation ·
Gaussian Tm-fitness transform (sigma = 4 K) · Shannon-entropy criterion weighting blended 0.5/0.5
with an AHP-style prior · TOPSIS · Grey Relational Analysis (zeta = 0.5) · Borda count ·
Kendall's W · heuristic clear-sky-reliability charging cap (`07b`, optional).

## Validation strategy actually present

Three layers exist in `era5-uttarakhand/`:

1. **Cross-source** — `03_plots_raw.py` / `03b_interactive_raw_qa.py` compute ERA5-vs-NASA-POWER
MBE / RMSE / Pearson *r* per variable and write `C_era5_vs_power_stats.csv`. There is **no**
agreement-analysis script, no bias-decision file, and no bias-correction branch anywhere in
`04_preprocess_uttarakhand.py`. The measured disagreement is reported but never acted upon.
2. **Internal statistical** — `04` step 13 hard gate; `05`'s BIC / silhouette / Davies-Bouldin /
Calinski-Harabasz table plus a K-Means silhouette comparison; `08`'s Kendall's W per cluster.
There is **no** bootstrap-ARI stability analysis and **no** Monte Carlo rank stability.
3. **Post-hoc verification suite** — `verify_01`…`verify_04` re-open the saved outputs and
regenerate independent diagnostics. See
`11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`, including two real defects in that suite.

**Not present for Uttarakhand:** physics-based simulation validation (Phase 7), external climate
classification (Köppen-Geiger / NBC-ECBC), bootstrap cluster stability, and Monte Carlo
uncertainty propagation.

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
| 4 — Regime Clustering | Discovered regimes, not hand-picked zones | GMM full covariance, K = 5 by manual selection from a BIC/silhouette table; lat/lon excluded | **Delivered** — clusters are spatially coherent without clustering on geography. **But** no bootstrap stability, no external classification, and soft membership collapsed to 1.000 |
| 5 — Feasibility Filtering | Corrected 42–70 °C SWH-specific PCM band | Band enforced; melting window [52, 65] °C at `Tm_target = 57` | **Partially delivered** — 3 of 5 Table-12 filters unimplemented; the corrosion veto cannot activate (all 55 candidates organic) |
| 6 — MCDM Ranking | Top-3 with explicit method-agreement reporting | TOPSIS + GRA, entropy/AHP weights, Gaussian Tm fitness, Borda, Kendall's W | **Partially delivered** — only 2 methods, no Monte Carlo, and the two methods are anti-correlated at rho = −0.930 |
| 7 — Physics Validation | Physics-validated ranking | — | **Not delivered** — no script exists |
| 8 — Recommendation Cards | Per-regime explainable output | 5 cards; population-weighted profiles; Top-3 with per-method scores and Kendall's W | **Delivered as code**; output not committed; no criterion-contribution breakdown |

### The central finding against the novelty claim

The framework's core proposition is that **different climate regimes should receive different PCM
recommendations.** For Uttarakhand this run does **not** demonstrate it: all five regimes return
the same 29 feasibility survivors and the same #1 PCM (RT60). The cause is traceable and stated
in-code — `Tm_target` is held constant at 57 °C by design, and `L_required` lands well below every
candidate's latent heat, so neither Phase 5 filter discriminates between regimes.

`08_mcdm_ranking.py` detects this and offers two honest framings, both reproduced in
`08_PHASE_6_AUDIT.md`. The one Objective 1 can defend today is: *Uttarakhand's climate regimes
differ more in solar reliability and cloud persistence than in delivery-relevant temperature, so
under the corrected `Tm_target` rule a single PCM family serves the whole state; differentiation
would have to appear in Phase 7's physics simulation, which was not built.*

### Phase -> broader-project mapping

| Phase | Feeds | Nature of the contribution |
|---|---|---|
| 1–2 | Climate-data foundation | A validated 10-year, 45-point, dual-source Uttarakhand climate dataset |
| 3–4 | Climate-aware recommendation (Objective 1's own gap) | Population-weighted regime discovery from a physically justified signature |
| 5–6 | Climate-aware recommendation | Per-regime PCM screening and multi-method ranking |
| 7 | Experimental/physics validation | **Not addressed** — no simulation exists |
| 8 | Design optimisation and hardware objectives | Per-regime PCM recommendations are the input those objectives would consume |
| — | Real-time DRL control | Not addressed — explicitly out of scope for Objective 1 |

### What this mapping does not claim

- That Objective 1 addresses the DRL-control, design-optimisation or hardware-prototype gaps — it
does not; it produces the input they consume.
- That the K = 5 partition is externally validated — no Köppen-Geiger or NBC/ECBC comparison exists.
- That the Top-3 ranking is physics-confirmed — Phase 7 was not built.
- That the identical-across-regimes result is a coding failure — it is the correct mathematical
outcome of a constant `Tm_target` and a non-binding latent-heat floor.

## What remains

Taken directly from `NEXT_STEPS.md` and `README.md`'s "Notes / known limitations", plus what this
audit confirmed against the artefacts. Full detail and priority ranking in
`12_FINAL_READINESS_REPORT.md`.

1. **Resolve the ERA5 GHI magnitude anomaly.** Raw mean noon `era5_GHI` is approximately 61 W/m²
and the cleaned whole-file mean 21.03 W/m² with max 702.74 W/m², against a NASA POWER MBE of
−211.4 W/m² and r = 0.432. Every solar-derived index inherits this. Verification is a single
inspection of a raw `*_accum.nc` file — see `04_PHASE_2_AUDIT.md` Part A.3.
2. **Replace the flat 1200 m elevation proxy** with real per-point elevation.
`README_PREPROCESSING.md` calls this "a real limitation here, not a footnote" for Uttarakhand's
~200–2000 m populated range, and 37.1 % of `era5_P_atm` values were NaN'd by a lowland-tuned
850 hPa lower bound before imputation — in the exact column `elev_proxy` is built from.
3. **Restore differentiation between regimes.** Run `07b_charging_feasibility.py` before `07`, or
report the convergence as a finding using `08`'s own wording.
4. **Implement Phase 7** (grey-box lumped-enthalpy tank) or state it as future work. Every input it
needs is already on disk — see `09_PHASE_7_AUDIT.md`.
5. **Add PROMETHEE II / VIKOR and Monte Carlo stability** — listed as stretch goals in `08`'s own
closing text. With TOPSIS and GRA anti-correlated, a third independent method would materially
strengthen the consensus.
6. **Commit the ~10 small result CSVs** so paper numbers trace to files rather than plot internals.
7. **Fix `monsoon_index`** by adding `PRECTOTCORR` to `01b`'s `POWER_PARAMETERS`, or keep reporting
it as a 3×/day ERA5 proxy — `NEXT_STEPS.md` explicitly says *don't* fix it now.

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
| `09_PHASE_7_AUDIT.md` | Physics validation — not implemented |
| `10_PHASE_8_AUDIT.md` | Recommendation cards |
| `11_LITERATURE_MAPPING.md` | The pipeline's complete citation footprint and the gaps to close |
| `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md` | Plot inventory, verification suite, and 13 figure defects |
| `12_FINAL_READINESS_REPORT.md` | Implementation issues, reproducibility audit, final verdict |
| `CONSOLIDATION_SUMMARY.md` | What was merged into what, and why |

# Source File 1: 01_PROJECT_CONTEXT(3).md
Source path: /mnt/data/01_PROJECT_CONTEXT(3).md
# 01 �?" Project Context

## Identity

"OBJECTIVE 1 �?" Climate-Region-Aware PCM Recommendation Framework," Group 12, B.Tech CSE Final
Year, Amrita School of Engineering. This documentation covers the **Uttarakhand** state pipeline,
implemented in `PCM-Selection-ML-model/era5-uttarakhand/`.

The Uttarakhand scripts cite the governing plan document as **v3.0** in every Phase 2�?"8 docstring
(`02b`, `04b`, `04`, `06`, `07`, `07b`, `08`, `09`, `05_cluster_uttarakhand`). The single
exception is `05_cluster_regions.py`, which still cites **v2.0 §7** �?" this is the unrun
multi-state script and its version lag is visible in its own docstring.

The plan document itself is **not present inside `era5-uttarakhand/`**. Every plan reference in
this documentation set is therefore recorded as "cited by the script", not verified against the
document.

## Scope decision recorded in the source files

`NEXT_STEPS.md`, line 3:

Scope decision for this sprint: **finish Objective 1 on Uttarakhand alone.** Cross-state
clustering (the original 4-state plan v3.0 design) is real future work, already documented and
state-parameterised, but it is not required to defend Objective 1 as a working framework. Don't
spend time trying to onboard another state's data yet.

`README_PREPROCESSING.md` restates it:

**You do not need to cluster across other states to finish Objective 1.** The objective
statement is "cluster meteorological data and identify Top-2/Top-3 PCM candidates per climatic
regime" �?" nothing requires those regimes to span state boundaries.

`05_cluster_uttarakhand.py`'s docstring gives the same justification and adds the expected
within-state structure: "the high-altitude Himalayan belt around Chamoli/Pithoragarh vs. the Doon
Valley around Dehradun vs. the Terai plains around Udham Singh Nagar/Haridwar are very plausibly
different regimes �?" elevation alone spans roughly 200-2000m of populated terrain here."

## Explicit "do not do this now" list

`NEXT_STEPS.md` §"What to explicitly not do right now" is unusually specific and is part of the
audit trail:

| Item | Instruction in the source file |
|---|---|
| Other states' data | "Don't onboard Rajasthan/Assam/Tamil Nadu data. `05_cluster_regions.py` stays untouched and ready for later." |
| TabTransformer/VAE encoder ablation | "Don't build" �?" "explicitly optional-only in the plan doc and adds nothing to Objective 1's core claim" |
| Per-point real elevation | **"Do"** think about it �?" "unlike the Tamil Nadu build �?�, Uttarakhand's 200m-2000m populated elevation range is exactly the case this repair was written for" |
| 5,000-draw Monte Carlo | "Don't run �?� unless Phase 5/6 finishes with time spare �?� it is genuinely optional" |
| Fixing `monsoon_index` via `PRECTOTCORR` | "Don't try" �?" "flag the proxy limitation in text instead, it costs no correctness in the ranking (monsoon_index isn't a ranking criterion, it's descriptive of the regime)" |

## Phase numbering �?" as used by the Uttarakhand scripts

| Phase | Name in the Uttarakhand docstrings | Script(s) |
|---|---|---|
| 0/1 | Sampling design + raw download | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` |
| 1 | Combine (ERA5 + NASA POWER merge) | `02_combine_uttarakhand.py` |
| 2 (Repair 1) | Daily-integral aggregates | `02b_build_daily_aggregates.py` |
| 2 | Preprocessing and Quality Control (13 steps) | `04_preprocess_uttarakhand.py` |
| 3 | Climate Signature Construction (Tier 1 + Tier 2) | `04b_climate_signature.py` |
| 4 | Climate Regime Clustering | `05_cluster_uttarakhand.py` (`05_cluster_regions.py` = multi-state, unrun) |
| 5 | PCM database + Feasibility Filtering | `06`, `07`, `07b` |
| 6 | Multi-Criteria Ranking Engine | `08_mcdm_ranking.py` |
| 7 | Physics-Based Validation | **no script in `era5-uttarakhand/`** |
| 8 | Explanation and Final Output | `09_recommendation_cards.py` |

Note the numbering quirk: the Uttarakhand `README.md` labels `02_combine_uttarakhand.py` as
"PHASE 2 �?" COMBINE" in its pipeline diagram but `README_PREPROCESSING.md` and
`PREPROCESSING_STEPS.md` label the same script "Phase 1". Both labellings appear in the source
files; neither is corrected here.

## Sprint status recorded in `NEXT_STEPS.md`

The status table in `NEXT_STEPS.md` was written mid-sprint and is **older than the artefacts in
`data/plots/`**. Reproduced verbatim in substance, with this audit's finding alongside:

| Phase | `NEXT_STEPS.md` status | What the committed artefacts show |
|---|---|---|
| 1. Data Collection | "**Done.** Points confirmed �?�, ~87.5% population coverage" | Confirmed �?" 45 points, 10,475,711 population |
| 2. Preprocessing & QC | "`02b` confirmed run (45/45 points, 0 skipped, 164,385 point-days). `04` code delivered �?" confirm it's actually been run" | `04` **has** been run: 489,105 output rows, 89 columns |
| 3. Climate Signature | "Code delivered �?�, **not yet confirmed run**" | Has been run �?" cluster/PCM artefacts downstream of it exist |
| 4. Clustering | "Code delivered �?�, **not yet confirmed run**" | Has been run at **K = 5**; sizes 12/9/3/7/14 |
| 5. Feasibility | "Code delivered, **not yet run**" | Has been run �?" 275-row survivors CSV |
| 6. MCDM Ranking | "Code delivered, **not yet run**" | Has been run �?" 15-row Top-3 CSV |
| 7. Physics Validation | "**Not written.**" | Still not written �?" no script exists |
| 8. Recommendation Cards | "Code delivered, **not yet run**" | Cannot be confirmed �?" output is git-ignored |

`NEXT_STEPS.md` should be treated as a **plan document that has been overtaken by the run**, not
as a current status report.

## Known internal inconsistency: PCM database size

Two source files in `era5-uttarakhand/` disagree about the PCM database:

- `06_build_pcm_database.py` (docstring, lines 1�?"21): **55 rows** �?" 24 Literature, 14 Rubitherm
Technologies, 7 Pluss Advanced Technologies, 5 PureTemp, 4 PCM Products Ltd., 1 CrodaTherm.
- `NEXT_STEPS.md` (line 17 and line 176): "**~25 candidates total**" and "PCM database is ~25
rows, not 40-60."
- `07_feasibility_filter.py` (line 158) prints "your database (25 rows) is thin for this" in its
low-survivor warning message.

**Resolution from the artefacts:** the committed
`PCM_data/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` has exactly **55 rows** with
exactly the manufacturer breakdown `06` claims, and the committed plot
`data/plots/verify_feasibility/06_summary.png` reports 55 PCM rows per cluster. The 25-row figures
in `NEXT_STEPS.md` and `07`'s warning string are stale text from an earlier database generation.
An earlier 25-row generation is independently evidenced �?" see `12_FINAL_READINESS_REPORT.md`.

## Uttarakhand-specific contextual notes from the source files

**Terrain is the defining constraint.** `README_PREPROCESSING.md` states it directly:

`02_combine_uttarakhand.py` uses a flat **1200m** proxy for every point's solar-geometry
calculations, not real per-point elevation. �?� Uttarakhand's populated terrain genuinely spans
roughly 200m (Terai plains near Udham Singh Nagar/Haridwar) to 2000m (hill towns), and elevation
drives both solar-geometry inputs (air mass, clear-sky irradiance) and the temperature-based
indices (HDD18/CDD24, Ta_mean) directly. This is plan v3.0's "Repair 2," written with
Uttarakhand specifically in mind.

**Small N.** With only 45 points, `README_PREPROCESSING.md` flags two QC steps for extra
scepticism: step 4's spatial-zone imputation fallback ("noticeably coarser zones with 45 points to
group") and step 11's VIF ("computed over fewer independent spatial samples"). It also warns that
a high silhouette is "more likely to mean an over-simple signature than a genuinely crisp regime
split" at this N.

**Corrosion mechanism.** `NEXT_STEPS.md` anticipates that "the corrosion veto [will] bite for
high-monsoon-humidity Uttarakhand clusters (Terai/valley points during Jun-Sep) �?� same veto,
different physical mechanism, worth noting in text." **This did not happen** �?" the corrosion veto
is not implemented in `07_feasibility_filter.py` at all (its docstring lists it under "NOT
applied"), so no corrosion-driven differentiation exists in the run.

**Constant `Tm_target`.** `04b_climate_signature.py` sets `Tm_target_C = 57` for every point by
design (`T_DELIVERY_C = 50` + `DT_APPROACH_C = 7`, "indirect-system assumption"). Because the
melting-window filter and the Gaussian Tm-fitness criterion are both driven by `Tm_target`, this
is the single largest reason the five regimes return identical survivor sets and an identical #1
PCM. `08_mcdm_ranking.py` detects and prints this explicitly rather than letting it pass silently.

## What this documentation set does not claim

- It does **not** import any number, PCM name, cluster count, methodology detail, or conclusion
from the Rajasthan, Tamil Nadu, or Assam pipelines.
- Where a value could not be verified inside `era5-uttarakhand/` it is marked **"not available in
the source files."**
- Row counts recovered from committed plot artefacts are labelled *(observed)*; counts derived
arithmetically from script constants are labelled *(expected)*.

# Source File 2: 02_DATA_SOURCES_AND_VARIABLES(3).md
Source path: /mnt/data/02_DATA_SOURCES_AND_VARIABLES(3).md
# 02 �?" Data Sources and Variables

All entries below are taken from the Uttarakhand scripts themselves
(`01_download_era5_uttarakhand.py`, `01b_download_nasapower.py`, `00a_build_population_grid.py`,
`02_combine_uttarakhand.py`, `02b_build_daily_aggregates.py`, `04_preprocess_uttarakhand.py`,
`04b_climate_signature.py`, `06_build_pcm_database.py`) or from the committed data artefacts.

## Primary data sources

### ERA5 (ECMWF Reanalysis v5)

- **Provider**: Copernicus Climate Data Store (CDS), accessed via `cdsapi`
- **Product**: `reanalysis-era5-single-levels`, hourly, `product_type: ["reanalysis"]`
- **Format requested**: `data_format: "netcdf"`, `download_format: "unarchived"`
- **Spatial extent**: the bounding envelope of `population_grid_points.csv`, padded **0.5°**
(`load_points_bbox(pad=0.5)`) �?" *not* the whole state
- **Temporal coverage**: 2016�?"2025 (10 full calendar years), all days
- **Hours requested**: computed dynamically from `suntimes.csv`, not fixed clock hours �?" three
circular (mod-24) windows around sunrise / solar noon / sunset, each padded `HOUR_MARGIN = 1`
- **Call structure**: 10 years �- 12 months �- 2 variable types = **240 API calls**
- **Output naming**: `data/raw/era5/points/era5_UK_points_{year}_{month}_{instant|accum}.nc`
- **Status tracking**: `data/raw/era5/download_status_points.csv` (fields: timestamp, year, month,
var_type, status, filepath, size_mb, note). Retry policy `MAX_RETRIES = 3`, `RETRY_WAIT = 60 s`.
A file under 50,000 bytes is treated as a corrupt download and removed.
- **Separation from an older archive**: `config.py` keeps `RAW_POINTS_DIR` /
`POINTS_DOWNLOAD_STATUS_FILE` distinct from `RAW_GRID_DIR` / `DOWNLOAD_STATUS_FILE`; the
docstring states the new pipeline "**does not touch** the old `data/raw/era5/grid/` archive".

**ERA5 variables �?" INSTANT group** (`INSTANT_VARS`, analysis fields):

| CDS variable | Short name | Unit as delivered | Converted to |
|---|---|---|---|
| `2m_temperature` | `t2m` | K | `era5_T_amb` (°C) |
| `2m_dewpoint_temperature` | `d2m` | K | `era5_T_dew` (°C), and `era5_RHum` via Magnus |
| `10m_u_component_of_wind` | `u10` | m/s | `era5_W_spd`, `era5_W_dir` |
| `10m_v_component_of_wind` | `v10` | m/s | `era5_W_spd`, `era5_W_dir` |
| `total_cloud_cover` | `tcc` | 0�?"1 fraction | `era5_cloud_cover` |
| `surface_pressure` | `sp` | Pa | `era5_P_atm` (hPa) |

**ERA5 variables �?" ACCUM group** (`ACCUM_VARS`, forecast fields):

| CDS variable | Short name | Unit as delivered | Converted to |
|---|---|---|---|
| `surface_solar_radiation_downwards` | `ssrd` | J/m² (accumulated) | `era5_GHI` (W/m²) |
| `mean_surface_direct_short_wave_radiation_flux` | `msdwswrf` | W/m² (mean rate) | `era5_DNI` (W/m²) |
| `surface_thermal_radiation_downwards` | `strd` | J/m² (accumulated) | `era5_LW_down` (W/m²) |
| `total_precipitation` | `tp` | m (accumulated) | `era5_precipitation` (mm) |

The accum request additionally downloads every target hour's **immediate predecessor**:
`ACCUM_HOURS = INSTANT_HOURS �^� {(h �^' 1) mod 24 for h in INSTANT_HOURS}`, because
`deaccumulate()` in `02_combine_uttarakhand.py` recovers hourly flux by `diff()`.

### NASA POWER

- **Provider**: NASA Langley Research Center, Prediction Of Worldwide Energy Resources
- **Endpoint**: `https://power.larc.nasa.gov/api/temporal/hourly/point` �?" **hourly** point data,
no API key required
- **Community**: `RE`
- **Time standard requested**: `UTC`
- **Parameters** (`POWER_PARAMETERS`): `ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M`
- **Coverage**: 2016�?"2025, one JSON per point per year �?' 45 �- 10 = **450 point-year caches**
- **Output naming**: `data/raw/nasapower/power_{point_id}_{year}.json`
- **Status tracking**: `data/raw/nasapower/download_status_power.csv`. `MAX_RETRIES = 3`,
`RETRY_WAIT = 20 s`, `REQUEST_SLEEP = 1.0 s` between successful calls, `REQUEST_TIMEOUT = 60 s`.
Files under 1,000 bytes are treated as corrupt.
- **Fill value handling**: `-999` is replaced with `NaN` in both `02_combine_uttarakhand.py` and
`02b_build_daily_aggregates.py`.
- **Dual role**: only 3 of the ~8,760 hours/year are consumed by `02`'s sun-event merge; the full
cache is re-read by `02b_build_daily_aggregates.py` to build true daily integrals.

**Stated limitation, in `01b`/`02b`/`04b` docstrings and `README.md`:** `POWER_PARAMETERS` does
**not** include precipitation (`PRECTOTCORR`). `monsoon_index` therefore remains an ERA5
3�-/day proxy and never receives a Tier-2 "true" version.

### Population raster

- **Source**: WorldPop unconstrained global mosaic, India, UN-adjusted, 100 m, **2020**
- **URL**: `https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_UNadj.tif`
- **Size**: ~1.5�?"2 GB, one-time, cached in `data/raw/population/`. Download auto-retries up to 5
attempts and resumes via HTTP `Range` requests.
- **Stated assumption** (`00a` docstring): "WorldPop doesn't publish a distinct India raster per
year at this resolution, so this pipeline uses a single static 2020 snapshot to weight sampling
locations across the whole 2016-2025 study period. That's a standard simplifying assumption �?�
not something this script tries to correct for."
- **Nodata handling**: `rio_mask(..., nodata=0, filled=True)` then `band[band < 0] = 0.0` �?"
WorldPop's negative nodata sentinels are zeroed.

### State boundary

- **Source**: GADM v4.1, India administrative level 1, GeoJSON
- **URL**: `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json`
- **Filter**: `NAME_1 == "Uttarakhand"`, `uk.geometry.iloc[0]` (first matching geometry)
- **Failure mode**: raises with the full list of available `NAME_1` values if not found

### PCM property data

- **Raw input**: `PCM_data/PCM_data/data/PCM_Properties_55records_42_70C_dense.csv` (55 records)
- **Cleaned output** consumed by the pipeline:
`PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` (55 rows �- 59 columns)
- **Cleaning method**: MICE + Random-Forest + Predictive Mean Matching
(`PCM_data/PCM_data/01_preprocess.py`, `N_ITER = 8`, `N_DONORS = 3`, `RANDOM_STATE = 42`)
- See `07_PHASE_5_AUDIT.md` for the full composition and imputation audit.

## Sampling grid

| Parameter | Value | Source |
|---|---|---|
| Grid resolution | `GRID_RES = 0.25°` | `00a_build_population_grid.py` |
| Grid origin | `ERA5_ORIGIN_LAT = 90.0`, `ERA5_ORIGIN_LON = -180.0` | `00a` |
| Coverage target | `COVERAGE_TARGET = 0.875` | `00a` |
| Points selected | **45** | observed: 45 markers/popups in `data/plots/comprehensive/maps/A2_population_map.html` |
| Point-ID format | `UKP_0001` �?� `UKP_0045` (contiguous, no gaps) | observed |
| Latitude range | 28.875° N �?" 30.625° N | observed from map marker coordinates |
| Longitude range | 77.875° E �?" 80.125° E | observed |
| Population covered | **10,475,711** (sum of the 45 cells' WorldPop 2020 values) | observed from map popups |
| Largest cell | `UKP_0001` = 1,061,041 | observed |
| Smallest cell | `UKP_0045` = 85,265 | observed |

The columns written are `point_id, lat, lon, population, weight`, with `weight` renormalised over
the selected subset only.

## Derived variables computed in `02_combine_uttarakhand.py`

| Output column | Derivation | Notes |
|---|---|---|
| `era5_T_amb` | `t2m �^' 273.15` | values `< �^'5` or `> 60` set to `NaN` in this script |
| `era5_T_dew` | `d2m �^' 273.15` | |
| `era5_RHum` | Magnus: `100·exp(17.625·Td/(243.04+Td)) / exp(17.625·T/(243.04+T))`, clipped 0�?"100 | |
| `era5_W_spd` | `�^s(u10² + v10²)` | m/s |
| `era5_W_dir` | `(degrees(atan2(u10, v10)) + 360) mod 360` | degrees |
| `era5_P_atm` | `sp / 100` | hPa |
| `era5_cloud_cover` | `tcc` passed through | 0�?"1 |
| `era5_GHI` | `deaccumulate(ssrd) / 3600`, clipped �?� 0 | `< 0 �?' 0`; `> 1400 �?' NaN` |
| `era5_LW_down` | `deaccumulate(strd) / 3600`, clipped �?� 0 | W/m² |
| `era5_precipitation` | `deaccumulate(tp) �- 1000`, clipped �?� 0 | mm |
| `era5_SZA` | pvlib `get_solarposition().zenith` | degrees |
| `era5_solar_azimuth` | pvlib `get_solarposition().azimuth` | degrees |
| `era5_GHI_clearsky` | pvlib `get_clearsky(model="ineichen").ghi` | W/m² |
| `era5_CSI` | `GHI / GHI_clearsky` clipped [0, 1.5]; forced 0 where `GHI_clearsky �?� 10` | |
| `era5_DNI` | `msdwswrf` clipped [0, 1400] (primary); else `GHI / cos(SZA)` fallback | see `04_PHASE_2_AUDIT.md` Part A.7 |
| `era5_DHI` | `(GHI �^' DNI·cos(SZA)).clip(0)` | closure residual, not measured |

`ETR` (extraterrestrial radiation) is computed by `compute_solar()` but is **not** in
`ERA5_OUTPUT_VARS`, so it is not written to the combined CSV.

## NASA POWER columns carried into the combined CSV

`POWER_VARS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]`, written with a
`power_` prefix: `power_ALLSKY_SFC_SW_DWN`, `power_CLRSKY_SFC_SW_DWN`, `power_T2M`, `power_RH2M`,
`power_WS10M`.

## Point metadata and calendar columns

Written per row by `process_point()`: `point_id`, `lat`, `lon`, `population`, `weight`, `date`,
`event`, `time_utc`, `grid_lat`, `grid_lon`, `month`, `DOY`, `year`, `season`, `season_code`.

## Season classification (`SEASON_MAP` in `02_combine_uttarakhand.py`)

| Months | Season | Code |
|---|---|---|
| Dec, Jan, Feb | **Winter** | 1 |
| Mar, Apr, May | **Summer** | 2 |
| Jun, Jul, Aug | **Monsoon** | 3 |
| Sep, Oct, Nov | **Retreat** | 4 |

Monsoon is **3 months (JJA)** in the season column. Note the inconsistency documented in
`03_PHASE_1_AUDIT.md`: `04b_climate_signature.py` computes `monsoon_index` over
**JJAS (Jun�?"Sep)**, which does not match `SEASON_MAP`'s JJA definition.

## Climate signature variables (`04b_climate_signature.py`)

### The 18 named indices

`INDEX_COLS` in `04b`, used for the correlation heatmap and distribution plots:

Ta_mean, Ta_p95, Ta_p05, DTR, GHI_daily_kWh, kt_mean, kt_std, SAI, CCI,
cloudy_frac, HDD18, CDD24, RH_mean, HSI, wind_mean, seasonality,
monsoon_index, elev_proxy

### Tier 1 �?" sun-event-only indices (computed in `build_signature_tier1`)

| Index | Derivation from the 3 sun-events/day |
|---|---|
| `Ta_mean_proxy` | mean of the daily mean of (sunrise, noon, sunset) `era5_T_amb` |
| `Ta_p95_proxy` / `Ta_p05_proxy` | 95th / 5th percentile of that daily mean |
| `DTR_proxy` | mean of `era5_T_amb_noon �^' era5_T_amb_sunrise` �?" **explicitly a proxy, not Tmax�^'Tmin** |
| `GHI_mean` | mean noon `era5_GHI` (W/m²) |
| `GHI_daily_kWh_proxy` | half-sine approximation `(2/�?) · GHI_noon(kW) · daylength_hours` |
| `kt_mean_proxy` / `kt_std_proxy` | mean / std of noon `era5_CSI` |
| `SAI_proxy` | `Σ era5_GHI / Σ era5_GHI_clearsky` over all rows |
| `cloudy_frac_proxy` | fraction of days with noon `CSI < KT_CLOUDY_THRESHOLD (0.35)` |
| `CCI_proxy` | longest consecutive run of cloudy days |
| `HDD18_proxy` / `CDD24_proxy` | `Σ max(0, 18 �^' Ta_daily)` / `Σ max(0, Ta_daily �^' 24)` |
| `RH_mean` | mean `era5_RHum` over all rows (**no Tier-2 override**) |
| `HSI` | `RH_mean �- fraction of rows with (T_amb �^' T_dew) < 3 K` (**no Tier-2 override**) |
| `wind_mean` | mean `era5_W_spd` (**no Tier-2 override**) |
| `seasonality_proxy` | `std / mean` of monthly-mean noon `era5_GHI` |
| `monsoon_index` | JJAS `era5_precipitation` sum ÷ total precipitation sum �?" **proxy only, permanently** |
| `elev_proxy` | `mean(era5_P_atm) / 1013.25` |

### Tier 2 �?" true daily-integral indices (from `02b_build_daily_aggregates.py`)

Written to `tier2_signature_uttarakhand.csv`, one row per `point_id`:

| Column | Derivation from the full NASA POWER hourly cache |
|---|---|
| `n_days_used` | days with �?� `MIN_HOURS_PER_DAY = 20` of 24 hours present |
| `GHI_daily_kWh_mean` | mean of daily `Σ ALLSKY_SFC_SW_DWN / 1000` |
| `kt_daily_mean` / `kt_daily_std` | daily `GHI/GHIcs` clipped [0, 1.5], guarded at `GHIcs > 0.05` |
| `SAI_true` | `Σ GHI_daily / Σ GHIcs_daily` |
| `cloudy_frac_true` | fraction of days with `kt_daily < 0.35` |
| `CCI_true` | longest consecutive cloudy-day run |
| `DTR_true_mean` | mean of daily `max(T2M) �^' min(T2M)` �?" **true diurnal range** |
| `Ta_mean_true`, `Ta_p95_true`, `Ta_p05_true` | mean / q95 / q05 of daily-mean `T2M` |
| `HDD18_true` / `CDD24_true` | degree-days from the true daily mean |
| `RH_mean_true`, `wind_mean_true` | mean of daily-mean `RH2M` / `WS10M` |
| `seasonality_true` | `std/mean` of monthly-mean daily GHI |

### Canonical-column rule (`CANON_MAP` in `04b`)

For each of `GHI_daily_kWh, DTR, kt_mean, kt_std, SAI, cloudy_frac, CCI, HDD18, CDD24, Ta_mean,
Ta_p95, Ta_p05, seasonality` the canonical column takes the **true Tier-2 value where present**
and falls back to the Tier-1 proxy otherwise. Both are kept side by side (`_proxy` / `_true`
suffixes) and both are **excluded from the clustering matrix** so only the canonical version
clusters.

`RH_mean`, `HSI`, `wind_mean`, `monsoon_index` and `elev_proxy` have **no** Tier-2 counterpart in
`CANON_MAP` and remain sun-event-derived. `wind_mean_true` and `RH_mean_true` are computed by
`02b` but are not mapped, so they are dropped from the clustering matrix by the `_true` suffix
rule.

### Derived PCM-facing quantities

T_DELIVERY_C  = 50.0
DT_APPROACH_C =  7.0
TM_TARGET_C   = T_DELIVERY_C + DT_APPROACH_C          # 57 °C, constant for every point

DRAW_RATE_KG_PER_S  = 60.0 / 1000 / 60                # = 0.001 kg/s
CP_WATER            = 4.186                           # kJ/kg·K
ASSUMED_PCM_MASS_KG = 50.0

sig["T_mains_est_C"]        = sig["Ta_mean"] - 2.0
q_night_kw                  = DRAW_RATE_KG_PER_S * CP_WATER * (T_DELIVERY_C - T_mains_est_C)
sig["L_required_kJ_per_kg"] = (q_night_kw * 3600 * 7) / ASSUMED_PCM_MASS_KG

Notes carried forward as caveats (see `05_PHASE_3_AUDIT.md`):
- The `�^' 2.0` K mains-temperature offset is **unsourced in-code**.
- There is **no `SHARE_PCM` factor** in this formula �?" the Uttarakhand `04b` sizes `L_required`
from a 7-hour draw at 0.001 kg/s against the full 50 kg PCM mass.
- `Tsoil_proxy_C = Ta_mean �^' 3.0` is defined only to feed the `int_wind_x_TaMinusTsoil`
interaction term and is dropped from the clustering matrix.

### 5 interaction terms

`int_GHI_x_ktstd`, `int_DTR_x_cloudyfrac`, `int_RH_x_TaMinusTm`, `int_wind_x_TaMinusTsoil`,
`int_CCI_x_1minusSAI`.

### PCA block

`PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]`,
`StandardScaler` then `PCA(n_components=0.95, random_state=42)`. Loadings written to
`pca_loadings.csv`. The number of retained components for the Uttarakhand run is **not available
in the source files** �?" `pca_loadings.csv` is under the git-ignored `data/processed/` tree.

## Physical bounds table (`BOUNDS` in `04_preprocess_uttarakhand.py`)

| Column | Lower | Upper |
|---|---|---|
| `era5_GHI` | 0 | 1400 W/m² |
| `era5_DNI` | 0 | 1400 W/m² |
| `era5_DHI` | 0 | 900 W/m² |
| `era5_GHI_clearsky` | 0 | 1400 W/m² |
| `era5_CSI` | 0 | 1.5 |
| `era5_LW_down` | **50** | **600** W/m² |
| `era5_T_amb` | �^'30 | 55 °C |
| `era5_T_dew` | �^'30 | 40 °C |
| `era5_RHum` | 0 | 100 % |
| `era5_W_spd` | 0 | 50 m/s |
| `era5_P_atm` | **850** | **1060** hPa |
| `era5_cloud_cover` | 0 | 1 |
| `era5_precipitation` | 0 | 200 mm |
| `era5_SZA` | 0 | 180 ° |
| `power_ALLSKY_SFC_SW_DWN` | 0 | 1400 W/m² |
| `power_CLRSKY_SFC_SW_DWN` | 0 | 1400 W/m² |
| `power_T2M` | �^'30 | 55 °C |
| `power_RH2M` | 0 | 100 % |
| `power_WS10M` | 0 | 50 m/s |

Out-of-range values become `NaN` (never silently clipped) and are then imputed by step 4.
The `era5_P_atm �?� 850 hPa` and `era5_LW_down �?� 50 W/m²` bounds are the two that bite hardest for
Uttarakhand �?" see `04_PHASE_2_AUDIT.md` Part B.

# Source File 3: 03_PHASE_1_AUDIT(3).md
Source path: /mnt/data/03_PHASE_1_AUDIT(3).md
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

loc = pvlib.location.Location(latitude=row.lat, longitude=row.lon, altitude=0, tz="UTC")
result = loc.get_sun_rise_set_transit(dates, method="spa")

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

GRID_RES = 0.25;  ERA5_ORIGIN_LAT = 90.0;  ERA5_ORIGIN_LON = -180.0

lon_cell_idx = floor((x − (−180)) / 0.25)     cell_lon = −180 + (lon_i + 0.5) · 0.25
lat_cell_idx = floor((90 − y)     / 0.25)     cell_lat =   90 − (lat_i + 0.5) · 0.25

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

**elevation note — this is a real limitation here, not a footnote:** `02_combine_uttarakhand.py`
uses a flat **1200m** proxy for every point's solar-geometry calculations, not real per-point
elevation. … Uttarakhand's populated terrain genuinely spans roughly 200m (Terai plains near
Udham Singh Nagar/Haridwar) to 2000m (hill towns), and elevation drives both solar-geometry
inputs (air mass, clear-sky irradiance) and the temperature-based indices (HDD18/CDD24, Ta_mean)
directly. This is plan v3.0's "Repair 2," written with Uttarakhand specifically in mind.

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

ist = df["time_utc"] + pd.Timedelta(hours=5, minutes=30)
df["ist_hour_decimal"] = ist.dt.hour + ist.dt.minute / 60 + ist.dt.second / 3600
df["solar_hour_angle"] = (df["ist_hour_decimal"] - 12) * 15

Uttarakhand spans ~77.9–80.1° E, so solar noon falls at roughly **06:40–06:50 UTC**. Any figure
shown to a general audience needs an explicit UTC→IST note at presentation time.

`solar_hour_angle` is derived from **IST clock time**, not true solar time — IST's 82.5° E
reference meridian is east of the whole state, so this column is a clock-hour-angle offset from
the true solar hour angle by roughly 9–18 minutes of longitude plus the equation of time. It is
used only as an engineered feature, never in a physics calculation.

### Sun-event times via pvlib SPA

`method="spa"` is pinned explicitly in `00b`. **Altitude 0 m** is used here, which differs from the
1200 m used for irradiance geometry in `02` — sunrise/sunset times are altitude-sensitive at the
minute scale, so the two assumptions are inconsistent, though the magnitude is small relative to
the ±1 h `HOUR_MARGIN` and the 3 h match tolerance.

(By contrast, `compute_solar()` in `02` calls `get_solarposition(times)` with **no** `method=`
argument — see `04_PHASE_2_AUDIT.md` Part A.6.)

### Cross-midnight UTC handling

Treated as a real case, not a hypothetical. From `00b`'s docstring:

**IMPORTANT — cross-midnight UTC dates are real, not a hypothetical edge case:** Uttarakhand's
sunrise can fall before 00:00 UTC (i.e. on the *previous* UTC calendar date) for eastern points
in summer — `time_utc` always reflects the true instant of the event; `date` is the nominal
(pvlib-assigned) calendar date the event belongs to.

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

"lag7" = the same sun-event 7 days earlier, not 7 hours earlier.

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

Dec, Jan, Feb  → Winter   (1)
Mar, Apr, May  → Summer   (2)
Jun, Jul, Aug  → Monsoon  (3)          ← JJA, three months
Sep, Oct, Nov  → Retreat  (4)

**But `04b_climate_signature.py` computes `monsoon_index` over JJAS — four months:**

jjas = precip[precip.index.month.isin([6, 7, 8, 9])].sum()
row["monsoon_index"] = jjas / total

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

# Source File 4: 04_PHASE_2_AUDIT(3).md
Source path: /mnt/data/04_PHASE_2_AUDIT(3).md
# 04 — Phase 2 Audit: Combine, Cross-Source Validation, and Quality Control

**Scripts**: `02_combine_uttarakhand.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`,
`03b_interactive_raw_qa.py`, `04_preprocess_uttarakhand.py`, `04c_postprocess_plots.py`,
`04c_interactive_postprocess_qc.py`

**Status**: **COMPLETE.** `climate_uttarakhand_points.csv` is confirmed at **493,155 rows**;
`uttarakhand_cleaned_physical.csv` at **489,105 rows × 89 columns** with zero residual missing
values.

This file contains everything Phase 2: the merge, the Tier-2 daily-integral repair, the ERA5
de-accumulation analysis, solar geometry and derived variables, the cross-source validation result,
and the 13-step quality-control sequence.

---

# PART A — Combine and Cross-Source Validation

## A.1 Purpose

Merge two independent climate products at the same 45 points and the same sun-event instants, then
repair the one thing a 3-samples-per-day schema cannot express: true daily integrals.

## A.2 Inputs

data/raw/era5/points/era5_UK_points_{year}_{month}_{instant,accum}.nc
data/raw/nasapower/power_{point_id}_{year}.json
data/processed/population_grid_points.csv
data/processed/suntimes.csv

## A.3 Processing

### ERA5 Accumulated Fields & De-accumulation — the load-bearing assumption

def deaccumulate(s):
    """
    ERA5 hourly reanalysis: accumulated values reset every 12 h.
    Resets happen at hours 1 and 13 UTC (start of each forecast run).
    diff() gives increments between consecutive downloaded hours; at reset
    hours the raw value is used directly since there's no valid predecessor.
    """
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    diff = s.diff()
    reset_mask = s.index.hour.isin([1, 13])
    diff[reset_mask] = s[reset_mask]
    return diff.clip(lower=0)

Applied to three fields:

df["GHI"]           = (deaccumulate(df["ssrd"]) / 3600).clip(0)
df["LW_down"]       = (deaccumulate(df["strd"]) / 3600).clip(0)
df["precipitation"] = (deaccumulate(df["tp"])   * 1000).clip(0)

**The stated model.** The docstrings in `01_download_era5_uttarakhand.py` and in `deaccumulate()`
assume the MARS convention: `ssrd` is cumulative since the last forecast reset (00 Z or 12 Z), so
the true one-hour flux is `value(h) − value(h−1)`, except at h ∈ {1, 13} where the raw value *is*
the first hour of a new cycle. The special case is argued as mathematically required, not an
optimisation: "hour 13's predecessor (hour 12) belongs to a *different* 12-hour accumulation cycle,
so diffing against it would produce garbage."

`avg_sdirswrf` **bypasses `deaccumulate()` entirely** — only `clip(0)` — consistent with it being a
mean-rate field.

**This assumption is not verified anywhere in `era5-uttarakhand/`, and three independent committed
artefacts indicate the resulting `era5_GHI` is far below physical expectation.**

#### Evidence 1 — raw event profile, before any cleaning

`data/plots/raw/B_event_profile.png` (from `03_plots_raw.py`, run directly on the merged CSV):

| Sun event | Mean `era5_GHI` (W/m²) | Mean `era5_T_amb` (°C) |
|---|---|---|
| sunrise | ≈ 1 | ≈ 15.3 |
| **noon** | **≈ 61** | ≈ 22.8 |
| sunset | ≈ 19 | ≈ 21.7 |

The **timezone check passes** — noon is the peak, which is exactly what check B exists to verify.
But a mean solar-noon GHI of ≈ 61 W/m² at 28.9–30.6 °N is roughly an order of magnitude below any
clear-sky-plus-cloud climatology.

#### Evidence 2 — cross-source disagreement, with two clean controls

`data/plots/raw/C_era5_vs_power_stats.csv`, computed over every row:

| Variable | n | MBE (ERA5 − POWER) | RMSE | Pearson *r* |
|---|---|---|---|---|
| **GHI (W/m²)** | 493,155 | **−211.406** | **369.323** | **0.4321** |
| Clear-sky GHI (W/m²) | 493,155 | +5.314 | 66.360 | **0.9923** |
| T_amb (°C) | 492,936 | −0.089 | 3.695 | 0.9020 |
| RHum (%) | 493,155 | +11.383 | 20.362 | 0.7399 |
| Wind speed (m/s) | 493,155 | −1.141 | 1.703 | 0.5396 |

The diagnostic value is in the **contrast between rows 1 and 2**. `era5_GHI_clearsky` is not an
ERA5 field at all — it is pvlib's Ineichen model evaluated locally at the same coordinates, the
same instants and `altitude = 1200 m` — and it agrees with NASA POWER's independently modelled
clear-sky product to within **+5.3 W/m² at r = 0.9923**. The coordinates, the sun-event time
matching, the nearest-hour lookup, the ERA5 grid snapping and the altitude assumption's effect on
the clear-sky model are therefore **all confirmed correct**. `era5_T_amb`, an instantaneous field
that never touches `deaccumulate()`, agrees to within 0.09 °C at r = 0.902.

Only the **all-sky ERA5 GHI** — the one solar quantity that passes through `deaccumulate()` —
disagrees, and it disagrees by −211 W/m² at r = 0.432. MBE −211.4 W/m² is ten times ERA5's own
whole-file mean of 21.03 W/m². An r of 0.432 says the two series differ in *shape*, not merely by
an offset.

#### Evidence 3 — downstream magnitudes

- Cleaned whole-file `era5_GHI`: mean **21.03 W/m²**, std 37.94, max **702.74 W/m²**.
- Per-cluster noon `GHI_mean`: **≈ 44.5 – 55.1 W/m²**.
- Per-cluster `GHI_daily_kWh_proxy` (half-sine daily integral from noon GHI):
**≈ 0.33 – 0.43 kWh/m²/day**, against a physically expected several kWh/m²/day.

#### Evidence 4 — a second fingerprint on the other de-accumulated field

`era5_LW_down` = `deaccumulate(strd)/3600`. `04`'s physical bound is `[50, 600]` W/m², and
`data/plots/post_preprocess/C_qc_flag_counts.csv` records **363,525 values (73.7 % of all rows)**
below it. A 50 W/m² floor is far below any plausible surface downwelling longwave value — clear
cold nights are still ~150–250 W/m² — so values falling under it is itself evidence that this
column is depressed in the same way `era5_GHI` is.

#### What can and cannot be concluded

**Can be concluded from `era5-uttarakhand/` alone:**
- `era5_GHI` disagrees with NASA POWER by MBE −211.4 W/m² at r = 0.432, while every
non-accumulated and locally-computed field agrees well.
- The anomaly is present in the **raw** merged data, so it originates in
`02_combine_uttarakhand.py` or upstream — **not** in `04_preprocess_uttarakhand.py`.
- The only transformation applied to `ssrd` and `strd` but not to the agreeing fields is
`deaccumulate()`.
- The pipeline **detected** the disagreement and **never acted on it**.

**Cannot be concluded:** the exact mechanism. Determining whether the CDS request configuration
used here returns cumulative-since-reset values (in which case `diff()` is right) or per-hour
accumulations (in which case `diff()` destroys most of the signal) requires opening one of the
`data/raw/era5/points/*_accum.nc` files. **Those are git-ignored and not in this repository**, so
that check could not be performed as part of this audit.

**Recommended verification, one command, no re-download.** Open any local
`era5_UK_points_2020_06_accum.nc`, extract `ssrd` for a single grid node across the downloaded
hours of one day, and check whether consecutive values **increase monotonically within each 12-hour
window** (→ cumulative, `diff()` correct) or whether each value is independently of order
10⁵–10⁶ J/m² (→ per-hour accumulation, `diff()` wrong, and the fix is a stateless non-negative clip
with no differencing). Compare the resulting W/m² against `power_ALLSKY_SFC_SW_DWN` for the same
instant.

### `02_combine_uttarakhand.py` — the merge/physics script

Four steps, from the docstring:

1. Per point: nearest-neighbour snap to the ERA5 grid, concatenate the full instant+accum hourly
series across all years, de-accumulate, compute solar geometry.
2. For each `(point_id, date, event)` row in `suntimes.csv`, pick the ERA5 hourly value nearest in
time to that event's exact UTC timestamp.
3. Same nearest-hour lookup against that point's cached NASA POWER hourly series.
4. Merge into one row per point/date/event and stream-write the output CSV.

Configuration:

DEFAULT_ALT_M   = 1200    # "Uttarakhand is mountainous; populated zones range roughly 200-2000m"
MAX_MATCH_HOURS = 3       # reject a nearest-hour match farther than 3 h from the event

**NetCDF handling.** `open_nc()` tries engines `netcdf4` → `scipy` → `h5netcdf`, then falls back to
`mask_and_scale=False, decode_cf=False, decode_times=False` ("Python 3.14 safe"). `safe_values()`
re-applies CF `scale_factor` / `add_offset` / `_FillValue` manually when that fallback was used.
`decode_time()` handles both `valid_time` and `time` coordinates.

**Unit conversions:**

| ERA5 field | Raw unit | Operation | Output |
|---|---|---|---|
| `ssrd` | J/m² (accum) | `deaccumulate() / 3600` | `era5_GHI` (W/m²) |
| `strd` | J/m² (accum) | `deaccumulate() / 3600` | `era5_LW_down` (W/m²) |
| `tp` | m (accum) | `deaccumulate() × 1000` | `era5_precipitation` (mm) |
| `t2m`, `d2m` | K | `− 273.15` | `era5_T_amb`, `era5_T_dew` (°C) |
| `t2m` + `d2m` | — | Magnus, clipped 0–100 | `era5_RHum` (%) |
| `u10`, `v10` | m/s | `√(u²+v²)`, `(deg(atan2(u,v))+360) mod 360` | `era5_W_spd`, `era5_W_dir` |
| `sp` | Pa | `/ 100` | `era5_P_atm` (hPa) |
| `tcc` | 0–1 | pass-through | `era5_cloud_cover` |
| `msdwswrf`/`fdir`/`msdrswrf` | see A.7 | `clip(0)` only | `avg_sdirswrf` → `era5_DNI` |

**In-script bounds applied before Phase 2 QC ever sees the data:**

| Rule | Note |
|---|---|
| `GHI < 0 → 0` | redundant with `deaccumulate`'s own clip |
| `GHI > 1400 → NaN` | never fires — observed max is 702.74 |
| `T_amb < −5 → NaN`, `T_amb > 60 → NaN` | **narrower than `04`'s `BOUNDS` (−30…55 °C)** |
| `RHum.clip(0, 100)` | silent clip |

The `T_amb < −5 °C` cut is Uttarakhand-relevant: sub-−5 °C high-altitude winter sunrise
temperatures are physically real for this state and are discarded at the merge step, **before any
QC accounting sees them**. The cleaned file's `era5_T_amb` minimum is exactly **−5.00 °C** — the
fingerprint of this rule.

**Temporal matching.** `nearest_row()` rejects any match farther than 3 h and is applied
independently to each source; a row is written if *either* matched. **The actually-matched
timestamp is not persisted** — only the requested `time_utc` — so per-row match quality is
unauditable from the output.

### `02b_build_daily_aggregates.py` — Tier-2 daily integrals (NASA POWER only)

The docstring is explicit that this is not optional polish:

`climate_uttarakhand_points.csv` keeps only 3 rows/day … Several signature indices genuinely
cannot be computed from three instantaneous samples: the true daily GHI energy integral, the
true diurnal temperature range (Tmax-Tmin, not noon-sunrise), heating/cooling degree-days from a
true daily mean, cloudy-day fraction, and the longest consecutive-cloudy-day run.

`README_PREPROCESSING.md` calls it "the single most important gap identified in plan v3.0 (Section
4.3, 'the repair that cannot be skipped')."

**Cost: zero new downloads.** It re-reads the full hourly POWER cache `01b` already wrote.

KT_CLOUDY_THRESHOLD = 0.35     # same threshold 04b uses for CCI/cloudy_frac
MIN_HOURS_PER_DAY   = 20       # else the day is dropped, not averaged short

Daily outputs: `GHI_daily_kWh`, `GHIcs_daily_kWh`, `kt_daily` (clipped [0, 1.5], guarded at
`GHIcs > 0.05`), `Ta_mean_true`/`Ta_max_true`/`Ta_min_true`, **`DTR_true` = true Tmax − Tmin**,
`RH_mean_true`, `wind_mean_true`.

Point-level Tier-2 outputs: `n_days_used`, `GHI_daily_kWh_mean`, `kt_daily_mean`, `kt_daily_std`,
`SAI_true`, `cloudy_frac_true`, `CCI_true`, `DTR_true_mean`, `Ta_mean_true`, `Ta_p95_true`,
`Ta_p05_true`, `HDD18_true`, `CDD24_true`, `RH_mean_true`, `wind_mean_true`, `seasonality_true`.

Two details for a methodology write-up:

- **`Ta_p95_true` / `Ta_p05_true` are percentiles of the daily *mean* temperature**, not of the
true daily maxima/minima — even though `Ta_max_true`/`Ta_min_true` are already computed in the
daily table. They are therefore not "design-day extremes" in the usual sense.
- **`CCI_true` is a run length in days**, not an index on [0, 1] — the longest consecutive cloudy
run, via a shift-cumsum run-ID and `transform("sum")`.

**Recorded run result**, quoted in `README_PREPROCESSING.md` from the author's own terminal output:

**Confirmed run**: `Points: 45`, all 45 processed with 0 skipped, `usable_days=3653` for every
sampled point shown in the log, `164,385` total point-days aggregated.

45 × 3,653 = 164,385 exactly. The ≥ 20-of-24-hours threshold excluded **no** days — "a good sign —
it means the … threshold this script uses wasn't a real bottleneck for your NASA POWER data."

**Stated limitation**: `01b`'s `POWER_PARAMETERS` never included `PRECTOTCORR`, so `monsoon_index`
is **not** upgraded to a true Tier-2 index and remains an ERA5 3×/day precipitation-fraction proxy
permanently. The docstring gives the one-line fix and flags it as "optional, not required for
Objective 1 to stand up." `NEXT_STEPS.md` explicitly instructs *not* to fix it now.

### `03_plots_raw.py` / `03b_interactive_raw_qa.py` — raw QA, before any cleaning

Read-only, run directly on `02`'s output. Six checks, mapped to plan Table 9:

| Check | Purpose | Output |
|---|---|---|
| A | Point map — is the sample actually population-weighted and covering the state? | `A_point_map.png` / `.html` |
| B | Event profile — **Table 9 check #2 (timezone)**: GHI/T_amb must peak at "noon" | `B_event_profile.png` / `.html` |
| C | ERA5 vs NASA POWER — **Table 9 check #7**: MBE/RMSE/*r* per variable | `C_era5_vs_power.png` / `.html` **+ `C_era5_vs_power_stats.csv`** |
| D | Missing-data heatmap per point × variable | `D_missing_heatmap.png` / `.html` |
| E | Seasonal boxplots against known climatology | `E_seasonal_boxplots.png` / `.html` |
| F | Multi-year trend — a step-change in one year would flag a download/unit bug | `F_yearly_trend.png` / `.html` |

All twelve outputs are committed. `C_era5_vs_power_stats.csv` is committed in both the static and
interactive variants with identical values, confirming both scripts ran on the same data.

## A.4 Code mapping

| Concern | Function / constant | File |
|---|---|---|
| NetCDF opening with engine fallbacks | `open_nc()`, `safe_values()`, `decode_time()` | `02` |
| Spatial snapping | `extract_nearest()` | `02` |
| Kelvin → °C | `kelvin_to_c()` | `02` |
| Relative humidity | `compute_rh()` (Magnus, a = 17.625, b = 243.04) | `02` |
| Accumulated → flux | `deaccumulate()` | `02` |
| Solar geometry | `compute_solar()` | `02` |
| Unit conversions + in-script bounds | `apply_unit_conversions()` | `02` |
| POWER cache load, `-999 → NaN` | `load_power_series()` | `02` |
| Nearest-in-time match | `nearest_row()`, `MAX_MATCH_HOURS = 3` | `02` |
| Per-point orchestration | `process_point_era5()`, `process_point()` | `02` |
| Daily integrals | `daily_from_hourly()` | `02b` |
| Point-level Tier 2 | `build_tier2_row()` | `02b` |

## A.5 Temporal Processing in the Merge

`nearest_row()` is applied independently to the ERA5 series and the POWER series with a 3-hour
rejection window. **Observed: zero rows lost** — the output has exactly 493,155 rows against a
theoretical maximum of 45 × 3,653 × 3 = 493,155. No `(point, date, event)` combination failed to
find both an ERA5 and a POWER reading within 3 hours across all 45 points and all 3,653 days.

That is a genuinely good coverage result. It also means the tolerance was never exercised as a
filter, so it provides no evidence about *typical* match quality — and because the matched
timestamp is not written, the offsets cannot be recovered. **A low-cost fix**: persist
`era5_matched_time_utc` and `power_matched_time_utc` alongside the requested `time_utc`.

Rows with a valid `era5_T_amb` are 492,936, i.e. **219 rows (0.044 %)** lost `era5_T_amb` to the
`< −5 °C` / `> 60 °C` cut or to a missing match.

Duplicate handling: `df[~df.index.duplicated(keep="first")]` is applied to the instant frame, the
accum frame, the joined frame, and the POWER frame — four separate de-duplications before any row
is written.

## A.6 Solar Geometry (why it's computed this way)

def compute_solar(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt, tz="UTC")
    sp = loc.get_solarposition(times)                    # no explicit method=
    cs = loc.get_clearsky(times, model="ineichen")
    df["SZA"], df["solar_azimuth"] = sp["zenith"], sp["azimuth"]
    df["ETR"]          = pvlib.irradiance.get_extra_radiation(times)
    df["GHI_clearsky"] = cs["ghi"]
    ...

**Solar-position method is not pinned.** `get_solarposition(times)` relies on the installed pvlib
version's default, whereas `00b_build_suntimes.py` *does* pin `method="spa"`. This is an
inconsistency inside one pipeline and a reproducibility gap: a pvlib version change could shift
`SZA`, `solar_azimuth`, `is_daytime`, and the `SZA ≥ 90` night-masking threshold in `04`'s step 2.
Pinning `method="spa"` in `compute_solar()` closes it at zero analytical cost.

**Clear-sky model: Ineichen with pvlib's default Linke-turbidity climatology**, no site-specific
turbidity. This choice is **independently validated by the pipeline's own statistics** — see the
+5.3 W/m² / r = 0.9923 result in A.3, which is the pipeline's strongest positive finding.

*Uttarakhand caveat:* the default Linke climatology is a coarse global lookup, and this state's
aerosol environment is strongly elevation-dependent — Indo-Gangetic-plain haze in the foothills
versus clean air above the boundary layer — while one 1200 m altitude and one climatological
turbidity are applied to all 45 points. The r = 0.9923 agreement is against another *model*, so it
confirms mutual consistency rather than absolute accuracy.

**Altitude: 1200 m for all 45 points**, feeding the Ineichen air-mass/turbidity correction. A Terai
point at ~200 m and a hill point at ~2000 m receive identical clear-sky curves. The value is **not
written to the output rows**, so the assumption is invisible in the data.

**Night-time handling and division-by-zero protection:**

| Guard | Rule |
|---|---|
| Clear-sky floor | `CSI = 0` wherever `GHI_clearsky ≤ 10 W/m²` |
| CSI ceiling | `clip(0, 1.5)` |
| Zenith clip for DNI division | `cos_z = cos(radians(SZA.clip(0, 89.9)))` — **only** for the DNI/DHI arithmetic; the unclipped `SZA` is what is written |
| Fallback DNI guard | `np.where(cos_z > 0.05, GHI/cos_z, 0)` |
| Night masking (in `04`) | all five solar fields forced to `0.0` where `era5_SZA ≥ 90°` |

A `CSI` of exactly 0 is **three-way ambiguous** in the output: true darkness, clear-sky below the
10 W/m² floor, or night-masked in `04`.

`04`'s night-masking rationale is schema-specific and well reasoned: "even though every row IS a
sun-event, the NEAREST-HOUR match … can land a few hours off true sun position (see 02_combine's
`MAX_MATCH_HOURS=3`), so a 'sunrise' row can occasionally have SZA > 90."

**`ETR` is computed and discarded** — it is not in `ERA5_OUTPUT_VARS`, so it never reaches the
combined CSV and no downstream script uses it.

**Latitude context** (arithmetic, not stated in the source files): the 45 points span
28.875–30.625 °N, so solar-noon zenith ranges from ~6–8° at the June solstice to ~52–54° at the
December solstice, and day length swings from ~10 h to ~14 h. `04b`'s half-sine daily-integral
proxy uses the actual `sunset − sunrise` interval rather than a nominal 12 h, which is the right
choice for a swing that large.

## A.7 Solar-Derived Variables (construction & assumptions)

### GHI

`GHI = deaccumulate(ssrd)/3600`, clipped ≥ 0, `> 1400 → NaN`. The pipeline's most consequential
derived variable — it feeds `CSI`, `DHI`, `cloud_opacity` and every Tier-1 solar signature index.
Observed magnitudes and the anomaly analysis are in A.3.

The `> 1400 → NaN` guard never fires: the observed maximum across 493,155 rows is 702.74 W/m².

### DNI — two-branch derivation

if "avg_sdirswrf" in df.columns:
    df["DNI"] = df["avg_sdirswrf"].clip(0, 1400)                              # primary
else:
    df["DNI"] = np.where(cos_z > 0.05, df["GHI"] / cos_z, 0).clip(0, 1400)    # fallback

**Branch 1 (primary) — and its unit-consistency caveat.** `avg_sdirswrf` is set upstream by a
three-name matcher:

fdir_col = next((c for c in df.columns if c in ("msdwswrf", "fdir", "msdrswrf")), None)
if fdir_col:
    df["avg_sdirswrf"] = df[fdir_col].astype(float).clip(0)

| Short name | ERA5 convention | Correct handling | What the code does |
|---|---|---|---|
| `msdwswrf` | mean-rate, W/m² | pass through | `clip(0)` ✓ |
| `msdrswrf` | mean-rate, W/m² | pass through | `clip(0)` ✓ |
| `fdir` | accumulated, J/m² | `/3600` | `clip(0)` ✗ — **would over-estimate by 3600×** |

`01_download_era5_uttarakhand.py` requests `mean_surface_direct_short_wave_radiation_flux`, which
maps to `msdwswrf` — a mean-rate field — so the no-conversion branch is almost certainly correct in
practice. **This audit could not verify the actual short name present**, because the NetCDF files
are git-ignored. Before presenting DNI as unit-validated, open one `*_accum.nc` and inspect
`ds.data_vars`.

**Branch 2 (fallback) — explicitly NOT a decomposition model.** `DNI = GHI / cos(SZA)` assumes a
zero diffuse component; it attributes all global horizontal irradiance to the beam. In cloudy
conditions it would over-estimate DNI substantially, and near the horizon it is numerically
unstable (hence the `cos_z > 0.05` guard). It should only execute if the ERA5 direct field is
absent, which `01` requests in every accum call — but **the pipeline does not record which branch
ran**, so this cannot be confirmed from the outputs.

Correct framing for a write-up: "DNI taken from ERA5's mean direct short-wave radiation flux where
available; a `GHI/cos(SZA)` closure fallback exists but is not expected to have been used" — not a
claim of decomposition-model provenance.

### DHI — closure residual

df["DHI"] = (df["GHI"] - df["DNI"] * cos_z).clip(0)

**Not independently derived.** The closure equation `GHI = DNI·cos(SZA) + DHI` is satisfied *by
construction*, so agreement with it is evidence of nothing, and any error in GHI or DNI propagates
entirely into DHI. Because GHI is anomalously low while DNI comes from a separate un-deaccumulated
field, the residual will frequently be negative and clipped to zero. `04`'s `BOUNDS` records **no**
`era5_DHI` flags, consistent with the column being dominated by clipped zeros.

**`DHI` is not used by `04b_climate_signature.py`.** Its only appearance downstream is in `04`'s
correlation and VIF reports. It should not be presented as a measured or modelled diffuse quantity.

### Clearness index (CSI / kt)

`CSI = GHI / GHI_clearsky` clipped [0, 1.5], forced to 0 below a 10 W/m² clear-sky floor and again
where `SZA ≥ 90°`.

Because GHI is anomalously low while `GHI_clearsky` is validated as correct, **`era5_CSI` is
correspondingly depressed**, and with it `kt_mean_proxy`, `kt_std_proxy`, `cloudy_frac_proxy`,
`CCI_proxy`, `SAI_proxy` and `era5_cloud_opacity`.

**The two-tier design contains this.** The canonical `kt_mean`, `kt_std`, `SAI`, `cloudy_frac`,
`CCI` and `GHI_daily_kWh` columns that reach the clustering matrix come from **NASA POWER via
`02b`**, and all the `_proxy` variants are excluded from the clustering matrix by `04b`'s suffix
rule. This is a real architectural benefit of the Tier-2 repair and should be reported as such.

**The one exception is `GHI_mean`** (mean noon `era5_GHI`), which carries no `_proxy` suffix and has
no Tier-2 override, so it enters the clustering matrix carrying the anomaly.

### Cloud cover, precipitation, longwave

| Variable | Derivation | Bound in `04` | Flags |
|---|---|---|---|
| `era5_cloud_cover` | `tcc` pass-through | [0, 1] | 0 |
| `era5_precipitation` | `deaccumulate(tp) × 1000` mm | [0, 200] | 0 |
| `era5_LW_down` | `deaccumulate(strd)/3600` | **[50, 600]** | **363,525 (73.7 %)** |

Cleaned `era5_precipitation`: mean **0.08 mm**, std 0.44, max **45.85 mm** — per-instant values at
three sun-event samples per day, not daily totals. It also passes through `deaccumulate()`, so it
carries the same unverified assumption; its only downstream use is `monsoon_index`, a **ratio** in
which a uniform multiplicative error would cancel (a non-uniform one would not).

`era5_LW_down` is not used by `04b`; its impact is limited to the correlation/VIF reports. It is
retained here as the second independent fingerprint of the de-accumulation problem.

### Physical bounds applied to derived solar variables

| Variable | Bound | Where | Flags observed |
|---|---|---|---|
| `GHI` | `<0 → 0`; `>1400 → NaN` | `02` | not counted |
| `GHI`, `DNI`, `GHI_clearsky` | `[0, 1400]` | `04` | 0 each |
| `DHI` | `[0, 900]` | `04` | 0 |
| `CSI` | `[0, 1.5]` | `04` | 0 |
| `LW_down` | `[50, 600]` | `04` | **363,525** |
| all five solar fields | forced `0.0` where `SZA ≥ 90°` | `04` | logged to `qc_report.txt`, not committed |

Not one of the five directly-solar columns triggered a physical-bounds flag. Read with the observed
magnitudes, that says the values sit comfortably *inside* their ranges because they are too small,
not because they are correct.

## A.8 Cross-Source Validation Decision — there isn't one

| Component | Status in `era5-uttarakhand/` |
|---|---|
| Cross-source statistics computed | **Yes** — `03` check C and its interactive twin |
| Statistics persisted | **Yes** — `C_era5_vs_power_stats.csv`, committed in both variants |
| Dedicated agreement-analysis script | **No.** No `03b_agreement_analysis*.py` of any name exists. |
| Bias decision file | **No.** No file records a BACKBONE / quantile-map decision. |
| Threshold-based decision logic | **No.** |
| Bias-correction / quantile-mapping step in `04` | **No.** The 13-step sequence contains no such step. |

**What the pipeline says it will do**, in three separate places:

`03_plots_raw.py`'s docstring: "quantifies exactly how much the two sources disagree, per variable,
**before you decide how (or whether) to bias-correct in 04.**"

`README.md`'s run-order block: "STOP AND LOOK at 03's output before continuing. … **check C: large
ERA5-vs-POWER MBE is expected and gets addressed in 04**."

`README_PREPROCESSING.md`: "If B shows noon isn't the peak, or C shows a large systematic MBE,
**stop and fix that before running `04`** — these are exactly the 'most silent failures at this
stage' the plan doc warns about."

**Check C shows a large systematic MBE. Nothing in `04` addresses it. The gate the source files
describe was not enforced.**

### Variable pairs compared

| ERA5 column | NASA POWER column |
|---|---|
| `era5_GHI` | `power_ALLSKY_SFC_SW_DWN` |
| `era5_GHI_clearsky` | `power_CLRSKY_SFC_SW_DWN` |
| `era5_T_amb` | `power_T2M` |
| `era5_RHum` | `power_RH2M` |
| `era5_W_spd` | `power_WS10M` |

Statistics are pooled over all rows — **no stratification by season, event, point or year**. There
is therefore no evidence about whether the GHI disagreement is uniform across the year or
concentrated in particular months. (Check F, the multi-year trend plot, is committed as a figure
but the statistics CSV carries no per-year breakdown.)

### Reading the other three rows

**RHum (+11.4 %, r = 0.740).** ERA5's relative humidity here is *derived* — the Magnus formula on
`t2m` and `d2m` — while POWER's `RH2M` is its own product. An 11-point offset between the two is a
plausible model-versus-model disagreement rather than a processing artefact, but it is
**unaddressed and reaches the clustering matrix**: `04b` takes `RH_mean` from the ERA5 side (there
is no `CANON_MAP` entry for it), and `RH_mean` is a `PCA_BLOCK` member and the basis of `HSI`.
`02b` computes an unused `RH_mean_true` from POWER.

**Wind (−1.14 m/s, r = 0.540).** Both products are nominally 10 m winds, so the heights match; the
disagreement reflects differing surface-roughness and orographic treatments over complex terrain,
which is exactly where 45 Himalayan-foothill points would show it. The mean disagreement is ~80 % of
the cleaned ERA5 mean of 1.43 m/s. `wind_mean` reaches the clustering matrix from the ERA5 side, and
`02b`'s `wind_mean_true` is likewise unused.

### Which side each clustering-matrix column comes from

| Signature column | Source | Affected by a measured disagreement? |
|---|---|---|
| `GHI_daily_kWh`, `kt_mean`, `kt_std`, `SAI`, `cloudy_frac`, `CCI` | NASA POWER (Tier 2) | No |
| `DTR`, `Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `seasonality` | NASA POWER (Tier 2) | No |
| `GHI_mean` | ERA5 noon GHI, no override | **Yes — the −211 W/m² problem** |
| `RH_mean` | ERA5 (Magnus-derived), no override | **Yes — +11.4 %** |
| `HSI` | ERA5 (`RH_mean` × dew-point-depression fraction) | **Yes** |
| `wind_mean` | ERA5, no override | **Yes — −1.14 m/s** |
| `monsoon_index` | ERA5 precipitation ratio, permanently proxy | Unquantified (no POWER precipitation) |
| `elev_proxy` | ERA5 `P_atm`, 37 % imputed | Not compared (POWER pressure not downloaded) |

**A concrete, cheap improvement:** adding `RH_mean` and `wind_mean` to `04b`'s `CANON_MAP` would
swap two ERA5-side columns for already-computed NASA POWER Tier-2 values at the cost of two
dictionary entries.

## A.9 Mathematical operations

Magnus RH · vector wind magnitude/direction · first-difference de-accumulation with 12-hourly reset
handling · pvlib SPA solar position · Ineichen clear-sky · clearness-index ratio with a low-light
floor · beam/diffuse closure · nearest-neighbour 1-D `argmin` snapping · nearest-in-time index
lookup with a tolerance · daily summation and min/max/mean aggregation · degree-day accumulation ·
run-length encoding for consecutive cloudy days · coefficient of variation for seasonality.

## A.10 Literature support

**None present in the source files for Phase 2.** `02_combine_uttarakhand.py` names `pvlib` and the
string `"ineichen"` but cites no paper; there is no ERA5 product citation, no NASA POWER citation,
no SPA citation, no clear-sky-model citation, and no decomposition-model reference anywhere in
`era5-uttarakhand/`. See `11_LITERATURE_MAPPING.md`.

## A.11 Validation

| Check | Result |
|---|---|
| Noon peaks GHI and T_amb (timezone) | **PASS** — 61 vs 1 vs 19 W/m²; 22.8 vs 15.3 vs 21.7 °C |
| Full point/day/event coverage | **PASS** — 493,155 = 45 × 3,653 × 3 exactly |
| Clear-sky cross-source agreement | **PASS** — MBE +5.3 W/m², r = 0.9923 |
| Temperature cross-source agreement | **PASS** — MBE −0.089 °C, r = 0.902 |
| All-sky GHI cross-source agreement | **FAIL** — MBE −211.4 W/m², r = 0.432, unaddressed |
| Humidity cross-source agreement | **MARGINAL** — MBE +11.4 %, r = 0.740, unaddressed |
| Wind cross-source agreement | **MARGINAL** — MBE −1.14 m/s, r = 0.540, unaddressed |
| Tier-2 daily coverage | **PASS** — 45/45 points, 0 skipped, 164,385 point-days |

## A.12 Outputs

| File | Rows | Committed? |
|---|---|---|
| `data/processed/climate_uttarakhand_points.csv` | **493,155** × 36 cols | No (git-ignored) |
| `data/processed/daily_aggregates_uttarakhand.csv` | ≤ 164,385 | No |
| `data/processed/tier2_signature_uttarakhand.csv` | ≤ 45 | No |
| `data/plots/raw/*.png` + `C_era5_vs_power_stats.csv` | 6 + 1 | **Yes** |
| `data/plots/raw_interactive/*.html` + `C_era5_vs_power_stats.csv` | 6 + 1 | **Yes** |

## A.13 Dependencies

`xarray`, `netCDF4` (and optionally `scipy`/`h5netcdf` as engine fallbacks), `pvlib`, `pandas`,
`numpy`; `matplotlib` + `seaborn` for `03`; `plotly` + `folium` + `branca` for `03b`.

---

# PART B — Phase 2: Preprocessing & Quality Control

**Script**: `04_preprocess_uttarakhand.py` (13 steps), with post-hoc QA by
`04c_postprocess_plots.py` and `04c_interactive_postprocess_qc.py`.

## B.1 Purpose and the schema note it leads with

The old `04_preprocess_uttarakhand.py` assumed one row per `(city, hour)` — a continuous 24 h/day
series. This dataset is one row per `(point_id, date, event)` with `event ∈ {sunrise, noon,
sunset}` — **3 samples/day, not 24**. Every "rolling"/"lag"/"delta" concept below is therefore
redefined over EVENT OCCURRENCES within a `(point_id, event)` group, sorted by date — e.g.
"lag7" = the same sun-event 7 days earlier, not 7 hours earlier.

This is the single most important fact to carry into any methodology write-up about this script.

## B.2 Steps 1–3b — inspection, physical validation, outlier flagging

**Step 1 — Dataset inspection.** Shape, dtype counts, duplicate `(point_id, date, event)` count
(dropping any), and the top-15 missing-% columns. Everything is appended to `report_lines` and
written to `qc_report.txt`.

**Step 2 — Physical validation.** Out-of-range values become **`NaN`, never clipped** — the in-code
comment: "matches your 'safer than clipping' rule."

| Column | Lower | Upper | Values flagged (observed) |
|---|---|---|---|
| `era5_LW_down` | **50 W/m²** | 600 W/m² | **363,525 (73.7 %)** |
| `era5_P_atm` | **850 hPa** | 1060 hPa | **182,899 (37.1 %)** |
| `era5_GHI`, `era5_DNI`, `era5_GHI_clearsky` | 0 | 1400 W/m² | 0 |
| `era5_DHI` | 0 | 900 W/m² | 0 |
| `era5_CSI` | 0 | 1.5 | 0 |
| `era5_T_amb` | −30 °C | 55 °C | 0 |
| `era5_T_dew` | −30 °C | 40 °C | 0 |
| `era5_RHum` | 0 % | 100 % | 0 |
| `era5_W_spd` | 0 m/s | 50 m/s | 0 |
| `era5_cloud_cover` | 0 | 1 | 0 |
| `era5_precipitation` | 0 mm | 200 mm | 0 |
| `era5_SZA` | 0° | 180° | 0 |
| `power_*` (5 columns) | see `02_DATA_SOURCES_AND_VARIABLES.md` | | 0 |

Counts are from `data/plots/post_preprocess/C_qc_flag_counts.csv`, which `04c` parses out of
`qc_report.txt`. Columns absent from that CSV had zero flags.

Then night masking: where `era5_SZA ≥ 90°`, all of `era5_GHI, era5_DNI, era5_DHI,
era5_GHI_clearsky, era5_CSI` are forced to `0.0`. Finally `era5_RHum` and `power_RH2M` are
hard-clipped to [0, 100].

**Step 3 — Hampel / MAD outlier flagging.**

HAMPEL_WINDOW = 15;  HAMPEL_N_SIGMA = 3.0
HAMPEL_COLS   = era5_GHI, era5_T_amb, era5_RHum, era5_W_spd, era5_cloud_cover
threshold     = 3.0 × 1.4826 × rolling_MAD          # 31-occurrence centred window, min_periods=5
is_outlier   &= roll_mad > 1e-6                     # skip flat/constant stretches

Per `(point_id, event)` series sorted by date — the window is ±15 occurrences of the *same sun
event*, roughly ±15 days. **Policy: flag → `NaN`, never delete.**

| Column | Flagged | % of 493,155 |
|---|---|---|
| `era5_cloud_cover` | **49,519** | 10.04 % |
| `era5_GHI` | **35,559** | 7.21 % |
| `era5_W_spd` | 11,350 | 2.30 % |
| `era5_T_amb` | 9,762 | 1.98 % |
| `era5_RHum` | 8,814 | 1.79 % |
| **Total** | **114,004** | |

**Both high rates have the same cause: univariate MAD filtering misapplied to variables whose
variance is the signal.** `era5_cloud_cover` is a bounded [0, 1] strongly bimodal variable (clear or
overcast), so a rolling median sits near an extreme, the MAD is small, the 3σ threshold is tight,
and genuine clear↔overcast transitions get flagged. `era5_GHI`'s day-to-day variability at a fixed
sun event is genuinely large in the monsoon, so real cloud-driven variation is winsorised as
outliers. Excluding those two columns from `HAMPEL_COLS`, or widening `HAMPEL_N_SIGMA` for them,
is the targeted fix — clouds are weather, not errors.

**Step 3b — Yeo-Johnson skew diagnostic**, report-only, on `era5_GHI, era5_W_spd,
era5_precipitation, era5_cloud_cover, era5_T_amb`. Writes `yeo_johnson_skew.csv`. **No column is
transformed.** Values for this run are **not available in the source files**.

## B.3 Step 4 — Hierarchical imputation

Four tiers, in order, over every column in `IMPUTE_COLS` (all numerics except `lat, lon,
population, weight, grid_lat, grid_lon, month, DOY, year, season_code`):

| Tier | Method | Scope |
|---|---|---|
| (a) | `interpolate(method="linear", limit=3, limit_area="inside")` | within each `(point_id, event)` series |
| (b) | `ffill(limit=3).bfill(limit=3)` | same |
| (c) | point median → `impute_zone` median → global median | progressively coarser |
| (d) | MICE (`IterativeImputer`, `max_iter=10`, `random_state=42`, `sample_posterior=False`) | fit on a ≤ 300,000-row sample |

**The `impute_zone` grouping** is a throwaway `KMeans(n_clusters=min(8, 45), random_state=42,
n_init=10)` on `lat`/`lon` only. The script is emphatic: "this is **NOT** the Phase 4 climate
clustering, just named `impute_zone` to avoid confusion with it." With 45 points and 8 zones, each
averages 5–6 points; `README_PREPROCESSING.md` warns this "will produce noticeably coarser zones
with 45 points to group."

**How many values reached each tier is not available in the source files** — `04` logs the
`Remaining after …` counts to `qc_report.txt`, which is git-ignored, and `04c`'s parser extracts
only the `physical_bounds` and `hampel_MAD` categories.

## B.4 Steps 5–9c — validation, feature engineering, occurrence-based features

**Step 5 — Temporal validation.** Warns for any `(point_id, event)` series with fewer than
`0.99 × expected_days` rows, and re-checks for duplicate keys. Per-series warnings for this run are
**not available in the source files**.

**Step 6 — Feature engineering** (7 features):

| Feature | Definition |
|---|---|
| `era5_W_dir_sin` / `era5_W_dir_cos` | `sin/cos(radians(W_dir)) × W_spd` |
| `era5_cloud_opacity` | `1 − CSI.clip(0,1)` |
| `era5_T_depression` | `T_amb − T_dew` |
| `is_daytime` | `(SZA < 90).astype(int)` |
| `ist_hour_decimal` | `time_utc + 5 h 30 m` as a decimal hour |
| `solar_hour_angle` | `(ist_hour_decimal − 12) × 15` degrees |

Computed per row "since there's no fixed 'hour' column in this schema — each sun-event happens at a
different UTC hour depending on point and date."

**Step 7 — Lag features.** `LAG_COLS` = `era5_GHI, era5_T_amb, era5_RHum, era5_W_spd,
era5_cloud_cover, era5_CSI`; `LAG_OCCURRENCES = [1, 7, 30]`, shifted within `(point_id, event)`
groups → **18 features**.

**Step 8 — Rolling stats.** `ROLL_OCCURRENCES = [7, 30]`, trailing mean + std (`min_periods=3`, std
`fillna(0)`) → **24 features**.

**Step 9 — Delta features.** 1-occurrence `diff()` (`fillna(0)`) for `era5_T_amb`, `era5_GHI`,
`era5_cloud_cover` → **3 features**.

18 + 24 + 3 = 45, matching the "Engineered features: 45" figure in
`data/plots/verify_preprocessing/07_preprocessing_summary.png` exactly.

**Step 9c — Lag-warm-up row drop.** Rows where `era5_GHI_lag30d` is `NaN` are dropped, "before
imputation/scaling see them, rather than let step 4's imputation quietly paper over what is
actually 'this occurrence is too early in this point's series to have a 30-days-prior lag'."

**Observed: 493,155 → 489,105 rows, i.e. exactly 4,050 dropped = 45 points × 3 events × 30
occurrences.** Every group lost precisely its 30-row warm-up, with none lost anywhere else.
Retention **99.2 %**.

**Step 9b — Savitzky-Golay diagnostic.** One sample point, `event == "noon"`, the median year; raw
vs `savgol_filter(polyorder=3)` with a window of up to 31. Visual QA only — the dataframe is
untouched. Not committed.

## B.5 Steps 10–11 — correlation and VIF

Pearson + Spearman on a ≤ 50,000-row sample of daytime rows over 15 columns, and
`variance_inflation_factor` on the same sample after dropping constant columns. **Nothing is
dropped on the basis of VIF** — it is reported only. Outputs `correlation_pearson.csv`,
`correlation_spearman.csv`, `correlation_heatmaps.png`, `vif_report.csv`, none committed, so the
actual values for this run are **not available in the source files**.

Both `README_PREPROCESSING.md` and `PREPROCESSING_STEPS.md` pre-empt the VIF result: near-infinite
VIF among `GHI/DNI/DHI/CSI` is expected and **structural**, because DNI and DHI are algebraically
derived from GHI. `README_PREPROCESSING.md` adds a small-N caveat: "step 11's VIF report is
computed over fewer independent spatial samples."

## B.6 Steps 12–13 — scaling and the hard gate

**Step 12 — leakage-safe MinMax scaling.** A separate `MinMaxScaler` **per column**, fitted on the
first `TRAIN_FRAC = 0.70` of the globally date-sorted rows, applied to the whole file. Scalers
pickled to `scalers.pkl`; output to `uttarakhand_cleaned_scaled.csv`. `SKIP_SCALE` excludes
identifiers, coordinates, calendar columns, `impute_zone` and `is_daytime`.

Because the sort is date-primary and the panel is balanced, this is a true chronological cut —
training is roughly 2016-02 to 2023-01.

The physical/scaled separation is enforced by design and `04b` reads only the physical file.
`PREPROCESSING_STEPS.md` gives the reason: "the signature indices (kWh/day, HDD18, etc.) are
non-linear functions of physical values and would be silently corrupted by pre-scaling."

**Step 13 — the hard gate.**

| Check | Criterion | Verifiable from committed artefacts? |
|---|---|---|
| Physical file: zero NaN in `IMPUTE_COLS` | `== 0` | **Yes — PASS** |
| Physical file: zero Inf | `== 0` | No |
| Scaled file: **train portion** within [0, 1] | `min ≥ −1e−6`, `max ≤ 1+1e−6` | No |
| Zero duplicate `(point_id, date, event)` | `== 0` | Indirect — the exact 489,105 = 493,155 − 4,050 arithmetic is only consistent with zero duplicates |
| All 8 required columns present | present | **Yes** — all 8 appear downstream |

The gate deliberately checks only the *training* portion of the scaled file, reporting the full-file
out-of-range fraction as **informational**: val/test rows may legitimately exceed [0, 1] "if the
val/test period contains a value more extreme than anything seen in training (e.g. a record hot day
in 2024 that wasn't in the 2016-2022 training window) — that's expected, not a bug."

**The final `RESULT: n/5 checks passed` line and `qc_report.txt` itself are not committed**, so the
gate's own verdict cannot be read directly. The two checks that *can* be corroborated both pass.

## B.7 Verified Phase 2 outcome

From `data/plots/verify_preprocessing/07_preprocessing_summary.png`:

| Metric | Value |
|---|---|
| Input records | 493,155 |
| Output records | **489,105** |
| Data retention | **99.2 %** |
| Input dimensions | 36 |
| Output dimensions | **89** |
| Core climate variables | 6 |
| Engineered features | 45 |
| Completeness of all 6 core variables | **100.0 % each** |
| Rows with no missing data | **489,105 (100.0 %)** |

**Zero residual missing data** — this independently confirms step 13's first gate condition passed,
and satisfies `04c`'s check A ("should be essentially all-zero; if not, step 4's imputation didn't
cover something and step 13's hard gate should already have failed").

## B.8 Cleaned-file distributions (observed)

From `data/plots/verify_preprocessing/01_climate_distributions.png`, over all 489,105 rows:

| Column | Mean | Std | Min | Max |
|---|---|---|---|---|
| `era5_T_amb` (°C) | 20.07 | 7.69 | **−5.00** | 42.22 |
| `era5_RHum` (%) | 68.99 | 19.00 | 8.96 | 100.00 |
| `era5_W_spd` (m/s) | 1.43 | 0.75 | 0.00 | 9.08 |
| `era5_P_atm` (hPa) | 901.90 | 47.66 | **850.00** | 1001.65 |
| `era5_GHI` (W/m²) | **21.03** | 37.94 | 0.00 | **702.74** |
| `era5_precipitation` (mm) | 0.08 | 0.44 | 0.00 | 45.85 |

Three observations for a write-up:

1. **`era5_T_amb` minimum is exactly −5.00 °C** — precisely the `02_combine` cut boundary, not a
climatological floor. Sub-−5 °C high-altitude winter sunrise values were removed at the merge
step and then imputed.
2. **`era5_P_atm` minimum is exactly 850.00 hPa** — precisely the `BOUNDS` lower limit. 850 hPa is
≈ 1,450 m in a standard atmosphere, so **37.1 % of readings from the pipeline's
higher-elevation points were destroyed and replaced by imputed values pulled toward
lower-elevation medians.** Only *low* values were removed, so the imputation is directionally
biased upward. The histogram is visibly multi-modal (peaks near 850–860, ~895, ~910 and
~965–980 hPa) — that is the real elevation stratification of the 45 points, truncated at its low
end with a large spike on the boundary. **`elev_proxy = mean(era5_P_atm)/1013.25` is a
`PCA_BLOCK` member and therefore feeds the clustering matrix**: the one signature index that
encodes elevation is computed from the column this bound compresses. Of every issue in this
pipeline, this is the one most specific to Uttarakhand.
3. **`era5_GHI` is anomalously low** — see Part A.3.

## B.9 Post-cleaning QA (`04c_postprocess_plots.py`)

Six checks, run **after** `04` on `uttarakhand_cleaned_physical.csv`. All six outputs are committed
to `data/plots/post_preprocess/`, five with interactive twins.

| Check | Purpose (from the docstring) | Output |
|---|---|---|
| A | Missing-data heatmap post-clean — "should be essentially all-zero" | `A_missing_post.png` / `.html` |
| B | Distribution sanity — "watch for imputation spikes (a suspicious mode exactly at the point/zone/global median)" | `B_distributions_post.png` / `.html` (43 MB) |
| C | Physical-bounds vs Hampel flag counts, parsed from `qc_report.txt` | `C_qc_flag_counts.png` **+ `C_qc_flag_counts.csv`** |
| D | Lag-feature sanity — GHI vs GHI-7-days-prior, "should be positive and clearly structured … not noise" | `D_lag_sanity.png` / `.html` |
| E | One point's cleaned noon-GHI series for one year with 7 d/30 d rolling means — "seasonal shape should look smooth, not flattened" | `E_point_timeseries.png` / `.html` |
| F | Post-clean correlation heatmap including the step-6 engineered features | `F_correlation_post.png` / `.html` |

`C_qc_flag_counts.csv` is the **only committed artefact anywhere in the repository that carries QC
counts**, and it carries the evidentiary weight of this entire Part B.
`04c_interactive_postprocess_qc.py` implements A, B, D, E, F only — "the qc_report.txt bar chart C
is trivial enough to leave as-is in the PNG script."

## B.10 Inputs, outputs, dependencies

**Inputs**: `data/processed/climate_uttarakhand_points.csv`.

**Outputs** (all under the git-ignored `data/preprocessed/`, none committed):
`uttarakhand_cleaned_physical.csv` (→ Phase 3), `uttarakhand_cleaned_scaled.csv`, `scalers.pkl`,
`qc_report.txt`, `correlation_pearson.csv`, `correlation_spearman.csv`, `correlation_heatmaps.png`,
`vif_report.csv`, `yeo_johnson_skew.csv`, `savitzky_golay_diagnostic.png`.

**Dependencies**: `pandas`, `numpy`, `scipy` (`stats`, `signal.savgol_filter`), `scikit-learn`
(`MinMaxScaler`, `KMeans`, `IterativeImputer`), `statsmodels` (VIF), `matplotlib`, `seaborn`;
`plotly` for `04c_interactive`.

---

# PART C — Combined Problems / Risks

Ranked by severity.

1. **`deaccumulate()`'s assumption is unverified and is associated with an order-of-magnitude GHI
deficit.** Highest-severity open item in the pipeline. `era5_GHI` feeds `era5_CSI`, `era5_DHI`,
`era5_cloud_opacity`, every Tier-1 solar index, and `GHI_mean` — which is in the clustering
matrix. Three independent artefacts corroborate the anomaly; two clean controls (clear-sky GHI
at r = 0.9923, T_amb at r = 0.902) isolate it to the de-accumulated fields.
2. **The cross-source disagreement was measured and never acted upon.** Three separate source files
state that a large MBE must be addressed before or in `04`; no such step exists. This is the
clearest process gap in the pipeline.
3. **`era5_P_atm`'s 850 hPa lower bound is mis-specified for Uttarakhand** and destroyed 37.1 % of
the column one-sidedly, in the exact variable `elev_proxy` is built from. State-specific, and
the highest-priority QC fix.
4. **`era5_LW_down`'s 50 W/m² bound destroyed 73.7 %** of that column. Harmless downstream, but a
second independent fingerprint of the same de-accumulation issue.
5. **The Hampel filter flagged 10.0 % of `era5_cloud_cover` and 7.2 % of `era5_GHI`** — a known
weakness of univariate MAD filtering on bounded bimodal and high-variance-by-nature variables.
114,004 values across five columns were replaced by imputation.
6. **Imputed and flagged cells are unmarked in the output.** With 114,004 Hampel-NaN'd values plus
546,424 bounds-NaN'd values all imputed and unlabelled, a consumer of
`uttarakhand_cleaned_physical.csv` cannot distinguish measured from reconstructed values. Adding
`{col}_imputed` booleans would cost little and would let `09`'s caveat text be specific. (The
*PCM* database does carry `*_imputed` flags; the climate data does not.)
7. **RHum's +11.4 % and wind's −1.14 m/s offsets reach the clustering matrix** while `02b`'s
already-computed `RH_mean_true` and `wind_mean_true` sit unused. A two-line `CANON_MAP` fix.
8. **`avg_sdirswrf`'s three-name matcher applies one unit convention to three fields** — a latent
3600× hazard, low-probability given what `01` requests, but unverified.
9. **`get_solarposition()`'s method is not pinned** in `compute_solar()` while it *is* pinned in
`00b`. One-line reproducibility fix.
10. **Bounds applied in `02` are narrower than those in `04` and are counted nowhere** — the
`T_amb < −5 °C` cut in particular is state-inappropriate.
11. **`CSI = 0` is three-way ambiguous** (true darkness / clear-sky floor / night mask).
12. **DHI is a closure residual with no independent basis** and should not be presented as a
modelled diffuse quantity.
13. **Matched timestamps are not persisted**, so per-row temporal match quality is unauditable.
14. **Cross-source statistics are pooled, not stratified** by season, event, point or year.
15. **`ETR` is computed and discarded.**
16. **None of the QC report artefacts are committed** — `qc_report.txt`, `vif_report.csv`,
`yeo_johnson_skew.csv`, the correlation CSVs and `pca_loadings.csv` are all git-ignored, so the
step-13 verdict and every diagnostic table are uncheckable from this repository.
`C_qc_flag_counts.csv` is the sole exception.

---

# PART D — Combined Status

**Phase 2 is COMPLETE and its structural results are strong.**

What went right, and is worth reporting positively:

- **Full coverage with zero loss at the merge**: 493,155 rows = 45 × 3,653 × 3 exactly. No
`(point, date, event)` failed the 3-hour match on either source.
- **The timezone/sun-event design works**: check B confirms noon peaks both GHI and T_amb.
- **Clear-sky modelling is independently corroborated** at r = 0.9923 / MBE +5.3 W/m² against NASA
POWER, validating coordinates, timing, grid snapping and the altitude assumption's effect on the
Ineichen model in one number.
- **The Tier-2 repair delivered**: 45/45 points, 0 skipped, 164,385 point-days, and it insulated
the clustering matrix's entire temperature and solar block from the ERA5 GHI problem.
- **Cleaning is surgical**: 99.2 % retention, with the only losses being exactly the 4,050-row
structural lag warm-up, and zero residual missing values afterwards.

What is not right, and blocks a final claim on any solar-derived quantity:

- **The all-sky ERA5 GHI is roughly an order of magnitude low**, the pipeline measured it, and
nothing corrected it. Verification requires one inspection of a raw `*_accum.nc` file.
- **The 850 hPa pressure bound compresses the one elevation-encoding signature index** for a state
whose entire methodological weak point is elevation.

Neither of these invalidates the Phase 3–6 chain — the two-tier design routed around the first, and
the second degrades rather than destroys `elev_proxy` — but both must be stated plainly wherever a
solar magnitude or an elevation-derived index is reported.

# Source File 5: 05_PHASE_3_AUDIT(3).md
Source path: /mnt/data/05_PHASE_3_AUDIT(3).md
# 05 — Phase 3 Audit: Climate Signature Construction

**Scripts**: `04b_climate_signature.py`, `04d_signature_interactive.py`

**Status**: **RUN.** Confirmed indirectly — every Phase 4–6 artefact consumes its output, and the
per-point signature values are visible in `data/plots/verify_clustering/05_cluster_profiles.png`.
The output CSV itself is under the git-ignored `data/processed/` tree and is **not present in this
repository**.

---

## Purpose

Collapse each point's entire 10-year, 3×-daily record into **one row per `point_id`**. That row is
the object Phase 4 actually clusters — not the raw data.

The v3.0 change this script implements is stated in its own docstring:

The earlier version only used the 3-events/day merged CSV and approximated `GHI_daily_kWh` with a
half-sine formula, and `DTR` as (noon − sunrise). Those are proxies, not measurements, and the
plan doc (v3.0 Section 4.3, "Repair 1") is explicit that this is the single highest-value
remaining data task.

## Hard gate

if not TIER2_FILE.exists():
    raise FileNotFoundError(
        f"{TIER2_FILE} not found. Run 02b_build_daily_aggregates.py first …
         This script cannot proceed without it (plan v3.0 Repair 1).")

`04b` will **not** run on Tier-1 proxies alone. This is a real, enforced dependency, not a comment.

## Inputs

- `data/preprocessed/uttarakhand_cleaned_physical.csv` — the **physical-units** file only. `04b`
never reads the scaled file, because the signature indices (kWh/day, HDD18, CDD24, …) are
non-linear functions of physical values and would be silently corrupted by pre-scaling.
- `data/processed/tier2_signature_uttarakhand.csv` — `02b`'s output.

## Processing — the six numbered stages

| Stage | What it does |
|---|---|
| [1/6] | Build Tier-1 sun-event signature vectors, one row per `point_id` |
| [2/6] | Left-join Tier-2, report `Points with Tier-2 coverage: n/45`, set canonical columns |
| [3/6] | Derive `Tm_target_C`, `T_mains_est_C`, `L_required_kJ_per_kg` |
| [4/6] | Add 5 interaction terms |
| [5/6] | PCA on the correlated temperature/pressure block; build the clustering column list |
| [6/6] | z-standardise the clustering matrix, join it back, write the output |

### Stage 1 — Tier-1 construction

`daily_frame()` pivots each point's records to one row per date with columns
`{era5_T_amb, era5_GHI, era5_CSI, era5_RHum, era5_precipitation, era5_T_dew} × {sunrise, noon,
sunset}`. The Tier-1 indices are computed from that pivot plus the long-form frame.

The `GHI_daily_kWh_proxy` half-sine formula, which is the one worth recording explicitly:

daylen_hours = (sunset_time_utc − sunrise_time_utc).total_seconds() / 3600
ghi_kw       = noon_GHI / 1000
daily_kwh    = (2.0 / π) · ghi_kw · daylen_hours

It uses the **actual** `sunset − sunrise` interval from `suntimes.csv`, not a nominal 12 h, which is
the right choice for a latitude band whose day length swings ~4 h across the year.

Two indices are **explicitly flagged as proxies** in the docstring:

- `DTR_proxy = noon T − sunrise T` — a lower bound on the true diurnal range, because true `Tmax`
typically lags solar noon by 1–3 h.
- `monsoon_index` — "a JJAS *fraction*, not an absolute rainfall total, since precipitation is only
sampled 3x/day."

### Stage 2 — the canonical merge

`CANON_MAP` has 13 entries. For each, the canonical column takes the **true Tier-2 value where
present** and falls back to the Tier-1 proxy otherwise:

sig[canon] = sig[true_col].where(sig[true_col].notna(), sig.get(f"{canon}_proxy", np.nan))

| Canonical column | Tier-2 source | Tier-1 fallback |
|---|---|---|
| `GHI_daily_kWh` | `GHI_daily_kWh_mean` | `GHI_daily_kWh_proxy` |
| `DTR` | `DTR_true_mean` | `DTR_proxy` |
| `kt_mean`, `kt_std` | `kt_daily_mean`, `kt_daily_std` | `kt_mean_proxy`, `kt_std_proxy` |
| `SAI` | `SAI_true` | `SAI_proxy` |
| `cloudy_frac` | `cloudy_frac_true` | `cloudy_frac_proxy` |
| `CCI` | `CCI_true` | `CCI_proxy` |
| `HDD18`, `CDD24` | `HDD18_true`, `CDD24_true` | `HDD18_proxy`, `CDD24_proxy` |
| `Ta_mean`, `Ta_p95`, `Ta_p05` | `Ta_mean_true`, `Ta_p95_true`, `Ta_p05_true` | `Ta_*_proxy` |
| `seasonality` | `seasonality_true` | `seasonality_proxy` |

Both versions are kept side by side "purely so you can report 'proxy vs. true agreement' in your
methodology," and **both are excluded from the clustering matrix** so only the canonical version
clusters.

**Five signature columns have no Tier-2 counterpart** and remain sun-event/ERA5-derived:
`RH_mean`, `HSI`, `wind_mean`, `monsoon_index`, `elev_proxy` — plus `GHI_mean` (mean noon
`era5_GHI`), which carries no `_proxy` suffix at all and therefore enters the clustering matrix
directly. Note that `02b` *does* compute `RH_mean_true` and `wind_mean_true`, but they have no
`CANON_MAP` entry and are dropped by the `_true` suffix rule — so two already-available Tier-2
values go unused. See `04_PHASE_2_AUDIT.md` Part A.8.

The script prints `Points with Tier-2 coverage: n/45` and warns for any point that fell back to a
proxy. **The actual coverage number is not available in the source files**, but `02b`'s confirmed
45/45-point, 164,385-point-day run implies full Tier-2 coverage.

### Stage 3 — derived PCM targets

T_DELIVERY_C  = 50.0
DT_APPROACH_C =  7.0
TM_TARGET_C   = 57.0                                  # constant for every point, by design

DRAW_RATE_KG_PER_S  = 60.0 / 1000 / 60                # = 0.001 kg/s
CP_WATER            = 4.186                           # kJ/kg·K
ASSUMED_PCM_MASS_KG = 50.0

sig["T_mains_est_C"]        = sig["Ta_mean"] - 2.0
q_night_kw                  = 0.001 × 4.186 × (50 − T_mains_est_C)
sig["L_required_kJ_per_kg"] = (q_night_kw × 3600 × 7) / 50

`PREPROCESSING_STEPS.md` explains the sign convention:

the corrected v2.0 rule: `Tm_target = T_delivery + delta_T_approach` (PCM sits *above* delivery
temperature so heat flows PCM→water during discharge; the earlier subtract-based rule had the
sign backwards). Comes out to a constant 57 C here (50 + 7, indirect-system assumption) — held
constant across all points **by design, not tuned per cluster**.

`04b` prints the resulting `L_required` range, but the values are **not available in the source
files**. They can be bounded from the observed cluster `Ta_mean` medians (≈ 13–25 °C, see
`06_PHASE_4_AUDIT.md`): **`L_required` ≈ 63–82 kJ/kg**, and the Phase 5 floor at 0.7× is
**≈ 44–58 kJ/kg**. The minimum latent heat in the whole 55-row PCM database is 128 kJ/kg, so the
floor is non-binding — which `08_mcdm_ranking.py`'s own diagnostic text confirms independently:
"every candidate's latent heat comfortably clearing L_required in every cluster."

Two things to note about this formula, both material for a write-up:

- **The `− 2.0` K mains-temperature offset is unsourced in-code.** No citation appears anywhere in
`era5-uttarakhand/`, and it drives `L_required` directly.
- **There is no `SHARE_PCM` fractional-contribution factor.** This `04b` sizes `L_required` from a
7-hour draw at 0.001 kg/s against the full 50 kg PCM mass — i.e. the PCM alone is assumed to
supply the whole night load. The resulting values happen to be small enough that the filter never
binds, so the assumption does not affect this run's outcome, but it should be stated rather than
left implicit.

### Stage 4 — 5 interaction terms

| Term | Definition |
|---|---|
| `int_GHI_x_ktstd` | `GHI_daily_kWh × kt_std` |
| `int_DTR_x_cloudyfrac` | `DTR × cloudy_frac` |
| `int_RH_x_TaMinusTm` | `RH_mean × (Ta_mean − Tm_target_C)` |
| `int_wind_x_TaMinusTsoil` | `wind_mean × (Ta_mean − Tsoil_proxy_C)`, where `Tsoil_proxy_C = Ta_mean − 3.0` |
| `int_CCI_x_1minusSAI` | `CCI × (1 − SAI)` |

`Tsoil_proxy_C` exists **only** to feed the fourth term and is dropped from the clustering matrix.
Note that `int_wind_x_TaMinusTsoil` therefore reduces algebraically to `3.0 × wind_mean` — it is a
rescaled copy of `wind_mean`, not an independent interaction. Since `wind_mean` is also in the
matrix, this effectively double-weights wind.

### Stage 5 — PCA and clustering-matrix construction

PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]
StandardScaler → PCA(n_components=0.95, random_state=42)      # retain 95% variance
loadings → pca_loadings.csv

**The number of retained components for this run is not available in the source files** —
`pca_loadings.csv` is git-ignored.

Columns removed from the clustering matrix (`DROP_FROM_CLUSTERING`):

- every `PCA_BLOCK` member (now represented by `PC1…PCn`)
- `lat`, `lon` — "never cluster on geography — plan v3.0 Section 6.2"; `05` re-prints this at run
time
- `population`, `T_mains_est_C`, `Tsoil_proxy_C`
- every column ending `_proxy`
- every column ending `_true` or `_true_mean`

Everything else is z-standardised with `StandardScaler` and appended with a `_z` suffix. The
resulting `_z` set comprises: the non-PCA canonical indices (`GHI_mean`, `kt_mean`, `kt_std`,
`SAI`, `CCI`, `cloudy_frac`, `DTR`, `GHI_daily_kWh`, `seasonality`, `HSI`, `wind_mean`,
`monsoon_index`), `Tm_target_C`, `L_required_kJ_per_kg`, the 5 interaction terms, and `PC1…PCn`.

**`Tm_target_C` is constant (57.0) across all 45 points**, so its z-score is a zero-variance
column. It contributes nothing to the clustering but is not excluded.

## Climate Signature Feature-to-PCM-Property Mapping

The design principle the two-tier signature is built on is that every index must earn its place by
constraining a PCM property. The Uttarakhand implementation's mapping:

### Tier 1 — sun-event statistics

| Feature | Physical mechanism | PCM property it constrains |
|---|---|---|
| `GHI_mean` | Mean solar irradiance at the charging instant | Charging-rate feasibility; upper bound on achievable `Tm` |
| `RH_mean` | Annual mean relative humidity → condensation risk at the PCM container | Corrosion-resistance requirement; encapsulation choice |
| `HSI` | `RH_mean × fraction(T_amb − T_dew < 3 K)` — combined humidity + near-saturation signal | Intended as the corrosion-veto trigger. **In this run it triggers nothing** — `07`'s corrosion veto is not implemented, and all 55 database candidates are organic. |
| `wind_mean` | Mean wind speed → convective loss from collector and tank | Tank/collector loss coefficient; indirectly the required storage margin |
| `monsoon_index` | JJAS share of annual precipitation → seasonal charging gap | Storage sizing for the monsoon under-charging window (descriptive, not a ranking criterion) |
| `elev_proxy` | `mean(P_atm)/1013.25` → atmospheric column mass | Air mass into the Ineichen clear-sky model; PCA thermodynamic block |

### Tier 2 — true daily-integral indices

| Feature | Physical mechanism | PCM property it constrains |
|---|---|---|
| `GHI_daily_kWh` | True daily charging energy available | `L_required` sizing — the latent-heat floor |
| `kt_mean` | Annual mean clearness index → solar resource quality | Charging reliability; the `07b` regime cap uses it directly |
| `kt_std` | Day-to-day clearness variability | Charging intermittency; feeds `int_GHI_x_ktstd` |
| `SAI` | `Σ GHI / Σ GHI_clearsky` → fraction of the clear-sky resource actually delivered | Latent-heat margin requirement |
| `cloudy_frac` | Fraction of days with `kt < 0.35` | Autonomy sizing — how often the PCM must carry the load alone |
| `CCI` | Longest consecutive cloudy-day run (days) | Worst-case autonomy; the binding case for storage capacity |
| `DTR` | True `Tmax − Tmin` → daily thermal cycling magnitude | Cycling-stability requirement (`cycles ≥ 300` in Phase 5) |
| `Ta_mean` | Annual mean ambient | `T_mains_est_C` → `L_required`; PCA block |
| `Ta_p95` | Hot design percentile | Upper end of the melting window; safety at extreme heat |
| `Ta_p05` | Cold design percentile | Night-discharge environment; low-temperature cycling stress |
| `HDD18` | Heating degree-days, base 18 °C | Seasonal demand context; PCA block |
| `CDD24` | Cooling degree-days, base 24 °C | Seasonal demand context; PCA block |
| `seasonality` | `std/mean` of monthly-mean daily GHI | Seasonal resource swing → sizing for the worst month |

### Derived targets (not in the clustering matrix as discriminators)

| Quantity | Role |
|---|---|
| `Tm_target_C` = 57 °C | Drives the Phase 5 melting window `[52, 65]` °C and the Phase 6 Gaussian `f_Tm` criterion. **Constant across all points**, so it discriminates nothing. |
| `L_required_kJ_per_kg` | Drives the Phase 5 latent-heat floor `L ≥ 0.7 × L_required`. Varies with `Ta_mean` but lands well below every candidate's latent heat, so it also discriminates nothing. |

### Why the two-tier design is necessary

Neither tier alone is sufficient, and the Uttarakhand run demonstrates exactly why:

- **Tier 1 alone underestimates.** `DTR_proxy = noon − sunrise` is a lower bound on the true
diurnal range. `GHI_daily_kWh_proxy` is a half-sine reconstruction from a single instantaneous
sample. Degree-days from a 3-point daily mean are not degree-days from a true daily mean.
- **Tier 2 alone loses the charge/discharge instants.** The sun-event samples are the only place
the pipeline observes conditions *at* the moments that matter thermally.
- **Tier 2 also rescued this run.** Because the canonical solar and temperature columns come from
NASA POWER via `02b`, the clustering matrix's entire solar block was insulated from the ERA5 GHI
magnitude anomaly documented in `04_PHASE_2_AUDIT.md` Part A.3. The `_proxy` variants carry the
anomaly but are excluded by the suffix rule. **This is the single largest practical payoff of the
Repair-1 design and should be reported as such.**

### PCA scope — and why the solar block is kept out

PCA is applied to `Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy` only — the mutually
correlated thermodynamic block. The solar and variability indices (`GHI_daily_kWh`, `kt_mean`,
`kt_std`, `SAI`, `CCI`, `cloudy_frac`, `DTR`, `seasonality`, `monsoon_index`, `HSI`, `wind_mean`)
are deliberately **kept out**, because they carry the discriminating signal for regime separation
and for PCM target derivation. Compressing them would reduce exactly the information the downstream
recommendation depends on.

### Indices that carry a known problem into the clustering matrix

| Index | Problem | Severity |
|---|---|---|
| `GHI_mean` | ERA5 noon GHI, no Tier-2 override — carries the −211 W/m² anomaly | High |
| `elev_proxy` | Built from `era5_P_atm`, 37.1 % of which was NaN'd one-sidedly by the 850 hPa bound and imputed | High for a montane state |
| `RH_mean` | ERA5-side, +11.4 % MBE vs POWER, unused `RH_mean_true` available | Moderate |
| `wind_mean` | ERA5-side, −1.14 m/s MBE vs POWER, unused `wind_mean_true` available | Moderate |
| `HSI` | Built on `RH_mean`, so inherits its offset | Moderate |
| `monsoon_index` | Permanently a 3×/day ERA5 precipitation *fraction*; JJAS here vs JJA in `SEASON_MAP` | Low (a ratio; descriptive only) |
| `int_wind_x_TaMinusTsoil` | Algebraically `3.0 × wind_mean` — a rescaled duplicate, not an interaction | Low |
| `Tm_target_C` | Zero-variance column | Cosmetic |

## `04d_signature_interactive.py` — explorer

Reads `climate_signature_uttarakhand.csv` and writes Folium/Plotly HTML to
`data/processed/signatures/interactive/`. Produces a multi-layer map with one toggleable layer per
index (`MAP_LAYERS = GHI_daily_kWh, Ta_mean, DTR, kt_mean, cloudy_frac, CCI, HDD18, CDD24, RH_mean,
HSI, monsoon_index, L_required_kJ_per_kg`), an interactive correlation heatmap, index-distribution
histograms, and a scatter matrix of the key PCM-facing indices "to eyeball the clustering structure
before `05` finds it formally."

**Its output directory is under the git-ignored `data/processed/` tree, so none of it is present in
this repository.**

## Outputs

| File | Contents | Committed? |
|---|---|---|
| `data/processed/signatures/climate_signature_uttarakhand.csv` | 45 rows: raw indices + `_z` columns | No |
| `data/processed/signatures/pca_loadings.csv` | PCA component loadings | No |
| `signature_correlation_heatmap.png` | 18-index correlation | No |
| `signature_distributions.png` | per-index histograms, with a constant-value special case | No |
| `point_signature_map.png` | lon/lat scatter coloured by `GHI_daily_kWh` and `monsoon_index` | No |
| `data/processed/signatures/interactive/*.html` | `04d` output | No |

None of Phase 3's own outputs are committed. The only surviving evidence of the signature values is
`data/plots/verify_clustering/05_cluster_profiles.png`, which plots six of them by cluster.

## Dependencies

`pandas`, `numpy`, `scikit-learn` (`PCA`, `StandardScaler`), `matplotlib`, `seaborn`;
`plotly` + `folium` + `branca` for `04d`.

## Validation

| Check | Result |
|---|---|
| Tier-2 file exists before running | **Enforced** — hard `FileNotFoundError` |
| Tier-2 coverage per point reported | Implemented; value not available in the source files |
| Reads the physical (unscaled) file only | **Confirmed** — `PHYSICAL_FILE` is the only climate input |
| lat/lon excluded from clustering | **Confirmed** — dropped in `DROP_FROM_CLUSTERING`, re-announced by `05` |
| PCA retains 95 % variance | Implemented (`n_components=0.95`); component count not available |
| Diagnostic plots handle degenerate columns | **Yes** — `signature_distributions.png` has an explicit constant-value branch, which is what `Tm_target_C` triggers |

## Problems / risks

1. **`Tm_target` is constant at 57 °C for every point.** A stated design decision, and the direct
cause of the identical survivor sets and identical #1 PCM in Phases 5 and 6. It means Phase 3
contributes no climate-driven differentiation to the PCM target itself — all differentiation
would have to come from `L_required`, which is non-binding.
2. **`T_mains_est_C = Ta_mean − 2.0` is unsourced in-code** and drives `L_required` directly.
3. **`L_required` has no `SHARE_PCM` fractional-contribution factor** — the PCM alone is implicitly
assumed to supply the whole night load. Non-binding in this run, but it should be stated.
4. **`GHI_mean` enters the clustering matrix carrying the ERA5 GHI anomaly** — the one solar column
the Tier-2 repair does not cover.
5. **`RH_mean` and `wind_mean` are taken from the ERA5 side despite Tier-2 equivalents existing**
(`RH_mean_true`, `wind_mean_true` are computed by `02b` and discarded). A two-entry `CANON_MAP`
addition would fix it.
6. **`int_wind_x_TaMinusTsoil` is a rescaled duplicate of `wind_mean`** (`= 3.0 × wind_mean`), so
wind is effectively double-weighted in the clustering matrix.
7. **`monsoon_index` uses JJAS while `SEASON_MAP` uses JJA** — unreconciled, and `monsoon_index` is
in the clustering matrix.
8. **`Tm_target_C` is a zero-variance column in the clustering matrix.** Harmless but untidy.
9. **`elev_proxy` is built from the column most damaged by Phase 2's physical bounds** (37.1 % of
`era5_P_atm` NaN'd one-sidedly and imputed) — see `04_PHASE_2_AUDIT.md` Part B.8. For a state
whose central methodological weakness is elevation, this is the most consequential inherited
defect in the signature.
10. **No Phase 3 output is committed**, so `pca_loadings.csv` — which `NEXT_STEPS.md` specifically
asks the student to inspect ("check how much weight `elev_proxy` carries") — cannot be examined
from this repository.

## Status

**COMPLETE.** The two-tier merge works as designed and demonstrably protected the clustering matrix
from the pipeline's largest data defect. The open items are the constant `Tm_target` (a design
choice with large downstream consequences), the unsourced mains-temperature offset, and the four
ERA5-side columns that could have used already-computed Tier-2 values.

# Source File 6: 06_PHASE_4_AUDIT(3).md
Source path: /mnt/data/06_PHASE_4_AUDIT(3).md
# 06 �?" Phase 4 Audit: Climate Regime Clustering

**Scripts**: `05_cluster_uttarakhand.py` (single-state, **run**),
`05b_cluster_interactive.py` (explorer), `05_cluster_regions.py` (multi-state, **not run**)

**Status**: **COMPLETE at K = 5.** Cluster assignments for all 45 points are recoverable from
`data/plots/uttarakhand_objective1/02_climate_regime_map_folium.html`.

---

## Why a single-state script exists

`05_cluster_uttarakhand.py`'s docstring:

`05_cluster_regions.py` was written for the ORIGINAL v3.0 scope: combine signature matrices from
FOUR states �?� and cluster across all of them together. �?� You are working on Uttarakhand only
right now. That cross-state comparison isn't required for Objective 1 to stand on its own: the
objective is "cluster meteorological data and identify Top-2/Top-3 PCM candidates per climatic
regime" �?" nothing in the objective statement requires those regimes to span multiple states.

The docstring names the regimes it expects to find within Uttarakhand: "the high-altitude
Himalayan belt around Chamoli/Pithoragarh vs. the Doon Valley around Dehradun vs. the Terai plains
around Udham Singh Nagar/Haridwar �?� elevation alone spans roughly 200-2000m of populated terrain
here." These are the script author's expectations, stated in prose �?" the pipeline does **not**
assign district names to clusters, and no committed artefact labels a cluster geographically.

## Algorithm choice: Gaussian Mixture, full covariance

GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)   # selection
GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=10)  # final fit

The justification given (repeated in `05_cluster_regions.py` and `README_PREPROCESSING.md`) is
that climate is a continuous gradient:

the boundary between "high-hill" and "valley/plains" Uttarakhand is not a hard line, and a point
near that boundary genuinely has partial membership in both. Soft membership probabilities are
kept and are what Phase 5/6 should read for boundary points.

`covariance_type="full"` is used without a separate justification in the Uttarakhand script.

## Model-selection configuration

K_CANDIDATES = list(range(2, 11))                       # K = 2 �?� 10
K_FINAL      = 5                                        # line 73 �?" set manually after review
SILHOUETTE_ACCEPT_LO, SILHOUETTE_ACCEPT_HI = 0.15, 0.40
RANDOM_STATE = 42

The 0.15�?"0.40 band is explicitly wider than the 0.15�?"0.35 band used by the four-state script,
with the reason given inline: "no artificial between-state gaps inflating it here."

`README_PREPROCESSING.md` sets the expectation and the warning:

Expected K for one state, and with only 45 points to work with: probably smaller than �?�
realistically 2-4 (e.g. high-Himalaya vs. Doon Valley vs. Terai plains). With 45 points, be
conservative about K: each additional cluster shrinks the average points-per-cluster fast, and a
GMM fit on very few points per component gets unstable.

**The run used K = 5, one above the top of that recommended range.** With 45 points that is an
average of 9 points per component, and the smallest component has only 3.

## Feature matrix

`X = sig[[c for c in sig.columns if c.endswith("_z")]].fillna(median).values`

Only the `_z` (standardised) columns from `04b` are used. `lat`/`lon` are absent by construction �?"
`04b` dropped them from the clustering column list, and `05` re-prints the reason at run time:
"(lat/lon are NOT among these �?" never cluster on geography, plan v3.0 Section 6.2)."

The exact number of `_z` columns for the Uttarakhand run is **not available in the source files**
(`climate_signature_uttarakhand.csv` is git-ignored). From `04b`'s `DROP_FROM_CLUSTERING` logic it
comprises: the non-PCA canonical indices (`GHI_mean`, `kt_mean`, `kt_std`, `SAI`, `CCI`,
`cloudy_frac`, `DTR`, `GHI_daily_kWh`, `seasonality`, `HSI`, `wind_mean`, `monsoon_index`), the
constant `Tm_target_C`, `L_required_kJ_per_kg`, the 5 interaction terms, and `PC1�?�PCn`.

**Note:** `Tm_target_C` is constant (57.0) across all 45 points. Its z-score is therefore
undefined/zero-variance; `StandardScaler` will emit a constant column. It contributes nothing to
the clustering but is not excluded.

## Model-selection outputs

Four metrics per K, written to `bic_selection_uttarakhand.csv`:
`BIC`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, plus an `in_accept_band` boolean.

A K-Means comparison (`KMeans(n_clusters=k, random_state=42, n_init=10)`, silhouette only) is
written to `kmeans_comparison_uttarakhand.csv`. Its purpose is stated as answering "the 'why not
K-Means' question with a number instead of an assertion."

**The contents of both CSVs are not available in the source files** �?" `data/processed/clustering/`
is git-ignored, and no committed plot renders the BIC or K-Means selection curves for the actual
run. (`05b_cluster_interactive.py` would render them, but its `interactive/` output directory is
git-ignored too.)

## Final fit and outputs

k_final_safe = min(K_FINAL, len(X) - 1)      # = 5
gmm_final    = GaussianMixture(5, covariance_type="full", random_state=42, n_init=10)
hard_labels  = gmm_final.fit_predict(X)
soft_probs   = gmm_final.predict_proba(X)

| Output file | Contents |
|---|---|
| `bic_selection_uttarakhand.csv` | K = 2�?�10 �- {BIC, silhouette, DB, CH, in_accept_band} |
| `kmeans_comparison_uttarakhand.csv` | K = 2�?�10 �- K-Means silhouette |
| `cluster_assignments_uttarakhand.csv` | `point_id, lat, lon, population, cluster_id, max_membership_prob, prob_cluster0�?�4` |
| `cluster_profiles_uttarakhand.csv` | one row per cluster: `cluster_id, n_points, total_population_covered`, plus the **population-weighted mean** of every non-`_z` numeric signature column |
| `cluster_map_uttarakhand.png` | scatter of lon/lat coloured by `cluster_id`, annotated `C0�?�C4` |

Population weighting uses `np.average(g[col], weights=g["population"])`, falling back to an
unweighted mean if the weight sum is zero.

`cluster_profiles_uttarakhand.csv` is what `07_feasibility_filter.py` and
`09_recommendation_cards.py` read. Critically, because `profile_cols` is "everything not
`point_id`/`cluster_id` and not ending `_z`", it carries `Tm_target_C` and `L_required_kJ_per_kg`
through �?" which is exactly what `07` checks for and errors on if absent.

---

## Observed results

### Cluster assignments (all 45 points)

Recovered from the popups in `data/plots/uttarakhand_objective1/02_climate_regime_map_folium.html`:

| Cluster | n_points | Member `point_id`s |
|---|---|---|
| **0** | **12** | 0003, 0004, 0005, 0006, 0007, 0014, 0019, 0020, 0031, 0034, 0037, 0044 |
| **1** | **9** | 0002, 0008, 0011, 0021, 0024, 0025, 0026, 0033, 0036 |
| **2** | **3** | 0023, 0040, 0041 |
| **3** | **7** | 0001, 0009, 0010, 0012, 0013, 0016, 0017 |
| **4** | **14** | 0015, 0018, 0022, 0027, 0028, 0029, 0030, 0032, 0035, 0038, 0039, 0042, 0043, 0045 |

Independently corroborated by `data/plots/verify_clustering/06_cluster_sizes.png`, which prints
12 / 9 / 3 / 7 / 14. Total 45. Max/min size ratio = 14 / 3 = **4.67**.

### Population and geographic extent per cluster

Computed by joining the cluster assignments to the per-point populations and coordinates embedded
in `data/plots/comprehensive/maps/A2_population_map.html`:

| Cluster | n | Population covered | Share | Latitude range (mean) | Longitude range (mean) |
|---|---|---|---|---|---|
| 0 | 12 | **3,432,283** | 32.8 % | 29.125 �?" 30.375 (29.562) | 77.875 �?" 79.875 (78.854) |
| 1 | 9 | **2,451,043** | 23.4 % | 30.125 �?" 30.625 (30.292) | 78.125 �?" 78.875 (78.486) |
| 2 | 3 | **330,779** | 3.2 % | 30.125 �?" 30.375 (30.292) | 79.125 �?" 79.375 (79.292) |
| 3 | 7 | **2,541,919** | 24.3 % | 28.875 �?" 29.875 (29.268) | 77.875 �?" 79.875 (78.804) |
| 4 | 14 | **1,719,687** | 16.4 % | 29.125 �?" 30.625 (29.696) | 77.875 �?" 80.125 (79.625) |
| **Total** | **45** | **10,475,711** | 100 % | | |

Cluster 2 is the smallest by both point count (3) and population (3.2 %), and is the most
spatially compact �?" a 0.25° �- 0.25° neighbourhood around 30.25° N, 79.25° E.

### Climate profile per cluster (observed medians)

From the boxplots in `data/plots/verify_clustering/05_cluster_profiles.png`, which plot the first
six numeric feature columns of the signature matrix. Values are read from the plot and are
therefore **approximate to the plotting resolution**:

| Index (Tier-1 proxy) | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| `Ta_mean_proxy` (°C) | �?^ 22.8 | �?^ 19.0 | **�?^ 13.4** | **�?^ 25.0** | �?^ 18.2 |
| `Ta_p95_proxy` (°C) | �?^ 29.8 | �?^ 25.6 | �?^ 20.4 | �?^ 32.8 | �?^ 24.3 |
| `Ta_p05_proxy` (°C) | �?^ 12.1 | �?^ 9.1 | �?^ 4.2 | �?^ 13.8 | �?^ 9.4 |
| `DTR_proxy` (K) | �?^ 7.9 | �?^ 7.8 | �?^ 7.1 | �?^ 7.9 | �?^ 7.2 |
| `GHI_mean` (W/m², noon) | �?^ 52.9 | �?^ 44.5 | �?^ 44.7 | �?^ 55.1 | �?^ 50.0 |
| `GHI_daily_kWh_proxy` (kWh/m²/day) | �?^ 0.404 | �?^ 0.342 | �?^ 0.335 | �?^ 0.428 | �?^ 0.380 |

The temperature ordering is monotone and coherent: **C3 (warmest) > C0 > C1 > C4 > C2 (coldest)**,
spanning �?^ 11.6 K of mean-temperature separation, with the same ordering reproduced in `Ta_p95` and
`Ta_p05`. Combined with the geographic extents above �?" C3 southernmost and lowest-latitude, C2 a
compact high-longitude/high-latitude pocket �?" the partition is internally consistent with an
elevation/latitude gradient.

**The source files do not assign geographic names to the clusters.** No committed artefact in
`era5-uttarakhand/` labels a cluster as "Terai", "Doon Valley" or "high Himalaya". Any such
labelling in a write-up would be interpretation added on top of the pipeline, not a pipeline
output.

**The `GHI_mean` and `GHI_daily_kWh_proxy` values above are affected by the ERA5 GHI magnitude
anomaly** documented in `04_PHASE_2_AUDIT.md` Part A.3. Their *relative* ordering across clusters is
still informative; their absolute magnitudes are not usable.

### Soft membership

Every one of the 45 popups reports `Prob: 1.000` �?" i.e. `max_membership_prob` rounds to 1.000 at
three decimal places for **every point**. The soft-clustering rationale in the docstring ("a point
near that boundary genuinely has partial membership in both") therefore did **not** materialise in
practice: the fitted GMM assigns every point to its cluster with essentially unit confidence.

This is the expected behaviour of a full-covariance GMM fitted to 45 samples in a
high-dimensional standardised space �?" each component can shape itself tightly around its members.
It means the `prob_cluster0�?�4` columns carry no usable boundary information for this run, and
`05b_cluster_interactive.py`'s boundary-point feature (a faint ring where `max prob < 1.5/K`) would
have highlighted nothing.

### Silhouette

`data/plots/verify_clustering/02_silhouette_plot.png` reports, for the **saved K = 5 labels**:

| Metric | Value |
|---|---|
| Average silhouette | **0.279** |
| Reference threshold drawn on the plot | 0.400 |
| Per-cluster spread | C0 �?^ 0 �?" 0.35, C1 �?^ 0 �?" 0.41, C2 �?^ 0 �?" 0.61, C3 �?^ 0 �?" 0.47, C4 �?^ �^'0.15 �?" 0.37 |

0.279 falls inside `05_cluster_uttarakhand.py`'s stated accept band of **0.15�?"0.40** and below the
0.4 "good" threshold used by `VERIFICATION_METHODOLOGY.md`'s success criteria. Cluster 4 (the
largest, n = 14) contains the only points with **negative** silhouette values, indicating a few
points closer to a neighbouring cluster's centroid than to their own.

**Caveat on this number.** `verify_02_clustering.py` computes silhouette on **its own** feature
matrix �?" every numeric column of `climate_signature_uttarakhand.csv` except
`point_id/cluster_id/lat/lon/population`, re-standardised �?" which includes the raw indices, the
`_proxy` and `_true` duplicates, the PCA-block members, **and** the `_z` columns. That is a
different and much larger space than the `_z`-only matrix the GMM was actually fitted in. The
0.279 figure is therefore a valid independent diagnostic but is **not** the silhouette that
`05_cluster_uttarakhand.py` itself wrote to `bic_selection_uttarakhand.csv` at K = 5. That value
is not available in the source files.

---

## What is absent from Phase 4

| Component | Status |
|---|---|
| Bootstrap / ARI cluster-stability analysis | **Not implemented.** No resampling of any kind appears in `05_cluster_uttarakhand.py`. |
| Fitted-model persistence (`joblib` scaler + GMM) | **Not implemented.** Neither the `StandardScaler` from `04b` nor the fitted `GaussianMixture` is saved. Re-running Phase 5�?"8 requires re-fitting. |
| `sklearn_version` recorded in outputs | **Not implemented.** |
| Canonical cluster relabelling (e.g. by ascending latitude) | **Not implemented.** Cluster IDs come straight from `GaussianMixture.fit_predict` and are therefore only stable because `random_state=42` is fixed. |
| External climate classification (Köppen-Geiger, NBC/ECBC) | **Not implemented.** The K = 5 partition rests entirely on internal statistics. |
| Automatic K selection | **Not implemented by design** �?" `K_FINAL` is a manually edited constant, and the script prints "update after reviewing this table, then re-run." |

---

## `05_cluster_regions.py` �?" multi-state, not run

Present but inert. `REGION_FILES` maps `"Uttarakhand"` to this pipeline's own signature file and
`"Rajasthan"` to `../era5-rajasthan/data/processed/signatures/climate_signature_rajasthan.csv`.
`main()` returns early with "Fewer than 2 regions available yet" unless at least two files load.

Its own settings differ from the single-state script: `K_CANDIDATES = range(3, 13)`,
`K_FINAL = 6`, silhouette band `0.15�?"0.35`, and it **re-standardises across the combined matrix**
before fitting. Its output filenames (`point_fingerprints.csv`, `bic_selection.csv`,
`cluster_assignments.csv`, `cluster_profiles.csv`) are un-suffixed and would collide with nothing
in the single-state set, but its `cluster_profiles.csv` is **not** the file `07`/`09` read.

The docstring cites plan **v2.0 §7** while every other Phase 2�?"8 script cites v3.0 �?" a visible
version lag in an unrun file.

---

## `05b_cluster_interactive.py` �?" explorer

Reads `cluster_assignments_uttarakhand.csv`, `cluster_profiles_uttarakhand.csv` and
`bic_selection_uttarakhand.csv`; writes Folium/Plotly HTML to
`data/processed/clustering/interactive/`. Features per the docstring: a cluster map whose popups
show the full soft-membership probability vector with boundary points (max membership below
`1.5/K`) drawn with a faint ring, a grouped-bar comparison of population-weighted profiles, a
population-share pie per regime, and BIC/silhouette K-selection curves.

**Its output directory is under the git-ignored `data/processed/` tree, so none of it is present
in this repository.**

---

## Phase 4 audit findings

1. **K = 5 exceeds the source files' own recommendation.** `README_PREPROCESSING.md` says
"realistically 2-4" for a 45-point single-state fit and warns that "a GMM fit on very few points
per component gets unstable." Cluster 2 has 3 points and cluster 3 has 7.
2. **Soft membership collapsed to 1.000 everywhere**, so the stated methodological reason for
choosing GMM over K-Means (partial membership at regime boundaries) is not realised in this run.
This should be reported rather than left implicit.
3. **No stability evidence exists.** With no bootstrap ARI, no model persistence and no external
classification, the only evidence for K = 5 is the (uncommitted) BIC/silhouette table and the
verification suite's 0.279 silhouette.
4. **Cluster ID stability depends solely on `random_state=42`.** There is no canonical relabelling
step, so any change to the signature matrix, sklearn version, or seed can permute cluster IDs
and silently invalidate downstream `cluster_id`-keyed joins in `07`, `08` and `09`.
5. **Cluster 2 is a 3-point regime carrying 3.2 % of population.** Any per-cluster statistic for it
�?" profile means, survivor counts, MCDM ranks �?" rests on three sampling points.
6. **`Tm_target_C` enters the clustering matrix as a zero-variance column.** Harmless but untidy;
worth excluding explicitly if the matrix is ever documented column-by-column.

# Source File 7: 07_PHASE_5_AUDIT(3).md
Source path: /mnt/data/07_PHASE_5_AUDIT(3).md
# 07 �?" Phase 5 Audit: PCM Database & Feasibility Filtering

**Scripts**: `PCM_data/PCM_data/01_preprocess.py`, `06_build_pcm_database.py`,
`07b_charging_feasibility.py` (optional), `07_feasibility_filter.py`

**Status**: **COMPLETE.** The PCM source CSVs are among the very few data files actually committed
in `era5-uttarakhand/`, so this phase is the most directly verifiable in the whole pipeline.

---

## PCM property cleaning �?" `PCM_data/PCM_data/01_preprocess.py`

### Method

MICE (chained-equation) imputation with a **Random Forest per column**, refined by **Predictive
Mean Matching**, so "every filled value is a REAL, previously-measured value donated from the most
physically-similar PCM �?" never a synthetic average."

IN_PATH      = data/PCM_Properties_55records_42_70C_dense.csv
OUT_LEAN     = data/PCM_Properties_cleaned_mice_pmm.csv
OUT_DETAILED = data/PCM_Properties_cleaned_mice_pmm_detailed.csv
N_ITER       = 8      # MICE refinement rounds
N_DONORS     = 3      # PMM donor pool size per missing cell
RANDOM_STATE = 42

The design rationale in the docstring targets a specific failure mode: several properties are
missing across an entire product line, so a naive nearest-neighbour fill has no donor. MICE+RF+PMM
avoids it because "the Random Forest for a given column trains ONLY on rows where that column is
actually observed �?" regardless of which product line they belong to", and PMM then ranks donors by
closeness of *model predictions*, not raw feature distance.

Every imputed numeric cell is donor-logged, "so cross-series borrowing can be verified directly in
the output rather than taken on faith."

**Documentation lag:** the docstring's worked example describes a dataset of "10/10" Rubitherm RT
rows and "8/8" Pluss savE/OM rows �?" an 18-row database. `IN_PATH` points at the 55-record file.
The narrative is stale; the code paths are current.

### Diagnostics produced (committed)

`PCM_data/PCM_data/data/`:
- `01_missingness_before_after.png`
- `02_cross_series_donor_audit.png`
- `03_imputed_vs_reported_sanity.png`
- `04_correlation_heatmap.png`
- `05_imputation_provenance.csv` (168 KB �?" the per-cell donor log)

---

## `06_build_pcm_database.py` �?" candidate database

### Input path resolution

INPUT_CSV = PROCESSED_DIR.parent.parent / "PCM_data" / "data" /
            "PCM_Properties_cleaned_mice_pmm_detailed.csv"

`PROCESSED_DIR` = `era5-uttarakhand/data/processed`, so `.parent.parent` = `era5-uttarakhand/`.
The resolved path is `era5-uttarakhand/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv`
�?" which **exists** (34,909 bytes).

`README.md` describes this as expecting "`PCM_data/` as a sibling folder **of this pipeline**",
but the code resolves to a **child** folder of `era5-uttarakhand/`. The code is what runs; the
README sentence is imprecise. Note also that the repository contains the file at **two** paths �?"
`PCM_data/data/�?�` (the one `06` reads) and `PCM_data/PCM_data/data/�?�` (the cleaner's own output
directory) �?" with identical size.

### Composition (verified directly against the committed CSV)

**55 rows �- 59 columns.**

| Manufacturer | Rows |
|---|---|
| Literature | **24** |
| Rubitherm Technologies | 14 |
| Pluss Advanced Technologies | 7 |
| PureTemp | 5 |
| PCM Products Ltd. | 4 |
| CrodaTherm | 1 |
| **Total** | **55** |

31 manufacturer rows + 24 literature rows �?" exactly as `06`'s docstring claims.

| `pcm_type` | Rows |
|---|---|
| Organic (RT-line) | 14 |
| Organic n-alkane | 11 |
| Organic | 7 |
| Organic PCM | 5 |
| Organic fatty acid | 4 |
| Organic bio-based PCM | 4 |
| Organic/composite blend | 3 |
| Organic blend | 3 |
| Organic/polymer blend | 2 |
| Organic commercial PCM | 1 |
| Organic/eutectic composite | 1 |

**Every row is organic.** There are no salt hydrates, no eutectic salts, and no inorganic PCMs of
any kind in the Uttarakhand database. This has a direct consequence in `07`: the
`corrosion_class` column is derived as `"check_manually" if "Inorganic" in pcm_type else
"low_organic"`, so it evaluates to `low_organic` for all 55 rows and carries zero discriminating
information.

### Property ranges (verified)

| Property | Range across the 55 rows |
|---|---|
| `Tm_melting` | **40.5 �?" 70.0 °C** |
| `latent_heat_melting` | **128 �?" 260 kJ/kg** |
| Rows inside the 42�?"70 °C absolute band | **54 of 55** (one row at 40.5 °C falls below) |
| Rows inside the [52, 65] °C melting window at `Tm_target = 57` | **29** |
| `cycles_tested_status` = "Reported by manufacturer" | **7** |
| `cycles_tested_status` = "Estimated via MICE-RF-PMM" | **48** |
| `flammability` = "Yes" | 45 |
| `flammability` = "No" | 10 |

### Imputation footprint (verified from the `*_imputed` flag columns)

**618 of 1,045** flagged property cells (55 rows �- 19 flagged properties) were imputed �?" **59.1 %**.
**All 55 rows** carry at least one imputed property, so `any_property_imputed` is `True` for every
candidate.

| Property | Rows imputed | Property | Rows imputed |
|---|---|---|---|
| `Tm_melting` | **0** | `TC_liquid` | 34 |
| `latent_heat_melting` | **3** | `TC_solid` | 39 |
| `density_solid` | 14 | `TC_both` | 36 |
| `density_liquid` | 14 | `cycles_tested` | 48 |
| `Cp_liquid` | 22 | `flammability` | 48 |
| `Cp_solid` | 24 | `appearance` | 48 |
| `Tm_freezing` | **29** | `volume_expansion` | 48 |
| `Tm_nucleation` | 54 | `max_op_temp` | 34 |
| `latent_heat_freezing` | 43 | `flash_point` | 39 |
| `heat_storage_Wh_kg` | 41 | | |

This is the single most important caveat for Phases 5 and 6:

- **`Tm_melting` is never imputed** (0/55) and **`latent_heat_melting` is imputed for only 3/55** �?"
the two properties that drive the melting-window filter and the latent-heat floor are almost
entirely measured.
- **`TC_W_mK` �?" an MCDM ranking criterion �?" is derived as `(TC_liquid + TC_solid)/2`, and those two
columns are imputed for 34 and 39 of 55 rows respectively.**
- **`cycles_confidence` �?" another MCDM ranking criterion �?" derives from `cycles_tested`, imputed
for 48 of 55 rows.**
- **`supercooling_K = Tm_C �^' Tm_freezing_C` �?" a feasibility filter �?" depends on `Tm_freezing`,
imputed for 29 of 55 rows.**
- **`rho_H_MJ_m3` �?" an MCDM criterion �?" depends on `density_solid`/`density_liquid`, imputed for
14 of 55 rows.**

### Column mapping and derived properties

out["Tm_C"]                = df["Tm_melting"]
out["latent_heat_kJ_kg"]   = df["latent_heat_melting"]
out["TC_W_mK"]             = (df["TC_liquid"] + df["TC_solid"]) / 2.0   # prefers per-phase
                                                                        # average over TC_both
out["supercooling_K"]      = out["Tm_C"] - out["Tm_freezing_C"]
out["n_properties_imputed"]= df[[c + "_imputed" for c in IMPUTABLE_PROPS]].sum(axis=1)
out["any_property_imputed"]= out["n_properties_imputed"] > 0
out["source"]              = "literature_MICE_RF_PMM_completed"      if manufacturer=="Literature"
                             else "manufacturer_datasheet_MICE_RF_PMM_completed"

rho_H_MJ_m3      = density_solid.fillna(density_liquid) �- latent_heat_kJ_kg / 1000
Cp_avg_kJ_kgK    = mean of Cp_liquid/Cp_solid with mutual fillna
cycles_confidence= log1p(cycles_tested) / log1p(max_cycles)          # NaN where cycles unknown
in_absolute_band = Tm_C.between(42.0, 70.0)
corrosion_class  = "check_manually" if "Inorganic" in pcm_type else "low_organic"

Family labels are assigned from `manufacturer` (Rubitherm RT, PLUSS savE, PCM Products, PureTemp,
CrodaTherm) or, for literature rows, from `pcm_type` (n-Alkane, Fatty acid, Composite, Blend,
Polymer blend, Eutectic composite, Organic PCM, Bio-based PCM, Commercial PCM, Organic, and
"Organic (RT-line)" �?' Rubitherm RT).

### Output

`data/processed/pcm/pcm_database_uttarakhand.csv`, sorted by `Tm_C`. Git-ignored �?" not committed.

---

## `07b_charging_feasibility.py` �?" optional regime-dependent Tm cap

### What it is for

The docstring is unusually candid, and the honesty note belongs in any write-up verbatim:

This is a **HEURISTIC PROXY, not a real collector thermal model.** A rigorous version needs the
cluster's 5th-percentile daily insolation fed through an actual collector efficiency curve
(`eta_th = F_R[S �^' U·(T_in �^' T_amb)/G]` �?�) �?" that's Phase 7 territory, not something to
improvise here under deadline pressure.

Its purpose is to break the constant-`Tm_target` degeneracy: "without it, every cluster shares the
same constant Tm_target and the same feasibility window, so every cluster gets an identical
survivor list (see 07's output �?" you'll have seen this if you ran it before this script)."

### Method

REFERENCE_GOOD_DAY_TEMP_C = 70.0     # stated assumption, not measured
MIN_ACHIEVABLE_TEMP_C     = 42.0
POOR_DAY_Z                = 1.28     # ~5th percentile under a normal approximation

poor_day_kt        = (kt_mean �^' 1.28 �- kt_std).clip(lower=0.05)
reliability_ratio  = (poor_day_kt / kt_mean).clip(0, 1)
achievable_temp    = 42 + reliability_ratio �- (70 �^' 42)
Tm_target_C_regime_capped = min(Tm_target_C, achievable_temp)

The 70 °C ceiling is described as "a generic collector-physics ceiling, not a Uttarakhand-specific
number", cited as "roughly consistent with Al-Mamun2023's cited FPC 25-100C operating band". This
is the **only external literature citation anywhere in `era5-uttarakhand/`'s pipeline code**.

The script also records an explicit Uttarakhand-specific reasoning step: "if anything, this state's
higher-altitude clusters see LESS reliable clear-sky access than the plains (more cloud/fog
persistence, not less), which is exactly what kt_mean/kt_std already capture per cluster."

### Side effect

It **overwrites `cluster_profiles_uttarakhand.csv` in place**, adding `poor_day_kt_estimate` and
`Tm_target_C_regime_capped`. `07_feasibility_filter.py` then silently prefers the capped column if
present:

tm_target = (prof["Tm_target_C_regime_capped"]
             if "Tm_target_C_regime_capped" in prof.index else prof["Tm_target_C"])

**Whether `07b` was run for the recorded Uttarakhand results is not available in the source
files.** The evidence points to **not run**: `07b`'s only purpose is to break the identical-
survivor-set degeneracy, and the observed run has an identical survivor set and an identical #1
PCM in all five clusters, which is precisely the outcome `07b` exists to prevent. This is inference
from the artefacts, not a direct observation.

---

## `07_feasibility_filter.py` �?" the feasibility filter

### Why filtering precedes ranking

This matters because MCDM is compensatory �?" a PCM with an unreachable melting point but great
latent heat can still score well in TOPSIS and be physically useless. Filtering first prevents
that.

### Constants

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX  = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7
CYCLES_FLOOR         = 300
SUPERCOOLING_MAX_K   = 8.0
MIN_SURVIVORS, MAX_RELAX_STEPS, RELAX_STEP_K = 5, 4, 2.0

### Filters actually applied

| # | Filter | Rule | Missing-data policy |
|---|---|---|---|
| 1 | Melting window | `Tm �^^ [Tm_target �^' 5, Tm_target + 8]` �?' **[52, 65] °C** at `Tm_target = 57` | �?" |
| 2 | Absolute band | `Tm �^^ [42, 70] °C` | �?" |
| 3 | Latent-heat floor | `L �?� 0.7 �- L_required` for that cluster | �?" |
| 4 | Cycling stability | `cycles_tested �?� 300` where reported | **retained and flagged** where unknown �?" "absence of data is not evidence of failure" |
| 5 | Supercooling veto | `|supercooling_K| �?� 8 K` where known | NaN passes through flagged, not excluded |

### Filters explicitly NOT applied (from the script's own docstring)

NOT applied (need data this project doesn't have yet �?" flagged as future work, not silently
skipped):
- Charging feasibility at the cluster's 5th-percentile insolation day (needs a full daily GHI
percentile per cluster, not just the mean in `cluster_profiles_uttarakhand.csv`)
- Corrosion veto against cluster HSI 75th percentile (needs a real `corrosion_class` per PCM;
the database currently only distinguishes "low_organic" vs "check_manually" for the one
inorganic PCM)
- Safety exclusion (no toxicity data in the current database)

Note the parenthetical about "the one inorganic PCM": the 55-row database this pipeline actually
consumes contains **zero** inorganic rows, so even that residual distinction is inert.

### Auto-relaxation

If a cluster keeps fewer than `MIN_SURVIVORS = 5`, the melting window is widened by `RELAX_STEP_K
= 2 K` and retried, up to `MAX_RELAX_STEPS = 4` (i.e. up to +8 K). If a cluster keeps more than 25,
that is reported but **not** narrowed �?" "Phase 6's ranking is what should separate them."

Status reported per cluster: `OK` for 5�?"25 survivors, `LOW` for < 5, `HIGH` for > 25.

### Output shape �?" an important detail

`07` writes **every** PCM �- cluster row, not just survivors:

result = filter_cluster(pcm_db, tm_target, l_required, window_relax=relax)   # all 55 rows
result.insert(0, "cluster_id", cid); �?�; all_rows.append(result)
full = pd.concat(all_rows, ignore_index=True); full.to_csv(OUT_FILE)

`feasibility_survivors_by_cluster.csv` therefore contains **55 �- 5 = 275 rows**, each carrying
per-filter booleans (`pass_melting_window`, `pass_absolute_band`, `pass_latent_heat`,
`pass_cycling`, `pass_supercooling`), the aggregate `passes_all`, and the window bounds
(`window_lo`, `window_hi`, `window_relax_applied`, `latent_heat_floor_used`).

The docstring calls this "the per-filter pass/fail detail kept alongside for your methodology
section's survivor-count table" �?" a deliberate design choice. **Consumers must filter on
`passes_all`.** `08_mcdm_ranking.py` does. Two of the four verification scripts do not (see
`11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`).

---

## Observed Phase 5 results

### Confirmed from committed artefacts

`data/plots/verify_feasibility/06_summary.png`:

Total Survivors: 275
Number of Clusters: 5
Avg Survivors per Cluster: 55.0
  Cluster 0: 55 PCMs   Cluster 1: 55 PCMs   Cluster 2: 55 PCMs
  Cluster 3: 55 PCMs   Cluster 4: 55 PCMs

These are **row counts, not survivor counts** �?" the verification script counted every row of the
275-row file. The same 55-per-cluster figure appears in
`data/plots/uttarakhand_objective1/05_pcm_survivors_per_cluster_interactive.html`, whose bars
encode the value 55 for each of clusters 0�?"4, because `generate_objective1_plots.py`'s `p05()`
uses `df.groupby("cluster_id").size()` without filtering `passes_all`.

**The actual `passes_all == True` count per cluster is not recorded in any committed artefact.**

### Reproduced survivor count

The four filters that do not depend on the un-committed `L_required` can be reproduced exactly
against the committed PCM CSV. Applying `Tm �^^ [52, 65]` **and** `Tm �^^ [42, 70]` **and**
`|Tm_melting �^' Tm_freezing| �?� 8 K` **and** `cycles_tested �?� 300`:

**29 candidates survive.** No candidate in the [52, 65] °C window fails on either supercooling or
cycling �?" every one of the 29 window-passers passes all four.

The fifth filter, `L �?� 0.7 �- L_required`, is non-binding: the observed cluster `Ta_mean` medians
(�?^ 13�?"25 °C, `06_PHASE_4_AUDIT.md`) put `L_required` in the �?^ 63�?"82 kJ/kg range and the floor at
�?^ 44�?"58 kJ/kg, while the *minimum* latent heat in the whole 55-row database is 128 kJ/kg.
`08_mcdm_ranking.py`'s own diagnostic text confirms this independently: "every candidate's latent
heat comfortably clearing L_required in every cluster."

**Therefore: 29 survivors in every cluster, identically.** Since 29 > 25, `07` would have printed
status `HIGH` for all five clusters and the auto-relaxation would never have triggered
(`window_relax_applied = 0.0` throughout).

### The 29 surviving candidates

| Name | Manufacturer | Tm (°C) | L (kJ/kg) | Supercool (K) | Cycles |
|---|---|---|---|---|---|
| n-Tetracosane (C24) | Literature | 52.0 | 255 | 2.7 | 1368 |
| PlusICE A52 | PCM Products Ltd. | 52.0 | 220 | 2.8 | 1447 |
| Paraffin/Expanded graphite (92 % paraffin) | Literature | 52.2 | 170 | 3.0 | 2000 |
| PureTemp 53 | PureTemp | 53.0 | 225 | 1.9 | 1686 |
| Myristic acid (C14) | Literature | 53.0 | 199 | 1.8 | 1686 |
| RT54HC | Rubitherm | 53.5 | 200 | 0.0 | 1474 |
| n-Pentacosane (C25) | Literature | 54.0 | 238 | 3.4 | 1404 |
| RT55 | Rubitherm | 54.0 | 170 | �^'2.5 | 2000 |
| Myristic acid/NBR-1.0 | Literature | 54.1 | **128** | 4.9 | 2000 |
| Myristic acid/NBR-0.5 | Literature | 54.6 | 142 | 4.1 | 2000 |
| **savE® OM55** | Pluss | 55.0 | 188 | 1.0 | 2000 |
| **Palmitic-stearic acid/Expanded graphite** | Literature | 55.2 | 176 | 0.3 | 2000 |
| **n-Hexacosane (C26)** | Literature | 56.5 | **256** | 0.3 | 1404 |
| RT57HC | Rubitherm | 56.5 | 240 | 0.0 | 1404 |
| **RT60** | Rubitherm | 58.0 | 160 | 0.0 | 2000 |
| **PureTemp 58** | PureTemp | 58.0 | 225 | �^'0.1 | 1620 |
| PlusICE A58 | PCM Products Ltd. | 58.0 | 215 | �^'0.2 | 1581 |
| n-Heptacosane (C27) | Literature | 59.0 | 236 | �^'0.7 | 1404 |
| CrodaTherm 60 | CrodaTherm | 59.8 | 217 | �^'1.7 | 1533 |
| Palmitic acid/Expanded graphite (80/20) | Literature | 60.9 | 148 | 0.1 | 2000 |
| PureTemp 60 | PureTemp | 61.0 | 220 | �^'0.5 | 1695 |
| RT65 | Rubitherm | 61.5 | 150 | 0.0 | 2000 |
| n-Octacosane (C28) | Literature | 61.6 | 253 | �^'0.7 | 1581 |
| PlusICE A62 | PCM Products Ltd. | 62.0 | 205 | 0.0 | 1581 |
| RT62HC | Rubitherm | 62.5 | 230 | 0.5 | 1404 |
| Palmitic acid (C16) | Literature | 62.6 | 198 | 0.5 | 1695 |
| PureTemp 63 | PureTemp | 63.0 | 206 | 1.0 | 1510 |
| n-Nonacosane (C29) | Literature | 64.0 | 240 | 1.7 | 1404 |
| RT64HC | Rubitherm | 64.0 | 250 | 1.5 | 1404 |

Bold rows are the five that appear in a Top-3 in Phase 6.

Survival rate: **29/55 = 52.7 %** of the database, identical in every cluster. Against
`VERIFICATION_METHODOLOGY.md`'s own success criterion of "10�?"50 % of candidates survive (not too
strict or loose)", this sits marginally above the upper bound.

---

## Phase 5 audit findings

1. **All five clusters have identical survivor sets.** This is a direct, unavoidable consequence
of `Tm_target = 57 °C` being constant and `L_required` being non-binding. Phase 5 contributes
**zero** climate-driven differentiation in this run. `07b_charging_feasibility.py` exists
precisely to fix this and appears not to have been run.
2. **Three of the five plan Table-12 filters are not implemented**, and the script says so in its
own docstring rather than hiding it: 5th-percentile-day charging feasibility, corrosion veto,
safety exclusion.
3. **The corrosion veto could not activate even if implemented** �?" every one of the 55 candidates
is organic, so `corrosion_class` is `low_organic` for all of them. `NEXT_STEPS.md`'s expectation
that "the corrosion veto [will] bite for high-monsoon-humidity Uttarakhand clusters" cannot be
realised with this database.
4. **`07`'s low-survivor warning string is stale**: it prints "your database (25 rows) is thin for
this" while the database is 55 rows. It would not have fired in this run anyway (29 > 5).
5. **Auto-relaxation never triggered** (29 �?� 5 in every cluster), so `window_relax_applied` is 0
throughout and the relaxation policy question is moot for this run.
6. **59.1 % of the PCM database's flagged property cells are MICE-RF-PMM estimates**, and three of
the five MCDM criteria (`TC_W_mK`, `cycles_confidence`, `rho_H_MJ_m3`) rest substantially on
them. The pipeline carries `any_property_imputed` and `n_properties_imputed` forward precisely
so this can be reported �?" `09_recommendation_cards.py`'s caveat text does mention it, but only
for "the literature-added candidates", which understates the scope: **all 55 rows** carry at
least one imputed property.
7. **The survivor count exceeds the pipeline's own upper comfort bound** (29 vs. the `OK` range of
5�?"25 and the verification suite's 10�?"50 % criterion). The melting window at
`Tm_target = 57 °C` (13 K wide) admits over half the database.

# Source File 8: 08_PHASE_6_AUDIT(3).md
Source path: /mnt/data/08_PHASE_6_AUDIT(3).md
# 08 �?" Phase 6 Audit: MCDM Ranking Engine

**Script**: `08_mcdm_ranking.py`

**Status**: **COMPLETE.** The Top-3 result for all five clusters, with per-method ranks and all
five candidates' properties, is fully recoverable from committed plot artefacts.

---

## Scope �?" what this script deliberately is and is not

The docstring is explicit that this is a reduced stack:

This is the "minimum viable MCDM stack" from your 4-day sprint plan: TOPSIS + GRA,
entropy-weighted per cluster, Borda-aggregated to a Top-3. **PROMETHEE II / VIKOR / CoCoSo and
the 5,000-draw Monte Carlo stability check are NOT implemented here** �?" they're real, documented
extensions �?�, add them if time remains, but this script alone already gives you a defensible,
falsifiable Top-3 per cluster.

So for Uttarakhand: **two methods, no Monte Carlo, no inclusion probabilities.**

## The Gaussian Tm-fitness transform

The script frames this as the correctness-critical step:

**THE ONE STEP EVERY PCM-MCDM PAPER GETS WRONG (plan v3.0 Section 9.2).** Melting temperature is
a TARGET-based criterion, not a benefit or cost �?" closer to `Tm_target` is better in both
directions. Feeding raw Tm into TOPSIS/GRA produces plausible-looking nonsense.

f_Tm(i) = exp( �^'(Tm_i �^' Tm_target)² / (2·�f²) ),   �f = SIGMA_TM = 4.0 K

`f_Tm` is then treated as an ordinary benefit criterion. �f = 4 K is cited to "plan v3.0 Section
9.2 �?" justified from HX approach temperature."

## Criteria

Five, all benefit-direction after the Tm transform:

| Criterion | Meaning | Source column |
|---|---|---|
| `f_Tm` | Gaussian melting-point fitness | computed from `Tm_C`, `Tm_target_C` |
| `latent_heat_kJ_kg` | gravimetric latent heat | PCM database |
| `rho_H_MJ_m3` | volumetric latent heat | `density �- L / 1000` |
| `TC_W_mK` | thermal conductivity | `(TC_liquid + TC_solid)/2` |
| `cycles_confidence` | log-scaled cycling stability | `log1p(cycles)/log1p(max_cycles)` |

Explicitly excluded, and stated as such: "**Corrosion class and cost are NOT included as ranking
criteria** �?" the database doesn't have reliable values for either yet �?� Say this explicitly in your
methodology rather than silently dropping them."

`cycles_confidence` NaNs are median-imputed **within each cluster's own candidate set**, with a
`cycles_confidence_imputed` boolean flag retained "(report, don't hide)". With
`cycles_tested` imputed for 48 of 55 database rows already, this flag will rarely fire �?" but the
underlying values are mostly MICE-RF-PMM estimates regardless (see `07_PHASE_5_AUDIT.md`).

## Weighting

ENTROPY_AHP_LAMBDA = 0.5

AHP_PRIOR = {                       # renormalised over the 5 criteria actually used,
    "f_Tm":              0.24/0.80, # from plan v3.0 Table 13's 8-criterion set with
    "latent_heat_kJ_kg": 0.20/0.80, # corrosion/cost/supercooling removed
    "rho_H_MJ_m3":       0.12/0.80,
    "TC_W_mK":           0.13/0.80,
    "cycles_confidence": 0.11/0.80,
}

w_final = 0.5 · w_entropy + 0.5 · w_ahp      # then renormalised to sum 1

Resolved AHP prior: `f_Tm` 0.300, `latent_heat_kJ_kg` 0.250, `TC_W_mK` 0.1625,
`rho_H_MJ_m3` 0.150, `cycles_confidence` 0.1375.

Shannon entropy weights are computed **per cluster from that cluster's own min-max-normalised
decision matrix**. The honesty note is explicit:

If you get 10 minutes with your guide for a real pairwise AHP matrix, replace `AHP_PRIOR` below
and rerun �?" until then this is an **honest placeholder, not a claimed AHP result**.

No pairwise elicitation was performed. There is no `AHP_PAIRWISE_MATRIX` variable in the
Uttarakhand script at all �?" only the fixed prior above.

## Normalisation and the two methods

Each criterion is min-max normalised to [0, 1] within the cluster's survivor set (constant columns
�?' 0.5), then:

**TOPSIS** �?" `norm = M / sqrt(Σ M²)` column-wise, weighted; ideal `v+ = max`, anti-ideal
`v�^' = min`; score `= s�^'/(s+ + s�^')`. All columns treated as benefit criteria, which is correct here
by construction.

**GRA** �?" reference = column max; `�" = |M �^' ref|`; coefficient
`(�"_min + ζ·�"_max)/(�" + ζ·�"_max)` with `GRA_ZETA = 0.5`; grade = weighted row sum.

Note: `delta_min`/`delta_max` are taken over the **whole matrix** (`delta.min()`, `delta.max()`),
not per-column. With min-max-normalised inputs `�"_min = 0` and `�"_max = 1` in almost every case,
so the coefficient reduces to `0.5/(�" + 0.5)` �?" a standard simplification, but worth stating if
GRA's formulation is written up.

## Consensus and agreement

borda = Σ over methods of (n �^' rank + 1)                    # higher = better
consensus_rank = borda.rank(ascending=False, method="min")  # ties share the lower rank

# Kendall's W over m = 2 rankers, n candidates
R = rowwise sum of ranks;  S = Σ(R �^' R�")²
W = 12·S / (m²·(n³ �^' n))

Kendall's W is written to every row as `kendall_w` and is reported per cluster. The script treats
low agreement as a finding, not a bug:

`[NOTE] Kendall's W < 0.6 for cluster(s) �?� �?" TOPSIS and GRA disagree meaningfully there. Per
plan v3.0 Section 9.5, this is a genuine, reportable finding (that regime's PCM choice is
ambiguous), not a bug to fix �?" discuss it rather than hide it.

## The constant-`Tm_target` diagnostic

`08` contains a purpose-built check for exactly the degeneracy this run exhibits:

top1_sets = topk[topk["consensus_rank"] == 1].groupby("cluster_id")["name"].first()
if top1_sets.nunique() == 1:
    print("[FINDING] Every cluster's #1 PCM is identical �?�")

and then offers two honest reporting options in full text:

(a) State it as a finding: Uttarakhand's climate regimes differ more in solar reliability/cloud
persistence than in delivery-relevant temperature, so a single PCM family serves the whole state
under the corrected `Tm_target` rule �?" differentiation would need to show up in Phase 7 physics
simulation (solar fraction per regime), not in the candidate list itself.
>
(b) Run `07b_charging_feasibility.py` (optional, heuristic regime-dependent upper bound on Tm)
before 07/08 to see if a real charging-feasibility constraint changes this.

Given the observed result (identical #1 in all five clusters), **this diagnostic fired.**

## Outputs

| File | Contents |
|---|---|
| `data/processed/pcm/mcdm_topk_by_cluster.csv` | Top-3 per cluster = **15 rows** |
| `data/processed/pcm/mcdm_full_scores_by_cluster.csv` | every survivor's full breakdown �?^ 5 �- 29 = **145 rows** |

Both are git-ignored. Clusters with fewer than 2 survivors are skipped with a message; that did
not occur here.

---

## Observed results

Recovered from three independent committed artefacts:
`data/plots/objective1/recommended_pcm_summary.html` (consensus ranks),
`data/plots/objective1/consensus_vs_topsis_agreement.html` (consensus vs TOPSIS rank pairs),
`data/plots/uttarakhand_objective1/07_bump_chart_ranks.html` (TOPSIS / GRA / consensus per
cluster), and `data/plots/uttarakhand_objective1/13_recommended_pcm_summary_interactive.html`
(per-candidate properties). All four agree.

### Clusters 0, 2 and 4 �?" identical Top-3

| Consensus rank | PCM | Family | Tm (°C) | L (kJ/kg) | ρ·H (MJ/m³) | TC (W/m·K) | Cycles | TOPSIS rank | GRA rank |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **RT60** | Rubitherm RT | 58.0 | 160 | 140.8 | 0.1695 | 2000 | 4 | 4 |
| **1** (tie) | **PureTemp 58** | PureTemp | 58.0 | 225 | 200.25 | 0.200 | 1620 | **1** | **7** |
| **3** | **n-Hexacosane (C26)** | n-Alkane | 56.5 | 256 | 197.12 | 0.238 | 1404 | **8** | �?" |

### Clusters 1 and 3 �?" identical Top-3

| Consensus rank | PCM | Family | Tm (°C) | L (kJ/kg) | ρ·H (MJ/m³) | TC (W/m·K) | Cycles | TOPSIS rank | GRA rank |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **RT60** | Rubitherm RT | 58.0 | 160 | 140.8 | 0.1695 | 2000 | 3 | 3 |
| **2** | **savE® OM55** | PLUSS savE | 55.0 | 188 | 175.78 | 0.130 | 2000 | 2 | 5 |
| **2** (tie) | **Palmitic-stearic acid / Expanded graphite** | Composite | 55.2 | 176 | 150.656 | 0.160 | 2000 | **1** | **6** |

All property values above are cross-checked against
`PCM_data/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` and match exactly.

### Frequency across clusters

From `data/plots/objective1/top3_inclusion_probability.html` (a **count** of clusters in which
each PCM reached the Top-3, not a probability �?" see the note below):

| PCM | Clusters in Top-3 |
|---|---|
| RT60 | **5** |
| PureTemp 58 | 3 |
| n-Hexacosane (C26) | 3 |
| savE® OM55 | 2 |
| Palmitic-stearic acid/Expanded graphite | 2 |

### Method agreement

From `data/plots/verify_ranking/06_summary.png` and
`data/plots/uttarakhand_objective1/08_method_rank_correlation_heatmap_interactive.html`
(identical values):

| Pair | Spearman ρ |
|---|---|
| TOPSIS vs GRA | **�^'0.930** |
| TOPSIS vs CONSENSUS | +0.376 |
| GRA vs CONSENSUS | �^'0.442 |

with `Number of ranked candidates: 15`, `Number of clusters: 5`, `Data completeness: 98.1 %`.

**Read these correlations carefully.** `verify_04_ranking.py` computes them across the **pooled
15 Top-3 rows from all five clusters at once**, not per cluster. They are therefore *not* the
per-cluster inter-method agreement statistic. The per-cluster statistic the pipeline itself
computes is Kendall's W, written to `mcdm_topk_by_cluster.csv` �?" and **that value is not
available in the source files**, because the CSV is git-ignored and no committed plot renders it.

Even with that caveat, the pattern within a single cluster is unambiguous from the bump chart. In
cluster 0, RT60 ranks 4th on TOPSIS and 4th on GRA, PureTemp 58 ranks **1st on TOPSIS and 7th on
GRA**, and n-Hexacosane C26 ranks **8th on TOPSIS**. In cluster 1, Palmitic-stearic/EG ranks
**1st on TOPSIS and 6th on GRA**. **TOPSIS and GRA disagree strongly, inside every cluster.**

### The consequence of that disagreement

Borda over two strongly-anti-correlated rankers produces near-ties. Concretely, in cluster 0 with
29 survivors:

- RT60: `(29 �^' 4 + 1) + (29 �^' 4 + 1) = 52`
- PureTemp 58: `(29 �^' 1 + 1) + (29 �^' 7 + 1) = 52`

�?" an exact tie, which is why both are reported at consensus rank 1 (`method="min"`). The same
mechanism produces the rank-2 tie in clusters 1 and 3. **The "winner" in each cluster is decided by
a tie, not by a margin.**

---

## What is absent from Phase 6

| Component | Status in `08_mcdm_ranking.py` |
|---|---|
| PROMETHEE II | **Not implemented** �?" listed in the closing text as a stretch goal ("~40 more lines") |
| VIKOR | **Not implemented** |
| CoCoSo | **Not implemented** |
| Copeland pairwise consensus | **Not implemented** (Borda only) |
| Monte Carlo weight/property perturbation | **Not implemented** �?" the closing text names a "5,000-draw" version as optional |
| Top-3 inclusion probability | **Not computed.** `generate_objective1_plots.py`'s `p09()` looks for `monte_carlo_stability.csv` or a `top3_inclusion_probability` column, finds neither, and prints "top3_inclusion_probability not found" �?" which is why **`09_monte_carlo_top3_probability.png` does not exist** in `data/plots/uttarakhand_objective1/`. |
| Analytical criterion contributions | **Not implemented** in `08` or `09` |
| AHP pairwise elicitation | **Not performed** �?" a fixed prior is used and labelled a placeholder |

---

## Phase 6 audit findings

1. **RT60 is consensus rank 1 in all five clusters.** This is the `[FINDING]` `08` is built to
detect, and it traces directly to `Tm_target = 57 °C` being constant. It is a correct
mathematical outcome of the inputs, not a bug �?" but it means Objective 1's "different PCM per
regime" claim is **not** demonstrated by this run.
2. **TOPSIS and GRA are strongly anti-correlated** (pooled Spearman �^'0.930), and the disagreement
is visible within individual clusters. Two methods that disagree this severely make a two-method
Borda consensus fragile: the consensus is essentially the arithmetic midpoint of two opposing
orderings.
3. **Every reported #1 (and the rank-2 slot in clusters 1/3) is a tie.** The tie-breaking is
`rank(method="min")`, i.e. positional, not substantive. Any write-up should present these as
joint recommendations rather than as a single winner.
4. **RT60 wins despite being mid-ranked by both methods.** It is 3rd�?"4th on TOPSIS and 3rd�?"4th on
GRA; it wins on Borda because it is the only candidate neither method places low. Its latent
heat (160 kJ/kg) is the **lowest** of the five Top-3 candidates and its ρ·H (140.8 MJ/m³) is the
lowest too; it leads on `cycles_confidence` (2000, the database maximum) and sits 1 K from
`Tm_target` on `f_Tm`. This is a defensible outcome but needs explaining, not asserting.
5. **No uncertainty quantification exists.** Without Monte Carlo, there is no evidence about how
stable these near-tied ranks are under small perturbations of the weights or of the
substantially-imputed `TC_W_mK` / `cycles_confidence` / `rho_H_MJ_m3` values.
6. **Kendall's W �?" the pipeline's own per-cluster agreement statistic �?" is not recoverable** from
any committed artefact. Given the pooled ρ of �^'0.930, it is very likely below `08`'s own 0.6
"ambiguous regime" threshold in every cluster, which would have triggered the `[NOTE]` block.
That cannot be confirmed from this repository.
7. **An earlier generation of this phase is preserved in the plot tree** with a completely
different Top-3 (RT54HC / RT55 / RT64HC) and a Spearman TOPSIS-vs-GRA of �^'1.000. See
`12_FINAL_READINESS_REPORT.md`.

# Source File 9: 09_PHASE_7_AUDIT(1).md
Source path: /mnt/data/09_PHASE_7_AUDIT(1).md
# 09 — Phase 7 Audit: Physics-Based Validation

**Script**: **none.**

**Status**: **NOT IMPLEMENTED.** There is no physics-validation script of any name in
`era5-uttarakhand/`.

---

## Purpose Phase 7 would have served

Independently test the Phase 6 MCDM ranking against a physics-based simulation, rather than against
another ranking method. The MCDM stack ranks PCMs on properties; a grey-box tank model ranks them
on *simulated delivered performance*. Comparing the two orderings (Spearman ρ) is the only way
Objective 1 can claim its ranking is more than internally consistent.

For Uttarakhand this would carry extra weight, because Phase 5 and Phase 6 return an **identical
survivor set and an identical #1 PCM in all five clusters** (see `07_PHASE_5_AUDIT.md` and
`08_PHASE_6_AUDIT.md`). `08_mcdm_ranking.py`'s own diagnostic names Phase 7 as the place that
differentiation would have to appear:

(a) State it as a finding: Uttarakhand's climate regimes differ more in solar reliability/cloud
persistence than in delivery-relevant temperature, so a single PCM family serves the whole state
under the corrected `Tm_target` rule — **differentiation would need to show up in Phase 7 physics
simulation (solar fraction per regime), not in the candidate list itself.**

## Confirmation that no script exists

| Evidence | Detail |
|---|---|
| Directory listing | `era5-uttarakhand/` contains `00_unzip_accum`, `00a`, `00b`, `01`, `01b`, `02`, `02b`, `03`, `03b`, `04`, `04b`, `04c` ×2, `04d`, `05`, `05b`, `05c`, `05d`, `05_cluster_regions`, `06`, `07`, `07b`, `08`, `09`, `comparison_plots_uttarakhand`, `generate_objective1_plots`, `verify_01`…`verify_04`, `config`. **There is no `10_*.py`.** |
| `README.md` pipeline map | Jumps directly from "PHASE 6 — MULTI-CRITERIA RANKING" to "PHASE 8 — FINAL OUTPUT" |
| `README.md`, explicit note | "(Phase 7 — physics-based validation via a grey-box lumped enthalpy tank model — **has no script in this repo yet**; see 'What's genuinely still open' at the bottom.)" |
| `NEXT_STEPS.md` status table | "7. Physics-Based Validation \| Grey-box lumped enthalpy tank model, Spearman rho vs. MCDM rank \| **Not written.**" |
| `08_mcdm_ranking.py` closing text | Lists "A minimal grey-box physics validation run per cluster's Top-1" under "What's still genuinely optional beyond this" |

## What the source files say should be built

`README.md`, "Notes / known limitations":

**Phase 7 (physics-based validation) has no script here.** A minimal single-PCM grey-box
lumped-enthalpy-tank simulation per cluster, compared against published annual-solar-fraction
benchmarks (54-84%), is enough to defensibly write "consistent with published benchmarks" — but
it isn't required for Objective 1 to stand as a working framework, and is explicitly an accepted
"future work" outcome if you don't get to it.

`NEXT_STEPS.md`, step 13:

If time allows: a **minimal** physics check — one grey-box lumped PCM tank simulation for just
the Top-1 PCM in 1-2 clusters, compared against published Table-16-style benchmarks (annual solar
fraction 54-84%). A single calibration run per cluster is enough to write "consistent with
published benchmarks" honestly. **If you can't fit this in, say explicitly in the paper that
physics validation is future work — an accepted outcome per the plan doc, not a weakness you
need to hide.**

`NEXT_STEPS.md`, "What's genuinely still open":

**Physics validation (Phase 7)** is not written. If you have a spare half-day, a single-PCM
grey-box run per cluster against the Table 16 benchmark ranges (annual solar fraction 54-84%) is
enough to write "consistent with published benchmarks" honestly — full validation across every
cluster is not required.

So the specification is consistent across three files: a **lumped-enthalpy grey-box tank model**,
run per cluster (or for 1–2 clusters at minimum), for at least the Top-1 PCM, with the annual solar
fraction checked against a published **54–84 %** band.

The only numeric anchor stated anywhere in `era5-uttarakhand/` is that **54–84 % band**, attributed
to "Table 16" of the plan document. The plan document is not present in the folder, so the band's
underlying references are **not available in the source files**.

## The one adjacent piece that does exist

`07b_charging_feasibility.py` is the closest thing in the repository to a physics calculation, and
its docstring explicitly marks the boundary:

This is a **HEURISTIC PROXY, not a real collector thermal model.** A rigorous version needs the
cluster's 5th-percentile daily insolation fed through an actual collector efficiency curve
(`eta_th = F_R[S − U·(T_in − T_amb)/G]`, as several of your literature summaries already have) —
**that's Phase 7 territory**, not something to improvise here under deadline pressure.

That efficiency-curve form is the most concrete statement anywhere in the folder of what the Phase 7
collector model would need. `07b` deliberately does not implement it. See `07_PHASE_5_AUDIT.md`.

## What the plot layer expected, and did not get

Two figure scripts were written against a Phase 7 that does not exist:

**`generate_objective1_plots.py`** declares
`PHYS_VAL = data/processed/pcm/physics_validation_results.csv` and its plot 11
("Agreement plot: physics rank vs consensus rank") checks for a
`hours_target_met_per_year` column. The file does not exist, so `p11()` falls through to its
degenerate branch and plots **TOPSIS rank against consensus rank** instead of simulated performance
against consensus rank. The committed `11_agreement_plot.png` and
`11_agreement_plot_interactive.html` are therefore **not** physics-validation figures despite their
titles saying "Simulated Performance vs MCDM Consensus Rank".

**`comparison_plots_uttarakhand.py`** declares the same `PHYS` path and its comparison 6 ("Physics
validation: hours_target_met vs MCDM rank") is skipped. That script never produced output at all —
see `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md` for its separate path defect.

**`generate_objective1_plots.py`'s plot 12** ("Tank temperature / melt-fraction profile") produces
`12_tank_temperature_melt_fraction.png` and `.html`, both committed. These are **synthetic
illustrations, not simulation output**:

Ta   = 28 + 14*np.sin((hrs-6)*np.pi/12)
tank = Tm - 6 + 18*np.sin((hrs-6)*np.pi/12)
melt = np.clip((tank - Tm + 5)/10, 0, 1)

Hard-coded sinusoids over a 24-hour axis, with only `Tm` taken from the real per-cluster
`Tm_target_C`. The ambient amplitude (28 ± 14 °C) does not correspond to any Uttarakhand cluster
profile. **These figures must not be presented as physics-validation results.**

## Consequence for Objective 1's claims

| Claim | Supportable for Uttarakhand? |
|---|---|
| "The MCDM ranking was tested against an independent physics simulation" | **No** — no simulation was run |
| "Annual solar fraction is consistent with published 54–84 % benchmarks" | **No** — no solar fraction was computed |
| "Different regimes favour different PCMs" | **No** — and Phase 7 was the designated place to demonstrate it |
| "The framework is complete through Phase 6, with Phase 7 as declared future work" | **Yes** — this is exactly how all three source files frame it |

The last row is the honest position, and it is the position the source files themselves take. It
should be stated in the paper as a declared limitation, not omitted.

## Inputs Phase 7 would need (all present or recoverable)

| Requirement | Availability |
|---|---|
| Per-cluster medoid point | `09_recommendation_cards.py` already computes an approximate medoid |
| Real daily climate for that point | `daily_aggregates_uttarakhand.csv` — 45 × 3,653 point-days from `02b` |
| Daily GHI, Tmax, Tmin | `GHI_daily_kWh`, `Ta_max_true`, `Ta_min_true` — all present in that file |
| Candidate PCM properties | `pcm_database_uttarakhand.csv` — `Tm_C`, `latent_heat_kJ_kg`, `density_*`, `Cp_*`, `TC_W_mK` |
| Survivors to simulate | `feasibility_survivors_by_cluster.csv`, or the 15 Top-3 rows from `08` |
| Ranking to correlate against | `mcdm_topk_by_cluster.csv` / `mcdm_full_scores_by_cluster.csv` |

**Every input a Phase 7 script would need already exists on disk.** The blocker is the script
itself, not the data.

One caveat that would apply to any Phase 7 built on this pipeline: a collector model driven by
`era5_GHI` would inherit the magnitude anomaly documented in `04_PHASE_2_AUDIT.md` Part A.3. It
should be driven by the **NASA POWER** `GHI_daily_kWh` from `02b` instead, which is the source the
canonical signature column already uses.

## Problems / risks

1. **Phase 7 is absent**, so Objective 1's ranking has no external validation of any kind for
Uttarakhand — only internal method agreement, which is itself poor (pooled TOPSIS-vs-GRA
Spearman ρ = −0.930, see `08_PHASE_6_AUDIT.md`).
2. **Two committed figures carry physics-sounding titles without physics behind them.**
`11_agreement_plot.png` silently plots TOPSIS-vs-consensus rank, and
`12_tank_temperature_melt_fraction.png` is a hard-coded sinusoid. Both are titled as if they
were simulation results. Neither should appear in a paper as validation evidence.
3. **The identical-across-clusters Phase 5/6 result has nowhere to be resolved.** `08`'s own
diagnostic nominates Phase 7 as the place regime differentiation would show up; with Phase 7
absent, the finding stands unresolved.
4. **The 54–84 % benchmark band has no traceable citation** inside `era5-uttarakhand/` — only a
reference to "Table 16" of an absent plan document.

## Status

**NOT IMPLEMENTED — and correctly declared as such in three separate source files.** This is not a
silent omission; `README.md`, `NEXT_STEPS.md` and `08_mcdm_ranking.py` all name it, specify what a
minimal version would look like, and state that recording it as future work is an accepted outcome.
Every input a minimal implementation would need is already on disk.

# Source File 10: 10_PHASE_8_AUDIT(1).md
Source path: /mnt/data/10_PHASE_8_AUDIT(1).md
# 10 — Phase 8 Audit: Explanation & Final Output (Recommendation Cards)

**Script**: `09_recommendation_cards.py`

**Status**: **CODE COMPLETE; OUTPUT NOT COMMITTED.** The script exists and is correct; its output
`recommendation_cards.md` lives under the git-ignored `data/processed/pcm/` tree and is **not
present in this repository**. Whether it was run cannot be confirmed directly — but every one of
its four inputs demonstrably exists, so nothing would have blocked it.

---

## Purpose

Turns everything Phases 4-6 produced into one recommendation card per cluster — this becomes your
results section directly (Table 18 in the plan doc). **Pure aggregation script, computes nothing
new.**

That last sentence is the key property: `09` introduces no new modelling assumption. Every number
on a card traces to `05`, `07` or `08`.

## Inputs and the early-exit guard

PROFILE_FILE   = data/processed/clustering/cluster_profiles_uttarakhand.csv     # from 05
ASSIGN_FILE    = data/processed/clustering/cluster_assignments_uttarakhand.csv  # from 05
TOPK_FILE      = data/processed/pcm/mcdm_topk_by_cluster.csv                    # from 08
SURVIVORS_FILE = data/processed/pcm/feasibility_survivors_by_cluster.csv        # from 07

for f in (PROFILE_FILE, ASSIGN_FILE, TOPK_FILE, SURVIVORS_FILE):
    if not f.exists():
        print(f"\n  ERROR: {f} not found — run the earlier phase scripts first.")
        return

All four are checked **before** anything is written, so a missing input produces a clear message
and **no partial output** — a design point `README.md` calls out explicitly:

`09_recommendation_cards.py` reads four files at once … and exits early with a clear "run the
earlier phase scripts first" message if any are missing — no partial output gets written.

## Output

`data/processed/pcm/recommendation_cards.md` — one markdown section per cluster, written with
`OUT_FILE.write_text("\n".join(lines), encoding="utf-8")`.

## Card structure (per cluster)

| Element | Source | Notes |
|---|---|---|
| Heading `## Cluster {cid}` | `cluster_profiles` | iterated `sort_values("cluster_id")` |
| **Points in regime** | `prof["n_points"]` | |
| **Population covered** | `prof["total_population_covered"]` | printed only if non-NaN |
| **Approx. medoid point** | computed from `cluster_assignments` | nearest member to the cluster's mean lat/lon |
| **Climate signature table** | `cluster_profiles`, `SIGNATURE_DISPLAY` list | population-weighted means, 3 dp |
| **Derived targets** | `prof["Tm_target_C"]`, `prof["L_required_kJ_per_kg"]` | |
| **Candidates screened** | `(survivors[cluster]["passes_all"]).sum()` | **correctly filters on `passes_all`** |
| **Top-3 PCM table** | `mcdm_topk_by_cluster` | rank, name, family, Tm, latent heat, TOPSIS, GRA |
| **Kendall's W + interpretation** | `cluster_top["kendall_w"].iloc[0]` | thresholded, see below |
| **Caveats** | hard-coded text | see below |

`SIGNATURE_DISPLAY = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac", "CCI", "HDD18",
"CDD24", "RH_mean", "HSI", "monsoon_index"]` — 11 of the 18 signature indices, chosen for
readability. Any column absent or NaN in the profile row is silently skipped.

### The medoid computation, and a fixed bug

medoid = members.loc[(members[["lat", "lon"]]
                       - members[["lat", "lon"]].mean()).pow(2).sum(axis=1).idxmin()]

The in-code comment records a real defect that was found and fixed:

`.loc[]`, not `.iloc[]` — `idxmin()` returns members' original ROW LABEL (inherited from the
un-reset `assign` dataframe this was boolean-filtered from), not a 0..len(members)-1 position.
Using `.iloc[]` here throws IndexError as soon as that label happens to exceed len(members),
which is exactly what you hit.

This is the only bug-fix history recorded anywhere in the Uttarakhand pipeline's code comments, and
it is worth citing as evidence of the project's self-audit process. Note the label "**Approx.**
medoid": this is the member nearest the cluster's **geographic centroid**, not a medoid in the
climate-signature space.

### Kendall's W interpretation thresholds

agreement_note = ("strong agreement"                                            if kw >= 0.8 else
                  "moderate agreement — discuss the disagreement"               if kw >= 0.6 else
                  "weak agreement — this regime's PCM choice is genuinely ambiguous")

This matches `08_mcdm_ranking.py`'s own 0.6 threshold for printing its `[NOTE]` block.

**The actual Kendall's W values for the Uttarakhand run are not available in the source files** —
`mcdm_topk_by_cluster.csv` is git-ignored and no committed plot renders the column. Given the
pooled TOPSIS-vs-GRA Spearman ρ of **−0.930** (`08_PHASE_6_AUDIT.md`), the "weak agreement —
genuinely ambiguous" branch is the likely one for every cluster, but that cannot be confirmed from
this repository.

### Empty-Top-3 branch

If a cluster has no ranked candidates, the card prints:

**No ranked candidates** — this cluster had <2 feasibility survivors. Widen the PCM database or
relax the melting window for this Tm_target before finalising.

This branch did not fire: all five clusters have 29 survivors.

### Caveats block (hard-coded, printed on every card)

**Caveats:** thermal conductivity / density / specific heat not reported in the source data for
the literature-added candidates (see `06_build_pcm_database.py`); cycling and corrosion vetoes
only partially applied (see `07_feasibility_filter.py`'s docstring for what wasn't checked yet).

Both halves are accurate but the first **understates the scope**. The verified imputation footprint
(`07_PHASE_5_AUDIT.md`) is that **618 of 1,045 flagged property cells across the whole 55-row
database are MICE-RF-PMM estimates, and all 55 rows carry at least one imputed property** — not
just "the literature-added candidates". Specifically: `TC_liquid` imputed in 34/55 rows,
`TC_solid` in 39/55, `cycles_tested` in 48/55, `Tm_freezing` (→ `supercooling_K`) in 29/55,
`density_solid` in 14/55. Three of the five MCDM criteria rest substantially on estimated values.

The database carries `any_property_imputed` and `n_properties_imputed` per row precisely so a card
could state this exactly, and `08`'s output carries `cycles_confidence_imputed` per candidate.
**Neither is read by `09`.** Surfacing them per recommended PCM would be a small change with real
explainability value.

## What the reconstructed cards would contain

Assembled from the committed artefacts documented in `06_PHASE_4_AUDIT.md`, `07_PHASE_5_AUDIT.md`
and `08_PHASE_6_AUDIT.md`. This is a reconstruction of what `recommendation_cards.md` holds, not a
transcript of it.

| Field | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Points in regime | 12 | 9 | **3** | 7 | 14 |
| Population covered | 3,432,283 | 2,451,043 | 330,779 | 2,541,919 | 1,719,687 |
| `Ta_mean_proxy` (°C, approx.) | 22.8 | 19.0 | **13.4** | **25.0** | 18.2 |
| `Tm_target_C` | 57.0 | 57.0 | 57.0 | 57.0 | 57.0 |
| `L_required` (kJ/kg, bounded estimate) | ≈ 63–82 across all clusters — exact values not available in the source files |
| Candidates screened (`passes_all`) | 29 | 29 | 29 | 29 | 29 |
| Top-3 #1 | RT60 | RT60 | RT60 | RT60 | RT60 |
| Top-3 #2 | PureTemp 58 (tied #1) | savE® OM55 | PureTemp 58 (tied #1) | savE® OM55 | PureTemp 58 (tied #1) |
| Top-3 #3 | n-Hexacosane (C26) | Palmitic-stearic acid/EG (tied #2) | n-Hexacosane (C26) | Palmitic-stearic acid/EG (tied #2) | n-Hexacosane (C26) |
| Kendall's W | not available in the source files |

**Every card names RT60 as the #1 recommendation, and every #1 is a Borda tie.** Clusters 0/2/4
share one card body and clusters 1/3 share another; only the point count, population, medoid and
climate-signature table differ between them.

## The finding a Phase 8 write-up must carry

`08_mcdm_ranking.py` detects the degeneracy and prints it, but **`09` does not propagate it into
the cards.** A reader of `recommendation_cards.md` alone sees five cards with the same top pick and
no explanation of why. The `[FINDING]` text from `08` — and the two honest reporting options it
offers — belongs in the cards, or at minimum in the paper section built from them:

Every cluster's #1 PCM is identical (`RT60`). This is a direct consequence of `Tm_target` being
held constant across all clusters (plan v3.0 Section 6.3's design rule) combined with every
candidate's latent heat comfortably clearing `L_required` in every cluster. **It is NOT a bug.**

## Dependencies

`pandas` only. No numerical or plotting libraries — consistent with "pure aggregation script,
computes nothing new."

## Validation

| Check | Result |
|---|---|
| All four inputs present before writing | **Implemented** — early exit, no partial output |
| `passes_all` correctly filtered for the survivor count | **Yes** — `09` is one of the few consumers that does this correctly |
| Medoid index bug | **Found and fixed**, with the reason recorded in-code |
| Cluster-ID consistency across the four inputs | **Not checked.** There is no provenance or fingerprint check; `09` joins on `cluster_id` and trusts it. |
| NaN-safe profile printing | **Yes** — `prof[col] == prof[col]` guards, and a `total_population_covered` NaN guard |
| Output committed | **No** |

## Problems / risks

1. **The output is not committed**, so the actual card content — and in particular the per-cluster
Kendall's W and `L_required` values — cannot be verified from this repository. These are the two
numbers most needed for a results section.
2. **No cross-phase provenance check.** `09` joins `cluster_profiles`, `cluster_assignments`,
`mcdm_topk` and `feasibility_survivors` on `cluster_id` with no verification that they came from
the same `05` run. Because `05_cluster_uttarakhand.py` has **no canonical cluster relabelling**
(see `06_PHASE_4_AUDIT.md`), a re-run of `05` with a different `K_FINAL`, a changed signature
matrix, or a different sklearn version can permute cluster IDs and silently produce cards that
mix regimes. `README.md` warns about the ordering ("if you re-run `05` … re-run `06`→`09` again
too, or your PCM rankings will be filtered against a stale set of clusters") but nothing
enforces it.
3. **The caveat text understates the imputation scope** — "the literature-added candidates" versus
the verified reality that all 55 rows carry at least one imputed property and three of the five
MCDM criteria rest substantially on estimates.
4. **`any_property_imputed`, `n_properties_imputed` and `cycles_confidence_imputed` are available
per candidate and are not surfaced** on the cards.
5. **The identical-#1 finding is not propagated** from `08`'s console output into the cards.
6. **Every recommended #1 is a Borda tie**, and the cards render `consensus_rank` without noting
the tie — a reader sees "1" and "1" in clusters 0/2/4 without explanation.
7. **No analytical criterion-contribution breakdown.** `mcdm_full_scores_by_cluster.csv` is written
by `08` precisely so a card can show per-criterion contributions — `08`'s docstring says "keep
this — it's what a recommendation card's 'criterion contributions' field needs" — but `09`
never reads that file.
8. **Phase 7 results have no slot on the card**, because Phase 7 does not exist
(`09_PHASE_7_AUDIT.md`).

## Status

**CODE COMPLETE, OUTPUT UNVERIFIED.** The script is well constructed: it validates all inputs
up-front, refuses to write partial output, correctly filters on `passes_all`, handles NaNs, and
carries a recorded bug fix. Its shortcomings are all about what it does *not* say — the imputation
scope, the tied ranks, the identical-#1 finding, and the per-criterion contributions it already has
the data for.

# Source File 11: 11_LITERATURE_MAPPING.md
Source path: /mnt/data/11_LITERATURE_MAPPING.md
# 17 — Literature Mapping

**Documentation note (2026-09-02):** Standalone concept files `10_TEMPORAL_PROCESSING.md` and
`11_SPATIAL_PROCESSING.md` have been consolidated into `03_PHASE_1_AUDIT.md` and
`04_PHASE_2_AUDIT.md` respectively, with full justification for each method. The research gap
mapping has been moved into `00_MASTER_OVERVIEW.md` under the new "Research gaps addressed
(N1–N6 novelty mapping)" section. This file (`17_LITERATURE_MAPPING.md`) remains the authoritative
reference for all methodology-component-to-source mappings.

## Method

Sources checked, in priority order: (1) `PCM-Selection-ML-model/Sources/` — 21 full paper summaries
(the project's own curated, previously-read literature), (2) the framework doc's own §15 IEEE
reference list, (3) `references.bib` (37 entries) and `.claude/references.md` (24 unique
ResearchRabbit entries + a duplicate of `references.bib`). Every citation below was checked against
one of these three, not asserted from general training knowledge alone, except where explicitly
marked "not independently verified in this project's bibliography" — those are standard, correct
citations for well-known methods (e.g. Reda & Andreas SPA, Ineichen clear-sky) that were not found in
this specific project's reference files during this audit and should be added before formal
submission.

## Methodology-component → implementation → literature matrix

| Component | Implementation | Supporting source | Strength |
|---|---|---|---|
| ERA5 reanalysis as climate backbone | Phase 1–2 | Hersbach et al. (2020), *QJRMS* — per framework doc §15 | Strong (product-defining citation) |
| NASA POWER as cross-check | Phase 1–2 | NASA POWER project documentation — per framework doc §15 | Strong |
| Solar position (SPA) | `pvlib`, `00b`/`02` | Reda & Andreas (2004), *Solar Energy* 76(5) | Strong, but not confirmed present in `references.bib`/`.claude/references.md` — add before submission |
| Clear-sky model (Ineichen) | `02_combine_rajasthan.py` | Ineichen & Perez (2002), *Solar Energy* 73(3) | Strong, not confirmed in project bib — add |
| pvlib software | throughout | Holmgren, Hansen & Mikofski (2018), *JOSS* 3(29) | Strong, per framework doc §15 |
| Humidity-stress index (HSI_sunrise) | `signature_lib.py` | Thom (1959), *Weatherwise* 12(2) — THI, correctly cited in-code | Strong, directly attributable |
| Night-discharge design basis (L_required) | `04_climate_signature_rajasthan.py` | Avargani et al. (2021), *J. Energy Storage* | Strong, direct citation with a corrected units interpretation (see `05_PHASE_3_AUDIT.md`) |
| Worst-month sizing cap (Tm_target_capped_C) | same | Durin et al. (2018), "Worst Month and Critical Period Methods..." | Strong, direct and appropriately applied |
| Field-evidence sanity check for the cap | same | Nahar (2003), tested at Jodhpur | Direct, present as a bare citation in `.claude/references.md` — needs a complete BibTeX entry |
| T_mains lag estimate | same | **none** — explicitly documented in-code as not derived from any published correlation | **Weak / open gap** — see recommendation below |
| GMM clustering, k-selection heuristics | `05_cluster_rajasthan.py` | *Building and Environment* (2024) India climate-classification study (silhouette 0.21 vs −0.2 NBC); a 2026 thermal-comfort clustering study (mean silhouette 0.235) | Moderate — cited with enough specificity to be traceable but full BibTeX entries not located in this pass |
| External classification validation | `05_cluster_rajasthan.py` | Beck et al. (2018), *Scientific Data* 5, DOI:10.1038/sdata.2018.214 (Köppen-Geiger) | Strong citation, **now wired in for real (2026-08-11)** — ARI=0.19/NMI=0.32 vs. GMM. NBC/ECBC remains unwired. |
| PCM candidate band (42–70°C) | Phase 5 | Framework doc Table 5, cross-referenced against Singh et al. (2025), *Solar Energy Materials and Solar Cells* 293 (states 40–70°C as the optimal SWH PCM band) | Strong, closely matching independent literature |
| PCM property values (RT-series validation) | PCM database | Martínez et al. (2025), *Heliyon* 11 — directly measures/validates RT54HC/RT55/RT64HC, the same product family in this project's database, and finds large literature-vs-measured discrepancies for some | Strong and directly relevant — should be cited as a caveat on manufacturer-datasheet trust, not just a property source |
| Gaussian Tm-fitness σ=4K | `08_mcdm_ranking_rajasthan.py` | Framework doc §9.2 only — "not independently literature-calibrated," per the code's own docstring | Weak/self-sourced — state plainly, do not overclaim external validation |
| PROMETHEE II q/p thresholds | same | Framework doc §9.4 | Implementation-defined, documented as such |
| TOPSIS unit-test fixture | same | Oluah (2020) — 72.12% thermal-conductivity domination cited as a cautionary comparator | Direct, used correctly as both a regression-test anchor and an interpretive comparator |
| MCDM method family (TOPSIS/PROMETHEE/VIKOR/GRA) | same | No dedicated MCDM-methodology paper found cross-referenced in `references.bib`/`.claude/references.md` | **Gap** — these are standard, well-established methods, but a formal write-up should cite each method's originating paper (Hwang & Yoon 1981 for TOPSIS; Brans & Vincke 1985 for PROMETHEE; Opricovic 1998 for VIKOR; Deng 1982 for GRA) |
| PCM database imputation (MICE-style + RF + custom PMM-like donor blend) | `PCM_data/01_preprocess.py` | No dedicated imputation-methodology paper found in this project's bibliography | **Gap** — cite the general MICE framework (van Buuren & Groothuis-Oudshoorn 2011) and note explicitly that the donor-blend step is a project-original variant, not textbook PMM (see `07_PHASE_5_AUDIT.md`) |
| Quantile mapping (bias correction) | `03b_agreement_analysis.py` | No dedicated citation found in this project's bibliography | **Gap** — cite Cannon et al. (2015) or an equivalent standard reference |
| Phase 7 lumped-enthalpy ODE structure (3-phase pre-melt/melt/post-melt) | `physics_lib.py` | Barqawi, F. A. (2025), *Muthanna J. Eng. Technol.* 13(3):1-14, doi:10.52113/3/eng/mjet/2025-13-03/-1-14 | Strong — already in `Sources/` (read in full pre-Phase-7), DOI independently re-verified this session, equations used directly (not paraphrased from memory) |
| Phase 7 model-class justification (lumped PCM-in-tank, the basis for TRNSYS Type 860) | same | Bony, J. & Citherlet, S. (2007), *Energy and Buildings* 39(9):1065-1072 | Strong — independently confirmed via web search this session (not previously in `Sources/`), cited for model-CLASS justification only, not claimed as a literal Type 860 replication |
| Phase 7 draw-profile SHAPE (two-peak, morning+evening) | same | ASHRAE Standard 90.2 §8.9.4 Table 8-4, built on Perlman & Mills (1985), *ASHRAE Transactions* | **Partial/honest gap** — the qualitative two-peak shape is real and cited, but the exact 24 published hourly fractions were not independently retrievable this session; `physics_lib.py`'s own docstring flags this explicitly as a parametric reconstruction of the documented SHAPE, not a verbatim reproduction of the standard's table — do not cite specific hourly percentages from this pipeline as if reproducing that table |
| Phase 7 draw-total volume (300 kg/day) | same | Avargani et al. (2021) — same citation Phase 3 already uses for `L_required_kJ_per_kg`'s 300 L/7h basis, reused as the FULL DAY total rather than a night-only ceiling (a different, explicitly stated use of the same cited figure) | Strong, cross-phase-consistent citation reuse |
| Phase 7 collector parameters (A_c, h_c, efficiency, PCM bed surface-to-volume ratio) | same | Barqawi (2025), same paper as above | Strong for the ORIGINAL values; **recalibrated** during Phase 7's own calibration pass (collector area, implicit loss coefficient) — recalibration reasoning documented in `physics_lib.py`'s CALIBRATION section, not silently changed |

## Sources/ folder papers — relevance summary (21 papers read in full)

The 21 papers in `Sources/` are overwhelmingly **PCM-material / PCM-SWH-system / AI-for-thermal-systems**
domain literature (Abdellatif 2025, Al-Mamun 2023, Assareh 2023, Barghi 2026, Barqawi 2025, Chen 2025,
Chopra 2023, Duraivel 2025, Eldokaishi 2022, Emami 2026, Ghodusinejad 2026, Hamzat 2025, Kou 2025, Liu
2025, Mansouri 2025, Martínez 2025, Mohammed 2025, Odoi-Yorke 2025, Singh 2025, Terfai 2025, Yan 2025)
— they substantiate this project's PCM-selection rationale, MCDM-in-PCM-context precedents (Assareh
2023's TOPSIS/LINMAP/AHP; Chen 2025's GRA), and ML-for-thermal-systems framing well. **None of them are
methodology-support papers for ERA5/reanalysis handling, pvlib solar geometry, quantile mapping, or
MCDM statistical foundations specifically** — this is a real, confirmed gap (searched by title
keyword against both `references.bib` and `.claude/references.md`; only two incidental matches, Chen
2025 for "grey relational" and Chopra 2023 for "Monte Carlo," both already counted above). Köppen
classification is now covered (Beck et al. 2018, above). **Barqawi (2025), already in this list as a
PCM-SWH domain paper, now ALSO serves as a direct methodology-support citation for Phase 7's
lumped-enthalpy simulation equations** — its equations are used directly, not just cited for
framing.

## Recommendation

Before formal submission, add a dedicated "Methods & Tools" reference block covering: Reda & Andreas
(2004), Ineichen & Perez (2002), Holmgren et al. (2018), Hwang & Yoon (1981), Opricovic (1998), Deng
(1982), Brans & Vincke (1985), van Buuren & Groothuis-Oudshoorn (2011), and a quantile-mapping
reference (e.g. Cannon et al. 2015) — none of these are currently in `references.bib` or
`.claude/references.md`, and all are directly load-bearing for claims this pipeline actually makes.
Also complete the bare Nahar (2003) citation note into a full BibTeX entry, and add Durin et al.
(2018) and a formal Thom (1959) entry, since both are directly quoted/used in code but not present in
either bibliography file. **New, added 2026-08-11**: add Bony & Citherlet (2007) — the Phase 7
model-class justification, independently confirmed via web search this session but not yet a formal
BibTeX entry in either bibliography file (Barqawi 2025 is already present via `Sources/`).
