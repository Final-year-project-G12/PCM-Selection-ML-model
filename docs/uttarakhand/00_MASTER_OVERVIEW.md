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

```
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
```

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
