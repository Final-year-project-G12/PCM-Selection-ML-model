# 00 — Master Overview: ERA5 Rajasthan Climate → PCM Selection Pipeline

## Project objective

Final-year B.Tech CSE project (Group 12, Amrita School of Engineering, Guide: Dr. T. Deepika):
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water
Heating."** Objective 1 (the scope of this audit) builds a **climate-region-aware PCM
recommendation framework**: turn 10 years of reanalysis climate data into population-weighted
climate regimes, derive PCM performance targets per regime, and rank candidate phase-change
materials against those targets with an auditable, multi-method, uncertainty-aware pipeline.

Governing document: `Objective1_PCM_Climate_Framework_Plan_v3.docx` ("the framework doc"),
version 3.0, which supersedes v2.0. It defines **Phase 1 through Phase 8** — there is no "Phase 0"
in the framework doc itself; the pipeline's own `phases.md` and file-naming use "Phase 0" informally
for the sampling-design step that precedes Phase 1. This documentation set follows the framework
doc's authoritative numbering and treats the sampling step as a Phase-1 prerequisite, not a
separate phase.

## What the ERA5 pipeline is trying to achieve

Rather than picking PCM candidates by hand for one nominal Indian climate, the project:
1. Samples Rajasthan at 320 population-weighted points (not a uniform grid, not a handful of
   named cities) so results are defensible against "why these locations?"
2. Pulls two independent climate data sources (ERA5 reanalysis, NASA POWER satellite/model
   product) for the *same* points and instants, and **validates one against the other before
   trusting either** — this caught a real bug (see below).
3. Reduces 10 years of hourly/daily data per point into a compact **two-tier climate signature**
   (instantaneous sun-event statistics + true daily-integral indices).
4. Clusters points into **climate regimes** (Gaussian Mixture Model, not hand-drawn zones) at two
   levels: spatial (Level A) and seasonal (Level B).
5. Derives a **per-regime PCM performance target** (melting point, required latent heat) from the
   regime's own climate signature, not a single national number.
6. Filters a PCM property database against physical/safety/economic constraints, then ranks
   survivors with **four independent MCDM methods** plus Monte Carlo uncertainty propagation, so
   the final recommendation is not an artifact of any one ranking method's assumptions.
7. Independently validates the MCDM ranking against a physics-based lumped-enthalpy tank
   simulation (Phase 7), and packages the whole result as per-cluster recommendation cards
   (Phase 8) — **both now implemented and run**, see the status table below. Phase 7's result is
   a genuine, honestly-reported NEGATIVE validation (all three clusters' Spearman rho ≤ 0.4) —
   see `19_PHASE_7_ONWARD.md` for the full completion report.

## Complete pipeline map (as actually implemented, not the generic assumption)

```
Phase 1 — DATA COLLECTION
  00a_build_population_grid.py   → population_grid_points.csv (320 pts, 87.5% pop coverage)
  00b_build_suntimes.py          → suntimes.csv (3,506,880 rows: 320 pts × 3653 days × 3 events)
  00c_attach_elevation.py        → population_grid_points.csv gains elevation_m (ERA5 geopotential)
  01_download_era5_rajasthan.py  → data/raw/era5/points/*.nc  (240 files, 816 MB, sun-event-aligned hours)
  01b_download_nasapower.py      → data/raw/nasapower/*.json  (3200 files, 2.47 GB)
  00_unzip_accum.py              → (fixes CDS zip-disguised-as-.nc quirk, both archives)
        ↓
Phase 2 — PREPROCESSING & CROSS-SOURCE VALIDATION
  02_combine_rajasthan.py        → climate_rajasthan_points.csv (unit conv., solar geometry, merge)
  02b_build_daily_aggregates.py  → daily_aggregates_rajasthan{,_summary}.csv (POWER-only daily integrals)
  03_verify_climate_csv.py       → stdout QA report (schema/coverage/nulls/range/agreement)
  03_qc_plots.py                 → outputs/qc_*.html (spatial + distributional QC)
  03b_agreement_analysis.py      → era5_power_agreement_rajasthan.csv, bias_decision_rajasthan.txt
        ↓  [DECISION: QUANTILE_MAP — see 14_ERA5_POWER_VALIDATION.md]
Phase 2.5 — QUALITY CHECK (undocumented until 2026-08-11 — see 15_QUALITY_CONTROL.md)
  03b_quality_check_rajasthan.py       → climate_rajasthan_points_clean.csv, quality_report_rajasthan.{md,json}
                                          (Hampel-filter outlier winsorizing on T_amb/RHum/W_spd ONLY —
                                           GHI/CSI deliberately excluded, see that script's docstring —
                                           + missing-data imputation)
  03b_validate_quality_fix_rajasthan.py → independent before/after re-verification (re-runs Phase 3)
  03c_plots_raw_rajasthan.py            → outputs/qc_raw_*.html (raw pre-QC visual sanity checks)
  03b_quality_check_plots_rajasthan.py  → outputs/qc_clean_*.html (post-QC visual sanity checks)
        ↓  Phase 3 now reads climate_rajasthan_points_CLEAN.csv, not 02's raw output directly
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
  signature_lib.py + 04_climate_signature_rajasthan.py → climate_signature_rajasthan.csv
  (Tier 1 sun-event indices + Tier 2 daily indices + Tm_target/L_required + 5 interaction terms
   + PCA(4 comps, 95% var) + standardized *_z clustering matrix + 2 QC plots)
        ↓
Phase 4 — CLIMATE REGIME CLUSTERING
  05_cluster_rajasthan.py → cluster_assignments_rajasthan_levelA/B.csv, bic_selection_rajasthan.csv,
                              cluster_profiles_rajasthan.csv, cluster_profile_cards_rajasthan.md
                              (k=3, GMM diag covariance, CANONICALLY RELABELED by ascending mean
                               latitude — fixed 2026-08-11, see 06_PHASE_4_AUDIT.md; Koppen-Geiger
                               external validation now actually wired in, not stubbed)
        ↓
Phase 5 — FEASIBILITY FILTERING  (+ shared PCM property database, run independently)
  01_preprocess.py (PCM_data/) → PCM_Properties_cleaned_mice_pmm{,_detailed}.csv (18 rows, MICE-RF-PMM)
  07_feasibility_filter_rajasthan.py → feasibility_survivors_rajasthan{,_kappa_calibrated}.csv
        ↓  [FINDING: 0 survivors at nominal kappa=0.7 — see 07_PHASE_5_AUDIT.md]
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking_rajasthan.py → mcdm_rankings_rajasthan.csv, mcdm_method_agreement_rajasthan.csv
  (TOPSIS + PROMETHEE II + VIKOR + GRA, Borda/Copeland/Kendall's W, 1000-draw Monte Carlo)
        ↓
Phase 7 — PHYSICS-BASED VALIDATION  ◄── CURRENT IMPLEMENTATION FRONTIER (complete)
  physics_lib.py + 09_physics_validation_rajasthan.py → physics_validation_rajasthan.csv,
    spearman_rho_by_cluster_rajasthan.csv, outputs/qc_calibration_check_rajasthan.html,
    physics_validation_summary_rajasthan.txt
  (lumped-enthalpy PCM+tank model, real hourly NASA POWER weather, cited draw profile;
   RESULT: genuine NEGATIVE validation, rho = -0.900 / -0.096 / -0.198 across the 3 clusters —
   see 19_PHASE_7_ONWARD.md)
        ↓
Phase 8 — RECOMMENDATION CARDS (complete)
  10_recommendation_cards_rajasthan.py → outputs/recommendation_cards_rajasthan.md
  (pure aggregation of Phases 4/6/7, one card per cluster + cross-cluster summary table, hard-fails
   on any cross-phase cluster-identity mismatch via provenance_lib.py)
```

**Orchestration**: `run_all_rajasthan.py` runs the entire reproducible chain (Phase 2 through
Phase 8) in one invocation, in the correct dependency order, stopping at the first core-stage
failure — see `21_REPRODUCIBILITY.md`.

## Phase 1–8 status at a glance

| Phase | Script(s) | Status | Headline finding |
|---|---|---|---|
| 1 — Data Collection | `00a/00b/00c`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 320 pts, 240/240 ERA5 files, 3200/3200 (1 retry) POWER files |
| 2 — Preprocessing & Validation | `02`, `02b`, `03_verify`, `03_qc_plots`, `03b` | **COMPLETE — with a caught-and-fixed critical bug** | Deaccumulation bug found & fixed; QUANTILE_MAP branch applied |
| 2.5 — Quality Check | `03b_quality_check`, `03b_validate_quality_fix` | **COMPLETE — 3 sequential corrections, see 15_QUALITY_CONTROL.md** | Hampel filter initially over-corrected genuine cloud-driven GHI/CSI variability; fixed by excluding those two variables from outlier detection entirely |
| 3 — Climate Signature | `signature_lib.py`, `04` | **COMPLETE — 5 documented corrections** | Tm_target=57°C fixed; Tm_target_capped varies by regime; now reads the Phase 2.5 CLEAN file |
| 4 — Regime Clustering | `05` | **COMPLETE — with 2 caught-and-fixed bugs** | k=3 (GMM `diag` covariance, fixed from `full`); GMM cluster-index instability fixed via canonical relabeling (2026-08-11); Koppen-Geiger external validation wired in (ARI=0.19, NMI=0.32 vs GMM) |
| 5 — Feasibility Filtering | `01_preprocess`, `07` | **COMPLETE but produces 0 survivors at nominal thresholds** | PCM database still 18/25 rows, not the target 40–60; latent-heat floor structurally unreachable at κ=0.7; now stamps a provenance fingerprint |
| 6 — MCDM Ranking | `08` | **COMPLETE — with 3 caught-and-fixed bugs, 1 documented deviation** | Runs on κ-relaxed survivor pool; N_DRAWS=1000 not 5000 (documented); AHP pairwise elicitation still a TODO stub; now hard-fails on a provenance mismatch |
| 7 — Physics Validation | `physics_lib.py`, `09` | **COMPLETE — 2 caught-and-fixed bugs in the solver, real calibration iteration, genuine NEGATIVE result** | Spearman rho = -0.900 (Cluster 0) / -0.096 (Cluster 1) / -0.198 (Cluster 2) — MCDM ranking is NOT confirmed by physics simulation for any cluster; see `19_PHASE_7_ONWARD.md` |
| 8 — Recommendation Cards | `10` | **COMPLETE** | Pure aggregation of Phases 4/6/7 into `recommendation_cards_rajasthan.md`, hard-fails on cross-phase cluster-identity mismatch |

## Current architecture

- **Language/stack**: Python, pandas/numpy/scikit-learn/scipy, `pvlib` for solar geometry,
  `cdsapi` for ERA5, `xarray`/`netCDF4` for NetCDF, `folium`/`plotly` for QC visualization,
  `geopandas`/`rasterio` for the population-grid step.
- **Path convention**: every script imports `config.py`, which anchors all paths to
  `era5-rajasthan/` regardless of working directory. No hardcoded absolute paths inside the
  numbered scripts themselves.
- **Resumability**: every download/compute stage has an idempotency mechanism (status-CSV
  logging + file-size/content checks) — see `21_REPRODUCIBILITY.md`. Every mechanism was
  independently ground-truthed against the actual files on disk, not just read from code.
- **State-parameterization**: `05_cluster_rajasthan.py` and `signature_lib.py` are explicitly
  written to be state-agnostic (`STATE_NAME` is the only hardcoded state string), anticipating the
  same pipeline running on Assam/Tamil Nadu/Uttarakhand and a future 4-state combined clustering run.

## Main datasets

| File | Rows | Grain | Produced by |
|---|---|---|---|
| `population_grid_points.csv` | 320 | 1 row/point | `00a` (+`00c` elevation) |
| `suntimes.csv` | 3,506,880 | 1 row/point/date/event | `00b` |
| `climate_rajasthan_points.csv` | ~3.5M (partial NaN for edge cases) | 1 row/point/date/event | `02` |
| `daily_aggregates_rajasthan.csv` | ~1.17M (320×3653) | 1 row/point/day | `02b` |
| `daily_aggregates_rajasthan_summary.csv` | 320 | 1 row/point | `02b` |
| `era5_power_agreement_rajasthan.csv` | 80 | 1 row/variable×season×event stratum | `03b` |
| `climate_signature_rajasthan.csv` | 320 | 1 row/point, 86 columns | `04` |
| `cluster_assignments_rajasthan_levelA.csv` | 320 | 1 row/point | `05` |
| `cluster_profiles_rajasthan.csv` | 3 | 1 row/cluster | `05` |
| `feasibility_survivors_rajasthan.csv` | 75 (3 clusters × 25 candidates) | 1 row/cluster×PCM | `07` |
| `mcdm_rankings_rajasthan.csv` | 20 (survivors across clusters) | 1 row/cluster×surviving PCM | `08` |
| `physics_validation_rajasthan.csv` | 20 | 1 row/cluster×simulated PCM | `09` |
| `spearman_rho_by_cluster_rajasthan.csv` | 3 | 1 row/cluster | `09` |
| `recommendation_cards_rajasthan.md` | 3 cards + 1 summary table | 1 card/cluster | `10` |

## Main algorithms

Solar geometry (pvlib SPA + Ineichen clear-sky) · Magnus-formula RH · Gaussian-mixture clustering
(diagonal covariance) with bootstrap-ARI stability · PCA (95% variance) · MICE-style chained-equation
imputation with a custom inverse-distance-weighted PMM-like donor blend (Random Forest, not sklearn's
`IterativeImputer`) · empirical quantile mapping · Shannon-entropy criterion weighting · TOPSIS ·
PROMETHEE II · VIKOR · Grey Relational Analysis · Borda count · Copeland pairwise · Kendall's W ·
Dirichlet/Gaussian Monte Carlo uncertainty propagation.

## Validation strategy

Two independent validation layers exist today: (1) **cross-source** — ERA5 vs NASA POWER agreement
analysis with a pre-registered decision rule (backbone / quantile-map / manual-review), and
(2) **internal statistical** — GMM bootstrap-ARI stability, silhouette/BIC/Davies-Bouldin/
Calinski-Harabasz for cluster count, Monte Carlo inclusion-probability for MCDM rank stability,
Kendall's W for cross-method ranking agreement. A third layer — **external classification
validation** (Köppen-Geiger, NBC/ECBC climate zones) — is specified and explicitly stubbed (`None`
values, not fabricated), and a fourth — **physics-based simulation validation** (Phase 7) — is
specified but not yet implemented.

## Current known issues (see `20_IMPLEMENTATION_ISSUES.md` for full detail)

1. **[FIXED, mandatory audit checkpoint]** ERA5 accumulated-field deaccumulation bug: an earlier
   `deaccumulate()` assumed classic MARS cumulative-since-reset semantics and diffed consecutive
   hours; this pipeline's actual CDS download already returns each hour as its own ~1-hour flux.
   The bug produced near-zero, physically implausible GHI (noon Pearson r≈0.01 against NASA POWER);
   the fix (`accum_to_flux()`, a stateless clip, no diffing) restored r=0.8102, MBE=10.95 W/m² at
   solar noon. **This is the single most important scientific-integrity finding in the pipeline.**
2. **[FIXED]** GMM covariance type: `full` → `diag`, root-caused as a covariance-parameter/sample-size
   underdetermination artifact that was saturating membership probabilities to ~1.0 regardless of
   true geometric separation.
3. **[FIXED]** VIKOR compromise-index sign inversion (was `(Sb-Sw)/(Rb-Rw)`, silently reversed rankings).
4. **[FIXED]** Entropy-weight inflation for sparse/all-NaN criteria (the `cost` criterion, always NaN
   in this database, was getting 64–75% entropy weight before the fix).
5. **[OPEN]** Feasibility filter's latent-heat floor (`L ≥ 0.7×L_required`) is structurally unreachable
   by every candidate in the current 18-row PCM database given the corrected `L_required` derivation
   (~610–643 kJ/kg ceiling vs best-case ~252 kJ/kg latent heat) — **0 of 75 cluster×candidate rows
   survive at the nominal κ=0.7**, and the pipeline currently relies on an ad hoc per-cluster
   κ-relaxation pass to produce any MCDM input at all.
6. **[OPEN]** PCM property database is 18 rows (25 counting literature-only rows in the vestigial
   TN-branch script), well short of the 40–60-row target the framework doc specifies for the
   corrected 42–70°C band — this is the same gap the user's independent PCM-database-expansion task
   is meant to close, and it is a **hard prerequisite** for Phase 5/6 results to be trustworthy.
7. **[OPEN, minor]** Inconsistent Monsoon month definitions between `02_combine_rajasthan.py`
   (Jun–Aug) and `02b_build_daily_aggregates.py` (Jun–Sep), feeding different downstream indices.
8. **[OPEN, minor]** `avg_sdirswrf` (direct-radiation surrogate) unit handling is inconsistent with
   `ssrd`/`strd` — never divided by 3600, regardless of whether the matched column name is an
   accumulated or mean-rate ERA5 field.
9. **[PARTIALLY RESOLVED]** External classification validation: Köppen-Geiger (Beck et al. 2018,
   doi:10.1038/sdata.2018.214) is now actually wired in (1-km raster, real per-point lookup) —
   ARI(GMM, Köppen)=0.19, NMI=0.32 (low-to-moderate agreement, read as "the GMM finds finer
   structure than Köppen's broad classes," not as a clustering failure). NBC/ECBC Indian
   climate-zone classification remains stubbed (`None` placeholders, no fabricated labels) —
   no local lookup exists in this project tree.
10. **[OPEN]** AHP pairwise elicitation + consistency-ratio check exists in code but is never invoked
    (`AHP_PAIRWISE_MATRIX = None`) — the "AHP" component of the blended MCDM weights is actually
    just the framework doc's indicative Table 13 priors, unmodified.
11. **[FIXED, high-impact]** GMM cluster-index instability: sklearn's GaussianMixture gives no
    guarantee that cluster label 0 refers to the same physical climate group across separate
    re-runs — Phase 5's and Phase 6's outputs were found (2026-08-11) to disagree cluster-by-
    cluster on which PCMs belonged to which cluster_id, because they'd been run from two different
    invocations of `05_cluster_rajasthan.py`. Fixed via (a) canonical relabeling by ascending mean
    latitude in `05_cluster_rajasthan.py`, and (b) a hard-fail provenance-fingerprint check
    (`provenance_lib.py`) that Phases 6/7/8 each run against `cluster_profiles_rajasthan.csv`
    before trusting their inputs. See `06_PHASE_4_AUDIT.md` and `19_PHASE_7_ONWARD.md`.
12. **[FIXED]** Two numerical bugs in `physics_lib.py`'s Phase 7 solver, both caught by that
    script's own required self-tests before any real result was trusted: a wrong closed-form
    backward-Euler solve (caused unbounded temperature blow-up) and a phase-transition energy-
    accounting bug (silently discarded the sensible-heat "overshoot" at melt onset). Energy
    conservation now holds to machine precision (~1e-13 relative residual) — see `physics_lib.py`'s
    own module docstring for the full diagnosis.

## Research gaps addressed

The framework doc frames its contribution via **novelty positions N1–N6** (discovered regimes vs
hand-picked zones; two-tier signature vs single temperature; corrected 42–70°C SWH band vs 18–28°C
building-comfort band; Top-3+consensus vs single winner; physics-validated vs self-referential MCDM;
population-weighted regimes), **not** the RG1–RG5 taxonomy used elsewhere in this project's literature
summaries (which belongs to a companion DRL/control objective document). See
`18_RESEARCH_GAP_MAPPING.md` for the full, disambiguated mapping — conflating the two systems would
misrepresent what Objective 1 itself claims to contribute.

## What remains

Phases 1–8 are all now implemented and have been run end-to-end (via `run_all_rajasthan.py`) from a
single consistent Phase 4 clustering pass. What remains is resolving what Phase 7's genuine
negative result means for the project's claims, not building more pipeline:

1. **Expand the PCM property database** to the 40–60-row, 42–70°C-band target (the user's parallel
   task) — still the single highest-leverage open item. Every Phase 6/7/8 output is still tagged
   `pcm_database_status = "PROVISIONAL — ~25-row database, not yet expanded to 40-60"`, and Phase 7's
   own inherited-caveats discussion (`09_physics_validation_rajasthan.py`'s docstring) explicitly
   flags that Cluster 0's negative rho may be better explained by its undersized candidate pool
   (n=5) than by a genuine MCDM/physics disagreement. **Re-running Phases 5-8 after the database
   expansion is not optional cleanup — it will likely change the result, not just the numbers.**
2. Decide and document the κ-relaxation policy for the latent-heat constraint (accept per-cluster
   calibrated κ, or rank-by-proximity-to-L_required instead of hard-gating, per Correction 4's own
   recommendation in `04_climate_signature_rajasthan.py`'s docstring).
3. NBC/ECBC Indian climate-zone validation remains stubbed (Köppen-Geiger is now wired in — see
   known issue 9 above).
4. Interpret and write up Phase 7's negative result properly (see `19_PHASE_7_ONWARD.md`) — this is
   itself a real, reportable finding, not a failure to hide: it means the MCDM ranking, as currently
   weighted, is not confirmed by the physics simulation at the pipeline's current PCM-database size,
   and the honest next step is diagnosis (which criterion's weight, or database expansion), not
   re-running the simulation hoping for a different number.

## Recommended next step

Expand the PCM property database (item 1 above), then re-run the full chain
(`python run_all_rajasthan.py --from 07_feasibility_filter_rajasthan.py`) and see whether Phase 7's
result changes. Phase 7 was deliberately run anyway against the current provisional database — see
`19_PHASE_7_ONWARD.md` for the reasoning and the full completion report, including why running it
now (rather than waiting) was itself informative.
