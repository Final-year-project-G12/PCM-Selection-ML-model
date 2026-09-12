# 00 — Master Overview: ERA5 Rajasthan Climate → PCM Selection Pipeline

⚠️ **CRITICAL UPDATE (2026-08-31): L_required Methodology Correction** — Phase 3's methodology was corrected 2026-08-31, halving L_required values and cascading through Phases 4–8. All outputs from Phases 5–8 documented in this overview are now STALE and must be regenerated. Documented results (κ calibrations, Spearman rho validation values, rankings) below are superseded. See CLAUDE.md §3.1 and `04b_climate_signature.py` docstring for full detail.

⚠️ **UNIFICATION UPDATE (2026-09-08): Phase 5 + Phase 6 unified with Tamil Nadu; scripts renumbered.**
- **Phase 5** — `07_feasibility_filter.py` now runs the same 8-constraint set / order / missing-value
  semantics / κ-calibration / provenance stamping as Tamil Nadu. Constraint 6 (charging feasibility,
  `Tm ≤ Tm_target_capped_C`) replaced the retired heuristic `07b_charging_feasibility.py`. Outputs
  renamed to Tamil Nadu's canonical names: `feasibility_survivors_by_cluster{,_kappa_calibrated}.csv`.
- **Phase 6** — `08_mcdm_ranking.py` is now byte-identical to Tamil Nadu's engine (apart from
  state/paths). 8 Table-13 criteria; latent-heat criterion is climate-relative
  (`latent_heat / L_required`); cycling is log-scaled `cycles_confidence`; **supercooling's
  entropy weight is capped at 2× its Table-13 prior (0.16)** to fix a known entropy-formula
  overweighting pathology. Outputs renamed: `mcdm_full_rankings.csv`, `mcdm_topk_by_cluster.csv`,
  `monte_carlo_stability.csv`, `mcdm_method_agreement.csv`, `qc_montecarlo_inclusion.html`.
- **Script renumber** — to match Tamil Nadu exactly: `10_physics_validation.py` →
  `10_physics_validation.py`; `09_recommendation_cards.py` → `09_recommendation_cards.py`.
  Both state folders now share Phase 5–8 basenames `07`/`08`/`09`/`10`/`11` (cards = 09 but runs
  LAST, after physics = 10).
- Fresh run pending; every Phase 5–8 number below is pre-unification and stale.

---

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
   see `09_PHASE_7_AUDIT.md` for the full completion report.

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
        ↓  [DECISION: QUANTILE_MAP — advisory; see 04_PHASE_2_AUDIT.md §A.8 / Part B]
Phase 2.5 — PREPROCESSING & QUALITY CONTROL (restructured 2026-09-08 onto the Tamil Nadu
            04_preprocess contract — see 04_PHASE_2_AUDIT.md Part B / §B.11)
  04_preprocess_rajasthan.py            → data/preprocessed/rajasthan_cleaned_physical.csv
                                          (BOUNDS physical screen + SZA>=90 solar night-mask +
                                           per-season ERA5→POWER quantile map PERSISTED +
                                           Hampel on GHI/T_amb/RHum/W_spd/cloud_cover +
                                           4-stage/IterativeImputer(MICE) imputation;
                                           also *_scaled.csv + scalers.pkl for later ML/DRL use)
  (diagnostic only, not in the core chain:)
  03b_quality_check_rajasthan.py       → climate_rajasthan_points_clean.csv, quality_report_rajasthan.{md,json}
                                          (leaner: Hampel winsorize on T_amb/RHum/W_spd ONLY —
                                           GHI/CSI excluded — + point-local imputation; SUPERSEDED,
                                           follow-up: check the GHI-Hampel trade-off, 04_PHASE_2_AUDIT.md Part C/D)
  03c_plots_raw_rajasthan.py            → outputs/qc_raw_*.html (raw pre-QC visual sanity checks)
  03b_quality_check_plots_rajasthan.py  → outputs/qc_clean_*.html (post-QC visual sanity checks)
        ↓  Phase 3 now reads data/preprocessed/rajasthan_cleaned_physical.csv, not 02's raw output directly
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
  signature_lib.py + 04b_climate_signature.py → climate_signature_rajasthan.csv
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
  01_preprocess.py (PCM_data/) → PCM_Properties_cleaned_mice_pmm{,_detailed}.csv (55 rows, MICE-RF-PMM —
                                   expanded 2026-08-12 from the prior 18-row database, see below)
  07_feasibility_filter.py → feasibility_survivors_by_cluster{,_kappa_calibrated}.csv
        ↓  [Pre-expansion FINDING: 0 survivors at nominal kappa=0.7 — see 07_PHASE_5_AUDIT.md.
            NOT yet re-verified against the expanded 55-row database — outputs on disk are stale.]
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking.py → mcdm_full_rankings.csv, mcdm_topk_by_cluster.csv,
    monte_carlo_stability.csv, mcdm_method_agreement.csv, outputs/qc_montecarlo_inclusion.html
  (TOPSIS + PROMETHEE II + VIKOR + GRA, Borda/Copeland/Kendall's W, pairwise method-agreement,
   1000-draw Monte Carlo; 8 Table-13 criteria; climate-relative latent heat; log-scaled cycling;
   supercooling entropy weight capped at 2× prior; UNIFIED with Tamil Nadu 2026-09-08)
        ↓
Phase 7 — PHYSICS-BASED VALIDATION
  physics_lib.py + 10_physics_validation.py (was 10_physics_validation.py) →
    physics_validation_rajasthan.csv, spearman_rho_by_cluster_rajasthan.csv,
    outputs/qc_calibration_check_rajasthan.html
  (lumped-enthalpy PCM+tank model, real hourly NASA POWER weather, cited draw profile, full
   calibration. Pre-unification RESULT: rho = -0.385 / +0.125 / -0.097; re-run pending against
   the unified, entropy-capped Phase 6 output — see 09_PHASE_7_AUDIT.md)
        ↓
Phase 8 — SUPERCOOLING PENALTY SENSITIVITY ANALYSIS
  08_phase8_supercooling_sweep.py → phase8_supercooling_sweep_rajasthan.csv
  (proportional h_p reduction for supercooling_K, sweep k ∈ [0.0,0.1,0.2,0.3]; pre-unification
   RESULT: penalty WORSENS physics/MCDM agreement — supporting evidence for the Phase 6 entropy
   cap; re-run pending — see 10_PHASE_8_AUDIT.md)
        ↓
Phase 8 deliverable — RECOMMENDATION CARDS
  09_recommendation_cards.py (was 09_recommendation_cards.py) →
    outputs/recommendation_cards_rajasthan.md
  (pure aggregation of Phases 4/6/7, one card per cluster + cross-cluster summary table, hard-fails
   on any cross-phase cluster-identity mismatch via provenance_lib.py. Numbered 09 to match Tamil
   Nadu; runs LAST, after 10_physics_validation.py)
```

**Orchestration**: `run_all_rajasthan.py` runs the entire reproducible chain (Phase 2 through
Phase 8) in one invocation, in the correct dependency order, stopping at the first core-stage
failure — see "Current architecture" → Resumability below.

## Phase 1–8 status at a glance

| Phase | Script(s) | Status | Headline finding |
|---|---|---|---|
| 1 — Data Collection | `00a/00b/00c`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 320 pts, 240/240 ERA5 files, 3200/3200 (1 retry) POWER files |
| 2 — Preprocessing & Validation | `02`, `02b`, `03_verify`, `03_qc_plots`, `03b` | **COMPLETE — with a caught-and-fixed critical bug** | Deaccumulation bug found & fixed; QUANTILE_MAP decision (advisory) — correction now persisted in Phase 2.5 |
| 2.5 — Preprocessing & Quality Control | `04_preprocess_rajasthan` (core); `03b_quality_check`, `03b_validate_quality_fix` (diagnostic/deprecated) | **RESTRUCTURED 2026-09-08 — converged onto the Tamil Nadu `04_preprocess` contract; re-run pending** | Phase 2.5 is now `04_preprocess_rajasthan.py` (BOUNDS + SZA night-mask + per-season quantile map persisted + Hampel + 4-stage/MICE imputation → `rajasthan_cleaned_physical.csv`). Follow-up: it Hampel-filters GHI/cloud_cover, which the retired `03b_quality_check` excluded — verify against the regenerated signature (04_PHASE_2_AUDIT.md Part C/D) |
| 3 — Climate Signature | `signature_lib.py`, `04b_climate_signature` | **COMPLETE — 5 documented corrections; re-run pending after the Phase 2.5 restructure** | Tm_target=57°C fixed; Tm_target_capped varies by regime; now reads `data/preprocessed/rajasthan_cleaned_physical.csv` |
| 4 — Regime Clustering | `05` | **COMPLETE — with 2 caught-and-fixed bugs** | k=3 (GMM `diag` covariance, fixed from `full`); GMM cluster-index instability fixed via canonical relabeling (2026-08-11); Koppen-Geiger external validation wired in (ARI=0.19, NMI=0.32 vs GMM) |
| 5 — Feasibility Filtering | `01_preprocess`, `07` | **UNIFIED with Tamil Nadu 2026-09-08; re-run pending** | 8 constraints in canonical order; Constraint 6 = `Tm ≤ Tm_target_capped_C` (replaces retired `07b_charging_feasibility.py`); κ-calibration; outputs `feasibility_survivors_by_cluster{,_kappa_calibrated}.csv`. 62-row shared PCM pool. |
| 6 — MCDM Ranking | `08` | **UNIFIED with Tamil Nadu 2026-09-08 (byte-identical engine); re-run pending** | 8 Table-13 criteria; climate-relative latent heat (`L/L_required`); log-scaled cycling; **supercooling entropy weight capped at 2× prior (0.16)** — fixes the entropy-formula overweighting Phase 7/8 diagnosed as the negative-correlation cause. PROMETHEE handles Tm natively (q=2K/p=8K). N_DRAWS=1000 (both states). Provenance hard-fail. AHP pairwise elicitation still a TODO stub. |
| 7 — Physics Validation | `physics_lib.py`, `10_physics_validation.py` | **RE-RUN PENDING (renumbered 09→10; unified Phase 6 input changed)** | Pre-unification Spearman rho = -0.385 / +0.125 / -0.097. The unified Phase 6's entropy cap should move these — the post-cap re-run is the actual test. SF ~65% (in the 54–84% benchmark band). |
| 8 — Supercooling Penalty | `physics_lib.py`, `08_phase8_supercooling_sweep.py` | **RE-RUN PENDING** | Pre-unification sweep worsened physics/MCDM agreement as k rose — this is the evidence the Phase 6 entropy cap is based on (supercooling was *over*-weighted). Energy conservation passes. |
| 8 (deliverable) — Recommendation Cards | `09_recommendation_cards.py` | **RE-RUN PENDING (renumbered 10→09)** | Pure aggregation of Phases 4/6/7 into `recommendation_cards_rajasthan.md`; hard-fails on cross-phase cluster-identity mismatch. Numbered 09 to match Tamil Nadu; runs LAST. |

## Current architecture

- **Language/stack**: Python, pandas/numpy/scikit-learn/scipy, `pvlib` for solar geometry,
  `cdsapi` for ERA5, `xarray`/`netCDF4` for NetCDF, `folium`/`plotly` for QC visualization,
  `geopandas`/`rasterio` for the population-grid step.
- **Path convention**: every script imports `config.py`, which anchors all paths to
  `era5-rajasthan/` regardless of working directory. No hardcoded absolute paths inside the
  numbered scripts themselves.
- **Resumability**: every download/compute stage has an idempotency mechanism (status-CSV
  logging + file-size/content checks), and `run_all_rajasthan.py` runs the core chain in dependency
  order with `--from <script>` resume support. Every mechanism was independently ground-truthed
  against the actual files on disk, not just read from code.
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
| `feasibility_survivors_by_cluster.csv` / `…_kappa_calibrated.csv` | 186 (3 clusters × 62 candidates) | 1 row/cluster×PCM | `07` |
| `mcdm_full_rankings.csv` | survivors across clusters (n≈41) | 1 row/cluster×surviving PCM | `08` |
| `mcdm_topk_by_cluster.csv` | 9 (3 clusters × Top-3) | 1 row/cluster×Top-3 PCM | `08` |
| `monte_carlo_stability.csv` / `mcdm_method_agreement.csv` | — | MC / method-pair diagnostics | `08` |
| `physics_validation_rajasthan.csv` | ≈41 | 1 row/cluster×simulated PCM | `10_physics_validation.py` |
| `spearman_rho_by_cluster_rajasthan.csv` | 3 | 1 row/cluster | `10_physics_validation.py` |
| `recommendation_cards_rajasthan.md` | 3 cards + 1 summary table | 1 card/cluster | `09_recommendation_cards.py` |

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
values, not fabricated), and a fourth — **physics-based simulation validation** (Phase 7,
`10_physics_validation.py`) — is implemented and run (pre-unification result: rho ≈ -0.4/+0.1/-0.1;
post-unification re-run pending).

## Current known issues

*(This section is the authoritative issue list for the Rajasthan pipeline; the phase audits and
`12_FINAL_READINESS_REPORT.md` refer back here.)*

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
5. **[RESOLVED]** Feasibility filter's latent-heat floor was structurally unreachable under the
   *all-latent* `L_required` derivation. After the 2026-08-31 combined-sensible+latent correction
   (`SHARE_PCM = 0.5`, `L_required ≈ 300–320 kJ/kg`) the fixed-κ=0.7 primary run now returns
   4 / 7 / 5 survivors per cluster, and the κ-calibration companion pass (κ = 0.5 / 0.6 / 0.5)
   yields 9 / 15 / 17 (n ≈ 41). κ-calibration is now a documented sensitivity step, not a rescue.
6. **[RESOLVED, prerequisite met 2026-08-12 — pipeline re-run still pending]** PCM property database
   expanded from 18 rows (25 counting literature-only rows in the vestigial TN-branch script) to
   **55 rows** (14 Rubitherm RT-line + 7 Pluss savE + 4 PCM Products Ltd/PlusICE + 5 PureTemp +
   1 CrodaTherm + 24 literature-sourced n-alkane/fatty-acid/composite rows), now inside the framework
   doc's 40–60-row target for the 42–70°C band. The row-count/manufacturer-diversity gap is closed.
   `PCM_Properties_cleaned_mice_pmm_detailed.csv` now exists on disk and is read by both
   `07_feasibility_filter.py` and `08_mcdm_ranking.py` (the 2026-09-08 runs used it). **What is still
   true**: zero rows are salt-hydrate/inorganic-typed, so the corrosion-veto (Constraint 7) is
   structurally inert. Both states now build the candidate pool from this one canonical file — 62
   rows (55 manufacturer + 7 Singh2025 literature), row-for-row identical between Rajasthan and
   Tamil Nadu. See `07_PHASE_5_AUDIT.md`.
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
    before trusting their inputs. See `06_PHASE_4_AUDIT.md` and `09_PHASE_7_AUDIT.md`
    ("Completion Report").
12. **[FIXED]** Two numerical bugs in `physics_lib.py`'s Phase 7 solver, both caught by that
    script's own required self-tests before any real result was trusted: a wrong closed-form
    backward-Euler solve (caused unbounded temperature blow-up) and a phase-transition energy-
    accounting bug (silently discarded the sensible-heat "overshoot" at melt onset). Energy
    conservation now holds to machine precision (~1e-13 relative residual) — see `physics_lib.py`'s
    own module docstring for the full diagnosis.
13. **[RESTRUCTURED 2026-09-08 — full re-run of Phases 2.5→8 pending]** Phase 2.5 converged onto the
    Tamil Nadu `04_preprocess` contract. `04_preprocess_rajasthan.py` replaces
    `03b_quality_check_rajasthan.py` in `run_all_rajasthan.py`'s core chain; `04b_climate_signature.py`
    now reads `data/preprocessed/rajasthan_cleaned_physical.csv`. This **resolves** the long-standing
    open item that the ERA5→NASA-POWER quantile-map correction was computed but never persisted —
    Step 2b of `04_preprocess_rajasthan.py` now writes the corrected `era5_GHI`/`era5_CSI` (applied
    unconditionally per season). It also fixes a latent runner bug: the core chain's old
    `04_climate_signature_rajasthan.py` entry named a nonexistent file, so `run_all` was silently
    skipping Phase 3. **Follow-up:** `04_preprocess_rajasthan.py` Hampel-filters `era5_GHI` /
    `era5_cloud_cover`, which `03b_quality_check_rajasthan.py` deliberately excluded after three
    empirical corrections — regenerate `climate_signature_rajasthan.csv` and Phases 4–8, diff
    against the pre-switch outputs, and confirm the GHI low-clearness tail is not being uniformly
    eroded. See `04_PHASE_2_AUDIT.md` Parts B/C/D.

## Research gaps addressed (N1–N6 novelty mapping)

### Important disambiguation

Two distinct gap/novelty systems exist:
- **N1–N6** (framework doc §3, Table 3): Objective 1's own novelty positioning, specific to this
  climate-signature/clustering/MCDM/validation pipeline.
- **RG1–RG5** (project-wide framing): research gaps for the broader multi-objective project
  (this objective plus downstream DRL-control and hardware-prototype objectives). RG1–RG5 do
  not appear in `Objective1_PCM_Climate_Framework_Plan_v3.docx` itself.

Conflating these two would misattribute claims — Objective 1 does not address all five RG gaps
directly (only RG5); the others are fed by this objective's output but addressed across multiple
objectives.

### Phase → N (novelty claim) mapping

| Phase | Primary N-claim(s) | How it contributes |
|---|---|---|
| 1 — Data Collection | N6 | Population-weighted, sun-event-aligned sampling — not a uniform grid or arbitrary city list |
| 2 — Preprocessing & Validation | (supports all) | The deaccumulation-bug catch and QUANTILE_MAP decision are the evidentiary basis for claiming the climate backbone (Phases 3+) is trustworthy — without this phase, none of N1–N5 would be defensible |
| 3 — Climate Signature | N2, N3 | Two-tier signature (not a single temperature); Tm_target/L_required corrected to the 42–70°C SWH band (not the 18–28°C comfort band a naive approach might reuse) |
| 4 — Regime Clustering | N1 | GMM-discovered regimes (k=3, statistically selected, not hand-picked); external validation now PARTIALLY wired in (Köppen-Geiger, ARI=0.19/NMI=0.32) — N1's "discovered, not hand-picked" claim is now supported by internal statistical measures PLUS one external classification cross-check (NBC/ECBC still open) |
| 5 — Feasibility Filtering | N3 (partial) | Enforces the corrected 42–70°C band and SWH-specific constraints; **database-size gap closed 2026-08-12** (18–25 → 55 rows, inside the 40–60 target) — N3's practical value depended on having enough real in-band candidates to filter; that prerequisite is now met, but Phase 5 has not yet been re-run against the expanded database, so N3's demonstrated value in the current on-disk output is still the pre-expansion result |
| 6 — MCDM Ranking | N4 | Four-method consensus + Monte Carlo, not a single TOPSIS winner; Kendall's W explicitly reports when consensus is *not* strong (Cluster 0, W=0.4375) rather than hiding disagreement — this honest reporting is itself part of N4's value proposition |
| 7 — Physics Validation (COMPLETE) | N5 | Independently validated the MCDM ranking against simulated solar fraction — **the result is a genuine NEGATIVE validation (Spearman rho ≤0.4, all 3 clusters)**, not a confirmation. This is itself evidence for N5 as a methodology (the validation was performed rigorously and reported honestly, exactly per the framework doc's own "write it out plainly" instruction) even though it does not currently confirm the MCDM ranking's output — N5's claim should read "the ranking WAS physics-tested, honestly, with a negative result attributable in part to the still-undersized PCM database" not "the ranking IS physics-validated." See `09_PHASE_7_AUDIT.md`. |
| 8 — Recommendation Cards (COMPLETE) | (packaging) | Aggregates N1–N5's evidence, including Phase 7's negative result and its caveats, into the final deliverable format — `09_recommendation_cards.py`'s own caveats section surfaces the physics-validation band per cluster, not just the MCDM Top-3 |

### Phase → RG (broader project research gap) mapping — explicitly indirect

Since RG1–RG5 belong to the *broader* multi-objective project rather than Objective 1 itself, this
mapping describes how Objective 1's output **feeds** the later objectives that directly address
RG1–RG4, and how Objective 1 itself directly addresses RG5:

| Phase | Related RG | Nature of contribution |
|---|---|---|
| 1–2 (Data Collection, Validation) | RG5 | Supplies the validated, uncertainty-characterized climate data a later predictive-optimization-under-uncertainty component (RG5, "no predictive optimization under climatic uncertainty") would need as its own input |
| 3–4 (Signature, Clustering) | RG5 | Climate regimes are themselves a climatic-uncertainty-aware framing (population-weighted, statistically validated) — a direct, not merely feeding, contribution to RG5 |
| 5–6 (Feasibility, MCDM) | RG5 | Monte Carlo uncertainty propagation over PCM property/weight perturbation is Objective 1's own predictive-optimization-under-uncertainty contribution |
| 7 (Physics Validation) | RG4 (indirect) | A grey-box simulation is not a real-world experiment, but it is Objective 1's step toward the experimental-validation direction RG4 (limited real-world experimental validation) ultimately calls for — the framework doc itself frames Phase 7 as "what makes the result publishable, not skippable" |
| 8 (Recommendation Cards) | RG2, RG3 (feeding, not addressing) | The per-regime PCM recommendation is the direct input a later hardware-prototype objective (RG2) and demand-alignment objective (RG3) would consume — Objective 1 does not itself build a prototype or model household demand |
| — | RG1 | **Not addressed by Objective 1 at all** — real-time adaptive control is explicitly out of this objective's scope (framework doc §1.2) |

### Important note

This mapping does not assert that Objective 1 "solves" RG1–RG4 — only RG5 is directly addressed by
this objective's own methodology (Monte Carlo uncertainty propagation, regime-level rather than
single-point climate targets). RG1–RG4 are gaps the *broader* project addresses across multiple
objectives, and Objective 1's role there is to produce a validated, regime-aware PCM recommendation
that the later objectives can build on — not to close those gaps itself. Presenting this mapping with
that distinction intact is more defensible in a viva than claiming Objective 1 single-handedly
addresses all five research gaps.

## What remains

Phases 1–8 are all now implemented and have been run end-to-end (via `run_all_rajasthan.py`) from a
single consistent Phase 4 clustering pass. What remains is resolving what Phase 7's genuine
negative result means for the project's claims, not building more pipeline:

1. **Regenerate `PCM_Properties_cleaned_mice_pmm_detailed.csv` and re-run Phases 5–8** against the
   now-expanded 55-row PCM database — this is now the single highest-leverage open item, replacing the
   database-expansion task itself (that part is done, see known issue 6 above). Every Phase 6/7/8
   output currently on disk is still tagged `pcm_database_status = "PROVISIONAL — ~25-row database, not
   yet expanded to 40-60"` because it predates the expansion, and Phase 7's own inherited-caveats
   discussion (`10_physics_validation.py`'s docstring) explicitly flags that Cluster 0's
   negative rho may be better explained by its undersized candidate pool (n=5) than by a genuine
   MCDM/physics disagreement. **Re-running Phases 5-8 is not optional cleanup — it will likely change
   the result, not just the numbers.** Concretely: `python PCM_data/PCM_data/01_preprocess.py`
   (regenerates the missing `_detailed.csv`), then `python run_all_rajasthan.py --from
   07_feasibility_filter.py`.
2. Decide and document the κ-relaxation policy for the latent-heat constraint (accept per-cluster
   calibrated κ, or rank-by-proximity-to-L_required instead of hard-gating, per Correction 4's own
   recommendation in `04b_climate_signature.py`'s docstring).
3. NBC/ECBC Indian climate-zone validation remains stubbed (Köppen-Geiger is now wired in — see
   known issue 9 above).
4. Interpret and write up Phase 7's negative result properly (see `09_PHASE_7_AUDIT.md`) — this is
   itself a real, reportable finding, not a failure to hide: it means the MCDM ranking, as currently
   weighted, is not confirmed by the physics simulation at the pipeline's current PCM-database size,
   and the honest next step is diagnosis (which criterion's weight, or database expansion), not
   re-running the simulation hoping for a different number.

## Recommended next step

Re-run the chain from Phase 5 against the **unified** Phase 5/6 engine (2026-09-08):
`python 07_feasibility_filter.py` → `08_mcdm_ranking.py` → `10_physics_validation.py` →
`09_recommendation_cards.py` → `11_seasonal_pcm_sensitivity.py` → `08_phase8_supercooling_sweep.py`
(or `python run_all_rajasthan.py --from 07_feasibility_filter.py`). The key question is whether
the Phase 6 **supercooling entropy-weight cap** (0.16, down from the ~0.48–0.64 the raw entropy
formula produced) improves Phase 7's Spearman rho vs the pre-cap -0.385 / +0.125 / -0.097. Every
number currently in `feasibility_survivors_by_cluster*.csv`, `mcdm_full_rankings.csv`,
`physics_validation_rajasthan.csv`, `spearman_rho_by_cluster_rajasthan.csv`, and
`recommendation_cards_rajasthan.md` is pre-unification and should be treated as superseded.
