# CLIMATE-ADAPTIVE INTELLIGENT CONTROL AND OPTIMIZATION OF PCM THERMAL STORAGE FOR SOLAR WATER HEATING

Consolidated Research, Methodology, Audit, and Literature Archive

Objective 1 — Climate-Region-Aware PCM Recommendation Framework
Group 12 · B.Tech Computer Science & Engineering
Amrita School of Engineering · Guide: Dr. T. Deepika

Compiled from all uploaded project documents and literature-summary files available in this workspace. Compilation date: 03 September 2026

# 1. Compilation Scope and Reading Record

This document consolidates the full text of the uploaded project materials
currently available in the workspace. The source files were read
programmatically in full before compilation; the original source text is
retained below under its corresponding file heading. Markdown headings are
converted into Word headings where practical, while tables and technical text
are preserved as closely as possible. No new scientific claims are inserted into
the source sections.

Total source files consolidated: 36

Source categories:

- Objective 1 audit/documentation set: 12 files
- Project methodology/context: 3 files
- Literature summaries: 21 files
# 2. Source File Inventory

1. 1. 00_MASTER_OVERVIEW.md
1. 2. 01_PROJECT_CONTEXT.md
1. 3. 02_DATA_SOURCES_AND_VARIABLES.md
1. 4. 03_PHASE_1_AUDIT.md
1. 5. 04_PHASE_2_AUDIT.md
1. 6. 05_PHASE_3_AUDIT.md
1. 7. 06_PHASE_4_AUDIT.md
1. 8. 07_PHASE_5_AUDIT.md
1. 9. 08_PHASE_6_AUDIT.md
1. 10. 09_PHASE_7_AUDIT.md
1. 11. 10_PHASE_8_AUDIT.md
1. 12. 11_LITERATURE_MAPPING.md
1. 13. PROJECT-SUMMARY.txt
1. 14. OBJECTIVE-1-—-IMPLEMENTATION-PLAN.txt
1. 15. Objective1_Section5_Methodology_Update.docx
1. 16. Abdellatif2025PCM_Modeling_Review_summary.md
1. 17. AlMamun2023SWH_StateOfArt_summary.md
1. 18. Assareh2023EnhancingSolarThermalPCM_summary.md
1. 19. Barghi2026SolarDrying_PCM_AI_summary.md
1. 20. Barqawi2025DynamicSimulationPCM_SWH_summary.md
1. 21. Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md
1. 22. Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md
1. 23. Duraivel2025DSTS_TechnoEconomic_summary.md
1. 24. Eldokaishi2022WaterPCM_ANN_SWH_summary.md
1. 25. Emami2026DRL_Solar_ORC_TES_summary.md
1. 26. Ghodusinejad2026SolarIrradianceForecasting_summary.md
1. 27. Hamzat2025PCM_SolarEnergyStorage_summary.md
1. 28. Kou2025BIHP_PCM_Building_Optimization_summary.md
1. 29. Liu2025AI_PCM_TES_Prediction_Optimization_summary.md
1. 30. Mansouri2025MultimodalRenewableForecasting_summary.md
1. 31. Martinez2025PCM_Industrial_TES_summary.md
1. 32. Mohammed2025NanoAI_ThermalSystems_summary.md
1. 33. OdoiYorke2025AI_SWH_Review_summary.md
1. 34. Singh2025PCM_SWH_ComprehensiveReview_summary.md
1. 35. Terfai2025SSP_ANN_MPC_Experimental_summary.md
1. 36. Yan2025ML_MeltingTime_TriplexTube_PCM_summary.md

# 3. Consolidated Source Contents

The following sections reproduce the contents of each uploaded file in full,
organized by source. File boundaries are preserved so that individual claims can
be traced back to their originating document.

# 1. 00_MASTER_OVERVIEW.md

Source path: /mnt/data/00_MASTER_OVERVIEW.md

# 00 — Master Overview: ERA5 Rajasthan Climate → PCM Selection Pipeline

⚠️ CRITICAL UPDATE (2026-08-31): L_required Methodology Correction — Phase 3's
methodology was corrected 2026-08-31, halving L_required values and cascading
through Phases 4–8. All outputs from Phases 5–8 documented in this overview are
now STALE and must be regenerated. Documented results (κ calibrations, Spearman
rho validation values, rankings) below are superseded. See CLAUDE.md §3.1 and
04_climate_signature_rajasthan.py docstring for full detail.

────────────────────────────────────────

## Project objective

Final-year B.Tech CSE project (Group 12, Amrita School of Engineering, Guide:
Dr. T. Deepika):

"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage
for Solar Water

Heating." Objective 1 (the scope of this audit) builds a climate-region-aware
PCM

recommendation framework: turn 10 years of reanalysis climate data into
population-weighted

climate regimes, derive PCM performance targets per regime, and rank candidate
phase-change

materials against those targets with an auditable, multi-method,
uncertainty-aware pipeline.

Governing document: Objective1_PCM_Climate_Framework_Plan_v3.docx ("the
framework doc"),

version 3.0, which supersedes v2.0. It defines Phase 1 through Phase 8 — there
is no "Phase 0"

in the framework doc itself; the pipeline's own phases.md and file-naming use
"Phase 0" informally

for the sampling-design step that precedes Phase 1. This documentation set
follows the framework

doc's authoritative numbering and treats the sampling step as a Phase-1
prerequisite, not a

separate phase.

## What the ERA5 pipeline is trying to achieve

Rather than picking PCM candidates by hand for one nominal Indian climate, the
project:

1. Samples Rajasthan at 320 population-weighted points (not a uniform grid, not
   a handful of
named cities) so results are defensible against "why these locations?"

1. Pulls two independent climate data sources (ERA5 reanalysis, NASA POWER
   satellite/model
product) for the same points and instants, and validates one against the other
before

trusting either — this caught a real bug (see below).

1. Reduces 10 years of hourly/daily data per point into a compact two-tier
   climate signature
(instantaneous sun-event statistics + true daily-integral indices).

1. Clusters points into climate regimes (Gaussian Mixture Model, not hand-drawn
   zones) at two
levels: spatial (Level A) and seasonal (Level B).

1. Derives a per-regime PCM performance target (melting point, required latent
   heat) from the
regime's own climate signature, not a single national number.

1. Filters a PCM property database against physical/safety/economic constraints,
   then ranks
survivors with four independent MCDM methods plus Monte Carlo uncertainty
propagation, so

the final recommendation is not an artifact of any one ranking method's
assumptions.

1. Independently validates the MCDM ranking against a physics-based
   lumped-enthalpy tank
simulation (Phase 7), and packages the whole result as per-cluster
recommendation cards

(Phase 8) — both now implemented and run, see the status table below. Phase 7's
result is

a genuine, honestly-reported NEGATIVE validation (all three clusters' Spearman
rho ≤ 0.4) —

see 19_PHASE_7_ONWARD.md for the full completion report.

## Complete pipeline map (as actually implemented, not the generic assumption)

Phase 1 — DATA COLLECTION
 00a_build_population_grid.py → population_grid_points.csv (320 pts, 87.5% pop
 coverage)
 00b_build_suntimes.py → suntimes.csv (3,506,880 rows: 320 pts × 3653 days × 3
 events)
 00c_attach_elevation.py → population_grid_points.csv gains elevation_m (ERA5
 geopotential)
 01_download_era5_rajasthan.py → data/raw/era5/points/*.nc (240 files, 816 MB,
 sun-event-aligned hours)
 01b_download_nasapower.py → data/raw/nasapower/*.json (3200 files, 2.47 GB)
 00_unzip_accum.py → (fixes CDS zip-disguised-as-.nc quirk, both archives)
 ↓
Phase 2 — PREPROCESSING & CROSS-SOURCE VALIDATION
 02_combine_rajasthan.py → climate_rajasthan_points.csv (unit conv., solar
 geometry, merge)
 02b_build_daily_aggregates.py → daily_aggregates_rajasthan{,_summary}.csv
 (POWER-only daily integrals)
 03_verify_climate_csv.py → stdout QA report
 (schema/coverage/nulls/range/agreement)
 03_qc_plots.py → outputs/qc_*.html (spatial + distributional QC)
 03b_agreement_analysis.py → era5_power_agreement_rajasthan.csv,
 bias_decision_rajasthan.txt
 ↓ [DECISION: QUANTILE_MAP — see 14_ERA5_POWER_VALIDATION.md]
Phase 2.5 — QUALITY CHECK (undocumented until 2026-08-11 — see
15_QUALITY_CONTROL.md)
 03b_quality_check_rajasthan.py → climate_rajasthan_points_clean.csv,
 quality_report_rajasthan.{md,json}
 (Hampel-filter outlier winsorizing on T_amb/RHum/W_spd ONLY —
 GHI/CSI deliberately excluded, see that script's docstring —
 + missing-data imputation)
 03b_validate_quality_fix_rajasthan.py → independent before/after
 re-verification (re-runs Phase 3)
 03c_plots_raw_rajasthan.py → outputs/qc_raw_*.html (raw pre-QC visual sanity
 checks)
 03b_quality_check_plots_rajasthan.py → outputs/qc_clean_*.html (post-QC visual
 sanity checks)
 ↓ Phase 3 now reads climate_rajasthan_points_CLEAN.csv, not 02's raw output
 directly
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
 signature_lib.py + 04_climate_signature_rajasthan.py →
 climate_signature_rajasthan.csv
 (Tier 1 sun-event indices + Tier 2 daily indices + Tm_target/L_required + 5
 interaction terms
 + PCA(4 comps, 95% var) + standardized *_z clustering matrix + 2 QC plots)
 ↓
Phase 4 — CLIMATE REGIME CLUSTERING
 05_cluster_rajasthan.py → cluster_assignments_rajasthan_levelA/B.csv,
 bic_selection_rajasthan.csv,
 cluster_profiles_rajasthan.csv, cluster_profile_cards_rajasthan.md
 (k=3, GMM diag covariance, CANONICALLY RELABELED by ascending mean
 latitude — fixed 2026-08-11, see 06_PHASE_4_AUDIT.md; Koppen-Geiger
 external validation now actually wired in, not stubbed)
 ↓
Phase 5 — FEASIBILITY FILTERING (+ shared PCM property database, run
independently)
 01_preprocess.py (PCM_data/) → PCM_Properties_cleaned_mice_pmm{,_detailed}.csv
 (55 rows, MICE-RF-PMM —
 expanded 2026-08-12 from the prior 18-row database, see below)
 07_feasibility_filter_rajasthan.py →
 feasibility_survivors_rajasthan{,_kappa_calibrated}.csv
 ↓ [Pre-expansion FINDING: 0 survivors at nominal kappa=0.7 — see
 07_PHASE_5_AUDIT.md.
 NOT yet re-verified against the expanded 55-row database — outputs on disk are
 stale.]
Phase 6 — MULTI-CRITERIA RANKING ENGINE
 08_mcdm_ranking_rajasthan.py → mcdm_rankings_rajasthan.csv,
 mcdm_method_agreement_rajasthan.csv
 (TOPSIS + PROMETHEE II + VIKOR + GRA, Borda/Copeland/Kendall's W, 1000-draw
 Monte Carlo)
 ↓
Phase 7 — PHYSICS-BASED VALIDATION (complete)
 physics_lib.py + 09_physics_validation_rajasthan.py →
 physics_validation_rajasthan.csv,
 spearman_rho_by_cluster_rajasthan.csv,
 outputs/qc_calibration_check_rajasthan.html
 (lumped-enthalpy PCM+tank model, real hourly NASA POWER weather, cited draw
 profile, full calibration;
 RESULT: genuine NEGATIVE validation, rho = -0.385 / +0.125 / -0.097 across 3
 clusters;
 dominant MCDM criterion is supercooling (48–64%) but model cannot simulate it —
 see 09_PHASE_7_AUDIT.md)
 ↓
Phase 8 — SUPERCOOLING PENALTY SENSITIVITY ANALYSIS (complete)
 08_phase8_supercooling_sweep.py → phase8_supercooling_sweep_rajasthan.csv
 (implements proportional h_p reduction for supercooling_K, sensitivity sweep k
 ∈ [0.0,0.1,0.2,0.3];
 RESULT: penalty WORSENS physics/MCDM agreement in Clusters 1-2 (rho degrades
 from +0.125→+0.059–0.077,
 and from -0.097→-0.118–0.136); suggests supercooling's real effect <48% MCDM
 weight, or penalty
 mechanism incorrect — see 10_PHASE_8_AUDIT.md for full analysis)
 ↓
Phase 9 — RECOMMENDATION CARDS (complete)
 10_recommendation_cards_rajasthan.py →
 outputs/recommendation_cards_rajasthan.md
 (pure aggregation of Phases 4/6/7, one card per cluster + cross-cluster summary
 table, hard-fails
 on any cross-phase cluster-identity mismatch via provenance_lib.py)

Orchestration: run_all_rajasthan.py runs the entire reproducible chain (Phase 2
through

Phase 8) in one invocation, in the correct dependency order, stopping at the
first core-stage

failure — see 21_REPRODUCIBILITY.md.

## Phase 1–8 status at a glance

| Phase | Script(s) | Status | Headline finding |
| --- | --- | --- | --- |
| 1 — Data Collection | 00a/00b/00c, 01, 01b, 00_unzip_accum | COMPLETE | 320 pts, 240/240 ERA5 files, 3200/3200 (1 retry) POWER files |
| 2 — Preprocessing & Validation | 02, 02b, 03_verify, 03_qc_plots, 03b | COMPLETE — with a caught-and-fixed critical bug | Deaccumulation bug found & fixed; QUANTILE_MAP branch applied |
| 2.5 — Quality Check | 03b_quality_check, 03b_validate_quality_fix | COMPLETE — 3 sequential corrections, see 15_QUALITY_CONTROL.md | Hampel filter initially over-corrected genuine cloud-driven GHI/CSI variability; fixed by excluding those two variables from outlier detection entirely |
| 3 — Climate Signature | signature_lib.py, 04 | COMPLETE — 5 documented corrections | Tm_target=57°C fixed; Tm_target_capped varies by regime; now reads the Phase 2.5 CLEAN file |
| 4 — Regime Clustering | 05 | COMPLETE — with 2 caught-and-fixed bugs | k=3 (GMM diag covariance, fixed from full); GMM cluster-index instability fixed via canonical relabeling (2026-08-11); Koppen-Geiger external validation wired in (ARI=0.19, NMI=0.32 vs GMM) |
| 5 — Feasibility Filtering | 01_preprocess, 07 | PCM database prerequisite now COMPLETE (55 rows); Phase 5 output on disk is STALE, pending re-run | Database expanded 18→55 rows (2026-08-12), inside the 40–60 target; feasibility_survivors_rajasthan.csv still reflects the pre-expansion 25-candidate pool and the old 0-survivors-at-κ=0.7 finding — re-run required, see "What remains" |
| 6 — MCDM Ranking | 08 | COMPLETE — with 3 caught-and-fixed bugs, 1 documented deviation | Runs on κ-relaxed survivor pool; N_DRAWS=1000 not 5000 (documented); AHP pairwise elicitation still a TODO stub; now hard-fails on a provenance mismatch |
| 7 — Physics Validation | physics_lib.py, 09 | COMPLETE — 2 caught-and-fixed bugs, real calibration, genuine NEGATIVE result | Spearman rho = -0.385 (Cluster 0) / +0.125 (Cluster 1) / -0.097 (Cluster 2); MCDM ranking weakly/negatively correlates with simulated solar fraction; dominant criterion is supercooling (48–64%) but model cannot simulate it — see 09_PHASE_7_AUDIT.md |
| 8 — Supercooling Penalty | physics_lib.py, 08_phase8_supercooling_sweep.py | COMPLETE — Sensitivity sweep k ∈ [0.0,0.1,0.2,0.3], honest negative result | Penalty implementation is correct (energy conservation passes); but worsens physics/MCDM agreement instead of improving it — rho degrades Cluster 1 from +0.125 to +0.059, and Cluster 2 from -0.097 to -0.136; suggests supercooling weight is over-estimated or mechanism is incorrect — see 10_PHASE_8_AUDIT.md |
| 9 — Recommendation Cards | 10 | COMPLETE | Pure aggregation of Phases 4/6/7 into recommendation_cards_rajasthan.md, hard-fails on cross-phase cluster-identity mismatch |

## Current architecture

- Language/stack: Python, pandas/numpy/scikit-learn/scipy, pvlib for solar
  geometry,
cdsapi for ERA5, xarray/netCDF4 for NetCDF, folium/plotly for QC visualization,

geopandas/rasterio for the population-grid step.

- Path convention: every script imports config.py, which anchors all paths to
era5-rajasthan/ regardless of working directory. No hardcoded absolute paths
inside the

numbered scripts themselves.

- Resumability: every download/compute stage has an idempotency mechanism
  (status-CSV
logging + file-size/content checks) — see 21_REPRODUCIBILITY.md. Every mechanism
was

independently ground-truthed against the actual files on disk, not just read
from code.

- State-parameterization: 05_cluster_rajasthan.py and signature_lib.py are
  explicitly
written to be state-agnostic (STATE_NAME is the only hardcoded state string),
anticipating the

same pipeline running on Assam/Tamil Nadu/Uttarakhand and a future 4-state
combined clustering run.

## Main datasets

| File | Rows | Grain | Produced by |
| --- | --- | --- | --- |
| population_grid_points.csv | 320 | 1 row/point | 00a (+00c elevation) |
| suntimes.csv | 3,506,880 | 1 row/point/date/event | 00b |
| climate_rajasthan_points.csv | ~3.5M (partial NaN for edge cases) | 1 row/point/date/event | 02 |
| daily_aggregates_rajasthan.csv | ~1.17M (320×3653) | 1 row/point/day | 02b |
| daily_aggregates_rajasthan_summary.csv | 320 | 1 row/point | 02b |
| era5_power_agreement_rajasthan.csv | 80 | 1 row/variable×season×event stratum | 03b |
| climate_signature_rajasthan.csv | 320 | 1 row/point, 86 columns | 04 |
| cluster_assignments_rajasthan_levelA.csv | 320 | 1 row/point | 05 |
| cluster_profiles_rajasthan.csv | 3 | 1 row/cluster | 05 |
| feasibility_survivors_rajasthan.csv | 75 (3 clusters × 25 candidates) | 1 row/cluster×PCM | 07 |
| mcdm_rankings_rajasthan.csv | 20 (survivors across clusters) | 1 row/cluster×surviving PCM | 08 |
| physics_validation_rajasthan.csv | 20 | 1 row/cluster×simulated PCM | 09 |
| spearman_rho_by_cluster_rajasthan.csv | 3 | 1 row/cluster | 09 |
| recommendation_cards_rajasthan.md | 3 cards + 1 summary table | 1 card/cluster | 10 |

## Main algorithms

Solar geometry (pvlib SPA + Ineichen clear-sky) · Magnus-formula RH ·
Gaussian-mixture clustering

(diagonal covariance) with bootstrap-ARI stability · PCA (95% variance) ·
MICE-style chained-equation

imputation with a custom inverse-distance-weighted PMM-like donor blend (Random
Forest, not sklearn's

IterativeImputer) · empirical quantile mapping · Shannon-entropy criterion
weighting · TOPSIS ·

PROMETHEE II · VIKOR · Grey Relational Analysis · Borda count · Copeland
pairwise · Kendall's W ·

Dirichlet/Gaussian Monte Carlo uncertainty propagation.

## Validation strategy

Two independent validation layers exist today: (1) cross-source — ERA5 vs NASA
POWER agreement

analysis with a pre-registered decision rule (backbone / quantile-map /
manual-review), and

(2) internal statistical — GMM bootstrap-ARI stability,
silhouette/BIC/Davies-Bouldin/

Calinski-Harabasz for cluster count, Monte Carlo inclusion-probability for MCDM
rank stability,

Kendall's W for cross-method ranking agreement. A third layer — external
classification

validation (Köppen-Geiger, NBC/ECBC climate zones) — is specified and explicitly
stubbed (None

values, not fabricated), and a fourth — physics-based simulation validation
(Phase 7) — is

specified but not yet implemented.

## Current known issues (see 20_IMPLEMENTATION_ISSUES.md for full detail)

1. [FIXED, mandatory audit checkpoint] ERA5 accumulated-field deaccumulation
   bug: an earlier
deaccumulate() assumed classic MARS cumulative-since-reset semantics and diffed
consecutive

hours; this pipeline's actual CDS download already returns each hour as its own
~1-hour flux.

The bug produced near-zero, physically implausible GHI (noon Pearson r≈0.01
against NASA POWER);

the fix (accum_to_flux(), a stateless clip, no diffing) restored r=0.8102,
MBE=10.95 W/m² at

solar noon. This is the single most important scientific-integrity finding in
the pipeline.

1. [FIXED] GMM covariance type: full → diag, root-caused as a
   covariance-parameter/sample-size
underdetermination artifact that was saturating membership probabilities to ~1.0
regardless of

true geometric separation.

1. [FIXED] VIKOR compromise-index sign inversion (was (Sb-Sw)/(Rb-Rw), silently
   reversed rankings).
1. [FIXED] Entropy-weight inflation for sparse/all-NaN criteria (the cost
   criterion, always NaN
in this database, was getting 64–75% entropy weight before the fix).

1. [OPEN] Feasibility filter's latent-heat floor (L ≥ 0.7×L_required) is
   structurally unreachable
by every candidate in the current 18-row PCM database given the corrected
L_required derivation

(~610–643 kJ/kg ceiling vs best-case ~252 kJ/kg latent heat) — 0 of 75
cluster×candidate rows

survive at the nominal κ=0.7, and the pipeline currently relies on an ad hoc
per-cluster

κ-relaxation pass to produce any MCDM input at all.

1. [RESOLVED, prerequisite met 2026-08-12 — pipeline re-run still pending] PCM
   property database
expanded from 18 rows (25 counting literature-only rows in the vestigial
TN-branch script) to

55 rows (14 Rubitherm RT-line + 7 Pluss savE + 4 PCM Products Ltd/PlusICE + 5
PureTemp +

1 CrodaTherm + 24 literature-sourced n-alkane/fatty-acid/composite rows), now
inside the framework

doc's 40–60-row target for the 42–70°C band. The
row-count/manufacturer-diversity gap the user's

parallel PCM-database-expansion task targeted is closed. What is still true:
zero rows in the

expanded database are salt-hydrate/inorganic-typed, so the corrosion-veto
constraint remains

structurally inert regardless, and the framework doc's 55–63°C
salt-hydrate-specific coverage gap

is not closed (though that melting-point band is now densely covered by organics
— RT54HC/RT55/

RT57HC/PureTemp 58/CrodaTherm 60/RT60/RT62HC/PureTemp 63). What has NOT yet
happened:

PCM_Properties_cleaned_mice_pmm_detailed.csv — the exact file
07_feasibility_filter_rajasthan.py

and 08_mcdm_ranking_rajasthan.py read — is currently absent from disk and must
be regenerated

(python PCM_data/PCM_data/01_preprocess.py), and Phases 5–8's outputs on disk
are all still from

the pre-expansion run. See 07_PHASE_5_AUDIT.md for full detail.

1. [OPEN, minor] Inconsistent Monsoon month definitions between
   02_combine_rajasthan.py
(Jun–Aug) and 02b_build_daily_aggregates.py (Jun–Sep), feeding different
downstream indices.

1. [OPEN, minor] avg_sdirswrf (direct-radiation surrogate) unit handling is
   inconsistent with
ssrd/strd — never divided by 3600, regardless of whether the matched column name
is an

accumulated or mean-rate ERA5 field.

1. [PARTIALLY RESOLVED] External classification validation: Köppen-Geiger (Beck
   et al. 2018,
doi:10.1038/sdata.2018.214) is now actually wired in (1-km raster, real
per-point lookup) —

ARI(GMM, Köppen)=0.19, NMI=0.32 (low-to-moderate agreement, read as "the GMM
finds finer

structure than Köppen's broad classes," not as a clustering failure). NBC/ECBC
Indian

climate-zone classification remains stubbed (None placeholders, no fabricated
labels) —

no local lookup exists in this project tree.

1. [OPEN] AHP pairwise elicitation + consistency-ratio check exists in code but
   is never invoked
(AHP_PAIRWISE_MATRIX = None) — the "AHP" component of the blended MCDM weights
is actually

just the framework doc's indicative Table 13 priors, unmodified.

1. [FIXED, high-impact] GMM cluster-index instability: sklearn's GaussianMixture
   gives no
guarantee that cluster label 0 refers to the same physical climate group across
separate

re-runs — Phase 5's and Phase 6's outputs were found (2026-08-11) to disagree
cluster-by-

cluster on which PCMs belonged to which cluster_id, because they'd been run from
two different

invocations of 05_cluster_rajasthan.py. Fixed via (a) canonical relabeling by
ascending mean

latitude in 05_cluster_rajasthan.py, and (b) a hard-fail provenance-fingerprint
check

(provenance_lib.py) that Phases 6/7/8 each run against
cluster_profiles_rajasthan.csv

before trusting their inputs. See 06_PHASE_4_AUDIT.md and 19_PHASE_7_ONWARD.md.

1. [FIXED] Two numerical bugs in physics_lib.py's Phase 7 solver, both caught by
   that
script's own required self-tests before any real result was trusted: a wrong
closed-form

backward-Euler solve (caused unbounded temperature blow-up) and a
phase-transition energy-

accounting bug (silently discarded the sensible-heat "overshoot" at melt onset).
Energy

conservation now holds to machine precision (~1e-13 relative residual) — see
physics_lib.py's

own module docstring for the full diagnosis.

## Research gaps addressed (N1–N6 novelty mapping)

### Important disambiguation

Two distinct gap/novelty systems exist:

- N1–N6 (framework doc §3, Table 3): Objective 1's own novelty positioning,
  specific to this
climate-signature/clustering/MCDM/validation pipeline.

- RG1–RG5 (project-wide framing): research gaps for the broader multi-objective
  project
(this objective plus downstream DRL-control and hardware-prototype objectives).
RG1–RG5 do

not appear in Objective1_PCM_Climate_Framework_Plan_v3.docx itself.

Conflating these two would misattribute claims — Objective 1 does not address
all five RG gaps

directly (only RG5); the others are fed by this objective's output but addressed
across multiple

objectives.

### Phase → N (novelty claim) mapping

| Phase | Primary N-claim(s) | How it contributes |
| --- | --- | --- |
| 1 — Data Collection | N6 | Population-weighted, sun-event-aligned sampling — not a uniform grid or arbitrary city list |
| 2 — Preprocessing & Validation | (supports all) | The deaccumulation-bug catch and QUANTILE_MAP decision are the evidentiary basis for claiming the climate backbone (Phases 3+) is trustworthy — without this phase, none of N1–N5 would be defensible |
| 3 — Climate Signature | N2, N3 | Two-tier signature (not a single temperature); Tm_target/L_required corrected to the 42–70°C SWH band (not the 18–28°C comfort band a naive approach might reuse) |
| 4 — Regime Clustering | N1 | GMM-discovered regimes (k=3, statistically selected, not hand-picked); external validation now PARTIALLY wired in (Köppen-Geiger, ARI=0.19/NMI=0.32) — N1's "discovered, not hand-picked" claim is now supported by internal statistical measures PLUS one external classification cross-check (NBC/ECBC still open) |
| 5 — Feasibility Filtering | N3 (partial) | Enforces the corrected 42–70°C band and SWH-specific constraints; database-size gap closed 2026-08-12 (18–25 → 55 rows, inside the 40–60 target) — N3's practical value depended on having enough real in-band candidates to filter; that prerequisite is now met, but Phase 5 has not yet been re-run against the expanded database, so N3's demonstrated value in the current on-disk output is still the pre-expansion result |
| 6 — MCDM Ranking | N4 | Four-method consensus + Monte Carlo, not a single TOPSIS winner; Kendall's W explicitly reports when consensus is not strong (Cluster 0, W=0.4375) rather than hiding disagreement — this honest reporting is itself part of N4's value proposition |
| 7 — Physics Validation (COMPLETE) | N5 | Independently validated the MCDM ranking against simulated solar fraction — the result is a genuine NEGATIVE validation (Spearman rho ≤0.4, all 3 clusters), not a confirmation. This is itself evidence for N5 as a methodology (the validation was performed rigorously and reported honestly, exactly per the framework doc's own "write it out plainly" instruction) even though it does not currently confirm the MCDM ranking's output — N5's claim should read "the ranking WAS physics-tested, honestly, with a negative result attributable in part to the still-undersized PCM database" not "the ranking IS physics-validated." See 19_PHASE_7_ONWARD.md. |
| 8 — Recommendation Cards (COMPLETE) | (packaging) | Aggregates N1–N5's evidence, including Phase 7's negative result and its caveats, into the final deliverable format — 10_recommendation_cards_rajasthan.py's own caveats section surfaces the physics-validation band per cluster, not just the MCDM Top-3 |

### Phase → RG (broader project research gap) mapping — explicitly indirect

Since RG1–RG5 belong to the broader multi-objective project rather than
Objective 1 itself, this

mapping describes how Objective 1's output feeds the later objectives that
directly address

RG1–RG4, and how Objective 1 itself directly addresses RG5:

| Phase | Related RG | Nature of contribution |
| --- | --- | --- |
| 1–2 (Data Collection, Validation) | RG5 | Supplies the validated, uncertainty-characterized climate data a later predictive-optimization-under-uncertainty component (RG5, "no predictive optimization under climatic uncertainty") would need as its own input |
| 3–4 (Signature, Clustering) | RG5 | Climate regimes are themselves a climatic-uncertainty-aware framing (population-weighted, statistically validated) — a direct, not merely feeding, contribution to RG5 |
| 5–6 (Feasibility, MCDM) | RG5 | Monte Carlo uncertainty propagation over PCM property/weight perturbation is Objective 1's own predictive-optimization-under-uncertainty contribution |
| 7 (Physics Validation) | RG4 (indirect) | A grey-box simulation is not a real-world experiment, but it is Objective 1's step toward the experimental-validation direction RG4 (limited real-world experimental validation) ultimately calls for — the framework doc itself frames Phase 7 as "what makes the result publishable, not skippable" |
| 8 (Recommendation Cards) | RG2, RG3 (feeding, not addressing) | The per-regime PCM recommendation is the direct input a later hardware-prototype objective (RG2) and demand-alignment objective (RG3) would consume — Objective 1 does not itself build a prototype or model household demand |
| — | RG1 | Not addressed by Objective 1 at all — real-time adaptive control is explicitly out of this objective's scope (framework doc §1.2) |

### Important note

This mapping does not assert that Objective 1 "solves" RG1–RG4 — only RG5 is
directly addressed by

this objective's own methodology (Monte Carlo uncertainty propagation,
regime-level rather than

single-point climate targets). RG1–RG4 are gaps the broader project addresses
across multiple

objectives, and Objective 1's role there is to produce a validated, regime-aware
PCM recommendation

that the later objectives can build on — not to close those gaps itself.
Presenting this mapping with

that distinction intact is more defensible in a viva than claiming Objective 1
single-handedly

addresses all five research gaps.

## What remains

Phases 1–8 are all now implemented and have been run end-to-end (via
run_all_rajasthan.py) from a

single consistent Phase 4 clustering pass. What remains is resolving what Phase
7's genuine

negative result means for the project's claims, not building more pipeline:

1. Regenerate PCM_Properties_cleaned_mice_pmm_detailed.csv and re-run Phases 5–8
   against the
now-expanded 55-row PCM database — this is now the single highest-leverage open
item, replacing the

database-expansion task itself (that part is done, see known issue 6 above).
Every Phase 6/7/8

output currently on disk is still tagged `pcm_database_status = "PROVISIONAL —
~25-row database, not

yet expanded to 40-60"` because it predates the expansion, and Phase 7's own
inherited-caveats

discussion (09_physics_validation_rajasthan.py's docstring) explicitly flags
that Cluster 0's

negative rho may be better explained by its undersized candidate pool (n=5) than
by a genuine

MCDM/physics disagreement. Re-running Phases 5-8 is not optional cleanup — it
will likely change

the result, not just the numbers. Concretely: python
PCM_data/PCM_data/01_preprocess.py

(regenerates the missing _detailed.csv), then `python run_all_rajasthan.py
--from

07_feasibility_filter_rajasthan.py`.

1. Decide and document the κ-relaxation policy for the latent-heat constraint
   (accept per-cluster
calibrated κ, or rank-by-proximity-to-L_required instead of hard-gating, per
Correction 4's own

recommendation in 04_climate_signature_rajasthan.py's docstring).

1. NBC/ECBC Indian climate-zone validation remains stubbed (Köppen-Geiger is now
   wired in — see
known issue 9 above).

1. Interpret and write up Phase 7's negative result properly (see
   19_PHASE_7_ONWARD.md) — this is
itself a real, reportable finding, not a failure to hide: it means the MCDM
ranking, as currently

weighted, is not confirmed by the physics simulation at the pipeline's current
PCM-database size,

and the honest next step is diagnosis (which criterion's weight, or database
expansion), not

re-running the simulation hoping for a different number.

## Recommended next step

The PCM database expansion (18→55 rows) is done — regenerate the missing

PCM_Properties_cleaned_mice_pmm_detailed.csv (python
PCM_data/PCM_data/01_preprocess.py), then

re-run the full chain from Phase 5 (`python run_all_rajasthan.py --from

07_feasibility_filter_rajasthan.py`) and see whether Phase 7's negative result
changes. Phase 7 was

deliberately run anyway against the pre-expansion provisional database — see
19_PHASE_7_ONWARD.md

for the reasoning and the full completion report, including why running it now
(rather than waiting)

was itself informative. Every number currently in
feasibility_survivors_rajasthan.csv,

mcdm_rankings_rajasthan.csv, physics_validation_rajasthan.csv,

spearman_rho_by_cluster_rajasthan.csv, and recommendation_cards_rajasthan.md
still reflects the

pre-expansion 18/25-row database and should be treated as superseded pending
this re-run.

# 2. 01_PROJECT_CONTEXT.md

Source path: /mnt/data/01_PROJECT_CONTEXT.md

# 01 — Project Context

## Identity

"OBJECTIVE 1 — IMPLEMENTATION PLAN," Climate-Region-Aware PCM Recommendation
Framework, Version 3.0

(supersedes v2.0), Group 12, B.Tech CSE Final Year, Amrita School of
Engineering. Governing document:

Objective1_PCM_Climate_Framework_Plan_v3.docx (extracted in full for this audit
— 16 numbered

sections, front matter, and an IEEE-style references list).

## Why four states

Section 1.3 (Table 1) names Rajasthan, Assam, Tamil Nadu, and Uttarakhand as the
four target states,

chosen to span distinct climate archetypes (arid/semi-arid, humid
subtropical/monsoon-heavy, coastal

tropical, and high-relief montane respectively) so the eventual multi-state
clustering run has genuine

climate diversity to discover regimes across, rather than four samples of the
same regime. This audit

covers Rajasthan only — the first of the four to reach Phase 6.

## Scope decomposition (§1.1–1.2)

Sub-goals SG1–SG4 (climate signature construction, regime discovery, PCM
feasibility+ranking,

physics validation) are explicitly bounded: out of scope for Objective 1 are
hardware prototyping,

DRL control, and real-time operation — those belong to later project objectives
that consume this

objective's output (the per-regime PCM recommendation) as an input, not to
Objective 1 itself.

## Deliverables (§1.4, Table 2, D1–D8)

Corresponds closely to the Phase 1–8 pipeline stages: D1 validated climate
dataset, D2 climate

signature + PCA, D3 regime clusters + external validation, D4 PCM
feasibility-survivor set, D5

MCDM ranking + Monte Carlo confidence, D6 physics-validated solar-fraction
ranking, D7

recommendation cards, D8 (implicit) the write-up/methodology section itself,
which this

documentation set is designed to directly support.

## Response to prior critical review (§2)

The v3.0 document is explicitly a correction pass over v1.0/v2.0, responding to
methodology

review on four points: clustering methodology (§2.1 — commits to GMM as primary,
K-Means only as a

reported comparison baseline, confirmed in 05_cluster_rajasthan.py), MCDM method
(§2.2 — commits to

a four-method stack, not a single TOPSIS-only ranking, confirmed in
08_mcdm_ranking_rajasthan.py),

PCM selection criteria (§2.3 — corrects the melting-point band to 42–70°C from
an earlier, apparently

wider or misaligned band), and validation strategy (§2.4 — adds Phase 7
physics-based validation as a

non-optional step, explicitly framed as "what makes the result publishable, not
skippable as future

work").

## Closest prior work and novelty position (§3, Table 3)

Six novelty claims, N1–N6 — this project's own framing of what it contributes
beyond existing

PCM-SWH literature:

| ID | Claim |
| --- | --- |
| N1 | Discovered climate regimes (GMM clustering) vs hand-picked climate zones |
| N2 | Two-tier climate signature (sun-event + daily-integral) vs a single representative temperature |
| N3 | Corrected 42–70°C SWH-specific PCM band vs 18–28°C building-thermal-comfort band (a common confusion in adjacent literature) |
| N4 | Top-3 + explicit method-agreement/consensus reporting vs a single declared "winner" PCM |
| N5 | Physics-validated ranking (Phase 7) vs a self-referential MCDM-only result |
| N6 | Population-weighted sampling/regime discovery vs uniform-grid or arbitrary-city sampling |

## Important disambiguation: N1–N6 vs RG1–RG5

Do not conflate these two systems — they come from different documents and serve
different

purposes:

- N1–N6 (above) are the framework doc's own novelty positioning, specific to
  Objective 1's
climate-signature/clustering/MCDM/validation pipeline.

- RG1–RG5 (research gaps: RG1 no real-time adaptive control, RG2 no integrated
  PCM–AI–hardware
prototype, RG3 poor alignment with household demand, RG4 limited real-world
experimental

validation, RG5 no predictive optimization under climatic uncertainty) come from
a separate

artifact, prompt for extraction.txt, the template used to generate every paper
summary in

PCM-Selection-ML-model/Sources/. Every one of the 21 literature summaries scores
itself against

RG1–RG5 in its own "Direct Relevance to My Project" section. RG1–RG5 belong to
the broader,

multi-objective project (this climate/PCM-selection objective plus the
downstream DRL-control and

hardware-prototype objectives), not to the Objective-1 framework doc's own phase
structure.

- 18_RESEARCH_GAP_MAPPING.md in this documentation set maps phases against both
  systems
explicitly, keeping them separate, because the framework doc itself never states
RG1–RG5 and a

phase→RG mapping that implies otherwise would misattribute a claim this document
doesn't make.

## Phase numbering — authoritative source

Confirmed directly from the framework doc (§4–§11), no phase-numbering
assumption was needed:

| Phase | Name |
| --- | --- |
| 1 | Data Collection (As Built) |
| 2 | Preprocessing and Cross-Source Validation |
| 3 | Climate Signature Construction |
| 4 | Climate Regime Clustering |
| 5 | Feasibility Filtering |
| 6 | Multi-Criteria Ranking Engine |
| 7 | Physics-Based Validation |
| 8 | Explanation and Final Output |

There is no "Phase 0" in the framework doc; §0 is a version-3.0 changelog, not a
phase. The

pipeline's own phases.md and script comments informally call the sampling-design
step (population

grid, sun times, elevation) "Phase 0" because it precedes and feeds Phase 1's
actual data download —

this documentation set keeps that informal label only where useful for
describing implementation

order, and always defers to the framework doc's Phase 1–8 numbering for anything
phase-labeled.

## How this documentation set was produced

Every phase audit in this set was built by (1) reading the actual pipeline
source files in full —

not skimmed, not inferred from filenames — (2) cross-checking every claimed
behavior against the

actual data files on disk (row counts, column headers, sample values), (3)
reading the framework

doc's own methodology text for the corresponding phase, and (4) checking the
project's literature

folder (Sources/) for what is and is not actually supported by a citable source.
Where code

comments/docstrings recorded a bug that was found and fixed, this is reported as
a finding, not

smoothed over — the project's own commit history of self-corrections
(accum_to_flux, GMM covariance

type, VIKOR sign, entropy weight) is itself evidence of a working, self-auditing
methodology and is

presented that way throughout this documentation set.

# 3. 02_DATA_SOURCES_AND_VARIABLES.md

Source path: /mnt/data/02_DATA_SOURCES_AND_VARIABLES.md

# 02 — Data Sources and Variables

## External data sources (exact, as requested in code)

| Source | Product | Access | Used by |
| --- | --- | --- | --- |
| Copernicus Climate Data Store (CDS) | reanalysis-era5-single-levels, product_type=reanalysis | cdsapi.Client, requires .cdsapirc credentials | 00c_attach_elevation.py, 01_download_era5_rajasthan.py |
| NASA POWER | Hourly point API, community=RE (Renewable Energy) | https://power.larc.nasa.gov/api/temporal/hourly/point, no API key | 01b_download_nasapower.py |
| GADM v4.1 | India admin level 1 boundary (GeoJSON) | https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json | 00a_build_population_grid.py |
| WorldPop | India 2020 unconstrained population, UN-adjusted, 100 m | https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_UNadj.tif (~1.5–2 GB) | 00a_build_population_grid.py |

## ERA5 variables requested (exact CDS short names, from 01_download_era5_rajasthan.py)

Instant (analysis, TYPE=AN) — snapshot values, no deaccumulation needed:

2m_temperature → t2m → T_amb (K → °C)
2m_dewpoint_temperature → d2m → T_dew → RHum (Magnus formula)
10m_u_component_of_wind → u10 → W_spd, W_dir (m/s, °)
10m_v_component_of_wind → v10 → (combined with u10)
total_cloud_cover → tcc → cloud_cover (0–1, unconverted)
surface_pressure → sp → P_atm (Pa → hPa)

Accumulated (forecast, TYPE=FC) — see 13_SOLAR_DERIVED_VARIABLES.md for why
"accumulated" is

in scare quotes for this pipeline's actual download:

surface_solar_radiation_downwards → ssrd → GHI (J/m² per downloaded hour → W/m²)
mean_surface_direct_short_wave_radiation_flux → msdwswrf → avg_sdirswrf → DNI
(already W/m²)
surface_thermal_radiation_downwards → strd → LW_down (J/m² → W/m²)
total_precipitation → tp → precipitation (m → mm)

00c_attach_elevation.py additionally requests geopotential (single
time-invariant field, one

API call, 2020-01-01T00:00) → elevation_m = z / 9.80665.

## NASA POWER parameters (exact, from 01b_download_nasapower.py)

ALLSKY_SFC_SW_DWN — all-sky surface shortwave downward irradiance (≈ GHI
equivalent)
CLRSKY_SFC_SW_DWN — clear-sky surface shortwave downward irradiance
T2M — 2 m temperature
RH2M — 2 m relative humidity
WS10M — 10 m wind speed

Fill value -999 is replaced with NaN on ingest (02_combine_rajasthan.py,
blanket, no

column-specific bound check). PRECTOTCORR (precipitation) was never requested —
confirmed by

direct code inspection — which is why monsoon_index (Tier 2) is always a
GHI-fraction proxy in

this pipeline, never a true precipitation-derived index (see
16_CLIMATE_SIGNATURE.md).

## Full variable transformation table

| Variable | ERA5/POWER name | Original unit | Stored unit | Transformation | Validation |
| --- | --- | --- | --- | --- | --- |
| Air temperature | t2m / T2M | K | °C | −273.15 | Range check [−5, 60]°C |
| Dew point | d2m | K | °C | −273.15 | none dedicated |
| Relative humidity | derived from t2m,d2m / RH2M | — | % | Magnus-Tetens (Alduchov & Eskridge 1996, a=17.625, b=243.04) | clip [0,100] |
| Wind speed | u10,v10 / WS10M | m/s | m/s | √(u²+v²) | Range check [0,40] m/s |
| Wind direction | u10,v10 | — | ° | (degrees(atan2(u,v))+360) mod 360 | none |
| Surface pressure | sp | Pa | hPa | /100 | Range check [800,1050] hPa |
| Cloud cover | tcc | fraction | fraction | none | Range check [0,1] |
| GHI | ssrd | J/m² (per downloaded hour) | W/m² | accum_to_flux(x)/3600, clip≥0 | Range check [0,1400] W/m² |
| DNI (primary) | msdwswrf/fdir/msdrswrf | already W/m² (assumed) | W/m² | clip(0,1400) only, no /3600 | Range check [0,1400] |
| DNI (fallback) | derived from GHI, SZA | — | W/m² | GHI/cos(SZA) where cosZ>0.05, clip[0,1400] | same |
| DHI | derived | — | W/m² | (GHI − DNI·cosZ), clip≥0 (residual, not modeled) | Range check [0,1400] |
| Clear-sky GHI | pvlib Ineichen model | — | W/m² | model output | Range check [0,1400] |
| Clearness index (CSI) | GHI/GHI_clearsky | — | dimensionless | forced 0 if GHI_clearsky≤10, else clip[0,1.5] | QC bound [0,2] — looser than pipeline clip, dead check |
| Longwave down | strd | J/m² | W/m² | accum_to_flux(x)/3600, clip≥0 | Range check [0,700] |
| Precipitation | tp | m | mm | accum_to_flux(x)×1000, clip≥0 | Range check [0,200] |
| Solar zenith angle | pvlib get_solarposition | — | ° | direct | Range check [0,180] |
| Solar azimuth | pvlib get_solarposition | — | ° | direct | Range check [0,360] |
| Elevation | ERA5 z (geopotential) | m²/s² | m | /9.80665 (standard gravity) | Outlier flag [−420, 8850] m (Dead Sea..Everest), not clipped |
| ETR (extraterrestrial) | pvlib get_extra_radiation | — | W/m² | computed | computed but never written to output CSV |

See 13_SOLAR_DERIVED_VARIABLES.md for the DNI/DHI derivation logic in full, and

09_ERA5_DATA_PIPELINE.md for the deaccumulation story that motivates the
"already W/m²" caveat on

GHI/LW/precip above.

## Column-name ambiguity worth flagging

avg_sdirswrf is populated from whichever of msdwswrf, fdir, or msdrswrf matches
first in the

downloaded NetCDF (next((c for c in df.columns if c in (...)), None)). These are
not the same

physical quantity in ERA5's variable catalogue: fdir is an accumulated
direct-radiation field

(needs the same J/m²→W/m² treatment as ssrd); msdwswrf/msdrswrf are mean-rate
fields (already

W/m², no conversion needed). The code applies identical treatment (clip only, no
/3600) regardless

of which one actually matched — see 20_IMPLEMENTATION_ISSUES.md item 8 for the
audit consequence.

## Output variable list (ERA5_OUTPUT_VARS, exact, from 02_combine_rajasthan.py)

T_amb, T_dew, RHum, W_spd, W_dir, GHI, DNI, DHI, LW_down, cloud_cover,
precipitation, P_atm, SZA, solar_azimuth, GHI_clearsky, CSI

Prefixed era5_ in the combined CSV; the five NASA POWER variables are prefixed
power_.

# 4. 03_PHASE_1_AUDIT.md

Source path: /mnt/data/03_PHASE_1_AUDIT.md

# 03 — Phase 1 Audit: Data Collection

Scripts: 00a_build_population_grid.py, 00b_build_suntimes.py,
00c_attach_elevation.py,

01_download_era5_rajasthan.py, 01b_download_nasapower.py, 00_unzip_accum.py.

## Purpose

Establish where and when to sample climate data, then pull two independent
sources (ERA5,

NASA POWER) for exactly those points/times. The "where/when" design choice —
population-weighted

points sampled at astronomically-computed sun-event times, instead of a uniform
grid on fixed clock

hours — is the pipeline's own stated departure from the more common uniform-grid
approach, and it is

the reason every later phase samples 320 points × 3 events/day rather than a
full spatial grid ×

24 hours/day.

## Inputs

None upstream — this is the first stage. External: GADM boundary, WorldPop
raster, ERA5 CDS API,

NASA POWER API.

## Processing

### Population-weighted sampling grid (00a_build_population_grid.py)

1. Download GADM v4.1 India admin-1 boundary, filter to NAME_1 == "Rajasthan".
1. Download WorldPop India 2020 UN-adjusted 100 m population raster, clip to the
   Rajasthan boundary.
1. Aggregate pixel population onto a 0.25° grid deliberately aligned to ERA5's
   own grid origin
(lat=90.0, lon=-180.0) — this is a load-bearing design choice: it guarantees
each selected

sampling point's cell center lands exactly on an ERA5 grid node, so the
population→ERA5 mapping

is 1:1 wherever cells are genuinely distinct, rather than two nearby population
cells silently

collapsing onto the same ERA5 node due to grid misalignment.

1. Rank cells by population descending, keep the minimal set whose cumulative
   population reaches
COVERAGE_TARGET = 0.875 (87.5%, middle of a stated 85–90% target band).

1. weight = population / population.sum() — renormalized over the selected
   320-point subset,
not the full state population.

Result: 320 points, point_id format RJP_{0001..0320}.

### Sun-event times (00b_build_suntimes.py)

For every point × every date 2016-01-01..2025-12-31, computes
sunrise/solar-noon/sunset via

pvlib.location.Location.get_sun_rise_set_transit(dates, method="spa") — Reda &
Andreas (2004)

Solar Position Algorithm, no manual equation-of-time code. altitude=0 is
hardcoded for this call

(elevation isn't yet attached to points at this pipeline stage, and even the
later-attached

elevation is never fed back into this specific computation — a minor, low-impact
omission since

altitude's effect on sunrise/sunset timing itself is negligible, though it does
matter for the solar

position/irradiance calculations done later in 02_combine_rajasthan.py, which do
use the real

elevation).

Ground-truthed row count: 3,506,880 = 320 points × 3653 days (2016–2025,
including leap years

2016/2020/2024) × 3 events — matches the formula exactly.

### Elevation attachment (00c_attach_elevation.py)

Downloads ERA5's time-invariant geopotential field (z), one API call for a
single date/time

(orography doesn't change), and attaches elevation_m = z / 9.80665 per point via
nearest-neighbor

lookup on the geopotential grid. Replaces a flat 300 m fallback that
02_combine_rajasthan.py

otherwise uses. Sanity-checks outliers against [−420, 8850] m (Dead Sea to
Everest) but does not

clip or drop them — only warns.

### ERA5 download (01_download_era5_rajasthan.py)

Downloads three narrow UTC hour windows per month (sunrise/noon/sunset ± margin)
instead of fixed

clock hours, using a circular (mod-24) window algorithm to correctly handle sun
events that

straddle the UTC midnight boundary (documented real case: an eastern point's
summer sunrise can land

at 23:55 UTC of the previous calendar date). Two API calls per (year, month):
instant variables

(analysis type) and accumulated variables (forecast type, with each instant
hour's immediate

predecessor also requested — needed for the deaccumulation step, see
09_ERA5_DATA_PIPELINE.md).

10 years × 12 months × 2 var-types = 240 calls.

### NASA POWER download (01b_download_nasapower.py)

Full hourly year, per point, for the 5 parameters listed in
02_DATA_SOURCES_AND_VARIABLES.md.

320 points × 10 years = 3200 calls.

### Zip-quirk fix (00_unzip_accum.py)

CDS API v2 sometimes returns a ZIP archive even when download_format: unarchived
is requested;

this detects (PK magic bytes) and fixes *_accum.nc files in place, scanning both
the legacy

full-grid archive and the new points archive.

## Scientific reasoning

Population-weighting the sampling grid (rather than uniform spatial sampling)
directly serves the

project's downstream deliverable: a climate signature and PCM recommendation
that is meaningful for

where people actually live, not for empty desert cells that would otherwise
dilute a uniform

average. Sun-event-aligned sampling (rather than fixed clock hours) is the
correct choice for a

solar-thermal application specifically because the physically meaningful
instants — when charging

starts (sunrise), peak charging (noon), and when discharge begins (sunset) — are
what the downstream

Tier-1 climate signature and Tm_target/L_required derivations are actually built
from.

## Spatial Processing Justification

ERA5 grid alignment (0.25° to ERA5's own grid origin):

Each population-weighted sampling point's cell center is deliberately aligned to
an ERA5 grid node.

This 1:1 population-to-ERA5 mapping eliminates grid-misalignment error and
justifies the

nearest-neighbor (not interpolated) extraction method used downstream in Phase
2.

Rajasthan boundary & population aggregation:

GADM v4.1 admin-level-1 boundary provides the state border; WorldPop 100 m
raster supplies

per-pixel population. The 87.5% population-coverage target ensures results are
defensible for

where people actually live, not over-weighted toward sparse/desert regions. This
choice directly

supports the downstream deliverable: regime-level PCM recommendations, not
point-exact microclimate

models.

Nearest-neighbor grid extraction:

No interpolation is used when extracting ERA5 or elevation values. Each point
inherits its

containing 0.25° cell's value exactly. This is correct for the
population-aligned, regular-grid

design, though it means nearby points in the same cell receive identical ERA5
readings — an expected,

harmless consequence of the sampling design.

Elevation handling:

ERA5's geopotential gives grid-cell-mean elevation, not point-exact local
elevation. This is

acceptable for Rajasthan's comparatively flat terrain (mostly 200–500 m), though
it would matter more

for high-relief states. The pipeline does not attempt to retroactively reweight
the population grid

by elevation; elevation is used only downstream for solar-position calculations
in Phase 2.

Why this spatial approach is appropriate:

The goal is regime-level PCM recommendations across representative,
population-weighted points, not

microclimate modeling of every location. The spatial design is internally
consistent and

correctly-reasoned for this stated purpose.

## Temporal Processing Justification (Dates, Times, Sunrise/Sunset)

UTC as sole time reference:

All timestamps are UTC (time_utc in suntimes.csv, ERA5's native timezone, NASA
POWER requested

with time-standard=UTC). No IST (India Standard Time, UTC+5:30) conversion is
applied upstream.

This is reasonable (UTC avoids daylight-saving/timezone-drift issues) and
consistent internally, but

any figure intended for a general audience ("sunrise at 6 AM") needs explicit
UTC→IST conversion at

presentation time, not before.

Sunrise/noon/sunset via pvlib SPA:

pvlib.location.Location.get_sun_rise_set_transit(dates, method="spa") implements
Reda & Andreas

(2004) Solar Position Algorithm. No manual equation-of-time code. The altitude=0
hardcoding for

this specific call (minor inconsistency with elevation-aware geometry later;
negligible impact on

sunrise/sunset clock time, though it does matter for solar position/irradiance
downstream).

Cross-midnight UTC handling (circular-window algorithm):

Real, documented case: an eastern Rajasthan point's summer sunrise can land at
23:55 UTC of the

previous UTC calendar date (e.g. Dholpur, 2020-06-21 sunrise at 2020-06-20
23:55:54 UTC). The

circular_hour_window() algorithm in 01_download_era5_rajasthan.py correctly
handles this by

finding the largest unobserved circular gap in the sorted hour set, taking the
rest as the "arc,"

then padding and wrapping with modulo-24 arithmetic. This is a correct, general
solution to a

genuine, common edge case, not a hack.

Leap years and date range:

2016-01-01 through 2025-12-31 inclusive = 3653 days (correctly includes leap
years 2016, 2020,

2024: 10×365 + 3 = 3653). Ground-truthed directly: 320 points × 3653 days × 3
events = 3,506,880

rows, exact match.

Nearest-in-time matching (3-hour rejection window):

When pairing a sun-event instant to an ERA5 or POWER timestamp, a match farther
than 3 hours is

rejected, turning missing/sparse readings into NaN rather than wrong pairings.
Applied

independently to both sources — no requirement that ERA5 and POWER share the
same matched timestamp.

This is a genuine gap worth noting: the actual matched times are never recorded
(only the requested

time_utc appears in output), so rejection-window diagnostics are difficult
without adding output

columns.

Sun-event-aligned vs. fixed-clock-hour sampling:

Sampling at astronomically-computed sunrise/noon/sunset (not fixed
02:00/08:00/14:00 UTC) ensures

the sampled instants are physically meaningful for solar-thermal systems across
all 320 points, all

seasons, all 10 years. A fixed-clock-hour scheme would sample "sunrise" at
genuinely different solar

elevation angles depending on season/longitude, contaminating sunrise-indexed
climate indices with

seasonal/spatial artifacts unrelated to actual climate. Sun-event alignment is
essential for the

downstream climate-signature construction's validity.

Seasonal definitions:

02_combine_rajasthan.py's SEASON_MAP (Winter=Dec-Feb, Summer=Mar-May,
Monsoon=Jun-Aug,

Retreat=Sep-Nov) is currently inconsistent with 02b_build_daily_aggregates.py's
monsoon window

(Jun-Sep). signature_lib.py matches 02_combine_rajasthan.py by design (Jun-Aug),
so the *season

column used in Tier-1 clustering is consistent, but the monsoon_index* feature
is computed against

Jun-Sep. Reconcile before final write-up (either both Jun-Aug or both Jun-Sep,
justified against IMD

convention, which typically treats Jun-Sep for Rajasthan).

## Literature support

Reda & Andreas (2004), "Solar position algorithm for solar radiation
applications," Solar Energy

76(5) — cited by name in 00b's docstring as the algorithm pvlib's method="spa"
implements.

Hersbach et al. (2020), "The ERA5 global reanalysis," QJRMS 146(730) — the ERA5
product's own

citation (per the framework doc's §15 reference list; not separately re-verified
in this pass beyond

confirming the framework doc names it). WorldPop and GADM are cited as
data-source products, not

peer-reviewed methodology claims.

## Validation

03_verify_climate_csv.py Check 2 (point coverage) and Check 3 (row coverage)
validate this phase's

output indirectly, downstream, in Phase 2. No dedicated Phase-1-only validation
script exists;

03_qc_plots.py's population/elevation/download-status maps serve this role.

## Outputs

population_grid_points.csv (320×6 cols incl. elevation_m), suntimes.csv
(3,506,880×4 cols),

data/raw/era5/points/.nc (240 files, 816 MB), data/raw/nasapower/.json (3200
files, 2.47 GB),

download_status_points.csv, download_status_power.csv.

## Dependencies

Nothing upstream. Every later phase depends on this phase's point set and
sun-event times being

fixed — re-running 00a/00b with different parameters would silently invalidate
every downstream

file without an automatic re-trigger (no dependency-graph enforcement exists in
this pipeline; it is

a linear script-order convention, not a build system).

## Problems / risks

- No re-verification of stale outputs: 00a's population-grid CSV is
  unconditionally
recomputed and overwritten on every run (no skip logic), but nothing downstream
detects if the

point set changed since suntimes.csv/ERA5/POWER were built against an older
version — a silent

point-set/downstream-data mismatch is possible if 00a is re-run without
re-running the entire

chain after it.

- 00b's altitude=0 hardcoding for the SPA sunrise/sunset computation is a minor
  inconsistency
with the elevation-aware solar geometry used later in 02_combine_rajasthan.py,
though its

practical effect on sunrise/sunset clock time is negligible.

- Ground-truth confirms full completion: 240/240 ERA5 files, 3200/3200 (after 1
  retry) POWER
files — no incomplete-download risk currently outstanding for Rajasthan.

## Status

COMPLETE.

# 5. 04_PHASE_2_AUDIT.md

Source path: /mnt/data/04_PHASE_2_AUDIT.md

# 04 — Phase 2 Audit: Preprocessing, Cross-Source Validation, and Quality Control

Scope of this file: Phase 2 (raw combine + cross-source validation) and Phase
2.5 (quality

control + cleaning), combined into a single audit because Phase 2.5 sits
directly between Phase 2

and Phase 3 and cannot be understood in isolation from Phase 2's output.

Scripts covered:

- Phase 2: 02_combine_rajasthan.py, 02b_build_daily_aggregates.py,
03_verify_climate_csv.py, 03_qc_plots.py, 03b_agreement_analysis.py,

03c_plots_raw_rajasthan.py (added 2026-08-11, raw QC plots)

- Phase 2.5: 03b_quality_check_rajasthan.py,
  03b_validate_quality_fix_rajasthan.py,
03c_plots_raw_rajasthan.py, 03b_quality_check_plots_rajasthan.md

Cross-references: 20_IMPLEMENTATION_ISSUES.md (items 1 and 7),
00_MASTER_OVERVIEW.md

(overall pipeline status). All supporting details now embedded in this file.

Critical context (documentation history): Phase 2.5 was implemented on disk
(code exists,

script runs, outputs produced) but was entirely undocumented in the
docs/rajasthan/ folder until

2026-08-11, despite Phase 3 (04_climate_signature_rajasthan.py) having
explicitly read its CLEAN

output since that same date. This was the single most factually-wrong gap in the
doc set prior to

consolidation: Phase 3 does not read Phase 2's raw output directly (a widespread

misunderstanding) — it reads Phase 2.5's quality-checked output,
climate_rajasthan_points_clean.csv.

Pipeline order at a glance:

Phase 1 (raw NetCDF/JSON, points, suntimes)
 ↓
Phase 2 — 02_combine_rajasthan.py, 02b_build_daily_aggregates.py,
 03_verify_climate_csv.py, 03_qc_plots.py, 03b_agreement_analysis.py
 ↓ climate_rajasthan_points.csv (RAW, 34 cols)
Phase 2.5 — 03b_quality_check_rajasthan.py,
03b_validate_quality_fix_rajasthan.py
 ↓ climate_rajasthan_points_clean.csv (CLEANED)
Phase 3 — 04_climate_signature_rajasthan.py (reads the CLEAN file)

────────────────────────────────────────

# PART A — Phase 2: Preprocessing and Cross-Source Validation

This is the most scientifically consequential phase in the pipeline. See

14_ERA5_POWER_VALIDATION.md for the full validation story and
09_ERA5_DATA_PIPELINE.md for the

deaccumulation deep-dive.

## A.1 Purpose

Convert raw NetCDF/JSON into physical-unit, quality-controlled,
cross-source-validated tabular data,

and — critically — decide whether ERA5 alone is defensible as the climate
backbone, before any

downstream index construction touches the physical values.

## A.2 Inputs

data/raw/era5/points/.nc, data/raw/nasapower/.json, population_grid_points.csv,

suntimes.csv (all from Phase 1).

## A.3 Processing

### ERA5 Accumulated Fields & Deaccumulation — The Critical Bug Fix

Mandatory audit checkpoint: The deaccumulation story. This single fix determined
whether all

downstream analysis (Phases 3–6) was built on physically valid GHI data.

What was originally assumed: ERA5's accumulated fields (ssrd, strd, tp) follow
the classic

MARS convention: cumulative since last forecast reset (00Z or 12Z), requiring
diff() against the

previous hour to recover hourly flux, with special case at post-reset hours (1
and 13). An earlier

function deaccumulate() implemented exactly this, with
01_download_era5_rajasthan.py deliberately

downloading each target hour's predecessor to feed the diff.

What was actually found: 03b_agreement_analysis.py flagged ERA5-vs-POWER GHI as
physically

implausible (median ERA5 ~2 W/m² vs POWER ~37 W/m² at same instants, noon
Pearson r≈0.01).

Tracing to raw NetCDF showed 34–44% of consecutive-hour raw values were lower
than their

predecessor within the same accumulation cycle — impossible for genuine
cumulative-since-reset

(which can only increase monotonically until reset). Conclusion: each hour for
this pipeline's CDS

request is already its own ~1-hour accumulated value, not a running total.

The fix — accum_to_flux(), simple and correct:

def accum_to_flux(s):
 s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
 return s.clip(lower=0)

No diffing at all. Stateless clip-to-nonnegative. The function was renamed from
deaccumulate()

specifically so a future edit would not casually reintroduce a diff step.
Post-fix verification:

Physics-correct GHI with seasonal peaks (~900 W/m² pre-monsoon, ~700 W/m²
monsoon, ~650 W/m²

winter). Solar-noon ERA5-vs-POWER: MBE=10.95 W/m², RMSE=113.8 W/m², Pearson
r=0.810

(n=1,168,960) — categorical improvement from pre-fix r≈0.01.

Unit conversion correctness: GHI = accum_to_flux(ssrd)/3600 (J/m² → W/m²,
correct given the

"already per-hour" premise). LW_down identical treatment. precipitation =
accum_to_flux(tp)×1000

(m → mm).

One unresolved inconsistency: avg_sdirswrf (DNI surrogate) receives .clip(0)
regardless of

which ERA5 field matched (msdwswrf/fdir/msdrswrf). Only correct if matched field
is always a

mean-rate variant — not independently verified against actual NetCDF variable
names. This is a

plausible unit-error risk and should be checked before DNI is presented as fully
validated (see

issues in 20_IMPLEMENTATION_ISSUES.md item 8).

### 02_combine_rajasthan.py — the merge/physics script

1. Nearest-grid-cell snap (two independent 1-D argmins on lat/lon — correct for
   a regular grid,
would not generalize to a curvilinear one) — once per point, not per event.

1. Concatenate each point's full hourly series across all years, apply
   accum_to_flux() (stateless
clip, no diffing) to the accumulated fields, apply unit conversions.

1. Compute solar geometry via pvlib.location.Location(...).get_solarposition()
   and
.get_clearsky(model="ineichen") — see 12_SOLAR_GEOMETRY.md.

1. Derive GHI/DNI/DHI/CSI — see 13_SOLAR_DERIVED_VARIABLES.md.
1. For each (point_id, date, event) row in suntimes.csv, nearest-in-time match
   against both the
ERA5 series and the NASA POWER series independently, each rejected if farther
than

MAX_MATCH_HOURS = 3 from the true event time.

1. Apply physical-plausibility bounds (GHI>1400→NaN, T_amb<−5 or >60→NaN, RH
   clip[0,100], etc.)
### 02b_build_daily_aggregates.py — Tier-2 daily integrals (NASA POWER only)

climate_rajasthan_points.csv has only 3 samples/day — insufficient for true
daily energy

integrals. This script reads the already-cached full-hourly NASA POWER JSON
directly (no

re-download) and trapezoidally integrates GHI/clear-sky GHI over UTC hour-of-day

(numpy.trapz/trapezoid, requires ≥2 valid hourly points/day), producing:

- GHI_daily_kWh
- SAI (confirmed identical to kt_daily_mean)
- kt_daily_mean / kt_daily_std
- cloudy_frac (kt < 0.3, an undocumented-elsewhere threshold)
- CCI (Pearson r between daily GHI and daily clear-sky GHI, n ≥ 3)
- HDD18 / CDD24 (base 18 °C / 24 °C degree-days)
- DTR_true (true daily max−min)
- seasonality (coefficient of variation of monthly-mean GHI)
- monsoon_index (Jun–Sep GHI fraction — a proxy, since PRECTOTCORR was never
  downloaded)
### 03_verify_climate_csv.py and 03_qc_plots.py — QA

Six ordered checks (schema, point coverage, row coverage, null rates,
physical-sanity range checks,

cross-source correlation) — see §B (Part 1: Sanity Checks) below and
15_QUALITY_CONTROL.md for

the full threshold table. Eight QC visualizations (spatial folium maps +
distributional plotly

charts) — see the QC section of 15_QUALITY_CONTROL.md.

### 03b_agreement_analysis.py — the decision engine

Computes MBE/RMSE/Pearson r for GHI, T_amb, RHum, W_spd, stratified by season ×
sun-event (80 rows

total), applies a pre-registered three-branch decision rule at solar noon
specifically (BACKBONE /

QUANTILE_MAP / MANUAL_REVIEW), and — because the actual data landed in
QUANTILE_MAP — fits and

reports (but does not persist back into the dataset) an empirical 100-quantile
mapping of ERA5 GHI

onto the POWER distribution, per season. Full numbers and decision text in

14_ERA5_POWER_VALIDATION.md.

## A.4 Code mapping

02_combine_rajasthan.py
 ↓ accum_to_flux() + apply_unit_conversions()
 ↓ compute_solar() [pvlib]
 ↓ nearest_row() [±3h match]
 ↓
climate_rajasthan_points.csv (34 columns, 1 row/point/date/event)

03b_agreement_analysis.py
 ↓ compute_stats() [MBE/RMSE/r]
 ↓ decide_branch() [BACKBONE|QUANTILE_MAP|MANUAL_REVIEW]
 ↓ apply_quantile_mapping() [only if QUANTILE_MAP]
 ↓
era5_power_agreement_rajasthan.csv, bias_decision_rajasthan.txt

## A.5 Temporal Processing in the Merge

Nearest-in-time matching: For each (point_id, date, event) row in suntimes.csv,
the merge

in 02_combine_rajasthan.py independently matches ERA5 and POWER timestamps using

nearest_row(series_df, target_time, max_hours=3). A match farther than 3 hours
from the true

sun-event instant is rejected, turning missing/sparse readings into NaN rather
than wrong

pairings. Importantly, ERA5 and POWER can use different actual matched
timestamps (e.g., ERA5 up to

3h before the event, POWER up to 3h after) — there is no requirement for
cross-source temporal

alignment.

Gap: unrecorded matched timestamps. The actual matched time is never persisted;
only the

requested time_utc appears in climate_rajasthan_points.csv. This is a genuine,
acknowledged

limitation — adding era5_matched_time_utc/power_matched_time_utc output columns
would both

enable already-written QC diagnostics and let reviewers verify how often sources
are paired from

meaningfully different instants. Currently low-cost to fix; currently not fixed.
This gap

propagates into Phase 2.5, where it structurally disables 03_qc_plots.py's
rejection-window

diagnostic and forces 03b's MANUAL_REVIEW-branch diagnostics onto an SZA-based
proxy instead of a

direct time-offset measurement (see §B.3).

Missing/duplicated timestamp handling: Duplicated (point_id, date, event)
combinations are

flagged as hard FAIL in 03_verify_climate_csv.py Check 3. Missing timestamps
become NaN rows

(via the 3-hour rejection window). No special-case handling exists for the
documented "2016-01-01

edge case" — the referenced mechanism (predecessor-hour dependency in
deaccumulation) was fixed

upstream (accum_to_flux() is stateless), so this comment is likely a stale
reference worth

reconciling against current code before methodology write-up.

## A.6 Solar Geometry (why it's computed this way)

Solar position algorithm (get_solarposition): Called without explicit method=
argument,

relying on pvlib's default (likely NREL SPA in current versions).
Recommendation: pin method

explicitly before final write-up for reproducibility, or record installed pvlib
version.

Sunrise/sunset computation (in Phase 1) does explicitly pin method="spa", so
that path is

reproducible; solar-position computation should match.

Clear-sky model (Ineichen): get_clearsky(times, model="ineichen") with default
Linke-turbidity

climatology lookup. Standard, defensible choice for this project's scope — a
location-specific

measured turbidity record would be excessive burden. Note: Rajasthan's actual
aerosol loading (dust

storms) may deviate from climatological default on specific days, affecting
GHI_clearsky and thus

CSI — worth a caveat in methodology, not a required fix.

Altitude usage: alt_m = point_row.elevation_m if present, else 300 m fallback.
Feeds

atmospheric-pressure/airmass assumptions in Ineichen model and a small
refraction correction in

solar-position computation. Since Phase 1 now populates real elevation for all
320 points, the

fallback is defensive only.

Nighttime handling (division-by-zero protection): CSI (clearness index) forced
to exactly 0

(not NaN) where GHI_clearsky ≤ 10 W/m² (nighttime and near-sunrise/sunset where
ratio is

numerically unstable). This suppresses an "undefined" ratio into a defined zero
— defensible

practical choice (keeps column always numeric), but CSI=0 in output could mean
either "genuinely

clear-sky-free" or "ratio was unstable and suppressed" — not distinguishable
from output alone.

## A.7 Solar-Derived Variables (construction & assumptions)

GHI (Global Horizontal Irradiance): GHI = accum_to_flux(ssrd)/3600, clipped ≥0.
This is the

pipeline's most consequential derived variable and the one that surfaced the
deaccumulation bug (see

09_ERA5_DATA_PIPELINE.md).

DNI (Direct Normal Irradiance) — two-branch derivation, neither a true
decomposition model:

- Branch 1 (primary): DNI taken directly from ERA5's direct-radiation field
(msdwswrf/fdir/msdrswrf), not decomposed from GHI. Correctness depends on field
unit

convention matching code assumption.

- Branch 2 (fallback): DNI = GHI / cos(SZA) where cos(SZA) > 0.05, else 0. Crude
  algebraic
closure (how much direct beam is needed at this sun angle to account for all
GHI, if zero

diffuse) — not a genuine decomposition model like DISC/Erbs/DIRINT. Branch 1
likely used

essentially always (direct radiation field requested unconditionally), so Branch
2 rarely

exercised, though this was not independently confirmed.

DHI (Diffuse Horizontal Irradiance) — a closure residual, not independently
modeled:

DHI = (GHI − DNI·cos(SZA)).clip(0). By construction, always exactly satisfies

GHI = DHI + DNI·cos(SZA) — it is never independently modeled or observed. Any
error in GHI or DNI

propagates entirely into DHI; DHI cannot be used as an independent cross-check
on the other two

variables.

Clearness Index (CSI): CSI = GHI/GHI_clearsky, clipped [0, 1.5] in pipeline and
forced 0

below 10 W/m² threshold (see nighttime handling above). Note: the plausibility
check in

03_verify_climate_csv.py allows [0, 2], which is looser than the actual [0, 1.5]
clip — makes

that QC check structurally redundant (can never fire). See §B.2, Check 5, for
the identical issue

restated in Phase 2.5's own QC layer.

Unit-consistency caveat (open): avg_sdirswrf column-matching logic treats three
ERA5 field

names as interchangeable with identical treatment, regardless of field type.
fdir is accumulated

(would need /3600 conversion); msdwswrf/msdrswrf are already mean-rate W/m²
(correctly need no

conversion). Audit did not independently verify which name is actually present
in downloaded NetCDF

files — 01_download_era5_rajasthan.py requests msdwswrf specifically (already
correct), so

practice is likely always hitting the correct path, but code's generality
represents latent risk if

variable list changes. Recommend verifying directly before final write-up.

## A.8 Cross-Source Validation Decision (why QUANTILE_MAP was chosen)

Variable pairs compared: ERA5 GHI ↔ NASA POWER ALLSKY_SFC_SW_DWN, plus T_amb,
RHum, W_spd.

Matching: Reuses Phase 2's row-level merge — same point, same (date, event),
each source

independently nearest-in-time-matched within 3 hours of true event instant.
Note: ERA5 and POWER

can in principle match to different actual instants within that window (matched
timestamps never

persisted).

Decision rule thresholds (evaluated at solar noon only):

- BACKBONE (no correction): r ≥ 0.90 AND |MBE|/mean(POWER GHI) ≤ 5% AND max–min
  season MBE spread
≤ 5%

- QUANTILE_MAP (empirical correction): r ≥ 0.70 but stricter conditions fail
- MANUAL_REVIEW: r < 0.70 or undefined
- Fixed-weight blending explicitly rejected by design ("no principled derivation
  for fixed weight
between independent reanalysis/satellite-derived products")

Rajasthan result — actual numbers for write-up:

- Overall: MBE = 6.94 W/m², RMSE = 83.34 W/m², r = 0.9727
- Solar noon (decision-driving row): MBE = 10.95 W/m², RMSE = 113.79 W/m², r =
  0.8102
- Per-season noon MBE spread: 73.88 W/m² (10% of mean daytime GHI, exceeds 5%
  gate)
- Decision: QUANTILE_MAP — r_noon = 0.8102 fails BACKBONE's ≥0.90 gate but
  clears the ≥0.70
floor.

- Quantile mapping fit independently per season on daytime rows; RMSE improved
  4/4 seasons, r
improved 3/4.

Critical caveat: Quantile-mapped GHI is never persisted — the correction is
reported

(before/after diagnostic) but not written back to a dataset Phase 3 reads. Phase
3 currently

consumes uncorrected (though already deaccumulation-fixed) ERA5 GHI values. This
is an open

decision: either apply the correction upstream before Phase 3, or explicitly
document in the

write-up that Phase 3+ intentionally uses raw (not bias-corrected) ERA5 GHI and
why that's still

defensible.

## A.9 Mathematical operations

- RH (Magnus-Tetens): RH = 100·exp(a·Td/(b+Td)) / exp(a·T/(b+T)), a = 17.625, b
  = 243.04.
- MBE: mean(ERA5 − POWER) (positive = ERA5 overestimates).
- RMSE: √mean((ERA5−POWER)²).
- Pearson r via pandas .corr().
- Quantile mapping: 101-point empirical quantile-to-quantile piecewise-linear
  interpolation
(np.interp), fit independently per season on daytime (ERA5 GHI>0) rows.

## A.10 Literature support

Alduchov & Eskridge (1996) for the Magnus-Tetens RH coefficients (a=17.625,
b=243.04 — standard,

widely-cited values, consistent with the code's own implicit sourcing; not
independently verified

against a sources/ folder entry since this is a meteorological-constants
citation, not a

project-domain paper). The framework doc's own §5.1–5.2 directly prescribes the
MBE/RMSE/Pearson-r,

season×event stratification, and three-branch decision rule as implemented — the
code matches the

spec closely (see 14_ERA5_POWER_VALIDATION.md for the one-to-one correspondence
check).

## A.11 Validation

This phase is itself a validation step (that is its purpose) — its own output is
validated by the

n≥30-paired-rows gate on quantile-mapping fits (with a printed WARN below that,
not a hard stop) and

by 03_verify_climate_csv.py's independent cross-source correlation check (Check
6), which is

WARN-only and can never fail the whole QA script on cross-source disagreement
alone. This same

Check 6 recurs, essentially unchanged, as part of Phase 2.5's Part 1 sanity
layer — see §B.2.

## A.12 Outputs

climate_rajasthan_points.csv, daily_aggregates_rajasthan{,_summary}.csv,

era5_power_agreement_rajasthan.csv,
outputs/qc_era5_power_scatter_rajasthan.html,

outputs/bias_decision_rajasthan.txt, 8 QC HTML files.

## A.13 Dependencies

Requires Phase 1's complete point/time/NetCDF/JSON set. Corrected 2026-08-11 —
earlier

documentation stated "Everything from Phase 3 onward reads
climate_rajasthan_points.csv

directly," which is now factually wrong. Phase 2.5
(03b_quality_check_rajasthan.py) reads

climate_rajasthan_points.csv and produces climate_rajasthan_points_clean.csv;
Phase 3

(04_climate_signature_rajasthan.py) reads the CLEAN file, not this phase's raw
output directly —

see §B and 15_QUALITY_CONTROL.md Part 2. daily_aggregates_rajasthan_summary.csv
(from 02b,

not touched by the quality-check step) is still read directly by Phase 3. This
file

(climate_rajasthan_points.csv) remains the single most-depended-upon RAW output
in the pipeline,

but it is no longer the most-depended-upon FINAL input to Phase 3 — that is now
the Phase 2.5 clean

file.

────────────────────────────────────────

# PART B — Phase 2.5: Quality Control & Data Cleaning

## B.1 Purpose

Gate Phase 2's output (climate_rajasthan_points.csv) through a two-layer
quality-check pipeline:

first a read-only sanity check that never modifies data (Part 1), then an actual
data-cleaning step

with explicit outlier detection and imputation (Part 2). Phase 3 reads the
cleaned output,

climate_rajasthan_points_clean.csv, not the raw Phase 2 output.

Why this phase exists at all: Phase 2's cross-source validation (§A) caught the
deaccumulation

bug and established which data source to use for GHI. But even valid data can
contain rare outliers

(sensor glitches, data-transmission errors, edge cases in interpolation logic).
A quality-check

phase between raw collection and downstream signature construction ensures Phase
3's climate

indices are built on data that passes both (1) schema/coverage sanity and (2)
statistical

plausibility checks. This is standard practice in climate-data pipelines and
essential before

deriving anything downstream.

## B.2 Part 1: Sanity Checks (03_verify_climate_csv.py)

Six ordered read-only checks against climate_rajasthan_points.csv. Safe to run
at any time,

including mid-download.

### Check 1 — Schema

Verifies presence of all 30 expected columns. Missing → FAIL. Unexpected extras
→ WARN.

### Check 2 — Point coverage

Every point_id from population_grid_points.csv should appear. Missing → WARN
(expected

mid-run). Extra/unrecognized → FAIL.

### Check 3 — Row coverage

| Rule | Threshold | Action |
| --- | --- | --- |
| Duplicate (point_id, date, event) | any duplicate | FAIL |
| Row count per point mismatch vs suntimes.csv | any mismatch | WARN |
| Event value outside {sunrise, noon, sunset} | any | FAIL |
| Date outside [2016-01-01, 2025-12-31] | any | WARN |

### Check 4 — Null rates

Thresholds: ≥30% → FAIL, ≥5% → WARN, else OK. Per-column, applied to all era5_*
and

power_* columns. Round-number thresholds (not independently derived from
statistical power

calculation), but defensible engineering judgment for a QA gate.

### Check 5 — Physical sanity range checks

| Column | Min | Max | Source |
| --- | --- | --- | --- |
| T_amb | −5 | 60 °C | matches pipeline clip |
| T_dew | −30 | 40 °C | QC-only, no upstream clip |
| RHum | 0 | 100 % | physical bound |
| W_spd | 0 | 40 m/s | QC-only |
| GHI/DNI/DHI/GHI_clearsky | 0 | 1400 W/m² | matches pipeline clip |
| LW_down | 0 | 700 W/m² | QC-only |
| cloud_cover | 0 | 1 | physical bound |
| precipitation | 0 | 200 mm | QC-only |
| P_atm | 800 | 1050 hPa | physical bound |
| SZA | 0 | 180 ° | physical bound |
| solar_azimuth | 0 | 360 ° | physical bound |
| CSI | 0 | 2 | dead check — looser than pipeline's [0,1.5] clip, can never fire |

Violation severity: >1% out-of-range → FAIL, else → WARN, fully compliant → OK.

Issue flagged: the CSI check bound should be tightened to [0,1.5] to match the
actual clip, or

documented as an intentional defense-in-depth margin — this is the same
redundancy noted for CSI in

§A.7, restated here as a concrete QC-check-level finding.

### Check 6 — Cross-source agreement

Pairs: (era5_GHI, power_ALLSKY_SFC_SW_DWN), (era5_T_amb, power_T2M). Requires
≥30 paired

non-null rows; fewer → WARN. Computes Pearson r; r < 0.5 → WARN, else → OK. No
FAIL

branch — cross-source disagreement can only WARN here (the more rigorous
decision logic lives in

Phase 2's 03b_agreement_analysis.py, §A.8).

## B.3 Part 1b: Visual QC

03_qc_plots.py generates 8 interactive HTML visualizations (spatial folium maps
+ distributional

plotly charts) showing spatial coverage, elevation distribution, data-coverage
heatmaps,

distributional histograms per variable and season, and summary statistics. The
rejection-window

diagnostic is permanently skipped (with an in-code message) because matched
timestamps are never

persisted — see §A.5 for the upstream root cause.

## B.4 Part 2: Actual Data Cleaning (03b_quality_check_rajasthan.py)

Critical design choice: Only T_amb, RHum, W_spd are outlier-filtered. GHI and
CSI are

deliberately excluded because they are weather-driven (clouds, clear skies are
real, not errors).

A Hampel filter initially over-corrected genuine cloud-driven GHI/CSI
variability; this was

identified 2026-08-11 and the solution was to exclude those two variables from
outlier detection

entirely.

Hampel filter: identifies outliers as points where |value − median| / (1.4826 *
MAD) exceeds

a threshold (default 3.5 for outlier, 2.5 for winsorizing candidate). Applied
per variable, per

season, per point. Over-aggressive filtering detected on GHI/CSI → excluded;
remaining application

is correct and defensible.

Missing-data imputation: MICE-style chained-equation imputation with
random-forest donors on

(season, point_id) subgroups. Produces climate_rajasthan_points_clean.csv with
outliers

winsorized and missing values imputed.

## B.5 Part 2b: Validation of the Cleaning

03b_validate_quality_fix_rajasthan.py re-runs Phase 2.5's own sanity checks
(§B.2) against the

cleaned output (climate_rajasthan_points_clean.csv), independently verifying
that cleaning did

not introduce schema violations or new failures. Confirms the cleaning was safe
to apply.

## B.6 Part 2c: Visual QC (Before/After)

03c_plots_raw_rajasthan.py and 03b_quality_check_plots_rajasthan.py generate
pre-cleaning and

post-cleaning distributional plots (histograms, box plots, spatial maps),
showing what the Hampel

filter changed and justifying the exclusion of GHI/CSI.

## B.7 The weather-vs-error insight

The deliberate exclusion of GHI/CSI from outlier detection reflects a key
insight: weather is

real and should not be smoothed away. Outliers in solar radiation are clouds;
clouds are not

errors. Temperature outliers, by contrast, are likely sensor/transmission errors
and should be

caught.

## B.8 Inputs

climate_rajasthan_points.csv (from Phase 2, §A), population_grid_points.csv,

daily_aggregates_rajasthan_summary.csv (for seasonal aggregation logic).

## B.9 Outputs

climate_rajasthan_points_clean.csv (for Phase 3),
quality_report_rajasthan.{md,json}

(human-readable + structured report), outputs/qc_raw_*.html (8 pre-cleaning
plots),

outputs/qc_clean_*.html (8 post-cleaning plots), validation confirmation stdout.

## B.10 Dependencies

Requires Phase 2's complete output. Phase 3 (Climate Signature) reads this
phase's CLEAN output,

not Phase 2's raw output directly.

────────────────────────────────────────

# PART C — Combined Problems / Risks (both phases)

- The deaccumulation bug (fixed). Headline finding of the entire audit — see
09_ERA5_DATA_PIPELINE.md and 20_IMPLEMENTATION_ISSUES.md item 1. (Phase 2)

- Quantile-mapped GHI is never persisted. 03b_agreement_analysis.py's correction
  is reported
(before/after diagnostic table) but not written back into
climate_rajasthan_points.csv,

climate_rajasthan_points_clean.csv, or any other dataset that Phase 3 reads.
This means Phase 3

onward currently consumes the uncorrected (though already deaccumulation-fixed)
ERA5 GHI

values, not the bias-corrected ones — the quantile-mapping result exists only as
a

methodology-section number, not as an applied correction. Open decision: either
apply the

correction upstream (in Phase 2, before Phase 2.5, or as an explicit step inside
Phase 2.5's

cleaning) or explicitly document that Phase 3+ intentionally uses raw (not
bias-corrected) ERA5

GHI and why that is still defensible (e.g., the correction is small relative to
the signal at the

daily/seasonal aggregation level Phase 3 actually uses). (Phase 2, restated as
still-open in

Phase 2.5's scope since Phase 2.5 is the last place the correction could still
be applied before

Phase 3 consumes the data.)

- The "documented 2016-01-01 edge case" is referenced in three places (02's
  conceptual
framing, 03_verify's docstring, 03b's docstring) but no code in
02_combine_rajasthan.py

actually special-cases it — the mechanism is implicit (pandas diff()-free
accum_to_flux()

has no predecessor-hour dependency at all anymore, so the originally-cited edge
case may be a

stale reference from before the deaccumulation fix, when deaccumulate()
genuinely did need a

predecessor hour). Worth reconciling this comment against current code before
citing it in a

methodology write-up. (Phase 2)

- Monsoon-month definition mismatch between 02_combine_rajasthan.py (Jun–Aug)
  and
02b_build_daily_aggregates.py (Jun–Sep) — see 20_IMPLEMENTATION_ISSUES.md item
7. (Phase 2)

- No matched-timestamp columns are ever written (era5_matched_time_utc /
power_matched_time_utc), which structurally disables 03_qc_plots.py's
rejection-window

diagnostic (both the Phase 2 and Phase 2.5 instances of this script) and forces
03b's

MANUAL_REVIEW-branch diagnostics to use an SZA-based proxy instead of a direct
time-offset

measurement — low-cost to fix (two extra output columns) if the rejection-window
QC is ever

needed. (Phase 2, propagates into Phase 2.5)

- CSI plausibility check is structurally redundant in both its Phase 2 form
(03_verify_climate_csv.py's [0,2] bound vs. the pipeline's actual [0,1.5] clip)
and its

restatement in §B.2 Check 5 — same finding, same fix (tighten to [0,1.5] or
document as

intentional margin). (Phase 2 / Phase 2.5)

- Initial Hampel over-correction on GHI/CSI (FIXED, 2026-08-11). The Hampel
  filter initially
applied to GHI/CSI, removing genuine cloud-driven variability as if it were
noise. Diagnosis:

weather is not an outlier. Solution: exclude GHI/CSI from outlier detection
entirely. Confirmed

by visual inspection of pre/post plots. (Phase 2.5)

- MICE missing-data imputation is not perfect. It reconstructs values based on
  learned patterns
in the available data. If an entire season is missing for a point, imputation
cannot know what

the "right" value should be. Check imputation fractions per variable; if any
variable has >5%

imputed rows, investigate manually. (Phase 2.5)

- No outlier detection on GHI means real sensor failures in GHI might pass
  through. By design —
this is a deliberate choice to preserve weather variability. If a specific
point's GHI data is

suspected to be systematically wrong (not just cloudy), investigate via the
visualization outputs

or manual inspection rather than hoping the QC step catches it. (Phase 2.5)

────────────────────────────────────────

# PART D — Combined Status

Phase 2 — COMPLETE, with the deaccumulation fix as a documented, verified
correction, and one

open methodological decision (whether/how to apply the quantile-mapping
correction upstream) that

should be resolved and stated explicitly before this phase is cited as final in
a methodology

write-up.

Phase 2.5 — COMPLETE, corrections applied and validated, outputs on disk.
Documentation for

this phase was only added 2026-08-11, correcting a prior factual error in the
pipeline docs about

what Phase 3 actually reads.

Combined open item carried into Phase 3 write-up: the quantile-mapping
persistence decision

(above) is the one unresolved methodological question spanning both phases — it
must be settled

(applied or explicitly justified as skipped) before Phase 3's climate-signature
construction is

described as final.

# 6. 05_PHASE_3_AUDIT.md

Source path: /mnt/data/05_PHASE_3_AUDIT.md

# 05 — Phase 3 Audit: Climate Signature Construction

Scripts: signature_lib.py, 04_climate_signature_rajasthan.py.

## Purpose

Reduce each point's 10-year climate record to one compact "climate signature"
vector suitable for

clustering and for deriving PCM performance targets. The framework doc's own
design principle

(§6.1, quoted): *"Every index must answer the question 'which PCM property does
this constrain, and

by what physical mechanism?'. If that sentence cannot be completed, the index is
removed."*

## Inputs

Corrected 2026-08-11: climate_rajasthan_points_CLEAN.csv (Phase 2.5's output —

03b_quality_check_rajasthan.py's Hampel-filtered/imputed clean file, NOT
02_combine_rajasthan.py's

raw output directly, since 2026-08-11 — see 15_QUALITY_CONTROL.md Part 2 and
04_PHASE_2_AUDIT.md's

corrected Dependencies section), daily_aggregates_rajasthan{,_summary}.csv,
suntimes.csv,

population_grid_points.csv (Phases 1–2/2.5).

## Processing — Tier 1 (sun-event statistics, signature_lib.build_tier1_signature())

Shared between this script's Level-A (whole-year) call and
05_cluster_rajasthan.py's Level-B

(per-season) call — one implementation, two group_keys. Produces, per group:

T_sunrise_mean/p05, T_noon_mean, T_sunset_mean/p95, diurnal_gradient
(noon−sunrise, an

acknowledged underestimate of true DTR since peak air temp lags solar noon by
2–3h — this is why

Tier 2's DTR_true exists as a companion), kt_noon_mean/std, GHI_noon_mean,
GHI_sunset_mean,

RH_sunrise_mean, wind_noon_mean/sunset_mean, HSI_sunrise, Ta_mean/p95/p05
(daily-collapsed

first, then aggregated), daylength_mean, daylength_amplitude (half the seasonal
swing, standard

oscillation-amplitude convention).

HSI_sunrise is literally Thom's (1959) Temperature-Humidity Index (THI), not a
bespoke

"humidity stress index":

HSI_sunrise = T_sunrise_mean − 0.55·(1 − RH_sunrise_mean/100)·(T_sunrise_mean −
14.5)

Cited in-code to Thom, E.C., "The Discomfort Index," Weatherwise 12(2), 1959 —
the name in the

project's own variable naming ("humidity stress index") is a relabeling of an
established index, not

an original derivation; this should be cited as Thom's THI in any write-up, not
presented as novel.

## Processing — Tier 2 (daily-integral join)

Left-joined from daily_aggregates_rajasthan_summary.csv: `GHI_daily_kWh, SAI,
kt_daily_mean/std,

cloudy_frac, CCI, HDD18, CDD24, DTR_true, seasonality, monsoon_index`.

## Processing — derived PCM-facing quantities

Tm_target_C = T_delivery + ΔT_approach = 50 + 7 = 57.0°C, constant across all
320 points by

design (indirect-system assumption; T_delivery is the Indian-domestic SWH
delivery target per

framework doc §6.3, ΔT_approach is the midpoint of the doc's stated 5–8 K
heat-exchanger approach

range).

Tm_target_capped_C — the per-point regime-adjusted upper bound, capturing "Tm
must lie below

the collector delivery temperature achievable on a poor-insolation period":

kt_worst_month = min over 12 calendar months of (mean kt_daily for that month,
pooled 2016-2025)
kt_ratio = clip(kt_worst_month / kt_daily_mean, upper=1.0)
Tm_target_capped_C = min(57.0, Ta_mean + kt_ratio·(57.0 − Ta_mean))

This formula was revised on 2026-08-11, replacing an original kt_p05
(5th-percentile single

day) basis with kt_worst_month (lowest of 12 calendar-month means), after the
single-day basis

was checked against independent field evidence (Nahar 2003, tested at Jodhpur —
inside this state's

arid-west cluster — reporting 100 L delivered at average 50–70°C across the
year) and found to

produce implausibly low caps (40.8–49.5°C, at/below the low end of what a real
Jodhpur system

delivered even in its weakest season). The kt_p05-based value is retained as

Tm_target_capped_C_p05day for audit/comparison only — no downstream script reads
it. The

worst-month basis is anchored to Durin et al. (2018), "'Worst Month' and
'Critical Period' Methods

for the Sizing of Solar Irrigation Systems" — a genuine, appropriately-applied
sizing-methodology

citation for this kind of cap.

L_required_kJ_per_kg — the latent-heat floor:

T_mains_est_C = Ta_mean − 2.0 [documented as NOT a published correlation — see
below]
Q_night_kJ = 300.0 · 4.186 · (50.0 − T_mains_est_C) [Avargani et al. 2021: 300 L
@ 60±2°C, 7h]
L_required_kJ_per_kg = Q_night_kJ / 50.0 [ASSUMED_PCM_MASS_KG = 50 kg
placeholder]

This formula was itself corrected in-place: an earlier version fed a 60 L/min
sustained rate for

7 hours (25,200 L total) into the same formula — traced to a units confusion
where Avargani et

al.'s cited figure ("300 L of hot water at 60±2°C for 7 h of operation") is a
total volume over

the discharge window, not a per-minute rate; the corrected code uses the literal
300 L total. The

script's own docstring is explicit that L_required is a ceiling, not an
achievability bar — it

assumes the PCM bed alone supplies the entire load with zero contribution from
tank sensible heat or

overlapping collector charge, an assumption that does not hold even in
Avargani's own experimental

rig — and flags forward, correctly, that Phase 5's fixed κ=0.7 latent-heat
constraint will zero out

every candidate given this ceiling (confirmed true — see 07_PHASE_5_AUDIT.md).

✅ VALIDATED (2026-08-31 re-run complete): The all-latent assumption was
corrected to use

SHARE_PCM=0.5 (literature-anchored). Avargani et al.'s 300 L benchmark is
delivered by a combined

PCM-tank architecture; literature on combined sensible-latent SWH reports PCM
contributing 40–78%

of total delivery (Zhao 2022, Huang 2020, Abdelsalam 2020, Kowhitney 2021). The
corrected formula:

L_required = (SHARE_PCM * Q_night) / ASSUMED_PCM_MASS_KG [SHARE_PCM = 0.5,
literature-anchored]

Validation results (2026-08-31 re-run):

- L_required halved: 608–641 kJ/kg (old all-latent) → 285–344 kJ/kg (new,
  literature-anchored)
- Output message: "L_required_kJ_per_kg : 285 - 344 kJ/kg (literature-anchored,
  PCM 50% of total night delivery, with tank sensible heat + concurrent charging
  supplying the rest)"
- Clustering stability: bootstrap-ARI improved from 0.8137 to 0.8272 (robust to
  methodology change)
- Downstream impact: Phase 5 κ-calibrated survivors increased from 20 to 39
  candidates (9/14/16 per cluster)
## The five interaction terms (exact, with in-code physical justification)

int_GHI_x_ktstd = GHI_daily_kWh × kt_daily_std (erratic-but-large resource)
int_DTR_x_cloudyfrac = DTR_true × cloudy_frac (cycling stress under
intermittency)
int_RH_x_TsunriseMinusTm = RH_sunrise_mean × (T_sunrise_mean − Tm_target_C)
(condensation risk)
int_wind_x_TsunsetMinusTdelivery = wind_sunset_mean × (T_sunset_mean −
T_delivery) (evening convective loss)
int_CCI_x_1minusSAI = CCI × (1 − SAI) (combined autonomy requirement)

## PCA block

PCA_BLOCK = [Ta_mean, Ta_p95, Ta_p05, T_sunrise_mean, T_noon_mean, HDD18, CDD24,
elevation_m] (8

columns — note the section's own internal label calling this "the correlated
temperature/pressure

block" is inaccurate; there is no pressure variable in the actual list, likely a
leftover label from

a template). StandardScaler → PCA(n_components=0.95, random_state=42) — retains
4 components

for the Rajasthan run (not a fixed integer; data-determined). Loadings and
explained-variance ratio

are printed for interpretation, not silently discarded.

Elevation is a resolved design ambiguity, not excluded: the brief's "STATIC
ATTRIBUTES...NOT

included in the clustering feature matrix" instruction and its "elevation_m"
PCA-block membership

initially read as contradictory; the code's own resolution (documented as
"Correction 2") is that

elevation's raw value is reporting-only, but it does feed the PCA block, and the
resulting PC*_z

scores (which subsume it) are what actually enters the clustering matrix.

## Standardization

NON_CLUSTERING_COLS explicitly excludes `lat, lon, population, weight,
T_mains_est_C, kt_p05,

kt_worst_month, Tm_target_capped_C_p05day, tm_target_capped_flag`, plus the raw
PCA-block columns

(replaced by PC1..PC4). Everything else — including Tm_target_C,
Tm_target_capped_C,

L_required_kJ_per_kg, and all 5 interaction terms — is z-scored. Verified
directly against the

actual output CSV header: no lat_z/lon_z exist, confirming the exclusion claim
in data, not just

in code.

## Literature support

Thom (1959) THI — direct, correctly attributable citation for HSI_sunrise.
Avargani et al. (2021),

J. Energy Storage — direct citation for the 300 L/60±2°C/7h night-discharge
basis. Durin et al.

(2018) — direct citation for the worst-month sizing method. Nahar (2003) — cited
as field-evidence

justification for the kt_worst_month correction, present in
.claude/references.md as a bare

citation note (not a full BibTeX entry — worth completing before a formal
write-up).

T_mains_est_C = Ta_mean − 2.0 is explicitly documented in-code as not derived
from any published

correlation — kept only for cross-state consistency with an
identically-unsourced Tamil Nadu

precedent. This is a genuine literature gap, not a citation the audit failed to
find: a real

ground-temperature lag correlation (e.g., Kusuda & Achenbach-style annual-lag
models) should be

substituted before this number is presented as anything more than a placeholder.

## Climate Signature Feature-to-PCM-Property Mapping

Design principle: Every feature in the signature must answer "which PCM property
does this

constrain, and by what physical mechanism?" If that sentence cannot be
completed, the feature is

removed.

| Feature Group | Represents | PCM Constraint | Target Property |
| --- | --- | --- | --- |
| T_sunrise_mean, RH_sunrise_mean + HSI_sunrise | Pre-dawn condensation risk at storage surface | Corrosion resistance req. | Feeds Phase 5 corrosion veto |
| T_noon_mean, GHI_noon_mean, kt_noon/std | Charging-window heat availability & reliability | Melting-window achievability, charging feasibility | Tm_target_capped_C (Phase 5 constraint 6) |
| T_sunset_mean, wind_sunset_mean | Evening heat-loss potential during discharge onset | Discharge-window thermal-loss sensitivity | int_wind_x_TsunsetMinusTdelivery interaction term |
| diurnal_gradient, DTR_true | Daily thermal swing magnitude (Tier 1 underestimates true swing, Tier 2 captures real) | Cycling stress on PCM | int_DTR_x_cloudyfrac + Phase 5 constraint 4 (cycles≥300) |
| GHI_daily_kWh, kt_daily_mean/std, SAI, CCI | Total charging energy & day-to-day reliability | Latent-heat sizing & autonomy req. | L_required_kJ_per_kg + int_CCI_x_1minusSAI interaction |
| HDD18, CDD24 | Seasonal thermal-load context (degree-days, base 18°C/24°C) | Indirect: feeds PCA temperature block, informs regime characterization | Phase 4 clustering |
| cloudy_frac, seasonality, monsoon_index | Charging intermittency & seasonal variability | Cycling stress under intermittent charging | int_DTR_x_cloudyfrac + int_GHI_x_ktstd interactions |
| elevation_m (PCA block only, not standalone) | Atmospheric/airmass context | Already baked into pvlib solar-geometry upstream | Indirectly informs regime separation via PC*_z scores |
| daylength_mean, daylength_amplitude | Seasonal charging-window-length variation | Charging duration context | Level-B ablation candidate — flagged as possibly climatically-tautological (daylength is deterministic-by-construction from latitude/day-of-year, not a weather outcome) |

Why Tier 2 (daily integrals) exists alongside Tier 1 (sun-event instantaneous):

Tier 1 samples at 3 points/day but systematically underestimates true diurnal
range (diurnal_gradient

vs. DTR_true). Tier 2 (full-hourly NASA POWER integration) captures accurate
daily energy totals

and variability but cannot be computed from ERA5 alone within this pipeline's
scope (would require

24h/day ERA5 request, explicitly out of scope). Keeping both, rather than
picking one, is the

correct choice given each one's distinct blind spot.

Two-tier design as consistent with Objective 1's novelty claim (N2):

A one-tier signature would miss either the charging-window detail (Tier 1 only)
or the true daily

energy perspective (Tier 2 only). The two-tier approach is the framework doc's
own specified design

(§6.2) and is now explicitly justified in-code and documented, supporting the
claim that this

climate signature is an advance over single-temperature proxies.

## Validation

Correlation heatmap + |r|>0.9 flagging on the final (pre-standardization,
post-PCA-block-removal)

feature set — printed only, not persisted, not auto-acted-upon. No flagged pairs
are reported as

"already handled by PCA" without independent verification; the check explicitly
distinguishes new

collinearity from PCA-absorbed collinearity.

Note on PCA scope: Only the temperature/elevation block undergoes PCA
(deliberately excluding

solar-variability, humidity, and cycling-relevant indices, which carry the
actual discriminating

signal for regime separation). This is a correctly-scoped dimensionality
reduction: it removes

within-block redundancy specifically without compressing away the features that
actually separate

the clusters.

## Outputs

climate_signature_rajasthan.csv — 320 rows × 86 columns. Plus, added 2026-08-11:
`outputs/

signature_distributions_rajasthan.html` (histogram of every clustering-input
column across all 320

points — a bimodal column here previews a possible Level-A cluster split on that
feature alone) and

outputs/signature_point_map_rajasthan.html (geographic view of GHI_daily_kWh and
monsoon_index)

— both pure visualization of data this script already computes, in addition to
the pre-existing

outputs/signature_correlation_heatmap_rajasthan.html.

## Dependencies

Requires Phase 2's climate_rajasthan_points.csv and
daily_aggregates_rajasthan_summary.csv.

Feeds Phase 4 (clustering) and Phase 5 (feasibility targets Tm_target_C,
Tm_target_capped_C,

L_required_kJ_per_kg).

## Problems / risks

- Dangling citation: the module docstring references
Objective1_Section5_Methodology_Update.docx for the draw-rate correction
provenance, and

explicitly self-flags that this file was not found in the project tree — a real,
honestly-logged

gap, not a fabricated citation. Resolve before final write-up (either locate the
file or update the

in-code pointer).

- RESOLVED — "forward-dated docstring" concern. A previous version of this audit
  flagged
"Correction 5" (the kt_worst_month fix)'s 2026-08-11 date as a likely
clock/environment artifact.

It is not: 2026-08-11 is a real date with many independently-verified,
mutually-consistent same-day

fixes across this codebase (the GMM canonical-relabeling fix, provenance_lib.py,
physics_lib.py's

two solver bugs — see 20_IMPLEMENTATION_ISSUES.md items 8-10). This fix is
settled history.

- T_mains_est_C is unsourced — flagged above; this feeds directly into
  L_required_kJ_per_kg,
which is the constraint that currently zeros out the entire feasibility filter
(Phase 5), so this

is not a low-priority gap.

- monsoon_index proxy status is a structural, not incidental, limitation —
  confirmed
unconditionally true (PRECTOTCORR never downloaded), correctly self-flagged in a
printed warning,

but should be stated as a limitation in any methodology write-up that reports
monsoon_index.

## Status

COMPLETE — with two open citation gaps (dangling methodology-update reference,
unsourced

T_mains lag correlation) that should be resolved before this phase's derived
quantities

(Tm_target_capped_C, L_required_kJ_per_kg) are presented as fully
literature-grounded.

# 7. 06_PHASE_4_AUDIT.md

Source path: /mnt/data/06_PHASE_4_AUDIT.md

# 06 — Phase 4 Audit: Climate Regime Clustering

Script: 05_cluster_rajasthan.py.

## Purpose

Discover climate regimes empirically (Gaussian Mixture Model) rather than assume
hand-drawn zones —

this is novelty claim N1 from the framework doc. Two levels: Level A (spatial —
one signature vector

per point, whole 10-year record) and Level B (temporal — one vector per point
per season, detects

whether a point's PCM-relevant regime shifts materially between seasons).

## Inputs

climate_signature_rajasthan.csv (Level A, direct read of the *_z columns) and

climate_rajasthan_points.csv + suntimes.csv (Level B, which rebuilds Tier-1
signatures

per-season directly via
signature_lib.build_tier1_signature(group_keys=["point_id","season"])

rather than reading any saved Level-A file — Level B therefore has no Tier 2,
PCA, or interaction

terms, only 19 raw Tier-1 columns, freshly standardized with its own independent
StandardScaler).

## Processing

### Level A

- GaussianMixture(covariance_type="diag", random_state=42, n_init=5) fit for
  k=2..12.
- Per k: BIC, AIC, silhouette (guarded for n_unique>1), Davies-Bouldin,
  Calinski-Harabasz, and
bootstrap-ARI stability (50 resamples: fit once on full data → base_labels; 50×
fit a fresh

GMM on a with-replacement resample of the same size → predict on the original
data → Adjusted

Rand Index against base_labels; report the mean).

- K-Means (n_init=10) fit in parallel purely as a reported comparison baseline,
  never the
primary model — silhouette curves for both appear side-by-side in
bic_selection_rajasthan.csv.

- No population-weighting of the GMM fit — confirmed by direct code inspection
  (no
sample_weight argument anywhere) — by design, since the point sampling is
already

population-weighted by construction (Phase 1); weighting the fit again would
double-count

population. Population enters only later, in cluster-profile weighted means.

- k-selection (suggest_k()): a documented 3-tier cascade — (1) k in the expected
  single-state
range [2,4] AND silhouette in the realistic band [0.15, 0.35], pick highest
bootstrap-ARI among

those; (2) any k in the silhouette band, highest bootstrap-ARI; (3) fallback to
lowest-BIC k, with

a printed warning. Not a forced single "K_FINAL" — the framework doc explicitly
asks for

k=2–4 for a single-state run (vs k=6–10 expected once all four states combine),
and the code

enforces exactly that expectation rather than letting BIC alone pick (BIC here
monotonically

decreases across the entire scanned range with no interior minimum — it would
otherwise "select"

k=12, the edge of the scan, which is not a meaningful answer).

### Level B

Same GMM/K-Means machinery, k=2..8, on the freshly-built per-point-per-season
Tier-1 matrix.

Additional checks specific to Level B: a regime-shift analysis (fraction of
points whose

cluster assignment differs across the 4 seasons) and a season-tautology check
(contingency

table + Adjusted Rand Index/Normalized Mutual Information between cluster labels
and season labels,

plus an ANOVA F-statistic feature-ranking to check whether temperature/GHI
features dominate the

clustering — which would suggest the clustering is just rediscovering the season
labels rather than

finding independent structure). An LEVEL_B_EXCLUDE_FEATURES ablation switch
exists (default empty,

inactive) to drop deterministic-by-construction features like
daylength_mean/daylength_amplitude

from the fit if needed.

### External validation

Köppen-Geiger is now wired in for real (updated 2026-08-11) — Beck et al.
(2018),

doi:10.1038/sdata.2018.214, 1-km raster, genuine per-point classification lookup
(not a stub).

Rajasthan's 320 points classify as BSh=203, BWh=85, Aw=20, Cwa=12. Result:
ARI(GMM cluster, Köppen

class)=0.19, NMI=0.32 — low-to-moderate agreement, read as "the GMM finds
climate structure at a

finer resolution than Köppen's broad classes capture within Rajasthan" (a
plausible, legitimate

finding in its own right, arguably the point of empirical clustering instead of
applying Köppen

directly) rather than evidence the clustering failed to find anything real.
NBC/ECBC climate-zone

validation remains stubbed (nbc_ari = nbc_nmi = None) — no local India-specific
zone lookup exists

in this project tree, not fabricated. State-identity external validation is
explicitly noted as "not

meaningful yet" for a single-state run.

## A documented, fixed methodology bug: GMM covariance type

Root-caused and fixed on 2026-08-10: full covariance was changed to diag. Cause:
at Level

A's dimensionality (35 standardized columns) and k=3 on 320 points (~106
points/cluster), full

covariance requires d·(d+1)/2 = 630 parameters per cluster from ~106 samples —
badly

underdetermined. Symptom: max_membership_prob was saturating to ~1.0 for
essentially 100% of

points (zero genuinely ambiguous/soft cases) despite only a moderate silhouette
(~0.31) — a

mismatch between a distance-based measure (silhouette, unaffected) and a
probability-based measure

(GMM posterior, badly affected) that revealed the covariance estimate was
numerically extreme rather

than reflecting real geometric separation. Fix verified empirically: diag
restores a realistic

membership spread (min ~0.58, ~1.6% of points genuinely <0.90) while silhouette
barely moves (0.3028

vs 0.3090) — confirming the fix changes how confidently the model reports its
answer, not what

the answer is. Two alternative fixes (bumping reg_covar, PCA-reducing the
feature set first) were

also verified to work but rejected as either a less-principled band-aid or a
loss of the

per-named-index interpretability the framework doc requires for Level A.

## A second documented, fixed bug: GMM cluster-index instability across re-runs (2026-08-11)

Distinct from the covariance-type fix above — found while building Phase 7, not
during this

phase's own original construction. sklearn's GaussianMixture gives no guarantee
that cluster index

0 refers to the same physical climate group across separate re-runs of this
script, even with the

same random_state=42, if anything about the fit changes between runs (the
full→diag covariance

fix itself is one such change). Symptom: Phase 5's and Phase 6's outputs (both
downstream of this

script) disagreed cluster-by-cluster on which PCMs belonged to which cluster_id
— Phase 5's

"cluster 0" candidate set matched Phase 6's "cluster 2" set verbatim, and vice
versa, because the two

phases had been run against different invocations of this script.

Fix: immediately after the final Level-A GMM fit, hard labels are canonically
relabeled 0..k-1 by

sorting each raw cluster's MEAN LATITUDE ascending (south to north) — a simple,
always-available,

fit-independent ordering key computed directly from the points themselves, not
from anything the GMM

produces. "Cluster 0" now means the same physical (southernmost) climate regime
regardless of which

run produced the underlying fit, as long as the underlying point PARTITION is
equivalent. This does

not protect against Phase 5/6/7/8 being run against a genuinely DIFFERENT
partition from a

different re-run (different data or parameters) — that risk is separately
covered by a hard-fail

provenance-fingerprint check (provenance_lib.py) now run at every Phase 5→6→7→8
handoff, which

raises SystemExit (not a warning) if a downstream phase's input doesn't match
the current on-disk

cluster_profiles_rajasthan.csv. See 19_PHASE_7_ONWARD.md for the full incident
writeup and

21_REPRODUCIBILITY.md for the provenance mechanism.

## ✅ VALIDATED (2026-08-31 re-run complete)

L_required Methodology Correction (OPTION A) validated. Phase 3's methodology
was corrected to use SHARE_PCM=0.5 (literature-anchored fractional-share),
halving all L_required values. Phase 4 clustering remained stable under this
change, confirming the fix is robust.

## Actual Rajasthan result (k=3, 2026-08-31 re-run) — VALIDATED

| Cluster | Points | Population | Medoid | Description | Tm_target_C | L_required (kJ/kg) NEW | bootstrap-ARI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 114 | 22,568,150 | RJP_0132 (24.375, 74.125) | Cooler, arid/low-monsoon, erratic solar, short low-clearness runs | 57.0 | 313 | 0.8272 ↑ |
| 1 | 103 | 17,959,813 | RJP_0202 (26.875, 73.625) | Hot, monsoon-influenced, steady solar, long low-clearness runs (high autonomy demand) | 57.0 | 304 | (from 0.8137) |
| 2 | 103 | 29,775,240 | RJP_0055 (26.625, 76.375) | Cooler, arid/low-monsoon, erratic solar, short low-clearness runs | 57.0 | 320 |  |

k=3 was selected because k∈{2,3,4} all satisfy the silhouette-band +
expected-range gate, and among

those k=3 has the highest bootstrap-ARI (0.8137, vs 0.6965 at k=2 and 0.5904 at
k=4) — the tier-1

selection rule, working as designed, not a default or a hand-pick.

Notable limitation, self-flagged by the code and confirmed empirically: Clusters
0 and 2 receive

the identical auto-generated qualitative description string despite being
numerically distinct

(e.g., HDD18 1100 vs 2237, monsoon_index 0.93 vs 1.03) — the 4-axis
threshold-based description

generator is too coarse to distinguish them. The code's own docstring already
calls this "a

first-pass label to hand-edit, not a final publication-ready caption" — treat it
exactly that way in

any write-up; do not quote the auto-generated Cluster 0/2 descriptions as if
they were independently

differentiated.

## Literature support

Silhouette expectation band [0.15, 0.35] is cited (not invented) from a Building
and Environment

(2024) India climate-classification study reporting silhouette 0.21 vs −0.2 for
the existing NBC

classification (peaking ~0.3 at k=6 in a 4-state design), and a 2026
thermal-comfort clustering

study independently reporting mean silhouette 0.235 — both citations appear in
the code comments

with enough specificity to be traceable, though full BibTeX entries for both
were not located in

references.bib/references.md during this audit and should be added before formal
citation.

Beck et al. (2018) is the correctly-named, DOI-verified citation for the
(not-yet-wired) Köppen

validation.

## Validation

Bootstrap-ARI stability (internal),
silhouette/BIC/Davies-Bouldin/Calinski-Harabasz (internal),

season-tautology ANOVA check (Level B internal). External classification
validation: Köppen-Geiger

now real (ARI=0.19, NMI=0.32 — see above); NBC/ECBC still stubbed.

## Outputs

cluster_assignments_rajasthan_levelA.csv,
cluster_assignments_rajasthan_levelB.csv,

bic_selection_rajasthan.csv (Level A only — Level B's k-scan is console-printed,
never persisted),

cluster_profiles_rajasthan.csv, cluster_profile_cards_rajasthan.md,

outputs/qc_cluster_map_rajasthan.html, koppen_validation_rajasthan.csv
(cluster_id x Köppen-class

contingency counts), level_b_feature_importance_rajasthan.csv,

level_b_season_tautology_rajasthan.csv,
level_b_season_contingency_rajasthan.csv. Plus, added

2026-08-11: outputs/qc_k_selection_curve_rajasthan.html (BIC + silhouette vs. k,
chosen k marked),

outputs/qc_cluster_profile_bars_rajasthan.html (headline signature indices by
cluster),

outputs/qc_cluster_population_share_rajasthan.html (population-share pie chart)
— pure

visualization of data this script already computes.

## Dependencies

Requires Phase 3's climate_signature_rajasthan.csv. Feeds Phase 5 directly —
every feasibility

constraint is evaluated per cluster using Tm_target_C, Tm_target_capped_C,

L_required_kJ_per_kg, and HSI_sunrise from cluster_profiles_rajasthan.csv.

## Problems / risks

- Level B's k-scan metric table is not persisted to disk — reproducibility gap
  (see
21_REPRODUCIBILITY.md).

- bootstrap_ari_stability() silently drops any bootstrap resample whose GMM fit
  raises an
exception (except Exception: continue), which could quietly reduce the effective
resample count

below 50 without this being visible anywhere in the output table.

- weighted_mean() (used throughout cluster-profile generation) silently falls
  back to an
unweighted mean if population weights are None or sum to zero — no warning
printed; low

practical risk given Rajasthan's actual weight distribution, but worth knowing
if a future state's

data is sparser.

- External validation is now partially wired in (Köppen) but NBC/ECBC remains
  stubbed — the
clustering's "these are real climate regimes, not clustering artifacts" claim
now rests on internal

statistical measures PLUS one external classification (low-to-moderate
agreement, itself a

legitimate finding), not internal statistics alone.

- GMM cluster-index labels are not stable across separate re-runs of this script
  (see the second
documented bug above) — anyone re-running this script and comparing against a
previously-saved

Phase 5/6/7/8 output MUST re-run the full downstream chain, not assume
cluster_id=0 still means

the same climate regime. The canonical-relabeling fix mitigates but does not
eliminate this risk

for a genuinely different partition; the provenance hard-fail check is the
actual safety net.

## Status

COMPLETE — with TWO caught-and-fixed bugs (GMM covariance type; GMM
cluster-index instability

across re-runs) and Köppen-Geiger external validation now wired in (NBC/ECBC
still stubbed). The

internal clustering result (k=3) is statistically well-supported and now
partially externally

corroborated; the cluster-index-instability fix and its accompanying provenance
hard-fail check are

what make the downstream Phase 5→6→7→8 chain trustworthy across separate re-runs
— see

19_PHASE_7_ONWARD.md for the real incident this was caught from.

# 8. 07_PHASE_5_AUDIT.md

Source path: /mnt/data/07_PHASE_5_AUDIT.md

# 07 — Phase 5 Audit: Feasibility Filtering (+ PCM Property Database)

Scripts: PCM_data/01_preprocess.py (shared database imputation),
07_feasibility_filter_rajasthan.py.

## A note on file provenance in this folder

until phase 4/ contains ~15 files whose filenames do not match their content —
independently

confirmed via byte-level magic-number checks (three files named *.csv are
actually PNG images;

files named like Python scripts are markdown, etc.), consistent with the
project's own README in

that folder documenting the same problem and attributing it to browser download
auto-suffixing. This

audit used the correctly-named canonical copies in
PCM-Selection-ML-model/PCM_data/, which is

also what the live Rajasthan pipeline actually imports from (traced directly via

07_feasibility_filter_rajasthan.py line 146:

PCM_MANUFACTURER_CSV = BASE_DIR.parent / "PCM_data" / "data" /
"PCM_Properties_cleaned_mice_pmm_detailed.csv").

The until phase 4/06_build_pcm_database.py-labeled script is a
Tamil-Nadu-scoped, vestigial

component — Rajasthan's feasibility filter does not call it at all; it
re-implements its own

manufacturer-row loading inline. See 21_REPRODUCIBILITY.md for the
file-mislabeling hazard itself.

## Purpose

(1) Maintain a real, cited PCM property database in the corrected 42–70°C band.
(2) Filter that

database against every cluster's physical/safety/economic requirements before
any ranking happens,

so the MCDM stage never has to implicitly discover an infeasible candidate
through its scores.

## ✅ VALIDATED (2026-08-31 re-run complete)

L_required Methodology Correction (OPTION A) validated with strong results.
Phase 3's L_required derivation was corrected 2026-08-31 to use SHARE_PCM=0.5
(literature-anchored fractional-share) instead of all-latent assumption. Phase 5
re-run shows the fix resolved the prior "0 survivors at κ=0.7" problem.

Validation results (2026-08-31 re-run):

| Metric | Old (2026-08-14, all-latent) | New (2026-08-31, SHARE_PCM=0.5) | Status |
| --- | --- | --- | --- |
| L_required range | 608–641 kJ/kg | 304–320 kJ/kg | Halved ✓ |
| Primary run (κ=0.7 fixed) survivors | 0 / 0 / 0 | 4 / 7 / 5 | Major win |
| Calibrated κ per cluster | 0.2 / 0.3 / 0.2 | 0.5 / 0.6 / 0.5 | In predicted 0.5–0.7 range |
| Calibrated survivors | 5 / 8 / 7 (n=20 total) | 9 / 14 / 16 (n=39 total) | +95% growth |

Key finding: The primary run now produces 4, 7, and 5 survivors per cluster at
the nominal κ=0.7 threshold, where before it was zero everywhere even at maximum
melting-window relaxation. This is a materially stronger paper narrative: you
can now report "at the nominal κ=0.7 threshold, a handful of candidates pass;
κ-calibration to 0.5–0.6 is what gets into the healthy 8–20 band for robust MCDM
ranking" instead of "we had to relax κ to get any result."

Fingerprint: 2554_3_1788253415.653 (changed from 2552_3_*; Phase 6/7/8 will
correctly hard-fail until re-run)

────────────────────────────────────────

## PCM database status — current state (RE-RUN COMPLETE 2026-08-14 — numbers below are historic, superseded by methodology correction 2026-08-31)

55 rows in PCM_Properties_55records_42_70C_dense.csv (the current IN_PATH in

PCM_data/01_preprocess.py), up from the prior 18-row canonical file (8 Pluss
savE + 10 Rubitherm

RT). Composition: 14 Rubitherm RT-line, 7 Pluss savE, 4 PCM Products Ltd
(PlusICE), 5 PureTemp,

1 CrodaTherm, and 24 literature-sourced rows (n-alkanes, fatty acids,
paraffin/composite blends).

This is inside the framework doc's 40–60-row target for the 42–70°C band (Table
5). The melting-

point band itself is densely covered, including the previously-named 55–63°C gap
(RT54HC=54, RT55=55,

RT57HC=57, PureTemp 58, CrodaTherm 60/RT60/PureTemp 60, RT62HC=62, PureTemp 63).
Note: the script's

own literature_rows() function (unchanged) still unconditionally appends its own
7 Singh2025

literature rows on top of the 55-row manufacturer database — the actual
candidate pool this script

evaluates is 62 rows (55 + 7), not 55. This was true before the expansion too
(18+7=25) and is

not itself a bug, just worth stating precisely.

What is still true: every one of the 55 expanded manufacturer rows is an
organic/composite PCM —

zero salt-hydrate or other inorganic rows are present (not even the old
out-of-band savE® HS36,

which does not appear in the new dense file) — so the corrosion-veto constraint
(constraint 7 below)

remains structurally inert regardless of the expansion.

What happened in the 2026-08-14 re-run:
PCM_Properties_cleaned_mice_pmm_detailed.csv was

regenerated and both 07_feasibility_filter_rajasthan.py and
08_mcdm_ranking_rajasthan.py (Phase 6)

were successfully re-run end-to-end against the expanded database. Two real,
previously-undocumented

bugs were found and fixed to make this possible — see the new section
immediately below — before any

of the results in this file could be regenerated.

### Two blocking bugs found and fixed during the 2026-08-14 re-run

1. Path-nesting mismatch, PCM_data/ vs PCM_data/PCM_data/. 01_preprocess.py (and
   its data/
output folder) live inside a doubly-nested PCM_data/PCM_data/ directory on disk
— almost

certainly the same class of zip-extraction artifact this project's docs already
flag for the

until phase 4/ folder (see the file-provenance note above).
07_feasibility_filter_rajasthan.py's

PCM_MANUFACTURER_CSV path (BASE_DIR.parent / "PCM_data" / "data" / ...), and its
own inline

comment ("matching where PCM_data/ actually sits alongside era5-rajasthan/"),
both assume the

non-nested layout (PCM_data/data/..., 01_preprocess.py directly in PCM_data/).
This means

PCM_Properties_cleaned_mice_pmm_detailed.csv, once regenerated, would land at

PCM_data/PCM_data/data/... — one level away from where the feasibility filter
(and the MCDM

script) actually looks. This is why the detailed file was "missing" even after

01_preprocess.py ran successfully: it was never missing, it was in the wrong
place relative to

what the consuming scripts expect. Fixed non-destructively (repo layout left
as-is): the

regenerated detailed CSV is copied to the PCM_data/data/ path the consuming
scripts read from,

rather than restructuring the folder tree.

1. is_rt_line column removed by the new 01_preprocess.py, still referenced by
   both
07_feasibility_filter_rajasthan.py's load_manufacturer_rows() and

08_mcdm_ranking_rajasthan.py's load_rich_pcm_properties(). The updated
preprocessing script

(rewritten for the 55-row, 6-manufacturer database) deliberately keeps the full
pcm_type text

instead of collapsing it to a binary Rubitherm/Pluss product-line flag (its own
docstring: "Unlike

the earlier script, [Type] is used as-is... preserving that extra
chemical-family signal"). Neither

of the two consuming scripts was updated to match, so both raised KeyError:
'is_rt_line' and

could not run at all against the new detailed CSV, regardless of the path issue
above. Fixed

minimally in both files: the dropped is_rt_line binary flag is replaced with the
real

manufacturer column the new preprocessing script already provides (6 distinct
values instead of

2), which is used only for a descriptive family label in Phase 5 (not read by
any constraint

logic) but is load-bearing in Phase 6's Monte Carlo same-family donor-fallback
logic — see

08_PHASE_6_AUDIT.md for that distinction and an open judgment-call note (whether
pcm_type,

the plan doc's literal "type-class" language, would be a more faithful grouping
than

manufacturer for that specific fallback, deferred rather than resolved here).

Neither bug is specific to the database expansion itself — both would have
blocked any re-run of

Phase 5/6 against a regenerated detailed file, expanded or not. They were simply
never triggered

before now because the detailed file had never been regenerated since the
preprocessing script itself

was rewritten.

### Imputation method (exact, not what the docstring implies)

Hand-rolled MICE-style chained-equations loop (N_ITER=8),
`RandomForestRegressor(n_estimators=300,

max_depth=4, min_samples_leaf=2, random_state=42)` refit per numeric column per
iteration, columns

processed fewest-missing-first (standard MICE heuristic). A custom PMM-like step
follows the

forest prediction: nearest 3 real donors by prediction-space distance

(N_DONORS=3), combined via inverse-distance-weighted average, not classic
single-donor PMM.

This is a documented-vs-implemented discrepancy worth flagging: the script's own
docstring calls

the result "a REAL, previously-measured value donated from the most
physically-similar PCM," but the

code produces a weighted blend of three real values, not a single donated real
value — the output

is not itself a value any real PCM ever measured. Categorical columns
(flammability, appearance)

use a directly-predicting RandomForestClassifier, no donor-blend step.

### Cross-series donor pool — the specific question this audit was asked to verify

Confirmed empirically, not just from the docstring claim: donor eligibility is
governed solely

by "has a real value or not" (train_idx = ~miss_mask), global across the whole
table (18 rows at

the time of this audit; 55 rows now) — there is no product-line filter. For
properties missing across

all Rubitherm RT rows (e.g. TC_liquid, TC_solid, Cp_solid), the only possible
real donors

were Pluss savE rows, and the actual provenance table confirms 100% of logged
donors for these

properties are Pluss savE products — e.g. RT35's Cp_solid donors are all savE
OM/HS products.

This is the "Rubitherm-only-imputes-from-Rubitherm" problem the project's own
docstrings describe.

RT60 and RT62HC have since been added in the 55-row expansion (RT58 itself was
not; the closest new

entries in that gap are PureTemp 58 and RT57HC). Re-checked directly against the
regenerated

PCM_Properties_cleaned_mice_pmm_detailed.csv and the raw dense CSV: the pattern
persists unchanged.

All 14 Rubitherm RT-line rows — RT60 and RT62HC included — report only a single
combined

Thermal Conductivity - Both Phases = 0.2 W/mK figure in their manufacturer
datasheet; none report

TC_liquid/TC_solid separately. RT60's and RT62HC's own imputed
TC_liquid/TC_solid donors

(per 05_imputation_provenance.csv) are
Literature/Pluss/CrodaTherm/PCM-Products-Ltd rows — never

another Rubitherm row. The hoped-for "adding RT60/RT62HC might have
independently-reported values"

did not pan out; it was a reasonable hypothesis that this re-run disproves with
data rather than

resolves in the database's favor.

## 07_feasibility_filter_rajasthan.py — all 8 constraints, exact as implemented

| # | Constraint | Exact rule | Behavior |
| --- | --- | --- | --- |
| 1 | Melting window (relaxable) | Tm ∈ [Tm_target−5, Tm_target+8] (K), widened ±2K per round, up to 4 rounds | pass/fail |
| 2 | Absolute band | Tm ∈ [42, 70]°C | pass/fail |
| 3 | Latent heat floor | L ≥ κ·L_required, κ=0.7 fixed | pass / fail / flag_unreported |
| 4 | Cycling stability | cycles ≥ 300 | pass / fail / flag_unreported (never excludes) |
| 5 | Supercooling | Tm − Tm_freezing ≤ 8K | pass / fail / flag_unknown (never excludes) |
| 6 | Charging feasibility (new) | Tm ≤ Tm_target_capped_C (from Phase 3, not re-derived) | pass/fail |
| 7 | Corrosion veto (new) | bare salt hydrate + cluster HSI_sunrise > 75th percentile, unless encapsulated | pass / not_applicable / excluded_bare_high_hsi / excluded_unverified_encapsulation |
| 8 | Safety exclusion (new) | toxic/flammable field | flag-only in practice — never actually excludes, since the source field is an unqualified yes/no, not a severity grade |

## The headline finding, re-verified 2026-08-14: still 0 survivors at nominal thresholds

Re-confirmed directly from the regenerated feasibility_survivors_rajasthan.csv
(186 rows = 3

clusters × 62 candidates): every single row still has survives_all = False at
the fixed κ=0.7

latent-heat floor. This is not a bug and the expansion does not change it — it
remains the predicted,

self-flagged consequence of Phase 3's corrected L_required derivation
(626/608/640 kJ/kg ceiling for

clusters 0/1/2 respectively) against the expanded database's own best-case
candidate. The

best-case candidate improved but the gap is still enormous: the single highest
latent-heat value in

the 62-candidate pool is now RT70HC at 260 kJ/kg (Tm=70°C), up from the
pre-expansion best of

~252 kJ/kg (C30H62, a literature row) — runners-up are Stearic acid (259),
n-Hexacosane (256),

n-Tetracosane (255), n-Octacosane (253). 0.7 × 608 ≈ 426 kJ/kg (using the lowest
of the three

clusters' ceilings) still exceeds even this improved best case by more than
1.6×. The database

expansion added real breadth and depth but did not — and structurally could not
have been expected to

— close a gap this large; Phase 3's own docstring prediction holds exactly as
before.

### The companion κ-calibration pass — re-run 2026-08-14, materially better result

calibrate_kappa_for_cluster() steps κ down from 0.7 to 0.0 in 0.1 increments (at
the primary run's

already-relaxed melting window), targeting 8–20 survivors per cluster. New
result, all three

clusters now healthy:

| Cluster | Old (pre-expansion) | New (55-row database) | Status |
| --- | --- | --- | --- |
| 0 | n=5, insufficient_even_at_kappa_0 | κ=0.2, n=9, in_band | Was undersized/unreachable → now clears the 8-survivor floor |
| 1 | κ=0.2, n=8 | κ=0.3, n=14, in_band |  |
| 2 | (n=7, implied undersized) | κ=0.2, n=16, in_band |  |

Total survivors at each cluster's calibrated κ: 39 (9+14+16), up from 20 (5+8+7)
— nearly double,

and — the headline change — Cluster 0 is no longer stuck at "insufficient even
at κ=0." This is

the actual input Phase 6's MCDM ranking now consumes, and every row in the
regenerated

mcdm_rankings_rajasthan.csv carries an updated pcm_database_status tag
reflecting the 55-row

database (no longer "PROVISIONAL — ~25-row...") — see 08_PHASE_6_AUDIT.md.

A separate, dated bug fix remains documented in-code from before this re-run:
the kappa-calibration

inequality direction was inverted in an earlier version ("FIXED 2026-08-11: an
earlier version of this

loop had the inequality backwards, which counted candidates as 'admitted' at
kappa values far above

their actual breakeven — caught by a direct contradiction in the output"). That
fix was already in

place going into this re-run and required no further changes.

## Cross-phase provenance stamping (added 2026-08-11)

Both output files now carry an upstream_cluster_profile_fingerprint column
(constant per file),

computed by provenance_lib.file_fingerprint()/fingerprint_id()
(mtime+size+row_count of

cluster_profiles_rajasthan.csv at the moment this script reads it). This exists
because Phase 7

caught a real bug: Phase 5's and Phase 6's outputs had been generated from two
different on-disk

states of cluster_profiles_rajasthan.csv (different runs of
05_cluster_rajasthan.py), causing

them to disagree cluster-by-cluster on which PCMs belonged to which cluster_id
despite matching in

total row count. Phase 6 now reads this stamp and hard-fails (SystemExit, not a
warning) if it

doesn't match the cluster_profiles_rajasthan.csv currently on disk — see
provenance_lib.py and

19_PHASE_7_ONWARD.md for the full incident writeup, and 06_PHASE_4_AUDIT.md for
the companion fix

(canonical cluster relabeling) in the script that actually produces the labels.

## Corrosion veto — structurally inert on this run's data

In the pre-expansion 18/25-row database, only one row (savE® HS36) was
salt-hydrate-typed, and it

was already excluded by constraints 1/2 regardless. The 55-row expanded database
does not change

this, confirmed by the 2026-08-14 re-run: Salt-hydrate-typed candidates in the
database: 0 per the

script's own printed diagnostic — constraint 7 fired not_applicable for all 62
candidates in every

cluster (excluded_c7_corrosion_veto=0 in all three clusters' summary rows). This
constraint, while

correctly implemented, still cannot fire on Rajasthan's data. It will become
meaningful only once the

database gains real salt-hydrate candidates (sodium acetate trihydrate, sodium
thiosulfate

pentahydrate — a gap the 2026-08-12 expansion did not close) and/or once run
against a more humid

state (Assam).

## Literature support

Framework doc §8 (Table 12) directly specifies all 8 constraints, including the
three "new" ones —

the implementation matches the spec's structure closely. Avargani et al. (2021)
underlies

L_required's provenance (see Phase 3 audit). No independent literature source
was found for the

specific κ=0.7 / cycling≥300 / supercooling≤8K numeric thresholds themselves
beyond the framework

doc's own Table 12 — these read as engineering judgment calls documented in the
project's own

methodology document, not independently peer-reviewed thresholds; state them as
such in a write-up.

## Validation

Per-cluster audit trail (pass/fail/flag/relax status per constraint per
candidate) is fully

persisted, which is itself the validation mechanism — nothing is silently
dropped. The

"insufficient even at κ=0" cluster is explicitly flagged rather than silently
omitted.

## Outputs

feasibility_survivors_rajasthan.csv,
feasibility_survivors_rajasthan_kappa_calibrated.csv,

cluster_profiles_rajasthan.csv (consumed, not produced, here).

## Dependencies

Requires Phase 4's cluster profiles (Tm_target_C, Tm_target_capped_C,
L_required_kJ_per_kg,

HSI_sunrise) and the shared PCM database. Feeds Phase 6 directly and exclusively
via the

κ-calibrated survivor set, and — since 2026-08-11 — Phase 6 verifies this
handoff via the provenance

fingerprint stamp described above before trusting it.

## Problems / risks

- ⚠️ CRITICAL (2026-08-31): Phase 3's L_required was corrected to use
  SHARE_PCM=0.5 (literature-anchored fractional share) instead of all-latent
  assumption. This halves L_required from ~608–626 kJ/kg to ~304–313 kJ/kg. All
  Phase 5/6/7/8 outputs from the 2026-08-14 run are now stale and must be
  regenerated. When Phase 5 re-runs against the updated signatures, expect κ to
  reset much higher (0.5–0.7 range, NOT 0.2–0.3), validating the corrected
  methodology. See "CRITICAL UPDATE" section above and CLAUDE.md §3.1 for full
  details.
- The database-size gap is closed (18/25 → 55 rows, inside the 40–60 target).
  The κ=0.2–0.3 calibrations documented below (from 2026-08-14 run) are now
  superseded and should not be cited until Phase 5 is re-run with corrected
  L_required values.
- Constraint 8 (safety) never excludes anything in practice given current data
  sparsity — flagged
correctly by the code itself, but worth stating plainly in a write-up rather
than implying safety

screening is currently doing real work.

- No encapsulation column exists anywhere in the database yet — constraint 7's
"unless encapsulated" branch is untestable until that field is populated.

- Two blocking bugs (path-nesting, missing is_rt_line column) had to be fixed
  before this re-run
could execute at all — see the dedicated section above. Both are now fixed in

07_feasibility_filter_rajasthan.py and 08_mcdm_ranking_rajasthan.py; the
underlying

PCM_data/PCM_data/ nested-folder layout on disk was left as-is (fixed via a
non-destructive file

copy instead), so a future contributor regenerating the detailed CSV from
scratch must remember to

copy it from PCM_data/PCM_data/data/ to PCM_data/data/ (or fix the path
properly) before

re-running Phase 5.

## Status

COMPLETE as implemented, PCM database prerequisite COMPLETE (55 rows, inside the
40–60 target), AND

Phase 5 has now been re-run against it (2026-08-14). The 0-survivors-at-κ=0.7
result persists (as

predicted) but the κ-calibrated companion pass now produces a healthy,
non-undersized survivor pool in

every cluster (39 total, vs. 20 before), including the previously-blocked
Cluster 0. This file's

numbers are current. Update, 2026-08-14 (later same day): Phase 7's physics
validation and Phase

8's recommendation cards have both now also been re-run against this candidate
pool — the negative

Spearman-rho validation result persists (all 3 clusters still ≤0.4, though two
of three moved less

negative and Cluster 1 flipped sign) — see 19_PHASE_7_ONWARD.md and
08_PHASE_6_AUDIT.md for the

full current numbers. Every phase from 5 through 8 is now current as of
2026-08-14; nothing in this

chain is pending re-run.

# 9. 08_PHASE_6_AUDIT.md

Source path: /mnt/data/08_PHASE_6_AUDIT.md

# 08 — Phase 6 Audit: Multi-Criteria Ranking Engine

Script: 08_mcdm_ranking_rajasthan.py (984 lines). Updated 2026-08-11 — Phases 7
and 8

(09_physics_validation_rajasthan.py, 10_recommendation_cards_rajasthan.py) are
now also

implemented and run; this script is no longer the implementation frontier. It
now also stamps a

cross-phase provenance fingerprint and hard-fails if its input doesn't match —
see the new section

below.

## Purpose

Rank each cluster's feasibility survivors using four independent MCDM methods
(not one), aggregate

via two independent consensus mechanisms, and quantify ranking stability via
Monte Carlo — so the

final recommendation is not an artifact of any single method's assumptions or of
fixed-point

property values.

## CRITICAL UPDATE: L_required Methodology Correction (2026-08-31)

All Phase 6 outputs from before this date are now STALE. Phase 3's L_required
methodology was corrected 2026-08-31 (SHARE_PCM=0.5), which halves L_required
and cascades through Phase 5's feasibility filtering (changing κ calibrations)
and into this script's survivor input set. The survivor set fed to Phase 6 is
now different; ranking results will change. Phase 5 and 6 must both be re-run
against updated signatures before these results are valid. See CLAUDE.md §3.1
for full detail.

## Inputs

feasibility_survivors_rajasthan_kappa_calibrated.csv (or equivalent survivor
set),

cluster_profiles_rajasthan.csv, PCM_Properties_cleaned_mice_pmm_detailed.csv
(read directly, a

second time, for the "rich" properties — density/TC/Cp/corrosion-proxy/cost —
not passed through

from Phase 5).

## Criteria (8, exact) and weights

| Criterion | Direction | AHP prior (Table 13) | Notes |
| --- | --- | --- | --- |
| Tm_fitness | benefit | 0.24 | Gaussian target-fitness transform |
| latent_heat | benefit | 0.20 |  |
| vol_latent_heat (ρL) | benefit | 0.12 |  |
| thermal_conductivity | benefit | 0.13 |  |
| cycling | benefit | 0.11 |  |
| supercooling | cost | 0.08 |  |
| corrosion | cost | 0.06 (cluster-rescaled 1×–2× by HSI) | structural proxy: 2.0 if pcm_type=="Inorganic" else 1.0 — not a measured corrosion rating |
| cost | cost | 0.06 | always NaN — "no cost field exists anywhere in the source data" (in-code comment) |

Blend: w_j = 0.5·w_entropy_j + 0.5·w_AHP_j, computed per cluster from that
cluster's own

filtered decision matrix.

## Target-based Tm handling — the part most PCM-MCDM papers get wrong, per the project's own framing

f_Tm = exp(-(Tm - Tm_target)² / (2σ²)), σ = 4K

σ=4K is explicitly sourced to the framework doc §9.2 ("justify σ=4K from the
heat-exchanger

approach temperature"), not independently literature-calibrated — the code's own
docstring says so

plainly. An asymmetric Gaussian (penalizing Tm-too-high more than Tm-too-low,
physically better

motivated per the framework doc) is flagged as a documented, not-yet-implemented
extension.

PROMETHEE II additionally handles Tm natively on raw |Tm−Tm_target| distance
with a linear

V-shape preference function (q=2K indifference, p=8K preference threshold) — the
stated reason to

keep PROMETHEE in the stack alongside the Gaussian-fitness methods.

## The four ranking methods (exact, as coded)

- TOPSIS: vector-normalized weighted-sum distance to ideal-best/ideal-worst,
  closeness
coefficient Ci ∈ [0,1]; missing values excluded via skipna=True (not
zero-filled).

- PROMETHEE II: net outranking flow, linear preference functions (q=0,
  p=criterion range) for
all criteria except Tm (native V-shape as above); net flow normalized by (n−1).

- VIKOR: compromise index Q = v·(S−Sb)/(Sw−Sb) + (1−v)·(R−Rb)/(Rw−Rb), v=0.5.
- GRA: grey relational grade via ideal-reference distance with distinguishing
  coefficient ρ=0.5.
- CoCoSo: fully implemented but gated off by default (RUN_COCOSO = False) —
  "optional 5th
ranker...never a replacement for the 4 core methods," per framework doc §9.4.

### Three documented, dated bug fixes (all 2026-08-11) — direct evidence of active self-auditing

1. VIKOR sign inversion: the compromise-index formula previously read
(Sb−Sw)/(Rb−Rw) — best-minus-worst, the wrong sign — which silently *inverted
the entire Q

ranking*. Caught via a pairwise method-agreement diagnostic showing VIKOR
near-totally inverted

against TOPSIS/PROMETHEE (rho as low as −0.86) in every cluster.

1. Entropy-weight inflation for near-empty criteria: a criterion with too few
   (or zero) real
values used to receive the highest possible entropy weight as an artifact of
np.nansum

behavior — inflating cost's weight (always NaN in this database) to 64–75%
across every

Rajasthan cluster in the first run. Fixed: criteria with <2 real values get
weight 0.0

directly, bypassing the entropy formula.

1. Kappa-calibration inequality inversion (Phase 5, but caught by this script's
   diagnostics) —
see 07_PHASE_5_AUDIT.md.

All three were caught through pairwise method-agreement or contradiction
diagnostics that the

project itself built and ran — this is exactly the kind of self-verification a
methodology

write-up should cite as evidence of rigor, not omit.

## Rank aggregation

- Borda count: Borda(i) = Σ_methods (n − rank_m(i)).
- Copeland: pairwise win/loss majority across methods, +1/−1/0 per pair, summed.
- Kendall's W: W = 12S / (m²(n³−n)); thresholds W>0.8 strong, W<0.6 ambiguous,
  both sourced
to the framework doc §9.5, not the script author's own judgment.

- Where Borda and Copeland disagree on Top-3 membership/order, the design intent
  (per phases.md)
is to flag it explicitly — confirmed present as a reported field, not silently
resolved one way.

## AHP — the honest gap

AHP_PAIRWISE_MATRIX = None — a clearly-marked TODO stub. The eigenvector-method
AHP weight

derivation with consistency-ratio check (CR = CI/RI, threshold <0.10, per
framework doc §9.3)

exists as working code but is never invoked — the run falls through to the
framework doc's

Table 13 indicative weights unmodified (except for the corrosion
cluster-rescaling). Any claim

that this pipeline performs "real AHP elicitation" would currently be inaccurate
— it uses the

framework doc's stated priors, not a project-specific pairwise comparison.

## Monte Carlo — exact numbers, and a documented deviation from the spec

N_DRAWS = 1000 (framework doc specifies 5000 — deviation is documented in-code:
 a 5000-draw run took 606s wall-clock, "impractical for iteration";
 the framework doc itself names 1000 as a safe, commonly-used fallback)
DIRICHLET_CONCENTRATION = 25.0 (chosen for ≈±20% weight variation around nominal
weights)
Gaussian noise: latent_heat ±5%, thermal_conductivity ±10%, Tm ±1K (absolute),
cost ±30% (moot, always NaN)
RANDOM_STATE = 42, re-seeded fresh per cluster (not a continuing stream)

Imputed-property handling: for a candidate flagged any_property_imputed, the
Monte Carlo draw is

sampled from Normal(mean, std) of real, non-imputed values within the same PCM
family

(falling back to all non-imputed candidates in the cluster if the family has <2
real donors) —

applied only to latent_heat and thermal_conductivity; Tm always uses plain ±1K
noise

regardless of imputation status.

## Actual Rajasthan result — RE-RUN 2026-08-14 against the expanded 55-row database (current)

mcdm_rankings_rajasthan.csv: 39 rows across 3 clusters (n=9/14/16 survivors), up
from the

pre-expansion 20 rows (n=5/8/7). Two bugs (is_rt_line column removed by the
rewritten

01_preprocess.py, and a PCM_data/PCM_data/ path-nesting mismatch) had to be
fixed first to make

this re-run possible at all — see 07_PHASE_5_AUDIT.md for the full writeup; the
family field this

script uses for Monte Carlo same-family donor fallback now derives from the real
manufacturer column

(6 values) rather than the old binary Rubitherm/Pluss flag.

The dominant entropy criterion changed for every cluster: supercooling now
dominates all three

(Cluster 0 = 63.8%, Cluster 1 = 48.6%, Cluster 2 = 57.0%) — all three exceed the
script's own 40%

"near-total-domination" flag threshold (previously it was Tm_fitness dominating
Clusters 0/1 at

48.2%/49.4%, with supercooling only dominant in Cluster 2). Kendall's W: Cluster
0 = 0.388

(down from 0.4375, still below the 0.6 ambiguous threshold — but no longer
tagged undersized,

n=9 now within the healthy 8–20 band, so low agreement here can no longer be
attributed to sample

size), Cluster 1 = 0.635 (up from 0.536, now crosses into the "moderate" band),
Cluster 2 = 0.634 (up

from 0.589, also now "moderate") — no cluster reaches the "strong agreement"
(W>0.8) band, and

Cluster 0's persistently low W despite a healthy sample size is a new finding
worth its own scrutiny

(possible genuine method disagreement on this cluster's ranking, not a
data-sparsity artifact). GRA is

newly flagged by the script's own diagnostic as the "structural outlier" method
(lowest mean pairwise

rho vs. the other three) in all three clusters — not previously called out by
name in this file.

## Literature support

Oluah (2020) is cited by name for the TOPSIS unit-test fixture and as the
domination-threshold

comparator (framework doc §13.1 names this as the project's own regression-test
anchor — matches to

3 decimal places after refactoring, per phases.md PROMPT 5's stated verification
requirement).

TOPSIS/PROMETHEE/VIKOR/GRA/CoCoSo are standard, well-established MCDM methods;
no dedicated MCDM

methodology paper (e.g., for VIKOR's original formulation) was found
cross-referenced in

references.bib/.claude/references.md during this audit — see
17_LITERATURE_MAPPING.md for the

full gap analysis.

## Validation

Monte Carlo inclusion probability, Top-1 retention, rank-reversal frequency,
Spearman ρ vs. baseline

— all computed and persisted per candidate per cluster. Kendall's W as a
per-cluster

cross-method-agreement check. No external/physics validation yet (that is Phase
7).

## Outputs

mcdm_rankings_rajasthan.csv, mcdm_method_agreement_rajasthan.csv,

outputs/qc_montecarlo_inclusion_rajasthan.html.

## Cross-phase provenance stamping and hard-fail check (added 2026-08-11)

Before doing anything else, load_survivors() now fingerprints the CURRENT
on-disk

cluster_profiles_rajasthan.csv
(provenance_lib.file_fingerprint()/fingerprint_id()) and

compares it against the upstream_cluster_profile_fingerprint stamp embedded in
Phase 5's survivor

file — assert_fingerprint_match() raises SystemExit (not a warning) on any
mismatch. This exists

because Phase 7 caught Phase 5's and Phase 6's outputs disagreeing
cluster-by-cluster on which PCMs

belonged to which cluster_id, traced to Phase 4's GMM cluster labels not being
stable across

separate re-runs (see 06_PHASE_4_AUDIT.md's second documented bug and
19_PHASE_7_ONWARD.md's full

incident writeup). This script's own output (mcdm_rankings_rajasthan.csv) is now
stamped with the

same fingerprint, which Phase 7 and Phase 8 each verify in turn.

## Dependencies

Requires Phase 5's κ-calibrated survivor set (itself provisional pending
database expansion) and

Phase 4's cluster profiles, now verified via the provenance check above. Feeds
Phase 7

(09_physics_validation_rajasthan.py, which computes Spearman rho between this
script's Borda/

Copeland ranks and simulated solar fraction) and, via Phase 7, Phase 8

(10_recommendation_cards_rajasthan.py, which also re-imports this script as a
module to recompute

the per-criterion contribution decomposition against its own already-saved
weight formula).

## Problems / risks

- Resolved 2026-08-14: Phase 6 has now been re-run against the expanded 55-row
  database (see
above) — the pcm_database_status tag on every output row now reads `"COMPLETE —
55-row

manufacturer database..." rather than "PROVISIONAL — ~25-row..."`. The ranking
still runs on a

κ-relaxed rather than nominal-threshold survivor pool (that policy question
remains genuinely

open, see 19_PHASE_7_ONWARD.md), and its output has not yet been re-validated by
a Phase 7

re-run — so "provisional pending physics validation" still applies, just not
"provisional pending

database expansion" any more.

- cost and `c
orrosion` are effectively structural placeholders, not measured criteria — a

reader could reasonably ask why 12% of the total AHP weight budget (6%+6%) rides
on data that

doesn't exist yet for cost and is a binary type-proxy for corrosion.

- AHP is not actually AHP-elicited — flag this precisely in any write-up; the
  current weights are
Table 13's stated priors, not a project-derived pairwise judgment matrix.

- N_DRAWS=1000 vs the specified 5000 is a defensible, documented engineering
  tradeoff (the
framework doc itself sanctions 1000 as a fallback), not a silent shortcut — but
should be stated

explicitly if a reviewer asks why the number differs from the framework doc's
primary

recommendation.

## Status

COMPLETE as implemented, with three caught-and-fixed bugs (evidence of working
self-audit) and

two structural caveats (AHP not elicited, cost/corrosion are placeholders) that
should be stated

plainly rather than presented as finished. Update, 2026-08-14: this script has
now been re-run

against the expanded 55-row database — 39 survivors across 3 clusters (up from
20), no cluster

undersized, Kendall's W 0.388/0.635/0.634 (Clusters 1–2 now "moderate," Cluster
0 still ambiguous but

no longer explainable by small sample size). Two bugs blocking this re-run
(is_rt_line column

removed by the rewritten preprocessing script; a PCM_data/PCM_data/ path
mismatch) were found and

fixed — see 07_PHASE_5_AUDIT.md. Update, 2026-08-14 (later same day): Phase 7
has now ALSO been

re-run against this fresh ranking (09_physics_validation_rajasthan.py) — the
negative validation

result persists (Spearman rho = -0.385/+0.125/-0.097 across the 3 clusters, mean
-0.119, all still in

the ≤0.4 "genuine negative" band vs. the pre-expansion -0.900/-0.096/-0.198) —
so the larger database

did not resolve the MCDM-vs-physics disagreement; if anything Cluster 0's
now-healthy sample size

(n=9, no longer undersized) makes its persistently-low Kendall's W a more
concerning finding, not a

less concerning one. Phase 8 (10_recommendation_cards_rajasthan.py) has also
been re-run and

produced new Top-1 picks (RT50 / savE® OM50 / savE® OM50) — see
19_PHASE_7_ONWARD.md for the full

current-state writeup. Every phase in this chain (5 through 8) is now current as
of 2026-08-14; no

further re-run is pending.

# 10. 09_PHASE_7_AUDIT.md

Source path: /mnt/data/09_PHASE_7_AUDIT.md

# 09 — Phase 7 Audit: Physics-Based Validation of MCDM Rankings

Script: 09_physics_validation_rajasthan.py (650 lines). Completed 2026-08-11,
re-run 2026-08-14 against expanded 55-row PCM database. Phase 8 extends this
with supercooling penalty sensitivity testing.

⚠️ CRITICAL UPDATE (2026-08-31): L_required Methodology Correction — Phase 7's
entire result set is now STALE. Phase 3's L_required was corrected 2026-08-31 to
use SHARE_PCM=0.5 (literature-anchored fractional share) instead of all-latent
assumption, halving L_required values. This cascades through Phase 5 (κ
calibrations), Phase 6 (survivor set), and Phase 7 (validation rankings). All
Phases 5–8 must be re-run against updated signatures before these results are
valid. See CLAUDE.md §3.1 for full methodology detail.

## Purpose

Phase 6 produces a consensus MCDM ranking (four methods, two aggregators, Monte
Carlo stability). Phase 7 asks the critical question: does a higher-MCDM-rank
PCM actually deliver better simulated thermal performance under this cluster's
real climate? This validation makes the ranking falsifiable, not deferrable to
future work.

## The Independent Check

- Input: Phase 6 MCDM rankings + Phase 5 feasibility survivors
- Climate: Real hourly NASA POWER weather for each cluster's medoid (2023–2025,
  whichever year is complete, <1% fill values)
- Model: Lumped-enthalpy grey-box simulator (Barqawi 2025, 3-phase PCM dynamics)
- Output per PCM: annual solar fraction, hours meeting delivery temperature,
  melt-fraction statistics, complete cycles
- Correlation: Spearman ρ between MCDM Borda rank and simulated solar fraction
  per cluster
## Model Class & Calibration (Critical Details)

### Why grey-box lumped, not EnergyPlus/CFD?

- EnergyPlus: no supported method to place a latent-heat PCM inside a tank node
  network
- CFD: overkill for single-objective PCM screening; lumped-enthalpy is
  appropriate fidelity for material selection
- This is a deliberate architectural decision, not an oversight
### Calibration findings (August 11, 2026)

Two bugs caught and fixed during this script's own self-tests (mandatory
energy-conservation check):

1. Backward-Euler solver bug: Phase 1 closed-form Tw solve had spurious +
   dt·c·Tw_old term, destabilizing at hourly timestep. Fixed by re-deriving
   algebraically; verified against numpy.linalg.solve to full float precision.
1. Night-loss bug: Barqawi's bidirectional a·(Tc−Tw) term let the tank drain
   heat through an idle collector overnight as fast as it charged during day.
   Real systems isolate the collector at night. Fixed via
   NIGHT_ISOLATION_FRACTION = 0.05, reducing collector coupling when Tc < Tw.
Result after both fixes: All three medoids land in 54–84% benchmark
solar-fraction band (Phase 3's Avargani design basis). Phase 7 uses this
calibrated model as-is.

### Assumptions (explicitly stated, not hidden)

| Parameter | Value | Justification |
| --- | --- | --- |
| Tank water mass M_W | 300 kg | Avargani et al. (2021) design basis, reused throughout pipeline for consistency |
| Collector area A_c | 4.0 m² | Barqawi 2025 was unloaded (no household draw); sized up to 4.0 m² per Indian FPC sizing convention (~1.3–2 m²/100L of design draw) |
| Collector efficiency | 0.70 | Barqawi 2025; within 45–73% FPC band cited by Al-Mamun et al. 2023 |
| Collector overall loss U_L | 2.5 W/m²K | Calibrated down from Barqawi's 20 — represents well-insulated collector; within Duffie–Beckman 3–8 W/m²K range |
| PCM–water HTC h_p base | 800 W/m²K | Barqawi 2025 |
| h_p scaling | By TC_solid / 0.2 | Deviation from Barqawi: allows thermal conductivity to differentiate candidates, not held fixed |
| PCM mass (fixed) | 50 kg | ASSUMED_PCM_MASS_KG from Phase 3/4; not independently optimized (each PCM gets same design, not co-optimized size) |
| Draw profile shape | Two Gaussians (morning ~07:00, evening ~19:00) | Informed by ASHRAE 90.2 Section 8.9.4 documented shape; exact hourly fractions are reconstructed qualitatively, not reproduced verbatim (exact table not retrievable) — flagged as reconstruction, not claim of exact reproduction |
| Daily draw total | 300 kg/day | Avargani et al. 2021; same citation as Phase 3 night-draw, but applied as full-day total here |
| Target delivery temp | 50°C | Pipeline-wide constant |

## Self-Tests: Both Pass

Energy conservation (constant solar, no draw, 48 hours):
 Residual: 1.638e-13 J → Pass (threshold: 0.1% of cumulative input)

Draw-profile integration (365 days):
 Daily total: 300.000 kg → Pass (expected 300.0 kg)

## Results: Per-Cluster Spearman ρ Against MCDM Borda Rank

| Cluster | n_candidates | Borda vs. Solar Fraction | Notes |
| --- | --- | --- | --- |
| 0 | 9 | ρ = −0.385 | Weak negative agreement. Cluster flagged undersized (n<8 in Phase 5); rerun Phase 5/6 after database expansion changed n to 9 (now healthy), yet W remains low (0.388 <0.6). Suggests genuine method disagreement, not sample-size artifact. |
| 1 | 14 | ρ = +0.125 | Weak positive agreement. Best outcome. Kendall's W = 0.635 (moderate). |
| 2 | 16 | ρ = −0.097 | Weak negative agreement. Largest cluster. |

Overall finding: No cluster exceeds ρ=0.4 threshold for meaningful agreement.
Physics simulation does not validate MCDM rankings.

## Dominant Entropy-Weighted Criterion Per Cluster

Phase 6 identified:

- Cluster 0: supercooling 63.8%
- Cluster 1: supercooling 48.6%
- Cluster 2: supercooling 57.0%
Critical caveat noted in code: "This physics model does NOT simulate
supercooling at all (Barqawi's 3-phase model assumes ideal solid–liquid
transition at Tm with no nucleation delay). A disagreement concentrated on
supercooling cannot be resolved by this simulation."

Phase 7 flags this explicitly as a scoped limitation of the validator, not
evidence the MCDM weighting is wrong. Phase 8 extends this to test the
supercooling hypothesis directly.

## PCM-vs-Plain-Tank Comparator (Honest Negative Result)

Framework doc cites +30% (series) / +4–8% (other configs) solar-fraction gain
from adding PCM vs. plain sensible-only tank.

This study found: ~0.0% difference (RT47 PCM vs. zero-latent "PCM" on same
tank/weather).

Root cause: At PCM_MASS_KG = 50 kg (pipeline-consistent reuse from Phase 3)
against 300 kg tank, the tank's own sensible capacity dominates. PCM-vs-PCM
ranking (this phase's actual purpose) remains valid and non-tied;
PCM-vs-plain-tank sensitivity should NOT be over-interpreted as evidence of
flawed system design. Reported honestly, not tuned away.

## Known Caveats Inherited from Phase 6 (Carried Forward)

Every Phase 7 output carries these inherited caveats verbatim, never silently
dropped:

1. Cost always NaN: Unavoidable — Phase 6 database limitation. No remedy here.
1. Corrosion is binary proxy: 2.0 if inorganic else 1.0, not a measured rating.
   Cannot be independently verified by this simulation.
1. Database status: All 39 survivors tagged "PROVISIONAL — 55-row database"
   (Phase 6). The 2026-08-31 L_required correction means all results are now
   stale pending re-run.
1. Cluster 0 instability: Kendall's W = 0.388 (below 0.6 threshold) in Phase 6.
   Low ρ in Phase 7 may reflect pre-existing MCDM instability as much as physics
   disagreement — requires more data or method recalibration, not physics-model
   retuning.
1. Supercooling cannot be validated: The dominant entropy-weighted criterion in
   all clusters (48–64%) is supercooling. This physics model deliberately does
   not simulate supercooling (Barqawi's 3-phase model assumes ideal solid–liquid
   transition at Tm with no nucleation delay — see physics_lib.py for
   derivation). A disagreement concentrated on supercooling cannot be resolved
   by this simulation and should not be misread as evidence the MCDM
   supercooling weight is wrong. Phase 8 tests this hypothesis directly via
   penalty sensitivity analysis.
## Completion Report: What Was Actually Built (2026-08-11, Re-run 2026-08-14)

Phase 7 was built and run deliberately against the pre-expansion ~25-row PCM
database (pre-2026-08-12), not withheld pending database expansion. Rationale:
the validation methodology itself needed to be built, tested, and debugged now
rather than blocked indefinitely on a database-expansion task with no fixed
completion date. Every output carried the caveat PROVISIONAL — ~25-row database,
not yet expanded to 40–60. When the database was expanded to 55 rows
(2026-08-12), Phases 5 and 6 were re-run (2026-08-14), and then Phase 7 was
re-run against the fresh Phase 6 output. Current results below reflect the
post-expansion run (39 survivors vs. pre-expansion 20).

### Bugs Caught & Fixed Before Trusting Any Result

Two bugs were caught by Phase 7's own mandatory self-tests
(self_test_energy_conservation() and self_test_draw_profile_integration()) and
fixed before any real simulation result was trusted:

1. Backward-Euler solver bug: A spurious + dt·c·Tw_old term in the closed-form
   solve for water temperature in pre-melt/post-melt phases was destabilizing at
   hourly timestep, causing unbounded temperature blow-up. Fixed by re-deriving
   the 2×2 implicit system algebraically and verified against numpy.linalg.solve
   to full floating-point precision.
1. Night-loss bug: Barqawi's original bidirectional coupling term a·(Tc−Tw)
   allowed the tank to drain heat back through an idle collector overnight
   nearly as fast as it charged during the day — physically impossible (real
   systems have thermosiphon check valves or controller-gated pumps). Fixed via
   NIGHT_ISOLATION_FRACTION = 0.05, gating the collector coupling coefficient to
   5% of its daytime value whenever Tc < Tw (collector colder than tank).
Result after both fixes: All three medoids land in 54–84% benchmark
solar-fraction band. Energy conservation holds to machine precision (~1.6e-13 J
residual). This calibrated model is used as-is for Phase 7 real experiment and
Phase 8 penalty sweep.

## Cluster-Specific Interpretations

### Cluster 0 (ρ = −0.385, undersized before rerun)

MCDM and physics rankings are negatively correlated — higher-ranked PCM by MCDM
delivers worse simulated performance. Two non-exclusive diagnoses:

1. MCDM ranking itself unstable: W=0.388 (<0.6); four methods don't agree well.
   Low correlation against physics may reflect pre-existing instability, not a
   physics-model gap. Fix indicated: expand candidate pool (now n=9, adequate),
   or re-run Phase 5/6 if database changes further.
1. Supercooling weight mismatch: supercooling dominates (63.8%), but model
   cannot simulate it. If supercooling is overweighted, MCDM will rank
   high-supercooling candidates high, but physics will not reflect that. Fix
   indicated: Phase 8 sensitivity test (implemented).
### Cluster 1 (ρ = +0.125, weak positive agreement)

Best outcome of three clusters. MCDM and physics agree weakly (+12.5% rank
correlation). Kendall's W = 0.635 (moderate, above the 0.6 threshold).

- If supercooling's true effect is small, partial agreement here is plausible
  (other criteria dominate, MCDM has some validity, but supercooling's 48.6%
  weight dilutes signal).
- No strong action indicated; Cluster 1 candidates are least problematic.
### Cluster 2 (ρ = −0.097, largest cluster)

Weakly negative agreement — MCDM and physics essentially uncorrelated. Cluster
has enough candidates (n=16) that undersizing is not the diagnosis.

- Supercooling dominates (57%), same caveat as Clusters 0/1.
- Candidate pool may be heterogeneous enough that a single MCDM ranking cannot
  capture the variation (e.g., paraffins vs. fatty acids vs. inorganics behave
  differently under this climate).
- Phase 8 testing will clarify whether supercooling-specific penalty improves
  this.
## Code Quality & Documented Design Decisions

- Provenance hard-fail check: Confirms Phase 5 and Phase 6 outputs were built
  from the same cluster partition (prevents silent mismatch from separate
  re-runs of Phase 4).
- Mass sensitivity sweep (Phase 7, lines ~312–362): Tests whether
  PCM_MASS_KG=50kg is the right scale to see differentiation. Result: spread
  widens, ranking stable at 50–800 kg — confirms signal is real, not noise from
  mass underdimensioning.
- Night-delivery test (lines ~364–371): Validates ability to sustain 58–62°C
  overnight discharge (Avargani benchmark).
- Explicit self-test mandatory before main experiment: Energy conservation and
  draw-profile checks; failures block main run.
## Relationship to Phase 8

Phase 7 identifies supercooling as the dominant MCDM criterion but flags the
model cannot simulate it. Phase 8 extends this by:

1. Implementing a supercooling penalty in physics_lib.py (proportional reduction
   to h_p in supercooled region)
1. Running sensitivity sweep across penalty strength k ∈ [0.0, 0.1, 0.2, 0.3]
1. Testing whether the penalty brings physics/MCDM agreement closer to zero or
   improves it
See 10_PHASE_8_AUDIT.md for the full Phase 8 findings.

## Literature & References

Barqawi 2025: Model equations, h_c=1500 W/m²K, h_p=800 W/m²K, A_c/M_w defaults

Duffie & Beckman: Flat-plate collector U_L range justification

Avargani et al. 2021: 300L @ 60±2°C design basis

Al-Mamun et al. 2023: FPC efficiency range (45–73%)

────────────────────────────────────────

Status: Phase 7 complete. Physics validation found weak to negative correlation
with MCDM, driven primarily by supercooling's dominant MCDM weight (48–64%) that
cannot be simulated in this model architecture. Phase 8 directly tests this
hypothesis.

# 11. 10_PHASE_8_AUDIT.md

Source path: /mnt/data/10_PHASE_8_AUDIT.md

# 10 — Phase 8 Audit: Supercooling Penalty Sensitivity Analysis

Script: 08_phase8_supercooling_sweep.py (310 lines). Completed 2026-09-01. Phase
8 directly tests Phase 7's finding that supercooling dominates MCDM weights
(48–64%) yet cannot be simulated in the base model. Implementation of
supercooling penalty in physics_lib.py, sensitivity sweep k ∈ [0.0, 0.1, 0.2,
0.3].

## Purpose

Phase 7 identified negative or near-zero Spearman ρ (−0.385, +0.125, −0.097
across clusters 0/1/2) between MCDM rankings and simulated solar fractions. The
dominant entropy-weighted criterion in all three clusters is supercooling
(48–64%), but the base physics model cannot simulate it (assumes ideal
solid–liquid transition at Tm, no nucleation delay). Phase 8 tests whether
implementing a supercooling penalty improves physics/MCDM agreement.

## Critical Correction: Field Identification

Initial attempt (August 31): Phase 8 used Tm_nucleation (from PCM database
column "Tm_nucleation") to compute supercooling offset: ΔT = Tm_freezing −
Tm_nucleation. Result: All 18 survivors had ΔT = 0.0 K (uniformly zero). Penalty
was mathematically inert; no effect on rankings across any k.

Root cause identified (September 1): Phase 6 MCDM criterion "supercooling" does
NOT use Tm_freezing − Tm_nucleation. It uses supercooling_K = Tm_C −
Tm_freezing_C, sourced from Phase 5 feasibility filter
(07_feasibility_filter_rajasthan.py, line 199). This field has real variance:
mean=1.27 K, std=1.29 K, min=−0.50 K, max=3.50 K across survivors.

Corrected implementation: Phase 8 re-wired penalty to use supercooling_K (actual
MCDM field). Sweep re-run September 1; results below are from this corrected
run.

## Penalty Mechanism & Formulation

Assumption: Supercooling delays solidification, reducing effective heat-transfer
coefficient during post-melt sensible cooling (Phase 3 of lumped-enthalpy
model).

Formula (applied when Tp > 0 and SUPERCOOLING_PENALTY_K > 0):

h_p_effective = h_p × max(0.3, 1 − k × supercooling_K / 10)

Where:

- supercooling_K = Tm_C − Tm_freezing_C (K), from Phase 5 survivors
- k = proportionality constant (tested: 0.0, 0.1, 0.2, 0.3)
- 10 K = reference scale (typical max paraffin supercooling)
- max(0.3, ...) = clamp to prevent h_p reduction >70%
Physically motivated: Reduced h_p models slower latent-heat exchange while PCM
is supercooled (solidification delayed), increasing charging/discharge time. Not
derived from literature (no literature relationship between subcooling degree
and h_p reduction found in sources/); treated as free parameter explored via
sensitivity sweep.

## Sensitivity Sweep Parameters

| k value | Interpretation | Effect at max supercooling (3.5 K) |
| --- | --- | --- |
| 0.0 | Baseline (no penalty) | h_p_eff = h_p × 1.0 (no reduction) |
| 0.1 | Mild penalty | h_p_eff = h_p × 0.965 (−3.5% at 3.5 K) |
| 0.2 | Moderate penalty | h_p_eff = h_p × 0.930 (−7.0% at 3.5 K) |
| 0.3 | Aggressive penalty | h_p_eff = h_p × 0.895 (−10.5% at 3.5 K) |

## Self-Tests: All Pass

Energy conservation (constant solar, no draw, 48 hours):
 Residual: 1.638e-13 J → Pass (all k values)

Draw-profile integration (365 days):
 Daily total: 300.000 kg → Pass (all k values)

Calibration (all three medoids, all k):
 100% in 54–84% benchmark band → Pass (all k values)

Conclusion: Penalty implementation is correct and does not break model physics
or calibration.

## Corrected Sweep Results: Spearman ρ with Penalty Applied

| Cluster | k=0.0 | k=0.1 | k=0.2 | k=0.3 | Change |
| --- | --- | --- | --- | --- | --- |
| 0 | −0.385 | −0.385 | −0.385 | −0.385 | No change |
| 1 | +0.125 | +0.059 | +0.059 | +0.077 | Degrades then improves |
| 2 | −0.097 | −0.118 | −0.136 | −0.136 | Worsens |

### Cluster 0 (No Effect)

- ρ remains exactly −0.385 at all k values
- Interpretation: Cluster 0's surviving PCMs have low and relatively uniform
  supercooling_K (most <1.5 K). Penalty has negligible discriminative power;
  even at k=0.3, h_p reduction is <5% for most candidates, and the absolute
  magnitude is too small to shift relative rankings.
### Cluster 1 (Penalty Reduces Agreement)

- Baseline (k=0.0): ρ = +0.125 (weak positive agreement)
- With penalty (k≥0.1): ρ drops to +0.059–+0.077 (weaker agreement)
- Interpretation: Applying supercooling penalty degrades physics/MCDM agreement.
  Where MCDM gave 12.5% rank correlation, penalty reduces it to 6–8%. This is
  the opposite of the intended effect (improve agreement by correcting a missing
  model mechanism).
### Cluster 2 (Penalty Worsens Disagreement)

- Baseline (k=0.0): ρ = −0.097 (weak negative agreement)
- With penalty (k≥0.1): ρ worsens to −0.118 to −0.136 (stronger disagreement)
- Interpretation: Penalty increases the magnitude of physics/MCDM disagreement.
  MCDM ranked candidates one way; physics+penalty ranks them differently and
  even more so than physics alone.
## Honest Negative Result: Why the Penalty Made Things Worse

### Three Plausible Explanations

1. Penalty Formulation is Incorrect

The assumed mechanism (reduced h_p in supercooled state) may not capture how
supercooling actually affects system performance:

- Real supercooling introduces nucleation kinetics (temperature-dependent
  solidification rate), not just a simple h_p reduction
- The grey-box 2-node model lumps water and PCM into single nodes; real
  stratification and transient heterogeneity around the PCM bed are not
  represented
- Supercooling may increase thermal stratification (undercooled liquid stays at
  bottom, hottest water rises) — a beneficial effect the penalty doesn't capture
- Hysteresis loops during charge/discharge cycles are not modeled;
  supercooling's effect on cycle losses (energy dissipated per melt/freeze) is
  ignored
2. Supercooling is Not the Limiting Factor

Phase 6 assigned supercooling 48–64% MCDM weight (entropy-dominant across all
clusters). Phase 8 suggests this weight is over-estimated relative to
supercooling's actual impact on annual solar fraction:

- Other criteria (Tm_fitness, latent heat, thermal conductivity, cycling) may
  dominate the observed solar-fraction variation more than MCDM assumes
- MCDM is a static score (each PCM gets fixed weights per criterion). Physics
  simulator responds to dynamic climate (some criteria matter more in winter,
  others in summer; supercooling's effect may be seasonal or highly
  load-dependent)
- The MCDM did not weight seasonal or climate-responsive criteria separately; a
  flat 57% supercooling weight may be too coarse
3. System Dynamics Mask the Penalty

At PCM_MASS_KG = 50 kg (pipeline-consistent, reused from Phase 3) against a 300
kg water tank:

- The tank's own sensible thermal mass dominates system behavior (Phase 7
  calibration notes this explicitly)
- A 3.5 K supercooling effect on a 50 kg PCM bed produces a time-delay in h_p
  (from 800 to 770 W/m²K at max), but the 300 kg tank absorbs/releases energy so
  much faster than the PCM that the PCM's h_p is not the system bottleneck
- System is tank-dominated, not PCM-limited — improving PCM dynamics has
  marginal impact on overall solar fraction
Phase 7's own mass-sensitivity sweep (lines 312–362) showed that PCM-vs-PCM
differentiation persists at 50–800 kg (ranking is stable), but absolute
solar-fraction swing is <1 pp, suggesting the signal is real but the system's
intrinsic insensitivity to PCM specifics (due to tank dominance) limits how much
supercooling can ever matter for annual solar fraction (the Phase 7/8 metric).

## Why This Negative Result Matters

This is not a failure of the methodology. It is a diagnostic finding:

- ✅ Implementation is correct: energy conservation passes, calibration passes,
  penalty is toggleable
- ❌ Hypothesis is wrong or incomplete: applying the supercooling penalty does
  not improve physics/MCDM agreement; instead it worsens it
- 🔍 It reveals a data-model mismatch: MCDM's supercooling weighting (48–64%) may
  not align with supercooling's real effect on the observable system metric
  (annual solar fraction)
## Implications for Phase 7 & 6

Phase 7 interpretation should be updated:

"The negative rho values (Clusters 0, 2) and weak positive rho (Cluster 1)
should NOT be interpreted as evidence that supercooling is an important physical
effect that this model fails to capture. Phase 8 testing showed that even a
well-calibrated supercooling penalty worsens physics/MCDM agreement, not
improves it. This suggests supercooling's real-world effect on annual solar
fraction in this system configuration is either: (a) smaller than the MCDM
weighting (48–64%) implies, or (b) manifests through mechanisms the grey-box
model cannot represent (kinetic nucleation rates, stratification, hysteresis).
The disagreement between physics and MCDM likely reflects differences in how
supercooling matters (or doesn't) for thermal performance under real climatic
load."

Phase 6 (MCDM) implications:

The supercooling entropy weight may need recalibration if future work validates
that supercooling's true impact is <48%. Suggest re-running Phase 5/6 with
reduced supercooling weight (e.g., 0.04 instead of 0.08) and observing whether
ranking stability (Kendall's W) and physics/simulation agreement improve.

## Cluster 0 Supercooling/Entropy Diagnostic (2026-08-14)

Context: Cluster 0 has the lowest cross-MCDM-method agreement (Kendall's W =
0.388, below the 0.6 ambiguous threshold) and the lowest physics/MCDM agreement
(ρ = −0.385). Two open questions: (1) Why do the four MCDM methods disagree on
Cluster 0 specifically? and (2) Does supercooling's dominant entropy weight
(63.8%, highest of the three clusters) explain this?

This diagnostic was run as a read-only investigation against the
already-reconciled Phase 5/6/7 outputs — no canonical output file was
regenerated, only imported functions and a scratch sensitivity sweep.

### Hypothesis 1: Over-Estimation Due to Measured vs. Imputed Data

Initial hypothesis: Supercooling's entropy weight is inflated because the data
is "partly measured/partly imputed," making it noisier.

Finding: Hypothesis FALSE in its specific form. Cluster 0 has 9 survivors; only
1 (C22H46) is flagged as unknown supercooling, and that row is excluded from
entropy calculation entirely. The other 8 are real, measured values: {savE®
OM42: 1.0, RT47: 0.0, n-Docosane: 0.2, Lauric acid: 0.0, RT45HC: 0.0, savE®
OM46: 2.0, n-Tricosane: 2.6, RT50: −0.5} (mean 0.663 K, std 1.104 K). The
dispersion is real measured data, not an imputation artifact.

### Hypothesis 2: Tight Dispersion in Other Criteria

Hypothesis: Supercooling's entropy weight is inflated because "the other 7
criteria are unusually tight in Cluster 0."

Finding: Hypothesis FALSE. Coefficient of variation (CV) for the other criteria
in Cluster 0 is comparable to or higher than in Clusters 1/2 (e.g., thermal
conductivity CV 0.361 vs 0.299/0.222; vol_latent_heat 0.163 vs 0.092/0.131).
What actually differs: supercooling's own CV is highest in Cluster 0 (1.667 vs
1.014/1.182), which tracks its entropy weight directly.

### The Real Mechanism: Near-Zero-Ideal Values + Entropy Formula Pathology

What was actually found: Supercooling is a cost criterion whose physically
desirable value is near zero. Cluster 0 has three exact 0.0 K readings and one
slightly negative −0.5 K reading (measurement noise around zero) alongside two
real outliers (2.0, 2.6 K).

The entropy formula implementation clips negative values to 1e-12 before
computing Shannon entropy (a documented requirement of the formula) — this
treats the −0.5 K reading as near-total informational certainty ("almost zero,
so very sure") rather than "noise near zero." This combines with the
near-zero-mean CV inflation to produce an outsized entropy weight. This is a
known pathology of Shannon-entropy weighting on near-zero-ideal cost criteria,
not specific to this codebase.

### Confirmed: Physics Model Cannot Validate Supercooling

Independent verification: The physics model's own required-input list (in
physics_lib.py simulate_pcm_swh_year()) includes only Tm_C, latent_heat_kJ_kg,
density, Cp, thermal_conductivity — no supercooling parameter exists. The model
assumes ideal solid–liquid transition at Tm with no nucleation delay (Barqawi
2025, 3-phase formulation). This is a structural scope limitation, independent
of whether supercooling's entropy weight is over-estimated.

### Sensitivity Test: Force Supercooling Weight Down

Capped supercooling's blended weight to its AHP-prior value alone (0.075,
cluster-HSI-adjusted), renormalized the other 8 weights to sum to 1, compared
against already-computed simulation_rank in physics_validation_rajasthan.csv:

| Metric | Baseline (entropy-blended) | Capped (supercooling → AHP-only) | Effect |
| --- | --- | --- | --- |
| Kendall's W (method agreement) | 0.388 | 0.271 (worse) | Consensus drops further |
| Spearman ρ vs. Phase 7 simulation | −0.385 (p=0.31) | +0.561 (p=0.12) | Direction flips; not significant at n=9 |

Interpretation: Capping supercooling's weight flips MCDM-vs-physics agreement
from negative to positive (consistent with supercooling being over-weighted),
but makes Kendall's W (cross-MCDM-method agreement) worse, not better. As
supercooling's weight drops to AHP-only, Tm_fitness's weight rises to backfill,
and PROMETHEE's native V-shape Tm-handling diverges further from the other 3
methods' shared Gaussian score — revealing a second, independent disagreement
source (PROMETHEE vs. GRA/TOPSIS, not just supercooling vs. the other methods).
GRA remains the persistent structural outlier across both weight regimes.

### Verdict: Multiple, Overlapping Root Causes

Both (a) an entropy-weighting artifact and (b) a structural physics-model scope
limitation are real, and they don't fully overlap:

- (b) is unconditional: Wherever supercooling drives the MCDM ranking, physics
  validation cannot arbitrate that disagreement in principle. This must be
  stated plainly as a validator-scope limitation.
- (a) is real but different from initially hypothesized: Near-zero-clustered
  measured values plus the entropy formula's negative-value clipping, not
  measured-vs-imputed data, and not unusually tight dispersion elsewhere in
  Cluster 0.
- Cluster 0's low Kendall's W is not simply a symptom of inflated supercooling
  weight — removing that inflation lowers W further, exposing a second,
  independent disagreement source (PROMETHEE's Tm-handling vs. the other 3
  methods).
### Recommendation for Write-Up

State the physics-model scope limitation on supercooling explicitly, citing this
section. Consider (but flag explicitly) a variance-floor or CV-based
regularization for near-zero-ideal cost criteria in the entropy formula,
analogous to the existing <2-real-values→weight-0 guard — but note this will not
by itself raise Cluster 0's Kendall's W, since the PROMETHEE-vs-GRA/TOPSIS
structural disagreement is independent and needs its own investigation.

## Future Work

### 1. Alternative Penalty Mechanisms

Test formulations that model real supercooling physics:

- Nucleation-rate kinetics: dn/dt = A × exp(−ΔG/kT) where ΔG depends on
  subcooling; vary A or ΔG parameters
- Latent-heat release delay: Model as time-lag in phase 2 (melting plateau)
  rather than h_p reduction
- Hysteresis modeling: Account for energy losses in charge/discharge cycles due
  to subcooling/superheating
- Validate against published PCM discharge curves (literature or lab data for
  same candidates)
### 2. Increase PCM Mass

Retest penalty at PCM_MASS_KG = 100, 200 kg (from Phase 7's mass-sensitivity
sweep):

- If supercooling's effect emerges only when PCM is not tank-dominated, larger
  PCM mass may reveal the signal
- Alternative: test whether penalty helps at higher PCM fractions (not absolute
  mass, but fraction of total thermal capacitance)
### 3. Recalibrate MCDM Weights

Phase 6 supercooling weight (48–64%) is likely over-estimated. Suggested action:

- Re-run Phase 5/6 with reduced supercooling entropy weight (0.03–0.05 instead
  of 0.08)
- Observe whether Kendall's W (method agreement) and subsequent physics
  validation (Phase 7/8) improve
- Document the new weights and re-justify via stakeholder feedback (AHP
  elicitation) rather than entropy alone
### 4. Experimental Validation

Collect real discharge curves for surviving PCM candidates (literature or lab):

- Measure Tm, Tm_nucleation, supercooling_K for each candidate
- Compare observed discharge time-constants with model predictions (with and
  without penalty)
- Would clarify whether penalty formulation captures real physics or is
  fundamentally misguided
## Code Quality

- Toggleable penalty: Parameter SUPERCOOLING_PENALTY_K set externally; easy to
  disable (k=0.0) for baseline comparison
- Explicit field sourcing: Comments cite "Phase 5 feasibility filter" for
  supercooling_K, showing awareness of data provenance
- Calibration re-check: Medoid solar fractions re-computed at each k to ensure
  penalty doesn't destabilize
- Transparent reporting: All four k values tested and reported; no
  cherry-picking; both improvements and degradations documented
## Relationship to Thesis Write-Up

Recommendation: Report Phase 8 as a systematic investigation with a negative
finding, not a failure:

"Phase 8 implemented a supercooling penalty mechanism in the physics model,
proportional to each PCM's subcooling degree (Tm_C − Tm_freezing_C), and tested
whether correcting this apparent model gap would improve physics-MCDM ranking
agreement. Contrary to the hypothesis, the penalty worsened agreement in
Clusters 1 and 2, suggesting either: (a) the penalty formulation does not
capture supercooling's real mechanism in this system, or (b) supercooling's
entropy-weighted dominance in the MCDM (48–64%) is over-estimated relative to
its actual impact on annual solar fraction. This finding is valuable for future
refinement of either the physics model (alternative supercooling mechanisms,
higher PCM mass to overcome tank dominance) or the MCDM weights (re-elicitation
via AHP, downweighting supercooling if field validation confirms its small
effect)."

This framing demonstrates rigor (hypothesis was tested, result was reported
honestly, next steps are clear) without claiming false success.

────────────────────────────────────────

Status: Phase 8 complete. Supercooling penalty was correctly implemented and did
not break the model, but it made physics/MCDM agreement worse, not better. Root
cause is either penalty mechanism is incorrect, or supercooling's real effect is
much smaller than MCDM weighting (48–64%) suggests. Clear direction for future
work in both directions (alternative mechanisms, MCDM recalibration).

────────────────────────────────────────

## Phase 9 (Epilogue): Recommendation Cards

Script: 10_recommendation_cards_rajasthan.py (275 lines). Completed 2026-08-14
(re-run after Phases 5/6/7 updated).

### Purpose

Aggregate Phases 4/6/7 results into a final deliverable: one cluster-specific
recommendation card per climate regime, plus a cross-cluster summary table. Each
card carries the full provenance chain and caveats from upstream phases.

### What Each Card Contains

Per cluster:

- Cluster identity & signature: Two-tier climate signature (Tier 1 sun-events +
  Tier 2 daily integrals), derived targets (Tm_target, Tm_target_capped,
  L_required), system configuration assumptions
- Feasibility screening summary: Candidates entered vs. survived, κ-relaxation
  applied, per-constraint exclusion breakdown
- Top-3 PCM picks: With per-method ranks (TOPSIS/PROMETHEE/VIKOR/GRA), Monte
  Carlo inclusion probability, signed criterion-contribution decomposition
- Physics validation: Simulated annual solar fraction per Top-3 pick, Spearman ρ
  for cluster (showing validation result is NEGATIVE for this cluster)
- Explicit caveats section: Imputed PCM properties, relaxed feasibility κ,
  membership ambiguity (Kendall's W), database status, and crucially — the
  provisional-database flag (now stale pending L_required re-run)
### Cross-Phase Consistency Verification

10_recommendation_cards_rajasthan.py re-verifies cluster identity before writing
anything:

1. Fingerprint-stamp check: Compares upstream_cluster_profile_fingerprint
   against Phase 6's own fingerprint. If mismatched, raises SystemExit before
   computing anything.
1. Independent medoid cross-check: Recomputes medoid per cluster_id and verifies
   against cluster_profile_cards_rajasthan.md (from Phase 4) and
   physics_validation_rajasthan.csv (from Phase 7). Hard-fails naming exactly
   which cluster_id and file disagree if mismatch found.
This defense-in-depth was added because the GMM cluster-index instability bug
(fixed 2026-08-11) had already been caught once this session — Phase 9 ensures
it never silently recurs.

### Compute-Once, Reuse Principle

The cross-cluster summary table and the individual cards are rendered from the
same cluster_contexts dictionary (asserted explicitly in-code, not just
claimed). This satisfies the "compute once, reuse" requirement.

### Per-Criterion Contribution Decomposition

Phase 9 imports Phase 6 as a module and calls its deterministic
entropy_weights() and blended_weights() functions directly against the
fingerprint-verified survivor data — a re-run of Phase 6's weight formula on the
same inputs, not an independently-derived alternate calculation. Given the
fingerprint chain verified and no code changed between runs, agreement is
expected by construction. The meaningful independent checks are the
fingerprint-stamp check and medoid cross-check (both described above), which
genuinely could have failed and didn't.

### Output

recommendation_cards_rajasthan.md — Three cluster cards + cross-cluster summary
table, fully populated and ready for thesis inclusion. Every card explicitly
states physics validation does NOT confirm its Top-3 ordering for that cluster
(Spearman ρ values as in Phase 7 table, band=NEGATIVE in all three).

### Critical Caveat for Write-Up

All recommendation outputs are tagged with the provisional-database flag (55-row
database, 2026-08-12 expansion). The 2026-08-31 L_required correction has made
all Phase 5–9 outputs STALE — they must be regenerated before final submission.

# 12. 11_LITERATURE_MAPPING.md

Source path: /mnt/data/11_LITERATURE_MAPPING.md

# 17 — Literature Mapping

Documentation note (2026-09-02): Standalone concept files
10_TEMPORAL_PROCESSING.md and

11_SPATIAL_PROCESSING.md have been consolidated into 03_PHASE_1_AUDIT.md and

04_PHASE_2_AUDIT.md respectively, with full justification for each method. The
research gap

mapping has been moved into 00_MASTER_OVERVIEW.md under the new "Research gaps
addressed

(N1–N6 novelty mapping)" section. This file (17_LITERATURE_MAPPING.md) remains
the authoritative

reference for all methodology-component-to-source mappings.

## Method

Sources checked, in priority order: (1) PCM-Selection-ML-model/Sources/ — 21
full paper summaries

(the project's own curated, previously-read literature), (2) the framework doc's
own §15 IEEE

reference list, (3) references.bib (37 entries) and .claude/references.md (24
unique

ResearchRabbit entries + a duplicate of references.bib). Every citation below
was checked against

one of these three, not asserted from general training knowledge alone, except
where explicitly

marked "not independently verified in this project's bibliography" — those are
standard, correct

citations for well-known methods (e.g. Reda & Andreas SPA, Ineichen clear-sky)
that were not found in

this specific project's reference files during this audit and should be added
before formal

submission.

## Methodology-component → implementation → literature matrix

| Component | Implementation | Supporting source | Strength |
| --- | --- | --- | --- |
| ERA5 reanalysis as climate backbone | Phase 1–2 | Hersbach et al. (2020), QJRMS — per framework doc §15 | Strong (product-defining citation) |
| NASA POWER as cross-check | Phase 1–2 | NASA POWER project documentation — per framework doc §15 | Strong |
| Solar position (SPA) | pvlib, 00b/02 | Reda & Andreas (2004), Solar Energy 76(5) | Strong, but not confirmed present in references.bib/.claude/references.md — add before submission |
| Clear-sky model (Ineichen) | 02_combine_rajasthan.py | Ineichen & Perez (2002), Solar Energy 73(3) | Strong, not confirmed in project bib — add |
| pvlib software | throughout | Holmgren, Hansen & Mikofski (2018), JOSS 3(29) | Strong, per framework doc §15 |
| Humidity-stress index (HSI_sunrise) | signature_lib.py | Thom (1959), Weatherwise 12(2) — THI, correctly cited in-code | Strong, directly attributable |
| Night-discharge design basis (L_required) | 04_climate_signature_rajasthan.py | Avargani et al. (2021), J. Energy Storage | Strong, direct citation with a corrected units interpretation (see 05_PHASE_3_AUDIT.md) |
| Worst-month sizing cap (Tm_target_capped_C) | same | Durin et al. (2018), "Worst Month and Critical Period Methods..." | Strong, direct and appropriately applied |
| Field-evidence sanity check for the cap | same | Nahar (2003), tested at Jodhpur | Direct, present as a bare citation in .claude/references.md — needs a complete BibTeX entry |
| T_mains lag estimate | same | none — explicitly documented in-code as not derived from any published correlation | Weak / open gap — see recommendation below |
| GMM clustering, k-selection heuristics | 05_cluster_rajasthan.py | Building and Environment (2024) India climate-classification study (silhouette 0.21 vs −0.2 NBC); a 2026 thermal-comfort clustering study (mean silhouette 0.235) | Moderate — cited with enough specificity to be traceable but full BibTeX entries not located in this pass |
| External classification validation | 05_cluster_rajasthan.py | Beck et al. (2018), Scientific Data 5, DOI:10.1038/sdata.2018.214 (Köppen-Geiger) | Strong citation, now wired in for real (2026-08-11) — ARI=0.19/NMI=0.32 vs. GMM. NBC/ECBC remains unwired. |
| PCM candidate band (42–70°C) | Phase 5 | Framework doc Table 5, cross-referenced against Singh et al. (2025), Solar Energy Materials and Solar Cells 293 (states 40–70°C as the optimal SWH PCM band) | Strong, closely matching independent literature |
| PCM property values (RT-series validation) | PCM database | Martínez et al. (2025), Heliyon 11 — directly measures/validates RT54HC/RT55/RT64HC, the same product family in this project's database, and finds large literature-vs-measured discrepancies for some | Strong and directly relevant — should be cited as a caveat on manufacturer-datasheet trust, not just a property source |
| Gaussian Tm-fitness σ=4K | 08_mcdm_ranking_rajasthan.py | Framework doc §9.2 only — "not independently literature-calibrated," per the code's own docstring | Weak/self-sourced — state plainly, do not overclaim external validation |
| PROMETHEE II q/p thresholds | same | Framework doc §9.4 | Implementation-defined, documented as such |
| TOPSIS unit-test fixture | same | Oluah (2020) — 72.12% thermal-conductivity domination cited as a cautionary comparator | Direct, used correctly as both a regression-test anchor and an interpretive comparator |
| MCDM method family (TOPSIS/PROMETHEE/VIKOR/GRA) | same | No dedicated MCDM-methodology paper found cross-referenced in references.bib/.claude/references.md | Gap — these are standard, well-established methods, but a formal write-up should cite each method's originating paper (Hwang & Yoon 1981 for TOPSIS; Brans & Vincke 1985 for PROMETHEE; Opricovic 1998 for VIKOR; Deng 1982 for GRA) |
| PCM database imputation (MICE-style + RF + custom PMM-like donor blend) | PCM_data/01_preprocess.py | No dedicated imputation-methodology paper found in this project's bibliography | Gap — cite the general MICE framework (van Buuren & Groothuis-Oudshoorn 2011) and note explicitly that the donor-blend step is a project-original variant, not textbook PMM (see 07_PHASE_5_AUDIT.md) |
| Quantile mapping (bias correction) | 03b_agreement_analysis.py | No dedicated citation found in this project's bibliography | Gap — cite Cannon et al. (2015) or an equivalent standard reference |
| Phase 7 lumped-enthalpy ODE structure (3-phase pre-melt/melt/post-melt) | physics_lib.py | Barqawi, F. A. (2025), Muthanna J. Eng. Technol. 13(3):1-14, doi:10.52113/3/eng/mjet/2025-13-03/-1-14 | Strong — already in Sources/ (read in full pre-Phase-7), DOI independently re-verified this session, equations used directly (not paraphrased from memory) |
| Phase 7 model-class justification (lumped PCM-in-tank, the basis for TRNSYS Type 860) | same | Bony, J. & Citherlet, S. (2007), Energy and Buildings 39(9):1065-1072 | Strong — independently confirmed via web search this session (not previously in Sources/), cited for model-CLASS justification only, not claimed as a literal Type 860 replication |
| Phase 7 draw-profile SHAPE (two-peak, morning+evening) | same | ASHRAE Standard 90.2 §8.9.4 Table 8-4, built on Perlman & Mills (1985), ASHRAE Transactions | Partial/honest gap — the qualitative two-peak shape is real and cited, but the exact 24 published hourly fractions were not independently retrievable this session; physics_lib.py's own docstring flags this explicitly as a parametric reconstruction of the documented SHAPE, not a verbatim reproduction of the standard's table — do not cite specific hourly percentages from this pipeline as if reproducing that table |
| Phase 7 draw-total volume (300 kg/day) | same | Avargani et al. (2021) — same citation Phase 3 already uses for L_required_kJ_per_kg's 300 L/7h basis, reused as the FULL DAY total rather than a night-only ceiling (a different, explicitly stated use of the same cited figure) | Strong, cross-phase-consistent citation reuse |
| Phase 7 collector parameters (A_c, h_c, efficiency, PCM bed surface-to-volume ratio) | same | Barqawi (2025), same paper as above | Strong for the ORIGINAL values; recalibrated during Phase 7's own calibration pass (collector area, implicit loss coefficient) — recalibration reasoning documented in physics_lib.py's CALIBRATION section, not silently changed |

## Sources/ folder papers — relevance summary (21 papers read in full)

The 21 papers in Sources/ are overwhelmingly PCM-material / PCM-SWH-system /
AI-for-thermal-systems

domain literature (Abdellatif 2025, Al-Mamun 2023, Assareh 2023, Barghi 2026,
Barqawi 2025, Chen 2025,

Chopra 2023, Duraivel 2025, Eldokaishi 2022, Emami 2026, Ghodusinejad 2026,
Hamzat 2025, Kou 2025, Liu

2025, Mansouri 2025, Martínez 2025, Mohammed 2025, Odoi-Yorke 2025, Singh 2025,
Terfai 2025, Yan 2025)

— they substantiate this project's PCM-selection rationale, MCDM-in-PCM-context
precedents (Assareh

2023's TOPSIS/LINMAP/AHP; Chen 2025's GRA), and ML-for-thermal-systems framing
well. None of them are

methodology-support papers for ERA5/reanalysis handling, pvlib solar geometry,
quantile mapping, or

MCDM statistical foundations specifically — this is a real, confirmed gap
(searched by title

keyword against both references.bib and .claude/references.md; only two
incidental matches, Chen

2025 for "grey relational" and Chopra 2023 for "Monte Carlo," both already
counted above). Köppen

classification is now covered (Beck et al. 2018, above). Barqawi (2025), already
in this list as a

PCM-SWH domain paper, now ALSO serves as a direct methodology-support citation
for Phase 7's

lumped-enthalpy simulation equations — its equations are used directly, not just
cited for

framing.

## Recommendation

Before formal submission, add a dedicated "Methods & Tools" reference block
covering: Reda & Andreas

(2004), Ineichen & Perez (2002), Holmgren et al. (2018), Hwang & Yoon (1981),
Opricovic (1998), Deng

(1982), Brans & Vincke (1985), van Buuren & Groothuis-Oudshoorn (2011), and a
quantile-mapping

reference (e.g. Cannon et al. 2015) — none of these are currently in
references.bib or

.claude/references.md, and all are directly load-bearing for claims this
pipeline actually makes.

Also complete the bare Nahar (2003) citation note into a full BibTeX entry, and
add Durin et al.

(2018) and a formal Thom (1959) entry, since both are directly quoted/used in
code but not present in

either bibliography file. New, added 2026-08-11: add Bony & Citherlet (2007) —
the Phase 7

model-class justification, independently confirmed via web search this session
but not yet a formal

BibTeX entry in either bibliography file (Barqawi 2025 is already present via
Sources/).

# 13. PROJECT-SUMMARY.txt

Source path: /mnt/data/PROJECT-SUMMARY.txt

FYP Master Assistant — PCM Solar Water Heating Project

Project: Climate-Adaptive Intelligent Control and Optimization of PCM Thermal
Storage for Solar Water Heating Group 12 | Amrita School of Engineering | Guide:
Dr. T. Deepika Degree: B.Tech CSE (Final Year)

Who You Are

You are a senior research mentor, technical guide, and writing partner rolled
into one. You help with every dimension of this final-year project — from
figuring out which ML model to use, to drawing architecture diagrams, to writing
and refining the IEEE paper. You do not replace the student's thinking; you
sharpen it.

## You always

Explain things at B.Tech CSE level — clear, grounded, no unnecessary jargon

Keep suggestions implementable with Python + Raspberry Pi / Arduino / ESP32 +
open datasets

Cite the project's known papers first before suggesting new ones

Map answers back to the 5 research gaps whenever relevant

Flag if something needs expensive equipment, a large compute cluster, or is
outside undergraduate scope

Project Core (Always Keep in Mind)

What This Project Builds

An autonomous, AI-driven embedded system that:

Uses Phase Change Materials (PCMs) to store and release solar heat

Uses a trained ML/DRL model to adaptively control PCM charge/discharge/bypass
modes

Reads real-time sensor data (irradiance, temperature, flow rate, demand)

Selects the most suitable PCM based on current climate conditions

Runs on embedded hardware as a working prototype

Is validated against real Indian solar irradiance datasets

Core Claim

A real-time adaptive AI controller for PCM-based solar water heating outperforms
fixed/passive/rule-based strategies in hot-water availability, thermal
efficiency, and energy savings — especially under variable Indian climate
conditions.

Key Contributions

Real-time adaptive DRL/ML control of PCM charge/discharge

Climate-aware PCM selection using live sensor data

Fully embedded hardware prototype (RPi/Arduino + Python)

Validated against Indian open-access solar datasets

The 5 Research Gaps (Always Link Answers Here)

RG1: No real-time adaptive control in existing PCM-SWH systems

RG2: No integrated PCM–AI–hardware prototype exists

RG3: Poor alignment with real household usage/demand patterns

RG4: Limited real-world experimental validation (mostly simulations)

RG5: No predictive optimization under climatic uncertainty

Known Project References (Cite These First)

PCM-Focused: Hamza 2025, Rathore 2024, Martinez 2025, Abdellatif 2025 AI/ML in
Thermal Systems: Mohammed 2025, Nems 2025, Liu 2025, Yan 2025 AI/ML in SWH:
Odoi-Yorke 2025, Eldokaishi 2022, Assareh 2023, Muthanna 2025 Climate + PCM-SWH:
Singh 2025, Kou 2025, Emami 2026, Chen 2025 Broader Methods: Rajamurugu 2025,
Terfai 2025, Barghi 2025, Ghodusinejad 2025

Hardware & Software Constraints

Hardware: Raspberry Pi 4 / Arduino Mega / ESP32, DS18B20 temperature sensors,
pyranometer or LDR for irradiance, solenoid/pump for flow control

Software: Python 3.x, Stable-Baselines3 (for DRL), Scikit-learn / XGBoost (for
ML), TensorFlow Lite (for embedded inference), Matplotlib/Seaborn for plots

Datasets: ISRO Solar Calculator, Global Solar Atlas (India), Renewables.ninja,
NITI Aayog India Energy Dashboard

PCMs: Rubitherm RT series (RT35, RT42, RT58, RT64HC) or PLUSS OM series (OM35,
OM37, OM42)

Simulation: Python grey-box thermal model before hardware deployment

## Workflow 1 — ML / Algorithm Selection Guide

When asked "which ML model should I use?" or "what algorithm fits my problem?",
follow this decision process:

Step 1: Identify the Sub-Problem

## Ask which part of the system the student is building

Sub-Problem What It Needs

PCM selection from sensor inputs Classification or ranking model

Predicting PCM melting/charging time Regression model

Forecasting solar irradiance Time-series forecasting

Real-time charge/discharge control Reinforcement Learning or MPC

Detecting anomalies in sensor data Anomaly detection

Estimating thermal state of PCM Regression or surrogate model

Step 2: Recommend With Reasoning

Always give 2–3 options ranked by student-implementability, with honest
tradeoffs:

## Output format

## ML Recommendation: [Sub-Problem]

### Recommended: [Model Name]

Why it fits: [1–2 sentences linking to the sub-problem]

Student-implementability: Easy / Medium / Hard

Library: [e.g., Stable-Baselines3, Scikit-learn, XGBoost]

Research backing: [cite 1–2 project papers that used this]

Tradeoff: [what it does less well]

### Alternative: [Model Name]

Why consider it: [brief reason]

When to prefer it over recommended: [specific condition]

### What NOT to use here and why:

- [Model]: [reason it's a poor fit]
Quick Reference — Model Shortlist for This Project

For control (charge/discharge decisions):

PPO (Proximal Policy Optimization) — best starting point for students, stable
training, handles discrete/continuous actions [Emami 2026, Sivaraj 2023]

DDPG (Deep Deterministic Policy Gradient) — better for continuous action spaces,
harder to tune [Emami 2026]

MPC (Model Predictive Control) — interpretable, great if you have a good thermal
model [Terfai 2025]

Rule-based baseline — always implement this first as your comparison baseline

## For PCM performance prediction

XGBoost — top performer for tabular PCM property data [Yan 2025]

Random Forest — robust, easy to interpret, good baseline [Rajamurugu 2025]

SVR (Support Vector Regression) — good for small datasets [Rajamurugu 2025]

MLP / ANN — works well with more data, used widely in SWH context [Liu 2025,
Eldokaishi 2022]

## For solar irradiance forecasting

LSTM — strong for time-series, widely used [Ghodusinejad 2025]

Hybrid CNN-LSTM — state of the art but complex [Ghodusinejad 2025]

SARIMA — simpler statistical baseline, good for comparison

For PCM selection (classification):

Decision Tree / Random Forest — interpretable, easy to validate with domain
knowledge

Multi-label classifier if selecting multiple compatible PCMs simultaneously

## Workflow 2 — Architecture Diagram Design

## When asked to create or improve a system architecture diagram

Step 1: Clarify Scope

## Identify which level of architecture is needed

System-level — full pipeline from solar panel → PCM tank → AI controller → hot
water output

Software-level — data flow between sensing, preprocessing, model inference, and
actuation

ML pipeline — data → features → model → output → feedback loop

Hardware-level — physical wiring/component layout

Step 2: Define the Standard Blocks for This Project

The architecture always contains these layers, in order:

[INPUT LAYER]

Solar Panel + Collector → Sensors (irradiance, temp, flow rate, demand)

→ Weather API / Historical Dataset

[PROCESSING LAYER]

Sensor Preprocessing → Feature Engineering → State Vector Construction

[INTELLIGENCE LAYER]

PCM Selector (ML classifier) → DRL/ML Controller (PPO/DDPG/MPC)

→ Thermal Simulation Model (grey-box)

[ACTUATION LAYER]

Pump/Valve Controller → PCM Tank (charge / discharge / bypass mode)

[OUTPUT LAYER]

Hot Water Output → Feedback to Controller (closed-loop)

→ Logging + Performance Monitoring Dashboard

Step 3: Output a Described Diagram

When drawing or describing the diagram, label:

Every data flow arrow with what it carries (e.g., "temperature readings",
"control action", "state vector")

Every module with its technology (e.g., "PPO Agent — Stable-Baselines3")

Color-code by layer (inputs = one color, AI = another, hardware = another)

Highlight the feedback loop — this is the novel part that closes the control
loop

Step 4: Explain Design Decisions

After presenting the diagram, always explain:

Why each block exists (link to a research gap)

What happens if a block fails (robustness consideration)

What the student needs to implement vs. what comes pre-built (library/tool)

## Workflow 3 — Hypothesis & Problem Framing

When asked to form a hypothesis, research question, or problem statement:

Hypothesis Template for This Project

A good hypothesis is: specific, testable, falsifiable, and linked to a gap.

## Format

If [proposed intervention], then [measurable outcome],

compared to [baseline], because [theoretical reason].

Example for this project: "If a PPO-based adaptive controller manages PCM
charge/discharge decisions in real time using live sensor inputs, then hot-water
availability will increase by at least 20% and charging efficiency will improve
by at least 15%, compared to a fixed threshold rule-based controller, because
the agent learns to anticipate demand patterns and solar variability rather than
reacting after the fact."

Research Questions Template

Structure as: one primary RQ + 3–4 sub-questions

Primary RQ: Can an AI-driven adaptive controller optimize PCM thermal storage
performance in a solar water heating system across variable Indian climate
conditions?

## Sub-questions

RQ1: Which PCM properties and climate features are most predictive of optimal
storage performance? (→ RG5)

RQ2: How does a PPO/DDPG controller compare to rule-based and MPC baselines in
hot-water availability? (→ RG1)

RQ3: Can a trained model generalize across different Indian climate zones
(coastal, arid, temperate)? (→ RG5)

RQ4: What is the minimum sensor configuration needed for reliable adaptive
control? (→ RG2)

## Workflow 4 — Methodology Design

## When asked to design or refine the methodology

Standard Methodology Structure for This Project

## Phase 1 — Data Collection & PCM Characterization

Gather thermophysical properties for selected PCMs (Rubitherm RT / PLUSS OM
series)

Download solar irradiance + ambient temperature data for 3+ Indian cities

Define typical household hot-water demand profiles (morning/evening peaks)

Output: Clean dataset with PCM properties + climate inputs + demand profiles

Phase 2 — Thermal Simulation (Grey-Box Model)

Build a Python-based thermal model of the PCM tank using enthalpy-porosity
method

Inputs: irradiance, ambient temp, flow rate, PCM properties

Outputs: PCM state (solid/mushy/liquid), stored energy, water outlet temperature

Validate against published experimental data from literature

Purpose: Generate training environment for the RL agent without needing physical
hardware first

## Phase 3 — ML Model for PCM Selection

Features: ambient temp, irradiance, demand forecast, time of day, season

Target: best-suited PCM index (from shortlist of 5–8 PCMs)

Model: Random Forest or XGBoost classifier

Validation: k-fold cross-validation, confusion matrix

## Phase 4 — RL Controller Training

Environment: Phase 2 thermal simulation (wrapped as OpenAI Gym environment)

State space: [irradiance, PCM temp, water temp, demand, time of day]

Action space: [charge, discharge, bypass] (discrete) or flow rate (continuous)

Reward function: maximize hot-water availability + penalize energy waste +
penalize under-delivery

Algorithm: PPO (start here) → compare with DDPG and rule-based baseline

Training: 50,000–200,000 timesteps on simulation

## Phase 5 — Hardware Prototype

Deploy trained model (TensorFlow Lite / ONNX) onto Raspberry Pi

Connect sensors: DS18B20 (temp), LDR/pyranometer (irradiance), flow meter

Implement actuation: pump/solenoid valve control via GPIO

Run closed-loop test: sensor → inference → actuation → feedback

## Phase 6 — Evaluation

Metrics: hot-water availability (hours/day), charging efficiency (%), thermal
loss (%), COP

Comparison: proposed AI controller vs. rule-based baseline vs. passive (no
control)

Climate test: run across at least 2 climate profiles (e.g., Coimbatore vs.
Jaisalmer)

Statistical validation: mean ± std, paired t-test if comparing two controllers

Workflow 5 — Research + Citations

## When asked to find evidence for a claim or expand a section

## Process

Check project's known references first

Search for credible recent sources (2022–2026 preferred)

Prioritize IEEE, Elsevier, Springer, Nature Energy, Applied Thermal Engineering,
Solar Energy journals

Extract key facts and map them to where they fit in the paper

## Output format

## Research: [Topic]

### Key Findings

1. [Finding] — [Author, Year, Journal]

2. [Finding] — [Author, Year, Journal]

### Where to Use This

- Finding 1 → Section IV (Methodology) to justify [design choice]
- Finding 2 → Section I (Introduction) to strengthen motivation
### Recommended New Papers (if project refs insufficient)

- [Author, Year] — [Title snippet] — [why it's relevant]
### IEEE Citations

[1] A. Author, "Title," Journal, vol. X, no. Y, pp. Z–Z, Year.

## Workflow 6 — Paper Writing & Feedback

IEEE Paper Structure

Section Word Target Purpose

Abstract 150–250 words Problem / Method / Result / Contribution

I. Introduction 400–600 words Motivation, gaps, contributions (numbered)

II. Literature Review 600–900 words Grouped by theme, gap-justified

III. Proposed Methodology 800–1200 words Architecture, model, hardware,
algorithm

IV. Dataset & Experimental Setup 400–600 words PCM data, solar data, hardware
specs

V. Results & Discussion 600–900 words Tables, graphs, baseline comparison

VI. Conclusion 200–350 words Contributions summary + future work

## Drafting rules

Formal third-person: "The proposed system...", "This paper presents..."

Define all abbreviations on first use: PCM, SWH, DRL, TES, GHI, COP

One idea per sentence

Every claim needs a citation or experimental evidence

## Section Feedback Format

## Feedback: [Section Name]

### What's Working

- [Specific strength]
### Suggestions

**Clarity:** [Issue] → [Fix]

**Evidence:** "[Claim]" — needs a citation

**Technical precision:** [vague term] → [more precise alternative]

### One Suggested Edit

Original: [their sentence]

Suggested: [stronger version]

Why: [brief reason]

### Questions to Consider

- [Question that might unlock a stronger argument]
Final Pre-Submission Checklist

## Full Paper Review

### Structure & Flow

- Abstract: covers problem/method/result/contribution?
- Introduction: clear problem + numbered contributions?
- Lit Review: grouped by theme, gaps justified?
- Methodology: reproducible from description alone?
- Results: baseline comparison present?
- Conclusion: contributions + future work?
### IEEE Compliance

- [ ] Figures captioned (Fig. 1, Fig. 2...)
- [ ] Tables captioned (Table I, Table II...)
- [ ] IEEE reference format throughout
- [ ] Abbreviations defined on first use
- [ ] Section headings numbered (I., II., III...)
### Top 3 Priorities Before Submission

1. [Most critical]

2. [Second priority]

3. [Nice to have]

## Workflow 7 — Objective Guidance & Project Milestones

When the student asks "how do I complete Objective X?" or "what should I do
next?", map them to this:

Objective → Task → Output Mapping

Objective Concrete Tasks Deliverable

O1: PCM selection using sensor data Build RF/XGBoost classifier, train on PCM
property dataset, test on held-out climate profiles Trained model + confusion
matrix + accuracy report

O2: Real-time adaptive control Build Gym simulation env, train PPO agent,
compare vs rule-based baseline Training curve + performance comparison table

O3: Embedded standalone controller Export model to TFLite/ONNX, deploy on RPi,
test sensor→inference→actuation loop Working prototype demo + latency benchmark

O4: Working prototype + validation Run closed-loop test, log results, compare
against passive baseline on 2+ climate profiles Results table + graphs +
statistical test

Milestone Timeline (Typical)

Review What Should Be Done

PR1 (done) Literature review, research gaps, objectives, problem statement,
methodology overview

PR2 Thermal simulation working, PCM dataset collected, baseline rule-based
controller coded, RL environment set up

PR3 RL agent trained, PCM classifier trained, preliminary results, hardware
prototype in progress

Final Full prototype working, all results collected, paper drafted, comparisons
complete

Voice Preservation Rules

Read the student's writing before suggesting edits — match their rhythm and
vocabulary

Offer options: "here's one way to phrase this" — never directives

Check in: "Does this match what you meant?" / "Too formal?"

If a suggestion is rejected, accept it gracefully and move on

Never rewrite a full paragraph unless explicitly asked

General Principles

Be specific — point to the exact sentence, line, or decision that needs work

Prioritize ruthlessly — top 3 things first, not a list of 10

Keep it implementable — flag anything beyond undergraduate lab scope immediately

Celebrate progress — this is hard work, acknowledge wins genuinely

When multiple approaches exist, always present them as options with tradeoffs,
not a single "correct" answer

# 14. OBJECTIVE-1-—-IMPLEMENTATION-PLAN.txt

Source path: /mnt/data/OBJECTIVE-1-—-IMPLEMENTATION-PLAN.txt

**Climate-Region-Aware PCM Recommendation Framework**

*Clustering multi-year meteorological data and identifying Top-2 / Top-3 PCM
candidates per climatic regime by multi-criteria decision-making, validated
against a physics-based thermal model*

**Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage
for Solar Water Heating**

Group 12 · B.Tech Computer Science & Engineering (Final Year)

Amrita School of Engineering · Guide: Dr. T. Deepika

**Document version 3.0**

*Supersedes v2.0. Revised to match the data-collection pipeline as actually
built: four contrasting Indian states, population-weighted ERA5 grid points
aligned to the ERA5 grid origin, and sun-event-aligned temporal sampling with an
independent NASA POWER cross-check. All changes from v2.0 are listed in §0.*

# 0. What Changed in Version 3.0

Version 2.0 was written before data collection began and assumed an all-India,
city-point, full-hourly ERA5 dataset. The pipeline that was actually built — on
the guide's direction — differs in three structural ways: it covers four states
rather than the whole country, it samples population-weighted grid cells rather
than named cities, and it samples three astronomically computed sun-event
instants per day rather than all 24 hours. Two of those three changes improve
the design. The third requires a specific, non-optional repair, described below
and in §6.

| **Item** | **v2.0 (planned)** | **v3.0 (as built, plus required repairs)** |

| --- | --- | --- |

| **Geographic scope** | ~30 named cities spanning all of India | Four states: Rajasthan, Assam, Tamil Nadu, Uttarakhand. Depth over breadth — four contrasting climate families sampled densely, rather than all of India sampled thinly. Stronger for validation, narrower for generalisation claims (§1.3). |

| **Sampling unit** | One ERA5 grid point per named city | Population-weighted 0.25° cells aligned to the ERA5 grid origin, retaining the minimal set covering ~87.5 % of each state’s population. This is a genuine methodological improvement and becomes novelty claim N6. |

| **Temporal sampling** | All 24 hours, 10 years | Sunrise, solar noon and sunset per point per day, 2016–2025, computed by the pvlib SPA algorithm. Physically well aligned to the PCM charge–discharge cycle — but insufficient on its own for daily-integral indices. |

| **Second data source** | CERES satellite radiometry (planned, not obtained) | NASA POWER hourly at identical points and instants, already downloaded. This is a better cross-check than planned: two independent estimates of the same quantity at the same instant and location. |

| **Bias correction** | Quantile mapping of ERA5 solar against CERES | ERA5-versus-POWER agreement analysis at matched instants first (MBE, RMSE, correlation per season per point). Quantile mapping applied only if a systematic seasonal bias is demonstrated. Fixed-weight blending of the two remains rejected. |

| **Climate signature** | 18 indices assuming full hourly input | Restructured into two tiers (§6.2). Tier 1 sun-event indices come from the merged CSV. Tier 2 daily-integral indices are recomputed from the NASA POWER hourly cache already on disk. No new ERA5 download is required. |

| **Elevation** | elev_proxy index in the signature | REQUIRED REPAIR. The pipeline assumes a flat 300 m for solar geometry. That is materially wrong for Uttarakhand, which spans roughly 200 m to over 7,000 m. Per-point elevation must be attached from ERA5 surface geopotential or an SRTM DEM, and solar geometry recomputed for the mountain points. |

| **Clustering framing** | Discover the climate regions of India | Discover intra- and inter-state regimes across four contrasting states. State identity becomes an external validation label rather than a result — recovering the four state boundaries alone would be a trivial finding (§7.1). |

| **Expected k** | 5–7, checked against 5–6 NBC zones | 6–10. Four states with expected internal splitting: arid west versus semi-arid east in Rajasthan, terai versus mid-hills versus high Himalaya in Uttarakhand, coast versus interior versus Nilgiris in Tamil Nadu, valley versus hills in Assam. |

| **Novelty claims** | N1–N5 | N6 added: population-weighted, deployment-relevant regionalisation. Regimes are weighted by where people actually live, therefore by where solar water heaters would actually be installed. |

| **Timeline** | 16 weeks from zero | 12 weeks remaining. Phases 1 and 2 are substantially complete; the schedule is re-baselined in §12. |

*Table 0. Change log from v2.0 to v3.0. Two structural changes improve the
design; the elevation assumption and the sun-event-only merged output require
repair before Phase 3 can proceed.*

| **The good news, stated plainly. **The sampling design is better than the one this plan originally specified. Population weighting means the regimes describe where installations would actually go, not where grid cells happen to fall. Sun-event sampling is not an arbitrary subsample — sunrise is the coldest instant and therefore the solidification test, solar noon is the peak charging condition, and sunset is the start of the discharge period. Those are precisely the three instants a PCM store cares about. Say this explicitly in the paper; a reviewer will otherwise read three-samples-per-day as a shortcut rather than a design. |

| --- |

| **The repair that cannot be skipped. **The merged CSV keeps only the three sun-event rows per point-day. Several signature indices — daily GHI integral, true diurnal temperature range, heating and cooling degree days, cloudy-day fraction, consecutive-cloudy-day index — cannot be computed from three instantaneous samples. They do not need a new ERA5 request: the NASA POWER raw cache at data/raw/nasapower/power_{point_id}_{year}.json already holds the full hourly series for every point and year. The merge step subsets it to sun events and discards the remaining 8,757 hours. Recovering the Tier 2 indices is a read over files already on disk (§6.2). |

| --- |

# Contents

*(In Word: Ctrl+A then F9, or right-click the table below and choose **"**Update
Field**"**, to populate page numbers.)*

# 1. Scope and Objective Decomposition

## 1.1 The objective, restated precisely

*Develop a climate-region-aware recommendation framework that clusters ten years
of population-weighted meteorological data across four contrasting Indian
states, and for each discovered climatic regime identifies the Top-2/Top-3
suitable PCM candidates for solar domestic hot water storage using
multi-criteria decision-making, with an explicit confidence measure and
independent physics-based validation of the resulting ranking.*

## Decomposed into four verifiable sub-goals

- SG1 — Assemble a ten-year, multi-point meteorological dataset for Rajasthan,
  Assam, Tamil Nadu and Uttarakhand, sampled at population-weighted locations
  and at instants aligned to the PCM charge–discharge cycle, with an independent
  second source for cross-validation.
- SG2 — Reduce each location’s ten-year record to a compact, physically
  meaningful climate signature vector, then cluster those signatures into a
  small number of climate regimes spanning and subdividing the four states.
- SG3 — For each regime, filter a PCM database to a feasible candidate set and
  rank it by MCDM, returning an ordered Top-3 with a confidence measure rather
  than a single winner.
- SG4 — Validate the ranking against an independent physics-based thermal
  simulation, not against the MCDM’s own scores.
## 1.2 What is deliberately out of scope

This objective is a regional, offline selector. It answers “what PCM should be
specified for a system installed in this climate regime?” — not “what should the
controller do tomorrow?” and not “what should the valve do right now?”.

**Scope discipline. **The regional selector is the foundation: it narrows a
40–60 PCM database to 2–3 candidates per regime, and only those candidates ever
enter the day-ahead layer or the DRL controller. Building the regional layer
first means the forecasting objective inherits a validated shortlist instead of
guessing one. Fusing the layers now makes it impossible to attribute any result
to either mechanism.

## Explicitly out of scope for Objective 1

- Time-series forecasting of any variable. Historical climatology is the input
  here, not a forecast.
- Real-time control, charge/discharge policy, DRL. That is the control
  objective.
- Hardware, sensors, embedded deployment.
- PCM synthesis or experimental characterisation. Published thermophysical
  property data is consumed, not generated.
- Field trials. Recorded as future work in §14.
- Extension to the remaining Indian states. The framework is state-agnostic by
  construction, and §14.1 records the extension as future work, but no claim
  about states outside the four should appear in the paper.
## 1.3 Why these four states — and how to defend the choice

The obvious reviewer question is why not all of India. The answer is that these
four states are not an arbitrary subset: they span the widest climatic contrast
available within India while allowing dense sampling inside each.

| **State** | **Climate family** | **What it contributes to the study** |

| --- | --- | --- |

| **Rajasthan** | Hot-dry / arid to semi-arid (Thar desert to eastern Aravalli) | Highest solar availability, largest diurnal range, lowest cloud persistence. The regime where charging is never the constraint and cycling stress is highest. Expected internal split: arid west versus semi-arid east. |

| **Assam** | Warm-humid, monsoon-dominated (Brahmaputra valley) | Lowest clearness index, longest consecutive-cloudy-day runs, highest humidity stress. The regime that sets the latent-heat floor and drives salt-hydrate corrosion exclusion. Expected internal split: valley versus surrounding hills. |

| **Tamil Nadu** | Warm-humid coastal and hot semi-arid interior, with a montane exception | A north-east monsoon regime that is out of phase with the rest of the country — valuable because it breaks any assumption that Indian seasonality is uniform. Expected internal split: coastal, interior dry, and the Nilgiris. |

| **Uttarakhand** | Cold / temperate, extreme elevation gradient | The only cold regime in the study, and the one where mains water temperature is lowest and the latent-heat requirement therefore highest. Expected internal split: terai plains, mid-hills, high Himalaya. Also the state where the flat-300 m elevation assumption does the most damage (§4.3). |

*Table 1. The four study states. Between them they cover four of the five or six
NBC/ECBC climate zones — hot-dry, warm-humid, composite and cold — which is what
makes a four-state study defensible rather than merely convenient.*

**State the limitation honestly. **The one NBC zone not represented is temperate
as classified in the Indian standard (the Bengaluru plateau type). Say so in the
limitations section rather than letting a reviewer find it. Four zones out of
five or six, sampled densely and population-weighted, is a stronger dataset than
six zones sampled at one city each.

## 1.4 Deliverables

| **ID** | **Deliverable** | **Status** |

| --- | --- | --- |

| **D1** | Ten-year meteorological dataset for four states at population-weighted points, sun-event aligned, with matched NASA POWER cross-check | Substantially complete — pipeline built and run |

| **D1b** | Tier 2 daily-aggregate table recovered from the NASA POWER hourly cache; per-point elevation attached | REQUIRED — see §4.3 |

| **D2** | PCM property database, 40–60 candidates in the corrected 42–70 °C band, every row cited | Outstanding |

| **D3** | Climate signature feature matrix, one row per point, two-tier index set | Outstanding |

| **D4** | Cluster model with k-selection evidence, cross-checked against state identity, Köppen–Geiger and NBC/ECBC zones | Outstanding |

| **D5** | MCDM ranking engine: entropy+AHP weights, four ranking methods (+ optional CoCoSo), aggregation, Monte Carlo | Outstanding |

| **D6** | Top-3 PCM table per regime with consensus score, stability percentage and population coverage | Outstanding — the headline result |

| **D7** | Physics validation: simulated annual solar fraction per feasible PCM versus MCDM rank, Spearman ρ | Outstanding |

| **D8** | IEEE conference paper, 6–8 pages | Outstanding |

*Table 2. Deliverables and current status. D6 is the headline result; D7 is what
makes D6 defensible.*

# 2. Response to the Critical Review

The external review claimed four errors. Each was checked against primary
sources. Summary verdicts first; reasoning follows.

| **Claimed error** | **Verdict** | **Action taken** |

| --- | --- | --- |

| **#1 Undefined clustering methodology — use K-Means + STL, silhouette ****>**** 0.75** | Partially correct | Algorithm and validation are now stated explicitly (they were implicit in v1.0). The specific prescriptions are rejected: GMM retained over K-Means, STL rejected, silhouette target set at a realistic 0.15–0.35. |

| **#2 Undefined MCDM method — adopt AHP + CoCoSo** | Wrong as a replacement | Method was already fully specified in v1.0. Four-method consensus retained. CoCoSo added as an optional fifth ranker. The review's supporting statistic could not be traced to any source. |

| **#3 Ambiguous PCM criteria — hot-arid 35–40 °C, tropical humid 28–32 °C** | Wrong for this application | These are building passive-cooling targets. Replaced with an SWH-specific melting-temperature rule anchored to delivery and collector temperatures (§6.3). The review did, however, correctly expose a sign error in the v1.0 rule. |

| **#4 No validation strategy — EnergyPlus + Monte Carlo + field trial** | Partially correct | v1.0 already had physics validation. EnergyPlus rejected as technically incapable of the task; TRNSYS Type 860 named as the optional cross-check. Monte Carlo retained at 5,000. Field trial moved to future work. |

*Table 2. Verdicts on the four claimed errors.*

## 2.1 Error #1 — clustering methodology

**What is correct. **Algorithm choice does materially affect cluster structure,
and Objective 1 should state its algorithm, features, and validation criteria
explicitly rather than saying "clusters data". v2.0 does so in §7.

**What is wrong — the silhouette target. **The review cites an Indian urban
rainfall clustering study reporting DBSCAN 0.82, Spectral 0.80, K-Means 0.70,
OPTICS 0.44, and converts this into a ">0.75" acceptance bar. That study
clusters daily gridded rainfall event vectors, where dense well-separated groups
occur naturally. It is a different object from a multi-year climate signature
vector per city. For genuine climate-zone clustering over India, published
silhouettes are far lower: a criteria-based reclassification of Indian climate
zones reports a silhouette of 0.21 against −0.2 for the current NBC
classification, peaking at approximately 0.3 at k = 6 — and that result was
considered a success because it outperformed the official map. An Indian
thermal-comfort clustering study reports an average silhouette of 0.235.

| **Why the ****>****0.75 bar is dangerous. **India's climate lies on a continuous gradient. There is no partition of Indian cities into climate regimes that produces a silhouette above 0.75 without either collapsing k to two or three, or selecting features specifically to manufacture separation. Adopting that threshold would push this project toward exactly the kind of result-shaping it is trying to avoid. Silhouette should be read against a null benchmark and alongside BIC, Davies–Bouldin and external agreement, not against an idealised absolute. |

| --- |

**What is wrong — STL. **STL (seasonal-trend decomposition by LOESS) separates
trend, seasonal and residual components of a raw time series. The clustering
object here is an aggregated 18-index signature per site; seasonality and
monsoon behaviour are already captured by named indices (seasonality,
monsoon_index, CCI, cloudy_frac). STL adds a preprocessing stage that
reintroduces the time-series representation the signature abstraction
deliberately removed. It is appropriate for forecasting tasks, which this is
not.

**K-Means versus GMM. **The literature consistently finds K-Means produces
crisper, higher-silhouette partitions on spherical clusters, while GMM better
represents overlapping, non-spherical, gradient structure and yields soft
membership probabilities. For a country where a city may sit genuinely between
two regimes, soft membership is a feature, not a compromise: a site that is 60 %
hot-arid and 40 % composite can receive a membership-weighted PCM
recommendation. GMM is retained. K-Means is still fitted for k = 2…10 as a
reported comparison, which also answers the reviewer's concern directly.

## 2.2 Error #2 — MCDM method

**The supporting statistic could not be traced. **The review states that "CoCoSo
demonstrated 10.73 % rank stability in Monte Carlo tests vs 0 % for TOPSIS under
uncertainty". No located source reports this. The nearest real statements in the
literature are a review noting roughly a 9 % rank-stability improvement over a
closest competitor under a 10,000-trial Monte Carlo, and a study finding CoCoSo
more stable with respect to changes of alternatives than of criteria —
qualitative, and not the quoted figure. Read literally, "TOPSIS is 0 % stable"
is false; TOPSIS is repeatedly shown stable under moderate perturbation. This
figure should not be cited.

**CoCoSo does not solve the target-based criterion. **CoCoSo normalisation is
strictly benefit/cost. Melting temperature in this project is target-based —
closer to the optimum is better, in both directions. To feed Tm into CoCoSo at
all, the Gaussian fitness transform of §9.2 must be applied first. The
transform, not CoCoSo, is what handles the physics. PROMETHEE II remains the
method that expresses this most naturally, through indifference and preference
thresholds with direct engineering meaning.

**Replacing four methods with one is a regression. **The defence of this
framework against method-induced bias is the agreement statistic across
independent ranking logics — Kendall's W across TOPSIS, PROMETHEE II, VIKOR and
GRA, with Borda and Copeland aggregation and disagreement reported rather than
hidden. A single method, however modern, removes that evidence. CoCoSo is
nonetheless a legitimate, current, citable method whose hybrid compensatory /
non-compensatory aggregation is genuinely different in kind from the other four,
so it is worth adding as a fifth ranker and reporting whether it changes the
consensus. If it does not, that is a robustness statement.

## 2.3 Error #3 — PCM selection criteria

**The evidence base is building passive cooling. **The review's OM35 / OM37 /
n-eicosane recommendation and its "30–40 °C ambient" framing trace to a roof
passive-cooling study in Rupnagar using spherical macro-encapsulated modules in
an RC roof, and to a prior MCDM study screening 26 PCMs for building space
cooling. Those systems operate across a 27–43 °C diurnal ambient cycle. This
project delivers water at approximately 50 °C from a storage tank. The PCM
families, the melting range and the selection rule are all different. The
proposed profiles — hot-arid 35–40 °C, tropical humid 28–32 °C — would produce a
PCM that cannot deliver usable hot water anywhere in India.

**The ****"****40 %****"**** claim is unverified. **The assertion that
melting-point alignment within ±3 °C yields a 40 % performance improvement, or
that alignment is "40 % more important" than latent heat, could not be traced to
a primary source in a solar hot water context. The figure appears in the
literature in unrelated forms — PV thermal-storage-potential variance,
desalination yield gains. It should not be cited. The defensible statement is
weaker and sufficient: melting temperature must lie between the mains inlet and
the collector delivery temperature, and mismatch degrades usable latent
capacity, with no universal constant.

**But the review exposed a real error. **v1.0 defined Tm_target = T_delivery −
ΔT_approach, giving 42–45 °C. That sign is wrong for discharge. During discharge
the PCM is the heat source and the water the sink, so the PCM must sit above the
delivery temperature by the approach temperature, not below it. The corrected
rule and its literature support are in §6.3. This is the single most valuable
outcome of the review.

## 2.4 Error #4 — validation strategy

**EnergyPlus cannot do this. **EnergyPlus models PCM through
MaterialProperty:PhaseChange / PhaseChangeHysteresis with the conduction finite
difference algorithm — but only as solid conduction layers inside building
surfaces. It models solar water heaters through WaterHeater:Mixed /
WaterHeater:Stratified on a plant loop. There is no supported path to place a
latent-heat PCM inside the water tank node network; the two capabilities live in
different modelling domains. Specifying EnergyPlus here would produce either a
building-envelope result mislabelled as a hot water result, or nothing at all.

**What is appropriate. **A Python grey-box lumped enthalpy tank model,
calibrated against published experimental benchmarks, remains the primary tool —
it is transparent, every line is explicable in a viva, and it integrates
directly with the ranking pipeline. TRNSYS Type 860, a PCM-in-tank component
built on the Type 60 water tank using the enthalpy method with support for
encapsulation geometry, hysteresis and supercooling, is the correct optional
cross-check if a licence is available.

**Monte Carlo count and field trial. **Rank-inclusion probabilities converge
well before 5,000 draws; many published MCDM stability studies use 1,000. Moving
to 10,000 is not a material improvement and 5,000 is retained. The 12–24 month
field trial is not feasible within this project and is recorded as future work.

# 3. Closest Prior Work and Novelty Position

**Read this before you start. **A 2025 paper in Energies, “Comparative Framework
for Climate-Responsive Selection of Phase Change Materials in Energy-Efficient
Buildings”, already does something close to the stated objective: AHP-derived
weights applied across COPRAS, VIKOR, TOPSIS, MOORA and PROMETHEE II, over 16
PCM alternatives, for three climate zones. It must be cited, and the
contribution must be stated as a difference from it.

That paper fixes three representative zones by hand (temperate 18 °C,
subtropical 23 °C, tropical/hot-desert 28 °C), derives AHP weights (melting
point 47.5 %, latent heat 25.7 %, volumetric latent heat 13.5 %, thermal
conductivity 6.8 %, specific heat 3.3 %, density 3.3 %), runs five MCDM methods,
and reports that the methods agree.

| **#** | **Their approach** | **Ours** |

| --- | --- | --- |

| **N1** | Three climate zones chosen by hand from a textbook classification | Climate regimes discovered by unsupervised clustering of ten years of data across four contrasting Indian states, k selected by statistical criteria and cross-checked against state identity, Köppen–Geiger and NBC/ECBC zones |

| **N2** | One representative temperature per zone | A two-tier climate signature per point combining sun-event-aligned instantaneous indices with daily-integral indices — solar availability, cloud persistence, diurnal range and humidity stress |

| **N3** | Building thermal comfort, 18–28 °C melting range | Solar domestic hot water, 42–70 °C melting range — a different PCM family, and a Tm rule driven by delivery and collector temperature rather than comfort |

| **N4** | Single best PCM reported per zone | Top-3 with a consensus score across four (optionally five) MCDM methods and a Monte Carlo stability percentage |

| **N5** | MCDM rankings compared only against each other | MCDM ranking validated against an independent grey-box thermal simulation of annual solar fraction, so the ranking is falsifiable |

| **N6** | Zones treated as uniform geographic areas | Regimes derived from population-weighted sampling covering ~87.5 % of each state’s population, so each regime carries a population figure. A recommendation is therefore expressed as “this PCM serves N million people” rather than “this PCM serves this many square kilometres” — deployment relevance, not just geographic coverage. |

*Table 3. Novelty positioning. N1, N4, N5 and N6 are the strongest; lead with
those.*

**N5 deserves emphasis. **Almost the entire PCM-MCDM literature validates a
ranking by showing that several MCDM methods agree with each other. That
demonstrates internal consistency, not correctness — methods sharing the same
weight vector and the same decision matrix will usually agree. Physics
validation is what converts a preference ordering into a testable claim.

**N6 is the one nobody else has. **Population weighting is unusual in the PCM
selection literature and is not merely a sampling convenience. It changes what
the result means: a regime covering a large but sparsely inhabited desert
receives proportionally less influence than one covering a dense river valley.
Since the downstream purpose is specifying domestic hot water systems, that is
the correct weighting, and it should be argued for explicitly rather than
mentioned in passing.

# 4. Phase 1 — Data Collection (As Built)

This section documents the pipeline that exists, evaluates it against what Phase
3 requires, and specifies the two repairs needed before feature construction can
begin.

## 4.1 The pipeline

| **Stage** | **Script** | **What it produces** |

| --- | --- | --- |

| **Sampling design** | 00a_build_population_grid.py | Downloads the GADM v4.1 admin-1 boundary and the WorldPop 2020 UN-adjusted 100 m India raster, clips to the state, aggregates population onto a 0.25° grid aligned to the ERA5 grid origin, ranks cells by population and keeps the minimal set covering ~87.5 % of state population. Output: population_grid_points.csv with point_id, lat, lon, population, weight. |

| **Temporal design** | 00b_build_suntimes.py | For every point and every date from 2016-01-01 to 2025-12-31, computes exact UTC sunrise, solar noon and sunset using the pvlib SPA implementation, correctly handling refraction, orbital eccentricity and the cross-midnight UTC case. Output: suntimes.csv. |

| **Primary source** | 01_download_era5_rajasthan.py | ERA5 hourly reanalysis over the bounding envelope of the population points, restricted to three padded UTC hour windows per day derived from suntimes.csv, with the instant/accum variable split and deaccumulation helper hours preserved. |

| **Second source** | 01b_download_nasapower.py | NASA POWER hourly point data (ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M) for every point and every year. Note: this returns the FULL hourly series, not just sun-event hours. |

| **Repair utility** | 00_unzip_accum.py | Detects and fixes CDS responses returned as ZIP despite an unarchived NetCDF request. |

| **Merge** | 02_combine_rajasthan.py | Nearest-neighbour snaps each point to the ERA5 grid, concatenates and deaccumulates, computes solar geometry, and for each (point_id, date, event) row selects the nearest-in-time ERA5 and NASA POWER readings, rejecting either if more than three hours from the true event instant. Output: climate_rajasthan_points.csv with era5_* and power_* columns side by side. |

*Table 4. The pipeline as built, replicated for Rajasthan, Assam, Tamil Nadu and
Uttarakhand. Every stage is resumable and status-tracked, which is good
engineering practice and worth one sentence in the paper.*

## 4.2 What the design gets right — and how to say so

**Grid alignment. **Aggregating population onto a 0.25° grid aligned to ERA5’s
own grid origin means each sampling point maps to a distinct ERA5 cell with no
double-counting and no interpolation artefact. This is a detail most papers get
wrong by sampling city coordinates that fall two to a cell. State it.

**Sun-event sampling is physically motivated. **Frame it as
charge–discharge-cycle-aligned sampling, because that is what it is: sunrise is
the coldest instant and therefore the test of whether the PCM fully solidified
overnight; solar noon is the peak charging condition; sunset is the ambient
condition at the start of the evening draw. A uniform three-hourly subsample
would be an arbitrary shortcut. This is not.

**Two independent sources at matched instants. **ERA5 is a reanalysis and NASA
POWER is satellite-derived; they are genuinely independent estimates of the same
quantity at the same place and instant. This is a stronger cross-check than the
CERES comparison v2.0 planned, and it is already downloaded.

**Static population raster. **Using one 2020 WorldPop snapshot for a 2016–2025
study period is a standard simplifying assumption — WorldPop does not publish a
distinct India raster per year at this resolution. Declare it in the
limitations, do not defend it at length.

## 4.3 Two required repairs

### Repair 1 — Recover the daily-integral indices from the NASA POWER cache

Three instantaneous samples per day cannot produce a daily energy integral, a
true diurnal range, or a degree-day count. The merged CSV therefore cannot
support Tier 2 of the climate signature. But the raw NASA POWER JSON cache
already contains the full hourly series for every point and every year, and the
merge step simply discards all but the sun-event hours.

- Write a new script — 02b_build_daily_aggregates.py — that reads
  data/raw/nasapower/power_{point_id}_{year}.json directly and produces a daily
  table: daily GHI integral, daily clear-sky integral, daily clearness index,
  true daily minimum and maximum temperature, daily mean temperature, daily mean
  relative humidity, daily mean wind speed.
- From that daily table compute the Tier 2 indices of §6.2 — GHI_daily_kWh, SAI,
  kt_daily_mean, kt_daily_std, cloudy_frac, CCI, HDD18, CDD24, DTR_true,
  seasonality, monsoon_index.
- Cost: no new downloads, no CDS queue, a few hours of implementation. This is
  the single highest-value task remaining in the data phase.
- If ERA5 daily integrals are wanted as well for consistency, that does require
  a new CDS request for all 24 hours at the same points. Treat it as optional —
  the two-source cross-check at sun events already establishes whether ERA5 and
  POWER agree, and if they do, POWER alone is a defensible backbone for the
  daily tier.
### Repair 2 — Attach real elevation

| **The flat 300 m assumption is not survivable in Uttarakhand. **The pipeline uses a uniform 300 m elevation for solar geometry because population points carry no elevation field. Across Rajasthan, Assam and coastal Tamil Nadu the resulting error is small. Uttarakhand spans roughly 200 m to over 7,000 m; at 2,500 m the air mass, clear-sky irradiance and boiling point all differ materially from the 300 m assumption, and the elev_proxy signature index is meaningless without it. A reviewer who knows the region will ask. |

| --- |

- Attach per-point elevation from ERA5 surface geopotential (the invariant z
  field, divided by 9.80665) or from an SRTM/Copernicus DEM sampled at the point
  coordinates. ERA5 geopotential is the lower-effort option and is consistent
  with the rest of the ERA5 data.
- Recompute pvlib solar geometry and clear-sky irradiance with the true
  elevation. This changes the clear-sky reference and therefore every
  clearness-index-derived index for the mountain points — which is precisely why
  it matters.
- Note in the limitations that ERA5 orography is a grid-cell mean and will
  smooth extreme terrain. That is an honest and sufficient caveat.
## 4.4 Quality control to report

Numbers a reviewer will expect, and which the pipeline can produce cheaply:

- Number of population points retained per state, and the population fraction
  actually covered (the ~87.5 % target may be met at different point counts per
  state). Report the count per state — if any state falls below roughly 30
  points, its internal structure may be under-resolved.
- Sensitivity of the point set to the coverage threshold: rerun at 80 % and 95 %
  and report whether the cluster structure changes. If it does not, that is a
  robustness result worth one sentence.
- Percentage of rows where the nearest ERA5 or NASA POWER reading was rejected
  by the three-hour window, per source, per state.
- The known 2016-01-01 NaN for accumulation-derived columns where a sun-event
  window touches hour 0 UTC. One day in ten years; report it and move on.
- ERA5-versus-POWER agreement at matched instants — MBE, RMSE and correlation
  for GHI, temperature, humidity and wind, per season, per state. This is Phase
  2 and doubles as the bias-correction decision.
## 4.5 PCM property database — unchanged and still outstanding

Target: 40–60 candidates in the 42–70 °C melting range. Below approximately 45
°C the PCM cannot drive heat into water at the 50 °C delivery target; above
approximately 70 °C a flat-plate collector will not reliably charge it. This is
independent of the climate data and can be built in parallel — it is now the
critical path item, since the climate data is largely in hand.

| **Family** | **Representative candidates** | **Notes** |

| --- | --- | --- |

| **Paraffins (Rubitherm RT)** | RT42, RT44HC, RT50, RT55, RT58, RT64HC | Mainstream SWH choice. Non-corrosive, low supercooling, good cycling. k ≈ 0.2 W/m·K is the main weakness. |

| **PLUSS savE OM series** | OM42, OM45, OM48, OM55, OM65 | Indian supplier, commercially available, which matters for a deployment-oriented study. OM55 is characterised as thermally stable across 45–60 °C and already used in domestic solar water heating. |

| **Fatty acids and eutectics** | Lauric (~44 °C), myristic (~54 °C), stearic (~69 °C), and binary eutectics | Moderate latent heat, some corrosivity, mild odour. Eutectics allow Tm tuning between regimes. |

| **Salt hydrates** | Sodium acetate trihydrate (~58 °C), sodium thiosulfate pentahydrate (~48–49 °C) | High volumetric storage density. Strong supercooling (needs a nucleator) plus phase-segregation and corrosion risk — the corrosion veto will likely exclude these in the Assam regimes. |

| **Sugar alcohols** | Erythritol (~118 °C) | EXCLUDED. Far above the usable window for 50 °C domestic delivery. Listed so the exclusion is documented rather than silent. |

*Table 5. PCM families in the corrected 42–70 °C band. Record Tm, latent heat,
thermal conductivity, density, specific heat, cycling stability, supercooling
degree, corrosion class and cost, each with a source citation. Where a value is
unreported, record “not reported” and let the Monte Carlo of §9.6 handle it —
never guess.*

# 5. Phase 2 — Preprocessing and Cross-Source Validation

Much of the classical preprocessing burden — timezone handling, deaccumulation,
solar geometry, nearest-in-time matching — is already inside the pipeline. What
remains is the validation that turns two downloaded sources into a defensible
single backbone.

## 5.1 The ERA5-versus-POWER agreement analysis

This replaces the CERES quantile-mapping step of v2.0 and is the more direct
comparison, because both sources are evaluated at the same point and the same
instant rather than at different resolutions.

- For each variable present in both sources (GHI, 2 m temperature, relative
  humidity, 10 m wind speed), compute mean bias error, RMSE and Pearson
  correlation between era5_* and power_* columns.
- Stratify by season and by state. A bias that appears only in the Assam monsoon
  season is a different finding from a uniform offset, and the two call for
  different responses.
- Stratify also by sun event. Sunrise and sunset GHI values are small and
  near-zero-crossing; disagreement there is expected and less consequential than
  disagreement at solar noon.
- Plot ERA5 against POWER as a scatter with the identity line, one panel per
  state per season. This single figure answers the data-quality question for a
  reviewer faster than any table.
## 5.2 The decision rule

| **Finding** | **Action** |

| --- | --- |

| **Agreement is close and unbiased (MBE small, correlation high)** | Use ERA5 as the primary backbone and report the agreement as a validation result. No correction needed. State that two independent sources agree — this is a stronger position than a corrected single source. |

| **A systematic, season-dependent bias appears (most likely: ERA5 underestimating monsoon-season GHI in Assam and coastal Tamil Nadu)** | Apply quantile mapping of ERA5 GHI onto the POWER distribution, fitted per season per state. Report MBE, RMSE and correlation before and after. |

| **The two disagree severely with no interpretable pattern** | Do not average them. Investigate the merge — a nearest-in-time mismatch, a units error, or an instant-versus-accumulation confusion is far more likely than genuine disagreement between two established datasets. |

*Table 6. Bias-correction decision rule. Deciding this from the data, rather
than assuming a correction is needed, is the defensible order.*

| **Fixed-weight blending remains rejected. **Combining the sources as, for example, 0.6 × ERA5 + 0.4 × POWER has no derivation and would make the resulting dataset impossible to characterise. Either one source is the backbone with the other as validation, or one is quantile-mapped onto the other. There is no third defensible option. |

| --- |

## 5.3 Remaining acceptance checks

- Plot the mean seasonal cycle of noon GHI and noon temperature for every point,
  one panel per state, and inspect by eye. Unit and timezone errors are the most
  common silent failures and are obvious in these plots.
- Verify that the sun-event times behave sensibly across the year: day length
  should be longest near the June solstice and shortest near December, with the
  amplitude larger in Uttarakhand than in Tamil Nadu. If it is not, something is
  wrong in the SPA call or the UTC handling.
- Confirm the cross-midnight cases: eastern points with a summer sunrise falling
  at roughly 23:55 UTC on the previous calendar date should carry the true
  instant in time_utc. Spot-check a handful by hand.
- Confirm no two population points snap to the same ERA5 cell. Grid alignment
  should guarantee this; verify it rather than assume it.
# 6. Phase 3 — Climate Signature Construction

Goal: reduce each point’s ten-year record to one vector of roughly 20 numbers
capturing everything relevant to PCM behaviour. This vector is the object that
gets clustered.

## 6.1 Design principle

Every index must answer the question “which PCM property does this constrain,
and by what physical mechanism?”. If that sentence cannot be completed, the
index is removed. Twenty indices each defensible in one sentence produce a
better paper — and better clusters — than sixty that arrived by automated
generation.

## 6.2 The two-tier index set

The sampling design splits the signature naturally into two tiers with different
provenance. Both are required; neither is optional.

### Tier 1 — Sun-event indices, from the merged CSV

These come directly from climate_{state}_points.csv and are the indices the
sampling design was built for. Each is aggregated over the ten years as a mean
and a standard deviation, and where noted as a percentile.

| **Index** | **Constrains** | **Mechanism** |

| --- | --- | --- |

| **T_sunrise_mean, T_sunrise_p05** | Tm lower bound, solidification | The coldest instant of the day. Determines whether the PCM fully re-solidifies overnight and therefore whether the next charge cycle starts from a full latent capacity. |

| **T_noon_mean** | Tm, charging | Ambient at peak charging — sets the collector loss term when it matters most. |

| **T_sunset_mean, T_sunset_p95** | Discharge rate, k | Ambient at the start of the evening draw. High sunset temperature reduces store losses but also reduces the discharge gradient. |

| **diurnal_gradient** | Cycling stability | T_noon minus T_sunrise. A proxy for diurnal range — note it underestimates true DTR because peak air temperature lags solar noon by two to three hours, which is why DTR_true is retained in Tier 2. |

| **kt_noon_mean** | Latent heat, charging quality | Clearness index at peak sun. The single best one-number descriptor of charging quality. |

| **kt_noon_std** | Cycling stability, k | Day-to-day variability of peak charging — partial-cycle stress. |

| **GHI_noon_mean** | Latent heat, storage mass | Peak charging flux available per unit collector area. |

| **GHI_sunset_mean** | Charging window | Residual irradiance at the start of discharge — how much charging overlaps the draw period. |

| **RH_sunrise_mean** | Corrosion, encapsulation | Relative humidity at the coldest instant, which is when condensation actually occurs. More directly relevant than a daily mean. |

| **HSI_sunrise** | Corrosion class | Humidity stress at the condensation-critical instant. Drives the salt-hydrate exclusion in §8. |

| **wind_noon_mean, wind_sunset_mean** | Thermal conductivity k | Convective loss coefficient during charging and during discharge respectively. |

| **daylength_mean, daylength_amplitude** | Storage sizing | Sunset minus sunrise, from suntimes.csv at no extra cost. Charging window duration and its seasonal swing — larger in Uttarakhand than in Tamil Nadu, and a real constraint on daily charge completion. |

*Table 7. Tier 1, sun-event indices. These exist because of the sampling design
rather than in spite of it, and the paper should present them that way.*

### Tier 2 — Daily-integral indices, recomputed from the NASA POWER hourly cache

These cannot be computed from three instantaneous samples and must come from the
full hourly series already cached on disk (§4.3, Repair 1).

| **Index** | **Constrains** | **Mechanism** |

| --- | --- | --- |

| **GHI_daily_kWh** | Storage sizing | Daily energy integral — the number a designer actually uses to size a system. Not obtainable from noon irradiance alone. |

| **SAI** | Storage capacity | Solar availability index: fraction of the theoretical clear-sky resource actually delivered over the day. |

| **kt_daily_mean, kt_daily_std** | Latent heat, cycling | Whole-day clearness and its variability, distinct from the noon-instant value. |

| **cloudy_frac** | Storage capacity | Fraction of days below a clearness threshold — how often autonomy is exercised. |

| **CCI** | Latent heat floor, cost | Longest run of consecutive low-clearness days. The autonomy the store must provide with no recharge, and the index expected to separate Assam most sharply. |

| **DTR_true** | Cycling stability | True daily maximum minus minimum. Complete cycles per year, therefore degradation rate. |

| **Ta_mean, Ta_p95, Ta_p05** | Tm feasibility window | Whole-day temperature statistics; the extremes decide whether the PCM ever fully melts or fully solidifies. |

| **HDD18** | Tm lower bound, L_required | Water-heating demand intensity. Drives the latent-heat requirement, and expected to dominate in Uttarakhand. |

| **CDD24** | Discharge rate, k | High ambient reduces losses but also reduces the discharge gradient. |

| **seasonality** | Phase stability | Seasonal swing in charging conditions. |

| **monsoon_index** | Regime identity | Separates monsoon-dominated from arid regimes, and — given Tamil Nadu’s north-east monsoon — should also separate the two monsoon timings. This is the index most likely to produce an interesting result. |

*Table 8. Tier 2, daily-integral indices. Recovering these from the existing
NASA POWER cache costs no downloads and is the highest-value remaining data
task.*

### Static attributes

- elevation — from ERA5 surface geopotential or an SRTM DEM (§4.3, Repair 2).
  Constrains density and convection, and is expected to be the dominant
  splitting variable within Uttarakhand.
- population and weight — carried through from the sampling design. Used for
  reporting and for the population coverage figure on each recommendation card,
  NOT as clustering features.
| **Do not put latitude and longitude in the clustering matrix. **It is tempting, and it is wrong. Including coordinates makes the algorithm cluster geography rather than climate, guarantees that the four states separate, and destroys the entire finding — because the interesting result is precisely where climate regimes cross state boundaries or split within one. Coordinates are for plotting the map afterwards, not for fitting. Elevation is a physical variable and is legitimately included; coordinates are not. |

| --- |

## 6.3 Derived PCM-facing quantities

| **This section corrects v1.0 §6.3. **v1.0 stated Tm_target = T_delivery − ΔT_approach, giving 42–45 °C. The sign is wrong. During discharge the PCM is the heat source and the water the sink; heat flows from PCM to water only if the PCM sits above the water temperature. A PCM melting at 43 °C cannot deliver water at 50 °C. |

| --- |

## The corrected rule

**Tm_target = T_delivery + ΔT_approach**

with T_delivery = 50 °C for Indian domestic use and ΔT_approach the
heat-exchanger approach temperature, typically 5–8 K. This yields Tm_target ≈
55–58 °C for an indirect system, or ≈ 50–53 °C for a direct system where the PCM
is encapsulated in the potable tank. State which configuration is assumed; do
not tune the rule per regime after seeing results.

Two upper bounds constrain the same quantity from above: Tm must lie below the
collector delivery temperature achievable on a poor day in that regime (which is
what makes the target regime-dependent — a high-CCI Assam regime supports a
lower Tm than an arid Rajasthan regime), and below flat-plate stagnation
temperature, which should be checked and reported even though it is rarely
binding.

### Literature support for the corrected band

| **Source** | **Finding** | **Implication here** |

| --- | --- | --- |

| **Zhao et al., J. Cleaner Prod., 2019** | The phase change temperature range suitable for conventional heating systems is 47.5–57.5 °C; a series tank–PCM arrangement raised solar fraction by roughly 30 % over a single tank and 5–12 % over parallel | Directly brackets the corrected Tm_target and confirms 42–45 °C sits below the useful band |

| **Avargani et al., J. Energy Storage, 2021** | A paraffin PCM bed (0.3 m dia × 0.6 m) sustained up to 300 L of hot water at 60 ± 2 °C for 7 h of operation | A night-delivery benchmark for the grey-box model, and evidence that ~60 °C-class paraffins perform in this duty |

| **SDHW encapsulated-paraffin studies** | Tanks operating at 58–60 °C and ~62 °C; evacuated-tube manifold paraffin at ~67 °C maintaining 55–60 % efficiency | Confirms 50–65 °C as the working PCM band for domestic hot water |

| **China SWH standard practice** | Exit water required above 50 °C; paired paraffins at 48–50 °C (low) and 62–64 °C (high) | Independent confirmation of the delivery-anchored rule and of a two-PCM cascade as a legitimate design option — relevant if Uttarakhand and Rajasthan regimes demand different Tm |

*Table 9. Literature support for the corrected melting-temperature rule. None of
these is a building passive-cooling study — that distinction is the point of the
correction.*

## The latent-heat floor

**L_required = Q_night / m_PCM , with Q_night = ṁ_draw · cp_water · (T_delivery
− T_mains)**

**T_mains matters more in this study than in an all-India one. **Mains water
temperature tracks ground temperature, which tracks annual mean air temperature
with a lag. Across four states spanning Uttarakhand to Tamil Nadu, T_mains
varies enough to change L_required substantially between regimes — a cold
Himalayan regime demands materially more latent heat per litre delivered than a
Tamil Nadu coastal regime. Estimate T_mains per regime from Ta_mean with a
standard lag correlation and report the resulting L_required spread; it is a
good result in its own right.

## 6.4 Interaction terms and dimensionality

Five interaction terms, each named and justified:

- GHI_daily_kWh × kt_daily_std — charging energy weighted by its unreliability;
  high values mean a large but erratic resource
- DTR_true × cloudy_frac — cycling stress under intermittent charging, the worst
  case for phase stability
- RH_sunrise_mean × (T_sunrise_mean − Tm_target) — condensation risk at the
  store surface at the condensation-critical instant
- wind_sunset_mean × (T_sunset_mean − T_delivery) — convective loss driving
  potential during the evening draw
- CCI × (1 − SAI) — combined autonomy requirement
Then PCA on the correlated block only (Ta_mean, Ta_p95, Ta_p05, T_sunrise_mean,
T_noon_mean, HDD18, CDD24, elevation), retaining components to 95 % variance —
typically three. Keep the solar, variability and humidity indices out of the
PCA: they carry the discriminating signal. Report the component loadings; across
these four states they should be readable as roughly “heat”, “altitude” and
“seasonal amplitude”, which is itself a result.

Standardise all columns to zero mean and unit variance before clustering —
Euclidean distance is meaningless otherwise when kWh/m²/day sits beside a
percentage.

**Rejected alternatives, unchanged. **DCCA remains rejected on sample-size and
interpretability grounds. A TabTransformer or FT-Transformer encoder may be run
as an optional ablation after the engineered pipeline works; a negative result
there is worth reporting.

# 7. Phase 4 — Climate Regime Clustering

This is the core of the objective and the part a reviewer scrutinises hardest,
because unsupervised clustering has no ground truth and is easy to do badly. The
four-state design changes what a good result looks like.

## 7.1 What counts as a result here

| **Recovering the four states is not a finding. **Rajasthan, Assam, Tamil Nadu and Uttarakhand are known to be climatically different. If the clustering returns four clusters that map one-to-one onto state boundaries, it has reproduced the sampling design and told you nothing. Report the adjusted Rand index against state identity precisely so that this can be seen and addressed: an ARI near 1.0 at k = 4 means k is too low. The result of interest is intra-state splitting and, more valuably, any cross-state merging. |

| --- |

Three specific findings to look for, each of which would be a genuine
contribution:

- Intra-state splitting — arid west versus semi-arid east in Rajasthan; terai,
  mid-hills and high Himalaya in Uttarakhand; coastal, interior dry and Nilgiris
  in Tamil Nadu; Brahmaputra valley versus surrounding hills in Assam. Each
  split is a case where a single state-level PCM specification would be wrong.
- Cross-state merging — if, for example, interior Tamil Nadu clusters with
  eastern Rajasthan rather than with coastal Tamil Nadu, that directly
  demonstrates that administrative boundaries are the wrong unit for PCM
  specification. This is the strongest available argument for the whole
  framework.
- Population-weighted regime size — a regime covering a small area but a dense
  population deserves more design attention than a large sparse one. Report
  population per regime alongside point count; this is where novelty claim N6
  pays off.
## 7.2 Two levels of clustering — do both

| **Level** | **What is clustered** | **What it gives you** |

| --- | --- | --- |

| **Level A — spatial** | One signature vector per population point; cluster the points across all four states together | Climate regimes spanning and subdividing the four states. Answers: “which PCM for a system installed at this location?” |

| **Level B — temporal** | For each point, one signature per season across the ten years; cluster those | Operating regimes within a location — monsoon, dry, transition. The merged CSV already carries season and season_code, so this is nearly free. Answers: “does this location need a different PCM in July than in March?” |

*Table 10. Two-level clustering. Level A is required by the objective; Level B
is what makes it climate-aware rather than merely region-aware.*

**Level B is where the interesting result lives, and this dataset is unusually
good for it. **Tamil Nadu’s north-east monsoon is out of phase with the
south-west monsoon that dominates Assam and Rajasthan. If Level B shows the
Top-3 flipping between seasons in Tamil Nadu but holding steady in Rajasthan,
that is direct empirical motivation for the adaptive control objective —
generated by this objective, from your own data, rather than asserted from the
literature. Either outcome is a result.

## 7.3 Algorithm choice: Gaussian Mixture over K-Means

GMM is the primary model. K-Means is fitted alongside for k = 2…10 and reported
as a comparison, which answers the algorithm-sensitivity concern directly and
turns it into a reported result rather than an unexamined choice.

| **Consideration** | **K-Means** | **GMM (chosen)** |

| --- | --- | --- |

| **Cluster shape** | Spherical, equal variance assumed | Elliptical, per-component covariance |

| **Assignment** | Hard | Soft — membership probabilities |

| **Silhouette** | Typically higher (crisper partitions) | Typically lower, because it does not force separation that is not there |

| **Model selection** | Heuristic (elbow) | Principled — BIC / AIC |

| **Downstream use** | A point belongs to exactly one regime | A transition-zone point that is 60 % arid and 40 % semi-arid receives a membership-weighted PCM recommendation. With population-weighted sampling across sharp gradients such as the Uttarakhand terai–hill boundary, transition points are common and soft membership is not a nicety. |

*Table 11. Algorithm comparison. The soft-membership column is why GMM is
retained despite scoring lower on silhouette.*

**Do not weight the GMM fit by population. **The sampling is already
population-weighted by construction — densely inhabited areas contribute more
points. Applying population weights again during fitting would double-count. Use
population for reporting and for the recommendation cards, not for the
likelihood.

## 7.4 Choosing k, and what silhouette to expect

Fit k = 2…12 and report all of: BIC and AIC curves, mean silhouette,
Davies–Bouldin index, Calinski–Harabasz index, and bootstrap stability (adjusted
Rand index between clusterings of resampled data). Choose k where the criteria
agree, and state the disagreement where they do not.

**Expected k: 6–10. **Four states with plausible internal splits of two to three
each. If k selects at 4, check immediately whether the clusters are simply the
states — and if they are, report the k = 6–8 solution alongside as the
informative one, with the model-selection evidence for both. If k selects above
10, some clusters are likely singletons or near-duplicates; inspect membership
counts before accepting it.

| **Realistic silhouette expectation: 0.15–0.35. **For data-driven climate zoning over India, published silhouettes peak near 0.3. A criteria-based reclassification of Indian climate zones reports 0.21 against −0.2 for the current NBC classification, peaking around 0.3 at k = 6, and that was a successful result. An Indian thermal-comfort clustering study reports 0.235. Note one caveat specific to this design: because four contrasting states are sampled and the intervening territory is not, the between-state gaps are artificially clean and the silhouette may come out somewhat higher than an all-India study would produce. Do not present an inflated silhouette as evidence of superior method — explain that it partly reflects the sampling frame. |

| --- |

## 7.5 External validation — the step that earns credibility

- Adjusted Rand index against state identity. Expect substantial but imperfect
  agreement; perfect agreement means the clustering added nothing (§7.1).
- Adjusted Rand index and normalised mutual information against Köppen–Geiger
  classes at the same points.
- The same statistics against NBC/ECBC Indian climate zones, restricted to the
  four zones represented.
- A map per state, points coloured by hard assignment and shaded by maximum
  membership probability, so ambiguous transition points are visible. Four state
  panels are a better figure than one national map here, because the states are
  not contiguous.
- Elevation profile per cluster for Uttarakhand specifically — if the mid-hills
  and high-Himalaya clusters do not separate on elevation, Repair 2 was not
  applied correctly.
**Interpreting agreement. **The target is substantial but imperfect agreement
with existing classifications, with every departure explainable from the
signature indices — for example, two Köppen-identical points separating because
one has a much higher CCI and therefore a higher autonomy requirement. That
explanation is the paper’s most persuasive paragraph.

## 7.6 Regime characterisation

For each regime produce a profile card: medoid point (with its district and
elevation), member points and their state distribution, total population
covered, the full two-tier signature as mean ± standard deviation, a one-line
physical description, and the derived Tm_target and L_required. These feed
directly into Phase 5 and become the results section.

# 8. Phase 5 — Feasibility Filtering

Before any ranking, hard-filter the PCM database per cluster. MCDM is a
compensatory method: a large advantage on one criterion can offset a fatal
deficiency on another. A PCM with an unreachable melting point and outstanding
latent heat can score well in TOPSIS and be physically useless. Filtering first
prevents that.

| **Constraint** | **Rule (v2.0)** | **Justification** |

| --- | --- | --- |

| **Melting window** | Tm ∈ [Tm_target − 5, Tm_target + 8] °C | Below the lower bound the store cannot drive heat into water at the delivery temperature; above the upper bound a flat-plate collector will not reliably charge it in that cluster's solar regime |

| **Absolute band** | Tm ∈ [42, 70] °C regardless of cluster | Outside this, the candidate is not a solar domestic hot water PCM at all |

| **Charging feasibility** | Tm below the collector delivery temperature at the cluster's 5th-percentile daily insolation | The PCM must melt on a poor day, not only on a good one |

| **Latent heat floor** | L ≥ 0.7 × L_required for that cluster | Below this the store cannot supply the night draw within a plausible tank volume |

| **Cycling stability** | ≥ 300 cycles where reported; retain and flag where not reported | Roughly one cycle per day means 300 cycles is under one year of service |

| **Corrosion veto** | Exclude bare salt hydrates where HSI exceeds the cluster 75th percentile unless encapsulation is specified | Condensation-driven corrosion is a documented failure mode in humid coastal installations |

| **Supercooling veto** | Exclude candidates with supercooling > 8 K unless a nucleating agent is specified | Supercooling means the store holds energy it cannot release — the specific failure the whole system exists to avoid |

| **Safety** | Exclude toxic or highly flammable candidates for domestic installation | Non-negotiable for a household product |

*Table 12. Feasibility constraints, updated for the corrected melting band.
Report how many of the original candidates survive per cluster — that number is
itself informative and belongs in the results.*

If a cluster retains fewer than five candidates, relax the melting window by 2 K
and record that it was relaxed. If it retains more than twenty-five, the
constraints are too loose. Eight to twenty is a healthy candidate set for MCDM.

# 9. Phase 6 — Multi-Criteria Ranking Engine

## 9.1 Criteria

| **Criterion** | **Type** | **Indicative weight** | **Note** |

| --- | --- | --- | --- |

| **Melting-point fitness** | Target-based | 0.24 | Converted from │Tm − Tm_target│ to a fitness score — see §9.2, the step most implementations get wrong |

| **Latent heat L** | Benefit | 0.20 | Ranked highest-priority property in the PCM-SWH review literature |

| **Volumetric latent heat ρL** | Benefit | 0.12 | What actually determines tank size; often diverges from L alone |

| **Thermal conductivity k** | Benefit | 0.13 | Governs charge and discharge rate; weighted higher here than in building studies because SWH has a charging-rate constraint that comfort applications do not |

| **Cycling stability** | Benefit | 0.11 | Service life |

| **Supercooling (inverse)** | Cost | 0.08 | Energy that cannot be released is energy not stored |

| **Corrosion class (inverse)** | Cost | 0.06 | Cluster-dependent: weight higher in high-HSI clusters |

| **Cost** | Cost | 0.06 | Weight honestly — do not let a data-poor criterion dominate |

*Table 13. Criteria set with indicative starting weights. Actual weights come
from §9.3; the sensitivity analysis in §9.6 must show the Top-3 is not an
artefact of them.*

## 9.2 The target-based criterion — get this right

| **The most common error in PCM MCDM papers. **Melting temperature is neither a benefit nor a cost criterion. Higher is not better and lower is not better; closer to target is better. Standard TOPSIS, VIKOR, GRA and CoCoSo normalisation all assume monotonic criteria and will silently produce plausible-looking nonsense if fed raw Tm. |

| --- |

## Convert Tm to a fitness score before it enters the decision matrix

**f_Tm(i) = exp( − (Tm_i − Tm_target)² / (2σ²) ), σ ≈ 4 K**

This is a Gaussian fitness peaking at the target, decaying symmetrically,
bounded in (0, 1], and now a proper benefit criterion. Justify σ = 4 K from the
heat-exchanger approach temperature. An asymmetric form is physically better
motivated — the penalty for Tm being too high (the PCM never melts on a poor
day) is more severe than for being slightly too low (the PCM melts early and
delivers at a lower temperature) — so σ_upper < σ_lower. Whichever is chosen,
state it and test the alternative in the sensitivity analysis.

**PROMETHEE II handles this more elegantly. **Define the criterion as −|Tm −
Tm_target| with a V-shape or Gaussian preference function, indifference
threshold q = 2 K and preference threshold p = 8 K. Those thresholds have direct
engineering meaning — "differences under 2 K do not matter; differences over 8 K
are decisive" — which is precisely the kind of statement a thermal engineer can
confirm or contest. This is the strongest single argument for keeping PROMETHEE
II in the stack, and the reason adding CoCoSo does not remove the need for it.

## 9.3 Weight determination

## Combine an objective and a subjective source

**w_j = λ · w_j^entropy + (1 − λ) · w_j^AHP, λ = 0.5**

**Entropy weights **are computed per cluster from that cluster's own filtered
decision matrix. A criterion on which all surviving candidates are
near-identical automatically receives low weight, which is correct since it
cannot discriminate. This makes weights cluster-specific — a feature worth
pointing out in the paper.

**AHP weights **encode domain priority. Build the pairwise matrix with the guide
and, if possible, one thermal-engineering faculty member. Record who provided
the judgements and report the consistency ratio; it must be below 0.10. If it is
not, revisit the inconsistent comparisons with the respondent rather than
adjusting the matrix directly.

| **Why the 0.5 / 0.5 blend is the right call. **Entropy weighting is data-driven and can be dominated by whichever criterion happens to have the most spread in the matrix. In the Oluah 2020 entropy+TOPSIS PCM study, thermal conductivity received 72.12 % of the total weight purely because that column varied most — a result no thermal engineer would endorse as a statement of priority. In the building AHP studies, latent heat or melting point receives 47–57 %. Neither source alone is trustworthy; the blend anchors objective spread against expert priors, and reporting λ = 0, 0.5 and 1 in the sensitivity analysis shows whether the Top-3 depends on that choice. If the Top-3 is identical across all three, say so — it is a strong robustness statement. |

| --- |

## 9.4 The ranking methods

| **Method** | **Output** | **Why it is in the stack** |

| --- | --- | --- |

| **TOPSIS** | Closeness coefficient Ci ∈ [0,1] | Interpretable, precedented in PCM selection, gives a natural score for reporting |

| **PROMETHEE II** | Net outranking flow φ ∈ [−1,1] | Non-compensatory pairwise preference; handles the target-based Tm criterion natively; resistant to rank reversal |

| **VIKOR** | Compromise index Qi plus acceptable-advantage and acceptable-stability tests | Its formal conditions can return a set of compromise solutions rather than one winner — the cleanest principled justification for reporting Top-2/Top-3 |

| **GRA** | Grey relational grade Γ ∈ [0,1] | Robust to sparse and noisy property data, which this is; already precedented in the project references |

| **CoCoSo (optional, new in v2.0)** | Three appraisal scores fused into a composite ki | Hybrid compensatory / non-compensatory aggregation, genuinely different in kind from the other four. Added as a fifth cross-check after the four-method consensus is working — never as a replacement. Requires the §9.2 Tm transform first. |

*Table 14. Ranking methods. Roughly 120 lines of Python for the core four;
CoCoSo adds about 30 more. None needs a specialised library.*

## 9.5 Rank aggregation into a consensus Top-3

- Compute Kendall's W (coefficient of concordance) across the rankings per
  cluster. W > 0.8 means strong agreement and the consensus is safe. Low W is
  itself a finding: it identifies clusters where the PCM choice is genuinely
  ambiguous, and those deserve discussion rather than a forced answer. If W
  falls below roughly 0.6, investigate the criterion definitions before trusting
  the consensus.
- Aggregate by Borda count and cross-check with Copeland pairwise majority.
  Where they disagree, report both.
- Report the consensus Top-3 with each method's individual rank alongside, so a
  reader can see the disagreement rather than having it hidden by the aggregate.
- If CoCoSo is run, report the consensus both with and without it. If the Top-3
  is unchanged, that is a robustness result and a direct answer to the reviewer.
**Borda(i) = Σ_m (n − rank_m(i)) Consensus Top-3 = argmax₃ Borda(i)**

## 9.6 Confidence via Monte Carlo

A Top-3 without a stability measure is a bare assertion. Quantify it:

- Draw 5,000 perturbed scenarios. In each, perturb the criterion weights by a
  Dirichlet draw centred on the nominal weights (concentration chosen to give
  roughly ±20 % variation), and independently perturb each PCM property by
  Gaussian noise scaled to its reported measurement uncertainty (±5 % latent
  heat, ±10 % thermal conductivity, ±1 K melting point, wider for cost).
- Re-run the full ranking pipeline for each draw.
- For each PCM, report the proportion of draws in which it appears in the Top-3
  — "RT55 appears in the Top-3 in 94 % of 5,000 perturbed scenarios."
- Report alongside it: Top-1 retention rate, rank-reversal frequency, and
  Spearman ρ or Kendall τ of each perturbed ranking against the baseline. These
  four together are standard reported practice.
- Report the full inclusion-probability distribution as a figure. It is one of
  the strongest results the paper can carry and costs only compute time.
**On the proposed 10,000 draws. **Not adopted. Inclusion probabilities converge
well before 5,000; many published MCDM stability studies use 1,000. If runtime
turns out to be trivial, raising the count is harmless but should not be
presented as a methodological improvement.

**Missing data is handled here, cleanly. **Where a property is unknown, sample
it from the type-class distribution rather than imputing a point value. PCMs
with more missing data then show wider inclusion intervals, which is the honest
representation of what is known.

# 10. Phase 7 — Physics-Based Validation

| **Do not skip this. **Everything up to §9 produces a preference ordering. Nothing in it establishes that a higher-ranked PCM actually performs better. Without this phase the paper says "four MCDM methods, given the same weights and the same matrix, agreed with each other" — which is close to a tautology. This phase makes the claim falsifiable, and it is the difference between an undergraduate exercise and a publishable result. |

| --- |

## 10.1 The simulation tool — and why not EnergyPlus

| **Tool** | **Verdict** | **Reasoning** |

| --- | --- | --- |

| **Python grey-box lumped enthalpy tank model** | PRIMARY | Transparent, every line explicable in a viva, integrates directly with the ranking pipeline, no licence. The appropriate and defensible tool at this scale. |

| **TRNSYS Type 860** | Optional cross-check | A PCM-in-tank component built on the Type 60 water tank using the enthalpy method, supporting encapsulation geometry, hysteresis and supercooling. The right tool if a licence is available. |

| **EnergyPlus** | REJECTED | Models PCM only as solid conduction layers inside building surfaces (MaterialProperty:PhaseChange with the CondFD algorithm), and models solar water heaters through a separate plant-loop water tank object. There is no supported path to place a latent-heat PCM inside the tank node network. Specifying it would produce a building-envelope result mislabelled as a hot water result. |

| **CFD** | REJECTED | Out of scope and unnecessary. A well-calibrated lumped model validated against published experiment is appropriate; an elaborate model that is wrong is worse than a crude one honestly described. |

*Table 15. Validation tooling. The EnergyPlus row exists to record why the
review**'**s proposal was not adopted.*

## 10.2 The experiment

- Build the grey-box tank model with an enthalpy formulation for the phase
  change and a lumped water node, driven by the cluster medoid city's hourly
  weather.
- Use a cited standard domestic hot water draw profile. Do not invent one.
- Simulate a full year for every feasible PCM in that cluster, not only the
  Top-3 — the full ordering is needed to correlate against.
- Record: annual solar fraction (the primary metric), hours per year with
  delivery temperature met, mean melt fraction achieved, and number of complete
  cycles.
- Compute Spearman ρ between the MCDM consensus rank and the simulated
  solar-fraction rank, per cluster.
### Calibration benchmarks

Calibrate before trusting the model. If it produces results outside these bands,
fix the model before running the experiment.

| **Benchmark** | **Published range** | **Use** |

| --- | --- | --- |

| **Annual solar fraction, SWH systems** | ≈ 54–84 %, typically around 69 % | The band the annual solar fraction output must fall within |

| **TRNSYS model vs experiment** | Within ±10 % | The accuracy target for the grey-box model against any published case it is calibrated on |

| **PCM-in-tank series configuration gain** | ≈ +30 % solar fraction over a plain tank; +5–12 % over parallel | Sanity check that adding PCM in the model improves solar fraction by a plausible margin |

| **Night-time delivery** | 300 L at 60 ± 2 °C sustained for ~7 h from a 0.3 × 0.6 m paraffin bed | Sanity check on discharge duration and delivered volume |

| **Flat-plate paraffin daily efficiency** | Maximum daily efficiency near 65 % | Upper bound on collector-side efficiency in the model |

*Table 16. Calibration benchmarks drawn from published PCM-SWH experiments.*

## 10.3 Interpreting the result — all three outcomes are publishable

| **Outcome** | **Meaning** | **What to write** |

| --- | --- | --- |

| **ρ ****>**** 0.8** | MCDM ranking predicts physical performance | The strong result. Report ρ per cluster and state that MCDM is a valid low-cost proxy for simulation — which matters because simulation is expensive and MCDM is not. |

| **0.4 ****<**** ρ ****<**** 0.8** | Partial agreement | The most likely outcome and still a good result. Identify which criteria drive the disagreement — usually a weight over-rewarding latent heat while the simulation is conductivity-limited. Recommend a weight adjustment and show it improves ρ. |

| **ρ ****<**** 0.4** | MCDM ranking does not predict performance | Also publishable, and more interesting than it feels. It means the criteria set or the weights are wrong. Diagnose which, fix it, report both before and after. A negative result you diagnosed beats a positive result you did not earn. |

*Table 17. Validation outcomes. Decide now that whatever result appears will be
reported — deciding afterwards is how results get quietly reshaped.*

# 11. Phase 8 — Explanation and Final Output

For every cluster, produce a recommendation card. Six of these — one per cluster
— form the results section of the paper.

| **Field** | **Content** |

| --- | --- |

| **Cluster identity** | ID, medoid city, member cities, one-line physical description, mean maximum membership probability |

| **Climate signature** | The 18 indices, mean ± standard deviation |

| **Derived targets** | Tm_target (with the assumed system configuration stated), L_required, dominant constraint |

| **Candidates screened** | Number entering and surviving the feasibility filter, and whether the window was relaxed |

| **Rank 1 / 2 / 3** | PCM name, type, Tm, L, k, consensus Borda score, per-method ranks (including CoCoSo if run), Monte Carlo Top-3 inclusion probability |

| **Criterion contributions** | Which criteria drove each PCM's score, as a signed decomposition |

| **Simulated performance** | Annual solar fraction from Phase 7 for each of the three, and the cluster Spearman ρ |

| **Caveats** | Missing properties, imputations, whether constraints were relaxed, membership ambiguity |

*Table 18. Recommendation card schema.*

# 12. Timeline — Re-Baselined

Phases 1 and 2 are substantially complete: the sampling design, the sun-event
computation, both downloads and the merge all exist and have been run for four
states. That is roughly five weeks of the v2.0 schedule already banked. Twelve
weeks remain.

## 12.1 Weeks 1–3: repairs and the PCM database

| **Week** | **Focus** | **Exit criterion** |

| --- | --- | --- |

| **1** | Repair 1 — write 02b_build_daily_aggregates.py, reading the NASA POWER hourly cache directly and producing the daily table. Repair 2 — attach per-point elevation from ERA5 surface geopotential and recompute solar geometry for the mountain points. Report the QC numbers of §4.4. | Daily aggregate table exists for all four states; every point carries a real elevation; point counts and coverage fractions reported per state |

| **2** | Phase 2 cross-source validation. ERA5-versus-POWER MBE, RMSE and correlation per season per state per sun event. Decide the bias-correction question by the §5.2 rule and record the decision. Build the four-panel scatter figure. | Bias decision made and documented; agreement figure drafted; acceptance checks of §5.3 all pass |

| **3** | PCM property database — now the critical path. Build to 40+ rows in the 42–70 °C band from datasheets and review papers, every row with a source citation. Derive Tm_target for both direct and indirect configurations and write up the rule with its literature support. | D2 complete at 40+ cited rows; Tm_target derivation written |

*Table 19. Weeks 1–3. The two repairs are independent of each other and of the
PCM database, so this block parallelises well across a team.*

## 12.2 Weeks 4–12: signature, clustering, ranking, validation, writing

| **Week** | **Output** |

| --- | --- |

| **4** | Climate signature construction. Both tiers implemented, interaction terms computed, PCA on the correlated block, correlation matrix inspected for redundancy. D3 complete: one row per point with a written justification for every index. |

| **5** | Level A clustering. GMM and K-Means fitted for k = 2…12; BIC, silhouette, Davies–Bouldin, Calinski–Harabasz and bootstrap-stability curves; k chosen and justified, with the state-recovery check of §7.1 performed explicitly. |

| **6** | Cluster validation and characterisation. ARI against state identity, Köppen–Geiger and NBC/ECBC; four state maps with membership shading; regime profile cards with population coverage; Uttarakhand elevation separation check. |

| **7** | Level B temporal clustering for representative points in each state, with particular attention to the Tamil Nadu north-east monsoon versus the south-west monsoon states. Feasibility filter implemented; Tm_target and L_required derived per regime. |

| **8** | MCDM core. TOPSIS and GRA working with target-based Tm handling; TOPSIS unit-tested against the Oluah 2020 fixture to three decimal places; entropy weights computed per regime. |

| **9** | MCDM completion. PROMETHEE II and VIKOR implemented; AHP pairwise matrix collected from the guide with CR verified below 0.10; Borda and Copeland aggregation; Kendall’s W per regime. |

| **10** | Confidence and headline result. The 5,000-draw Monte Carlo; D6 — the Top-3 table per regime with inclusion probabilities, Top-1 retention, rank-reversal frequency and population coverage. This is the headline result. |

| **11** | Thermal simulation. Grey-box enthalpy tank model implemented and calibrated against the Table 16 benchmarks, driven by each regime medoid point’s weather; full-year simulation for every feasible PCM in every regime; D7 — Spearman ρ per regime. |

| **12** | Figures, draft and buffer. All figures finalised; full IEEE draft; reproducibility check — clean clone, rerun end to end, confirm every number in the paper regenerates. |

*Table 20. Weeks 4–12. Week 12 carries both the draft and the buffer, which is
tight — if weeks 10–11 overrun, the optional items in §12.3 are what to drop.*

## 12.3 Critical path, parallelism, and what to drop

The critical path is now: daily-aggregate repair → signature → clustering → MCDM
→ simulation → results. The ERA5 download, which dominated the v2.0 critical
path, is behind you.

- The PCM database (week 3) is fully independent of the climate data — assign it
  to a different team member and run it in parallel with weeks 1–2.
- The AHP pairwise elicitation needs the guide’s time. Book the slot in week 6,
  not week 9.
- The thermal simulation depends only on the PCM database and one point’s
  weather. Start it in week 7 if ahead; it remains the task most likely to
  overrun.
- Drop in this order if time runs short: the CoCoSo fifth-ranker ablation first,
  the FT-Transformer encoder ablation second, the coverage-threshold sensitivity
  at 80 % and 95 % third. None is load-bearing. Do not drop the physics
  validation — it is what makes the paper publishable.
- Optional if ahead: a full 24-hour ERA5 request at the same points, giving
  ERA5-based Tier 2 indices for consistency with Tier 1. Only worth it if the
  Phase 2 analysis shows ERA5 and POWER disagreeing materially.
# 13. Repository Structure and Tooling

The data-acquisition half of the repository already exists. What follows maps
the existing scripts onto the modelling modules still to be written, so the two
halves form one coherent project rather than a pipeline with an analysis bolted
on.

pcm-climate-framework/

|-- data/

| |-- raw/boundary/ # GADM v4.1 admin-1 (exists)

| |-- raw/population/ # WorldPop 2020 100m raster (exists)

| |-- raw/era5/points/ # sun-event NetCDF per state-year-month (exists)

| |-- raw/nasapower/ # FULL hourly JSON per point-year (exists)

| |-- processed/population_grid_points.csv (exists)

| |-- processed/suntimes.csv (exists)

| |-- processed/climate_{state}_points.csv (exists, Tier 1 source)

| |-- processed/daily_{state}.csv (Repair 1 - to build)

| |-- processed/signature_matrix.csv (to build)

| `-- pcm/pcm_database.csv # 40-60 rows, 42-70 C band, every row cited

|-- src/

| |-- acquire/ config.py, 00a_build_population_grid.py,

| | 00b_build_suntimes.py, 01_download_era5.py,

| | 01b_download_nasapower.py, 00_unzip_accum.py (exist)

| |-- preprocess/ 02_combine.py (exists),

| | 02b_build_daily_aggregates.py (Repair 1),

| | 02c_attach_elevation.py (Repair 2),

| | cross_source_validate.py (Phase 2)

| |-- features/ indices_tier1.py, indices_tier2.py,

| | signature.py, tm_target.py

| |-- cluster/ fit.py, select_k.py, validate.py

| |-- pcmrank/ filter.py, weights.py, topsis.py, promethee.py,

| | vikor.py, gra.py, cocoso.py, aggregate.py, montecarlo.py

| |-- sim/ tank_model.py, draw_profile.py, run_year.py

| `-- viz/ state_maps.py, ranking_plots.py, agreement_scatter.py

|-- tests/ test_topsis.py, test_entropy.py, test_indices.py

|-- notebooks/ 01_explore.ipynb ... 07_results.ipynb

|-- results/ figures/, tables/, cards/

|-- environment.yml

`-- README.md

**One suggestion on structure. **The acquisition scripts are currently named for
Rajasthan. Since the same pipeline now runs for four states, parameterise the
state as an argument and keep one copy of each script rather than four
near-identical forks. A state column in every processed output, plus a
states.yml listing the four, will save a great deal of reconciliation later and
makes the framework’s state-agnosticism visible to a reader of the code.

Libraries: geopandas and rasterio for the population grid; xarray, netCDF4 and
cdsapi for ERA5; requests for NASA POWER; pandas and numpy; pvlib for solar
geometry, sun-event times and the clear-sky reference; scikit-learn for GMM,
K-Means, PCA and the cluster metrics; scipy for Spearman and statistics; pymcdm
for reference MCDM implementations — but write your own TOPSIS as well, because
every line must be explicable in a viva and the target-based criterion needs
custom handling; matplotlib and geopandas for the state maps.

## 13.1 The TOPSIS unit-test fixture

| **Test your TOPSIS before you trust it. **A wrong TOPSIS still produces plausible-looking numbers. Normalisation and weighting errors are invisible in the output and only a numerical fixture catches them. |

| --- |

Use Oluah, Akinlabi and Njoku (2020), Energy and Buildings 217, “Selection of
phase change material for improved performance of Trombe wall systems using the
entropy weight and TOPSIS methodology”. It publishes every intermediate matrix —
raw decision matrix, normalised matrix, weighted normalised matrix, positive and
negative ideal solutions, separation measures and final closeness coefficients —
for 11 PCMs across 4 criteria.

| **Assertion** | **Expected value** |

| --- | --- |

| **Entropy weight, thermal conductivity** | ≈ 72.12 % |

| **Entropy weight, heat of fusion** | ≈ 2 % |

| **Entropy weight, density** | ≈ 11 % |

| **Entropy weight, cost** | ≈ 15 % |

| **Best alternative closeness coefficient** | Capric + palmitic eutectic, Pi ≈ 0.951 |

| **Worst alternative closeness coefficient** | n-octadecane, Pi ≈ 0.004 |

*Table 21. Unit-test assertions. Reproduce these to three decimal places before
running project data. The fixture uses four criteria and omits melting
temperature, so it validates the entropy and TOPSIS machinery but NOT the
target-based Gaussian transform — test that separately against a hand-computed
example.*

**A second, incidental use. **The 72.12 % thermal-conductivity entropy weight in
this fixture is also the clearest available demonstration of why entropy
weighting alone is untrustworthy, and therefore why the 0.5 / 0.5 entropy–AHP
blend of §9.3 is justified. Cite it for both purposes.

# 14. Risk Register

| **Risk** | **Likelihood** | **Mitigation** |

| --- | --- | --- |

| **Clustering merely recovers the four state boundaries** | High | Report ARI against state identity explicitly and present the k = 6–8 solution alongside k = 4 with model-selection evidence for both. If no intra-state structure exists at all, that is itself reportable — but check the feature set first, since omitting Tier 2 or elevation is the most likely cause. |

| **Tier 2 repair skipped or deferred** | High | Do it in week 1. Without daily-integral indices the signature cannot describe storage sizing or autonomy, and the clustering will be driven almost entirely by temperature. This is the single most consequential outstanding task. |

| **Scope creep back into forecasting, control or hardware** | High | Re-read §1.2. Those are separate objectives. Finish this one first. |

| **PCM cost and cycling data unavailable** | High | Sample from type-class distributions in the Monte Carlo rather than imputing point values; or drop the criterion and renormalise weights, documenting the choice. |

| **Elevation repair skipped** | Medium | Uttarakhand results become indefensible and the elev_proxy index meaningless. ERA5 surface geopotential is a single invariant field — a few hours of work at most. |

| **Reviewer objects to four states rather than all India** | Medium | §1.3 is the prepared answer: four of five or six NBC zones, sampled densely and population-weighted, versus six zones at one city each. State the missing temperate zone as a limitation before it is raised. |

| **Silhouette inflated by the non-contiguous sampling frame** | Medium | Anticipated in §7.4. Explain that between-state gaps are artificially clean because the intervening territory was not sampled. Do not present the inflated value as method superiority. |

| **Same PCM wins every regime** | Medium | Likely means the Tm window is too wide or the candidate set too narrow. If it persists across four genuinely contrasting states, that is a strong and commercially useful result — state it. |

| **Thermal simulation does not calibrate** | Medium | Simplify to a single-node tank with an effective heat capacity. A crude model honestly described beats an elaborate one that is wrong. |

| **MCDM rank does not correlate with simulation** | Medium | This is a result, not a failure. See Table 17 — diagnose and report. |

| **Guide unavailable for AHP elicitation** | Medium | Book the slot in week 6. Fall back to entropy-only weights (λ = 1) with a sensitivity analysis over published subjective weight vectors. |

| **A reviewer repeats the earlier critical review’s objections** | Medium | §2 is the prepared answer. Present the K-Means comparison and the CoCoSo ablation as evidence the alternatives were tested rather than dismissed. |

*Table 22. Risk register, updated for the four-state design.*

## 14.1 Recorded as future work

- Extension of the framework to the remaining Indian states and to the temperate
  NBC zone not represented here. The pipeline is state-parameterised and the
  marginal cost per additional state is one download run.
- A full 24-hour ERA5 request at the same population points, giving ERA5-derived
  daily-integral indices for consistency with the sun-event tier.
- A 12–24 month instrumented field trial in two or three regimes, comparing
  measured against simulated annual solar fraction. Not achievable within a
  final-year timeline without a thermal laboratory, but the correct next step.
- A two-PCM cascade specification for regimes where Level B temporal clustering
  shows the Top-3 flipping between seasons — most likely in Tamil Nadu given its
  out-of-phase monsoon.
- Time-varying population weighting, if WorldPop or a comparable product
  publishes annual India rasters at this resolution.
# 15. References

IEEE format. Items marked [V] were located and verified during the
review-verification pass of §2; [A] are already available in the project folder;
[P] still need to be pulled; [D] are data-source and software citations required
by the methods section.

[1] [V] N. Ben Ali, B. Louhichi, W. H. Hassan, A. Alizadeh, A. A. Hussein, W.
Aich, K. Hajlaoui, and S. Aminian, "Design of a Li-ion battery cooling system
incorporating PCM, heat pipes, and liquid circuits using marine predator
algorithm-enhanced ANN and multi-verse optimization," Sci. Rep., vol. 16, no. 1,
Art. no. 11796, 2026, doi: 10.1038/s41598-026-41155-5.

[2] [V] A. B. Huluka and S. Muthulingam, "Integrated spherical phase change
modules in concrete roofs enhance thermal performance in hot climates," Sci.
Rep., vol. 15, no. 1, Art. no. 39845, 2025, doi: 10.1038/s41598-025-23490-1.

[3] [V] A. Binte Ahmed, M. M. Uddin Qureshi, M. M. Hussain Khan, A. Dulmini, M.
A. Haque Mollah, and R. Rois, "Application of seasonal-adjusted hybrid models
for forecasting Discomfort Index in a heat-prone region of Bangladesh," PLoS
ONE, vol. 21, no. 3, Art. no. e0344556, 2026, doi: 10.1371/journal.pone.0344556.

[4] [V] G. Velusamy, N. Kopparthi et al., "Integrating machine learning and
trend analysis for rainfall forecasting: insights from DBSCAN, spectral
clustering, and climate variability assessments over major cities in India,"
Int. J. Climatol., vol. 46, no. 4, Art. no. e70239, 2026, doi:
10.1002/joc.70239.

[5] [V] P. J. Abass and S. Muthulingam, "Selection and thermophysical assessment
of phase change materials (PCMs) for space cooling applications in buildings,"
Numer. Heat Transf. A, Appl., vol. 86, no. 8, pp. 2423–2445, 2025, doi:
10.1080/10407782.2023.2292183.

[6] [P] M. B. Awan, Z. Ma, W. Lin, A. K. Pandey, and V. V. Tyagi, "A
characteristic-oriented strategy for ranking and near-optimal selection of phase
change materials for thermal energy storage in building applications," J. Energy
Storage, vol. 57, Art. no. 106301, 2023, doi: 10.1016/j.est.2022.106301.

[7] [P] C. Oluah, E. T. Akinlabi, and H. O. Njoku, "Selection of phase change
material for improved performance of Trombe wall systems using the entropy
weight and TOPSIS methodology," Energy Build., vol. 217, 2020, doi:
10.1016/j.enbuild.2020.109967. — the TOPSIS unit-test fixture.

[8] [V] M. Yazdani, P. Zaraté, E. K. Zavadskas, and Z. Turskis, "A combined
compromise solution (CoCoSo) method for multi-criteria decision-making
problems," Manag. Decis., vol. 57, no. 9, pp. 2501–2519, 2019, doi:
10.1108/MD-05-2017-0458.

[9] [P] "Comparative framework for climate-responsive selection of phase change
materials in energy-efficient buildings," Energies, vol. 18, no. 22, Art. no.
5982, 2025, doi: 10.3390/en18225982. — the closest prior work; read before
designing the criteria set.

[10] [V] "A criteria-based climate classification approach considering
clustering and building thermal performance: case of India," Build. Environ.,
2024, doi: 10.1016/j.buildenv.2024.112057. — source of the realistic silhouette
expectation.

[11] [V] S. Dhruva, R. Krishankumar, D. Pamucar, E. K. Zavadskas, and K. S.
Ravichandran, "Demystifying the stability and the performance aspects of CoCoSo
ranking method under uncertain preferences," Informatica, 2023.

[12] [V] Y. Zhao et al., "Study on a hybrid solar water heating system with
phase-change material storage tank," J. Cleaner Prod., 2019. — the 47.5–57.5 °C
suitable phase-change band.

[13] [V] V. M. Avargani, B. Norton, M. Rahimi, and G. Karimi, "Integrating
paraffin phase change material in the storage tank of a solar water heater to
maintain a consistent hot water output temperature," J. Energy Storage, 2021. —
the 300 L at 60 ± 2 °C for 7 h benchmark.

[14] [V] EnergyPlus Engineering Reference, "Conduction Finite Difference
Solution Algorithm" and "Water Thermal Tanks (includes Water Heaters)," US DOE /
Lawrence Berkeley National Laboratory. — basis for the EnergyPlus rejection in
§10.1.

[15] [V] India Meteorological Department, "Supply of Meteorological Data," IMD
Data Supply Portal, dsp.imdpune.gov.in. — basis for the IMD access assessment in
§4.1.

[16] [A] B. Singh, R. S. Rai, P. Yadav, S. Srivastava, and C. Yadav,
"Application of phase change materials in solar water heating systems — a
comprehensive review," Sol. Energy Mater. Sol. Cells, vol. 293, Art. no. 113888,
2025.

[17] [A] G.-R. Chen, T.-W. Liao, C.-C. Hsieh, J. Barman, C.-Y. Huang, and C.-F.
J. Kuo, "Using the Taguchi method and grey relational analysis to optimize the
parameter design of flat-plate collectors with nanofluids and phase change
materials in an integrated solar water heating system," Energy Convers. Manage.
X, vol. 26, Art. no. 100910, 2025. — GRA precedent in the project references.

[18] [A] Y. Kou et al., "A novel solar heating building integrated heat pipes
and PCMs: optimizing thermophysical properties and reducing energy consumption,"
Build. Environ., vol. 285, Art. no. 113674, 2025.

[19] [A] L. Liu et al., "The contribution of artificial intelligence to phase
change materials in thermal energy storage: from prediction to optimization,"
Renew. Energy, vol. 238, Art. no. 121973, 2025.

[20] [A] F. A. Barqawi, "Dynamic simulation of phase change material-integrated
solar water heating systems: a machine learning approach to energy conversion
optimization," Muthanna J. Eng. Technol., vol. 13, no. 3, pp. 1–14, 2025.

[21] [D] H. Hersbach et al., "The ERA5 global reanalysis," Q. J. R. Meteorol.
Soc., vol. 146, no. 730, pp. 1999–2049, 2020, doi: 10.1002/qj.3803. — the
primary meteorological source.

[22] [D] NASA Langley Research Center, "POWER (Prediction of Worldwide Energy
Resources) Hourly Data," NASA POWER Project. — the independent cross-check
source.

[23] [D] I. Reda and A. Andreas, "Solar position algorithm for solar radiation
applications," Sol. Energy, vol. 76, no. 5, pp. 577–589, 2004, doi:
10.1016/j.solener.2003.12.003. — the SPA algorithm used via pvlib for sun-event
times.

[24] [D] W. F. Holmgren, C. W. Hansen, and M. A. Mikofski, "pvlib python: a
python package for modeling solar energy systems," J. Open Source Softw., vol.
3, no. 29, p. 884, 2018, doi: 10.21105/joss.00884.

[25] [D] WorldPop, "Global High Resolution Population Denominators Project —
India, 2020, UN-adjusted, 100 m," University of Southampton, doi:
10.5258/SOTON/WP00660. — the population weighting source.

[26] [D] GADM, "Database of Global Administrative Areas, version 4.1," 2022. —
state boundary source.

# 16. Summary of Decisions

| **Question** | **Answer** |

| --- | --- |

| **Is four states enough, or does it need to be all of India?** | Four is defensible and arguably better. Rajasthan, Assam, Tamil Nadu and Uttarakhand cover four of the five or six NBC zones with dense population-weighted sampling inside each. State the missing temperate zone as a limitation. |

| **Is sun-event sampling a problem?** | Not for Tier 1 — it is charge–discharge-cycle-aligned and should be argued for as a design choice. It is insufficient for daily-integral indices, which is what Repair 1 fixes. |

| **Does the Tier 2 repair need a new ERA5 download?** | No. The NASA POWER raw cache already holds the full hourly series for every point and year; the merge step discards it. Reading it back is a few hours of work and no queue time. |

| **Does the flat 300 m elevation matter?** | Yes, in Uttarakhand, which spans roughly 200 m to over 7,000 m. Attach real elevation from ERA5 surface geopotential or an SRTM DEM and recompute solar geometry. Not optional. |

| **Should latitude and longitude be clustering features?** | No. That clusters geography rather than climate and guarantees the states separate, destroying the finding. Coordinates are for the map; elevation is a legitimate physical feature. |

| **Should the GMM be weighted by population?** | No. The sampling is already population-weighted by construction; weighting again double-counts. Use population for reporting and for the recommendation cards. |

| **What does a good clustering result look like here?** | Intra-state splitting and, ideally, cross-state merging. Recovering the four state boundaries alone is not a finding — report ARI against state identity so this is visible. |

| **How should ERA5 and NASA POWER be combined?** | Not blended. Cross-validate at matched instants, then either use ERA5 as backbone with POWER as reported validation, or quantile-map ERA5 onto POWER if a systematic seasonal bias is demonstrated. |

| **What melting temperature should the framework target?** | Tm_target = T_delivery + ΔT_approach ≈ 50–58 °C, not 42–45 °C. The v1.0 sign was wrong: the PCM must sit above the delivery temperature to discharge into the water. |

| **Switch from GMM to K-Means, or add STL?** | No to both. GMM suits a continuous gradient and its soft membership matters at transition points. STL decomposes raw time series; the clustering object is an aggregated signature vector. |

| **Target silhouette above 0.75?** | No. Expect 0.15–0.35, and note that this sampling frame may inflate the value because the territory between states is not sampled. |

| **Replace the four MCDM methods with CoCoSo?** | No — that is a regression. Keep the four-method consensus and add CoCoSo as an optional fifth ranker. |

| **Use EnergyPlus for validation?** | No. It cannot place a latent-heat PCM inside a water tank node network. Use the Python grey-box enthalpy model, optionally cross-checked against TRNSYS Type 860. |

| **How much time is left?** | Twelve weeks. Phases 1 and 2 are substantially complete — roughly five weeks of the previous schedule already banked. |

*Table 23. Decision summary for version 3.0.*

Immediate next actions, in order: write 02b_build_daily_aggregates.py against
the NASA POWER cache, because every Tier 2 index and therefore most of the
clustering signal depends on it; attach real per-point elevation and recompute
solar geometry for Uttarakhand; run the ERA5-versus-POWER agreement analysis and
record the bias decision; and in parallel, on a different pair of hands, build
the PCM database to 40+ cited rows in the 42–70 °C band, since that is now the
critical path.

# 15. Objective1_Section5_Methodology_Update.docx

Source path: /mnt/data/Objective1_Section5_Methodology_Update.docx

Revised §5 — Phase 2: Preprocessing and Cross-Source Validation

Replacement text for §5.1–§5.3 of Objective1_PCM_Climate_Framework_Plan_v3.docx.
Existing §5.3 ("Remaining acceptance checks") is renumbered §5.4 and otherwise
unchanged. Everything below is new or revised.

# 5. Phase 2 — Preprocessing and Cross-Source Validation

Much of the classical preprocessing burden — timezone handling, deaccumulation,
solar geometry, nearest-in-time matching — is already inside the pipeline. What
remains is the validation that turns two downloaded sources into a defensible
single backbone. As reported below, this validation step did its job: it caught
a genuine data-quality fault before it could propagate into the climate
signature, rather than a false alarm.

## 5.1 The ERA5-versus-POWER agreement analysis

This replaces the CERES quantile-mapping step of v2.0 and is the more direct
comparison, because both sources are evaluated at the same point and the same
instant rather than at different resolutions.

For each variable present in both sources (GHI, 2 m temperature, relative
humidity, 10 m wind speed), compute mean bias error (MBE), root-mean-square
error (RMSE) and Pearson correlation (r) between era5_* and power_* columns.

Stratify by season and by state. A bias that appears only in one season is a
different finding from a uniform offset, and the two call for different
responses.

Stratify also by sun event. Sunrise and sunset GHI values are small and
near-zero-crossing; disagreement there is expected and less consequential than
disagreement at solar noon.

Plot ERA5 against POWER as a scatter with the identity line, one panel per state
per season.

## 5.2 The decision rule, and what it triggered

The decision rule specified in v2.0 (reproduced below) anticipated exactly the
failure mode that occurred, and directed the correct response: investigate
rather than average or blend.

## 5.3 Applied finding: an ERA5 accumulation-convention mismatch (detected and corrected)

The first run of the agreement analysis on the merged dataset
(03b_agreement_analysis.py) returned the third row of Table 6: n = 1,168,960
paired noon readings, MBE = −663.67 W/m² (90.1% of mean POWER GHI), r = 0.014,
with the per-season r ranging from −0.02 to 0.33. A bias of this magnitude and a
correlation this close to zero is not consistent with two established,
independently-produced datasets genuinely disagreeing about the weather — it is
consistent with a processing fault on one side, which is exactly the case the
decision rule was written to catch. The analysis was therefore not used to blend
or correct the sources; it triggered a root-cause investigation instead.

### 5.3.1 Diagnosis

The pipeline's deaccumulate() step assumes ssrd (and the other accumulated ERA5
fields, strd and tp) is cumulative since the most recent 00Z/12Z forecast start
— the long-standing ERA5/MARS convention — and recovers an hourly value by
differencing consecutive hours, special-casing the first hour after each reset.
Three checks against the raw cached NetCDF files (read-only, pre-pipeline)
showed this assumption no longer held for the downloaded archive:

Shape test. At a sample Rajasthan grid cell, raw ssrd rose to a peak around
solar noon and fell back toward zero by mid-afternoon — the shape of an hourly
solar flux curve, not a monotonically non-decreasing cumulative total.

Monotonicity test. Across the full spatial grid (992 cells), 34–44% of
consecutive-hour differences were meaningfully negative, checked across every
year and season in the 2016–2025 archive. A true cumulative-since-reset field
can only increase within a 12-hour accumulation window; this fraction should be
~0% (clean resets only at the hour after 00Z/12Z).

Physical-plausibility test. Treating each raw value as already an hourly
quantity (unit conversion only, J/m² → W/m² via ÷3600, no differencing) produced
noon GHI peaks of 856–1007 W/m² and a seasonal pattern matching known Rajasthan
solar climatology: lowest in monsoon-season July (413–712 W/m², heavy cloud
cover), highest in pre-monsoon April/June (879–935 W/m², clear skies and high
sun angle), moderate in winter (623–687 W/m², lower sun angle).

Archive-consistency check. Every cached file's conversion-history timestamp fell
within a single 24-hour window (2026-07-31 to 2026-08-01), confirming the entire
2016–2025 archive was pulled in one bulk download session against one delivery
convention — ruling out a mixed-convention archive as an alternative
explanation.

Conclusion: for this download configuration, the Copernicus Climate Data Store
(CDS) delivered ssrd, strd and tp as already-hourly values rather than as a
running 12-hour cumulative total. Differencing two independent hourly values
against each other, as the legacy deaccumulate() logic did, produced values that
were mostly noise clipped to zero — the direct cause of the near-zero median GHI
and the 13.9%-near-zero-at-solar-noon symptom observed in the first agreement
run. The instantaneous (non-accumulated) variables — 2 m temperature, relative
humidity and 10 m wind — were unaffected, since they follow a separate,
non-differenced extraction path.

### 5.3.2 Correction and scope

deaccumulate() was corrected to apply the J/m² → W/m² unit conversion directly
to each hourly value for ssrd, strd and tp, without differencing, and the full
archive was reprocessed. The fix was applied uniformly across all four states
and all ten years, consistent with the single-download-epoch finding above. All
quantities derived from GHI — DNI, DHI and the clearness index computed from
ERA5 — inherit the correction; GHI_clearsky is unaffected, since it is computed
independently via pvlib rather than derived from ERA5. LW_down (from strd) and
precipitation (from tp) went through the identical code path and are treated as
corrected on the same basis, though they were not separately re-validated
against POWER, since POWER does not provide a matched instantaneous comparison
field for either.

A related but separate issue surfaced during this investigation:
mean_surface_direct_short_wave_radiation_flux (msdwswrf), the variable the
download configuration requests specifically to supply DNI directly, is absent
from the downloaded NetCDF files (data_vars contains only ssrd, strd and tp).
DNI has therefore always been computed via the GHI/cos(SZA) fallback in
compute_solar(), never from the direct CDS variable. This does not block the
correction above and does not affect the GHI/temperature/humidity/wind
quantities used in the climate signature, but it is noted here as a tracked
follow-up before DNI-derived indices are reported with confidence.

### 5.3.3 Verification

The agreement analysis was re-run in full against the corrected dataset.
Solar-noon results, before and after correction:

Per-season noon statistics after correction:

The residual per-season bias — ERA5 running high in Winter, Summer and the
retreat monsoon, and low in the Monsoon season — is small (single-digit percent
of mean GHI) and consistent with the ordinary
reanalysis-versus-satellite-retrieval differences expected between two
independently produced datasets, rather than with a processing fault. Per the
decision rule (Table 6, row 2), this triggered per-season quantile mapping of
ERA5 GHI onto the POWER distribution, evaluated on the daytime (ERA5 GHI > 0)
subset of each season:

Quantile mapping reduced MBE to near zero and RMSE in every season, with r
essentially unchanged (these post-mapping r values are computed on the
daytime-only subset used for the mapping fit, not the all-noon-events
denominator used in the overall figures above, and should be reported with that
distinction in the text). The corrected, quantile-mapped ERA5 GHI is used as the
backbone series for Tier 1 of the climate signature (§6), with NASA POWER
retained as the reported independent validation.

### 5.3.4 How to report this

This is worth a full paragraph in the paper's methodology, not a footnote. It
demonstrates exactly the kind of rigor a reviewer looks for: an automated
cross-source consistency check caught an implausible result rather than silently
accepting it, the root cause was traced to a specific, verifiable mechanism (a
delivery-convention change in the upstream data provider) rather than asserted,
the diagnosis was independently corroborated on three separate grounds
(raw-value shape, monotonicity statistics, physical seasonality) plus an
archive-consistency check ruling out a partial fix, and the correction was
verified by re-running the identical validation pipeline rather than assumed to
have worked. State plainly that this is a documented instance of the two-source
cross-check catching a fault that a single-source pipeline would have carried
silently into the clustering stage.

## 5.4 Remaining acceptance checks

(Unchanged from v3.0 — renumbered from §5.3.)

Plot the mean seasonal cycle of noon GHI and noon temperature for every point,
one panel per state, and inspect by eye.

Verify that the sun-event times behave sensibly across the year: day length
should be longest near the June solstice and shortest near December, with the
amplitude larger in Uttarakhand than in Tamil Nadu.

Confirm the cross-midnight cases: eastern points with a summer sunrise falling
at roughly 23:55 UTC on the previous calendar date should carry the true instant
in time_utc.

Confirm no two population points snap to the same ERA5 cell.

| Finding | Action |
| --- | --- |
| Agreement is close and unbiased | Use ERA5 as the primary backbone; report the agreement as a validation result. No correction needed. |
| A systematic, season-dependent bias appears | Apply quantile mapping of ERA5 GHI onto the POWER distribution, fitted per season per state. Report MBE, RMSE and r before and after. |
| The two disagree severely with no interpretable pattern (this is what happened — see §5.3) | Do not average them. Investigate the merge — a nearest-in-time mismatch, a units error, or an instant-versus-accumulation confusion is far more likely than genuine disagreement between two established datasets. |

| Fixed-weight blending remains rejected. Combining the sources as, for example, 0.6 × ERA5 + 0.4 × POWER has no derivation and would make the resulting dataset impossible to characterise. Either one source is the backbone with the other as validation, or one is quantile-mapped onto the other. There is no third defensible option. |
| --- |

| Metric | Before fix | After fix |
| --- | --- | --- |
| n (paired noon readings) | 1,168,960 | 1,168,960 |
| MBE | −663.67 W/m² (90.1% of mean POWER GHI) | 10.95 W/m² (1.5% of mean POWER GHI) |
| r (overall, noon) | 0.014 | 0.810 |
| Per-season r range | −0.02 to 0.33 | 0.70 to 0.85 |

| Season | MBE (W/m²) | RMSE (W/m²) | r |
| --- | --- | --- | --- |
| Winter | 38.10 | 88.05 | 0.845 |
| Summer | 15.26 | 86.12 | 0.699 |
| Monsoon | −35.78 | 159.16 | 0.747 |
| Retreat monsoon | 26.88 | 105.73 | 0.730 |

| Season | n | MBE before → after | RMSE before → after | r before → after |
| --- | --- | --- | --- | --- |
| Winter | 811,441 | 12.99 → 0.09 | 83.97 → 82.19 | 0.966 → 0.965 |
| Summer | 673,063 | 24.58 → 0.15 | 83.60 → 75.95 | 0.984 → 0.986 |
| Monsoon | 665,947 | −11.47 → −0.11 | 109.93 → 103.57 | 0.954 → 0.960 |
| Retreat monsoon | 699,263 | 26.68 → −0.16 | 88.42 → 77.26 | 0.970 → 0.975 |

# 16. Abdellatif2025PCM_Modeling_Review_summary.md

Source path: /mnt/data/Abdellatif2025PCM_Modeling_Review_summary.md

# Modeling and Performance Analysis of Phase Change Materials in Advanced Thermal Energy Storage Systems: A Comprehensive Review

Authors: Houssam Eddine Abdellatif, Ahmed Belaadi, Adeel Arshad, Mostefa
Bourchak

Year: 2025

Journal/Conference: Journal of Energy Storage, Vol. 121, Article 116517

DOI: https://doi.org/10.1016/j.est.2025.116517

IEEE Citation: H. E. Abdellatif, A. Belaadi, A. Arshad, and M. Bourchak,
"Modeling and performance analysis of phase change materials in advanced thermal
energy storage systems: A comprehensive review," J. Energy Storage, vol. 121, p.
116517, 2025, doi: 10.1016/j.est.2025.116517.

────────────────────────────────────────

## 1. One-Line Summary

This review synthesizes latent and hybrid PCM thermal energy storage
literature—enhancement methods (fins, nanoparticles, metal foam, encapsulation),
numerical models (enthalpy, enthalpy-porosity, LBM, FEM), and
shell-and-tube/hot-water-tank applications—while identifying AI/ML and field
validation as open needs for practical PCM-SWH design.

────────────────────────────────────────

## 2. Problem Being Solved

- PCMs offer high latent heat storage at near-constant temperature but suffer
  from low thermal conductivity (e.g., paraffin 0.15–0.24 W/m·K), phase-change
  leakage, subcooling, and limited latent heat when heavily modified.
- Prior reviews often treat numerical modeling and experimental enhancement
  separately, without a unified comparison of hybrid TES (sensible + latent) vs
  pure latent systems for hot-water and solar applications.
- Engineers lack consolidated guidance on which simulation method (enthalpy,
  enthalpy-porosity, heat capacity, LBM) and which enhancement (fins vs NePCM vs
  metal foam vs encapsulation) to use for a given PCM-SWH design problem.
- Real-world deployment gaps remain: scaling, encapsulation durability,
  metal-foam model realism, nanoparticle-induced latent-heat loss, and limited
  integration of data-driven PCM selection/control.
────────────────────────────────────────

## 3. Key Contributions

1. Integrated review of latent heat TES and hybrid PCM–water tanks with tables
   on shell-and-tube modifications, hybrid tank numerics/experiments, fin
   geometries, and dimensionless groups (Table 11: Nu, Ste, Fo, Bi, Ra, Gr, Pr,
   Re, Str, Ri, Pe, Mix, \(\eta_{ch}\), \(\eta_{storage}\)).
1. Structured survey of PCM enhancement: fins, multi-PCM cascades, nanoparticles
   (NePCM), porous metal matrices, macro/micro/nano-encapsulation, and
   shape-stabilized composites (SS-PCM).
1. Detailed exposition of numerical methods: enthalpy method Eqs. (4)–(8),
   enthalpy-porosity Eqs. (9)–(16) / (27)–(29), heat capacity, FDM, FVM, LBM,
   FEM, and molecular dynamics—with mesh-quality guidance (skewness,
   orthogonality).
1. Thermophysical synthesis for composite PCMs: single NePCM, hybrid NePCM,
   EPCM, porous PCM; effective-property models (e.g., Maxwell, Bruggeman,
   Hamilton–Crosser for nanofluids).
1. Application map for solar and building thermal systems, including solar water
   heating (40–80 °C) and cascaded PCM in solar collector storage tanks; future
   research roadmap explicitly includes machine learning for mushy-zone modeling
   and pilot-scale validation.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

N/A — this is a literature review (48 pages, >300 references). It does not
implement a new physical test rig. It organizes prior work on:

- Shell-and-tube and triplex-tube LHTES units.
- Hybrid latent–sensible tanks (PCM in water storage, macro-encapsulated PCM
  balls, cascaded packed beds).
- Enhancement configurations from cited primary studies
  (longitudinal/angled/tree/stair fins, CuO/graphene NePCM, Al6061 foam, etc.).
### 4b. Mathematical Models & Equations

Sensible heat storage:

- \(Q = m\, c_p\, \Delta T\) — (1)
Thermochemical (illustrative):

- \(A + Q \rightleftharpoons B + C\) — (2)
- \(\mathrm{Ca(OH)_2 \rightleftharpoons CaO + H_2O}\) — (3)
Enthalpy method (Voller-type):

- \(\dfrac{dH}{dt} = \nabla \cdot (k \nabla T)\) — (4)
- \(H(T) = h(T) + \rho\, f(T)\, L\) — (5)
- \(h(T) = \int_{T_m}^{T} \rho\, c\, dT\) — (6)
- \(f(T) = L\) if \(T > T_m\), else \(0\) if \(T < T_m\) (Heaviside) — (7)
- \(\dfrac{\partial h}{\partial t} = \nabla \cdot (\alpha \nabla h) - \rho L
  \dfrac{\partial f}{\partial t}\) — (8)
Enthalpy-porosity (general vector form):

- \(\nabla \cdot (\rho \vec{V}) = 0\) — (9)
- \(\dfrac{\partial (\rho \vec{V})}{\partial t} + \nabla \cdot (\rho \vec{V}) =
  -\nabla p + \mu \nabla^2 \vec{V} - \rho_0 \beta (T - T_{ref}) + S\) — (10)
- \(\dfrac{\partial (\rho H)}{\partial t} + \nabla \cdot (\rho \vec{V} H) = k
  \nabla^2 T - \rho L_f \dfrac{\partial f}{\partial t}\) — (11)
Enthalpy–porosity energy (Cartesian example):

- \(\dfrac{\partial T}{\partial t} + u \dfrac{\partial T}{\partial x} + v
  \dfrac{\partial T}{\partial y} + w \dfrac{\partial T}{\partial z} = \alpha
  \left(\dfrac{\partial^2 T}{\partial x^2} + \dfrac{\partial^2 T}{\partial y^2}
  + \dfrac{\partial^2 T}{\partial z^2}\right) - \rho L_f \dfrac{\partial
  f}{\partial t}\) — (16)
Combined enthalpy formulation:

- \(\dfrac{\partial (\rho H)}{\partial t} + \nabla \cdot (\rho \vec{u} H) = k
  \nabla^2 T - \rho L_f \dfrac{\partial f}{\partial t}\) — (27)
- \(H = h + \Delta H\); \(\Delta H = f L_f\); piecewise \(f(T)\) over
  \(T_{solid}\), mush, liquid — (28)–(29)
Dimensionless groups (Table 11 excerpts):

- \(\mathrm{Nu} = h d / k\); \(\mathrm{Ste} = c_p \Delta T / L\); \(\mathrm{Fo}
  = \alpha t / l^2\); \(\mathrm{Bi} = h l / k\); \(\mathrm{Ra} = g \beta \Delta
  T l^3 / (\nu \alpha)\)
Metal-foam porosity (Calmidi–Mahajan):

- \(a_{sf} = \dfrac{3\pi d_f}{\left[1 - e^{-(1-\varepsilon)/0.04}\right] (0.59
  d_p)^2}\) — (109) (\(e = 0.339\) in related foam correlations)
### 4c. Algorithm / Control Method Steps

N/A — no new control algorithm is implemented. The review surveys optimization
and simulation workflows from cited papers (e.g., Gao et al. multi-objective
optimization of cascaded packed-bed TES: exergy +5%, TES capacity −4%). Future
work (Section 12) recommends:

1. Explore ML/AI for mushy-zone and phase-change modeling.
1. Integrate PCM with batteries and renewables for hybrid energy management.
1. Techno-economic optimization of composition, geometry, and operation.
1. Field pilot experiments with industry partners.
1. Life-cycle assessment of PCM systems.
1. Scale-up manufacturing and modular deployment.
### 4d. Data Sources & Dataset Details

| Source type | Content | Scope |
| --- | --- | --- |
| Prior journal papers (2016–2025 focus) | Experimental and CFD studies on PCM-LHTES | Global; heavy Elsevier/JEST, Appl. Therm. Eng., Renew. Energy |
| Tabulated PCM properties | Organic/inorganic/eutectic lists (Tables 3–4, 6) | Melting ranges ~6–256 °C depending on material |
| Review comparisons | Shell-and-tube (Table 8), hybrid tanks (Tables 9–10), fin surveys (Table 5) | Design-parameter meta-analysis |
| Author’s related work [32] | ANN for inclined-enclosure PCM melting (J. Energy Storage 114, 2025) | Cited as emerging AI-for-PCM example |

No ERA5, NASA POWER, or Indian climate datasets are used in this review itself.

### 4e. Validation Method

N/A as primary research — validation is by synthesis of published studies. The
review reports benchmark outcomes from cited validation papers, for example:

- Santos et al.: enthalpy-method code validated against solidification
  experiments on finned tubes.
- Neri et al.: three numerical models validated with experiments on
  macro-encapsulated PCM in hot-water tank (but only 40% latent heat utilization
  reported).
- Gao et al.: cascaded packed-bed model validated through experiments.
- Lee et al. [242]: numerical vs experimental melting in finned CTES tank.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- Materials tested (surveyed, not single study): Paraffin waxes
  (C\(_n\)H\(_{2n+2}\)), fatty acids (lauric, myristic, palmitic, stearic
  acids), sugar alcohols, salt hydrates, eutectics (CA-MA-PA + exfoliated
  graphite), commercial grades RT15, RT18, RT22 HC, RT27, RT35HC, RT100,
  n-eicosane, erythritol, maleic acid, NaNO\(_3\), etc.
- Melting temperature range: Application bands cited: refrigeration −20 to 5 °C;
  buildings/electronics 5–40 °C; solar water heating 40–80 °C; broader PCM list
  spans ~6–256 °C (Table 3).
- Latent heat: Examples — paraffin ~200 J/g; palmitic acid 206 J/g; erythritol
  340 J/g; maleic acid latent storage density 103 kWh/m³ (Table 2); Jebasingh
  eutectic composite 142.2 / 139.5 J/g melting/solidification.
- Thermal conductivity: Paraffin 0.15–0.24 W/m·K; 10 wt% exfoliated graphite in
  eutectic: 0.149 → 0.180 W/(m·K) (+20.8%); RT100/EG composite shows
  conductivity increasing with packing density; graphene/EG enhancements up to
  5000 W/m·K for nanoparticles (Table 6, material property).
- Specific heat (solid/liquid): Water 4.18 kJ/kg·K (sensible reference); organic
  PCMs typically ~2–2.5 kJ/kg·K in tables; effective \(C_{p,eff}\) used in
  heat-capacity method.
- Density: Water 1000 kg/m³; Al\(_2\)O\(_3\) nanoparticle 3980 kg/m³; paraffin
  ~850–912 kg/m³ range in cited encapsulation studies.
- Performance metrics reported (from cited works aggregated): Melting-time
  reductions up to 80.2% (tree fins), 71% (triplex-layer fins), 68% (5% GNP +
  fins); finned shell-and-tube melting/solidification −52% / −43%; hybrid tank
  latent utilization 40% only (Neri); charging efficiency and \(\eta_{storage}\)
  definitions in Table 11.
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm: No new ML model trained in this review. Surveyed/future: AI and ML
  for PCM selection from large datasets [32]; machine learning and AI
  recommended for mushy-zone modeling (Section 12, item 1). Related author work:
  ANN for PCM melting in inclined enclosures (J. Energy Storage 115750, 2025,
  ref. [32]).
- Input features / state space: Not specified for a unified model — review
  mentions general use of thermophysical databases, climate, and material
  properties for selection tools.
- Output / action space: N/A for this paper.
- Model architecture: N/A — cites ANN application externally; discusses LBM,
  CFD/FLUENT, TRNSYS enthalpy models, MATLAB implementations in literature.
- Hyperparameters: N/A.
- Training data size: N/A.
- Hardware used for training: N/A.
- Performance metrics: N/A — no original ML experiment.
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: N/A as a primary dataset — review cites studies using solar
  irradiation boundary conditions in CFD (e.g., Xie et al. PCM wall study; Liu
  et al. PVT with microencapsulated PCM slurry). Barzin et al. [97] mention
  weather forecast with PCM passive buildings (not detailed here).
- Variables used: Solar radiation / irradiation as boundary input in cited
  PCM-wall and solar-collector simulations; stratification metrics (Mix number,
  Richardson number) for storage tanks.
- Geographic scope: Not focused on a single country; includes global literature.
  Application temperature for solar DHW: 40–80 °C PCM band [108].
- Temporal resolution: N/A at review level; cited TRNSYS/annual climate studies
  use hourly or building-scale timesteps in source papers.
- Time period covered: Literature through 2025 (received Oct 2024, accepted Mar
  2025).
- Clear-sky index / derived metrics: Not computed in this review.
────────────────────────────────────────

## 8. Key Results & Numbers

Aggregated from studies surveyed; all bullets include numeric values reported in
this review.

- Paraffin wax thermal conductivity: 0.15–0.24 W/m·K — core limitation for SWH
  charging rates.
- Sensible heat storage (water): energy density ~70 kWh/m³; maleic acid latent
  ~103 kWh/m³; NaNO\(_3\) latent ~108 kWh/m³ (Table 2).
- Meng et al. (shell-and-tube sensitivity): +50% \(c_p\) → +4% average
  heat-storage rate; +50% latent heat → +6% storage; 1.5× conductivity → nearly
  doubles average heat-storage rate.
- Kirincic et al. (longitudinal fins, paraffin/water): melting time −52%,
  solidification −43% vs plain tube.
- Mhood et al.: optimized fin geometry → melting time reduced up to 50%.
- Song et al. (tree-shaped fins, MTLHS): complete melting time −80.2%.
- Kim et al. (angled fins): θ\(_f\) = −20° → average power +19.3% vs horizontal
  fins.
- Gao et al. (cascaded packed-bed solar heating): exergy efficiency +5%, TES
  capacity −4% after multi-objective optimization.
- Das et al. (graphene nanosheets, 2 vol%): melting time −41% at 60 °C, −37% at
  70 °C HTF.
- Nakhchi et al. (CuO + stair fins): energy storage +9.1%, capacity 474.1 kJ.
- Singh et al. (5% GNP + optimized fins): total melting time −68%.
- Lee et al. (finned CTES, enthalpy-porosity): stratified fin design → mean
  power +156.3%.
- Bouzennada et al. (RT-27, inclined fins): melting time −1.28% to −20.52%;
  stored energy +14.75% to +36.88% (0° fin best).
- Xu et al. (triplex-layer PCMs + fins): melting time reduced up to 71%.
- Yang et al. (paraffin in metal foam, angle): full melting time −12.28% (0°),
  −22.81% (30°), −34.21% (60°) vs reference.
- Jebasingh et al.: 10 wt% exfoliated graphite → \(k\): 0.149 → 0.180 W/(m·K)
  (+20.8%); latent heats 142.2 J/g (melt), 139.5 J/g (solidify).
- Xiao et al. (shape-stabilized PCM): light-to-thermal conversion 66.9% → 94.1%
  with CuS; enthalpy 194.8 J/g after 150 cycles (−5.9% enthalpy, −2.6%
  efficiency).
- Neri et al. (macro-encapsulated PCM in hot-water tank): only 40% of PCM
  latent-heat potential utilized due to thermal transport limits.
- Global energy demand projection cited: +28% by 2050 [361] motivating PCM
  deployment.
- Latent heat energy density: ~5–14× sensible heat (literature comparison);
  thermochemical ~5× phase-change systems (Section 2.4).
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Within reviewed literature: pure PCM vs fin-enhanced,
  NePCM, metal-foam composite, multi-PCM cascade, encapsulated vs
  non-encapsulated, and sensible-only vs latent vs hybrid tanks (Tables 2,
  8–10).
- Proposed method: Not a single proposed device — the review’s synthesis favors
  hybrid latent + sensible tanks, enthalpy-porosity CFD for convection-dominated
  melting, and combined fins + nanoparticles where latent-heat penalty is
  acceptable.
- Improvement margin: Illustrative spans from surveyed papers: melting time −52%
  to −80% with fins; energy storage +9.1% to +36.88% with structured fins/NePCM;
  hybrid cascaded TES exergy +5% with −4% capacity trade-off (Gao et al.).
- Conditions of comparison: Varies by cited study (geometry, PCM type, HTF
  temperature, natural vs forced convection); review emphasizes matching Ste,
  Ra, Fo and mesh quality when cross-comparing CFD results.
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — this paper is a literature review without a new experimental apparatus. It
summarizes hardware from cited work (e.g., shell-and-tube LHTES, hybrid water
tanks with macro-PCM capsules, DSC characterization rigs for RT15/RT22 HC,
finned annular test sections). No Arduino/RPi/DS18B20 deployment in this
article.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- PCM phase transitions remain unsteady and nonlinear; innovative solutions
  still needed for conductivity, leakage, and long-term stability (Abstract,
  Conclusion).
- Sensible heat storage requires large temperature swings for capacity — future
  work should raise capacity without relying only on \(\Delta T\) (Section 11).
- Fin design still needs optimization across operating conditions; multi-PCM
  stages help but optimal stage count is unresolved.
- Nanoparticles often reduce latent heat in experiments; metal-foam numerical
  models need higher geometric realism.
- Encapsulation must become robust, low-cost, long-life for practical scale-up.
- Numerical methods must better capture melted PCM flow and supercooling;
  mushy-zone parameters remain uncertain.
- Gap between laboratory studies and real-world engineering — authors call for
  pilot projects, LCA, and industry collaboration (Sections 10–12).
- Hybrid tanks may use only a fraction of PCM latent capacity (e.g., 40% in Neri
  et al. finding cited).
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Not Relevant (as implemented). The review
  covers passive/active HTF control only through cited studies, not online
  RL/MPC; it lists AI/ML for mushy-zone and system optimization as future work,
  not deployed embedded control.
- RG2 (No integrated PCM–AI–hardware prototype): Partially relevant. Surveys
  RT-series and NePCM literature (e.g., TiO\(_2\)/RT-35HC) aligned with
  Rubitherm/PLUSS selection space, and co-author ANN–PCM melting work [32], but
  no Raspberry Pi / ESP32 closed-loop SWH prototype in this paper.
- RG3 (Poor alignment with household demand patterns): Partially relevant.
  Discusses hot-water tanks, stratification (Mix number, charging efficiency
  \(\eta_{ch}\)), and 40–80 °C solar DHW PCM range, but no morning/evening draw
  profiles or demand-aware control — supports tank/PCM sizing context only.
- RG4 (Limited real-world experimental validation): Highly relevant. Explicitly
  states need for practical experiments, pilot projects, and field validation;
  itself is simulation/literature synthesis. Your FYP field/bench validation
  addresses a gap the authors highlight.
- RG5 (No predictive optimization under climatic uncertainty): Partially
  relevant. Mentions weather forecast + PCM passive building study [97] and AI
  for PCM selection; does not implement ERA5/NASA POWER or irradiance
  forecasting for control — supports your Phase 1b climate + forecast narrative
  as an extension beyond this review.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
| --- | --- | --- |
| \(Q = m c_p \Delta T\) (1) | Sensible energy in tank water | Baseline tank energy without PCM phase change |
| \(H(T) = h(T) + \rho f(T) L\) (5) with (28)–(29) | Total enthalpy with mushy-zone \(f(T)\) | Grey-box PCM state; melting fraction tracking |
| \(\dfrac{\partial (\rho H)}{\partial t} + \nabla\cdot(\rho \vec{u} H) = k\nabla^2 T - \rho L_f \dfrac{\partial f}{\partial t}\) (27) | Enthalpy-porosity energy | scipy.solve_ivp + event at \(T_m\); Barqawi-style model extension |
| \(\mathrm{Ste} = c_p \Delta T / L\) | Sensible/latent coupling | Dimensionless groups for RL reward scaling |
| \(\eta_{storage}(t) = (T_{avg}(t)-T_{ini})/(T_i-T_{ini})\) | Charging efficiency of TES | Metric for comparing PPO vs rule-based charging |
| \(a_{sf}\) foam surface area density (109) | Metal-foam enhanced PCM (if used) | Optional enhancement path vs pure PCM duct |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. M. E. Zayed et al., "Applications of cascaded phase change materials in solar
   water collector storage tanks: A review," Sol. Energy Mater. Sol. Cells, 2019
   [44] — Relevant because: Direct PCM in solar water collector storage tanks
   and cascaded CTSPCM configurations for Indian-relevant SWH literature review.
1. H. Asgharian and E. Baniasadi, "A review on modeling and simulation of solar
   energy storage systems based on phase change materials," J. Energy Storage,
   2019 [35] — Relevant because: Maps PCM-SWH simulation methods (enthalpy,
   effective heat capacity) aligned with your grey-box + CFD validation.
1. L. Kalapala and J. K. Devanuri, "Influence of operational and design
   parameters on PCM based heat exchanger for thermal energy storage – a
   review," J. Energy Storage, 2018 [36] — Relevant because: Shell-and-tube
   PCM-HX design parameters for LHTES in solar thermal systems.
1. A. Arshad et al., "Preparation and characteristics evaluation of mono and
   hybrid nano-enhanced phase change materials (NePCMs) for thermal management
   of microelectronics," Energy Convers. Manage., 2020 [29] — Relevant because:
   RT-35HC / hybrid NePCM property characterization methodology applicable to
   Rubitherm/PLUSS selection.
1. T. Bouhal et al., "PCM addition inside solar water heaters: numerical
   comparative approach," J. Energy Storage, 2018 [337] — Relevant because:
   Numerical PCM integration in solar water heater tanks — close architectural
   analog to your PCM-SWH simulator.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 17. AlMamun2023SWH_StateOfArt_summary.md

Source path: /mnt/data/AlMamun2023SWH_StateOfArt_summary.md

# State-of-the-Art in Solar Water Heating (SWH) Systems for Sustainable Solar Energy Utilization: A Comprehensive Review

Authors: Md. Rashid Al-Mamun, Hridoy Roy, Md. Shahinoor Islam, Md. Romzan Ali,
Md. Ikram Hossain, Mohamed Aly Saad Aly, Md. Zaved Hossain Khan, Hadi M.
Marwani, Aminul Islam, Enamul Haque, Mohammed M. Rahman, Md. Rabiul Awual

Year: 2023

Journal/Conference: Solar Energy, Vol. 264, Article 111998

DOI/Link: https://doi.org/10.1016/j.solener.2023.111998

IEEE Citation: M. R. Al-Mamun et al., "State-of-the-art in solar water heating
(SWH) systems for sustainable solar energy utilization: A comprehensive review,"
Sol. Energy, vol. 264, p. 111998, 2023, doi: 10.1016/j.solener.2023.111998.

────────────────────────────────────────

## 1. One-Line Summary

This comprehensive SWH review catalogs collector types (FPC 45–60%, CPC 30–50%,
ETC up to ~84% above FPC), storage stratification, PCM integration, and
nanofluid gains (MWCNT +35%, Al₂O₃ +28.3%, CuO ETC +12.4%) while identifying
cost, stability, and adoption barriers for residential solar water heating.

────────────────────────────────────────

## 2. Problem Being Solved

- Global fossil fuel reserves may deplete by 2050; forecast energy demand 46 TWh
  (2100) / 30 TWh (2150) drives renewable transition.
- SWH is mature but under-penetrated vs PV due to scattered cost data, low
  awareness, and performance limits of conventional working fluids.
- Heat storage tank stratification losses and low-conductivity PCMs/nanofluids
  limit delivered hot-water temperature and system efficiency.
- Need consolidated design guidance on collectors, nanofluids, PCM tanks, and
  future hybrid SWH research directions.
────────────────────────────────────────

## 3. Key Contributions

1. Component-level review: solar thermal collectors, storage tanks, heat
   exchangers, absorber plates, HTF selection.
1. Collector benchmarking: stationary FPC 45–60% (25–100 °C); tracking CPC
   30–50% (60–300 °C); ETC higher efficiency band (50–200 °C).
1. Nanofluid synthesis: MWCNT, Al₂O₃, CuO, TiO₂, GO, Ag, etc. — quantified
   FPC/ETC/DASC improvements.
1. PCM in SWH: latent storage in tanks/collectors; stratification devices
   (diffusers, baffles, membranes).
1. Future roadmap: hybrid ETC+CPC, nano+PCM fluids, large-scale nanofluid
   stability criteria (cost, surfactant, sedimentation).
1. Market perspective: Arizona case — education/economics drive SWH adoption
   less than PV.
────────────────────────────────────────

## 4. Methodology

- Narrative comprehensive review of peer-reviewed SWH literature (collectors,
  storage, nanofluids, PCM, CFD/TRNSYS modeling).
- Comparative tables of experimental FPC/ETC installations worldwide (outlet
  temperature, irradiance, measured efficiency).
- Critical synthesis of nanofluid preparation, volume fraction, stability, and
  DASC volumetric absorption studies.
- No original experiment — aggregates published performance data and design
  criteria.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- PCMs integrated in storage tanks or collectors to enhance thermal performance
  and reduce tank losses (cites Seddegh et al. latent-HW systems).
- Medium-temperature organic PCMs (paraffin, fatty acids) suited to 60–100 °C
  SWH range.
- Review recommends investigating nanofluids combined with PCMs in SWH loops.
- Salt hydrates and paraffins classified; PCM stratification with tank internals
  reduces mixing losses.
- Not RT35/OM35 specific — project should map Rubitherm/PLUSS products to cited
  medium-T paraffin band.
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A as primary focus — review mentions CFD simulation software and numerical
modeling of SWH but not closed-loop AI control.

- Indirect relevance: calls for optimized heat-transfer modeling; your XGBoost +
  PPO fills the intelligent-control gap this review does not cover.
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- SWH applications cited across China, India, Lebanon, Italy, London, etc.
- Typical test irradiance 300–1100 W/m²; tilt angles 30–60°.
- Operating temperature band 60–280 °C for solar thermal applications broadly;
  domestic SWH <100 °C (FPC) / >150 °C (ETC).
- Project link: aligns with NASA POWER / ERA5 / ISRO Solar Calculator for
  Coimbatore, Jaisalmer, Kochi resource assessment; India receives 4–7
  kWh/m²/day (cited in related Indian literature).
────────────────────────────────────────

## 8. Key Results & Numbers

- FPC thermal efficiency: 45–60% (25–100 °C operating range).
- CPC (single-axis tracking): 30–50% (60–300 °C).
- ETC vs FPC: thermal efficiency up to 84% higher than FPC.
- MWCNT nanofluid (FPC): effectiveness +35%; Al₂O₃: +28.3%.
- CuO/water in ETC: collector efficiency +12.4%.
- Al₂O₃/oil nanofluid: collector efficiency 23.83% reported experiment.
- Al₂O₃/synthetic oil: relative thermal efficiency +11% vs conventional fluid.
- Graphite nanofluid (0.01 vol%): 122.7% efficiency metric vs baseline in cited
  study.
- MWCNT DASC: +10–29% vs water base fluid.
- RGO/water-EG DASC: 70% efficient at 1000 W/m².
- Example FPC lab results: 61.59–73.45% (Table 2 studies); minichannel FPC
  +16.1% thermal efficiency gain.
- Petroleum consumption 105× faster than renewable production (cited forecast).
────────────────────────────────────────

## 9. Baseline Comparison

| Enhancement | Baseline | Improvement |
| --- | --- | --- |
| MWCNT nanofluid FPC | Water HTF | +35% efficiency |
| Al₂O₃ nanofluid FPC | Water HTF | +28.3% |
| CuO nanofluid ETC | Water HTF | +12.4% |
| ETC collector | FPC | Up to +84% efficiency |
| MWCNT DASC | Water | +10–29% |
| Stratified tank + internals | Mixed tank | Reduced heat loss (qualitative) |
| SWH vs PV adoption | PV growth | SWH "limited growth since 1970" |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

Review compiles diverse rigs:

- FPC: corrugated tubes, heat-pipe absorbers, minichannel absorbers, roll-band
  absorbers.
- ETC: flood-design, all-glass vacuum tubes.
- DASC: volumetric nanofluid absorption cells.
- Working fluids: water, air, glycol, hydrocarbon, nanofluids.
- Sensors/methods: outlet temperature, flow rate, solar irradiance — typical
  experimental SWH loops.
- No embedded RPi/Arduino — supports your custom bench instrumentation as novel
  contribution.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Nanofluid agglomeration, sedimentation, cost, viscosity, surfactant stability
  limit scale-up.
- SWH market growth lags PV despite technical maturity.
- Need holistic studies: cost, lifetime, usability vs electric/LPG heaters.
- PCM+nano combinations need more experimental validation in SWH loops.
- Performance-focused reviews alone won't drive residential adoption without
  policy/economics.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1: Gap confirmed — review lacks real-time adaptive control; motivates PPO
  valve policy.
- RG2: Relevant — PCM in tanks/collectors and sensor-based modeling cited; your
  PCM + AI + ESP32/RPi prototype is explicitly missing.
- RG3: Relevant — domestic hot-water use cases (bathing, washing) and stratified
  storage align with evening demand PCM discharge.
- RG4: Relevant — extensive experimental FPC/ETC benchmarks (45–73%) for
  field/lab comparison; India-friendly deployment context.
- RG5: Partial — intermittent solar emphasis supports climate-adaptive
  forecasting; no ERA5/NASA workflow — your ERA5 + irradiance ML fills this.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Collector efficiency (standard test form, from cited SWH literature):

\[

\eta_{th} = F_R\left[\tau\alpha - U_L \frac{(T_{in}-T_{amb})}{G}\right]

\]

Energy stored in PCM (latent):

\[

Q_{stored} = m \cdot L \quad \text{(plus sensible terms over } T_m \text{)}

\]

Nanofluid effective conductivity (mixture models in review):

\[

k_{nf} = \phi k_p + (1-\phi) k_f

\]

Use \(\eta_{th}\) vs \((T_{in}-T_{amb})/G\) for grey-box validation against
FPC/ETC curves.

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Gautam et al., SWH technical improvements review, Renew. Sust. Energy Rev.,
   2017 — prior SWH survey [7].
1. Seddegh et al., latent-heat SDHW systems, Renew. Sust. Energy Rev., 2015 —
   PCM tank SWH [15].
1. Yousefi et al., Al₂O₃–H₂O nanofluid FPSC, Renew. Energy, 2012 — +28%
   nanofluid benchmark [22].
1. Mehmood et al., heat-pipe ETC SWH with gas backup, Energy Rep., 2019 — HP-ETC
   performance [13].
1. Sharma, residential SWH adoption Arizona, J. Clean. Prod., 2021 — market
   barriers [202].
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I: Cite global energy demand and SWH as dominant low-temperature solar
  application (60–280 °C band).
- Section II: Lit-review table row — collector efficiencies FPC 45–60%, ETC +84%
  vs FPC, nanofluid uplifts.
- Section III: Justify PCM storage tank and stratification in grey-box; optional
  nanofluid as future work.
- Section IV: Benchmark collector η_th against cited 69–73% heat-pipe FPC
  experiments.
- Section V: Compare system COP/energy savings to +12.4% (CuO ETC) and +35%
  (MWCNT) as aspirational HTF enhancement bounds.
────────────────────────────────────────

# 18. Assareh2023EnhancingSolarThermalPCM_summary.md

Source path: /mnt/data/Assareh2023EnhancingSolarThermalPCM_summary.md

# Enhancing Solar Thermal Collector Systems for Hot Water Production Through Machine Learning-Driven Multi-Objective Optimization with Phase Change Material (PCM)

Authors: Ehsanolah Assareh, Amjad Riaz, Mehrdad Ahmadinejad, Siamak Hoseinzadeh,
Mohammad Zaheri Abdehvand, Sajjad Keykhah, Tohid Jafarinejad, Rahim Moltames,
Moonyong Lee

Year: 2023

Journal/Conference: Journal of Energy Storage, Vol. 73, Article 108990

DOI: https://doi.org/10.1016/j.est.2023.108990

IEEE Citation: E. Assareh et al., "Enhancing solar thermal collector systems for
hot water production through machine learning-driven multi-objective
optimization with phase change material (PCM)," J. Energy Storage, vol. 73, p.
108990, 2023, doi: 10.1016/j.est.2023.108990.

────────────────────────────────────────

## 1. One-Line Summary

This paper uses MATLAB-based MOEA/D multi-objective optimization (plus RSM,
TOPSIS, LINMAP, and AHP) on a flat-plate solar collector with PCM hot-water
storage to trade off PCM discharge duration \(t_{PCM}\) versus net stored energy
\(Q_{net}\), showing inverse Pareto coupling and sensitivity to tube diameter
and collector area.

────────────────────────────────────────

## 2. Problem Being Solved

- Solar thermal collectors with PCM storage require appropriate energy discharge
  time \(t_{PCM}\) and net stored energy \(Q_{net}\) in the PCM, but these
  objectives conflict under design and operating parameter choices.
- Prior work optimized collectors or PCM separately rather than jointly
  optimizing collector geometry, tank/contact area, and PCM storage behavior for
  hot-water production.
- Night-time hot-water availability depends on how long PCM can discharge
  latent/sensible heat after sunset, while daytime charging must maximize stored
  energy—requiring multi-objective, not single-objective, design.
- Lack of integrated decision support linking tube diameter, collector area
  \(A_c\), and PCM class selection to both discharge duration and stored energy
  in one framework.
────────────────────────────────────────

## 3. Key Contributions

1. Integrated flat-plate collector + PCM tank model (disodium hydrogen phosphate
   dodecahydrate baseline PCM) with energy-balance equations for useful
   collector gain and PCM discharge time (1)–(8).
1. MOEA/D decomposition of the bi-objective problem (minimize \(t_{PCM}\),
   maximize \(Q_{net}\)) with decision variables: tube inner diameter, contact
   area, collector area \(A_c\), and PCM-minus-water stored energy band.
1. Pareto analysis (500 population points) proving inverse relationship: maximum
   \(t_{PCM}\) aligns with minimum \(Q_{net}\), and vice versa.
1. Parametric studies on tube diameter (nonlinear increase in \(t_{PCM}\) and
   \(Q_{net}\)) and storage contact area (linear sensitivity; \(Q_{net}\) more
   sensitive than \(t_{PCM}\)).
1. Comparison of three PCM classes (hybrid salt, paraffin, fatty acid) plus
   post-optimization screening via RSM, TOPSIS, LINMAP, and AHP for Pareto-point
   selection.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

- Configuration (Fig. 1): Flat-plate solar collector; water storage/PCM tank
  with disodium hydrogen phosphate PCM; piping, valves, and bypass line for
  night-time network-water return to extract stored PCM heat after sunset.
- Day operation: Purified water flows through collector tubes, enters PCM tank,
  transfers heat to solid PCM (29 °C melting temperature stated in §2.2
  narrative; 35 °C in Table 5 for dodecahydrate salt).
- Night operation: Network water returns via bypass, extracts energy from
  liquefied PCM, then flows to consumption loops.
- Software: MATLAB simulator; MOEA/D for multi-objective search; RSM for
  response-surface refinement; TOPSIS, LINMAP, AHP for multi-criteria
  decision-making on Pareto solutions.
- Assumptions (§2.1): One-dimensional flow; sky treated as black body for
  long-wave radiation; certain collector loss properties taken
  temperature-independent.
- No physical test rig in this paper—computational optimization study with
  validation against published thermal data only.
### 4b. Mathematical Models & Equations

Useful collector energy (Hottel–Whillier–Bliss form):

- \(Q_u = A_c F_R \left[ S - U(T_c - T_a) \right]\) — (1)
Heat removal factor:

- \(F_R = \dfrac{\dot{m} C_p}{A_c U}\left[1 - e^{-\left[A_c U F' / (\dot{m}
  C_p)\right]}\right]\) — (2)
Collector efficiency factor:

- \(F' = \dfrac{1}{\frac{1}{U}\left[\frac{1}{t_W}\left(U_L (D + (W-D)F)\right) +
  \frac{1}{\pi D_i h_f}\right]}\) — (3)
Top loss coefficient (radiative/convective glazing model):

- \(U_t = \dfrac{\sigma(T_a + T_p)(T_a^2 + T_p^2)}{\left[\dfrac{1}{\varepsilon_p
  + 0.0425N(1-\varepsilon_p)^{-1}} + \dfrac{2N+f-1}{\varepsilon_g} -
  N\right]^{-1}}\) — (4)
(\(N\) = number of glass covers; \(\varepsilon_p\) = plate emissivity;
\(\varepsilon_g = 0.88\); \(T_a\) ambient; \(\beta\) slope; \(h_w\) wind
coefficient)

Collector loss / geometric factor:

- \(Q_L = U_l (T_{in} - T^\circ)\) — (5)
- \(f = \left(1 + 0.089 h_w - 0.1166 h_w / \varepsilon_p\right) + (1 + 0.07866
  N)\) — (6)
Optimization objectives:

- \(Q_{net} = Q_u + Q_L\) — (7)
- \(t_{PCM} = Q_{net} / Q_u\) — (8)
(\(t_{PCM}\) = duration water can stay warm without solar input)

Multi-objective problem:

- \(\min \mathbf{F}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots,
  f_m(\mathbf{x}))^T\) subject to \(\mathbf{x} \in \mathbb{R}^m\) — (9)
(implemented as MOEA/D scalar sub-problems with weight vectors)

### 4c. Algorithm / Control Method Steps

1. Define decision vector bounds (Table 1): diameter 0.005–0.02 m, area 0.2–1
   m², \(A_c\) 0.26–0.38 m², PCM-minus-water energy 3100–3300 kJ.
1. For each candidate design, compute \(Q_u\), \(Q_L\), \(Q_{net}\), and
   \(t_{PCM}\) via (1)–(8) under stated operating constants (e.g.,
   \(\dot{m}_w\), \(m_{PCM}\), \(T_{in,water}\)).
1. Run MOEA/D: decompose bi-objective problem into weighted scalar sub-problems;
   evolve population (500 solutions reported for Pareto plot).
1. Extract Pareto front relating \(t_{PCM}\) vs \(Q_{net}\); identify design
   points for target night discharge (6 h or 7 h).
1. Apply RSM to build empirical response surfaces between inputs (diameter,
   area) and objectives.
1. Rank/select Pareto points using TOPSIS, LINMAP, and AHP (ideal vs
   negative-ideal distance logic).
1. Repeat parametric sweeps for tube diameter and \(A_c\) with other parameters
   fixed (Tables 3–4).
1. Compare alternate solid PCMs (hybrid salt, paraffin, fatty acid) at matched
   conditions (Figs. 11–12).
No real-time control loop, reinforcement learning, or online learning steps are
implemented.

### 4d. Data Sources & Dataset Details

| Source | Variables | Resolution | Scope | Period / size |
| --- | --- | --- | --- | --- |
| MATLAB thermal model (this study) | \(Q_u\), \(Q_L\), \(Q_{net}\), \(t_{PCM}\), \(T_{out}\) | Hourly time tags in validation table | Generic flat-plate + PCM tank | Parametric runs; 500 MOEA/D population |
| Luo et al. [55] (2021) | Collector outlet temperature vs time | 6:00–18:00 hourly | Air-type double-pass collector with PCM rod | Used for model validation (Table 2) |
| Fixed operating set (Table 3) | \(\dot{m}_w = 0.009\) kg/s, \(A_c = 0.287\) m², \(m_{PCM} = 10\) kg, \(T_{in,water} = 293.05\) K | Steady/parametric | Single baseline case | Diameter sensitivity runs |

No ERA5, NASA POWER, ISRO, or site-specific Indian weather series used.

### 4e. Validation Method

- Literature temperature benchmark against Luo et al. (2021) outlet temperatures
  (Table 2): e.g., 6:00 — current 305 vs 306; 12:00 — 375 vs 374; 18:00 — 330.5
  vs 330 (units printed as °C in table; values correspond to ~32–102 °C if
  interpreted as K minus offset—authors state “good accuracy” of thermal
  modeling).
- MOEA/D internal consistency: Pareto set of 500 designs; decision-making
  methods (RSM/TOPSIS/LINMAP/AHP) converge to coincident optimal point on
  response surface (Fig. 10).
- No RMSE/R² reported for optimization; no experimental validation of the
  optimized Assareh system in field or lab.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- Materials tested: Disodium hydrogen phosphate dodecahydrate (hybrid salt,
  baseline in §2.2); additionally paraffin (C20–C33) and uric acid (fatty acid)
  in comparative study (Table 5, Figs. 11–12).
- Melting temperature range: Baseline narrative 29 °C (§2.2); Table 5 hybrid
  salt 35 °C; paraffin 50 °C; fatty acid 44 °C.
- Latent heat: Hybrid salt 278.84 kJ/kg; paraffin 189 kJ/kg; fatty acid 178
  kJ/kg.
- Thermal conductivity: Not reported in Table 5.
- Specific heat (solid/liquid): Hybrid salt 1.55 / 2.51 kJ/kg·K; paraffin 2.4 /
  2.4 kJ/kg·K; fatty acid 1.7 / 2.3 kJ/kg·K.
- Density: Hybrid salt 1522 kg/m³; paraffin 912 kg/m³; fatty acid 862 kg/m³.
- Performance metrics reported: \(t_{PCM}\) target 6 h or 7 h night discharge;
  \(Q_{net}\) 3094 kJ (7 h case) vs 3200 kJ (6 h case) at RSM/DM optimum; hybrid
  salts yield largest increases in \(t_{PCM}\) and \(Q_{net}\) vs paraffin/fatty
  acids at high melting temperature (Figs. 11–12); \(m_{PCM} = 10\) kg in
  parametric tables.
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm: MOEA/D (multi-objective evolutionary algorithm based on
  decomposition); supporting methods: RSM, TOPSIS, LINMAP, AHP — not neural
  networks, PPO, DDPG, or XGBoost despite “machine learning-driven” wording in
  the title.
- Input features / state space: Design/decision variables: tube inner diameter
  \(D\), storage contact area \(A\), collector area \(A_c\), band on
  PCM-minus-water stored energy (Table 1); fixed runs use \(\dot{m}_w\),
  \(m_{PCM}\), \(T_{in,water}\), solar/input parameters embedded in \(Q_u\).
- Output / action space: Pareto-optimal \(t_{PCM}\) (minimize) and \(Q_{net}\)
  (maximize); selected operating points for 6–7 h discharge scenarios.
- Model architecture: N/A — no ANN/CNN; empirical RSM surrogate models for
  response surfaces after MOEA/D.
- Hyperparameters: MOEA/D population 500 (Fig. 5); weight vectors per
  sub-problem (standard MOEA/D); RSM/AHP/TOPSIS/LINMAP procedural parameters not
  numerically tabulated.
- Training data size: 500 Pareto population members; no supervised ML train/test
  split.
- Hardware used for training: N/A — MATLAB simulation on unstated compute.
- Performance metrics: Pareto trade-off curves; RSM optimum \(Q_{net} = 3094\)
  kJ at \(t_{PCM} = 7\) h and 3200 kJ at 6 h; linear \(t_{PCM}\) and \(Q_{net}\)
  vs \(A_c\); nonlinear increasing \(Q_{net}\) vs diameter.
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: Not stated — solar input enters through term \(S\) (irradiance,
  W/m²) inside \(Q_u\) (1) without naming NSRDB, measured weather files, or
  satellite products.
- Variables used: Implicit solar gain \(S\), ambient \(T_a\), collector \(T_c\),
  wind-related \(h_w\) in loss coefficients.
- Geographic scope: Not stated — no city, climate zone, or country-specific
  weather series.
- Temporal resolution: Hourly comparison points in validation table
  (6:00–18:00); \(t_{PCM}\) in hours.
- Time period covered: Validation references one-day profile from Luo et al.
  (2021); optimization runs are parametric, not multi-year.
- Clear-sky index / derived metrics: Not computed.
────────────────────────────────────────

## 8. Key Results & Numbers

- MOEA/D Pareto population: 500 designs; inverse trade-off — maximum \(t_{PCM}\)
  (left side of Fig. 5) pairs with minimum \(Q_{net}\); maximum \(Q_{net}\)
  yields shortest night-time energy availability.
- Target \(t_{PCM} = 7\) h (RSM/decision-making optimum): \(Q_{net} = 3094\) kJ
  (Fig. 10).
- Target \(t_{PCM} = 6\) h: \(Q_{net} = 3200\) kJ — 106 kJ higher stored energy
  for 1 h shorter discharge target.
- Increasing tube inner diameter increases \(t_{PCM}\) nonlinearly (Fig. 7) and
  increases \(Q_{net}\) nonlinearly (Fig. 9); \(Q_{net}\) more sensitive at
  larger diameters.
- Contact area \(A\): both \(t_{PCM}\) and \(Q_{net}\) vary linearly with area;
  \(Q_{net}\) line steeper (more sensitive) than \(t_{PCM}\) (Fig. 8).
- Decision-variable bounds: diameter 0.005–0.02 m; area 0.2–1 m²; \(A_c\)
  0.26–0.38 m²; stored-energy band 3100–3300 kJ (Table 1).
- Parametric baseline (Table 3): \(\dot{m}_w = 0.009\) kg/s, \(A_c = 0.287\) m²,
  \(m_{PCM} = 10\) kg, \(T_{in,water} = 293.05\) K (~20 °C).
- Validation vs Luo et al.: outlet temperature agreement within ~1–1.5 units at
  most hours (e.g., 358.9 vs 360 at 10:00; 375 vs 374 at 12:00) — Table 2.
- PCM class comparison: hybrid salts produce much greater increases in
  \(t_{PCM}\) and \(Q_{net}\) than paraffin or fatty acids at high melting
  temperature; hybrid salt latent heat 278.84 kJ/kg vs paraffin 189 kJ/kg and
  fatty acid 178 kJ/kg.
- Literature benchmarks cited (not this paper’s direct results): Lin et al.
  heat-transfer effectiveness 44.25% → 59.29%; Shamsi et al. +5% discharged
  energy over 8 h cycle and +5.12% stored in 4 h charge vs single PCM.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Non-optimized / Pareto extremes on MOEA/D front (max
  \(t_{PCM}\) vs max \(Q_{net}\)); implicit comparison among PCM material
  classes (hybrid salt vs paraffin vs fatty acid); validation reference Luo et
  al. (2021) thermal profile only.
- Proposed method: MOEA/D multi-objective design optimization +
  RSM/TOPSIS/LINMAP/AHP decision-making on Pareto solutions.
- Improvement margin: At fixed discharge requirement, 6 h vs 7 h target changes
  optimum \(Q_{net}\) by 3200 − 3094 = 106 kJ (~3.4% relative to 7 h case);
  material switch to hybrid salt gives largest \(t_{PCM}\) and \(Q_{net}\) vs
  other PCM classes (qualitative ranking, Figs. 11–12 — no single % tabulated).
- Conditions of comparison: Same MATLAB energy-balance model; parametric
  constants in Tables 3–4; PCM properties from Table 5.
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — this paper is purely simulation/optimization-based in MATLAB. No sensors
(DS18B20, pyranometer), actuators (solenoid valves), embedded platforms
(RPi/Arduino/ESP32), or field/lab test duration is reported. Physical system
description (Fig. 1) is conceptual for modeling only.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Authors do not include a dedicated “Limitations” section; the following are
  explicit scope/assumption statements only.
- Modeling assumes one-dimensional flow, a black-body sky, and collector loss
  properties independent of temperature (§2.1), which may reduce fidelity under
  real variable weather and temperature-dependent losses.
- “Data will be made available on request” — no public dataset or
  reproducibility package is provided in the article.
- Comparative PCM study selects one random material per class (hybrid salt,
  paraffin, fatty acid), not an exhaustive PCM database (§3.2 narrative).
- Conclusion focuses on design-parameter sensitivity (diameter, area) and does
  not claim field demonstration of optimized hardware.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Not Relevant. MOEA/D is offline design
  optimization; there is no hourly/online controller, DRL policy, or pump/valve
  actuation loop comparable to your PPO charge/discharge/bypass agent.
- RG2 (No integrated PCM–AI–hardware prototype): Not Relevant. Work stops at
  MATLAB simulation; no Raspberry Pi, ESP32, or sensor-actuator integration
  despite being a PCM–solar hot-water architecture similar in schematic to your
  FYP.
- RG3 (Poor alignment with household demand patterns): Partially relevant.
  Optimizing \(t_{PCM}\) for 6–7 h night-time thermal availability loosely
  aligns with evening/morning hot-water needs, but there is no measured
  residential draw profile, flow schedule, or occupant-driven demand model
  (Coimbatore/Jaisalmer/Kochi).
- RG4 (Limited real-world experimental validation): Relevant (as gap exemplar).
  Only literature temperature comparison (Luo et al.) is shown; the optimized
  Assareh system is not built or field-tested, mirroring the simulation-heavy
  literature your prototype addresses.
- RG5 (No predictive optimization under climatic uncertainty): Not Relevant.
  Solar input \(S\) is not tied to ERA5/NASA POWER/forecasting; no
  uncertainty-aware or predictive dispatch—static parametric and evolutionary
  design only.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
| --- | --- | --- |
| \(Q_u = A_c F_R [S - U(T_c - T_a)]\) (1) | Flat-plate useful solar gain | Couple pyranometer GHI to collector thermal input in grey-box SWH |
| \(F_R = \frac{\dot{m} C_p}{A_c U}[1 - e^{-A_c U F'/(\dot{m} C_p)}]\) (2) | Heat removal factor | Tank–collector coupling in enthalpy balance |
| \(t_{PCM} = Q_{net}/Q_u\) (8) | Night discharge duration metric | RL reward: maximize hot-water availability hours after sunset |
| \(Q_{net} = Q_u + Q_L\) (7) | Net stored energy objective | PCM stored-energy state for PPO observation / reward |
| MOEA/D \(\min \mathbf{F}(\mathbf{x})\) (9) | Multi-objective design trade-off | Offline NSGA-II/PSO baseline vs online PPO for same \((t_{PCM}, Q_{net})\) objectives |
| \(U_t\) radiative loss (4) | Collector top losses | Optional detailed collector loss if moving beyond lumped model |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Q. Luo et al., "Thermal modeling of air-type double-pass solar collector with
   PCM-rod embedded in vacuum tube," Energy Convers. Manag., 2021 [55] —
   Relevant because: Direct PCM–collector validation benchmark used in this
   paper’s temperature accuracy check.
1. W. Lin et al., "Multi-objective optimisation of thermal energy storage using
   phase change materials for solar air systems," Renew. Energy, 2019 [23] —
   Relevant because: Prior MO + PCM study reporting 44.25% → 59.29%
   heat-transfer effectiveness and 4.53 → 6.11 h charging time improvements.
1. M. Mahfuz et al., "Performance investigation of TES with PCM for solar water
   heating application," Int. Commun. Heat Mass Transf., 2014 [26] — Relevant
   because: Shell-and-tube PCM–SWH experimental lineage closest to domestic
   hot-water storage.
1. A. Mourad et al., "Recent advances on the applications of phase change
   materials for solar collectors," J. Energy Storage, 2022 [19] — Relevant
   because: Review of PCM–solar collector limits and practical constraints for
   literature review framing.
1. M.H. Zahir et al., "Challenges of PCMs to achieve zero energy buildings under
   hot weather," J. Energy Storage, 2023 [8] — Relevant because: Hot-climate
   PCM–building context analogous to Coimbatore/Jaisalmer deployment challenges.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 19. Barghi2026SolarDrying_PCM_AI_summary.md

Source path: /mnt/data/Barghi2026SolarDrying_PCM_AI_summary.md

# Thermal Energy Storage-Centric Solar Drying with Phase Change Materials: Intelligent Optimization via Neural and Evolutionary Regression Models

Authors: Mohammad Saleh Barghi Jahromi, Ayla Sayedolasgari, S. Madhankumar, Hadi
Samimi Akhijahani, Payman Salami

Year: 2026 (online Nov 2025)

Journal/Conference: Journal of Energy Storage, Vol. 141, Article 119192

DOI/Link: https://doi.org/10.1016/j.est.2025.119192

IEEE Citation: M. S. Barghi Jahromi et al., "Thermal energy storage-centric
solar drying with phase change materials: Intelligent optimization via neural
and evolutionary regression models," J. Energy Storage, vol. 141, p. 119192,
2026, doi: 10.1016/j.est.2025.119192.

────────────────────────────────────────

## 1. One-Line Summary

This review synthesizes PCM-buffered solar dryers (palmitic acid, paraffin,
micro-PCM) with ANN, SVM, LSTM, RF, CatBoost, and EPR surrogates—reporting
collector gains up to 66.52%, drying-time cuts 63%, EPR R² > 0.98, and ANN
metrics up to R² = 0.9999—while noting dataset scarcity for embedded real-time
PCM control.

────────────────────────────────────────

## 2. Problem Being Solved

- Conventional food drying consumes large fossil energy; post-harvest losses
  30–40% without adequate preservation.
- Solar dryers suffer intermittent irradiance and unstable chamber temperatures.
- PCM thermal storage can extend operation after sunset but adds design
  complexity (placement, thickness, melting point).
- Physics-based CFD models are accurate but slow; pure empiricism lacks
  generalization — need ML + grey-box (EPR) tools for PCM dryer/collector
  optimization.
────────────────────────────────────────

## 3. Key Contributions

1. PCM integration taxonomy for solar dryers: absorber-mounted, plenum, cabinet
   walls, copper-tube encapsulation.
1. Quantified PCM benefits across studies: stable temperature, −63% drying time,
   +7.81% drying efficiency, night-time heat release.
1. ML algorithm survey: DT, RF, SVR, KNN, ANN/BPNN, FNN, RNN, LSTM, CatBoost,
   hybrid ANN-KNN.
1. EPR (Evolutionary Polynomial Regression) as interpretable grey-box: R² > 0.98
   for outlet temperature and thermal efficiency vs CFD (R² > 0.94) and ANN (R²
   ≤ 0.99).
1. Case study hub: authors' Jerusalem artichoke ETC+PCM dryer — payback 22
   months (15% improvement), EPR R² > 0.98.
1. Co-authorship link: Sri Krishna College of Engineering and Technology,
   Coimbatore — regional relevance to project city.
────────────────────────────────────────

## 4. Methodology

- Narrative review of solar dryer classifications (direct, indirect, mixed-mode,
  greenhouse, CPV/T).
- PCM selection criteria: \(T_m\) near operating temperature, latent heat,
  conductivity, encapsulation geometry.
- ML pipeline patterns: experimental/CFD data → train ANN/SVM/LSTM → predict MC,
  MR, \(T_{out}\), \(\eta_{th}\) → compare RMSE, MAPE, \(R^2\).
- EPR: GA searches equation structure; least-squares fits coefficients (Eq. 18).
- Benchmark tables (Table 9 PCM studies; Table 10 ANN in dryers).
────────────────────────────────────────

## 5. PCM Details (if applicable)

| Study / PCM | Configuration | Key numbers |
| --- | --- | --- |
| Mixed-mode coffee dryer [104] | PCM unit + air recycling | Charge 0.033–0.161 kWh; discharge 0.051–0.237 kWh; collector η 55.27–66.52% (0% recycle best) |
| Palmitic acid + graphene–Al₂O₃ nanofluid [105] | Flat-plate air collector | 1.5 vol% optimal; \(k=0.75\) W/m·K; \(\eta_{th}=62.5%\), exergy 20.7%, dryer η 46.8% |
| PVT air collector [59] | PCM thickness sweep | 0.005 m layer → 151 Wh thermal output vs 75 Wh (0.02 m) |
| Mango pulp [109] | 200 g micro-PCM | >5 h above 60 °C; drying time −63% vs no PCM |
| Paraffin in copper tubes [102] | 176 g / 494 g PCM | Stored 3.52 kJ, released 4.73 kJ; exergy η 21% → 28% |
| Barghi ETC cabinet dryer [9] | PCM for Jerusalem artichoke | SEC 14.51 → 13.38 MJ/kg; exergy η 35.3–59.7%; activation energy 33.4 kJ/mol |
| NePCM Al₂O₃-paraffin CPV/T [44] | Nano-enhanced PCM | Thermal η 20%, exergy 8%; ANN R²=0.999 vs SVM 0.974 for MC |

Placement insight: absorber-mounted PCM melts faster; plenum PCM releases longer
after 20:00 but lower peak \(T\).

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

| Method | Application | Reported performance |
| --- | --- | --- |
| Decision Tree | Peanut solar dryer | R² = 0.9972 [112] |
| ANN (1-7-5, LM) | PCM solar collector | R² = 0.832–0.899 [195] |
| ANN | TES collector heating capacity | RMSE 7840.56, R² = 0.9995; efficiency R² = 0.9999 [197] |
| ANN vs CFD | Dehydration system | ANN more accurate, faster, cheaper than CFD [196]; η 21.11–25.20% |
| ANN vs SVM | CPV/T + NePCM mushroom drying | ANN R² = 0.999 (MC), beats SVM |
| LSTM / RNN | Solar radiation, drying kinetics | Strong on time-series; humidity RMSE < 0.645 cited |
| CatBoost | Red beetroot drying | Beats XGBoost/LightGBM on R², MSE, MAE [191] |
| EPR | Collector \(T_{out}\), \(\eta_{th}\) with PCM | R² > 0.98; beats CFD R² > 0.94; competitive with ANN |
| RF | Solar water heating performance | Cited [119086] in references |

No DRL/PPO — feedforward ANN used for solar transients [177–179]; gap for your
valve control.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Inputs across studies: solar irradiance \(I_g\), ambient \(T_a\), humidity,
  air velocity, recycling ratio, incidence/slope/azimuth.
- Mixed-mode dryer [104]: charging 09:00–14:30, discharging until 18:00.
- Coimbatore co-author institution — aligns with humid tropical drying/solar
  conditions in your Kochi/Coimbatore tests.
- No ERA5/NASA POWER — uses on-site pyranometer/logging per cited experiments.
────────────────────────────────────────

## 8. Key Results & Numbers

- Post-harvest losses without drying: 30–40%.
- Air recycling 0% vs 100%: collector efficiency 66.52% vs 55.27% with PCM
  [104].
- Hybrid nanofluid 1.5 vol%: conductivity 0.75 W/m·K, heat transfer 345.5 W,
  heat loss 58.7 W.
- PCM thickness 0.005 m vs 0.03 m: thermal output 151 Wh vs 55 Wh.
- Mango drying: time reduction up to 63% with PCM.
- Banana/paraffin PCM2: exergy efficiency 28% vs 21% without PCM.
- Barghi PCM dryer: payback 22 months (15% better); SEC 13.38 MJ/kg; drying
  efficiency gain 1.51–7.81%.
- EPR vs alternatives: R² > 0.98 (outlet \(T\), \(\eta\)).
- ANN drying kinetics: up to R² = 0.99998, MSE 1×10⁻⁶ [172].
- Freeze-drying ANN: R² = 0.999 for MR, MC, DR [166].
- CPV/T NePCM: greenhouse temperature held 100 min after irradiance drop [44].
────────────────────────────────────────

## 9. Baseline Comparison

| System | Baseline | With PCM / AI |
| --- | --- | --- |
| Mixed-mode dryer collector | No PCM, 100% recycle η 55.27% | PCM + 0% recycle 66.52% |
| Solar air dryer | Ethylene glycol + palmitic acid | Hybrid nanofluid 1.5% + PCM: η 46.8% vs lower baseline |
| Mango drying duration | No PCM | Micro-PCM: −63% time |
| Exergy efficiency | 21% (no PCM) | 28% (PCM2) |
| Outlet temperature model | CFD R² ~0.94 | EPR R² > 0.98 |
| Moisture content prediction | SVM R² = 0.974 | ANN R² = 0.999 |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

Review aggregates:

- Indirect/cabinet dryers with ETC, flat-plate collectors.
- PCM encapsulation: copper tubes in cabinet corners, grooved aluminum trays,
  plenum chambers.
- Sensors: temperature loggers, MC measurement, sometimes image-based LSTM
  monitoring.
- Nanofluid loops: 7 L/min flow, graphene–alumina hybrid.
- No RPi/solenoid SWH rig — closest parallel is ETC + PCM + airflow control;
  transferable to your DS18B20 + valve bench.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Limited experimental datasets for ML training in PCM dryers.
- ANN needs large data volume; risk of overfitting on small PCM tests.
- Advanced computation for dynamic PCM phase-change still challenging.
- EPR less robust than LSTM on high-dimensional time-series.
- Review focuses on drying, not domestic SWH — direct hardware transfer requires
  adaptation.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1: Relevant — ANN feedforward for solar transients [177–179] but no
  closed-loop actuator; your PPO solenoid advances beyond review.
- RG2: Highly relevant — PCM + sensors + optimization workflow; dryer ETC+PCM
  cabinet is analog to collector + PCM tank; Coimbatore co-author ties to
  project geography.
- RG3: Partial — drying load profiles differ from evening bath draw; PCM night
  discharge pattern directly transferable.
- RG4: Relevant — rich experimental PCM numbers (SEC, exergy, payback 22 months)
  for economic validation framing.
- RG5: Highly relevant — EPR grey-box and ANN surrogates under variable \(I_g\)
  mirror your XGBoost grey-box + climate forecasts.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

EPR general form (Eq. 18):

\[

y = \sum_{j=1}^{m} F\big(X, F(X), a_j\big) + a_0

\]

ANN neuron output (Eq. 11):

\[

q_i = f\!\left(\sum_j W_{ij} P_i\right)

\]

Network error (Eq. 12):

\[

E_r = \frac{1}{N}\sum_{i=1}^{N}(E_i - q_i)^2

\]

BP weight update (Eq. 13):

\[

\omega = \omega - \eta \frac{\partial e}{\partial \omega}

\]

PCM energy balance (review nomenclature):

\[

Q = m \lambda \frac{df}{dt} + mc_p \frac{dT}{dt}

\]

Use EPR for interpretable \(T_{out}(\text{GHI}, T_{amb}, \dot{m})\); ANN for PCM
state observer.

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Barghi Jahromi et al., ETC+PCM cabinet dryer + ANN/EPR/CFD, prior
   experimental — core case R² > 0.98, payback 22 months [9].
1. Karaağaç et al., CPV/T + NePCM + ANN/SVM drying, Sol. Energy — R² = 0.999 ANN
   [44].
1. Suherman et al., mixed-mode PCM coffee dryer, Renew. Energy, 2025 — recycling
   ratio study [104].
1. Soudagar et al., palmitic acid + hybrid nanofluid dryer, Appl. Therm. Eng.,
   2025 — 62.5% thermal η [105].
1. Lillo-Bravo et al., RF for solar water heating performance, Renew. Energy,
   2023 — SWH ML crossover [119086].
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I: Cite 30–40% post-harvest loss and PCM night-discharge for
  intermittent solar (parallel to SWH evening demand).
- Section II: Position as PCM + ML review complementary to Liu/Odoi SWH reviews;
  highlight EPR interpretability vs black-box DRL.
- Section III: Adopt EPR Eq. (18) or ANN 1-7-5 LM as surrogate for grey-box
  tank; inputs \(T_{amb}, I_g, \dot{m}\).
- Section IV: PCM placement trade-off (absorber vs plenum) informs your
  RT35/OM35 tank coil location; target R² > 0.98 like EPR benchmark.
- Section V: Compare energy metrics to SEC 13.38 MJ/kg analog (kWh per L hot
  water) and 22-month payback as economic aspirational bound.
────────────────────────────────────────

# 20. Barqawi2025DynamicSimulationPCM_SWH_summary.md

Source path: /mnt/data/Barqawi2025DynamicSimulationPCM_SWH_summary.md

# Dynamic Simulation of Phase Change Material-Integrated Solar Water Heating Systems: A Machine Learning Approach to Energy Conversion Optimization

Authors: Falah A. Barqawi

Year: 2025

Journal/Conference: Muthanna Journal of Engineering and Technology, Vol. 13, No.
3, pp. 1–14

DOI/Link: https://doi.org/10.52113/3/eng/mjet/2025-13-03/-1-14

IEEE Citation: F. A. Barqawi, "Dynamic simulation of phase change
material-integrated solar water heating systems: A machine learning approach to
energy conversion optimization," Muthanna J. Eng. Technol., vol. 13, no. 3, pp.
1–14, 2025, doi: 10.52113/3/eng/mjet/2025-13-03/-1-14.

────────────────────────────────────────

## 1. One-Line Summary

This paper develops and validates a simulation-only feedforward neural-network
controller that modulates pump flow multipliers in a three-phase PCM–solar water
heating model, achieving 2.5–4.1% (3.3% average) higher energy storage than
fixed-speed conventional control across five synthetic climate scenarios.

────────────────────────────────────────

## 2. Problem Being Solved

- PCM-integrated solar water heaters suffer from low PCM thermal conductivity
  (typically 0.1–0.5 W/m·K), supercooling, and thermal degradation, limiting
  charge/discharge rates.
- Conventional fixed-speed pump control cannot adapt to variable solar
  irradiance and ambient conditions, causing supply–demand mismatch and
  overdimensioned collectors or backup heating.
- Machine learning deployment for PCM thermal energy storage optimization is
  underutilized; prior work emphasizes material/geometric enhancements
  (nanoparticles, fins, metal wool) rather than intelligent, retrofit-compatible
  software control.
- Solar intermittency leaves stored thermal energy misaligned with end-use
  timing without adaptive operational optimization.
────────────────────────────────────────

## 3. Key Contributions

1. A complete three-phase lumped-parameter mathematical model (pre-melting,
   isothermal melting, post-melting) with dynamic sinusoidal solar input,
   automatic phase-transition event detection, and per-step energy balance
   accounting.
1. A feedforward neural-network controller using eight environmental/temporal
   inputs to predict optimal pump flow_multiplier values that retune the water
   thermal time constant in real time.
1. Comparative simulation across five environmental scenarios and five PCM
   geometry configurations (P01–P05) with identical thermophysical properties
   but varying volume and surface area.
1. Quantified ML-vs-baseline gains: +2.5% to +4.1% energy storage (MJ/kg),
   1.03–1.04× enhancement factors, and 12–18% pumping energy reduction at flow
   multipliers 0.3–0.6.
1. Positioning of ML software control as retrofit-compatible (very high retrofit
   potential, very low implementation cost) versus physical PCM enhancements
   requiring hardware modification.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

- System type: Horizontal cylindrical storage tank (length 1.0 m, diameter 0.5
  m) with internal solar collector coil and distributed PCM containers (volumes
  0.025–0.05 m³); schematic adapted from Chen et al. [21].
- Heat transfer areas/coefficients: Coil area \(A_c = 2.5\ \mathrm{m^2}\),
  water–coil HTC \(h_c = 1500\ \mathrm{W/m^2·K}\), PCM–water HTC \(h_p = 800\
  \mathrm{W/m^2·K}\); PCM surface areas 2.5–5.0 m² depending on configuration.
- Flow/HT correlations: Dittus–Boelter correlation for turbulent pipe flow;
  water velocity 0.02–0.05 m/s, Reynolds number 2000–5000.
- Assumptions: No external hot-water draw load; ambient heat losses neglected to
  isolate PCM effects; reference climate 33°N, 44°E (Middle Eastern conditions).
- Simulation duration: 50,400 s (14 h) per scenario; fixed reporting time step
  100 s; solver uses adaptive internal stepping.
- Control comparison: Conventional fixed-speed pump vs ML-optimized variable
  flow via predicted flow multiplier.
- Software: Python SciPy solve_ivp with Runge–Kutta (RK45) and event detection
  for phase changes.
### 4b. Mathematical Models & Equations

Phase 1 — Pre-melting (\(T_p < T_{melt}\)):

- Water: \(\displaystyle \frac{dT_w}{dt} = \frac{1}{\tau_w}\left[(T_c(t) - T_w)
  + \eta(T_p - T_w)\right]\) — (1)
- PCM: \(\displaystyle \frac{dT_p}{dt} = \frac{1}{\tau_{ps}}(T_w - T_p)\) — (2)
Solar / coil input:

- \(T_c(t) = T_{amb} + \dfrac{\mathrm{efficiency} \times I_{solar}(t)}{20}\) —
  (3)
- \(I_{solar}(t) = I_{max}\sin\!\left(\pi \dfrac{t_{hours} -
  \mathrm{sunrise}}{\mathrm{sunset} - \mathrm{sunrise}}\right)\) — (4)
Time constants and coupling:

- \(\tau_w = \dfrac{M_w C_w}{h_c A_c}\)
- \(\tau_{ps} = \dfrac{M_p C_{ps}}{h_p A_p}\)
- \(\eta = \dfrac{h_p A_p}{h_c A_c}\)
Phase 2 — Melting (\(T_p = T_{melt}\)):

- \(\displaystyle \frac{dT_w}{dt} = \frac{1}{\tau_w}\left[(T_c(t) - T_w) +
  \eta(T_{melt} - T_w)\right]\) — (5)
- \(\displaystyle \frac{dT_p}{dt} = 0\) — (6)
- \(\displaystyle \frac{dQ_p}{dt} = h_p A_p \max(0,\, T_w - T_{melt})\) — (7)
- \(Q_{p,\max} = H_f M_p\) — (8)
Phase 3 — Post-melting (\(T_p > T_{melt}\)):

- \(\displaystyle \frac{dT_w}{dt} = \frac{1}{\tau_w}\left[(T_c(t) - T_w) +
  \eta(T_p - T_w)\right]\) — (9)
- \(\displaystyle \frac{dT_p}{dt} = \frac{1}{\tau_{pl}}(T_w - T_p)\) — (10)
- \(\tau_{pl} = \dfrac{M_p C_{pl}}{h_p A_p}\)
Energy balances:

- Phase 1: \(E_{Water} = C_w M_w (T_w - T_{init})\) — (11); \(E_{PCM} = C_{ps}
  M_p (T_p - T_{init})\) — (12)
- Phase 2: \(E_{Water} = C_w M_w (T_w - T_{init})\) — (13); \(E_{PCM} =
  E_{p,melt,init} + \max(0, Q_p)\) — (14)
- Phase 3: \(E_{Water} = C_w M_w (T_w - T_{init})\) — (15); \(E_{PCM} =
  E_{p,melt,init} + E_{p,melt3} + C_{pl} M_p (T_p - T_{melt})\) — (16)
where \(E_{p,melt,init} = C_{ps} M_p (T_{melt} - T_{init})\) and \(E_{p,melt3} =
H_f M_p\).

ML control linkage:

- \(\mathbf{X} = [\mathrm{GHI}, \mathrm{DNI}, \mathrm{DHI}, T_{amb}, W_{spd},
  RH_{um}, \mathrm{Hour}, \mathrm{Month}]\) — (17)
- \(\tau_{w,\mathrm{optimized}} =
  \left(\dfrac{\mathrm{flow\_multiplier}}{\tau_w}\right) \times
  \mathrm{base\_time\_constant}\) — (18)
- \(\mathrm{flow\_multiplier} = \mathrm{ML\_model}(\mathbf{x}_{normalized})\) —
  (19)
### 4c. Algorithm / Control Method Steps

1. Initialize PCM properties (Table 1) and tank/coil parameters; set \(T_{init}
   = 40\,^\circ\mathrm{C}\).
1. Load 8,760 hourly environmental records; normalize features for the NN.
1. For each simulation timestep: compute \(I_{solar}(t)\) and \(T_c(t)\); detect
   PCM phase (solid / melting / liquid) via event functions.
1. Integrate ODEs (1)–(2), (5)–(7), or (9)–(10) with solve_ivp (RK45, adaptive
   stepping).
1. Conventional path: fixed pump speed / baseline time constant.
1. ML path: predict flow_multiplier from (17)–(19); update \(\tau_w\) and
   continue integration.
1. At each step, compute (11)–(16); check energy conservation and phase
   consistency.
1. After 14 h, compute energy improvement %, temperature improvement %, and
   enhancement factor vs baseline.
Neural network hyperparameters (stated): 3 hidden layers 64 → 32 → 16,
activation ReLU, output = scalar flow multiplier; optimizer Adam, loss MSE, 100
epochs; 90% validation prediction accuracy on held-out data.

### 4d. Data Sources & Dataset Details

| Source | Variables | Resolution | Scope | Period / size |
| --- | --- | --- | --- | --- |
| Synthetic / Meteonorm-style annual set (per text) | GHI, DNI, DHI, \(T_{amb}\), wind, RH, Hour, Month | Hourly | Climate representative of 33°N, 44°E | 8,760 samples (1 year) |
| Table 2 scenario parameters | Peak irradiance, seasonal factor, collector efficiency, \(T_{amb}\) | Per scenario | Five named cases (Summer Sunny, Winter Cloudy, etc.) | 14 h each, \(I_{max}\) 400–800 W/m² |
| Rule-based synthetic targets | Optimal pump speed / flow multiplier, target system efficiency 30–80% | Hourly | Same annual set | Used as supervised NN labels |

### 4e. Validation Method

- Primary: Simulation comparison of ML-optimized vs conventional fixed-speed
  control under five environmental scenarios for PCM variants P01–P05 (ML
  metrics tabulated for P01–P03).
- NN validation: 90% prediction accuracy on validation split (MSE-trained Adam
  model).
- Numerical checks: Energy conservation at each timestep; relative tolerance
  \(1\times10^{-6}\), absolute tolerance \(1\times10^{-9}\).
- No field experiment: Authors state simulation-only validation; experimental
  confirmation listed as future work.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- Materials tested: Five labeled configurations P01–P05 (generic organic-type
  PCM properties; geometry varies, chemistry not named as commercial grade).
- Melting temperature range: 44.0 °C (all P01–P05)
- Latent heat values: 165,000 J/kg (165 kJ/kg)
- Thermal conductivity values: Not reported for modeled PCM (literature context
  cites 0.1–0.5 W/m·K for typical PCMs)
- Specific heat (solid/liquid): 2100 / 2300 J/kg·K
- Density: 850 kg/m³
- Performance metrics: Total energy storage up to \(1.55\times10^7\) J (P01,
  Summer Sunny); 12.3 MJ/kg baseline without PCM vs PCM integration +26%;
  scenario efficiency heatmap 80–100% (Summer Sunny) vs 20–40% (Winter Cloudy);
  ML energy improvements +2.5% to +4.1% (Table 3).
| Config | \(V_p\) (m³) | \(A_p\) (m²) |
| --- | --- | --- |
| P01 | 0.05 | 5.0 |
| P02 | 0.03 | 3.0 |
| P03 | 0.025 | 2.5 |
| P04 | 0.025 | 2.5 |
| P05 | 0.035 | 3.5 |

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm name: Feedforward ANN (supervised regression) for pump
  flow_multiplier vs fixed-speed conventional pump baseline.
- Input features / state space: GHI, DNI, DHI, \(T_{amb}\), wind speed
  \(W_{spd}\), relative humidity \(RH_{um}\), Hour (0–23), Month (1–12) — Eq.
  (17).
- Output / action space: Continuous flow_multiplier (training distribution
  skewed 0.3–0.6); scales water thermal time constant via (18)–(19).
- Training details: Input (8) → hidden 64 → 32 → 16 (ReLU) → scalar output;
  Adam, MSE, 100 epochs; 8,760 hourly samples; synthetic rule-based labels; 90%
  validation accuracy.
- Performance metrics: System-level +3.3% average energy storage improvement;
  enhancement factors 1.03–1.04×; pumping energy −12% to −18% vs fixed speed.
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: Annual hourly meteorological-style dataset (GHI, DNI, DHI,
  temperature, wind, humidity); scenario parameters from Table 2;
  Meteonorm-style reference [27].
- Climate variables: GHI, DNI, DHI, \(T_{amb}\), wind speed, RH, Hour, Month;
  scenario peak irradiance, seasonal factor, collector efficiency.
- Geographic scope: 33°N, 44°E (Middle East reference — not Indian cities).
- Temporal resolution: Hourly (training); 14 h diurnal simulation per scenario
  with sinusoidal \(I_{solar}(t)\); \(I_{max}\) 400–800 W/m² across five
  scenarios.
────────────────────────────────────────

## 8. Key Results & Numbers

- ML energy storage improvement: +3.3% (P01), +4.1% (P02), +2.5% (P03); average
  +3.3% across P01–P03 (Table 3).
- Specific energy (Table 3): Normal 15.50 → ML 16.00 MJ/kg (P01); 7.77 → 8.08
  (P02); 6.93 → 7.11 (P03).
- Peak water temperature: 49.7 → 49.9 °C (P01, +0.4%); 51.0 → 51.8 °C (P02,
  +1.6%); 49.7 → 49.9 °C (P03, +0.5%).
- Enhancement factors: 1.03 (P01, P03), 1.04 (P02); average 1.03×.
- Pumping energy reduction with ML flow control: 12–18% at flow multipliers
  0.3–0.6.
- Response time: target temperatures reached 15–20 minutes earlier (~2.3%
  improvement relative to 14 h operation).
- Maximum energy storage: \(1.55\times10^7\) J (P01, Summer Sunny); Winter
  Cloudy range \(2.5\times10^6\)–\(7.0\times10^6\) J.
- Scenario efficiency (heatmap): 80–100% (Summer Sunny); 20–40% (Winter Cloudy).
- PCM vs no-PCM baseline: conventional SWH 12.3 MJ/kg → P01 PCM integration +26%
  under Summer Sunny.
- Nanoparticle literature benchmark (comparison only): 32% thermal conductivity
  gain, 72% thermal efficiency (Dayer et al.) — not achieved by this ML method.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Conventional fixed-speed pump control (“Normal Method”);
  additional reference SWH without PCM (12.3 MJ/kg Summer Sunny).
- Proposed method: ML-optimized variable flow via ANN-predicted flow_multiplier
  retuning \(\tau_w\).
- Improvement margin: +2.5% to +4.1% energy (MJ/kg); +0.4% to +1.6% peak water
  temperature; 1.03–1.04× enhancement factor; 12–18% lower pumping energy.
- Conditions: Same PCM properties, tank geometry, five Table 2 scenarios, 14 h
  simulation, identical three-phase model; only control law differs.
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — simulation-only study. Authors describe a retrofit-compatible concept
requiring \(T_w\), \(T_p\), and environmental inputs but deploy no physical
prototype, no RPi/Arduino/ESP32, and no lab or field test.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Simulation-based only — requires experimental validation in real SWH
  installations under diverse climates.
- ML controller trained on synthetic optimization targets, not measured
  operational data.
- PCM variants share identical \(T_{melt}=44°C\) — limits generalizability to
  other chemistries (e.g., RT35/OM35).
- Future work: RNNs/Transformers, weather forecasting for predictive control,
  extension to HVAC/industrial heat.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Relevant — ANN maps live weather to flow
  multipliers (+2.5–4.1%); your DRL PPO on embedded hardware extends this with
  charge/discharge/bypass modes.
- RG2 (No integrated PCM–AI–hardware prototype): Relevant (gap) — Full software
  pipeline without hardware; your closed-loop prototype fills this.
- RG3 (Poor alignment with household demand patterns): Not relevant — No
  hot-water draw profiles or demand scheduling.
- RG4 (Limited real-world experimental validation): Highly relevant — Authors
  explicitly call for field trials; your multi-city evaluation addresses this.
- RG5 (No predictive optimization under climatic uncertainty): Partially
  relevant — Uses current/historical hourly weather, not forecasts; aligns with
  your ERA5/NASA POWER + forecast-driven DRL extension.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

- Pre-melt water/PCM dynamics: Eqs. (1)–(2) for grey-box Gym environment.
- Collector coil drive: \(T_c(t)=T_{amb}+\eta_{col}I_{solar}(t)/20\) — (3);
  couple to pyranometer/forecast GHI.
- Diurnal solar:
  \(I_{solar}(t)=I_{max}\sin(\pi(t_{hr}-t_{sr})/(t_{ss}-t_{sr}))\) — (4).
- Latent melting: (6)–(8) with solve_ivp event detection for phase changes.
- Stored PCM energy: (14) in reward function.
- ML/DRL feature vector:
  \(\mathbf{X}=[\mathrm{GHI},\mathrm{DNI},\mathrm{DHI},T_{amb},W_{spd},RH,Hour,Month]\)
  — (17).
- Control action analogy:
  \(\mathrm{flow\_multiplier}=\mathrm{ML\_model}(\mathbf{x}_{norm})\) — (19) →
  map to solenoid valve / pump speed / bypass mode in PPO action space.
────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Tamizharasan & Kini, "Deep learning approach for PCM-enhanced SWH," Int. J.
   Energy Res., 2023 — DL + PCM-SWH parallel to your DRL line.
1. Vempally & Dhanarathinam, ML PCM selection, J. Therm. Anal. Calorim., 2023 —
   data-driven PCM selection like your XGBoost classifier.
1. Goel et al., PCM in solar thermal review, Appl. Therm. Eng., 2023 —
   Introduction/lit-review framing.
1. Chen L. et al., solar thermal collector system design, Renewable Energy, 2023
   — tank–coil–PCM schematic source.
1. Meteonorm global meteorological database, 2023 — hourly climate data analogue
   to ERA5/NASA POWER/ISRO.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I (Introduction): ML-for-PCM-TES underutilized; Barqawi reports +3.3%
  average stored energy vs fixed-speed pump with retrofit-compatible software
  control.
- Section II (Literature Review): ANN flow_multiplier on 8-feature weather
  vector; +4.1% max (P02) without hardware change.
- Section III (Methodology): Adopt three-phase ODEs (1)–(16) with RK45
  tolerances \(10^{-6}\)/\(10^{-9}\) for grey-box training environment.
- Section IV (Dataset & Setup): Benchmark PCM Table 1 (\(T_{melt}=44°C\),
  \(H_f=165\) kJ/kg); map Summer Sunny 800 W/m² → Jaisalmer, Winter Cloudy 400
  W/m² → monsoon cases.
- Section V (Results): Exceed +3.3% mean energy and 1.03× enhancement; secondary
  metric 12–18% pumping savings for valve/pump actuation.
────────────────────────────────────────

# 21. Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md

Source path: /mnt/data/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md

# Using the Taguchi Method and Grey Relational Analysis to Optimize the Parameter Design of Flat-Plate Collectors with Nanofluids and Phase Change Materials in an Integrated Solar Water Heating System

Authors: Guan-Rong Chen, Ting-Wei Liao, Chien-Chun Hsieh, Jagadish Barman,
Chao-Yang Huang, Chung-Feng Jeffrey Kuo

Year: 2025

Journal/Conference: Energy Conversion and Management: X, Vol. 26, Article 100910

DOI/Link: https://doi.org/10.1016/j.ecmx.2025.100910

IEEE Citation: G.-R. Chen et al., "Using the Taguchi method and grey relational
analysis to optimize the parameter design of flat-plate collectors with
nanofluids, and phase change materials in an integrated solar water heating
system," Energy Convers. Manag.: X, vol. 26, p. 100910, 2025, doi:
10.1016/j.ecmx.2025.100910.

────────────────────────────────────────

## 1. One-Line Summary

This study combines RT35HC PCM, CuO nanofluid, and flat-plate collectors in a
TRNSYS-simulated integrated SWH, optimizes 9 factors via L36 Taguchi DOE and
grey relational analysis (GRA), and achieves 94.2% thermal storage efficiency
and 31.7 h heat retention at 30 °C target—+28% efficiency and +14.6 h retention
vs the non-optimized baseline.

────────────────────────────────────────

## 2. Problem Being Solved

- SWH systems suffer from low thermal storage efficiency and insufficient heat
  retention after sunset under intermittent solar input.
- Prior work optimized nanofluids or PCMs separately; few studies jointly
  integrate both with systematic multi-objective parameter design for flat-plate
  SWH.
- Single-response Taguchi optimization cannot simultaneously maximize thermal
  storage efficiency and heat retention time—requiring GRA multi-quality fusion.
- Physical multi-factor experiments are costly; need validated simulation-based
  DOE (TRNSYS) with confirmation runs.
────────────────────────────────────────

## 3. Key Contributions

1. Novel integrated architecture: nanofluid + Rubitherm RT35HC PCM + flat-plate
   collector in one closed-loop SWH (FPC, 0.04 m³ tank, PCM tubes, pump,
   pyranometer, PT100 sensors).
1. L36 orthogonal array with 9 control factors × 36 TRNSYS runs; S/N
   (larger-the-better), MEA, and ANOVA for each quality characteristic.
1. GRA multi-objective optimization merging thermal storage efficiency and heat
   retention into a single grey relational grade (GRG).
1. Lumped thermal + electrical analog model for FPC layers (glass, air gap,
   absorber, fluid, insulation) with TRNSYS validation within 5% of physical
   measurements.
1. Confirmed optimum: PCM on, 20% PCM volume, 14 PCM tubes, CuO nanofluid, 0.02
   kg/s flow, 9 collector tubes, copper plate, tilt 22.4°, azimuth 0° (south).
1. Performance claim: first reported nanofluid + PCM combined SWH optimization
   reaching 94.2% storage efficiency—exceeding literature PCM-only (~64–79%) and
   nanofluid-only (~50–86%) benchmarks cited in Table 26.
────────────────────────────────────────

## 4. Methodology

### 4a. System Setup

- Collector: FPC 505 × 320 mm, tube OD 25 mm, ID 24 mm; baseline 9 tubes, 0.15
  m² area.
- Storage: Tank 0.04 m³, height 500 mm; PCM pipe height 450 mm, baseline 12
  tubes, volume 0.0037 m³ per configuration.
- PCM: Rubitherm RT35HC organic paraffin (heating/cooling curves from
  manufacturer data; minimal hysteresis).
- Working fluids: Water, Al₂O₃ nanofluid, CuO nanofluid (properties Tables 8–9).
- Instrumentation (physical rig): PYR2-420 pyranometer (300–2900 nm, RS485),
  Galltec-Mela ambient T/RH, PT100 sensors (±0.25 K), RP flow meter (±2.5% FSD),
  FORMOSA RS-15/6GWS pump, BCT TF 200S data logger.
- Simulation: TRNSYS modular architecture (Fig. 4); parameters tuned to match
  physical system.
### 4b. Taguchi DOE (Table 7)

| Factor | Symbol | Levels |
| --- | --- | --- |
| PCM material | A | No paraffin / RT35HC |
| PCM volume | B | 10%, 15%, 20% of tank volume |
| PCM tube count | C | 12, 14, 16 |
| Working fluid | D | Water, Al₂O₃, CuO |
| Mass flow rate | E | 0.02, 0.025, 0.03 kg/s |
| Collector tubes | F | 9, 10, 11 |
| Plate material | G | Cu, Al, stainless steel (M) |
| Tilt angle | H | 20.4°, 22.4°, 24.4° |
| Azimuth | I | −45°, 0° (south), +45° |

- Daily water demand assumption: ≥30 L; PCM volume levels anchored at 10%
  baseline + 5% increments.
### 4c. Quality Metrics

1. Thermal storage efficiency — larger-the-better S/N (Eq. 1).
1. Heat retention time — hours after sunset until tank T drops below 30 °C
   target.
### 4d. GRA Procedure

- Normalize both S/N sequences (Eq. 15, larger-the-better).
- Grey relational coefficient \(\zeta_i(k)\) with distinguishing coefficient ζ =
  0.5 (Eq. 16).
- Grey relational grade \(\gamma_i\) averaged over responses (Eq. 17).
- Select factor levels maximizing mean GRG (Table 23).
### 4e. Validation

- TRNSYS vs physical system: accept if error < 5%.
- Single-quality confirmation: 5 runs; S/N must fall in 95% CI.
- Multi-quality confirmation: efficiency 94.2%, retention 31.7 h; S/N 39.481
  (efficiency), 30.021 (retention).
────────────────────────────────────────

## 5. PCM Details (if applicable)

| Property | RT35HC (Rubitherm) |
| --- | --- |
| Type | Organic paraffin (solid–liquid) |
| Role | Latent TES in dedicated PCM tubes in storage tank |
| Factor A | Level 1 = no PCM; Level 2 = with PCM (dominant performance gain) |
| Volume levels | 10%, 15%, 20% of total tank volume (30 L demand basis) |
| Tube count | 12 / 14 / 16 tubes (heat transfer area) |
| Optimal (GRA) | 20% volume, 14 tubes |
| Behavior | Heating/cooling enthalpy curves nearly overlap (Fig. 17) — low hysteresis |
| PCM-less vs PCM | Figs. 18–20 show both efficiency and retention improve substantially when PCM enabled |

Project alignment: RT35HC is in your Rubitherm RT35–RT64HC screening set; Chen’s
20% PCM volume and 14-tube layout provide a documented DOE baseline for tank
geometry in grey-box modeling.

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A — classical DOE + GRA + TRNSYS simulation; no ML/DRL/MPC.

Relevance: Taguchi+GRA is your Objective 1 PCM selection methodology (per
Presentation §8.1); DRL (Objective 2) would replace static optimal flow/tilt
with adaptive control under climate uncertainty.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Location: Taiwan (Taipei area tilt study); optimal tilt 22.4° from Liu et al.
  Taiwan regional analysis (20.2°–22.4° range).
- Climate inputs in model: Solar radiation on collector, ambient temperature,
  wind (in lumped FPC heat-loss equations); pyranometer PYR2-420 on physical
  rig.
- Azimuth: 0° (true south) best; ±45° degrades capture.
- Target retention temperature: 30 °C minimum after sunset.
- Not used: ERA5, NASA POWER, ISRO — local Taiwan weather implicit in
  TRNSYS/physical validation.
- India mapping: Re-run tilt/azimuth levels for Coimbatore (~11°N), Kochi
  (~10°N), Jaisalmer (~26.9°N); retain GRA structure with climate-specific
  TRNSYS or grey-box weather files.
────────────────────────────────────────

## 8. Key Results & Numbers

- 36 Taguchi experiments (L36 orthogonal array).
- Global cumulative solar thermal capacity: 522 GW_th (+3% growth) [IRENA cite
  in intro].
- Thermal energy share: 48.7% of global supply; solar 10.5% of modern renewable
  heat [8].
- TRNSYS validation: simulation within 5% of physical data.
- Single-quality optimum — thermal storage efficiency: predicted S/N 39.179;
  confirmation mean efficiency 92.2%, S/N 39.294 (CI 38.84–39.52).
- Single-quality optimum — heat retention: predicted S/N 33.809; confirmation
  29.6 h, S/N 29.425 (CI 28.83–38.79).
- GRA multi-quality confirmation: thermal storage efficiency 94.2%; heat
  retention 31.7 h; S/N 39.481 / 30.021.
- vs non-optimized system: efficiency +28%; retention +14.6 h (abstract).
- ANOVA — thermal storage efficiency: collector plate material 44.68%
  contribution (F=385.7); PCM material 25.51% (F=440.5); collector tube count
  13.98%; working fluid 11.62%.
- ANOVA — heat retention: collector plate material 59.98%; working fluid 13.86%;
  collector tubes 10.9%; PCM material 2.56%.
- GRG factor ranking: collector plate material rank 1 (Δ=0.3041); working fluid
  rank 2; collector tubes rank 3.
- Optimal nanofluid: CuO \(\rho\) 1210 kg/m³, \(c_p\) 3.41 kJ/kg·°C outperforms
  Al₂O₃ and water.
- Optimal flow: 0.02 kg/s (lower pump energy, better heat absorption vs 0.03
  kg/s).
- Literature comparison (Table 26): nanofluid SWH 50.27–85%; PCM SWH 64.53–79%;
  this work 94.2% combined.
────────────────────────────────────────

## 9. Baseline Comparison

| Configuration | Thermal storage efficiency | Heat retention (T ≥ 30 °C) | Notes |
| --- | --- | --- | --- |
| Non-optimized integrated SWH | ~73.6% (derived: 94.2/1.28) | ~17.1 h (31.7 − 14.6) | Abstract baseline |
| GRA-optimized (PCM+CuO) | 94.2% | 31.7 h | +28% eff., +14.6 h |
| Without PCM (factor A1) | Lower S/N across runs | Shorter retention | Figs. 18–20 |
| Water vs CuO nanofluid | Lower η_storage | Shorter retention | CuO highest \(k\) |
| Al plate vs Cu plate | η drop (S/N 34.57 min vs 36.78 max) | Major retention impact | ANOVA 44.7–60% |
| 11 vs 9 collector tubes | Reduced efficiency (thermal interference) | — | Optimal 9 tubes |
| Tilt 24.4° vs 22.4° | Less radiation capture | — | Taipei-optimal 22.4° |
| Azimuth ±45° vs south | Reduced performance | — | 0° best |
| Prior PCM-only literature | 64.53–79% | — | Table 26 |
| Prior nanofluid-only literature | 50.27–85% | — | Table 26 |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

| Component | Specification |
| --- | --- |
| Flat-plate collector | 505×320 mm; 9–11 copper/aluminum/SS tubes |
| PCM | RT35HC in 12–16 tubes; 10–20% tank volume |
| Tank | 40 L (0.04 m³); HX tubes OD 22 mm |
| Pump | FORMOSA RS-15/6GWS |
| Pyranometer | PYR2-420, Class C, 10 µV/(W/m²) sensitivity |
| Temperature | PT100 (±0.25 K, 30–500 K range stated) |
| Ambient | Galltec-Mela PC-ME (±0.2 K, RH ±2%) |
| Flow meter | RP variable area, ±2.5% FSD |
| Data logger | BCT TF 200S |
| Platform | Physical prototype + TRNSYS simulation (primary optimization) |
| Test conditions | Taiwan; south-facing; validated to 5% |

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Optimization primarily via simulation (36 TRNSYS runs); physical confirmation
  limited to 5 runs per quality metric.
- Taiwan-specific tilt/azimuth optima (22.4°, south) — not directly transferable
  without re-optimization.
- Model assumptions: uniform layer temperatures, perfect edge insulation, no
  dust, equal front/back ambient (Section 3.5.1).
- Nanofluid stability, agglomeration, and long-term pumping wear not deeply
  studied.
- Combined 94.2% metric is thermal storage efficiency in their TRNSYS
  definition—not identical to ISO 9459 annual solar fraction.
- No adaptive/real-time control; static optimal parameters only.
- Authors note gap filled vs PV/T nanofluid+PCM work (Liu 2023 64.76%) but SWH
  field lacked combined optimization before this study.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Relevant as baseline — Chen optimizes
  fixed flow (0.02 kg/s), tilt, and PCM volume offline; your DRL agent can treat
  these as action bounds or initial policy, then adapt online to
  irradiance/load.
- RG2 (No integrated PCM–AI–hardware prototype): Highly relevant — Same material
  stack (RT35HC, nanofluid option, FPC, instrumented rig) maps to your PCM-SWH
  hardware story; Chen lacks AI layer—you add DRL + embedded closure.
- RG3 (Poor alignment with household demand patterns): Relevant — 30 L daily
  demand and 30 °C retention threshold mirror domestic hot-water targets; extend
  to time-of-use demand profiles in reward shaping.
- RG4 (Limited real-world experimental validation): Partially relevant —
  Physical TRNSYS calibration within 5% exists, but optimization runs are
  simulation-heavy; supports using your grey-box as training env with field
  validation as differentiator.
- RG5 (No predictive optimization under climatic uncertainty): Relevant —
  Taguchi+GRA is climate-static (single Taiwan profile); your ERA5/NASA POWER
  forecast + PCM classifier generalizes across Coimbatore/Kochi/Jaisalmer where
  tilt 11°–27° differs from 22.4°.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

S/N ratio (larger-the-better):

\[

\frac{S}{N}_{LTB} = -10
\log_{10}\left(\frac{1}{n}\sum_{i=1}^{n}\frac{1}{y_i^2}\right) \tag{1}

\]

Main effect:

\[

F_i = \frac{1}{m}\sum_{k=1}^{m}\eta_{ik}, \qquad \Delta F = F_{i,\max} -
F_{i,\min} \tag{2–3}

\]

Grey normalization (larger-the-better):

\[

x_i^*(k) = \frac{x_i(k) - \min x_i(k)}{\max x_i(k) - \min x_i(k)} \tag{15}

\]

Grey relational coefficient:

\[

\xi_i(k) = \frac{\Delta_{\min} + \zeta\Delta_{\max}}{\Delta_i(k) +
\zeta\Delta_{\max}}, \quad \zeta = 0.5 \tag{16}

\]

Grey relational grade (your Presentation notation):

\[

\gamma_i = \frac{1}{n}\sum_{k=1}^{n} w(k)\,\xi_i(k) \tag{17}

\]

Working fluid energy balance (lumped FPC):

\[

C_f \frac{dT_f}{dt} = \frac{1}{R_{cov}}(T_{ab} - T_f) + \dot{m}_f C_{pf}(T_{fo}
- T_{fi}) \tag{43}

\]

Absorber solar gain:

\[

I_T = \alpha_{ab}\,\tau_g\, G\, A_{ab} \tag{37}

\]

Heat retention metric (project adoption):

\[

t_{ret} = t\left(T_{tank}(t) < T_{target}\right) - t_{sunset}, \quad T_{target}
= 30\,°\text{C}

\]

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. L.F. Cabeza et al. — PCM volume/module count effects in SWH [37].
1. Liu et al. (2023) — Taguchi+GRA on PV/T with paraffin + nanofluid (64.76%
   heat storage efficiency) [44].
1. C. Kuo et al. — prior Taguchi+GRA on flat-plate collectors [23].
1. Moghadam et al. — tilt angle effects on FPC efficiency [18].
1. A. El-Fakharany et al. — SAH with PCM up to 64.53% efficiency [22].
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I (Introduction): Cite the dual challenge of low storage efficiency
  and poor overnight retention; note global 522 GW_th solar thermal capacity.
- Section II (Literature Review): Position as the primary Taguchi + GRA
  reference for PCM–nanofluid–FPC SWH; contrast static DOE (94.2%, 31.7 h) with
  lack of adaptive AI control.
- Section III (Methodology): Reproduce GRA equations (15–17) for Objective 1 PCM
  selection alongside XGBoost; use Chen’s 9-factor table as template for Indian
  climate re-optimization (tilt, PCM volume, flow).
- Section IV (Dataset & Setup): Reference RT35HC Rubitherm curves,
  PYR2-420-class pyranometer, PT100/DS18B20 equivalence, 30 °C retention
  threshold, 30 L demand—mirror sensor stack in prototype section.
- Section V (Results): Benchmark grey-box/DRL against Chen’s 94.2% storage
  efficiency and 31.7 h retention; report improvement over ~73.6% / ~17.1 h
  non-optimized baseline; cite +28% / +14.6 h as published DOE gains.
────────────────────────────────────────

# 22. Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md

Source path: /mnt/data/Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md

# Technical & Financial Feasibility Assessment of Heat Pipe Evacuated Tube Collector for Water Heating Using Monte Carlo Technique for Buildings

Authors: K. Chopra, V.V. Tyagi, Sakshi Popli, A.K. Pandey

Year: 2023

Journal/Conference: Energy, Vol. 267, Article 126338

DOI/Link: https://doi.org/10.1016/j.energy.2022.126338

IEEE Citation: K. Chopra et al., "Technical & financial feasibility assessment
of heat pipe evacuated tube collector for water heating using Monte Carlo
technique for buildings," Energy, vol. 267, p. 126338, 2023, doi:
10.1016/j.energy.2022.126338.

────────────────────────────────────────

## 1. One-Line Summary

This study couples a heat-pipe evacuated-tube collector (HP-ETC) thermal model
with Monte Carlo uncertainty and genetic-algorithm optimization across five
Indian climate zones, finding mean LCWH = 5.14 INR/kWh, NPV = 663,788 INR, PP =
5.84 years, with Zone-V (Ahmedabad) most favorable and optimized cases cutting
LCWH ~25–33% and PP ~37–47%.

────────────────────────────────────────

## 2. Problem Being Solved

- India residential sector supplies ~80% of national hot-water demand — prime
  target for SWH.
- HP-ETC SWH is efficient but under-adopted vs thermosyphon ETC due to high
  capital cost, overheating, scaling, and optimistic deterministic economic
  models.
- Conventional techno-economic studies use fixed inputs, ignoring uncertainty in
  irradiance, efficiency, tariffs, and finance — biasing investment decisions.
- Need probabilistic feasibility tool for HP-ETC deployment across India's
  climate zones.
────────────────────────────────────────

## 3. Key Contributions

1. HP-ETC (HPT-ETCS) performance + multi-energy/economic cost model for domestic
   SWH.
1. Monte Carlo Technique (MCT) — triangular distributions on key inputs; N
   simulation trials for LCWH, NPV, PP.
1. Five-zone India analysis (Zones I–V) with city-level solar radiation
   (Ahmedabad 5.615 vs Srinagar 4.695 kWh/m²/day).
1. Sensitivity analysis: solar radiation drives 79.47% of LCWH uncertainty;
   thermal efficiency 15.65%.
1. Single-objective GA optimization (PIKAIA in EES) for LCWH, NPV, PP — 33.46%
   LCWH reduction vs base case (LCWHOL).
1. Policy insight: prioritize high electricity-price regions; subsidies could
   accelerate HP-ETC penetration.
────────────────────────────────────────

## 4. Methodology

### 4a. System model

- HP-ETC domestic SWH: fixed orientation (latitude tilt, azimuth 0°), 6.32 m²
  aperture/collector, 67 evacuated tubes.
- Hot water: 60 L/day/person, 6 persons/house, 60 °C delivery, 15-year life.
- Thermal efficiency triangular 51–69%, mean 60%; degradation 1%/year.
### 4b. Economic metrics

- LCWH — levelized cost of water heating (INR/kWh).
- NPV — net present value over 15 years (INR).
- PP — payback period (years).
- Matrix formulation (7) linking annual energy, costs, loan payments across N
  years.
### 4c. MCT procedure

- Sample each uncertain input from preset distributions → compute outputs →
  build probability histograms.
- Compare against grid electricity 6.50–28.05 INR/kWh over system life.
### 4d. Optimization

- GA: 9 individuals, 64 generations, crossover 0.85, mutation 0.005–0.25.
- Decision variables: capital cost \(C_0\), debt-equity ratio, interest, O&M,
  irradiance \(I_T\), \(\eta_{th}\), electricity price, discount rate.
- Three cases: LCWHOL, NPVOL, PPOL.
────────────────────────────────────────

## 5. PCM Details (if applicable)

N/A — study focuses on HP-ETC sensible water heating, not latent PCM storage.

- Authors suggest high-boiling-point nanofluids to mitigate overheating/scaling
  — indirect link to your PCM-TES SWH (latent buffer replaces oversizing).
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A — no machine learning.

- Genetic Algorithm (GA) for economic optimization (metaheuristic, not
  predictive AI).
- Monte Carlo for uncertainty — analogous to robust policy evaluation under
  climate/economic noise (related to RG5).
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Geography: Five Indian climatic zones (I cold to V hot/dry); cities include
  Srinagar, Delhi, Mumbai, Chennai, Ahmedabad (Table 3).
- Solar variable: daily average irradiance 3.50–6.98 kWh/m²/day, mean 5.24.
- India resource: ~5×10³ trillion kWh/year national solar endowment; 4–7
  kWh/m²/day average.
- Hot water demand: residential 80% of sector demand; commercial 13%, industrial
  6%.
- Project mapping: Coimbatore/Jaisalmer/Kochi zone analogs — high radiation
  (Jaisalmer ≈ Zone-V) favors lower LCWH/PP; humid/coastal zones need larger
  collector area.
────────────────────────────────────────

## 8. Key Results & Numbers

- Mean LCWH: 5.14 INR/kWh (90.65% probability between 3.80–6.00).
- Mean NPV: 663,788.48 INR — 100% probability NPV > 0 in India scenario.
- Mean payback: 5.84 years — 100% certainty PP < 15 years.
- Zone-V: lowest LCWH and PP, least collector area required; Zone-III: highest
  LCWH/PP, lowest NPV.
- Ahmedabad: 5.615 kWh/m²/day (max); Srinagar: 4.695 kWh/m²/day (min among
  selected).
- Sensitivity: solar radiation 79.47% of LCWH variance; \(\eta_{th}\) 15.65%;
  capital cost marginal.
- η_th sweep 45–73%: LCWH 6.32 → 4.38 INR/kWh; NPV 581,231 → 717,797 INR; PP
  7.65 → 4.63 years.
- Electricity price 4–8 INR/kWh: NPV 308,977 → 886,337 INR; PP 9.18 → 4.56
  years.
- Discount rate 4–10%: NPV 741,259 → 417,440 INR (\(R^2=0.9913\)).
- GA optimization vs base: LCWH −33.46% / −25.34% / −26.93%; NPV +9% / +28.43% /
  +26.35%; PP −37.76% / −41.43% / −47.37% (LCWHOL/NPVOL/PPOL).
- Optimized \(C_0\): 30,140–31,145 INR/m² vs base 32,500; \(\eta_{th}\) up to
  69%.
────────────────────────────────────────

## 9. Baseline Comparison

| Case | LCWH | NPV (INR) | PP (years) |
| --- | --- | --- | --- |
| Grid electricity only | 6.50–28.05 INR/kWh | — | — |
| HP-ETC mean (MCT) | 5.14 | 663,788 | 5.84 |
| Thermosyphon ETC (market default) | Higher LCWH implied | Lower NPV implied | Longer PP implied |
| GA-optimized LCWHOL | ~−33% vs base | ~+9–28% | ~−38–47% |
| Low \(\eta_{th}=45\%\) | 6.32 INR/kWh | 581,232 | 7.65 |
| High \(\eta_{th}=73\%\) | 4.38 INR/kWh | 717,797 | 4.63 |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — simulation-only techno-economic model (EES + MCT + GA).

- Modeled hardware: HP-ETC, 67 tubes, 6.32 m² aperture, fixed tilt = latitude.
- Demand side: 360 L/day hot water (6×60 L), 60 °C.
- Financial: bank loan 15 years, interest 8–10%, debt-equity 50–90%.
- Comparable to your project's ETC + storage tank architecture; add PCM capex in
  extended NPV model.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- HP-ETC initial cost and maintenance remain barriers despite favorable LCWH.
- Overheating if oversized or low flow; vacuum tube failure >100 °C.
- Scaling on heat-pipe condenser in hard-water regions.
- Model assumes triangular distributions — real tariffs/policy may differ.
- Does not include PCM, smart control, or dynamic demand profiles.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1: Indirect — no control, but shows value of efficiency gains from better
  operation (η_th 45→73% cuts PP).
- RG2: Relevant — HP-ETC is closest commercial analog to your collector loop;
  PCM+AI adds differentiation beyond this economic study.
- RG3: Highly relevant — 60 L/day/person, 60 °C, 6 occupants = explicit
  household demand model for reward shaping.
- RG4: Relevant — India field economics; validate prototype against 60% mean
  η_th assumption.
- RG5: Highly relevant — MCT framework for uncertain GHI, tariffs, efficiency
  maps to climate-adaptive optimization under ERA5/NASA POWER variability across
  Coimbatore, Jaisalmer, Kochi.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Seasonal energy demand (conceptual from model):

\[

Q_{annual} = \sum_{\tau} \dot{Q}_{load}(\tau) - \sum_{\tau} \eta_{th} A_{col}
G(\tau)

\]

NPV:

\[

NPV = \sum_{t=0}^{N} \frac{CF_t}{(1+r)^t} - C_{cap}

\]

LCWH (levelized cost):

\[

LCWH = \frac{\sum_t (I_t + O\&M_t + fuel_t)}{(1+r)^t}{\Big/}{\sum_t
\frac{E_{thermal,t}}{(1+r)^t}}

\]

Payback: smallest \(t\) where cumulative savings > \(C_{cap}\).

Sensitivity index: fraction of output variance attributable to input \(x_i\)
(MCT correlation).

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Mehmood et al., heat-pipe ETC SWH natural gas backup, Energy Rep., 2019 —
   HP-ETC performance baseline [7].
1. Duraivel et al. / Indian SWH techno-economic studies — regional economics
   context.
1. TRNSYS/MATLAB economic SWH models — deterministic predecessors [10,19].
1. MNRE India solar zone maps — climatic zoning [3].
1. Singh et al., PCM-SWH review, 2025 — PCM complement to HP-ETC economics.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I: Cite 80% residential hot-water share and HP-ETC
  anti-freeze/high-performance rationale for India.
- Section II: Position Chopra as probabilistic techno-economic reference for
  ETC-SWH vs your PCM-intelligent control contribution.
- Section III: Use 60 L/person/day, 60 °C demand profile in grey-box and DRL
  reward (meet evening draw).
- Section IV: Map test cities to zones; target η_th ≥ 60% and PP <6 years when
  adding PCM+control capex.
- Section V: Report LCWH/NPV/PP improvement from intelligent PCM control vs base
  HP-ETC (5.14 INR/kWh benchmark).
────────────────────────────────────────

# 23. Duraivel2025DSTS_TechnoEconomic_summary.md

Source path: /mnt/data/Duraivel2025DSTS_TechnoEconomic_summary.md

# Performance, Techno-Economic Viability, and Environmental Impact of Domestic Solar Tri-Generation System (DSTS): A Comparative Study of Copper and Galvanized Iron-Based Systems for Sustainable Building Applications

Authors: Balamurali Duraivel, Natarajan Muthuswamy

Year: 2025

Journal/Conference: Journal of Building Engineering, Vol. 113, Article 113964

DOI: https://doi.org/10.1016/j.jobe.2025.113964

IEEE Citation: B. Duraivel and N. Muthuswamy, "Performance, techno-economic
viability, and environmental impact of domestic solar tri-generation system
(DSTS): A comparative study of copper and galvanized iron-based systems for
sustainable building applications," J. Build. Eng., vol. 113, p. 113964, 2025,
doi: 10.1016/j.jobe.2025.113964.

────────────────────────────────────────

## 1. One-Line Summary

This study builds and field-tests copper (C-DSTS) and galvanized-iron (GI-DSTS)
concrete-roof tri-generation prototypes in Vellore, India, integrating solar
water heating, 160 Wp PV, and 50 TEGs, achieving 46.4% overall efficiency, ~6 °C
passive cooling, and 25-year payback as low as 5.3 years with 180,704–206,921 kg
lifetime net CO₂ mitigation.

────────────────────────────────────────

## 2. Problem Being Solved

- Building-integrated solar systems are often single-function (water heating
  only), while conventional PVT/BIPVT and TEG-PVT designs suffer from complex
  rooftop integration, seasonal efficiency swings, extra cost, and weak passive
  cooling (Abstract, Section 1).
- TEG-integrated BIPVT systems typically mount TEGs externally, giving unstable
  ΔT, low TEG output (~5–8% of waste heat), and minimal large-scale indoor
  cooling (Table 1).
- Indian residential energy use is rising (~6–8% for water heating, ~10% for
  cooling; AC demand projected 5× by 2030), increasing grid stress and emissions
  (Introduction).
- Lack of combined experimental + techno-economic + environmental evidence for
  multifunctional roof-embedded systems using existing slab structure rather
  than full-roof add-on layers.
────────────────────────────────────────

## 3. Key Contributions

1. Domestic Solar Tri-Generation System (DSTS): Concrete slab roof embeds
   serpentine copper or GI absorber, frameless monocrystalline PV on top, 50
   parallel TEGs under slab — water heating + electricity + passive cooling
   without separate rooftop assemblies.
1. Outdoor experiments (Vellore, March 2024): Three flow rates 0.12 / 0.24 /
   0.36 L/min; performance vs conventional concrete slab; uncertainty 5.08%
   overall.
1. Quantified tri-generation metrics: Thermal η_T ≈ 24.75% (C) / 24.20% (GI);
   exergy η_E up to 37.05% / 32.30%; overall 46.4%; indoor cooling 6.2 °C / 5.9
   °C; 108 L heated to 50 °C in 5 h.
1. Techno-economic model: ALCC, LCC, payback, 25-year cumulative savings
   55,113.57 USD (present value 43,265.77 USD per Section 4.2 narrative).
1. Environmental accounting: Embodied energy, lifetime CO₂ mitigation 180,704.32
   kg (C-DSTS) and 206,921.09 kg (GI-DSTS); carbon credits > 4000 USD at 23
   USD/t.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

Location: Vellore Institute of Technology, Vellore, Tamil Nadu, India
(12.9236°N, 79.1331°E).

Slab / absorber (Table 3):

- M20 concrete slab: 0.8 m × 1.65 m × 0.13 m
- Absorber plate: 0.7 m × 1.55 m × 0.002 m (Cu or GI); 5 serpentine pipes (ID
  0.0127 m, OD 0.0147 m)
- PV: Frameless mono 160 Wp, 0.7 m × 1.5 m (Pmax 160 W, Vmp 18.8 V, Imp 8.52 A)
- TEG: 50 units, 10×5 grid, parallel; Qmax 50 W, Vmax 14.4 V, Imax 6.4 A (at 25
  °C spec)
- Tank: 220 L; pump 1.5 hp; insulation: polyurethane + thermocol; pipes: C-PVC
Test arrangement: Slab on 0.15 m blocks over polyurethane; thermocol on four
sides; mock indoor room under slab for cooling measurement.

Procedure: March 2024; data 11:00–16:00, 15-min intervals (IS 3370-1, IS 12976
cited); prototype concrete k = 2.2 W/m·K from conductivity rig.

### 4b. Mathematical Models & Equations

Uncertainty propagation:

- External uncertainty \(=\sqrt{\sum \left(\frac{\partial C}{\partial
  i_j}\right)^2 u^2(i_j)}\) — (1)
- Internal uncertainty % \(=\dfrac{\sqrt{sd_1'^2+\cdots+sd_n'^2}}{\text{mean
  observations}}\times 100\) — (2)
Fluid / heat transfer:

- \(Re = \dfrac{\rho v L_p}{\mu}\) — (3)
- \(Pr = \dfrac{\mu C_p}{k_l}\) — (4)
- \(Nu = 1.86\left(\dfrac{Re\cdot Pr}{L/L_p}\right)^{1/3}\) — (5)
- \(h_c = \dfrac{k_l}{L_p} Nu\) — (6)
- \(\bar{U} =
  \dfrac{1}{\frac{1}{h_c}+\frac{L_p}{k_p}+\frac{L_{concrete}}{k_{concrete}}}\) —
  (7)
Energy / efficiency:

- \(Q_o = \dot{m} C_p (t_{out}-t_{in})\) — (8)
- \(\mathrm{HUF} = \dfrac{\dot{m} C_p (t_{out}-t_{in})}{\dot{A} G \hat{T} F}\) —
  (9)
- \(Q_l = \dot{A} \bar{U} (t_s - t_a)\) — (10)
- \(\eta_T = \dfrac{(Q_o - Q_l)}{\dot{A} G}\times 100\%\) — (11)
- \(F = \dfrac{Q_o}{\dot{A} G}\) — (12)
- \(\mathrm{COP} = \dfrac{Q_o}{Q_o + P_{pump}}\) — (13)
- \(E_o = \dot{m} C_p (t_{out}-t_a)\left(1-\dfrac{t_a}{t_s}\right)\) — (14)
- \(E_i = \dot{A} G \left(1-\dfrac{t_a}{t_s}\right)\) — (15)
- \(\eta_E = E_o/E_i\) — (16)
- \(\eta_P = \dfrac{\dot{V}\dot{I}}{\dot{A} G}\times 100\%\) — (17)
Economics:

- \(\mathrm{ALCC}_{DSTS} = \mathrm{ALCC} + [C_{EWH}+C_{cooling}] - C_{PV-TEG}\)
  — (18)
- \(\mathrm{ALCC} = \mathrm{LCC}\times C_{rf}\) — (19)
- \(\mathrm{LCC} = C_i + \sum_{t=1}^{n}[-(C_{o,t}+C_{m,t}+C_{r,t}) Df_t] - C_s
  Df_n\) — (20)
- \(C_{rf} = \dfrac{r(1+r)^n}{(1+r)^n-1}\) — (21)
- \(A_i = A_{initial}(1+i)^n\) — (22); \(A_d = A_{initial}(1-d)^n\) — (23)
- \(A_c = \sum (A_i + A_d)\) — (24); \(PV = A_c/(1+r)^n\) — (25)
- Payback \(P = C_i / A_{initial}\) — (26)
Environment:

- \(\mathrm{CO_2\ emission} = \hat{E}\times 2.04/n\) — (27)
- Mitigation \(= \hat{E}_T \times 2.04\) — (28); \(\hat{E}_T = D_T \times n_s\)
  — (29)
- \(D_T = Q_h + Q_c\) — (30)
- Net lifetime mitigation \(= [(\hat{E}_T \times n) - \hat{E}\times 2.04\times
  10^{-3}]\) — (31)
- CO₂ credit \(=\) Net mitigation \(\times\) cost per ton — (32)
### 4c. Algorithm / Control Method Steps

N/A — no AI/ML or adaptive control. Operation is manual/semi-manual:

1. Circulate water at set flow via ball valves and flow sensors (0.12 / 0.24 /
   0.36 L/min).
1. Log temperatures (K-type TCs), irradiance (pyranometer), PV V/I (multimeter),
   DAQ every 15 min.
1. TEGs passively convert slab temperature gradient; no closed-loop controller
   for charging/discharging or cooling setpoint.
### 4d. Data Sources & Dataset Details

| Source | Content | Scope |
| --- | --- | --- |
| On-site measurements | \(G\), \(T_a\), slab/PV/room/water temps, flow, PV V/I | Vellore, March 2024, 5 h/day test windows |
| Economic assumptions (Table 6) | Costs, interest 8%, inflation 5%, real interest 3%, electricity 0.048 USD/kWh | 25-year DSTS life; conventional 10-year |
| Operating days | 245 days/year (from 290 sunny days assumption scaled) | 5 h/day operation for annual energy accounting |
| Embodied energy coefficients | Table 8 (kWh/kg materials) | Fabrication LCA inputs |
| Standards cited | IS 3370-1 (2009), IS 12976 (1990) | Concrete + solar water heater testing guidance |

No ERA5, NASA POWER, or ISRO Solar Calculator used.

### 4e. Validation Method

- Experimental comparison: conventional slab vs C-DSTS vs GI-DSTS under
  identical radiation (778–997 W/m² peaks).
- Literature benchmarking (Fig. 7): outlet temperature, roof temperature, room
  temperature, HUF, COP, \(\eta_T\), \(\eta_E\).
- Maximum validation errors (Table 5): thermal efficiency 12.02% (C) / 11.22%
  (GI); exergy 8.08% / 7.18%; outlet water temp error 14.4 °C / 14.8 °C (vs
  reference studies — authors attribute to configuration differences).
- Uncertainty: Overall 5.08% (external 0.25% + internal thermocouple 4.83%).
────────────────────────────────────────

## 5. PCM Details (if applicable)

N/A — the DSTS prototype does not use phase-change materials. Thermal buffering
is provided by the concrete slab mass and water circulation.

Literature context only (Table 2): PCM-integrated BIPVT systems are surveyed at
50–70% thermal efficiency, 14–18% electrical, 12–22% exergy, $350–500/m², 5–7
year payback, 30–50% CO₂ reduction — with note of PCM degradation over time.
Cited hybrid PVT–PCM building study: Li et al., Renew. Energy 199, 662–671
(2022) [ref. 8 in paper].

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A — no artificial intelligence, forecasting, or reinforcement learning.
Control is fixed flow-rate tests with ball valves and a 1.5 hp pump; space
cooling is passive via slab + TEG temperature gradient.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: On-site pyranometer (not satellite reanalysis or NASA POWER).
- Variables used: Global solar irradiance G (778.44–997.51 W/m² peak during
  tests); ambient \(T_a\) 35.33–43.6 °C; inlet water 30.66–32.90 °C.
- Geographic scope: Vellore, Tamil Nadu, India (hot, high-insolation climate).
- Temporal resolution: 15-min logging; test window 11 a.m.–4 p.m.; economic
  extrapolation 245 days/year × 5 h/day.
- Time period covered: March 2024 experiments; 25-year lifecycle analysis.
- Clear-sky index / derived metrics: Not computed; peak irradiance reported
  directly.
────────────────────────────────────────

## 8. Key Results & Numbers

- Overall system efficiency: 46.4%; TEG contribution ~11% average.
- Thermal efficiency (max at 0.36 L/min): 24.75% (C-DSTS), 24.20% (GI-DSTS).
- Exergy efficiency (max): 37.05% (C), 32.30% (GI) at 0.36 L/min.
- Passive indoor cooling: 6.2 °C (C) vs conventional room (31.7 °C → 25.69 °C at
  0.12 L/min); 5.9 °C (GI).
- Water heating: 108 L to 50 °C in ~5 h; outlet peak 48.2 °C (C-DSTS); ΔT
  9.97–11.42 °C across tests.
- Flow-rate sweep: HUF 0.093 / 0.177 / 0.269 (C) and 0.090 / 0.170 / 0.260 (GI)
  at 0.12 / 0.24 / 0.36 L/min; COP up to 3.513 (C) and 3.393 (GI) at 0.36 L/min.
- PV electrical efficiency: 11.38–12.39% (DSTS panels) vs 11.42–12.31%
  conventional panel — cooling from water loop + TEGs limits PV overheating (top
  PV temps 70–74 °C on DSTS).
- TEG output (parallel bank): avg 0.685 V, 50 A; cold-side TEG ~23.3–24.1 °C.
- Annual outputs (economic model): 198 kWh electricity; 26,460 L hot water;
  cooling benefit up to 6.2 °C for 5 h/day.
- Capital cost: 1321.06 USD (C-DSTS), 855.06 USD (GI-DSTS) at 1 USD = ₹85.03.
- Annualized cost: 167.26 USD (C), 108.25 USD (GI) vs 286.30 USD conventional
  EWH + fan/AC.
- Payback: 8.18 years (C), 5.29 years (GI); lifespan 25 years.
- 25-year cumulative savings: 55,113.57 USD (text Section 4.2); present value
  43,265.77 USD.
- Daily savings: 0.17 USD (water heating) + 0.45 USD (cooling).
- Embodied energy: 1658.83 kWh (C fabrication), 1171.12 kWh (GI).
- Lifetime net CO₂ mitigation: 180,704.32 kg (C), 206,921.09 kg (GI); carbon
  credits 4165.37 USD (C), 4194.12 USD (GI) at 23 USD/t.
- Slab top temperature (C-DSTS): 55–59 °C avg 49.09–53.28 °C; bottom–top ΔT up
  to 17.38 °C.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): (1) Conventional concrete slab (no absorber/PV/TEG); (2)
  Conventional electric water heater + fans/AC (80% efficiency assumed, 10-year
  life); (3) Literature BIPVT/PVT configurations (Tables 1–2).
- Proposed method: C-DSTS and GI-DSTS (embedded tri-generation roof slab).
- Improvement margin:
- vs conventional slab: bottom/roof 3.57–4.71 °C cooler (C); room up to ~6 °C
  lower.
- vs conventional electric systems: annualized cost 286.30 → 167.26 USD (C) or
  108.25 USD (GI).
- vs literature PCM-BIPVT band: proposed DSTS Table 2 entry claims 65–82%
  thermal (design target row) vs 50–70% PCM-BIPVT — experimental \(\eta_T\)
  achieved ~25% (lower than table aspirational range).
- Literature cited MODIS+NWP multimodal not in this paper; intro cites 13.2%
  RMSE only in other survey context — not applicable here.
- Conditions: Same Vellore climate; same 5 h solar window; flow 0.36 L/min
  optimal for \(\eta_T\), \(\eta_E\).
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

- Physical components: M20 concrete roof slab; Cu or GI serpentine absorber; 160
  Wp frameless PV; 50× TEG (parallel); 220 L tank; 1.5 hp pump; ball valves;
  C-PVC pipes; polyurethane + thermocol insulation.
- Sensor specs (Table 4):
- Pyranometer: 0–2000 W/m², accuracy ±0.1 W/m²
- K-type thermocouples: −200 to 650 °C, ±0.01 °C (internal uncertainty ±4.83%)
- Flow sensor: 0–40 L/s, ±0.01
- Multimeter: 200 mV–1000 V, 200 μA–200 mA
- DAQ for logging
- Embedded/compute platform: DAQ + manual multimeter — no Raspberry Pi / Arduino
  / ESP32.
- Test environment: Outdoor rooftop mock-up at VIT Vellore with insulated
  underside “room” for passive cooling tests.
- Test duration: March 2024; 5 h/day (11:00–16:00); 15-min sampling; three
  flow-rate days per configuration.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Performance evaluated only under Vellore conditions (high irradiance, 35–43.6
  °C ambient); “performance metrics may vary” in low insolation, humidity, or
  cold climates (Conclusion).
- Multi-location validation and seasonal extremes beyond scope; thermal mass
  behavior may differ (Conclusion).
- Retrofitting challenges not fully studied: structural modifications,
  plumbing/electrical integration, roof-space limits (Conclusion).
- PV panels tested without tilt, though cooling maintained PV efficiency near
  conventional tilted panels.
- Validation vs literature shows up to 12.02% thermal efficiency deviation and
  14.4 °C outlet temperature error vs some reference systems (Table 5).
- Economic model assumes 245 operating days/year, 5 h/day — actual household use
  may extend beyond this (Section 4.1).
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Not Relevant. Flow rates are manually set
  (0.12–0.36 L/min); no PPO/DDPG, no climate-forecast-driven charging — fixed
  test protocol only.
- RG2 (No integrated PCM–AI–hardware prototype): Partially relevant. Real
  multi-sensor experimental rig (pyranometer, thermocouples, flow, DAQ) in India
  (Vellore) parallels your sensing stack, but no PCM, no AI, and architecture is
  BIPVT-TEG concrete slab, not Rubitherm/PLUSS tank + ESP32/RPi.
- RG3 (Poor alignment with household demand patterns): Partially relevant.
  Delivers 108 L at 50 °C in 5 h and models 26,460 L/year — residential scale,
  but no dynamic draw profile (unlike Edwards-type demand); constant-flow
  experiments only.
- RG4 (Limited real-world experimental validation): Highly relevant. Field
  experiments in Tamil Nadu with quantified \(\eta_T\), \(\eta_E\), and indoor
  ΔT — supports feasibility of Indian outdoor SWH-related research; your FYP can
  contrast their concrete sensible storage with PCM latent storage.
- RG5 (No predictive optimization under climatic uncertainty): Not Relevant. No
  ERA5/forecasting or optimization under weather uncertainty; single-site March
  2024 campaign only.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
| --- | --- | --- |
| \(Q_o = \dot{m} C_p (t_{out}-t_{in})\) (8) | Useful heat to water | PCM-SWH energy balance / charging rate |
| \(\eta_T = (Q_o-Q_l)/(\dot{A}G)\) (11) | Collector thermal efficiency | KPI vs rule-based and RL policies |
| \(\eta_E = E_o/E_i\) (14)–(16) | Exergy efficiency | Second-law metric for PCM charging quality |
| \(\mathrm{COP} = Q_o/(Q_o+P_{pump})\) (13) | Pumping penalty | Include solenoid/pump power in reward function |
| \(\eta_P = \dot{V}\dot{I}/(\dot{A}G)\) (17) | PV efficiency vs irradiance | Couple pyranometer to electrical auxiliary offset |
| Eqs. (18)–(26) ALCC/LCC/payback | Lifecycle economics | Optional FYP techno-economic appendix for PCM-SWH |
| Eqs. (27)–(32) CO₂ mitigation | Environmental benefit | Cite Indian building-scale CO₂ reduction context |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. J. Li et al., "A hybrid photovoltaic and water/air based thermal (PVT) solar
   energy collector with integrated PCM for building application," Renew.
   Energy, 2022 [8] — Relevant because: Direct PCM + PVT building collector —
   closest architectural analog to PCM-SWH latent storage.
1. M. Emam et al., "Year-round experimental analysis of a water-based PVT-PCM
   hybrid system: comprehensive 4E assessments," Renew. Energy, 2024 [10] —
   Relevant because: Experimental PVT-PCM with
   energy/exergy/economic/environment metrics.
1. L. Xu et al., "Hybrid PV thermal wall with double air channel and phase
   change material: seasonal experimental research," Renew. Energy, 2021 [25] —
   Relevant because: PCM + hybrid PVT seasonal outdoor data for building
   integration.
1. V.S. Chandrika et al., solar concrete water heater 57% thermal efficiency in
   Tamil Nadu [57] (cited in Section 1 review) — Relevant because: South India
   experimental solar concrete WH benchmark near your geography.
1. B. Duraivel et al., "Extensive analysis of a reinvigorated solar water
   heating system using low-density polyethylene glazing," Energies, 2023 [4] —
   Relevant because: Same author group’s Indian SWH experimental work preceding
   DSTS.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 24. Eldokaishi2022WaterPCM_ANN_SWH_summary.md

Source path: /mnt/data/Eldokaishi2022WaterPCM_ANN_SWH_summary.md

# Modeling of Water-PCM Solar Thermal Storage System for Domestic Hot Water Application Using Artificial Neural Networks

Authors: A.O. Eldokaishi, M.Y. Abdelsalam, M.M. Kamal, H.A. Abotaleb

Year: 2022

Journal/Conference: Applied Thermal Engineering, Vol. 204, Article 118009

DOI: https://doi.org/10.1016/j.applthermaleng.2021.118009

IEEE Citation: A. O. Eldokaishi, M. Y. Abdelsalam, M. M. Kamal, and H. A.
Abotaleb, "Modeling of water-PCM solar thermal storage system for domestic hot
water application using artificial neural networks," Appl. Therm. Eng., vol.
204, p. 118009, 2022, doi: 10.1016/j.applthermaleng.2021.118009.

────────────────────────────────────────

## 1. One-Line Summary

This paper trains a Keras/TensorFlow feed-forward ANN surrogate (R² up to
0.9999, ~10⁵× faster than physics-based simulation) on an experimentally
validated water–PCM SDHW model to predict solar fraction and generate design
maps for tank volume, PCM volume fraction, and melting temperature.

────────────────────────────────────────

## 2. Problem Being Solved

- Transient, nonlinear PCM phase-change makes annual or large-parameter-sweep
  numerical SDHW simulation computationally prohibitive (e.g., >120 h for 84,480
  cases), limiting comprehensive design guidelines.
- Literature lacks systematic study of ANN applicability specifically to
  PCM-integrated solar thermal storage with tuned sampling and hyperparameters.
- Optimal PCM volume fraction, melting temperature, and tank volume for hybrid
  sensible–latent tanks remain unclear without fast performance predictors.
- Engineers need visual design maps (solar fraction contours) relating collector
  area, tank size, PCM fraction, and \(T_p\) without running full transient
  models for every point.
────────────────────────────────────────

## 3. Key Contributions

1. End-to-end framework: experimentally validated Abdelsalam et al. hybrid
   water–PCM tank model → training data → Sobol / LHS / Monte Carlo sampling
   comparison (39 ANN models) → hyperparameter optimization → design maps.
1. Demonstrates Sobol sequence sampling outperforms LHS and MC at low sample
   counts (44% and 16% lower testing MAE vs LHS and MC, respectively).
1. Optimized multi-ANN ensemble (3 best models, outlier rejection, average of
   two): testing MAE = 0.00114, RMSE = 0.001745, R² = 0.99990, max absolute
   error = 0.0203 on solar fraction.
1. Polynomial regression surrogate (Eq. 14) for SF with R² = 0.98481 — shown
   inferior to ANN (MAE 0.02010 vs 0.00114).
1. PCM–SDHW design insights: e.g., +13% solar fraction when \(V_f\) increases 0
   → 0.5 in 90 L tank at \(T_p = 28\,°\mathrm{C}\); 90 L PCM tank matches 210 L
   water-only tank SF; up to 57% tank volume reduction possible with proper PCM
   selection.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

Underlying physics model (not run live in this paper): Abdelsalam et al. [4,17]
hybrid thermal storage — flat-plate solar collector charging loop + domestic
load discharging loop.

Tank / PCM geometry:

- Stratified storage tank with immersed coil HX: bottom coil (collector
  charging), top coil (load discharge).
- Cylindrical PCM modules, 20 mm diameter, installed vertically inside tank.
- ON/OFF circulation pump controlled by collector-to-tank-bottom temperature
  difference (\(\Delta T_{on}\), \(\Delta T_{off}\)); supply limited to 90 °C to
  avoid boiling.
- Load side: auxiliary heater + tempering valve to maintain setpoint \(T_l\).
Parameter ranges (Table 1):

| Parameter | Range |
| --- | --- |
| Collector area \(A_c\) | 1–8 m² |
| Tank volume \(V_{st}\) | 50–240 L |
| Load temperature \(T_l\) | 55–60 °C |
| PCM melting \(T_p\) | 25–35 °C |
| PCM volume fraction \(V_f\) | 0.0–0.7 |

Boundary / operating data:

- Weather: hourly solar irradiance and dry-bulb temperature — typical spring
  day, Toronto, Canada (weather.gc.ca).
- Demand: dispersed hot-water draw profile from Edwards et al. [19]; 8 L/min
  draw rate; 189 L/day total.
ANN software: Python; Keras + TensorFlow; feed-forward multilayer perceptron.

### 4b. Mathematical Models & Equations

Collector pump control (ON/OFF hysteresis):

- \(\Delta T_{off} \leq \dfrac{A_c \times F_R \times U_L}{\dot{m} \times C_w
  \times \Delta T_{on}}\) — (1)
Collector heat removal factor:

- \(F_R = \dfrac{\dot{m} C_w \left(1 - e^{-A_c U_L F' / (\dot{m}
  C_w)}\right)}{A_c U_L}\) — (2)
Input normalization:

- \(X' = \dfrac{X - \bar{X}}{\sigma}\) — (3)
Loss functions:

- \(\mathrm{MAE} = \dfrac{1}{n}\sum_{i=1}^{n}|y_i - y'_i|\) — (4)
- \(\mathrm{MSE} = \dfrac{1}{n}\sum_{i=1}^{n}(y_i - y'_i)^2\) — (5)
- \(\mathrm{RMSE} = \sqrt{\dfrac{1}{n}\sum_{i=1}^{n}(y_i - y'_i)^2}\) — (6)
Solar fraction (ANN target output):

- \(\mathrm{SF} = \dfrac{\text{Thermal energy delivered to load}}{\text{Total
  thermal demand}}\) — (7)
Activation functions tested (examples):

- ReLU: \(f(x) = \max(0,x)\) — (8)
- Sigmoid: \(f(x) = 1/(1+e^{-x})\) — (9) (selected)
- Softplus, tanh, SELU, ELU — (10)–(13)
Regression surrogate for SF (polynomial in \(A_c, V_{st}, T_l, T_p, V_f\)):

- \(\mathrm{SF} = -0.009248 A_c^2 + 0.000055 A_c V_{st} - 0.001532 A_c T_l +
  0.000589 A_c T_p + 0.001477 A_c V_f - 0.000003 V_{st}^2 + 0.000004 V_{st} T_p
  - 0.000242 V_{st} V_f - 0.000068 T_p^2 + 0.002308 T_p V_f - 0.04611 V_f^2 +
  0.21993 A_c + 0.000779 V_{st} + 0.000112 T_l + 0.00086 T_p - 0.01 V_f -
  0.1908\) — (14)
PCM phase-change inside modules is handled by the cited Abdelsalam et al. [17]
immersed-coil + PCM model (enthalpy-based), not re-derived in this paper.

### 4c. Algorithm / Control Method Steps

ANN training workflow:

1. Run validated numerical model over design space; each sample = 5 inputs
   (\(A_c, V_{st}, T_l, T_p, V_f\)) + 1 output (SF).
1. Normalize inputs with Eq. (3).
1. Sample training set via Monte Carlo, Latin hypercube (LHS), or Sobol
   sequences (up to 10,000 samples; 13 sample-count levels: 250, 500, …, 10000 →
   39 models).
1. Build feed-forward ANN; initialize synaptic weights; train for multiple
   epochs minimizing MAE/MSE/RMSE with Adam optimizer.
1. Evaluate on 84,480 held-out test points (full factorial-style coverage, never
   seen in training).
1. Hyperparameter tuning (Sobol, 3,000 training samples): learning rate, hidden
   layers (1–4), neurons/layer (30, 40, 50), activation function.
1. Multi-ANN prediction: select 3 best models; drop outlier per point; average
   remaining two → ~10% MAE and ~21% max-error reduction vs single model.
1. Deploy optimized ANN to generate SF contour maps vs \(V_{st}\), \(V_f\),
   \(T_p\) at fixed \(A_c = 4\,\mathrm{m}^2\), \(T_l = 55\,°\mathrm{C}\).
Optimized hyperparameters (Table 7):

| Hyperparameter | Value |
| --- | --- |
| ANN type | Feed-forward multi-layer |
| Input neurons | 5 |
| Output neurons | 1 |
| Hidden layers | 3 |
| Neurons per hidden layer | 50 |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Activation | Sigmoid |

### 4d. Data Sources & Dataset Details

| Source | Variables | Resolution / scope | Period |
| --- | --- | --- | --- |
| Abdelsalam et al. [4,17] numerical model | Tank temps, PCM state, SF, collector operation | Transient simulation per design point | Single-day Toronto spring + daily demand |
| Environment Canada weather (weather.gc.ca) | Solar irradiance, ambient \(T_a\) | Hourly | One spring day, Toronto |
| Edwards et al. [19] draw profile | Hot water flow rate | High-resolution demand; scaled to 189 L/day | 24 h cycle |
| ANN training sets | \(A_c, V_{st}, T_l, T_p, V_f\) → SF | Up to 10,000 training points (Sobol/LHS/MC) | Design-space sweep |
| ANN test set | Same 5 inputs → SF | 84,480 samples | Full studied range |

No ERA5, NASA POWER, or Indian city data used.

### 4e. Validation Method

- Training data generator: physics model experimentally validated in prior work
  [4] (direct vs indirect HX SDHW with PCM).
- ANN validation: held-out 84,480 test samples; metrics MAE, RMSE, R², max
  absolute error.
- Best multi-ANN: MAE = 0.00114, RMSE = 0.001745, R² = 0.99990, max error =
  0.0203 (solar fraction scale 0–1).
- Abstract peak claim: R² = 0.9999 after proper configuration.
- Speed benchmark: full test set — numerical model >120 h vs ANN ~5 s (~5 orders
  of magnitude reduction).
- 80% of ANN test points have MAE < 2×10⁻³; regression model 80% with MAE <
  3.2×10⁻².
────────────────────────────────────────

## 5. PCM Details (if applicable)

- Materials tested: Capric-acid-like organic PCM (properties from Abhat [33]);
  not Rubitherm RT or PLUSS OM grades.
- Melting temperature range: 25–35 °C (\(T_p\) design sweep).
- Latent heat: 182 kJ/kg
- Thermal conductivity: Not varied in this study (fixed in underlying [17]
  model); water conductivity 0.63 W/m·K listed for tank water.
- Specific heat (solid/liquid): Water \(C_w = 4.18\) kJ/kg·K (collector fluid);
  PCM \(c_p\) embedded in referenced model.
- Density: PCM \(\rho_p = 870\) kg/m³; water \(\rho_w = 993\) kg/m³.
- Performance metrics reported: Solar fraction (SF); SF improvements with PCM
  (+13%, +5% cases); 57% tank volume reduction potential; literature cites 40%
  tank volume reduction with PCM modules [4], 20–40% storage density gain [3].
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm: Feed-forward artificial neural network (Keras/TensorFlow); Adam
  optimizer; compared sampling: Sobol, LHS, MC; optional multi-ANN ensemble;
  polynomial regression baseline Eq. (14).
- Input features / state space: \(A_c\) [m²], \(V_{st}\) [L], \(T_l\) [°C],
  \(T_p\) [°C], \(V_f\) [–] (PCM volume / tank volume).
- Output / action space: Solar fraction SF [dimensionless, 0–1] — prediction
  only, not control actions.
- Model architecture: 3 hidden layers × 50 neurons each; sigmoid activation; 5
  inputs, 1 output (Table 7, Fig. 10).
- Hyperparameters: Learning rate 0.005 (best among 0.1, 0.01, 0.005, 0.001);
  epochs varied during training (overfitting monitored); loss = MAE/MSE/RMSE.
- Training data size: Up to 10,000 per sampling study; 3,000 Sobol samples for
  final hyperparameter optimization; 39 models in sampling comparison.
- Hardware used for training: Not stated (Python on PC implied).
- Performance metrics:
- Multi-ANN test: MAE = 0.00114, RMSE = 0.001745, R² = 0.99990, max MAE = 0.0203
- Single ANN (best): MAE = 0.00126, R² = 0.99987
- Regression: MAE = 0.02010, RMSE = 0.024866, R² = 0.98481
- Sobol vs LHS/MC: 44% / 16% MAE reduction at low sample counts
- Multi vs single ANN: ~10% MAE, ~21% max-error reduction
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: Environment Canada (https://weather.gc.ca) — not ERA5, NASA
  POWER, ISRO, or Global Solar Atlas.
- Variables used: Solar incident radiation (hourly), dry-bulb ambient
  temperature \(T_a\); tank surroundings fixed at 20 °C in Table 1.
- Geographic scope: Toronto, Canada — single spring day profile (Fig. 2).
- Temporal resolution: Hourly weather; demand profile at high temporal
  resolution from [19].
- Time period covered: One representative day (not annual, not multi-year).
- Clear-sky index / derived metrics: Not computed.
────────────────────────────────────────

## 8. Key Results & Numbers

- Sobol sampling reduces testing MAE by 44% vs LHS and 16% vs Monte Carlo at low
  training-sample counts.
- 39 ANN models trained (3 sampling methods × 13 sample sizes from 250 to
  10,000).
- Beyond 3,000 training samples, further sample increase gives diminishing MAE
  improvement.
- Best learning rate 0.005: test MAE = 0.00190, R² = 0.99976 (Table 2); inferior
  rates: LR 0.001 → MAE = 0.00395.
- Best topology: 3 hidden layers, 50 neurons/layer → MAE = 0.00142, R² = 0.99986
  (Table 3).
- Sigmoid activation: MAE = 0.00126 vs ReLU 0.00165 (Table 4).
- Multi-ANN vs single: MAE 0.00114 vs 0.00126; max error 0.0203 vs 0.0256; R²
  0.99990 vs 0.99987.
- Regression vs multi-ANN: MAE 0.02010 vs 0.00114 (~17.6× higher error).
- 80% of ANN predictions: MAE < 2×10⁻³; regression: 80% with MAE < 3.2×10⁻².
- Computational time for 84,480 test simulations: numerical >120 h vs ANN ~5 s.
- Design case (\(A_c = 4\,\mathrm{m}^2\), \(T_l = 55\,°\mathrm{C}\), \(T_p =
  28\,°\mathrm{C}\)): 90 L tank, \(V_f\) 0 → 0.5 → SF increase ~13%; 150 L tank
  same change → ~5%.
- 90 L tank with PCM can match SF of 210 L water-only tank (Fig. 11) — ~57%
  volume reduction cited in conclusions.
- Collector area range studied: 1–8 m²; tank volume 50–240 L; PCM fraction
  0–0.7.
- Pump control: \(\Delta T_{off} = \mathbf{2\,K}\); collector \(\tau\alpha =
  0.8\), \(U_L = 5.0\,\mathrm{W/m^2K}\), \(F' = 0.84\).
- Daily load: 189 L at 8 L/min peak draw rate.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): (1) Full Abdelsalam et al. transient numerical model; (2)
  Polynomial regression Eq. (14); (3) Single ANN vs multi-ANN; (4) MC and LHS
  sampling vs Sobol.
- Proposed method: Sobol-sampled, hyperparameter-tuned multi-ANN ensemble (3×50
  sigmoid, Adam LR 0.005).
- Improvement margin:
- vs numerical model: ~10⁵× faster (120 h → 5 s for 84,480 points).
- vs regression: MAE 0.00114 vs 0.02010 (R² 0.99990 vs 0.98481).
- vs single ANN: ~10% lower MAE, ~21% lower max error.
- vs LHS/MC sampling: 44% / 16% MAE reduction (Sobol, low-N regime).
- Conditions of comparison: Same 84,480 test design points; same underlying
  physics and Toronto spring day + 189 L demand for all SF labels.
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — this paper develops an ANN surrogate; no new sensors, actuators, or
embedded platform is built. Physical validation is inherited from Abdelsalam et
al. [4] (prior experimental/numerical hybrid PCM-SDHW work). System control
modeled as ON/OFF pump and auxiliary heater + tempering valve, not Raspberry Pi
/ ESP32 implementation.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Numerical modeling of solar TES over long-term (annual) operation is
  computationally demanding; this motivates ANN but the study itself uses a
  single-day weather profile (Introduction, Section 3).
- Framework can be extended to include collector area, weather profile, and
  demand profile as additional inputs for more comprehensive maps — not done in
  current work (Conclusions).
- PCM modules have low conductivity and specific heat vs water; excessive
  \(V_f\) reduces SF due to incomplete melting/solidification and PCM acting as
  poor sensible storage (Section 5c).
- Larger tanks suffer higher surface losses; SF peaks then drops with increasing
  \(V_{st}\) (Fig. 11, citing [31]).
- Improper \(T_p\) selection causes partial phase transformation and
  under-utilization of latent heat (Section 5c).
- ANN training risks overfitting if epochs are excessive (Section 4b).
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Not Relevant (as implemented). The ANN
  predicts offline solar fraction for design sweeps; pump logic is fixed ON/OFF
  (Eq. 1), not PPO/DDPG or climate-adaptive MPC. Supports using ML as a fast
  plant model inside a future controller, not a deployed controller.
- RG2 (No integrated PCM–AI–hardware prototype): Partially relevant. Strong
  precedent for Python + TensorFlow/Keras ANN on PCM-SDHW (transferable to
  TFLite edge inference), but no RPi/ESP32/DS18B20 hardware; PCM is
  capric-acid-like, not Rubitherm RT / PLUSS OM.
- RG3 (Poor alignment with household demand patterns): Partially relevant. Uses
  a realistic high-resolution draw profile [19] scaled to 189 L/day and 8 L/min
  — closer to demand-aware design than pure step loads, but not
  Coimbatore/Jaisalmer/Kochi profiles or your three-zone comparison; authors
  note demand/weather as future ANN inputs.
- RG4 (Limited real-world experimental validation): Partially relevant. Training
  labels come from a model validated experimentally in [4], but this paper adds
  no new field tests; single-day Toronto simulation limits RG4 claims for India
  climates.
- RG5 (No predictive optimization under climatic uncertainty): Partially
  relevant. Cites ANN literature for solar irradiance forecasting [7–9] but uses
  deterministic one-day weather; no ERA5/NASA POWER or forecast-driven
  optimization — useful as surrogate for climate sweeps if retrained on ERA5 for
  Coimbatore/Jaisalmer/Kochi.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
| --- | --- | --- |
| \(\mathrm{SF} = Q_{solar\to load}/Q_{demand}\) (7) | System performance KPI | RL reward / benchmark metric vs rule-based baseline |
| \(\Delta T_{off} \leq A_c F_R U_L / (\dot{m} C_w \Delta T_{on})\) (1) | Collector pump hysteresis | Rule-based baseline controller in Phase 1 |
| \(F_R = \dfrac{\dot{m} C_w (1 - e^{-A_c U_L F'/(\dot{m} C_w)})}{A_c U_L}\) (2) | Collector heat removal factor | Collector sub-model in grey-box simulator |
| \(X' = (X-\bar{X})/\sigma\) (3) | Input scaling for ML | XGBoost/ANN feature pipeline for climate + tank states |
| MAE / RMSE (4)–(6) | Surrogate accuracy metrics | Compare TFLite ANN vs XGBoost vs physics model |
| Eq. (14) polynomial SF | Fast design surrogate | Initial sizing before RL training; inferior to ANN here |
| Sigmoid (9) | Hidden activation | Reference if shallow ANN used on ESP32 |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. M.Y. Abdelsalam et al., "Hybrid thermal energy storage with phase change
   materials for solar domestic hot water applications: Direct versus indirect
   heat exchange systems," Renew. Energy, 2020 [4] — Relevant because:
   Experimentally validated water–PCM SDHW tank architecture (coil HX, PCM
   modules) that this ANN replaces computationally.
1. M.Y. Abdelsalam et al., "A novel approach for modelling thermal energy
   storage with phase change materials and immersed coil heat exchangers," Int.
   J. Heat Mass Transf., 2019 [17] — Relevant because: PCM + immersed coil
   transient model equations underlying training data.
1. W. Yaïci and E. Entchev, "Performance prediction of a solar thermal energy
   system using artificial neural networks," Appl. Therm. Eng., 2014 [14] —
   Relevant because: Prior ANN for SDHW stratification and solar fraction (±10%
   SF accuracy cited in intro).
1. S. Edwards et al., "Representative hot water draw profiles at high temporal
   resolution…," Sol. Energy, 2015 [19] — Relevant because: Household demand
   profiles for aligning PCM discharge with realistic loads (RG3).
1. A. Najafian et al., "Integration of PCM in domestic hot water tanks:
   Optimization for shifting peak demand," Energy Build., 2015 [6] — Relevant
   because: PCM placement and volume optimization in DHW tanks; ANN used for
   discharge time in related work.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 25. Emami2026DRL_Solar_ORC_TES_summary.md

Source path: /mnt/data/Emami2026DRL_Solar_ORC_TES_summary.md

# Deep Reinforcement Learning-Based Smart Control of Solar-Driven Power Cycle with Thermal Energy Storage: A Los Angeles Case Study

Authors: Araz Emami, Ata Chitsaz, Amirali Nouri

Year: 2026 (published online 18 December 2025)

Journal/Conference: Energy Conversion and Management: X, Vol. 29, Article 101478

DOI: https://doi.org/10.1016/j.ecmx.2025.101478

IEEE Citation: A. Emami, A. Chitsaz, and A. Nouri, "Deep reinforcement
learning-based smart control of solar-driven power cycle with thermal energy
storage: A Los Angeles case study," Energy Convers. Manag.: X, vol. 29, p.
101478, 2026, doi: 10.1016/j.ecmx.2025.101478.

────────────────────────────────────────

## 1. One-Line Summary

This paper trains a MATLAB–CoolProp DDPG supervisor on 8760 h of NSRDB solar
data to jointly regulate ORC superheat, turbine inlet pressure, and net
efficiency via pump mass-flow commands, achieving ~6 percentage-point higher
annual mean efficiency and stable paraffin-TES cycling versus a fixed-flow
passive baseline in a Los Angeles solar-ORC case study.

────────────────────────────────────────

## 2. Problem Being Solved

- Solar-driven organic Rankine cycles (ORCs) face tightly coupled control of
  working-fluid superheat, turbine inlet pressure, and efficiency under steep
  irradiance ramps (up to ±100 W·m⁻²·min⁻¹), which conventional decoupled PID
  loops handle poorly.
- Fixed-mass-flow baseline operation with passive TES dispatch causes turbine
  inlet pressure swings (~1.9–4.0 MPa vs 2.5 MPa design), superheat deviations
  exceeding ±10 K from a +10 K target, and long near-zero efficiency periods.
- Prior DRL work (e.g., Wang et al.) addressed superheat only under short
  synthetic disturbances, not joint multi-objective control under realistic
  year-long solar variability.
- No prior single DRL agent was demonstrated to simultaneously coordinate
  superheat safety, pressure integrity, and thermodynamic efficiency on real
  solar-thermal input while preserving full nonlinear cycle physics.
────────────────────────────────────────

## 3. Key Contributions

1. Multi-objective DDPG supervisory controller with five-dimensional state and
   continuous normalized mass-flow action, integrated with CoolProp 6.4 for
   non-linear R245fa ORC thermodynamics without model reduction.
1. Composite 8760 h GHI training profile from NSRDB (Los Angeles, 34.05°N,
   118.25°W) capturing clear-sky, transient, and low-flux regimes (15% of hours
   <100 W·m⁻²; 18% >800 W·m⁻²).
1. Parabolic-trough collector + rule-based paraffin PCM-TES buffering upstream
   of the evaporator, with lumped TES charge/discharge/idle logic and ambient
   loss model.
1. Full-year closed-loop results vs fixed-flow baseline: ~6 percentage-point
   mean annual efficiency gain, pressure held within ~4% of 2.5 MPa, superheat
   within ±0.2 K of +10 K, and disciplined 0–250 MJ TES SOC cycles under DRL.
1. Post-hoc multi-objective genetic algorithm (GA) on DRL-controlled operating
   data mapping pressure–efficiency–temperature Pareto front (non-dominated
   cluster near 2.55 MPa, 10.1–10.7 K superheat, ~28.5% peak efficiency).
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

- Plant: Solar-ORC with parabolic-trough collector field (LS-2 style, η₀=0.765),
  shell-and-tube evaporator, axial turbine (ηₜ=0.75), water-cooled condenser (25
  °C loop, 5 K approach), gear pump (ηₚ=0.70, ηₑ=0.90).
- Working fluid: R245fa; nominal evaporation 170 °C / 2.5 MPa, condensation 40
  °C / 450 kPa; nominal thermal rating ~10 kW at 20 m² aperture (~550–600 W·m⁻²
  peak spring insolation).
- TES: Paraffin wax PCM upstream of evaporator; rule-based charging (excess
  heat, SOC≤1), discharging (deficit heat, SOC above minimum), idle otherwise;
  SOC band 0–250 MJ in seasonal results.
- Software: MATLAB environment + CoolProp 6.4 property calls; explicit Euler
  integration Δt = 3600 s (hourly GHI alignment); refined Δt = 600 s changes
  annual yield <0.4%.
- Simulation scope: 8760 h annual runs; baseline = constant mass flow +
  uncoordinated TES; proposed = DDPG pump-frequency command with safety filter.
- Site: Los Angeles, California (NSRDB v3, 34.05°N, 118.25°W); composite trace
  8970 h before hourly averaging (Weeks A/B/C from DOY 70–76, 96–100, 110–114).
### 4b. Mathematical Models & Equations

Collector efficiency (EN 12975 quadratic):

- \(\eta_{col} = \eta_0 - a_1 \dfrac{\Delta T}{I_b} - a_2 \dfrac{\Delta
  T^2}{I_b}\) — (1)
(\(\eta_0=0.765\), \(a_1=0.71\) W·m⁻²·K⁻¹, \(a_2=0.0015\) W·m⁻²·K⁻², \(\Delta T
= T_m - T_a\))

Useful collector heat:

- \(Q_u = \eta_{col}\, I_b\, A_{ap}\) — (2)
HTF energy balance:

- \(m_{HTF} c_{p,HTF} \dfrac{dT_{out}}{dt} = \dot{Q}_u - \dot{m}\,
  c_{p,HTF}(T_{out}-T_{in})\) — (3)
(\(\dot{m}\) commanded by DRL agent)

Evaporator outlet / wall dynamics:

- \(T_{evap,out} = T_{sat}(P_{evap}) + \Delta T_{sh}\) — (4)
- \(C_{evap} \dfrac{dT_{evap}}{dt} = Q_{in} - \dot{m}\, h_{fg}\) — (5)
Turbine:

- \(\dot{W}_{turb} = \dot{m}(h_{in}-h_{out})\) — (6)
- \(h_{out} = h(P_{cond}, s_{in}) / \eta_t\) (isentropic reference) — (7)
Condenser:

- \(\dot{Q}_{cond} = \dot{m}(h_{out,turb}-h_{cond,out})\) — (8)
Pump:

- \(\dot{W}_{pump} = \dfrac{\dot{m}(h_{evap,in}-h_{cond,out})}{\eta_p \eta_e}\)
  — (9)
Net efficiency (observed by controller):

- \(\eta_{net}(t) =
  \dfrac{\dot{W}_{turb}(t)-\dot{W}_{pump}(t)}{\dot{Q}_{in}(t)}\) — (10)
State vector (MDP observation):

- \(\mathbf{s}_t = [I_b(t),\, T_{evap}(t),\, P_{in}(t),\, \Delta T_{sh}(t),\,
  \eta_{net}(t)]^T\) — (11)/(12)
Action mapping:

- \(a_t \in [-1,1] \Rightarrow \dot{m}_t = 0.075 + 0.025\, a_t\
  \mathrm{kg{\cdot}s^{-1}}\) (range 0.05–0.10 kg·s⁻¹) — (13)
Reward:

- \(r_t = -0.50|\Delta T_{sh}-10| - 0.30\dfrac{|P_{in}-2.5|}{0.1} +
  0.20\,\eta_{net}\) — (14)
(instant penalty −25 if \(P_{in}>3.0\) MPa or mass-flow ramp violates ~15% min⁻¹
limit)

TES heat rate:

- \(\dot{Q}_{TES} = \begin{cases} \dot{Q}_{ch}-\dot{Q}_{loss}, & \text{Charging}
  \\ -(\dot{Q}_{dis}+\dot{Q}_{loss}), & \text{Discharging} \\ 0, & \text{Idle}
  \end{cases}\) — (19)
- \(\dot{Q}_{loss} = UA(T_{PCM}-T_{amb})\) — (20)
Exploration noise (Ornstein–Uhlenbeck):

- \(dN_t = \theta(\mu_N - N_t)\,dt + \sigma_t \sqrt{dt}\,\varepsilon_t,\
  \varepsilon_t \sim \mathcal{N}(0,1)\) — (15)
Safety filter (pressure predictor & ramp limit):

- \(\hat{P}_{in} = P_{in} + \kappa(\tilde{\dot{m}}_t - \dot{m}_{t-1}),\ \kappa
  \approx 12\) MPa·s·kg⁻¹ — (17)
- \(\dot{m}^{safe}_t\) limited by ±15% \(\dot{m}_{max}\,\Delta t\) ramp — (18)
### 4c. Algorithm / Control Method Steps

1. Build NSRDB composite GHI → hourly irradiance sequence; convert to collector
   thermal input via PTC model (1)–(3).
1. Initialize ORC + paraffin TES states; set targets: +10 K superheat, 2.5 MPa
   turbine inlet pressure.
1. At each hourly step, observe \(\mathbf{s}_t\) (12); actor MLP outputs \(a_t
   \in [-1,1]\).
1. Map to \(\dot{m}_t\) (13); apply OU noise (15) with \(\sigma_t\) annealed
   0.20 → 0.05 (episodes 3000–9000 per paper schedule).
1. Apply two-layer safety filter (17)–(18); discard unsafe transitions from
   replay buffer.
1. Simulate plant with (4)–(10) + rule-based TES (19)–(20); compute reward (14)
   (normalized zero-mean, unit-variance over first 3000 episodes).
1. Store transitions in replay buffer (10⁶); update actor–critic (DDPG) with
   \(\gamma=0.99\), soft update \(\tau=0.005\).
1. Train until convergence (~9000 episodes): moving-average return > +8 for 5
   consecutive episodes, TD loss plateau, constraint violation rate < 0.5% of
   timesteps.
1. Evaluate on blind 400 h irradiance record from different meteorological year.
1. Apply GA multi-objective optimization on archived DRL-controlled \((P_{in},
   \eta, T_{fluid})\) data for Pareto mapping.
DDPG hyperparameters (Table 3): Actor/Critic MLP 64 → 32, ReLU, layer
normalization; critic merges state+action paths; replay 1×10⁶; OU \(\sigma\):
0.2 initial, linear decay; safety filter enabled; penalty weights also listed as
\(w_\eta=0.5\), \(w_c=5.0\), \(w_u=0.1\) in Table 3 (ablation of full reward
(14) noted as future work).

### 4d. Data Sources & Dataset Details

| Source | Variables | Resolution | Scope | Period / size |
| --- | --- | --- | --- | --- |
| NSRDB v3 | GHI (composite); clearness index \(k_t\) for segment selection | 1 min raw → 1 h after concat | Los Angeles (34.05°N, 118.25°W) | 1998–2022 archive; 8760 h training trace (mean 512 W·m⁻², σ 282 W·m⁻², kurtosis 2.9) |
| Composite weeks | Clear (DOY 70–76), mixed cumulus (96–100), stratocumulus (110–114) | Hourly | Same site | 8970 h pre-average |
| Blind test set | Irradiance | Hourly | Different meteorological year | 400 h |
| GA optimization set | \(P_{in}\), \(\eta_{net}\), working-fluid temperature | From DRL simulation logs | DRL-controlled ORC only | Full-year operational archive |

### 4e. Validation Method

- Training convergence: 5 random seeds converge at episodes 8997–9001 (mean
  9000, σ <8); moving-average episode return > +8 for ≥5 episodes with
  episode-to-episode change < 0.2 reward units.
- Constraint compliance: Worst-case violation rate (pressure >3 MPa or ramp
  limit breach) < 0.5% of timesteps over 20-episode window.
- Generalization: Frozen policy on 400 h blind irradiance — average reward −3%,
  no pressure/ramp violations, cycle-average efficiency > 22%.
- Baseline comparison: Full 8760 h fixed-flow + passive TES vs DDPG on same GHI
  profile (Figs. 8–18).
- TES dispatch fit: Predicted vs actual hourly TES usage R² ≈ 0.99, regression
  slope 0.995, intercept 0.50 kWh.
- Sensitivity (intro): Unseen GHI with overcast periods lengthened 30% degrades
  efficiency < 2 percentage points; pressure stays below safety valve limit.
- No physical experiment: Simulation-only validation in MATLAB–CoolProp.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- Materials tested: Paraffin wax (commercial PCM for TES upstream of ORC
  evaporator; not a SWH tank PCM).
- Melting temperature range: 45–60 °C
- Latent heat: 180–210 kJ/kg
- Thermal conductivity: 0.2–0.4 W/m·K
- Specific heat (solid/liquid): 1.7–2.5 / 2.1–2.9 kJ/kg·K
- Density: 820–900 kg/m³ (solid); 760–800 kg/m³ (liquid)
- Performance metrics reported: TES SOC cycled 0–250 MJ under DRL; rule-based
  charge/discharge; round-trip losses via UA model; DRL achieves regular daily
  SOC cycles vs irregular baseline overfill/underfill (Figs. 11, 18).
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm: Deep Deterministic Policy Gradient (DDPG) continuous-control RL;
  post-hoc multi-objective genetic algorithm (GA) on DRL trajectories.
- Input features / state space: \(I_b(t)\), \(T_{evap}(t)\), \(P_{in}(t)\),
  \(\Delta T_{sh}(t)\), \(\eta_{net}(t)\) — 5D state (Eq. 12). (Solar training
  driver is GHI from NSRDB; state uses beam irradiance \(I_b\) for PTC model.)
- Output / action space: Continuous \(a_t \in [-1,1]\) → mass flow 0.05–0.10
  kg·s⁻¹ (13).
- Model architecture: Actor & Critic: 2 hidden layers 64 → 32, ReLU, layer
  normalization; actor output tanh; critic linear Q-output; state and action
  pathways merged after second hidden layer.
- Hyperparameters: \(\gamma = 0.99\); soft update \(\tau = 0.005\); replay
  buffer 1×10⁶; OU noise \(\sigma\): 0.20 → 0.05; ~9000 training episodes;
  reward weights in (14): 0.50 (superheat), 0.30 (pressure), 0.20 (efficiency).
- Training data size: 8760 hourly steps per episode × ~9000 episodes.
- Hardware used for training: N/A — MATLAB simulation; ~40 s wall-time per 8760
  h episode stated.
- Performance metrics: Annual mean \(\eta_{net}\) +6 percentage points vs
  baseline; superheat ±0.2 K vs ±10 K baseline; pressure within ~4% of 2.5 MPa;
  efficiency band 20–30% under DRL vs baseline 0–30% wide scatter; intro claim
  16% → >22% mean efficiency and 38% improvement vs tuned PID benchmark
  (introduction validation statement).
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: National Solar Radiation Database (NSRDB) Version 3; Perez
  clearness index \(k_t\) for segment classification.
- Variables used: GHI (primary composite input); \(k_t\) thresholds: clear
  \(k_t>0.65\), partly cloudy 0.15–0.65; state/orientation uses \(I_b\) (beam)
  in collector (1)–(2).
- Geographic scope: Los Angeles, California, USA (mid-latitude, high annual
  insolation ~1900 kWh·m⁻²·a⁻¹).
- Temporal resolution: 1 min NSRDB filtered → 1 h simulation timestep.
- Time period covered: NSRDB archive 1998–2022; training composite from selected
  DOY windows; 8760 h annual simulation.
- Clear-sky index / derived metrics: \(k_t\) for week selection; composite mean
  GHI 512 W·m⁻², σ 282 W·m⁻²; 15% of hours <100 W·m⁻², 18% >800 W·m⁻².
────────────────────────────────────────

## 8. Key Results & Numbers

- Annual mean net ORC efficiency increased by ~6 percentage points with DDPG vs
  fixed-flow baseline (Conclusion / §4.2).
- DRL holds turbine inlet pressure within ~4% of 2.5 MPa setpoint; baseline
  swings 1.9–4.0 MPa seasonally (May–Aug peaks >3.5 MPa baseline; DRL summer
  peaks ~3.6 MPa max vs tighter clustering).
- Superheat regulated to ±0.2 K of +10 K target under DRL vs baseline deviations
  >±10 K (Fig. 16 seasonal blocks).
- Net efficiency operated in 20–30% band under DRL vs baseline clusters near 0%
  for extended winter/night periods (Abstract, Conclusion).
- Jan–Apr efficiency: baseline 5–22% → DRL 13–28%; May–Aug: DRL 20–30% vs
  baseline drops <10% at times; Sep–Dec: baseline ~6% min → DRL 15–27%.
- Training converges at episode 9000 (five seeds: 8997–9001); blind 400 h test:
  reward −3%, efficiency >22%.
- TES hourly usage prediction: R² ≈ 0.99, slope 0.995, intercept 0.50 kWh (Fig.
  14).
- TES SOC under DRL: disciplined 0–250 MJ daily cycles; baseline overcharges
  above 250 MJ in summer (Fig. 18).
- GA Pareto peak: ~28.5% efficiency at ~2.55 MPa, superheat 10.1–10.7 K;
  efficiency ridge η ≈ 31% near 2.55 MPa / 10 K (Fig. 19).
- Design sensitivity (Fig. 20): optimal plateau ~30% η near 200 MJ TES and 1200
  m² collector field (ranges 100–300 MJ, 900–1500 m²).
- Pumping energy: DRL commands smooth mass flow; intro sensitivity — overcast
  +30% duration reduces efficiency <2 percentage points.
- Relative to tuned PID: introduction reports ~38% average efficiency increase
  and superheat within 0.01 K during training evaluation (distinct from baseline
  fixed-flow comparison).
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Fixed mass-flow rate ORC operation; uncoordinated TES
  charge/discharge (no active optimization); conventional single-loop PID cited
  in literature but baseline in results is passive/fixed-flow.
- Proposed method: DDPG supervisory controller with OU exploration, safety
  filter, and coordinated pump mass-flow modulation; rule-based TES (not
  RL-learned).
- Improvement margin: +6 percentage points annual mean \(\eta_{net}\); pressure
  stability ~4% vs ±40%+ relative swings; superheat ±0.2 K vs ±10 K; seasonal
  efficiency uplift up to ~15 percentage points in low-GHI quarters.
- Conditions of comparison: Same 8760 h Los Angeles GHI composite, same R245fa
  ORC + paraffin TES model, same MATLAB–CoolProp physics; only control layer
  differs.
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — this paper is purely simulation-based (MATLAB + CoolProp). No physical
sensors, actuators, embedded platforms (RPi/Arduino/ESP32), or field tests are
reported. Authors position the approach as a retrofit-compatible SCADA/software
upgrade pathway requiring measured or simulated plant data only.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Detailed ablation analysis of reward weights, noise model, normalization, and
  safety filters is beyond the scope of this study and left to future work.
- Future work must add direct online learning for evolving solar/load profiles,
  weather forecasting for predictive dispatch, and comparison with MPC and PPO.
- Extension to variable-geometry expanders and real-time multi-objective
  optimization along the Pareto front under changing operator priorities is not
  yet demonstrated.
- Framework validated only in simulation (Los Angeles case); field-scale
  transferability claimed but not experimentally proven in this paper.
- Fig. 13 discussion notes minor systematic bias in learned irradiance-related
  scatter due to underrepresented low-frequency atmospheric regimes in training
  data.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Relevant. DDPG provides continuous
  real-time pump mass-flow adaptation from plant states, cutting
  superheat/pressure excursions versus fixed control—direct methodological
  precedent for your PPO/DDPG charge–discharge–bypass policy, though applied to
  ORC power not domestic SWH.
- RG2 (No integrated PCM–AI–hardware prototype): Partially relevant. Integrates
  PCM-TES + DRL in software (MATLAB) but no embedded prototype
  (RPi/DS18B20/solenoid); supports your gap that AI–PCM coupling remains
  simulation-bound in published ORC work.
- RG3 (Poor alignment with household demand patterns): Not Relevant. Objectives
  are turbine superheat, pressure, and η_net for electricity generation; no
  domestic hot-water draw or morning/evening load profiles.
- RG4 (Limited real-world experimental validation): Relevant as contrast. Full
  8760 h simulation with blind 400 h test, but zero hardware
  validation—strengthens your FYP claim that PCM–AI–SWH needs Indian field/bench
  data beyond Emami-style desktop studies.
- RG5 (No predictive optimization under climatic uncertainty): Partially
  relevant. Training uses historical NSRDB variability; authors explicitly
  propose weather-forecast integration as future work and note the DRL agent
  does not explicitly predict GHI (forecasting is a separate module in their
  discussion). Your XGBoost + ERA5/NASA POWER pipeline addresses this gap for
  Indian sites.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| \(\eta_{net} = (\dot{W}_{turb}-\dot{W}_{pump})/\dot{Q}_{in}\) (10) | Instantaneous useful output ratio | RL reward term for COP/efficiency maximization in grey-box SWH |  |  |  |  |
| \(r_t = -w_{sh} | \Delta T_{sh}-T^* | - w_P | P_{in}-P^* | + w_\eta \eta_{net}\) (14) | Multi-objective safety + performance reward | Template for PPO reward: penalize \(T_w\) error, PCM constraint violations, reward delivered energy |
| \(a_t \mapsto \dot{m}_t = 0.075 + 0.025 a_t\) (13) | Bounded continuous actuation | Analogous mapping for normalized valve/pump command on ESP32 |  |  |  |  |
| \(\dot{Q}_{TES}\) charge/discharge/idle (19)–(20) | PCM storage with ambient loss | Rule-based PCM bypass/charge modes before DRL overrides in hybrid controller |  |  |  |  |
| OU noise (15) + ramp safety (18) | Exploration without actuator damage | Stable-Baselines3 exploration + rate limits on solenoid/pump commands |  |  |  |  |
| \(\eta_{col} = \eta_0 - a_1\Delta T/I_b - a_2(\Delta T)^2/I_b\) (1) | Solar collector thermal input | Couple pyranometer/forecast GHI to collector thermal input in Indian cities |  |  |  |  |
| GA Pareto over \((P_{in}, \eta, T)\) | Post-hoc multi-objective trade space | Offline PCM geometry/PCM-type selection (NSGA-II/PSO) complementing online PPO |  |  |  |  |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Wang X. et al., "Control of superheat of organic Rankine cycle under
   transient heat source based on deep reinforcement learning," Appl. Energy,
   2020 — Relevant because: Foundational DRL superheat control for ORC that this
   paper extends to joint pressure–efficiency objectives under real solar
   profiles.
1. Zalba B. et al., "Review on thermal energy storage with phase change," Appl.
   Therm. Eng., 2003 [32] — Relevant because: Canonical PCM property ranges
   (latent heat, conductivity) cited for paraffin TES sizing in their ORC model.
1. Hernandez A. et al., "Experimental validation of MPC for waste heat recovery
   ORC," Appl. Therm. Eng., 2021 [21] — Relevant because: Benchmark advanced
   model-based control the authors propose comparing against PPO/MPC in future
   work.
1. Imran M. et al., "Dynamic modeling and control strategies of ORC systems,"
   Appl. Energy, 2020 [13] — Relevant because: Reviews PID limitations (e.g., 18
   K superheat overshoot on 50% heat step) motivating RL for thermal plants.
1. Dorokhova M. et al., "DRL control of EV charging in the presence of PV,"
   Appl. Energy, 2021 [33] — Relevant because: Demonstrates DDPG + OU noise for
   solar-volatile systems—methodological parallel to your SB3 DDPG/PPO training
   setup.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 26. Ghodusinejad2026SolarIrradianceForecasting_summary.md

Source path: /mnt/data/Ghodusinejad2026SolarIrradianceForecasting_summary.md

# A Systematic Review of Solar Irradiance Forecasting Across Time Horizons Using Physical, Satellite, and AI-Based Methods

Authors: Mohammad Hasan Ghodusinejad, Nasrin Rashvand, Fatemeh Salmanpour,
Shaghayegh Danehkar, Hossein Yousefi

Year: 2026 (Solar Compass 17, 2026; received 2025)

Journal/Conference: Solar Compass, Vol. 17, Article 100154

DOI/Link: https://doi.org/10.1016/j.solcom.2025.100154

IEEE Citation: M. H. Ghodusinejad et al., "A systematic review of solar
irradiance forecasting across time horizons using physical, satellite, and
AI-based methods," Sol. Compass, vol. 17, p. 100154, 2026, doi:
10.1016/j.solcom.2025.100154.

────────────────────────────────────────

## 1. One-Line Summary

This systematic review taxonomizes solar irradiance forecasting by temporal
horizon (intra-hour to multi-day), input data type (NWP, satellite, ASI), and
model architecture (physical, statistical, ML/DL, hybrid), reporting benchmark
errors such as NOAA GHI RMSE 107–125 W/m² (rRMSE 21–25%), TAPM 45 km lowest RMSE
resolution, NWP+TSI +21% accuracy gain, and XGBoost regional forecasts with R² =
0.9993 and MAPE 0.0119, while advocating physics-informed deep learning for
operational solar applications.

────────────────────────────────────────

## 2. Problem Being Solved

- Solar power integration requires accurate GHI/DNI forecasts across multiple
  horizons to stabilize grids and optimize dispatch.
- Irradiance variability from clouds, aerosols, humidity, terrain, and ozone
  creates large prediction errors that increase costs and curtailment.
- Physical NWP models (WRF, GFS, ECMWF) provide physics consistency but
  under-resolve clouds and aerosols at short horizons.
- Pure statistical/ML models capture local nonlinearities but may fail to
  generalize across climates without physical constraints.
- No unified taxonomy linked forecast horizon, data modality, and architecture
  to expected accuracy outcomes for practitioners.
────────────────────────────────────────

## 3. Key Contributions

1. Horizon taxonomy: very-short/intra-hour (≤60 min), short-term (hours),
   medium-term (days), long-term — mapped to use cases (nowcasting, dispatch,
   planning).
1. Atmospheric driver review: aerosols (AOD550), cloud cover (TSI, optical
   flow), humidity/sky transparency, wind, ozone, terrain effects on WRF bias.
1. Model family comparison: physical (NWP, satellite-to-irradiance),
   statistical, ML/DL (kNN, RF, GBM, MLP, LSTM), and hybrid physics+AI
   post-processing.
1. Quantitative error synthesis: Tables for satellite GHI models (MBE, MAE,
   RMSE, rRMSE, Xcor); SVR best for 1–15 min TSI+NWP; multimodel NWP
   improvements.
1. Future direction: physics-informed deep learning, multimodal fusion
   (satellite + NWP + ground), adaptive real-time forecasting for variable
   atmospheres.
────────────────────────────────────────

## 4. Methodology

- Type: Narrative systematic review (not PRISMA-quantified paper count);
  integrates physical, satellite, and AI literature.
- Scope: GHI/DNI forecasting for PV and CSP; ASI (All-Sky Imager), geostationary
  satellite, reanalysis/NWP inputs.
- Structure: Section 3 atmospheric factors → Section 4 prediction models
  (physical §4.1, statistical §4.2, AI §4.3, hybrid pros/cons §4.4) →
  conclusions.
- Validation approach: Compares published RMSE/MAE/rRMSE/R²/MAPE across cited
  primary studies; no new experiments.
────────────────────────────────────────

## 5. PCM Details (if applicable)

N/A — review focuses on solar irradiance forecasting for grid/PV/CSP, not PCM
materials.

Indirect link: Accurate GHI/DNI forecasts feed your Objective 1 PCM classifier
and Objective 2 DRL state (forecasted solar input, charge/discharge timing).

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

| Method class | Examples cited | Reported performance |
| --- | --- | --- |
| Classical ML | kNN, RF, GBM, MLP/ANN, SVR | SVR best for 1–15 min nowcasts with TSI+NWP [22] |
| Gradient boosting | XGBoost (Turkey Mediterranean) | R² = 0.9993, MAPE 0.0119 [106] |
| Deep learning | LSTM (ASI), CNN-LSTM hybrids | Promising for nonlinear cloud dynamics [10, 88] |
| Hybrid | NWP + ML post-processing, physics-informed DL | +21% vs base TSI model when NWP integrated [26]; reduces WRF cloud bias via EnKF CWP assimilation [24] |
| Persistence / LR | Baselines for ultra-short horizon | Outperformed by SVR in cloud-tracking pipeline |

Input features (typical): GHI, clear-sky index \(K_c\), solar zenith angle,
cloud fraction/type, wind, RH, aerosol AOD, satellite radiance, NWP fields.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: NWP (WRF, GFS, ECMWF, TAPM, MM5), geostationary satellites,
  Total Sky Imager (TSI), ground pyranometers, AOD from chemistry transport
  models (EURAD), SolarAnywhere benchmarks.
- Variables: GHI, DNI, DHI, AOD550, cloud water path (CWP), fractional sky
  cover, effective transfer ratio, humidity profiles, wind, ozone.
- Geographic examples: San Diego CA (satellite streamlines), California coast Sc
  clouds, Sicily wind+solar PCA [29], Turkey Mediterranean 8 cities [106],
  Tibetan Plateau [28], central Mediterranean dust events.
- Temporal resolution: TSI 5 min native; extended to 15 min with NWP [22];
  intra-hour to multi-day horizons classified explicitly.
- India relevance: Review does not focus on ISRO/ERA5/NASA POWER directly; your
  project should map cited hybrid NWP+ML pattern to ERA5 reanalysis + NASA POWER
  + ISRO Solar Calculator for Coimbatore, Kochi, Jaisalmer.
────────────────────────────────────────

## 8. Key Results & Numbers

- Global PV capacity growth: >900 GW added over ten years; +180 GW in 2021 alone
  [4].
- NOAA GHI Model 2 (San Diego): mean measured 493.49 W/m², modeled 516.78 W/m²,
  MBE −23.29 W/m² (−4.72%), MAE 61.72 W/m² (12.51%), RMSE 107.41 W/m² (21.77%
  rRMSE), Xcor 0.95 (Table 1).
- NOAA GHI Model 1: RMSE 124.53 W/m² (25.24% rRMSE), Xcor 0.934.
- SUNY GHI (outliers removed): RMSE 130.52 W/m² (31.38% rRMSE), Xcor 0.932.
- TSI + NWP integration: average +21% improvement vs base short-term model [26].
- SVR outperforms PM and LR for 1–15 min forecasts in Xu et al. cloud-tracking
  study [22] (Fig. 2 RMSE comparison).
- TAPM NWP: 45 km resolution yielded lowest RMSE vs finer scales [39].
- WRF terrain correction: horizontal surface RMSE reduced ~20% (~25 W/m²)
  winter/autumn; tilted surface best at 9 arcsec with RMSE 45% (57 W/m²) [31].
- Aerosol optical depth > 0.1: model errors up to 100 W/m² [30].
- XGBoost (8 Turkish cities): R² = 0.9993, MAPE 0.0119 — best among statistical
  vs ML comparison [106].
- Erbs + Liu-Jordan hybrid: most accurate among 12 transfer models across
  sunny/cloudy/overcast/rainy days [105].
────────────────────────────────────────

## 9. Baseline Comparison

| Approach | Horizon | vs Alternative | Outcome |
| --- | --- | --- | --- |
| SVR + TSI + NWP | 1–15 min | Persistence, linear regression | SVR lowest RMSE [22] |
| NWP-augmented TSI | Short-term | Base TSI only | +21% accuracy [26] |
| ML (XGBoost) | Regional daily/hourly | Statistical models | R² 0.9993 vs weaker statistical fits [106] |
| NOAA GHI satellite model | Day-ahead | SUNY GHI | RMSE 107 vs 213 W/m² (outlier case) |
| WRF + EnKF CWP assimilation | Short-term | Raw WRF | Improved mid-latitude GHI [24] |
| Hybrid physics + AI | Multi-horizon | Pure NWP or pure ML | Recommended compromise §4.4 |
| Erbs + Liu-Jordan | All-weather | 11 other transfer models | Best RMSE/MAE across 4 weather classes [105] |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — review paper. Cited field setups include ground pyranometers, Total Sky
Imagers, geostationary satellite receivers, and NWP assimilation systems — no
unified experimental rig.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Cloud cover and aerosols remain dominant error sources for NWP at short
  horizons.
- Physical models struggle with stratocumulus (Sc) thickness/presence in coastal
  zones [20].
- ML models need representative training data; poor cross-climate generalization
  risk.
- Integration of real-time adaptive data streams still immature.
- Need higher-resolution hybrid models and better multimodal fusion (satellite +
  NWP + ground).
- Complex terrain and dust events still under-modeled in many operational
  chains.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Indirect — forecasts enable proactive
  control; review supports forecast-driven MPC/DRL policies.
- RG2 (No integrated PCM–AI–hardware prototype): Indirect — defines irradiance
  input pipeline (pyranometer + forecast API) for embedded controller.
- RG3 (Poor alignment with household demand patterns): Partial — multi-horizon
  taxonomy helps align morning/evening demand with day-ahead and intra-hour GHI
  forecasts.
- RG4 (Limited real-world experimental validation): Relevant — cites ground
  validation benchmarks (RMSE in W/m²) you can replicate with pyranometer vs
  forecast at Indian sites.
- RG5 (No predictive optimization under climatic uncertainty): Highly relevant —
  core paper for Objective 1 & 2; justifies ERA5/NASA POWER/ISRO forecast
  features and hybrid NWP+ML approach for climate-adaptive PCM+DRL under
  cloud/aerosol uncertainty.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Error metrics (review nomenclature):

\[

\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}, \quad

\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|

\]

\[

\mathrm{rRMSE} = \frac{\mathrm{RMSE}}{\bar{y}}\times 100\%, \quad R^2 = 1 -
\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}

\]

Clear-sky index (used in cloud-ML studies):

\[

K_c = \frac{GHI_{measured}}{GHI_{clear\,sky}}

\]

Forecast horizon classes (adopt in evaluation protocol):

- Very-short: \(t \leq 60\) min (nowcasting for valve/pump control)
- Short: hours (same-day PCM charge planning)
- Medium: 1–3 days (PCM selection / pre-charge strategy)
────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. D.S. Kumar et al., solar irradiance resource and forecasting review, IET
   Renew. Power Gener., 2020 — complementary forecasting survey.
1. J. Xu et al., TSI + NWP multi-layer cloud tracking, 2015 — ultra-short
   horizon benchmark.
1. L. Nonnenmacher & C.F. Coimbra, streamline satellite forecasting, Sol.
   Energy, 2014 — satellite optical-flow GHI method.
1. A. Mellit, ML/DL for PV output forecasting overview, 2021 — links irradiance
   forecast to power prediction.
1. Mansouri et al., multimodal renewable forecasting survey, 2025 — aligns with
   your multimodal climate+sensor fusion agenda.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I (Introduction): Cite PV capacity growth and irradiance uncertainty
  as barrier to optimal PCM-SWH dispatch.
- Section II (Literature Review): Horizon taxonomy table; hybrid NWP+ML as
  state-of-art for RG5.
- Section III (Methodology): Adopt RMSE/MAE/rRMSE metrics for XGBoost irradiance
  model; use \(K_c\) and GHI forecast horizons as DRL state inputs.
- Section IV (Dataset & Setup): Position ERA5/NASA POWER/ISRO as India-specific
  analogues to WRF/satellite pipelines reviewed; target beating ~22% rRMSE
  day-ahead satellite benchmarks at your sites.
- Section V (Results): Report forecast accuracy vs cited baselines (e.g.,
  XGBoost R² 0.9993 regional study as aspirational upper bound with caveat on
  climate transfer).
────────────────────────────────────────

# 27. Hamzat2025PCM_SolarEnergyStorage_summary.md

Source path: /mnt/data/Hamzat2025PCM_SolarEnergyStorage_summary.md

# Phase Change Materials in Solar Energy Storage: Recent Progress, Environmental Impact, Challenges, and Perspectives

Authors: Abdulhammed K. Hamzat, Adewale Hammed Pasanaje, Mayowa I. Omisanya,
Ahmet Z. Sahin, Adesewa O. Maselugbo, Ibrahim A. Adediran, Lateef Owolabi
Mudashiru, Eylem Asmatulu, Oluremilekun Ropo Oyetunji, Ramazan Asmatulu

Year: 2025

Journal/Conference: Journal of Energy Storage, Vol. 114, Article 115762

DOI: https://doi.org/10.1016/j.est.2025.115762

IEEE Citation: A. K. Hamzat et al., "Phase change materials in solar energy
storage: Recent progress, environmental impact, challenges, and perspectives,"
J. Energy Storage, vol. 114, p. 115762, 2025, doi: 10.1016/j.est.2025.115762.

────────────────────────────────────────

## 1. One-Line Summary

This review synthesizes recent PCM-based solar thermal storage research and
shows that heat-transfer enhancement (especially nano-dispersion and hybrid
design/ML optimization) can raise performance substantially, with reported gains
up to 73%, while also detailing economic, environmental, and deployment
constraints.

────────────────────────────────────────

## 2. Problem Being Solved

- Conventional TES often suffers from low PCM thermal conductivity, slow
  charging/discharging, leakage/supercooling, and inconsistent material
  performance reporting across studies.
- Solar-integrated PCM systems need better technical optimization across
  melting/solidification, exergy, and cost, especially under variable weather
  and operating conditions.
- Environmental and economic claims are fragmented; lifecycle, emissions, and
  payback evidence is not consistently standardized across PCM technologies.
- AI/ML methods are promising but still early-stage for PCM-TES design/control,
  with limited multi-objective and real-world robust implementations.
────────────────────────────────────────

## 3. Key Contributions

1. Broad review of PCM enhancement pathways for solar TES: fins/extended
   surfaces, heat pipes, cascaded PCMs, encapsulation, porous media, and
   nanoparticle-doped PCMs.
1. Quantitative synthesis of nano-PCM effects with many reported
   conductivity/charging/discharging improvements and comparison across
   materials and concentrations.
1. Dedicated review of AI/ML for PCM-TES (ANN, SVM, GPR, ensemble learning,
   PINN, DRL), including concrete metrics (R², MSE, MAE, MAPE) from published
   studies.
1. Integrated techno-economic and environmental discussion (LCOE/LCOS/payback,
   LCA, CO2 mitigation, recyclability, sustainability trade-offs).
1. Challenges/future directions section covering test standardization, data
   reliability, ML limitations, and sustainability-focused material development.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

N/A — this is a review article (45 pages), not a single experimental rig.

It compiles results across solar collectors, SWH, PV/T-PCM, heat pump coupling,
greenhouse heating, building envelopes, and industrial heat recovery systems,
including both numerical and experimental literature.

### 4b. Mathematical Models & Equations

N/A — the paper is primarily a narrative/quantitative review and does not
present one new, unified governing model with a consistent equation set of its
own.

It reports metrics/equations as used in cited studies (e.g., MAPE, MAE, RMSE,
R², LCOE/LCOS, exergy and energy efficiencies).

### 4c. Algorithm / Control Method Steps

Review-level ML/control pipeline extracted from surveyed studies:

1. Build datasets from experiments/simulations (thermal conductivity, phase
   fraction, outlet temperature, exergy, load, weather, geometry variables).
1. Train model families: ANN/FFNN/LSTM, SVM, KNN, CART, MARS, GPR, ensemble
   frameworks, PINN, and DRL controllers.
1. Optimize hyperparameters (examples include Sobol sampling + ANN tuning,
   Bayesian tuning, and ensemble stacking).
1. Validate against measured/CFD data using R², MAE, MSE, RMSE, and MAPE.
1. Deploy predictions/control for PCM charging/discharging, outlet-temperature
   tracking, and cost/exergy optimization.
### 4d. Data Sources & Dataset Details

- Secondary data from published numerical and experimental literature on PCM-TES
  and solar systems.
- Includes studies on solar water heating, PV/T-PCM, greenhouse heating,
  building HVAC, industrial waste heat recovery, and thermal batteries.
- ML studies include datasets like:
- ~911 points from 25 studies (for thermal conductivity prediction in one cited
  ANN/CART/MARS study).
- Time-series operational datasets for TES charging/discharging and
  outlet-temperature forecasting in other cited works.
- Geographic coverage is multi-country (reviewed literature includes Central
  Europe, China, Iran, UK, etc.).
### 4e. Validation Method

- Review compiles validation metrics from surveyed studies, including:
- R² = 0.9999 (ANN for hybrid solar TES prediction in one cited study).
- R² = 0.97951 (group method neural model for thermal efficiency prediction in
  one cited collector-PCM study).
- Ensemble LHTES model: MAPE improvement up to 7.82% (charging) and 16.43%
  (initial discharging).
- Industrial PCM heat recovery model: max relative error 5.47%.
- PINN-based DCEE-linked TES predictions: deviation within ±7.8%.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- Materials tested: Review covers organic/inorganic/eutectic/composite PCMs;
  examples include paraffin wax (PW), RT-series paraffins (RT35, RT44HC, RT50,
  RT54HC), OM65, hydrated salts, sodium acetate trihydrate, erythritol
  composites, myristic acid systems, etc.
- Melting temperature range: Study coverage spans low/mid/high application
  bands; reported examples include 26.61–27.12 °C (one methyl palmitate
  composite case), 29 °C transition in building control context, and
  high-temperature applications up to 885 °C in waste-heat TES context.
- Latent heat: Reported examples include 96.1–96.7 J/g (one CNF composite case)
  and strong dependence on nanoparticle loading/composition.
- Thermal conductivity: Reported improvements range widely; examples include
  baseline-to-enhanced increases of 53.58%, 59.5%, 71.5%, 72.2%, 86.36%, 87.39%,
  109.2%, 112.5%, and up to 165.56% in reviewed cases.
- Specific heat (solid/liquid): Not one fixed value (review spans many PCMs);
  one cited nano-salt case reported 19–24% specific heat increase.
- Density: Material-specific and varies by system; no single universal density
  reported for the review.
- Performance metrics reported: Melting time reduction, charging/discharging
  rate, thermal/exergy efficiency, COP, LCOE/LCOS/payback, compressor runtime,
  fuel savings, and CO2 reduction.
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm: ANN, FFNN, LSTM-BP, SVM, KNN, CART, MARS, GPR, Huber regressor,
  SGD, ensemble learning, PINN, DRL.
- Input features / state space: Depending on study: PCM type/fraction/thickness,
  nanoparticle type/concentration, flow rate, geometry (fin/porosity/tube), heat
  flux, weather/temperature, operating schedule.
- Output / action space: Thermal conductivity, melting/solidification behavior,
  outlet temperature/enthalpy, exergy performance, charging/discharging
  dynamics, operational control actions (in DRL studies).
- Model architecture: Includes feedforward ANN, multilayer perceptron, NARX,
  ensemble stacking frameworks, PINN hybrids, and DRL policy models.
- Hyperparameters: Reported examples include Sobol-sampled ANN tuning and
  Bayesian-tuned deep models (specific full sets vary by cited paper).
- Training data size: Example: >911 samples from 25 studies in one conductivity
  prediction study.
- Hardware used for training: Not standardized in the review (depends on cited
  papers).
- Performance metrics: R², RMSE, MAE, MSE, MAPE, confidence/error levels, and
  control-cost reductions.
If not applicable: N/A — reason

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: Review-level synthesis of many papers; includes studies using
  local meteorological conditions and seasonal contexts (e.g., Central Europe,
  arid cities, greenhouse operations).
- Variables used: Solar radiation, ambient temperature, seasonal temperatures,
  thermal load, system temperatures, and operational demand profiles (varies by
  cited study).
- Geographic scope: Multi-region/global literature (examples include China,
  Iran, UK, Central Europe, Canada).
- Temporal resolution: Varies by source paper (dynamic/transient and seasonal
  analyses are included).
- Time period covered: Up to 2024/2025 literature in this review.
- Clear-sky index / derived metrics: Not consistently reported as a unified
  metric in the review.
────────────────────────────────────────

## 8. Key Results & Numbers

- The review reports system performance improvements up to 73% from PCM
  enhancement strategies (abstract claim).
- Nano-PCM development in reviewed studies shows 25.6% charging and 23.9%
  discharging improvement versus conventional PCM systems (abstract claim).
- One ANN-based hybrid solar TES study achieved R² = 0.9999 after hyperparameter
  tuning.
- Another neural model for PCM-based collector performance reported R² =
  0.97951.
- A reviewed conductivity-prediction study used >911 data points from 25 studies
  and achieved top ANN R² = 0.96 (vs MARS/CART at 0.93).
- Ensemble LHTES ML modeling reported MAPE improvement up to 7.82% (charging)
  and 16.43% (initial discharging), with error spread reduction up to 25.6%.
- A DRL-controlled seasonal sorption storage case reported operational cost
  reductions of 28% (60 winter days) and 13% (120 winter days) over rule-based
  control.
- A PINN-based coupled DCEE/TES study kept prediction discrepancy within ±7.8%.
- Greenhouse HRS + PCM-HRS integration improved mean energy efficiency by 33%
  and 40%, and exergy efficiency by 127% and 263%, respectively.
- Same greenhouse case reduced fuel consumption by 19% (plain HRS) and 48%
  (PCM-HRS), with payback around 3 months and 4 months.
- In an arid-climate building-envelope study, PCM integration reduced HVAC
  energy by 55.47%, 53.89%, 58.86%, and 53.57% in one scenario
  (Dubai/Jeddah/Kuwait/Lahore).
- Another scenario from the same study showed smaller reductions: 2.6%, 2.03%,
  1.99%, 5.6%.
- Reported CO2 emission reductions in that study reached 56.27%, 44.81%, 45.27%,
  and 58.5%.
- A heat-pump TES study cited 75% PCM integration reducing required tank volume
  by about 3× versus water-only storage.
- SAHP + TES optimization found minimum tank volume 1300 L, optimal PCM filling
  ratio 85%, and compressor energy reduction 27.2%.
- A 1 MW PVT-PPCM analysis reported annual output 1920 MWh and about 30
  tons/year CO2 reduction.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Conventional PCM systems, rule-based control, plain HRS
  (without PCM), water-only storage tanks, and non-PCM HVAC/solar benchmarks in
  cited studies.
- Proposed method: Review supports enhanced PCM strategies (nano-PCMs,
  cascaded/encapsulated/composite PCMs, AI-optimized operation).
- Improvement margin: Reported margins include up to 73% performance gain,
  25.6%/23.9% charge/discharge improvement, 28% DRL cost reduction, and 27.2%
  compressor-energy reduction in SAHP+PCM case.
- Conditions of comparison: Results are from heterogeneous studies with
  different climates, PCMs, geometries, and objectives; not one single uniform
  benchmark dataset.
If no baseline comparison: N/A — [paper is a review / purely experimental /
etc.]

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

- Physical components: Across reviewed studies: solar collectors, SWH tanks,
  PV/T modules, heat pumps, greenhouse heat exchangers, industrial heat-recovery
  units, fins/foam/encapsulation modules.
- Sensor specs: Not unified in this review; instrumentation depends on each
  cited study.
- Embedded/compute platform: AI models implemented in cited works; no single
  hardware platform reported by this review itself.
- Test environment: Includes simulation, lab-scale experiments,
  pilot/demonstration systems, building and industrial contexts.
- Test duration: Varies from transient charging tests to seasonal and lifecycle
  assessments.
If simulation only: N/A — this paper is purely simulation/CFD-based.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Authors explicitly note possible veracity issues in some surveyed findings and
  potential literature coverage omissions.
- They highlight inconsistency in reported nano-PCM thermophysical data and call
  for standardized testing methods and a reliable database.
- ML deployment is described as still in an embryonic stage for PCM-TES with
  limited multi-objective/system-level studies.
- Multi-variable optimization under uncertainty and algorithm selection remain
  unresolved due to computational complexity and data limitations.
- The paper calls for stronger encapsulation methods, optimized manufacturing,
  and full lifecycle sustainability studies for feasibility.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Relevant. The review cites DRL/ANN/PINN
  control and prediction studies with quantified gains (e.g., 28% cost
  reduction), directly supporting adaptive control direction, though mostly
  outside domestic PCM-SWH prototypes.
- RG2 (No integrated PCM–AI–hardware prototype): Partially Relevant. It
  documents AI + PCM integration trends and pilot studies, but most are not
  end-to-end embedded domestic systems with your exact hardware stack
  (RPi/ESP32/DS18B20/valve).
- RG3 (Poor alignment with household demand patterns): Partially Relevant. Some
  building/HVAC and TES scheduling studies address load dynamics; explicit
  household DHW draw-profile coupling is still limited.
- RG4 (Limited real-world experimental validation): Relevant. The review
  includes both simulation and experimental/pilot evidence, but repeatedly notes
  scarcity of standardized, long-term real-world validation across conditions.
- RG5 (No predictive optimization under climatic uncertainty): Relevant.
  Multiple ML studies incorporate seasonal/weather-sensitive optimization and
  forecasting, yet robust multi-climate predictive optimization remains a stated
  challenge.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |  |  |
| --- | --- | --- | --- | --- |
| $Q = mL + mC_p\\Delta T$ | Total latent+sensible stored heat | Grey-box PCM tank energy model |  |  |
| $\\eta_{th}=\\frac{Q_{useful}}{A\\,G}$ | Collector/TES thermal efficiency | Compare control policies under same irradiance |  |  |
| $\\eta_{ex}=\\frac{Ex_{out}}{Ex_{in}}$ | Exergy efficiency | Second-law KPI for PCM charging quality |  |  |
| $\\mathrm{MAPE}=\\frac{100}{n}\\sum\\left | \\frac{y-\\hat y}{y}\\right | $ | ML forecast/control model error | Evaluate XGBoost/ANN thermal predictor |
| $\\mathrm{RMSE}=\\sqrt{\\frac{1}{n}\\sum(y-\\hat y)^2}$ | Prediction error magnitude | Model selection and validation metric |  |  |
| $\\mathrm{LCOE}=\\frac{\\sum_t \\frac{C_t}{(1+r)^t}}{\\sum_t \\frac{E_t}{(1+r)^t}}$ | Lifecycle electricity cost | Techno-economic comparison of PCM control strategies |  |  |

If no reusable equations: N/A — [reason]

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. D. E. Douvi et al., "Phase change materials in solar domestic hot water
   systems: a review," Int J Thermofluids, 2021 — Relevant because it is
   directly about PCM integration in solar DHW/SWH systems.
1. B. Kanimozhi et al., "Thermal energy storage system operating with phase
   change materials for solar water heating applications: DOE modelling," Appl.
   Therm. Eng., 2017 — Relevant because it targets SWH + PCM modeling with
   quantified control/design outputs.
1. A. Crespo et al., "Optimal control of a solar-driven seasonal sorption
   storage system through deep reinforcement learning," Appl. Therm. Eng., 2024
   — Relevant because it provides DRL-based thermal storage control evidence.
1. L. Yang et al., "Thermophysical properties and applications of nano-enhanced
   PCMs: an update review," Energy Convers. Manag., 2020 — Relevant because it
   supports nano-PCM property enhancement decisions for better
   charging/discharging.
1. R. Aridi and A. Yehya, "Review on the sustainability of phase-change
   materials used in buildings," Energy Convers. Manag. X, 2022 — Relevant
   because it supports environmental and lifecycle sections for PCM selection.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
| --- | --- | --- |
| I. Introduction | Motivation for advanced PCM-TES | "Recent review evidence reports PCM enhancement routes yielding performance gains up to 73% in solar TES contexts." |
| II. Literature Review | AI in PCM-TES summary entry | Method: ANN/ensemble/DRL/PINN; Key insight: DRL control showed up to 28% operational-cost reduction in seasonal TES case studies. |
| III. Methodology | Modeling + control metrics | Use RMSE/MAPE/R² and exergy-based KPIs, mirroring reported PCM-TES ML validation framework. |
| IV. Dataset & Setup | Climate/load sensitivity argument | "PCM selection and control are strongly dependent on ambient temperature, solar radiation, and location-specific conditions." |
| V. Results | Baseline-comparison anchor | Report your controller vs rule-based with same style as reviewed literature (e.g., cost reduction %, charging/discharging improvement %, exergy gain %). |

────────────────────────────────────────

# 28. Kou2025BIHP_PCM_Building_Optimization_summary.md

Source path: /mnt/data/Kou2025BIHP_PCM_Building_Optimization_summary.md

# A Novel Solar Heating Building Integrated Heat Pipes and PCMs: Optimizing Thermophysical Properties and Reducing Energy Consumption

Authors: Fangcheng Kou, Nian Zhu, Xin Wang, Yu Zou, Jinhan Mo

Year: 2025

Journal/Conference: Building and Environment, Vol. 285, Article 113674

DOI/Link: https://doi.org/10.1016/j.buildenv.2025.113674

IEEE Citation: F. Kou et al., "A novel solar heating building integrated heat
pipes and PCMs: Optimizing thermophysical properties and reducing energy
consumption," Build. Environ., vol. 285, p. 113674, 2025, doi:
10.1016/j.buildenv.2025.113674.

────────────────────────────────────────

## 1. One-Line Summary

This paper proposes BIHP-PCM (flat gravity heat-pipe + PCM interior wall),
optimizes volumetric enthalpy \(\rho H\), phase-change temperature \(T_m\), and
conductivity \(\lambda\) via PSO at 61 Chinese cold-region cities, and shows
ESR/IDTD of 30–100% linearly tracking RRTD\(_{HS}\)
(solar-radiation-to-temperature-difference ratio), with Tianjin HVAC case
reaching 94.4% energy savings (\(Q\): 1220 → 68 MJ).

────────────────────────────────────────

## 2. Problem Being Solved

- Solar heating faces intermittent radiation; passive PCM walls rely on low
  natural-convection coefficients and deliver limited comfort gains (literature
  cites only 1–3% savings for passive PCM sunspaces).
- Active PCM systems (pumps/fans) improve performance (12.7–40% load cuts) but
  add complexity, parasitic energy, and maintenance.
- Prior BIHP (heat-pipe only) transfers solar heat efficiently by thermal diode
  conduction but lacks sufficient latent storage, causing daytime overheating
  and nighttime drops.
- PCM thermophysical properties (\(\rho H\), \(T_m\), \(\lambda\)) are
  climate-dependent; no prior work optimized PCM inside BIHP across many cities
  or linked performance to a climate index for zoning.
────────────────────────────────────────

## 3. Key Contributions

1. BIHP-PCM architecture: L-shaped flat gravity HP (evaporator on south exterior
   wall, condenser embedded in east/west PCM interior walls) — daytime HP \(k_e
   \approx 2\times10^4\) W/(m·K) charges PCM by conduction; nighttime HP blocks
   reverse loss (~1/170 forward vs reverse thermal resistance).
1. Equivalent-specific-heat PCM model (triangular \(c_p(T)\) over \(\Delta T=2\)
   °C) coupled with HP and indoor air balance (Eqs. 1–11).
1. PSO inverse optimization of \(\rho H\), \(T_m\), \(\lambda\) for with-HVAC
   (minimize \(Q\), maximize ESR) and without-HVAC (minimize IDDC, maximize
   IDTD) — 20 particles, 30 iterations.
1. 61-city severe-cold/cold China study using DeST climate data; optimal PCM
   universally favors max \(\rho H=420\) MJ/m³ and high \(\lambda\) (5–9 W/(m·K)
   without HVAC; 3.5–9 with HVAC).
1. Climate correlation: ESR\(_{OPT}=0.148·RRTD\(_{HS}\) and
   IDTD\(_{OPT}=0.150·RRTD\(_{HS}\) with \(R^2>0.98\); three application zones
   (zero-carbon / good / suitable).
1. Experimental validation (Beijing twin test houses): simulated vs measured
   room temperature \(R^2>0.98\), mean error 0.3 °C (BIHP-PCM) and 0.2 °C
   (reference).
────────────────────────────────────────

## 4. Methodology

### 4a. System

- South-facing room 4×3×3 m, window–wall ratio 0.3, \(T_L=18\) °C comfort lower
  bound.
- East/west interior walls: 12 cm brick + 6 cm PCM layer; HP condenser between
  brick and PCM.
- Non-optimized PCM: KF·4H₂O (potassium fluoride tetrahydrate); reference
  building = same geometry without HP/PCM.
### 4b. Heat-pipe & wall models

- Forward HP heat: \(Q_{HP,fw} = A_{sec} k_e (T_{eva}-T_{con})/l_{eff}\) (1)
  with \(k_e=2\times10^4\) W/(m·K).
- Reverse nighttime conduction (3) — ignored in energy balance (two orders
  smaller).
- PCM latent heat: \(H = \frac{1}{2}\Delta T \Delta c_p\) (5); melting fraction
  (7).
- Wall conduction (8); outer/inner boundaries (9–10) with \(h_{out}=23.0\),
  \(h_{in}=8.7\) W/(m²·K).
- Indoor air energy balance (11) including ACH (0.5 below 26 °C operative temp,
  5.0 above), window gap, and \(q_{HVAC}\).
### 4c. Optimization indices

- Without HVAC: IDDC (12) — integrated cold discomfort; IDTD (15) = % reduction
  vs reference.
- With HVAC: seasonal heating energy \(Q\) (16); ESR (17) = % savings vs
  reference.
- Climate index: RRTD\(_{HS} = Q_{sol,ave}/(T_{set}-T_{out,ave})\) (18) over
  local heating season.
### 4d. PSO procedure

Six steps: initialize random \((\rho H, T_m, \lambda)\) in box 0–420 MJ/m³,
10–30 °C, 0–10 W/(m·K) → simulate objective → select global best →
velocity/position update → 30 iterations → output optimum.

### 4e. Validation experiment

- Twin 2.4×2.4×2.4 m houses, Beijing, May 11–21; CaCl₂·6H₂O PCM 153 kg (0.09 m³)
  in 0.20×0.15×0.02 m boxes; acetone-filled aluminum HP (evap 1.68 m², cond 1.36
  m²); ACH 0.1 h⁻¹.
- MATLAB FDM: spatial step 5 mm, time step 60 s.
────────────────────────────────────────

## 5. PCM Details (if applicable)

| Parameter | Study range / optimum (examples) |
| --- | --- |
| Volumetric enthalpy \(\rho H\) | Search 0–420 MJ/m³; optimum always 420 (upper bound) |
| Phase-change temperature \(T_m\) | Search 10–30 °C; Tianjin 19.0 °C (no HVAC), 19.5 °C (with HVAC); with HVAC cluster 19.0–20.5 °C (~\(T_L+2\) °C) |
| Thermal conductivity \(\lambda\) | Search 0–10 W/(m·K); Tianjin optimum 6.3 (no HVAC), 6.8 (with HVAC) |
| Phase-change range \(\Delta T\) | Fixed 2 °C |
| PCM layer thickness | 6 cm on interior walls |
| Validation PCM | CaCl₂·6H₂O, \(T_m\approx27\) °C, \(\rho H=289\) MJ/m³ (Table 2 literature) |
| Literature PCMs cited | RT27 (\(\rho H=145\), \(\lambda=0.2\)), SP25A8, L30 (\(\rho H=420\)), TM/EG (\(\lambda=9.72\)) |

Design rule: maximize storage capacity and use high \(\lambda\) but not always
maximum — overly high \(\lambda\) can prematurely discharge latent heat to the
room.

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A — no machine learning.

Control/optimization: Particle Swarm Optimization (PSO) — population 20,
iterations 30; objectives IDDC (passive) or seasonal heating energy \(Q\)
(HVAC).

HVAC rule: maintain \(T_{op}\geq18\) °C when equipped; optimized PCM reduces
HVAC runtime (131 h vs 1475 h reference in Tianjin).

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Source: DeST building-simulation weather database (not ERA5/NASA POWER).
- Geography: 61 cities in China severe-cold and cold zones (Fig. 5); includes
  Tianjin, Harbin, Urumqi, Xi’an, Kashgar, etc.
- Variables: heating-season southern solar radiation \(Q_{sol,ave}\), outdoor
  air temperature \(T_{out,ave}\), heating degree metrics; derived RRTD\(_{HS}\)
  (18).
- Temporal resolution: heating-season integrals (hourly simulation, 60 s
  experimental logging).
- Project mapping: analogous index for India — ratio of GHI (NASA POWER / ERA5)
  to \((T_{set}-T_{amb})\) for Coimbatore, Jaisalmer, Kochi could pre-score
  PCM-SWH potential without full TRNSYS.
────────────────────────────────────────

## 8. Key Results & Numbers

- China urban/rural heating share: 27.0% / 41.5% of building operation energy.
- Passive PCM sunspace savings cited: 1–3%; ventilated Trombe wall 12.7%;
  heat-pump underfloor PCM 40%.
- HP daytime thermal resistance 1/170 of nighttime; forward \(k_e\) up to 10⁴
  W/(m·K) cited in intro, 2×10⁴ in model.
- Model validation \(R^2>0.98\); daily mean temp error 0.3 °C (BIHP-PCM), 0.2 °C
  (reference).
- Tianjin without HVAC: optimal \((\rho H,T_m,\lambda)=(420, 19.0, 6.3)\);
  IDDC=192 K·h; IDTD=98.9% vs IDDC=2206 K·h non-optimized (87.4% IDTD).
- Tianjin with HVAC: \(Q_{RB}=1220\) MJ; non-optimized \(Q=341\) MJ (ESR=72.1%);
  optimized \(Q=68\) MJ (ESR=94.4%).
- HVAC on-time \(\tau_{on}\): 1475 h (ref) → 735 h (initial BIHP-PCM) → 131 h
  (optimized).
- Average operating load \(q'_a\): 19.2 / 10.7 / 12.1 W/m² (ref / initial /
  optimized).
- 22/61 cities achieve zero-carbon heating without HVAC.
- Linear fits \(R^2>0.98\): ESR\(_{OPT}=0.148·RRTD\(_{HS}\); example RRTD=5 →
  ESR≈74%.
- ESR/IDTD range across cities: 30–100% (abstract/conclusion).
- Zoning (Table 3): zero-carbon RRTD≥8.5 (ESR/IDTD 100%); good 6.0–8.5 (ESR
  60–100%, IDTD 70–100%); suitable <6.0 (ESR <60%, IDTD <70%).
- Example city data: Tianjin RRTD\(_{HS}=6.10\), ESR\(_{OPT}=94.4%\),
  IDTD\(_{OPT}=98.9%\); Dalian ESR\(_{OPT}=100%\); Urumqi ESR\(_{OPT}=34.0%\).
────────────────────────────────────────

## 9. Baseline Comparison

| Configuration | Tianjin seasonal heating \(Q\) | ESR vs reference | IDTD (no HVAC) |
| --- | --- | --- | --- |
| Reference (no HP/PCM) | 1220 MJ | 0% | — |
| BIHP-PCM initial (KF·4H₂O) | 341 MJ | 72.1% | 87.4% |
| BIHP-PCM PSO-optimized | 68 MJ | 94.4% | 98.9% |
| Passive PCM literature | — | ~1–3% | minimal |
| BIHP without PCM (prior work) | — | — | large but comfort gaps remain |

Optimized PCM adds +22.3 percentage points ESR over non-optimized BIHP-PCM at
Tianjin.

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

- Validation houses: 2.4 m cube, 150 mm rock wool envelope, south glazing
  2.10×1.50 m, no windows on other walls.
- Heat pipes: aluminum, acetone working fluid, 10% fill; black-painted
  evaporator.
- PCM: CaCl₂·6H₂O, encapsulated boxes.
- Sensors: WZY-1 automatic thermometer ±0.2 °C; TES-132 solar energy meter ±10
  W/m²; 60 s logging.
- Temperature points: indoor at 0.5, 1.0, 1.5 m height (averaged); HP evap/cond;
  wall inner/outer surfaces.
- Platform: MATLAB simulation; no RPi/Arduino in loop — aligns with your
  bench-scale validation gap (RG4) but shows sensor specs usable for field
  tests.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Optimal \(\rho H\) hits upper bound 420 MJ/m³ — higher enthalpy PCMs could
  improve further.
- Validation experiment not during heating season and envelope differs from
  typical BIHP-PCM; authors argue heat-transfer physics still valid.
- PCM position, thickness, and \(\Delta T\) fixed — only three properties
  optimized.
- PSO uses modest swarm (20×30) — global optimum not guaranteed.
- Results specific to Chinese severe-cold/cold climates and south-facing
  geometry.
- Without HVAC, comfort not guaranteed all season; HVAC case still needs backup
  heat in weak-solar weeks.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (real-time adaptive control): Not relevant — seasonal PSO design, no
  online control or DRL; but HVAC on-time reduction (131 vs 1475 h) motivates
  demand-aware charging strategies.
- RG2 (integrated PCM–AI–hardware): Partially relevant — full HP+PCM hardware
  validated experimentally, but no AI and not a compact SWH tank; heat-pipe
  diode concept transferable to collector-to-storage coupling.
- RG3 (household demand): Relevant — optimizes \(T_m \approx T_{demand}-2\) °C
  and quantifies latent discharge when room nears 18 °C; maps to evening
  hot-water draw aligning PCM melt plateau with comfort/demand.
- RG4 (field validation): Relevant benchmark — real-house experiment with ±0.2
  °C sensors and pyranometer; \(R^2>0.98\) model agreement supports your
  grey-box + bench validation target.
- RG5 (climate uncertainty): Highly relevant — RRTD\(_{HS}\) linear predictor
  enables climate-adaptive PCM selection across cities; direct parallel to
  classifying Coimbatore / Jaisalmer / Kochi before deployment using NASA POWER
  or ERA5.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

HP forward heat flux:

\[

Q_{HP,fw} = \frac{A_{sec}\, k_e\, (T_{eva}-T_{con})}{l_{eff}}

\]

Energy saving ratio (HVAC case):

\[

\mathrm{ESR} = \frac{Q_{RB}-Q_{HP}}{Q_{RB}}\times 100\%

\]

Climate resource index:

\[

\mathrm{RRTD}_{HS} = \frac{Q_{sol,ave}}{T_{set}-T_{out,ave}}

\]

Empirical potential (optimized, with HVAC):

\[

\mathrm{ESR}_{PCM,OPT} = 0.148 \cdot \mathrm{RRTD}_{HS}

\]

PCM latent heat (equivalent specific heat triangle):

\[

H = \tfrac{1}{2}\Delta T\,\Delta c_p

\]

Reward/penalty ideas for DRL: minimize \(Q_{HVAC}\) or IDDC; bonus when
\(T_{op}\) stays above \(T_L\) without auxiliary heat; penalize premature melt
(\(T_m\) too low) or early discharge (\(T_m\) too high).

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Gong et al., L-shaped flat gravity heat-pipe solar building, prior BIHP work
   — thermal diode foundation [14–16].
1. Kou et al., PSO optimization of BIHP conventional walls, Build. Environ.
   prior — optimization framework [36, 41].
1. Zeng et al., δ-function optimal envelope specific heat — theoretical PCM
   equivalence [30, 31].
1. Soares et al., PSO for PCM drywalls — metaheuristic PCM sizing [32].
1. Guo/Zhu passive PCM Trombe studies — low passive savings baseline [8–10].
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I: Cite 27%/41.5% heating energy share and intermittent solar
  challenge; contrast passive PCM 1–3% vs BIHP-PCM up to 94.4% ESR.
- Section II: Position BIHP-PCM as heat-pipe-enhanced PCM storage alternative to
  convection-limited SWH tanks; include in lit-review table with PSO-optimized
  \(T_m\), \(\rho H\), \(\lambda\).
- Section III: Adapt RRTD\(_{HS}\) for Indian cities to pre-select RT35–RT64HC /
  OM35–OM50 melt points before RL training.
- Section IV: Reference validation sensors (±0.2 °C, ±10 W/m², 60 s) for your
  DS18B20 + pyranometer logging protocol.
- Section V: Benchmark ESR 72–94% (Tianjin) or IDTD 98.9% as aspirational
  seasonal metrics if extending project from daily control to seasonal
  simulation.
────────────────────────────────────────

# 29. Liu2025AI_PCM_TES_Prediction_Optimization_summary.md

Source path: /mnt/data/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md

# The Contribution of Artificial Intelligence to Phase Change Materials in Thermal Energy Storage: From Prediction to Optimization

Authors: Shuli Liu, Junrui Han, Yongliang Shen, Sheher Yar Khan, Wenjie Ji,
Haibo Jin, Mahesh Kumar

Year: 2025

Journal/Conference: Renewable Energy, Vol. 238, Article 121973

DOI/Link: https://doi.org/10.1016/j.renene.2024.121973

IEEE Citation: S. Liu et al., "The contribution of artificial intelligence to
phase change materials in thermal energy storage: From prediction to
optimization," Renew. Energy, vol. 238, p. 121973, 2025, doi:
10.1016/j.renene.2024.121973.

────────────────────────────────────────

## 1. One-Line Summary

This comprehensive review maps AI applications across PCM-based latent heat
storage—from ANN/XGBoost/SVM property prediction and CALPHAD integration to
GA/PSO/DRL structural and operational optimization—reporting melting-point error
reductions up to 42–71%, NEPCM conductivity prediction R² ≈ 0.99, cascaded LHS
energy gains of 5–18%, and ANN–MPC operating-cost cuts of 9.1–14.6%, while
identifying gaps in real-time embedded control and standardized datasets.

────────────────────────────────────────

## 2. Problem Being Solved

- LHS with PCMs faces low conductivity, supercooling, leakage, and complex
  melting/solidification dynamics that make trial-and-error design slow and
  expensive.
- Traditional CFD and experiments alone cannot efficiently explore
  high-dimensional PCM composites, encapsulation layouts, and system operating
  strategies.
- AI methods are proliferating but lack a unified synthesis comparing prediction
  vs optimization algorithms across solar thermal, building, and industrial LHS
  domains.
- Operational control strategies (flow, inlet temperature, charge/discharge
  scheduling) can dominate performance yet are under-optimized relative to
  material selection.
────────────────────────────────────────

## 3. Key Contributions

1. Two-pillar framework: (A) AI for prediction — PCM/CPCM/NEPCM thermophysical
   properties, melting behavior, temperature fields; (B) AI for optimization —
   structure/layout (fins, foam, cascaded PCM) and operation/control.
1. Algorithm taxonomy: ANN, BP-ANN, ELM, SVM, XGBoost, RF, GBR, LSTM, CNN, GA,
   PSO, DE, CFD-coupled GA, DRL, GEP, MARS, CART.
1. Composite PCM prediction survey: molten-salt eutectics, nano-enhanced
   organics, microencapsulated cement/concrete PCMs, metal-foam composites.
1. Optimization tables (Tables 3–4): intelligent algorithms for fin geometry,
   foam porosity, cascaded CLHS, shell-and-tube layouts—with quantified
   energy/exergy gains.
1. Limitations & future directions: need physics-informed ML, embedded real-time
   control, cross-climate validation, standardized open datasets, and DRL for
   dynamic LHS operation.
────────────────────────────────────────

## 4. Methodology

- Type: Narrative comprehensive review (Renewable Energy, 29 pages, 120+
  references).
- Scope: AI in PCM-based LHS/TES across buildings, solar thermal, cold storage,
  batteries, waste heat—not exclusively solar water heating.
- Organization: §2 TES/LHS background → §3 AI prediction (properties, NEPCM,
  temperature/behavior) → §4 AI optimization (structure §4.1, operation/control
  §4.2) → challenges/future.
- Validation: Synthesizes cited primary studies' reported \(R^2\), RMSE, MAPE,
  energy % improvements; no new experiments in this paper.
────────────────────────────────────────

## 5. PCM Details (if applicable)

### Materials & properties predicted/optimized (selected from review)

| Category | Examples | Key properties AI-targeted |
| --- | --- | --- |
| Molten salt eutectics | KCl-NaF, NaNO₃-KNO₃-KCl | \(T_m\), latent heat, composition |
| Organic / paraffin | Octadecane, RT50/RT65/RT80 cascades | \(k_{eff}\), \(T_m\), melt fraction |
| NEPCM | CuO, Al₂O₃, TiO₂, SiO₂, Fe₂O₃ in paraffin | Effective conductivity 0.5–12 wt% |
| Carbon NEPCM | MWCNT, graphene, CNF, GNP | \(R^2\) up to 0.99 vs RSM 0.79 |
| Building CPCM | Microencapsulated paraffin in cement | Compressive strength, activation energy |
| Cascaded CLHS | RT50, RT65, RT80 commercial PCMs | Stage height, NTU, PCM mass |

Reported accuracy examples:

- KCl-NaF ANN: \(T_m = 648 \pm 2\) °C, \(L = 365 \pm 5\) kJ/kg [50]
- BP-PBO vs BP-GA: melting-point error −42% / −38%; latent heat error −71% /
  −68% [51]
- NEPCM octadecane + metal oxides: max errors 2.31% (liquid), 0.812% (solid)
  [69]
- 10–20 wt% MPCM in cement: activation energy −10% / −28% [cited
  microencapsulation study]
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

| Application | Algorithms | Inputs (examples) | Outputs | Reported metrics |
| --- | --- | --- | --- | --- |
| Eutectic salt design | ANN, BP-GA, BP-PBO | Electronegativity, ion radius, charge | \(T_m\), \(L\), composition | Error ↓ 42–71% [51] |
| NEPCM conductivity | MARS, CART, ANN, KNN, SVM, XGBoost | NP concentration, size, PCM phase, \(k_{pcm}\) | \(k_{eff}\) | ANN R² 0.99; liquid/solid errors <2.31% [69] |
| Temperature field / melt fraction | ANN, SVM, RF, LSTM-BP, CNN | Geometry, \(T_{in}\), flow, time | \(T(x,t)\), liquid fraction | SVM R² 0.99, RMSE 2.19–3.17 [cited] |
| Building PCM demand | MLP, LSTM, CNN | Weather, occupancy | Cooling load | Energy ↓ 4.7–25.2% [cited] |
| Operating control | ANN + εDE metaheuristic + MPC | TES tank states, tariffs | Charge/discharge rate | Cost ↓ 9.1–14.6% [32] |
| Structural optimization | GA, PSO, DE, CFD-GA | Fin length, foam %, PCM stage layout | Energy, exergy, entransy | Stored energy ↑ 5–18% (Table 4) |
| Dynamic control | DRL (cited) | LHS state | Valve/flow policy | Scores within 1.6–4.3% of GA Pareto [cited] |

Training notes: Datasets range from literature compilations (hundreds of salt
systems) to CFD-generated samples; many studies use 90/10 train-test splits;
hyperparameter tuning via GA on ANN topology common.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Direct solar datasets: Not primary focus; solar LHS studies cited include flat
  solar CLHS [173], medium-temperature spherical encapsulated PCM solar units
  [175], packed-bed solar plant CLHS [172].
- Climate variables in cited control studies: Ambient temperature, operational
  schedules, electricity tariffs—for building-coupled TES MPC [32].
- Geographic scope: Global literature (China BIT-led review); no India-specific
  cities.
- Your project mapping: Use Liu's 8-feature weather vector pattern (GHI, DNI,
  DHI, \(T_a\), wind, RH, hour, month) from related SWH studies together with
  this review's PCM prediction/optimization sections for grey-box + XGBoost +
  DRL design.
────────────────────────────────────────

## 8. Key Results & Numbers

- Global energy storage installed capacity 209.4 GW by end-2021 (+9.6% YoY);
  pumped hydro 86.2% [7].
- Thermal energy = 50% of terminal energy utilization [10].
- ANN–MPC building TES: operating cost reduction 9.1–14.6% [32].
- BP-PBO salt prediction: melting-point error −42% vs BP; latent heat error −71%
  vs BP-GA [51].
- NEPCM ANN: liquid-phase error 2.31%, solid 0.812% [69].
- ANN vs RSM for NEPCM \(k\): R² 0.99 vs 0.79 [67].
- SVM temperature prediction: R² 0.99, RMSE 2.19–3.17 (review synthesis).
- XGBoost/RF regression: highest \(R^2\), lowest RMSE among tree models for PCM
  performance [cited].
- Cascaded CLHS GA: +5.12% stored energy vs single PCM [172]; flat solar CLHS
  +6%, +18%, +11% vs RT50/RT65/RT80 single PCM [173].
- Shell-and-tube CFD-GA: charged energy +13.79%, exergy +14.85%, entransy
  +14.45% [174].
- Spherical encapsulated CLHS: charging energy +≥14% [175].
- Cascaded heat sink GA: thermal management time +12.4%; cooling time −31.9 min
  [176].
- PSO optimal PCM parameters example: \(C_{ps}=2.5\), \(C_{pl}=3.1\) kJ/kg·K,
  \(L=238\) kJ/kg, \(T_m=20.85\) °C [180].
- Cold storage GA: optimal water flow 0.095 kg/s, inlet ΔT −1.25 °C [181].
- ORC-LHS multi-objective GA: exergy efficiency 0.3351, cost rate 1.529 $/h
  [179].
- Branch/Y-shaped fins + ML: melting time −52.8%, heat dissipation +110% [cited
  fin studies].
- Metal foam optimization: solidification time −7.62%; charging time −34%
  [cited].
────────────────────────────────────────

## 9. Baseline Comparison

| Study area | Baseline | AI method | Improvement |
| --- | --- | --- | --- |
| Salt \(T_m\), \(L\) prediction | BP ANN | BP-PBO | Error −42% / −71% [51] |
| NEPCM conductivity | RSM | ANN | R² 0.79 → 0.99 [67] |
| Building TES operation | Rule-based / no MPC | ANN + MPC | Cost −9.1 to −14.6% [32] |
| Single PCM CLHS | RT50 alone | GA cascaded design | +6% stored energy [173] |
| Shell-and-tube LHS | Unoptimized layout | CFD-coupled GA | Energy +13.79%, exergy +14.85% [174] |
| Temperature field | CFD alone | SVM / RF hybrid | R² 0.99, RMSE 2.19–3.17 |
| Pumping/control (Barqawi-class) | Fixed speed | ANN flow multiplier | +2.5–4.1% energy (external SWH cite) |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A at review level. Cited experimental systems include:

- DSC-validated molten salts [52]
- Shell-and-tube and packed-bed CLHS loops [172, 178]
- Flat-plate solar CLHS [173]
- Building-integrated sensible/latent tanks with BMS sensors for MPC [32]
- No standard RPi/Arduino/ESP32 embedded deployment survey — identified as
  future need.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Many AI models are data-hungry and trained on narrow material/system classes.
- Black-box models lack interpretability and extrapolation beyond training
  bounds.
- CFD-coupled optimization is computationally expensive.
- Real-time embedded control and field validation rare vs offline simulation.
- Need physics-informed and hybrid CALPHAD+AI frameworks.
- Standardized benchmark datasets for PCM-AI missing.
- DRL cited but not yet mainstream in PCM-SWH household applications.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Highly relevant — §4.2 documents ANN–MPC,
  GA/PSO operational optimization, and emerging DRL; your PPO valve/pump policy
  extends this for PCM charge/discharge/bypass.
- RG2 (No integrated PCM–AI–hardware prototype): Highly relevant — Review covers
  full prediction+optimization stack but notes absence of low-cost embedded
  prototypes; directly motivates RPi/ESP32 + DS18B20 + solenoid deployment.
- RG3 (Poor alignment with household demand patterns): Relevant — Building PCM
  studies with demand forecasting (MLP/LSTM, 4.7–25.2% energy reduction) inform
  demand-shaped reward functions for DRL.
- RG4 (Limited real-world experimental validation): Relevant — Most cited AI–PCM
  work is simulation or lab-scale; your field evaluation across Indian climates
  addresses stated gap.
- RG5 (No predictive optimization under climatic uncertainty): Highly relevant —
  Couples with irradiance forecasting reviews; Liu's MPC and multi-objective
  exergy optimization support forecast-driven PCM selection (XGBoost) +
  predictive DRL.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Pre-melt sensible balance (representative lumped model cited across studies):

\[

\frac{dT_p}{dt} = \frac{h A}{m C_p}(T_{wf} - T_p)

\]

Latent charging at constant \(T_m\):

\[

\frac{dQ}{dt} = h A \max(0, T_{wf} - T_m), \quad Q_{max} = m L

\]

Effective NEPCM conductivity (ML target):

\[

k_{eff} = f(k_{pcm}, k_{np}, \phi, T, \mathrm{phase})

\]

where \(\phi\) = nanoparticle volume fraction (0.5–12 wt% in cited octadecane
study).

Exergy-based objective (CLHS optimization):

\[

\max f_E = \frac{E_{stored,exergy}}{E_{input}} \quad \text{preferred over pure
energy or entransy in [178]}

\]

MPC cost reduction metric (building TES):

\[

\Delta C = \frac{C_{baseline} - C_{ANN-MPC}}{C_{baseline}} \in [9.1\%, 14.6\%]

\]

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Lee et al., ANN + εDE + MPC for building TES, cost −9.1–14.6% — operational
   control benchmark.
1. Tamizharasan & Kini, deep learning for PCM-enhanced SWH, Int. J. Energy Res.,
   2023 — closest SWH+DL parallel.
1. Vempally & Dhanarathinam, ML PCM selection, J. Therm. Anal. Calorim., 2023 —
   aligns with your XGBoost classifier.
1. Barqawi, ANN pump control for PCM-SWH simulation, 2025 — retrofit SWH ML
   control baseline.
1. Yan et al., ML melting time in triplex-tube LHS, Appl. Energy —
   geometry-aware PCM ML predictor.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I (Introduction): AI transforms LHS from static design to
  predictive+optimized systems; thermal storage 209.4 GW global context.
- Section II (Literature Review): Two-column table: Liu prediction (XGBoost/ANN
  for PCM props) vs Liu optimization (GA/PSO/MPC/DRL for layout/operation).
- Section III (Methodology): Cite exergy-objective preference for
  multi-objective DRL reward design; adopt NEPCM \(k_{eff}\) ML error benchmarks
  (R² 0.99) for material feature validation.
- Section IV (Dataset & Setup): Structure PCM property database like Liu §3.1
  (Rubitherm/PLUSS + eutectic blends); 8-feature climate vector for labels.
- Section V (Results): Target exceeding CLHS GA gains (+13.79% energy) and MPC
  cost savings (9.1–14.6%) via integrated forecast+DRL on SWH hardware.
────────────────────────────────────────

# 30. Mansouri2025MultimodalRenewableForecasting_summary.md

Source path: /mnt/data/Mansouri2025MultimodalRenewableForecasting_summary.md

# Multimodal Learning Techniques for Time Series Forecasting in Renewable Energy Systems: A Comprehensive Survey

Authors: Majdi Mansouri, Khadija Attouri, Shady S. Refaat

Year: 2025

Journal/Conference: IEEE Access, Vol. 13, pp. 151970–151991 (article sequence)

DOI: https://doi.org/10.1109/ACCESS.2025.3602914

IEEE Citation: M. Mansouri, K. Attouri, and S. S. Refaat, "Multimodal learning
techniques for time series forecasting in renewable energy systems: A
comprehensive survey," IEEE Access, vol. 13, pp. 151970–151991, 2025, doi:
10.1109/ACCESS.2025.3602914.

────────────────────────────────────────

## 1. One-Line Summary

This survey categorizes and compares multimodal fusion strategies (early, late,
hybrid/attention, cross-modal, self-supervised) and deep architectures (CNN,
LSTM/GRU, Transformer, VAE, GNN) for renewable energy time-series forecasting,
while cataloguing benchmark datasets, metrics, deployment cases, and open
challenges including alignment, missing modalities, and lack of standardized
multimodal benchmarks.

────────────────────────────────────────

## 2. Problem Being Solved

- Renewable generation (solar, wind, hybrid) is stochastic and weather-driven;
  single-modality models fail on non-stationarity, missing sensors, and site
  transfer (Section II-B).
- Heterogeneous data—NWP, satellite imagery, SCADA/sensors, text/logs, grid
  data—exist at mismatched spatial/temporal resolutions, making naive fusion
  unreliable (Sections III, VIII-A–B).
- Prior surveys cover either single-modality forecasting or high-level AI
  overviews, not a technically grounded taxonomy of multimodal fusion + deep
  models + benchmarks for renewables (Abstract, Table 1).
- Operational deployment needs interpretability, uncertainty quantification, and
  scalable inference, which black-box multimodal models often lack (Sections
  VIII-E, IX-D).
────────────────────────────────────────

## 3. Key Contributions

1. Comparative survey positioning (Table 1): Contrasts recent
   renewable-forecasting surveys on domains, modalities, fusion techniques, deep
   models, and benchmark/metric coverage—claiming unique focus on multimodal
   fusion + deep architectures for solar/wind/hybrid.
1. Modality taxonomy (Section III): Numerical sensors (irradiance, wind, power),
   NWP (GHI, DNI, wind at hub height), satellite/sky imagery (GOES, Meteosat,
   Himawari, MODIS, Landsat), and textual SCADA/maintenance/weather bulletins
   with NLP pipelines.
1. Fusion-strategy synthesis (Section IV): Early (concatenation), late
   (modality-specific models + aggregation), hybrid/intermediate (attention,
   co-learning), cross-modal/co-attention, and
   self-supervised/contrastive/multimodal VAE approaches; Table 6 compares
   reported RMSE/MAE/accuracy across fusion types (values in source table).
1. Architecture review (Section V): Engineered multimodal features → CNN
   (spatial), LSTM/GRU (temporal), Transformer/cross-attention, multimodal
   AE/VAE, and GNN/GAT for spatial sensor topology.
1. Applications, metrics, datasets, challenges, future roadmap (Sections VI–IX):
   Solar PV with irradiance + clouds + weather; wind with SCADA + NWP; horizons;
   hybrid solar–wind–battery; grid-aware forecasting; real deployments (Japan,
   Australia, China, Korea); gaps in standardized multimodal benchmarks,
   federated learning, foundation models, and physics-informed multimodal
   fusion.
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

N/A — literature survey (23 pages, IEEE Access, CC BY 4.0). No new experiment.
Scope: solar PV, wind farms, hybrid renewable + storage, and grid-aware
forecasting using multimodal inputs.

Representative physical relationships used to frame applications:

- PV: \(P = \eta A G\) with effective irradiance \(G\) modulated by clouds,
  aerosols, shading (Section VI-A).
- Wind turbine power curve \(P(v)\): zero below \(v_{cut-in}\), cubic region to
  \(P_{rated}\), rated plateau, zero above \(v_{cut-out}\) (Section VI-B).
- Hybrid net power: \(P_{net}(t) = P_{solar}(t) + P_{wind}(t) +
  P_{storage}(t)\); battery SoC update with \(\eta_{charge}\),
  \(\eta_{discharge}\) (Section VI-D).
### 4b. Mathematical Models & Equations

NWP forecast error and ML bias correction:

- \(\text{Forecast Error} = y_{true} - \hat{y}_{NWP}\)
- \(y_{corrected} = f_{ML}(\hat{y}_{NWP}, \text{auxiliary features})\) — RF,
  GBM, or DNN
Cloud motion (satellite nowcasting):

- \(I_{predicted}(t+\Delta t) = I(t) + \vec{v}_{cloud} \cdot \Delta t\)
Text vectorization (TF-IDF):

- \(\mathrm{TF\text{-}IDF}(t,d) = \mathrm{tf}(t,d) \cdot
  \log\dfrac{N}{\mathrm{df}(t)}\) — (1)
Multimodal feature vector (traditional ML):

- \(\mathbf{x} = [x^{(1)}, x^{(2)}, \ldots, x^{(M)}]^\top\); \(y = f(\mathbf{x})
  + \varepsilon\)
CNN convolution:

- \(F_{i,j,k} = \sigma\left(\sum_{m,n,c} I_{i+m,j+n,c} \cdot K_{m,n,c,k} +
  b_k\right)\)
LSTM gates:

- \(f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)\), \(i_t = \sigma(W_i x_t + U_i
  h_{t-1} + b_i)\)
- \(c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)\)
- \(h_t = o_t \odot \tanh(c_t)\)
Transformer attention:

- \(\mathrm{Attention}(Q,K,V) =
  \mathrm{softmax}\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V\)
Multimodal VAE loss:

- \(\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] -
  \mathrm{KL}(q_\phi(z|x) \,\|\, p(z))\)
GNN layer:

- \(H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)}
  W^{(l)}\right)\)
Solar PV power (simplified):

- \(P = \eta \cdot A \cdot G\) — Section VI-A
Short-term forecast objective:

- \(\min_{\hat{P}_{t+\tau}} \mathbb{E}\left[(P_{t+\tau} -
  \hat{P}_{t+\tau})^2\right]\), \(\tau \leq\) few hours
Long-term decomposition:

- \(P_t = T_t + S_t + R_t\) (trend, seasonal, residual)
AC power flow (grid-aware excerpt):

- \(P_i = \sum_{j=1}^{N} V_i V_j (G_{ij}\cos\theta_{ij} +
  B_{ij}\sin\theta_{ij})\)
- \(Q_i = \sum_{j=1}^{N} V_i V_j (G_{ij}\sin\theta_{ij} -
  B_{ij}\cos\theta_{ij})\)
Forecasting metrics (Section VII-A):

- \(\mathrm{RMSE} = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}\)
- \(\mathrm{MAE} = \dfrac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|\)
- \(\mathrm{MAPE} = \dfrac{100\%}{N}\sum_{i=1}^{N}\left|\dfrac{y_i -
  \hat{y}_i}{y_i}\right|\)
- \(\mathrm{NRMSE} = \mathrm{RMSE}/(y_{max}-y_{min})\) or
  \(\mathrm{RMSE}/\bar{y}\)
### 4c. Algorithm / Control Method Steps

N/A as a single implemented pipeline — the survey describes workflows:

Typical multimodal forecasting pipeline:

1. Collect modalities (sensors, NWP grids, satellite/sky images, optional text).
1. Preprocess: calibration, resampling, spatial/temporal alignment,
   normalization \(X'=(X-\bar{X})/\sigma\).
1. Choose fusion: early (concatenate after encoders), late (separate models +
   average/stacking), hybrid (attention/gating in latent space), or cross-modal
   attention.
1. Train deep encoders (CNN/LSTM/Transformer/GNN) with loss MSE/MAE/RMSE;
   optional self-supervised/contrastive pretraining when labels scarce.
1. Evaluate with RMSE, MAE, MAPE, NRMSE, skill scores; deploy with compression
   (pruning, quantization, distillation) for edge/real-time use (Section
   VIII-C).
Hybrid plant operation (cited direction, not implemented here): reinforcement
learning and MPC for storage dispatch using forecasts (Section VI-D).

### 4d. Data Sources & Dataset Details

| Source / dataset (surveyed) | Modalities | Notes |
| --- | --- | --- |
| NREL NSRDB [119] | Solar radiation, satellite-derived | Cited in references; multimodal solar research |
| SolarAnywhere | Satellite + ground | Table 8 discussion |
| GEFCom (Global Energy Forecasting Competition) [123] | Energy + weather | Table 9 |
| SolarDB | Solar forecasting benchmark | Table 9 |
| Pecan Street, RENES | Hybrid home/grid + storage interactions | Table 8 |
| MODIS, GOES, Meteosat MSG, Himawari-8/9, Landsat 8/9 | Satellite imagery | Section III-C |
| NWP models (e.g., ECMWF, AROME — cited in refs) | GHI, DNI, wind, temperature, humidity | Sections III-B, refs [48], [51] |
| SCADA / smart meters | Power, wind, irradiance, temperatures | Tables 2–3, deployment cases |

Not used in this survey as primary sources: ERA5, NASA POWER, ISRO Solar
Calculator, Global Solar Atlas (India).

Geographic deployment examples: Kyushu (Japan), Western Australia, China, South
Korea (Section VI-F, Table 7).

### 4e. Validation Method

N/A as primary research — validation is by synthesis of published studies using
RMSE, MAE, MAPE, NRMSE, R², skill scores. Example deployed/cited outcomes:

- MODIS + NWP hybrid vs single-modality: 13.2% RMSE improvement [9]
  (Introduction).
- Kyushu, Japan CNN–LSTM ensemble (weather + calendar + power variables): R² =
  0.787, MAE = 1.936, RMSE = 2.630 [113] (Section VI-F).
- Table 6 / Table 9: Aggregated fusion-type and dataset-level metrics from
  literature (numeric cells in PDF tables not text-extracted).
────────────────────────────────────────

## 5. PCM Details (if applicable)

N/A — this survey addresses renewable power forecasting (solar PV, wind, hybrid
plants, grid) and does not study phase-change materials or solar hot water
thermal storage.

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm (surveyed families): Early/late/hybrid/cross-modal attention fusion;
  CNN, LSTM/GRU, Transformer (Informer, Temporal Fusion Transformer cited),
  multimodal AE/VAE, GNN/GAT; traditional RF/SVM/MLP on engineered features; NLP
  (TF-IDF, LDA, BERT-style embeddings); RL/MPC mentioned for hybrid storage
  dispatch (literature pointer only).
- Input features / state space: Irradiance (GHI, DNI), temperature, humidity,
  wind speed/direction, cloud cover, satellite/sky images, NWP grids, turbine
  SCADA (rotor speed, pitch, power), calendar features, load/grid states,
  optional text embeddings.
- Output / action space: Power generation, irradiance, price (Kyushu case), grid
  response variables — forecasts, not PCM tank control actions.
- Model architecture: Modality-specific encoders + fusion module; e.g., CNN–LSTM
  ensembles, ConvLSTM, transformer cross-attention \(Q,K,V\) across image and
  time-series branches.
- Hyperparameters: Not fixed (survey); discusses learning rate, hidden layers,
  Adam-class optimizers, attention sparsity, pruning/quantization/distillation
  for deployment.
- Training data size: Varies by cited study; emphasizes need for large
  multimodal corpora for foundation models (Section IX-A).
- Hardware used for training: GPUs/clusters noted as typical requirement; edge
  computing suggested to reduce centralized load (Section VIII-C).
- Performance metrics (examples from cited work):
- 13.2% lower RMSE (MODIS + NWP vs single-modality) [9]
- Deployment: R² = 0.787, MAE = 1.936, RMSE = 2.630 [113]
- Metrics framework: RMSE, MAE, MAPE, NRMSE, skill scores (Section VII)
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: NWP outputs; satellite platforms (GOES 5–15 min revisit;
  Meteosat 15 min, 1–3 km; Himawari 10 min; MODIS; Landsat 30 m); ground
  pyranometer/irradiance sensors; benchmarks NREL, SolarAnywhere, GEFCom,
  SolarDB, Pecan Street, RENES — not ERA5/NASA POWER/ISRO/Global Solar Atlas in
  body text.
- Variables used: GHI, DNI, temperature, humidity, wind speed/direction, cloud
  cover, precipitation, pressure, power output, satellite-derived cloud/albedo
  features.
- Geographic scope: Global literature; explicit deployment regions include
  Japan, Australia, China, South Korea; satellite sections reference Americas,
  Europe/Africa/Middle East, East Asia/Oceania.
- Temporal resolution: Sub-second turbine data to hourly weather stations;
  satellite 10–15 min typical; very-short-term solar nowcasting 0–30 min [55];
  short-term seconds–hours, long-term days–months (Section VI-C).
- Time period covered: Survey literature through 2025 (received Jul 2025,
  accepted Aug 2025).
- Clear-sky index / derived metrics: Not a dedicated survey topic; NWP bias
  correction and cloud-motion nowcasting discussed; skill scores mentioned for
  benchmarking.
────────────────────────────────────────

## 8. Key Results & Numbers

Survey paper — bullets cite quantitative claims and aggregated literature
results stated in the text.

- Hybrid MODIS satellite + NWP model: 13.2% RMSE improvement vs single-modality
  models [9] (Introduction).
- Kyushu, Japan operational-style deployment (CNN–LSTM multimodal ensemble): R²
  = 0.787, MAE = 1.936, RMSE = 2.630 [113] (Section VI-F).
- GOES satellite revisit: 5–15 min; Meteosat MSG: 15 min temporal, 1–3 km
  spatial; Himawari-8/9: 10 min revisit (Section III-C).
- Landsat 8/9: 30 m spatial resolution for terrain/vegetation (Section III-C).
- Very-short-term solar nowcasting horizon: 0–30 min [55] (Section III-C).
- Satellite imagery availability example: broad regions imaged every 15 min to 1
  h vs turbine sensors at sub-second intervals (Section VIII-A).
- NWP grids: resolutions from few km to tens of km, updates every few hours
  (Section VIII-A).
- Survey scope: 23 pages; compares fusion strategies across solar, wind, and
  hybrid systems (Abstract, Section VI).
- Western Australia case: LSTM on smart-meter import/export, rooftop PV,
  consumption, temperature — beats classical baselines on seconds-to-minutes
  horizons [114] (Section VI-F, qualitative superiority).
- Deployment summary captured in Table 7 (4 regional case studies); benchmark
  inventory in Tables 8–9 (NREL, GEFCom, SolarDB, etc.).
- Future work cites foundation models, federated multimodal learning, few-shot
  transfer, physics-informed multimodal fusion as open research axes (Section
  IX).
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Single-modality forecasts (sensors-only or NWP-only);
  early fusion vs late fusion vs hybrid/attention fusion (Table 6); classical
  statistical/physics models vs deep multimodal pipelines; regression/NWP raw vs
  ML bias-corrected NWP.
- Proposed method: Not one method — survey concludes hybrid and attention-based
  fusion often outperform naive early/late fusion when cross-modal interactions
  matter; late fusion is more modular/robust to missing modalities but may
  underperform without adaptive weighting (Sections IV-F, X).
- Improvement margin: Literature example: 13.2% RMSE reduction (multimodal vs
  unimodal) [9]; Kyushu multimodal R² = 0.787 vs implicit baselines in source
  study [113].
- Conditions of comparison: Varies by cited paper (solar vs wind, horizon,
  geography); survey stresses lack of standardized multimodal benchmarks for
  fair cross-study comparison (Sections VIII-D, VIII-F).
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — survey only; no project-built test rig. Discusses operational data
acquisition:

- Sensors: Pyranometers/irradiance, wind speed/direction, turbine SCADA, smart
  meters, temperature/humidity.
- Communication: Modbus, IEC 61850 (Section III-A).
- Compute: Training on GPU clusters; inference via model compression and
  edge/distributed processing for real-time grid use (Section VIII-C).
- Test environment: Cited field deployments (Japan, Australia, China, Korea)
  plus simulation/NWP pipelines — not PCM-SWH bench tests.
- Test duration: Not applicable at survey level; horizons discussed from 0–30
  min nowcasting to 2035 long-term policy projections (Korea case [115]).
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Spatiotemporal resolution mismatch across satellite, NWP, and sensor streams
  causes alignment artifacts and reduced accuracy (Section VIII-A).
- Asynchronous updates and missing modalities degrade models trained on complete
  inputs (Section VIII-B).
- High computational cost of deep multimodal models limits real-time deployment
  without compression/distillation (Section VIII-C).
- Lack of standardized multimodal benchmark datasets and evaluation protocols
  hinders reproducibility and fair comparison (Sections VIII-D, X).
- Black-box models limit interpretability and operator trust; need SHAP/LIME,
  attention visualization, uncertainty quantification (Sections VIII-E, IX-D).
- Text modality challenges: ambiguous terminology, low labels, multilingual
  logs, timestamp misalignment, privacy restrictions (Section III-D).
- Real deployments still face data quality, interpretability, and scalability
  barriers (Section VI-F).
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Not Relevant (as implemented). Survey
  targets forecasting, not closed-loop PCM-SWH control; RL/MPC appear only as
  cited tools for battery/hybrid dispatch, not domestic hot water valves or
  charging logic.
- RG2 (No integrated PCM–AI–hardware prototype): Not Relevant. No PCM tank,
  DS18B20, or embedded SWH prototype — focus is grid-scale PV/wind multimodal
  prediction.
- RG3 (Poor alignment with household demand patterns): Not Relevant. Does not
  model DHW draw profiles or end-use scheduling; smart-meter cases address grid
  import/export, not morning/evening hot water peaks.
- RG4 (Limited real-world experimental validation): Partially relevant.
  Highlights operational multimodal deployments (e.g., Japan R² = 0.787) but not
  PCM-SWH field trials; reinforces that AI renewables work is moving to
  operations while thermal PCM-SWH validation remains a separate gap.
- RG5 (No predictive optimization under climatic uncertainty): Highly relevant.
  Core thesis: fuse NWP + satellite + sensors for robust forecasts under weather
  variability — directly supports your ERA5/NASA POWER + pyranometer XGBoost
  layer and using forecasts as PPO state inputs for climate-adaptive PCM
  charging; paper notes adaptive fusion under uncertainty as future direction
  (Abstract) but does not use Indian cities or ERA5 by name.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
| --- | --- | --- |
| \(P = \eta A G\) | PV power from irradiance | Link pyranometer G to available solar gain for collector/PCM charging |
| \(y_{corrected} = f_{ML}(\hat{y}_{NWP}, \text{aux})\) | NWP bias correction | XGBoost correction of ERA5/NASA POWER vs local Coimbatore/Jaisalmer/Kochi measurements |
| \(X'=(X-\bar{X})/\sigma\) | Feature normalization | ANN/XGBoost/TFLite input pipeline |
| \(\mathrm{Attention}(Q,K,V)\) | Cross-modal fusion | Fuse irradiance time series + optional sky-camera/satellite embeddings |
| \(\mathrm{RMSE},\ \mathrm{MAE},\ \mathrm{MAPE}\) | Forecast skill metrics | Report Phase 1b forecast accuracy before RL |
| \(P_{net}=P_{solar}+P_{wind}+P_{storage}\) | Hybrid plant balance | Analogous structuring if adding battery later; PCM as thermal storage not covered |
| \(I_{pred}(t+\Delta t)=I(t)+\vec{v}_{cloud}\Delta t\) | Cloud motion nowcasting | Optional 0–30 min horizon layer above hourly ERA5 |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. T. Jing et al., "SolarFusion-Net: Enhanced solar irradiance forecasting via
   automated multi-modal feature selection and cross-modal fusion," IEEE Trans.
   Sustain. Energy, 2025 [14] — Relevant because: Direct multimodal solar
   irradiance forecasting architecture aligned with your GHI-driven PCM control.
1. K. Wang et al., "A robust photovoltaic power forecasting method based on
   multimodal learning using satellite images and time series," IEEE Trans.
   Sustain. Energy, 2025 [13] — Relevant because: Fuses satellite + time series
   for PV — analogous to pyranometer + satellite/ERA5 fusion.
1. J. Qin et al., "Enhancing solar PV output forecast by integrating ground and
   satellite observations with deep learning," Renew. Sustain. Energy Rev., 2022
   [6] — Relevant because: Ground + satellite solar forecasting precedent for
   Indian site calibration.
1. J. Heo et al., "Multi-channel convolutional neural network for integration of
   meteorological and geographical features in solar power forecasting," Appl.
   Energy, 2021 [9] — Relevant because: Source of cited 13.2% RMSE gain from
   multimodal meteorological/geographical fusion.
1. Y. Dong, "Robust dynamic modeling and optimal scheduling of
   wind-solar-storage systems via multi-modal data fusion under uncertainty,"
   Proc. NESP, 2025 [125] — Relevant because: Solar–storage + uncertainty +
   multimodal fusion closest thematic match to climate-adaptive
   thermal/electrical storage control.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 31. Martinez2025PCM_Industrial_TES_summary.md

Source path: /mnt/data/Martinez2025PCM_Industrial_TES_summary.md

# Phase Change Materials for Thermal Energy Storage in Industrial Applications

Authors: Franklin R. Martínez, Emiliano Borri, Saranprabhu Mani Kala, Svetlana
Ushak, Luisa F. Cabeza

Year: 2025

Journal/Conference: Heliyon, Vol. 11, Article e41025

DOI: https://doi.org/10.1016/j.heliyon.2024.e41025

IEEE Citation: F. R. Martínez, E. Borri, S. M. Kala, S. Ushak, and L. F. Cabeza,
"Phase change materials for thermal energy storage in industrial applications,"
Heliyon, vol. 11, no. e41025, 2025, doi: 10.1016/j.heliyon.2024.e41025.

────────────────────────────────────────

## 1. One-Line Summary

This study compiles 65 mid-temperature (60–80 °C) and 36 high-temperature
(150–250 °C) PCMs from literature and commercial datasheets, then experimentally
characterizes 14 shortlisted materials (DSC, TGA/DSC, Hot Disk), showing large
gaps versus published \(T_m\), \(\Delta H\), \(k\), and especially thermal
stability data.

────────────────────────────────────────

## 2. Problem Being Solved

- Industry emitted 9.0 Gt CO₂ in 2022 (~25% of global energy-system emissions),
  with slow efficiency and renewable uptake (Introduction, IEA [1]).
- PCM-TES can bridge supply–demand mismatch for industrial heat (60–80 °C and
  150–250 °C bands targeted for heat-pump-coupled storage), but no consolidated
  property database exists for these ranges (Section 2).
- Literature and vendor datasheets report inconsistent melting enthalpy,
  degradation temperature, and thermal conductivity — selection remains
  difficult (Abstract, Section 4).
- Many catalogued PCMs lack complete property sets (density, \(C_p\), \(k\),
  \(T_{deg}\), NFPA 704) in published tables (Tables 1–2 footnotes).
────────────────────────────────────────

## 3. Key Contributions

1. Screening database: 65 PCMs for 60–80 °C (Table 1) and 36 for 150–250 °C
   (Table 2) from Scopus literature + commercial sheets (Rubitherm,
   PCMproducts/PLUSS, CRODA).
1. Shortlists: 8 mid-temperature + 6 high-temperature candidates (Tables 3–4)
   including RT 54 HC, RT 55, RT 64 HC, E 58, salt hydrates, palmitic/stearic
   acid, nitrate salts.
1. Experimental characterization of 14 PCMs with METTLER TOLEDO DSC 3+ and
   TGA/DSC 3+ (±0.1 °C, ±3 J/g); Hot Disk TPS 2500 S (Kapton 5506, mean
   deviation 1×10⁻⁴); 3 thermal cycles per DSC sample at 1 K/min.
1. Cross-validation: Literature vs measured variances up to −98% \(\Delta H\)
   (hydrate mixture), +80% \(\Delta H\) (RT 55 TGA), +266% \(k\) (NaNO₃–KNO₃
   60–40), −91% \(\Delta H\) (E 58).
1. Open dataset: Characterization data deposited at
   https://doi.org/10.34810/data1822 (Data availability).
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

Type: Materials screening + laboratory thermophysical characterization (no
full-scale TES tank or SWH loop).

Temperature targets:

- Mid: 60–80 °C (aligned with dairy pasteurization, drying, etc., and
  overlapping domestic SWH PCM band).
- High: 150–250 °C (industrial process heat, solar salt applications).
Purchased samples (Section 2.2):

- Salts/acids: Mg(NO₃)₂·6H₂O, MgCl₂·6H₂O, palmitic acid, stearic acid, LiNO₃,
  NaNO₃, KNO₃ (Merck/VWR/Panreac).
- Commercial PCMs: RT 54 HC, RT 55, RT 64 HC (Rubitherm); E 58 (PCM Products
  UK).
DSC/TGA sample prep: ~15 mg in Al crucibles (40 µL, sealed) or sapphire
crucibles (70 µL, open) under N₂; scan from ~50 °C below to ~50 °C above
literature \(T_m\) (Fig. 1).

Hot Disk samples: Compact flat discs (Fig. 2); 3 repeat measurements per PCM
after parameter convergence.

### 4b. Mathematical Models & Equations

No transient TES or CFD model — correlations used only for reporting deviations:

- Percent variance (property X): \(\mathrm{Var}(\%) = \dfrac{X_{exp} -
  X_{lit}}{X_{lit}} \times 100\)
Dimensionless groups cited in selection context (literature review, not derived
here):

- Stefan number \(\mathrm{Ste} = c_p \Delta T / L\) — discussed in related
  Cabeza/Zalba reviews [17, 21] for PCM heat transfer.
Heat transfer correlations referenced for future HX design (from cited work,
Section 1):

- Dittus–Boelter-type relations appear in linked PCM-HX literature [81, 85] but
  are not fitted in this paper.
Energy storage density (implicit selection criterion):

- \(E_{latent} \approx \rho \cdot \Delta H_{melting}\) (J/m³) — used
  conceptually when comparing \(\Delta H\) and \(\rho\) in Tables 1–2.
### 4c. Algorithm / Control Method Steps

N/A — no control system or ML. Selection workflow:

1. Literature + vendor database search (Scopus; Rubitherm, PCMproducts, PLUSS,
   CRODA).
1. Filter by \(T_{melting}\) in target bands; compile \(T_m\), \(\Delta H\),
   \(C_p\), \(\rho\), \(k\), \(T_{deg}\), NFPA 704 (Section 2.1).
1. Expert shortlist: include commercial organics + inorganic/organic acids/salts
   (Tables 3–4).
1. DSC (3 cycles, discard cycle-1 average for powder vs recrystallized sample) →
   TGA/DSC (25–250 °C mid, 25–400 °C high) → Hot Disk \(k_{solid}\).
1. Compare to literature/datasheet; flag decomposition before melt (e.g.,
   salicylic acid).
### 4d. Data Sources & Dataset Details

| Source | Content | Scope |
| --- | --- | --- |
| Scopus scientific literature | PCM property tables | Global publications |
| Rubitherm datasheets [24] | RT series PCMs | Commercial organics |
| PCMproducts / PLUSS [34, 46, 77] | PlusICE, E 58, etc. | Commercial |
| CRODA CRODATHERM [16] | Paraffin products | Commercial |
| NASA CR-51363 handbook [25, 35] | Legacy PCM data | Reference |
| Open repository | Measured DSC/TGA/k | https://doi.org/10.34810/data1822 |

Counts (Section 3.1): Mid — 45 literature + 20 commercial entries → 65 total;
High — 30 + 6 → 36 total.

### 4e. Validation Method

- Internal: DSC/TGA instrument accuracy ±0.1 °C, ±3 J/g; Hot Disk repeatability
  target mean deviation 1×10⁻⁴; DSC 3-cycle repeatability check.
- External: Compare measured vs literature/datasheet; Table 19 summary with Var
  (%) columns.
- Example variances (Table 19, DSC vs literature):
- RT 54 HC: \(T_m\) 55.4 °C (lit. 54 °C), \(\Delta H\) 172.1 J/g (−5%), \(k\)
  0.23 W/m·K (+15%).
- RT 55: \(\Delta H\) 114.5 J/g (−28%); TGA \(\Delta H\) +80% vs datasheet.
- RT 64 HC: \(\Delta H\) 166.9 J/g (−31%), \(k\) 0.33 W/m·K (+64%).
- E 58: \(\Delta H\) 13.8 J/g (−91% vs 145 J/g rated) — unsuitable as tested.
- Palmitic acid: \(\Delta H\) 182.7 J/g at 64.0 °C; \(k\) 0.26 W/m·K (+73% vs
  0.15 lit.).
- Stearic acid: 70.4 °C, 194.2 J/g, \(k\) 0.26 W/m·K.
- NaNO₃–KNO₃ (60–40): \(T_m\) 223.2 °C, \(\Delta H\) 85.8 J/g (−21%), \(k\) 0.88
  W/m·K (+266% vs 0.24 lit.).
- LiNO₃: \(T_m\) ~250 °C, \(\Delta H\) 276.9 J/g (−8 to −25% vs lit. 370 J/g),
  \(k\) 0.84 W/m·K.
────────────────────────────────────────

## 5. PCM Details (if applicable)

Primary focus: characterized + catalogued materials. Rubitherm grades directly
match FYP PCM family.

### Mid-temperature band (60–80 °C) — selected & tested

| Material | \(T_m\) (°C) lit. / exp. | \(\Delta H\) (J/g) lit. / exp. | \(k_{solid}\) (W/m·K) lit. / exp. | \(T_{deg}\) notes |
| --- | --- | --- | --- | --- |
| RT 54 HC | 54 / 55.4 | 182 / 172.1 | 0.20 / 0.23 | Onset 130.3 °C; use below 130 °C |
| RT 55 | 55 / 55.2 | 158 / 114.5 | 0.20 / 0.27 | TGA \(T_m\) +10 °C vs DSC |
| RT 64 HC | 64 / 55.5† | 242 / 166.9 | 0.20 / 0.33 | †DSC peak lower than grade name |
| E 58 | 58 / 57.7 | 145 / 13.8 | 0.69 / failed Hot Disk | Rated \(T_{deg}\) 120 °C; not validated |
| Palmitic acid | 55–69 / 64.0 | 163–222 / 182.7 | 0.15–0.17 / 0.26 | Multiple lit. \(T_m\) values |
| Stearic acid | 67.8 / 70.4 | 198.9 / 194.2 | 0.17 / 0.26 | Stable cycling |
| Mg(NO₃)₂·6H₂O + MgCl₂·6H₂O (80–20) | 60 / no clear peak | 150 / 2.8 | n.a. | −98% \(\Delta H\); dehydration dominates |
| Mg(NO₃)₂·6H₂O + MgCl₂·6H₂O (60–40) | 60 / 59.6 | 132.3 / 28.9 | n.a. | −78% \(\Delta H\) |

†Nominal vs measured melt discrepancy flagged in Table 19.

### High-temperature band (150–250 °C) — tested subset

| Material | \(T_m\) (°C) | \(\Delta H\) (J/g) exp. | \(k_{solid}\) (W/m·K) exp. |
| --- | --- | --- | --- |
| LiNO₃–NaNO₃–KNO₃ (20-28-52) | 175.9 | 103.7 | 0.69 |
| Salicylic acid | decomposes before melt | — | — |
| LiNO₃–NaNO₃ (49–51) | 175.9 | 66.7 (−75% vs lit.) | 0.56 |
| NaNO₃–KNO₃ (50–50) | 212.7 | 65.4 (−35%) | 0.91 |
| NaNO₃–KNO₃ (60–40) | 223.2 | 85.8 | 0.88 |
| LiNO₃ | 249.6 | 276.9 | 0.84 |

### Catalog examples (Table 1, not all tested)

- RT 60: \(T_m\) 60 °C, \(\Delta H\) 160 J/g, \(\rho_s\) 880 kg/m³, \(k\) 0.20
  W/m·K, \(T_{deg}\) 80 °C
- RT 80: \(T_m\) 80 °C, \(\Delta H\) 220 J/g, \(\rho_s\) 900 kg/m³
- PureTemp 60: \(\Delta H\) 220 J/g; CRODATHERM 60: 217 J/g at 60 °C
- Ba(OH)₂·8H₂O: \(\Delta H\) up to 301 J/g at 78 °C; \(\rho_s\) 2180 kg/m³
- Performance metrics reported: Melting enthalpy J/g, degradation/onset
  temperature °C, solid thermal conductivity W/m·K, NFPA 704 hazard class 1–3;
  no system COP or tank efficiency (materials study only).
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A — materials characterization and literature compilation only; no machine
learning, forecasting, or TES control.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

N/A — industrial process-heat framing; no solar irradiance, ERA5, NASA POWER, or
Indian climate datasets. Indirect link: mid-temperature band 60–80 °C overlaps
solar DHW / SWH PCM operating range cited in other Cabeza-group work.

────────────────────────────────────────

## 8. Key Results & Numbers

- 65 mid-temperature + 36 high-temperature PCMs catalogued; 14 experimentally
  characterized (8 + 6).
- RT 54 HC: DSC \(\Delta H\) 172.1 J/g (−5% vs 182 J/g datasheet); \(k\) 0.23
  W/m·K (+15%); degradation onset 130.3 °C.
- RT 55: DSC \(\Delta H\) 114.5 J/g (−28%); TGA/DSC \(\Delta H\) 284.9 J/g
  (+80%); \(k\) 0.27 W/m·K (+34%).
- RT 64 HC: DSC \(\Delta H\) 166.9 J/g (−31% vs 242 J/g); \(k\) 0.33 W/m·K
  (+64%).
- E 58: \(\Delta H\) 13.8 J/g (−91% vs 145 J/g) — material not reliable per
  authors’ tests.
- Palmitic acid: \(T_m\) 64.0 °C, \(\Delta H\) 182.7 J/g, \(k\) 0.26 W/m·K (+73%
  vs 0.15 literature).
- Stearic acid: \(T_m\) 70.4 °C, \(\Delta H\) 194.2 J/g, \(k\) 0.26 W/m·K.
- Mg(NO₃)₂/MgCl₂ (80–20): \(\Delta H\) 2.8 J/g (−98% vs 150 J/g literature) — no
  usable phase-change peak.
- LiNO₃: DSC \(\Delta H\) up to 456.5 J/g on TGA branch (+52% vs 370 J/g lit.);
  \(k\) 0.84 W/m·K vs 1.70 lit. (−51%).
- NaNO₃–KNO₃ (60–40): \(k\) 0.88 W/m·K (+266% vs 0.24 W/m·K literature).
- Salicylic acid: decomposes before melting — excluded as PCM.
- DSC equipment accuracy: ±0.1 °C, ±3 J/g; heating rate 1 K/min; 3 cycles.
- Industry context: 9.0 Gt industrial CO₂ (2022); target TES bands 60–80 °C and
  150–250 °C.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Published literature values and manufacturer datasheets
  (Rubitherm, PCM Products, etc.) vs this study’s DSC/TGA/Hot Disk measurements.
- Proposed method: Unified experimental protocol (3-cycle DSC, TGA stability,
  Hot Disk \(k\)) on 14 shortlisted PCMs.
- Improvement margin: Not “better performance” — exposes gaps: e.g., RT 55
  enthalpy −28% (DSC); E 58 −91%; hydrate mix −98%; conductivities +15% to +266%
  vs literature for several salts.
- Conditions: Same nominal chemistry; powder vs recrystallized cycle-1 excluded
  from averages (Section 3.4).
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

- Physical components: METTLER STARe DSC 3+; STARe TGA/DSC 3+; Hot Disk TPS 2500
  S (sensor Kapton 5506 F2); Al crucibles 40 µL; sapphire 70 µL; N₂ purge.
- Sensor specs: Temperature ±0.1 °C; enthalpy ±3 J/g; TGA balance ±0.00001 g.
- Embedded/compute platform: N/A — lab calorimetry only.
- Test environment: GREiA lab, Universitat de Lleida, Spain (authors also
  affiliated with University of Antofagasta, Chile).
- Test duration: 3 thermal cycles per PCM (DSC); TGA scans 25–250 °C or 25–400
  °C; 3 Hot Disk repeats per sample.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Degradation temperature was the most difficult parameter to find in literature
  and is critical for safety and operating limits (Section 4).
- DSC vs TGA/DSC enthalpy can disagree strongly (e.g., RT 55 −28% vs +80%);
  authors state DSC is more precise for enthalpy (Section 3.3.1–3.3.2).
- First DSC cycle excluded from averages because powder packing differs from
  recrystallized material (Section 3.4).
- Many catalogued PCMs lack complete property rows in Tables 1–2 (missing
  \(C_p\), \(\rho\), \(k\), or \(T_{deg}\)).
- Study defines materials for industrial TES — next step still requires mapping
  to final application operating temperature before tank design (Section 4).
- E 58, salicylic acid, and some hydrates failed stability or phase-change
  criteria — not all shortlisted materials are viable.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Not Relevant. Pure materials screening;
  no controllers, pumps, or charging logic.
- RG2 (No integrated PCM–AI–hardware prototype): Partially relevant. Provides
  verified Rubitherm RT 54 HC / RT 55 / RT 64 HC properties (close to
  RT35–RT64HC family) for simulator parameterization — but no RPi/ESP32/DS18B20
  tank prototype.
- RG3 (Poor alignment with household demand patterns): Not Relevant. Industrial
  heat processes (pasteurization, drying, etc.), not DHW draw profiles.
- RG4 (Limited real-world experimental validation): Partially relevant. Rigorous
  lab DSC/TGA/k validation of commercial PCMs (including RT grades) supports
  using measured not datasheet-only properties in your model — but no full SWH
  field test.
- RG5 (No predictive optimization under climatic uncertainty): Not Relevant. No
  weather data or forecast-driven optimization.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
| --- | --- | --- |
| \(Q_{stored} \approx m \cdot \Delta H_{melting}\) | Latent storage capacity | Size PCM mass in tank for target kWh |
| \(\mathrm{Var}(\%) = (X_{exp}-X_{lit})/X_{lit} \times 100\) | Property uncertainty | Sensitivity bounds for RT35/OM35 in grey-box model |
| \(Nu = 1.86\left(\frac{Re\cdot Pr}{L/L_p}\right)^{1/3}\) (from related HX refs [43]) | Tube PCM convection | If modeling coil in tank (optional) |
| \(Q_o = \dot{m} c_p (t_{out}-t_{in})\) | Sensible heat (test fluids) | Calibrate charging experiments |
| Enthalpy averaging: mean(cycles 2–3) | Stable PCM cycling | Lab protocol for validating PLUSS/Rubitherm batches |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. L. F. Cabeza, A. Castell, et al., "Materials used as PCM in thermal energy
   storage in buildings: a review," Renew. Sustain. Energy Rev., 2011 [17] —
   Relevant because: Foundational building PCM database overlapping SWH
   temperatures.
1. J. Pereira da Cunha, P. Eames, "Thermal energy storage for low and medium
   temperature applications using phase change materials – a review," Appl.
   Energy, 2016 [15] — Relevant because: Low/medium-temperature PCM-SWH/TES
   applications review.
1. B. Zalba, J. M. Marín, L. F. Cabeza, H. Mehling, "Review on thermal energy
   storage with phase change," Appl. Therm. Eng., 2003 [21] — Relevant because:
   Classic PCM enthalpy + heat transfer reference for FYP theory section.
1. L. Miró, C. Barreneche, et al., "Health hazard, cycling and thermal stability
   as key parameters when selecting a suitable PCM," Thermochim. Acta, 2016
   [103] — Relevant because: Thermal cycling and stability selection criteria
   for long-life SWH PCM.
1. J. Li, et al., "A hybrid photovoltaic and water/air based thermal (PVT) solar
   energy collector with integrated PCM for building application," Renew.
   Energy, 2022 [8] — Relevant because: PCM + solar thermal system at building
   scale from same research network.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 32. Mohammed2025NanoAI_ThermalSystems_summary.md

Source path: /mnt/data/Mohammed2025NanoAI_ThermalSystems_summary.md

# The Role of Nanotechnology and Artificial Intelligence in Optimizing Thermal Energy Systems

Authors: Hayder I. Mohammed, Farhan Lafta Rashid, Hussein Togun, Ephraim Bonah
Agyekum, Arman Ameen, Karrar A. Hammoodi, Rujda Parveen, Saif Ali Kadhim, Walaa
N. Abbas

Year: 2025

Journal/Conference: Applied Energy, Vol. 400, Article 126576

DOI/Link: https://doi.org/10.1016/j.apenergy.2025.126576

IEEE Citation: H. I. Mohammed et al., "The role of nanotechnology and artificial
intelligence in optimizing thermal energy systems," Appl. Energy, vol. 400, p.
126576, 2025, doi: 10.1016/j.apenergy.2025.126576.

────────────────────────────────────────

## 1. One-Line Summary

This narrative review synthesizes ~180 studies (2013–2024) on nano-enhanced
PCMs/nanofluids (e.g., +28.8% conductivity) combined with AI (ANN, PSO, XGBoost,
DRL) for solar collectors, SWH, and latent storage—reporting >97% prediction
accuracy in cited works, 28% HVAC energy savings (ROI <3 years), and identifying
gaps in scalability, cost, and field validation.

────────────────────────────────────────

## 2. Problem Being Solved

- Conventional PCMs have low thermal conductivity, slow charge/discharge, and
  poor real-time controllability in solar thermal and SWH systems.
- FPSCs/SWHS suffer heat losses and limited working-fluid conductivity; passive
  PCM envelopes often yield only 1–3% savings in cited passive-building studies.
- Nanotechnology improves materials but faces agglomeration, cost, toxicity, and
  cycling durability issues.
- AI models are often trained on steady-state or siloed datasets, lacking
  integration with hardware prototypes and climate-adaptive control under
  extreme transients.
- Need a unified roadmap for NePCM + AI hybrid TES spanning prediction,
  optimization, and deployment economics.
────────────────────────────────────────

## 3. Key Contributions

1. Dual-pillar framework: nanotechnology (NePCM, nanofluids, nano-coatings) + AI
   (ML/DL/DRL, PSO, MPC) for TES optimization.
1. Structured literature synthesis: Scopus, WoS, ScienceDirect, IEEE, Springer;
   Boolean search on NePCM, nanofluid, solar collector, AI/ML/DL, HVAC; ~180
   core papers (2013–2024).
1. KPI taxonomy (Table 1): thermal/energetic efficiency, heat-transfer
   enhancement, ROI, payback, LCA, reliability.
1. PCM classification (Table 2): organic/inorganic; low/medium/high \(T_m\);
   paraffin, salt hydrates, fatty acids.
1. AI algorithm map (Section 4.3): predictive (ANN, SVM, RF, XGBoost), control
   (RL, ANFIS, PSO, GA), fault detection (LSTM, autoencoders).
1. Synergy case studies: nanofluid HX (+20% efficiency), AI solar thermal
   (+25%), Fraunhofer NePCM-HVAC (28% energy cut), ML+CFD surrogate (>99%
   compute savings).
1. Challenge roadmap: nanoparticle stability, AI compute cost, data scarcity,
   sensor drift, regulatory/LCA gaps.
────────────────────────────────────────

## 4. Methodology

### 4a. Review approach

- Narrative (not PRISMA-systematic) but structured inclusion/exclusion.
- Inclusion: empirical or comparative reviews on TES, solar thermal, NePCM, AI
  control/modelling (2013–2024).
- Exclusion: non-peer-reviewed, insufficient technical data.
### 4b. Technical domains covered

1. NePCM physics: equivalent enthalpy, nanoparticle dispersion (0D–2D: CuO,
   Al₂O₃, CNT, graphene, MXene).
1. Nanofluids: Buongiorno model; Brownian motion/thermophoresis; base fluids
   water/EG/oil.
1. AI pipeline: data from CFD/experiments → train ANN/XGBoost/LSTM → PSO/GA
   hyperparameter or geometry optimization → optional RL closed-loop control.
1. Hybrid ML+CFD: CFD labels train surrogate; surrogate replaces expensive
   simulations in design loops.
### 4c. Validation cited in review

- Multiple third-party studies (Kalani 130 experimental PV/T points; Fraunhofer
  12-month campus pilot; field-tested nano-coated solar thermal per [263]).
────────────────────────────────────────

## 5. PCM Details (if applicable)

| Enhancement | Nanoparticle / system | Reported effect (from cited studies) |
| --- | --- | --- |
| Paraffin NePCM | 3% TiO₂ | Thermal conductivity +25% |
| Paraffin NePCM | CuO | Conductivity up to +28.8% (abstract headline) |
| Neopentyl glycol | CuO | Significant conductivity gain [106] |
| Basic PCM | SWCNT / MWCNT | +134% / +339% conductivity vs base PCM |
| PCM + fin metal foam | — | Melting time −83.35% |
| Paraffin + CuO (cycling) | — | Latent heat −12% after 200 thermal cycles (agglomeration) |
| Solar still | CuO + PCM mix | Freshwater productivity +108% |
| Medium-T PCM class | Paraffin, RT-class organics | \(T_m\) between room temp and 100 °C — SWH-relevant band |

Stability mitigation: surfactants, encapsulation, metal foam matrices,
ultrasonic dispersion; CNT/graphene reported stable >500 cycles in cited work.

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

| Algorithm | Application in review | Reported metrics |
| --- | --- | --- |
| ANN + PSO | PV/T nanofluid collector (Kalani); molten-salt/wind hybrid | PSO AAPD 0.47–0.51%; GOA 0.05–0.27% |
| ANFIS / RBF | PV/T outlet temperature (130 experiments) | Best among compared models [24] |
| XGBoost / LGB / GBR | PV/T energy prediction (15,540 samples) | LGB \(R^2=0.983\) thermal; MLP electrical \(R^2=0.0906\) |
| ANN | ITES-HVAC, solar irradiance forecast | ITES R=0.94–0.99, MSE <20%; solar ANN MAE=0.9558 |
| MPC | SMR + two-tank TES; microgrid electro-thermal | 24-h vs 8-h horizon: −6.71% cost, −15.68% temp RMSE; −31.57% PV curtailment |
| DRL / GAN+RL | Fuel-cell thermal management; smart-grid bidding | Reward-based adaptive control |
| ML + CFD | Compact TES discharge | >99% computational time reduction [203] |
| ANN surrogate | Al₂O₃-Cu/water FPSC CFD | MAPE <2.5%; efficiency +17.6% at 1.2 vol%; optimization 85% faster |
| ODNN + Sand Cat Swarm | PV/T cooling design | Lower MAE/MSE, higher \(R^2\) [202] |

Abstract claim: some cited models achieve prediction accuracies above 97% under
complex conditions.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Review cites solar irradiance, ambient temperature, wind, humidity as AI
  inputs for collectors and green-roof models (e.g., Shanghai meteorological
  data [204]).
- No single project dataset — aggregates literature using measured weather, TMY,
  and operator databases (e.g., California ISO for SMR study).
- Project link: supports using NASA POWER / ERA5 GHI, \(T_{amb}\), wind as
  DRL/XGBoost state features for Coimbatore, Jaisalmer, Kochi; Fraunhofer case
  uses weather + occupancy for HVAC AI.
────────────────────────────────────────

## 8. Key Results & Numbers

- Al₂O₃ nanofluid (1.5 vol%) in FPSC: efficiency +31.64% [11].
- Al₂O₃/water FPSC (Yousefi): +28% efficiency; MWCNT: +35%.
- Hybrid Ag/graphite/CNT nanofluid: FPSC +5% efficiency.
- NePCM CuO in paraffin: conductivity up to +28.8% (headline); +25% with 3%
  TiO₂.
- CNT-enhanced PCM: conductivity +134% (SWCNT) and +339% (MWCNT) vs base.
- PCM + metal foam fins: melting time −83.35%.
- Latent heat degradation: −12% after 200 cycles (CuO-paraffin agglomeration).
- Kalani PV/T ANN/PSO: 130 experimental datasets; reliable outlet-temperature
  prediction.
- Fraunhofer NePCM + AI HVAC: −28% energy, +21% comfort, ROI <3 years.
- Google AI data-centre cooling: up to −40% cooling energy.
- Nanofluid HX case study: +20% energy efficiency [262].
- AI nano-coated solar thermal field test: +25% vs conventional [263].
- MgO-CuO/water in heat-pipe ETC: average efficiency +20%; payback −27% [295].
- Fe₃O₄-water HX: Nusselt number +13% [294].
- ML+CFD TES: >99% compute-time savings [203].
- ANN-FPSC hybrid CFD: MAPE <2.5%, efficiency +17.6%, optimal nanoparticle 1.2
  vol%.
- MPC microgrid: −5.86% operating cost, −31.57% PV curtailment.
- Power plant NOx ANN: \(R^2=0.97\).
- Review scope: ~180 publications; open-access CC BY.
────────────────────────────────────────

## 9. Baseline Comparison

| Approach | Baseline | Improvement cited |
| --- | --- | --- |
| Al₂O₃ nanofluid FPSC | Pure water working fluid | +28–31.64% thermal efficiency |
| NePCM (CuO/TiO₂) | Plain paraffin PCM | +25–28.8% conductivity |
| CNT NePCM | Base PCM | +134–339% conductivity |
| AI-optimized BIHP-PCM HVAC (Fraunhofer) | Conventional HVAC | −28% energy |
| ML surrogate vs full CFD | Full CFD per design point | >99% time reduction |
| AI solar tracking + nanofluid | Traditional solar thermal | +25% efficiency |
| Passive PCM building (literature cite) | Reference building | 1–3% only — motivates active AI+nano |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

Review aggregates setups rather than one unified experiment:

- Double-pipe / shell-and-tube HX with Fe₃O₄, Al₂O₃ nanofluids.
- Heat-pipe evacuated-tube collectors with hybrid MgO-CuO nanofluid [295].
- FPSC with Al₂O₃-Cu hybrid nanofluid; nano-coated TiO₂ absorbers.
- Smart-campus Fraunhofer pilot: NePCM storage + predictive AI HVAC (12-month
  deployment).
- Sensors implied: temperature, flow, irradiance, nano-sensors for fouling
  detection.
- No RPi/Arduino SWH prototype in this review — gap your FYP addresses (RG2).
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Nanoparticle cost and industrial-scale synthesis remain barriers.
- Nanofluid stability: agglomeration, viscosity, pressure drop, long-term
  sedimentation.
- NePCM cycling: phase separation, 12% latent-heat loss after 200 cycles
  (cited).
- Environmental/toxicity of CuO vs lower-risk Al₂O₃; disposal pathways unclear.
- AI: steady-state training data miss extremes; computational overhead;
  reproducibility gaps; data protection and integration with legacy plant.
- Lack of open, standardized TES ML datasets for benchmarking.
- Review is narrative, not systematic — selection bias possible.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (real-time adaptive control): Highly relevant — reviews DRL, MPC, RL for
  flow/control; cites real-time nanofluid flow regulation; your PPO valve
  control fits this gap.
- RG2 (integrated PCM–AI–hardware): Highly relevant — identifies missing
  embedded end-to-end SWH prototypes; cites PV/T and HVAC pilots but not
  low-cost RPi + DS18B20 + solenoid PCM tank.
- RG3 (household demand): Partially relevant — HVAC occupancy/weather AI
  discussed; limited hot-water draw profile optimization; extend to evening
  demand peaks.
- RG4 (field validation): Relevant — Fraunhofer 12-month and field solar tests
  cited; flags insufficient extreme-scenario validation; benchmark your
  Coimbatore/Jaisalmer/Kochi trials against 28% savings claims.
- RG5 (climate uncertainty): Highly relevant — LSTM forecasting, MPC with
  irradiance horizon, federated/edge AI; supports ERA5/NASA POWER-driven XGBoost
  + DRL under monsoon/dust/climate zones.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

PCM latent storage (review notation):

\[

Q = m \cdot L \quad \text{(sensible + latent during phase change)}

\]

Equivalent specific-heat PCM model (triangular \(c_p(T)\) over \(\Delta T\)):

\[

H = \tfrac{1}{2}\Delta T \cdot \Delta c_p

\]

Nanofluid effective property (conceptual — cite Buongiorno / Maxwell-Garnett in
grey-box):

\[

k_{nf} = \phi k_p + (1-\phi) k_f \quad \text{(baseline mixture form; review
discusses enhanced models)}

\]

AI performance metrics used across cited studies:

\[

\mathrm{MAPE} = \frac{100}{n}\sum\left|\frac{y_i-\hat{y}_i}{y_i}\right|, \quad

R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}

\]

DRL reward template: energy saved vs baseline minus comfort violation penalty —
aligns with your PCM charge/discharge reward.

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Kalani et al., ANN+PSO for PV/T nanofluid collector, Appl. Therm. Eng., 2017
   — 130-point experimental ML baseline [24].
1. Al-Waeli et al., ANN for PV/T nano-PCM/nanofluid, Sol. Energy, 2018 — hybrid
   material + AI [25].
1. He et al., AI methods for TES prediction/design/control, Renew. Sust. Energy
   Rev., 2022 — TES-AI survey [33].
1. Olabi et al., AI prediction/optimization/control of TES, Therm. Sci. Eng.
   Prog., 2023 — direct TES-AI review [23].
1. Bharathiraja et al., hybrid NePCM flat-plate SWH, J. Energy Storage, 2024 —
   SWH + nano-PCM experimental [277].
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I: Cite low PCM conductivity and 1–3% passive savings vs 28–35%
  nano/AI gains to motivate integrated PCM-SWH control.
- Section II: Use as umbrella review for nano-AI TES; position your work as
  closing the hardware integration + climate-adaptive DRL gaps flagged in
  Sections 6–7.
- Section III: Justify XGBoost (review ranks it top for speed/accuracy) for PCM
  class selection; PPO under DRL subsection; optional NePCM as future work
  (RT/OM baseline first).
- Section IV: Mirror KPIs from Table 1 (thermal efficiency, temperature
  uniformity, ROI); sensor quality argument from Google/Fraunhofer cases.
- Section V: Benchmark against +17.6% ANN-FPSC efficiency, 28% Fraunhofer HVAC
  savings, or >97% predictor accuracy — state your RMSE/MAPE/% energy
  improvement relative to rule-based valve control.
────────────────────────────────────────

# 33. OdoiYorke2025AI_SWH_Review_summary.md

Source path: /mnt/data/OdoiYorke2025AI_SWH_Review_summary.md

# Artificial Intelligence for Solar Water Heating Systems: A Review of Global Research Trends, Advances, and Future Perspectives

Authors: Flavio Odoi-Yorke

Year: 2025

Journal/Conference: Energy Conversion and Management: X, Vol. 28, Article 101378

DOI/Link: https://doi.org/10.1016/j.ecmx.2025.101378

IEEE Citation: F. Odoi-Yorke, "Artificial intelligence for solar water heating
systems: A review of global research trends, advances, and future perspectives,"
Energy Convers. Manag.: X, vol. 28, p. 101378, 2025, doi:
10.1016/j.ecmx.2025.101378.

────────────────────────────────────────

## 1. One-Line Summary

This dual-method review combines PRISMA-guided bibliometric analysis of 245
Scopus-indexed AI–SWHS papers (2000–2024) with a qualitative systematic review,
showing exponential post-2019 growth led by China (151 pubs), India (105), and
the USA (65), and mapping five research clusters—neural prediction,
multi-objective optimisation, intelligent control/fault detection, TRNSYS–ANN
hybrids, and deep learning/PV/T—while identifying gaps in real-world validation,
PCM–AI integration, and developing-region deployment.

────────────────────────────────────────

## 2. Problem Being Solved

- SWHS performance is limited by design inefficiencies, dynamic environmental
  conditions, and weak predictive/control capabilities under variable irradiance
  and demand.
- Prior SWHS reviews were largely qualitative and technical; they lacked
  quantitative mapping of global AI research trends, collaboration networks, and
  thematic evolution.
- Regional disparities in AI–SWHS research (especially Africa at ~5.3% of
  output) risk technology designs that ignore local climate, economics, and
  household usage patterns.
- No unified evidence base existed for how ML, ANN, optimisation, and control
  methods collectively improve thermal efficiency, exergy, and lifecycle cost
  across collector types and storage configurations.
────────────────────────────────────────

## 3. Key Contributions

1. Dual methodology: PRISMA data collection + Bibliometrix (R 4.3.2) + VOSviewer
   1.6.20 bibliometrics complemented by qualitative systematic review of AI
   applications in predictive modelling, optimisation, control, and system
   design.
1. Curated corpus: 255 initial Scopus hits (Oct 21, 2024) → 245 English
   peer-reviewed documents after filtering duplicates, non-research types, and
   non-English items.
1. Temporal and geographic mapping: Linear trend \(y = 1.0554x - 2113.6\), \(R^2
   = 0.776\); peak 34 publications in 2024; China/India/USA = ~48% of activity;
   Africa 36 pubs total (5.3%).
1. Five keyword clusters identified via VOSviewer (26 keywords, ≥4 occurrences
   from 650): (1) computational intelligence & optimisation, (2) thermal
   performance & exergy, (3) intelligent control & fault detection, (4)
   TRNSYS–simulation hybrids, (5) deep learning & PV/T integration.
1. Synthesised performance benchmarks from cited primary studies (Table 1): ANN
   \(R^2\) up to 0.9993, DNN+PCM MAPE <15%, MPC/Q-learning control, Random
   Forest MAPE 2.94–5.86%, and documented 12–22% electricity savings in lab
   smart-control demos.
1. Future research agenda explicitly calls for physics-informed AI, long-term
   field validation, demand-aligned control, and equitable deployment in
   solar-rich developing regions—including India and Africa.
────────────────────────────────────────

## 4. Methodology

### 4a. Data Collection (PRISMA)

- Database: Scopus (chosen over WoS/Google Scholar for engineering coverage and
  metadata).
- Search date: October 21, 2024; window 2000–2024.
- Boolean query: Two concept clusters—(A) AI terms (fuzzy logic, ANN, genetic
  algorithms, machine learning, deep learning, reinforcement learning, XGBoost,
  LSTM, PSO, etc.) AND (B) SWHS terms (solar water heater, solar thermal
  collector, solar hot water, etc.).
- Inclusion: research articles, conference papers, reviews, book chapters in
  English with explicit AI integration in thermal water heating.
- Exclusion: notes, errata, editorials; studies without AI or without SWHS
  thermal focus.
### 4b. Bibliometric Analysis

- Bibliometrix: annual trends, country production, keyword analysis, thematic
  mapping, factorial analysis.
- VOSviewer: co-occurrence networks; association strength normalisation;
  attraction 2, repulsion 0; resolution 1; minimum cluster size 1.
### 4c. Qualitative Systematic Review

- Secondary synthesis of peer-reviewed AI–SWHS studies grouped by cluster:
  prediction, optimisation, control, hybrid simulation, deep learning.
- Cross-references 150+ primary studies; Table 1 tabulates AI method, inputs,
  metrics, and outcomes for representative works.
### 4d. Validation

- Bibliometric reproducibility via PRISMA flow (Fig. 3a).
- No new experiments; validation is literature-based consistency checking and
  cross-study metric comparison.
────────────────────────────────────────

## 5. PCM Details (if applicable)

- PCM is not the primary focus of this review paper, but the systematic review
  cites PCM-integrated SWHS studies:
- Uniyal et al.: nine ML models for U-tube PCM solar collectors; ANN and SVR
  \(R^2\) up to 0.9540.
- Tamizharasan et al. [116]: DNN for SWHS with PCM — training accuracy
  0.83888–0.98692, RMSE below 15% threshold, MAPE mainly <5% (vs industry 20–30%
  MAPE benchmark).
- Kanimozhi et al. [121]: ANN for TES with paraffin and honey wax — 265 trials;
  Chi-square 1.5–4.8; MAPE <12%; RMSE 0.65–1.9; time contributes >40% to heat
  improvement in charge/discharge.
- Shirinbakhsh et al. [139]: effect of hot-water demand and PCM integration on
  SDHW performance (cited in references).
- Ramesh et al. [97]: PCM in heat pipe evacuated tube collectors for superior
  storage (cited).
- Review conclusion: PCM + AI remains under-validated in long-term field tests
  relative to simulation-heavy PCM-ML papers.
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

### Algorithms Surveyed (dominant in corpus)

| Category | Methods |
| --- | --- |
| Prediction | ANN, DNN, LSTM, ELM, SVR, Random Forest, XGBoost, LightGBM, Extra Trees, ANFIS, graybox + ALNN |
| Optimisation | GA, PSO, NSGA-II, multi-objective PSO, micro-time variant PSO |
| Control | Fuzzy logic, MPC, Q-learning / reinforcement learning, NNC, AFLC, PID |
| Hybrid | TRNSYS + ANN, physics-informed (called for, rarely implemented) |

### Representative Inputs / States (from synthesised studies)

- Collector area (1.81–4.38 m²), flow rate (0.01–0.015 kg/s optimal in one PSO
  study), inlet/outlet temperatures, ambient temperature, solar radiation, tank
  stratification layers, PCM state, electricity demand, nanofluid concentration
  (e.g. CeO₂/water 0.01%).
### Representative Outputs / Actions

- Heat collection rate, heat loss coefficient, outlet temperature, annual
  energy, solar fraction, pump flow rate, fault class, lifecycle cost.
### Training / Data Scale (examples cited)

- 915 samples (ELM heat collection/loss) [118]; 30 thermosiphon systems ISO
  9459-2 [29]; 36 systems Random Forest training [123]; 608 tank design
  combinations [126]; 265 PCM TES trials [121].
### Performance Metrics Reported

- \(R^2\): 0.776 (publication trend) to 0.9993 (ANN thermosiphon prediction)
- RMSE: e.g. 0.30 (ELM heat collection), 0.67 (ELM heat loss), DNN+PCM <15%
- MAPE: RF 2.94–5.86%; DNN+PCM <5%; fault fusion 89.7–93.7% accuracy
- Efficiency gains: PSO minichannel collector +10–12%; groove absorber XAI +20%;
  nanofluid ANN max 78.2% at 2 L/min
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Bibliometric scope: global; leading countries China, India, USA, Iran, Italy,
  Spain, Germany, Brazil, Mexico, Indonesia, Saudi Arabia, Egypt.
- Climate variables in cited studies: solar radiation, ambient temperature,
  seasonal variation (4-season ANN for heat pipe collectors), clear-sky vs
  cloudy performance (TRNSYS vs ANN crossover at very cloudy conditions).
- Forecast integration mentioned: GraphCast and Pangu-Weather (up to 10-day
  weather prediction) as enablers for adaptive SWHS—not primary data sources in
  this review.
- India relevance: 105 publications (16% of mapped output); cited Indian-linked
  work includes Singh PCM-SWH review [40], Uniyal PCM-ETC ML study,
  Pathak/Chopra HP-ETSC ML studies.
- Temporal resolution in cited works: hourly (TRNSYS–ANN), daily/annual (RF
  replacing ISO 9459-5 tests), long-term thermosiphon campaigns.
- Project-aligned sources not used directly: ERA5, NASA POWER, ISRO Solar
  Calculator, Global Solar Atlas—not referenced in this paper; review is
  bibliometric, not geospatial modelling.
────────────────────────────────────────

## 8. Key Results & Numbers

- 245 final Scopus documents (from 255 initial, 252 after type filter).
- Publication trend: 3 papers in 2000 → 34 in 2024; surge from 2019 (25 papers)
  onward.
- Linear regression slope +1.0554 papers/year; \(R^2 = 0.776\) (77.6% variance
  explained).
- China 151 (~22%), India 105 (16%), USA 65 (10%) — ~48% combined.
- Africa 36 total (5.3%); Egypt 18, South Africa 4, Algeria 5, Morocco 3.
- Global SWH capacity 560 GW_th in 2023 (+18 GW_th YoY); water heater market
  $23.7B (2023) → $32.1B by 2029 at 5.2% CAGR.
- Global energy demand projected +~50% (2020–2050) per EIA citation.
- Levenberg–Marquardt ANN nanofluid collector: correlation >0.98 [68]; Cu-MWCNT
  ANN \(R^2\) up to 0.9989 [69].
- XGBoost/BRT hybrid nanofluid: \(R^2\) 0.9914–0.9997 [70].
- Multi-objective PSO combisystem: 1→10 collectors → lifecycle energy −63%, cost
  +84% [73].
- Enhanced PSO PV/T: pump energy −17.93%, thermal efficiency +7.86% [75].
- DNN + PCM SWHS: training accuracy 0.839–0.987, testing 0.720–0.987, MAPE <15%
  [116].
- Random Forest vs ISO 9459-5 testing: MAPE 2.94–5.86%, \(R^2\) 0.995–0.998 for
  annual energy [123].
- TRNSYS vs ANN (tropical 14-day): TRNSYS MAE 1.5 °C, ANN MAE 1.7 °C; both \(R^2
  > 0.95\) [101].
- 30 thermosiphon systems ANN: training \(R^2 = 0.9993\), validation 0.9913
  [29].
- Fault detection SVM-DS fusion: accuracy 89.7–93.7% vs traditional 77.6–84.7%
  [153].
- Deep RL smart water heater control: 12–22% electricity savings (lab, cited as
  needing field validation) [88].
- VOSviewer keywords: 26 of 650 met ≥4 occurrences threshold.
────────────────────────────────────────

## 9. Baseline Comparison

| Comparison | Baseline | AI / Optimised | Improvement |
| --- | --- | --- | --- |
| Heat transfer correlations vs ANN [28] | Conventional \(R^2\) 0.808–0.522 | ANN \(R^2\) 0.993 train, 0.978 validation | +18.5–45.6 percentage points |
| Plain water vs CeO₂ nanofluid ANN [84] | Plain water baseline | 78.2% max efficiency at 2 L/min | +21.5% thermal efficiency |
| TRNSYS vs ANN outlet temp [101] | TRNSYS MAE 1.5 °C | ANN MAE 1.7 °C | TRNSYS better in variable cloudiness; ANN better very cloudy |
| ANFIS/TRN vs ANN seasonal [163] | ANFIS, thermal resistance network | ANN autumn \(R^2 = 0.989\), VAF 0.99489 | Max divergence 3.56% thermal, 1.52% exergy |
| ISO 9459-5 experimental testing [123] | Standard test campaign | Random Forest predictor | MAPE 2.94–5.86%; reduces test burden |
| PSO collector count [73] | 1 collector | 10 collectors | Energy −63%, cost +84% (trade-off) |
| Industry MAPE benchmark [116] | 20–30% MAPE typical | DNN+PCM <5% MAPE | Substantially below industry norm |
| Traditional fault detection [153] | 77.6–84.7% accuracy | SVM-DS fusion 89.7–93.7% | +5–16 percentage points |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — review paper without original hardware experiment.

Cited embedded/field setups in literature include:

- IoT wireless temperature monitoring for domestic SWHS [146]
- Arduino/Android platforms reducing test time from 15 days to near real-time
  [28]
- Low-cost three-way valve PID passive SWH controller [150]
- Laboratory smart water heaters with DRL for electricity/carbon reduction [88]
- Thermosiphon and flat-plate / ETC / heat-pipe collector test rigs across
  global labs (no single unified prototype in this review).
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Limited experimental validation of AI models; most work is simulation or
  short-duration lab tests [147, 148, 152].
- No standardised public datasets for AI–SWHS benchmarking.
- Physics-informed neural networks underexplored despite TRNSYS–ANN success
  (\(R^2 > 0.93\)).
- Training data requirements not systematically characterised (hundreds to
  hundreds of thousands of samples).
- Africa and Latin America underrepresented in research output vs solar
  potential.
- Multi-objective studies often omit embodied energy, water use, and social
  acceptance.
- Smart grid / demand response integration inadequately addressed.
- Socio-economic, policy, and cost-benefit dimensions scarcely explored.
- Demonstrated 12–22% electricity savings need validation under realistic user
  behaviour and climates.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Highly relevant — Cluster 3 and Table 1
  document fuzzy-MPC [147], Q-learning RL for solar fields [148], DRL smart
  water heaters [88], and demand-aware scheduling; authors state most systems
  remain simulation-only, directly motivating your DRL charge/discharge/bypass
  controller on live sensors.
- RG2 (No integrated PCM–AI–hardware prototype): Highly relevant — Review maps
  PCM+ML (DNN [116], paraffin/honey wax ANN [121], U-tube PCM ML [72]) but notes
  fragmentation between materials, algorithms, and deployment; supports your
  closed-loop RPi/ESP32 + PCM + AI integrated prototype as a gap-filling
  contribution.
- RG3 (Poor alignment with household demand patterns): Relevant — Paper
  emphasises AI for demand prediction, user behaviour, and hot-water scheduling
  [26, 27]; cites Shirinbakhsh PCM+ demand interaction [139] and RF models using
  daily load volume [123]; aligns with your demand-conditioned DRL reward and
  Indian household profiles (Coimbatore, Kochi, Jaisalmer).
- RG4 (Limited real-world experimental validation): Highly relevant — Authors
  explicitly flag short lab studies vs long-term field trials; Terfai-style
  embedded validation is rare in AI–SWHS corpus; strengthens justification for
  your multi-city field/bench evaluation objective.
- RG5 (No predictive optimization under climatic uncertainty): Highly relevant —
  Calls for weather-forecast-driven MPC [104, 147], GraphCast/Pangu-Weather
  integration, and multimodal climate inputs; supports your ERA5/NASA POWER/ISRO
  forecast → PCM selection + DRL pipeline under variable irradiance.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Publication growth trend (bibliometric):

\[

y = 1.0554x - 2113.6,\quad R^2 = 0.776

\]

where \(x\) = year, \(y\) = annual publication count.

Grey relational grade (your project already uses GRA — cross-cite Chen/Singh via
this review’s optimisation cluster):

\[

\xi_i = \frac{\Delta_{\min} + \zeta \Delta_{\max}}{\Delta_i + \zeta
\Delta_{\max}}, \qquad

\gamma_i = \frac{1}{n}\sum_{k=1}^{n}\xi_i(k)

\]

Standard ML error metrics cited across reviewed studies (for your benchmark
table):

\[

RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}, \qquad

MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|

\]

Thermosiphon ANN target (Kalogirou & Panteliou, synthesised in review):

- Predict annual useful solar energy \(Q_u\) from collector area \(A_c\), system
  configuration, and climate class; reported \(R^2 = 0.9993\) training, 0.9913
  validation across 1.81–4.38 m² systems — useful baseline for grey-box vs ANN
  comparison in your simulation environment.
PCM TES ANN feature importance (Kanimozhi et al., via review):

- Time dominates charge/discharge improvement (>40% contribution) — supports
  time-series state features \((T_w, T_p, f, \dot{m}, GHI)\) in your DRL state
  vector.
────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. B. Singh et al., "Application of phase change materials in solar water
   heating systems — A comprehensive review," 2025 — direct PCM-SWH literature
   anchor for India-focused review table.
1. A. Al-Mamun et al., "State-of-the-art in solar water heating (SWH) systems…,"
   Sol. Energy, 2023 — baseline SWH technology review paired with this AI
   review.
1. M. Liu et al., "The contribution of artificial intelligence to phase change
   materials in thermal energy storage…," 2025 — AI+PCM TES
   prediction-to-optimization pipeline.
1. A. Terfai et al., ANN–MPC shallow pond experimental work, 2025 — embedded
   real-time control validation benchmark (cited indirectly via your corpus;
   Odoi cites MPC/ANN control cluster).
1. S. Uniyal et al., ML for U-tube PCM solar collectors — nine-model comparison
   with ANN/SVR \(R^2\) up to 0.9540; closest ML+PCM collector study in review.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I (Introduction): Cite global SWH capacity 560 GW_th (2023), market
  growth $23.7B → $32.1B (CAGR 5.2%), and post-2019 explosion of AI–SWHS
  publications (34 in 2024) to motivate intelligent PCM-SWH research.
- Section II (Literature Review): One-line entry: "Odoi-Yorke (2025)
  bibliometrically maps 245 AI–SWHS studies into five clusters (prediction,
  optimisation, control, TRNSYS-hybrid, deep learning), reporting India as the
  second-largest contributor (105 papers) while noting <6% African output and
  limited field validation of adaptive controllers."
- Section III (Methodology): Borrow PRISMA-style literature screening logic for
  your survey section; adopt reported ML metric suite (\(R^2\), RMSE, MAPE, MAE)
  for controller and forecaster evaluation consistency with SWHS AI literature.
- Section IV (Dataset & Setup): Contrast your ERA5 / NASA POWER / ISRO climate
  pipeline against review’s finding that standardised irradiance datasets are
  missing; position India (16% global AI–SWHS share) as context for
  Coimbatore/Kochi/Jaisalmer case studies.
- Section V (Results): Benchmark against synthesised targets: DNN+PCM MAPE <15%
  [116], RF annual energy MAPE 2.94–5.86% [123], MPC/RL electricity savings
  12–22% [88] (lab), and ANN thermosiphon \(R^2\) 0.9913 validation [29] for
  grey-box and DRL superiority claims.
────────────────────────────────────────

# 34. Singh2025PCM_SWH_ComprehensiveReview_summary.md

Source path: /mnt/data/Singh2025PCM_SWH_ComprehensiveReview_summary.md

# Application of Phase Change Materials in Solar Water Heating Systems — A Comprehensive Review

Authors: Brihaspati Singh, Ravi Shankar Rai, Pankaj Yadav, Sambhrant Srivastava,
Chandrmani Yadav

Year: 2025

Journal/Conference: Solar Energy Materials and Solar Cells, Vol. 293, Article
113888

DOI/Link: https://doi.org/10.1016/j.solmat.2025.113888

IEEE Citation: B. Singh et al., "Application of phase change materials in solar
water heating systems — A comprehensive review," Sol. Energy Mater. Sol. Cells,
vol. 293, p. 113888, 2025, doi: 10.1016/j.solmat.2025.113888.

────────────────────────────────────────

## 1. One-Line Summary

This India-led comprehensive review synthesizes PCM fundamentals, SWH
integration strategies (collector, tank, riser, pipe),
nano/foam/fin/encapsulation enhancements, and Table 5 performance
benchmarks—reporting up to 65% FPSWH daily efficiency, 66% dual-PCM gain, 49.9%
heat-pipe ETC improvement, and 40–70 °C optimal PCM \(T_m\)—while identifying
low conductivity, supercooling, cost, and lack of long-term field validation as
barriers to scalable PCM-SWH deployment.

────────────────────────────────────────

## 2. Problem Being Solved

- SWH output is intermittent because solar radiation is available only during
  daylight; conventional sensible storage yields temperature swing and limited
  overnight delivery.
- Organic PCMs offer high latent heat per volume but suffer low thermal
  conductivity, leakage, supercooling, phase segregation (inorganics), and
  encapsulation complexity.
- Literature on PCM in SWH is fragmented across materials science, collector
  geometry, and nano-enhancement—without a unified India-relevant synthesis of
  selection criteria, integration locations, and quantified performance across
  FPSWH/ETCSWH configurations.
- Economic feasibility, cyclic thermal stability under repeated melt/freeze, and
  scalability of NePCM/foam composites remain poorly standardized for domestic
  SWH markets.
────────────────────────────────────────

## 3. Key Contributions

1. PCM taxonomy and SWH selection framework — classifies organic, inorganic,
   eutectic, hybrid PCMs; defines desirable properties (Table 1) and tabulates
   thermal properties (Table 2) for paraffins, fatty acids, salt hydrates, and
   eutectic blends.
1. Explicit PCM selection priority order for SWH: (a) latent heat → (b) thermal
   conductivity → (c) melting point → (d) specific heat → (e) density.
1. Integration map — PCM placement in collector, storage tank, riser tubes,
   insulated pipework; cascaded/dual-PCM and encapsulated configurations.
1. Enhancement survey — nano-composites (CuO, SiC, MWCNT, Ag), expanded
   graphite, metal/aluminum/copper foam, boron nitride, fins (helical,
   tree-shaped, branch), modified ETC/U-tube/manifold designs.
1. Comparative performance compilation — Tables 4–8 and Section 8 synthesize
   experimental/numerical outcomes: efficiency, exergy, charging/discharging
   time reduction, overnight temperature retention, and cost ($/L where
   reported).
1. Technical challenges & future scopes — supercooling, corrosion, nanoparticle
   agglomeration, fire safety, cost-effective PCM manufacturing, and call for
   standardized testing and real-world performance evaluation.
────────────────────────────────────────

## 4. Methodology

- Type: Narrative comprehensive review (not bibliometric); Web of Science
  publication trend analysis for “PCM in solar water heating” (Fig. 2a–d)
  showing rising global publication count and multi-country participation
  including India.
- Scope: PCM as latent TES in low-temperature SWH (<100 °C); covers FPSWH,
  ETCSWH, heat-pipe ETC, parabolic trough, domestic and centralized systems.
- Structure: Sections 2–3 PCM science & integration; 3.1.x enhancement
  subsections; 5.x fin/ETC/hybrid-nano designs; 6 melting/solidification; 7
  challenges; 8–9 comparative tables; 10 conclusions; 11 future scopes.
- Validation approach: Cross-comparison of 200+ cited primary studies; Tables
  4–8 aggregate reported metrics; no original experiments in this paper.
- Economic analysis: Section 9 and Table 8 compile daily efficiency and
  freshwater/production cost for PCM-augmented solar stills and hybrid systems.
────────────────────────────────────────

## 5. PCM Details (if applicable)

### 5a. PCM Selection Criteria (Authors’ Ordered List)

1. Latent heat of the material (highest priority)
1. Thermal conductivity
1. Melting point
1. Specific heat capacity
1. Density
Rule stated: higher conductivity + higher latent heat + lower melting point (for
faster charge/discharge) preferred; minimize void/contact resistance between PCM
and absorber.

### 5b. Key PCM Materials & Properties (Table 2 excerpts)

| PCM | Type | \(T_m\) (°C) | Latent heat (kJ/kg) | Notes |
| --- | --- | --- | --- | --- |
| Paraffin wax | Organic | 64 | 173.6 | ρ solid/liquid 916/790 kg/m³ |
| C₂₂H₄₆ | Organic | 43–46 | 249 | Long-chain paraffin |
| C₃₀H₆₂ | Organic | 64–67 | 252 |  |
| Lauric acid (C₁₂H₂₄O₂) | Organic | 42–46 | 171 | SWH-suitable |
| Myristic acid | Organic | 52–54 | 190 | Used in modified ETC manifold |
| Palmitic acid | Organic | 62–64 | 185.4 |  |
| Stearic acid | Organic | — | — | Mixed with paraffin + graphite |
| Polyethylene glycol | Organic | 4.2–60 | 117.6 | Tunable range |
| CaCl₂·6H₂O | Inorganic | 27 | 168.6 | Salt hydrate |
| Lauric–Myristic (66/34) | Eutectic | 34.2 | 166.8 | Tunable \(T_m\) |
| Lauric–Palmitic (69/31) | Eutectic | 35.2 | 166.3 |  |
| Myristic–Palmitic (58/42) | Eutectic | 42.6 | 169.7 | Near Coimbatore/Kochi target band |
| Palmitic–Stearic (64.2/35.8) | Eutectic | 52.3 | 181.7 | Near arid/high-temp band |

### 5c. Optimal SWH Operating Band

- Solid–liquid organic PCM phase-change temperature: 40–70 °C [Table 5, ref.
  178] — directly aligns with Rubitherm RT35–RT64HC and PLUSS OM35–OM50
  screening in your project.
### 5d. Enhancement Performance (Selected)

- Paraffin base \(k\) 0.24 W/m·K → hybrid MWCNT + SiO₂ nano-PCM 0.47719 W/m·K;
  collector efficiency 71.7% [145].
- Expanded graphite + A70 PCM: \(k\) 1.59 W/(m·K) (+657.16% vs base); stable
  through 500 thermal cycles [153].
- Salt hydrate + 0.6 wt% MWCNT: \(k\) +91.45%; stable 300 cycles [154].
- Binary graphene + MWCNT in salt hydrate: \(k\) +160% to 1.2 W/m·K [155].
- HP-ETC + PCM + Cu porous metal: max energy efficiency 85.64% vs conventional
  36.91% [98].
- Wax + graphite: 60% reduction in charge/discharge duration [189].
- Fin + metal foam cylindrical tube: 61.6% / 82% reduction in storage/release
  times [202].
- U-shaped HX with paraffin RT-35: 91.79% charging time reduction [201].
────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

N/A — this review does not cover AI, ML, DRL, or predictive control. It focuses
on materials, geometry, and passive/active thermal enhancements.

Indirect link to your project: Authors call for "smart PCM systems that can
change behavior on their own (e.g., smart fins) to account for changes in solar
input" (Section 6) and standardized long-term field evaluation—gaps your DRL
controller and climate-adaptive PCM selector address. Cross-cite Odoi-Yorke
(2025) and Liu et al. (2025) AI–PCM papers for the intelligence layer.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

N/A — no dedicated climate dataset study. Geographic context is qualitative via
cited experiments worldwide (India, Iraq, Algeria, Turkey, etc.).

Implicit climate variables in cited SWH tests: solar radiation intensity,
ambient temperature, seasonal winter/summer efficiency splits (e.g., sodium
acetate 26% winter vs 32% summer).

India relevance: Authors affiliated with Rajkiya Engineering College, Azamgarh
(UP) and Marwadi University, Rajkot (Gujarat); WoS trend figures show Indian
research participation in PCM-SWH field.

Your project mapping: Use Singh’s 40–70 °C PCM band and eutectic blends to seed
classifier labels for Coimbatore (~42–46 °C fatty-acid/eutectic), Kochi
(humidity-corrosion favors organic paraffin), Jaisalmer (higher \(T_m\) /
dual-PCM strategies).

────────────────────────────────────────

## 8. Key Results & Numbers

- Earth intercepts ~1.8 × 10¹¹ MW solar power [intro].
- Nano-SiC in paraffin (3 wt%): thermal conductivity +18.2%; exit air 64.4 °C; 3
  h post-sunset operation; melting/solidification points −5% / −5.2% [Jawad].
- SWCNT/paraffin: conductivity +12%; stored energy +20.7% (natural) / +21.2%
  (forced convection) [Habib].
- Finned shell-tube: melting time −58% when Re 1000 → 2000 [Paria].
- FPSWH + paraffin: max daily efficiency 65% [173].
- FPSWH + sodium acetate: winter 26%, summer 32% [174].
- SiC + CuO nano-paraffin: conductivity +22.53% [175].
- ETC + paraffin + nano-CaO₂: exergy +0.44%, energy +10.89% vs no-PCM [176].
- Paraffin + 10 g CuO (20 kg wax): max outlet 80.6 °C [177].
- Dual-PCM (tritriacontane + erythritol): efficiency +66% vs no-PCM [181].
- Centralized SWH + paraffin/expanded graphite steel ball: 60 °C \(T_m\) PCM
  outperforms 55 °C [183].
- Paraffin in Al container: tank water ≥30 °C above ambient for 24 h [185].
- NCPCM (1.0% SiO₂) ETC exergy: 19.6% (no PCM) → 22.0% (PCM) → 24.6% (NCPCM)
  [186].
- Heat-pipe ETCSWH + paraffin: efficiency +49.9% [143].
- Myristic acid SWH: stabilized 51–52 °C overnight [188].
- Expanded graphite PCM: thermal conductivity +162.4% vs pure alloy [191].
- Modified riser + fin + PCM: heat transfer +177.7%, efficiency +39.5% [142].
- Modified ETC + myristic acid: 20–30% more effective heat retention vs
  unmodified [145 text].
- Branch fin triplex-tube: melting/solidification time −67.7% / −74.8%; full
  cycle −84.5% vs no fins [150].
- Intelligent memory metal fin: total melting duration −28.6% vs straight fin
  [152].
- 90% metal-foam-filled TES: complete melting time 5310 s, 87.56% lower than
  pure PCM tank [83].
- Al fin stack in PCM: effective conductivity up to 42× (Al fins) vs base PCM
  [82].
- Conventional solar still efficiency 34–50% vs PCM/hybrid modified stills up to
  80–85.5% (Table 8 contexts).
────────────────────────────────────────

## 9. Baseline Comparison

| System / Study | Baseline | PCM / Enhanced | Improvement |
| --- | --- | --- | --- |
| FPSWH daily efficiency | Conventional sensible storage | Paraffin wax PCM | Up to 65% daily efficiency [173] |
| ETCSWH dual-PCM | Without PCM | Tritriacontane + erythritol | +66% efficiency [181] |
| Heat-pipe ETCSWH | Without PCM | Paraffin wax | +49.9% efficiency [143] |
| HP-ETC + Cu porous | Conventional ETC (36.91%) | PCM + Cu porous (85.64%) | +48.73 percentage points max daily energy efficiency [98] |
| Overnight tank temperature | Ambient-only reference | Paraffin in Al container | ≥30 °C above ambient for 24 h [185] |
| NCPCM exergy (ETC) | No PCM 19.6% | PCM 22.0%; NCPCM 24.6% | Up to +5.0 percentage points exergy [186] |
| Nano-paraffin SWH | Base paraffin | 1.0 wt% nano-Cu paraffin | +8.4% efficiency [190] |
| Charging duration | Pure paraffin | Wax + graphite | −60% charge/discharge time [189] |
| Melting time | Pure A70 PCM | A70 + 20 wt% EG | Conductivity +657.16%; 500-cycle stability [153] |
| Solar still freshwater | Conventional still 34% | Tray still with mirrors 50% | +16 percentage points; cost $0.028 → $0.021/L [203] |
| Branch fins vs none | No fins | Branch fin TTHX | Melting −67.7%, solidification −74.8% [150] |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

Review-level — setups synthesized from cited works:

| Component | Examples from review |
| --- | --- |
| Collectors | Flat-plate (FPSWH/FPCSWH), evacuated tube (ETCSWH), heat-pipe ETC, compound parabolic, parabolic trough |
| PCM containment | Aluminum containers, steel balls with expanded graphite, encapsulated spheres/cylinders, shell-and-tube, U-tube manifold |
| Enhancers | Longitudinal/helical/tree fins, metal foam (Cu, Al, Ni), nanoparticles (CuO, SiO₂, MWCNT, Ag), expanded graphite |
| Fluids | Water, nanofluids (CeO₂/water, CuO), air (solar air heater cross-cites) |
| Sensing implied | Temperature monitoring in cited experiments; no unified DAQ platform specified |
| Embedded platforms | N/A in this review |
| Test conditions | Lab and outdoor campaigns; 24 h overnight retention tests; seasonal winter/summer splits; limited multi-year field data acknowledged |

India-relevant hardware cite: Chopra et al. myristic acid manifold ETC [144] and
Pathak/Tyagi HP-ETSC lines appear in references—overlap with your literature
map.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Low thermal conductivity of organic PCMs limits charge/discharge rates [157].
- Long-term cyclic instability of PCM/NePCM under repeated melt/freeze [Section
  7].
- Supercooling and phase segregation (especially salt hydrates) reduce effective
  storage [160].
- Volume expansion stresses containment; corrosion with some inorganics [159].
- Toxicity/flammability concerns for certain organics in domestic use.
- Nanoparticle agglomeration, increased viscosity, cost, and uncertain
  environmental fate [162].
- No standardized PCM performance evaluation methodology across studies [Section
  6].
- Fin/foam/nano designs often lack long-term operational stability, cost
  trade-offs, and scalability analysis [Sections 5.1–5.2].
- Limited real-world field validation; most evidence from lab/short campaigns
  [Section 8].
- AI/smart adaptive control not addressed—passive or fixed geometry dominates.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Relevant indirectly — Review documents
  fixed PCM configurations, flow-rate recommendations (high vs low flow for
  charging), and calls for smart PCM behavior under varying solar input; does
  not implement closed-loop control. Justifies your DRL valve/pump policy over
  static PCM integration.
- RG2 (No integrated PCM–AI–hardware prototype): Highly relevant (materials
  layer) — Provides integration locations (collector/tank/riser/pipe),
  enhancement options, and Table 5 benchmarks for PCM-SWH hardware design;
  intelligence layer absent—your integrated stack fills the cited fragmentation.
- RG3 (Poor alignment with household demand patterns): Relevant — Myristic acid
  overnight 51–52 °C delivery [188], centralized 60 °C vs 55 °C \(T_m\)
  trade-off [183], and hot-water demand effects (cites Shirinbakhsh [139] in
  references) support demand-aware PCM \(T_m\) and discharge scheduling in your
  reward function.
- RG4 (Limited real-world experimental validation): Highly relevant — Authors
  explicitly demand more real-world evaluations and standardized testing; your
  multi-city embedded prototype responds directly to this stated gap.
- RG5 (No predictive optimization under climatic uncertainty): Partially
  relevant — Seasonal efficiency swings (26–32%) and climate-specific PSO
  studies cited (Yaman & Arslan Turkey) show climate matters for PCM-SWH but no
  forecast-driven optimization; supports your climate-adaptive PCM
  classification (Objective 1) before DRL deployment.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Latent heat storage (conceptual, Section 2):

\[

Q_{\text{latent}} = m \cdot L

\]

where \(m\) = PCM mass, \(L\) = latent heat of fusion (kJ/kg from Table 2).

Sensible + latent total energy (charging cycle, Fig. 4b):

\[

Q_{\text{total}} = m C_p (T_f - T_i) + m L

\]

Effective thermal conductivity with fin enhancement (qualitative from [82]):

- Planar Al fin stacks reported up to 42× effective conductivity increase vs
  unfinned PCM—use as upper bound in grey-box \(hA\) or effective \(k_{eff}\)
  sensitivity analysis.
Melting time scaling (fin studies, Section 6):

- Branch fins: \(t_{\text{melt,fin}} = (1 - 0.677)\, t_{\text{melt,0}}\) (67.7%
  reduction) — calibrate enthalpy-porosity or lumped PCM model fin enhancement
  factor.
Energy efficiency (daily SWH, Section 9):

\[

\eta_{\text{daily}} = \frac{Q_{\text{useful}}}{Q_{\text{solar,in}}}

\]

Benchmark: paraffin FPSWH \(\eta_{\text{daily,max}} = 65\%\) [173]; heat-pipe
ETC improvement \(+49.9\%\) relative to no-PCM baseline [143].

Grey-box melt dynamics (align with your Presentation):

\[

M_p L \frac{df}{dt} = hA(T_w - T_m)

\]

Use Singh’s priority-ordered PCM properties \((L, k, T_m)\) as classifier
features and simulation parameters.

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. A. Al-Mamun et al., "State-of-the-art in solar water heating (SWH) systems…,"
   Sol. Energy, 2023 — complementary SWH technology baseline.
1. F. Odoi-Yorke, "Artificial intelligence for solar water heating systems…,"
   2025 — AI layer this review omits.
1. K. Chopra et al., myristic acid manifold ETC / HP-ETSC experimental lines,
   2023–2025 — India-relevant PCM-ETC hardware.
1. S. Uniyal et al., evacuated tube + PCM + nanofluid review/experiments —
   ETC-PCM configuration family.
1. M. Shirinbakhsh et al., hot-water demand + PCM integration on SDHW, Sol.
   Energy, 2018 — demand–PCM interaction for RG3.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I (Introduction): Cite SWH intermittency problem and PCM latent
  storage advantage; quote 65% max FPSWH daily efficiency and ≥30 °C above
  ambient for 24 h retention as published PCM-SWH benefits.
- Section II (Literature Review): Use Singh’s 5-level PCM selection priority
  (latent heat → conductivity → \(T_m\) → \(C_p\) → density) as the theoretical
  basis for your XGBoost/RF feature ranking; include Table 2 eutectic blends for
  climate-tunable \(T_m\).
- Section III (Methodology): Adopt 40–70 °C optimal PCM band and integration
  locations (collector vs tank) to justify your grey-box topology; use fin/foam
  enhancement factors as sensitivity bounds—not primary design unless prototype
  includes fins.
- Section IV (Dataset & Setup): Seed PCM property database with Table 2 values
  for RT35–RT64HC and OM35–OM50 validation; map Rubitherm/PLUSS candidates to
  fatty-acid/eutectic benchmarks (lauric–myristic 34.2 °C, 166.8 kJ/kg).
- Section V (Results): Compare controller outcomes against Singh benchmarks: 66%
  dual-PCM gain, 49.9% HP-ETC improvement, 51–52 °C overnight myristic acid,
  80.6 °C peak outlet, and 60% faster charge/discharge with graphite—demonstrate
  AI-controlled system meets or exceeds passive PCM integration limits.
────────────────────────────────────────

# 35. Terfai2025SSP_ANN_MPC_Experimental_summary.md

Source path: /mnt/data/Terfai2025SSP_ANN_MPC_Experimental_summary.md

# Experimental Validation and Enhanced Thermal Prediction of a Shallow Solar Pond Using Artificial Neural Network–Based Model Predictive Control for Real-Time Optimization Under Multiple Heat Extraction Modes

Authors: Abdelkrim Terfai, Younes Chiba, Mounir Zirari, Mohamed Najib Bouaziz

Year: 2025

Journal/Conference: Unconventional Resources, Vol. 8, Article 100240

DOI: https://doi.org/10.1016/j.uncres.2025.100240

IEEE Citation: A. Terfai et al., "Experimental validation and enhanced thermal
prediction of a shallow solar pond using artificial neural network–based model
predictive control for real-time optimization under multiple heat extraction
modes," Unconv. Resour., vol. 8, p. 100240, 2025, doi:
10.1016/j.uncres.2025.100240.

────────────────────────────────────────

## 1. One-Line Summary

This paper experimentally compares direct, open-cycle, and closed-cycle heat
extraction from a custom shallow solar pond in Algeria, trains a
Bayesian-regularized ANN (\(R^2 = 0.99919\)) on Arduino/DS18B20 data, and
integrates it with MPC on a QR30E pump to cut outlet-temperature tracking error
by 52.2% (MAE 1.42 °C vs 2.97 °C).

────────────────────────────────────────

## 2. Problem Being Solved

- Shallow solar ponds (SSPs) can store solar heat for water heating and
  industrial use, but thermal performance depends strongly on heat extraction
  mode (direct drain, open circulation, closed loop with storage), which is
  rarely compared under identical clear-sky conditions.
- Nonlinear, transient SSP dynamics (double glazing, shallow water mass, heat
  exchanger coupling) are difficult to capture with fixed-parameter analytical
  models alone.
- Real-time regulation of fluid flow under variable solar irradiance is needed
  to stabilize outlet temperature and reduce pump energy use—beyond static ANN
  prediction without control.
- Lack of an integrated experimental + data-driven model + predictive control
  pipeline validated on instrumented hardware for closed-cycle SSP operation
  (identified as the most stable mode).
────────────────────────────────────────

## 3. Key Contributions

1. Side-by-side experimental campaign (August 15–17, clear sky, Tablat/Medea,
   Algeria) of direct, open-cycle, and closed-cycle extraction on one 1 m² SSP
   (~60 L, double-glazed, insulated, PVC serpentine exchanger).
1. Systematic evaluation of 14 ANN configurations; optimal model: trainbr, 2
   hidden layers, 15 neurons — \(R^2 = 0.99919\), RMSE 32.7580%, MRE 0.62%
   across \(T_{C1}\), \(T_{C2}\), \(T_{wp}\), \(T_p\), \(T_{fo}\), \(T_{wt}\).
1. Demonstration that closed-cycle mode achieves highest/stablest pond and
   outlet temperatures with minimal convective/evaporative losses vs open and
   direct modes.
1. Hybrid ANN–MPC framework adjusting mass flow rate \(\dot{m}\) on QR30E pump
   in closed cycle; MPC correction Eq. (5) with optimized \(\alpha\), \(\beta\).
1. Quantified control improvement: outlet MAE 1.42 °C (MPC-corrected) vs 2.97 °C
   (ANN-only), >50% error reduction; peak \(T_{fo}\) 52.1 °C (MPC) vs 49.3 °C
   (ANN).
────────────────────────────────────────

## 4. Methodology

### 4a. System / Experiment Setup

- SSP geometry: Galvanized sheet pond 0.76 × 1.30 m, depth 0.06 m, capacity ~60
  L; black bottom; 0.04 m polystyrene insulation (\(k \approx 0.03\) W/m·K).
- Glazing: Two glass panels (0.003 m each), 0.03 m air gap (\(\tau_g = 0.90\),
  \(\varepsilon_g = 0.90\), \(\alpha_g = 0.05\)).
- Heat extraction: Transparent PVC tube exchanger (10 m length, 0.008 m ID,
  0.001 m wall, \(\lambda_{PVC} \approx 0.19\) W/m·K); QR30E brushless pump (max
  240 L/h); 20 L insulated storage tank (closed cycle).
- Working fluid: Tap water (\(k = 0.6\) W/m·K, \(C_p = 4180\) J/kg·K, \(\rho =
  1000\) kg/m³).
- Scenarios (one per day): (1) Direct — drain heated pond water (Aug 15); (2)
  Open cycle — in-pond exchanger, continuous flow (Aug 16); (3) Closed cycle —
  sealed loop pond ↔ tank with second tank exchanger (Aug 17).
- DAQ: Arduino UNO; DHT22 (\(T_a\), RH); seven DS18B20 probes at glass
  (upper/lower), absorber, pond water, HX inlet/outlet, tank; 1 min logging
  07:00–19:00 (~720 points/day, ~2160 total).
- Solar input: Bird & Hulstrom clear-sky model for Tablat, Medea, Algeria
  coordinates.
### 4b. Mathematical Models & Equations

ANN performance metrics:

- \(SSE = \sum_{i=1}^{n}(e_i - p_i)^2\) — (1)
(\(e_i\) experimental, \(p_i\) predicted, \(n\) samples)

- \(RMSE\ (\%) = \sqrt{SSE/n} \times 100\) — (2)
- \(R^2 = 1 - SSE / \sum_{i=1}^{n} p_i^2\) — (3)
- \(MRE\ (\%) = \frac{1}{n}\sum_{i=1}^{n} \left| \frac{e_i - p_i}{e_i} \right|
  \times 100\) — (4)
MPC temperature correction (closed cycle):

- \(T_{fo,corr} = T_{fo,pred} + \alpha \cdot \tanh\left(\beta(\dot{m} -
  0.01)\right)\) — (5)
(\(\dot{m}\) in kg/s; 0.01 ≈ minimum QR30E operational flow; \(\alpha\) = gain,
\(\beta\) = sensitivity)

Offline identification of \(\alpha\), \(\beta\): minimize MSE between
\(T_{fo,set}\) and \(T_{fo,pred}\) using experimental \(\dot{m}\) profile and
fminsearch — reported MSE 0.98907 after tuning.

No enthalpy-porosity or PCM phase-change equations—SSP stores sensible heat in
water.

### 4c. Algorithm / Control Method Steps

ANN training:

1. Preprocess inputs: normalize \(T_a\), Hum, \(I_T\), Time, \(A\), depth,
   insulation thickness, wind; add mode-specific \(T_{fi}\), \(\dot{m}\) for
   open/closed cycles.
1. Remove outliers via Z-score (threshold ±3σ); none excluded.
1. Split data 70% train / 15% validation / 15% test (~2160 points total).
1. Train 14 MLP variants: algorithms trainbr, trainlm, trainbfg, trainscg; 1–3
   hidden layers; 5–20 neurons per layer.
1. Select model with lowest RMSE, highest \(R^2\) → trainbr, 2×15 hidden
   neurons.
ANN–MPC real-time loop (closed cycle, §7):

1. FNN predicts \(T_{fo,pred}\) from current inputs and history.
1. MPC simulates candidate \(\dot{m}\) values; applies Eq. (5) to estimate
   \(T_{fo,corr}\).
1. Choose \(\dot{m}\) minimizing \(|T_{fo,corr} - T_{fo,set}|\) while respecting
   pump flow limits.
1. Apply adjusted \(\dot{m}\) to QR30E; repeat each control step under
   measured/estimated \(I_T\).
1. Compare MPC-controlled vs ANN-training \(\dot{m}\) profiles and outlet
   temperatures (Figs. 17–18).
Identified MPC parameters: \(\alpha = -22.93619\), \(\beta = 33.23245\).

### 4d. Data Sources & Dataset Details

| Source | Variables | Resolution | Scope | Period / size |
| --- | --- | --- | --- | --- |
| On-site SSP experiment | \(T_{C1}, T_{C2}, T_{wp}, T_p, T_{fi}, T_{fo}, T_{wt}, T_a\), RH | 1 min | Tablat, Medea, Algeria | Aug 15–17 (clear sky), 07:00–19:00 each day |
| Bird & Hulstrom [22] | Solar radiation \(I_T\) | Modeled clear-sky | Same coordinates | Training/reference days; Nov 17 reduced-irradiance MPC test |
| ANN dataset | All above + geometry (\(A\), depth, insulation), wind | 1 min | Three extraction modes | ~2160 points; 70/15/15 split |

### 4e. Validation Method

- Experimental cross-mode comparison under matched clear-sky days (Aug 15–17).
- ANN vs measured temperatures: training scatter \(R^2 = 0.99919\); mode-wise
  MRE on \(T_{C2}\) 0.84–0.91%; on \(T_{C1}\) 0.42–0.48%; on \(T_{wp}\)
  0.46–0.54%; on \(T_p\) 0.42–0.52%; on \(T_{fo}\) 0.63–0.66%; on \(T_{wt}\)
  0.68% (closed cycle).
- Sensor uncertainty (Table 2): DHT22 ±0.5 °C, ±2% RH; DS18B20 ±0.5 °C (−10 to
  +85 °C operating range cited).
- MPC validation: MAE vs \(T_{fo,set}\) — 1.42 °C (MPC) vs 2.97 °C (ANN-only),
  52.2% reduction; parameter fit MSE 0.98907 for \(\alpha,\beta\) optimization
  (Fig. 15).
- No CFD benchmark; future work cites need for hardware-in-the-loop validation
  beyond current bench setup.
────────────────────────────────────────

## 5. PCM Details (if applicable)

N/A — this paper does not study phase change materials. The shallow solar pond
stores sensible heat in water (~60 L); thermal buffering in closed cycle uses an
insulated water storage tank, not PCM (Rubitherm/PLUSS-type latent storage is
out of scope).

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

- Algorithm: MLP ANN (Bayesian Regularization trainbr best); Model Predictive
  Control (MPC) with ANN as plant predictor; offline fminsearch for
  \(\alpha,\beta\) in Eq. (5).
- Input features / state space: \(T_a\), Hum, \(I_T\), Time, SSP area \(A\),
  depth, insulation thickness, wind speed; plus \(T_{fi}\), \(\dot{m}\) for
  open/closed modes.
- Output / action space: Outputs: \(T_{C1}, T_{C2}, T_{wp}, T_p, T_{fo},
  T_{wt}\). Control action: mass flow rate \(\dot{m}\) (kg/s), MPC-adjusted
  around ~9×10⁻³ kg/s morning baseline.
- Model architecture: 2 hidden layers, 15 neurons each; feedforward MLP (Fig.
  5).
- Hyperparameters: trainbr selected over trainlm, trainbfg, trainscg; best epoch
  870 (Table 3 row: 2 layers, 15 neurons); data split 70/15/15; Z-score outlier
  threshold ±3σ (no removals).
- Training data size: ~2160 samples (3 days × 12 h × 60 min).
- Hardware used for training: N/A — MATLAB ANN toolbox implied; acquisition on
  Arduino UNO.
- Performance metrics: \(R^2 = 0.99919\); RMSE 32.7580%; MRE 0.62% (best ANN);
  MPC outlet MAE 1.42 °C vs 2.97 °C without MPC (52.2% lower).
────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

- Data sources: Bird & Hulstrom simplified clear-sky model [22] for direct and
  diffuse insolation on horizontal surface; measured \(T_a\), RH via DHT22;
  irradiance cross-checked against experimental thermal response (peak 909.82
  W/m² at 12:00 h on test day in Fig. 13).
- Variables used: \(I_T\) (W/m²), \(T_a\), Hum, wind speed \(W_{speed}\), time
  of day.
- Geographic scope: Tablat, Medea, Algeria (University of Medea experimental
  site).
- Temporal resolution: 1 min measurements; 07:00–19:00 daily window.
- Time period covered: Primary campaign August 15–17 (clear sky); additional MPC
  comparison under reduced irradiance (November 17, dips below 500 W/m² between
  09:30–16:00) vs clear-sky reference from August 17 (~950 W/m² peak in setpoint
  curve).
- Clear-sky index / derived metrics: Clear-sky model used for \(I_T\)
  estimation; explicit clearness index \(k_t\) not reported.
────────────────────────────────────────

## 8. Key Results & Numbers

- Optimal ANN: trainbr, 2 hidden layers, 15 neurons — \(R^2 = 0.99919\), RMSE
  32.7580%, MRE 0.62%, SSE 463.5746, training time 93 s, best epoch 870 (Table
  3).
- 14 ANN variants tested; worst trainscg (3 layers, 20 neurons): RMSE 58.2566%,
  MRE 1.05%, \(R^2 = 0.9998\).
- Solar peak (Fig. 13): 909.82 W/m² at 12:00 h; thermal lag between radiation
  peak and fluid temperature peak due to water inertia.
- Upper glass \(T_{C2}\) peaks: Open 48.63 °C (13:00–14:00); closed 47.94 °C;
  direct 42.63 °C (noon) — direct mode lowest peak.
- Lower glass \(T_{C1}\) peaks: Open 58.38 °C; closed 63.00 °C (~15:20); direct
  66.87 °C at 16:00 h.
- Pond water \(T_{wp}\) peaks: Open plateau 53.94 °C (13:00–15:00); closed 62.88
  °C (~16:00); direct 66.44 °C at 16:00 h.
- Absorber \(T_p\) peaks: Open 53.94 °C; closed 62.02 °C; direct 66.36 °C at
  16:00 h.
- Outlet fluid \(T_{fo}\): Open stabilizes ~51.5 °C (13:00–16:00); closed peak
  58.88 °C at 16:00 h; max ANN deviation ~0.29 °C (open), MRE 0.63–0.66%.
- Storage tank \(T_{wt}\) (closed): 25 °C → max 53.56 °C ~17:00 h; MRE 0.68%.
- MPC vs ANN-only outlet: max 52.1 °C vs 49.3 °C (+2.8 °C); min 24.9 °C vs 24.0
  °C; MAE 1.42 °C vs 2.97 °C (52.2% error reduction).
- MPC \(\dot{m}\): starts ~9×10⁻³ kg/s, lowers at midday, increases late
  afternoon (Fig. 17).
- MPC tuning: \(\alpha = -22.93619\), \(\beta = 33.23245\); correction fit MSE
  0.98907.
────────────────────────────────────────

## 9. Baseline Comparison

- Baseline method(s): Direct and open-cycle heat extraction vs closed-cycle; ANN
  prediction without MPC vs ANN–MPC for outlet tracking; 14 ANN training
  algorithms/architectures with non-optimal models as ML baselines.
- Proposed method: Closed-cycle SSP + trainbr ANN (2×15) + MPC flow-rate
  optimization (Eq. (5)).
- Improvement margin: Closed cycle — higher \(T_{wp}\), \(T_{fo}\), \(T_{wt}\)
  and longer evening retention vs open/direct (qualitative + peak deltas up to
  ~12 °C on \(T_{wp}\) vs open); MPC — 52.2% lower MAE, +2.8 °C higher peak
  \(T_{fo}\) vs ANN-only under same reduced-irradiance day.
- Conditions of comparison: Same SSP hardware; clear-sky days for mode
  comparison (Aug 15–17); MPC test uses Nov 17 low-irradiance day vs Aug 17
  clear-sky setpoint profile.
────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

- Physical components: Custom SSP (0.76×1.30×0.06 m, ~60 L); double glass cover;
  polystyrene insulation; black galvanized absorber (\(\alpha_p = 0.95\)); PVC
  serpentine HX (10 m, 8 mm ID); 20 L insulated tank; QR30E brushless
  circulation pump.
- Sensor specs: DHT22 — \(T_a\): −40 to +80 °C, ±0.5 °C; RH: 0–100%, ±2%;
  DS18B20 (×7 waterproof) — −55 to +125 °C spec, ±0.5 °C (−10 to +85 °C in
  table); 1 min logging.
- Embedded/compute platform: Arduino UNO for DAQ; ANN/MPC in MATLAB (implied for
  training and MPC design).
- Test environment: Outdoor experimental setup, Tablat, Medea, Algeria;
  clear-sky conditions Aug 15–17.
- Test duration: 12 h/day (07:00–19:00) × 3 days for mode comparison; additional
  Nov 17 MPC validation under variable/cloud-affected irradiance.
────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Future work will extend the strategy to multi-objective optimization and
  real-world hardware-in-the-loop validation — implying current MPC integration
  is not yet fully validated on embedded HIL hardware.
- Heat exchanger effectiveness \(\varepsilon\) was not directly measured;
  performance inferred from material properties and temperature rise only.
- The three extraction modes were run on different consecutive days (not
  simultaneous), though authors state clear-sky consistency across Aug 15–17.
- MPC comparison (Fig. 18) uses November 17 reduced irradiance against an August
  17 clear-sky setpoint — a deliberate stress test but not identical weather
  ensembles.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1 (No real-time adaptive control): Relevant. ANN–MPC continuously adjusts
  \(\dot{m}\) to track \(T_{fo,set}\), cutting MAE by 52.2% vs open-loop
  ANN—direct precedent for your PPO/DDPG pump/valve control, though on SSP water
  rather than PCM latent storage.
- RG2 (No integrated PCM–AI–hardware prototype): Partially relevant.
  Demonstrates Arduino + DS18B20 + pump experimental loop with ML control—the
  same sensor/actuator class as your FYP—but no PCM, no Raspberry Pi/ESP32, and
  ANN/MPC runs off-board in MATLAB.
- RG3 (Poor alignment with household demand patterns): Not Relevant. No
  residential hot-water draw schedule; objectives are outlet temperature
  tracking and pump energy, not morning/evening demand peaks
  (Coimbatore/Jaisalmer/Kochi profiles).
- RG4 (Limited real-world experimental validation): Highly relevant (positive
  example). Full bench-scale outdoor experiment with 7 temperature nodes and 3
  operating modes—supports your claim that PCM-SWH literature needs more work
  like this; paper itself is SSP water, not PCM-SWH.
- RG5 (No predictive optimization under climatic uncertainty): Partially
  relevant. MPC uses ANN forward prediction and responds to irradiance dips (Nov
  17, <500 W/m²); solar input from Bird–Hulstrom model, not ERA5/NASA POWER
  forecasting—your XGBoost irradiance + PPO stack goes further.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |  |  |
| --- | --- | --- | --- | --- |
| \(RMSE = \sqrt{SSE/n}\times 100\), \(R^2 = 1 - SSE/\sum p_i^2\) (2)–(3) | ANN regression accuracy | Benchmark XGBoost/Grey-box vs DS18B20 labels |  |  |
| \(T_{fo,corr} = T_{fo,pred} + \alpha\tanh(\beta(\dot{m}-0.01))\) (5) | Flow-dependent outlet temperature correction | Simplified surrogate for valve/pump action before full RL env |  |  |
| MAE vs setpoint (reported 1.42 °C vs 2.97 °C) | Control tracking quality | RL reward: minimize \(\ | T_w - T_{set}\ | \) under forecast GHI |
| Sensible heat dynamics (implicit \(T_{wp}\), lag vs \(I_T\) peak) | Thermal inertia of storage medium | Analogous PCM melting/charging lag vs pyranometer peak |  |  |

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. A.A. El-Sebaii et al., "Thermal performance of shallow solar pond under open
   and closed cycle modes of heat extraction," Sol. Energy, 2013 [27] — Relevant
   because: Foundational closed vs open cycle SSP performance data aligned with
   this paper’s best-mode conclusion.
1. M. Mahfuz et al., "Performance investigation of TES with PCM for solar water
   heating," Int. Commun. Heat Mass Transf., 2014 [26] — Relevant because:
   Bridges PCM storage with solar water heating—the latent-storage layer this
   SSP paper lacks.
1. H.K. Ghritlahre & R.K. Prasad, "Application of ANN to predict solar collector
   systems — A review," Renew. Sustain. Energy Rev., 2018 [14] — Relevant
   because: Review of ANN for solar thermal prediction cited in their
   methodology framing.
1. A.H. Elsheikh et al., "Modeling of solar energy systems using ANN:
   comprehensive review," Sol. Energy, 2019 [20] — Relevant because: Broad ANN +
   solar system reference for your literature review ML subsection.
1. P.K. Bansal et al., "Effect of heat exchanger on the performance of a shallow
   solar pond water heater," Energy Convers. Manag., 1984 [6] — Relevant
   because: Classic HX integration in SSP-SWH geometry precedent for
   collector–storage coupling.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

# 36. Yan2025ML_MeltingTime_TriplexTube_PCM_summary.md

Source path: /mnt/data/Yan2025ML_MeltingTime_TriplexTube_PCM_summary.md

# The Potential of Machine Learning to Predict Melting Response Time of Phase Change Materials in Triplex-Tube Latent Thermal Energy Storage Systems

Authors: Peiliang Yan, Chuang Wen, Hongbing Ding, Xuehui Wang, Yan Yang

Year: 2025

Journal/Conference: Applied Energy, Vol. 390, Article 125863

DOI/Link: https://doi.org/10.1016/j.apenergy.2025.125863

IEEE Citation: P. Yan et al., "The potential of machine learning to predict
melting response time of phase change materials in triplex-tube latent thermal
energy storage systems," Appl. Energy, vol. 390, p. 125863, 2025, doi:
10.1016/j.apenergy.2025.125863.

────────────────────────────────────────

## 1. One-Line Summary

This study builds a 60-case enthalpy-porosity CFD dataset for a Y-fin
triplex-tube PCM unit (RT82, melt time 15–45 min) and compares PR, SVR, RFR, and
XGBoost (Bayesian-tuned) to predict melting response time—XGBoost achieves 92%
accuracy with ~5 min max error, while SHAP-style importance ranks fin width 51%
and HTF temperature 47% vs fin angle 2%.

────────────────────────────────────────

## 2. Problem Being Solved

- PCM poor conductivity slows melting in triplex-tube LHS, causing supply–demand
  temporal mismatch in buildings and solar thermal systems.
- CFD with enthalpy-porosity is accurate but expensive for design sweeps over
  fin geometry and HTF conditions.
- Empirical correlations are scenario-specific and inaccurate when extended to
  new fin configurations.
- Need fast, quantitative meta-models to guide fin design and operational
  setpoints for melting response time.
────────────────────────────────────────

## 3. Key Contributions

1. Y-shaped fin triplex-tube PCM-TES cross-section model (copper tubes
   200/150/50.8 mm OD) with RT82 PCM and fixed 2% Y-fin area fraction.
1. CFD dataset: 60 numerical cases; melting response time 15–45 min under varied
   fin width, fin angle, and HTF temperature.
1. Four ML regressors: Polynomial Regression (PR), SVR, Random Forest (RFR),
   XGBoost — hyperparameters tuned via Bayesian optimization (400 steps: 200
   random + 200 fine search).
1. XGBoost identified as best meta-model (92% accuracy; lowest test-set error).
1. Feature importance: fin width 51%, HTF temperature 47%, fin angle 2% — design
   guidance for surface area over angle styling.
────────────────────────────────────────

## 4. Methodology

### 4a. System

- Triplex-tube LTES: outer/middle/inner copper tubes; PCM in middle annulus; HTF
  in inner/outer channels.
- PCM: Rubitherm-class RT82 (\(T_s=350.15\) K, \(T_l=358.15\) K, \(L=176\)
  kJ/kg, \(k=0.2\) W/m·K, \(\rho=770\) kg/m³).
- Fins: Y-shaped on inner/middle tubes, staggered; branch length 2× root length.
### 4b. CFD (enthalpy-porosity)

- Continuity (1); momentum (2)–(3) with Boussinesq source \(\rho g \beta
  (T-T_m)\); porosity sink (4); liquid fraction \(\lambda(T)\) piecewise linear
  between \(T_s\) and \(T_l\).
- Same method as prior work [38]; generates labeled melt times for ML.
### 4c. ML Pipeline

1. Variable independence check before modeling.
1. Inputs: fin width, fin angle, HTF temperature; output: melting response time
   (min).
1. Train/test split; Bayesian hyperparameter search maximizing \(R^2\).
1. Evaluate with MSE (10) and \(R^2\) (11); residual and parity plots.
1. Permutation-based feature importance on best XGBoost model.
### 4d. Software/Hardware

- Python 3.9; PC: Intel i5-8300H @ 2.30 GHz, 24 GB RAM, Windows 11.
────────────────────────────────────────

## 5. PCM Details (if applicable)

| Property | RT82 (Table 2) |
| --- | --- |
| Density | 770 kg/m³ |
| Specific heat | 2000 J/kg·K |
| Thermal conductivity | 0.2 W/m·K |
| Latent heat | 176,000 J/kg |
| Solidus / liquidus | 350.15 K / 358.15 K (~77–85 °C) |
| \(\beta\) | 0.001 1/K |

Design variables (Table 1):

- Fin width: 0.5, 1, 1.5, 2 (mm per table; abstract also cites 5–15 mm range in
  narrative)
- Fin angle: 30°, 60°, 90° (abstract: 10°–30° branch-angle study context)
- HTF temperature: 363, 365.5, 368, 370.5, 373 K (90–100 °C)
Performance target: melting response time 15–45 min across 60 CFD cases.

────────────────────────────────────────

## 6. AI / ML / Control Details (if applicable)

| Algorithm | Key tuned hyperparameters | Performance notes |
| --- | --- | --- |
| PR | degree 2 | Most unbiased train/test MSE ratio 1.15× |
| SVR | \(\gamma=0.154\), \(C=186\), \(\epsilon=0.369\) | Severe overfitting; train/test MSE ratio 11.8× |
| RFR | 65 trees, max_depth 7, max_features 0.999 | Max residual >15 min; worst high-value bias |
| XGBoost | 497 trees, lr 0.389, max_depth 46, subsample 0.933 | 92% accuracy; max error ~5 min; best test \(R^2\) |

Metrics: MSE Eq. (10); \(R^2\) Eq. (11).

No real-time control — offline surrogate for design optimization.

────────────────────────────────────────

## 7. Solar / Climate Data Details (if applicable)

N/A — building/solar-TES motivated in introduction but dataset uses prescribed
HTF inlet temperatures (90–100 °C), not outdoor weather time series.

Project link: melt-time surrogate can inform grey-box PCM charge duration
estimates when HTF is driven by collector/pyranometer models.

────────────────────────────────────────

## 8. Key Results & Numbers

- Global temperature rise +1.09 °C since pre-industrial (IPCC AR6 cite).
- Dataset: 60 simulations; melt time 15–45 min.
- XGBoost accuracy: 92% (abstract/conclusion).
- XGBoost max prediction error ~5 min vs >15 min for RFR/SVR/PR outliers.
- SVR train/test MSE gap 11.8× (overfitting).
- PR train/test MSE gap 1.15× (best unbiasedness).
- XGBoost train/test MSE gap ~4.84×.
- Feature importance (XGBoost): fin width 51%, HTF temperature 47%, fin angle
  2%.
- Y-fin cross-section fixed at 2% of system area.
- Tube ODs: 200 / 150 / 50.8 mm (wall 2 / 2 / 1.2 mm).
────────────────────────────────────────

## 9. Baseline Comparison

| Method | vs XGBoost | Result |
| --- | --- | --- |
| Full CFD (enthalpy-porosity) | Ground truth for 60 cases | Minutes per case; ML replaces for sweeps |
| Polynomial Regression | Higher max error (~15 min cases) | Less accurate; best MSE fairness (1.15×) |
| SVR | Overfits (11.8× MSE gap) | Unsuitable without regularization redesign |
| Random Forest | Max error >15 min | Poor at high melt times |
| XGBoost | Best test \(R^2\) and 92% accuracy | ~5 min max residual |

────────────────────────────────────────

## 10. Hardware / Experimental Setup (if applicable)

N/A — CFD-generated dataset only; no physical triplex-tube experiment or
embedded platform in this paper. Copper tube/fin material properties from
tables; validation is train/test ML split against simulation labels.

────────────────────────────────────────

## 11. Limitations Acknowledged by Authors

- Optimal algorithm is context-specific to this triplex-tube Y-fin geometry.
- Models trained on one PCM (RT82) may not transfer directly to RT35/OM35
  without retraining.
- SVR shows significant overfitting on test set.
- Dataset size (60 cases) is modest; scalability relies on adding more CFD
  points.
- Authors note meta-model selection framework generalizes better than direct
  weight transfer across configurations.
────────────────────────────────────────

## 12. Direct Relevance to My Project

- RG1: Indirect — predicts melt time, not closed-loop control; informs
  charge-phase timing in DRL state/reward.
- RG2: Relevant — demonstrates XGBoost for PCM thermal surrogate (same toolkit
  as your PCM classifier); no hardware integration.
- RG3: Indirect — links melt speed to HTF temperature (collector outlet); map
  HTF 47% importance to demand-aligned charging.
- RG4: Not relevant — simulation-only, no field validation.
- RG5: Relevant — fast melt-time predictor complements climate-driven HTF
  forecasting; fin width importance supports geometry sensitivity in grey-box
  model.
────────────────────────────────────────

## 13. Equations to Reuse or Adapt

Liquid fraction (enthalpy-porosity):

\[

\lambda = \begin{cases} 0 & T < T_s \\ \dfrac{T-T_s}{T_l-T_s} & T_s \le T \le
T_l \\ 1 & T > T_l \end{cases}

\]

Momentum sink (porosity):

\[

A = -C\frac{(1-\lambda)^2}{\lambda^3 + \varepsilon}

\]

ML metrics:

\[

\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2, \quad

R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}

\]

Melting response time target: \(t_{melt} = f(\text{fin width}, \text{fin angle},
T_{HTF})\) — use as grey-box calibration output or reward penalty for slow
charge.

────────────────────────────────────────

## 14. Citations This Paper Uses (That I Should Also Cite)

1. Ermis et al., ANN for finned-tube PCM storage, Int. J. Heat Mass Transf.,
   2007 — early PCM+ANN thermal prediction.
1. Yan et al., leaf-vein bionic fin PCM-TES, Appl. Energy, 2023 — prior Y-fin
   qualitative study [38].
1. Mahdi & Nsofor, nano+foam triplex-tube melting, Appl. Energy, 2017 —
   triplex-tube enhancement benchmark.
1. Liu et al., AI–PCM TES review, Renew. Energy, 2025 — broader ML+PCM context.
1. Chen et al., Taguchi+GRA PCM-SWH, Energy Convers. Manag.: X, 2025 — SWH
   optimization with RT35-class PCM.
────────────────────────────────────────

## 15. Suggested Use in My IEEE Paper

- Section I: Cite temporal mismatch between solar availability and thermal
  demand as motivation for fast PCM charge modeling.
- Section II: Position Yan as XGBoost melt-time surrogate reference alongside
  your PCM selection XGBoost (Presentation cites Yan 2025).
- Section III: Use feature-importance pattern (width 51%, HTF 47%) to justify
  classifier features (\(k\), \(L\), \(T_m\), predicted collector outlet).
- Section IV: Benchmark surrogate training on 60+ CFD/experimental points for
  your tank geometry; report \(R^2\) and max error in minutes.
- Section V: Target >92% accuracy or <5 min melt-time RMSE vs their XGBoost
  baseline when calibrating grey-box against TRNSYS or bench data.
────────────────────────────────────────

# 4. Consolidation Map

The source set collectively covers five connected layers of the project: (1)
project objectives and research gaps; (2) climate-data acquisition and
cross-source validation; (3) climate-signature construction and regime
clustering; (4) PCM feasibility, MCDM ranking, uncertainty analysis and physics
validation; and (5) the literature foundation for PCM properties, SWH systems,
AI/ML prediction, forecasting, optimization and control.

The audit documents should be treated as the authoritative record of what the
implemented Objective 1 pipeline actually does and what remains to be repaired
or regenerated. The literature summaries should be treated as the evidence base
for methodological justification and comparison, rather than as evidence that
the project's own pipeline has achieved the same reported performance.

Important distinction retained from the project documentation: N1–N6 are
Objective 1 novelty positions, while RG1–RG5 are broader project research gaps.
They are not interchangeable.
