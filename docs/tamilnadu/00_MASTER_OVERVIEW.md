# 00 — Master Overview: ERA5 Tamil Nadu Climate → PCM Selection Pipeline

## Project Objective
Final-year B.Tech CSE project:
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating"**
Objective 1 builds a **climate-region-aware PCM recommendation framework**: turning 10 years of reanalysis climate data into population-weighted climate regimes, deriving PCM performance targets per regime, and ranking candidate phase-change materials against those targets with an auditable, multi-method, uncertainty-aware pipeline.

Governing document: `Objective1_PCM_Climate_Framework_Plan_v3.docx` ("the framework doc"), version 3.0. The **Tamil Nadu pipeline has been fully implemented from Phase 1 through Phase 8**. Critical v3.0 bugs were corrected in **v3.1** (August 2026). A cross-check against the Rajasthan pipeline's documented bug history then found that v3.1's Phase 7 fix had not actually taken effect — two further solver bugs were reproducing the pre-fix symptom. Both are corrected in **v3.2**; see `19_PHASE_7_8_AUDIT.md` and `20_IMPLEMENTATION_ISSUES.md` (#6, #7).

## What the ERA5 Pipeline Achieves
1. **Population-Weighted Sampling**: Samples Tamil Nadu at 133 population-weighted points (representing 87.5% of the state's population) to ensure findings are representative of where domestic demand actually resides.
2. **Double-Source Validation**: Pulls ERA5 reanalysis and NASA POWER satellite/model data for the same coordinates and times, validating one against the other.
3. **Two-Tier Climate Signature**: Redefines 10 years of hourly/daily data into instantaneous sun-event statistics (Tier 1) and true daily-integral indices (Tier 2).
4. **Climate Regimes (Level A & B)**: Clusters points into spatial climate regimes (Level A, `05_cluster_tamilnadu.py`) using Gaussian Mixture Models (GMM), then re-clusters per point per season to detect regime shifts (Level B, `05a_level_b_regime_shift_tamilnadu.py`). A separate post-Phase-6 script (`11_seasonal_pcm_sensitivity.py`) checks whether the recommended PCM flips by season.
5. **PCM Feasibility & Screening**: Filters the current 62-candidate database (55 manufacturer-derived + 7 literature) against physical, corrosion, and safety constraints. The feasibility CSV retains a full per-candidate audit; current runs retain 9-15 candidates per cluster.
6. **Multi-Criteria Decision Making (MCDM)**: Ranks feasibility survivors using four independent methods (TOPSIS, GRA, PROMETHEE II, VIKOR) with Monte Carlo uncertainty propagation.
7. **Grey-Box Physics Validation**: Solves a lumped-enthalpy tank simulation using backward Euler, driven by the real 10-year daily weather of each regime's medoid point, evaluating Spearman rank concordance.
8. **Recommendation Cards**: Generates markdown summary cards for each climate regime.

<<<<<<< HEAD
=======
---

## Research Gaps Addressed (N1–N6 Novelty & RG1–RG5 Mapping)

Objective 1 addresses the broader final-year project goals. The framework doc establishes six specific novelty positions (**N1–N6**) and maps how each pipeline phase contributes to resolving key research gaps (**RG1–RG5**):

### Disambiguation of Novelties (N1–N6) and Research Gaps (RG1–RG5)
- **Novelty Positions (N1–N6)**: Specific methodological innovations of this framework (population-weighting, two-tier signatures, 42–70°C SWH melting band, 4-method MCDM, falsifiable physics validation).
- **Research Gaps (RG1–RG5)**: Broader challenges in solar-thermal literature addressed across the overall multi-objective project.

### Phase → Research Gap Mapping Table

| Phase | Research Gap | How It Contributes | Key References |
|---|---|---|---|
| **Phase 1 — Data Collection** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Establishes the population-weighted grid (133 points, 87.5% population) and temporal solar windows to represent real meteorological stress. | WorldPop 2020; GADM v4.1 |
| **Phase 2 — Preprocessing & QA** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Filters, cleans, and imputes historical weather with 13-step QC and Step 2b quantile mapping, providing high-fidelity data. | Ghodusinejad (2026); Mansouri (2025) |
| **Phase 3 — Climate Signature** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Distills raw weather into PCM-facing thermal targets (`Tm_target`, `L_required` with `SHARE_PCM=0.5`). | Avargani (2021); Singh (2025) |
| **Phase 4 — GMM Clustering** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Discovers 5 spatial climate regimes (`covariance_type="diag"`), replacing arbitrary administrative boundaries with GMM profiles. | Liu et al. (2025) |
| **Phase 5 — Feasibility Filter** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Implements 8 physical screening constraints (Table 12) to prevent compensatory MCDM errors. | Martinez (2025); Abdellatif (2025) |
| **Phase 6 — MCDM Ranking** | **RG5**: Lack of predictive optimization under climatic uncertainty. | 4-method Borda + Monte Carlo (N_DRAWS=1000) uncertainty propagation over 8 Table-13 criteria identifies Top-3 candidates robust to parameter uncertainty; unified with Rajasthan. | Chen et al. (2025); Chopra (2023) |
| **Phase 7 — Physics Validation** | **RG4**: Limited real-world experimental / dynamic physical validation. | Provides a grey-box lumped-enthalpy tank simulation (`UA_TANK_W_K=2.0`) to verify that MCDM rankings correlate with physical solar fraction. | Barqawi (2025) |
| **Phase 8 — Rec Cards** | **RG3**: Poor alignment with household demand. | Distills final recommendations into actionable cards aligned to domestic hot water profiles (300 L/day). | Odoi & Yorke (2025) |
| **Phase 9+ — DRL Controller** | **RG1**: Lack of real-time adaptive control. | Future work: DRL uses discovered regimes to optimize PCM charging online. | Emami (2026); Terfai (2025) |

---

>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637
## Complete Pipeline Map
```
Phase 1 — DATA COLLECTION
  00a_build_population_grid.py  → population_grid_points.csv (133 pts, 87.5% pop coverage)
  00b_build_suntimes.py         → suntimes.csv (1,457,547 rows: 133 pts × 3653 days × 3 events)
  01_download_era5_tamilnadu.py → data/raw/era5/points/*.nc (instant + accum NetCDF)
  01b_download_nasapower.py     → data/raw/nasapower/*.json (1330 files, full hourly cache)
  00_unzip_accum.py             → (fixes CDS zip-disguised-as-.nc quirk)
        ↓
Phase 2 — PREPROCESSING & CROSS-SOURCE VALIDATION
  02_combine_tamilnadu.py       → climate_tamilnadu_points.csv (accum_to_flux, v3.1)
  02b_build_daily_aggregates.py → daily_aggregates_tamilnadu.csv (POWER-only daily integrals)
  03_plots_raw.py               → raw diagnostic plots & C_era5_vs_power_stats.csv
  03b_agreement_analysis.py     → era5_power_agreement_tamilnadu.csv, bias decision (NEW v3.1)
  03b_interactive_raw_qa.py     → interactive Plotly/Folium HTML maps/plots
  04_preprocess_tamilnadu.py    → tamilnadu_cleaned_physical.csv (13-step QC + Step 2b QM)
  04c_postprocess_plots.py      → post-cleaning QA plots
        ↓
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
  04b_climate_signature.py      → climate_signature_tamilnadu.csv (300 L/day draw, SHARE_PCM=0.5)
  04d_signature_interactive.py  → interactive signature exploration maps
        ↓
Phase 4 — CLIMATE REGIME CLUSTERING
  05_cluster_tamilnadu.py       → cluster_assignments (k=3 via cluster_lib.suggest_k cascade, covariance_type=diag)
  05a_level_b_regime_shift_tamilnadu.py → Level B: cluster_assignments_tamilnadu_levelB.csv + regime-shift Sankey
  05b_cluster_interactive.py    → interactive GMM cluster map
        ↓
Phase 5 — FEASIBILITY FILTERING
<<<<<<< HEAD
  07_feasibility_filter.py      → feasibility_survivors_by_cluster.csv (8 Table-12 filters)
        ↓
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking.py            → mcdm_topk_by_cluster.csv, monte_carlo_stability.csv
=======
  06_build_pcm_database.py      → pcm_database_tamilnadu.csv (55 manufacturer + 7 literature = 62 rows)
  07_feasibility_filter.py      → feasibility_survivors_by_cluster.csv (8 Table-12 constraints, fixed κ=0.7)
                                 + feasibility_survivors_by_cluster_kappa_calibrated.csv (κ-calibrated companion)
                                 [07b_charging_feasibility.py RETIRED 2026-09-08 — folded into Constraint 6]
        ↓
Phase 6 — MULTI-CRITERIA RANKING ENGINE  (UNIFIED with Rajasthan 2026-09-08 — byte-identical engine)
  08_mcdm_ranking.py            → mcdm_full_rankings.csv, mcdm_topk_by_cluster.csv, monte_carlo_stability.csv,
                                  mcdm_method_agreement.csv, outputs/qc_montecarlo_inclusion.html
                                  (8 Table-13 criteria; climate-relative latent heat; log-scaled cycling;
                                   supercooling entropy weight capped at 2× prior; PROMETHEE native Tm;
                                   Kendall's W + pairwise agreement; N_DRAWS=1000)
>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637
        ↓
Phase 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py      → physics_validation_results.csv (UA_TANK=2.0 W/K + corrected backward-Euler solve + night isolation, v3.2)
        ↓
Phase 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py    → recommendation_cards.md
```

## Phase 1–8 Status and Headline Findings
| Phase | Script(s) | Status | Headline Finding |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 133 points, 240 NetCDF files, 1330 NASA POWER JSON files. |
| 2 — Preprocessing & QA | `02`, `02b`, `03`, `03b`, `04`, `04c` | **COMPLETE (v3.1 fixes applied)** | Deaccumulation replaced with `accum_to_flux()`. Per-season quantile mapping in Step 2b. Re-run required for new outputs. |
<<<<<<< HEAD
| 3 — Climate Signature | `04b`, `04d` | **COMPLETE** | 300 L/day draw with `SHARE_PCM=0.5`; current generated cluster targets are approximately 301-326 kJ/kg. |
| 4 — GMM Clustering | `05`, `05b`, `11` | **COMPLETE (v3.1 fixes applied)** | K=5 regimes, `covariance_type="diag"`. Level B seasonal re-rank uses corrected draw volume. |
| 5 — Feasibility | `06`, `07` | **COMPLETE** | 62 PCM records are audited per cluster; current pass counts are 9-15 and vary by cluster. |
| 6 — MCDM Ranking | `08` | **COMPLETE** | 4-method Borda + 5000-draw Monte Carlo. |
| 7 — Physics Validation | `10` | **COMPLETE (v3.2 fixes applied)** | Tank ambient heat loss active (`UA_TANK_W_K=2.0`); backward-Euler solve error and missing night-isolation both fixed. Mean Spearman ρ = +0.177 (was -0.151); 41% of runs now in the 54–84% benchmark band (was 0%). |
| 8 — Rec Cards | `09` | **COMPLETE** | Aggregates Phases 4–7 into `recommendation_cards.md`. |
=======
| 3 — Climate Signature | `04b`, `04d` | **Analysis complete; clean re-run pending** | 300 L/day draw with `SHARE_PCM=0.5` (now defined in `config.py`); completed-run cluster targets ≈ 301-326 kJ/kg. |
| 4 — GMM Clustering | `05`, `05a`, `05b`, `cluster_lib.py` | **COMPLETE (unified with Rajasthan 2026-09-08)** | **k=3** via the shared 3-tier `suggest_k` cascade (bootstrap-ARI tiebreak), `covariance_type="diag"`, Köppen-Geiger external validation (ARI 0.067), canonical latitude relabel, `provenance_lib` hard-fail wired into 07/08/10/09. `05a` = Level B regime-shift re-clustering (k=4, 90.2% shift). `11_seasonal_pcm_sensitivity.py` (post-Phase-6, reads `08` outputs) is now listed under Phase 5-8, not here. |
| 5 — Feasibility | `06`, `07` | **Code unified with Rajasthan 2026-09-08; clean re-run pending** | `06` builds 62 records (55 manufacturer + 7 literature) from the one canonical MICE/PMM output; `07` applies 8 constraints (Constraint 6 = `Tm ≤ Tm_target_capped_C` from Phase 3) + κ-calibration, emitting `feasibility_survivors_by_cluster.csv` and `…_kappa_calibrated.csv`. `07b_charging_feasibility.py` deleted. Pre-unification pass counts 15/9/13/13/9 are stale. |
| 6 — MCDM Ranking | `08` | **UNIFIED with Rajasthan 2026-09-08 (byte-identical engine); re-run pending** | 8 Table-13 criteria; climate-relative latent heat; log-scaled cycling; **supercooling entropy weight capped at 2× prior (0.16)**; PROMETHEE native Tm (q=2K/p=8K); Kendall's W + pairwise method-agreement; N_DRAWS=1000. 2026-09-08 run: k=3, `Tm_fitness`-dominant, Top-1 = Myristic acid / n-Tetracosane / Palmitic-Stearic. |
| 7 — Physics Validation | `10` | **Analysis complete; clean re-run pending** | `UA_TANK_W_K=2.0` active. Completed run: mean Spearman ρ = **+0.177** (per cluster −0.016/+0.717/+0.355/−0.171/−0.000); 24/59 sims in 54-84% band; cycles 3-260/yr. |
| 8 — Rec Cards | `09` | **Re-run pending** | Aggregates Phases 4–7 into `recommendation_cards.md` (k=3 → 3 cluster cards after the unified Phase 4). |
>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637

## Corrected Issues (v3.1 — August 2026; v3.2 — physics solver)
All five critical bugs from the v3.0 audit are fixed in source code (v3.1). Two further Phase 7 solver bugs, which had silently kept the v3.1 physics fix from taking effect, are fixed as v3.2. See `20_IMPLEMENTATION_ISSUES.md` for full details and numerical proof.

<<<<<<< HEAD
1. **Deaccumulation** → `accum_to_flux()` in `02_combine_tamilnadu.py` (v3.1)
2. **Quantile mapping** → Step 2b in `04_preprocess_tamilnadu.py` + `03b_agreement_analysis.py` (v3.1)
3. **1000× flow rate** → 300 L/day in `04b_climate_signature.py` and `11_level_b_seasonal_analysis.py` (v3.1)
4. **GMM overfitting** → `covariance_type="diag"` in `05_cluster_tamilnadu.py` (v3.1)
5. **Tank heat loss** → `UA_TANK_W_K=2.0` in `10_physics_validation.py` (v3.1, but ineffective until v3.2)
6. **Backward-Euler closed-form solve error** → numerator corrected to use old `Tp` only, in `10_physics_validation.py` (v3.2)
7. **Missing night/idle collector-coupling isolation** → `NIGHT_ISOLATION_FRACTION=0.05` gates the collector coupling when Tc<Tw, in `10_physics_validation.py` (v3.2)
=======
## Corrected Issues

The five v3.0 critical bugs plus three blocking script/orchestrator errors found on 2026-09-07 are fixed in source:

1. **Deaccumulation** → `accum_to_flux()` in `02_combine_tamilnadu.py`
2. **Quantile mapping** → Step 2b in `04_preprocess_tamilnadu.py` + `03b_agreement_analysis.py`
3. **1000× flow rate** → 300 L/day in `04b_climate_signature.py` and `11_seasonal_pcm_sensitivity.py`
4. **GMM overfitting** → `covariance_type="diag"` in `05_cluster_tamilnadu.py`
5. **Tank heat loss** → `UA_TANK_W_K=2.0` in `10_physics_validation.py`
6. **Missing `config.py` symbols** (ImportError in `04b`/`07`/`11`) → `SHARE_PCM = 0.5` and `latent_heat_floor_kj_kg()` added to `config.py` (2026-09-07)
7. **`06_build_pcm_database.py` input path** resolved to a non-existent `era5-tamilnadu/PCM_data/` → fixed to repo-root `PCM_data/data/` (2026-09-07)
8. **`run_all_tamilnadu.py`** had `02_combine` commented out of CORE, ran `11` before its inputs, and cited non-existent guides → reconciled (2026-09-07)

---

## Reproducibility Checklist & Execution Guide (formerly `21_REPRODUCIBILITY.md`)

### One-Command Runner
`python run_all_tamilnadu.py` runs the CORE chain in dependency order (stops on first required failure).
- `--include-setup`: also runs Phase-1 downloads first.
- `--with-optional`: adds QA/plot scripts.
- `--dry-run`: prints the resolved order.
- `--from <script>`: resumes mid-chain.
- `05c_explore_interactive.py`: launch separately via `streamlit run 05c_explore_interactive.py`.

### Step-by-Step Chronological Execution Checklist
- [ ] **Phase 1**: `python 00a_build_population_grid.py` → 133 points (`TNP_0001`–`TNP_0133`)
- [ ] **Phase 1**: `python 00b_build_suntimes.py` → 1,457,547 rows
- [ ] **Phase 1**: `python 01_download_era5_tamilnadu.py` → 240 NetCDF files
- [ ] **Phase 1**: `python 01b_download_nasapower.py` → 1330 JSON files
- [ ] **Phase 1**: `python 00_unzip_accum.py` → fix NetCDF zip-disguise
- [ ] **Phase 2**: `python 02_combine_tamilnadu.py` → uses `accum_to_flux()`
- [ ] **Phase 2**: `python 02b_build_daily_aggregates.py` → daily integrals + `tier2_signature_tamilnadu.csv`
- [ ] **Phase 2**: `python 03_plots_raw.py` → raw QA plots (optional)
- [ ] **Phase 2**: `python 03b_agreement_analysis.py` → cross-source decision
- [ ] **Phase 2**: `python 04_preprocess_tamilnadu.py` → Step 2b quantile mapping
- [ ] **Phase 3**: `python 04b_climate_signature.py` → 300 L/day draw; `SHARE_PCM` imported from `config.py`
- [ ] **Phase 4**: `python 05_cluster_tamilnadu.py` → `covariance_type="diag"`, `K_FINAL=5`
- [ ] **Phase 5**: `python 06_build_pcm_database.py` → 62 PCMs (55 manufacturer + 7 literature)
- [ ] **Phase 5**: `python 07_feasibility_filter.py` → `feasibility_survivors_by_cluster.csv` + `feasibility_survivors_by_cluster_kappa_calibrated.csv` (62 audited per cluster, 8 constraints, κ-calibrated companion)
- [ ] **Phase 6**: `python 08_mcdm_ranking.py` → 8 criteria, N_DRAWS=1000 (raise to 5000 for final); writes `mcdm_full_rankings.csv` + `mcdm_topk_by_cluster.csv` + `monte_carlo_stability.csv` + `mcdm_method_agreement.csv`
- [ ] **Phase 7**: `python 10_physics_validation.py` → `UA_TANK=2.0 W/K`; writes `physics_validation_results.csv`
- [ ] **Phase 8**: `python 09_recommendation_cards.py` → `recommendation_cards.md`
- [ ] **Phase 4 Level B (regime shift)**: `python 05a_level_b_regime_shift_tamilnadu.py` → runs in Phase-4 order, non-blocking
- [ ] **Seasonal PCM sensitivity**: `python 11_seasonal_pcm_sensitivity.py` → runs LAST; reads `08` outputs

### Environmental Configuration & Random Seeds
- **pvlib version**: `pvlib >= 0.9`
- **CDS API**: `.cdsapirc` in `era5-tamilnadu/`
- **`config.py` PCM constants**: `SHARE_PCM = 0.5`; `latent_heat_floor_kj_kg(l_required, fraction=0.7, absolute_min_kj_kg=100.0)`
- **Random seeds**: `KMeans(random_state=42)`, `GaussianMixture(random_state=42)`, `run_monte_carlo(seed=42)`
- **Python dependencies**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`

---

## Data-Layout Note (Clean Re-run Pending)
The `data/processed/processed/` path-duplication bug is **fixed** in `config.py` / `04b_climate_signature.py`, and the stale `era5-tamilnadu/data/processed/processed/` mirror tree has been **deleted** (2026-09-08). The canonical `data/processed/` tree still holds a superseded pre-unification Phase 5 run (7-constraint schema); one clean re-run of the CORE chain — after the unified Phase 3 (which must produce `Tm_target_capped_C` via `kt_worst_month`) — regenerates `pcm_database_tamilnadu.csv` (62 rows), `feasibility_survivors_by_cluster.csv`, `feasibility_survivors_by_cluster_kappa_calibrated.csv`, and everything downstream, in the canonical location. Do not quote pre-unification survivor counts.

---
>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637

## Still Open
See `22_FINAL_READINESS_REPORT.md`: PCM database expansion, external cluster validation, elevation proxy, monsoon precipitation download, full Level-B GMM, bootstrap-ARI k-selection, cross-phase provenance hard-fail, residual Phase 7 tank/collector calibration.

## Plot Documentation
See `23_PLOTS_GUIDE.md` for the interpretation and exact location of plots produced by the raw QA, preprocessing, climate-signature, clustering, comprehensive, Objective 1, and comparison scripts.

## Literature Support
| Pipeline Component | Key Reference | Source File |
|---|---|---|
| Population grid | GADM + WorldPop | `03_PHASE_1_AUDIT.md` |
| Solar geometry | Reda & Andreas (2004) SPA | `12_SOLAR_GEOMETRY.md` |
| Cross-source validation | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Climate signature / sizing | Avargani et al. (2021), Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| MCDM stack | Chen et al. (2025) Taguchi+GRA | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
| Physics validation | Barqawi (2025) | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
| Full matrix | — | `17_LITERATURE_MAPPING.md` |
