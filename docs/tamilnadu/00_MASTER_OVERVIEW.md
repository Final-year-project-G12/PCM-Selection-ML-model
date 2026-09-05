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
4. **Climate Regimes (Level A & B)**: Clusters points into spatial climate regimes (Level A) using Gaussian Mixture Models (GMM) and performs seasonal sensitivity analysis (Level B).
5. **PCM Feasibility & Screening**: Filters the current 62-candidate database (55 manufacturer-derived + 7 literature) against physical, corrosion, and safety constraints. The feasibility CSV retains a full per-candidate audit; current runs retain 9-15 candidates per cluster.
6. **Multi-Criteria Decision Making (MCDM)**: Ranks feasibility survivors using four independent methods (TOPSIS, GRA, PROMETHEE II, VIKOR) with Monte Carlo uncertainty propagation.
7. **Grey-Box Physics Validation**: Solves a lumped-enthalpy tank simulation using backward Euler, driven by the real 10-year daily weather of each regime's medoid point, evaluating Spearman rank concordance.
8. **Recommendation Cards**: Generates markdown summary cards for each climate regime.

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
  05_cluster_tamilnadu.py       → cluster_assignments (K_FINAL=5, covariance_type=diag)
  05b_cluster_interactive.py    → interactive GMM cluster map
  11_level_b_seasonal_analysis.py → level_b_seasonal_topk.csv, level_b_seasonal_summary.md
        ↓
Phase 5 — FEASIBILITY FILTERING
  07_feasibility_filter.py      → feasibility_survivors_by_cluster.csv (8 Table-12 filters)
        ↓
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking.py            → mcdm_topk_by_cluster.csv, monte_carlo_stability.csv
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
| 3 — Climate Signature | `04b`, `04d` | **COMPLETE** | 300 L/day draw with `SHARE_PCM=0.5`; current generated cluster targets are approximately 301-326 kJ/kg. |
| 4 — GMM Clustering | `05`, `05b`, `11` | **COMPLETE (v3.1 fixes applied)** | K=5 regimes, `covariance_type="diag"`. Level B seasonal re-rank uses corrected draw volume. |
| 5 — Feasibility | `06`, `07` | **COMPLETE** | 62 PCM records are audited per cluster; current pass counts are 9-15 and vary by cluster. |
| 6 — MCDM Ranking | `08` | **COMPLETE** | 4-method Borda + 5000-draw Monte Carlo. |
| 7 — Physics Validation | `10` | **COMPLETE (v3.2 fixes applied)** | Tank ambient heat loss active (`UA_TANK_W_K=2.0`); backward-Euler solve error and missing night-isolation both fixed. Mean Spearman ρ = +0.177 (was -0.151); 41% of runs now in the 54–84% benchmark band (was 0%). |
| 8 — Rec Cards | `09` | **COMPLETE** | Aggregates Phases 4–7 into `recommendation_cards.md`. |

## Corrected Issues (v3.1 — August 2026; v3.2 — physics solver)
All five critical bugs from the v3.0 audit are fixed in source code (v3.1). Two further Phase 7 solver bugs, which had silently kept the v3.1 physics fix from taking effect, are fixed as v3.2. See `20_IMPLEMENTATION_ISSUES.md` for full details and numerical proof.

1. **Deaccumulation** → `accum_to_flux()` in `02_combine_tamilnadu.py` (v3.1)
2. **Quantile mapping** → Step 2b in `04_preprocess_tamilnadu.py` + `03b_agreement_analysis.py` (v3.1)
3. **1000× flow rate** → 300 L/day in `04b_climate_signature.py` and `11_level_b_seasonal_analysis.py` (v3.1)
4. **GMM overfitting** → `covariance_type="diag"` in `05_cluster_tamilnadu.py` (v3.1)
5. **Tank heat loss** → `UA_TANK_W_K=2.0` in `10_physics_validation.py` (v3.1, but ineffective until v3.2)
6. **Backward-Euler closed-form solve error** → numerator corrected to use old `Tp` only, in `10_physics_validation.py` (v3.2)
7. **Missing night/idle collector-coupling isolation** → `NIGHT_ISOLATION_FRACTION=0.05` gates the collector coupling when Tc<Tw, in `10_physics_validation.py` (v3.2)

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
