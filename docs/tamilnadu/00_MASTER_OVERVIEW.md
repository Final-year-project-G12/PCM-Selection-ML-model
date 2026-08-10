# 00 — Master Overview: ERA5 Tamil Nadu Climate → PCM Selection Pipeline

## Project Objective
Final-year B.Tech CSE project:
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating"**
Objective 1 builds a **climate-region-aware PCM recommendation framework**: turning 10 years of reanalysis climate data into population-weighted climate regimes, deriving PCM performance targets per regime, and ranking candidate phase-change materials against those targets with an auditable, multi-method, uncertainty-aware pipeline.

Governing document: `Objective1_PCM_Climate_Framework_Plan_v3.docx` ("the framework doc"), version 3.0. Unlike the Rajasthan pipeline (where Phase 7 and 8 were planned but not implemented), the **Tamil Nadu pipeline has been fully implemented from Phase 1 through Phase 8**.

## What the ERA5 Pipeline Achieves
1. **Population-Weighted Sampling**: Samples Tamil Nadu at 133 population-weighted points (representing 87.5% of the state's population) to ensure findings are representative of where domestic demand actually resides.
2. **Double-Source Validation**: Pulls ERA5 reanalysis and NASA POWER satellite/model data for the same coordinates and times, validating one against the other.
3. **Two-Tier Climate Signature**: Redefines 10 years of hourly/daily data into instantaneous sun-event statistics (Tier 1) and true daily-integral indices (Tier 2).
4. **Climate Regimes (Level A & B)**: Clusters points into spatial climate regimes (Level A) using Gaussian Mixture Models (GMM) and performs seasonal sensitivity analysis (Level B).
5. **PCM Feasibility & Screening**: Filters a 25-candidate database (18 manufacturer + 7 literature) against physical, corrosion, and safety constraints.
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
  02_combine_tamilnadu.py       → climate_tamilnadu_points.csv (merges sun-event instants)
  02b_build_daily_aggregates.py → daily_aggregates_tamilnadu.csv (POWER-only daily integrals)
  03_plots_raw.py               → raw diagnostic plots & C_era5_vs_power_stats.csv
  03b_interactive_raw_qa.py     → interactive Plotly/Folium HTML maps/plots
  04_preprocess_tamilnadu.py    → tamilnadu_cleaned_physical.csv (13-step QC and imputation)
  04c_postprocess_plots.py      → post-cleaning QA plots
        ↓
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
  04b_climate_signature.py      → climate_signature_tamilnadu.csv (combines Tier 1 & 2, PCA, targets)
  04d_signature_interactive.py  → interactive signature exploration maps
        ↓
Phase 4 — CLIMATE REGIME CLUSTERING
  05_cluster_tamilnadu.py       → cluster_assignments_tamilnadu.csv, cluster_profiles_tamilnadu.csv (K_FINAL=5)
  05b_cluster_interactive.py    → interactive GMM cluster map
  11_level_b_seasonal_analysis.py → level_b_seasonal_topk.csv, level_b_seasonal_summary.md
        ↓
Phase 5 — FEASIBILITY FILTERING
  07_feasibility_filter.py      → feasibility_survivors_by_cluster.csv (8 physical/corrosion/safety filters)
        ↓
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking.py            → mcdm_topk_by_cluster.csv, mcdm_full_scores_by_cluster.csv,
                                  monte_carlo_stability.csv (TOPSIS, GRA, PROMETHEE II, VIKOR, 5000 MC draws)
        ↓
Phase 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py      → physics_validation_results.csv, physics_validation_spearman.csv
        ↓
Phase 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py    → recommendation_cards.md (per-cluster card report)
```

## Phase 1–8 Status and Headline Findings
| Phase | Script(s) | Status | Headline Finding |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 133 points, 240 NetCDF files, 1330 NASA POWER JSON files. |
| 2 — Preprocessing & QA | `02`, `02b`, `03`, `03b`, `04`, `04c` | **COMPLETE - WITH SILENT BUGS** | **Active Deaccumulation Bug**: `deaccumulate()` uses `diff()`, producing near-zero GHI (MBE = -231.89 W/m², r = 0.396). **Missing Quantile Mapping**: QM correction is mentioned but not implemented in `04_preprocess`. |
| 3 — Climate Signature | `04b`, `04d` | **COMPLETE - WITH UNITS BUG** | **1000x Water Flow Rate Error**: `DRAW_RATE_KG_PER_S` is off by 1000x, underestimating night draw (25.2 kg instead of 25,200 kg or 300 kg), leading to a very low target `L_required` (~52 kJ/kg). |
| 4 — GMM Clustering | `05`, `05b`, `11` | **COMPLETE - OVERFITTING RISK** | Clusters TN into `K_FINAL = 5` regimes. Uses `covariance_type="full"`, which overfits on 133 points. Level B seasonal analysis shows flips in #1 PCM. |
| 5 — Feasibility | `07` | **COMPLETE - SILENT BYPASS** | Low `L_required` (~52 kJ/kg) makes the latent-heat floor (~36 kJ/kg) a no-op; 7 candidates survive in all clusters without relaxation. |
| 6 — MCDM Ranking | `08` | **COMPLETE** | Ranks survivors across all 4 methods. 5000-draw Monte Carlo provides stability probabilities. |
| 7 — Physics Validation | `10` | **COMPLETE - DISAGREEMENT CAUGHT** | Solves grey-box model. Spearman rho is very low (r = 0.18–0.54, "weak agreement") due to signature/filter errors. Solar fractions are systematically high (~95%) with 0–1 cycles/year. |
| 8 — Rec Cards | `09` | **COMPLETE** | Successfully aggregates Phases 4–7 into `recommendation_cards.md`. |

## Current Known Issues
1. **Deaccumulation Bug in `02_combine_tamilnadu.py`**: The script uses a diff-based `deaccumulate()`, but the CDS API downloads for Tamil Nadu already return hourly fluxes, not cumulative totals. This makes the ERA5-derived GHI near-zero.
2. **Missing Quantile-Mapping Correction**: In `04_preprocess_tamilnadu.py`, the quantile mapping is not applied. Thus, the near-zero GHI is normalized and directly clustered as `GHI_mean_z`.
3. **1000x Flow Rate Error in `04b_climate_signature.py`**: Flow rate is calculated as `60.0 / 1000 / 60` (which is `0.001` kg/s), underestimating a 60 L/min flow rate (1.0 kg/s) by 1000x. The resulting `L_required` is 51–54 kJ/kg, rendering the latent-heat filter useless.
4. **GMM Covariance Overfitting**: Using `covariance_type="full"` with 133 samples and 27 dimensions overdetermines the model.
5. **Physics Tank Model Simplifications**: The lumped tank model ignores ambient heat losses, leading to artificially high solar fractions (~95%) and 0–1 cycles/year (the PCM never freezes/melts dynamically).
