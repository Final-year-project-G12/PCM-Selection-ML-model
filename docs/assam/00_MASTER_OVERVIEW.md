# 00 — Master Overview: ERA5 Assam Climate → PCM Selection Pipeline

## Project Objective

Final-year B.Tech CSE project (Group 12, Amrita School of Engineering, Guide: Dr. T. Deepika):
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating."** 

Objective 1 builds an end-to-end **climate-region-aware PCM recommendation and validation framework**: transforming 10 years of reanalysis climate data into population-weighted climate regimes, deriving engineering performance targets per regime, screening candidate phase-change materials against those targets, and conducting an independent, multi-year dynamic physics validation.

Governing framework: Objective 1 Framework Plan (v3.0), expanded through comprehensive audits and implementation into a **complete 11-phase architecture**. This documentation covers **Assam** — the third state in the multi-state comparative study, characterized by a humid subtropical monsoon regime with pronounced seasonal cloud cover and high humidity.

---

## Complete Pipeline Architecture (Phases 1–11)

The Assam pipeline has evolved from its initial exploratory prototype into a fully locked, auditable 11-phase research framework:

```
Phase 1 — POPULATION-WEIGHTED SAMPLING & SPATIAL GRID
  00a_build_population_grid.py   → population_grid_points.csv (129 points, 87.8% population coverage)
  00b_build_suntimes.py          → suntimes.csv (solar noon, sunrise, sunset for 10 years)
  01_download_era5_assam.py      → data/raw/era5/points/*.nc (hourly ERA5 reanalysis)
  01b_download_nasapower.py      → data/raw/nasapower/*.json (10 years hourly NASA POWER)
        ↓
Phase 2 — CLIMATE PREPROCESSING & CROSS-SOURCE VALIDATION
  02_combine_assam.py            → climate_assam_points.csv (solar geometry, unit conversions)
  02b_build_daily_aggregates_assam.py → daily_aggregates_assam.csv (467,367 daily rows)
  03b_agreement_analysis_assam.py → bias_decision_assam.txt (BACKBONE decision, 1.1% GHI MBE)
        ↓
Phase 2.5 — QUALITY CONTROL & OUTLIER DETECTION
  04_preprocess_assam.py         → preprocessed/parquet/{point_id}.parquet (129 files)
                                   (Physical bounds checks, IsolationForest multivariate flagging)
        ↓
Phase 3 — CLIMATE REGIME CLUSTERING (LOCKED K=3 MODEL)
  04b_climate_signature.py       → climate_signatures_raw.csv (18 indices across 129 sites)
  05_cluster_assam.py            → clustering/cluster_assignments_assam.csv (K=3 GMM, full covariance)
                                   (5 core features: GHI_mean, Ta_mean, DTR, RH_mean, wind_mean;
                                    min BIC=1574.94 at K=3; medoids: ASP_0012, ASP_0092, ASP_0028)
        ↓
Phase 4 — SOLAR WATER HEATING (SWH) DESIGN SPECIFICATION
  05_cluster_assam.py            → clustering/cluster_profiles_assam.csv
                                   (50 kg PCM, 100 kg water, 100 L/day demand, 50 L AM + 50 L PM,
                                    Tm_target = 44.0°C, approach ΔT = 6 K, T_delivery = 50.0°C)
        ↓
Phase 5 — CURATED PCM PROPERTY DATABASE
  06_build_pcm_database.py       → pcm/pcm_database_final.csv (58 deduplicated PCMs × 41 columns)
                                   (Strict provenance: source_type, value_status; strict Cp average)
                                   [Historical prototype: pcm_database_assam.csv, 25 rows]
        ↓
Phase 6 — FEASIBILITY FILTERING ENGINE
  07_feasibility_filter.py       → Final K=3 Governance: n_confirmed = [0, 0, 0], 1 conditional
                                   (n-Tetracosane C24, Tm=52°C in C0); 0 auto-relaxation.
                                   [Historical K=4 screening preserved 8 unique candidate PCMs]
        ↓
Phase 7 — MULTI-CRITERIA DECISION MAKING (MCDM)
  08_mcdm_ranking.py             → Final K=3 Governance: NOT PERFORMED (n_confirmed = 0)
                                   [Historical K=4 pre-audit ranking preserved as reference benchmark:
                                    TOPSIS, GRA, PROMETHEE II, VIKOR, Borda consensus]
        ↓
Phase 8 — MONTE CARLO UNCERTAINTY ANALYSIS
  08_mcdm_ranking.py             → Final K=3 Governance: SKIPPED (n_draws = 0)
                                   [Historical K=4 pre-audit 5,000-draw stability preserved as reference]
        ↓
Phase 9 — SUB-HOURLY 10-YEAR DYNAMIC PHYSICS VALIDATION
  10_physics_validation.py       → pcm/physics_validation_results_assam.csv (24 simulations)
                                   (10-year sub-hourly Δt=300s/150s, 3 K=3 medoids, 8 historical PCMs,
                                    4-state path-dependent enthalpy, First-Law error = 0.0000%,
                                    SSRD duration-overlap error = 0.000000%, 100% spin-up convergence)
        ↓
Phase 10 — VALIDATION & COMPARISON: MCDM VS. PHYSICS
  10_validation_comparison.py    → Dual-level assessment: Level 1 confirms K=3 MCDM NOT PERFORMED;
                                   Level 2 retrospective comparison reveals negative rank correlation
                                   (Spearman ρ = -0.52 to -0.64, Top-1 agreement = 0.0%).
                                   Scientific verdict: NOT PHYSICALLY SUPPORTED.
        ↓
Phase 11 — FINAL OUTPUTS AUDIT & CONSOLIDATION
  consolidate_final_outputs.py   → final_output_manifest.csv (31 entries: 27 Active, 4 Historical)
  generate_phase11_figures.py    → 10 publication-ready tables, 10 publication-ready figures
  final_project_verification.py  → Automated master test suite (PASSED 100%)
```

---

## Status at a Glance Across All Phases

| Phase | Core Scripts | Status | Key Authoritative Metric / Outcome |
|---|---|---|---|
| **Phase 1: Spatial Grid** | `00a`, `00b`, `01`, `01b` | **COMPLETE** | 129 points (`ASP_0001`–`ASP_0129`), 87.8% population coverage |
| **Phase 2: Preprocessing** | `02`, `02b`, `03b` | **COMPLETE** | 467,367 daily rows; BACKBONE decision (1.1% GHI MBE) |
| **Phase 2.5: Quality Control** | `04` | **COMPLETE** | 129 parquet files; IsolationForest multivariate outlier flagging |
| **Phase 3: Climate Clustering** | `04b`, `05` | **LOCKED (FINAL)** | K=3 GMM (full covariance, 5 features); min BIC=1574.94; ARI=0.6289 |
| **Phase 4: SWH Specification** | `05` | **COMPLETE** | 50 kg PCM, 100 kg water, 100 L/day demand, Tm_target=44.0°C |
| **Phase 5: PCM Database** | `06` | **LOCKED (FINAL)** | 58 PCMs × 41 columns (`pcm_database_final.csv`); strict provenance |
| **Phase 6: Feasibility** | `07` | **GOVERNED** | Final K=3: $n_{\text{confirmed}}=[0,0,0]$; 1 conditional (`n-Tetracosane C24`) |
| **Phase 7: MCDM Ranking** | `08` | **GOVERNED** | Final K=3: **NOT PERFORMED**; Historical K=4 preserved as reference |
| **Phase 8: Monte Carlo** | `08` | **GOVERNED** | Final K=3: **SKIPPED** ($n_{\text{draws}}=0$); Historical K=4 5k-draw reference |
| **Phase 9: Physics Validation** | `10` | **COMPLETE** | 10-year sub-hourly simulation; First-Law error = 0.0000%; 24 runs |
| **Phase 10: Comparison** | `10_val` | **COMPLETE** | Scientific verdict: **NOT PHYSICALLY SUPPORTED** ($\rho = -0.52$ to $-0.64$) |
| **Phase 11: Consolidation** | `final_proj` | **VERIFIED** | 31 manifest items, 10 thesis tables, 10 figures; 100% test pass |

---

## Critical Methodological Distinctions

### 1. Final K=3 Model vs. Historical K=4 Pipeline Version
- **Historical Pipeline ($K=4$)**: Initially, an exploratory GMM clustering yielded 4 clusters on 128 preliminary grid points. A preliminary 25-row PCM database (`pcm_database_assam.csv`) underwent feasibility screening, resulting in 8 survivors. Formal MCDM (TOPSIS, GRA, PROMETHEE II, VIKOR) ranked `RT44HC` #1 across all 4 clusters.
- **Final Locked Pipeline ($K=3$)**: Rigorous BIC evaluation across 5 core physical features (`GHI_mean`, `Ta_mean`, `DTR`, `RH_mean`, `wind_mean`) confirmed an absolute BIC minimum at **$K=3$** ($\text{BIC}=1574.94$), with mean bootstrap ARI = 0.6289. The spatial grid was finalized at **129 points**.
- **Governance Boundary**: Under final $K=3$ cluster forcing and the audited 58-row PCM database without arbitrary relaxation, confirmed feasible candidates $n_{\text{confirmed}} = [0, 0, 0]$. Consequently, formal $K=3$ MCDM ranking was **`NOT PERFORMED`**, and Monte Carlo was **`SKIPPED`**. Historical $K=4$ outputs are explicitly labeled as **historical pre-audit artifacts**.

### 2. Candidate Universes: Historical Survivors vs. Conditional Candidate
- **Historical 8-PCM Survivor Set**: Extracted from the historical $K=4$ screening and passed into the Phase 9 dynamic physics simulation:
  1. `Myristic-Palmitic eutectic (58/42)` ($T_m = 42.6^\circ\text{C}$)
  2. `RT44HC` ($T_m = 43.0^\circ\text{C}$)
  3. `savE® OM42` ($T_m = 44.0^\circ\text{C}$)
  4. `C22H46 (docosane-class paraffin)` ($T_m = 44.5^\circ\text{C}$)
  5. `savE® OM46` ($T_m = 47.0^\circ\text{C}$)
  6. `RT45HC` ($T_m = 47.0^\circ\text{C}$)
  7. `savE® OM50` ($T_m = 50.0^\circ\text{C}$)
  8. `savE® OM48` ($T_m = 51.0^\circ\text{C}$)
  *(Note: Neither `RT47` nor `n-Tetracosane` belongs to this historical 8-PCM simulation set).*
- **Final K=3 Conditional Candidate**: In the audited 58-row database under $K=3$ forcing, `n-Tetracosane (C24)` ($T_m = 52.0^\circ\text{C}$) qualified as a **Conditional candidate** in Cluster 0 ($L = 255.0\text{ kJ/kg} \ge L_{\text{req}} = 252.0\text{ kJ/kg}$), but is not an MCDM-ranked PCM.

### 3. Independent Physics Validation & Phase 10 Finding
- Phase 9 simulates a 100 kg water + 50 kg PCM storage tank driven by true 10-year sub-hourly ERA5 climate forcing at the 3 final medoids (`ASP_0012`, `ASP_0092`, `ASP_0028`).
- Incorporates a 4-state path-dependent enthalpy model with supercooling hysteresis and duration-overlap SSRD reconstruction (First-Law cumulative error $= 0.0000\%$).
- Phase 10 compared the historical MCDM rankings against the physical solar fraction and delivery performance. The result demonstrated **negative rank correlation** (delivery-rank Spearman $\rho = -0.52$ to $-0.64$, Top-1 agreement $= 0.0\%$, Top-3 overlap $= 0.0\%$).
- **Scientific Verdict**: **`NOT PHYSICALLY SUPPORTED`**. This divergence is physically explained by the system's operational threshold: delivering domestic water at $50.0^\circ\text{C}$ favors PCMs melting close to $50^\circ\text{C}$ (`savE OM48`, $T_m=51.0^\circ\text{C}$), whereas the MCDM Gaussian fitness penalized candidates deviating from the $44.0^\circ\text{C}$ target.

---

## Authoritative Master Deliverables (Phase 11)

The consolidated deliverables are indexed in `final_output_manifest.csv` (31 total entries: 27 Active/Final, 4 Locked Historical):

### Consolidated Thesis Tables (`final_outputs/tables/`)
1. `table01_climate_signatures.csv`: Summary statistics of 18 climate signature indices across 129 points.
2. `table02_pca_loadings.csv`: Principal component loadings for the thermodynamic index block.
3. `table03_gmm_selection.csv`: Clustering diagnostic metrics ($K=2$ to $K=6$) establishing minimum BIC at $K=3$.
4. `table04_cluster_profiles_k3.csv`: Final $K=3$ regime profiles, population distributions, and medoids.
5. `table05_pcm_database_summary.csv`: Summary of the 58-row curated PCM database by family and status.
6. `table06_feasibility_survivors.csv`: Feasibility screening status showing $n_{\text{confirmed}}=[0,0,0]$ and 1 conditional candidate.
7. `table07_historical_mcdm_rankings_k4.csv`: Historical pre-audit $K=4$ MCDM rankings (Borda, TOPSIS, GRA, PROMETHEE, VIKOR).
8. `table08_monte_carlo_stability_k3.csv`: Governance record confirming $K=3$ Monte Carlo was skipped ($n_{\text{draws}}=0$).
9. `table09_physics_performance_k3.csv`: 10-year sub-hourly dynamic simulation results for 8 PCMs across 3 medoids.
10. `table10_mcdm_vs_physics_comparison.csv`: Dual-level comparison metrics, Spearman correlations, and divergence diagnostics.

### Consolidated Thesis Figures (`final_outputs/visuals/`)
1. `fig01_gmm_bic_selection.png`: GMM BIC model selection curve showing global minimum at $K=3$.
2. `fig02_gmm_silhouette_curve.png`: Silhouette coefficient curve across candidate cluster counts.
3. `fig03_gmm_davies_bouldin.png`: Davies-Bouldin cluster separation index.
4. `fig04_gmm_calinski_harabasz.png`: Calinski-Harabasz variance ratio criterion.
5. `fig05_mcdm_vs_delivery_rank.png`: Rank comparison scatter plot: MCDM rank vs. physics delivery temperature rank.
6. `fig06_mcdm_vs_solar_fraction_rank.png`: Rank comparison scatter plot: MCDM rank vs. physics solar fraction rank.
7. `fig07_mcdm_vs_cycling_rank.png`: Rank comparison scatter plot: MCDM rank vs. annual thermal cycling count.
8. `fig08_tm_vs_physics_delivery_mechanism.png`: Thermodynamic mechanism plot demonstrating why $T_m \approx 51^\circ\text{C}$ outperforms $T_m = 44^\circ\text{C}$.
9. `fig09_final_k3_climate_regime_map.png` (and interactive `.html`): Publication-grade geographic map of the 129 grid points colored by final $K=3$ regime.
10. `fig10_final_k3_pca_projection.png`: 2D PCA projection of the 129 points showing the $K=3$ GMM regime ellipsoids.
