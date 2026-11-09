# 00 — Master Overview: ERA5 Assam Climate → PCM Selection Pipeline

## Corrections applied 2026-11-10

Source: `.claude/Objective_1_and_Objective_2_corrected_methodology_.docx`, each item checked against the file named. **Documentation and comments only:** no pipeline was re-run and no result file changed. Assam is the **legacy** O1 variant (its own 11-phase pipeline); it is not on the Rajasthan method.

### Delivery and melting-target basis (versioned table, v2026-11-10)

| State | O1 delivery basis | O1 Tm target (nominal) | Capped targets actually used for filtering | O2 delivery target | PCM limit |
|---|---|---|---|---|---|
| Assam | 50 C, own constants (`era5-assam/04b_climate_signature.py:59`, `10_physics_validation.py:105`); NOT imported from `pcm_shared_config.py`, which says 60 C | 56 C (`04b_climate_signature.py:61`; `04c_charging_ceiling_assam.py:48`; `10_physics_validation.py:106`; `data/processed/design/swh_design_specification.csv`: 50 + 6 K) | 50.31 / 49.36 / 49.41 C for clusters 0/1/2 (`Tm_target_capped_C`, `data/processed/clustering/cluster_profiles_assam.csv`) | 45 C (`OBJECTIVE2/Objective---2/objective2-assam/configs/system_config_shared.yaml:54`) | 65 C (same file, line 50) |

Two different things are called "delivery". Here O1 uses 50 C both as the sizing basis and as the success threshold in its physics run (a draw counts only if the tank reaches 50 C), while O2 counts hot water as delivered above 45 C. The Rajasthan / Tamil Nadu O1 basis is 60 C (`pcm_shared_config.py:50`). The capped targets used here (49.4-50.3 C) are below the 50 C delivery floor, which the current physics report itself flags as a reason delivery ranks need not follow the MCDM ranking. The 56 C nominal target is below the 65 C PCM limit. Whether Assam adopts the 60 C basis needs a full re-run and is an open decision (b).

### What differs from the files (earlier text corrected where edited, otherwise flagged with a banner)

- **K and covariance.** `05_cluster_assam.py:95` has `K_FINAL = 3` and `covariance_type="diag"` (lines 162, 210, 272). "Full covariance" in earlier text is wrong for the current script.
- **BIC / stability.** The on-disk `data/processed/clustering/bic_selection_assam.csv` gives BIC 1030.97 at K = 3; its lowest BIC over K = 2-10 is at K = 9 (983.63), so K = 3 is not the BIC minimum and is a chosen constant, which is consistent with the caveat that BIC picks a density model, not necessarily a meaningful number of climate regimes (Baudry et al. 2010; see also Baldo and Locatelli 2022; Babanov, Semenov and Mokhov 2023). `gmm_bootstrap_stability.csv`: mean bootstrap ARI 0.694 (500 bootstraps). The earlier BIC = 1574.94 / ARI = 0.6289 are not reproduced by any file on disk.
- **Tm target.** 56 C nominal (50 + 6), not 44 C (earlier text).
- **Physics run.** `data/processed/pcm/physics_validation_assam.csv` has 108 rows (36 candidates x 3 medoids), not 24 runs. The comparison with MCDM (`data/preprocessed/validation_comparison_report.txt`, 2026-10-28) ranks 35 candidates per cluster and gives Spearman rho against the delivery rank -0.284 / -0.300 / -0.281 (p = 0.098 / 0.080 / 0.102) and against the solar-fraction rank +0.626 / +0.621 / +0.639 (p <= 0.0001), Top-1 agreement 0 of 3, Top-3 overlap 0 of 3; the report's own verdict is "partially supported / inconclusive". The "-0.52 to -0.64 / NOT PHYSICALLY SUPPORTED" figures in the older documents (`23_PHASE_10_AUDIT.md`: -0.5238, -0.5238, -0.6429 over 8 candidates) are not reproduced by the current files; a scratch recomputation from `final_outputs/tables/table10_mcdm_vs_physics_comparison.csv` (9 candidates per cluster) also gives different values, so that table and the report disagree with each other too. **Resolved:** `final_outputs/tables/table10` and `table09` are plain copies in `generate_phase11_consolidation.py` of `data/processed/pcm/mcdm_vs_physics_comparison.csv` and `physics_validation_assam.csv`, but the files on disk in `final_outputs/tables/` are older (medoids ASP_0012 / ASP_0092 / ASP_0028, 9 candidates per cluster, 15 physics rows) than their sources (medoids ASP_0003 / ASP_0036 / ASP_0080, 35 candidates per cluster, 108 rows). The current report's numbers reproduce exactly from the current source (35 candidates per cluster, delivery rho -0.284 / -0.300 / -0.281); the `final_outputs/` tables, and the docs that quote them, are stale and need regenerating with `generate_phase11_consolidation.py`.
- **No K = 3 MCDM ranking exists** (`data/processed/pcm/mcdm_cluster_eligibility_summary.csv`: confirmed = 0 in all three clusters, "NOT PERFORMED"). The physics comparison is therefore against the **historical K = 4 pre-audit** ranking (limitations 1-3 of the report itself). It tests no ranking O1 currently stands behind, and does not show that any ranking is "physics-validated".
- **Weights.** `08_mcdm_ranking_final.py:66-75` defines fixed stated priors (`PRIOR_WEIGHTS`, f_Tm 0.30 / latent-heat margin 0.25 / rho_H 0.15 / k 0.15 / cycles 0.15), commented "Equal-Prior AHP Weights"; no pairwise elicitation exists, so never "AHP-weighted". The weight columns in the historical `mcdm_full_scores_assam.csv` vary slightly by cluster (f_Tm 0.341 / 0.344 / 0.342), so which blend that run used is not verified.
- **Point count resolved: 80, not 129.** Current files: `data/processed/population_grid_points.csv` 80 rows, `clustering/cluster_assignments_assam.csv` 80 rows, regimes 27 / 28 / 25, physics medoids ASP_0003 / ASP_0036 / ASP_0080. The 129 points (`ASP_0001`-`ASP_0129`, 87.8 %) in these documents and in the O2 Assam input copy (33 / 61 / 35, medoids ASP_0012 / ASP_0092 / ASP_0028) belong to an older grid; the docs written before the re-run were not updated. Population coverage of the 80 points: not verified.
- **Population.** "Population-weighted point selection (80 points in the current files; 129 in older text) and cluster profiles", never "population-weighted clustering": `sample_weight` appears nowhere in `05_cluster_assam.py`.
- **External validation.** No Koppen-Geiger or NBC/ECBC comparison exists for Assam (no such files); only the internal bootstrap ARI above.
- **Which PCMs O2 Assam used.** Not from an O1 MCDM ranking. `OBJECTIVE2/Objective---2/objective2-assam/configs/states/assam.yaml` (lines 12-20) states that O1 produced no confirmed-feasible K = 3 ranking, that the historical K = 4 ranking was set aside, and that the shortlist (savE OM48, savE OM50, savE OM46) comes from the O1 Phase 9/10 10-year physics-model performance (OM48 first and OM50 second on delivery). That is a ranking by the physics model, not a validation of those PCMs.

### Already correct (verified, no edit)

Final K = 3 produced no confirmed survivors and no formal MCDM or Monte Carlo (`mcdm_cluster_eligibility_summary.csv`); the historical K = 4 outputs are labelled as historical; 129 population-weighted points; the "NOT PERFORMED" governance rule.

### Still open

(b) 60 C versus 50 C basis; (c) re-run Assam on the current method or keep it labelled legacy; the physics comparison and its older documents disagree and need regenerating together; no external classification validation.

---

## Project Objective

Final-year B.Tech CSE project (Group 12, Amrita School of Engineering, Guide: Dr. T. Deepika):
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating."** 

Objective 1 builds an end-to-end **climate-region-aware PCM recommendation and validation framework**: transforming 10 years of reanalysis climate data (sampled at population-weighted points) into climate regimes, deriving engineering performance targets per regime, screening candidate phase-change materials against those targets, and conducting an independent, multi-year dynamic physics validation.

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
  04b_climate_signature.py       → climate_signatures_raw.csv (19 indices across 129 sites, all
                                   computed from event-sampled ERA5 data — see 02_DATA_SOURCES_AND_VARIABLES.md)
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
                                   (older text: Spearman ρ = -0.52 to -0.64; current report: about -0.3 against delivery rank, about +0.6 against solar-fraction rank; Top-1 agreement = 0.0%).
                                   Report verdict now: partially supported / inconclusive (older text: NOT PHYSICALLY SUPPORTED).
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
| **Phase 3: Climate Clustering** | `04b`, `05` | **LOCKED (FINAL)** | K=3 GMM (diagonal covariance, 5 features; `05_cluster_assam.py:95,162`); on-disk BIC at K=3 = 1030.97, mean bootstrap ARI = 0.694 (earlier text: 1574.94 / 0.6289, not reproduced) |
| **Phase 4: SWH Specification** | `05` | **COMPLETE** | 50 kg PCM, 100 kg water, 100 L/day demand, Tm_target=56.0°C nominal (49.4-50.3°C capped; earlier text said 44.0°C) |
| **Phase 5: PCM Database** | `06` | **LOCKED (FINAL)** | 58 PCMs × 41 columns (`pcm_database_final.csv`); strict provenance |
| **Phase 6: Feasibility** | `07` | **GOVERNED** | Final K=3: $n_{\text{confirmed}}=[0,0,0]$; 1 conditional (`n-Tetracosane C24`) |
| **Phase 7: MCDM Ranking** | `08` | **GOVERNED** | Final K=3: **NOT PERFORMED**; Historical K=4 preserved as reference |
| **Phase 8: Monte Carlo** | `08` | **GOVERNED** | Final K=3: **SKIPPED** ($n_{\text{draws}}=0$); Historical K=4 5k-draw reference |
| **Phase 9: Physics Validation** | `10` | **COMPLETE** | 10-year sub-hourly simulation; First-Law error = 0.0000%; 108 rows on disk (36 candidates x 3 medoids; earlier text said 24 runs) |
| **Phase 10: Comparison** | `10_val` | **COMPLETE** | Report verdict now "partially supported / inconclusive" (older text: **NOT PHYSICALLY SUPPORTED**); no K=3 MCDM ranking exists to validate ($\rho = -0.28$ to $-0.30$ against the delivery rank and $+0.62$ to $+0.64$ against the solar-fraction rank (current `validation_comparison_report.txt`, 2026-10-28; earlier text said $-0.52$ to $-0.64$)) |
| **Phase 11: Consolidation** | `final_proj` | **VERIFIED** | 31 manifest items, 10 thesis tables, 10 figures; 100% test pass |

---

## Critical Methodological Distinctions

### 1. Final K=3 Model vs. Historical K=4 Pipeline Version
- **Historical Pipeline ($K=4$)**: Initially, an exploratory GMM clustering yielded 4 clusters on 128 preliminary grid points. A preliminary 25-row PCM database (`pcm_database_assam.csv`) underwent feasibility screening, resulting in 8 survivors. Formal MCDM (TOPSIS, GRA, PROMETHEE II, VIKOR) ranked `RT44HC` #1 across all 4 clusters.
- **Final Locked Pipeline ($K=3$)**: Rigorous BIC evaluation across 5 core physical features (`GHI_mean`, `Ta_mean`, `DTR`, `RH_mean`, `wind_mean`) selected **$K=3$** as `K_FINAL` (on-disk BIC at K=3 = 1030.97, not the BIC minimum; mean bootstrap ARI = 0.694; earlier text said BIC = 1574.94, ARI = 0.6289, which no file reproduces). The spatial grid was finalized at **129 points**.
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
- Phase 10 compared the historical MCDM rankings against the physical solar fraction and delivery performance. The result demonstrated **negative rank correlation** (delivery-rank Spearman $\rho = -0.28$ to $-0.30$ against the delivery rank and $+0.62$ to $+0.64$ against the solar-fraction rank (current `validation_comparison_report.txt`, 2026-10-28; earlier text said $-0.52$ to $-0.64$), Top-1 agreement $= 0.0\%$, Top-3 overlap $= 0.0\%$).
- **Scientific Verdict (corrected 2026-11-10)**: the current report says **partially supported / inconclusive**: delivery-rank rho is negative and not significant, solar-fraction rho is positive and significant, Top-1/Top-3 agreement is 0, and the ranking compared is the historical K=4 one. This divergence is physically explained by the system's operational threshold: delivering domestic water at $50.0^\circ\text{C}$ favors PCMs melting close to $50^\circ\text{C}$ (`savE OM48`, $T_m=51.0^\circ\text{C}$), whereas the MCDM Gaussian fitness penalized candidates deviating from its Tm target (44.0 C in the historical run; 49.4-50.3 C capped now).

---

## Authoritative Master Deliverables (Phase 11)

The consolidated deliverables are indexed in `final_output_manifest.csv` (31 total entries: 27 Active/Final, 4 Locked Historical):

### Consolidated Thesis Tables (`final_outputs/tables/`)
1. `table01_climate_signatures.csv`: Summary statistics of 19 climate signature indices across 129 points.
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
1. `k_selection_bic.png`: GMM BIC model selection curve showing global minimum at $K=3$.
2. `k_selection_silhouette.png`: Silhouette coefficient curve across candidate cluster counts.
3. `k_selection_davies_bouldin.png`: Davies-Bouldin cluster separation index.
4. `k_selection_calinski_harabasz.png`: Calinski-Harabasz variance ratio criterion.
5. `11b_physics_vs_mcdm_delivery_rank.png`: Rank comparison scatter plot: MCDM rank vs. physics delivery temperature rank.
6. `11b_physics_vs_mcdm_solar_fraction_rank.png`: Rank comparison scatter plot: MCDM rank vs. physics solar fraction rank.
7. `11b_physics_vs_mcdm_cycling_rank.png`: Rank comparison scatter plot: MCDM rank vs. annual thermal cycling count.
8. `11c_tm_vs_physics_delivery.png`: Thermodynamic mechanism plot demonstrating why $T_m \approx 51^\circ\text{C}$ outperforms $T_m = 44^\circ\text{C}$.
9. `cluster_map.png` (and interactive `.html`): Publication-grade geographic map of the 129 grid points colored by final $K=3$ regime.
10. `pca_projection.png`: 2D PCA projection of the 129 points showing the $K=3$ GMM regime ellipsoids.
