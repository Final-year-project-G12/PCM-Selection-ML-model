# 00 — Master Overview: ERA5 Assam Climate → PCM Selection Pipeline

 
> **CURRENT DATASET UPDATE (2026-11-11):** This overview reflects the latest Assam Objective 1 K=3 chain verified in the read-only report. The active spatial grid contains 80 points (`ASP_0001`–`ASP_0080`), not the historical 129-point grid. Latest GMM covariance is diagonal; latest medoids are `ASP_0003`, `ASP_0036`, and `ASP_0080`. Legacy K=4 artifacts are retained only where explicitly labelled historical. The latest K=3 Monte Carlo run is verified as performed (5,000 draws per cluster); the latest K=3 MCDM–physics comparison remains **NOT VERIFIED** because the on-disk comparison artifacts retain legacy medoid IDs.

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
  00a_build_population_grid.py   → population_grid_points.csv (80 points, 87.8% population coverage)
  00b_build_suntimes.py          → suntimes.csv (solar noon, sunrise, sunset for 10 years)
  01_download_era5_assam.py      → data/raw/era5/points/*.nc (hourly ERA5 reanalysis)
  01b_download_nasapower.py      → data/raw/nasapower/*.json (10 years hourly NASA POWER)
        ↓
Phase 2 — CLIMATE PREPROCESSING & CROSS-SOURCE VALIDATION
  02_combine_assam.py            → climate_assam_points.csv (solar geometry, unit conversions)
  02b_build_daily_aggregates_assam.py → daily_aggregates_assam.csv (292,160 rows; 80 points × 3,652 days, 2016-01-02 to 2025-12-31)
  03b_agreement_analysis_assam.py → bias_decision_assam.txt (BACKBONE decision, 1.1% GHI MBE)
        ↓
Phase 2.5 — QUALITY CONTROL & OUTLIER DETECTION
  04_preprocess_assam.py         → preprocessed/parquet/{point_id}.parquet (80 files)
                                   (Physical bounds checks, IsolationForest multivariate flagging)
        ↓
Phase 3 — CLIMATE REGIME CLUSTERING (LOCKED K=3 MODEL)
  04b_climate_signature.py       → climate_signatures_raw.csv (19 indices across 80 sites, all
                                   computed from event-sampled ERA5 data — see 02_DATA_SOURCES_AND_VARIABLES.md)
  05_cluster_assam.py            → clustering/cluster_assignments_assam.csv (K=3 GMM, diagonal covariance)
                                   (5 core features: GHI_mean, Ta_mean, DTR, RH_mean, wind_mean;
                                    min BIC=1030.97 at K=3; mean bootstrap ARI=0.6939 (median=0.7255); medoids: ASP_0003, ASP_0036, ASP_0080)
        ↓
Phase 4 — SOLAR WATER HEATING (SWH) DESIGN SPECIFICATION
  05_cluster_assam.py            → clustering/cluster_profiles_assam.csv
                                   (50 kg PCM, 100 kg water, 100 L/day demand, 50 L AM + 50 L PM,
                                    nominal Tm_target = 56.0°C; capped targets: C0=49.659°C, C1=49.395°C, C2=49.437°C; approach ΔT = 6 K, T_delivery = 50.0°C)
        ↓
Phase 5 — CURATED PCM PROPERTY DATABASE
  06_build_pcm_database.py       → pcm/pcm_database_final.csv (58 deduplicated PCMs × 41 columns)
                                   (Strict provenance: source_type, value_status; strict Cp average)
                                   [Historical prototype: pcm_database_assam.csv, 25 rows]
        ↓
Phase 6 — FEASIBILITY FILTERING ENGINE
  07_feasibility_filter.py       → Final K=3: n_confirmed = [2, 2, 2]; n_deployable_survivors = [5, 5, 5].
                                   [Historical K=4 screening preserved separately as a legacy result.]
        ↓
Phase 7 — MULTI-CRITERIA DECISION MAKING (MCDM)
  08_mcdm_ranking.py             → K=3 MCDM PERFORMED; consensus Top-1 = savE® OM49 in all 3 clusters.
                                   [Historical K=4 rankings retained only as historical reference.]
        ↓
Phase 8 — MONTE CARLO UNCERTAINTY ANALYSIS
  08b_monte_carlo_stability_final.py → monte_carlo_stability_assam.csv (15 rows; 5 PCMs × 3 clusters; 5,000 draws per cluster; all COMPLETED; Top-1 retention 86.6%; mean Spearman ρ=0.7601)
        ↓
Phase 9 — SUB-HOURLY 10-YEAR DYNAMIC PHYSICS VALIDATION
  10_physics_validation.py       → pcm/physics_validation_assam.csv (15 simulations)
                                   (5 K=3 deployable survivors × 3 medoids ASP_0003/ASP_0036/ASP_0080;
                                    4-state path-dependent enthalpy, First-Law error = 0.0000%,
                                    SSRD duration-overlap error = 0.000000%, 100% spin-up convergence)
        ↓
Phase 10 — VALIDATION & COMPARISON: MCDM VS. PHYSICS
  10_validation_comparison.py    → Dual-level assessment: K=3 comparison currently NOT VERIFIED: comparison artifacts retain legacy medoid IDs.
                                   Do not report correlation ranges or a scientific verdict until comparison is regenerated/validated.
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
| **Phase 1: Spatial Grid** | `00a`, `00b`, `01`, `01b` | **COMPLETE** | 80 points (`ASP_0001`–`ASP_0080`), 87.8% population coverage |
| **Phase 2: Preprocessing** | `02`, `02b`, `03b` | **COMPLETE** | 292,160 daily rows (80 points × 3,652 days; 2016-01-02 to 2025-12-31); BACKBONE decision (1.1% GHI MBE) |
| **Phase 2.5: Quality Control** | `04` | **COMPLETE** | 80 parquet files; IsolationForest multivariate outlier flagging |
| **Phase 3: Climate Clustering** | `04b`, `05` | **LOCKED (FINAL)** | K=3 GMM (diagonal covariance, 5 features); BIC=1030.97; mean ARI=0.6939, median ARI=0.7255; clusters=27/28/25 |
| **Phase 4: SWH Specification** | `05` | **COMPLETE** | 50 kg PCM, 100 kg water, 100 L/day demand; nominal Tm_target=56.0°C (cluster-capped values below) |
| **Phase 5: PCM Database** | `06` | **LOCKED (FINAL)** | 58 PCMs × 41 columns (`pcm_database_final.csv`); strict provenance |
| **Phase 6: Feasibility** | `07` | **COMPLETE** | $n_{\text{confirmed}}=[2,2,2]$; 5 deployable survivors per cluster |
| **Phase 7: MCDM Ranking** | `08` | **PERFORMED** | K=3 consensus Top-1 = savE® OM49 in all 3 clusters |
| **Phase 8: Monte Carlo** | `08` | **PERFORMED** | 5,000 draws per cluster across all 3 clusters; 15 completed rows; savE® OM49 Top-1 retention 86.6%; mean Spearman ρ=0.7601 |
| **Phase 9: Physics Validation** | `10` | **COMPLETE** | Latest file `physics_validation_assam.csv`: 15 rows (5 PCMs × 3 medoids) |
| **Phase 10: Comparison** | `10_val` | **NOT VERIFIED** | Comparison files retain legacy medoid IDs; latest K=3 correlations/verdict not established |
| **Phase 11: Consolidation** | `final_proj` | **VERIFIED** | 31 manifest items, 10 thesis tables, 10 figures; 100% test pass |

---

## Critical Methodological Distinctions

### 1. Final K=3 Model vs. Historical K=4 Pipeline Version
- **Historical Pipeline ($K=4$)**: Initially, an exploratory GMM clustering yielded 4 clusters on 128 preliminary grid points. A preliminary 25-row PCM database (`pcm_database_assam.csv`) underwent feasibility screening, resulting in 8 survivors. Formal MCDM (TOPSIS, GRA, PROMETHEE II, VIKOR) ranked `RT44HC` #1 across all 4 clusters.
- **Final Locked Pipeline ($K=3$)**: Rigorous BIC evaluation across 5 core physical features (`GHI_mean`, `Ta_mean`, `DTR`, `RH_mean`, `wind_mean`) confirmed an absolute BIC minimum at **$K=3$** ($\text{BIC}=1030.97$), with mean bootstrap ARI = 0.6939 and median ARI = 0.7255. The spatial grid was finalized at **80 points**.
- **Governance Boundary**: Under final $K=3$ cluster forcing, confirmed feasible candidates are $n_{\text{confirmed}} = [2, 2, 2]$, with 5 deployable survivors per cluster. K=3 MCDM was **PERFORMED**, with savE® OM49 consensus rank 1 in all clusters. Monte Carlo was **PERFORMED** for the latest K=3 pipeline: 5,000 draws per cluster across all three clusters, with 86.6% Top-1 retention for savE® OM49 and mean Spearman ρ=0.7601. Historical $K=4$ outputs are explicitly labeled as **historical pre-audit artifacts**.

### 2. Candidate Universes: Historical K=4 Set vs. Current K=3 Survivors
- **Historical 8-PCM Survivor Set (K=4 legacy only)**: Extracted from the historical $K=4$ screening; this list is retained for traceability and is NOT the current K=3 physics input:
  1. `Myristic-Palmitic eutectic (58/42)` ($T_m = 42.6^\circ\text{C}$)
  2. `RT44HC` ($T_m = 43.0^\circ\text{C}$)
  3. `savE® OM42` ($T_m = 44.0^\circ\text{C}$)
  4. `C22H46 (docosane-class paraffin)` ($T_m = 44.5^\circ\text{C}$)
  5. `savE® OM46` ($T_m = 47.0^\circ\text{C}$)
  6. `RT45HC` ($T_m = 47.0^\circ\text{C}$)
  7. `savE® OM50` ($T_m = 50.0^\circ\text{C}$)
  8. `savE® OM48` ($T_m = 51.0^\circ\text{C}$)
  *(Note: Neither `RT47` nor `n-Tetracosane` belongs to this historical 8-PCM simulation set).*
- **Current K=3 deployable survivor set**: Five candidates per cluster passed the current governance summary. The consensus Top-1 is `savE® OM49` in all three clusters. The complete per-cluster survivor membership should be taken from the current feasibility/MCDM CSVs, not inferred from the historical list above.

### 3. Independent Physics Validation & Phase 10 Finding
- The latest Phase 9 result file is `physics_validation_assam.csv` with 15 rows (5 current K=3 survivors × 3 medoids: `ASP_0003`, `ASP_0036`, `ASP_0080`). `physics_validation_results_assam.csv` is a legacy K=4 file and must not be used as the latest result.
- Incorporates a 4-state path-dependent enthalpy model with supercooling hysteresis and duration-overlap SSRD reconstruction (First-Law cumulative error $= 0.0000\%$).
- Phase 10 comparison is **NOT VERIFIED for the latest K=3 run**. The available comparison table still contains the old medoid IDs (`ASP_0012`, `ASP_0092`, `ASP_0028`). Do not quote correlation values, Top-1/Top-3 agreement, or a scientific verdict until the comparison is regenerated or its exact inputs are confirmed.
- **Scientific Verdict**: **PENDING**. The old `NOT PHYSICALLY SUPPORTED` verdict and its $ρ=-0.52$ to $-0.64$ range belong to a legacy K=4 comparison and are not valid conclusions for the current K=3 chain.

---

## Current verification caveats (do not remove before final audit)

1. **Daily aggregate row count:** `daily_aggregates_assam.csv` contains **292,160 data rows** (292,161 lines including the header), covering 80 points from 2016-01-02 to 2025-12-31 (3,652 days). The former 467,367-row claim is incorrect.
2. **Monte Carlo:** The latest K=3 run is verified as performed: `monte_carlo_stability_assam.csv` contains 15 completed rows (5 PCMs × 3 clusters), with 5,000 draws per cluster. The report records 86.6% Top-1 retention for savE® OM49 and mean Spearman ρ=0.7601. The older `monte_carlo_stability_assam_stage1.csv` is the historical K=4 run.
3. **MCDM–physics comparison:** the comparison output still carries legacy medoid IDs (`ASP_0012`, `ASP_0092`, `ASP_0028`), whereas the current K=3 physics validation uses (`ASP_0003`, `ASP_0036`, `ASP_0080`). The reported legacy correlation range (delivery ρ = −0.52 to −0.64) and verdict must not be presented as current K=3 findings. Latest K=3 comparison metrics and scientific verdict are **NOT VERIFIED**.
4. **Figures:** `cluster_map.png` and `pca_projection.png` were identified as stale in the audit notes. Regenerate or verify them against the 80-point, diagonal-covariance K=3 assignments before using them in a thesis or presentation.
5. **Historical files:** `physics_validation_results_assam.csv` and historical K=4 MCDM artifacts are retained for traceability only. The current Phase 9 physics file is `physics_validation_assam.csv` (15 rows).

---

## Authoritative Master Deliverables (Phase 11)

The consolidated deliverables are indexed in `final_output_manifest.csv` (31 total entries: 27 Active/Final, 4 Locked Historical):

### Consolidated Thesis Tables (`final_outputs/tables/`)
1. `table01_climate_signatures.csv`: Summary statistics of 19 climate signature indices across 80 points.
2. `table02_pca_loadings.csv`: Principal component loadings for the thermodynamic index block.
3. `table03_gmm_selection.csv`: Clustering diagnostic metrics ($K=2$ to $K=6$) establishing minimum BIC at $K=3$.
4. `table04_cluster_profiles_k3.csv`: Final $K=3$ regime profiles (cluster sizes 27/28/25), population distributions, capped targets, and medoids.
5. `table05_pcm_database_summary.csv`: Summary of the 58-row curated PCM database by family and status.
6. `table06_feasibility_survivors.csv`: Feasibility screening status showing $n_{\text{confirmed}}=[2,2,2]$ and 5 deployable survivors per cluster.
7. `table07_historical_mcdm_rankings_k4.csv`: Historical pre-audit $K=4$ MCDM rankings (Borda, TOPSIS, GRA, PROMETHEE, VIKOR).
8. `table08_monte_carlo_stability_k3.csv`: K=3 Monte Carlo stability output (5,000 draws per cluster; all three clusters completed; savE® OM49 Top-1 retention 86.6%; mean Spearman ρ=0.7601).
9. `table09_physics_performance_k3.csv`: Latest K=3 physics validation: 15 rows (5 PCMs × 3 medoids).
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
9. `cluster_map.png` (and interactive `.html`): Publication-grade geographic map of the 80 grid points (verify figure freshness before publication) colored by final $K=3$ regime.
10. `pca_projection.png`: 2D PCA projection of the 80 points (verify figure freshness before publication) showing the $K=3$ GMM regime ellipsoids.
