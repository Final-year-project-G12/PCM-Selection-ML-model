# 00 — Master Overview: ERA5 Tamil Nadu Climate → PCM Selection Pipeline

## Project Objective
Final-year B.Tech CSE project:
**"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating"**
Objective 1 builds a **climate-region-aware PCM recommendation framework**: turning 10 years of reanalysis climate data into population-weighted climate regimes, deriving PCM performance targets per regime, and ranking candidate phase-change materials against those targets with an auditable, multi-method, uncertainty-aware pipeline.

Governing document: `Objective1_PCM_Climate_Framework_Plan_v3.docx` ("the framework doc"), version 3.0. The **Tamil Nadu pipeline is implemented from Phase 1 through Phase 8** and a complete 62-PCM run exists. Critical v3.0 bugs were corrected in **v3.1** (August 2026); three further blocking script/orchestrator errors were fixed in the **2026-09-07 reconciliation** (see "Corrected Issues" below and `12_FINAL_READINESS_REPORT.md`).

---

## What the ERA5 Pipeline Achieves
1. **Population-Weighted Sampling**: Samples Tamil Nadu at 133 population-weighted points (representing 87.5% of the state's population) to ensure findings are representative of where domestic demand actually resides.
2. **Double-Source Validation**: Pulls ERA5 reanalysis and NASA POWER satellite/model data for the same coordinates and times, validating one against the other.
3. **Two-Tier Climate Signature**: Redefines 10 years of hourly/daily data into instantaneous sun-event statistics (Tier 1) and true daily-integral indices (Tier 2).
4. **Climate Regimes (Level A & B)**: Clusters points into spatial climate regimes (Level A) using Gaussian Mixture Models (GMM) and performs seasonal sensitivity analysis (Level B).
5. **PCM Feasibility & Screening**: Filters the current 62-candidate database (55 manufacturer-derived + 7 literature) against physical, corrosion, and safety constraints. The feasibility CSV retains a full per-candidate audit; current runs retain 9-15 candidates per cluster.
6. **Multi-Criteria Decision Making (MCDM)**: Ranks feasibility survivors using four independent methods (TOPSIS, GRA, PROMETHEE II, VIKOR) with Monte Carlo uncertainty propagation.
7. **Grey-Box Physics Validation**: Solves a lumped-enthalpy tank simulation using backward Euler, driven by the real 10-year daily weather of each regime's medoid point, evaluating Spearman rank concordance.
8. **Recommendation Cards**: Generates markdown summary cards for each climate regime.

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
| **Phase 6 — MCDM Ranking** | **RG5**: Lack of predictive optimization under climatic uncertainty. | 4-method Borda + 5,000-draw Monte Carlo uncertainty propagation identifies Top-3 candidates robust to parameter uncertainty. | Chen et al. (2025); Chopra (2023) |
| **Phase 7 — Physics Validation** | **RG4**: Limited real-world experimental / dynamic physical validation. | Provides a grey-box lumped-enthalpy tank simulation (`UA_TANK_W_K=2.0`) to verify that MCDM rankings correlate with physical solar fraction. | Barqawi (2025) |
| **Phase 8 — Rec Cards** | **RG3**: Poor alignment with household demand. | Distills final recommendations into actionable cards aligned to domestic hot water profiles (300 L/day). | Odoi & Yorke (2025) |
| **Phase 9+ — DRL Controller** | **RG1**: Lack of real-time adaptive control. | Future work: DRL uses discovered regimes to optimize PCM charging online. | Emami (2026); Terfai (2025) |

---

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
  06_build_pcm_database.py      → pcm_database_tamilnadu.csv (55 manufacturer + 7 literature = 62 rows)
  07b_charging_feasibility.py   → (optional) adds Tm_target_C_regime_capped to cluster_profiles
  07_feasibility_filter.py      → feasibility_survivors_by_cluster.csv (8 Table-12 filters)
        ↓
Phase 6 — MULTI-CRITERIA RANKING ENGINE
  08_mcdm_ranking.py            → mcdm_topk_by_cluster.csv, mcdm_full_scores_by_cluster.csv, monte_carlo_stability.csv
        ↓
Phase 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py      → physics_validation_results.csv, physics_validation_spearman.csv (UA_TANK=2.0 W/K, v3.1)
        ↓
Phase 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py    → recommendation_cards.md

  run_all_tamilnadu.py          → runs the whole CORE chain above in dependency order in one command
```

---

## Phase 1–8 Status and Headline Findings

| Phase | Script(s) | Status | Headline Finding |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 133 points, 240 NetCDF files, 1330 NASA POWER JSON files. |
| 2 — Preprocessing & QA | `02`, `02b`, `03`, `03b`, `04`, `04c` | **COMPLETE (v3.1 fixes applied)** | Deaccumulation replaced with `accum_to_flux()`. Per-season quantile mapping in Step 2b. Re-run required for new outputs. |
| 3 — Climate Signature | `04b`, `04d` | **Analysis complete; clean re-run pending** | 300 L/day draw with `SHARE_PCM=0.5` (now defined in `config.py`); completed-run cluster targets ≈ 301-326 kJ/kg. |
| 4 — GMM Clustering | `05`, `05b`, `11` | **COMPLETE (v3.1 fixes applied)** | K=5 regimes, `covariance_type="diag"`. `11` (Level B) runs after Phase 6 — it reads `08`'s `mcdm_full_scores_by_cluster.csv`. |
| 5 — Feasibility | `06`, `07b`, `07` | **Analysis complete; clean re-run pending** | `06` builds 62 records (55 manufacturer + 7 literature); `07` audits all 62 per cluster, pass counts 15/9/13/13/9 for clusters 0-4. `06` input-path fixed 2026-09-07. |
| 6 — MCDM Ranking | `08` | **COMPLETE** | 4-method Borda + 5000-draw Monte Carlo. Consensus rank-1 = `n-Octacosane (C28)` in all 5 clusters. |
| 7 — Physics Validation | `10` | **Analysis complete; clean re-run pending** | `UA_TANK_W_K=2.0` active. Completed run: mean Spearman ρ = **+0.177** (per cluster −0.016/+0.717/+0.355/−0.171/−0.000); 24/59 sims in 54-84% band; cycles 3-260/yr. |
| 8 — Rec Cards | `09` | **Analysis complete; clean re-run pending** | Aggregates Phases 4–7 into `recommendation_cards.md` (5 cluster cards). |

---

## Corrected Issues

The five v3.0 critical bugs plus three blocking script/orchestrator errors found on 2026-09-07 are fixed in source:

1. **Deaccumulation** → `accum_to_flux()` in `02_combine_tamilnadu.py`
2. **Quantile mapping** → Step 2b in `04_preprocess_tamilnadu.py` + `03b_agreement_analysis.py`
3. **1000× flow rate** → 300 L/day in `04b_climate_signature.py` and `11_level_b_seasonal_analysis.py`
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
- [ ] **Phase 5**: `python 07b_charging_feasibility.py` → optional; adds `Tm_target_C_regime_capped`
- [ ] **Phase 5**: `python 07_feasibility_filter.py` → `feasibility_survivors_by_cluster.csv` (62 audited per cluster)
- [ ] **Phase 6**: `python 08_mcdm_ranking.py` → 5,000 MC draws; writes `mcdm_topk_by_cluster.csv`
- [ ] **Phase 7**: `python 10_physics_validation.py` → `UA_TANK=2.0 W/K`; writes `physics_validation_results.csv`
- [ ] **Phase 8**: `python 09_recommendation_cards.py` → `recommendation_cards.md`
- [ ] **Phase 4 Level B**: `python 11_level_b_seasonal_analysis.py` → runs LAST; reads `08` outputs

### Environmental Configuration & Random Seeds
- **pvlib version**: `pvlib >= 0.9`
- **CDS API**: `.cdsapirc` in `era5-tamilnadu/`
- **`config.py` PCM constants**: `SHARE_PCM = 0.5`; `latent_heat_floor_kj_kg(l_required, fraction=0.7, absolute_min_kj_kg=100.0)`
- **Random seeds**: `KMeans(random_state=42)`, `GaussianMixture(random_state=42)`, `run_monte_carlo(seed=42)`
- **Python dependencies**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`

---

## Data-Layout Note (Clean Re-run Pending)
The completed 62-PCM run's artifacts (feasibility, MCDM, Monte Carlo, physics, cards, Level B) currently sit in the **non-canonical** `era5-tamilnadu/data/processed/processed/` tree; the canonical `data/processed/` tree still holds a superseded 25-PCM run (`pcm_database_tamilnadu.csv` = 25 rows there, no physics/cards). No script reads `processed/processed/`. One clean re-run of the CORE chain (now that Issues 6–8 are fixed) regenerates everything in the canonical location; the stale files can then be deleted. Read run-specific numbers from `processed/processed/` until then.

---

## Still Open
See `12_FINAL_READINESS_REPORT.md`: PCM database expansion, external cluster validation, elevation proxy, monsoon precipitation download, full Level-B GMM.

## Plot Documentation
See `11_PLOTS_GUIDE.md` for the interpretation and exact location of plots produced by raw QA, preprocessing, climate-signature, clustering, comprehensive, Objective 1, and comparison scripts.

## Literature Support Matrix
See `13_LITERATURE_MAPPING.md` for the complete method-to-paper mapping matrix.
