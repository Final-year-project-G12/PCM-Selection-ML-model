# 12 — Final Readiness Report: Tamil Nadu

> **UPDATE 2026-09-16**: the "clean re-run pending" state described throughout this report is
> **resolved** — the full CORE chain (Phases 1-8) has been re-run against elevation-corrected
> data and a corrected `TM_TARGET_C=67°C` (was 57°C — a citation-accuracy fix, see `CHANGELOG.md`).
> Current per-phase results live in `00_MASTER_OVERVIEW.md`'s status table and the individual
> phase audit docs (`06_PHASE_4_AUDIT.md`, `08_PHASE_6_AUDIT.md`, `09_PHASE_7_AUDIT.md`). This
> report's own numbers below (pre-2026-09-16) are historical context, not current state — in
> particular the Spearman ρ figures quoted below have since gone **negative** in Cluster 0
> (ρ=-0.595) under the corrected target, a diagnosed, open finding — see `09_PHASE_7_AUDIT.md`.

## Current Implementation Status
The Tamil Nadu recommendation pipeline is **implemented from Phase 1 through Phase 8**. All five v3.0 critical bugs are corrected in source; the missing `config.py` symbols (`SHARE_PCM`, `latent_heat_floor_kj_kg`) and the wrong PCM-input path in `06_build_pcm_database.py` are fixed. **Phase 5 was unified with Rajasthan on 2026-09-08**: `07_feasibility_filter.py` now applies 8 constraints in Rajasthan's order (Constraint 6 = `Tm ≤ Tm_target_capped_C`), runs the κ-calibration companion pass, and emits `feasibility_survivors_by_cluster.csv` + `feasibility_survivors_by_cluster_kappa_calibrated.csv`; `07b_charging_feasibility.py` was retired; the `data/processed/processed/` path bug is fixed and its stale mirror tree deleted. **Outstanding**: a single clean re-run of the CORE chain (after the unified Phase 3, which must produce `Tm_target_capped_C` via `kt_worst_month`) regenerates everything in the canonical `data/processed/` tree. Pre-unification survivor/physics numbers below are stale.

---

## Strongest Components
- **Full Phase Implementation**: Complete operational loop from data download to physics-based grey-box validation and card generation.
- **Uncertainty Propagation**: The Monte Carlo stack (N_DRAWS=1000; 5000 for the final reported run) provides a robust confidence metric for the Top-3 ranks. Phase 6 unified with Rajasthan 2026-09-08 (8 Table-13 criteria; supercooling entropy weight capped at 2× prior; PROMETHEE native Tm; Kendall's W + pairwise method-agreement).
- **Level B Seasonal Sensitivity**: Analyzes monsoon-dependent PCM rank flips (`11_level_b_seasonal_analysis.py`).
- **Current PCM Run**: 62 records are screened, ranked, physics-tested, and summarized in recommendation cards.
- **Level B Seasonal Sensitivity Findings** *(2026-09-08 unified run, k=3)*: 4 of 12 (cluster, season) combinations change their #1 PCM — Cluster 0's annual pick flips to `n-Tetracosane (C24)` in all four seasons. (Pre-unification, k=5: 4 of 20, `savE® OM55` in Summer/Monsoon for clusters 2–3.)
- **Validation Transparency** - **CURRENT figures, corrected 2026-11-09.** The 2026-09-16 run (k=3, n=26) gives **mean Spearman rho = +0.169** (per cluster **+0.381** / +0.103 / +0.024) and **6 / 26 (23 %)** simulations inside the 54-84 % benchmark band. **None of the per-cluster correlations is statistically significant** (n=8-10 per cluster, p = 0.12-0.96), so no value here - positive or negative - is distinguishable from zero; treat them as descriptive point estimates. Interpret cluster-by-cluster, not as a global pass/fail. *(Superseded, do not quote: the pre-unification k=5 run's rho = +0.177 over 5 clusters and 24/59 in band.)*

---

## Corrected Issues

| Issue | Script | Status |
|---|---|---|
| Deaccumulation bug | `02_combine_tamilnadu.py` | **Fixed** — `accum_to_flux()` |
| Quantile mapping | `04_preprocess_tamilnadu.py` + `03b_agreement_analysis.py` | **Fixed** — Step 2b per-season QM |
| 1000× flow rate | `04b_climate_signature.py`, `11_level_b_seasonal_analysis.py` | **Fixed** — 300 L/day draw |
| GMM overfitting | `05_cluster_tamilnadu.py` | **Fixed** — `covariance_type="diag"` |
| Missing tank heat loss | `10_physics_validation.py` | **Fixed** — `UA_TANK_W_K = 2.0` |
| Missing `config.py` symbols (ImportError in Phases 3/5/Level B) | `config.py` (used by `04b`, `07`, `11`) | **Fixed 2026-09-07** — `SHARE_PCM = 0.5`, `latent_heat_floor_kj_kg()` |
| Wrong PCM input path (`06` could not find its CSV) | `06_build_pcm_database.py` | **Fixed 2026-09-07** — `INPUT_CSV` now resolves to repo-root `PCM_data/data/` |
| `run_all_tamilnadu.py`: `02_combine` commented out of CORE; `11` sequenced before inputs | `run_all_tamilnadu.py` | **Fixed 2026-09-07** |

---

## Validation Verdict
- **VERDICT (corrected 2026-11-09): CODE OPERATIONAL - CLEAN RE-RUN COMPLETED 2026-09-16; PHYSICS AGREEMENT WEAK AND NOT STATISTICALLY SIGNIFICANT; INTERPRET CLUSTER-BY-CLUSTER.** The "re-run pending" verdict this line used to carry was left over from before the 2026-09-16 run.
- **Reasoning (updated 2026-11-09)**: the scripts import and resolve their inputs correctly, Phase 5 is unified with Rajasthan, and **the clean CORE re-run was completed 2026-09-16** - the "no unified run has been executed yet" statement this line used to carry is obsolete. Current grey-box physics agreement is weak and not statistically significant (mean rho = +0.169; 6/26 in the 54-84 % band; p = 0.12-0.96), so PCM recommendations must be reported per cluster with stated model assumptions and **never as "physics-validated"**. The MCDM/physics disagreement is a documented finding, not a blocker - see `09_PHASE_7_AUDIT.md` section 5.

---

## Recommended Re-Run Order
```bash
python 02_combine_tamilnadu.py
python 02b_build_daily_aggregates.py
python 03_plots_raw.py
python 03b_agreement_analysis.py          # optional cross-source decision report
python 04_preprocess_tamilnadu.py         # includes Step 2b quantile mapping
python 04b_climate_signature.py
python 05_cluster_tamilnadu.py
python 06_build_pcm_database.py           # reads repo-root PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv (55 rows) -> 62-row DB
python 07_feasibility_filter.py           # 8 constraints (C6 = Tm <= Tm_target_capped_C) + kappa calibration -> feasibility_survivors_by_cluster{,_kappa_calibrated}.csv  [07b_charging_feasibility.py RETIRED 2026-09-08]
python 08_mcdm_ranking.py                 # UNIFIED with Rajasthan; 8 criteria + supercooling entropy cap; reads ..._kappa_calibrated.csv -> mcdm_full_rankings.csv, mcdm_topk_by_cluster.csv, monte_carlo_stability.csv, mcdm_method_agreement.csv
python 10_physics_validation.py           # reads mcdm_full_rankings.csv
python 09_recommendation_cards.py         # reads mcdm_topk_by_cluster.csv + feasibility_survivors_by_cluster_kappa_calibrated.csv
python 11_seasonal_pcm_sensitivity.py     # runs last: reads 08's mcdm_full_rankings.csv
```

`python run_all_tamilnadu.py` runs this whole CORE chain in this exact order in one command (`--include-setup` also runs Phase-1 downloads; `--with-optional` adds QA/plot scripts).

---

## Still Open (Not Blocking Code Readiness)
1. **PCM database coverage** — Current database has 62 rows (55 manufacturer-derived + 7 literature), meeting the former 40–60-row target. Additional salt hydrates remain optional expansion.
2. **External cluster validation** — ARI vs Köppen-Geiger / NBC-ECBC zones not implemented.
3. ~~**Elevation proxy**~~ — **RESOLVED 2026-09-16**: `00c_attach_elevation.py` added, real per-point elevation now attached (-0.0–1,283.4 m, mean 282.4 m across 133 points) and consumed by `02_combine_tamilnadu.py`'s solar geometry. No longer a flat 150 m assumption.
4. **`monsoon_index`** — Proxy-only; NASA POWER precipitation not downloaded.
5. **Charging feasibility** — Now Constraint 6 in `07_feasibility_filter.py`: `Tm ≤ Tm_target_capped_C`, using Phase 3's literature-anchored `kt_worst_month` ceiling (not the literal 5th-percentile daily-insolation mechanism). The old heuristic `07b_charging_feasibility.py` was retired 2026-09-08. Phase 7's grey-box model remains the stronger downstream check.
6. **Full Level-B GMM** — Current `11_level_b_seasonal_analysis.py` is seasonal re-rank, not independent per-season clustering.

---

## Literature Support
See `13_LITERATURE_MAPPING.md` for the full method-to-paper matrix. Key references for readiness criteria:
- **Cross-source validation**: Ghodusinejad et al. (2026) — `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md`
- **PCM-SWH sizing**: Singh et al. (2025) — `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md`
- **Physics validation**: Barqawi (2025) — `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md`
- **MCDM consensus**: Chen et al. (2025) — `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md`
