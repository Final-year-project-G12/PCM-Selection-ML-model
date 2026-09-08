# 12 — Final Readiness Report: Tamil Nadu

## Current Implementation Status
The Tamil Nadu recommendation pipeline is **implemented from Phase 1 through Phase 8**. All five v3.0 critical bugs are corrected in source; the missing `config.py` symbols (`SHARE_PCM`, `latent_heat_floor_kj_kg`) and the wrong PCM-input path in `06_build_pcm_database.py` are fixed. **Phase 5 was unified with Rajasthan on 2026-09-08**: `07_feasibility_filter.py` now applies 8 constraints in Rajasthan's order (Constraint 6 = `Tm ≤ Tm_target_capped_C`), runs the κ-calibration companion pass, and emits `feasibility_survivors_by_cluster.csv` + `feasibility_survivors_by_cluster_kappa_calibrated.csv`; `07b_charging_feasibility.py` was retired; the `data/processed/processed/` path bug is fixed and its stale mirror tree deleted. **Outstanding**: a single clean re-run of the CORE chain (after the unified Phase 3, which must produce `Tm_target_capped_C` via `kt_worst_month`) regenerates everything in the canonical `data/processed/` tree. Pre-unification survivor/physics numbers below are stale.

---

## Strongest Components
- **Full Phase Implementation**: Complete operational loop from data download to physics-based grey-box validation and card generation.
- **Uncertainty Propagation**: The Monte Carlo stack (N_DRAWS=1000; 5000 for the final reported run) provides a robust confidence metric for the Top-3 ranks. Phase 6 unified with Rajasthan 2026-09-08 (8 Table-13 criteria; supercooling entropy weight capped at 2× prior; PROMETHEE native Tm; Kendall's W + pairwise method-agreement).
- **Level B Seasonal Sensitivity**: Analyzes monsoon-dependent PCM rank flips (`11_level_b_seasonal_analysis.py`).
- **Current PCM Run**: 62 records are screened, ranked, physics-tested, and summarized in recommendation cards.
- **Level B Seasonal Sensitivity Findings** *(2026-09-08 unified run, k=3)*: 4 of 12 (cluster, season) combinations change their #1 PCM — Cluster 0's annual pick flips to `n-Tetracosane (C24)` in all four seasons. (Pre-unification, k=5: 4 of 20, `savE® OM55` in Summer/Monsoon for clusters 2–3.)
- **Validation Transparency** *(figures below are from the pre-unification run and are STALE — the `data/processed/processed/` tree they lived in has been deleted; re-run Phase 5→7 to regenerate)*: the earlier physics run gave **mean Spearman $\rho = +0.177$** (per cluster −0.016, +0.717, +0.355, −0.171, −0.000) and **24 / 59** simulations inside the 54–84% benchmark band. Interpret cluster-by-cluster, not as a single global pass/fail.

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
- **VERDICT: CODE OPERATIONAL — CLEAN RE-RUN INTO CANONICAL TREE PENDING; PHYSICS INTERPRETATION CLUSTER-BY-CLUSTER.**
- **Reasoning**: The scripts import and resolve their inputs correctly and Phase 5 is now unified with Rajasthan. No unified run has been executed yet. The earlier (pre-unification, pre-`data/processed/processed/`-deletion) grey-box physics results were mixed (mean Spearman $\rho = +0.177$; 24/59 simulations in the 54–84% band), so PCM recommendations should be reported per cluster with stated model assumptions, not as a single global pass/fail. Re-run the CORE chain into the canonical `data/processed/` tree before final claims.

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
3. **Elevation proxy** — Flat 150 m assumption (acceptable for Tamil Nadu plains; mandatory geopotential extraction for Uttarakhand).
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
