# 12 — Final Readiness Report: Tamil Nadu

## Current Implementation Status
The Tamil Nadu recommendation pipeline is **implemented from Phase 1 through Phase 8** and a complete 62-PCM run exists (PCM DB, feasibility, ranking, Monte Carlo, physics validation, recommendation cards, Level B). All five v3.0 critical bugs are corrected in source. Two further blocking errors found in the 2026-09-07 reconciliation — missing `config.py` symbols (`SHARE_PCM`, `latent_heat_floor_kj_kg`) and a wrong PCM-input path in `06_build_pcm_database.py` — are also now fixed (see `12_FINAL_READINESS_REPORT.md` §14 and `00_MASTER_OVERVIEW.md`). **Outstanding**: the completed 62-PCM artifacts currently live in the non-canonical `data/processed/processed/` tree; the canonical `data/processed/` tree holds a superseded 25-PCM run. A single clean re-run of the chain (now that import/path errors are fixed) regenerates everything in the canonical location.

---

## Strongest Components
- **Full Phase Implementation**: Complete operational loop from data download to physics-based grey-box validation and card generation.
- **Uncertainty Propagation**: The 5000-draw Monte Carlo stack provides a robust confidence metric for the Top-3 ranks.
- **Level B Seasonal Sensitivity**: Analyzes monsoon-dependent PCM rank flips (`11_level_b_seasonal_analysis.py`).
- **Current PCM Run**: 62 records are screened, ranked, physics-tested, and summarized in recommendation cards.
- **Level B Seasonal Sensitivity Findings**: Four of 20 cluster-season combinations change their #1 PCM; `savE® OM55` replaces the annual winner in Summer and Monsoon for clusters 2 and 3.
- **Validation Transparency**: The v3.1-fixed physics run (`data/processed/processed/pcm/physics_validation_*`) gives **mean Spearman $\rho = +0.177$** (per cluster −0.016, +0.717, +0.355, −0.171, −0.000) and **24 / 59** simulations inside the 54–84% benchmark band. Cluster 1 shows partial rank agreement; the other four are weak/near-zero. Rank-1 `n-Octacosane (C28)` is in band for clusters 0–2, below band for 3–4. Interpret cluster-by-cluster, not as a single global pass/fail.

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
- **Reasoning**: The scripts now import and resolve their inputs correctly. A completed 62-PCM run exists in `data/processed/processed/`. Its grey-box physics results are mixed (mean Spearman $\rho = +0.177$; 24/59 simulations in the 54–84% band; realistic cycling 3–260/yr), so PCM recommendations should be reported per cluster with stated model assumptions, not as a single global pass/fail. Regenerate in the canonical `data/processed/` tree before final claims.

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
python 07b_charging_feasibility.py        # optional; must precede 07 to take effect
python 07_feasibility_filter.py
python 08_mcdm_ranking.py
python 10_physics_validation.py
python 09_recommendation_cards.py
python 11_level_b_seasonal_analysis.py    # runs last: reads 08's mcdm_full_scores_by_cluster.csv
```

`python run_all_tamilnadu.py` runs this whole CORE chain in this exact order in one command (`--include-setup` also runs Phase-1 downloads; `--with-optional` adds QA/plot scripts).

---

## Still Open (Not Blocking Code Readiness)
1. **PCM database coverage** — Current database has 62 rows (55 manufacturer-derived + 7 literature), meeting the former 40–60-row target. Additional salt hydrates remain optional expansion.
2. **External cluster validation** — ARI vs Köppen-Geiger / NBC-ECBC zones not implemented.
3. **Elevation proxy** — Flat 150 m assumption (acceptable for Tamil Nadu plains; mandatory geopotential extraction for Uttarakhand).
4. **`monsoon_index`** — Proxy-only; NASA POWER precipitation not downloaded.
5. **5th-percentile insolation charging filter** — Heuristic substitute in `07b_charging_feasibility.py`.
6. **Full Level-B GMM** — Current `11_level_b_seasonal_analysis.py` is seasonal re-rank, not independent per-season clustering.

---

## Literature Support
See `13_LITERATURE_MAPPING.md` for the full method-to-paper matrix. Key references for readiness criteria:
- **Cross-source validation**: Ghodusinejad et al. (2026) — `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md`
- **PCM-SWH sizing**: Singh et al. (2025) — `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md`
- **Physics validation**: Barqawi (2025) — `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md`
- **MCDM consensus**: Chen et al. (2025) — `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md`
