# 21 — Reproducibility Audit

## Checklist for Pipeline Verification (v3.1)
Run scripts in this chronological order. Steps marked **(v3.1)** were added or corrected in the bug-fix release.

- [ ] **Phase 1**: `python 00a_build_population_grid.py` → 133 points
- [ ] **Phase 1**: `python 00b_build_suntimes.py` → 1,457,547 rows
- [ ] **Phase 1**: `python 01_download_era5_tamilnadu.py` → 240 NetCDF files
- [ ] **Phase 1**: `python 01b_download_nasapower.py` → 1330 JSON files
- [ ] **Phase 1**: `python 00_unzip_accum.py` → fix NetCDF zip-disguise
- [ ] **Phase 2**: `python 02_combine_tamilnadu.py` → **(v3.1)** uses `accum_to_flux()`
- [ ] **Phase 2**: `python 02b_build_daily_aggregates.py` → daily integrals
- [ ] **Phase 2**: `python 03_plots_raw.py` → raw QA plots
- [ ] **Phase 2**: `python 03b_agreement_analysis.py` → **(v3.1 NEW)** cross-source decision
- [ ] **Phase 2**: `python 04_preprocess_tamilnadu.py` → **(v3.1)** Step 2b quantile mapping
- [ ] **Phase 3**: `python 04b_climate_signature.py` → **(v3.1)** 300 L/day draw
- [ ] **Phase 4**: `python 05_cluster_tamilnadu.py` → **(v3.1)** covariance_type=diag
- [ ] **Phase 4**: `python 11_level_b_seasonal_analysis.py` → **(v3.1)** corrected L_required
- [ ] **Phase 5**: `python 06_build_pcm_database.py` → 25 PCMs
- [ ] **Phase 5**: `python 07_feasibility_filter.py` → screening
- [ ] **Phase 6**: `python 08_mcdm_ranking.py` → 5000 MC draws
- [ ] **Phase 7**: `python 10_physics_validation.py` → **(v3.1)** UA_TANK=2.0 W/K
- [ ] **Phase 8**: `python 09_recommendation_cards.py` → recommendation cards

## Environmental Configuration
- pvlib version: `pvlib >= 0.9`
- CDS API: `.cdsapirc` in project root
- Random seeds: `KMeans(42)`, `GaussianMixture(42)`, `run_monte_carlo(seed=42)`
- Python dependencies: numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, plotly

## Literature Support
| Requirement | Reference | Source |
|---|---|---|
| Reproducible random seeds | Standard ML practice | Framework doc §21 |
| Cross-source validation gate | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Full pipeline traceability | Framework doc D1–D8 | `01_PROJECT_CONTEXT.md` |
