# 21 — Reproducibility Audit

## Checklist for Pipeline Verification
To verify the Tamil Nadu pipeline, run the scripts in this chronological order:

- [ ] **Phase 1**: `python 00a_build_population_grid.py` -> Creates `population_grid_points.csv` (133 points).
- [ ] **Phase 1**: `python 00b_build_suntimes.py` -> Computes `suntimes.csv` (1,457,547 rows).
- [ ] **Phase 1**: `python 01_download_era5_tamilnadu.py` -> Downloads NetCDF files (240 files).
- [ ] **Phase 1**: `python 01b_download_nasapower.py` -> Downloads NASA POWER JSON files (1330 files).
- [ ] **Phase 1**: `python 00_unzip_accum.py` -> Fixes NetCDF zip-disguise.
- [ ] **Phase 2**: `python 02_combine_tamilnadu.py` -> Combines ERA5 & POWER into `climate_tamilnadu_points.csv`.
- [ ] **Phase 2**: `python 02b_build_daily_aggregates.py` -> Creates `daily_aggregates_tamilnadu.csv`.
- [ ] **Phase 2**: `python 03_plots_raw.py` -> Generates raw plots and `C_era5_vs_power_stats.csv`.
- [ ] **Phase 2**: `python 04_preprocess_tamilnadu.py` -> Creates cleaned data.
- [ ] **Phase 3**: `python 04b_climate_signature.py` -> Generates climate signatures.
- [ ] **Phase 4**: `python 05_cluster_tamilnadu.py` -> Fits GMM (K_FINAL=5).
- [ ] **Phase 4**: `python 11_level_b_seasonal_analysis.py` -> Evaluates seasonal re-rankings.
- [ ] **Phase 5**: `python 06_build_pcm_database.py` -> Builds database (25 PCMs).
- [ ] **Phase 5**: `python 07_feasibility_filter.py` -> Screening constraints.
- [ ] **Phase 6**: `python 08_mcdm_ranking.py` -> MCDM ranking with 5000 MC draws.
- [ ] **Phase 7**: `python 10_physics_validation.py` -> Grey-box tank simulation.
- [ ] **Phase 8**: `python 09_recommendation_cards.py` -> Compiles recommendation cards.

## Environmental Configuration
- Pylib version: `pvlib >= 0.9`
- CDS API configuration: CDS API key saved in `.cdsapirc`.
- Random seeds: `KMeans(random_state=42)`, `GaussianMixture(random_state=42)`, `run_monte_carlo(seed=42)`.
