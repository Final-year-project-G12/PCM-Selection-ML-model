# Assam PCM Pipeline Plotting & Verification Guide

This guide details the full visualization, verification, and cross-pipeline comparison suite created for the **Assam ERA5 PCM Selection Pipeline** (`era5-assam`).

---

## 📁 Directory Structure & Generated Outputs

All outputs are saved automatically under `era5-assam/data/plots/`:

```
era5-assam/data/plots/
├── assam_objective1/                     # Objective 1 climate figures & interactive HTML cards
│   ├── 01_raw_vs_preprocessed_radiation.png
│   ├── 02_climate_regime_map.png
│   └── 02_climate_regime_map_folium.html
├── comparison/                           # Cross-pipeline comparison charts
│   ├── 01_comparison_cluster_ghi.png
│   ├── 02_comparison_temp_vs_tm_target.png
│   ├── 03_comparison_mcdm_methods.png
│   ├── 04_comparison_mc_vs_rank.png
│   ├── 05_comparison_latent_heat_distribution.png
│   ├── 06_comparison_physics_vs_rank.png
│   ├── 07_comparison_cross_cluster_top_pcm.png
│   ├── 08_comparison_rank_sensitivity.png
│   └── 09_physics_diurnal_thermal_cycle.png
├── raw/                                  # Raw Data QA & Cross-Source (ERA5 vs NASA POWER)
│   ├── B_event_profile.png               # Diurnal sun-event profile (Sunrise, Noon, Sunset)
│   ├── C_era5_vs_power.png               # 5-panel multi-variable scatter (GHI, T2M, RH, etc.)
│   ├── C_era5_vs_power_stats.csv         # Detailed MBE, RMSE, r error statistics
│   ├── E_seasonal_boxplots.png           # Seasonal GHI & temperature distributions
│   └── F_multiyear_trend.png             # 10-year decadal calibration trend (2016-2025)
├── verify_preprocessing/                 # Verification 01: Preprocessing QA
│   ├── 01_climate_distributions.png
│   ├── 02_data_completeness.png
│   ├── 03_population_grid_map.png
│   ├── 05_era5_vs_nasa_power_agreement.png
│   ├── 06_correlation_analysis.png
│   ├── 07_preprocessing_summary.png
│   ├── 08_era5_vs_nasa_power_multivariable_scatter.png
│   ├── 09_era5_vs_nasa_power_seasonal_scatter.png
│   ├── 10_event_profile_era5_vs_power.png
│   ├── 11_seasonal_boxplots_era5_vs_power.png
│   └── 12_multiyear_trend_era5_vs_power.png
├── verify_clustering/                    # Verification 02: GMM & Zoning QA
│   ├── 01_elbow_curves.png
│   ├── 02_pca_scree_variance.png
│   ├── 03_pca_projection.png
│   ├── 04_geographic_map.png
│   ├── 05_cluster_profiles.png
│   └── 06_cluster_sizes.png
├── verify_feasibility/                   # Verification 03: PCM Constraints QA
│   ├── 01_survival_rate_by_cluster.png
│   ├── 02_feasible_property_space.png
│   ├── 04_constraint_analysis.png
│   ├── 05_property_distributions.png
│   └── 06_feasibility_summary.png
└── verify_ranking/                       # Verification 04: MCDM & Stability QA
    ├── 01_method_correlation.png
    ├── 02_top3_inclusion_probability.png
    ├── 03_rank_distributions.png
    ├── 05_method_agreement.png
    └── 06_ranking_summary.png
```

---

## 🛠️ Execution Commands

Run all scripts from the workspace root or from `era5-assam/`:

```powershell
# 1. Generate Objective 1 plots and interactive Folium maps
python era5-assam/generate_assam_plots.py

# 2. Generate 9 cross-pipeline comparison charts
python era5-assam/comparison_plots_assam.py

# 3. Generate cross-source ERA5 vs NASA POWER comparison plots & interactive dashboard
python era5-assam/generate_era5_nasa_comparison_plots.py

# 4. Generate regional parity diagrams (NASA agreement, population grid, PCA scree, diurnal physics)
python era5-assam/generate_missing_assam_diagrams.py

# 5. Preprocessing Verification Suite
python era5-assam/verify_01_preprocessing_assam.py

# 6. Clustering & Zoning Verification Suite
python era5-assam/verify_02_clustering_assam.py

# 7. Feasibility Filter Verification Suite
python era5-assam/verify_03_feasibility_assam.py

# 8. MCDM Ranking & Monte Carlo Verification Suite
python era5-assam/verify_04_ranking_assam.py
```

---

## 📊 Summary of Verification Criteria

| Suite | Focus | Key Checks | Status |
| :--- | :--- | :--- | :--- |
| **Verify 01** | Data Preprocessing | Continuous distributions, zero nulls post-imputation, valid correlations | ✅ PASS |
| **Verify 02** | GMM Clustering | BIC curve inflection, non-overlapping PCA space, clear climate profiles | ✅ PASS |
| **Verify 03** | Feasibility Filter | Bounded physical property space, non-zero survivors per regime | ✅ PASS |
| **Verify 04** | MCDM Ranking | High inter-method Spearman correlation ($>0.85$), stable Monte Carlo rank probabilities | ✅ PASS |
