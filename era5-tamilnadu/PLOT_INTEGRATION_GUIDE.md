# PLOT INTEGRATION GUIDE — Tamil Nadu PCM Pipeline
## Mapping Available Visualizations to Documentation Sections

**Total Plots Available:** 84 files across 12 directories  
**Status:** Ready to integrate into enhanced audit document  
**Format:** PNG for static plots, HTML for interactive viewers

---

## PHASE 1: DATA COLLECTION
**Scripts:** `00a_build_population_grid.py`, `00b_build_suntimes.py`, `01_download_era5_tamilnadu.py`, `01b_download_nasapower.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| Population distribution map | `A_point_map.png` | `data/plots/raw/` | PNG | ✓ READY | Embed as "Population-Weighted Sampling Map" |
| Hourly weather profiles | `B_event_profile.png` | `data/plots/raw/` | PNG | ✓ READY | Embed as "Sample Hourly Climate Profile (TNP_0001)" |
| Annual GHI trend | `F_yearly_trend.png` | `data/plots/raw/` | PNG | ✓ READY | Embed as "10-Year GHI Time Series (2016-2025)" |
| Data coverage overview | `A_point_map.png` | `data/plots/raw/` | PNG | ✓ READY | (Use for coverage visualization) |

### Interactive Alternatives

| Plot | Location | File Count |
|---|---|---|
| Interactive point explorer | `data/plots/raw_interactive/` | 6 HTML files |
| Folium cluster map | `data/plots/interactive_explorer/` | 1 HTML file |

---

## PHASE 2: PREPROCESSING & CROSS-SOURCE VALIDATION
**Scripts:** `02_combine_tamilnadu.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`, `03b_agreement_analysis.py`, `04_preprocess_tamilnadu.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| ERA5 vs NASA POWER scatter | `C_era5_vs_power.png` | `data/plots/raw/` | PNG | ✓ READY | Embed as "GHI Cross-Source Comparison (Pre-QM)" |
| Missing data heatmap | `D_missing_heatmap.png` | `data/plots/raw/` | PNG | ✓ READY | Embed as "Missing Data Pattern Analysis" |
| Seasonal boxplots | `E_seasonal_boxplots.png` | `data/plots/raw/` | PNG | ✓ READY | Embed as "Seasonal Temperature/GHI Distributions" |
| Statistical QC summary | `03_statistical_summary.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Pre-QC Statistical Summary" |
| Feature engineering results | `04_feature_engineering.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Interaction Term Distributions" |
| Correlation heatmap | `05_correlation_analysis.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Feature Correlation Matrix" |
| Data completeness | `02_data_completeness.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Data Completeness After QC" |

### Interactive Alternatives

| Plot | Location | File Count |
|---|---|---|
| Interactive QA maps | `data/plots/post_preprocess_interactive/` | 5 HTML files |
| Post-preprocess verification plots | `data/plots/post_preprocess/` | 6 PNG files |

---

## PHASE 3: CLIMATE SIGNATURE CONSTRUCTION
**Scripts:** `04b_climate_signature.py`, `04d_signature_interactive.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| Climate distributions | `01_climate_distributions.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Climate Signature Variable PDFs" |
| Data completeness | `02_data_completeness.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Data Availability by Point" |
| Feature engineering | `04_feature_engineering.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Derived Interaction Term Distributions" |
| Correlation analysis | `05_correlation_analysis.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Signature Feature Correlations" |
| Quality metrics | `06_data_quality_metrics.png` | `data/plots/verify_preprocessing/` | PNG | ✓ READY | Embed as "Quality Metrics by Point" |

### Interactive Alternatives

| Plot | Location | File Count |
|---|---|---|
| Interactive signature explorer | `data/plots/post_preprocess_interactive/` | 5 HTML files |
| Signature cluster browser | `data/plots/interactive_explorer/` | 1 HTML file |

---

## PHASE 4: CLIMATE REGIME CLUSTERING
**Scripts:** `05_cluster_tamilnadu.py`, `05b_cluster_interactive.py`, `11_level_b_seasonal_analysis.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| **Geographic cluster map** | `04_geographic_map.png` | `data/plots/verify_clustering/` | PNG | ✓ READY | **CRITICAL: Embed as "Tamil Nadu Climate Regimes (GMM K=5)" — Central visual** |
| Elbow curve (BIC/silhouette) | `01_elbow_curves.png` | `data/plots/verify_clustering/` | PNG | ✓ READY | Embed as "Model Selection: BIC & Silhouette vs K" |
| Silhouette plot | `02_silhouette_plot.png` | `data/plots/verify_clustering/` | PNG | ✓ READY | Embed as "Silhouette Analysis (K=5 Selected)" |
| PCA projection | `03_pca_projection.png` | `data/plots/verify_clustering/` | PNG | ✓ READY | Embed as "Cluster Separation in PCA Space" |
| Cluster profiles (radar) | `05_cluster_profiles.png` | `data/plots/verify_clustering/` | PNG | ✓ READY | Embed as "Cluster Climate Profiles (Radar Charts)" |
| Cluster sizes | `06_cluster_sizes.png` | `data/plots/verify_clustering/` | PNG | ✓ READY | Embed as "Point Distribution Across 5 Clusters" |
| Comprehensive climate regime map | `02_climate_regime_map.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "Integrated Climate Regime Map" |

### Interactive Alternatives

| Plot | Location | File Count |
|---|---|---|
| Interactive cluster boundary explorer | `data/plots/post_preprocess_interactive/` | 5 HTML files |

---

## PHASE 5: FEASIBILITY FILTERING
**Scripts:** `06_build_pcm_database.py`, `07_feasibility_filter.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| **Survivor rate by cluster** | `01_survival_rate_by_cluster.png` | `data/plots/verify_feasibility/` | PNG | ✓ READY | **CRITICAL: Embed as "Feasibility Filter Results (Cluster-wise Pass Rates)"** |
| Feasible property space | `02_feasible_property_space.png` | `data/plots/verify_feasibility/` | PNG | ✓ READY | Embed as "Melting Point × Latent Heat Property Space" |
| Top candidates per cluster | `03_top_candidates_per_cluster.png` | `data/plots/verify_feasibility/` | PNG | ✓ READY | Embed as "Top Feasible PCM Candidates by Cluster" |
| Constraint analysis | `04_constraint_analysis.png` | `data/plots/verify_feasibility/` | PNG | ✓ READY | Embed as "Screening Constraint Effectiveness" |
| Property distributions | `05_property_distributions.png` | `data/plots/verify_feasibility/` | PNG | ✓ READY | Embed as "PCM Property Distributions (All 62 Candidates)" |
| Melting point vs latent heat | `03_melting_point_vs_latent_heat.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "PCM Property Scatter (Tm × L)" |
| Survivors highlighted | `04_feasible_candidates_highlighted.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "Feasible Candidates in Property Space" |
| PCM survivors per cluster | `05_pcm_survivors_per_cluster.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "Feasibility Survivors per Cluster" |

---

## PHASE 6: MULTI-CRITERIA RANKING ENGINE
**Scripts:** `08_mcdm_ranking.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| **Monte Carlo Top-3 probability** | `09_monte_carlo_top3_probability.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | **CRITICAL: Embed as "Monte Carlo Top-3 Inclusion Probability (5K draws)"** |
| Method correlation heatmap | `01_method_correlation.png` | `data/plots/verify_ranking/` | PNG | ✓ READY | Embed as "4-Method MCDM Correlation Matrix" |
| Top-3 inclusion probability | `02_top3_inclusion_probability.png` | `data/plots/verify_ranking/` | PNG | ✓ READY | Embed as "MC Stability: Top-3 Inclusion Rate by PCM" |
| Rank distributions | `03_rank_distributions.png` | `data/plots/verify_ranking/` | PNG | ✓ READY | Embed as "Rank Distribution Across Methods" |
| Rank reversal frequency | `04_rank_reversal_frequency.png` | `data/plots/verify_ranking/` | PNG | ✓ READY | Embed as "Method Agreement: Rank Reversal Analysis" |
| Method agreement | `05_method_agreement.png` | `data/plots/verify_ranking/` | PNG | ✓ READY | Embed as "Kendall's W (4-Method Concordance)" |
| Rank correlation heatmap | `08_method_rank_correlation_heatmap.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "Rank Correlation Among Methods" |
| Bump chart (rank trajectories) | `07_bump_chart_ranks.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "PCM Rank Trajectories Across 4 Methods" |
| Rank reversal violin plot | `10_rank_reversal_violin_bar.png` | `data/plots/tamilnadu_objective1/` | PNG | ✓ READY | Embed as "Rank Variance Distribution (Sensitivity Analysis)" |

---

## PHASE 7: PHYSICS-BASED VALIDATION
**Scripts:** `10_physics_validation.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| Solar fraction distribution | (Not yet explicitly named in list) | `data/plots/tamilnadu_objective1/` | PNG | ? | **NEED: Boxplot of annual solar fraction by cluster** |
| Spearman correlation | (Not yet explicitly named in list) | `data/plots/tamilnadu_objective1/` | PNG | ? | **NEED: Bar chart of Spearman ρ per cluster** |
| Complete cycles/year | (Not yet explicitly named in list) | `data/plots/tamilnadu_objective1/` | PNG | ? | **NEED: Histogram of cycles/year distribution** |
| Tank temperature trajectory | (Not yet explicitly named in list) | `data/plots/tamilnadu_objective1/` | PNG | ? | **NEED: Time series plot (T_tank vs T_ambient for sample year)** |

### Status
⚠ **Partial:** Some physics validation plots may exist in `tamilnadu_objective1/` but need verification. **ACTION: Check remaining 17 files in `tamilnadu_objective1/` for physics plots.**

---

## PHASE 8: RECOMMENDATION CARDS
**Scripts:** `09_recommendation_cards.py`

### Available Plots → Documentation Mapping

| Required Plot | Available File | Location | Type | Status | Recommendation |
|---|---|---|---|---|---|
| **Cluster profile radar overlays** | (Not yet explicitly named) | `data/plots/tamilnadu_objective1/` | PNG | ? | **CRITICAL: 5-cluster radar comparison — check remaining files** |
| PCM recommendation matrix | (Not yet explicitly named) | `data/plots/tamilnadu_objective1/` | PNG | ? | **Heatmap: Top-3 PCM recommendations × 5 clusters** |
| Population distribution pie | `Cluster sizes` (may be in verify_clustering) | `data/plots/` | PNG | ? | **Population % per cluster** |

---

## COMPREHENSIVE COMPARISON PLOTS
**Location:** `data/plots/comparison/` (8 files)

These plots are designed to compare across multiple phases:
- Raw vs. preprocessed radiation data
- Era5 vs. NASA POWER bias
- Before/after quantile mapping
- Clustering comparison (K=2-10)
- Multi-method ranking comparison
- Physics validation vs. MCDM

**Recommendation:** Use for validation sections and appendix

---

## SUMMARY: PLOT CHECKLIST FOR DOCUMENTATION

### ✓ READY TO EMBED (Status = Ready)
- **Phase 1:** 3/3 plots ready (A, B, F)
- **Phase 2:** 7/7 plots ready (C, D, E + 4 verify plots)
- **Phase 3:** 5/5 plots ready (from verify_preprocessing)
- **Phase 4:** 7/7 plots ready (6 clustering + 1 comprehensive)
- **Phase 5:** 8/8 plots ready (6 feasibility + 2 comprehensive)
- **Phase 6:** 9/9 plots ready (6 ranking verify + 3 comprehensive)
- **Phase 7:** 0/4 plots ready (**NEED TO VERIFY in tamilnadu_objective1/**)
- **Phase 8:** 0/3 plots ready (**NEED TO VERIFY in tamilnadu_objective1/**)

### ⚠ ACTION ITEMS
1. **List remaining 17 files** in `data/plots/tamilnadu_objective1/` to find Phase 7-8 plots
2. **Verify plot naming** — Some files may have generic names (e.g., `11_*.png`, `12_*.png`)
3. **Generate missing plots** if Phase 7-8 visualizations don't exist:
   - Solar fraction by cluster (boxplot)
   - Spearman ρ by cluster (bar chart)
   - Complete cycles/year (histogram)
   - Tank temperature simulation (time series)
   - Cluster radar overlays (5 radars on 1 plot)
   - Top-3 PCM heatmap (clusters × ranks)

---

## EMBEDDING INSTRUCTIONS FOR DOCX

When converting to DOCX:

1. **High-Priority Embeds** (CRITICAL):
   - `04_geographic_map.png` → Phase 4 header
   - `01_survival_rate_by_cluster.png` → Phase 5 header
   - `09_monte_carlo_top3_probability.png` → Phase 6 header

2. **Standard Embeds** (Include all):
   - Static PNG plots: ~50 files
   - Size recommendation: 6" width (300 dpi for print)
   - Alignment: Center with captions
   - Caption format: `Figure X.Y: [Title] — [Data source]`

3. **Interactive Embeds** (Optional):
   - HTML files not directly embeddable in DOCX
   - Use QR codes linking to HTML viewers
   - Alternatively, convert to static screenshots

4. **Plot Ordering** (by phase):
   ```
   Phase 1: Raw data collection
   ├─ A_point_map.png
   ├─ B_event_profile.png
   └─ F_yearly_trend.png
   
   Phase 2: Preprocessing QA
   ├─ C_era5_vs_power.png
   ├─ D_missing_heatmap.png
   ├─ E_seasonal_boxplots.png
   ├─ verify_preprocessing/*.png (7 files)
   └─ post_preprocess/*.png (6 files)
   
   [Continue for Phases 3-8...]
   ```

---

## FILE LISTING: Complete Plot Inventory

**Total: 84 files (81 PNG + 12 HTML)**

```
data/plots/
├── comparison/ (8 PNG)
├── comprehensive/ (0 PNG) ← EMPTY, may need generation
├── interactive_explorer/ (1 HTML)
├── post_preprocess/ (6 PNG)
├── post_preprocess_interactive/ (5 HTML)
├── raw/ (6 PNG) ✓
├── raw_interactive/ (6 HTML)
├── tamilnadu_objective1/ (27 PNG) ✓ — PRIMARY SOURCE for comprehensive plots
├── verify_clustering/ (6 PNG) ✓
├── verify_feasibility/ (6 PNG) ✓
├── verify_preprocessing/ (7 PNG) ✓
└── verify_ranking/ (6 PNG) ✓
```

---

## NEXT STEPS

1. **Verify Phase 7-8 plots** in `tamilnadu_objective1/` (remaining 17 files)
2. **Generate missing visualizations** if needed
3. **Create comprehensive/ plots** if not already done (currently empty)
4. **Organize images by phase** into folders for easy DOCX embedding
5. **Create image captions** with figure numbers and data sources
6. **Convert to DOCX** with embedded figures and proper formatting

---

**Document Status:** Ready for Phase 1-6; Needs Phase 7-8 plot verification  
**Last Updated:** 2026-09-03  
**Author:** Enhanced Audit v2.0
