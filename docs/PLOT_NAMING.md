# Plot naming convention (all four states)

Every figure a script writes goes under `era5-<state>/data/plots/`, in the
same category folder and with the same file name in every state. File names
never carry the state name; the state folder already says which state it is.
A figure that only one state produces still follows the scheme. It keeps its
own slot number, so a gap in another state's numbering means "not produced
there", not "missing".

Scripts stay where they are (Rajasthan's plotting scripts still live in
`PLOTSV2/`); only their output paths follow this layout. The pipeline scripts
get their folders from `config.py` (`PLOTS_DIR`, `QC_PLOTS_DIR`,
`PREPROCESS_PLOTS_DIR`, `SIGNATURE_PLOTS_DIR`, `CLUSTER_PLOTS_DIR`).

## Layout

| Folder | Files | Written by |
|---|---|---|
| `raw/` | `A_point_map`, `B_event_profile`, `C_era5_vs_power`, `D_missing_heatmap`, `E_seasonal_boxplots`, `F_yearly_trend` (.png) | `03_plots_raw.py` (Assam: B, C, E, F also overwritten by `generate_era5_nasa_comparison_plots.py` with the two-source versions) |
| `raw_interactive/` | same six names, .html | `03b_interactive_raw_qa.py` |
| `qc/` | `qc_*.html` interactive QC checks: `qc_era5_power_scatter`, `qc_population_grid`, `qc_clean_*`, `qc_cluster_map`, `qc_k_selection_curve`, `qc_cluster_profile_bars`, `qc_cluster_population_share`, `qc_level_b_regime_shift_sankey`, `qc_montecarlo_inclusion`, `qc_calibration_check`, `qc_seasonal_pcm_flip_heatmap` | the core-chain script for that stage |
| `preprocess/` | `correlation_heatmaps.png`, `savitzky_golay_diagnostic.png` | `04_preprocess_<state>.py` |
| `post_preprocess/`, `post_preprocess_interactive/` | `A_missing_post` … `F_correlation_post` | `04c_postprocess_plots.py`, `04c_interactive_postprocess_qc.py` |
| `signatures/` | `point_signature_map`, `signature_correlation_heatmap`, `signature_distributions` (.png; .html from `04b` in Rajasthan and Tamil Nadu) | `04b_climate_signature.py` (Uttarakhand .png), `PLOTSV2/phase3_climate_signature_rajasthan.py` (Rajasthan .png) |
| `signatures/interactive/` | `A_signature_layers`, `B_correlation`, `C_distributions`, `D_scatter_matrix` | `04d_signature_interactive.py` |
| `clustering/` | `cluster_map.png`; Assam also `k_selection_bic`, `k_selection_silhouette`, `k_selection_davies_bouldin`, `k_selection_calinski_harabasz`, `cluster_map.html`, `pca_projection.png` | `05_cluster_<state>.py`; Assam `generate_phase11_figures.py` |
| `clustering/interactive/` | `A_cluster_map`, `B_cluster_profiles`, `C_population_share`, `D_k_selection` | `05b_cluster_interactive.py` |
| `interactive_explorer/` | `location_map.html` | `05c_explore_interactive.py` (Streamlit) |
| `comprehensive/` | `maps/A0–A3`, `timeseries/B1–B4`, `statistics/C1–C4`, `solar_resource/D1–D2` | `05d_plots_comprehensive.py` |
| `mcdm/` | `mcdm_clusters_plotly.html`, `mcdm_clusters_folium.html` | `12_mcdm_interactive_plots.py` |
| `verify_preprocessing/` | see numbering below | `verify_01_*` |
| `verify_clustering/` | see numbering below | `verify_02_*` |
| `verify_feasibility/` | see numbering below | `verify_03_*` |
| `verify_ranking/` | see numbering below | `verify_04_*` |
| `comparison/` | see numbering below | `comparison_plots_<state>.py` |
| `objective1/` | see numbering below | `generate_<state>_plots.py` / `generate_objective1_plots.py` |
| `ppt/` | curated copies for slides, one folder per phase (`1 Data collection/` … `6 PCM Recommendation and Output/`), same file names as the sources | `PLOTSV2/build_plots_folder_rajasthan.py`, `build_plots_assam_ppt.py` |

## Numbered sets

PNG and HTML twins share a stem; the HTML twin ends in `_interactive.html`
inside `objective1/`. Per-cluster figures end in `_cluster_<id>`.

**`objective1/`**

| # | File stem | States |
|---|---|---|
| 01 | `01_raw_vs_preprocessed_radiation` | all |
| 02 | `02_climate_regime_map` (+ `_folium.html`) | all |
| 03 | `03_melting_point_vs_latent_heat` | RJ, TN, UK |
| 04 | `04_feasible_candidates_highlighted` | RJ, TN, UK |
| 05 | `05_pcm_survivors_per_cluster` | RJ, TN, UK |
| 06 | `06_pcm_feasibility_scatter_and_survivors` | RJ, TN, UK |
| 07 | `07_bump_chart_ranks`, `07_bump_chart_ranks_cluster_<id>` | all (RJ per-cluster only) |
| 08 | `08_method_rank_correlation_heatmap` | RJ, TN, UK |
| 09 | `09_monte_carlo_top3_probability` | RJ, TN, UK |
| 10 | `10_rank_reversal_violin_bar` | RJ, TN, UK |
| 11 | `11_agreement_plot` | RJ, TN, UK |
| 11b | `11b_physics_vs_mcdm_all_clusters` (MCDM rank vs simulated performance, one panel per cluster) | RJ (.html), TN |
| 11b | `11b_physics_vs_mcdm_delivery_rank`, `11b_physics_vs_mcdm_solar_fraction_rank`, `11b_physics_vs_mcdm_cycling_rank` (same question, one figure per physics metric) | AS |
| 11c | `11c_tm_vs_physics_delivery` | AS |
| 12 | `12_tank_temperature_melt_fraction` — **illustrative schematic** (idealised sine curves, not simulation output; titled as such) | RJ, TN, UK (AS in `ppt/` only) |
| 13 | `13_recommended_pcm_summary` | RJ, TN, UK |
| 14 | `14_mcdm_winner_map` | TN |

**`verify_preprocessing/`**: `01_climate_distributions`, `02_data_completeness`,
`03_statistical_summary`, `04_feature_engineering`, `05_correlation_analysis`,
`06_data_quality_metrics`, `07_preprocessing_summary`. Assam only:
`08_era5_vs_nasa_power_agreement`, `09_era5_vs_nasa_power_multivariable_scatter`,
`10_era5_vs_nasa_power_seasonal_scatter`, `11_event_profile_era5_vs_power`,
`12_seasonal_boxplots_era5_vs_power`, `13_multiyear_trend_era5_vs_power`,
`14_population_grid_map`.

**`verify_clustering/`**: `01_elbow_curves`, `02_silhouette_plot`,
`03_pca_projection`, `04_geographic_map`, `05_cluster_profiles`,
`06_cluster_sizes`. Assam only: `07_pca_scree_variance`.

**`verify_feasibility/`**: `01_survival_rate_by_cluster`,
`02_feasible_property_space`, `03_top_candidates_per_cluster`,
`04_constraint_analysis`, `05_property_distributions`, `06_feasibility_summary`.

**`verify_ranking/`**: `01_method_correlation`, `02_top3_inclusion_probability`,
`03_rank_distributions`, `04_rank_reversal_frequency`, `05_method_agreement`,
`06_ranking_summary`.

**`comparison/`**: `01_comparison_cluster_ghi`, `02_comparison_temp_vs_tm_target`,
`03_comparison_mcdm_methods`, `04_comparison_mc_vs_rank`,
`05_comparison_latent_heat_distribution`, `06_comparison_physics_vs_rank`,
`07_comparison_cross_cluster_top_pcm`, `08_comparison_rank_sensitivity`.
Assam only: `09_comparison_physics_diurnal_thermal_cycle` (an illustrative
schematic from idealised forcing, not physics-model output; titled as such). Uttarakhand only:
`10_comparison_topsis_vs_gra_agreement`. Rajasthan only (before/after
method-change checks): `phase3_tmcap_old_vs_new/tmcap_methodology_comparison.html`,
`phase5_lrequired_before_after/lrequired_before_after.html`.

## Assam Phase 11 package

`final_outputs/visuals/` is Assam's Phase 11 deliverable, a copy of figures
that already exist under `data/plots/`. The copies use the same file names
as their sources (`k_selection_bic.png`, `11b_physics_vs_mcdm_delivery_rank.png`,
and so on); the earlier `fig01_` … `fig10_` names are retired.

## Renames applied on 2026-10-28

| Old | New |
|---|---|
| `PLOTSV2/<folder>/` (Rajasthan outputs) | `data/plots/<folder>/` |
| `PLOTSV2/comparison_plots/` | `data/plots/comparison/` |
| `PLOTSV2/clustering_interactive/` | `data/plots/clustering/interactive/` |
| `PLOTSV2/phase3_climate_signature/` | `data/plots/signatures/` |
| `PLOTSV2/physics_validation/mcdm_vs_physics_agreement_rajasthan.html` | `data/plots/objective1/11b_physics_vs_mcdm_all_clusters.html` |
| `PLOTSV2/Plots/`, `plots_assam_ppt/<phase>/Assam/` | `data/plots/ppt/<phase>/` |
| `data/plots/<state>_objective1/` | `data/plots/objective1/` |
| `07_bump_chart_ranks_cluster<id>` (TN, UK) | `07_bump_chart_ranks_cluster_<id>` |
| `outputs/qc_*_<state>.html`, `outputs/qc_montecarlo_inclusion.html` | `data/plots/qc/qc_*.html` |
| `outputs/grid_plot.html` (Rajasthan) | `data/plots/qc/qc_population_grid.html` |
| `outputs/signature_*_<state>.html` | `data/plots/signatures/signature_*.html` |
| `data/preprocessed/{correlation_heatmaps,savitzky_golay_diagnostic}.png` | `data/plots/preprocess/` |
| `data/processed/signatures/*.png`, `.../signatures/interactive/` | `data/plots/signatures/`, `data/plots/signatures/interactive/` |
| `data/processed/clustering/cluster_map_<state>.png` | `data/plots/clustering/cluster_map.png` |
| `data/processed/clustering/interactive/` | `data/plots/clustering/interactive/` |
| Assam `data/processed/clustering/gmm_{bic,silhouette,davies_bouldin,calinski_harabasz}.png` | `data/plots/clustering/k_selection_*.png` |
| Assam `phase10_visualizations/01–03_mcdm_vs_*_rank.png` | `data/plots/objective1/11b_physics_vs_mcdm_*_rank.png` |
| Assam `phase10_visualizations/04_tm_vs_physics_delivery.png` | `data/plots/objective1/11c_tm_vs_physics_delivery.png` |
| Assam `final_outputs/visuals/fig09_final_k3_climate_regime_map.*`, `fig10_final_k3_pca_projection.png` | `data/plots/clustering/cluster_map.*`, `pca_projection.png` (copied to `final_outputs/visuals/` under the same names) |
| Assam `raw/F_multiyear_trend.png` | `raw/F_yearly_trend.png` |
| UK `verify_preprocessing/05_data_quality_metrics`, `06_correlation_analysis`; Assam `06_correlation_analysis` | `06_data_quality_metrics`, `05_correlation_analysis` |
| Assam `verify_preprocessing/03_population_grid_map`, `05_era5_vs_nasa_power_agreement`, `08–12_*` | `14_population_grid_map`, `08_…`, `09–13_*` |
| Assam `verify_clustering/02_pca_scree_variance` | `07_pca_scree_variance` |
| RJ, TN, UK `verify_feasibility/06_summary`, `verify_ranking/06_summary` | `06_feasibility_summary`, `06_ranking_summary` |
| Assam `comparison/09_physics_diurnal_thermal_cycle` | `09_comparison_physics_diurnal_thermal_cycle` |
| UK `comparison/04_comparison_topsis_vs_gra_agreement` | `10_comparison_topsis_vs_gra_agreement` |

Removed as duplicates: Assam's `plots/` mirror (every file also written to
`data/plots/`), Rajasthan's `PLOTSV2/phase1_data_collection_rajasthan.py`
(same six figures as `03_plots_raw.py`) and
`PLOTSV2/interactive_plots/03b_interactive_raw_qa.py` (copy of the root
`03b_interactive_raw_qa.py`).
