# 23 — Plot Guide: Tamil Nadu Pipeline

This guide documents what the plotting scripts show, what each result can be used to infer, and where the generated files are written. Paths are relative to the repository root. The plots are diagnostic or explanatory unless stated otherwise; a plot does not by itself prove causality or model accuracy.

## How To Run

Run scripts from the repository root. Output folders are created automatically.

```text
python 03_plots_raw.py
python 03b_interactive_raw_qa.py
python 04_preprocess_tamilnadu.py
python 04c_postprocess_plots.py
python 04b_climate_signature.py
python 04d_signature_interactive.py
python 05_cluster_tamilnadu.py
python 05b_cluster_interactive.py
python 05d_plots_comprehensive.py
python plots/generate_tamilnadu_plots.py
python plots/comparison_plots_tamilnadu.py
```

The scripts in `Objective 1/` are duplicate/older-location copies of several root scripts. The tables below document the root scripts that are used by the current pipeline.

## 1. Raw Data QA: `03_plots_raw.py`

Location: `data/plots/raw/`

| File | What it shows | What it helps infer |
|---|---|---|
| `A_point_map.png` | The 133 population-weighted sample points, sized and colored by population. | Whether the spatial sample covers populated Tamil Nadu and whether high-population cells dominate the sample as intended. |
| `B_event_profile.png` | Mean GHI and ambient temperature at sunrise, solar noon, and sunset. | A timezone/event-order error is suspected if GHI does not peak around noon. |
| `C_era5_vs_power.png` | ERA5 versus NASA POWER scatter plots for GHI, clear-sky GHI, temperature, humidity, and wind. | Agreement, systematic bias, scatter, and outliers between the two sources. The 1:1 line is a visual reference, not a calibration result. |
| `C_era5_vs_power_stats.csv` | MBE, RMSE, correlation, and sample count for each comparison variable. | The numerical evidence used to decide whether ERA5 can remain the backbone or needs quantile mapping. |
| `D_missing_heatmap.png` | Missing percentage by point and variable. | Whether gaps are localized to particular points or variables, and whether missingness is larger than expected before cleaning. |
| `E_seasonal_boxplots.png` | Noon GHI and ambient-temperature distributions by Winter, Summer, Monsoon, and Retreat seasons. | Seasonal spread, outliers, and whether the broad Tamil Nadu hot/monsoon pattern is plausible. |
| `F_yearly_trend.png` | Mean noon GHI and ambient temperature for each year from 2016 to 2025. | Abrupt year-to-year steps that could indicate a download, unit, or processing discontinuity. |

Interactive equivalents are written by `03b_interactive_raw_qa.py` to `data/plots/raw_interactive/` with the same A-F names and `.html` extensions. `03b_agreement_analysis.py` additionally writes `data/processed/era5_power_agreement_tamilnadu.csv`, `outputs/qc_era5_power_scatter_tamilnadu.html`, and `outputs/bias_decision_tamilnadu.txt`.

## 2. Preprocessing QA: `04_preprocess_tamilnadu.py` and `04c_postprocess_plots.py`

The first script writes diagnostics to `data/preprocessed/`:

| File | What it shows | What it helps infer |
|---|---|---|
| `savitzky_golay_diagnostic.png` | The smoothing diagnostic used during preprocessing. | Whether smoothing preserves the shape of the radiation signal rather than flattening peaks. |
| `correlation_heatmaps.png` | Correlation structure of the climate variables before/around feature engineering. | Whether expected relationships, such as temperature-humidity opposition, are present and whether a variable is suspiciously disconnected. |

`04c_postprocess_plots.py` writes to `data/plots/post_preprocess/`:

| File | What it shows | What it helps infer |
|---|---|---|
| `A_missing_post.png` | Residual missingness after physical filtering and imputation. | Whether the cleaned file is ready for signature construction; non-zero blocks require investigation. |
| `B_distributions_post.png` | Histograms of cleaned physical variables. | Whether cleaning introduced implausible spikes or collapsed a distribution around an imputed value. |
| `C_qc_flag_counts.png` | Counts flagged by physical bounds and Hampel filtering. | Which variables drove quality-control changes and whether one variable dominates the exclusions. |
| `C_qc_flag_counts.csv` | The numeric version of the QC counts. | Reproducible reporting of the plot values. |
| `D_lag_sanity.png` | Noon GHI against the seven-day lagged value. | Positive but imperfect structure supports temporal continuity; a near-zero or broken pattern suggests lag construction trouble. |
| `E_point_timeseries.png` | One cleaned point-year with raw-looking GHI and rolling means. | Whether the annual seasonal shape survives cleaning and whether smoothing is overly aggressive. |
| `F_correlation_post.png` | Correlations among cleaned and engineered daytime variables. | How feature engineering changes relationships used by the climate signature and clustering stages. |

Interactive postprocessing equivalents are in `data/plots/post_preprocess_interactive/` as HTML files.

## 3. Climate Signature: `04b_climate_signature.py`

Location: `data/processed/signatures/`

| File | What it shows | What it helps infer |
|---|---|---|
| `signature_correlation_heatmap.png` | Correlations among the canonical climate-signature indices. | Redundant or strongly related climate descriptors and whether PCA/standardization is warranted. |
| `signature_distributions.png` | Distributions of signature variables such as temperature, GHI, cloudiness, and humidity indices. | Skew, outliers, and the range of climate conditions passed into regime discovery. |
| `point_signature_map.png` | Spatial maps of mean GHI and monsoon-related signature indices. | Geographic gradients in solar resource and monsoon behavior that may explain different climate regimes. |

Interactive equivalents are in `data/processed/signatures/interactive/`: `A_signature_layers.html`, `B_correlation.html`, `C_distributions.html`, and `D_scatter_matrix.html`.

The signature stage also derives `Tm_target` and `L_required`. Current sizing uses a 300 L/day draw and `SHARE_PCM=0.5`; the generated `L_required` values are run-specific and should be read from the signature or feasibility CSV rather than inferred from a plot.

## 4. Climate Regimes: `05_cluster_tamilnadu.py`

Location: `data/processed/clustering/`

| File | What it shows | What it helps infer |
|---|---|---|
| `cluster_map_tamilnadu.png` | GMM cluster labels mapped to the population-weighted points. | The geographic arrangement and spatial continuity of the climate regimes used for PCM recommendations. | 

The non-plot files `bic_selection_tamilnadu.csv` and `kmeans_comparison_tamilnadu.csv` support the choice of `K=5`; they are evidence for model selection, not visualizations.

Interactive cluster outputs from `05b_cluster_interactive.py` are in `data/processed/clustering/interactive/`:

| File | What it shows | What it helps infer |
|---|---|---|
| `A_cluster_map.html` | Hard labels and soft GMM membership probabilities. | Boundary uncertainty and points that do not belong strongly to one regime. |
| `B_cluster_profiles.html` | Population-weighted climate properties by cluster. | The physical climate differences behind each cluster label. |
| `C_population_share.html` | Population share assigned to each cluster. | The demand relevance of each regime, not just its geographic area. |
| `D_k_selection.html` | BIC and silhouette values over candidate cluster counts. | Whether K=5 is supported by the selected model diagnostics. |

`05c_explore_interactive.py` is a Streamlit explorer rather than a batch plot generator. It displays raw/processed time series and a property-colored map and caches the map as `data/plots/interactive_explorer/location_map.html`.

## 5. Comprehensive Climate Plots: `05d_plots_comprehensive.py`

Location: `data/plots/comprehensive/`. By default this script reads the processed climate file; set `USE_PROCESSED=False` in the script for raw-file plots.

### Maps

| File | Meaning |
|---|---|
| `maps/A0_all_points_overview.html` | All sampled points, with population and summary values available on hover. |
| `maps/A1_GHI_mean_map.html` | Spatial pattern of mean noon GHI, with a heatmap overlay. |
| `maps/A2_population_map.html` | Population weighting represented spatially. |
| `maps/A3_india_context.html` | Tamil Nadu points shown in national geographic context. |

These maps show spatial coverage and gradients. They do not establish that climate causes a PCM ranking difference without the downstream target and ranking evidence.

### Time series and statistics

| File | What it shows | Main inference |
|---|---|---|
| `timeseries/B1_noon_GHI_sample_points.png` | Seven-day rolling noon GHI for 12 sample points. | Compare seasonal timing and variability among representative locations. |
| `timeseries/B2_noon_GHI_all_points.png` | All point traces plus the Tamil Nadu mean. | Distinguish statewide behavior from local variability. |
| `timeseries/B3_Tamb_vs_GHI_scatter.png` | Ambient temperature versus GHI. | Inspect whether hotter conditions tend to coincide with stronger/weaker radiation in the sampled data. |
| `timeseries/B4_annual_cycle_GHI.png` | Annual cycle of noon GHI. | Identify the seasonal solar-resource pattern. |
| `statistics/C1_correlation_matrix.png` | Correlations among selected climate variables. | Identify covariate redundancy and unexpected relationships. |
| `statistics/C2_GHI_violin_season.png` | Seasonal GHI distributions. | Compare medians and spread between seasons. |
| `statistics/C3_diurnal_profile_season.png` | Sunrise/noon/sunset profiles by season. | Check event timing and seasonal diurnal differences using the three sampled events. |
| `statistics/C4_cloud_vs_GHI_density.png` | Cloud-related variable versus GHI density/relationship. | Inspect how cloudiness is associated with radiation availability. |
| `solar_resource/D1_CSI_distribution.png` | Clear-sky index distribution. | Describe cloud attenuation and radiation intermittency. |
| `solar_resource/D2_top20_points_GHI.png` | Highest-noon-GHI points. | Locate the strongest sampled solar-resource points; this is not a population-weighted ranking unless population is encoded in the script. |

## 6. Objective 1 PCM Plots: `plots/generate_tamilnadu_plots.py`

Location: `data/plots/tamilnadu_objective1/`. Each numbered plot has a static PNG where implemented and usually an interactive HTML counterpart.

| Plot | What it shows | What it helps infer |
|---|---|---|
| `01_raw_vs_preprocessed_radiation` | Raw and cleaned GHI for one point. | Whether preprocessing changed the radiation series materially. |
| `02_climate_regime_map` | GMM cluster labels geographically. | Where each climate regime occurs and how confidently points are assigned. |
| `03_melting_point_vs_latent_heat` | Feasibility records in Tm-latent-heat space, with cluster windows/floors. | Which candidates satisfy the temperature and latent-heat constraints for each cluster. |
| `04_feasible_candidates_highlighted` | All PCM database records versus feasibility records. | How hard screening reduces the candidate set relative to the full 62-record database. |
| `05_pcm_survivors_per_cluster` | Count of feasibility-audit rows by cluster. | Use only `passes_all=True` rows for survivor counts; the CSV itself retains all audited candidates. |
| `06_pcm_feasibility_scatter_and_survivors` | Combined scatter and count summary. | A compact view of candidate properties and cluster-level filtering. |
| `07_bump_chart_ranks` | Rank of leading PCMs across TOPSIS, GRA, PROMETHEE, VIKOR, and consensus. | Agreement or rank reversal between decision methods. |
| `08_method_rank_correlation_heatmap` | Spearman and Kendall correlations among method ranks. | Whether methods produce broadly consistent orderings. |
| `09_monte_carlo_top3_probability` | Top-3 inclusion probability from 5,000 uncertainty draws. | Ranking stability under perturbed weights and PCM properties. |
| `10_rank_reversal_violin_bar` | Rank distributions and rank spread across methods. | Which candidates are sensitive to the MCDM method. |
| `11_agreement_plot` | Simulated performance rank versus consensus rank. | Whether higher MCDM rank tends to correspond to better simulated performance. |
| `12_tank_temperature_melt_fraction` | A representative synthetic daily tank temperature and melt-fraction profile. | Illustrates the intended charging/melting/discharging phases; it is explanatory and is not the full 10-year physics-validation output. |
| `13_recommended_pcm_summary` | Top-3 consensus recommendations and their properties per cluster. | Communicates the final candidate shortlist; stability and physics evidence should be checked alongside it. |

Files use the corresponding names with `.png` or `_interactive.html`. The generator also writes the canonical `pcm_feasibility_scatter.png` and `pcm_survivors_per_cluster.png`.

## 7. Cross-Step PCM Comparisons: `plots/comparison_plots_tamilnadu.py`

Location: `data/plots/comparison/`

| File | What it shows | What it helps infer |
|---|---|---|
| `01_comparison_cluster_ghi.png` | Mean GHI and standard deviation by cluster. | Whether regimes differ in solar resource. |
| `02_comparison_temp_vs_tm_target.png` | Cluster mean temperature versus PCM target melting point. | Whether the target has a consistent offset from climate temperature. |
| `03_comparison_mcdm_methods.png` | Top-five ranks from each MCDM method, side by side. | Method agreement and disagreements in the candidates users may select. |
| `04_comparison_mc_vs_rank.png` | Consensus rank versus Monte Carlo Top-3 probability. | Whether nominal rank is supported by uncertainty stability. |
| `05_comparison_latent_heat_distribution.png` | Latent-heat distributions for all database records and feasibility rows. | How screening changes the material-property distribution. |
| `06_comparison_physics_vs_rank.png` | Consensus rank against simulated hours target met and complete cycles/year. | Whether decision rankings align with physical performance and cycling. |
| `07_comparison_cross_cluster_top_pcm.png` | Properties of each cluster's consensus rank-1 PCM. | Whether recommended material properties change across regimes. |
| `08_comparison_rank_sensitivity.png` | Rank response to selected weight shifts. | How sensitive the result is to weighting assumptions. |

## Reading Rules

1. Use the plot and its source CSV together. Plots summarize the data and may hide rows, use samples, or show only Top-3 records.
2. Treat `data/processed/pcm/feasibility_survivors_by_cluster.csv` as a full filter audit. Count survivors with `passes_all=True`, not by counting all rows in the file.
3. Treat the current PCM database as 62 records: 55 manufacturer-derived MICE+RF+PMM-completed rows and 7 literature rows with genuinely unreported properties left missing.
4. Treat values such as `L_required`, Top-3 probabilities, Spearman correlation, and solar fractions as run-specific outputs. Regenerate downstream artifacts after changing climate inputs, configuration, or PCM data.
5. A visualization can reveal patterns and quality problems; it cannot replace the numerical tests, uncertainty analysis, or physics-validation tables.
