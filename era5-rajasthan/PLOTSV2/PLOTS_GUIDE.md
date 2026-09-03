# Rajasthan PCM Pipeline — PLOTSV2 Plots & Verification Guide

> **Project:** Climate-Adaptive PCM Thermal Storage for Solar Water Heating — Objective 1 (climate-region-aware PCM recommendation)
> **State:** Rajasthan | **Plot set:** V2 — the same 13 objective-1 plots + 4 verification suites that `tamilnadu_pipeline/plots/` and `era5-uttarakhand/` produce, regenerated on Rajasthan data with the same figure sizes, palette and filenames.

---

## 1. Quick reference

| Script | What it generates | Output directory |
| :--- | :--- | :--- |
| `generate_rajasthan_plots.py` | **13 objective-1 plots** (static PNG + interactive Plotly/Folium HTML) | `rajasthan_objective1/` |
| `verify_01_preprocessing_rajasthan.py` | **Preprocessing & QC verification** (7 plots) | `verify_preprocessing/` |
| `verify_02_clustering_rajasthan.py` | **Clustering validation** (6 plots) | `verify_clustering/` |
| `verify_03_feasibility_rajasthan.py` | **Feasibility-filter validation** (6 plots) | `verify_feasibility/` |
| `verify_04_ranking_rajasthan.py` | **MCDM ranking validation** (6 plots) | `verify_ranking/` |
| `phase1_data_collection_rajasthan.py` | **Raw-data QA** (6 plots + MBE/RMSE CSV) | `phase1_data_collection/` |
| `phase3_climate_signature_rajasthan.py` | **Climate-signature diagnostics** (3 plots) | `phase3_climate_signature/` |
| `build_plots_folder_rajasthan.py` | Assembles the curated 6-phase folder (§6) | `Plots/` |
| `run_all_plots_v2.py` | Runs all of the above in pipeline order | — |

```
python run_all_plots_v2.py              # everything, then assembles Plots/
python run_all_plots_v2.py objective1   # just the 13 plots
python run_all_plots_v2.py verify       # just the four verification suites
python run_all_plots_v2.py phases       # just the phase-1 / phase-3 figures
python run_all_plots_v2.py plots        # just re-assemble Plots/
```

Phase 1 reads the ~1.4 GB raw points CSV, so a full run takes a few minutes.

Cross-step **comparison plots** are not duplicated here — `../plotting/comparison_plots_rajasthan.py`
already produces those 8 figures into `../outputs/objective1_plots_rajasthan/comparison_plots/`.

---

## 2. Data these plots read

All paths relative to `era5-rajasthan/`:

| Input | File |
| :--- | :--- |
| Raw ERA5 point series | `data/processed/climate_rajasthan_points.csv` |
| Preprocessed physical series | `data/preprocessed/rajasthan_cleaned_physical.csv` |
| Climate signature (per point) | `data/processed/climate_signature_rajasthan.csv` |
| Cluster assignments (**Level A**, annual GMM, k=3) | `data/processed/cluster_assignments_rajasthan_levelA.csv` |
| Cluster profiles | `data/processed/cluster_profiles_rajasthan.csv` |
| Feasibility evaluation (kappa-calibrated) | `data/processed/feasibility_survivors_rajasthan_kappa_calibrated.csv` |
| MCDM rankings | `data/processed/mcdm_rankings_rajasthan.csv` |
| Physics validation | `data/processed/physics_validation_rajasthan.csv` |
| PCM property database | `../PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` |

### Schema note

Rajasthan names its columns differently from Uttarakhand / Tamil Nadu
(`TOPSIS_rank` vs `topsis_rank`, `pcm_id` vs `name`, a `borda_score` instead of a
stored `consensus_rank`). All of that is handled in one `RENAME` map plus
`load_topk()` / `load_feasibility()` at the top of `generate_rajasthan_plots.py` —
the 13 plot functions below it are unchanged from the Uttarakhand original.

Two Rajasthan-specific things worth knowing:

- **The feasibility CSV is an evaluation table, not a survivor list.** It holds all
  62 candidates × 3 clusters = 186 rows with a boolean `survives_all`. Survivors
  are the 39 flagged rows (9 / 14 / 16 per cluster). Anything that plots
  "survivors" must filter on that column first.
- **Consensus rank is derived**, not stored: Borda score ranked high-to-low
  within each cluster. Ties are real — cluster 0 has two candidates tied at
  Borda 20, so its "Top-3" panel legitimately shows 4 bars.

---

## 3. The 13 objective-1 plots (`rajasthan_objective1/`)

### 1. Raw vs. Preprocessed Radiation
`01_raw_vs_preprocessed_radiation.png` · `..._interactive.html`
**Source:** raw ERA5 points vs `rajasthan_cleaned_physical.csv`, point `RJP_0001`.
**Why:** shows the effect of Hampel outlier filtering, gap imputation and quantile-mapping bias correction against NASA POWER.
**Verify:** no negative GHI after preprocessing; noon peaks around 900–1050 W/m² (physically right for Rajasthan). The two files are exported with different row ordering (raw is point-major, preprocessed is date-major), so the script clips both to their overlapping date window before plotting — the two panels must cover the same span.

### 2. Climate-Regime Map
`02_climate_regime_map.png` · `..._interactive.html` · `..._folium.html`
**Source:** `cluster_assignments_rajasthan_levelA.csv` (320 population-weighted grid points, 23.1–29.9 °N, 71.1–78.1 °E).
**Verify:** the three regimes should be spatially coherent, not salt-and-pepper — cluster 1 picks out the arid western Thar belt, cluster 2 the north-east/Shekhawati side, cluster 0 the southern/south-east block.

### 3. Melting Point vs. Latent Heat
`03_melting_point_vs_latent_heat.png` · `..._interactive.html`
**Source:** kappa-calibrated survivors.
**Verify:** the shaded band is the filter's *real* melting window. All three clusters exhausted the same 4 relaxation rounds (widen ±8 K on a `Tm_target` of 57 °C), so the window is **[44.0, 73.0] °C** for all three and the three bands coincide — that is correct, not a plotting bug. Every survivor sits above the 100 kJ/kg reference line.

### 4. Feasible Candidates Highlighted
`04_feasible_candidates_highlighted.png`
**Verify:** grey = the full 62-candidate evaluated pool, coloured = survivors per cluster. The rejected mass should sit outside the Tm window or below the latent-heat floor.

### 5. Feasible Candidates per Climate Regime
`05_pcm_survivors_per_cluster.png` · `..._interactive.html`
**Verify:** 9 / 14 / 16 survivors for clusters 0 / 1 / 2 — i.e. 14.5 % / 22.6 % / 25.8 % of the pool, inside the 10–50 % selectivity band.

### 6. Feasibility Scatter + Survivor Count (canonical filenames)
`06_pcm_feasibility_scatter_and_survivors.png` · `pcm_feasibility_scatter.png` · `pcm_survivors_per_cluster.png`
Same content as 3 and 5, under the filenames the Uttarakhand/Tamil Nadu sets use, so the three states' folders stay directly comparable.

### 7. Bump Chart — ranks across MCDM methods (**one per cluster**)
`07_bump_chart_ranks_cluster_0.png` · `_cluster_1.png` · `_cluster_2.png` (+ matching `.html`)
**Source:** TOPSIS / GRA / PROMETHEE II / VIKOR ranks + consensus (Borda).

This is the one place the Rajasthan set deliberately departs from the Uttarakhand
layout. Uttarakhand draws a **single pooled chart** of the top 12 by consensus rank;
MCDM ranks are assigned *within* a cluster, so pooling puts three different
candidates at rank 1 on one pair of axes and overlays lines that are not on a
common scale. Rajasthan splits per cluster (9 / 14 / 16 candidates), showing every
candidate in that regime — the same layout `plotting/05_bump_chart.py` uses, with
the same style and axes as the rest of this set.

**Verify:** flat lines = unanimous ranking; crossings = sensitivity to the
aggregation method. `savE® OM50` holds rank 1 under TOPSIS, PROMETHEE II, VIKOR and
consensus in clusters 1 and 2, dipping only to 2–3 under GRA; `RT50` does the same
in cluster 0 (rank 1 everywhere except GRA, where it falls to 8). GRA is visibly
the outlier method in all three charts, which is the same disagreement plot 8
quantifies.

> **Do not use `outputs/objective1_plots_rajasthan/04_mcdm_agreement/bump_chart_cluster_*.html`
> for the paper.** That version derives its consensus column with
> `borda_score.rank()`, which defaults to *ascending* — but Borda score is
> higher = better, so its consensus axis is inverted and the best candidate is
> drawn last. `physics_validation_rajasthan.csv`'s own `mcdm_borda_rank` column
> confirms the correct direction (RT50 = rank 1 in cluster 0); PLOTSV2 matches it.

### 8. Method Rank Correlation Heatmap
`08_method_rank_correlation_heatmap.png` · `..._interactive.html`
**Verify:** Spearman ρ and Kendall τ between every method pair. TOPSIS↔PROMETHEE II is the strongest pair (~0.77–0.84 within cluster); GRA is the outlier and is *negatively* correlated with TOPSIS in cluster 0 — that disagreement is a real finding, worth a sentence in the paper rather than smoothing over.

### 9. Monte Carlo Top-3 Inclusion Probability
`09_monte_carlo_top3_probability.png` · `..._interactive.html`
**Source:** `mc_top3_inclusion_pct` (weight-perturbation draws).
**Verify:** ≥ 80 % = robust under decision-maker weight uncertainty. `savE® OM50` clears it in both clusters 1 (83.2 %) and 2 (93.9 %); `RT50` clears it in cluster 0 (90.8 %).

### 10. Rank-Reversal Frequency (violin + bar)
`10_rank_reversal_violin_bar.png` · `..._interactive.html`
**Verify:** left = rank distribution per method per cluster; right = the 15 candidates with the widest max−min rank spread. Small spread for the recommended PCMs is what you want.

### 11. Agreement Plot — simulated performance vs. MCDM consensus
`11_agreement_plot.png` · `..._interactive.html`
**Source:** `hours_target_met_per_year` from the physics validation, ranked per cluster, against the Borda consensus rank.
**Verify:** points near the red 1:1 line mean the MCDM ranking agrees with the simulated thermal performance. Scatter away from the line is the honest result to report — it is the same disagreement the `rank_gap_abs` column in `physics_validation_rajasthan.csv` quantifies.

### 12. Tank Temperature / Melt-Fraction Profile
`12_tank_temperature_melt_fraction.png` · `..._interactive.html`
**Source:** per-cluster `Tm_target_capped_C` (48.4 / 52.3 / 51.1 °C) driving an idealised representative diurnal cycle.
**Verify:** tank temperature crosses Tm during the solar window and the melt fraction runs 0 → 1 and back. This is an illustrative day-cycle schematic, **not** the enthalpy-porosity simulation — cite it as such.

### 13. Recommended PCM Summary per Cluster
`13_recommended_pcm_summary.png` · `..._interactive.html`
**Verify:** top-3 by consensus rank per cluster, annotated with Tm.

| Cluster | Top-3 by Borda consensus |
| :--- | :--- |
| 0 | RT50 · RT45HC · (n-Docosane C22 and Lauric acid C12 tied at rank 3) |
| 1 | savE® OM50 · Paraffin/HDPE PCM3 · Paraffin/HDPE PCM6 |
| 2 | savE® OM50 · Paraffin/HDPE PCM3 · Paraffin/HDPE PCM6 |

---

## 4. Verification suites

### A. Preprocessing (`verify_preprocessing/`)
`01_climate_distributions` · `02_data_completeness` · `03_statistical_summary` · `04_feature_engineering` · `05_correlation_analysis` · `06_data_quality_metrics` · `07_preprocessing_summary`

Reads `data/preprocessed/rajasthan_cleaned_physical.csv` — the only file carrying the engineered lag / rolling / delta columns, so plot 4 renders here (the older `plotting/verify_01_preprocessing_rajasthan.py` reads `climate_rajasthan_points_clean.csv` instead and silently skips it).

### B. Clustering (`verify_clustering/`)
`01_elbow_curves` · `02_silhouette_plot` · `03_pca_projection` · `04_geographic_map` · `05_cluster_profiles` · `06_cluster_sizes`

Runs on **Level A** (k = 3, one row per grid point) because that is the partition the whole PCM chain downstream uses. Set `LEVEL = "B"` at the top of the script to check the seasonal k = 8 partition instead.

Standardised `*_z` and `PC*` columns are excluded from the feature matrix — keeping them alongside their raw counterparts would double-count the same signal.

**Current result:** k = 3, average silhouette **0.313** (cluster 0: 0.296, cluster 1: 0.287, cluster 2: 0.359), Davies-Bouldin 1.130, Calinski-Harabasz 172.0.
The usual ">0.35 confirms valid separation" rule of thumb is **not** met at the state level. That is a genuine limitation to state plainly in the paper: Rajasthan's climate is comparatively homogeneous, so the regimes are contiguous gradients rather than well-separated blobs. The stronger evidence for k = 3 is the bootstrap stability already reported in `cluster_profiles_rajasthan.csv` (mean ARI **0.827** over 50 resamples); the Köppen cross-check there is weak on its own (ARI 0.189, NMI 0.319) and should be quoted as partial agreement, not validation.

### C. Feasibility (`verify_feasibility/`)
`01_survival_rate_by_cluster` · `02_feasible_property_space` · `03_top_candidates_per_cluster` · `04_constraint_analysis` · `05_property_distributions` · `06_summary`

Two corrections against `plotting/verify_03_feasibility_rajasthan.py`:
1. it plotted all 186 evaluation rows as survivors, so plot 1 reported "62 survivors" in every cluster instead of 9 / 14 / 16;
2. it looked for `pass_*` constraint columns, which Rajasthan does not have (they are `c1_melting_window` … `c8_safety` holding `pass`/`fail`/`not_applicable`/`flag_*` strings), so plot 4 was an empty placeholder.

**Constraint breakdown (186 evaluations):** `c6_charging_feasibility` is by far the tightest gate (~55 pass / 131 fail), then `c3_latent_heat` (~160 pass). `c7_corrosion_veto` and `c8_safety` are entirely `not_applicable`/flagged for this pool.

### D. Ranking (`verify_ranking/`)
`01_method_correlation` · `02_top3_inclusion_probability` · `03_rank_distributions` · `04_rank_reversal_frequency` · `05_method_agreement` · `06_summary`

Correlations are computed **within each cluster and then averaged** — pooling a 9-candidate cluster's ranks with a 16-candidate cluster's would mix two different scales. `borda_score` is used only to derive `consensus_rank` and is then dropped from the method list (the older script left it in, producing a spurious −1.00 row in the heatmap).

---

## 5. The curated `Plots/` folder (paper-facing set)

`build_plots_folder_rajasthan.py` assembles `PLOTSV2/Plots/` in the same 6-phase
layout as the project-root `Plots/` folder, one figure set per pipeline stage —
28 figures. It matches the Uttarakhand subfolder file-for-file except in phase 5,
where the single pooled bump chart becomes three per-cluster ones (see §3, plot 7):

| Phase folder | Figures |
| :--- | :--- |
| `1 Data collection/Rajasthan/` | `A_point_map` · `C_era5_vs_power` · `F_yearly_trend` |
| `2 Data Preprocessing/Rajasthan/` | `01_raw_vs_preprocessed_radiation` · `02_data_completeness` · `05_data_quality_metrics` · `06_correlation_analysis` · `07_preprocessing_summary` |
| `3 Climate Feature Engineering (Climate Signature)/Rajasthan/` | `point_signature_map` · `signature_correlation_heatmap` · `signature_distributions` |
| `4 Climate Region Discovery (Clustering)/Rajasthan/` | `01_elbow_curves` · `02_silhouette_plot` · `05_cluster_profiles` · `06_cluster_sizes` |
| `5 PCM Suitability Evaluation (MCDA)/Rajasthan/` | `03_melting_point_vs_latent_heat` · `04_constraint_analysis` · `04_feasible_candidates_highlighted` · `05_pcm_survivors_per_cluster` · `05_property_distributions` · `07_bump_chart_ranks_cluster_0/1/2` · `08_method_rank_correlation_heatmap` · `10_rank_reversal_violin_bar` |
| `6 PCM Recommendation and Output/Rajasthan/` | `11_agreement_plot` · `12_tank_temperature_melt_fraction` · `13_recommended_pcm_summary` |

`--mirror` also copies the tree into the project-root `Plots/` folder, dropping a
`Rajasthan/` subfolder alongside the existing `Tamilnadu/` and `Uttarakhand/` ones.
Without the flag nothing outside `PLOTSV2/` is touched.

Three things worth knowing about this folder:

- **Phase 2 numbering is deliberately swapped.** Uttarakhand's preprocessing suite
  writes `05_data_quality_metrics` / `06_correlation_analysis`; Tamil Nadu's and
  Rajasthan's write those two the other way round. The curated folder follows
  Uttarakhand's numbering, so the assembler renames them on copy — the
  Rajasthan `05_data_quality_metrics.png` here is `verify_preprocessing/06_data_quality_metrics.png`.
- **Phases 1 and 3 needed new scripts.** Rajasthan's own `03c_plots_raw_rajasthan.py`
  and `04_climate_signature_rajasthan.py` write these figures as interactive Plotly
  HTML only (that pipeline's convention), so there was no PNG to curate.
  `phase1_data_collection_rajasthan.py` and `phase3_climate_signature_rajasthan.py`
  are matplotlib ports of the Uttarakhand/Tamil Nadu originals, same filenames.
- **`Clustering.jpeg` has no Rajasthan equivalent.** The Uttarakhand phase-4 folder
  contains a hand-made conceptual diagram with no source script; nothing in this
  repo can regenerate it for Rajasthan.

### Phase-1 QA results (all checks pass)

| Check | Result |
| :--- | :--- |
| Sampling design | 320 population-weighted points, 70.3 M people covered, 2016–2025 |
| Timezone (noon must peak) | ✔ noon GHI 747.8 W/m² vs sunrise 1.1, sunset 56.0 |
| Missing data | ✔ 0.00 % across all 7 checked variables |
| Year-to-year discontinuity | ✔ mean noon GHI 732–758 W/m², no step change |
| ERA5 vs NASA POWER | GHI MBE +6.9 W/m², RMSE 83.3, r = 0.973; T_amb MBE +0.50 °C, r = 0.912; RHum MBE +7.7 %, r = 0.830 — this is the bias the quantile-mapping step in `04_preprocess_rajasthan.py` corrects |

---

## 6. How to sanity-check the numbers

1. **Physical plausibility**
   - Cluster mean ambient 26.5–27.8 °C, with the p05–p95 envelope roughly 17.5–35.6 °C (from `cluster_profiles_rajasthan.csv`).
   - Noon GHI peaks ≈ 900–1050 W/m²; daily total ≈ 5.2 kWh/m².
   - A usable PCM melts 15–25 °C above the cluster mean ambient — the capped targets of 48–52 °C fit.
2. **Statistical robustness**
   - Survival rate per cluster inside 10–50 % ✔ (14.5 / 22.6 / 25.8 %).
   - Monte Carlo top-3 stability ≥ 80 % for the rank-1 pick in every cluster ✔.
   - Method concordance ρ ≥ 0.70 holds for TOPSIS↔PROMETHEE II but **not** for GRA ✘ — report it, do not hide it.
   - Silhouette > 0.35 ✘ at 0.313 — see §4B.
3. **Provenance** — every downstream CSV carries `upstream_cluster_profile_fingerprint`. If it stops matching `cluster_profiles_rajasthan.csv`, the plots are stale; re-run the pipeline before re-running this folder.
