# 01 — Filename Correspondence (Critical Reading Aid)

Every file in `D:\Final Year Project\tamilnadu\` has a filename that does not match its content.
This is confirmed three independent ways: (1) the project's own `README.md` documents it with a
35-row table and a stated root cause ("consistent with files having been downloaded one-at-a-time...
a browser auto-suffixes repeat downloads as `name (1).ext`, `name (2).ext`... and then re-associated
with the wrong original filename afterwards — the `(1)`, `(2)`, `(3)` suffixes... are the browser's
duplicate-download markers, not intentional versioning"); (2) independent spot-checks during this
audit (2/2 confirmed exactly, first lines of `00_unzip_accum (3).py` and `NEXT_STEPS (3).md` matched
the claimed true content); (3) the four research passes behind this documentation set each
independently opened and read every file's actual content in full.

**If you intend to run this pipeline, rename every file per the table below first.** The project's
own README states this is safe — file timestamps show nothing has been executed under the wrong
name yet.

## Full correspondence table (disk filename → true content → correct name)

| Disk filename | Actual content | Correct name |
|---|---|---|
| `00_unzip_accum (3).py` | Sun-event time table builder (pvlib SPA) | `00b_build_suntimes.py` |
| `00b_build_suntimes (3).py` | Population-weighted sampling-grid builder (WorldPop+GADM → ~133 pts) | `00a_build_population_grid.py` |
| `01_download_era5_tamilnadu (3).py` | NASA POWER hourly downloader | `01b_download_nasapower.py` |
| `01_preprocess (1).py` | 4-day sprint plan / project status doc (markdown) | `NEXT_STEPS.md` |
| `01b_download_nasapower (3).py` | ERA5 accumulated-field ZIP-file fixer | `00_unzip_accum.py` |
| `02_combine_tamilnadu (3).py` | True daily GHI/DTR/HDD/CDD integral builder from full NASA POWER hourly cache | `02b_build_daily_aggregates.py` |
| `02_cross_series_donor_audit (1).png` | **Python source**, not an image — PCM property-table imputation (MICE+RF+PMM) | `01_preprocess.py` |
| `02b_build_daily_aggregates (3).py` | ERA5 downloader (population points, sun-event hours, 10-yr) | `01_download_era5_tamilnadu.py` |
| `03_plots_raw (3).py` | Interactive (Plotly/Folium) raw-data QA dashboard | `03b_interactive_raw_qa.py` |
| `03b_interactive_raw_qa (3).py` | ERA5+NASA POWER combine script (deaccumulation, solar geometry, merge) | `02_combine_tamilnadu.py` |
| `04_correlation_heatmap (2).png` | **Genuine PNG** — correlation heatmap of the final imputed PCM property table | correctly named (content-wise) |
| `04_preprocess_tamilnadu (3).py` | Interactive explorer for the climate-signature output | `04d_signature_interactive.py` |
| `04b_climate_signature (3).py` | Raw-data QA plots (matplotlib PNGs) | `03_plots_raw.py` |
| `04c_interactive_postprocess_qc (3).py` | **Phase 3 — climate signature construction** | `04b_climate_signature.py` |
| `04c_postprocess_plots (3).py` | Interactive post-preprocessing QA dashboard | `04c_interactive_postprocess_qc.py` |
| `04d_signature_interactive (3).py` | Post-preprocessing QA plots (matplotlib PNGs) | `04c_postprocess_plots.py` |
| `05_cluster_regions (2).py` | Comprehensive visualization batch (Folium + matplotlib) | `07_plots_comprehensive.py` |
| `05_cluster_tamilnadu (3).py` | **Multi-region** GMM clustering (needs ≥2 states; not runnable — confirmed) | `05_cluster_regions.py` |
| `05_imputation_provenance (2).csv` | **Genuine PNG** — "missingness before/after" heatmap | `01_missingness_before_after.png` |
| `05b_cluster_interactive (3).py` | **Phase 2 — preprocessing & QC**, 13-step pipeline | `04_preprocess_tamilnadu.py` |
| `05c_explore_interactive (2).py` | Interactive cluster explorer | `05b_cluster_interactive.py` |
| `05d_plots_comprehensive (2).py` | **Streamlit** interactive explorer app | `06_explore_interactive.py` |
| `06_build_pcm_database (2).py` | **Phase 4 — climate regime clustering, single-state (the script actually used)** | `05_cluster_tamilnadu.py` |
| `07_feasibility_filter (2).py` | *Optional* charging-feasibility heuristic | `07b_charging_feasibility.py` |
| `07b_charging_feasibility (2).py` | **Phase 5 prep — PCM property database builder v2** | `06_build_pcm_database.py` |
| `08_mcdm_ranking (2).py` | **Phase 5 — feasibility filtering** | `07_feasibility_filter.py` |
| `09_recommendation_cards (2).py` | **Phase 6 — MCDM ranking engine** (TOPSIS + GRA) | `08_mcdm_ranking.py` |
| `NEXT_STEPS (3).md` | Shared config module (paths + CDS credential loader) | `config.py` |
| `PCM_Properties (2).csv` | **PNG image** — an imputation diagnostic plot | `02_cross_series_donor_audit.png` or `03_imputed_vs_reported_sanity.png` |
| `PCM_Properties_cleaned_mice_pmm (2).csv` | **PNG image** — the other diagnostic plot | (as above) |
| `PCM_Properties_cleaned_mice_pmm_detailed (2).csv` | Real CSV — imputation provenance table (donor traceability) | `05_imputation_provenance.csv` |
| `PREPROCESSING_STEPS (3).md` | Real CSV — raw, as-scraped manufacturer PCM property table | `PCM_Properties.csv` |
| `README (3).md` | Real CSV — lean cleaned PCM property table | `PCM_Properties_cleaned_mice_pmm.csv` |
| `README_PREPROCESSING (3).md` | Real CSV — detailed cleaned PCM table (value + imputed flag + original text) | `PCM_Properties_cleaned_mice_pmm_detailed.csv` |
| `config (3).py` | **Phase 8 — recommendation card generator** | `09_recommendation_cards.py` |

## What this audit independently added to the table

The project's own README already had this full table. This audit additionally verified, by full
read (not spot-check) of every script named above: exact function signatures, every hardcoded
constant, every formula, and cross-checked several TN scripts' outputs/logic directly against the
equivalent Rajasthan script where one exists — which is how the `L_required` bug (see
`05_PHASE_3_AUDIT.md`) was found: it is not mentioned in either `README.md` or `FIXES.md`, and only
surfaces when the TN and Rajasthan formulas are compared line-by-line, which this audit did.

## Practical guidance

Every phase-audit file in this documentation set (`03` through `09`) refers to scripts **by their
correct name only** (e.g. "`04b_climate_signature.py`"), with the disk filename noted once at the
top of the relevant section — consistent with how the project's own README chose to present it, and
for the same reason: this document should stay useful after the files are renamed.
