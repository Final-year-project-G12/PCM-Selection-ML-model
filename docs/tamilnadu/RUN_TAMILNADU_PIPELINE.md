# Tamil Nadu Pipeline — Run Guide

This file shows the recommended way to run the Tamil Nadu pipeline in order.

## 1) Quickest way: run the whole pipeline

From the project root:

```powershell
cd "c:\Users\jaswi\OneDrive\Desktop\ALL\SEM_7\project\jammalamadugu\new_obj\tamilnadu_pipeline"
python run_all_tamilnadu.py
```

This runs the main core pipeline only.

### Optional variants

```powershell
python run_all_tamilnadu.py --with-optional
```
Runs the main pipeline plus the diagnostic/plotting scripts.

```powershell
python run_all_tamilnadu.py --include-setup
```
Runs the raw-data acquisition scripts first as well.

```powershell
python run_all_tamilnadu.py --dry-run
```
Prints the execution order without running anything.

```powershell
python run_all_tamilnadu.py --from 05_cluster_tamilnadu.py
```
Starts from a later stage and skips earlier ones.

---

## 2) Manual run order for each script

If you want to run scripts one by one, use this order:

### Setup / data download (only if needed)

```powershell
python 00a_build_population_grid.py
python 00c_attach_elevation.py
python 00b_build_suntimes.py
python 01_download_era5_tamilnadu.py
python 01b_download_nasapower.py
python 00_unzip_accum.py
```

### Main processing chain

```powershell
python 02_combine_tamilnadu.py
python 02b_build_daily_aggregates.py
python 04_preprocess_tamilnadu.py
python 04b_climate_signature.py
python 05_cluster_tamilnadu.py
python 06_build_pcm_database.py
python 07_feasibility_filter.py
python 08_mcdm_ranking.py
python 10_physics_validation.py
python 09_recommendation_cards.py
python 11_seasonal_pcm_sensitivity.py
```

### Optional diagnostic / plotting scripts

```powershell
python 03_plots_raw.py
python 03b_agreement_analysis.py
python 03b_interactive_raw_qa.py
python 04c_postprocess_plots.py
python 04c_interactive_postprocess_qc.py
python 05b_cluster_interactive.py
python 05d_plots_comprehensive.py
python plots/generate_tamilnadu_plots.py
python plots/comparison_plots_tamilnadu.py
python plots/verify_01_preprocessing_tamilnadu.py
python plots/verify_02_clustering_tamilnadu.py
python plots/verify_03_feasibility_tamilnadu.py
python plots/verify_04_ranking_tamilnadu.py
```

---

## 3) Script purpose summary

| Script | Purpose | Notes |
|---|---|---|
| 00a_build_population_grid.py | Build population-weighted sampling points | One-time setup |
| 00c_attach_elevation.py | Attach real per-point elevation (added 2026-09-16) | One-time setup, after 00a |
| 00b_build_suntimes.py | Compute sunrise/noon/sunset times | One-time setup |
| 01_download_era5_tamilnadu.py | Download ERA5 data | One-time setup |
| 01b_download_nasapower.py | Download NASA POWER data | One-time setup |
| 00_unzip_accum.py | Fix zip-disguised .nc files | One-time setup |
| 02_combine_tamilnadu.py | Merge ERA5 + NASA POWER into final point dataset | Core |
| 02b_build_daily_aggregates.py | Build daily aggregates | Core |
| 04_preprocess_tamilnadu.py | Clean and QC data | Core |
| 04b_climate_signature.py | Create climate signatures | Core |
| 05_cluster_tamilnadu.py | Cluster regions (k auto-selected) | Core |
| 06_build_pcm_database.py | Build PCM candidate database | Core |
| 07_feasibility_filter.py | Filter feasible candidates (includes charging feasibility as Constraint 6 — the old separate `07b_charging_feasibility.py` was retired 2026-09-08 and deleted) | Core |
| 08_mcdm_ranking.py | Rank options with multi-criteria decision methods | Core |
| 10_physics_validation.py | Validate with physics simulation | Core |
| 09_recommendation_cards.py | Create recommendation outputs | Core |
| 11_seasonal_pcm_sensitivity.py | Seasonal sensitivity analysis (renamed 2026-09-08 from `11_level_b_seasonal_analysis.py`, which is deleted) | Run last |
| 03_plots_raw.py | Raw-data QC plots | Diagnostic |
| 03b_agreement_analysis.py | Source agreement analysis | Diagnostic |
| 03b_interactive_raw_qa.py | Interactive QA | Diagnostic |
| 04c_postprocess_plots.py | Postprocess plots | Diagnostic |
| 04c_interactive_postprocess_qc.py | Interactive QC | Diagnostic |
| 05b_cluster_interactive.py | Interactive cluster explorer | Diagnostic |
| 05d_plots_comprehensive.py | Full cluster plots | Diagnostic |
| plots/generate_tamilnadu_plots.py | Objective 1 PCM plot gallery (13 plots) | Diagnostic/deliverable |
| plots/comparison_plots_tamilnadu.py, plots/verify_01..04_*.py | Cross-step comparison + per-phase verification plots | Diagnostic |

---

## 4) Recommended workflow

For a normal run:

```powershell
python run_all_tamilnadu.py
```

If you want plots and QA outputs too:

```powershell
python run_all_tamilnadu.py --with-optional
```

If you are re-running after an earlier stage failed:

```powershell
python run_all_tamilnadu.py --from 05_cluster_tamilnadu.py
```

---

## 5) Important notes

- `run_all_tamilnadu.py` is the safest single entry point.
- The setup scripts download external data and may take a long time
  (`02_combine_tamilnadu.py` — solar-geometry computation over the full
  10-year record — is typically the single slowest core stage).
- `00c_attach_elevation.py` must run after `00a_build_population_grid.py`
  and before `02_combine_tamilnadu.py`; it's a single small CDS request
  (geopotential is time-invariant), not a per-year download.
- `11_seasonal_pcm_sensitivity.py` is meant to run last.
- The core pipeline is sequential: each step depends on outputs from the previous one.

---

## 6) If you want to run from PowerShell directly

```powershell
Set-Location "c:\Users\jaswi\OneDrive\Desktop\ALL\SEM_7\project\jammalamadugu\new_obj\tamilnadu_pipeline"
python .\run_all_tamilnadu.py --with-optional
```

This is the easiest way to run everything in the project.
