# PCM-Selection-ML-model — Objective 1

Objective 1 of the Group 12 final-year project *Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating* (Amrita School of Engineering, guide Dr. T. Deepika).

**Objective 1 — climate-region-aware PCM recommendation.** Cluster ERA5 + NASA POWER weather data for population-weighted points in a state into climate regimes, then rank phase-change materials (PCMs) for each regime with multi-criteria decision-making (MCDM) and check the ranking against a grey-box tank simulation.

The same pipeline is run separately for four Indian states. Each state has its own folder, parameters and results; do not mix numbers across states.

| Folder | State | Runner |
|---|---|---|
| `era5-rajasthan/` | Rajasthan | `run_all_rajasthan.py` |
| `era5-tamilnadu/` | Tamil Nadu | `run_all_tamilnadu.py` |
| `era5-uttarakhand/` | Uttarakhand | `run_all_uttarakhand.py` |
| `era5-assam/` | Assam | `run_all_assam.py` |

Objective 2 (`../OBJECTIVE2/Objective---2/`) reads this repo's outputs, so file names under each state's `data/processed/` and `outputs/` folders should not be renamed casually.

---

## Pipeline phases

Every state follows the same numbered stages. Script names carry the stage number, but numbering is not always run order (for example `10_physics_validation.py` runs before `09_recommendation_cards.py`); each runner states its own order.

| Phase | Scripts | What it does |
|---|---|---|
| 0–1 Data collection | `00a_build_population_grid.py`, `00b_build_suntimes.py`, `00c_attach_elevation.py`, `01_download_era5_<state>.py`, `01b_download_nasapower.py`, `00_unzip_accum.py` | Population-weighted sample points, sunrise/noon/sunset times, ERA5 and NASA POWER downloads |
| 1 Merge | `02_combine_<state>.py` | ERA5 + NASA POWER at the sun-event hours → `climate_<state>_points.csv` |
| 2 Daily integrals | `02b_build_daily_aggregates*.py` | True daily totals from the NASA POWER hourly cache |
| 2.5 Preprocessing | `04_preprocess_<state>.py` | Physical bounds, night mask, outlier handling, imputation |
| 3 Climate signature | `04b_climate_signature.py` | One signature vector per point, including `Tm_target` and `L_required` |
| 4 Clustering | `05_cluster_<state>.py` | Gaussian Mixture climate regimes |
| 5 PCM database + feasibility | `06_build_pcm_database*.py`, `07_feasibility_filter*.py` | Builds the candidate list and filters it per regime |
| 6 MCDM ranking | `08_mcdm_ranking*.py` | TOPSIS, GRA, PROMETHEE II and VIKOR with a consensus rank, plus Monte Carlo stability |
| 7 Physics validation | `10_physics_validation.py` | Grey-box tank simulation of the surviving PCMs on each regime's real weather |
| 8 Recommendation | `09_recommendation_cards.py` | Final per-regime recommendation cards |

States differ in places, and each runner's docstring explains why:

- **Assam** keeps two Phase 5–8 chains (a historical K=4 chain and the final K=3 `_final` chain) and adds Phase 10–11 validation and consolidation scripts.
- **Uttarakhand** runs `09b_monte_carlo_stability.py` as a separate step and `07b_charging_feasibility.py` as an optional one.
- **Rajasthan** has extra post-Phase-6 sensitivity scripts.

---

## How to run a state

From inside the state folder:

```bash
python run_all_<state>.py --dry-run          # print the run order, run nothing
python run_all_<state>.py                    # core pipeline (Phase 1 merge onward)
python run_all_<state>.py --with-optional    # core + QC / diagnostic plot scripts
python run_all_<state>.py --from <script>.py # resume the core chain from one stage
python run_all_<state>.py --include-setup    # ALSO re-download raw data first (hours; needs credentials)
```

Rajasthan and Assam also take `--with-plots`, which runs the full figure set. For Tamil Nadu and Uttarakhand, the Objective-1 figures and verification plots are separate scripts:

- **Tamil Nadu:** `plots/generate_tamilnadu_plots.py`, `plots/comparison_plots_tamilnadu.py` and `plots/verify_0*_tamilnadu.py`.
- **Uttarakhand:** `generate_objective1_plots.py`, `comparison_plots_uttarakhand.py` and `verify_0*.py`.

`05c_explore_interactive.py` (and Rajasthan's `PLOTSV2/interactive_plots/`) are Streamlit apps: start them with `streamlit run <file>`.

**Raw data downloads** need a free Copernicus CDS account. Put the key in a `.cdsapirc` file in the state folder; it is git-ignored, never commit it.

---

## Shared inputs

- `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` — the manufacturer PCM property table every state's PCM database is built from. It is produced by the mini-pipeline `PCM_data/PCM_data/01_preprocess.py`, which the runners do not run. Uttarakhand reads its own copy in `era5-uttarakhand/PCM_data/data/`.
- `pcm_shared_config.py` — constants shared by Rajasthan and Tamil Nadu (population coverage target, delivery temperature, PCM share of night delivery, number of signature PCA components). Uttarakhand and Assam keep their own constants in their `config.py`.

---

## Where outputs go

| State | Data products | Figures |
|---|---|---|
| Rajasthan | `data/processed/`, `outputs/` (recommendation and cluster cards) | `data/plots/`; curated set in `data/plots/ppt/` |
| Tamil Nadu | `data/processed/`, `outputs/` | `data/plots/` |
| Uttarakhand | `data/processed/`, `outputs/` | `data/plots/` |
| Assam | `data/processed/`, `final_outputs/tables/` | `data/plots/`; curated set in `data/plots/ppt/`; Phase 11 copies in `final_outputs/visuals/` |

Every state uses the same figure folders and file names under `data/plots/`, so the same figure has the same path in every state. The full layout and numbering is in [`docs/PLOT_NAMING.md`](docs/PLOT_NAMING.md).

Raw downloads, processed CSVs and most generated plots are git-ignored (multi-GB; see `.gitignore`). A fresh clone needs the raw data downloaded or copied in before the core chain will run.

---

## Documentation

- `docs/<STATE>_FULL_PIPELINE.md` — the complete per-state write-up.
- `docs/<state>/` — per-phase audits, master overview and changelog for each state.
- `docs/METHODS.md` — method choice and rejected alternatives per script.
- `docs/PLOT_NAMING.md` — the plot folder layout and file names every state follows.
- `docs/references.bib` — bibliography; `docs/Literature Review.md` — literature review.
- `OBJECTIVE1_*.md` — consolidated explanation, summary, methods lookup and quick reference for Objective 1.
- `Sources/` — paper summaries (`*_summary.md`), extracted text (`extracted_papers/`) and the original PDFs (`pdfs/`).
- Each state folder has its own `README.md` and/or `PLOTS_GUIDE.md`.

---

## Requirements

Python 3.10+, then:

```bash
pip install numpy pandas scipy scikit-learn statsmodels xarray netCDF4 cdsapi requests pvlib geopandas shapely rasterio matplotlib seaborn plotly folium branca streamlit joblib
```

`geopandas`, `rasterio` and `cdsapi` are only needed for the Phase 0–1 setup scripts.
