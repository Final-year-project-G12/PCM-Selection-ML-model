# ERA5 Tamil Nadu Solar Climate Dataset
## Multimodal Learning for Solar Energy Forecasting

> Based on: *"Multimodal Learning Techniques for Time Series Forecasting in Renewable Energy Systems"*  
> Mansouri et al., IEEE Access 2025

---

## Project Overview

This pipeline downloads, combines, preprocesses, and visualises two years of ERA5 hourly solar-climate data (2024–2025) across **222 locations in Tamil Nadu**, covering 8 climate zones. The cleaned, feature-engineered dataset is used to train multimodal deep learning models for GHI (Global Horizontal Irradiance) forecasting.

---

## Folder Structure

```
project/
├── data/
│   ├── raw/
│   │   └── era5/
│   │       └── grid/               ← Downloaded NetCDF files from CDS API
│   ├── processed/
│   │   └── climate_tamilnadu_all.csv   ← 3.9M rows × 37 cols, all 222 cities
│   └── preprocessed/               ← Output of 04_preprocess_tamilnadu.py
│       ├── train.csv               ← 70% temporal split
│       ├── val.csv                 ← 15% temporal split
│       ├── test.csv                ← 15% temporal split
│       ├── full_preprocessed.csv   ← All rows, normalised + engineered
│       ├── scalers.pkl             ← MinMaxScaler per column (for inference)
│       ├── feature_list.txt        ← All feature names used for modelling
│       └── preprocessing_report.txt
├── data/plots/                     ← Output of 05_plot_tamilnadu.py
│   ├── maps/                       ← Interactive HTML maps (open in browser)
│   ├── timeseries/                 ← PNG time series charts
│   ├── statistics/                 ← PNG statistical / distribution charts
│   ├── features/                   ← PNG feature engineering verification
│   └── solar_resource/             ← PNG solar quality charts
│
├── 00_unzip_accum.py               ← Fix ZIP-disguised .nc files from CDS
├── 01_download_era5_tamilnadu.py   ← Download ERA5 data via CDS API
├── 02_combine_tamilnadu.py         ← Combine NetCDF → single CSV
├── 04_preprocess_tamilnadu.py      ← Full preprocessing pipeline
├── 05_plot_tamilnadu.py            ← All visualisations
└── README.md
```

---

## Scripts — What Each One Does

### `00_unzip_accum.py`
CDS API sometimes downloads accumulated-variable `.nc` files that are actually ZIP archives. This script:
- Scans `data/raw/era5/grid/` for all `*_accum.nc` files
- Checks if each file is actually a ZIP (PK header)
- Extracts the real `.nc` from inside and replaces the fake file
- Safe to re-run — already-valid NetCDF files are skipped

**Run once before 02.**

---

### `01_download_era5_tamilnadu.py`
Downloads ERA5 reanalysis data from the Copernicus CDS API for the full Tamil Nadu bounding box.

| Parameter | Value |
|---|---|
| Bounding box | N=13.75, W=75.75, S=7.75, E=81.25 |
| Resolution | 0.25° × 0.25° (~25 km, ERA5 native) |
| Period | 2024-01-01 → 2025-12-31 |
| Total API calls | 48 (2 years × 12 months × 2 variable types) |

Two separate API calls per month (ERA5 rule):
- **Instant** (`_instant.nc`) — analysis variables: temperature, wind, humidity, pressure
- **Accum** (`_accum.nc`) — forecast variables: GHI, DNI, LW radiation, precipitation

Requires a `.cdsapirc` file with your CDS API key. Re-running is safe — completed files are skipped.

---

### `02_combine_tamilnadu.py`
Reads all downloaded NetCDF files, extracts the 222 named Tamil Nadu locations, computes derived variables (DHI, CSI, ETR, season, climate zone, etc.) and writes the final combined CSV.

**Output:** `data/processed/climate_tamilnadu_all.csv`  
- ~3.9 million rows × 37 columns  
- 222 cities × 17,520 hourly timestamps (2 years)

---

### `04_preprocess_tamilnadu.py`
Full preprocessing pipeline. Steps:

| Step | What it does |
|---|---|
| 1 | Load CSV with `engine="python"` to handle commas in city names |
| 2 | Impute missing values (solar=0 at night, weather=ffill, sunrise/sunset from solar geometry) |
| 3 | Physical bounds enforcement (clip outliers to realistic ranges) |
| 4 | Temporal alignment check (verify uniform 1-hour spacing) |
| 5 | Feature engineering: cyclical encoding, lag features (1/3/6/12/24h), rolling stats (3/6/24h windows), wind decomposition, daily statistics |
| 6 | Drop lag warmup rows (first 24h per city) |
| 7 | MinMaxScaler normalisation per column (scalers saved as `.pkl`) |
| 8 | Temporal train/val/test split: 70% / 15% / 15% |
| 9 | Save all outputs |

**Total features after engineering:** ~115 columns  
**Target variable:** GHI (W/m²)

---

### `05_plot_tamilnadu.py`
All visualisations. Produces:

**A. Interactive maps (HTML — open in browser)**
| File | Description |
|---|---|
| `A0_all_222_locations_overview.html` | **All 222 data locations**, colour-coded by climate zone, with popup for each city |
| `A1_GHI_mean_map.html` | Colour + size scaled by mean GHI, with GHI heatmap layer |
| `A2_climate_zone_map.html` | Each dot coloured by climate zone with legend |
| `A3_district_solar_resource.html` | District-level aggregated GHI |
| `A4_all_locations_india_context.html` | All 222 locations shown on India map |

**B. Time series (PNG)**
| File | Description |
|---|---|
| `B1_daily_GHI_districts.png` | 7-day rolling mean GHI for all districts |
| `B2_GHI_vs_clearsky.png` | Actual vs clearsky GHI for sample cities |
| `B3_Tamb_vs_GHI_scatter.png` | Temperature vs GHI scatter by climate zone |
| `B4_annual_cycle_GHI.png` | Monthly mean GHI for each climate zone |
| `B5_daily_GHI_all_cities.png` | All 222 city traces overlaid, TN mean highlighted |

**C. Statistical (PNG)**
| File | Description |
|---|---|
| `C1_correlation_matrix.png` | Feature correlation heatmap (daytime only) |
| `C2_GHI_violin_climate_zone.png` | GHI distribution violin plot by climate zone |
| `C3_diurnal_profile.png` | Hourly mean GHI by season |
| `C4_cloud_vs_GHI_density.png` | Cloud cover vs GHI 2D density |

**D. Feature engineering (PNG, needs `04_...` run first)**
| File | Description |
|---|---|
| `D1_lag_correlations.png` | Pearson correlation of lag features with GHI |
| `D2_rolling_mean.png` | Raw vs smoothed GHI comparison |
| `D3_train_val_test_split.png` | Timeline showing train/val/test boundaries |

**E. Solar resource quality (PNG)**
| File | Description |
|---|---|
| `E1_RRTDHS_heatmap.png` | Solar resource score — top 30 cities by month |
| `E2_CSI_distribution.png` | Clear Sky Index distribution overall + by season |
| `E3_top20_GHI_cities.png` | Horizontal bar chart of top 20 locations |

---

## How to Run

### Option A — VS Code / Local Jupyter

```bash
# 1. Install requirements
pip install cdsapi netCDF4 pandas numpy scikit-learn matplotlib seaborn folium branca

# 2. Download ERA5 data (needs CDS API key in ~/.cdsapirc)
python 01_download_era5_tamilnadu.py

# 3. Fix any ZIP-disguised accum files
python 00_unzip_accum.py

# 4. Combine NetCDF → CSV
python 02_combine_tamilnadu.py

# 5. Preprocess
python 04_preprocess_tamilnadu.py

# 6. Plot (HTML maps open automatically in your browser)
python 05_plot_tamilnadu.py
```

To convert to Jupyter notebook:
```bash
pip install jupytext
jupytext --to notebook 04_preprocess_tamilnadu.py
jupytext --to notebook 05_plot_tamilnadu.py
```

Or in VS Code: right-click the `.py` file → *Open With* → *Jupyter Notebook*

---

### Option B — Google Colab

**Step 1** — Upload `climate_tamilnadu_all.csv`:
```python
from google.colab import files
files.upload()   # select climate_tamilnadu_all.csv → lands at /content/
```
Or drag-drop it into the Files panel (folder icon, left sidebar).

**Step 2** — Set `COLAB = True` at the top of `04_preprocess_tamilnadu.py`, paste the whole script into a cell and run.

**Step 3** — Set `COLAB = True` at the top of `05_plot_tamilnadu.py`, paste and run.  
HTML maps display **inline** inside the notebook. PNG plots are saved to `/content/data/plots/`.

**Outputs are at:**
```
/content/data/preprocessed/   ← all preprocessing outputs
/content/data/plots/           ← all plots
```

---

## Dataset Variables (37 columns in combined CSV)

| Column | Unit | Description |
|---|---|---|
| `timestamp` | UTC | Hourly timestamp |
| `city` | — | Location name |
| `lat`, `lon` | degrees | ERA5 grid point coordinates |
| `altitude_m` | m | Elevation above sea level |
| `district` | — | Tamil Nadu district |
| `climate_zone` | — | One of 8 climate zone categories |
| `GHI` | W/m² | Global Horizontal Irradiance |
| `DNI` | W/m² | Direct Normal Irradiance |
| `DHI` | W/m² | Diffuse Horizontal Irradiance |
| `GHI_clearsky` | W/m² | Theoretical clear-sky GHI |
| `CSI` | — | Clear Sky Index (GHI / GHI_clearsky) |
| `ETR` | W/m² | Extraterrestrial Radiation |
| `avg_sdirswrf` | W/m² | Mean surface direct SW radiation flux |
| `LW_down` | W/m² | Downwelling longwave radiation |
| `T_amb` | °C | 2m air temperature |
| `T_dew` | °C | 2m dewpoint temperature |
| `RHum` | % | Relative humidity |
| `W_spd` | m/s | 10m wind speed |
| `W_dir` | degrees | 10m wind direction |
| `P_atm` | hPa | Surface pressure |
| `cloud_cover` | 0–1 | Total cloud cover fraction |
| `precipitation` | mm | Total precipitation |
| `SZA` | degrees | Solar Zenith Angle |
| `solar_azimuth` | degrees | Solar azimuth angle |
| `hour`, `month`, `DOY`, `year` | — | Time components |
| `season` | — | Winter / Summer / Monsoon / Retreat |
| `season_code` | 0–3 | Numeric season encoding |
| `sunrise_hour`, `sunset_hour` | hours | Computed from solar geometry |
| `RRTDHS` | 0–1 | Relative solar resource score |
| `high_solar_resource` | 0/1 | Flag: GHI > threshold |
| `T_set` | °C | Reference temperature for solar panels |

---

## Climate Zones in the Dataset

| Zone | Colour | Description |
|---|---|---|
| hot-humid-coastal | 🔵 blue | Coastal districts, high humidity |
| hot-humid | 🔵 light blue | Inland humid regions |
| hot-semi-arid | 🟠 orange | Semi-arid central TN |
| semi-arid | 🟡 yellow | Drier inland areas |
| hot-arid-coastal | 🔴 red | Arid coastal strips |
| hot-arid | 🟥 dark red | Arid inland |
| cool-hilly | 🟢 green | Western Ghats highlands (Ooty, Kodai) |
| semi-arid-elevated | 🩵 light teal | Elevated semi-arid terrain |
| hot-humid-elevated | 🟩 teal | Humid elevated areas |

---

## Key Design Decisions

**Why `engine="python"` in `pd.read_csv`?**  
Some city/district names in the CSV contain commas (e.g. `"Salem, TN"`). The default C parser treats this as an extra column and raises `ParserError: Expected 37 fields, saw 38`. The Python engine handles quoted fields correctly.

**Why compute `sunrise_hour`/`sunset_hour` from solar geometry?**  
ERA5 does not include sunrise/sunset as direct variables. The CDS output for these columns is 100% NaN. We compute them from solar declination and latitude using the standard hour-angle formula.

**Why temporal (not random) train/val/test split?**  
Random splitting causes data leakage — lag features computed from future timestamps would appear in the training set. Temporal splitting (first 70% of time for train) prevents this and reflects real forecasting conditions.

---

## Requirements

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.1
matplotlib >= 3.6
seaborn >= 0.12
folium >= 0.14
branca >= 0.6
cdsapi >= 0.6        (for download script only)
netCDF4 >= 1.6       (for combine script only)
```

Install all:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium branca cdsapi netCDF4
```

---

## Reference

> Mansouri, M., et al. (2025). *Multimodal Learning Techniques for Time Series Forecasting in Renewable Energy Systems*. IEEE Access.