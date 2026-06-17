# PCM-SWH Data Pipeline
## Group 12 | Amrita School of Engineering | Guide: Dr. T. Deepika

### Objective 1: Forecast next-day solar irradiance & select optimal PCM
**Addresses: RG2 (no embedded prototype) + RG5 (climatic uncertainty)**

---

## Pipeline Overview

```
ERA5 API  ──►  01_download_era5.py   ──► data/raw/era5/*.nc
                      │
                      ▼
               02_process_era5.py   ──► data/processed/climate_all_cities.csv
                      │                  (GHI, DNI, DHI, T_amb, RHum, W_spd,
                      │                   CSI, SZA, ETR, RRTDHS, season, etc.)
Your PCM CSV ─►  03_process_pcm.py   ──► data/processed/pcm_cleaned.csv
                      │                  data/processed/pcm_label_mapping.csv
                      │
                      ▼
               04_fuse_data.py       ──► data/processed/climate_pcm_fused_hourly.csv
                      │                  data/processed/classifier_dataset.csv
                      │                  data/processed/pcm_label_encoder.csv
                      ▼
               05_train_forecaster.py ─► models/lstm_irradiance_forecaster/
                      │                  (predicts GHI_next_day, T_amb_next_day)
                      ▼
               06_train_pcm_classifier.py ► models/xgboost_pcm_classifier/
                                           (selects optimal PCM from climate inputs)
```

---

## File Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── era5/           ← ERA5 NetCDF files (downloaded)
│   │   └── pcm_data.csv    ← YOUR PCM CSV (copy here)
│   └── processed/
│       ├── climate_all_cities.csv
│       ├── pcm_cleaned.csv
│       ├── pcm_label_mapping.csv
│       ├── climate_pcm_fused_hourly.csv
│       ├── classifier_dataset.csv
│       └── pcm_label_encoder.csv
├── models/
│   ├── lstm_irradiance_forecaster/
│   └── xgboost_pcm_classifier/
├── pipeline/
│   ├── 01_download_era5.py
│   ├── 02_process_era5.py
│   ├── 03_process_pcm.py
│   ├── 04_fuse_data.py
│   ├── 05_train_forecaster.py   ← coming next
│   └── 06_train_pcm_classifier.py ← coming next
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up ERA5 access
1. Register at: https://cds.climate.copernicus.eu/
2. Accept ERA5 licence terms
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR-UID:YOUR-API-KEY
   ```

### 3. Place your PCM CSV
```bash
cp your_pcm_data.csv data/raw/pcm_data.csv
```

### 4. Run the pipeline (in order)
```bash
python pipeline/01_download_era5.py     # ~10-30 min (downloads ~500MB per city)
python pipeline/02_process_era5.py      # ~2-5 min
python pipeline/03_process_pcm.py       # ~30 seconds
python pipeline/04_fuse_data.py         # ~2-5 min
```

---

## Climate Features Produced (source-verified)

| Feature | Source Paper |
|---------|-------------|
| GHI, DNI, DHI (W/m²) | Barqawi 2025, Ghodusinejad 2026 |
| T_amb (°C) | All papers |
| RHum (%) | Barqawi 2025, Ghodusinejad 2026 |
| W_spd (m/s) | Barqawi 2025, Chen 2025 |
| Cloud cover | Ghodusinejad 2026 |
| CSI (Clear Sky Index) | Ghodusinejad 2026 |
| SZA, ETR | Ghodusinejad 2026 |
| RRTDHS index | **Kou 2025** (key PCM selector feature) |
| Season, DOY, Hour, Month | Barqawi 2025, Ghodusinejad 2026 |
| Sunrise/Sunset hour | Barqawi 2025 (Eq. 4) |
| Demand profile | **RG3 novel contribution** |
| Lag features (1/2/7 day) | Ghodusinejad 2026 |

## PCM Features Produced

| Feature | Source Paper |
|---------|-------------|
| Tm_melting, Tm_freezing | Singh 2025 |
| latent_heat_melting | Singh 2025, Kou 2025 (highest priority) |
| TC_liquid, TC_solid, TC_both | Kou 2025 (λ 5–9 W/mK optimal) |
| density_solid, density_liquid | Singh 2025 |
| Cp_liquid, Cp_solid | Singh 2025 |
| rho_H_MJ_m3 (volumetric enthalpy) | Kou 2025 (0–420 MJ/m³ range) |
| pcm_suitability_score | Singh 2025 priority weighting |
| pcm_label_code (TARGET) | Kou 2025 RRTDHS-based selection |

---

## Cities Validated

| City | Climate Zone | T_set | Lat/Lon |
|------|-------------|-------|---------|
| Coimbatore | hot-humid (Tamil Nadu) | 45°C | 11.0°N, 77.0°E |
| Jaisalmer | hot-arid (Rajasthan) | 55°C | 26.9°N, 70.9°E |

Add more cities by editing `CITIES` in `01_download_era5.py` and `02_process_era5.py`.
