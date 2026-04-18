# Preprocessing Pipeline - Tamil Nadu ERA5 Data

## Overview

This preprocessing pipeline transforms raw Tamil Nadu ERA5 climate data (3,434,544 rows × 9 columns) into a production-ready dataset (3,425,160 rows × ~58 columns) suitable for:
- XGBoost PCM classifier training
- LSTM time-series forecasting
- RPi/TFLite model deployment

## Pipeline Architecture

```
INPUT FILES (tamilnadu_data/data/)
  ├── era5_climate_tamilnadu_2024.csv          (3,434,544 × 9)
  └── era5_climate_tamilnadu_2024_features.csv (3,434,544 × 33)
          ↓
   [STEP 0: Audit]           → Verify data integrity
          ↓
   [STEP 1: Conversions]     → Unit conversions + derived features
          ↓
   [STEP 2: Temporal]        → Cyclical, lags, rolling statistics
          ↓
   [STEP 3: Alignment]       → Handle lag NaNs, temporal continuity
          ↓
   [STEP 4: Outliers]        → Bounds checking, anomaly detection
          ↓
   [STEP 5: Spatial]         → Extract regional climate CSVs
          ↓
   [STEP 6: Scaling]         → Apply three-scaler strategy
          ↓
   [STEP 8: Final Output]    → Comprehensive audit & save final CSV
          ↓
OUTPUT FILES (tamilnadu_data/data/)
  ├── era5_tamilnadu_preprocessed_final.csv    (3,425,160 × 58) ⭐
  ├── scaler_minmax_tamilnadu.pkl              (for deployment)
  ├── scaler_standard_tamilnadu.pkl            (for deployment)
  └── scaler_robust_tamilnadu.pkl              (for deployment)
```

## Quick Start

### Option A: Run All Steps (Recommended)

```bash
cd f:\FInal Year Project\PCM-Selection-ML-model\tamilnadu_data\processing

python run_all_steps.py
```

This executes all 8 steps sequentially with error handling and progress reporting.

**Expected execution time:** ~10 hours (optimized to ~3 hours with parallel I/O possible)

### Option B: Run Individual Steps

```bash
# Step 0: Audit input files
python step_0_audit.py

# Step 1: Unit conversions
python step_1_conversions.py

# Step 2: Temporal features
python step_2_temporal_features.py

# Step 3: Temporal alignment
python step_3_alignment.py

# Step 4: Outlier detection
python step_4_outliers.py

# Step 5: Spatial joins
python step_5_spatial.py

# Step 6: Normalization
python step_6_scaling.py

# Step 8: Final output
python step_8_final_output.py
```

### Option C: Resume from a Specific Step

If a step fails, fix the issue and re-run from that step (intermediate CSVs are saved).

Example: If Step 4 fails, run steps 4+ after fixing:

```bash
python step_4_outliers.py
python step_5_spatial.py
python step_6_scaling.py
python step_8_final_output.py
```

## Step Descriptions

### Step 0: Audit & Validate (30 min)
- Load both input CSVs
- Verify zero NaNs, 391 locations, 8,784 hourly timestamps
- Generate audit report
- **Output:** `step_0_audit_report.txt`

### Step 1: Unit Conversions & Derived Features (1 hour)
- Convert ERA5 raw units (K → °C, Pa → hPa, etc.)
- Compute derived features (RH via Magnus, altitude via hypsometric, etc.)
- Validate against features.csv for consistency
- **Output:** `step_1_converted.csv`

### Step 2: Temporal Feature Engineering (1.5 hours)
- Cyclical encoding (sin/cos hour, month, DOY)
- Lagged features (GHI, T_amb at 1, 2, 3, 6, 12, 24 hours)
  - ⚠️ **CRITICAL:** Grouped by (lat, lon) to preserve spatial independence
- Rolling statistics (3h, 6h, 24h means/stds)
- Season one-hot encoding
- **Output:** `step_2_temporal_features.csv`
- **Note:** First 24 rows per location = NaN (handled in Step 3)

### Step 3: Temporal Alignment & Lag Handling (1 hour)
- Verify temporal continuity (8,784 unique timestamps, no gaps)
- **DROP option (chosen):** Remove first 24 rows per location
  - 391 × 24 = 9,384 rows dropped
  - Result: 3,434,544 - 9,384 = **3,425,160 rows**
- Per-location linear interpolation (max 6-hour gap)
- Create daytime subset (GHI > 10 W/m²) for XGBoost training
- **Output:** 
  - `step_3_aligned.csv` (full dataset)
  - `step_3_aligned_daytime.csv` (daytime only, ~50% of rows)

### Step 4: Outlier Detection & Cleaning (1.5 hours)
- Physical bounds validation (16 variables, specific ranges per variable)
- Isolation Forest anomaly detection on (GHI, DNI, DHI) triplet
  - Flags anomalies as feature (does NOT remove rows)
- Per-location linear interpolation for flagged NaNs
- **Output:** `step_4_cleaned.csv` (zero NaNs guaranteed)

### Step 5: Spatial Joins (Nearest Neighbor) (1.5 hours)
- Extract climate data for target installations:
  - Coimbatore SWH prototype: 11.0°N, 76.96°E
  - Jaisalmer reference: 26.92°N, 70.9°E
  - Chennai alternative: 13.08°N, 80.27°E
- Find nearest ERA5 grid cell via Euclidean distance
- **Output:**
  - `climate_Coimbatore_SWH.csv` (~8,784 rows)
  - `climate_Jaisalmer_SWH.csv` (~8,784 rows)
  - `climate_Chennai_SWH.csv` (~8,784 rows)
- **Use case:** Prototype environment setup, simulation input validation

### Step 6: Normalization & Feature Scaling (2 hours)
- **Three-scaler strategy** (data-leakage prevention):
  
  **MinMaxScaler [0,1]** (11 columns):
  - GHI, DNI, DHI, LW_down, SZA, CSI, cloud_cover, RH_percent, ETR, GHI_clearsky, solar_azimuth
  
  **StandardScaler (Z-score)** (23 columns):
  - T_amb, T_dew, T_depression, T_pcm_delta, P_atm_hPa, RRTDHS, GHI_clearsky_diff, altitude_m
  - Lagged features (12), rolling statistics (4), RRTDHS (1)
  
  **RobustScaler (IQR)** (3 columns):
  - Precip_mm, Wind_speed_ms, Wind_dir_deg (skewed, heavy-tail)
  
  **No scaling** (21 columns):
  - Cyclical: sin_hour, cos_hour, sin_month, cos_month, sin_DOY, cos_DOY
  - Categorical: hour, month, DOY, year, season_code, season_one-hots
  - Flags: solar_anomaly_flag, high_solar_resource
  - Spatial: lat, lon, timestamp

- **Train/test split:** Fit on Jan-Sep (before Oct 1), apply to full year
  - Prevents data leakage into test set
  - Training: 2,740,320 rows (~80%)
  - Test: 684,840 rows (~20%)

- **Output:**
  - `step_6_scaled.csv` (all features normalized)
  - `scaler_minmax_tamilnadu.pkl` (load on RPi for inference)
  - `scaler_standard_tamilnadu.pkl`
  - `scaler_robust_tamilnadu.pkl`

### Step 8: Final Output & Verification (30 min)
- Comprehensive audit of final dataset
- Verify:
  - Row count (expected 3,425,160)
  - Column count (~58)
  - NaN count (expected 0)
  - Temporal coverage (2024 full year)
  - Spatial coverage (391 grid points)
  - Scaler files saved
- **Output:**
  - `era5_tamilnadu_preprocessed_final.csv` ⭐ **MAIN OUTPUT**
  - `step_8_final_audit.txt` (detailed verification report)
  - `preprocessing_complete.log` (execution summary)

## Output Specification

### Final Dataset: `era5_tamilnadu_preprocessed_final.csv`

**Shape:** 3,425,160 rows × ~58 columns

**Column Inventory (58 total):**

| Category | Columns | Count |
|----------|---------|-------|
| Identifiers | timestamp, lat, lon | 3 |
| Raw Climate | T_amb, T_dew, RH_percent, Wind_speed_ms, Wind_dir_deg, GHI, DNI, DHI, LW_down, cloud_cover, Precip_mm, P_atm_hPa | 12 |
| Derived | T_depression, T_pcm_delta, altitude_m, GHI_clearsky_diff, CSI, RRTDHS | 6 |
| Solar Geometry | SZA, solar_azimuth, ETR, GHI_clearsky | 4 |
| Temporal Raw | hour, month, DOY, year, season_code | 5 |
| Temporal Cyclical | sin_hour, cos_hour, sin_month, cos_month, sin_DOY, cos_DOY | 6 |
| Season One-Hot | season_Winter, season_Summer, season_Monsoon, season_NE | 4 |
| Lagged GHI | GHI_lag1–24 | 6 |
| Lagged T_amb | T_amb_lag1–24 | 6 |
| Rolling Stats | GHI_roll3h_mean, GHI_roll6h_mean, GHI_roll24h_std, T_roll24h_mean | 4 |
| Anomaly Flags | solar_anomaly_flag | 1 |
| Targets | T_set, high_solar_resource | 2 |
| **TOTAL** | | **~58** |

**Data Quality:**
- ✅ Zero NaNs across all 3.4M rows
- ✅ No data leakage (scalers fit on training split only)
- ✅ Complete temporal coverage (2024-01-01 to 2024-12-31, hourly)
- ✅ Complete spatial coverage (391 grid points, Tamil Nadu 0.25° resolution)

**File Size:** ~1.6 GB

## Deployment

### On RPi / TFLite Pipeline

```python
import joblib
import pandas as pd

# Load scalers (saved during Step 6)
mm = joblib.load('scaler_minmax_tamilnadu.pkl')
ss = joblib.load('scaler_standard_tamilnadu.pkl')
rs = joblib.load('scaler_robust_tamilnadu.pkl')

# Prepare input sensor readings
sensor_array = np.array([[T_amb, T_dew, RH, ...]])  # 1 × n_features

# Apply scalers
scaled_array = mm.transform(sensor_array[:, minmax_cols])
scaled_array = ss.transform(sensor_array[:, standard_cols])
scaled_array = rs.transform(sensor_array[:, robust_cols])

# Feed to XGBoost/TFLite model
prediction = model.predict(scaled_array)
```

## Troubleshooting

### Step X Fails

1. Check the log file in `processing/step_X_*_report.txt`
2. Verify input file exists in `tamilnadu_data/data/`
3. Ensure sufficient RAM (16+ GB recommended)
4. Fix the issue and re-run from that step

### Memory Issues

- Process files in chunks (modify input steps to use `pd.read_csv(..., chunksize=100000)`)
- Or reduce to daytime subset only (`step_3_aligned_daytime.csv`)

### NaN Issues in Output

- Step 4 uses per-location interpolation (limit=6 hours)
- If gaps > 6 hours, small NaN sections may remain
- These are interpolated by forward/backward fill

## References

- **Temporal feature design:** Ghodusinejad et al. (2025)
- **Spatiotemporal resolution mismatch:** Mansouri et al. (2025) Section VIII.A
- **Solar geometry:** Spencer (1971), Haurwitz (1945), Bourges (1985)
- **Outlier detection:** Kou et al. (2025) — Isolation Forest methodology
- **Feature scaling:** Scikit-learn best practices

## File Structure

```
tamilnadu_data/
├── data/
│   ├── era5_climate_tamilnadu_2024.csv              (input)
│   ├── era5_climate_tamilnadu_2024_features.csv    (input)
│   ├── era5_tamilnadu_preprocessed_final.csv       (output) ⭐
│   ├── scaler_minmax_tamilnadu.pkl                 (output)
│   ├── scaler_standard_tamilnadu.pkl               (output)
│   └── scaler_robust_tamilnadu.pkl                 (output)
│
└── processing/                   ← YOU ARE HERE
    ├── step_0_audit.py           (verify input integrity)
    ├── step_1_conversions.py     (unit conversions)
    ├── step_2_temporal_features.py    (temporal engineering)
    ├── step_3_alignment.py       (temporal alignment)
    ├── step_4_outliers.py        (outlier detection)
    ├── step_5_spatial.py         (regional extraction)
    ├── step_6_scaling.py         (normalization)
    ├── step_8_final_output.py    (final audit)
    ├── run_all_steps.py          ← RUN THIS TO EXECUTE PIPELINE
    │
    ├── step_0_audit_report.txt
    ├── step_1_conversion_report.txt
    ├── step_2_temporal_report.txt
    ├── step_3_alignment_report.txt
    ├── step_4_outlier_report.txt
    ├── step_5_spatial_report.txt
    ├── step_6_scaling_report.txt
    ├── step_8_final_audit.txt
    └── preprocessing_complete.log    ← CHECK THIS AFTER PIPELINE FINISHES
```

## Next Steps After Preprocessing

1. **XGBoost PCM Classifier:**
   - Load `era5_tamilnadu_preprocessed_final.csv` or daytime subset
   - Train/test split: use Oct-Dec as held-out test set
   - Features: all except timestamps and identifiers

2. **Deployment on RPi:**
   - Load `scaler_*.pkl` files
   - Quantize/convert XGBoost → ONNX → TFLite
   - Deploy on RPi with live sensor inputs

3. **Thermal Simulation:**
   - Use `climate_Coimbatore_SWH.csv` as environment input
   - Run grey-box Barqawi model in Gym environment

---

**Status:** ✅ Ready to execute  
**Last Updated:** 2024-03-27  
**Author:** Preprocessing Pipeline v1.0
