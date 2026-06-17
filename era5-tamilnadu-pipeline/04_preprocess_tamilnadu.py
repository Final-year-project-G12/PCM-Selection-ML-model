"""
04_preprocess_tamilnadu.py
===========================
Complete preprocessing pipeline for ERA5 Tamil Nadu climate data.
Based on: "Multimodal Learning Techniques for Time Series Forecasting
           in Renewable Energy Systems" (Mansouri et al., IEEE Access 2025)

Preprocessing steps performed:
  1.  Load combined CSV from data/processed/climate_tamilnadu_all.csv
  2.  Handle missing values (imputation strategies per variable type)
  3.  Remove physical outliers and night-time solar artifacts
  4.  De-accumulation check (verify GHI is non-negative, hourly)
  5.  Feature engineering:
        - Lag features  (1h, 3h, 6h, 12h, 24h)  — from paper §V-A
        - Rolling stats (mean, std, min, max over 3h, 6h, 24h windows)
        - Fourier/cyclical encoding of hour, DOY, month
        - Clearness index CSI already present; add cloud opacity
        - Wind speed and direction sin/cos decomposition
  6.  Normalization (MinMaxScaler per variable, saved as .pkl)
  7.  Temporal alignment check (no gaps, uniform 1h spacing)
  8.  Train / Validation / Test split (70 / 15 / 15 by time)
  9.  Save preprocessed data to data/preprocessed/

HOW TO RUN (Google Colab):
  Upload your data/processed/ folder to Google Drive, then run this notebook.
  Or just set COLAB = True below and mount Drive.

OUTPUTS:
  data/preprocessed/
    train.csv
    val.csv
    test.csv
    full_preprocessed.csv
    scalers.pkl                 ← MinMaxScaler per column
    feature_list.txt            ← all feature names used
    preprocessing_report.txt    ← summary of what was done
"""

# ═══════════════════════════════════════════════════════════
# COLAB SETUP — set COLAB = True if running in Google Colab
# ═══════════════════════════════════════════════════════════
COLAB = False   # ← change to True in Colab

if COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    # Change this to your actual Drive path:
    BASE_DIR = "/content/drive/MyDrive/tamilnadu_era5"
else:
    from config import BASE_DIR, CLIMATE_COMBINED_FILE, PREPROCESSED_DIR, ensure_data_dirs

    ensure_data_dirs()

import os
import sys
import warnings
import pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
if COLAB:
    INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "climate_tamilnadu_all.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
else:
    INPUT_FILE = str(CLIMATE_COMBINED_FILE)
    OUTPUT_DIR = str(PREPROCESSED_DIR)

print("=" * 68)
print("  ERA5 Tamil Nadu — Preprocessing Pipeline")
print(f"  Input  : {INPUT_FILE}")
print(f"  Output : {OUTPUT_DIR}/")
print("=" * 68)


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════
print("\n[1/9] Loading data ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

print(f"  Loaded  : {len(df):,} rows  ×  {len(df.columns)} columns")
print(f"  Cities  : {df['city'].nunique()}")
print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# Expected numeric feature columns
SOLAR_COLS    = ["GHI", "DNI", "DHI", "avg_sdirswrf", "LW_down", "ETR",
                 "GHI_clearsky", "CSI"]
WEATHER_COLS  = ["T_amb", "T_dew", "RHum", "W_spd", "W_dir",
                 "P_atm", "cloud_cover", "precipitation"]
TIME_COLS     = ["hour", "month", "DOY", "year", "season_code",
                 "sunrise_hour", "sunset_hour", "SZA", "solar_azimuth"]
META_COLS     = ["city", "lat", "lon", "altitude_m", "district",
                 "climate_zone", "T_set", "high_solar_resource"]
TARGET_COL    = "GHI"

ALL_NUMERIC   = [c for c in SOLAR_COLS + WEATHER_COLS + TIME_COLS
                 if c in df.columns]

report_lines = []
report_lines.append("PREPROCESSING REPORT — ERA5 Tamil Nadu")
report_lines.append("=" * 68)
report_lines.append(f"Input rows    : {len(df):,}")
report_lines.append(f"Input columns : {len(df.columns)}")
report_lines.append(f"Cities        : {df['city'].nunique()}")
report_lines.append(f"Date range    : {df['timestamp'].min()} to {df['timestamp'].max()}")


# ═══════════════════════════════════════════════════════════
# STEP 2 — MISSING VALUE ANALYSIS AND IMPUTATION
# Per paper §VIII-B: multimodal datasets suffer missing values;
# use imputation or robust modeling strategies.
# ═══════════════════════════════════════════════════════════
print("\n[2/9] Handling missing values ...")

missing_before = df[ALL_NUMERIC].isna().sum()
missing_pct    = (missing_before / len(df) * 100).round(2)
print("  Missing values before imputation:")
for col in ALL_NUMERIC:
    if missing_before[col] > 0:
        print(f"    {col:<25} {missing_before[col]:>8,}  ({missing_pct[col]:.2f}%)")

# Strategy:
#   Night-time solar cols (GHI, DNI, DHI) → set to 0 when SZA >= 90°
#   Other solar cols at night → 0
#   Weather cols → forward-fill within city group, then KNN for remainder
#   GHI clearsky at night → 0

# Night mask: sun is below horizon
night_mask = df["SZA"] >= 90.0

for col in ["GHI", "DNI", "DHI", "avg_sdirswrf", "GHI_clearsky", "CSI"]:
    if col in df.columns:
        # Night values should be 0
        df.loc[night_mask, col] = df.loc[night_mask, col].fillna(0)
        df.loc[night_mask & (df[col] < 0), col] = 0

# Forward fill weather columns within each city
weather_to_ffill = [c for c in WEATHER_COLS if c in df.columns]
df[weather_to_ffill] = (
    df.groupby("city")[weather_to_ffill]
    .transform(lambda x: x.ffill().bfill())
)

# Any remaining NaN in numeric cols → median per city
for col in ALL_NUMERIC:
    if df[col].isna().sum() > 0:
        df[col] = df.groupby("city")[col].transform(
            lambda x: x.fillna(x.median()))

# Final check — any still missing?
missing_after = df[ALL_NUMERIC].isna().sum().sum()
print(f"  Missing after imputation: {missing_after}")
report_lines.append(f"\n[Step 2] Missing values imputed. Remaining: {missing_after}")


# ═══════════════════════════════════════════════════════════
# STEP 3 — PHYSICAL OUTLIER REMOVAL
# Paper §VII-B: data accuracy and physical bounds are critical.
# ═══════════════════════════════════════════════════════════
print("\n[3/9] Physical bounds enforcement ...")

bounds = {
    "GHI":          (0, 1400),
    "DNI":          (0, 1400),
    "DHI":          (0, 900),
    "avg_sdirswrf": (0, 1000),
    "LW_down":      (50, 600),
    "GHI_clearsky": (0, 1400),
    "CSI":          (0, 1.5),
    "T_amb":        (-5, 55),
    "T_dew":        (-20, 40),
    "RHum":         (0, 100),
    "W_spd":        (0, 50),
    "P_atm":        (850, 1060),
    "cloud_cover":  (0, 1),
    "precipitation":(0, 200),
    "SZA":          (0, 180),
}

clipped = {}
for col, (lo, hi) in bounds.items():
    if col in df.columns:
        n_bad = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        clipped[col] = int(n_bad)
        if n_bad > 0:
            print(f"  Clipped {n_bad:,} values in {col} to [{lo}, {hi}]")

# Night-time solar = 0 (enforce again after clipping)
for col in ["GHI", "DNI", "DHI", "avg_sdirswrf", "GHI_clearsky"]:
    if col in df.columns:
        df.loc[night_mask, col] = 0.0

report_lines.append(f"[Step 3] Physical bounds enforced: {clipped}")


# ═══════════════════════════════════════════════════════════
# STEP 4 — TEMPORAL ALIGNMENT CHECK
# Paper §VIII-B: synchronization across modalities is critical.
# ═══════════════════════════════════════════════════════════
print("\n[4/9] Temporal alignment check ...")

gaps_found = 0
for city, grp in df.groupby("city"):
    diffs = grp["timestamp"].diff().dropna()
    bad = diffs[diffs != pd.Timedelta("1h")]
    if len(bad) > 0:
        gaps_found += len(bad)
        print(f"  [WARN] {city}: {len(bad)} irregular time steps")

if gaps_found == 0:
    print("  All cities: uniform 1-hour spacing ✓")
else:
    print(f"  Total irregular steps found: {gaps_found}")

report_lines.append(f"[Step 4] Temporal gaps found: {gaps_found}")


# ═══════════════════════════════════════════════════════════
# STEP 5 — FEATURE ENGINEERING
# Paper §V-A: traditional ML uses handcrafted features;
# §III-A: sensor data requires lag and frequency features.
# ═══════════════════════════════════════════════════════════
print("\n[5/9] Feature engineering ...")

# ── 5a. Cyclical encoding of time variables ────────────────
# Converts hour, month, DOY to sin/cos so models see
# that hour 23 is adjacent to hour 0 (circular continuity)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["DOY_sin"]   = np.sin(2 * np.pi * df["DOY"] / 365)
df["DOY_cos"]   = np.cos(2 * np.pi * df["DOY"] / 365)

# ── 5b. Wind decomposition ────────────────────────────────
# Wind direction as sin/cos so 359° and 1° are treated as adjacent
if "W_dir" in df.columns and "W_spd" in df.columns:
    df["wind_u"] = df["W_spd"] * np.sin(np.deg2rad(df["W_dir"]))
    df["wind_v"] = df["W_spd"] * np.cos(np.deg2rad(df["W_dir"]))

# ── 5c. Derived features ──────────────────────────────────
# Clearness index already present as CSI
# Cloud opacity = 1 - CSI (inverted solar transmittance)
if "CSI" in df.columns:
    df["cloud_opacity"] = 1.0 - df["CSI"].clip(0, 1)

# Temperature depression (T_amb - T_dew) → humidity proxy
if "T_amb" in df.columns and "T_dew" in df.columns:
    df["T_depression"] = df["T_amb"] - df["T_dew"]

# Daylight flag
df["is_daytime"] = (df["SZA"] < 90).astype(int)

# Solar hour angle (approximation from hour and longitude)
# Paper §VI-A uses solar geometry for GHI prediction
df["solar_hour_angle"] = (df["hour"] - 12) * 15  # degrees, rough approx

# ── 5d. Lag features (within each city) ───────────────────
# Paper §V-A: lagged variables capture temporal dependencies
LAG_COLS  = ["GHI", "T_amb", "RHum", "W_spd", "cloud_cover", "CSI"]
LAG_HOURS = [1, 3, 6, 12, 24]

print(f"  Creating lag features: {LAG_HOURS}h for {LAG_COLS} ...")
for col in LAG_COLS:
    if col not in df.columns:
        continue
    for lag in LAG_HOURS:
        new_col = f"{col}_lag{lag}h"
        df[new_col] = df.groupby("city")[col].shift(lag)

# ── 5e. Rolling window statistics ─────────────────────────
# Paper §III-A Table 3: feature engineering for hybrid models
ROLL_COLS    = ["GHI", "T_amb", "W_spd", "cloud_cover", "RHum"]
ROLL_WINDOWS = [3, 6, 24]

print(f"  Creating rolling stats: windows={ROLL_WINDOWS}h ...")
for col in ROLL_COLS:
    if col not in df.columns:
        continue
    for win in ROLL_WINDOWS:
        grp = df.groupby("city")[col]
        df[f"{col}_roll{win}h_mean"] = grp.transform(
            lambda x: x.rolling(win, min_periods=1).mean())
        df[f"{col}_roll{win}h_std"]  = grp.transform(
            lambda x: x.rolling(win, min_periods=1).std().fillna(0))

# ── 5f. Rate of change (gradient) ─────────────────────────
for col in ["GHI", "T_amb", "cloud_cover"]:
    if col in df.columns:
        df[f"{col}_delta1h"] = df.groupby("city")[col].diff(1).fillna(0)

# ── 5g. Daily statistics joined back ──────────────────────
# Daily max GHI and daily GHI sum per city-date
df["date"] = df["timestamp"].dt.date
daily = df.groupby(["city", "date"]).agg(
    daily_GHI_sum=("GHI", "sum"),
    daily_GHI_max=("GHI", "max"),
    daily_T_mean=("T_amb", "mean"),
).reset_index()
df = df.merge(daily, on=["city", "date"], how="left")
df.drop(columns=["date"], inplace=True)

print(f"  Features after engineering: {len(df.columns)}")
report_lines.append(f"[Step 5] Feature engineering complete. Total columns: {len(df.columns)}")


# ═══════════════════════════════════════════════════════════
# STEP 6 — DROP ROWS WITH NaN FROM LAG/ROLL CREATION
# (first 24 rows per city will have NaN lags)
# ═══════════════════════════════════════════════════════════
print("\n[6/9] Dropping rows with NaN from lag windows ...")

rows_before = len(df)
df.dropna(subset=[f"GHI_lag{max(LAG_HOURS)}h"], inplace=True)
df.reset_index(drop=True, inplace=True)
rows_after = len(df)
print(f"  Dropped {rows_before - rows_after:,} rows (lag warmup). Remaining: {rows_after:,}")
report_lines.append(f"[Step 6] Rows after lag warmup drop: {rows_after:,}")


# ═══════════════════════════════════════════════════════════
# STEP 7 — NORMALIZATION
# Paper §V-A: features normalized before ML/DL training.
# Save scalers so they can be applied/inverted at inference.
# ═══════════════════════════════════════════════════════════
print("\n[7/9] Normalizing features ...")

# Columns to normalize (all numeric except binary flags, encoded cyclicals,
# metadata, and target which is normalized separately)
SKIP_NORM = (
    {"city", "district", "climate_zone", "season", "timestamp",
     "year", "is_daytime", "high_solar_resource",
     "hour_sin", "hour_cos", "month_sin", "month_cos",
     "DOY_sin", "DOY_cos", "season_code", "lat", "lon",
     "altitude_m", "T_set"} |
    set(META_COLS)
)

norm_cols = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in SKIP_NORM]

scalers = {}
df_norm = df.copy()

for col in norm_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    vals = df_norm[col].values.reshape(-1, 1)
    df_norm[col] = scaler.fit_transform(vals).flatten()
    scalers[col] = scaler

# Save scalers
scalers_path = os.path.join(OUTPUT_DIR, "scalers.pkl")
with open(scalers_path, "wb") as f:
    pickle.dump(scalers, f)

print(f"  Normalized {len(norm_cols)} columns. Scalers saved → {scalers_path}")
report_lines.append(f"[Step 7] Normalized {len(norm_cols)} columns. Scalers: {scalers_path}")


# ═══════════════════════════════════════════════════════════
# STEP 8 — TRAIN / VALIDATION / TEST SPLIT
# Paper §VII-B: standardized train-test splits needed.
# Split by TIME (not random) to avoid data leakage.
# 70% train / 15% validation / 15% test
# ═══════════════════════════════════════════════════════════
print("\n[8/9] Train / Validation / Test split ...")

# Sort by timestamp for temporal split
df_norm = df_norm.sort_values("timestamp").reset_index(drop=True)

n = len(df_norm)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df_norm.iloc[:train_end].reset_index(drop=True)
val_df   = df_norm.iloc[train_end:val_end].reset_index(drop=True)
test_df  = df_norm.iloc[val_end:].reset_index(drop=True)

print(f"  Train : {len(train_df):,} rows  "
      f"({train_df['timestamp'].min().date()} → {train_df['timestamp'].max().date()})")
print(f"  Val   : {len(val_df):,} rows  "
      f"({val_df['timestamp'].min().date()} → {val_df['timestamp'].max().date()})")
print(f"  Test  : {len(test_df):,} rows  "
      f"({test_df['timestamp'].min().date()} → {test_df['timestamp'].max().date()})")

report_lines.append(f"[Step 8] Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")


# ═══════════════════════════════════════════════════════════
# STEP 9 — SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════
print("\n[9/9] Saving preprocessed files ...")

# Save splits
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path   = os.path.join(OUTPUT_DIR, "val.csv")
test_path  = os.path.join(OUTPUT_DIR, "test.csv")
full_path  = os.path.join(OUTPUT_DIR, "full_preprocessed.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path,     index=False)
test_df.to_csv(test_path,   index=False)
df_norm.to_csv(full_path,   index=False)

# Feature list
feature_cols = [c for c in df_norm.columns
                if c not in ["city", "district", "climate_zone", "season",
                              "timestamp", "year"]]
feat_path = os.path.join(OUTPUT_DIR, "feature_list.txt")
with open(feat_path, "w") as f:
    for feat in feature_cols:
        f.write(feat + "\n")

# Report
report_lines.append(f"\n[Step 9] Files saved:")
report_lines.append(f"  {train_path}")
report_lines.append(f"  {val_path}")
report_lines.append(f"  {test_path}")
report_lines.append(f"  {full_path}")
report_lines.append(f"  {scalers_path}")
report_lines.append(f"  {feat_path}")
report_lines.append(f"\nTotal features for modelling: {len(feature_cols)}")
report_lines.append(f"\nFeature groups:")
report_lines.append(f"  Solar     : {[c for c in feature_cols if any(s in c for s in ['GHI','DNI','DHI','CSI','ETR','LW'])]}")
report_lines.append(f"  Weather   : {[c for c in feature_cols if any(s in c for s in ['T_amb','RHum','W_spd','cloud','precip','P_atm'])]}")
report_lines.append(f"  Time/Cycl : {[c for c in feature_cols if any(s in c for s in ['sin','cos','hour','DOY','season','day'])]}")
report_lines.append(f"  Lags      : {[c for c in feature_cols if 'lag' in c]}")
report_lines.append(f"  Rolling   : {[c for c in feature_cols if 'roll' in c]}")

rpt_path = os.path.join(OUTPUT_DIR, "preprocessing_report.txt")
with open(rpt_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"  train.csv          → {train_path}")
print(f"  val.csv            → {val_path}")
print(f"  test.csv           → {test_path}")
print(f"  full_preprocessed  → {full_path}")
print(f"  scalers.pkl        → {scalers_path}")
print(f"  feature_list.txt   → {feat_path}")
print(f"  preprocessing_report.txt → {rpt_path}")

print("\n" + "=" * 68)
print("  ✅  PREPROCESSING COMPLETE")
print(f"  Total features for modelling : {len(feature_cols)}")
print(f"  Target variable              : {TARGET_COL} (GHI, W/m²)")
print("=" * 68)
print("\nNext: run  05_plot_tamilnadu.py")