# ╔══════════════════════════════════════════════════════════════════════╗
# ║  04_preprocess_tamilnadu.py                                          ║
# ║  ERA5 Tamil Nadu — Complete Preprocessing Pipeline                   ║
# ║                                                                      ║
# ║  Based on: "Multimodal Learning Techniques for Time Series           ║
# ║  Forecasting in Renewable Energy Systems"                            ║
# ║  Mansouri et al., IEEE Access 2025                                   ║
# ║                                                                      ║
# ║  HOW TO RUN                                                          ║
# ║  ──────────────────────────────────────────────────────────────────  ║
# ║  Option A — VS Code / Jupyter (.ipynb)                               ║
# ║    1. Open this file as a notebook (right-click → Open With →        ║
# ║       Jupyter Notebook) or convert:                                  ║
# ║         jupyter nbconvert --to notebook 04_preprocess_tamilnadu.py   ║
# ║    2. Set COLAB = False (default)                                    ║
# ║    3. Place climate_tamilnadu_all.csv at data/processed/ relative    ║
# ║       to this file and run all cells.                                ║
# ║                                                                      ║
# ║  Option B — Google Colab                                             ║
# ║    1. Upload climate_tamilnadu_all.csv via the Files panel           ║
# ║       (left sidebar → Upload) so it lands at /content/              ║
# ║    2. Set COLAB = True                                               ║
# ║    3. Run all cells                                                  ║
# ║                                                                      ║
# ║  OUTPUTS → data/preprocessed/  (local) or                           ║
# ║            /content/data/preprocessed/  (Colab)                     ║
# ║    train.csv, val.csv, test.csv, full_preprocessed.csv              ║
# ║    scalers.pkl, feature_list.txt, preprocessing_report.txt          ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 0. Mode switch ────────────────────────────────────────────────────
COLAB = False   # ← set True when running in Google Colab

# ── 1. Imports ────────────────────────────────────────────────────────
import os, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── 2. Path configuration ─────────────────────────────────────────────
if COLAB:
    # ── Auto-discover CSV (handles direct upload and Drive) ──────────
    _SEARCH = [
        "/content/climate_tamilnadu_all.csv",                               # direct upload
        "/content/drive/MyDrive/climate_tamilnadu_all.csv",                 # Drive root
        "/content/drive/MyDrive/tamilnadu_era5/data/processed/climate_tamilnadu_all.csv",
    ]
    INPUT_FILE = next((p for p in _SEARCH if os.path.exists(p)), None)
    if INPUT_FILE is None:
        try:
            from google.colab import files as _gf
            print("CSV not found — opening upload dialog …")
            _up = _gf.upload()
            INPUT_FILE = "/content/" + list(_up.keys())[0] if _up else None
        except Exception:
            pass
    if INPUT_FILE is None:
        raise FileNotFoundError(
            "\n❌  climate_tamilnadu_all.csv not found!\n"
            "Upload it via the Files panel or:\n"
            "    from google.colab import files; files.upload()"
        )
    OUTPUT_DIR = "/content/data/preprocessed"
else:
    # ── Local (VS Code / Jupyter) ─────────────────────────────────────
    _HERE      = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    INPUT_FILE = os.path.join(_HERE, "data", "processed", "climate_tamilnadu_all.csv")
    OUTPUT_DIR = os.path.join(_HERE, "data", "preprocessed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 68)
print("  ERA5 Tamil Nadu — Preprocessing Pipeline")
print(f"  Input  : {INPUT_FILE}")
print(f"  Output : {OUTPUT_DIR}/")
print("=" * 68)


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ───────────────────────────────────────────────────────────
# WHY engine="python" + on_bad_lines="warn"?
#   The CSV produced by 02_combine_tamilnadu.py sometimes has city
#   or district names containing a literal comma (e.g. "Salem, TN").
#   The default C parser treats that comma as an extra column separator,
#   counts 38 fields instead of 37 and raises:
#       ParserError: Expected 37 fields in line N, saw 38
#   engine="python" handles quoted fields with commas correctly.
#   on_bad_lines="warn" skips any truly malformed rows instead of crashing.
# ═══════════════════════════════════════════════════════════
print("\n[1/9] Loading data ...")

df = pd.read_csv(
    INPUT_FILE,
    engine="python",        # handles quoted commas inside field values
    on_bad_lines="warn",    # warn + skip malformed rows, don't crash
    parse_dates=["timestamp"],
)

# Force datetime in case parse_dates silently failed on some rows
if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

bad_ts = df["timestamp"].isna().sum()
if bad_ts > 0:
    print(f"  ⚠  Dropped {bad_ts} rows with unparseable timestamps")
    df = df[df["timestamp"].notna()].copy()

df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

print(f"  Loaded  : {len(df):,} rows  ×  {len(df.columns)} columns")
print(f"  Cities  : {df['city'].nunique()}")
print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# ── Column groups ──────────────────────────────────────────
SOLAR_COLS   = ["GHI","DNI","DHI","avg_sdirswrf","LW_down","ETR","GHI_clearsky","CSI"]
WEATHER_COLS = ["T_amb","T_dew","RHum","W_spd","W_dir","P_atm","cloud_cover","precipitation"]
TIME_COLS    = ["hour","month","DOY","year","season_code",
                "sunrise_hour","sunset_hour","SZA","solar_azimuth"]
META_COLS    = ["city","lat","lon","altitude_m","district",
                "climate_zone","T_set","high_solar_resource"]
TARGET_COL   = "GHI"
ALL_NUMERIC  = [c for c in SOLAR_COLS + WEATHER_COLS + TIME_COLS if c in df.columns]

report_lines = [
    "PREPROCESSING REPORT — ERA5 Tamil Nadu", "=" * 68,
    f"Input rows    : {len(df):,}", f"Input columns : {len(df.columns)}",
    f"Cities        : {df['city'].nunique()}",
    f"Date range    : {df['timestamp'].min()} to {df['timestamp'].max()}",
]


# ═══════════════════════════════════════════════════════════
# STEP 2 — MISSING VALUES
# ═══════════════════════════════════════════════════════════
print("\n[2/9] Handling missing values ...")

missing_before = df[ALL_NUMERIC].isna().sum()
missing_pct    = (missing_before / len(df) * 100).round(2)
print("  Missing values before imputation:")
for col in ALL_NUMERIC:
    if missing_before[col] > 0:
        print(f"    {col:<25} {missing_before[col]:>8,}  ({missing_pct[col]:.2f}%)")

# Night mask
if "SZA" in df.columns and df["SZA"].notna().any():
    night_mask = df["SZA"] >= 90.0
else:
    night_mask = ~df["hour"].between(6, 18)

# Solar = 0 at night
for col in ["GHI","DNI","DHI","avg_sdirswrf","GHI_clearsky","CSI"]:
    if col in df.columns:
        df.loc[night_mask, col] = df.loc[night_mask, col].fillna(0)
        df.loc[night_mask & (df[col] < 0), col] = 0

# Weather → ffill/bfill within city
weather_to_fill = [c for c in WEATHER_COLS if c in df.columns]
df[weather_to_fill] = df.groupby("city")[weather_to_fill].transform(
    lambda x: x.ffill().bfill())

# sunrise_hour / sunset_hour: 100% missing → compute from solar geometry
for col in ["sunrise_hour", "sunset_hour"]:
    if col in df.columns and df[col].isna().all():
        if "lat" in df.columns and "DOY" in df.columns:
            decl   = 23.45 * np.sin(np.deg2rad(360 / 365 * (df["DOY"] - 81)))
            cos_ha = (-np.tan(np.deg2rad(df["lat"])) *
                       np.tan(np.deg2rad(decl))).clip(-1, 1)
            ha_deg = np.degrees(np.arccos(cos_ha))
            df["sunrise_hour"] = (12 - ha_deg / 15).round(2)
            df["sunset_hour"]  = (12 + ha_deg / 15).round(2)
        else:
            df["sunrise_hour"] = 6.0
            df["sunset_hour"]  = 18.0

# Remaining NaN → city median → global median
fill_cols = [c for c in ALL_NUMERIC if c not in ("sunrise_hour","sunset_hour")]
for col in fill_cols:
    if df[col].isna().sum() > 0:
        df[col] = df.groupby("city")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

missing_after = df[ALL_NUMERIC].isna().sum().sum()
print(f"  Missing after imputation: {missing_after}")
report_lines.append(f"\n[Step 2] Missing values imputed. Remaining: {missing_after}")


# ═══════════════════════════════════════════════════════════
# STEP 3 — PHYSICAL BOUNDS ENFORCEMENT
# ═══════════════════════════════════════════════════════════
print("\n[3/9] Physical bounds enforcement ...")

bounds = {
    "GHI":(0,1400),"DNI":(0,1400),"DHI":(0,900),"avg_sdirswrf":(0,1000),
    "LW_down":(50,600),"GHI_clearsky":(0,1400),"CSI":(0,1.5),
    "T_amb":(-5,55),"T_dew":(-20,40),"RHum":(0,100),"W_spd":(0,50),
    "P_atm":(850,1060),"cloud_cover":(0,1),"precipitation":(0,200),"SZA":(0,180),
}
clipped = {}
for col, (lo, hi) in bounds.items():
    if col in df.columns:
        n_bad = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        clipped[col] = int(n_bad)
        if n_bad > 0:
            print(f"  Clipped {n_bad:,} values in {col} to [{lo}, {hi}]")

for col in ["GHI","DNI","DHI","avg_sdirswrf","GHI_clearsky"]:
    if col in df.columns:
        df.loc[night_mask, col] = 0.0

report_lines.append(f"[Step 3] Physical bounds enforced: {clipped}")


# ═══════════════════════════════════════════════════════════
# STEP 4 — TEMPORAL ALIGNMENT CHECK
# ═══════════════════════════════════════════════════════════
print("\n[4/9] Temporal alignment check ...")

if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

gaps_found = 0
for city, grp in df.groupby("city"):
    diffs    = grp["timestamp"].sort_values().diff().dropna()
    non_hour = diffs[diffs != pd.Timedelta("1h")]
    gaps_found += len(non_hour)

if gaps_found == 0:
    print("  All cities: uniform 1-hour spacing ✓")
else:
    print(f"  WARNING: {gaps_found} non-1h gaps found (pipeline continues)")
report_lines.append(f"[Step 4] Temporal gaps (non-1h): {gaps_found}")


# ═══════════════════════════════════════════════════════════
# STEP 5 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print("\n[5/9] Feature engineering ...")

# Cyclical encoding
for col, period in [("hour",24),("month",12),("DOY",365)]:
    if col in df.columns:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

# Season
if "season" not in df.columns and "month" in df.columns:
    _sm = {12:"Winter",1:"Winter",2:"Winter",
            3:"Summer",4:"Summer",5:"Summer",
            6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
           10:"Retreat",11:"Retreat"}
    df["season"]      = df["month"].map(_sm)
    df["season_code"] = df["season"].map(
        {"Winter":0,"Summer":1,"Monsoon":2,"Retreat":3})

# Wind decomposition
if "W_dir" in df.columns and "W_spd" in df.columns:
    df["W_dir_sin"] = np.sin(np.deg2rad(df["W_dir"])) * df["W_spd"]
    df["W_dir_cos"] = np.cos(np.deg2rad(df["W_dir"])) * df["W_spd"]

# Derived features
if "CSI" in df.columns:
    df["cloud_opacity"] = 1.0 - df["CSI"].clip(0, 1)
if "T_amb" in df.columns and "T_dew" in df.columns:
    df["T_depression"] = df["T_amb"] - df["T_dew"]
if "SZA" in df.columns:
    df["is_daytime"] = (df["SZA"] < 90).astype(int)
else:
    df["is_daytime"] = df["hour"].between(6, 18).astype(int)
if "hour" in df.columns:
    df["solar_hour_angle"] = (df["hour"] - 12) * 15

# Lag features
LAG_COLS  = [c for c in ["GHI","T_amb","RHum","W_spd","cloud_cover","CSI"] if c in df.columns]
LAG_HOURS = [1, 3, 6, 12, 24]
print(f"  Creating lag features: {LAG_HOURS}h for {LAG_COLS} ...")
for col in LAG_COLS:
    for lag in LAG_HOURS:
        df[f"{col}_lag{lag}h"] = df.groupby("city")[col].shift(lag)

# Rolling stats
ROLL_COLS    = [c for c in ["GHI","T_amb","W_spd","cloud_cover","RHum"] if c in df.columns]
ROLL_WINDOWS = [3, 6, 24]
print(f"  Creating rolling stats: windows={ROLL_WINDOWS}h ...")
for col in ROLL_COLS:
    for win in ROLL_WINDOWS:
        grp = df.groupby("city")[col]
        df[f"{col}_roll{win}h_mean"] = grp.transform(
            lambda x, w=win: x.rolling(w, min_periods=1).mean())
        df[f"{col}_roll{win}h_std"]  = grp.transform(
            lambda x, w=win: x.rolling(w, min_periods=1).std().fillna(0))

# Rate of change
for col in [c for c in ["GHI","T_amb","cloud_cover"] if c in df.columns]:
    df[f"{col}_delta1h"] = df.groupby("city")[col].diff(1).fillna(0)

# Daily statistics
df["_date"] = df["timestamp"].dt.date
daily = df.groupby(["city","_date"]).agg(
    daily_GHI_sum=("GHI","sum"),
    daily_GHI_max=("GHI","max"),
    daily_T_mean=("T_amb","mean"),
).reset_index()
df = df.merge(daily, on=["city","_date"], how="left")
df.drop(columns=["_date"], inplace=True)

print(f"  Features after engineering: {len(df.columns)}")
report_lines.append(f"[Step 5] Feature engineering complete. Total columns: {len(df.columns)}")


# ═══════════════════════════════════════════════════════════
# STEP 6 — DROP NaN FROM LAG WARMUP
# ═══════════════════════════════════════════════════════════
print("\n[6/9] Dropping rows with NaN from lag windows ...")

rows_before   = len(df)
lag_check_col = f"GHI_lag{max(LAG_HOURS)}h"
if lag_check_col in df.columns:
    df.dropna(subset=[lag_check_col], inplace=True)
df.reset_index(drop=True, inplace=True)
rows_after = len(df)
print(f"  Dropped {rows_before - rows_after:,} rows (lag warmup). Remaining: {rows_after:,}")
report_lines.append(f"[Step 6] Rows after lag warmup drop: {rows_after:,}")


# ═══════════════════════════════════════════════════════════
# STEP 7 — NORMALIZATION
# ═══════════════════════════════════════════════════════════
print("\n[7/9] Normalizing features ...")

SKIP_NORM = (
    {"city","district","climate_zone","season","timestamp","year",
     "is_daytime","high_solar_resource","hour_sin","hour_cos",
     "month_sin","month_cos","DOY_sin","DOY_cos","season_code",
     "lat","lon","altitude_m","T_set"} | set(META_COLS)
)
norm_cols = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in SKIP_NORM]

scalers = {}
df_norm = df.copy()
for col in norm_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_norm[col] = scaler.fit_transform(
        df_norm[col].values.reshape(-1, 1)).flatten()
    scalers[col] = scaler

scalers_path = os.path.join(OUTPUT_DIR, "scalers.pkl")
with open(scalers_path, "wb") as f:
    pickle.dump(scalers, f)
print(f"  Normalized {len(norm_cols)} columns. Scalers saved → {scalers_path}")
report_lines.append(f"[Step 7] Normalized {len(norm_cols)} columns.")


# ═══════════════════════════════════════════════════════════
# STEP 8 — TRAIN / VAL / TEST SPLIT (70/15/15, temporal)
# ═══════════════════════════════════════════════════════════
print("\n[8/9] Train / Validation / Test split ...")

df_norm   = df_norm.sort_values("timestamp").reset_index(drop=True)
n         = len(df_norm)
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
report_lines.append(
    f"[Step 8] Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")


# ═══════════════════════════════════════════════════════════
# STEP 9 — SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════
print("\n[9/9] Saving preprocessed files ...")

train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path   = os.path.join(OUTPUT_DIR, "val.csv")
test_path  = os.path.join(OUTPUT_DIR, "test.csv")
full_path  = os.path.join(OUTPUT_DIR, "full_preprocessed.csv")
feat_path  = os.path.join(OUTPUT_DIR, "feature_list.txt")
rpt_path   = os.path.join(OUTPUT_DIR, "preprocessing_report.txt")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path,     index=False)
test_df.to_csv(test_path,   index=False)
df_norm.to_csv(full_path,   index=False)

feature_cols = [c for c in df_norm.columns
                if c not in {"city","district","climate_zone",
                              "season","timestamp","year"}]
with open(feat_path, "w") as f:
    f.write("\n".join(feature_cols))

report_lines += [
    "\n[Step 9] Files saved:",
    f"  {train_path}", f"  {val_path}", f"  {test_path}",
    f"  {full_path}", f"  {scalers_path}", f"  {feat_path}",
    f"\nTotal features for modelling: {len(feature_cols)}",
    f"  Lags   : {[c for c in feature_cols if 'lag' in c]}",
    f"  Rolling: {[c for c in feature_cols if 'roll' in c]}",
]
with open(rpt_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"  train.csv              → {train_path}")
print(f"  val.csv                → {val_path}")
print(f"  test.csv               → {test_path}")
print(f"  full_preprocessed.csv  → {full_path}")
print(f"  scalers.pkl            → {scalers_path}")
print(f"  feature_list.txt       → {feat_path}")
print(f"  preprocessing_report   → {rpt_path}")

print("\n" + "=" * 68)
print("  ✅  PREPROCESSING COMPLETE")
print(f"  Total features : {len(feature_cols)}")
print(f"  Target         : {TARGET_COL} (GHI, W/m²)")
print("=" * 68)
print("\nNext: run  05_plot_tamilnadu.py")