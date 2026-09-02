"""
04_preprocess_rajasthan.py
============================
PHASE 2 — PREPROCESSING AND QUALITY CONTROL
(Objective 1 plan, Section 5; your 13-step methodology table)

INPUT  : data/processed/climate_rajasthan_points.csv   (02_combine output)
OUTPUT : data/preprocessed/
           rajasthan_cleaned_physical.csv   <- Phase 3 (04b) reads THIS.
                                                Physical units, QC-passed,
                                                imputed. NOT scaled — Phase 3
                                                indices (GHI_daily_kWh, HDD18,
                                                CDD24, ...) are non-linear
                                                functions of physical values;
                                                scaling first would silently
                                                corrupt them (plan doc §5.2).
           rajasthan_cleaned_scaled.csv     <- Same rows, MinMax-scaled
                                                feature columns, for any
                                                later ML/DRL use (mirrors
                                                what the old 04_preprocess
                                                did for the encoder).
           scalers.pkl                      <- per-column MinMaxScaler,
                                                fit on the first 70% of
                                                chronologically-sorted rows
                                                only (no leakage).
           qc_report.txt                    <- every check, every count.
           correlation_pearson.csv / _spearman.csv / heatmaps
           vif_report.csv
           yeo_johnson_skew.csv
           savitzky_golay_diagnostic.png

IMPORTANT SCHEMA DIFFERENCE FROM THE OLD PIPELINE
---------------------------------------------------
The old 04_preprocess_rajasthan.py assumed one row per (city, hour) —
a continuous 24h/day series. This dataset is one row per
(point_id, date, event) with event in {sunrise, noon, sunset} — 3
samples/day, not 24. Every "rolling"/"lag"/"delta" concept below is
therefore redefined over EVENT OCCURRENCES within a (point_id, event)
group, sorted by date — e.g. "lag7" = the same sun-event 7 days earlier,
not 7 hours earlier. This is called out at each step so it's traceable
in your methodology write-up.

HOW TO RUN:
  python 04_preprocess_rajasthan.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# Disable parallelization to avoid Windows subprocess issues
os.environ["SKLEARN_N_JOBS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import COMBINED_POINTS_FILE, PREPROCESSED_DIR

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ORDER = ["sunrise", "noon", "sunset"]
TRAIN_FRAC = 0.70  # chronological — matches the rest of your pipeline's convention

# Physical bounds (Table 9, plan doc + your original 04's bounds dict)
BOUNDS = {
    "era5_GHI": (0, 1400), "era5_DNI": (0, 1400), "era5_DHI": (0, 900),
    "era5_GHI_clearsky": (0, 1400), "era5_CSI": (0, 1.5),
    "era5_LW_down": (50, 600),
    "era5_T_amb": (-30, 55), "era5_T_dew": (-30, 40), "era5_RHum": (0, 100),
    "era5_W_spd": (0, 50), "era5_P_atm": (850, 1060),
    "era5_cloud_cover": (0, 1), "era5_precipitation": (0, 200),
    "era5_SZA": (0, 180),
    "power_ALLSKY_SFC_SW_DWN": (0, 1400), "power_CLRSKY_SFC_SW_DWN": (0, 1400),
    "power_T2M": (-30, 55), "power_RH2M": (0, 100), "power_WS10M": (0, 50),
}

ERA5_SOLAR_COLS = ["era5_GHI", "era5_DNI", "era5_DHI", "era5_GHI_clearsky", "era5_CSI"]

LAG_COLS = ["era5_GHI", "era5_T_amb", "era5_RHum", "era5_W_spd",
            "era5_cloud_cover", "era5_CSI"]
LAG_OCCURRENCES = [1, 7, 30]        # in event-occurrences: 1 day, ~1 week, ~1 month prior
ROLL_OCCURRENCES = [7, 30]          # trailing window, in event-occurrences

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(str(msg))


# ═══════════════════════════════════════════════════════════
# STEP 1 — DATASET INSPECTION
# ═══════════════════════════════════════════════════════════
log("=" * 68)
log("  PHASE 2 — PREPROCESSING & QUALITY CONTROL — Rajasthan")
log("=" * 68)
log("\n[1/13] Dataset inspection ...")

df = pd.read_csv(COMBINED_POINTS_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)

log(f"  Shape: {df.shape}")
log(f"  Dtypes:\n{df.dtypes.value_counts().to_string()}")

dup_mask = df.duplicated(subset=["point_id", "date", "event"])
log(f"  Duplicate (point_id, date, event) rows: {dup_mask.sum()}")
if dup_mask.sum() > 0:
    df = df[~dup_mask].copy()
    log("  -> Dropped duplicates.")

numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
missing_before = df[numeric_cols_all].isna().mean() * 100
log("\n  Missing % (top 15, before any cleaning):")
log(missing_before.sort_values(ascending=False).head(15).round(2).to_string())


# ═══════════════════════════════════════════════════════════
# STEP 2 — PHYSICAL VALIDATION
# ═══════════════════════════════════════════════════════════
log("\n[2/13] Physical validation ...")

clipped_counts = {}
for col, (lo, hi) in BOUNDS.items():
    if col not in df.columns:
        continue
    bad = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
    n_bad = int(bad.sum())
    clipped_counts[col] = n_bad
    df.loc[bad, col] = np.nan   # NaN, not silent clip — matches your "safer than clipping" rule
    if n_bad > 0:
        log(f"  {col}: {n_bad:,} out-of-range values -> set to NaN "
            f"(bounds [{lo}, {hi}])")

# Night-masking equivalent: even though every row IS a sun-event, the
# NEAREST-HOUR match to a real ERA5/POWER reading can land a few hours off
# true sun position (see 02_combine's MAX_MATCH_HOURS=3), so a "sunrise"
# row can occasionally have SZA > 90 (still below horizon) or a "sunset"
# row SZA < 90. Force solar fields to 0 whenever SZA says the sun is down
# — this is Table 9 check #4, applied at row level instead of hour level.
if "era5_SZA" in df.columns:
    below_horizon = df["era5_SZA"] >= 90.0
    for col in ERA5_SOLAR_COLS:
        if col in df.columns:
            n_forced = int((below_horizon & (df[col] > 0)).sum())
            df.loc[below_horizon, col] = 0.0
            if n_forced:
                log(f"  {col}: {n_forced:,} values forced to 0 (SZA >= 90, sun below horizon)")

if "era5_RHum" in df.columns:
    df["era5_RHum"] = df["era5_RHum"].clip(0, 100)
if "power_RH2M" in df.columns:
    df["power_RH2M"] = df["power_RH2M"].clip(0, 100)

log(f"\n  Total values NaN'd by physical validation: {sum(clipped_counts.values()):,}")


# ═══════════════════════════════════════════════════════════
# STEP 2b — PER-SEASON QUANTILE MAPPING (ERA5 GHI → NASA POWER)
# ═══════════════════════════════════════════════════════════
log("\n[2b/13] Per-season quantile mapping (daytime era5_GHI > 0) ...")

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
N_QUANTILES = 100
ERA5_GHI_COL = "era5_GHI"
POWER_GHI_COL = "power_ALLSKY_SFC_SW_DWN"

if ERA5_GHI_COL in df.columns and POWER_GHI_COL in df.columns and "season" in df.columns:
    df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
    qmap_rows = []

    def _fit_quantile_mapper(era5_vals, power_vals):
        qs = np.linspace(0, 1, N_QUANTILES + 1)
        era5_q = np.quantile(era5_vals, qs)
        power_q = np.quantile(power_vals, qs)
        era5_q_u, idx = np.unique(era5_q, return_index=True)
        power_q_u = power_q[idx]

        def mapper(x):
            return np.interp(x, era5_q_u, power_q_u, left=power_q_u[0], right=power_q_u[-1])

        return mapper

    for season in SEASON_ORDER:
        mask = (
            (df["season"] == season)
            & (df[ERA5_GHI_COL] > 0)
            & df[ERA5_GHI_COL].notna()
            & df[POWER_GHI_COL].notna()
        )
        paired = df.loc[mask]
        if len(paired) < 30:
            log(f"  {season}: only {len(paired)} paired daytime rows — skipping QM")
            continue
        mapper = _fit_quantile_mapper(paired[ERA5_GHI_COL].values, paired[POWER_GHI_COL].values)
        before_mbe = float((paired[ERA5_GHI_COL] - paired[POWER_GHI_COL]).mean())
        corrected = mapper(paired[ERA5_GHI_COL].values)
        after_mbe = float((corrected - paired[POWER_GHI_COL]).mean())
        df.loc[paired.index, ERA5_GHI_COL] = corrected
        if "era5_CSI" in df.columns and "era5_GHI_clearsky" in df.columns:
            cs = df.loc[paired.index, "era5_GHI_clearsky"]
            df.loc[paired.index, "era5_CSI"] = np.where(
                cs > 10, (corrected / cs).clip(0, 1.5), 0)
        qmap_rows.append({"season": season, "n": len(paired),
                          "MBE_before": round(before_mbe, 2),
                          "MBE_after": round(after_mbe, 2)})
        log(f"  {season}: n={len(paired):,}  MBE {before_mbe:>7.2f} -> {after_mbe:>7.2f} W/m²")

    if qmap_rows:
        qmap_df = pd.DataFrame(qmap_rows)
        qmap_df.to_csv(PREPROCESSED_DIR / "ghi_quantile_mapping_report.csv", index=False)
        log("  Saved: ghi_quantile_mapping_report.csv")
else:
    log("  [SKIP] era5_GHI / power_ALLSKY_SFC_SW_DWN / season not all present")


# ═══════════════════════════════════════════════════════════
# STEP 3 — HAMPEL FILTER / MAD  (per point, per event, over date order)
# ═══════════════════════════════════════════════════════════
log("\n[3/13] Hampel filter (MAD-based outlier flagging) ...")
log("  Window = trailing+leading 15 occurrences of the SAME (point_id, event) "
    "series, sorted by date (not hours — see module docstring).")

HAMPEL_WINDOW = 15   # occurrences each side
HAMPEL_N_SIGMA = 3.0
HAMPEL_COLS = ["era5_GHI", "era5_T_amb", "era5_RHum", "era5_W_spd", "era5_cloud_cover"]

df = df.sort_values(["point_id", "event", "date"]).reset_index(drop=True)

hampel_flag_counts = {}
for col in HAMPEL_COLS:
    if col not in df.columns:
        continue
    grp = df.groupby(["point_id", "event"], observed=True)[col]
    roll_median = grp.transform(lambda s: s.rolling(HAMPEL_WINDOW * 2 + 1,
                                                      center=True, min_periods=5).median())
    roll_mad = grp.transform(
        lambda s: (s - s.rolling(HAMPEL_WINDOW * 2 + 1, center=True, min_periods=5).median())
        .abs().rolling(HAMPEL_WINDOW * 2 + 1, center=True, min_periods=5).median())
    threshold = HAMPEL_N_SIGMA * 1.4826 * roll_mad  # 1.4826 = MAD->sigma consistency constant
    is_outlier = (df[col] - roll_median).abs() > threshold
    is_outlier &= roll_mad > 1e-6  # skip flat/constant stretches (mad=0 would flag everything)
    n_flagged = int(is_outlier.sum())
    hampel_flag_counts[col] = n_flagged
    df.loc[is_outlier, col] = np.nan   # flag -> NaN, handled by imputation next (never auto-deleted)
    log(f"  {col}: {n_flagged:,} points flagged as Hampel outliers -> NaN "
        f"({100*n_flagged/len(df):.3f}% of rows)")


# ═══════════════════════════════════════════════════════════
# STEP 3b — YEO-JOHNSON SKEW DIAGNOSTIC  (report only, does not touch data)
# ═══════════════════════════════════════════════════════════
log("\n[3b/13] Yeo-Johnson skew diagnostic (report-only) ...")

YJ_COLS = ["era5_GHI", "era5_W_spd", "era5_precipitation", "era5_cloud_cover", "era5_T_amb"]
yj_rows = []
for col in YJ_COLS:
    if col not in df.columns:
        continue
    vals = df[col].dropna().values
    if len(vals) < 100:
        continue
    skew_before = float(stats.skew(vals))
    try:
        transformed, lam = stats.yeojohnson(vals)
        skew_after = float(stats.skew(transformed))
    except Exception:
        lam, skew_after = np.nan, np.nan
    yj_rows.append({"column": col, "skew_before": round(skew_before, 3),
                     "yeo_johnson_lambda": round(lam, 4) if lam == lam else None,
                     "skew_after": round(skew_after, 3) if skew_after == skew_after else None})
yj_df = pd.DataFrame(yj_rows)
yj_df.to_csv(PREPROCESSED_DIR / "yeo_johnson_skew.csv", index=False)
log(yj_df.to_string(index=False))
log("  Saved: yeo_johnson_skew.csv  (diagnostic only — no columns transformed)")


# ═══════════════════════════════════════════════════════════
# STEP 4 — HIERARCHICAL IMPUTATION + MICE
# ═══════════════════════════════════════════════════════════
log("\n[4/13] Hierarchical imputation (interpolate -> ffill/bfill -> zone/global median -> MICE) ...")

IMPUTE_COLS = [c for c in numeric_cols_all if c in df.columns and c not in
               ("lat", "lon", "population", "weight", "grid_lat", "grid_lon",
                "month", "DOY", "year", "season_code")]

missing_pre_impute = df[IMPUTE_COLS].isna().sum().sum()
log(f"  Missing cells entering imputation: {missing_pre_impute:,}")

# (a) Linear interpolation for short interior gaps (<=3 occurrences) within
# each (point_id, event) series
df = df.sort_values(["point_id", "event", "date"]).reset_index(drop=True)
for col in IMPUTE_COLS:
    df[col] = df.groupby(["point_id", "event"], observed=True)[col].transform(
        lambda s: s.interpolate(method="linear", limit=3, limit_area="inside"))

# (b) Forward/backward fill within the same series (edge gaps interpolation can't reach)
for col in IMPUTE_COLS:
    df[col] = df.groupby(["point_id", "event"], observed=True)[col].transform(
        lambda s: s.ffill(limit=3).bfill(limit=3))

remaining_after_ab = df[IMPUTE_COLS].isna().sum().sum()
log(f"  Remaining after interpolate+ffill/bfill: {remaining_after_ab:,}")

# (c) point -> impute_zone -> global median.
# No district/climate_zone metadata exists for population points (unlike the
# old named-city dict), so a lightweight spatial grouping is built here
# PURELY as an imputation fallback grouping — this is NOT the Phase 4
# climate clustering, just named `impute_zone` to avoid confusion with it.
point_coords = df.groupby("point_id")[["lat", "lon"]].first()
N_IMPUTE_ZONES = min(8, len(point_coords))
km = KMeans(n_clusters=N_IMPUTE_ZONES, random_state=42, n_init=10)
point_coords["impute_zone"] = km.fit_predict(point_coords[["lat", "lon"]])
df["impute_zone"] = df["point_id"].map(point_coords["impute_zone"])

for col in IMPUTE_COLS:
    df[col] = df.groupby("point_id")[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df.groupby("impute_zone")[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df[col].fillna(df[col].median())

remaining_after_c = df[IMPUTE_COLS].isna().sum().sum()
log(f"  Remaining after point/zone/global median: {remaining_after_c:,}")

# (d) MICE (IterativeImputer) for any still-missing cells — should be ~0
# after (a)-(c) given the population-point pipeline's typical completeness,
# but kept as the documented final fallback per your spec.
still_missing = df[IMPUTE_COLS].isna().sum().sum()
if still_missing > 0:
    log(f"  Running MICE on remaining {still_missing:,} cells "
        f"(fit on a 300k-row sample) ...")
    mice_cols = [c for c in IMPUTE_COLS if df[c].isna().any()]
    fit_cols = [c for c in IMPUTE_COLS if df[c].notna().all() or c in mice_cols]
    sample = df[fit_cols].sample(min(300_000, len(df)), random_state=42)
    imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False)
    imputer.fit(sample)
    df[fit_cols] = imputer.transform(df[fit_cols])
    log(f"  MICE complete. Remaining missing: {df[IMPUTE_COLS].isna().sum().sum():,}")
else:
    log("  No residual missing cells — MICE not needed.")

log(f"\n  Total imputed cells (steps a-d combined): {missing_pre_impute:,} "
    f"({100*missing_pre_impute/(len(df)*len(IMPUTE_COLS)):.3f}% of the imputable matrix)")


# ═══════════════════════════════════════════════════════════
# STEP 5 — TEMPORAL VALIDATION
# ═══════════════════════════════════════════════════════════
log("\n[5/13] Temporal validation ...")

expected_days = (df["date"].max() - df["date"].min()).days + 1
for point_id, g in df.groupby("point_id"):
    for event in EVENT_ORDER:
        n = (g["event"] == event).sum()
        if n < expected_days * 0.99:
            log(f"  [WARN] {point_id} / {event}: {n} rows vs {expected_days} expected days "
                f"({100*n/expected_days:.1f}% coverage)")

dup_check = df.duplicated(subset=["point_id", "date", "event"]).sum()
log(f"  Duplicate (point_id, date, event) after all cleaning: {dup_check}")
log(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()} "
    f"({expected_days} days)")


# ═══════════════════════════════════════════════════════════
# STEP 6 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
log("\n[6/13] Feature engineering ...")

if "era5_W_dir" in df.columns and "era5_W_spd" in df.columns:
    df["era5_W_dir_sin"] = np.sin(np.deg2rad(df["era5_W_dir"])) * df["era5_W_spd"]
    df["era5_W_dir_cos"] = np.cos(np.deg2rad(df["era5_W_dir"])) * df["era5_W_spd"]
if "era5_CSI" in df.columns:
    df["era5_cloud_opacity"] = 1.0 - df["era5_CSI"].clip(0, 1)
if "era5_T_amb" in df.columns and "era5_T_dew" in df.columns:
    df["era5_T_depression"] = df["era5_T_amb"] - df["era5_T_dew"]
if "era5_SZA" in df.columns:
    df["is_daytime"] = (df["era5_SZA"] < 90).astype(int)

# Local (IST) decimal hour of each event, used instead of a fixed "hour"
# column (which doesn't exist here — sun-event hour varies per point/date).
df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
ist = df["time_utc"] + pd.Timedelta(hours=5, minutes=30)
df["ist_hour_decimal"] = ist.dt.hour + ist.dt.minute / 60 + ist.dt.second / 3600
df["solar_hour_angle"] = (df["ist_hour_decimal"] - 12) * 15

engineered = [c for c in ["era5_W_dir_sin", "era5_W_dir_cos", "era5_cloud_opacity",
                           "era5_T_depression", "is_daytime", "ist_hour_decimal",
                           "solar_hour_angle"] if c in df.columns]
log(f"  Added: {engineered}")


# ═══════════════════════════════════════════════════════════
# STEP 7 — LAG FEATURES  (in event-occurrences: 1d, 7d, 30d prior — NOT hours)
# ═══════════════════════════════════════════════════════════
log(f"\n[7/13] Lag features: occurrences={LAG_OCCURRENCES} on {LAG_COLS} "
    f"(within point_id x event groups, i.e. 1/7/30 DAYS prior at the SAME sun event) ...")

df = df.sort_values(["point_id", "event", "date"]).reset_index(drop=True)
for col in LAG_COLS:
    if col not in df.columns:
        continue
    for lag in LAG_OCCURRENCES:
        df[f"{col}_lag{lag}d"] = df.groupby(["point_id", "event"], observed=True)[col].shift(lag)


# ═══════════════════════════════════════════════════════════
# STEP 8 — ROLLING STATS  (trailing 7/30 occurrences)
# ═══════════════════════════════════════════════════════════
log(f"[8/13] Rolling stats: windows={ROLL_OCCURRENCES} occurrences on {LAG_COLS} ...")

for col in LAG_COLS:
    if col not in df.columns:
        continue
    for win in ROLL_OCCURRENCES:
        grp = df.groupby(["point_id", "event"], observed=True)[col]
        df[f"{col}_roll{win}d_mean"] = grp.transform(
            lambda x, w=win: x.rolling(w, min_periods=3).mean())
        df[f"{col}_roll{win}d_std"] = grp.transform(
            lambda x, w=win: x.rolling(w, min_periods=3).std().fillna(0))


# ═══════════════════════════════════════════════════════════
# STEP 9 — DELTA FEATURES  (1-occurrence rate of change)
# ═══════════════════════════════════════════════════════════
log("[9/13] Delta features (1-day-prior rate of change, same event) ...")

for col in ["era5_T_amb", "era5_GHI", "era5_cloud_cover"]:
    if col not in df.columns:
        continue
    df[f"{col}_delta1d"] = df.groupby(["point_id", "event"], observed=True)[col].diff(1).fillna(0)


# ═══════════════════════════════════════════════════════════
# STEP 9c — DROP LAG-WARMUP ROWS
# ═══════════════════════════════════════════════════════════
# Steps 7-8 shift/roll within each (point_id, event) group's chronological
# occurrences — the first max(LAG_OCCURRENCES)=30 occurrences of every
# group have no prior history to draw from, so those lag/rolling columns
# come out NaN there by construction (not a data quality problem — a
# structural warmup period, same concept as the old hourly pipeline's
# lag-warmup drop). Drop those rows now, before imputation/scaling see
# them, rather than let step 4's imputation quietly paper over what is
# actually "this occurrence is too early in this point's series to have
# a 30-days-prior lag," which is a different thing from a real gap.
rows_before_warmup_drop = len(df)
deepest_lag_col = f"era5_GHI_lag{max(LAG_OCCURRENCES)}d"
if deepest_lag_col in df.columns:
    df = df.dropna(subset=[deepest_lag_col]).reset_index(drop=True)
rows_after_warmup_drop = len(df)
log(f"\n  [9c/13] Dropped {rows_before_warmup_drop - rows_after_warmup_drop:,} lag-warmup rows "
    f"(first {max(LAG_OCCURRENCES)} occurrences per point_id x event group). "
    f"Remaining: {rows_after_warmup_drop:,}")


# ═══════════════════════════════════════════════════════════
# STEP 9b — SAVITZKY-GOLAY DIAGNOSTIC  (visual QA only, one sample point/year)
# ═══════════════════════════════════════════════════════════
log("\n[9b/13] Savitzky-Golay smoothing diagnostic (visual QA only) ...")

sample_point = df["point_id"].iloc[0]
sample_year = int(df["year"].median())
sub = df[(df["point_id"] == sample_point) & (df["event"] == "noon") &
         (df["year"] == sample_year)].sort_values("date")

if len(sub) > 31:
    window = min(31, len(sub) - (1 - len(sub) % 2))
    if window % 2 == 0:
        window -= 1
    smoothed = savgol_filter(sub["era5_GHI"].values, window_length=window, polyorder=3)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sub["date"], sub["era5_GHI"], color="#888", alpha=0.5, linewidth=0.8, label="Raw noon GHI")
    ax.plot(sub["date"], smoothed, color="#f3722c", linewidth=1.8, label=f"Savitzky-Golay (window={window})")
    ax.set_title(f"Savitzky-Golay Diagnostic — {sample_point}, noon GHI, {sample_year}\n"
                 f"(peak shape preserved, unlike a moving average — QA only, data untouched)")
    ax.set_ylabel("GHI (W/m²)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PREPROCESSED_DIR / "savitzky_golay_diagnostic.png", dpi=130, bbox_inches="tight")
    plt.close()
    log(f"  Saved: savitzky_golay_diagnostic.png  ({sample_point}, {sample_year})")
else:
    log("  [SKIP] Not enough same-year noon rows for the sample point.")


# ═══════════════════════════════════════════════════════════
# STEP 10 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════
log("\n[10/13] Correlation analysis (Pearson + Spearman) ...")

CORR_COLS = [c for c in ["era5_GHI", "era5_DNI", "era5_DHI", "era5_GHI_clearsky",
                          "era5_CSI", "era5_T_amb", "era5_T_dew", "era5_RHum",
                          "era5_W_spd", "era5_cloud_cover", "era5_precipitation",
                          "era5_P_atm", "era5_SZA", "era5_cloud_opacity",
                          "era5_T_depression"] if c in df.columns]

day_sub = df[df["is_daytime"] == 1] if "is_daytime" in df.columns else df
corr_sample = day_sub[CORR_COLS].sample(min(50_000, len(day_sub)), random_state=42)

pearson_corr = corr_sample.corr(method="pearson")
spearman_corr = corr_sample.corr(method="spearman")
pearson_corr.to_csv(PREPROCESSED_DIR / "correlation_pearson.csv")
spearman_corr.to_csv(PREPROCESSED_DIR / "correlation_spearman.csv")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
for ax, corr, title in zip(axes, [pearson_corr, spearman_corr], ["Pearson", "Spearman"]):
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, annot_kws={"size": 7})
    ax.set_title(f"{title} Correlation — daytime rows")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PREPROCESSED_DIR / "correlation_heatmaps.png", dpi=130, bbox_inches="tight")
plt.close()
log("  Saved: correlation_pearson.csv, correlation_spearman.csv, correlation_heatmaps.png")


# ═══════════════════════════════════════════════════════════
# STEP 11 — VIF
# ═══════════════════════════════════════════════════════════
log("\n[11/13] Variance Inflation Factor ...")

vif_sample = day_sub[CORR_COLS].dropna().sample(min(50_000, len(day_sub)), random_state=42)
vif_sample = vif_sample.loc[:, vif_sample.std() > 1e-8]  # drop any constant column (VIF undefined)
vif_rows = []
for i, col in enumerate(vif_sample.columns):
    try:
        vif_val = variance_inflation_factor(vif_sample.values, i)
    except Exception:
        vif_val = np.nan
    vif_rows.append({"feature": col, "VIF": round(float(vif_val), 2) if vif_val == vif_val else None})
vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif_df.to_csv(PREPROCESSED_DIR / "vif_report.csv", index=False)
log(vif_df.to_string(index=False))
log("  Saved: vif_report.csv  (VIF > 10 = high multicollinearity — reported, nothing dropped)")


# ═══════════════════════════════════════════════════════════
# SAVE PHYSICAL-UNITS OUTPUT  (Phase 3 / 04b reads this one)
# ═══════════════════════════════════════════════════════════
physical_path = PREPROCESSED_DIR / "rajasthan_cleaned_physical.csv"
df_sorted = df.sort_values(["date", "point_id", "event"]).reset_index(drop=True)
df_sorted.to_csv(physical_path, index=False)
log(f"\n  Saved (physical units, for Phase 3): {physical_path}")


# ═══════════════════════════════════════════════════════════
# STEP 12 — SCALING  (separate artifact — Phase 3 must NOT use this one)
# ═══════════════════════════════════════════════════════════
log("\n[12/13] MinMax scaling (train-only fit, leakage-safe) ...")

SKIP_SCALE = {"point_id", "lat", "lon", "population", "weight", "date", "event",
              "time_utc", "grid_lat", "grid_lon", "month", "DOY", "year",
              "season", "season_code", "impute_zone", "is_daytime"}
scale_cols = [c for c in df_sorted.select_dtypes(include=[np.number]).columns
              if c not in SKIP_SCALE]

df_scaled = df_sorted.copy()
n = len(df_scaled)
train_end = int(n * TRAIN_FRAC)   # rows are globally date-sorted, so this is a true chronological cut

scalers = {}
for col in scale_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_scaled[col].values[:train_end].reshape(-1, 1))
    df_scaled[col] = scaler.transform(df_scaled[col].values.reshape(-1, 1)).flatten()
    scalers[col] = scaler

scalers_path = PREPROCESSED_DIR / "scalers.pkl"
with open(scalers_path, "wb") as f:
    pickle.dump(scalers, f)

scaled_path = PREPROCESSED_DIR / "rajasthan_cleaned_scaled.csv"
df_scaled.to_csv(scaled_path, index=False)
log(f"  Scaled {len(scale_cols)} columns, fit on first {TRAIN_FRAC*100:.0f}% of rows "
    f"({train_end:,}/{n:,})")
log(f"  Saved: {scaled_path}")
log(f"  Saved: {scalers_path}")


# ═══════════════════════════════════════════════════════════
# STEP 13 — FINAL VALIDATION  (hard gate)
# ═══════════════════════════════════════════════════════════
log("\n[13/13] Final validation ...")

checks = []
def check(name, ok, detail=""):
    checks.append((name, ok))
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

n_nan_physical = int(df_sorted[IMPUTE_COLS].isna().sum().sum())
check("Physical file: zero NaN in imputable columns", n_nan_physical == 0,
      f"{n_nan_physical} NaN cells remain")

n_inf = int(np.isinf(df_sorted[IMPUTE_COLS].select_dtypes(include=[np.number])).sum().sum())
check("Physical file: zero Inf", n_inf == 0, f"{n_inf} Inf cells")

# NOTE: the scaler is intentionally fit on the TRAIN portion only (leakage
# prevention). Rows after train_end can legitimately fall outside [0,1] if
# the val/test period contains a value more extreme than anything seen in
# training (e.g. a record hot day in 2024 that wasn't in the 2016-2022
# training window) — that's expected, not a bug. The hard gate below only
# checks the TRAIN portion, which is scaled by construction; the full-file
# out-of-range fraction is reported as an informational stat, not a failure.
train_scaled = df_scaled[scale_cols].values[:train_end]
train_range_ok = bool((train_scaled.min() >= -1e-6) and (train_scaled.max() <= 1 + 1e-6))
full_out_of_range_frac = float(
    ((df_scaled[scale_cols] < -1e-6) | (df_scaled[scale_cols] > 1 + 1e-6)).values.mean())
check("Scaled file: TRAIN portion within [0,1]", train_range_ok)
log(f"  Informational: {100*full_out_of_range_frac:.3f}% of scaled cells across the FULL "
    f"file (train+val+test) fall outside [0,1] — expected when val/test contains a more "
    f"extreme value than training ever saw; not a failure by itself.")

n_dup_final = int(df_sorted.duplicated(subset=["point_id", "date", "event"]).sum())
check("Zero duplicate (point_id, date, event) rows", n_dup_final == 0, f"{n_dup_final} duplicates")

required_cols = ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
                  "era5_CSI", "era5_precipitation", "era5_P_atm", "era5_W_spd"]
missing_required = [c for c in required_cols if c not in df_sorted.columns]
check("All required columns present", len(missing_required) == 0,
      f"missing: {missing_required}" if missing_required else "")

n_pass = sum(1 for _, ok in checks if ok)
log(f"\n  RESULT: {n_pass}/{len(checks)} checks passed")

report_path = PREPROCESSED_DIR / "qc_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n" + "=" * 68)
print("  PHASE 2 COMPLETE" if n_pass == len(checks) else "  PHASE 2 FINISHED WITH FAILED CHECKS — inspect qc_report.txt")
print(f"  Physical (Phase 3 input) : {physical_path}")
print(f"  Scaled (ML/DRL use)      : {scaled_path}")
print(f"  QC report                : {report_path}")
print("=" * 68)
print("\nNext: python 04b_climate_signature_rajasthan.py")
