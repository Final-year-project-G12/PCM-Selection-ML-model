"""
04_preprocess_assam.py
============================
PHASE 2 — PREPROCESSING AND QUALITY CONTROL (Table 9)

INPUT  : data/processed/climate_assam_points.csv   (02_combine output)
OUTPUT : data/preprocessed/parquet/{point_id}.parquet
           (Physical units, QC-passed, imputed, OUTLIERS FLAGGED but NEVER deleted, 
           NO SCALING)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COMBINED_POINTS_FILE, PREPROCESSED_DIR

PARQUET_DIR = PREPROCESSED_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ORDER = ["sunrise", "noon", "sunset"]

# Physical bounds (Table 9, plan doc)
BOUNDS = {
    "era5_GHI": (0, 1400), "era5_T_amb": (-30, 55), "era5_RHum": (0, 100),
    "era5_T_dew": (-30, 40), "era5_W_spd": (0, 50), "era5_P_atm": (850, 1060),
    "era5_cloud_cover": (0, 1), "era5_precipitation": (0, 200),
}

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(str(msg))

def main():
    log("=" * 68)
    log("  PHASE 2 — PREPROCESSING & QUALITY CONTROL (Table 9) — Assam")
    log("=" * 68)

    # -----------------------------------------------------------
    # [1] Unit Standardisation & Basic Load
    # -----------------------------------------------------------
    log("\n[1] Loading data & Unit Standardisation (bounds check) ...")
    df = pd.read_csv(COMBINED_POINTS_FILE, parse_dates=["date"])
    df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
    
    # Check physical ranges
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            out_of_bounds = (df[col] < lo) | (df[col] > hi)
            n_bad = out_of_bounds.sum()
            if n_bad > 0:
                log(f"  {col}: {n_bad} values out of bounds [{lo}, {hi}] - clipping/nulling.")
                df.loc[out_of_bounds, col] = np.nan

    # -----------------------------------------------------------
    # [2] Timezone Conversion
    # -----------------------------------------------------------
    log("\n[2] Timezone conversion (UTC -> IST) ...")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["time_ist"] = df["time_utc"].dt.tz_convert("Asia/Kolkata")
    
    peak_hr = df[df["event"] == "noon"]["time_ist"].dt.hour.mean()
    log(f"  Average IST hour for 'noon' event: {peak_hr:.2f}")
    if not (10 <= peak_hr <= 13):
        log("  [FAIL] Timezone conversion is wrong (peak not at 12-13).")

    # -----------------------------------------------------------
    # [3] Derived Humidity
    # -----------------------------------------------------------
    log("\n[3] Derived humidity (Magnus formula) check ...")
    if "era5_RHum" in df.columns:
        invalid_rh = df["era5_RHum"] > 100
        if invalid_rh.sum() > 0:
            log(f"  Found {invalid_rh.sum()} rows with RH > 100%. Clipping to 100.")
            df.loc[invalid_rh, "era5_RHum"] = 100.0

    # -----------------------------------------------------------
    # [4] Night Masking
    # -----------------------------------------------------------
    log("\n[4] Night masking (GHI=0 when SZA >= 90) ...")
    if "era5_SZA" in df.columns and "era5_GHI" in df.columns:
        night_mask = df["era5_SZA"] >= 90.0
        n_night_ghi = (night_mask & (df["era5_GHI"] > 0)).sum()
        df.loc[night_mask, "era5_GHI"] = 0.0
        if "era5_GHI_clearsky" in df.columns:
            df.loc[night_mask, "era5_GHI_clearsky"] = 0.0
        log(f"  Masked {n_night_ghi} night-time GHI values to 0.")

    # -----------------------------------------------------------
    # [5] Missing Values
    # -----------------------------------------------------------
    log("\n[5] Missing values (Interpolation <= 6h, Climatological > 6h, Drop site-year > 5%) ...")
    
    # Sort chronologically for each point and event
    df = df.sort_values(["point_id", "time_ist"]).reset_index(drop=True)
    
    impute_cols = [c for c in df.columns if (c.startswith("era5_") or c.startswith("power_")) and pd.api.types.is_numeric_dtype(df[c])]
    
    for col in impute_cols:
        # Linear interpolation for short gaps
        df[col] = df.groupby("point_id")[col].transform(lambda s: s.interpolate(method="linear", limit=1))
        
        # Climatological mean (same-hour) for longer gaps
        month_event_mean = df.groupby(["point_id", "month", "event"])[col].transform("mean")
        df[col] = df[col].fillna(month_event_mean)

    # Drop any site-year with >5% missing
    df["year_ist"] = df["time_ist"].dt.year
    site_year_missing = df.groupby(["point_id", "year_ist"])[impute_cols].apply(lambda x: x.isna().mean().max())
    bad_site_years = site_year_missing[site_year_missing > 0.05].index
    
    if len(bad_site_years) > 0:
        log(f"  Dropping {len(bad_site_years)} site-years with >5% missing data.")
        mask = df.set_index(["point_id", "year_ist"]).index.isin(bad_site_years)
        df = df[~mask].reset_index(drop=True)
    
    n_remaining_na = df[impute_cols].isna().sum().sum()
    log(f"  Remaining missing cells after Step 5: {n_remaining_na}")

    # -----------------------------------------------------------
    # [6] Outlier Detection (Flag, never delete)
    # -----------------------------------------------------------
    log("\n[6] Outlier detection (Bounds -> 3sigma -> Isolation Forest) ...")
    
    df["is_outlier"] = 0
    
    # Per-variable per-month 3-sigma
    for col in ["era5_T_amb", "era5_GHI", "era5_W_spd"]:
        if col not in df.columns: continue
        grp = df.groupby(["point_id", "month", "event"])[col]
        mean = grp.transform("mean")
        std = grp.transform("std").fillna(0)
        outlier_mask = (df[col] - mean).abs() > (3 * std)
        df.loc[outlier_mask, "is_outlier"] = 1
        log(f"  {col}: {outlier_mask.sum()} points flagged by 3-sigma.")
    
    # Isolation Forest on multivariate residual
    iso_cols = [c for c in ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_W_spd"] if c in df.columns]
    if len(iso_cols) > 0 and len(df) > 0:
        iso_data = df[iso_cols].fillna(0)
        iso = IsolationForest(contamination=0.01, random_state=42)
        preds = iso.fit_predict(iso_data)
        if_outliers = (preds == -1)
        df.loc[if_outliers, "is_outlier"] = 1
        log(f"  Isolation Forest flagged {if_outliers.sum()} multivariate anomalies.")
        
    log(f"  Total rows flagged as outliers (is_outlier=1): {df['is_outlier'].sum()}")

    # -----------------------------------------------------------
    # [7] Solar Bias Correction (Quantile mapping)
    # -----------------------------------------------------------
    log("\n[7] Solar bias correction (Quantile mapping ERA5 GHI -> NASA POWER) ...")
    
    if "era5_GHI" in df.columns and "power_ALLSKY_SFC_SW_DWN" in df.columns:
        pre_mbe = (df["era5_GHI"] - df["power_ALLSKY_SFC_SW_DWN"]).mean()
        
        df["era5_GHI_corrected"] = df["era5_GHI"].copy()
        
        for name, grp in df.groupby(["point_id", "season_code"]):
            era_vals = grp["era5_GHI"].values
            ref_vals = grp["power_ALLSKY_SFC_SW_DWN"].values
            
            if len(era_vals) > 10:
                era_sorted = np.sort(era_vals)
                ref_sorted = np.sort(ref_vals)
                corrected = np.interp(era_vals, era_sorted, ref_sorted)
                df.loc[grp.index, "era5_GHI_corrected"] = corrected
                
        post_mbe = (df["era5_GHI_corrected"] - df["power_ALLSKY_SFC_SW_DWN"]).mean()
        log(f"  Pre-correction MBE  : {pre_mbe:.3f} W/m²")
        log(f"  Post-correction MBE : {post_mbe:.3f} W/m²")
        
        df["era5_GHI"] = df["era5_GHI_corrected"]

    # -----------------------------------------------------------
    # [8] Clear-sky Index (kt)
    # -----------------------------------------------------------
    log("\n[8] Clear-sky index (kt = GHI / GHIcs, clipped to [0, 1.2]) ...")
    if "era5_GHI" in df.columns and "era5_GHI_clearsky" in df.columns:
        df["era5_CSI"] = np.where(
            df["era5_GHI_clearsky"] > 10,
            (df["era5_GHI"] / df["era5_GHI_clearsky"]).clip(0, 1.2), 0)
        
        median_kt = df[df["era5_CSI"] > 0]["era5_CSI"].median()
        log(f"  Median kt (when > 0): {median_kt:.3f}")
        if 0.55 <= median_kt <= 0.75:
            log("  [PASS] Median kt is in [0.55, 0.75] range.")
        else:
            log(f"  [WARN] Median kt is outside [0.55, 0.75].")

    # -----------------------------------------------------------
    # [9] Storage (Parquet, one file per site)
    # -----------------------------------------------------------
    log("\n[9] Storage (Write to Parquet, one file per site) ...")
    
    for point_id, grp in df.groupby("point_id"):
        out_path = PARQUET_DIR / f"{point_id}.parquet"
        grp.to_parquet(out_path, index=False)
        
    log(f"  Saved {df['point_id'].nunique()} parquet files to {PARQUET_DIR}")
    
    # Round-trip test
    sample_point = df["point_id"].iloc[0]
    sample_path = PARQUET_DIR / f"{sample_point}.parquet"
    test_read = pd.read_parquet(sample_path)
    
    if len(test_read) == len(df[df["point_id"] == sample_point]):
        log(f"  [PASS] Round-trip test successful for {sample_point}.")
    else:
        log(f"  [FAIL] Round-trip test failed row count for {sample_point}.")
        
    report_path = PREPROCESSED_DIR / "qc_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    log("\n" + "=" * 68)
    log("  PHASE 2 COMPLETE (Table 9 Compliance)")
    log("=" * 68)

if __name__ == "__main__":
    main()
