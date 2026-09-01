"""
02b_build_daily_aggregates.py
================================
PHASE 2 REPAIR 1 (Objective 1 Plan v3.0, Section 4.3) — UTTARAKHAND

climate_uttarakhand_points.csv keeps only 3 rows/day (sunrise, noon, sunset).
Several signature indices genuinely cannot be computed from three
instantaneous samples: the true daily GHI energy integral, the true
diurnal temperature range (Tmax-Tmin, not noon-sunrise), heating/cooling
degree-days from a true daily mean, cloudy-day fraction, and the longest
consecutive-cloudy-day run.

This does NOT need a new download. 01b_download_nasapower.py already
cached the FULL hourly series for every point/year at
  data/raw/nasapower/power_{point_id}_{year}.json
02_combine_uttarakhand.py only ever read 3 of those ~8760 hours per year.
This script reads the same files again, in full, and builds the daily
integrals the merge step discarded. No CDS/NASA queue time is spent.

LIMITATION (say this in your methodology, don't hide it)
-----------------------------------------------------------
01b's POWER_PARAMETERS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M"
does NOT include precipitation (PRECTOTCORR). `monsoon_index` therefore
still comes from the ERA5 3x/day precipitation-fraction proxy in
04b_climate_signature.py — it is NOT recomputed here as a true Tier-2
index. If you want a real one, add PRECTOTCORR to POWER_PARAMETERS in
01b_download_nasapower.py and re-run just that script (a few hours,
no CDS involved) — optional, not required for Objective 1 to stand up.

INPUT  : data/raw/nasapower/power_{point_id}_{year}.json  (already on disk)
         data/processed/population_grid_points.csv
OUTPUT : data/processed/daily_aggregates_uttarakhand.csv
           one row per (point_id, date): true daily GHI/clearsky integrals,
           true Tmax/Tmin/DTR, true daily-mean T/RH/wind
         data/processed/tier2_signature_uttarakhand.csv
           one row per point_id — the point-level Tier-2 indices, ready to
           be merged onto the Tier-1 sun-event signature in
           04b_climate_signature.py

A day is only used if >=20 of its 24 hours were actually returned by NASA
POWER (occasional gaps exist in the raw service) — days with heavier
gaps are dropped rather than silently averaged over fewer hours.

HOW TO RUN:
  python 02b_build_daily_aggregates.py
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import RAW_POWER_DIR, POPULATION_GRID_FILE, PROCESSED_DIR, ensure_data_dirs

ensure_data_dirs()

YEARS = [str(y) for y in range(2016, 2026)]
KT_CLOUDY_THRESHOLD = 0.35          # same threshold 04b uses for CCI/cloudy_frac
MIN_HOURS_PER_DAY = 20              # else the day is dropped, not averaged short

OUT_DAILY = PROCESSED_DIR / "daily_aggregates_uttarakhand.csv"
OUT_TIER2 = PROCESSED_DIR / "tier2_signature_uttarakhand.csv"


def load_power_hourly(point_id):
    """Concatenate every cached NASA POWER year for one point — full
    hourly series, not the sun-event subset 02_combine used."""
    frames = []
    for year in YEARS:
        fp = os.path.join(str(RAW_POWER_DIR), f"power_{point_id}_{year}.json")
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        params = data.get("properties", {}).get("parameter", {})
        if not params:
            continue
        idx = None
        cols = {}
        for var, series in params.items():
            if idx is None:
                idx = pd.to_datetime(list(series.keys()), format="%Y%m%d%H", utc=True)
            cols[var] = list(series.values())
        frames.append(pd.DataFrame(cols, index=idx))
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.replace(-999, np.nan)
    return df


def daily_from_hourly(hourly):
    hourly = hourly.copy()
    hourly["date"] = hourly.index.date
    counts = hourly.groupby("date").size()
    g = hourly.groupby("date")

    out = pd.DataFrame(index=counts.index)
    out["n_hours"] = counts

    if "ALLSKY_SFC_SW_DWN" in hourly.columns:
        out["GHI_daily_kWh"] = g["ALLSKY_SFC_SW_DWN"].sum() / 1000.0   # W/m^2 * 1h -> Wh -> kWh
    if "CLRSKY_SFC_SW_DWN" in hourly.columns:
        out["GHIcs_daily_kWh"] = g["CLRSKY_SFC_SW_DWN"].sum() / 1000.0
    if {"GHI_daily_kWh", "GHIcs_daily_kWh"}.issubset(out.columns):
        out["kt_daily"] = np.where(out["GHIcs_daily_kWh"] > 0.05,
                                    (out["GHI_daily_kWh"] / out["GHIcs_daily_kWh"]).clip(0, 1.5),
                                    np.nan)
    if "T2M" in hourly.columns:
        out["Ta_mean_true"] = g["T2M"].mean()
        out["Ta_max_true"] = g["T2M"].max()
        out["Ta_min_true"] = g["T2M"].min()
        out["DTR_true"] = out["Ta_max_true"] - out["Ta_min_true"]
    if "RH2M" in hourly.columns:
        out["RH_mean_true"] = g["RH2M"].mean()
    if "WS10M" in hourly.columns:
        out["wind_mean_true"] = g["WS10M"].mean()

    out = out[out["n_hours"] >= MIN_HOURS_PER_DAY].drop(columns=["n_hours"])
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out.reset_index()


def build_tier2_row(point_id, daily_df):
    row = {"point_id": point_id, "n_days_used": len(daily_df)}

    row["GHI_daily_kWh_mean"] = daily_df["GHI_daily_kWh"].mean()
    row["kt_daily_mean"] = daily_df["kt_daily"].mean()
    row["kt_daily_std"] = daily_df["kt_daily"].std()

    ghi_sum = daily_df["GHI_daily_kWh"].sum()
    ghics_sum = daily_df["GHIcs_daily_kWh"].sum()
    row["SAI_true"] = ghi_sum / ghics_sum if ghics_sum > 0 else np.nan

    is_cloudy = daily_df["kt_daily"] < KT_CLOUDY_THRESHOLD
    row["cloudy_frac_true"] = is_cloudy.mean()
    run_id = (is_cloudy != is_cloudy.shift()).cumsum()
    run_lengths = is_cloudy.astype(int).groupby(run_id).transform("sum")
    row["CCI_true"] = int((run_lengths * is_cloudy.astype(int)).max()) if len(run_lengths) else 0

    row["DTR_true_mean"] = daily_df["DTR_true"].mean()
    row["Ta_mean_true"] = daily_df["Ta_mean_true"].mean()
    row["Ta_p95_true"] = daily_df["Ta_mean_true"].quantile(0.95)
    row["Ta_p05_true"] = daily_df["Ta_mean_true"].quantile(0.05)
    row["HDD18_true"] = np.maximum(0, 18 - daily_df["Ta_mean_true"]).sum()
    row["CDD24_true"] = np.maximum(0, daily_df["Ta_mean_true"] - 24).sum()
    row["RH_mean_true"] = daily_df["RH_mean_true"].mean()
    row["wind_mean_true"] = daily_df["wind_mean_true"].mean()

    dt = pd.to_datetime(daily_df["date"])
    monthly_ghi = daily_df.groupby(dt.dt.month)["GHI_daily_kWh"].mean()
    row["seasonality_true"] = (monthly_ghi.std() / monthly_ghi.mean()
                                if monthly_ghi.mean() > 0 else np.nan)
    return row


def main():
    print("=" * 68)
    print("  PHASE 2 REPAIR 1 — Daily-Integral (Tier 2) Indices — Uttarakhand")
    print("=" * 68)

    points_df = pd.read_csv(POPULATION_GRID_FILE)
    print(f"  Points: {len(points_df)}")

    daily_frames, tier2_rows = [], []
    n_skipped = 0
    for i, r in enumerate(points_df.itertuples(index=False), start=1):
        hourly = load_power_hourly(r.point_id)
        if hourly is None or hourly.empty:
            print(f"  [{i}/{len(points_df)}] {r.point_id}  [SKIP] no POWER cache on disk")
            n_skipped += 1
            continue
        daily = daily_from_hourly(hourly)
        if daily.empty:
            print(f"  [{i}/{len(points_df)}] {r.point_id}  [SKIP] no day met "
                  f">= {MIN_HOURS_PER_DAY}h coverage")
            n_skipped += 1
            continue
        daily.insert(0, "point_id", r.point_id)
        daily_frames.append(daily)
        tier2_rows.append(build_tier2_row(r.point_id, daily))
        if i % 20 == 0 or i == len(points_df):
            print(f"  [{i}/{len(points_df)}] {r.point_id}  usable_days={len(daily)}")

    if not daily_frames:
        print("\n  ERROR: no usable NASA POWER data found — run 01b_download_nasapower.py first.")
        return

    daily_all = pd.concat(daily_frames, ignore_index=True)
    daily_all.to_csv(OUT_DAILY, index=False)

    tier2_df = pd.DataFrame(tier2_rows)
    tier2_df.to_csv(OUT_TIER2, index=False)

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Points processed : {len(tier2_rows)}/{len(points_df)}  ({n_skipped} skipped)")
    print(f"  Point-days       : {len(daily_all):,}")
    print(f"  Saved: {OUT_DAILY}")
    print(f"  Saved: {OUT_TIER2}")
    print("=" * 68)
    print("\nNext: run 04_preprocess_uttarakhand.py (Phase 2) if not already done, "
          "then the UPDATED 04b_climate_signature.py, which merges this Tier-2 "
          "table onto the Tier-1 sun-event signature.")


if __name__ == "__main__":
    main()
