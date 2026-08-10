"""
DAILY AGGREGATES — RAJASTHAN POPULATION POINTS  (NASA POWER only)
=============================================================================
climate_rajasthan_points.csv (02_combine_rajasthan.py) only has 3
instantaneous samples/day (sunrise, noon, sunset), which can't produce daily
energy integrals, true diurnal temperature range, or degree-day counts. But
data/raw/nasapower/power_{point_id}_{year}.json already contains the FULL
hourly series for every point/year — the merge step just never uses most of
it. This script reads those cached files directly (no re-download) and
builds true daily aggregates plus a set of per-point Tier 2 indices from
them.

ERA5 is intentionally NOT used here: computing daily integrals from ERA5
would need all 24 hours/day downloaded (a new CDS request), which is out of
scope for this script — see README.

Reads  :
  data/raw/nasapower/power_{point_id}_{year}.json   (2016-2025, per point)
  data/processed/population_grid_points.csv
Writes :
  data/processed/daily_aggregates_rajasthan.csv          — one row per point/day
  data/processed/daily_aggregates_rajasthan_summary.csv  — one row per point (Tier 2 indices)
  data/processed/daily_aggregates_status.csv              — per-point status tracking

Per-point/day table columns:
  GHI_Wh_m2, GHI_clearsky_Wh_m2   — trapezoidal 24h integral of
                                     ALLSKY_SFC_SW_DWN / CLRSKY_SFC_SW_DWN
                                     (both are hourly-average W/m2, so
                                     integrating over hour-of-day gives Wh/m2)
  kt_daily                        — GHI_Wh_m2 / GHI_clearsky_Wh_m2
  T2M_min, T2M_max, T2M_mean      — true daily temperature stats (not
                                     sun-event samples)
  RH2M_mean, WS10M_mean
  n_hours                         — hourly readings available that day (QC)

Per-point Tier 2 summary columns (definitions confirmed with the project
owner where not already pinned down by GHI_daily_kWh/kt_daily/HDD/CDD/DTR's
standard formulas):
  GHI_daily_kWh   — mean daily GHI integral, kWh/m2/day
  SAI             — solar availability index = kt_daily_mean (confirmed
                     definition: same value as kt_daily_mean, kept as a
                     separate named column per the spec)
  kt_daily_mean, kt_daily_std
  cloudy_frac     — fraction of days with kt_daily < CLOUDY_KT_THRESHOLD.
                     No threshold is defined elsewhere in the repo — 0.3 is
                     an assumption, documented here and in the README.
  CCI             — clear-sky correlation index = Pearson correlation
                     between the daily GHI_Wh_m2 series and the daily
                     GHI_clearsky_Wh_m2 series for that point
  HDD18           — heating degree days, base 18C: sum(max(0, 18 - T2M_mean))
  CDD24           — cooling degree days, base 24C: sum(max(0, T2M_mean - 24))
  DTR_true        — mean(T2M_max - T2M_min) across days
  seasonality     — coefficient of variation of monthly-mean GHI_Wh_m2
                     (std across the 12 monthly means / mean of those means)
  monsoon_index    — mean(GHI_Wh_m2, Jun-Sep) / mean(GHI_Wh_m2, full year)

HOW TO RUN:
  python 02b_build_daily_aggregates.py

Resumable: points already marked OK in daily_aggregates_status.csv are
skipped (their rows stay in daily_aggregates_rajasthan.csv untouched); the
summary table is always recomputed from the full on-disk daily table at the
end, so a resumed run still produces a complete, correct summary.
"""

import os
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    RAW_POWER_DIR,
    POPULATION_GRID_FILE,
    DAILY_AGGREGATES_FILE,
    DAILY_AGGREGATES_SUMMARY_FILE,
    DAILY_AGGREGATES_STATUS_FILE,
    ensure_data_dirs,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

YEARS = [str(y) for y in range(2016, 2026)]
POWER_DIR = str(RAW_POWER_DIR)
OUTPUT_FILE = str(DAILY_AGGREGATES_FILE)
SUMMARY_FILE = str(DAILY_AGGREGATES_SUMMARY_FILE)
STATUS_FILE = str(DAILY_AGGREGATES_STATUS_FILE)
ensure_data_dirs()

# Not defined anywhere else in the repo — documented assumption (see
# module docstring / README).
CLOUDY_KT_THRESHOLD = 0.3

MONSOON_MONTHS = [6, 7, 8, 9]


# ═══════════════════════════════════════════════════════════
# STATUS TRACKER  (same pattern as 01b_download_nasapower.py)
# ═══════════════════════════════════════════════════════════

class StatusTracker:
    FIELDS = ["timestamp", "point_id", "n_days", "status", "note"]

    def __init__(self, filepath):
        self.filepath = filepath
        self.records = []
        self._done_set = set()
        if os.path.exists(filepath):
            with open(filepath, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.records.append(row)
                    if row["status"] == "OK":
                        self._done_set.add(row["point_id"])

    def is_done(self, point_id):
        return point_id in self._done_set

    def log(self, point_id, n_days, status, note=""):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "point_id": point_id, "n_days": n_days,
            "status": status, "note": str(note)[:300],
        }
        self.records.append(row)
        if status == "OK":
            self._done_set.add(point_id)
        self._flush()

    def _flush(self):
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.records)

    def summary(self):
        ok = sum(1 for r in self.records if r["status"] == "OK")
        fail = sum(1 for r in self.records if r["status"] == "FAIL")
        return f"OK={ok}  FAIL={fail}  Total={len(self.records)}"


# ═══════════════════════════════════════════════════════════
# LOAD RAW NASA POWER HOURLY CACHE  (read-only — never downloads)
# ═══════════════════════════════════════════════════════════

def load_point_hourly(point_id):
    """Concatenate every cached NASA POWER year for one point into a single
    hourly time-indexed DataFrame. Returns None if no cache files exist."""
    frames = []
    for year in YEARS:
        fp = os.path.join(POWER_DIR, f"power_{point_id}_{year}.json")
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

        cols, idx = {}, None
        for var, series in params.items():
            if idx is None:
                idx = pd.to_datetime(list(series.keys()), format="%Y%m%d%H", utc=True)
            cols[var] = list(series.values())
        frames.append(pd.DataFrame(cols, index=idx))

    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.replace(-999, np.nan)  # NASA POWER's fill value for missing data
    return df


# ═══════════════════════════════════════════════════════════
# PER-DAY AGGREGATION
# ═══════════════════════════════════════════════════════════

def _daily_integral(hour_of_day, values):
    """Trapezoidal integral of an hourly-average W/m2 series over one UTC
    day, in Wh/m2. Needs >=2 valid hourly points; returns NaN otherwise."""
    s = pd.Series(np.asarray(values, dtype=float), index=np.asarray(hour_of_day, dtype=float))
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    s = s.sort_index()
    trapezoid = getattr(np, "trapezoid", np.trapz)  # numpy >=2.0 renamed trapz -> trapezoid
    return float(trapezoid(s.values, s.index.values))


def build_daily_table(point_id, hourly_df):
    df = hourly_df.copy()
    df["date"] = df.index.date
    df["hour"] = df.index.hour + df.index.minute / 60.0

    rows = []
    for date, day_df in df.groupby("date"):
        n_hours = len(day_df)

        ghi_wh = (_daily_integral(day_df["hour"], day_df["ALLSKY_SFC_SW_DWN"])
                  if "ALLSKY_SFC_SW_DWN" in day_df.columns else np.nan)
        csky_wh = (_daily_integral(day_df["hour"], day_df["CLRSKY_SFC_SW_DWN"])
                   if "CLRSKY_SFC_SW_DWN" in day_df.columns else np.nan)
        kt = ghi_wh / csky_wh if csky_wh and csky_wh > 1e-6 else np.nan

        t2m = day_df["T2M"] if "T2M" in day_df.columns else pd.Series(dtype=float)
        t2m_valid = t2m.dropna()

        rows.append({
            "point_id": point_id,
            "date": date,
            "n_hours": n_hours,
            "GHI_Wh_m2": ghi_wh,
            "GHI_clearsky_Wh_m2": csky_wh,
            "kt_daily": kt,
            "T2M_min": t2m_valid.min() if not t2m_valid.empty else np.nan,
            "T2M_max": t2m_valid.max() if not t2m_valid.empty else np.nan,
            "T2M_mean": t2m.mean(),
            "RH2M_mean": day_df["RH2M"].mean() if "RH2M" in day_df.columns else np.nan,
            "WS10M_mean": day_df["WS10M"].mean() if "WS10M" in day_df.columns else np.nan,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# PER-POINT TIER 2 SUMMARY
# ═══════════════════════════════════════════════════════════

def build_summary(daily_df):
    rows = []
    for point_id, g in daily_df.groupby("point_id"):
        g = g.copy()
        g["month"] = pd.to_datetime(g["date"]).dt.month

        ghi = g["GHI_Wh_m2"].dropna()
        kt = g["kt_daily"].dropna()

        GHI_daily_kWh = (ghi / 1000.0).mean() if not ghi.empty else np.nan
        kt_daily_mean = kt.mean() if not kt.empty else np.nan
        kt_daily_std = kt.std() if not kt.empty else np.nan
        SAI = kt_daily_mean  # confirmed definition: SAI == kt_daily_mean
        cloudy_frac = (kt < CLOUDY_KT_THRESHOLD).mean() if not kt.empty else np.nan

        gg = g.dropna(subset=["GHI_Wh_m2", "GHI_clearsky_Wh_m2"])
        CCI = gg["GHI_Wh_m2"].corr(gg["GHI_clearsky_Wh_m2"]) if len(gg) >= 3 else np.nan

        t_mean = g["T2M_mean"].dropna()
        HDD18 = (18.0 - t_mean).clip(lower=0).sum() if not t_mean.empty else np.nan
        CDD24 = (t_mean - 24.0).clip(lower=0).sum() if not t_mean.empty else np.nan

        dtr = (g["T2M_max"] - g["T2M_min"]).dropna()
        DTR_true = dtr.mean() if not dtr.empty else np.nan

        monthly_mean_ghi = g.dropna(subset=["GHI_Wh_m2"]).groupby("month")["GHI_Wh_m2"].mean()
        seasonality = (monthly_mean_ghi.std() / monthly_mean_ghi.mean()
                       if len(monthly_mean_ghi) >= 2 and monthly_mean_ghi.mean() else np.nan)

        annual_mean = ghi.mean() if not ghi.empty else np.nan
        monsoon_mean = g.loc[g["month"].isin(MONSOON_MONTHS), "GHI_Wh_m2"].dropna().mean()
        monsoon_index = (monsoon_mean / annual_mean
                          if annual_mean and not np.isnan(annual_mean) else np.nan)

        rows.append({
            "point_id": point_id,
            "n_days": len(g),
            "GHI_daily_kWh": GHI_daily_kWh,
            "SAI": SAI,
            "kt_daily_mean": kt_daily_mean,
            "kt_daily_std": kt_daily_std,
            "cloudy_frac": cloudy_frac,
            "CCI": CCI,
            "HDD18": HDD18,
            "CDD24": CDD24,
            "DTR_true": DTR_true,
            "seasonality": seasonality,
            "monsoon_index": monsoon_index,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 68)
    print("  Daily Aggregates — Rajasthan Population Points  (NASA POWER)")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input   : {POWER_DIR}/")
    print(f"  Output  : {OUTPUT_FILE}")
    print(f"          : {SUMMARY_FILE}")
    print(f"  Status  : {STATUS_FILE}")
    print("═" * 68)

    if not POPULATION_GRID_FILE.exists():
        print(f"\n  ERROR: {POPULATION_GRID_FILE} not found — "
              "run 00a_build_population_grid.py first.")
        raise SystemExit(1)

    points_df = pd.read_csv(POPULATION_GRID_FILE)
    tracker = StatusTracker(STATUS_FILE)

    already_done = sum(1 for pid in points_df["point_id"] if tracker.is_done(pid))
    fresh_run = already_done == 0 or not os.path.exists(OUTPUT_FILE)
    mode = "w" if fresh_run else "a"
    header_written = not fresh_run

    print(f"\n  Points  : {len(points_df)}  ({already_done} already OK, resuming)")

    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as fout:
        for i, point_row in enumerate(points_df.itertuples(index=False), start=1):
            point_id = point_row.point_id

            if tracker.is_done(point_id):
                if i % 25 == 0 or i == len(points_df):
                    print(f"  [{i}/{len(points_df)}] {point_id}  [SKIP] already OK")
                continue

            hourly = load_point_hourly(point_id)
            if hourly is None:
                print(f"  [{i}/{len(points_df)}] {point_id}  [FAIL] no cached NASA POWER data")
                tracker.log(point_id, 0, "FAIL", "no cached NASA POWER JSON files found")
                continue

            daily = build_daily_table(point_id, hourly)
            daily.to_csv(fout, index=False, header=not header_written)
            fout.flush()
            header_written = True

            tracker.log(point_id, len(daily), "OK")
            if i % 25 == 0 or i == len(points_df):
                print(f"  [{i}/{len(points_df)}] {point_id}  "
                      f"{len(daily)} days  |  {tracker.summary()}")

    print("\n" + "─" * 68)
    print("  Building per-point Tier 2 summary from the full daily table ...")
    daily_full = pd.read_csv(OUTPUT_FILE)
    summary_df = build_summary(daily_full)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"  Wrote {len(summary_df)} point summaries -> {SUMMARY_FILE}")

    print("\n" + "═" * 68)
    print("  SANITY CHECK — Tier 2 indices (across all points)")
    for col in ["GHI_daily_kWh", "SAI", "kt_daily_mean", "kt_daily_std",
                "cloudy_frac", "CCI", "HDD18", "CDD24", "DTR_true",
                "seasonality", "monsoon_index"]:
        s = summary_df[col].dropna()
        if s.empty:
            print(f"    {col:16s}  no valid values")
            continue
        print(f"    {col:16s}  min={s.min():10.4f}  max={s.max():10.4f}  mean={s.mean():10.4f}")

    flags = []
    if (daily_full["GHI_Wh_m2"].dropna() < 0).any():
        flags.append("negative GHI_Wh_m2 values in daily table")
    if (daily_full["GHI_clearsky_Wh_m2"].dropna() < 0).any():
        flags.append("negative GHI_clearsky_Wh_m2 values in daily table")
    if "kt_daily_mean" in summary_df and (summary_df["kt_daily_mean"].dropna() > 1.2).any():
        flags.append("kt_daily_mean > 1.2 for some point(s) — check clear-sky model / units")
    if "DTR_true" in summary_df and (summary_df["DTR_true"].dropna() > 40).any():
        flags.append("DTR_true > 40C for some point(s) — check T2M outliers")
    if flags:
        print("\n  ⚠️  FLAGGED:")
        for f in flags:
            print(f"    - {f}")
    else:
        print("\n  ✅ no obviously-wrong values detected")
    print("═" * 68)

    print(f"\n  {tracker.summary()}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext: use {os.path.basename(OUTPUT_FILE)} / "
          f"{os.path.basename(SUMMARY_FILE)} alongside climate_rajasthan_points.csv.")


if __name__ == "__main__":
    main()
