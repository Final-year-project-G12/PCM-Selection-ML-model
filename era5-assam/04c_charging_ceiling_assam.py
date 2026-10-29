"""
04c_charging_ceiling_assam.py  - Assam SWH Project
=============================================================================
Derives the per-point charging-feasibility ceiling Tm_target_capped_C using
EXACTLY the calculation in era5-rajasthan/04_climate_signature_rajasthan.py
(correction #5, worst-MONTH basis), applied to Assam's own daily clearness data.

Reference chain (Rajasthan 04, section "6/9 Derived PCM-facing quantities"):
  daily kt_daily (per point, per day, from 02b daily aggregates)
    -> month = date.month
    -> monthly_mean_kt = mean(kt_daily) per (point_id, calendar month), all years pooled
    -> kt_worst_month  = min over the 12 monthly means, per point
    -> kt_ratio        = clip(kt_worst_month / kt_daily_mean, upper=1.0)
    -> tm_cap          = Ta_mean + kt_ratio * (Tm_target_C - Ta_mean)
    -> Tm_target_capped_C = min(Tm_target_C, tm_cap)          [per point]
  05_cluster_* then aggregates the per-point values into cluster_profiles.

Assam differences (data only): Tm_target_C = 56.0 (Assam design basis), and
Ta_mean / kt data are Assam's. Nothing is read from or copied from Rajasthan.

ADDITIVE: reads existing Assam files, writes ONE new file; does not modify
climate_signatures_raw.csv or any upstream result.

INPUTS (all under era5-assam/data/processed/):
  climate_signatures_raw.csv      needs point_id, Ta_mean  (kt_daily_mean optional)
  daily_aggregates*.csv           the per-point/day table (NOT the *summary* file);
                                  needs point_id, date, kt_daily
OUTPUT:
  tm_cap_by_point_assam.csv

Run:  python 04c_charging_ceiling_assam.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROC = BASE_DIR / "data" / "processed"
RAW_SIG_FILE = PROC / "climate_signatures_raw.csv"
OUT_FILE = PROC / "tm_cap_by_point_assam.csv"

TM_TARGET_C = 56.0   # 50 C delivery + 6 K approach (Assam design basis)


def find_daily_file():
    cands = sorted(p for p in PROC.glob("daily_aggregates*.csv") if "summary" not in p.name.lower())
    if len(cands) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one per-point/day file matching {PROC}/daily_aggregates*.csv "
            f"(excluding *summary*), found {[c.name for c in cands]}. Set DAILY_FILE explicitly.")
    return cands[0]


def main():
    daily_file = find_daily_file()
    if not RAW_SIG_FILE.exists():
        raise SystemExit(f"ERROR: {RAW_SIG_FILE} not found.")

    sig = pd.read_csv(RAW_SIG_FILE)
    daily = pd.read_csv(daily_file)
    for df, name, cols in ((sig, RAW_SIG_FILE.name, ["point_id", "Ta_mean"]),
                           (daily, daily_file.name, ["point_id", "date", "kt_daily"])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"ERROR: {name} lacks {missing}. Columns present: {list(df.columns)}")

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month

    monthly = daily.groupby(["point_id", "month"])["kt_daily"].mean()
    kt_worst_month = monthly.groupby("point_id").min().rename("kt_worst_month")
    n_months = monthly.groupby("point_id").size()
    if (n_months < 12).any():
        print(f"  [WARN] {(n_months < 12).sum()} point(s) have <12 calendar months of kt data; "
              f"their kt_worst_month is a min over fewer months.")

    out = sig[["point_id", "Ta_mean"]].copy()
    if "kt_daily_mean" in sig.columns:
        out["kt_daily_mean"] = sig["kt_daily_mean"].values
        kt_src = f"{RAW_SIG_FILE.name}:kt_daily_mean"
    else:
        out = out.merge(daily.groupby("point_id")["kt_daily"].mean().rename("kt_daily_mean"),
                        on="point_id", how="left")
        kt_src = f"mean of {daily_file.name}:kt_daily (no kt_daily_mean in signatures)"
    out = out.merge(kt_worst_month, on="point_id", how="left")

    out["Tm_target_C"] = TM_TARGET_C
    kt_ratio = (out["kt_worst_month"] / out["kt_daily_mean"]).clip(upper=1.0)
    tm_cap = out["Ta_mean"] + kt_ratio * (out["Tm_target_C"] - out["Ta_mean"])
    out["tm_target_capped_flag"] = out["Tm_target_C"] > tm_cap
    out["Tm_target_capped_C"] = np.minimum(out["Tm_target_C"], tm_cap)

    if out["Tm_target_capped_C"].isna().any():
        bad = out.loc[out["Tm_target_capped_C"].isna(), "point_id"].tolist()
        raise SystemExit(f"ERROR: could not compute the ceiling for points {bad[:10]} (missing kt/Ta).")

    out.to_csv(OUT_FILE, index=False)
    print(f"  points: {len(out)} | daily file: {daily_file.name} | kt_daily_mean from {kt_src}")
    print(f"  kt_worst_month   : {out['kt_worst_month'].min():.3f} - {out['kt_worst_month'].max():.3f}")
    print(f"  Tm_target_capped_C: {out['Tm_target_capped_C'].min():.1f} - "
          f"{out['Tm_target_capped_C'].max():.1f} C "
          f"({int(out['tm_target_capped_flag'].sum())}/{len(out)} points below the 56 C target)")
    print(f"  Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()