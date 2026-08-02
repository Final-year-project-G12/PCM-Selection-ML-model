"""
SUN-EVENT TIME TABLE — RAJASTHAN
=============================================================================
For every point in population_grid_points.csv and every date 2016-01-01
through 2025-12-31, computes the exact UTC sunrise, solar noon (transit),
and sunset using pvlib's SPA algorithm — no manual equation-of-time code.

Output: data/processed/suntimes.csv
  point_id, date, event (sunrise|noon|sunset), time_utc

This table drives everything downstream: 01_download_era5_rajasthan.py uses
it to compute which UTC hour windows to request, and 02_combine_rajasthan.py
uses it to pick the nearest ERA5/NASA-POWER hourly reading for each event.

IMPORTANT — cross-midnight UTC dates are real, not a hypothetical edge case:
Rajasthan's sunrise can fall before 00:00 UTC (i.e. on the *previous* UTC
calendar date) for eastern points in summer — e.g. Dholpur's 2020-06-21
sunrise lands at 2020-06-20 23:55:54 UTC. `time_utc` always reflects the
true instant of the event; `date` is the nominal (pvlib-assigned) calendar
date the event belongs to, which may differ from time_utc's own UTC date
for events near midnight.

HOW TO RUN:
  python 00b_build_suntimes.py [--force]

Safe to re-run — skipped entirely if suntimes.csv already covers every
point_id currently in population_grid_points.csv (pass --force to rebuild).
"""

import argparse

import pandas as pd
import pvlib

from config import POPULATION_GRID_FILE, SUNTIMES_FILE, ensure_data_dirs

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

ensure_data_dirs()


def already_done(points_df):
    if not SUNTIMES_FILE.exists():
        return False
    try:
        existing = pd.read_csv(SUNTIMES_FILE, usecols=["point_id"])
    except Exception:
        return False
    have = set(existing["point_id"].unique())
    need = set(points_df["point_id"])
    return need.issubset(have)


def build_suntimes(points_df, dates):
    frames = []
    n = len(points_df)
    for i, row in enumerate(points_df.itertuples(index=False), start=1):
        loc = pvlib.location.Location(
            latitude=row.lat, longitude=row.lon, altitude=0, tz="UTC")
        result = loc.get_sun_rise_set_transit(dates, method="spa")

        long_df = pd.DataFrame({
            "point_id": row.point_id,
            "date": list(dates.date) * 3,
            "event": ["sunrise"] * len(dates) + ["noon"] * len(dates) + ["sunset"] * len(dates),
            "time_utc": list(result["sunrise"].values) + list(result["transit"].values) + list(result["sunset"].values),
        })
        frames.append(long_df)

        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] {row.point_id}  lat={row.lat:.3f}  lon={row.lon:.3f}")

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Recompute even if suntimes.csv already covers all points")
    args = parser.parse_args()

    print("=" * 68)
    print("  Sun-Event Time Table — Rajasthan")
    print("=" * 68)

    points_df = pd.read_csv(POPULATION_GRID_FILE)
    print(f"  Points : {len(points_df)}  (from {POPULATION_GRID_FILE})")
    print(f"  Dates  : {START_DATE} .. {END_DATE}")

    if not args.force and already_done(points_df):
        print(f"\n  [SKIP] {SUNTIMES_FILE} already covers all current points.")
        print("  Pass --force to rebuild anyway.")
        return

    dates = pd.date_range(START_DATE, END_DATE, freq="D", tz="UTC")
    print(f"  {len(dates)} dates x {len(points_df)} points x 3 events "
          f"= {len(dates) * len(points_df) * 3:,} rows\n")

    suntimes = build_suntimes(points_df, dates)
    suntimes.to_csv(SUNTIMES_FILE, index=False)

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Rows   : {len(suntimes):,}")
    print(f"  Output : {SUNTIMES_FILE}")
    print("=" * 68)
    print("\nNext step: run  01_download_era5_rajasthan.py  and  01b_download_nasapower.py")


if __name__ == "__main__":
    main()
