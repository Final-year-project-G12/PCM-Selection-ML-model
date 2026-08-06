"""
SUN-EVENT TIME TABLE — TAMIL NADU
=============================================================================
For every point in population_grid_points.csv and every date 2016-01-01
through 2025-12-31, computes the exact UTC sunrise, solar noon (transit),
and sunset using pvlib's SPA algorithm — the same "3 times a day" sampling
scheme as the Rajasthan pipeline, so both regions' timestamps are directly
comparable for the later clustering step.

Output: data/processed/suntimes.csv
  point_id, date, event (sunrise|noon|sunset), time_utc

This table drives everything downstream: 01_download_era5_tamilnadu.py uses
it to compute which UTC hour windows to request, and 02_combine_tamilnadu.py
uses it to pick the nearest ERA5/NASA-POWER hourly reading for each event.

Tamil Nadu sits close enough to 80°E that sun events land mostly within a
single UTC calendar day (unlike Rajasthan's western edge, which can push an
event across midnight UTC) — but the exact-instant `time_utc` / nominal
`date` split is kept identical to the Rajasthan script for consistency and
in case a point falls on an unusual edge case near the boundary.

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
    print("  Sun-Event Time Table — Tamil Nadu")
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
    print("\nNext step: run  01_download_era5_tamilnadu.py  and  01b_download_nasapower.py")


if __name__ == "__main__":
    main()
