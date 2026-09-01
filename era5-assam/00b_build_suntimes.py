"""
SUN-EVENT TIME TABLE — ASSAM
=============================================================================
For every point in population_grid_points.csv and every date 2016-01-01
through 2025-12-31, computes exact UTC sunrise, solar noon (transit),
and sunset using pvlib's SPA algorithm.

Output: data/processed/suntimes.csv
  point_id, date, event (sunrise|noon|sunset), time_utc

HOW TO RUN:
  python 00b_build_suntimes.py [--force]
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
    print("  Sun-Event Time Table — Assam")
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
    print("\nNext step: run  01_download_era5_assam.py  and  01b_download_nasapower.py")


if __name__ == "__main__":
    main()
