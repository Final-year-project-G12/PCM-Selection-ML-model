"""
NASA POWER DOWNLOAD — ASSAM POPULATION POINTS  (cross-check source)
=============================================================================
For every point in population_grid_points.csv and every year 2016-2025,
downloads NASA POWER hourly point data for:
  ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M

Endpoint: https://power.larc.nasa.gov/api/temporal/hourly/point
  (public NASA Langley service — no Earthdata login / API key needed)

HOW TO RUN:
  python 01b_download_nasapower.py
"""

import os
import csv
import time
from datetime import datetime

import requests
import pandas as pd

from config import (
    POPULATION_GRID_FILE,
    RAW_POWER_DIR,
    POWER_DOWNLOAD_STATUS_FILE,
    ensure_data_dirs,
)

YEARS = [str(y) for y in range(2016, 2026)]

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
POWER_COMMUNITY = "RE"
POWER_PARAMETERS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M"

OUTPUT_DIR = str(RAW_POWER_DIR)
STATUS_FILE = str(POWER_DOWNLOAD_STATUS_FILE)
ensure_data_dirs()

MAX_RETRIES = 3
RETRY_WAIT = 20
REQUEST_SLEEP = 1.0
REQUEST_TIMEOUT = 60


class StatusTracker:
    FIELDS = ["timestamp", "point_id", "year",
              "status", "filepath", "size_kb", "note"]

    def __init__(self, filepath):
        self.filepath = filepath
        self.records = []
        self._done_set = set()
        if os.path.exists(filepath):
            with open(filepath, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.records.append(row)
                    if row["status"] == "OK":
                        self._done_set.add((row["point_id"], row["year"]))

    def is_done(self, point_id, year):
        return (point_id, year) in self._done_set

    def log(self, point_id, year, status, filepath, size_kb=0.0, note=""):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "point_id": point_id, "year": year,
            "status": status, "filepath": filepath,
            "size_kb": f"{size_kb:.1f}", "note": str(note)[:300],
        }
        self.records.append(row)
        if status == "OK":
            self._done_set.add((point_id, year))
        self._flush()

    def _flush(self):
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.records)

    def summary(self):
        ok = sum(1 for r in self.records if r["status"] == "OK")
        skip = sum(1 for r in self.records if r["status"] == "SKIP")
        fail = sum(1 for r in self.records if r["status"] == "FAIL")
        return f"OK={ok}  SKIP={skip}  FAIL={fail}  Total={len(self.records)}"

    def failed(self):
        return [(r["point_id"], r["year"])
                for r in self.records if r["status"] == "FAIL"]


def download_one(session, point_id, year, lat, lon, filepath, tracker):
    if tracker.is_done(point_id, year):
        print(f"  [SKIP-LOG]  {point_id}  {year}  (already OK in status CSV)")
        return "SKIP"

    if os.path.exists(filepath):
        sz = os.path.getsize(filepath)
        if sz > 1000:
            print(f"  [SKIP-FILE] {point_id}  {year}  ({sz/1e3:.1f} KB)")
            tracker.log(point_id, year, "SKIP", filepath, sz / 1e3, "file existed")
            return "SKIP"
        print(f"  [REMOVE]   tiny/corrupt file ({sz} B) — re-downloading")
        os.remove(filepath)

    params = {
        "start": f"{year}0101",
        "end": f"{year}1231",
        "latitude": lat,
        "longitude": lon,
        "community": POWER_COMMUNITY,
        "parameters": POWER_PARAMETERS,
        "format": "JSON",
        "time-standard": "UTC",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(POWER_BASE, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            body = resp.text
            if len(body) < 1000 or '"properties"' not in body:
                raise RuntimeError("Response body too small or missing 'properties'")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(body)

            sz_kb = os.path.getsize(filepath) / 1e3
            print(f"  [OK]  {point_id}  {year}  ({sz_kb:.1f} KB) [OK]")
            tracker.log(point_id, year, "OK", filepath, sz_kb)
            time.sleep(REQUEST_SLEEP)
            return "OK"

        except Exception as exc:
            msg = str(exc)
            print(f"  [FAIL {attempt}/{MAX_RETRIES}]  {point_id} {year} — {msg[:200]}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                tracker.log(point_id, year, "FAIL", filepath, 0, msg[:200])
                return "FAIL"


def main():
    if not POPULATION_GRID_FILE.exists():
        raise FileNotFoundError(
            f"{POPULATION_GRID_FILE} not found -- run 00a_build_population_grid.py first.")
    points_df = pd.read_csv(POPULATION_GRID_FILE)

    total_calls = len(points_df) * len(YEARS)
    print("\n" + "=" * 68)
    print("  NASA POWER Assam -- Population Points Download  (10 years)")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Points  : {len(points_df)}  (from {POPULATION_GRID_FILE.name})")
    print(f"  Years   : {YEARS[0]}--{YEARS[-1]}  ({len(YEARS)} years)")
    print(f"  Calls   : {total_calls}  ({len(points_df)} points x {len(YEARS)} years)")
    print(f"  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print("=" * 68)

    tracker = StatusTracker(STATUS_FILE)
    session = requests.Session()
    session.headers.update({"User-Agent": "AssamClimatePipeline/1.0"})

    for row in points_df.itertuples(index=False):
        pid = row.point_id
        plat = float(row.lat)
        plon = float(row.lon)
        print(f"\n-- {pid}  lat={plat:.3f} lon={plon:.3f} --")

        for yr in YEARS:
            filepath = os.path.join(OUTPUT_DIR, f"power_{pid}_{yr}.json")
            download_one(session, pid, yr, plat, plon, filepath, tracker)

        print(f"  Progress: {tracker.summary()}")

    print("\n" + "=" * 68)
    print("  NASA POWER DOWNLOAD COMPLETE")
    print(f"  {tracker.summary()}")

    failed = tracker.failed()
    if failed:
        print(f"\n  FAILED ({len(failed)}) -- re-run script to retry:")
        for pid, yr in failed:
            print(f"    {pid}  {yr}")
    else:
        print("  [OK] All files downloaded successfully!")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)


if __name__ == "__main__":
    main()
