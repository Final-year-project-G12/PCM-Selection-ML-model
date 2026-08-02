"""
NASA POWER DOWNLOAD — RAJASTHAN POPULATION POINTS  (cross-check source)
=============================================================================
For every point in population_grid_points.csv and every year 2016-2025,
downloads NASA POWER hourly point data for:
  ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M, WS10M

This is an independent source from ERA5 — 02_combine_rajasthan.py merges
both so ERA5 and NASA POWER values can be cross-checked against each other
for the same point/date/event.

Endpoint: https://power.larc.nasa.gov/api/temporal/hourly/point
  (public NASA Langley service — no Earthdata login / API key needed)

Caching: one raw JSON response per point/year, written to
  data/raw/nasapower/power_{point_id}_{year}.json
tracked in a CSV status file with the same skip-if-done / retry-on-fail
pattern as 01_download_era5_rajasthan.py's StatusTracker.

HOW TO RUN:
  python 01b_download_nasapower.py

Re-running is safe — completed point/year combos are skipped automatically.
"""

import os
import csv
import time
from datetime import datetime

import requests

from config import (
    POPULATION_GRID_FILE,
    RAW_POWER_DIR,
    POWER_DOWNLOAD_STATUS_FILE,
    ensure_data_dirs,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

YEARS = [str(y) for y in range(2016, 2026)]

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
POWER_COMMUNITY = "RE"
POWER_PARAMETERS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M"

OUTPUT_DIR = str(RAW_POWER_DIR)
STATUS_FILE = str(POWER_DOWNLOAD_STATUS_FILE)
ensure_data_dirs()

MAX_RETRIES = 3
RETRY_WAIT = 20          # seconds between retries on failure
REQUEST_SLEEP = 1.0      # seconds between successful calls (avoid rate-limit)
REQUEST_TIMEOUT = 60


# ═══════════════════════════════════════════════════════════
# STATUS TRACKER  (same pattern as 01_download_era5_rajasthan.py)
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════

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
            data = resp.json()

            params_out = data.get("properties", {}).get("parameter", {})
            if not params_out or not any(params_out.values()):
                raise RuntimeError("Response has no parameter data")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(resp.text)

            sz = os.path.getsize(filepath)
            if sz < 1000:
                raise RuntimeError(f"File too small ({sz} bytes) — corrupt response")

            size_kb = sz / 1e3
            print(f"  [OK]  {point_id}  {year}  {size_kb:.1f} KB ✓")
            tracker.log(point_id, year, "OK", filepath, size_kb)
            time.sleep(REQUEST_SLEEP)
            return "OK"

        except Exception as exc:
            msg = str(exc)
            print(f"  [FAIL {attempt}/{MAX_RETRIES}]  {point_id}  {year}  {msg[:300]}")
            if os.path.exists(filepath):
                os.remove(filepath)
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_WAIT}s ...")
                time.sleep(RETRY_WAIT)
            else:
                tracker.log(point_id, year, "FAIL", filepath, 0, msg[:300])
                return "FAIL"


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    import pandas as pd
    points_df = pd.read_csv(POPULATION_GRID_FILE)

    print("\n" + "═" * 68)
    print("  NASA POWER Download — Rajasthan Population Points")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Points  : {len(points_df)}")
    print(f"  Years   : {YEARS[0]}–{YEARS[-1]}  ({len(YEARS)} years)")
    print(f"  Params  : {POWER_PARAMETERS}")
    total_calls = len(points_df) * len(YEARS)
    print(f"  Calls   : {total_calls}  ({len(points_df)} points × {len(YEARS)} years)")
    print(f"  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print("═" * 68)

    tracker = StatusTracker(STATUS_FILE)
    session = requests.Session()

    for i, row in enumerate(points_df.itertuples(index=False), start=1):
        for year in YEARS:
            fp = os.path.join(OUTPUT_DIR, f"power_{row.point_id}_{year}.json")
            download_one(session, row.point_id, year, row.lat, row.lon, fp, tracker)

        if i % 10 == 0 or i == len(points_df):
            print(f"\n  Progress: {i}/{len(points_df)} points  |  {tracker.summary()}")

    print("\n" + "═" * 68)
    print("  DOWNLOAD COMPLETE")
    print(f"  {tracker.summary()}")

    failed = tracker.failed()
    if failed:
        print(f"\n  FAILED ({len(failed)}) — re-run script to retry:")
        for pid, yr in failed[:20]:
            print(f"    {pid}  {yr}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")
    else:
        print("  ✅ All files downloaded successfully!")

    print(f"\n  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 68)
    print("\nNext step: run 01_download_era5_rajasthan.py (if not already done), "
          "then 00_unzip_accum.py, then 02_combine_rajasthan.py")


if __name__ == "__main__":
    main()
