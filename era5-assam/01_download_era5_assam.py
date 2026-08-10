"""
ERA5 DOWNLOAD — ASSAM POPULATION-WEIGHTED POINTS  (sun-event-aligned, 10-year history)
=============================================================================
Downloads ERA5 over the bounding envelope of population_grid_points.csv for Assam
at UTC hours computed from suntimes.csv (built by 00b_build_suntimes.py).

Bounding box  : envelope of population_grid_points.csv lat/lon, padded 0.5°
Years         : 2016–2025  (past 10 full calendar years)
API calls     : 10 years × 12 months × 2 var-types = 240 total calls

HOW TO RUN:
  1. Save your API key in era5-assam/.cdsapirc
  2. Run 00a_build_population_grid.py and 00b_build_suntimes.py first.
  3. python 01_download_era5_assam.py
"""

import os
import csv
import time
from datetime import datetime

import pandas as pd

from config import (
    POPULATION_GRID_FILE,
    SUNTIMES_FILE,
    RAW_POINTS_DIR,
    POINTS_DOWNLOAD_STATUS_FILE,
    get_cdsapi_client,
    ensure_data_dirs,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

YEARS = [str(y) for y in range(2016, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

# Expand sun-event window by this many hours on each end
HOUR_MARGIN = 1

OUTPUT_DIR = str(RAW_POINTS_DIR)
STATUS_FILE = str(POINTS_DOWNLOAD_STATUS_FILE)
ensure_data_dirs()

MAX_RETRIES = 3
RETRY_WAIT = 30

INSTANT_VARS = [
    "2m_temperature",               # t2m  → T_amb (K → °C)
    "2m_dewpoint_temperature",      # d2m  → T_dew → RH
    "10m_u_component_of_wind",      # u10  → wind U (m/s)
    "10m_v_component_of_wind",      # v10  → wind V (m/s)
    "total_cloud_cover",            # tcc  → cloud fraction (0–1)
    "surface_pressure",             # sp   → P_atm (Pa → hPa)
]

ACCUM_VARS = [
    "surface_solar_radiation_downwards",              # ssrd  → GHI (J/m², accum)
    "surface_solar_radiation_downward_clear_sky",     # ssrdc → GHI_clearsky_era5 (J/m², accum)
                                                      #   Phase 1 spec: CSI = ssrd/ssrdc; backbone of
                                                      #   every solar-availability index.
    "mean_surface_direct_short_wave_radiation_flux",  # msdwswrf → avg DNI (W/m², mean rate)
    "surface_thermal_radiation_downwards",             # strd  → LW_down (J/m², accum)
    "total_precipitation",                             # tp    → rain (m, accum)
]


# ═══════════════════════════════════════════════════════════
# POPULATION-POINT BBOX + SUN-EVENT HOUR WINDOWS
# ═══════════════════════════════════════════════════════════

def load_points_bbox(pad=0.5):
    if not POPULATION_GRID_FILE.exists():
        raise FileNotFoundError(
            f"{POPULATION_GRID_FILE} not found — run 00a_build_population_grid.py first.")
    points_df = pd.read_csv(POPULATION_GRID_FILE)
    north = min(90.0, points_df["lat"].max() + pad)
    south = max(-90.0, points_df["lat"].min() - pad)
    east  = min(180.0, points_df["lon"].max() + pad)
    west  = max(-180.0, points_df["lon"].min() - pad)
    return [north, west, south, east], len(points_df)


def circular_hour_window(hours_observed, margin=HOUR_MARGIN):
    obs = sorted(set(int(h) for h in hours_observed))
    if not obs:
        return []
    if len(obs) >= 24:
        return list(range(24))

    best_gap = -1
    arc_start, arc_end = obs[0], obs[-1]
    for i in range(len(obs)):
        a = obs[i]
        b = obs[(i + 1) % len(obs)]
        gap = (b - a - 1) % 24
        if gap > best_gap:
            best_gap = gap
            arc_start, arc_end = b, a

    arc_len = 24 - best_gap
    lo = (arc_start - margin) % 24
    total_len = arc_len + 2 * margin
    if total_len >= 24:
        return list(range(24))
    return sorted(set((lo + i) % 24 for i in range(total_len)))


def compute_hour_windows():
    if not SUNTIMES_FILE.exists():
        raise FileNotFoundError(
            f"{SUNTIMES_FILE} not found — run 00b_build_suntimes.py first.")
    sun_df = pd.read_csv(SUNTIMES_FILE)
    sun_df["time_utc"] = pd.to_datetime(sun_df["time_utc"], utc=True)
    sun_df["hour"] = sun_df["time_utc"].dt.hour

    windows = {}
    for event in ("sunrise", "noon", "sunset"):
        hours_observed = set(sun_df.loc[sun_df["event"] == event, "hour"].unique())
        windows[event] = circular_hour_window(hours_observed)

    instant_hours = sorted(set().union(*windows.values()))
    accum_hours = sorted(set(instant_hours) | {(h - 1) % 24 for h in instant_hours})

    fmt = lambda hours: [f"{h:02d}:00" for h in hours]
    return fmt(instant_hours), fmt(accum_hours), windows


AS_BBOX, N_POINTS = load_points_bbox()
INSTANT_HOURS, ACCUM_HOURS, EVENT_WINDOWS = compute_hour_windows()


# ═══════════════════════════════════════════════════════════
# STATUS TRACKER
# ═══════════════════════════════════════════════════════════

class StatusTracker:
    FIELDS = ["timestamp", "year", "month", "var_type",
              "status", "filepath", "size_mb", "note"]

    def __init__(self, filepath):
        self.filepath  = filepath
        self.records   = []
        self._done_set = set()
        if os.path.exists(filepath):
            with open(filepath, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.records.append(row)
                    if row["status"] == "OK":
                        self._done_set.add(
                            (row["year"], row["month"], row["var_type"].strip()))

    def is_done(self, year, month, var_type):
        return (year, month, var_type.strip()) in self._done_set

    def log(self, year, month, var_type, status, filepath, size_mb=0.0, note=""):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "year": year, "month": month, "var_type": var_type.strip(),
            "status": status, "filepath": filepath,
            "size_mb": f"{size_mb:.2f}", "note": str(note)[:300],
        }
        self.records.append(row)
        if status == "OK":
            self._done_set.add((year, month, var_type.strip()))
        self._flush()

    def _flush(self):
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.records)

    def summary(self):
        ok   = sum(1 for r in self.records if r["status"] == "OK")
        skip = sum(1 for r in self.records if r["status"] == "SKIP")
        fail = sum(1 for r in self.records if r["status"] == "FAIL")
        return f"OK={ok}  SKIP={skip}  FAIL={fail}  Total={len(self.records)}"

    def failed(self):
        return [(r["year"], r["month"], r["var_type"])
                for r in self.records if r["status"] == "FAIL"]


# ═══════════════════════════════════════════════════════════
# DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════

def download_one(c, year, month, var_type, variables, hours, filepath, tracker):
    vt = var_type.strip()

    if tracker.is_done(year, month, vt):
        print(f"  [SKIP-LOG]  {year}-{month}  {vt}  (already OK in status CSV)")
        return "SKIP"

    if os.path.exists(filepath):
        sz = os.path.getsize(filepath)
        if sz > 50_000:
            print(f"  [SKIP-FILE] {year}-{month}  {vt}  ({sz/1e6:.1f} MB)")
            tracker.log(year, month, vt, "SKIP", filepath, sz/1e6, "file existed")
            return "SKIP"
        else:
            print(f"  [REMOVE]   tiny/corrupt file ({sz} B) -- re-downloading")
            os.remove(filepath)

    print(f"\n  -- {year}-{month}  [{vt}]  hours={hours} --")
    for v in variables:
        print(f"     {v}")
    print(f"  -> {filepath}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type":    ["reanalysis"],
                    "variable":        variables,
                    "year":            [year],
                    "month":           [month],
                    "day":             DAYS,
                    "time":            hours,
                    "area":            AS_BBOX,
                    "data_format":     "netcdf",
                    "download_format": "unarchived",
                },
                filepath,
            )

            if not os.path.exists(filepath):
                raise RuntimeError("File not created after retrieve()")
            sz = os.path.getsize(filepath)
            if sz < 50_000:
                raise RuntimeError(f"File too small ({sz} bytes) -- corrupt download")

            size_mb = sz / 1e6
            print(f"  [OK]  {year}-{month}  {vt}  {size_mb:.1f} MB [OK]")
            tracker.log(year, month, vt, "OK", filepath, size_mb)
            return "OK"

        except Exception as exc:
            msg = str(exc)
            print(f"  [FAIL {attempt}/{MAX_RETRIES}]  {msg[:300]}")
            if os.path.exists(filepath):
                os.remove(filepath)
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_WAIT}s ...")
                time.sleep(RETRY_WAIT)
            else:
                tracker.log(year, month, vt, "FAIL", filepath, 0, msg[:300])
                return "FAIL"


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 68)
    print("  ERA5 Assam -- Population Points Download  (sun-event hours, 10 years)")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Points  : {N_POINTS}  (from {POPULATION_GRID_FILE.name})")
    print(f"  BBox    : N={AS_BBOX[0]:.2f} W={AS_BBOX[1]:.2f} "
          f"S={AS_BBOX[2]:.2f} E={AS_BBOX[3]:.2f}")
    print(f"  Years   : {YEARS[0]}--{YEARS[-1]}  ({len(YEARS)} years)")
    for event, hours in EVENT_WINDOWS.items():
        print(f"  {event:8s} window : {[f'{h:02d}:00' for h in hours]}")
    print(f"  Instant hours : {INSTANT_HOURS}  ({len(INSTANT_HOURS)} hours/day)")
    print(f"  Accum hours   : {ACCUM_HOURS}  ({len(ACCUM_HOURS)} hours/day, "
          f"includes deaccumulation predecessor hours)")
    total_calls = len(YEARS) * len(MONTHS) * 2
    print(f"  Calls   : {total_calls}  ({len(YEARS)} years x 12 months x 2 types)")
    print(f"  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print("=" * 68)

    tracker = StatusTracker(STATUS_FILE)
    c = get_cdsapi_client()

    for year in YEARS:
        for month in MONTHS:
            print(f"\n{'-'*56}")
            print(f"  Processing: {year}-{month}")
            print(f"{'-'*56}")

            fi = os.path.join(OUTPUT_DIR,
                              f"era5_AS_points_{year}_{month}_instant.nc")
            download_one(c, year, month, "instant", INSTANT_VARS,
                        INSTANT_HOURS, fi, tracker)

            fa = os.path.join(OUTPUT_DIR,
                              f"era5_AS_points_{year}_{month}_accum.nc")
            download_one(c, year, month, "accum", ACCUM_VARS,
                        ACCUM_HOURS, fa, tracker)

            print(f"\n  Progress: {tracker.summary()}")

    print("\n" + "=" * 68)
    print("  DOWNLOAD COMPLETE")
    print(f"  {tracker.summary()}")

    failed = tracker.failed()
    if failed:
        print(f"\n  FAILED ({len(failed)}) -- re-run script to retry:")
        for yr, mo, vt in failed:
            print(f"    {yr}-{mo}  {vt}")
    else:
        print("  [OK] All files downloaded successfully!")

    print(f"\n  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print("\nNext step: run  01b_download_nasapower.py,  then  00_unzip_accum.py,  then  02_combine_assam.py")


if __name__ == "__main__":
    main()
