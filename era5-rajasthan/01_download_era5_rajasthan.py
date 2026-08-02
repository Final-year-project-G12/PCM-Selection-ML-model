"""
ERA5 DOWNLOAD — RAJASTHAN POPULATION-WEIGHTED POINTS  (sun-event-aligned, 10-year history)
=============================================================================
Downloads ERA5 over the bounding envelope of population_grid_points.csv
(built by 00a_build_population_grid.py) — the population-weighted subset of
Rajasthan, not the full state — at UTC hours computed from suntimes.csv
(built by 00b_build_suntimes.py) rather than fixed clock hours.

Bounding box  : envelope of population_grid_points.csv lat/lon, padded 0.5°
Years         : 2016–2025  (past 10 full calendar years)
API calls     : 10 years × 12 months × 2 var-types = 240 total calls

SUN-EVENT-ALIGNED HOURS instead of fixed 3 readings/day
---------------------------------------------------------
Rather than fixed 02:00/08:00/14:00 UTC, this computes three separate
narrow UTC hour windows — one around sunrise, one around solar noon, one
around sunset — each wide enough to cover the seasonal + longitude spread
of that event across every point in population_grid_points.csv, all year,
plus a ~1hr margin. These three windows are unioned into INSTANT_HOURS.

Cross-midnight UTC dates are real here, not hypothetical: Rajasthan's
longitude span means an eastern point's summer sunrise can land at ~23:55
UTC of the *previous* UTC calendar date. The hour-window computation below
handles this with a circular (mod-24) window rather than a naive min/max,
since a plain numeric min/max across hours like {23, 0, 1, 2} would be
nonsensical (min=0, max=23 spans the whole day).

Two separate API calls per month (ERA5 rule), same as before:
  • instant  — analysis variables (TYPE=AN): temperature, wind, humidity, pressure
      Needs exactly INSTANT_HOURS — snapshots don't depend on neighbours.
  • accum    — forecast variables (TYPE=FC): solar radiation, precipitation
      ERA5 accumulated fields are cumulative since the last forecast reset
      (00 UTC or 12 UTC). To recover the true 1-hour flux via diff() (see
      deaccumulate() in 02_combine_rajasthan.py, unchanged), every target
      hour's immediate predecessor is also downloaded:
          ACCUM_HOURS = INSTANT_HOURS ∪ {(h - 1) mod 24 for h in INSTANT_HOURS}
      deaccumulate()'s reset_mask = hour.isin([1, 13]) already generalizes
      correctly to this wider/contiguous hour set — see that function's
      docstring for why hours 1 and 13 (the first step after each 00Z/12Z
      reset) must use the raw value directly rather than diff().

      One true edge case: 2016-01-01 has no 2015-12-31 file to supply hour
      23 as hour 0's predecessor. Because 02_combine_rajasthan.py
      concatenates every month/year into one continuous sorted series per
      point *before* calling deaccumulate(), every other month boundary is
      bridged automatically (hour 23 is downloaded on every day of every
      month, including each month's last day) — only that single first day
      of the whole 10-year dataset is short a predecessor, and pandas'
      diff() naturally yields NaN there rather than a silently wrong value.

HOW TO RUN:
  1. Save your API key in era5-rajasthan/.cdsapirc  (see .cdsapirc.example)
  2. Run 00a_build_population_grid.py and 00b_build_suntimes.py first.
  3. python 01_download_era5_rajasthan.py

  Scripts can be run from any folder; paths resolve via config.py.

Re-running is safe — completed files are skipped automatically. This writes
to RAW_POINTS_DIR / POINTS_DOWNLOAD_STATUS_FILE — distinct paths from the
old full-state-grid download (RAW_GRID_DIR / DOWNLOAD_STATUS_FILE), so nothing
here touches the ~7.5 years of NetCDF already downloaded under the old
fixed-hour scheme.
"""

import os
import csv
import time
from datetime import datetime

import pandas as pd

from config import (
    RAW_POINTS_DIR,
    POINTS_DOWNLOAD_STATUS_FILE,
    POPULATION_GRID_FILE,
    SUNTIMES_FILE,
    get_cdsapi_client,
    ensure_data_dirs,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

YEARS  = [str(y) for y in range(2016, 2026)]   # past 10 full calendar years
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]

HOUR_MARGIN = 1   # hours of padding on each side of each event's window

OUTPUT_DIR  = str(RAW_POINTS_DIR)
STATUS_FILE = str(POINTS_DOWNLOAD_STATUS_FILE)
ensure_data_dirs()

MAX_RETRIES = 3
RETRY_WAIT  = 60  # seconds between retries

# ═══════════════════════════════════════════════════════════
# VARIABLE GROUPS  (unchanged from the original state-grid pipeline)
#
# INSTANT (analysis/AN) — snapshot at each selected hour
# ACCUM (forecast/FC) — accumulated since forecast start, MUST be a
#   separate request from instant vars.
# DHI is computed in 02_combine_rajasthan.py as: DHI = GHI - DNI × cos(SZA)
# ═══════════════════════════════════════════════════════════

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
    """Minimal circular (mod-24) span of hours covering every hour in
    `hours_observed`, expanded by `margin` hours on each end.

    A plain min()/max() over raw hour-of-day integers breaks whenever the
    observed hours straddle the UTC midnight boundary (e.g. {23, 0, 1, 2}
    would numerically look like it spans the whole day). This instead finds
    the largest unobserved circular gap and takes everything else.
    """
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
        gap = (b - a - 1) % 24  # hours strictly between a and b, going forward
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


RJ_BBOX, N_POINTS = load_points_bbox()
INSTANT_HOURS, ACCUM_HOURS, EVENT_WINDOWS = compute_hour_windows()


# ═══════════════════════════════════════════════════════════
# STATUS TRACKER  (unchanged pattern from the original script)
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
# DOWNLOAD FUNCTION  (unchanged from the original script)
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
            print(f"  [REMOVE]   tiny/corrupt file ({sz} B) — re-downloading")
            os.remove(filepath)

    print(f"\n  ── {year}-{month}  [{vt}]  hours={hours} ──")
    for v in variables:
        print(f"     {v}")
    print(f"  → {filepath}")

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
                    "area":            RJ_BBOX,
                    "data_format":     "netcdf",
                    "download_format": "unarchived",
                },
                filepath,
            )

            if not os.path.exists(filepath):
                raise RuntimeError("File not created after retrieve()")
            sz = os.path.getsize(filepath)
            if sz < 50_000:
                raise RuntimeError(f"File too small ({sz} bytes) — corrupt download")

            size_mb = sz / 1e6
            print(f"  [OK]  {year}-{month}  {vt}  {size_mb:.1f} MB ✓")
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
    print("\n" + "═" * 68)
    print("  ERA5 Rajasthan — Population Points Download  (sun-event hours, 10 years)")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Points  : {N_POINTS}  (from {POPULATION_GRID_FILE.name})")
    print(f"  BBox    : N={RJ_BBOX[0]:.2f} W={RJ_BBOX[1]:.2f} "
          f"S={RJ_BBOX[2]:.2f} E={RJ_BBOX[3]:.2f}")
    print(f"  Years   : {YEARS[0]}–{YEARS[-1]}  ({len(YEARS)} years)")
    for event, hours in EVENT_WINDOWS.items():
        print(f"  {event:8s} window : {[f'{h:02d}:00' for h in hours]}")
    print(f"  Instant hours : {INSTANT_HOURS}  ({len(INSTANT_HOURS)} hours/day)")
    print(f"  Accum hours   : {ACCUM_HOURS}  ({len(ACCUM_HOURS)} hours/day, "
          f"includes deaccumulation predecessor hours)")
    total_calls = len(YEARS) * len(MONTHS) * 2
    print(f"  Calls   : {total_calls}  ({len(YEARS)} years × 12 months × 2 types)")
    print(f"  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print("═" * 68)

    tracker = StatusTracker(STATUS_FILE)
    c = get_cdsapi_client()

    for year in YEARS:
        for month in MONTHS:
            print(f"\n{'─'*56}")
            print(f"  Processing: {year}-{month}")
            print(f"{'─'*56}")

            fi = os.path.join(OUTPUT_DIR,
                              f"era5_RJ_points_{year}_{month}_instant.nc")
            download_one(c, year, month, "instant", INSTANT_VARS,
                        INSTANT_HOURS, fi, tracker)

            fa = os.path.join(OUTPUT_DIR,
                              f"era5_RJ_points_{year}_{month}_accum.nc")
            download_one(c, year, month, "accum", ACCUM_VARS,
                        ACCUM_HOURS, fa, tracker)

            print(f"\n  Progress: {tracker.summary()}")

    print("\n" + "═" * 68)
    print("  DOWNLOAD COMPLETE")
    print(f"  {tracker.summary()}")

    failed = tracker.failed()
    if failed:
        print(f"\n  FAILED ({len(failed)}) — re-run script to retry:")
        for yr, mo, vt in failed:
            print(f"    {yr}-{mo}  {vt}")
    else:
        print("  ✅ All files downloaded successfully!")

    print(f"\n  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 68)
    print("\nNext step: run  01b_download_nasapower.py  (if not already done), "
          "then  00_unzip_accum.py,  then  02_combine_rajasthan.py")


if __name__ == "__main__":
    main()
