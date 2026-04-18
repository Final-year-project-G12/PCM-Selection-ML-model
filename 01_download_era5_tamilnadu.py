"""
ERA5 DOWNLOAD — TAMIL NADU STATE GRID
======================================
Downloads the entire Tamil Nadu state as a single bounding-box grid.

Bounding box  : N=13.75, W=75.75, S=7.75, E=81.25  (covers all of TN + buffer)
Resolution    : 0.25° × 0.25° (~25 km, ERA5 native)
Grid points   : ~24 lat × 22 lon = ~528 points
API calls     : 2 years × 12 months × 2 var-types = 48 total calls

Two separate API calls per month (ERA5 rule):
  • instant  — analysis variables (TYPE=AN): temperature, wind, humidity, pressure
  • accum    — forecast variables (TYPE=FC): solar radiation, precipitation

HOW TO RUN:
  python 01_download_era5_tamilnadu.py 2>&1 | Tee-Object -FilePath download_log.txt

Re-running is safe — completed files are skipped automatically.

IMPORTANT: You deleted the accum files. Re-run this script to get them back.
The accum files are REQUIRED for GHI, DNI, DHI, LW_down, precipitation.
"""

import cdsapi
import os
import csv
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

YEARS  = ["2024", "2025"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
HOURS  = [f"{h:02d}:00" for h in range(0, 24)]

# Full Tamil Nadu bounding box [North, West, South, East]
# Slightly wider than TN borders to catch all edge locations
TN_BBOX = [13.75, 75.75, 7.75, 81.25]

OUTPUT_DIR  = "data/raw/era5/grid"
STATUS_FILE = "data/raw/era5/download_status.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)

MAX_RETRIES = 3
RETRY_WAIT  = 60  # seconds between retries

# ═══════════════════════════════════════════════════════════
# VARIABLE GROUPS
#
# INSTANT (analysis/AN) — snapshot at each hour
#   These are all in one request.
#
# ACCUM (forecast/FC) — accumulated since forecast start
#   MUST be a separate request from instant vars.
#   ERA5 reanalysis accumulations reset every 12 hours
#   (at 00 UTC and 12 UTC forecast runs).
#
# REMOVED: "surface_diffuse_solar_radiation_downwards"
#   — this variable does NOT exist in ERA5. MARS rejects it.
#   DHI is computed in script 02 as: DHI = GHI - DNI × cos(SZA)
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
    "surface_thermal_radiation_downwards",            # strd  → LW_down (J/m², accum)
    "total_precipitation",                            # tp    → rain (m, accum)
]


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

def download_one(c, year, month, var_type, variables, filepath, tracker):
    vt = var_type.strip()

    # Skip if already logged as OK
    if tracker.is_done(year, month, vt):
        print(f"  [SKIP-LOG]  {year}-{month}  {vt}  (already OK in status CSV)")
        return "SKIP"

    # Skip if file already exists and is large enough
    if os.path.exists(filepath):
        sz = os.path.getsize(filepath)
        if sz > 50_000:
            print(f"  [SKIP-FILE] {year}-{month}  {vt}  ({sz/1e6:.1f} MB)")
            tracker.log(year, month, vt, "SKIP", filepath, sz/1e6, "file existed")
            return "SKIP"
        else:
            print(f"  [REMOVE]   tiny/corrupt file ({sz} B) — re-downloading")
            os.remove(filepath)

    print(f"\n  ── {year}-{month}  [{vt}] ──")
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
                    "time":            HOURS,
                    "area":            TN_BBOX,
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
    print("  ERA5 Tamil Nadu — State Grid Download")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  BBox    : N={TN_BBOX[0]} W={TN_BBOX[1]} S={TN_BBOX[2]} E={TN_BBOX[3]}")
    print(f"  Years   : {YEARS}")
    total_calls = len(YEARS) * len(MONTHS) * 2
    print(f"  Calls   : {total_calls}  (2 years × 12 months × 2 types)")
    print(f"  Output  : {OUTPUT_DIR}/")
    print(f"  Status  : {STATUS_FILE}")
    print()
    print("  NOTE: If you deleted accum files, this will re-download them.")
    print("  Instant files already on disk will be SKIPPED automatically.")
    print("═" * 68)

    tracker = StatusTracker(STATUS_FILE)
    c = cdsapi.Client()

    for year in YEARS:
        for month in MONTHS:
            print(f"\n{'─'*56}")
            print(f"  Processing: {year}-{month}")
            print(f"{'─'*56}")

            # ── Request 1: Instant (analysis) variables ──
            fi = os.path.join(OUTPUT_DIR,
                              f"era5_TN_grid_{year}_{month}_instant.nc")
            download_one(c, year, month, "instant", INSTANT_VARS, fi, tracker)

            # ── Request 2: Accumulated (forecast) variables ──
            fa = os.path.join(OUTPUT_DIR,
                              f"era5_TN_grid_{year}_{month}_accum.nc")
            download_one(c, year, month, "accum", ACCUM_VARS, fa, tracker)

            print(f"\n  Progress: {tracker.summary()}")

    # ── Final summary ──
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
    print("\nNext step: run  02_combine_tamilnadu.py")


if __name__ == "__main__":
    main()