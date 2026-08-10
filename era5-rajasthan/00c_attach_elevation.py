"""
ELEVATION ATTACHMENT — RAJASTHAN POPULATION POINTS
=============================================================================
Population points carry no elevation field, so 02_combine_rajasthan.py used
to fall back to a flat 300m approximation for every point when calling
pvlib for solar geometry / clear-sky irradiance. That's a reasonable
simplification for a state like Rajasthan (mostly 200-500m), but breaks
down anywhere terrain varies a lot within the study area.

This script downloads ERA5's time-invariant surface geopotential field (z)
over the bounding envelope of population_grid_points.csv — the same
envelope 01_download_era5_rajasthan.py uses — and attaches a per-point
`elevation_m` column, computed as:

    elevation_m = z / 9.80665      (WMO standard gravity, geopotential -> height)

Geopotential is time-invariant in ERA5 (it's the surface orography, not a
weather field), so this is ONE CDS request for a single date/time, not a
per-year download — cached separately under data/raw/era5/invariant/ so it
never touches (or triggers a re-download of) the sun-event instant/accum
files under data/raw/era5/points/.

Output:
  data/raw/era5/invariant/era5_RJ_geopotential.nc   (raw cache)
  population_grid_points.csv gains an `elevation_m` column (updated in place)

HOW TO RUN:
  python 00c_attach_elevation.py

Safe to re-run — skips the CDS request if the cached NetCDF already exists,
and skips rewriting population_grid_points.csv if every point already has
elevation_m populated.

Caveat (see also README "Notes / known limitations"): ERA5's native grid is
~0.25 deg (~28km), so its orography is a grid-cell MEAN elevation. In areas
with sharp terrain (e.g. Uttarakhand's 200m-7000m+ range), a single cell
value smooths out real local relief. This is an accepted, documented
limitation, not something this script tries to fix further.
"""

import os
import time

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    RAW_INVARIANT_DIR,
    GEOPOTENTIAL_FILE,
    POPULATION_GRID_FILE,
    get_cdsapi_client,
    ensure_data_dirs,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

G0 = 9.80665  # standard gravity (m/s^2) — WMO geopotential -> geopotential-height conversion

# Geopotential/orography is time-invariant in ERA5 — any single valid
# date/time returns the same field. Picking a date well inside the
# reanalysis record purely for API-call hygiene.
INVARIANT_REQUEST_DATE = {"year": "2020", "month": "01", "day": "01", "time": "00:00"}

BBOX_PAD_DEG = 0.5   # same padding 01_download_era5_rajasthan.py uses

MAX_RETRIES = 3
RETRY_WAIT = 60  # seconds between retries

ensure_data_dirs()


# ═══════════════════════════════════════════════════════════
# BBOX  (same logic as 01_download_era5_rajasthan.py's load_points_bbox)
# ═══════════════════════════════════════════════════════════

def load_points_bbox(points_df, pad=BBOX_PAD_DEG):
    north = min(90.0, points_df["lat"].max() + pad)
    south = max(-90.0, points_df["lat"].min() - pad)
    east = min(180.0, points_df["lon"].max() + pad)
    west = max(-180.0, points_df["lon"].min() - pad)
    return [north, west, south, east]


# ═══════════════════════════════════════════════════════════
# DOWNLOAD  (single request, same retry pattern as the other download scripts)
# ═══════════════════════════════════════════════════════════

def download_geopotential(bbox):
    fp = str(GEOPOTENTIAL_FILE)

    if os.path.exists(fp) and os.path.getsize(fp) > 5_000:
        print(f"  [SKIP-FILE] geopotential already downloaded "
              f"({os.path.getsize(fp)/1e3:.1f} KB)")
        return fp

    print(f"  BBox: N={bbox[0]:.2f} W={bbox[1]:.2f} S={bbox[2]:.2f} E={bbox[3]:.2f}")
    c = get_cdsapi_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": ["geopotential"],
                    "year": [INVARIANT_REQUEST_DATE["year"]],
                    "month": [INVARIANT_REQUEST_DATE["month"]],
                    "day": [INVARIANT_REQUEST_DATE["day"]],
                    "time": [INVARIANT_REQUEST_DATE["time"]],
                    "area": bbox,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                fp,
            )
            if not os.path.exists(fp) or os.path.getsize(fp) < 5_000:
                raise RuntimeError("File missing or too small after retrieve()")

            print(f"  [OK]  geopotential  {os.path.getsize(fp)/1e3:.1f} KB")
            return fp

        except Exception as exc:
            print(f"  [FAIL {attempt}/{MAX_RETRIES}]  {str(exc)[:300]}")
            if os.path.exists(fp):
                os.remove(fp)
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_WAIT}s ...")
                time.sleep(RETRY_WAIT)
            else:
                raise


# ═══════════════════════════════════════════════════════════
# EXTRACT ELEVATION AT EACH POINT
# ═══════════════════════════════════════════════════════════

def open_nc(fpath):
    """Same multi-engine fallback as 02_combine_rajasthan.py's open_nc."""
    for engine in ("netcdf4", "scipy", "h5netcdf"):
        try:
            return xr.open_dataset(fpath, engine=engine)
        except Exception:
            pass
    return xr.open_dataset(
        fpath, engine="netcdf4", mask_and_scale=False, decode_cf=False, decode_times=False)


def attach_elevation(points_df, geopotential_fp):
    ds = open_nc(geopotential_fp)

    lat_name = next((c for c in list(ds.coords) + list(ds.dims) if "lat" in c.lower()), None)
    lon_name = next((c for c in list(ds.coords) + list(ds.dims) if "lon" in c.lower()), None)
    z_name = next((v for v in ds.data_vars if v.lower() in ("z", "geopotential")), None)
    if lat_name is None or lon_name is None or z_name is None:
        raise RuntimeError(
            f"Couldn't find lat/lon/geopotential in {geopotential_fp}; "
            f"data_vars={list(ds.data_vars)}")

    lat_arr = ds[lat_name].values.astype(float)
    lon_arr = ds[lon_name].values.astype(float)
    z = np.asarray(ds[z_name].values, dtype=float).squeeze()  # drop time -> (lat, lon)

    elevations = np.empty(len(points_df), dtype=float)
    for i, row in enumerate(points_df.itertuples(index=False)):
        li = int(np.argmin(np.abs(lat_arr - row.lat)))
        lo = int(np.argmin(np.abs(lon_arr - row.lon)))
        elevations[i] = float(z[li, lo]) / G0

    points_df = points_df.copy()
    points_df["elevation_m"] = elevations
    return points_df


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 68)
    print("  ERA5 Elevation Attachment — Rajasthan Population Points")
    print("═" * 68)

    if not POPULATION_GRID_FILE.exists():
        print(f"\n  ERROR: {POPULATION_GRID_FILE} not found — "
              "run 00a_build_population_grid.py first.")
        raise SystemExit(1)

    points_df = pd.read_csv(POPULATION_GRID_FILE)

    if "elevation_m" in points_df.columns and points_df["elevation_m"].notna().all():
        print(f"\n  [SKIP] elevation_m already populated for all "
              f"{len(points_df)} points in {POPULATION_GRID_FILE.name}.")
        elevs = points_df["elevation_m"]
    else:
        bbox = load_points_bbox(points_df)
        print(f"\n  Points  : {len(points_df)}")
        print(f"  Output  : {GEOPOTENTIAL_FILE}")

        geopotential_fp = download_geopotential(bbox)

        print("\n  Attaching per-point elevation ...")
        points_df = attach_elevation(points_df, geopotential_fp)
        points_df.to_csv(POPULATION_GRID_FILE, index=False)
        elevs = points_df["elevation_m"]
        print(f"  Wrote elevation_m -> {POPULATION_GRID_FILE}")

    print("\n" + "─" * 68)
    print("  SANITY CHECK — elevation_m")
    print(f"    min  = {elevs.min():.1f} m")
    print(f"    max  = {elevs.max():.1f} m")
    print(f"    mean = {elevs.mean():.1f} m")
    outliers = points_df[(elevs < -420) | (elevs > 8850)]  # Dead Sea .. Everest
    if not outliers.empty:
        print(f"    ⚠️  {len(outliers)} point(s) outside plausible Earth-surface "
              f"range (-420m .. 8850m) — check {GEOPOTENTIAL_FILE.name}:")
        print(outliers[["point_id", "lat", "lon", "elevation_m"]].to_string(index=False))
    else:
        print("    ✅ all values within plausible range")
    print("─" * 68)

    print("\nNext step: run 02_combine_rajasthan.py (it will now pick up "
          "elevation_m instead of the flat 300m default).")


if __name__ == "__main__":
    main()
