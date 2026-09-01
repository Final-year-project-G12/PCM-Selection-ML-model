"""
POPULATION-WEIGHTED GRID POINTS — ASSAM
=============================================================================
Builds the set of sampling points for Assam, using population-weighted
selection anchored to ERA5's 0.25° native grid.

Sources:
  • Boundary   : GADM v4.1, India admin level 1 (states), GeoJSON
                 https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json
  • Population : WorldPop unconstrained global mosaic, India, UN-adjusted,
                 100m, 2020
                 https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_UNadj.tif

METHOD
------
1. Clip the WorldPop raster to the Assam boundary polygon (GADM NAME_1 == "Assam").
2. Aggregate pixel population onto a 0.25° lat/lon grid anchored to ERA5's origin
   (lat=90.0, lon=-180.0, multiples of 0.25°).
3. Rank cells by population descending, keep the minimal prefix whose
   cumulative population covers >= COVERAGE_TARGET (87.5%) of the state total.
4. Write population_grid_points.csv: point_id, lat, lon, population, weight.

HOW TO RUN:
  python 00a_build_population_grid.py
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

from config import (
    RAW_BOUNDARY_DIR,
    RAW_POPULATION_DIR,
    POPULATION_GRID_FILE,
    ensure_data_dirs,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

STATE_NAME = "Assam"
POINT_ID_PREFIX = "ASP_"

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json"
BOUNDARY_FILE = RAW_BOUNDARY_DIR / "gadm41_IND_1.json"

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/"
    "ind_ppp_2020_UNadj.tif"
)
POPULATION_RASTER_FILE = RAW_POPULATION_DIR / "ind_ppp_2020_UNadj.tif"

GRID_RES = 0.25
ERA5_ORIGIN_LAT = 90.0
ERA5_ORIGIN_LON = -180.0

# Minimal set of highest-population cells covering ~87.5% - 90% of total state population
COVERAGE_TARGET = 0.875

ensure_data_dirs()


# ═══════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════

def download_file(url, dest_path, min_size_bytes=1024, desc=""):
    """Stream-download url to dest_path. Skip if already present."""
    if dest_path.exists():
        sz = dest_path.stat().st_size
        if sz > min_size_bytes:
            print(f"  [SKIP] {desc or dest_path.name}  ({sz/1e6:.1f} MB, already cached)")
            return dest_path
        print(f"  [REMOVE] tiny/corrupt cached file ({sz} B) -- re-downloading")
        dest_path.unlink()

    print(f"  [DOWNLOAD] {desc or dest_path.name}")
    print(f"    <- {url}")
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    max_attempts = 50

    for attempt in range(1, max_attempts + 1):
        resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        mode = "ab" if resume_from else "wb"

        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
                if resume_from and resp.status_code == 200:
                    resume_from = 0
                    mode = "wb"
                resp.raise_for_status()

                content_length = int(resp.headers.get("content-length", 0))
                total = resume_from + content_length if resume_from else content_length
                downloaded = resume_from

                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = 100 * downloaded / total
                                print(f"    {downloaded/1e6:8.1f} / {total/1e6:.1f} MB  ({pct:4.1f}%)",
                                      end="\r")
            print()
            break

        except (requests.exceptions.RequestException, IOError) as exc:
            print(f"\n    [WARN] Download interrupted (attempt {attempt}/{max_attempts}): {str(exc)[:200]}")
            if attempt == max_attempts:
                raise RuntimeError(f"Download failed after {max_attempts} attempts: {url}") from exc
            import time
            time.sleep(5)
            print(f"    Retrying (resuming from {tmp_path.stat().st_size / 1e6:.1f} MB if possible) ...")

    os.replace(tmp_path, dest_path)
    sz = dest_path.stat().st_size
    if sz <= min_size_bytes:
        raise RuntimeError(f"Downloaded file too small ({sz} bytes): {dest_path}")
    print(f"    [OK] {sz/1e6:.1f} MB -> {dest_path}")
    return dest_path


def load_assam_boundary():
    download_file(GADM_URL, BOUNDARY_FILE, min_size_bytes=10_000,
                  desc="GADM India admin-1 boundary")
    gdf = gpd.read_file(BOUNDARY_FILE)
    state = gdf[gdf["NAME_1"] == STATE_NAME]
    if state.empty:
        raise RuntimeError(
            f"'{STATE_NAME}' not found in GADM NAME_1 column -- "
            f"available values: {sorted(gdf['NAME_1'].unique())}")
    return state.geometry.iloc[0]


# ═══════════════════════════════════════════════════════════
# POPULATION AGGREGATION
# ═══════════════════════════════════════════════════════════

def aggregate_population(raster_path, boundary_geom):
    print("  Opening + clipping population raster ...")
    with rasterio.open(raster_path) as src:
        clipped, clipped_transform = rio_mask(
            src, [boundary_geom], crop=True, nodata=0, filled=True)
        band = clipped[0].astype(np.float64)
        band[band < 0] = 0.0

        height, width = band.shape
        cols = np.arange(width)
        rows = np.arange(height)
        xs = clipped_transform.c + (cols + 0.5) * clipped_transform.a
        ys = clipped_transform.f + (rows + 0.5) * clipped_transform.e

    print(f"  Raster clipped: {width} x {height} px, "
          f"{band.sum():,.0f} total population (raw pixel sum)")

    lon_cell_idx = np.floor((xs - ERA5_ORIGIN_LON) / GRID_RES).astype(np.int64)
    lat_cell_idx = np.floor((ERA5_ORIGIN_LAT - ys) / GRID_RES).astype(np.int64)

    lon_min, lon_max = lon_cell_idx.min(), lon_cell_idx.max()
    lat_min, lat_max = lat_cell_idx.min(), lat_cell_idx.max()
    n_lon_cells = int(lon_max - lon_min + 1)
    n_lat_cells = int(lat_max - lat_min + 1)
    lon_offset = (lon_cell_idx - lon_min).astype(np.intp)

    accumulator = np.zeros((n_lat_cells, n_lon_cells), dtype=np.float64)
    for r in range(height):
        row = band[r]
        if not row.any():
            continue
        row_sums = np.bincount(lon_offset, weights=row, minlength=n_lon_cells)
        accumulator[lat_cell_idx[r] - lat_min] += row_sums

    lat_i, lon_i = np.nonzero(accumulator)
    pops = accumulator[lat_i, lon_i]

    cell_lat = ERA5_ORIGIN_LAT - (lat_i + lat_min + 0.5) * GRID_RES
    cell_lon = ERA5_ORIGIN_LON + (lon_i + lon_min + 0.5) * GRID_RES

    df = pd.DataFrame({"lat": cell_lat, "lon": cell_lon, "population": pops})
    return df.sort_values("population", ascending=False).reset_index(drop=True)


def select_top_coverage(df, target=COVERAGE_TARGET):
    total = df["population"].sum()
    cumulative = df["population"].cumsum()
    cutoff = (cumulative / total >= target).idxmax()
    selected = df.iloc[:cutoff + 1].copy()
    covered_pct = 100 * selected["population"].sum() / total
    return selected, total, covered_pct


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print(f"  Population-Weighted Grid Points — {STATE_NAME}")
    print("=" * 68)

    print(f"\n[1/4] {STATE_NAME} boundary (GADM level-1) ...")
    boundary = load_assam_boundary()

    print("\n[2/4] WorldPop population raster (India, 2020, UN-adjusted) ...")
    download_file(WORLDPOP_URL, POPULATION_RASTER_FILE, min_size_bytes=10_000_000,
                  desc="WorldPop India population raster")

    print(f"\n[3/4] Aggregating onto a {GRID_RES}° grid (ERA5-aligned) ...")
    grid_df = aggregate_population(POPULATION_RASTER_FILE, boundary)
    print(f"  {len(grid_df)} populated cells inside {STATE_NAME} boundary")

    print(f"\n[4/4] Selecting minimal set covering >= {COVERAGE_TARGET*100:.1f}% of population ...")
    selected, total_pop, covered_pct = select_top_coverage(grid_df)
    selected = selected.reset_index(drop=True)
    selected["point_id"] = [f"{POINT_ID_PREFIX}{i+1:04d}" for i in range(len(selected))]
    selected["weight"] = selected["population"] / selected["population"].sum()
    selected = selected[["point_id", "lat", "lon", "population", "weight"]]

    selected.to_csv(POPULATION_GRID_FILE, index=False)

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Total state population (raster sum): {total_pop:,.0f}")
    print(f"  Points selected  : {len(selected)}")
    print(f"  Coverage achieved: {covered_pct:.1f}%")
    print(f"  Lat range        : {selected['lat'].min():.2f} .. {selected['lat'].max():.2f}")
    print(f"  Lon range        : {selected['lon'].min():.2f} .. {selected['lon'].max():.2f}")
    print(f"  Output           : {POPULATION_GRID_FILE}")
    print("=" * 68)
    print("\nNext step: run  00b_build_suntimes.py")


if __name__ == "__main__":
    main()
