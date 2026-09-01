"""
COMBINE — ASSAM POPULATION POINTS  (ERA5 + NASA POWER cross-check)
=============================================================================
Reads  :
  data/raw/era5/points/era5_AS_points_{year}_{month}_{instant,accum}.nc
  data/raw/nasapower/power_{point_id}_{year}.json
  data/processed/population_grid_points.csv
  data/processed/suntimes.csv
Writes :
  data/processed/climate_assam_points.csv

HOW TO RUN:
  python 02_combine_assam.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import pvlib

from config import (
    RAW_POINTS_DIR,
    RAW_POWER_DIR,
    POPULATION_GRID_FILE,
    SUNTIMES_FILE,
    COMBINED_POINTS_FILE,
    ensure_data_dirs,
)

# ===========================================================
# CONFIG
# ===========================================================

INPUT_DIR = str(RAW_POINTS_DIR)
POWER_DIR = str(RAW_POWER_DIR)
ensure_data_dirs()

YEARS  = [str(y) for y in range(2016, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Default elevation approximation (Assam valley/plains baseline ~100m)
DEFAULT_ALT_M = 100

MAX_MATCH_HOURS = 3

SEASON_MAP = {
    12: ("Winter", 1),  1: ("Winter", 1),  2: ("Winter", 1),
     3: ("Pre-Monsoon", 2), 4: ("Pre-Monsoon", 2), 5: ("Pre-Monsoon", 2),
     6: ("Monsoon", 3), 7: ("Monsoon", 3), 8: ("Monsoon", 3), 9: ("Monsoon", 3),
    10: ("Post-Monsoon", 4), 11: ("Post-Monsoon", 4),
}

ERA5_OUTPUT_VARS = [
    "T_amb", "T_dew", "RHum", "W_spd", "W_dir",
    "GHI", "DNI", "DHI", "LW_down", "cloud_cover", "precipitation", "P_atm",
    "SZA", "solar_azimuth",
    "GHI_clearsky",       # final clear-sky GHI used for CSI (era5_ssrdc if available, else pvlib)
    "GHI_clearsky_era5",  # ERA5 ssrdc deaccumulated (W/m²) — direct from reanalysis
    "clearsky_source",    # 'era5_ssrdc' | 'pvlib_ineichen' — audit trail
    "CSI",
]
POWER_VARS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]


# ===========================================================
# FILE OPENING
# ===========================================================

def open_nc(fpath):
    if not os.path.exists(fpath):
        return None
    for engine in ("netcdf4", "scipy", "h5netcdf"):
        try:
            return xr.open_dataset(fpath, engine=engine)
        except Exception:
            pass
    try:
        return xr.open_dataset(
            fpath, engine="netcdf4",
            mask_and_scale=False, decode_cf=False, decode_times=False)
    except Exception as e:
        print(f"    [WARN] Cannot open {os.path.basename(fpath)}: {e}")
        return None


def safe_values(da):
    arr = da.values.astype(float)
    attrs = getattr(da, "attrs", {})
    if "scale_factor" in attrs:
        arr = arr * float(attrs["scale_factor"])
    if "add_offset" in attrs:
        arr = arr + float(attrs["add_offset"])
    if "_FillValue" in attrs:
        fv = float(attrs["_FillValue"])
        arr = np.where(np.abs(arr - fv) < 1e-4, np.nan, arr)
    return arr


def decode_time(ds_pt):
    for tc in ("valid_time", "time"):
        if tc not in ds_pt.coords and tc not in ds_pt.dims:
            continue
        tv = ds_pt[tc].values
        try:
            t = pd.to_datetime(tv)
            return t.tz_localize(None) if t.tz is not None else t
        except Exception:
            pass
        try:
            units = ds_pt[tc].attrs.get("units", "hours since 1900-01-01")
            t = pd.to_datetime(
                xr.coding.times.decode_cf_datetime(tv, units), utc=True
            ).tz_localize(None)
            return t
        except Exception:
            pass
    return None


def extract_nearest(ds, lat, lon):
    lat_name = next((c for c in list(ds.coords) + list(ds.dims)
                     if "lat" in c.lower()), None)
    lon_name = next((c for c in list(ds.coords) + list(ds.dims)
                     if "lon" in c.lower()), None)
    if not lat_name or not lon_name:
        return None

    lat_arr = ds[lat_name].values.astype(float)
    lon_arr = ds[lon_name].values.astype(float)
    li = int(np.argmin(np.abs(lat_arr - lat)))
    lo = int(np.argmin(np.abs(lon_arr - lon)))

    ds_pt = ds.isel({lat_name: li, lon_name: lo})
    time_coord = decode_time(ds_pt)
    if time_coord is None:
        return None

    records = {}
    for var in ds_pt.data_vars:
        try:
            arr = safe_values(ds_pt[var]).squeeze()
            if arr.shape == time_coord.shape:
                records[var] = arr
        except Exception:
            pass

    if not records:
        return None

    df = pd.DataFrame(records, index=time_coord)
    df.index.name = "time"
    df.attrs["grid_lat"] = float(lat_arr[li])
    df.attrs["grid_lon"] = float(lon_arr[lo])
    return df.sort_index()


# ===========================================================
# PHYSICS FUNCTIONS
# ===========================================================

def kelvin_to_c(arr):
    return np.asarray(arr, dtype=float) - 273.15


def compute_rh(T_c, Td_c):
    a, b = 17.625, 243.04
    return np.clip(
        100 * np.exp(a * Td_c / (b + Td_c)) / np.exp(a * T_c / (b + T_c)),
        0, 100)


def deaccumulate(s):
    return pd.Series(np.asarray(s, dtype=float), index=s.index).copy()


def compute_solar(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon,
                                  altitude=alt, tz="UTC")
    times = pd.DatetimeIndex(df.index)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")

    df["SZA"]           = sp["zenith"].values
    df["solar_azimuth"] = sp["azimuth"].values
    df["ETR"]           = pvlib.irradiance.get_extra_radiation(times).values

    # -- Clear-sky GHI source selection --------------------------------------
    # Phase 1 spec: "Request ssrd and ssrdc together — the ratio is your
    # clear-sky index, which is the backbone of every solar-availability index."
    # Prefer ERA5 ssrdc (deaccumulated in apply_unit_conversions as
    # GHI_clearsky_era5).  Fall back to pvlib Ineichen when ssrdc is absent
    # from the NetCDF (e.g., files downloaded before ssrdc was added to
    # ACCUM_VARS); the fallback keeps the pipeline running during the transition.
    if "GHI_clearsky_era5" in df.columns and df["GHI_clearsky_era5"].notna().any():
        df["GHI_clearsky"]    = df["GHI_clearsky_era5"]
        df["clearsky_source"] = "era5_ssrdc"
    else:
        df["GHI_clearsky"]    = cs["ghi"].values
        df["clearsky_source"] = "pvlib_ineichen"

    if "GHI" in df.columns:
        df["CSI"] = np.where(
            df["GHI_clearsky"] > 10,
            (df["GHI"] / df["GHI_clearsky"]).clip(0, 1.2), 0)
        cos_z = np.cos(np.deg2rad(df["SZA"].clip(0, 89.9)))
        if "avg_sdirswrf" in df.columns:
            df["DNI"] = df["avg_sdirswrf"].clip(0, 1400)
        else:
            df["DNI"] = np.where(cos_z > 0.05,
                                 df["GHI"] / cos_z, 0).clip(0, 1400)
        df["DHI"] = (df["GHI"] - df["DNI"] * cos_z).clip(0)
    return df


def apply_unit_conversions(df):
    if "t2m" in df.columns:
        df["T_amb"] = kelvin_to_c(df["t2m"])
    if "d2m" in df.columns:
        df["T_dew"] = kelvin_to_c(df["d2m"])
    if "T_amb" in df.columns and "T_dew" in df.columns:
        df["RHum"] = compute_rh(df["T_amb"], df["T_dew"])
    if "u10" in df.columns and "v10" in df.columns:
        u = df["u10"].astype(float)
        v = df["v10"].astype(float)
        df["W_spd"] = np.sqrt(u**2 + v**2)
        df["W_dir"] = (np.degrees(np.arctan2(u, v)) + 360) % 360
    if "sp" in df.columns:
        df["P_atm"] = df["sp"].astype(float) / 100.0
    if "tcc" in df.columns:
        df["cloud_cover"] = df["tcc"].astype(float)

    ssrd_col  = next((c for c in df.columns if c == "ssrd"),  None)
    ssrdc_col = next((c for c in df.columns if c == "ssrdc"), None)
    fdir_col  = next((c for c in df.columns
                      if c in ("msdwswrf", "fdir", "msdrswrf")), None)
    strd_col  = next((c for c in df.columns if c == "strd"),  None)
    tp_col    = next((c for c in df.columns if c == "tp"),    None)

    if ssrd_col:
        df["GHI"]             = (deaccumulate(df[ssrd_col].astype(float))  / 3600).clip(0)
    if ssrdc_col:
        # Phase 1 spec: ssrdc is the ERA5 clear-sky equivalent of ssrd.
        # Same deaccumulation logic as ssrd (accumulated J/m² ÷ 3600 → W/m²).
        df["GHI_clearsky_era5"] = (deaccumulate(df[ssrdc_col].astype(float)) / 3600).clip(0)
    if fdir_col:
        df["avg_sdirswrf"]    = df[fdir_col].astype(float).clip(0)
    if strd_col:
        df["LW_down"]         = (deaccumulate(df[strd_col].astype(float))  / 3600).clip(0)
    if tp_col:
        df["precipitation"]   = (deaccumulate(df[tp_col].astype(float))    * 1000).clip(0)

    if "GHI" in df.columns:
        df.loc[df["GHI"] < 0,    "GHI"] = 0
        df.loc[df["GHI"] > 1400, "GHI"] = np.nan
    if "T_amb" in df.columns:
        df.loc[df["T_amb"] < -10, "T_amb"] = np.nan
        df.loc[df["T_amb"] > 60,  "T_amb"] = np.nan
    if "RHum" in df.columns:
        df["RHum"] = df["RHum"].clip(0, 100)

    return df


# ===========================================================
# LOAD ALL ERA5 NETCDF FILES
# ===========================================================

def load_all_files():
    print("  Loading NetCDF files into memory ...")
    instant_ds, accum_ds = {}, {}

    for year in YEARS:
        instant_ds[year] = {}
        accum_ds[year]   = {}
        for month in MONTHS:
            fi = os.path.join(INPUT_DIR, f"era5_AS_points_{year}_{month}_instant.nc")
            fa = os.path.join(INPUT_DIR, f"era5_AS_points_{year}_{month}_accum.nc")
            if os.path.exists(fi):
                ds = open_nc(fi)
                if ds is not None:
                    instant_ds[year][month] = ds.load()
            if os.path.exists(fa):
                ds = open_nc(fa)
                if ds is not None:
                    accum_ds[year][month] = ds.load()

    ni = sum(len(v) for v in instant_ds.values())
    na = sum(len(v) for v in accum_ds.values())
    expected = len(YEARS) * len(MONTHS)
    print(f"  Opened: instant={ni}/{expected}  accum={na}/{expected}")
    return instant_ds, accum_ds


# ===========================================================
# NASA POWER CACHE LOADING
# ===========================================================

def load_power_series(point_id):
    frames = []
    for year in YEARS:
        fp = os.path.join(POWER_DIR, f"power_{point_id}_{year}.json")
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        params = data.get("properties", {}).get("parameter", {})
        if not params:
            continue

        cols, idx = {}, None
        for var, series in params.items():
            if idx is None:
                idx = pd.to_datetime(list(series.keys()), format="%Y%m%d%H", utc=True)
            cols[var] = list(series.values())
        frames.append(pd.DataFrame(cols, index=idx))

    if not frames:
        return None
    power_df = pd.concat(frames).sort_index()
    power_df = power_df[~power_df.index.duplicated(keep="first")]
    power_df = power_df.replace(-999, np.nan)
    return power_df


# ===========================================================
# NEAREST-HOUR MATCHING
# ===========================================================

def nearest_row(series_df, target_time, max_hours=MAX_MATCH_HOURS):
    if series_df is None or series_df.empty:
        return None
    pos = series_df.index.get_indexer([target_time], method="nearest")[0]
    if pos == -1:
        return None
    nearest_time = series_df.index[pos]
    if abs((nearest_time - target_time).total_seconds()) > max_hours * 3600:
        return None
    return series_df.iloc[pos]


# ===========================================================
# PROCESS ONE POINT
# ===========================================================

def process_point_era5(lat, lon, instant_ds, accum_ds):
    fi_frames, fa_frames = [], []
    grid_lat_used = grid_lon_used = None

    for year in YEARS:
        for month in MONTHS:
            if month in instant_ds.get(year, {}):
                df_i = extract_nearest(instant_ds[year][month], lat, lon)
                if df_i is not None and len(df_i) > 0:
                    if grid_lat_used is None:
                        grid_lat_used = df_i.attrs.get("grid_lat")
                        grid_lon_used = df_i.attrs.get("grid_lon")
                    fi_frames.append(df_i)
            if month in accum_ds.get(year, {}):
                df_a = extract_nearest(accum_ds[year][month], lat, lon)
                if df_a is not None and len(df_a) > 0:
                    fa_frames.append(df_a)

    if not fi_frames and not fa_frames:
        return None, None, None

    df_i = pd.concat(fi_frames).sort_index() if fi_frames else pd.DataFrame()
    df_a = pd.concat(fa_frames).sort_index() if fa_frames else pd.DataFrame()
    df_i = df_i[~df_i.index.duplicated(keep="first")] if not df_i.empty else df_i
    df_a = df_a[~df_a.index.duplicated(keep="first")] if not df_a.empty else df_a

    if df_i.empty:
        df = df_a.copy()
    elif df_a.empty:
        df = df_i.copy()
    else:
        df = df_i.join(df_a, how="outer", lsuffix="", rsuffix="_a")
        df.drop(columns=[c for c in df.columns if c.endswith("_a")],
                errors="ignore", inplace=True)

    df = df[~df.index.duplicated(keep="first")]
    df = apply_unit_conversions(df)
    df = compute_solar(df, lat, lon, alt=DEFAULT_ALT_M)

    df.index = df.index.tz_localize("UTC")
    return df, grid_lat_used, grid_lon_used


def process_point(point_row, instant_ds, accum_ds, sun_df):
    point_id = point_row.point_id
    era5_df, grid_lat, grid_lon = process_point_era5(
        point_row.lat, point_row.lon, instant_ds, accum_ds)
    power_df = load_power_series(point_id)

    if era5_df is None and power_df is None:
        return None

    point_sun = sun_df[sun_df["point_id"] == point_id]
    rows = []
    for r in point_sun.itertuples(index=False):
        target = r.time_utc
        era5_row = nearest_row(era5_df, target)
        power_row = nearest_row(power_df, target)
        if era5_row is None and power_row is None:
            continue

        rec = {
            "point_id": point_id,
            "lat": point_row.lat, "lon": point_row.lon,
            "population": point_row.population, "weight": point_row.weight,
            "date": r.date, "event": r.event, "time_utc": target,
            "grid_lat": grid_lat, "grid_lon": grid_lon,
        }
        if era5_row is not None:
            for col in ERA5_OUTPUT_VARS:
                if col in era5_row.index:
                    rec[f"era5_{col}"] = era5_row[col]
        if power_row is not None:
            for col in POWER_VARS:
                if col in power_row.index:
                    rec[f"power_{col}"] = power_row[col]
        rows.append(rec)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    dt = pd.to_datetime(df["date"])
    df["month"]       = dt.dt.month
    df["DOY"]         = dt.dt.dayofyear
    df["year"]        = dt.dt.year
    df["season"]      = df["month"].map(lambda m: SEASON_MAP[m][0])
    df["season_code"] = df["month"].map(lambda m: SEASON_MAP[m][1])
    return df


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    print("=" * 68)
    print("  ERA5 + NASA POWER COMBINE — ASSAM POPULATION POINTS")
    print(f"  ERA5 input  : {INPUT_DIR}/")
    print(f"  POWER input : {POWER_DIR}/")
    print(f"  Output      : {COMBINED_POINTS_FILE}")
    print("=" * 68)

    if not POPULATION_GRID_FILE.exists():
        print(f"\n  ERROR: {POPULATION_GRID_FILE} not found — "
              "run 00a_build_population_grid.py first.")
        raise SystemExit(1)
    if not SUNTIMES_FILE.exists():
        print(f"\n  ERROR: {SUNTIMES_FILE} not found — "
              "run 00b_build_suntimes.py first.")
        raise SystemExit(1)

    points_df = pd.read_csv(POPULATION_GRID_FILE)
    sun_df = pd.read_csv(SUNTIMES_FILE)
    sun_df["time_utc"] = pd.to_datetime(sun_df["time_utc"], utc=True)

    instant_count = sum(
        1 for y in YEARS for m in MONTHS
        if os.path.exists(os.path.join(INPUT_DIR, f"era5_AS_points_{y}_{m}_instant.nc")))
    accum_count = sum(
        1 for y in YEARS for m in MONTHS
        if os.path.exists(os.path.join(INPUT_DIR, f"era5_AS_points_{y}_{m}_accum.nc")))
    expected = len(YEARS) * len(MONTHS)
    power_count = sum(
        1 for pid in points_df["point_id"] for y in YEARS
        if os.path.exists(os.path.join(POWER_DIR, f"power_{pid}_{y}.json")))
    power_expected = len(points_df) * len(YEARS)

    print(f"\n  Points        : {len(points_df)}")
    print(f"  ERA5 files    : instant={instant_count}/{expected}  accum={accum_count}/{expected}")
    print(f"  NASA POWER    : {power_count}/{power_expected} point-year caches")

    if instant_count == 0 and accum_count == 0:
        print("\n  ⚠️  No ERA5 files found — run 01_download_era5_assam.py first.")
    if power_count == 0:
        print("\n  ⚠️  No NASA POWER cache found — run 01b_download_nasapower.py first.")
        print("     Continuing with ERA5-only output ...")

    print("\n  Loading ERA5 NetCDF files ...")
    instant_ds, accum_ds = load_all_files()

    print(f"\n  Processing {len(points_df)} points ...")
    print("-" * 68)

    total_rows = 0
    points_ok = 0
    era5_hits = 0
    power_hits = 0
    header_written = False

    with open(COMBINED_POINTS_FILE, "w", newline="", encoding="utf-8") as fout:
        for i, point_row in enumerate(points_df.itertuples(index=False), start=1):
            df = process_point(point_row, instant_ds, accum_ds, sun_df)
            if df is None:
                print(f"  [{i}/{len(points_df)}] {point_row.point_id}  [SKIP] no data")
                continue

            era5_hits += df.filter(like="era5_").notna().any(axis=1).sum()
            power_hits += df.filter(like="power_").notna().any(axis=1).sum()

            df.to_csv(fout, index=False, header=not header_written)
            header_written = True
            total_rows += len(df)
            points_ok += 1

            if i % 10 == 0 or i == len(points_df):
                print(f"  [{i}/{len(points_df)}] {point_row.point_id}  "
                      f"rows_so_far={total_rows:,}")

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Points processed : {points_ok}/{len(points_df)}")
    print(f"  Rows written     : {total_rows:,}")
    if total_rows:
        print(f"  Rows with ERA5 data  : {era5_hits:,}  ({100*era5_hits/total_rows:.1f}%)")
        print(f"  Rows with POWER data : {power_hits:,}  ({100*power_hits/total_rows:.1f}%)")
    print(f"  Output           : {COMBINED_POINTS_FILE}")
    print("=" * 68)
    print("\nNext: load climate_assam_points.csv for training / cross-checking.")
