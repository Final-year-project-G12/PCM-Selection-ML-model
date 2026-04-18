import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import re
import zipfile
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: unzip if needed
# ─────────────────────────────────────────────────────────────────────────────

def unzip_if_needed(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4)
    if header[:2] != b'PK':
        return [filepath]                             # already a real NC file

    print(f"  [WARNING] {filepath} is a ZIP -- extracting all .nc inside ...")
    zip_path = filepath.replace('.nc', '_tmp.zip')
    os.rename(filepath, zip_path)

    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        nc_files_inside = [n for n in z.namelist() if n.endswith('.nc')]
        if not nc_files_inside:
            print(f"  [ERROR] No .nc found inside {zip_path}.")
            os.rename(zip_path, filepath)
            return []
        for nc_name in nc_files_inside:
            z.extract(nc_name, SCRIPT_DIR)
            basename = os.path.splitext(os.path.basename(nc_name))[0]
            dest = filepath.replace('.nc', f'__{basename}.nc')
            os.rename(os.path.join(SCRIPT_DIR, nc_name), dest)
            extracted.append(dest)
            print(f"  [OK] Unzipped -> {dest}")

    os.remove(zip_path)
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: open ONE NC file → xarray Dataset (cleaned up)
# ─────────────────────────────────────────────────────────────────────────────

def open_nc(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4)

    if header[:4] == b'\x89HDF':
        engine = 'netcdf4'
    elif header[:3] == b'CDF':
        engine = 'scipy'
    else:
        for eng in ['netcdf4', 'scipy', 'h5netcdf']:
            try:
                return xr.open_dataset(filepath, engine=eng)
            except Exception:
                continue
        print(f"  [ERROR] Cannot open {filepath}")
        return None

    ds = xr.open_dataset(filepath, engine=engine)

    # Drop degenerate scalar dimensions
    for dim in ['number', 'surface', 'step']:
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim, errors='ignore')

    # Handle expver split (take first)
    if 'expver' in ds.dims:
        ds = ds.sel(expver=ds.expver[0], drop=True)

    # Normalise time coord name → 'valid_time'
    for tname in ['time', 'valid_time']:
        if tname in ds.coords and tname != 'valid_time':
            ds = ds.rename({tname: 'valid_time'})
            break

    print(f"    [{os.path.basename(filepath)}] vars={list(ds.data_vars)}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load ALL nc parts for one month → merged Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_month_datasets(nc_paths):
    """
    ERA5 CDS sometimes splits one month into two NC files:
      - stepType-instant  → t2m, d2m, u10, v10, tcc
      - stepType-accum    → ssrd, tp
    Opens all parts and merges them into one Dataset.
    """
    datasets = []
    for p in nc_paths:
        ds = open_nc(p)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]

    try:
        merged = xr.merge(datasets, join='inner', compat='override')
        print(f"    Merged {len(datasets)} files successfully.")
        return merged
    except Exception as e:
        print(f"    ❌ xr.merge failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Dataset → DataFrame with all variables, ALL grid points preserved
# ─────────────────────────────────────────────────────────────────────────────

def dataset_to_df(ds, month_label=""):
    """
    Converts the merged xarray Dataset to a flat DataFrame.
    ALL (latitude, longitude) grid points are kept — no spatial averaging.
    This preserves Tamil Nadu's spatial variation across ~391 grid points.
    """
    # Flatten to DataFrame — index will be (valid_time, latitude, longitude)
    df = ds.to_dataframe().reset_index()
    ds.close()

    # Rename lat/lon aliases
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    print(f"  Columns available: {list(df.columns)}")
    print(f"  Grid points: {df[['latitude','longitude']].drop_duplicates().shape[0]}")

    # ── Timestamp ────────────────────────────────────────────────────────────
    for tcol in ['valid_time', 'time']:
        if tcol in df.columns:
            df['timestamp'] = pd.to_datetime(df[tcol])
            break
    else:
        print(f"  [ERROR] No time column found {month_label}")
        return None

    # ── GHI (W/m²) — deaccumulate ssrd ──────────────────────────────────────
    if 'solar_radiation' in df.columns:
        df['GHI_Wm2'] = df['solar_radiation'].clip(lower=0)
        print("  ℹ️  GHI ← solar_radiation (W/m²)")
    elif 'ssrd' in df.columns:
        # ssrd is accumulated J/m² — deaccumulate per grid point then convert
        df = df.sort_values(['latitude', 'longitude', 'timestamp'])
        df['ssrd'] = df.groupby(['latitude', 'longitude'])['ssrd'].diff().clip(lower=0)
        df['GHI_Wm2'] = (df['ssrd'] / 3600.0).clip(lower=0)
        print("  ℹ️  GHI ← ssrd deaccumulated per grid point ÷ 3600")
    elif 'msdwswrf' in df.columns:
        df['GHI_Wm2'] = df['msdwswrf'].clip(lower=0)
        print("  ℹ️  GHI ← msdwswrf")
    else:
        print(f"  [WARNING] GHI variable not found {month_label}")
        df['GHI_Wm2'] = np.nan

    # ── Temperature (K → °C) ─────────────────────────────────────────────────
    if 't2m' in df.columns:
        df['T_ambient_C'] = df['t2m'] - 273.15
    else:
        print(f"  [WARNING] t2m not found {month_label}")
        df['T_ambient_C'] = np.nan

    # ── Wind Speed (m/s) ─────────────────────────────────────────────────────
    if 'u10' in df.columns and 'v10' in df.columns:
        df['Wind_speed_ms'] = np.sqrt(df['u10']**2 + df['v10']**2)
    else:
        df['Wind_speed_ms'] = np.nan

    # ── Precipitation (m → mm) ───────────────────────────────────────────────
    if 'tp' in df.columns:
        df['Precip_mm'] = (df['tp'] * 1000.0).clip(lower=0)
        print("  ℹ️  Precip ← tp")
    else:
        print(f"  [WARNING] tp not found {month_label}")
        df['Precip_mm'] = np.nan

    # ── Cloud Cover (0–1) ────────────────────────────────────────────────────
    if 'tcc' in df.columns:
        df['Cloud_cover_fraction'] = df['tcc'].clip(0, 1)
    else:
        df['Cloud_cover_fraction'] = np.nan

    # ── Relative Humidity (Magnus formula) ───────────────────────────────────
    if 'd2m' in df.columns and 't2m' in df.columns:
        T  = df['t2m'] - 273.15
        Td = df['d2m'] - 273.15
        df['RH_percent'] = (
            100 * np.exp((17.625 * Td) / (243.04 + Td))
                / np.exp((17.625 * T)  / (243.04 + T))
        ).clip(0, 100)
    else:
        df['RH_percent'] = np.nan

    # ── DNI (W/m²) — deaccumulate fdir ───────────────────────────────────────
    if 'fdir' in df.columns:
        df = df.sort_values(['latitude', 'longitude', 'timestamp'])
        df['fdir'] = df.groupby(['latitude', 'longitude'])['fdir'].diff().clip(lower=0)
        df['DNI_Wm2'] = (df['fdir'] / 3600.0).clip(lower=0)
        print("  ℹ️  DNI ← fdir deaccumulated per grid point ÷ 3600")
    else:
        print(f"  [WARNING] fdir (DNI) not found {month_label}")
        df['DNI_Wm2'] = np.nan

    # ── DHI (W/m²) — deaccumulate fsdss ──────────────────────────────────────
    if 'fsdss' in df.columns:
        df = df.sort_values(['latitude', 'longitude', 'timestamp'])
        df['fsdss'] = df.groupby(['latitude', 'longitude'])['fsdss'].diff().clip(lower=0)
        df['DHI_Wm2'] = (df['fsdss'] / 3600.0).clip(lower=0)
        print("  ℹ️  DHI ← fsdss deaccumulated per grid point ÷ 3600")
    else:
        print(f"  [WARNING] fsdss (DHI) not found {month_label}")
        df['DHI_Wm2'] = np.nan

    # ── Surface Pressure (Pa) ─────────────────────────────────────────────────
    if 'sp' in df.columns:
        df['surface_pressure_Pa'] = df['sp']
        print("  ℹ️  surface_pressure_Pa ← sp")
    else:
        print(f"  [WARNING] sp (surface_pressure) not found {month_label}")
        df['surface_pressure_Pa'] = np.nan

    # ── Select final columns ─────────────────────────────────────────────────
    geo_cols  = [c for c in ['latitude', 'longitude'] if c in df.columns]
    data_cols = ['GHI_Wm2', 'DNI_Wm2', 'DHI_Wm2',
                 'T_ambient_C', 'Wind_speed_ms',
                 'Precip_mm', 'Cloud_cover_fraction', 'RH_percent',
                 'surface_pressure_Pa']

    return df[['timestamp'] + geo_cols + data_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("ERA5 → CSV CONVERTER  (Tamil Nadu — all grid points preserved)")
print("=" * 65)

# ── Group files by month ───────────────────────────────────────────────────
all_nc = sorted(glob.glob(os.path.join(SCRIPT_DIR, "era5_2024_*.nc")))
if not all_nc:
    print("❌ No era5_2024_*.nc files found.")
    print(f"   Script directory: {SCRIPT_DIR}")
    exit()

month_files = defaultdict(list)
for f in all_nc:
    m = re.match(r'era5_\d{4}_(\d{2})', os.path.basename(f))
    if m:
        month_files[m.group(1)].append(f)

print(f"Found files for {len(month_files)} months:")
for mm in sorted(month_files):
    print(f"  Month {mm}: {[os.path.basename(p) for p in month_files[mm]]}")
print()

all_dfs = []
failed  = []

for mm in sorted(month_files.keys()):
    paths = month_files[mm]
    print(f"\n[Processing] Processing month {mm} ({len(paths)} file(s))")

    # Unzip any ZIPs still in ZIP format
    real_paths = []
    for p in paths:
        real_paths.extend(unzip_if_needed(p))

    if not real_paths:
        failed.append(mm)
        continue

    try:
        ds = load_month_datasets(real_paths)
        if ds is None:
            failed.append(mm)
            continue

        df = dataset_to_df(ds, month_label=f"(month {mm})")
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            grid_pts = df[['latitude','longitude']].drop_duplicates().shape[0]
            print(f"  [OK] {len(df):,} rows extracted for month {mm}  ({grid_pts} grid points x {len(df)//grid_pts} hours)")
        else:
            failed.append(mm)
    except Exception as e:
        print(f"  [ERROR] Error processing month {mm}: {e}")
        import traceback; traceback.print_exc()
        failed.append(mm)

# ── Merge all months ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MERGING ALL MONTHS")
print("=" * 65)

if not all_dfs:
    print("[ERROR] No data to merge.")
else:
    climate_df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate and sort
    climate_df = (
        climate_df
        .drop_duplicates(subset=['timestamp', 'latitude', 'longitude'])
        .sort_values(['timestamp', 'latitude', 'longitude'])
        .reset_index(drop=True)
    )

    # Fill first-row NaNs from deaccumulation (one NaN per grid point per month)
    climate_df['GHI_Wm2']   = climate_df['GHI_Wm2'].fillna(0).clip(lower=0)
    climate_df['DNI_Wm2']   = climate_df['DNI_Wm2'].fillna(0).clip(lower=0)
    climate_df['DHI_Wm2']   = climate_df['DHI_Wm2'].fillna(0).clip(lower=0)
    climate_df['Precip_mm'] = climate_df['Precip_mm'].clip(lower=0)

    output = os.path.join(SCRIPT_DIR, 'era5_climate_tamilnadu_2024.csv')
    print(f"\nWriting CSV (this may take a moment for large spatial datasets)...")
    climate_df.to_csv(output, index=False)

    # Summary
    grid_pts   = climate_df[['latitude','longitude']].drop_duplicates().shape[0]
    timestamps = climate_df['timestamp'].nunique()
    nan_summary = climate_df.isna().sum()

    print(f"\n[SUCCESS] SUCCESS!")
    print(f"   File        : {output}")
    print(f"   Total rows  : {len(climate_df):,}")
    print(f"   Grid points : {grid_pts}  (lat/lon cells across Tamil Nadu)")
    print(f"   Timestamps  : {timestamps}  (hourly 2024)")
    print(f"   Date range  : {climate_df['timestamp'].min()}  →  {climate_df['timestamp'].max()}")
    print(f"   Columns     : {list(climate_df.columns)}")
    print(f"   File size   : ~{os.path.getsize(output)/1e6:.1f} MB")
    print(f"\nNaN audit:")
    print(nan_summary.to_string())
    print(f"\n{'─'*65}")
    print("SAMPLE (first 5 rows):")
    print(climate_df.head().to_string(index=False))
    print(f"\n{'─'*65}")
    print("STATISTICS:")
    print(climate_df.drop(columns='timestamp').describe().round(2).to_string())

if failed:
    print(f"\n[WARNING] Failed months: {failed}")
