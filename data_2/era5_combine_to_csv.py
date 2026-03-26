import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import re
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: unzip if needed
# ─────────────────────────────────────────────────────────────────────────────

def unzip_if_needed(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4)
    if header[:2] != b'PK':
        return [filepath]                             # already a real NC file

    print(f"  ⚠️  {filepath} is a ZIP — extracting all .nc inside ...")
    zip_path = filepath.replace('.nc', '_tmp.zip')
    os.rename(filepath, zip_path)

    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        nc_files_inside = [n for n in z.namelist() if n.endswith('.nc')]
        if not nc_files_inside:
            print(f"  ❌ No .nc found inside {zip_path}.")
            os.rename(zip_path, filepath)
            return []
        for nc_name in nc_files_inside:
            z.extract(nc_name, '.')
            basename = os.path.splitext(os.path.basename(nc_name))[0]
            dest = filepath.replace('.nc', f'__{basename}.nc')
            os.rename(nc_name, dest)
            extracted.append(dest)
            print(f"  ✅ Unzipped → {dest}")

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
        else:
            print(f"  ❌ Cannot open {filepath}")
            return None

    ds = xr.open_dataset(filepath, engine=engine)

    # Drop degenerate scalar dimensions
    for dim in ['number', 'surface', 'step']:
        if dim in ds.dims and ds.dims[dim] == 1:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim, errors='ignore')

    # Handle expver split (take first non-null)
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
# HELPER: load ALL nc parts for one month → merged Dataset → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def load_month_files(nc_paths):
    """
    ERA5 CDS sometimes splits one month into two NC files:
      - stepType-instant  → t2m, d2m, u10, v10, tcc
      - stepType-accum    → ssrd (solar_radiation), tp
    This function opens all parts and merges them into one Dataset.
    """
    datasets = []
    for p in nc_paths:
        ds = open_nc(p)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        return None

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        # Merge on shared dimensions (valid_time, latitude, longitude)
        try:
            merged = xr.merge(datasets, join='inner', compat='override')
            print(f"    Merged {len(datasets)} files successfully.")
        except Exception as e:
            print(f"    ⚠️  xr.merge failed ({e}), trying concat fallback ...")
            # Fallback: convert each to DataFrame and pandas-merge
            dfs = []
            for ds in datasets:
                spatial = [d for d in ds.dims if d in ('latitude', 'longitude', 'lat', 'lon')]
                if spatial:
                    ds = ds.mean(dim=spatial)
                df_ = ds.to_dataframe().reset_index()
                dfs.append(df_)
            merged_df = dfs[0]
            for df_ in dfs[1:]:
                time_col = next((c for c in df_.columns if 'time' in c.lower()), None)
                merged_df = pd.merge(merged_df, df_, on=time_col, how='outer',
                                     suffixes=('', '_dup'))
                dup_cols = [c for c in merged_df.columns if c.endswith('_dup')]
                merged_df.drop(columns=dup_cols, inplace=True)
            return merged_df   # return early — already a DataFrame

    # ── Flatten Dataset to DataFrame ─────────────────────────────────────────
    spatial_dims = [d for d in merged.dims if d in ('latitude', 'longitude', 'lat', 'lon')]
    if spatial_dims:
        # Average over any spatial spread in the small bounding box
        ds_spatial_mean = merged.mean(dim=spatial_dims, keep_attrs=True)
        # Also compute the representative lat/lon (mean of grid points)
        coords_dict = {}
        for sd in spatial_dims:
            coords_dict[sd] = float(merged[sd].mean())
        df = ds_spatial_mean.to_dataframe().reset_index()
        for sd, val in coords_dict.items():
            df[sd] = val
    else:
        df = merged.to_dataframe().reset_index()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build final columns from a raw merged DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_final_df(df, source_label=""):
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    print(f"  Columns available: {list(df.columns)}")

    # ── Timestamp ────────────────────────────────────────────────────────────
    for tcol in ['valid_time', 'time']:
        if tcol in df.columns:
            df['timestamp'] = pd.to_datetime(df[tcol])
            break
    else:
        print(f"  ❌ No time column found {source_label}")
        return None

    # ── GHI (W/m²) ───────────────────────────────────────────────────────────
    if 'solar_radiation' in df.columns:
        df['GHI_Wm2'] = df['solar_radiation'].clip(lower=0)
        print("  ℹ️  GHI ← solar_radiation (W/m²)")
    elif 'ssrd' in df.columns:
        # ssrd is accumulated J/m² — deaccumulate then convert to W/m²
        df = df.sort_values('valid_time' if 'valid_time' in df.columns else 'timestamp')
        df['ssrd'] = df['ssrd'].diff().clip(lower=0)
        df['GHI_Wm2'] = (df['ssrd'] / 3600.0).clip(lower=0)
        print("  ℹ️  GHI ← ssrd deaccumulated ÷ 3600")
    elif 'msdwswrf' in df.columns:
        df['GHI_Wm2'] = df['msdwswrf'].clip(lower=0)
        print("  ℹ️  GHI ← msdwswrf")
    else:
        print(f"  ⚠️  GHI variable not found {source_label}")
        df['GHI_Wm2'] = np.nan

    # ── Temperature (K → °C) ─────────────────────────────────────────────────
    if 't2m' in df.columns:
        df['T_ambient_C'] = df['t2m'] - 273.15
    else:
        print(f"  ⚠️  t2m not found {source_label}")
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
        print(f"  ⚠️  tp not found {source_label}")
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

    # ── Select final columns ─────────────────────────────────────────────────
    geo_cols  = [c for c in ['latitude', 'longitude'] if c in df.columns]
    data_cols = ['GHI_Wm2', 'T_ambient_C', 'Wind_speed_ms',
                 'Precip_mm', 'Cloud_cover_fraction', 'RH_percent']

    return df[['timestamp'] + geo_cols + data_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("ERA5 → CSV CONVERTER  (handles split instant+accum ZIP files)")
print("=" * 65)

# ── Group files by month ──────────────────────────────────────────────────────
# Matches both:
#   era5_2024_01.nc                          (single-file month)
#   era5_2024_01__data_stream-oper_*.nc      (split-file month from ZIP)

all_nc = sorted(glob.glob(os.path.join(SCRIPT_DIR, "era5_2024_*.nc")))
if not all_nc:
    print("❌ No era5_2024_*.nc files found.")
    print(f"   Script directory: {SCRIPT_DIR}")
    exit()

# Build month → [list of files] mapping
from collections import defaultdict
month_files = defaultdict(list)

for f in all_nc:
    # extract the MM part robustly via regex
    m = re.match(r'era5_\d{4}_(\d{2})', os.path.basename(f))
    if m:
        month_files[m.group(1)].append(f)

print(f"Found files for {len(month_files)} months:")
for mm in sorted(month_files):
    print(f"  Month {mm}: {month_files[mm]}")
print()

all_dfs = []
failed  = []

for mm in sorted(month_files.keys()):
    paths = month_files[mm]
    print(f"\n📂 Processing month {mm} ({len(paths)} file(s)): {paths}")

    # Unzip any ZIPs that are still ZIPs
    real_paths = []
    for p in paths:
        result = unzip_if_needed(p)
        real_paths.extend(result)

    if not real_paths:
        failed.append(mm)
        continue

    try:
        raw_df = load_month_files(real_paths)
        if raw_df is None:
            failed.append(mm)
            continue

        df = build_final_df(raw_df, source_label=f"(month {mm})")
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f"  ✅ {len(df):,} rows extracted for month {mm}")
        else:
            failed.append(mm)
    except Exception as e:
        print(f"  ❌ Error processing month {mm}: {e}")
        import traceback; traceback.print_exc()
        failed.append(mm)

# ── Merge all months ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MERGING ALL MONTHS")
print("=" * 65)

if not all_dfs:
    print("❌ No data to merge.")
else:
    climate_df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate and sort
    dedup_cols = ['timestamp']
    if 'latitude'  in climate_df.columns: dedup_cols.append('latitude')
    if 'longitude' in climate_df.columns: dedup_cols.append('longitude')

    climate_df = (
        climate_df
        .drop_duplicates(subset=dedup_cols)
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    climate_df['GHI_Wm2']   = climate_df['GHI_Wm2'].clip(lower=0)
    climate_df['Precip_mm'] = climate_df['Precip_mm'].clip(lower=0)

    output = os.path.join(SCRIPT_DIR, 'era5_climate_coimbatore_2024.csv')
    climate_df.to_csv(output, index=False)

    # NaN audit
    nan_summary = climate_df.isna().sum()
    print(f"\n🎉 SUCCESS!")
    print(f"   File       : {output}")
    print(f"   Rows       : {len(climate_df):,}  (expect ~8,784 for leap year 2024)")
    print(f"   Date range : {climate_df['timestamp'].min()}  →  {climate_df['timestamp'].max()}")
    print(f"   Columns    : {list(climate_df.columns)}")
    print(f"\nNaN audit:")
    print(nan_summary.to_string())
    print(f"\n{'─'*65}")
    print("SAMPLE (first 5 rows):")
    print(climate_df.head().to_string(index=False))
    print(f"\n{'─'*65}")
    print("STATISTICS:")
    print(climate_df.drop(columns='timestamp').describe().round(2).to_string())

if failed:
    print(f"\n⚠️  Failed months: {failed}")
