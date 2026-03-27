import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import re
from collections import defaultdict

# Merges DNI, DHI, surface_pressure from *_extra*.nc files into the existing CSV.
# Run after:  python .\era5_download_extra.py

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXISTING_CSV = os.path.join(SCRIPT_DIR, 'era5_climate_tamilnadu_2024.csv')
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, 'era5_climate_tamilnadu_2024.csv')  # overwrite in-place

print("=" * 65)
print("MERGING EXTRA VARIABLES INTO EXISTING CSV")
print("=" * 65)

# ── Load existing CSV ──────────────────────────────────────────────────────────
if not os.path.exists(EXISTING_CSV):
    print(f"❌ Existing CSV not found: {EXISTING_CSV}")
    exit()

print(f"\n📂 Loading existing CSV: {EXISTING_CSV}")
climate_df = pd.read_csv(EXISTING_CSV, parse_dates=['timestamp'])
print(f"   Shape: {climate_df.shape}  |  Columns: {list(climate_df.columns)}")

# ── Find _extra*.nc files and group by month ────────────────────────────────
extra_nc = sorted(glob.glob(os.path.join(SCRIPT_DIR, 'era5_2024_*_extra*.nc')))
if not extra_nc:
    print("❌ No era5_2024_*_extra*.nc files found. Run era5_download_extra.py first.")
    exit()

month_files = defaultdict(list)
for f in extra_nc:
    m = re.match(r'era5_(\d{4})_(\d{2})_extra', os.path.basename(f))
    if m:
        month_files[m.group(2)].append(f)

print(f"\nFound extra files for {len(month_files)} months:")
for mm in sorted(month_files):
    print(f"  Month {mm}: {[os.path.basename(p) for p in month_files[mm]]}")

# ── Helper: open and clean a NetCDF file ──────────────────────────────────────
def open_nc(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4)
    engine = 'netcdf4' if header[:4] == b'\x89HDF' else 'scipy' if header[:3] == b'CDF' else None
    if engine is None:
        for eng in ['netcdf4', 'scipy', 'h5netcdf']:
            try:
                ds = xr.open_dataset(filepath, engine=eng); engine = eng; break
            except Exception: continue
        else:
            print(f"  ❌ Cannot open {filepath}"); return None
    ds = xr.open_dataset(filepath, engine=engine)
    for dim in ['number', 'surface', 'step']:
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim, errors='ignore')
    if 'expver' in ds.dims:
        ds = ds.sel(expver=ds.expver[0], drop=True)
    for tname in ['time', 'valid_time']:
        if tname in ds.coords and tname != 'valid_time':
            ds = ds.rename({tname: 'valid_time'}); break
    print(f"    [{os.path.basename(filepath)}] vars={list(ds.data_vars)}")
    return ds

# ── Process each month's extra files ──────────────────────────────────────────
extra_dfs = []

for mm in sorted(month_files.keys()):
    paths = month_files[mm]
    print(f"\n📂 Processing month {mm} ({len(paths)} file(s))")

    datasets = [ds for ds in (open_nc(p) for p in paths) if ds is not None]
    if not datasets:
        print(f"  ❌ No datasets loaded for month {mm}"); continue

    try:
        merged = xr.merge(datasets, join='inner', compat='override') if len(datasets) > 1 else datasets[0]
        if len(datasets) > 1:
            print(f"    Merged {len(datasets)} files.")
    except Exception as e:
        print(f"  ❌ Merge failed: {e}"); continue

    df = merged.to_dataframe().reset_index()
    merged.close()
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    # Timestamp
    for tcol in ['valid_time', 'time']:
        if tcol in df.columns:
            df['timestamp'] = pd.to_datetime(df[tcol]); break

    df = df.sort_values(['latitude', 'longitude', 'timestamp'])

    # fdir = total_sky_direct_solar_radiation_at_surface (accumulated J/m²)
    # → deaccumulate per grid point → convert to W/m² (÷3600)
    # This gives Direct Horizontal Irradiance (DHI_direct = fdir_Wm2)
    # DHI (diffuse) = GHI - fdir_Wm2  (computed in merge step using existing GHI)
    if 'fdir' in df.columns:
        df['fdir'] = df.groupby(['latitude', 'longitude'])['fdir'].diff().clip(lower=0)
        df['fdir_Wm2'] = (df['fdir'] / 3600.0).clip(lower=0)
        print("  ℹ️  fdir (direct horizontal) deaccumulated ÷ 3600")
    else:
        df['fdir_Wm2'] = np.nan
        print("  ⚠️  fdir not found")

    # Surface pressure (instantaneous — no deaccumulation needed)
    if 'sp' in df.columns:
        df['surface_pressure_Pa'] = df['sp']
        print("  ℹ️  surface_pressure_Pa ← sp")
    else:
        df['surface_pressure_Pa'] = np.nan
        print("  ⚠️  sp not found")

    keep = ['timestamp', 'latitude', 'longitude', 'fdir_Wm2', 'surface_pressure_Pa']
    extra_dfs.append(df[[c for c in keep if c in df.columns]])
    print(f"  ✅ {len(df):,} rows for month {mm}")

if not extra_dfs:
    print("❌ No extra data extracted."); exit()

# ── Combine all extra months ───────────────────────────────────────────────────
extra_combined = pd.concat(extra_dfs, ignore_index=True)
extra_combined['fdir_Wm2'] = extra_combined['fdir_Wm2'].fillna(0).clip(lower=0)

print(f"\n✅ Extra data combined: {extra_combined.shape}")

# ── Merge with existing CSV ────────────────────────────────────────────────────
print("\n🔗 Merging into existing CSV ...")

# Round lat/lon to avoid float precision mismatches
for df_ in [climate_df, extra_combined]:
    df_['latitude']  = df_['latitude'].round(4)
    df_['longitude'] = df_['longitude'].round(4)

merged_df = climate_df.merge(
    extra_combined,
    on=['timestamp', 'latitude', 'longitude'],
    how='left'
)

# Derive DNI and DHI from fdir and existing GHI
# fdir = direct radiation on horizontal surface (W/m²)
# DHI (diffuse) = GHI - fdir
# DNI proxy     = fdir (stored as direct horizontal; true DNI needs cos(SZA) via pvlib)
if 'fdir_Wm2' in merged_df.columns and 'GHI_Wm2' in merged_df.columns:
    merged_df['DHI_Wm2'] = (merged_df['GHI_Wm2'] - merged_df['fdir_Wm2']).clip(lower=0)
    merged_df.rename(columns={'fdir_Wm2': 'DNI_Wm2'}, inplace=True)
    print("  ✅ DNI_Wm2 = fdir_Wm2 (direct horizontal)")
    print("  ✅ DHI_Wm2 = GHI - fdir (diffuse horizontal)")

# Reorder columns
desired_order = [
    'timestamp', 'latitude', 'longitude',
    'GHI_Wm2', 'DNI_Wm2', 'DHI_Wm2',
    'T_ambient_C', 'Wind_speed_ms', 'Precip_mm',
    'Cloud_cover_fraction', 'RH_percent', 'surface_pressure_Pa'
]
final_cols = [c for c in desired_order if c in merged_df.columns]
merged_df = merged_df[final_cols]

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n💾 Writing to {OUTPUT_CSV} ...")
merged_df.to_csv(OUTPUT_CSV, index=False)

nan_summary = merged_df.isna().sum()
match_rate  = merged_df['DNI_Wm2'].notna().mean() * 100 if 'DNI_Wm2' in merged_df.columns else 0

print(f"\n🎉 Done!")
print(f"   Rows    : {len(merged_df):,}")
print(f"   Columns : {list(merged_df.columns)}")
print(f"   DNI/DHI match rate: {match_rate:.1f}% (should be ~100%)")
print(f"\nNaN audit:\n{nan_summary.to_string()}")
print(f"\nSample:\n{merged_df.head(3).to_string(index=False)}")
