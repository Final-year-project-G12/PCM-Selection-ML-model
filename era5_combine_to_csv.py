import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import zipfile

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: unzip any files that are still ZIPs
# ─────────────────────────────────────────────────────────────────────────────

def unzip_if_needed(filepath):
    """If the .nc file is actually a ZIP, extract the real .nc from inside."""
    with open(filepath, 'rb') as f:
        header = f.read(4)

    if header[:2] != b'PK':
        return filepath                               # already a real NC file

    print(f"  ⚠️  {filepath} is a ZIP — extracting...")
    zip_path = filepath.replace('.nc', '_tmp.zip')
    os.rename(filepath, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        nc_files_inside = [n for n in z.namelist() if n.endswith('.nc')]
        if not nc_files_inside:
            print(f"  ❌ No .nc found inside {zip_path}. Skipping.")
            os.rename(zip_path, filepath)
            return None
        z.extract(nc_files_inside[0], '.')
        extracted = nc_files_inside[0]

    os.rename(extracted, filepath)
    os.remove(zip_path)
    print(f"  ✅ Unzipped → {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: open one NC file and return a clean DataFrame (with lat/lon)
# ─────────────────────────────────────────────────────────────────────────────

def nc_to_dataframe(filepath):
    """
    Opens an ERA5 NetCDF file and extracts all variables.
    Preserves latitude and longitude columns.
    Handles: solar_radiation, ssrd, msdwswrf for GHI.
    Handles: tp, expver-split files for precipitation.
    Drops degenerate dimensions: number, surface, step.
    """
    # ── Choose engine ────────────────────────────────────────────────────────
    with open(filepath, 'rb') as f:
        header = f.read(4)

    if header[:4] == b'\x89HDF':
        engine = 'netcdf4'
    elif header[:3] == b'CDF':
        engine = 'scipy'
    else:
        for eng in ['netcdf4', 'scipy', 'h5netcdf']:
            try:
                xr.open_dataset(filepath, engine=eng)
                engine = eng
                break
            except Exception:
                continue
        else:
            print(f"  ❌ Cannot open {filepath} with any engine.")
            return None

    ds = xr.open_dataset(filepath, engine=engine)
    print(f"  Dimensions  : {dict(ds.dims)}")
    print(f"  Variables   : {list(ds.data_vars)}")
    print(f"  Coordinates : {list(ds.coords)}")

    # ── Drop degenerate / unwanted dimensions ────────────────────────────────
    # 'number' = ensemble member (size 1), 'surface' = level (size 1), 'step' = forecast step
    for dim in ['number', 'surface', 'step', 'expver']:
        if dim in ds.dims and ds.dims[dim] == 1:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim, errors='ignore')

    # ── If multiple expver values exist, take the first (non-NaN preference) ─
    if 'expver' in ds.dims:
        # Try to merge by taking the first non-null along expver
        ds = ds.sel(expver=ds.expver[0], drop=True)

    # ── Flatten to DataFrame (keeping lat/lon as columns) ───────────────────
    df = ds.to_dataframe().reset_index()
    ds.close()

    print(f"  Columns in file: {list(df.columns)}")

    # ── Normalise column names (ERA5 uses both lat/latitude, lon/longitude) ──
    df = df.rename(columns={
        'lat': 'latitude',
        'lon': 'longitude',
        'valid_time': 'valid_time',   # keep as-is
    })

    # ── Timestamp ────────────────────────────────────────────────────────────
    for tcol in ['valid_time', 'time']:
        if tcol in df.columns:
            df['timestamp'] = pd.to_datetime(df[tcol])
            break
    else:
        print(f"  ❌ No time column in {filepath}")
        return None

    # ── GHI (W/m²) ───────────────────────────────────────────────────────────
    # Priority: solar_radiation > ssrd (J/m² ÷ 3600) > msdwswrf
    if 'solar_radiation' in df.columns:
        # CDS new API returns it already in W/m²
        df['GHI_Wm2'] = df['solar_radiation'].clip(lower=0)
        print("  ℹ️  GHI source: solar_radiation (W/m²)")
    elif 'ssrd' in df.columns:
        df['GHI_Wm2'] = (df['ssrd'] / 3600.0).clip(lower=0)
        print("  ℹ️  GHI source: ssrd ÷ 3600 (J/m² → W/m²)")
    elif 'msdwswrf' in df.columns:
        df['GHI_Wm2'] = df['msdwswrf'].clip(lower=0)
        print("  ℹ️  GHI source: msdwswrf (W/m²)")
    else:
        print(f"  ⚠️  GHI variable not found in {filepath}")
        df['GHI_Wm2'] = np.nan

    # ── Ambient Temperature: t2m (K → °C) ────────────────────────────────────
    if 't2m' in df.columns:
        df['T_ambient_C'] = df['t2m'] - 273.15
    else:
        print(f"  ⚠️  t2m not found in {filepath}")
        df['T_ambient_C'] = np.nan

    # ── Wind Speed: u10 + v10 → scalar (m/s) ─────────────────────────────────
    if 'u10' in df.columns and 'v10' in df.columns:
        df['Wind_speed_ms'] = np.sqrt(df['u10']**2 + df['v10']**2)
    else:
        df['Wind_speed_ms'] = np.nan

    # ── Precipitation: tp (m → mm) ────────────────────────────────────────────
    if 'tp' in df.columns:
        df['Precip_mm'] = (df['tp'] * 1000.0).clip(lower=0)
        print("  ℹ️  Precipitation source: tp")
    else:
        print(f"  ⚠️  tp (precipitation) not found in {filepath}")
        df['Precip_mm'] = np.nan

    # ── Cloud Cover: tcc (fraction 0–1) ──────────────────────────────────────
    if 'tcc' in df.columns:
        df['Cloud_cover_fraction'] = df['tcc'].clip(0, 1)
    else:
        df['Cloud_cover_fraction'] = np.nan

    # ── Relative Humidity from Dew Point (Magnus formula) ────────────────────
    if 'd2m' in df.columns and 't2m' in df.columns:
        T  = df['t2m'] - 273.15
        Td = df['d2m'] - 273.15
        df['RH_percent'] = (
            100 * np.exp((17.625 * Td) / (243.04 + Td))
                / np.exp((17.625 * T)  / (243.04 + T))
        ).clip(0, 100)
    else:
        df['RH_percent'] = np.nan

    # ── Select final columns (include lat/lon if present) ────────────────────
    base_cols   = ['timestamp', 'latitude', 'longitude']
    data_cols   = ['GHI_Wm2', 'T_ambient_C', 'Wind_speed_ms',
                   'Precip_mm', 'Cloud_cover_fraction', 'RH_percent']

    # Only keep lat/lon columns if they actually exist
    keep_geo = [c for c in base_cols if c in df.columns]
    final_cols = keep_geo + data_cols

    return df[final_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: process all monthly files → single CSV
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("ERA5 → CSV CONVERTER  (with lat/lon + solar_radiation)")
print("=" * 60)

nc_files = sorted(glob.glob("era5_2024_*.nc"))

if not nc_files:
    print(f"❌ No era5_2024_*.nc files found.")
    print(f"   Current directory: {os.getcwd()}")
    print("   Move this script to the same folder as your .nc files.")
    exit()

print(f"Found {len(nc_files)} files: {nc_files}\n")

all_dfs = []
failed  = []

for filepath in nc_files:
    print(f"\n📂 Processing: {filepath}")

    filepath = unzip_if_needed(filepath)
    if filepath is None:
        failed.append(filepath)
        continue

    try:
        df = nc_to_dataframe(filepath)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f"  ✅ {len(df):,} rows extracted")
        else:
            failed.append(filepath)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback; traceback.print_exc()
        failed.append(filepath)

# ── Merge all months ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MERGING ALL MONTHS")
print("=" * 60)

if not all_dfs:
    print("❌ No data to merge. Check errors above.")
else:
    climate_df = (
        pd.concat(all_dfs, ignore_index=True)
          .sort_values(['timestamp', 'latitude', 'longitude'])
          .reset_index(drop=True)
    )

    # If only one lat/lon point exists, drop duplicates on timestamp
    unique_points = climate_df[['latitude', 'longitude']].drop_duplicates()
    if len(unique_points) == 1:
        climate_df = climate_df.drop_duplicates(subset=['timestamp'])
        print(f"  Single grid point: lat={unique_points.iloc[0]['latitude']:.4f}, "
              f"lon={unique_points.iloc[0]['longitude']:.4f}")
    else:
        climate_df = climate_df.drop_duplicates(subset=['timestamp', 'latitude', 'longitude'])
        print(f"  Multiple grid points: {len(unique_points)}")

    climate_df = climate_df.sort_values('timestamp').reset_index(drop=True)

    # Final cleanup
    climate_df['GHI_Wm2']   = climate_df['GHI_Wm2'].clip(lower=0)
    climate_df['Precip_mm'] = climate_df['Precip_mm'].clip(lower=0)

    output = 'era5_climate_coimbatore_2024.csv'
    climate_df.to_csv(output, index=False)

    print(f"\n🎉 SUCCESS!")
    print(f"   File       : {output}")
    print(f"   Rows       : {len(climate_df):,}")
    print(f"   Date range : {climate_df['timestamp'].min()}  →  {climate_df['timestamp'].max()}")
    print(f"   Columns    : {list(climate_df.columns)}")
    print(f"\n{'─'*60}")
    print("SAMPLE (first 5 rows):")
    print(climate_df.head().to_string(index=False))
    print(f"\n{'─'*60}")
    print("STATISTICS:")
    print(climate_df.drop(columns='timestamp').describe().round(2).to_string())

if failed:
    print(f"\n⚠️  Failed files ({len(failed)}): {failed}")
