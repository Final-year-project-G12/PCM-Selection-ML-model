"""
diagnose_nc.py
==============
Examine NetCDF files to diagnose missing data issues
"""
import os, glob
import xarray as xr
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("NETCDF DIAGNOSTIC REPORT")
print("=" * 70)

# Check instant files
print("\n[INSTANT FILES (.._instant.nc)]:")
instant_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*instant*.nc')))
print(f"   Found {len(instant_files)} files\n")

for f in instant_files[:2]:  # Check first 2
    print(f"   {os.path.basename(f)}:")
    try:
        ds = xr.open_dataset(f)
        print(f"      Dimensions: {dict(ds.dims)}")
        print(f"      Variables: {list(ds.data_vars.keys())}")
        print(f"      Coords: {list(ds.coords.keys())}")
        df = ds.to_dataframe().reset_index()
        print(f"      Shape: {df.shape}")
        print(f"      Columns: {list(df.columns)}")
        ds.close()
    except Exception as e:
        print(f"      ERROR: {e}")

# Check extra files
print("\n[EXTRA FILES (.._extra*.nc)]:")
extra_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*_extra*.nc')))
print(f"   Found {len(extra_files)} files\n")

for f in extra_files[:4]:  # Check first 4 (2 accum + 2 instant)
    print(f"   {os.path.basename(f)}:")
    try:
        ds = xr.open_dataset(f)
        print(f"      Dimensions: {dict(ds.dims)}")
        print(f"      Variables: {list(ds.data_vars.keys())}")
        print(f"      Coords: {list(ds.coords.keys())}")
        df = ds.to_dataframe().reset_index()
        print(f"      Shape: {df.shape}")
        print(f"      Columns: {list(df.columns)}")
        ds.close()
    except Exception as e:
        print(f"      ERROR: {e}")

# Check base CSV
print("\n[BASE CSV]:")
csv_file = os.path.join(SCRIPT_DIR, 'era5_climate_tamilnadu_2024.csv')
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, nrows=5)
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {pd.read_csv(csv_file).shape}")
else:
    print(f"   NOT FOUND")

print("\n" + "=" * 70)
