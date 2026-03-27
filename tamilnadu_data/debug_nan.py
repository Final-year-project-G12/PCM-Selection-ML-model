#!/usr/bin/env python
import pandas as pd
import xarray as xr
import glob
import os

print("=" * 70)
print("DIAGNOSING NaN ISSUES")
print("=" * 70)

# Check base CSV
df = pd.read_csv('era5_climate_tamilnadu_2024.csv')
print(f"\n📋 Base CSV columns: {list(df.columns)}")
print(f"   Shape: {df.shape}")

# Check a P_atm/sp extraction
print(f"\n🔍 Checking P_atm extraction from extra instant files...")
extra_instant_files = sorted(glob.glob('*_extra*instant*.nc'))
if extra_instant_files:
    print(f"   Found {len(extra_instant_files)} extra instant files")
    
    for f in extra_instant_files[:2]:
        print(f"\n   {os.path.basename(f)}:")
        try:
            ds = xr.open_dataset(f)
            print(f"      Variables: {list(ds.data_vars.keys())}")
            df_test = ds.to_dataframe().reset_index()
            print(f"      Shape: {df_test.shape}")
            print(f"      Columns: {list(df_test.columns)}")
            
            if 'sp' in df_test.columns:
                print(f"      'sp' non-null count: {df_test['sp'].notna().sum()}")
                print(f"      'sp' value range: {df_test['sp'].min():.0f} - {df_test['sp'].max():.0f}")
            
            if 'valid_time' in df_test.columns:
                print(f"      'valid_time' sample: {df_test['valid_time'].iloc[0]}")
            
            ds.close()
        except Exception as e:
            print(f"      ERROR: {e}")

# Now re-run the merge logic in simplified form
print(f"\n🔄 Simulating merge logic...")
print(f"   Base CSV has {len(df)} rows")

# Load instant files
instant_files = sorted(glob.glob('*instant*.nc'))[:2]  # Just first 2 for testing
raw_dfs = []
for f in instant_files:
    ds = xr.open_dataset(f)
    raw = ds.to_dataframe().reset_index()
    ds.close()
    raw = raw.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    for tcol in ['valid_time', 'time']:
        if tcol in raw.columns:
            raw['timestamp'] = pd.to_datetime(raw[tcol]); break
    
    keep = ['timestamp', 'latitude', 'longitude']
    for c in ['d2m', 'u10', 'v10']:
        if c in raw.columns: keep.append(c)
    
    if len(keep) > 3:
        raw_dfs.append(raw[[k for k in keep if k in raw.columns]])
        
if raw_dfs:
    raw_combined = pd.concat(raw_dfs, ignore_index=True)
    print(f"   Instant merge would have {len(raw_combined)} rows")

# Load extra files
extra_files = sorted(glob.glob('*_extra*.nc'))[:4]  # Just first 4 for testing
extra_dfs = []
for f in extra_files:
    ds = xr.open_dataset(f)
    raw = ds.to_dataframe().reset_index()
    ds.close()
    print(f"   {os.path.basename(f)}: vars={list(ds.data_vars.keys())}, shape={raw.shape}")
    raw = raw.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    for tcol in ['valid_time', 'time']:
        if tcol in raw.columns:
            raw['timestamp'] = pd.to_datetime(raw[tcol]); break
    extra_dfs.append(raw)

if extra_dfs:
    extra_combined = pd.concat(extra_dfs, ignore_index=True)
    print(f"   Extra merge would have {len(extra_combined)} rows")
    print(f"   'sp' in extra_combined: {'sp' in extra_combined.columns}")
    if 'sp' in extra_combined.columns:
        print(f"   'sp' non-null: {extra_combined['sp'].notna().sum()}")

print("\n" + "=" * 70)
