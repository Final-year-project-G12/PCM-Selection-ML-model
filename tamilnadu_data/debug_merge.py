#!/usr/bin/env python
import pandas as pd
import xarray as xr
import glob
import os

print("=" * 70)
print("MERGE KEY ANALYSIS")
print("=" * 70)

# Load base CSV and check sample
base = pd.read_csv('era5_climate_tamilnadu_2024.csv', nrows=1000)
base['latitude'] = base['latitude'].round(4)
base['longitude'] = base['longitude'].round(4)
base['timestamp'] = pd.to_datetime(base['timestamp'])

print("\n📋 BASE CSV (first 1000 rows):")
print(f"   Unique timestamps: {base['timestamp'].nunique()}")
print(f"   Unique (lat,lon) pairs: {len(base[['latitude', 'longitude']].drop_duplicates())}")
print(f"   Sample timestamp: {base['timestamp'].iloc[0]}")
print(f"   Sample lat/lon: {base['latitude'].iloc[0]}, {base['longitude'].iloc[0]}")
print(f"   Lat range: {base['latitude'].min():.4f} - {base['latitude'].max():.4f}")
print(f"   Lon range: {base['longitude'].min():.4f} - {base['longitude'].max():.4f}")

# Load extra instant file and check
ds = xr.open_dataset('era5_2024_01_extra__data_stream-oper_stepType-instant.nc')
extra = ds.to_dataframe().reset_index()
extra = extra.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
extra['latitude'] = extra['latitude'].round(4)
extra['longitude'] = extra['longitude'].round(4)
extra['timestamp'] = pd.to_datetime(extra['valid_time'])
ds.close()

print("\n🔍 EXTRA INSTANT FILE:")
print(f"   Unique timestamps: {extra['timestamp'].nunique()}")
print(f"   Unique (lat,lon) pairs: {len(extra[['latitude', 'longitude']].drop_duplicates())}")
print(f"   Sample timestamp: {extra['timestamp'].iloc[0]}")
print(f"   Sample lat/lon: {extra['latitude'].iloc[0]}, {extra['longitude'].iloc[0]}")
print(f"   Lat range: {extra['latitude'].min():.4f} - {extra['latitude'].max():.4f}")
print(f"   Lon range: {extra['longitude'].min():.4f} - {extra['longitude'].max():.4f}")

# Try a manual merge test
print("\n🔗 MANUAL MERGE TEST:")
test_base = base[['timestamp', 'latitude', 'longitude']].drop_duplicates().head(10)
test_extra = extra[['timestamp', 'latitude', 'longitude']].drop_duplicates()

# Check if any match
matches = 0
for idx, row in test_base.iterrows():
    ts, lat, lon = row['timestamp'], row['latitude'], row['longitude']
    matching = test_extra[
        (test_extra['timestamp'] == ts) &
        (test_extra['latitude'] == lat) &
        (test_extra['longitude'] == lon)
    ]
    if len(matching) > 0:
        matches += 1
        print(f"   ✓ Match found: {ts} @ ({lat}, {lon})")
    else:
        print(f"   ✗ No match: {ts} @ ({lat}, {lon})")

print(f"\n   Total matches: {matches} / {len(test_base)}")

print("\n" + "=" * 70)
