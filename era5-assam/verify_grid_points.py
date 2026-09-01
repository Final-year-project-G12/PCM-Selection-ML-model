import csv

rows = []
with open(r'm:/Final_year_pro/PCM-Selection-ML-model/era5-assam/data/processed/population_grid_points.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({'id': r['point_id'], 'lat': float(r['lat']), 'lon': float(r['lon'])})

print(f'Total points: {len(rows)}')
lats = sorted(set(r["lat"] for r in rows))
lons = sorted(set(r["lon"] for r in rows))
print(f'Lat range: {min(lats):.3f} to {max(lats):.3f}')
print(f'Lon range: {min(lons):.3f} to {max(lons):.3f}')
print()
print(f'Unique latitudes  ({len(lats)}): {lats}')
print(f'Unique longitudes ({len(lons)}): {lons}')

# Assam bounding box check (generous bounds)
ALAT_MIN, ALAT_MAX = 24.1, 28.2
ALON_MIN, ALON_MAX = 89.6, 96.1
outside = [r for r in rows if r['lat'] < ALAT_MIN or r['lat'] > ALAT_MAX or r['lon'] < ALON_MIN or r['lon'] > ALON_MAX]
print(f'\nPoints outside Assam bounding box [{ALAT_MIN}-{ALAT_MAX}N, {ALON_MIN}-{ALON_MAX}E]: {len(outside)}')
for r in outside:
    print(f'  {r["id"]}  lat={r["lat"]}  lon={r["lon"]}')

print('\n--- All 129 grid points ---')
print(f'{"ID":<12} {"Lat":>8} {"Lon":>8}')
print('-' * 32)
for r in rows:
    print(f'{r["id"]:<12} {r["lat"]:>8.3f} {r["lon"]:>8.3f}')
