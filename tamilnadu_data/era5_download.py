import cdsapi
import zipfile
import os
import argparse

# Usage examples:
#   python .\era5_download.py                  # download all months (skip already done)
#   python .\era5_download.py --month 03       # download only March
#   python .\era5_download.py --start 03 --end 06   # download March through June

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Tamil Nadu bounding box (ERA5 area = [N, W, S, E]) ───────────────────────
# Covers the entire state at 0.25° resolution → ~23 lat × 17 lon = ~391 grid points
AREA = [13.6, 76.2, 8.0, 80.4]

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Download ERA5 data month by month')
parser.add_argument('--month', type=str, default=None, help='Single month to download (e.g. 03)')
parser.add_argument('--start', type=str, default='01',  help='Start month (default: 01)')
parser.add_argument('--end',   type=str, default='12',  help='End month   (default: 12)')
args = parser.parse_args()

if args.month:
    months = [args.month.zfill(2)]
else:
    months = [str(i).zfill(2) for i in range(int(args.start), int(args.end) + 1)]

print(f"📋 Months to process: {months}")

c = cdsapi.Client()

for m in months:
    print(f"\n📥 Downloading month: {m} ...")

    output_nc  = os.path.join(SCRIPT_DIR, f'era5_2024_{m}.nc')
    output_zip = os.path.join(SCRIPT_DIR, f'era5_2024_{m}.zip')

    # ── Skip if already downloaded ────────────────────────────────────────────
    import glob as _glob
    already = _glob.glob(os.path.join(SCRIPT_DIR, f'era5_2024_{m}*.nc'))
    if already:
        print(f"  ⏭️  Skipping month {m} — already downloaded: {[os.path.basename(f) for f in already]}")
        continue

    # ── Download ─────────────────────────────────────────────────────────────
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',                               # t2m  → T_ambient_C
                '2m_dewpoint_temperature',                      # d2m  → RH_percent
                '10m_u_component_of_wind',                      # u10  → Wind_speed_ms
                '10m_v_component_of_wind',                      # v10  → Wind_speed_ms
                'total_precipitation',                          # tp   → Precip_mm
                'surface_solar_radiation_downwards',            # ssrd → GHI_Wm2
                'total_cloud_cover',                            # tcc  → Cloud_cover_fraction
                'surface_direct_normal_irradiance',             # fdir → DNI_Wm2  [NEW]
                'surface_diffuse_solar_radiation_downwards',    # fsdss→ DHI_Wm2  [NEW]
                'surface_pressure',                             # sp   → surface_pressure_Pa [NEW]
            ],
            'year': '2024',
            'month': m,
            'day':  [str(i).zfill(2) for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'area': AREA,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        output_nc
    )

    # ── Check if CDS delivered a ZIP (happens when variables split by stepType) ──
    with open(output_nc, 'rb') as f:
        header = f.read(4)

    if header[:2] == b'PK':
        print(f"  ⚠️  Got ZIP archive — extracting ALL .nc files inside ...")
        os.rename(output_nc, output_zip)

        with zipfile.ZipFile(output_zip, 'r') as z:
            names = z.namelist()
            print(f"     ZIP contains: {names}")
            nc_inside = [n for n in names if n.endswith('.nc')]

            if len(nc_inside) == 0:
                print(f"  ❌ No .nc found inside zip.")

            elif len(nc_inside) == 1:
                z.extract(nc_inside[0], SCRIPT_DIR)
                os.rename(os.path.join(SCRIPT_DIR, nc_inside[0]), output_nc)
                print(f"  ✅ Extracted → {output_nc}")

            else:
                # ERA5 splits instant vars (t2m, u10, v10, tcc, d2m)
                # and accumulated vars (ssrd, tp) into separate files.
                for nc_name in nc_inside:
                    z.extract(nc_name, SCRIPT_DIR)
                    basename = os.path.splitext(os.path.basename(nc_name))[0]
                    dest = os.path.join(SCRIPT_DIR, f'era5_2024_{m}__{basename}.nc')
                    os.rename(os.path.join(SCRIPT_DIR, nc_name), dest)
                    print(f"  ✅ Extracted → {dest}")

        os.remove(output_zip)
    else:
        print(f"  ✅ Valid NetCDF saved → {output_nc}")

print("\n🎉 All months downloaded successfully!")
print("\nNOTE: If any month produced two files (e.g. era5_2024_01__*instant*.nc and")
print("      era5_2024_01__*accum*.nc), the combine script will automatically merge them.")
print(f"\nExpected file size: ~391 grid points × 8,784 hours × 7 variables")
print(f"Output CSV will be large (~300MB+ uncompressed). Consider chunking if memory is limited.")
