import cdsapi
import zipfile
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

c = cdsapi.Client()

months = [str(i).zfill(2) for i in range(1, 13)]

for m in months:
    print(f"\n📥 Downloading month: {m} ...")

    output_nc  = os.path.join(SCRIPT_DIR, f'era5_2024_{m}.nc')
    output_zip = os.path.join(SCRIPT_DIR, f'era5_2024_{m}.zip')

    # ── Download ─────────────────────────────────────────────────────────────
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',
                '2m_dewpoint_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'total_precipitation',
                'surface_solar_radiation_downwards',
                'total_cloud_cover',
            ],
            'year': '2024',
            'month': m,
            'day':  [str(i).zfill(2) for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'area': [11.12, 76.85, 10.91, 77.06],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        output_nc
    )

    # ── Check if CDS delivered a ZIP (happens when variables split by stepType)
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
                z.extract(nc_inside[0], '.')
                os.rename(nc_inside[0], output_nc)
                print(f"  ✅ Extracted → {output_nc}")

            else:
                # ERA5 splits instant vars (t2m, u10, v10, tcc, d2m)
                # and accumulated vars (ssrd, tp) into separate files.
                # Save BOTH with descriptive names so the combine script can merge them.
                for nc_name in nc_inside:
                    z.extract(nc_name, SCRIPT_DIR)
                    basename = os.path.splitext(os.path.basename(nc_name))[0]
                    dest = os.path.join(SCRIPT_DIR, f'era5_2024_{m}__{basename}.nc')
                    os.rename(os.path.join(SCRIPT_DIR, nc_name), dest)
                    print(f"  ✅ Extracted → {dest}")
                # The plain era5_2024_MM.nc is NOT created when there are 2 files.
                # The combine script will glob both parts and merge them.

        os.remove(output_zip)
    else:
        print(f"  ✅ Valid NetCDF saved → {output_nc}")

print("\n🎉 All months downloaded successfully!")
print("\nNOTE: If any month produced two files (e.g. era5_2024_01__*instant*.nc and")
print("      era5_2024_01__*accum*.nc), the combine script will automatically merge them.")
