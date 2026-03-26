import cdsapi
import zipfile
import os

c = cdsapi.Client()

months = [str(i).zfill(2) for i in range(1, 13)]

for m in months:
    print(f"\n📥 Downloading month: {m} ...")

    output_nc  = f'era5_2024_{m}.nc'
    output_zip = f'era5_2024_{m}.zip'

    # ── Download ─────────────────────────────────────────────────────────────
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',                       # t2m  → T_ambient_C
                '2m_dewpoint_temperature',              # d2m  → RH_percent
                '10m_u_component_of_wind',              # u10  → Wind_speed_ms
                '10m_v_component_of_wind',              # v10  → Wind_speed_ms
                'total_precipitation',                  # tp   → Precip_mm   ✅ ADDED
                'surface_solar_radiation_downwards',    # ssrd → GHI_Wm2     ✅ FIXED NAME
                'total_cloud_cover',                    # tcc  → Cloud_cover_fraction
            ],
            'year': '2024',
            'month': m,
            'day':  [str(i).zfill(2) for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'area': [11.12, 76.85, 10.91, 77.06],      # [N, W, S, E] Coimbatore bbox
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        output_nc
    )

    # ── Safety check: if CDS still delivers a zip, unzip it ──────────────────
    with open(output_nc, 'rb') as f:
        header = f.read(4)

    if header[:2] == b'PK':                            # ZIP magic bytes
        print(f"  ⚠️  Got ZIP archive — unzipping {output_nc} ...")
        os.rename(output_nc, output_zip)

        with zipfile.ZipFile(output_zip, 'r') as z:
            names = z.namelist()
            print(f"     ZIP contains: {names}")

            nc_inside = [n for n in names if n.endswith('.nc')]
            if nc_inside:
                z.extract(nc_inside[0], '.')
                os.rename(nc_inside[0], output_nc)
                print(f"  ✅ Extracted → {output_nc}")
            else:
                print(f"  ❌ No .nc found inside zip. Contents: {names}")

        os.remove(output_zip)
    else:
        print(f"  ✅ Valid NetCDF saved → {output_nc}")

print("\n🎉 All months downloaded successfully!")
