import cdsapi
import zipfile
import os
import glob
import argparse

# Downloads ONLY the 3 extra variables — DNI, DHI, surface_pressure
# Output files are named:  era5_2024_MM_extra.nc  (or split ZIPs)
#
# Usage:
#   python .\era5_download_extra.py               # all 12 months
#   python .\era5_download_extra.py --month 03    # just March
#   python .\era5_download_extra.py --start 01 --end 06

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AREA = [13.6, 76.2, 8.0, 80.4]   # Tamil Nadu [N, W, S, E]

parser = argparse.ArgumentParser(description='Download extra ERA5 variables (DNI, DHI, pressure)')
parser.add_argument('--month', type=str, default=None)
parser.add_argument('--start', type=str, default='01')
parser.add_argument('--end',   type=str, default='12')
args = parser.parse_args()

if args.month:
    months = [args.month.zfill(2)]
else:
    months = [str(i).zfill(2) for i in range(int(args.start), int(args.end) + 1)]

print(f"[Extra-variable download for months]: {months}")
print("   Variables: DNI (fdir), DHI (fsdss), surface_pressure (sp)")

c = cdsapi.Client()

for m in months:
    print(f"\n[Downloading extra vars for month]: {m} ...")

    output_nc  = os.path.join(SCRIPT_DIR, f'era5_2024_{m}_extra.nc')
    output_zip = os.path.join(SCRIPT_DIR, f'era5_2024_{m}_extra.zip')

    # Skip if already downloaded
    already = glob.glob(os.path.join(SCRIPT_DIR, f'era5_2024_{m}_extra*.nc'))
    if already:
        print(f"  [Skipping] Already have: {[os.path.basename(f) for f in already]}")
        continue

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'total_sky_direct_solar_radiation_at_surface',  # fdir  → DNI proxy
                'surface_thermal_radiation_downwards',          # strd  → LW_down
                'surface_pressure',                             # sp    → P_atm
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

    # Handle ZIP (CDS sometimes splits files by stepType)
    with open(output_nc, 'rb') as f:
        header = f.read(4)

    if header[:2] == b'PK':
        print(f"  [WARNING] Got ZIP — extracting ...")
        os.rename(output_nc, output_zip)
        with zipfile.ZipFile(output_zip, 'r') as z:
            names = z.namelist()
            nc_inside = [n for n in names if n.endswith('.nc')]
            print(f"     ZIP contains: {names}")
            if len(nc_inside) == 1:
                z.extract(nc_inside[0], SCRIPT_DIR)
                os.rename(os.path.join(SCRIPT_DIR, nc_inside[0]), output_nc)
                print(f"  [OK] Extracted -> {output_nc}")
            else:
                for nc_name in nc_inside:
                    z.extract(nc_name, SCRIPT_DIR)
                    basename = os.path.splitext(os.path.basename(nc_name))[0]
                    dest = os.path.join(SCRIPT_DIR, f'era5_2024_{m}_extra__{basename}.nc')
                    os.rename(os.path.join(SCRIPT_DIR, nc_name), dest)
                    print(f"  [OK] Extracted -> {dest}")
        os.remove(output_zip)
    else:
        print(f"  [OK] Valid NetCDF saved -> {output_nc}")

print("\n[DONE] Extra variable download complete!")
print("Now run:  python .\\era5_merge_extra.py")
