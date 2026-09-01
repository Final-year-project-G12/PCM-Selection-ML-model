"""
00_unzip_accum.py — Run this BEFORE 02_combine_assam.py
============================================================
CDS API v2 sometimes downloads files as .zip even when
"download_format": "unarchived" is requested.

This script:
  1. Finds every accum .nc file that is actually a ZIP archive
  2. Extracts the real .nc file from inside it
  3. Replaces the fake .nc with the real one

Run once:
  python 00_unzip_accum.py

Safe to re-run — already-valid NetCDF files are skipped.
"""

import os
import zipfile
import shutil
import tempfile

from config import RAW_GRID_DIR, RAW_POINTS_DIR

INPUT_DIRS = [str(RAW_GRID_DIR), str(RAW_POINTS_DIR)]

def is_zip(filepath):
    """Check if a file is actually a ZIP archive."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
        return header[:2] == b"PK"
    except Exception:
        return False

def is_netcdf(filepath):
    """Check if a file is a valid NetCDF file."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
        return header[:3] == b"CDF" or header[:4] == b"\x89HDF"
    except Exception:
        return False

def fix_file(filepath):
    """If filepath is a ZIP, extract the .nc inside and replace it."""
    fname = os.path.basename(filepath)

    if is_netcdf(filepath):
        print(f"  [OK]      {fname}  (already valid NetCDF)")
        return "ok"

    if not is_zip(filepath):
        size = os.path.getsize(filepath)
        print(f"  [SKIP]    {fname}  (not ZIP, not NetCDF — size={size}B, skipping)")
        return "skip"

    print(f"  [FIXING]  {fname}  (is ZIP archive — extracting real .nc ...)")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(filepath, "r") as z:
                nc_names = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_names:
                    nc_names = z.namelist()
                if not nc_names:
                    print(f"  [ERROR]   {fname} — ZIP is empty!")
                    return "error"
                z.extract(nc_names[0], tmpdir)
                extracted_path = os.path.join(tmpdir, nc_names[0])

            if is_netcdf(extracted_path):
                shutil.move(extracted_path, filepath)
                new_size = os.path.getsize(filepath)
                print(f"  [FIXED]   {fname}  (extracted successfully, size={new_size/1e6:.2f}MB)")
                return "fixed"
            else:
                print(f"  [ERROR]   {fname} — extracted file is not NetCDF")
                return "error"
        except Exception as e:
            print(f"  [ERROR]   {fname} — extraction failed: {e}")
            return "error"

def main():
    print("=" * 68)
    print("  00_unzip_accum.py — Unzipping ERA5 archives ...")
    print("=" * 68)

    fixed_cnt = 0
    ok_cnt = 0
    err_cnt = 0
    skip_cnt = 0

    for d in INPUT_DIRS:
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))
        nc_files = [f for f in files if f.endswith(".nc")]
        print(f"\nScanning {d} ({len(nc_files)} NetCDF files) ...")

        for f in nc_files:
            fp = os.path.join(d, f)
            res = fix_file(fp)
            if res == "fixed":
                fixed_cnt += 1
            elif res == "ok":
                ok_cnt += 1
            elif res == "error":
                err_cnt += 1
            else:
                skip_cnt += 1

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print(f"  Already valid NetCDF : {ok_cnt}")
    print(f"  Fixed (unzipped)     : {fixed_cnt}")
    print(f"  Skipped (non-NC/ZIP) : {skip_cnt}")
    print(f"  Errors               : {err_cnt}")
    print("=" * 68)

if __name__ == "__main__":
    main()
