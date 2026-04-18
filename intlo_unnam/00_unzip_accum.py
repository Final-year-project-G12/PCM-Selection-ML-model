"""
00_unzip_accum.py — Run this BEFORE 02_combine_tamilnadu.py
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

INPUT_DIR = "data/raw/era5/grid"

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
        # NetCDF3 magic: b'CDF\x01' or b'CDF\x02'
        # NetCDF4/HDF5 magic: b'\x89HDF'
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

    print(f"  [UNZIP]   {fname}  ...", end=" ")

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            members = zf.namelist()
            # Find the .nc file inside the zip
            nc_members = [m for m in members if m.lower().endswith(".nc")]
            if not nc_members:
                print(f"\n  [ERROR]   No .nc file found inside {fname}")
                print(f"            Contents: {members}")
                return "error"
            # Extract all nc files (usually just one)
            for member in nc_members:
                zf.extract(member, tmpdir)
            extracted = os.path.join(tmpdir, nc_members[0])

        # Verify the extracted file is a valid NetCDF
        if not is_netcdf(extracted):
            print(f"\n  [ERROR]   Extracted file is not valid NetCDF: {nc_members[0]}")
            return "error"

        # Replace the zip-disguised file with the real NetCDF
        os.replace(extracted, filepath)
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"✓  {size_mb:.2f} MB")
        return "fixed"

    except zipfile.BadZipFile:
        print(f"\n  [ERROR]   {fname} is not a valid ZIP file either.")
        print(f"            The download may be corrupt. Delete and re-download.")
        return "error"
    except Exception as e:
        print(f"\n  [ERROR]   {e}")
        return "error"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print("=" * 60)
    print("  ERA5 Accum File Unzipper")
    print(f"  Directory: {INPUT_DIR}")
    print("=" * 60)

    if not os.path.exists(INPUT_DIR):
        print(f"\n  ERROR: Directory not found: {INPUT_DIR}")
        return

    # Find all accum .nc files
    accum_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith("_accum.nc")
    ])

    if not accum_files:
        print("  No accum .nc files found.")
        return

    print(f"\n  Found {len(accum_files)} accum files\n")

    counts = {"ok": 0, "fixed": 0, "skip": 0, "error": 0}
    for fp in accum_files:
        result = fix_file(fp)
        counts[result] += 1

    print("\n" + "=" * 60)
    print(f"  Results:")
    print(f"    Already valid NetCDF : {counts['ok']}")
    print(f"    Fixed (unzipped)     : {counts['fixed']}")
    print(f"    Skipped              : {counts['skip']}")
    print(f"    Errors               : {counts['error']}")

    if counts["error"] > 0:
        print("\n  ⚠️  Some files had errors — delete them and re-run 01_download...")
    elif counts["fixed"] > 0:
        print("\n  ✅  All ZIP files extracted. Now run: python 02_combine_tamilnadu.py")
    else:
        print("\n  ✅  All files were already valid NetCDF.")
        print("     Now run: python 02_combine_tamilnadu.py")
    print("=" * 60)


if __name__ == "__main__":
    main()