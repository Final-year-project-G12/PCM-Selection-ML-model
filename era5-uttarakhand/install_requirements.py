"""
Installs all third-party packages required by the era5-uttarakhand pipeline
(everything imported across 00_*.py through 12_*.py, run_all_uttarakhand.py,
and config.py). Run once per environment:

    python install_requirements.py
"""

import subprocess
import sys

PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "requests",
    "xarray",
    "pvlib",
    "cdsapi",
    "geopandas",
    "rasterio",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "folium",
    "branca",
    "plotly",
    "streamlit",
]


def main():
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *PACKAGES]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("\nAll packages installed successfully.")


if __name__ == "__main__":
    main()
