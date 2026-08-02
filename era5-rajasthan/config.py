"""
Shared paths and CDS API settings for the ERA5 Rajasthan pipeline.

All paths are anchored to this folder, so scripts work regardless of the
current working directory.

CDS credentials are read from the local .cdsapirc file, with environment
variable fallback for convenience.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_ERA5_DIR = DATA_DIR / "raw" / "era5"
RAW_GRID_DIR = RAW_ERA5_DIR / "grid"
DOWNLOAD_STATUS_FILE = RAW_ERA5_DIR / "download_status.csv"

# Population-weighted points pipeline (distinct from the old full-state grid
# above — different bbox, different hours, kept separate on purpose so the
# old grid/ archive and its status file are never touched by the new code).
RAW_POINTS_DIR = RAW_ERA5_DIR / "points"
POINTS_DOWNLOAD_STATUS_FILE = RAW_ERA5_DIR / "download_status_points.csv"

RAW_POPULATION_DIR = DATA_DIR / "raw" / "population"
RAW_BOUNDARY_DIR = DATA_DIR / "raw" / "boundary"
RAW_POWER_DIR = DATA_DIR / "raw" / "nasapower"
POWER_DOWNLOAD_STATUS_FILE = RAW_POWER_DIR / "download_status_power.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_NAMED_DIR = PROCESSED_DIR / "by_location"
PROCESSED_GRID_DIR = PROCESSED_DIR / "grid"
CLIMATE_COMBINED_FILE = PROCESSED_DIR / "climate_rajasthan_all.csv"

POPULATION_GRID_FILE = PROCESSED_DIR / "population_grid_points.csv"
SUNTIMES_FILE = PROCESSED_DIR / "suntimes.csv"
COMBINED_POINTS_FILE = PROCESSED_DIR / "climate_rajasthan_points.csv"

PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PLOTS_DIR = DATA_DIR / "plots"

CDSAPI_RC = BASE_DIR / ".cdsapirc"


def ensure_data_dirs():
    for directory in (
        RAW_GRID_DIR,
        RAW_POINTS_DIR,
        RAW_POPULATION_DIR,
        RAW_BOUNDARY_DIR,
        RAW_POWER_DIR,
        PROCESSED_NAMED_DIR,
        PROCESSED_GRID_DIR,
        PREPROCESSED_DIR,
        PLOTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_cds_credentials():
    """Read url and key from .cdsapirc or environment variables."""
    import os

    env_url = os.getenv("CDSAPI_URL")
    env_key = os.getenv("CDSAPI_KEY")
    if env_url and env_key:
        return env_url.strip(), env_key.strip()

    if not CDSAPI_RC.is_file():
        raise FileNotFoundError(
            f"CDS API config not found: {CDSAPI_RC}\n"
            "Add a .cdsapirc file in the pipeline folder, or set CDSAPI_URL and CDSAPI_KEY."
        )

    text = CDSAPI_RC.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(
            f"CDS config file is empty: {CDSAPI_RC}\n"
            "Copy the contents of .cdsapirc.example into this file, or set CDSAPI_URL and CDSAPI_KEY."
        )

    url = None
    key = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("url:"):
            url = line.split(":", 1)[1].strip()
        elif line.startswith("key:"):
            key = line.split(":", 1)[1].strip()

    if not url or not key:
        raise ValueError(
            f"Invalid or incomplete CDS config: {CDSAPI_RC}\n"
            "Expected two lines or set CDSAPI_URL and CDSAPI_KEY:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <your-copernicus-api-key>"
        )

    return url, key


def get_cdsapi_client():
    import cdsapi

    url, key = load_cds_credentials()
    return cdsapi.Client(url=url, key=key)
