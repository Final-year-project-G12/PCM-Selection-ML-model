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

# Time-invariant fields (e.g. surface geopotential / orography) — one cached
# file each, not per-year — kept separate from points/ so elevation lookups
# never touch the sun-event instant/accum cache.
RAW_INVARIANT_DIR = RAW_ERA5_DIR / "invariant"
GEOPOTENTIAL_FILE = RAW_INVARIANT_DIR / "era5_RJ_geopotential.nc"

RAW_POPULATION_DIR = DATA_DIR / "raw" / "population"
RAW_BOUNDARY_DIR = DATA_DIR / "raw" / "boundary"

# Beck et al. 2018 Koppen-Geiger present-climate classification, 1-km
# resolution GeoTIFF ("present" = 1980-2016 climatology), used by
# 05_cluster_rajasthan.py for external validation of the Level A GMM
# clusters. DOI:10.1038/sdata.2018.214; source ZIP (Beck_KG_V1.zip)
# downloaded from https://ndownloader.figshare.com/files/12407516 (figshare
# article 6396959) and cached here, same one-time-download-then-cache
# pattern as RAW_POPULATION_DIR above.
RAW_KOPPEN_DIR = DATA_DIR / "raw" / "koppen"
KOPPEN_RASTER_FILE = RAW_KOPPEN_DIR / "Beck_KG_V1_present_0p0083.tif"
KOPPEN_LEGEND_FILE = RAW_KOPPEN_DIR / "legend.txt"
RAW_POWER_DIR = DATA_DIR / "raw" / "nasapower"
POWER_DOWNLOAD_STATUS_FILE = RAW_POWER_DIR / "download_status_power.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_NAMED_DIR = PROCESSED_DIR / "by_location"
PROCESSED_GRID_DIR = PROCESSED_DIR / "grid"
CLIMATE_COMBINED_FILE = PROCESSED_DIR / "climate_rajasthan_all.csv"

POPULATION_GRID_FILE = PROCESSED_DIR / "population_grid_points.csv"
SUNTIMES_FILE = PROCESSED_DIR / "suntimes.csv"
COMBINED_POINTS_FILE = PROCESSED_DIR / "climate_rajasthan_points.csv"

# 03b_quality_check_rajasthan.py's output — Hampel-filtered/winsorized +
# gap-imputed version of COMBINED_POINTS_FILE, same schema plus per-
# variable *_outlier_flag columns. 04_climate_signature_rajasthan.py reads
# THIS file, not COMBINED_POINTS_FILE directly, as of that quality-check
# script's introduction.
CLEANED_POINTS_FILE = PROCESSED_DIR / "climate_rajasthan_points_clean.csv"

QUALITY_REPORT_MD_FILE = PROCESSED_DIR / "quality_report_rajasthan.md"
QUALITY_REPORT_JSON_FILE = PROCESSED_DIR / "quality_report_rajasthan.json"

DAILY_AGGREGATES_FILE = PROCESSED_DIR / "daily_aggregates_rajasthan.csv"
DAILY_AGGREGATES_SUMMARY_FILE = PROCESSED_DIR / "daily_aggregates_rajasthan_summary.csv"
DAILY_AGGREGATES_STATUS_FILE = PROCESSED_DIR / "daily_aggregates_status.csv"

CLIMATE_SIGNATURE_FILE = PROCESSED_DIR / "climate_signature_rajasthan.csv"

PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PLOTS_DIR = DATA_DIR / "plots"

# QC plots (03_qc_plots.py) — kept top-level, separate from data/, since
# these are throwaway sanity-check artifacts, not pipeline data.
OUTPUTS_DIR = BASE_DIR / "outputs"

CDSAPI_RC = BASE_DIR / ".cdsapirc"


def ensure_data_dirs():
    for directory in (
        RAW_GRID_DIR,
        RAW_POINTS_DIR,
        RAW_INVARIANT_DIR,
        RAW_POPULATION_DIR,
        RAW_BOUNDARY_DIR,
        RAW_KOPPEN_DIR,
        RAW_POWER_DIR,
        PROCESSED_NAMED_DIR,
        PROCESSED_GRID_DIR,
        PREPROCESSED_DIR,
        PLOTS_DIR,
        OUTPUTS_DIR,
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
