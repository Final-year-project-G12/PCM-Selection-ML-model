"""
Shared paths and CDS API settings for the intlo_unna pipeline.

All paths are anchored to this folder (intlo_unna/), so scripts work
regardless of the current working directory.

CDS credentials are read from intlo_unna/.cdsapirc (not ~/.cdsapirc).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_ERA5_DIR = DATA_DIR / "raw" / "era5"
RAW_GRID_DIR = RAW_ERA5_DIR / "grid"
DOWNLOAD_STATUS_FILE = RAW_ERA5_DIR / "download_status.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_NAMED_DIR = PROCESSED_DIR / "by_location"
PROCESSED_GRID_DIR = PROCESSED_DIR / "grid"
CLIMATE_COMBINED_FILE = PROCESSED_DIR / "climate_tamilnadu_all.csv"

PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PLOTS_DIR = DATA_DIR / "plots"

CDSAPI_RC = BASE_DIR / ".cdsapirc"


def ensure_data_dirs():
    for directory in (
        RAW_GRID_DIR,
        PROCESSED_NAMED_DIR,
        PROCESSED_GRID_DIR,
        PREPROCESSED_DIR,
        PLOTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_cds_credentials():
    """Read url and key from intlo_unna/.cdsapirc."""
    if not CDSAPI_RC.is_file():
        raise FileNotFoundError(
            f"CDS API config not found: {CDSAPI_RC}\n"
            "Add a .cdsapirc file in the intlo_unna folder with your url and key."
        )

    text = CDSAPI_RC.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(
            f"CDS config file is empty: {CDSAPI_RC}\n"
            "Save the file in your editor, then add:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <your-copernicus-api-key>"
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
            "Expected two lines:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <your-copernicus-api-key>"
        )

    return url, key


def get_cdsapi_client():
    import cdsapi

    url, key = load_cds_credentials()
    return cdsapi.Client(url=url, key=key)

'''cd intlo_unna
python -m jupyter nbconvert --to notebook --execute 05_plot_tamilnadu.ipynb --output 05_plot_tamilnadu_executed.ipynb'''