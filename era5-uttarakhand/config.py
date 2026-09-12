"""
Shared paths and CDS API settings for the ERA5 Uttarakhand pipeline.

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

RAW_INVARIANT_DIR = RAW_ERA5_DIR / "invariant"
GEOPOTENTIAL_FILE = RAW_INVARIANT_DIR / "era5_UK_geopotential.nc"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_NAMED_DIR = PROCESSED_DIR / "by_location"
PROCESSED_GRID_DIR = PROCESSED_DIR / "grid"
CLIMATE_COMBINED_FILE = PROCESSED_DIR / "climate_uttarakhand_all.csv"

POPULATION_GRID_FILE = PROCESSED_DIR / "population_grid_points.csv"
SUNTIMES_FILE = PROCESSED_DIR / "suntimes.csv"
COMBINED_POINTS_FILE = PROCESSED_DIR / "climate_uttarakhand_points.csv"

PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PLOTS_DIR = DATA_DIR / "plots"
OUTPUTS_DIR = BASE_DIR / "outputs"

CDSAPI_RC = BASE_DIR / ".cdsapirc"


def ensure_data_dirs():
    for directory in (
        RAW_GRID_DIR,
        RAW_POINTS_DIR,
        RAW_POPULATION_DIR,
        RAW_BOUNDARY_DIR,
        RAW_POWER_DIR,
        RAW_INVARIANT_DIR,
        PROCESSED_NAMED_DIR,
        PROCESSED_GRID_DIR,
        PREPROCESSED_DIR,
        PLOTS_DIR,
        OUTPUTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# PCM sizing shared across Phase 3 (04b) and Level B (11).
# SHARE_PCM: literature-anchored fraction of overnight delivery supplied by
# PCM latent heat (remainder from tank sensible heat + concurrent charging).
# See 04b_climate_signature.py's draw-sizing comment for the full citation
# list and the bug this fixes (previous DRAW_RATE_KG_PER_S formula was
# missing water's density factor, making L_required ~1000x too small and
# the feasibility floor below a no-op).
SHARE_PCM = 0.5

# Latent-heat feasibility floor, used by 07_feasibility_filter.py and
# 11_level_b_seasonal_analysis.py. Independent of how L_required itself is
# computed (see 04b_climate_signature.py for that) — this just enforces a
# practical absolute minimum on top of whatever fraction-of-L_required rule
# a given script applies.
LATENT_HEAT_FRACTION = 0.7
LATENT_HEAT_ABSOLUTE_MIN_KJ_KG = 100.0


def latent_heat_floor_kj_kg(l_required_kj_per_kg,
                            fraction=LATENT_HEAT_FRACTION,
                            absolute_min=LATENT_HEAT_ABSOLUTE_MIN_KJ_KG):
    """L >= max(100 kJ/kg, 0.7 x L_required)."""
    return max(absolute_min, fraction * l_required_kj_per_kg)


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
    # pyrefly: ignore [missing-import]
    import cdsapi

    url, key = load_cds_credentials()
    return cdsapi.Client(url=url, key=key)