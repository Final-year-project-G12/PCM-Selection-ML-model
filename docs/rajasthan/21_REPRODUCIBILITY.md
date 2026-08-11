# 21 — Reproducibility Audit

## Checklist

| Item | Status | Notes |
|---|---|---|
| Random seeds | **PASS** | `random_state=42` used consistently across `01_preprocess.py` (RF imputation), `04_climate_signature_rajasthan.py` (PCA), `05_cluster_rajasthan.py` (GMM/K-Means/bootstrap), `08_mcdm_ranking_rajasthan.py` (Monte Carlo, though re-seeded fresh per cluster, not a continuing stream — document this explicitly if exact draw-by-draw reproduction across clusters is ever needed) |
| Dataset version | **PARTIAL** | ERA5 product/version not explicitly pinned in any config (relies on whatever `reanalysis-era5-single-levels` currently serves via CDS — ERA5 itself is occasionally reprocessed/updated by ECMWF; no download-date-stamped snapshot is recorded per file beyond the file's own OS timestamp) |
| Download dates | **PARTIAL** | `download_status_points.csv`/`download_status_power.csv` record a `timestamp` per download event — good — but this is operational logging, not a single pinned "dataset version" statement suitable for a methods section |
| Geographic coordinates | **PASS** | Deterministic, reproducible from GADM+WorldPop+the fixed 0.25° ERA5-aligned grid algorithm — same inputs would reproduce the same 320 points (modulo WorldPop/GADM's own version stability, which is outside this pipeline's control) |
| API parameters | **PASS** | Exact CDS variable lists, NASA POWER parameter string, hour-window computation — all in version-controlled `.py` files, not ad hoc notebook cells |
| Time ranges | **PASS** | `2016-01-01`..`2025-12-31`, hardcoded consistently |
| Variable names / units | **MOSTLY PASS** | See `02_DATA_SOURCES_AND_VARIABLES.md`; the one open ambiguity is the `avg_sdirswrf` column-matching issue (item 16 in `20_IMPLEMENTATION_ISSUES.md`) |
| Preprocessing rules | **PASS** | Deterministic, code-defined, not manual/notebook-based |
| Dependency versions | **FAIL** | `README.md`'s `pip install ...` list has **no version pins** — `geopandas`, `rasterio`, `pvlib`, `scikit-learn`, etc. are all unpinned. Given at least one solar-geometry call (`get_solarposition`) relies on a library default that could change across versions (see `12_SOLAR_GEOMETRY.md`), this is a genuine reproducibility risk, not just hygiene |
| Environment | **FAIL** | No `requirements.txt`/`environment.yml`/lockfile found in `era5-rajasthan/`; only the README's prose pip-install line |
| Output naming | **PASS** | Consistent `{artifact}_rajasthan.csv` convention throughout, state-parameterized where the code is designed to be state-agnostic |
| Logging | **PASS** | Every download/compute stage with a resumability concern has a status CSV; console output is informative (branch decisions, warnings) |
| Metadata | **PARTIAL** | Point metadata (population, weight, elevation) is well-tracked; no single manifest file records "this run used ERA5 pulled on date X against CDS API version Y" |
| Intermediate files | **PASS** | Raw/processed separation is clean (`data/raw/` vs `data/processed/`), `.gitignore` correctly excludes generated data from version control while keeping code tracked |
| Random-state Monte Carlo reproducibility | **PASS with a caveat** | Fixed seed reproduces the exact same 1000-draw sequence per cluster, but the fresh-reseed-per-cluster design means the *order* clusters are processed in doesn't affect results (a good property) at the cost of clusters not being a single continuing random stream (a neutral, documented design choice, not a defect) |
| Cross-phase provenance consistency (added 2026-08-11) | **PASS — a real bug was caught and fixed here** | `provenance_lib.py` fingerprint-stamps (mtime+size+row_count of `cluster_profiles_rajasthan.csv`) every Phase 5/6/7/8 output and HARD-FAILS (`SystemExit`, not a warning) if a downstream phase's input doesn't match the file currently on disk. This closes a real hazard found this session: Phase 5's and Phase 6's outputs had been generated from two different Phase 4 runs and silently disagreed on cluster_id/pcm_id pairing until this check existed. See `19_PHASE_7_ONWARD.md` for the incident and `06_PHASE_4_AUDIT.md` for the companion canonical-relabeling fix. |
| Single-invocation full-chain reproduction (added 2026-08-11) | **PASS** | `run_all_rajasthan.py` runs Phase 2 through Phase 8 in one invocation, in dependency order, stopping at the first core-stage failure — the most reliable way to guarantee every phase in a given run was actually produced from the same upstream state (directly reduces the risk the provenance-fingerprint row above exists to catch in the first place). |

## The one reproducibility hazard worth calling out specifically

The `until phase 4/` folder's file-mislabeling (confirmed via byte-level PNG-magic-number checks on
files named `.csv`) is a **real hazard if anyone — including a future version of this same project
team — cites a file path from that folder without first verifying its actual content.** The live
Rajasthan pipeline itself does not depend on any file in that folder (it reads from the correctly-
named `PCM_data/` copies instead), so this does not affect the *pipeline's* reproducibility — but it
would affect anyone trying to reproduce or audit the PCM-database preprocessing step by following a
path name from `until phase 4/` at face value. Recommend either deleting/archiving that folder
outside the working tree, or adding a prominent note in its own README (which, per the extraction
agent's finding, already partially documents the mislabeling) directing readers to the canonical
`PCM_data/` copies exclusively.

## Recommended fixes, in order of effort/impact

1. **Add a pinned `requirements.txt`** (even an unpinned-to-pinned `pip freeze` snapshot from the
   current working environment) to `era5-rajasthan/` — lowest effort, closes the single biggest
   reproducibility gap found.
2. **Record the ERA5 product version/pull date** in a small manifest file alongside
   `download_status_points.csv` — CDS reanalysis products are versioned/reprocessed occasionally;
   a future re-download against a different backend version could silently differ from this run's
   data.
3. **Pin `get_solarposition(method=...)`** explicitly (see `12_SOLAR_GEOMETRY.md`) rather than
   relying on a library default.
4. **Archive or clearly re-flag `until phase 4/`** so its mislabeled files cannot be mistaken for
   canonical sources by a future reader.
