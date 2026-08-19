# 22 — Reproducibility Audit (Assam)

## Checklist

| Item | Status | Notes |
|---|---|---|
| Random seeds | **PASS** | `random_state=42` set on GMM, K-Means, bootstrap in `05_cluster_assam.py`; `10_physics_validation.py` deterministic (no random component) |
| GMM/scaler persistence | **PASS** | `scaler_assam.joblib` and `gmm_model_assam.joblib` saved — any future re-run of Phase 5–8 can reload them and reproduce cluster assignments without re-fitting |
| sklearn version recorded | **PASS** | Every cluster output CSV has a `sklearn_version` column; value = **1.9.0** |
| Monte Carlo draws | **PASS** | `N_MONTE_CARLO_DRAWS = 5000`, matches plan spec; random seed consistent within cluster runs |
| Dataset version | **PARTIAL** | ERA5 product/version not pinned beyond CDS API; ERA5 is occasionally reprocessed by ECMWF; no download-date manifest per file |
| Download dates | **PARTIAL** | `download_status_points.csv` / `download_status_power.csv` track timestamp per download event; operational logging, not a pinned version statement |
| Geographic coordinates | **PASS** | Deterministic from GADM+WorldPop+fixed 0.25° ERA5-aligned grid — same inputs reproduce same 128 points |
| API parameters | **PASS** | CDS variable lists, POWER parameter strings, hour-window computation all in version-controlled `.py` files |
| Time ranges | **PASS** | `2016-01-01` through `2025-12-31`, hardcoded consistently in all scripts |
| Dependency versions | **FAIL** | No `requirements.txt` or pinned environment file found in `era5-assam/`; same gap as Rajasthan |
| Environment | **FAIL** | No `requirements.txt`/`environment.yml`/lockfile in `era5-assam/`; only README prose pip-install |
| Output naming | **PASS** | Consistent `{artifact}_assam.csv` convention throughout |
| Logging | **PASS** | Download stages have status CSVs; console output is informative |
| Parquet output per-point | **PASS** | `preprocessed/parquet/{point_id}.parquet` — one file per point, consistent naming |
| Canonical cluster relabeling | **PASS** | Clusters relabeled by ascending mean latitude right after GMM fit — same fix as Rajasthan; cluster 0–3 refer to same physical groups across re-runs |
| Cross-phase provenance | **PARTIAL** | No `provenance_lib.py` equivalent for Assam; cluster ID consistency is checked inline in `09_recommendation_cards.py` only |
| Full-chain orchestration | **NOT PRESENT** | No `run_all_assam.py`; scripts must be run manually in order |

## The main reproducibility gaps unique to Assam

### 1. No `run_all_assam.py`

Rajasthan has `run_all_rajasthan.py` — a single-invocation script that runs Phase 2 through Phase 8
in dependency order, stopping at the first core-stage failure. Assam has no equivalent. Running
`08_mcdm_ranking.py` after a `05_cluster_assam.py` re-run without also re-running `07_feasibility_filter.py`
in between would produce silently inconsistent results. Risk is mitigated by the `joblib` model
persistence, but an orchestration script would eliminate it entirely.

### 2. No `provenance_lib.py` cross-phase fingerprinting

Rajasthan's `provenance_lib.py` hard-fails (`SystemExit`) if a downstream phase's input file
doesn't match the cluster profiles file it was originally computed from. This caught the
cluster-label instability bug. Assam's `09_recommendation_cards.py` checks cluster ID consistency
inline, but there is no equivalent hard-fail provenance check at Phases 5, 6, and 7 entry.

### 3. No pinned `requirements.txt`

Same gap as Rajasthan and Tamil Nadu. The `get_solarposition()` method-pin issue
(see `15_SOLAR_GEOMETRY.md`) applies here too.

## Recommended fixes (in order of effort/impact)

1. **Add `requirements.txt`** — `pip freeze > requirements.txt` from the working environment.
   Closes the single biggest reproducibility gap, zero code changes.
2. **Create `run_all_assam.py`** — same structure as `run_all_rajasthan.py`:
   `02_combine_assam.py → 02b → 04_preprocess → 04b → 05_cluster → 06_build_pcm → 07_feasibility → 08_mcdm → 10_physics → 09_recommendation_cards`.
3. **Add provenance fingerprinting** — adapt `provenance_lib.py` from Rajasthan, or add inline
   mtime+size+rowcount checks at entry to `07_feasibility_filter.py`, `08_mcdm_ranking.py`,
   `10_physics_validation.py`.
4. **Record ERA5 pull date** in a manifest file alongside `download_status_points.csv`.
5. **Pin `get_solarposition(method="spa")`** in `02_combine_assam.py`.
