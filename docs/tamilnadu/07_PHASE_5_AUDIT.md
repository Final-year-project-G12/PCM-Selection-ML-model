# 07 — Phase 5 Audit: Feasibility Filtering

Scripts: `06_build_pcm_database.py`, `07b_charging_feasibility.py` (optional), `07_feasibility_filter.py`.

## Purpose
Hard-screen candidate PCMs from a database against each cluster's climate-adaptive targets (melting point and latent heat) to ensure only physically viable PCMs proceed to ranking.

## The PCM Database
- Imputes missing manufacturer properties (Rubitherm RT, Pluss savE) via MICE+RF+PMM blend.
- Appends 7 literature PCMs (fatty acids, paraffins).
- Total candidates: **62 PCMs**: **55 manufacturer-derived records** completed from the MICE+RF+PMM detailed input plus **7 literature records** from Singh et al. Table 2. Manufacturer imputation flags and provenance are retained; genuinely unreported literature properties remain missing.

## `07b_charging_feasibility.py` (optional, Phase 5 pre-step)
- **Purpose**: implements the one Table 12 filter that `07` does not — "Tm must lie below the collector delivery temperature achievable on a poor day in that regime" — as a **heuristic proxy** (not a collector thermal model). Estimates each cluster's poor-day solar reliability from `kt_mean`/`kt_std` in `climate_signature_tamilnadu.csv` and scales a reference good-day achievable temperature down for less reliable clusters. Scaling constants (`REFERENCE_GOOD_DAY_TEMP`, `MIN_ACHIEVABLE_TEMP`) are stated assumptions.
- **Input**: `data/processed/signatures/climate_signature_tamilnadu.csv`, `data/processed/clustering/cluster_profiles_tamilnadu.csv`.
- **Output**: adds one column `Tm_target_C_regime_capped` to `cluster_profiles_tamilnadu.csv`; the original `Tm_target_C` is left untouched. `07_feasibility_filter.py` uses `Tm_target_C_regime_capped` instead of `Tm_target_C` when the column is present.
- **Dependencies**: `config.PROCESSED_DIR` only. Must run **after** `05`/`04b` and **before** `07`.
- **Status**: OPTIONAL. Without it every cluster shares one constant `Tm_target` and the feasibility windows are identical across clusters. `run_all_tamilnadu.py` sequences it as a non-blocking core step before `07`; its failure does not stop the run. Phase 7's grey-box model supersedes the need for it in practice.

## Screen Constraints (Table 12)
1. Melting window: `Tm ∈ [Tm_target − 5, Tm_target + 8]°C` (relaxable ±2K, up to 4 steps).
2. Absolute band: `Tm ∈ [42, 70]°C`.
3. Latent heat floor: `L ≥ 0.7 × L_required` — **now binding after v3.1 L_required fix**.
4. Cycling stability: `cycles ≥ 300` (flagged if NaN).
5. Supercooling veto: `supercooling ≤ 8K` (flagged if NaN).
6. Corrosion veto: excludes `check_manually` in high-HSI clusters.
7. Safety exclusion: flammability keyword veto.

## Current Finding
- The feasibility output audits **62 candidates per cluster** (310 rows = 62 × 5 in the completed run's `data/processed/processed/pcm/feasibility_survivors_by_cluster.csv`), with pass/fail detail for every filter. Despite its filename, the CSV is not survivors-only.
- Actual survivors (`passes_all=True`) in the completed run are **15, 9, 13, 13, and 9** for clusters 0-4 respectively.
- Completed-run cluster `L_required` values are approximately **301-326 kJ/kg**. The latent-heat floor is `latent_heat_floor_kj_kg(L_required, 0.7) = max(100, 0.7 × L_required)` (e.g. 211 kJ/kg for cluster 0), achievable for a subset of candidates.
- **Stale artifact warning**: the canonical `data/processed/pcm/feasibility_survivors_by_cluster.csv` currently holds a **superseded 25-PCM run** (125 rows, 7 survivors in every cluster). Use `data/processed/processed/` numbers until the clean re-run.

## CRITICAL UPDATE: L_required Methodology Correction (2026-08-31, OPTION A)

**The v3.1 L_required fix documented above has been superseded by a more fundamental methodology correction (2026-08-31).** Phase 3's all-latent assumption (PCM supplies 100% of night discharge alone) was replaced with a literature-anchored fractional-share model: **SHARE_PCM = 0.5**, meaning PCM supplies ~50% of delivery, tank sensible heat + concurrent charging supply the remainder (per Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021).

**Current interpretation:** `SHARE_PCM = 0.5` is active in the upstream sizing calculation. The older ≈2500 kJ/kg all-latent value and the ≈1250 kJ/kg planning estimate are superseded by the values written to the signature and feasibility artifacts. `SHARE_PCM` is defined in `config.py` (added 2026-09-07 — it was previously imported but undefined, so `04b`/`11` raised `ImportError`); `latent_heat_floor_kj_kg()` is also in `config.py` and imported by `07` and `11`. See `20_IMPLEMENTATION_ISSUES.md` §6.

## Status
**Analysis COMPLETE (62-PCM run in `data/processed/processed/`) — clean re-run PENDING.**
- `06_build_pcm_database.py` `INPUT_CSV` path was wrong (resolved to a non-existent `era5-tamilnadu/PCM_data/`); fixed 2026-09-07 to the repo-root `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` (55 manufacturer rows) — see `20_IMPLEMENTATION_ISSUES.md` §7.
- Re-run `06` → `07b` (optional) → `07` (after the upstream chain) to regenerate `pcm_database_tamilnadu.csv` (62 rows) and `feasibility_survivors_by_cluster.csv` in the canonical `data/processed/` tree, then delete the superseded files there. Re-run whenever the PCM source or upstream climate signatures change.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| PCM property database | Martinez (2025) — Rubitherm measured data | `sources/Martinez2025PCM_Industrial_TES_summary.md` |
| Literature PCMs Table 2 | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Melting band 42–70°C SWH | Abdellatif (2025) PCM modeling review | `sources/Abdellatif2025PCM_Modeling_Review_summary.md` |
| Corrosion in humid climates | Hamzat (2025) PCM solar storage | `sources/Hamzat2025PCM_SolarEnergyStorage_summary.md` |
| Property imputation | Eldokaishi (2022) ANN SWH | `sources/Eldokaishi2022WaterPCM_ANN_SWH_summary.md` |
