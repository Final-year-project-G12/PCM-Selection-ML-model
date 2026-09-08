# 07 — Phase 5 Audit: Feasibility Filtering

Scripts: `06_build_pcm_database.py`, `07_feasibility_filter.py`.

> **UNIFIED 2026-09-08 with Rajasthan.** `07_feasibility_filter.py` now uses the
> same 8-constraint set, constraint order, missing-value semantics,
> kappa-calibration procedure and provenance stamping as
> `era5-rajasthan/07_feasibility_filter.py`. `07b_charging_feasibility.py` has
> been **retired** (its heuristic regime-cap is folded into Constraint 6, which
> uses Phase 3's `Tm_target_capped_C`). Output filenames
> (`feasibility_survivors_by_cluster{,_kappa_calibrated}.csv`) are the canonical
> naming for both states — Rajasthan's `feasibility_survivors_rajasthan*.csv`
> were renamed to match. **The pipeline has not been re-run since these changes;
> the survivor counts below are from the pre-unification run and are stale.**

## Purpose
Hard-screen candidate PCMs from a database against each cluster's climate-adaptive targets (melting point and latent heat) to ensure only physically viable PCMs proceed to ranking.

## The PCM Database
- `06_build_pcm_database.py` is a **thin builder**: it consumes the single canonical MICE+RF+PMM preprocessing output (`PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv`, produced by `PCM_data/PCM_data/01_preprocess.py`) and only adds derived columns. It does **not** re-impute anything — there is one imputation pipeline for the shared manufacturer data, shared with Rajasthan.
- Appends 7 literature PCMs (fatty acids, paraffins) from Singh et al. Table 2.
- Total candidates: **62 PCMs** = **55 manufacturer-derived records** + **7 literature records**. Row-for-row identical to Rajasthan's pool on all shared columns (same canonical source + same literature rows).

## Screen Constraints (Table 12) — 8 constraints, exact order (matches Rajasthan)
1. **Melting window**: `Tm ∈ [Tm_target − 5, Tm_target + 8]°C`, relaxable ±2K, up to 4 rounds.
2. **Absolute band**: `Tm ∈ [42, 70]°C`.
3. **Latent heat floor**: `L ≥ κ · L_required`, `κ = 0.7` nominal. `pass` / `fail` / `flag_unreported`.
4. **Cycling stability**: `cycles ≥ 300`; unreported → `flag_unreported` (never excludes).
5. **Supercooling**: `≤ 8K`; unknown → `flag_unknown` (never excludes).
6. **Charging feasibility**: `Tm ≤ Tm_target_capped_C` (Phase 3's `kt_worst_month` ceiling, referenced directly — the single charging-feasibility path; the old `REFERENCE_GOOD_DAY_TEMP` / `MIN_ACHIEVABLE_TEMP` heuristic is gone).
7. **Corrosion veto**: bare salt hydrate + cluster `HSI_sunrise` > 75th percentile, unless encapsulated. **Implemented but currently excludes zero candidates because the shared PCM database contains zero salt-hydrate candidates.**
8. **Safety exclusion**: flag-only; does not currently exclude any candidate.

## Kappa calibration
`calibrate_kappa_for_cluster()` (ported from Rajasthan) steps κ down `0.7 → 0.0` in 0.1 increments, targeting **8–20 survivors per cluster**, evaluated at each cluster's melting window from the primary run's final relaxation round. Produces `feasibility_survivors_by_cluster_kappa_calibrated.csv`. The Phase 5 report must state **both** the nominal κ=0.7 survivor count per cluster and the final calibrated-κ survivor count per cluster.

## Current Finding (STALE — pre-unification run, pending re-run)
- The feasibility output audits **62 candidates per cluster** (all rows kept with per-constraint verdicts; use `survives_all == True` / `passes_all == True` for actual survivors).
- Pre-unification survivor counts were **15, 9, 13, 13, 9** for clusters 0-4 (7-constraint schema, no Constraint 6, no kappa calibration). **These will change after the re-run** and must not be quoted for the unified pipeline.
- `L_required` values are ~**301–326 kJ/kg** (combined sensible+latent basis, Phase 3 OPTION A).
- **Path bug fixed**: the `data/processed/processed/` duplication is resolved in `config.py` / `04b_climate_signature.py`. The stale `data/processed/processed/` mirror tree has been deleted. The stale 7-constraint `data/processed/pcm/feasibility_survivors_by_cluster.csv` will be overwritten by the re-run (delete it first for a clean tree if preferred).

## CRITICAL UPDATE: L_required Methodology Correction (2026-08-31, OPTION A)

**The v3.1 L_required fix documented above has been superseded by a more fundamental methodology correction (2026-08-31).** Phase 3's all-latent assumption (PCM supplies 100% of night discharge alone) was replaced with a literature-anchored fractional-share model: **SHARE_PCM = 0.5**, meaning PCM supplies ~50% of delivery, tank sensible heat + concurrent charging supply the remainder (per Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021).

**Current interpretation:** `SHARE_PCM = 0.5` is active in the upstream sizing calculation. The older ≈2500 kJ/kg all-latent value and the ≈1250 kJ/kg planning estimate are superseded by the values written to the signature and feasibility artifacts. `SHARE_PCM` is defined in `config.py` (added 2026-09-07 — it was previously imported but undefined, so `04b`/`11` raised `ImportError`); `latent_heat_floor_kj_kg()` is also in `config.py` and imported by `07` and `11`. See `20_IMPLEMENTATION_ISSUES.md` §6.

## Status
**Code UNIFIED with Rajasthan (2026-09-08) — clean re-run PENDING (not executed as part of the unification).**
- `06_build_pcm_database.py` `INPUT_CSV` resolves to the repo-root `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` (55 manufacturer rows); the `is_rt_line` reference was replaced with `manufacturer` (matches Rajasthan).
- `07b_charging_feasibility.py` **deleted**; removed from `run_all_tamilnadu.py`.
- Re-run order after the unified upstream chain (Phase 3 must have produced `Tm_target_capped_C` via the `kt_worst_month` method): `06_build_pcm_database.py` → `07_feasibility_filter.py` → `08_mcdm_ranking.py` (+ `10_physics_validation.py`, `09_recommendation_cards.py`). `07` now emits **both** `feasibility_survivors_by_cluster.csv` (fixed κ=0.7) and `feasibility_survivors_by_cluster_kappa_calibrated.csv`; `08`/`09` read the `_kappa_calibrated` one.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| PCM property database | Martinez (2025) — Rubitherm measured data | `sources/Martinez2025PCM_Industrial_TES_summary.md` |
| Literature PCMs Table 2 | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Melting band 42–70°C SWH | Abdellatif (2025) PCM modeling review | `sources/Abdellatif2025PCM_Modeling_Review_summary.md` |
| Corrosion in humid climates | Hamzat (2025) PCM solar storage | `sources/Hamzat2025PCM_SolarEnergyStorage_summary.md` |
| Property imputation | Eldokaishi (2022) ANN SWH | `sources/Eldokaishi2022WaterPCM_ANN_SWH_summary.md` |
