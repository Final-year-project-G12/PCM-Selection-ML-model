# 07 — Phase 5 Audit: Feasibility Filtering

Scripts: `06_build_pcm_database.py`, `07_feasibility_filter.py`.

## Purpose
Hard-screen candidate PCMs from a database against each cluster's climate-adaptive targets (melting point and latent heat) to ensure only physically viable PCMs proceed to ranking.

## The PCM Database
- Imputes missing manufacturer properties (Rubitherm RT, Pluss savE) via MICE+RF+PMM blend.
- Appends 7 literature PCMs (fatty acids, paraffins).
- Total candidates: **62 PCMs**: **55 manufacturer-derived records** completed from the MICE+RF+PMM detailed input plus **7 literature records** from Singh et al. Table 2. Manufacturer imputation flags and provenance are retained; genuinely unreported literature properties remain missing.

## Screen Constraints (Table 12)
1. Melting window: `Tm ∈ [Tm_target − 5, Tm_target + 8]°C` (relaxable ±2K, up to 4 steps).
2. Absolute band: `Tm ∈ [42, 70]°C`.
3. Latent heat floor: `L ≥ 0.7 × L_required` — **now binding after v3.1 L_required fix**.
4. Cycling stability: `cycles ≥ 300` (flagged if NaN).
5. Supercooling veto: `supercooling ≤ 8K` (flagged if NaN).
6. Corrosion veto: excludes `check_manually` in high-HSI clusters.
7. Safety exclusion: flammability keyword veto.

## Current Finding
- The feasibility output audits **62 candidates per cluster**, with pass/fail detail for every filter. Despite its filename, `feasibility_survivors_by_cluster.csv` is not survivors-only.
- Current actual survivors (`passes_all=True`) are **15, 9, 13, 13, and 9** for clusters 0-4 respectively.
- Current cluster `L_required` values are approximately **301-326 kJ/kg**. The latent-heat floor is `max(100, 0.7 × L_required)` and is achievable for a subset of candidates.

## CRITICAL UPDATE: L_required Methodology Correction (2026-08-31, OPTION A)

**The v3.1 L_required fix documented above has been superseded by a more fundamental methodology correction (2026-08-31).** Phase 3's all-latent assumption (PCM supplies 100% of night discharge alone) was replaced with a literature-anchored fractional-share model: **SHARE_PCM = 0.5**, meaning PCM supplies ~50% of delivery, tank sensible heat + concurrent charging supply the remainder (per Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021).

**Current interpretation:** `SHARE_PCM = 0.5` is active in the upstream sizing calculation. The older approximately 2500 kJ/kg all-latent value and the approximately 1250 kJ/kg planning estimate are superseded by the values written to the current signature and feasibility artifacts. See `04b_climate_signature.py` and `config.py` for the active implementation.

## Status
**COMPLETE for the current generated artifacts.** Re-run `06_build_pcm_database.py` and `07_feasibility_filter.py` whenever the PCM source or upstream climate signatures change.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| PCM property database | Martinez (2025) — Rubitherm measured data | `sources/Martinez2025PCM_Industrial_TES_summary.md` |
| Literature PCMs Table 2 | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Melting band 42–70°C SWH | Abdellatif (2025) PCM modeling review | `sources/Abdellatif2025PCM_Modeling_Review_summary.md` |
| Corrosion in humid climates | Hamzat (2025) PCM solar storage | `sources/Hamzat2025PCM_SolarEnergyStorage_summary.md` |
| Property imputation | Eldokaishi (2022) ANN SWH | `sources/Eldokaishi2022WaterPCM_ANN_SWH_summary.md` |
