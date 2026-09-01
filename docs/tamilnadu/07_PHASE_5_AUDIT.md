# 07 — Phase 5 Audit: Feasibility Filtering

Scripts: `06_build_pcm_database.py`, `07_feasibility_filter.py`.

## Purpose
Hard-screen candidate PCMs from a database against each cluster's climate-adaptive targets (melting point and latent heat) to ensure only physically viable PCMs proceed to ranking.

## The PCM Database
- Imputes missing manufacturer properties (Rubitherm RT, Pluss savE) via MICE+RF+PMM blend.
- Appends 7 literature PCMs (fatty acids, paraffins).
- Total candidates: **25 PCMs** (target: 40–60 — still open).

## Screen Constraints (Table 12)
1. Melting window: `Tm ∈ [Tm_target − 5, Tm_target + 8]°C` (relaxable ±2K, up to 4 steps).
2. Absolute band: `Tm ∈ [42, 70]°C`.
3. Latent heat floor: `L ≥ 0.7 × L_required` — **now binding after v3.1 L_required fix**.
4. Cycling stability: `cycles ≥ 300` (flagged if NaN).
5. Supercooling veto: `supercooling ≤ 8K` (flagged if NaN).
6. Corrosion veto: excludes `check_manually` in high-HSI clusters.
7. Safety exclusion: flammability keyword veto.

## Corrected Finding (v3.1)
- **Was**: Pre-fix `L_required` ≈ 52 kJ/kg → floor = 36 kJ/kg → all 25 candidates passed → exactly 7 survivors per cluster (melting window only).
- **After v3.1 fix**: `L_required` ≈ 2500 kJ/kg → floor ≈ 1750 kJ/kg → latent-heat filter becomes **binding**; survivor count will drop significantly after re-run.

## CRITICAL UPDATE: L_required Methodology Correction (2026-08-31, OPTION A)

**The v3.1 L_required fix documented above has been superseded by a more fundamental methodology correction (2026-08-31).** Phase 3's all-latent assumption (PCM supplies 100% of night discharge alone) was replaced with a literature-anchored fractional-share model: **SHARE_PCM = 0.5**, meaning PCM supplies ~50% of delivery, tank sensible heat + concurrent charging supply the remainder (per Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021).

**What this means for Tamil Nadu:**
- **Old v3.1 L_required:** ~2500 kJ/kg (all-latent ceiling; unreachable by any candidate)
- **New 2026-08-31 L_required:** ~1250 kJ/kg (halved, since SHARE_PCM=0.5 bakes upstream into Phase 3)
- **Phase 5 re-run expectation:** Latent-heat filter remains binding but is now achievable by some candidates; κ calibration should land in 0.5–0.7 range (not the near-zero values v3.1 implied)

**See CLAUDE.md §3.1 and `04b_climate_signature.py` docstring (corrections #4–5) for full methodology detail.** Phase 5 must be re-run against updated signatures to generate valid results.

## Status
**STALE (as of 2026-08-31) — Phase 5 must be re-run against updated signatures with corrected L_required (SHARE_PCM=0.5).** Previous v3.1 findings (all-latent L_required ≈ 2500 kJ/kg) are now superseded. Expect latent-heat filter to remain binding but achievable after re-run, with κ resetting to 0.5–0.7 range.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| PCM property database | Martinez (2025) — Rubitherm measured data | `sources/Martinez2025PCM_Industrial_TES_summary.md` |
| Literature PCMs Table 2 | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Melting band 42–70°C SWH | Abdellatif (2025) PCM modeling review | `sources/Abdellatif2025PCM_Modeling_Review_summary.md` |
| Corrosion in humid climates | Hamzat (2025) PCM solar storage | `sources/Hamzat2025PCM_SolarEnergyStorage_summary.md` |
| Property imputation | Eldokaishi (2022) ANN SWH | `sources/Eldokaishi2022WaterPCM_ANN_SWH_summary.md` |
