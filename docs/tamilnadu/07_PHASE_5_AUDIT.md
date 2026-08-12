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
- **After fix**: `L_required` ≈ 2500 kJ/kg → floor ≈ 1750 kJ/kg → latent-heat filter becomes **binding**; survivor count will drop significantly after re-run.

## Status
**COMPLETE — re-run `07` after Phase 3 signature re-run**

## Literature Support
| Component | Reference | Source |
|---|---|---|
| PCM property database | Martinez (2025) — Rubitherm measured data | `sources/Martinez2025PCM_Industrial_TES_summary.md` |
| Literature PCMs Table 2 | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Melting band 42–70°C SWH | Abdellatif (2025) PCM modeling review | `sources/Abdellatif2025PCM_Modeling_Review_summary.md` |
| Corrosion in humid climates | Hamzat (2025) PCM solar storage | `sources/Hamzat2025PCM_SolarEnergyStorage_summary.md` |
| Property imputation | Eldokaishi (2022) ANN SWH | `sources/Eldokaishi2022WaterPCM_ANN_SWH_summary.md` |
