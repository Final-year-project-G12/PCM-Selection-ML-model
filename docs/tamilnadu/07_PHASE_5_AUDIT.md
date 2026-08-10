# 07 — Phase 5 Audit: Feasibility Filtering

Scripts: `06_build_pcm_database.py`, `07_feasibility_filter.py`.

## Purpose
Hard-screen candidate PCMs from a database against each cluster's climate-adaptive targets (melting point and latent heat) to ensure only physically viable PCMs proceed to ranking.

## The PCM Database
- Imputes missing manufacturer properties (Rubitherm RT and Pluss savE) using a custom 3-donor PMM-like blend with Random Forests.
- Appends 7 literature PCMs (fatty acids and paraffins) with unknown values left as NaN.
- Total candidates: **25 PCMs**.

## Screen Constraints (Table 12)
1. **Melting window**: `Tm ∈ [Tm_target - 5, Tm_target + 8]°C` (relaxable by ±2K up to 4 steps if survivors < 5).
2. **Absolute band**: `Tm ∈ [42, 70]°C`.
3. **Latent heat floor**: `L >= 0.7 * L_required`.
4. **Cycling stability**: `cycles >= 300` (flagged, not excluded if NaN).
5. **Supercooling veto**: `supercooling <= 8K` (flagged, not excluded if NaN).
6. **Corrosion veto**: Excludes `check_manually` class PCMs in high-humidity clusters (`HSI` > 75th percentile).
7. **Safety exclusion**: Vetoes flammability keywords (toxic, highly flammable).

## Critical Audit Findings
- **All Clusters Have Exactly 7 Survivors**:
  Because `L_required` was calculated with the 1000x underestimation error (~52 kJ/kg), the latent heat floor is only `0.7 * 52 = 36.4` kJ/kg. Since the lowest latent heat in the database is 160 kJ/kg, **all 25 candidates easily pass the latent-heat filter**.
- The survivors are determined solely by the melting window and absolute band constraints:
  - `Tm_target = 57.0°C`
  - Nominal melting window: `[52.0, 65.0]°C`
  - Candidates in this window: `'Palmitic-Stearic eutectic (64.2/35.8)'` (52.3°C), `'Myristic acid'` (53.0°C), `'RT54HC'` (54.0°C), `'RT55'` (55.0°C), `'Palmitic acid'` (63.0°C), `'RT64HC'` (64.0°C), and `'Paraffin wax (generic)'` (64.0°C).
  - This yields exactly **7 survivors** for all 5 clusters. No kappa-relaxation was triggered.
- **Corrosion/Safety Vetoes**: Salt hydrates are missing from the database (only `savE HS36` is present but excluded by temperature), so the corrosion veto never active.

## Status
**COMPLETE BUT SILENTLY BYPASSED** (Filter logic works but was fed a buggy target, bypassing the latent heat constraint).
