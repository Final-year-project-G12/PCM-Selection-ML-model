# 05 — Phase 3 Audit: Climate Signature Construction

Script: `04b_climate_signature.py`, `04d_signature_interactive.py`.

## Purpose
Collapse each point's 10-year hourly/daily weather into a single climate signature vector, which defines the location's climatology and determines the PCM performance targets.

## Processing Details
1. **Tier 1 (Sun-Event Statistics)**: Means and percentiles of sun-event temperatures, GHI, humidity, wind. HSI (Thom 1959 Discomfort Index).
2. **Tier 2 (Daily-Integral Merge)**: True daily integrals from `02b` — GHI, SAI, cloudy fraction, CCI, HDD18, CDD24, DTR.
3. **Derived targets (v3.1 corrected)**:
   - `Tm_target = 50.0 + 7.0 = 57.0°C`
   - `L_required = (DRAW_MASS_KG × CP_WATER × ΔT) / ASSUMED_PCM_MASS_KG`
   - `DRAW_VOLUME_L = 300` (Avargani et al. 2021 domestic baseline)
4. **Five Interaction Terms**: GHI×kt_std, DTR×cloudy_frac, RH×(Ta−Tm), wind×(Ta−Tsoil), CCI×(1−SAI).
5. **PCA Reduction**: 4 components on temperature/climate block (>95% variance).
6. **Standardization**: z-scoring for GMM clustering matrix.

## Corrected Finding (v3.1 — 1000× Flow Rate Bug)
- **Was**: `DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60` → 0.001 kg/s → `L_required` ≈ 52 kJ/kg (latent-heat filter bypassed).
- **Fixed**: `DRAW_VOLUME_L = 300`, `DRAW_MASS_KG = 300 kg` → `L_required` ≈ 2500 kJ/kg (realistic domestic scale).
- **Also fixed in**: `11_level_b_seasonal_analysis.py` (seasonal L_required uses same formula).

## Status
**COMPLETE (v3.1 fixes applied — re-run `04b` for updated signatures)**

## Literature Support
| Component | Reference | Source |
|---|---|---|
| 300 L/day draw volume | Avargani et al. (2021) | `17_LITERATURE_MAPPING.md` |
| HSI / discomfort index | Thom (1959) | `17_LITERATURE_MAPPING.md` |
| PCM melting band 42–70°C | Singh et al. (2025) Table 2 | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Worst-month sizing | Durin et al. (2018) | `17_LITERATURE_MAPPING.md` |
| Climate-feature → PCM mapping | Liu et al. (2025) | `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
