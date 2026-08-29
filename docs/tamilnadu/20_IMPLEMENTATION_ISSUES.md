# 20 — Implementation Issues and Troubleshooting

All five critical issues identified in the v3.0 audit have been **corrected in code (v3.1)**. Processed outputs under `data/processed/` still reflect pre-fix runs until the pipeline is re-executed from `02_combine_tamilnadu.py` onward.

### 1. Deaccumulation Bug (Phase 2 / Step 2) — **CORRECTED**
- **Was**: `deaccumulate()` in `02_combine_tamilnadu.py` applied `diff()` between consecutive downloaded hours. CDS point downloads already return hourly fluxes, so diffing corrupted GHI (noon r ≈ 0.40 vs NASA POWER).
- **Fix applied**: Replaced with `accum_to_flux(s) = s.clip(lower=0)` in `02_combine_tamilnadu.py`, matching the Rajasthan fix.
- **Re-run required**: `02_combine_tamilnadu.py` → full downstream chain.

### 2. Missing Quantile-Mapping Bias Correction (Phase 2 / Step 7) — **CORRECTED**
- **Was**: `04_preprocess_tamilnadu.py` did not implement quantile mapping; corrupted GHI was normalized and clustered directly.
- **Fix applied**: Step 2b per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution in `04_preprocess_tamilnadu.py`. New script `03b_agreement_analysis.py` provides the cross-source decision gate (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW).
- **Re-run required**: `03b_agreement_analysis.py` (optional QA) → `04_preprocess_tamilnadu.py` → downstream.

### 3. 1000× Flow Rate Unit Error (Phase 3 / Target Derivation) — **CORRECTED**
- **Was**: `DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60` (0.001 kg/s) made `L_required` ≈ 52 kJ/kg, bypassing the latent-heat floor.
- **Fix applied**: `04b_climate_signature.py` now uses `DRAW_VOLUME_L = 300` (Avargani et al. 2021). `11_level_b_seasonal_analysis.py` updated to the same formula.
- **Re-run required**: `04b_climate_signature.py` → downstream.

### 4. GMM Covariance Overfitting (Phase 4 / Clustering) — **CORRECTED**
- **Was**: `covariance_type="full"` on 133×27 features overdetermined the model (membership saturation).
- **Fix applied**: `05_cluster_tamilnadu.py` uses `covariance_type="diag"`.
- **Re-run required**: `05_cluster_tamilnadu.py` → downstream.

### 5. Physics Tank Model Simplification (Phase 7 / Physics Validation) — **CORRECTED**
- **Was**: `10_physics_validation.py` omitted ambient tank heat loss → solar fractions 90–99%, 0–1 PCM cycles/year.
- **Fix applied**: Convective loss term `UA_TANK_W_K = 2.0 W/K` added to the backward-Euler solver.
- **Re-run required**: `10_physics_validation.py` → `09_recommendation_cards.py`.

## Literature Support
| Issue | Corrective Method | Source |
|---|---|---|
| Deaccumulation | CDS point-download flux convention | Ghodusinejad et al. (2026) — reanalysis vs satellite GHI validation; `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Quantile mapping | Empirical bias correction | Mansouri et al. (2025) — multimodal renewable forecasting; `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
| Draw volume | 300 L/day domestic baseline | Avargani et al. (2021); `sources/` (cited in `17_LITERATURE_MAPPING.md`) |
| GMM covariance | Diagonal regularization for small-n | Liu et al. (2025) — AI PCM TES prediction; `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
| Tank heat loss | Grey-box lumped-enthalpy ODE | Barqawi (2025); `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
