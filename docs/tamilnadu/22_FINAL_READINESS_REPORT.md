# 22 — Final Readiness Report: Tamil Nadu

## Current Implementation Status
The Tamil Nadu recommendation pipeline is **fully implemented from Phase 1 through Phase 8**. Unlike the Rajasthan pipeline, the validation code (`10_physics_validation.py`) and reporting cards (`09_recommendation_cards.py`) are fully built and operational.

## Strongest Components
- **Full Phase Implementation**: Complete operational loop from data download to physics-based grey-box validation and card generation.
- **Uncertainty Propagation**: The 5000-draw Monte Carlo stack provides a robust confidence metric for the Top-3 ranks.
- **Level B Seasonal Sensitivity**: Analyzes monsoon-dependent swings.

## Weakest Components / Critical Bugs
- **Active Deaccumulation Bug**: Causes ERA5 GHI to be near-zero, producing poor raw correlation (r = 0.396).
- **Quantile-Mapping Omission**: Corrupted GHI features were directly clustered.
- **1000x Flow Rate Error**: Bypassed the latent-heat floor constraint.
- **Oversimplified Physics Model**: Missing tank ambient heat loss leads to high solar fractions (~95%) and prevents PCM cycling.

## Validation Verdict
- **VERDICT: NOT READY — MAJOR FIXES REQUIRED**
- **Reasoning**: The pipeline is operationally complete but scientifically compromised. Due to the active deaccumulation bug, GMM regimes are clustered on invalid GHI coordinates. Due to the 1000x flow-rate unit error, the latent-heat filter was bypassed, and rankings were evaluated on a relaxed pool. The weak Spearman correlation (r = 0.18–0.54) in the physics validation is a direct symptom of these errors.

## Recommended Next Steps
1. **Fix Deaccumulation**: Replace `deaccumulate()` with stateless clipping in `02_combine_tamilnadu.py`.
2. **Apply Quantile Mapping**: Implement empirical quantile mapping in preprocessing.
3. **Correct Water Flow Rate**: Fix the unit error in `04b_climate_signature.py` (either correct `DRAW_RATE_KG_PER_S` to `1.0` or use a flat 300 L daily volume).
4. **Change GMM Covariance**: Change GMM covariance from `full` to `diag` in `05_cluster_tamilnadu.py`.
5. **Add Tank Heat Loss**: Add convective loss to ambient air in `10_physics_validation.py`.
