# 19 — Phase 7 & 8 Audit: Physics-Based Validation and Output

The Tamil Nadu pipeline has fully implemented both phases.

## Phase 7: Grey-Box Physics Validation (`10_physics_validation.py`) — current run
1. **Model Structure**: 3-phase lumped-enthalpy tank (Barqawi 2025): sensible solid → isothermal melting → sensible liquid.
2. **Numerical Method**: Backward Euler (implicit), hourly `dt = 3600 s`.
3. **v3.1 fix**: Ambient tank heat loss `UA_TANK_W_K = 2.0 W/K` added to prevent artificially high solar fractions and enable PCM cycling.
4. **Current validation outcome**:
   - Spearman ρ by cluster is approximately **-0.471 to 0.094**, with mean **-0.151**. This is weak agreement and does not validate the MCDM ordering.
   - Solar fractions are approximately **85.3-99.6%**; **0%** of simulations fall within the published 54-84% benchmark band.
   - Complete cycles/year remain **0-1**, so the tank assumptions require diagnosis before treating the simulated performance as calibrated.

## Corrected Root Causes (v3.1)
| Cause | Fix |
|---|---|
| Disabled latent-heat constraint | 300 L/day draw → realistic L_required |
| Missing tank heat loss | UA_TANK_W_K = 2.0 W/K |
| GHI feature contamination | accum_to_flux + quantile mapping |

## Phase 8: Recommendation Cards (`09_recommendation_cards.py`)
- Aggregates cluster profiles, MCDM rankings, physics validation, Monte Carlo stability into `recommendation_cards.md`.
- Re-run `09` after `10` to include updated Spearman ρ and solar fractions.
- The current cards were regenerated after the updated ranking and physics runs and contain five cluster recommendations.

## Status
**COMPLETE for the current generated artifacts.** Re-run `10` → `09` whenever the PCM database, climate signatures, or ranking outputs change.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| Grey-box tank ODE | Barqawi (2025) | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
| Solar fraction benchmark 54–84% | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Spearman rank validation | Framework doc §10 | `17_LITERATURE_MAPPING.md` |
| Backward Euler stability | Ghodusinejad (2026) — physics-informed models | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Recommendation cards | Odoi & Yorke (2025) AI SWH review | `sources/OdoiYorke2025AI_SWH_Review_summary.md` |
