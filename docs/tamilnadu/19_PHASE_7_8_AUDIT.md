# 19 — Phase 7 & 8 Audit: Physics-Based Validation and Output

The Tamil Nadu pipeline has fully implemented both phases.

## Phase 7: Grey-Box Physics Validation (`10_physics_validation.py`) — current run
1. **Model Structure**: 3-phase lumped-enthalpy tank (Barqawi 2025): sensible solid → isothermal melting → sensible liquid.
2. **Numerical Method**: Backward Euler (implicit), hourly `dt = 3600 s`.
3. **v3.1 fix**: Ambient tank heat loss `UA_TANK_W_K = 2.0 W/K` added to prevent artificially high solar fractions and enable PCM cycling.
4. **v3.2 fix (this audit)**: the v3.1 fix alone did **not** actually work — two independent bugs in the backward-Euler solver were silently overriding it and reproducing the exact pre-v3.1 failure signature. Both are now fixed; see `20_IMPLEMENTATION_ISSUES.md` issues 6–7 for the full derivation and numerical proof.
   - A spurious extra term in the closed-form `Tw_new` solve (phases 1 and 3) — the same bug class the Rajasthan pipeline's `physics_lib.py` documents as "a wrong closed-form backward-Euler solve... caused unbounded temperature blow-up."
   - No night/idle isolation of the collector-tank coupling, letting the tank drain heat back out through the idle collector loop overnight at nearly the daytime charging rate — the second bug class from the same Rajasthan audit ("Barqawi's bidirectional a·(Tc−Tw) term").
5. **Current validation outcome (post v3.2 fix, current on-disk artifacts)**:
   - Spearman ρ by cluster: cluster 0 = **-0.016**, cluster 1 = **+0.717** (partial agreement, p=0.030), cluster 2 = **+0.355**, cluster 3 = **-0.171**, cluster 4 = **0.000**. Mean **+0.177** across clusters (was mean **-0.151** pre-fix). Overall still weak/mixed agreement — an honestly-reportable finding per Table 17, not evidence the fix failed.
   - Solar fractions now span approximately **30.5%–80.1%** (was pinned at 85.3–99.6%); **41%** of simulations fall within the published 54–84% benchmark band (was 0%).
   - Complete cycles/year now range **3–260** (was 0–1), consistent with real annual PCM freeze-melt cycling.
   - The remaining 59% outside the benchmark band is a parameter-calibration question (stated tank/collector assumptions — `M_W_KG`, `A_C_M2`, `COLLECTOR_EFF`, draw schedule — are literature-anchored, not fit to this pipeline's own data), not a further solver bug. Per this script's own printed guidance, do not hand-tune these purely to force more runs into band.

## Corrected Root Causes (v3.1 + v3.2)
| Cause | Fix | Version |
|---|---|---|
| Disabled latent-heat constraint | 300 L/day draw → realistic L_required | v3.1 |
| Missing tank heat loss | UA_TANK_W_K = 2.0 W/K | v3.1 |
| GHI feature contamination | accum_to_flux + quantile mapping | v3.1 |
| Spurious term in backward-Euler `Tw_new` closed form | Numerator corrected to use old `Tp` only | v3.2 |
| No night/idle collector-coupling isolation | `NIGHT_ISOLATION_FRACTION = 0.05` gates `a` when Tc < Tw | v3.2 |

## Phase 8: Recommendation Cards (`09_recommendation_cards.py`)
- Aggregates cluster profiles, MCDM rankings, physics validation, Monte Carlo stability into `recommendation_cards.md`.
- Re-run `09` after `10` to include updated Spearman ρ and solar fractions.
- The current cards were regenerated after the updated ranking and physics runs and contain five cluster recommendations.

## Status
**COMPLETE for the current generated artifacts (v3.2).** Re-run `10` → `09` whenever the PCM database, climate signatures, or ranking outputs change.

## Still Open (physics validation specifically)
- Solar-fraction calibration: 59% of runs still fall outside the 54–84% benchmark band; the model is now structurally correct (no solver bugs) but the stated tank/collector parameters are not empirically fit to Tamil Nadu deployments.
- No mandatory self-test (energy-conservation check under constant solar / no draw) exists in this script, unlike Rajasthan's `physics_lib.py`, which runs one before trusting any real result. Recommended if further changes are made to the solver.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| Grey-box tank ODE | Barqawi (2025) | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
| Solar fraction benchmark 54–84% | Singh et al. (2025) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Spearman rank validation | Framework doc §10 | `17_LITERATURE_MAPPING.md` |
| Backward Euler stability | Ghodusinejad (2026) — physics-informed models | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Recommendation cards | Odoi & Yorke (2025) AI SWH review | `sources/OdoiYorke2025AI_SWH_Review_summary.md` |
