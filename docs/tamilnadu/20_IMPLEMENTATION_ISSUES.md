# 20 — Implementation Issues and Troubleshooting

All five critical issues identified in the v3.0 audit have been **corrected in code** (v3.1). A cross-check against the Rajasthan pipeline's documented bug history (2026) then found two further issues in the Phase 7 physics solver that v3.1 did not actually fix — the "missing tank heat loss" symptom persisted even after `UA_TANK_W_K` was added, because two independent solver bugs were overriding it. Both are corrected below as **v3.2**. The current processed PCM artifacts reflect the updated database, fractional-share sizing model, and corrected physics solver; regenerate the full downstream chain whenever upstream climate data or PCM inputs change.

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

### 5. Physics Tank Model Simplification (Phase 7 / Physics Validation) — **CORRECTED (v3.1), BUT INEFFECTIVE UNTIL v3.2**
- **Was**: `10_physics_validation.py` omitted ambient tank heat loss → solar fractions 90–99%, 0–1 PCM cycles/year.
- **v3.1 fix applied**: Convective loss term `UA_TANK_W_K = 2.0 W/K` added to the backward-Euler solver.
- **Discovered during this audit**: this fix alone did not change the observed output at all — every simulated PCM was still landing at 85–100% solar fraction with 0–1 cycles/year (0% in the 54–84% benchmark band), i.e. exactly the pre-fix symptom. Root cause was two further solver bugs (issues 6 and 7 below), not the loss term itself.
- **Re-run required**: `10_physics_validation.py` → `09_recommendation_cards.py`.

### 6. Backward-Euler closed-form solve error (Phase 7 / Physics Validation) — **CORRECTED (v3.2)**
- **Was**: in `simulate_pcm_swh_year()`, the pre-melt (phase 1) and post-melt (phase 3) branches solved the coupled tank/PCM implicit system with
  `Tw_new = ((Tw + dt*a*tc + loss_coeff*tamb)*(1+dt*c) + dt*b*(Tp + dt*c*Tw)) / (denom1*(1+dt*c) - dt*b*dt*c)`.
  Algebraically eliminating `Tp_new` from the two implicit equations
  (`Tw_new*denom1 = Tw + dt*a*tc + loss_coeff*tamb + dt*b*Tp_new` and
  `Tp_new*(1+dt*c) = Tp + dt*c*Tw_new`) gives a numerator of
  `(Tw + dt*a*tc + loss_coeff*tamb)*(1+dt*c) + dt*b*Tp` — using the OLD
  `Tp` alone. The `+ dt*c*Tw` term inside the `dt*b*(...)` factor is
  spurious and does not appear in the correct closed form. This is the
  same bug class the Rajasthan pipeline's `physics_lib.py` documents:
  "a wrong closed-form backward-Euler solve... caused unbounded
  temperature blow-up."
- **Numerically verified** (script's own default parameters, Tw=Tp=35°C,
  Tc=45°C, Tamb=30°C, UA=2.0 W/K, dt=3600s): the buggy formula gives
  `Tw_new = 69.2°C` — exceeding the 45°C collector, the sole heat source
  for this step, which is thermodynamically impossible for a passive
  linear coupling. The corrected formula gives `Tw_new = 44.5°C`.
- **Fix applied**: numerator corrected to `... + dt*b*Tp` (both phase 1
  and phase 3 branches).
- **Re-run required**: `10_physics_validation.py` → `09_recommendation_cards.py`.

### 7. Missing night/idle collector-coupling isolation (Phase 7 / Physics Validation) — **CORRECTED (v3.2)**
- **Was**: the collector-tank coupling coefficient `a` (used in both the
  `denom1` term and the `dt*a*tc` drive term, all three phases) was
  applied identically at every hour. At night `tc` collapses to ambient
  (`isolar=0` in `build_hourly_drivers`), so the un-isolated coupling let
  the tank drain heat back out through the idle collector loop at nearly
  the same rate it charges during the day — on top of the separate
  `UA_TANK_W_K` ambient-loss term, double-counting overnight losses. This
  is the second bug class documented in the Rajasthan audit: "Barqawi's
  bidirectional a·(Tc−Tw) term let the tank drain heat through an idle
  collector overnight nearly as fast as it charged during the day."
- **Fix applied**: added `NIGHT_ISOLATION_FRACTION = 0.05`; each hourly
  step now computes `a_eff = a if tc >= Tw else a * NIGHT_ISOLATION_FRACTION`
  and uses `a_eff` (not `a`) in `denom1`/`denom` and the `dt*a*tc` drive
  term, in all three phases. Only the collector coupling is gated — the
  PCM-tank coupling `b` is an internal exchange, not a valved external
  loop, and is left unchanged, matching Rajasthan's fix exactly.
- **Combined effect of issues 6+7**: solar fractions moved from pinned
  85.3–99.6% (0% in benchmark band) to a physical 30.5–80.1% spread
  (41% in the 54–84% benchmark band); complete cycles/year moved from
  0–1 to 3–260; mean Spearman ρ across clusters moved from -0.151 to
  +0.177.
- **Re-run required**: `10_physics_validation.py` → `09_recommendation_cards.py` (already re-run; current on-disk artifacts reflect this fix).

### 8. K_FINAL not selected by a data-driven tie-break (Phase 4 / Clustering) — **OPEN, not fixed this round**
- `05_cluster_tamilnadu.py` reports BIC/silhouette/Davies-Bouldin/Calinski-Harabasz for k=2..10 but has no bootstrap-ARI stability step, unlike Rajasthan's three-tier k-selection rule. `K_FINAL=5` is hard-coded; in the current run, k=6 (silhouette 0.305) and k=9 (0.312) both score higher than k=5 (0.262) within the accepted 0.15–0.40 band.
- **Not changed here**: re-clustering at a different k cascades through every downstream phase and changes the headline per-regime PCM recommendations (D6) — a decision for the project owner, not something to silently redo mid-audit.

### 9. No cross-phase provenance/fingerprint hard-fail (Phases 4→5→6→7→8) — **OPEN, not fixed this round**
- Rajasthan's pipeline hard-fails if a downstream phase's cluster input doesn't match what's currently on disk (sklearn's GMM cluster-index order is not guaranteed stable across separate re-runs). No equivalent check exists in the Tamil Nadu scripts. No evidence this has actually caused a mismatch in the current artifacts, but it is a real reproducibility gap if `05` is ever re-run without also re-running 07→10 in the same pass.

### 10. MCDM criteria reduced to 5 of the framework doc's 8 (Phase 6 / MCDM Ranking) — **DOCUMENTED DEVIATION, not a bug**
- `08_mcdm_ranking.py` drops `cost`, `corrosion`, and `supercooling` entirely (the database has no real cost data and only one corrosion-relevant candidate) rather than carrying them as near-zero-weight criteria the way Rajasthan does. Deliberate, self-documented in the script, but means Rajasthan's "supercooling dominates the entropy weight and the physics model can't simulate it" finding does not — and cannot — recur in this pipeline's own Phase 7 diagnosis.

## Literature Support
| Issue | Corrective Method | Source |
|---|---|---|
| Deaccumulation | CDS point-download flux convention | Ghodusinejad et al. (2026) — reanalysis vs satellite GHI validation; `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Quantile mapping | Empirical bias correction | Mansouri et al. (2025) — multimodal renewable forecasting; `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
| Draw volume | 300 L/day domestic baseline | Avargani et al. (2021); `sources/` (cited in `17_LITERATURE_MAPPING.md`) |
| GMM covariance | Diagonal regularization for small-n | Liu et al. (2025) — AI PCM TES prediction; `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
| Tank heat loss | Grey-box lumped-enthalpy ODE | Barqawi (2025); `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
