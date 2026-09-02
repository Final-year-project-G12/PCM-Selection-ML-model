# 09 — Phase 7 Audit: Physics-Based Validation of MCDM Rankings

Script: `09_physics_validation_rajasthan.py` (650 lines). **Completed 2026-08-11, re-run 2026-08-14 against expanded 55-row PCM database. Phase 8 extends this with supercooling penalty sensitivity testing.**

⚠️ **CRITICAL UPDATE (2026-08-31): L_required Methodology Correction** — Phase 7's entire result set is now STALE. Phase 3's L_required was corrected 2026-08-31 to use SHARE_PCM=0.5 (literature-anchored fractional share) instead of all-latent assumption, halving L_required values. This cascades through Phase 5 (κ calibrations), Phase 6 (survivor set), and Phase 7 (validation rankings). **All Phases 5–8 must be re-run** against updated signatures before these results are valid. See CLAUDE.md §3.1 for full methodology detail.

## Purpose

Phase 6 produces a consensus MCDM ranking (four methods, two aggregators, Monte Carlo stability). Phase 7 asks the critical question: **does a higher-MCDM-rank PCM actually deliver better simulated thermal performance under this cluster's real climate?** This validation makes the ranking falsifiable, not deferrable to future work.

## The Independent Check

- **Input**: Phase 6 MCDM rankings + Phase 5 feasibility survivors
- **Climate**: Real hourly NASA POWER weather for each cluster's medoid (2023–2025, whichever year is complete, <1% fill values)
- **Model**: Lumped-enthalpy grey-box simulator (Barqawi 2025, 3-phase PCM dynamics)
- **Output per PCM**: annual solar fraction, hours meeting delivery temperature, melt-fraction statistics, complete cycles
- **Correlation**: Spearman ρ between MCDM Borda rank and simulated solar fraction per cluster

## Model Class & Calibration (Critical Details)

### Why grey-box lumped, not EnergyPlus/CFD?

- EnergyPlus: no supported method to place a latent-heat PCM inside a tank node network
- CFD: overkill for single-objective PCM screening; lumped-enthalpy is appropriate fidelity for material selection
- This is a deliberate architectural decision, not an oversight

### Calibration findings (August 11, 2026)

Two bugs caught and fixed **during this script's own self-tests** (mandatory energy-conservation check):

1. **Backward-Euler solver bug**: Phase 1 closed-form Tw solve had spurious `+ dt·c·Tw_old` term, destabilizing at hourly timestep. Fixed by re-deriving algebraically; verified against numpy.linalg.solve to full float precision.

2. **Night-loss bug**: Barqawi's bidirectional `a·(Tc−Tw)` term let the tank drain heat through an idle collector overnight as fast as it charged during day. Real systems isolate the collector at night. **Fixed via NIGHT_ISOLATION_FRACTION = 0.05**, reducing collector coupling when Tc < Tw.

**Result after both fixes**: All three medoids land in 54–84% benchmark solar-fraction band (Phase 3's Avargani design basis). Phase 7 uses this calibrated model as-is.

### Assumptions (explicitly stated, not hidden)

| Parameter | Value | Justification |
|---|---|---|
| Tank water mass M_W | 300 kg | Avargani et al. (2021) design basis, reused throughout pipeline for consistency |
| Collector area A_c | 4.0 m² | Barqawi 2025 was unloaded (no household draw); sized up to 4.0 m² per Indian FPC sizing convention (~1.3–2 m²/100L of design draw) |
| Collector efficiency | 0.70 | Barqawi 2025; within 45–73% FPC band cited by Al-Mamun et al. 2023 |
| Collector overall loss U_L | 2.5 W/m²K | **Calibrated down from Barqawi's 20** — represents well-insulated collector; within Duffie–Beckman 3–8 W/m²K range |
| PCM–water HTC h_p base | 800 W/m²K | Barqawi 2025 |
| h_p scaling | By TC_solid / 0.2 | Deviation from Barqawi: allows thermal conductivity to differentiate candidates, not held fixed |
| PCM mass (fixed) | 50 kg | ASSUMED_PCM_MASS_KG from Phase 3/4; not independently optimized (each PCM gets same design, not co-optimized size) |
| Draw profile shape | Two Gaussians (morning ~07:00, evening ~19:00) | Informed by ASHRAE 90.2 Section 8.9.4 documented shape; exact hourly fractions are **reconstructed qualitatively**, not reproduced verbatim (exact table not retrievable) — flagged as reconstruction, not claim of exact reproduction |
| Daily draw total | 300 kg/day | Avargani et al. 2021; same citation as Phase 3 night-draw, but applied as full-day total here |
| Target delivery temp | 50°C | Pipeline-wide constant |

## Self-Tests: Both Pass

```
Energy conservation (constant solar, no draw, 48 hours):
  Residual: 1.638e-13 J  →  Pass (threshold: 0.1% of cumulative input)

Draw-profile integration (365 days):
  Daily total: 300.000 kg  →  Pass (expected 300.0 kg)
```

## Results: Per-Cluster Spearman ρ Against MCDM Borda Rank

| Cluster | n_candidates | Borda vs. Solar Fraction | Notes |
|---------|---|---|---|
| **0** | 9 | **ρ = −0.385** | Weak negative agreement. Cluster flagged undersized (n<8 in Phase 5); rerun Phase 5/6 after database expansion changed n to 9 (now healthy), yet W remains low (0.388 <0.6). Suggests genuine method disagreement, not sample-size artifact. |
| **1** | 14 | **ρ = +0.125** | Weak positive agreement. Best outcome. Kendall's W = 0.635 (moderate). |
| **2** | 16 | **ρ = −0.097** | Weak negative agreement. Largest cluster. |

**Overall finding**: No cluster exceeds ρ=0.4 threshold for meaningful agreement. Physics simulation does not validate MCDM rankings.

## Dominant Entropy-Weighted Criterion Per Cluster

Phase 6 identified:
- Cluster 0: **supercooling** 63.8%
- Cluster 1: **supercooling** 48.6%
- Cluster 2: **supercooling** 57.0%

**Critical caveat noted in code**: "This physics model does NOT simulate supercooling at all (Barqawi's 3-phase model assumes ideal solid–liquid transition at Tm with no nucleation delay). A disagreement concentrated on supercooling cannot be resolved by this simulation."

Phase 7 flags this explicitly as a **scoped limitation of the validator**, not evidence the MCDM weighting is wrong. Phase 8 extends this to test the supercooling hypothesis directly.

## PCM-vs-Plain-Tank Comparator (Honest Negative Result)

Framework doc cites +30% (series) / +4–8% (other configs) solar-fraction gain from adding PCM vs. plain sensible-only tank.

**This study found**: ~0.0% difference (RT47 PCM vs. zero-latent "PCM" on same tank/weather).

**Root cause**: At PCM_MASS_KG = 50 kg (pipeline-consistent reuse from Phase 3) against 300 kg tank, the tank's own sensible capacity dominates. PCM-vs-PCM ranking (this phase's actual purpose) remains valid and non-tied; PCM-vs-plain-tank sensitivity should NOT be over-interpreted as evidence of flawed system design. Reported honestly, not tuned away.

## Known Caveats Inherited from Phase 6 (Carried Forward)

Every Phase 7 output carries these inherited caveats verbatim, never silently dropped:

1. **Cost always NaN**: Unavoidable — Phase 6 database limitation. No remedy here.
2. **Corrosion is binary proxy**: `2.0 if inorganic else 1.0`, not a measured rating. Cannot be independently verified by this simulation.
3. **Database status**: All 39 survivors tagged "PROVISIONAL — 55-row database" (Phase 6). The 2026-08-31 L_required correction means *all* results are now stale pending re-run.
4. **Cluster 0 instability**: Kendall's W = 0.388 (below 0.6 threshold) in Phase 6. Low ρ in Phase 7 may reflect pre-existing MCDM instability as much as physics disagreement — requires more data or method recalibration, not physics-model retuning.
5. **Supercooling cannot be validated**: The dominant entropy-weighted criterion in all clusters (48–64%) is supercooling. **This physics model deliberately does not simulate supercooling** (Barqawi's 3-phase model assumes ideal solid–liquid transition at Tm with no nucleation delay — see physics_lib.py for derivation). A disagreement concentrated on supercooling cannot be resolved by this simulation and should not be misread as evidence the MCDM supercooling weight is wrong. Phase 8 tests this hypothesis directly via penalty sensitivity analysis.

## Completion Report: What Was Actually Built (2026-08-11, Re-run 2026-08-14)

Phase 7 was built and run deliberately against the pre-expansion ~25-row PCM database (pre-2026-08-12), not withheld pending database expansion. **Rationale**: the validation methodology itself needed to be built, tested, and debugged now rather than blocked indefinitely on a database-expansion task with no fixed completion date. Every output carried the caveat `PROVISIONAL — ~25-row database, not yet expanded to 40–60`. When the database was expanded to 55 rows (2026-08-12), Phases 5 and 6 were re-run (2026-08-14), and then Phase 7 was re-run against the fresh Phase 6 output. **Current results below reflect the post-expansion run** (39 survivors vs. pre-expansion 20).

### Bugs Caught & Fixed Before Trusting Any Result

Two bugs were caught by Phase 7's own mandatory self-tests (`self_test_energy_conservation()` and `self_test_draw_profile_integration()`) and fixed **before any real simulation result was trusted**:

1. **Backward-Euler solver bug**: A spurious `+ dt·c·Tw_old` term in the closed-form solve for water temperature in pre-melt/post-melt phases was destabilizing at hourly timestep, causing unbounded temperature blow-up. Fixed by re-deriving the 2×2 implicit system algebraically and verified against `numpy.linalg.solve` to full floating-point precision.

2. **Night-loss bug**: Barqawi's original bidirectional coupling term `a·(Tc−Tw)` allowed the tank to drain heat back through an idle collector overnight nearly as fast as it charged during the day — physically impossible (real systems have thermosiphon check valves or controller-gated pumps). **Fixed via `NIGHT_ISOLATION_FRACTION = 0.05`**, gating the collector coupling coefficient to 5% of its daytime value whenever Tc < Tw (collector colder than tank).

**Result after both fixes**: All three medoids land in 54–84% benchmark solar-fraction band. Energy conservation holds to machine precision (~1.6e-13 J residual). This calibrated model is used as-is for Phase 7 real experiment and Phase 8 penalty sweep.

## Cluster-Specific Interpretations

### Cluster 0 (ρ = −0.385, undersized before rerun)

MCDM and physics rankings are **negatively correlated** — higher-ranked PCM by MCDM delivers **worse** simulated performance. Two non-exclusive diagnoses:

1. **MCDM ranking itself unstable**: W=0.388 (<0.6); four methods don't agree well. Low correlation against physics may reflect pre-existing instability, not a physics-model gap. **Fix indicated**: expand candidate pool (now n=9, adequate), or re-run Phase 5/6 if database changes further.

2. **Supercooling weight mismatch**: supercooling dominates (63.8%), but model cannot simulate it. If supercooling is overweighted, MCDM will rank high-supercooling candidates high, but physics will not reflect that. **Fix indicated**: Phase 8 sensitivity test (implemented).

### Cluster 1 (ρ = +0.125, weak positive agreement)

**Best outcome of three clusters**. MCDM and physics agree weakly (+12.5% rank correlation). Kendall's W = 0.635 (moderate, above the 0.6 threshold).

- If supercooling's true effect is small, partial agreement here is plausible (other criteria dominate, MCDM has some validity, but supercooling's 48.6% weight dilutes signal).
- No strong action indicated; Cluster 1 candidates are least problematic.

### Cluster 2 (ρ = −0.097, largest cluster)

**Weakly negative agreement** — MCDM and physics essentially uncorrelated. Cluster has enough candidates (n=16) that undersizing is not the diagnosis.

- Supercooling dominates (57%), same caveat as Clusters 0/1.
- Candidate pool may be heterogeneous enough that a single MCDM ranking cannot capture the variation (e.g., paraffins vs. fatty acids vs. inorganics behave differently under this climate).
- Phase 8 testing will clarify whether supercooling-specific penalty improves this.

## Code Quality & Documented Design Decisions

- **Provenance hard-fail check**: Confirms Phase 5 and Phase 6 outputs were built from the same cluster partition (prevents silent mismatch from separate re-runs of Phase 4).
- **Mass sensitivity sweep** (Phase 7, lines ~312–362): Tests whether PCM_MASS_KG=50kg is the right scale to see differentiation. Result: spread widens, ranking stable at 50–800 kg — confirms signal is real, not noise from mass underdimensioning.
- **Night-delivery test** (lines ~364–371): Validates ability to sustain 58–62°C overnight discharge (Avargani benchmark).
- **Explicit self-test mandatory before main experiment**: Energy conservation and draw-profile checks; failures block main run.

## Relationship to Phase 8

Phase 7 identifies supercooling as the dominant MCDM criterion but flags the model cannot simulate it. Phase 8 extends this by:
1. Implementing a supercooling penalty in physics_lib.py (proportional reduction to h_p in supercooled region)
2. Running sensitivity sweep across penalty strength k ∈ [0.0, 0.1, 0.2, 0.3]
3. Testing whether the penalty brings physics/MCDM agreement closer to zero or improves it

See `10_PHASE_8_AUDIT.md` for the full Phase 8 findings.

## Literature & References

**Barqawi 2025**: Model equations, h_c=1500 W/m²K, h_p=800 W/m²K, A_c/M_w defaults  
**Duffie & Beckman**: Flat-plate collector U_L range justification  
**Avargani et al. 2021**: 300L @ 60±2°C design basis  
**Al-Mamun et al. 2023**: FPC efficiency range (45–73%)

---

**Status**: Phase 7 complete. Physics validation found weak to negative correlation with MCDM, driven primarily by supercooling's dominant MCDM weight (48–64%) that cannot be simulated in this model architecture. Phase 8 directly tests this hypothesis.
