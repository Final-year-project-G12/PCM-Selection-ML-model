# 11 — Implementation Issues: Consolidated List

Ranked by severity.

## Critical — fix before first run

1. **`L_required_kJ_per_kg` unit-conversion bug, unfixed** (`04b_climate_signature.py`). Uses
   `DRAW_RATE_KG_PER_S = 60.0/1000/60` — the exact pre-correction formula Rajasthan's own code
   diagnoses and fixes. Understates the latent-heat design ceiling by roughly an order of magnitude,
   which will make Phase 5's feasibility filter pass candidates it should not. **Fix**: port
   Rajasthan's corrected formula (`NIGHT_DRAW_TOTAL_KG=300`, total-volume basis, Avargani et al.
   2021) directly. See `05_PHASE_3_AUDIT.md`.

## High priority

2. **Pipeline has never been executed end-to-end.** No `data/` folder exists. Every quantitative
   claim in this documentation set is code-derived, not measured. First real run should include
   explicit checks for items 3 and 4 below.
3. **Deaccumulation function's correctness for TN's actual CDS response is unverified.** The logic
   (`diff()` + reset-hour override) is plausible and not obviously wrong, but Rajasthan's own history
   shows this exact category of assumption can fail for a specific pipeline's download behavior.
   Should be checked with the same raw-vs-diffed diagnostic Rajasthan used, the first time TN's
   ERA5 data is actually downloaded.
4. **`covariance_type="full"` in GMM clustering has not been stress-tested at TN's dimensionality/
   sample-size ratio.** Rajasthan found `full` covariance saturates membership probabilities at
   comparable or even more favorable per-cluster sample sizes. TN's `K_FINAL=5` on ~133 points ≈ 27
   points/cluster on average — smaller than Rajasthan's pre-fix 320/3≈107 points/cluster — so the
   same underdetermination risk is plausible, not less likely.
5. **`K_FINAL=5` (single-state) / `6` (multi-region) are hardcoded placeholders**, not yet validated
   against any real BIC/silhouette scan (none exists on disk).
6. **MCDM stack is TOPSIS+GRA only**, no Monte Carlo, no PROMETHEE/VIKOR — a real, non-hidden scope
   gap relative to Rajasthan, worth closing before treating TN's results as equivalent to Rajasthan's
   in rigor.

## Moderate priority

7. **Multi-region clustering script's hardcoded Rajasthan signature-file path does not match
   Rajasthan's actual output location** — a small, concrete bug that would need fixing before a
   genuine 2-state combined run could succeed.
8. **Two unreconciled elevation concepts**: a flat 150 m constant (solar geometry) and a separate
   pressure-ratio pseudo-elevation (PCA feature) — neither is real per-point elevation, and they are
   never checked against each other.
9. **PCM database undersized**: 18–25 rows vs. 40–60 target, identical gap to Rajasthan.
10. **Charging feasibility, corrosion veto, safety exclusion are absent** from the default feasibility
    filter (honestly documented, not silently skipped) — an optional heuristic exists for charging
    feasibility only.
11. **HSI formula is unsourced** in Tamil Nadu's signature script (unlike Rajasthan's correctly-cited
    Thom 1959 THI) — a different formula despite the similar variable name; should be either sourced
    or explicitly labeled as an original index.
12. **`Tsoil_proxy_C = Ta_mean − 3.0`** (interaction-term helper) is unsourced, same category of gap
    as Rajasthan's equally-unsourced `T_mains_est_C = Ta_mean − 2.0`.
13. **GRA's `delta_min`/`delta_max` computed as global scalars, not per-criterion column-wise** — a
    specific, checkable implementation choice that should be disclosed in a methods write-up.
14. **AHP is not real pairwise elicitation** — a renormalized subset of the framework doc's Table 13
    priors, honestly labeled as a placeholder in-code, but still not project-specific AHP.

## Low priority / informational

15. `INPUT_CSV` in the PCM database builder requires manual path editing before running — a
    documented, minor first-run friction point.
16. Rolling-std and delta-feature first-occurrence NaNs are silently zero-filled in the Phase-2
    preprocessing pipeline (same pattern noted in Rajasthan's docs).
17. RH is clipped (not NaN'd) in the physical-bounds step, inconsistent with the pipeline's stated
    "never clip, always NaN" policy for other bounds violations.
18. No Level B (seasonal) clustering exists for Tamil Nadu (Rajasthan has it).
19. No external classification validation structure exists at all (Rajasthan at least has a stubbed
    `None`-valued placeholder with a cited TODO).
20. `era5-tamilnadu-pipeline`/`intlo_unna` is a separate, unrelated project (Mansouri et al.
    multimodal forecasting) that shares a similarly-named folder structure and could be confused with
    the Objective-1 Tamil Nadu pipeline documented here — worth a clarifying note wherever both are
    referenced.
