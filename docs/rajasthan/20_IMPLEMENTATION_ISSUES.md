# 20 — Implementation Issues: Consolidated List

Ranked by severity. "Fixed" items are included because they are direct evidence of the project's own
working self-audit process and belong in a methodology write-up as such — omitting them would
understate the rigor actually demonstrated.

## Fixed, high-impact (report these as evidence of rigor)

1. **ERA5 deaccumulation bug** (`02_combine_rajasthan.py`). Assumed classic MARS
   cumulative-since-reset semantics; this pipeline's actual CDS download already returns per-hour
   flux values. Symptom: noon GHI Pearson r≈0.01 against NASA POWER. Fix (`accum_to_flux()`,
   stateless clip, no diffing): r=0.8102 post-fix. **The single most important finding in this
   audit.** See `09_ERA5_DATA_PIPELINE.md`, `14_ERA5_POWER_VALIDATION.md`.
2. **GMM covariance type** (`05_cluster_rajasthan.py`). `full`→`diag`, fixing a
   parameter-count/sample-size underdetermination that was saturating membership probabilities to
   ~1.0 regardless of true geometric separation. See `06_PHASE_4_AUDIT.md`.
3. **VIKOR sign inversion** (`08_mcdm_ranking_rajasthan.py`, 2026-08-11). Compromise index formula
   had best/worst reversed, silently inverting the entire ranking. Caught via pairwise
   method-agreement diagnostic (rho as low as −0.86 vs. TOPSIS/PROMETHEE). See `08_PHASE_6_AUDIT.md`.
4. **Entropy-weight inflation for near-empty criteria** (same file, 2026-08-11). The always-NaN
   `cost` criterion was receiving 64–75% entropy weight before the fix. See `08_PHASE_6_AUDIT.md`.
5. **Kappa-calibration inequality inversion** (`07_feasibility_filter_rajasthan.py`, 2026-08-11).
   See `07_PHASE_5_AUDIT.md`.
6. **Draw-rate units error** (`04_climate_signature_rajasthan.py`). An intermediate fix corrected a
   1000× unit error, then was itself superseded same-day by the correct total-volume (not rate)
   interpretation of the Avargani et al. (2021) benchmark. See `05_PHASE_3_AUDIT.md`.
7. **Tm_target_capped_C basis revision** (same file, 2026-08-11). `kt_p05` (worst single day) →
   `kt_worst_month` (worst calendar-month mean), after field-evidence cross-check (Nahar 2003)
   showed the single-day basis produced implausibly low caps. See `05_PHASE_3_AUDIT.md`.

## Open, high-priority

8. **PCM database undersized**: 18 rows canonical (25 counting a vestigial branch's literature
   additions), vs. the 40–60-row / 42–70°C-band target. Directly causes the next item.
9. **Zero survivors at nominal feasibility thresholds**: `L ≥ 0.7×L_required` is unreachable by any
   current candidate given the corrected `L_required` ceiling (~610–643 kJ/kg vs. best-case ~252
   kJ/kg latent heat). The pipeline currently runs on an ad hoc per-cluster κ-relaxation pass, not a
   settled policy. See `07_PHASE_5_AUDIT.md`.
10. **Quantile-mapping correction not persisted**: Phase 2's bias-correction is computed and
    reported but never written back into `climate_rajasthan_points.csv` — Phase 3+ consumes
    uncorrected (though deaccumulation-fixed) GHI. See `04_PHASE_2_AUDIT.md`, `14_ERA5_POWER_VALIDATION.md`.
11. **`T_mains_est_C = Ta_mean − 2.0` is explicitly unsourced** in-code — feeds directly into
    `L_required_kJ_per_kg`, the constraint currently driving finding #9. See `05_PHASE_3_AUDIT.md`.
12. **AHP is not actually elicited**: `AHP_PAIRWISE_MATRIX = None`; the working eigenvector/CR-check
    code exists but is never invoked. Current "AHP" weights are the framework doc's stated priors,
    unmodified. See `08_PHASE_6_AUDIT.md`.
13. **External classification validation fully stubbed**: Köppen-Geiger and NBC/ECBC ARI/NMI are
    hardcoded `None`, not fabricated but not computed either. See `06_PHASE_4_AUDIT.md`.
14. **`until phase 4/` file mislabeling**: ~15 files with filenames not matching content (three
    `.csv`-named files are PNGs; a `.py`-named file is markdown), independently confirmed via
    byte-level magic-number checks. Reproducibility/citation-integrity hazard if anyone cites a file
    path from that folder without checking actual content first. See `07_PHASE_5_AUDIT.md`,
    `21_REPRODUCIBILITY.md`.

## Open, moderate priority

15. **Monsoon month-range inconsistency**: `02_combine_rajasthan.py`'s `SEASON_MAP` (Jun–Aug) vs.
    `02b_build_daily_aggregates.py`'s `MONSOON_MONTHS` (Jun–Sep) — feeds different downstream
    features inconsistently. See `10_TEMPORAL_PROCESSING.md`.
16. **`avg_sdirswrf` unit-handling ambiguity**: three ERA5 field names (`msdwswrf`/`fdir`/
    `msdrswrf`) treated interchangeably despite differing accumulation conventions; not
    independently verified which field actually matches in the downloaded data. See
    `13_SOLAR_DERIVED_VARIABLES.md`.
17. **Dangling citation**: `04_climate_signature_rajasthan.py` references
    `Objective1_Section5_Methodology_Update.docx`, self-flagged in-code as not found in the project
    tree. See `05_PHASE_3_AUDIT.md`.
18. **Forward-dated docstring**: Correction 5 in the same file is dated 2026-08-11, one day after
    this audit's stated reference date — likely a clock/environment artifact, worth a quick
    file-timestamp sanity check.
19. **"2016-01-01 edge case" referenced but not implemented**: mentioned in three files' comments,
    but the mechanism that would have produced it (predecessor-hour dependency) no longer exists in
    the code after the deaccumulation fix. Likely a stale comment; re-verify before citing. See
    `10_TEMPORAL_PROCESSING.md`.
20. **No matched-timestamp output columns**: disables the already-written rejection-window QC plot
    and forces a cruder SZA-based proxy diagnostic in the MANUAL_REVIEW branch. See
    `10_TEMPORAL_PROCESSING.md`.
21. **`era5_CSI` QC bound looser than the pipeline's own clip** ([0,2] vs. [0,1.5]) — a dead check
    that can never fire. See `15_QUALITY_CONTROL.md`.
22. **pvlib `get_solarposition()` method not explicitly pinned**; likely SPA by default but not
    guaranteed across pvlib versions. See `12_SOLAR_GEOMETRY.md`.
23. **Duplicate cluster descriptions**: Clusters 0 and 2 receive the identical auto-generated
    qualitative label despite distinct numeric signatures — a self-flagged limitation of the
    4-axis threshold description generator. See `06_PHASE_4_AUDIT.md`.
24. **N_DRAWS=1000, not the framework doc's primary 5000**, a documented, defensible performance
    tradeoff (the framework doc itself names 1000 as an acceptable fallback) — state explicitly if
    asked. See `08_PHASE_6_AUDIT.md`.

## Open, low priority / informational

25. `ETR` (extraterrestrial radiation) computed but never written to output.
26. `bootstrap_ari_stability()` silently drops failed bootstrap resamples from its mean, without
    reporting the effective resample count.
27. `weighted_mean()` silently falls back to unweighted mean on degenerate (zero/None) weights.
28. Level B's k-scan metric table (`bic_selection`-equivalent) is never persisted to disk, unlike
    Level A's.
29. CoCoSo is fully implemented but gated off by default (`RUN_COCOSO=False`) — correct per spec,
    not an issue, listed here only for completeness of the "what exists but isn't active" inventory.
