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
8. **GMM cluster-index instability across re-runs** (`05_cluster_rajasthan.py`, 2026-08-11 —
   distinct from item 2's covariance-type fix). sklearn's `GaussianMixture` gives no guarantee that
   cluster index 0 refers to the same physical climate group across separate re-runs. Symptom, found
   while building Phase 7: Phase 5's and Phase 6's outputs disagreed cluster-by-cluster on which
   PCMs belonged to which `cluster_id` — Phase 5's "cluster 0" candidate set matched Phase 6's
   "cluster 2" set verbatim, and vice versa. Fix: canonical relabeling by ascending mean latitude
   right after the GMM fit, plus a hard-fail provenance-fingerprint check (`provenance_lib.py`) at
   every Phase 5→6→7→8 handoff. **Arguably the highest-impact bug caught this session** — an
   uncaught version would have silently compared one climate regime's MCDM ranking against a
   different regime's simulated physics. See `06_PHASE_4_AUDIT.md`, `19_PHASE_7_ONWARD.md`.
9. **Wrong closed-form ODE solve** (`physics_lib.py`, 2026-08-11). A backward-Euler pair-solve
   copied from a precedent script had an algebraic error, causing the simulated water temperature to
   blow up unboundedly. Caught immediately by the script's own mandatory
   `self_test_energy_conservation()` self-test (run before any real simulation is allowed). Fixed by
   re-deriving the 2x2 implicit system from scratch and verifying against an independent
   `numpy.linalg.solve`. See `19_PHASE_7_ONWARD.md`.
10. **Phase-transition energy-accounting bug** (`physics_lib.py`, 2026-08-11). The PCM's "overshoot"
    sensible energy at melt onset was being silently discarded rather than credited to the latent-
    heat accumulator, producing a ~1.4% cumulative energy-conservation error. Fixed; energy
    conservation now holds to ~2e-13 relative residual (machine precision). See `19_PHASE_7_ONWARD.md`.

## Open, high-priority

11. **[RESOLVED, prerequisite met 2026-08-12 — re-run of Phases 5-8 still pending]** PCM database
    expanded from 18 rows canonical (25 counting a vestigial branch's literature additions) to **55
    rows** (14 Rubitherm RT-line + 7 Pluss savE + 4 PCM Products Ltd + 5 PureTemp + 1 CrodaTherm + 24
    literature-sourced rows), now inside the 40–60-row / 42–70°C-band target. Still zero salt-hydrate
    or other inorganic rows, so the corrosion-veto-inertness issue (see `07_PHASE_5_AUDIT.md`) is
    unaffected. This directly caused item 12 (below) and was the leading suspect for Phase 7's
    negative Cluster-0 result (undersized n=5 pool, Kendall's W=0.4375) — **neither of those
    downstream effects has been re-checked yet**, because Phases 5–8 have not been re-run against the
    expanded database, and the imputation script's `PCM_Properties_cleaned_mice_pmm_detailed.csv`
    output (which Phase 5/6 read directly) is currently missing from disk and needs regenerating
    first (`python PCM_data/PCM_data/01_preprocess.py`). This is still the single item most likely to
    change every downstream phase's numeric result — the expansion just moved the blocker from
    "database too small" to "database expanded but not yet propagated through the pipeline."
12. **Zero survivors at nominal feasibility thresholds (pre-expansion result, not yet re-checked)**:
    `L ≥ 0.7×L_required` was unreachable by any candidate in the pre-expansion 18/25-row database
    given the corrected `L_required` ceiling (~610–643 kJ/kg vs. best-case ~252 kJ/kg latent heat).
    Whether the expanded 55-row database (item 11) contains any candidate closer to that ceiling is
    unknown until Phase 5 is re-run — not assumed either way. The pipeline currently runs on an ad hoc
    per-cluster κ-relaxation pass, not a settled policy, regardless of the outcome. See
    `07_PHASE_5_AUDIT.md`.
13. **Quantile-mapping correction not persisted**: Phase 2's bias-correction is computed and
    reported but never written back into `climate_rajasthan_points.csv` — Phase 3+ consumes
    uncorrected (though deaccumulation-fixed) GHI. See `04_PHASE_2_AUDIT.md`, `14_ERA5_POWER_VALIDATION.md`.
14. **`T_mains_est_C = Ta_mean − 2.0` is explicitly unsourced** in-code — feeds directly into
    `L_required_kJ_per_kg`, the constraint currently driving finding #12 (and, downstream, Phase 7's
    mains-temperature draw calculation too). See `05_PHASE_3_AUDIT.md`.
15. **AHP is not actually elicited**: `AHP_PAIRWISE_MATRIX = None`; the working eigenvector/CR-check
    code exists but is never invoked. Current "AHP" weights are the framework doc's stated priors,
    unmodified. See `08_PHASE_6_AUDIT.md`.
16. **External classification validation partially stubbed**: Köppen-Geiger is now wired in (ARI=0.19,
    NMI=0.32 vs GMM, real per-point lookup against Beck et al. 2018's 1-km raster) — **RESOLVED for
    Köppen**. NBC/ECBC ARI/NMI remain hardcoded `None`, not fabricated but not computed either. See
    `06_PHASE_4_AUDIT.md`.
17. **`until phase 4/` file mislabeling**: ~15 files with filenames not matching content (three
    `.csv`-named files are PNGs; a `.py`-named file is markdown), independently confirmed via
    byte-level magic-number checks. Reproducibility/citation-integrity hazard if anyone cites a file
    path from that folder without checking actual content first. See `07_PHASE_5_AUDIT.md`,
    `21_REPRODUCIBILITY.md`.

## Open, moderate priority

18. **Monsoon month-range inconsistency**: `02_combine_rajasthan.py`'s `SEASON_MAP` (Jun–Aug) vs.
    `02b_build_daily_aggregates.py`'s `MONSOON_MONTHS` (Jun–Sep) — feeds different downstream
    features inconsistently. See `10_TEMPORAL_PROCESSING.md`.
19. **`avg_sdirswrf` unit-handling ambiguity**: three ERA5 field names (`msdwswrf`/`fdir`/
    `msdrswrf`) treated interchangeably despite differing accumulation conventions; not
    independently verified which field actually matches in the downloaded data. See
    `13_SOLAR_DERIVED_VARIABLES.md`.
20. **Dangling citation**: `04_climate_signature_rajasthan.py` references
    `Objective1_Section5_Methodology_Update.docx`, self-flagged in-code as not found in the project
    tree. See `05_PHASE_3_AUDIT.md`.
21. **RESOLVED — "forward-dated docstring" concern.** A previous version of this audit flagged
    Correction 5's 2026-08-11 date as a likely clock/environment artifact. It is not: 2026-08-11 is a
    real date with many legitimate same-day fixes across this codebase (items 8-10 above, plus the
    VIKOR/entropy-weight/kappa fixes), confirmed by the volume and mutual consistency of same-day
    work across multiple independent files. No further action needed.
22. **"2016-01-01 edge case" referenced but not implemented**: mentioned in three files' comments,
    but the mechanism that would have produced it (predecessor-hour dependency) no longer exists in
    the code after the deaccumulation fix. Likely a stale comment; re-verify before citing. See
    `10_TEMPORAL_PROCESSING.md`.
23. **No matched-timestamp output columns**: disables the already-written rejection-window QC plot
    and forces a cruder SZA-based proxy diagnostic in the MANUAL_REVIEW branch. See
    `10_TEMPORAL_PROCESSING.md`.
24. **`era5_CSI` QC bound looser than the pipeline's own clip** ([0,2] vs. [0,1.5]) — a dead check
    that can never fire. See `15_QUALITY_CONTROL.md`.
25. **pvlib `get_solarposition()` method not explicitly pinned**; likely SPA by default but not
    guaranteed across pvlib versions. See `12_SOLAR_GEOMETRY.md`.
26. **Duplicate cluster descriptions**: two clusters can receive the identical auto-generated
    qualitative label despite distinct numeric signatures — a self-flagged limitation of the
    4-axis threshold description generator. See `06_PHASE_4_AUDIT.md`.
27. **N_DRAWS=1000, not the framework doc's primary 5000**, a documented, defensible performance
    tradeoff (the framework doc itself names 1000 as an acceptable fallback) — state explicitly if
    asked. See `08_PHASE_6_AUDIT.md`.
28. **Phase 7's PCM-vs-plain-tank comparator measured ~0%** against the framework doc's cited
    +30%/+4-8% literature range. Diagnosed (tank-dominated system at the pipeline's fixed 50 kg PCM
    mass), reported honestly, not tuned to match the citation — not treated as a defect, but worth
    stating explicitly in any write-up that cites the framework doc's comparator numbers. See
    `19_PHASE_7_ONWARD.md`.
29. **TRNSYS Type 860 calibration cross-check not attempted** (Phase 7) — no license/installation
    available, and no published Type 860 case with enough reported parameter detail to replicate to
    ±10% was found. Flagged explicitly rather than silently skipped; the primary calibration gate
    (54-84% solar fraction band) was still satisfied without it. See `19_PHASE_7_ONWARD.md`.

## Open, low priority / informational

30. `ETR` (extraterrestrial radiation) computed but never written to output.
31. `bootstrap_ari_stability()` silently drops failed bootstrap resamples from its mean, without
    reporting the effective resample count.
32. `weighted_mean()` silently falls back to unweighted mean on degenerate (zero/None) weights.
33. Level B's k-scan metric table (`bic_selection`-equivalent) is never persisted to disk, unlike
    Level A's.
34. CoCoSo is fully implemented but gated off by default (`RUN_COCOSO=False`) — correct per spec,
    not an issue, listed here only for completeness of the "what exists but isn't active" inventory.
35. `run_all_rajasthan.py`'s `CORE_SCRIPTS` list has `02_combine_rajasthan.py` commented out by
    design (raw-combine is treated as a one-time step once `climate_rajasthan_points.csv` exists,
    consistent with the runner excluding the 00/01 acquisition scripts too) — not a bug, but worth
    confirming intentional before relying on the runner for a from-scratch reproduction.
