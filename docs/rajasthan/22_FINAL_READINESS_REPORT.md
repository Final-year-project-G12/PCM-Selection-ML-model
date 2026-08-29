# 22 — Final Readiness Report

## Current implementation status

**All 8 phases (Data Collection → Recommendation Cards) are now implemented and have been run
end-to-end on real Rajasthan data**, via `run_all_rajasthan.py`, from one consistent Phase 4
clustering pass. Phase 7 (Physics Validation) returns a genuine, honestly-reported NEGATIVE result
(all three clusters' Spearman rho ≤ 0.4) — this is a real finding, not an implementation gap, and it
changes what can be claimed about the pipeline's own MCDM ranking (see "What cannot yet be claimed"
below).

**Update, 2026-08-12**: the PCM property database prerequisite that this report previously named as
the single blocking gap has been closed — expanded from 18/25 rows to 55 rows, inside the 40–60-row
target (see `07_PHASE_5_AUDIT.md`). Phases 5–8 have not yet been re-run against the expanded
database, so every phase-5-through-8 number in this report below is still the pre-expansion result.
The blocking item has moved from "expand the database" to "regenerate
`PCM_Properties_cleaned_mice_pmm_detailed.csv` (currently missing from disk) and re-run Phases 5–8" —
see "Prerequisites for a FINAL (non-provisional) result" and "Recommended next implementation" below.

## Completed phases

Phase 1 (Data Collection) — complete, 320/320 points, 240/240 ERA5 files, 3200/3200 POWER files.
Phase 2 (Preprocessing & Validation) — complete, including a caught-and-fixed critical bug. Phase 2.5
(Quality Check, previously undocumented) — complete, three sequential corrections (Hampel filter
initially over-corrected genuine cloud-driven GHI/CSI variability; fixed by excluding those two
variables from outlier detection). Phase 3 (Climate Signature) — complete, five documented
corrections, now reads the Phase 2.5 clean file. Phase 4 (Clustering) — complete, TWO
caught-and-fixed bugs (GMM covariance type, and GMM cluster-index instability across re-runs — the
second one found while building Phase 7); Köppen-Geiger external validation now wired in. Phase 5
(Feasibility Filtering) — complete as code, but its practical output (0 survivors at nominal
thresholds) directly exposes the PCM-database prerequisite gap; now stamps a cross-phase provenance
fingerprint. Phase 6 (MCDM Ranking) — complete, three caught-and-fixed bugs, running on self-flagged
provisional input; now hard-fails on a provenance mismatch. Phase 7 (Physics Validation) — complete,
two caught-and-fixed bugs in the simulation solver itself, real calibration iteration, genuine
negative result (rho = -0.900/-0.096/-0.198). Phase 8 (Recommendation Cards) — complete, pure
aggregation with its own independent cross-phase consistency re-verification.

## Strongest components

1. **The ERA5-vs-POWER cross-source validation pipeline (Phase 2).** This is the strongest single
   piece of evidence for the project's scientific rigor: it caught a real, high-impact
   preprocessing fault (deaccumulation bug, noon r≈0.01→0.81) before it silently propagated into
   every downstream climate index. This should be a headline methodology-section story, not a
   footnote.
2. **The self-auditing culture visible across the codebase, now with Phase 7 as its strongest
   example.** Seven-plus independently dated bug fixes across the whole pipeline (accum_to_flux, GMM
   covariance, VIKOR sign, entropy weight, GMM cluster-index instability, a wrong closed-form ODE
   solve, a phase-transition energy-accounting bug), each caught via a specific diagnostic the
   project built for itself — including, in Phase 7's case, MANDATORY self-tests
   (`self_test_energy_conservation()`, `self_test_draw_profile_integration()`) that must pass before
   the real simulation is even allowed to run. Each fix is documented in-code with the symptom, root
   cause, and verification. This is the kind of evidence a viva panel responds well to.
3. **The honest reporting of ambiguous and negative results, now including a genuine negative
   validation outcome.** Cluster 0's Kendall's W=0.4375, "insufficient even at κ=0" feasibility
   status, AND Phase 7's negative Spearman rho across all three clusters (-0.900/-0.096/-0.198) are
   all reported plainly, with caveat-aware interpretation logic (e.g. distinguishing "MCDM is wrong"
   from "MCDM was already unstable" for Cluster 0), not smoothed over or hidden. The PCM-vs-plain-
   tank comparator (measured ~0% against a cited +30%/+4-8% literature range) was reported as-is
   rather than tuned to match the citation.
4. **The two-tier climate signature and multi-method MCDM stack**, both correctly implemented to
   specification and both directly traceable to the framework doc's own methodological reasoning.
5. **Cross-phase provenance enforcement (new).** A real bug — Phase 5's and Phase 6's outputs
   disagreeing on which PCMs belonged to which cluster_id, traced to GMM cluster-label instability
   across separate re-runs — was caught, root-caused, and fixed with a genuine hard-fail mechanism
   (`provenance_lib.py`), not a warning that lets execution continue. `10_recommendation_cards_
   rajasthan.py` goes further and independently re-verifies the same consistency with a second,
   different check (fresh medoid recomputation) before writing anything.

## Weakest components

1. **The PCM property database — row-count gap now closed (2026-08-12), pipeline re-run still
   pending.** Expanded from 18–25 rows to 55 rows, inside the 40–60-row target. The pre-expansion
   database was structurally unable to satisfy its own nominal latent-heat feasibility constraint;
   whether the expanded database still is remains unverified, because Phases 5–8 have not yet been
   re-run against it, and the detailed imputation output
   (`PCM_Properties_cleaned_mice_pmm_detailed.csv`) both scripts read is currently missing from disk.
   This re-run is now the single item most likely to change Phase 5/6's actual numeric results.
2. **AHP weighting is not actually elicited** — presented as a TODO in code, but this distinction
   needs to be equally explicit in any write-up that describes the weighting methodology.
3. **External classification validation** — Köppen-Geiger is now wired in (real per-point lookup,
   ARI=0.19/NMI=0.32 vs. GMM); NBC/ECBC remains stubbed. Phase 4's "these are real climate regimes"
   claim now rests on internal statistics PLUS one external classification, not internal statistics
   alone — a genuine improvement, though NBC/ECBC (the India-specific classification) is still open.
4. **Two unsourced numeric choices** feed directly into load-bearing quantities: `T_mains_est_C`'s
   `Ta_mean − 2.0` offset (feeds `L_required_kJ_per_kg`, which currently drives the zero-survivor
   finding) and the Gaussian Tm-fitness `σ=4K` (feeds every MCDM method's melting-point criterion).

## Critical bugs

All bugs found during development were **fixed before being relied upon** — seven are now on record
(deaccumulation, GMM covariance, VIKOR sign, entropy weight, GMM cluster-index instability, the
Phase 7 closed-form-solve bug, the Phase 7 phase-transition energy-accounting bug) — see
`20_IMPLEMENTATION_ISSUES.md` for the full list. No unfixed critical bug is currently known in the
code. The "zero survivors" outcome (Phase 5) and the negative Spearman rho (Phase 7) are not bugs —
both are correct, self-predicted-or-honestly-reported consequences of a genuinely under-populated
PCM database, and treating them as data-completeness/database-size findings rather than code defects
is the accurate framing.

## Non-critical issues

See `20_IMPLEMENTATION_ISSUES.md` items 18–35 — monsoon-month mismatch, `avg_sdirswrf` unit
ambiguity, dangling citation, stale edge-case comment, missing matched-timestamp columns, a dead QC
bound, an unpinned pvlib call, duplicate cluster descriptions, Phase 7's near-0% PCM-vs-plain-tank
comparator and un-attempted TRNSYS cross-check (both diagnosed and reported honestly, not defects),
and several silent-fallback patterns that are low-risk given current data but worth tightening.
(The previously-listed "forward-dated docstring" item is now resolved — see item 21 there.)

## Scientific risks

The two most consequential open scientific decisions are: (1) whether to apply the Phase-2
quantile-mapping GHI correction upstream before Phase 3 consumes it (currently not applied), and (2)
what the permanent policy should be for the latent-heat feasibility constraint given it is currently
unreachable at its nominal threshold (accept calibrated-κ, or switch to rank-by-proximity). Neither
is a defect in what exists — both are unresolved methodological choices that should be made
explicitly, with a stated justification, before Phase 6's output is presented as final.

## Reproducibility risks

No pinned dependency versions (`requirements.txt` absent), no explicit ERA5 product-version/pull-date
manifest, one unpinned `pvlib` solar-position method call, and the `until phase 4/` folder's
file-content mislabeling (real, but does not affect the live pipeline's own reproducibility since it
doesn't read from that folder). See `21_REPRODUCIBILITY.md` for the full checklist and fixes.

## Missing validation

External climate-classification validation: Köppen-Geiger is now wired in (ARI=0.19, NMI=0.32 vs
GMM); NBC/ECBC Indian climate-zone classification remains stubbed. Physics-based simulation
validation (Phase 7) is now implemented and run — but returned a NEGATIVE result, which itself
becomes a claims-boundary item (see below), not a gap to fill. The framework doc's own stated bar for
a publishable result ("externally and physically validated") is now partially met: physically
validated (yes, with a negative outcome that needs honest treatment), externally classification-
validated (partially — Köppen only).

## Missing literature support

No dedicated methodology citations currently in `references.bib`/`.claude/references.md` for: SPA
(Reda & Andreas 2004), Ineichen clear-sky (Ineichen & Perez 2002), pvlib (Holmgren et al. 2018),
TOPSIS/PROMETHEE/VIKOR/GRA originating papers, MICE imputation (van Buuren & Groothuis-Oudshoorn
2011), or quantile mapping (e.g. Cannon et al. 2015) — see `17_LITERATURE_MAPPING.md` for the full
gap list and recommended additions. The PCM-domain literature base (`Sources/`, 21 papers) is strong
and well-matched to the project's PCM-selection claims specifically. **Phase 7's own citations are
now real and verified this session**: Barqawi (2025, DOI-verified) for the lumped-enthalpy ODE
structure, Bony & Citherlet (2007, independently confirmed) for the model-class justification — both
added to `17_LITERATURE_MAPPING.md`.

## What can already be used in the thesis

The full Phase 1–8 methodology narrative, the deaccumulation-bug-catch story (a genuinely strong
result), the k=3 Rajasthan clustering result with its statistical justification, the MCDM
methodology description (four methods, Monte Carlo, honest Kendall's W reporting), the Phase 7
physics-simulation methodology (lumped-enthalpy model, two self-caught numerical bugs, real
calibration iteration against literature bands) — all of this is real, defensible, and ready to write
up, **with the caveats above stated explicitly rather than omitted.** Phase 7's negative result is
itself a legitimate, reportable methodological finding: the validation was performed rigorously and
the result was not reshaped to look more favorable than the numbers support — this is exactly the
kind of honest-negative-result reporting the framework doc's own §10 asked for.

## What cannot yet be claimed

That the current Top-3 PCM recommendation per cluster is final (it is provisional pending re-run
against the now-expanded 55-row database — the expansion itself is complete, the re-run is not), that
the clustering result is externally validated against a complete set of independent
classifications (Köppen only, NBC/ECBC still stubbed), that AHP pairwise elicitation informed the
criterion weights (it did not — Table 13 priors were used unmodified), or — the change from the
previous version of this report — **that the MCDM ranking has been confirmed by physics simulation**.
It has been TESTED (Phase 7 exists and ran), but the result does not confirm it: Spearman rho is
≤0.4 (a genuine negative result) for all three clusters. The correct claim is "the MCDM ranking was
physics-validated and the validation returned a negative result at the pipeline's current PCM-
database size," not "the MCDM ranking is physics-validated."

## Prerequisites for a FINAL (non-provisional) result

(1) **PCM database expansion to the 40–60-row target — DONE (2026-08-12): 55 rows.** What is not yet
done is propagating that expansion through the pipeline: regenerating
`PCM_Properties_cleaned_mice_pmm_detailed.csv` (currently missing from disk) and re-running Phases
5–8 against it. This is now the single blocking item, replacing the expansion task itself — with
Phase 7 evidence that it matters even more than previously known: Cluster 0's negative rho may be
attributable to its undersized (pre-expansion) candidate pool (n=5) rather than a genuine MCDM/physics
mismatch, and the PCM-mass sensitivity check shows the physics ranking is stable regardless of PCM
sizing — meaning the database, not a simulation parameter, is the likely lever that would actually
change the result. (2) A settled feasibility-constraint policy (accept calibrated κ, or
rank-by-proximity). (3) Ideally, resolution of the quantile-mapping-correction-application question,
since Phase 7's calibration benchmarks are sensitive to the GHI values driving the simulated solar
resource. (4) NBC/ECBC external validation, if time permits.

## Recommended next implementation

All 8 phases exist and run end-to-end. The PCM database expansion is finished (55 rows). In order:
regenerate `PCM_Properties_cleaned_mice_pmm_detailed.csv` (`python PCM_data/PCM_data/01_preprocess.py`
— currently missing from disk) → re-run `07 → 08 → 09 → 10` (`python run_all_rajasthan.py --from
07_feasibility_filter_rajasthan.py`) against the expanded database → see whether the negative Phase 7
rho persists → decide and document the feasibility-constraint policy → (optional) wire in NBC/ECBC
external validation for Phase 4.

## Final verdict

**READY WITH MINOR FIXES** for Phases 1–4 (the ERA5/climate-signature/clustering pipeline this audit
was centrally scoped around) — the deaccumulation-bug catch and its fix, the GMM covariance and
cluster-relabeling fixes, and the overall two-tier signature/clustering methodology are sound,
well-validated internally and (partially) externally, and substantively ready for a methodology
write-up with the stated open citations added.

**READY, WITH THE NEGATIVE RESULT STATED PLAINLY** for Phase 7 as a piece of methodology — the
simulation code itself is now well-validated (self-tests, calibration against literature bands,
two caught-and-fixed numerical bugs) and ready to describe in full. What is NOT ready is treating its
OUTPUT (the negative Spearman rho) as final, because it rests on the same undersized PCM database
that already limits Phases 5–6.

**NOT READY YET, BUT THE BLOCKING FIX IS NOW ONE RE-RUN AWAY** for Phases 5–8 as a *final* result —
not because the code is wrong (all four phases run correctly and their bugs are fixed), and no longer
because the PCM database input is too small (that gap closed 2026-08-12, 18/25→55 rows). What remains
is mechanical, not scientific: regenerate the missing `PCM_Properties_cleaned_mice_pmm_detailed.csv`
and re-run Phases 5–8 against the expanded database. Every number currently on disk (zero survivors at
nominal thresholds, provisional tags on every Phase 6/7/8 row, the negative physics-validation result)
is from the pre-expansion run and should be treated as superseded, not final, until that re-run
happens. This is not a discouraging finding: it is the pipeline correctly reporting its own current
limitation, at every layer it was asked to check, which is exactly what a well-instrumented
methodology should do — the limitation has just moved from "not enough data" to "haven't re-run with
the new data yet."
