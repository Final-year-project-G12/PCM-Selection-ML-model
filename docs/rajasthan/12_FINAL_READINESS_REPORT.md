# 12 — Final Readiness Report

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

⚠️ **CRITICAL UPDATE (2026-08-31): L_required Methodology Correction** — Phase 3's L_required methodology was corrected 2026-08-31, halving L_required values (600–650 kJ/kg → 300–325 kJ/kg) and cascading through Phases 5–8. **All results documented in this report (κ calibrations, Spearman rho values, recommendations) are now STALE.** Phases 5–8 must be re-run against updated signatures. This supersedes the "regenerate preprocessing output and re-run" item above; the NEWER prerequisite is "re-run Phases 3 (climate signature only), 4, 5–8 in sequence." See CLAUDE.md §3.1 and `04b_climate_signature.py` docstring for full methodology detail.

✅ **CRITICAL UPDATE #2 (2026-09-13): Delivery temperature corrected to match Avargani, full re-run
COMPLETE.** `T_DELIVERY_C` was `50.0°C` throughout the pipeline (Phase 3's `Tm_target_C`/`L_required`
formulas AND `physics_lib.py`'s simulator), but Avargani et al. (2021)'s cited 300 L/7h benchmark is
validated at 60±2°C — corrected to `60.0°C` in the shared `pcm_shared_config.py`. **This
supersedes every number in this report from before 2026-09-13, including the 2026-08-31 update's
own (stale) figures above.** Current state (04b → 05 → 05a → 07 → 08 → 10 → 09 re-run complete):
- `Tm_target_C`: 57.0 → **67.0°C**; `L_required`: 285–344 → **410–469 kJ/kg**.
- Phase 5 fixed-κ=0.7 survivors: back to **0/0/0** (raised ceiling); κ-calibrated survivors
  **4/8/11** (n=23 total, down from 39) — cluster 0 bottoms out `insufficient_even_at_kappa_0`.
- Phase 7 Spearman rho: **0.105 / -0.095 / -0.091** (clusters 0/1/2) — still negative-band;
  cluster 0's reading is on an undersized n=4 pool.
- Phase 8 supercooling sweep: agreement still worsens as the penalty k rises at every cluster,
  same qualitative finding as pre-fix, now on the smaller n=4/8/11 pool.
- Top-1 picks changed: Palmitic-stearic acid/Expanded graphite / PureTemp 60 / n-Heptacosane (C27)
  (clusters 0/1/2) — see `outputs/recommendation_cards_rajasthan.md` for the full cards.
See `05_PHASE_3_AUDIT.md`, `07_PHASE_5_AUDIT.md`, `08_PHASE_6_AUDIT.md`, `09_PHASE_7_AUDIT.md`, and
`10_PHASE_8_AUDIT.md` for each phase's detailed superseding section.

✅ **CRITICAL UPDATE #3 (2026-09-13, same session): Tank mass decoupled from Avargani + re-run
COMPLETE.** `M_W_KG` (simulator's static tank mass) was also wrongly reused from Avargani's 300L
figure — but that figure is a continuous-FLOW throughput volume, not a static tank capacity, in a
system that has no tank in the sense `M_W_KG` represents. Corrected 300→200 kg, grounded instead in
Eldokaishi et al. (2022)'s hybrid-PCM-SWH tank-sizing literature (50–240L for a 1–8 m² collector,
matching this pipeline's own 4 m² design case). Required re-tuning `COLLECTOR_UL_WM2K` (2.5→2.0)
and `NIGHT_ISOLATION_FRACTION` (0.05→0.03) to keep calibration in the 54–84% benchmark band (now
64.0/65.8/64.3%, closer to the 69% target than before). Phase 7 rho: 0.105/-0.190/-0.091. See
`09_PHASE_7_AUDIT.md` and `10_PHASE_8_AUDIT.md`.

✅ **CRITICAL UPDATE #4 (2026-09-13, same session): seasonal PCM sensitivity degeneracy — root
cause found and fixed; result reframed as a POSITIVE finding.** The "< 2 survivors everywhere"
result above traced to a real bug in `11_seasonal_pcm_sensitivity.py`: it re-filtered each
cluster's already-calibrated survivor pool with a hardcoded `LATENT_HEAT_FRACTION=0.7` instead of
that cluster's own Phase 5 calibrated kappa (0.0/0.5/0.3). Fixed to use the per-cluster calibrated
kappa. Re-run result: no longer degenerate (all 11 cluster-season cells rank cleanly, 4-11
survivors each) — **0/11 flip from the annual #1 pick**, a genuine null finding, not censored data:
Rajasthan's delivery-anchored `Tm_target` rule is seasonally robust.

**This does NOT weaken Objective 3's case.** Re-checked against how comparable DRL-for-solar-
thermal papers actually motivate their controllers: Emami et al. (2025/2026) — already an extracted
project source (`sources/Emami2026DRL_Solar_ORC_TES_summary.md`) — motivates their DDPG controller
purely by real-time weather stochasticity (year-long irradiance variability a fixed-flow baseline
cannot track), not by any PCM-selection instability. The corrected framing: **O1's material
selection is stable per climate region (this finding); O3's job is the hour-by-hour operating
decision under weather/demand variability that a fixed rule-based controller cannot react to** —
a stronger, better-precedented motivation than the original "PCM ranking flips" plan. Heidari et
al. (2022), cited by the reviewing party as a second precedent (stochastic demand + weather), is
NOT yet a verified project source — do not cite its specific claims in the paper until it has been
read and extracted per CLAUDE.md §0. Full detail and the recommended paper framing in
`09_PHASE_7_AUDIT.md` and `Objective1_Fixes_SourceVerified.md` Fix 6.

✅ **Fix 5 disclosures applied (2026-09-13):** `physics_lib.py`'s tank-mass/collector-calibration
documentation was rewritten to state plainly that (a) `M_W_KG=200` is chosen WITHIN Eldokaishi et
al. (2022)'s validated range, not a value that paper tests; (b) only the tank-VOLUME range is
borrowed from Eldokaishi, not its PCM material, backup heater, or climate/demand context; (c) the
`COLLECTOR_UL_WM2K=2.0` retune is calibration to restore the benchmark band, no longer
Duffie-Beckman-justified; (d) the tank-mass fix and the UL/night-isolation retune are documented as
one linked calibration event. No numeric change from this disclosure pass.

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
negative result — **rho = 0.105 / -0.190 / -0.091** (clusters 0/1/2, current post-2026-09-13 numbers;
see CRITICAL UPDATE #3 above — an earlier pre-correction reading of -0.900/-0.096/-0.198 has been
superseded throughout this report). Phase 8 (Recommendation Cards) — complete, pure aggregation with
its own independent cross-phase consistency re-verification.

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
   status, AND Phase 7's negative Spearman rho across all three clusters (current: 0.105/-0.190/-0.091,
   clusters 0/1/2 — see CRITICAL UPDATE #3) are all reported plainly, with caveat-aware interpretation
   logic (e.g. distinguishing "MCDM is wrong"
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

1. **The PCM property database — row-count gap closed (2026-08-12), pipeline re-run COMPLETE
   (2026-09-13).** Expanded from 18–25 rows to a 62-row evaluated pool (55 manufacturer + 7
   literature rows), inside the 40–60-row target. `PCM_Properties_cleaned_mice_pmm_detailed.csv`
   exists on disk (`PCM_data/data/`, 34,909 bytes) and Phases 5–8 have been re-run against it — see
   CRITICAL UPDATE #2/#3 above and `07_PHASE_5_AUDIT.md`. The remaining open item is not the database
   size but Cluster 0's small κ-calibrated survivor pool (n=4/62) — see "Scientific risks" below.
2. **AHP weighting is not actually elicited** — presented as a TODO in code, but this distinction
   needs to be equally explicit in any write-up that describes the weighting methodology.
3. **External classification validation** — Köppen-Geiger is now wired in (real per-point lookup,
   ARI=0.2787/NMI=0.3817 vs. GMM, current on-disk value); NBC/ECBC remains stubbed. Phase 4's "these are real climate regimes"
   claim now rests on internal statistics PLUS one external classification, not internal statistics
   alone — a genuine improvement, though NBC/ECBC (the India-specific classification) is still open.
4. **Two unsourced numeric choices** feed directly into load-bearing quantities: `T_mains_est_C`'s
   `Ta_mean − 2.0` offset (feeds `L_required_kJ_per_kg`, which currently drives the zero-survivor
   finding) and the Gaussian Tm-fitness `σ=4K` (feeds every MCDM method's melting-point criterion).

## Critical bugs

All bugs found during development were **fixed before being relied upon** — seven are now on record
(deaccumulation, GMM covariance, VIKOR sign, entropy weight, GMM cluster-index instability, the
Phase 7 closed-form-solve bug, the Phase 7 phase-transition energy-accounting bug) — see `00_MASTER_OVERVIEW.md` ("Current known
issues") for the full list. No unfixed critical bug is currently known in the
code. The "zero survivors" outcome (Phase 5) and the negative Spearman rho (Phase 7) are not bugs —
both are correct, self-predicted-or-honestly-reported consequences of a genuinely under-populated
PCM database, and treating them as data-completeness/database-size findings rather than code defects
is the accurate framing.

## Non-critical issues

See `00_MASTER_OVERVIEW.md` ("Current known issues") — monsoon-month mismatch, `avg_sdirswrf` unit
ambiguity, dangling citation, stale edge-case comment, missing matched-timestamp columns, a dead QC
bound, an unpinned pvlib call, duplicate cluster descriptions, Phase 7's near-0% PCM-vs-plain-tank
comparator and un-attempted TRNSYS cross-check (both diagnosed and reported honestly, not defects),
and several silent-fallback patterns that are low-risk given current data but worth tightening.
(The previously-listed "forward-dated docstring" item is now resolved — see item 21 there.)

## Scientific risks

**(1) GHI quantile-mapping — RESOLVED (confirmed 2026-09-17 audit).** This was previously listed as
an open decision ("whether to apply the Phase-2 quantile-mapping GHI correction upstream before
Phase 3 consumes it"), but code inspection shows it is already implemented and wired in:
`04_preprocess_rajasthan.py` (Phase 2.5, ~lines 170–224) fits a per-season quantile mapper (ERA5 GHI
→ NASA POWER GHI) and applies it **in place** to `era5_GHI` (propagating into `era5_CSI`), writing
the corrected values into `rajasthan_cleaned_physical.csv`; `04b_climate_signature.py` reads that
same file (`PHYSICAL_FILE = PREPROCESSED_DIR / "rajasthan_cleaned_physical.csv"`), so Phase 3
**does** consume the quantile-mapped GHI, not raw ERA5. Before/after numbers are in
`ghi_quantile_mapping_report.csv`. No further action needed; this superseded the identical stale
claim previously repeated in `CONSOLIDATION_SUMMARY.md`.

**(2) Latent-heat feasibility constraint policy — DECIDED (2026-09-17): accept calibrated-κ.**
Cluster 0's n=4 κ-calibrated survivor pool (out of 62 evaluated candidates) is accepted as the
Top-3 basis for that cluster, rather than switching to a rank-by-proximity fallback. Justification:
per-cluster investigation (`07_PHASE_5_AUDIT.md`, and the follow-up check recorded in CLAUDE.md §3.4)
confirmed Cluster 0's small pool is a genuine climate-driven finding, not a calibration artifact —
only 4 of 62 candidates clear the melting-window/charging-feasibility gate regardless of κ, and
`calibrated_kappa=0.0` correctly reports that latent heat was never the real bottleneck for this
cluster. Rank-by-proximity was rejected because it would rank candidates that fail the physical
feasibility gate rather than being transparent about the gate itself. **Stated limitation for the
write-up:** a Top-3 claim drawn from an n=4 pool has materially less statistical support than
Clusters 1/2's n=8/n=11 pools — the methodology section should flag this explicitly rather than
presenting all three clusters' Top-3 rankings as equally well-supported.

## Reproducibility risks

No pinned dependency versions (`requirements.txt` absent), no explicit ERA5 product-version/pull-date
manifest, one unpinned `pvlib` solar-position method call, and a doubly-nested `PCM_data/PCM_data/`
folder layout on disk that the PCM-database consuming scripts do not expect (worked around by a
non-destructive file copy — see `07_PHASE_5_AUDIT.md`, "Two blocking bugs"). The
resumability/provenance mechanisms are summarised in `00_MASTER_OVERVIEW.md` ("Current architecture"
→ Resumability, and known issue 11).

## Missing validation

External climate-classification validation: Köppen-Geiger is now wired in (ARI=0.2787, NMI=0.3817 vs
GMM, current on-disk value); NBC/ECBC Indian climate-zone classification remains stubbed. Physics-based simulation
validation (Phase 7) is now implemented and run — but returned a NEGATIVE result, which itself
becomes a claims-boundary item (see below), not a gap to fill. The framework doc's own stated bar for
a publishable result ("externally and physically validated") is now partially met: physically
validated (yes, with a negative outcome that needs honest treatment), externally classification-
validated (partially — Köppen only).

## Missing literature support

No dedicated methodology citations currently in `references.bib`/`.claude/references.md` for: SPA
(Reda & Andreas 2004), Ineichen clear-sky (Ineichen & Perez 2002), pvlib (Holmgren et al. 2018),
TOPSIS/PROMETHEE/VIKOR/GRA originating papers, MICE imputation (van Buuren & Groothuis-Oudshoorn
2011), or quantile mapping (e.g. Cannon et al. 2015) — see `13_LITERATURE_MAPPING.md` for the full
gap list and recommended additions. The PCM-domain literature base (`Sources/`, 21 papers) is strong
and well-matched to the project's PCM-selection claims specifically. **Phase 7's own citations are
now real and verified this session**: Barqawi (2025, DOI-verified) for the lumped-enthalpy ODE
structure, Bony & Citherlet (2007, independently confirmed) for the model-class justification — both
added to `13_LITERATURE_MAPPING.md`.

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

That the current Top-3 PCM recommendation per cluster rests on an equally well-supported candidate
pool in every cluster (Cluster 0's n=4 pool is accepted policy — see "Scientific risks" — but is
smaller than Clusters 1/2's n=8/n=11), that the clustering result is externally validated against a
complete set of independent classifications (Köppen only, NBC/ECBC still stubbed), that AHP pairwise
elicitation informed the criterion weights (it did not — Table 13 priors were used unmodified), or
**that the MCDM ranking has been confirmed by physics simulation**. It has been TESTED (Phase 7
exists and has been re-run against the 62-row database, current numbers rho = 0.105/-0.190/-0.091),
but the result does not confirm it: Spearman rho is ≤0.4 (a genuine negative result) for all three
clusters. The correct claim is "the MCDM ranking was physics-validated against the final 62-row
database and the validation returned a negative result," not "the MCDM ranking is physics-validated."

## Prerequisites for a FINAL (non-provisional) result

(1) **PCM database expansion to the 40–60-row target — DONE (2026-08-12): 62-row evaluated pool.**
`PCM_Properties_cleaned_mice_pmm_detailed.csv` is regenerated and on disk, and Phases 5–8 have been
re-run against it (CRITICAL UPDATE #2/#3, 2026-09-13) — **DONE.** (2) A settled feasibility-constraint
policy — **DONE (2026-09-17): accept calibrated-κ**, see "Scientific risks" above. (3) Resolution of
the quantile-mapping-correction-application question — **DONE**, confirmed already implemented and
consumed by Phase 3, see "Scientific risks" above. (4) NBC/ECBC external validation, if time permits
— still open, non-blocking (Köppen-Geiger external validation is already wired in).

## Recommended next implementation

All 8 phases exist, run end-to-end, and have been re-run against the final 62-row PCM database
(CRITICAL UPDATE #2/#3). The feasibility-constraint policy and the GHI quantile-mapping question are
both now decided/resolved (see "Scientific risks"). What remains is non-blocking: (optional) wire in
NBC/ECBC external validation for Phase 4, and reflect the "n=4 pool for Cluster 0" limitation
explicitly wherever the Top-3 results table is presented in the write-up.

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

**READY, WITH TWO EXPLICIT CAVEATS STATED** for Phases 5–8 as a *final* result (updated 2026-09-17;
supersedes the "NOT READY YET" verdict below, kept for change-history only). The blocking re-run is
complete: Phases 5–8 have been run end-to-end against the final 62-row PCM database (CRITICAL UPDATE
#2/#3, 2026-09-13), and both open scientific-risk items — the GHI quantile-mapping question and the
latent-heat feasibility-constraint policy — are now resolved/decided (see "Scientific risks"). The
two caveats that should still be stated plainly in the write-up: (a) Cluster 0's Top-3 rests on an
n=4 candidate pool, smaller than Clusters 1/2's n=8/n=11 — accepted policy, but a genuinely weaker
statistical basis; (b) Phase 7's physics validation returned a negative result (rho =
0.105/-0.190/-0.091), which is a real, honestly-reported finding, not a code defect, and should be
described as such rather than as confirmation of the MCDM ranking.

*(Historical verdict, superseded 2026-09-17 — kept for change-history only:)* "NOT READY YET, BUT THE
BLOCKING FIX IS NOW ONE RE-RUN AWAY" for Phases 5–8 as a final result — not because the code was
wrong, and no longer because the PCM database input was too small (that gap closed 2026-08-12,
18/25→55 rows). What remained was mechanical, not scientific: regenerate the missing
`PCM_Properties_cleaned_mice_pmm_detailed.csv` and re-run Phases 5–8 against the expanded database.
That re-run is now complete, per the update above.
