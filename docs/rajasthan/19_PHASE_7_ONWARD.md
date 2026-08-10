# 19 — Phase 7 Onward: What Should Happen Next

This is derived from three consistent sources — the framework doc's own §10–§11, `phases.md`'s
PROMPT 6/7 (a Rajasthan-specific elaboration of the same spec), and this audit's own findings about
what Phase 5/6 currently produce — not invented independently. Where this audit disagrees with or
adds a caveat to the existing spec, it is marked explicitly.

## THE PREREQUISITE — before Phase 7, not instead of it

**Do not run Phase 7 (physics-based validation) against the current Phase 5/6 output.** Every
survivor set Phase 6 ranks today is explicitly self-tagged
`pcm_database_status = "PROVISIONAL — ~25-row database, not yet expanded to 40-60"`, and Phase 5's
nominal-threshold filter produces **zero** survivors before ad hoc κ-relaxation is applied (see
`07_PHASE_5_AUDIT.md`). Running an expensive full-year grey-box simulation (Phase 7's own cost
estimate, per cluster per surviving candidate) against a candidate pool that will very likely change
once the PCM database reaches its 40–60-row target would mean redoing that simulation work. **The
PCM-database-expansion task (adding RT58/RT60/RT62HC, OM55/OM65, a real salt-hydrate row, confirming
fatty-acid/eutectic coverage) is therefore the actual next step, not Phase 7 itself** — it is a Phase
5 prerequisite already identified by this project's own Phase-3 docstring (Correction 4) and is
exactly the task already in progress in parallel with this audit (per the `.claude/phases.md` PROMPT
4a specification).

Second prerequisite: resolve the κ-relaxation policy explicitly (accept the calibrated-κ survivor
set as final, or switch to ranking-by-proximity-to-L_required instead of hard-gating, per Correction
4's own recommendation) — Phase 7 needs a settled, non-provisional survivor definition to simulate.

## Phase 7 — Physics-Based Validation (as specified, not yet built)

**Purpose** (framework doc §10, quoted): "Everything up to §9 produces a preference ordering.
Nothing in it establishes that a higher-ranked PCM actually performs better... This phase makes the
claim falsifiable."

**Tool choice**: a Python grey-box **lumped-enthalpy** PCM tank model — enthalpy formulation handles
latent-heat release/absorption across the melting range without tracking a moving solid-liquid front
explicitly, coupled to a lumped water-node energy balance. **EnergyPlus explicitly rejected** (no
supported way to place a latent-heat PCM inside a water-tank node network). **CFD explicitly rejected**
(out of scope for this study's compute budget and timeline). TRNSYS Type 860 named as an **optional**
cross-check, not the primary tool, "if a case with enough reported detail to replicate" can be found.

**Inputs**: the cluster medoid point's actual weather (full hourly series — pull from NASA POWER
cache since it has genuine hourly coverage, unlike ERA5's 3-samples/day scheme in this pipeline), a
**cited** standard domestic hot-water draw profile (an ASHRAE or IS-standard residential pattern —
must not be an invented schedule), and every PCM that survived Phase 5's filter for that cluster
(**not just the Top-3** — the full ordering is needed to compute the Spearman-rho validation against
the MCDM rank).

**Calibration gates — must pass before real experiments run**:
- Annual solar fraction for a calibration case in ~54–84% (typically ~69%) against a published
  benchmark.
- Within ±10% of a TRNSYS Type 860 or equivalent published case, if a sufficiently-detailed one can
  be found to replicate.
- A series PCM-tank configuration showing ~+30% solar fraction over a plain tank, +5–12% over a
  parallel configuration.
- A paraffin PCM bed sustaining ~300 L at 60±2°C for ~7 hours in a night-delivery test — **this is
  the same Avargani et al. (2021) benchmark already used to derive `L_required_kJ_per_kg` in Phase 3**,
  so Phase 7's calibration target and Phase 3's design basis are the same literature source, which is
  methodologically consistent and worth stating explicitly as a cross-phase consistency check.
- Max daily flat-plate-with-paraffin efficiency peaking near 65%.

**Real experiment**: full-year simulation per cluster per surviving PCM; record annual solar fraction
(primary metric), hours/year meeting delivery temperature, mean melt fraction, complete
charge/discharge cycle count. Compute Spearman's ρ between the MCDM consensus rank and the simulated
solar-fraction rank, per cluster.

**Interpretation — all three outcomes must be handled, not just the favorable one**:
- ρ>0.8: strong validation, MCDM is a valid low-cost proxy for full simulation.
- 0.4<ρ<0.8: partial agreement — identify which criteria likely drive the disagreement (e.g., is the
  simulation conductivity-limited while the MCDM weighting favors latent heat?).
- ρ<0.4: genuine negative result, diagnosed not discarded — identify which criterion's weight is the
  likely culprit based on where the rankings diverge most. The framework doc is explicit: "write it
  out plainly — don't reshape the interpretation to look more positive than the number supports." This
  audit endorses that instruction as good scientific practice and notes Phase 6 has already
  demonstrated this project's willingness to report an unfavorable number honestly (Cluster 0's
  Kendall's W=0.4375, below its own "ambiguous" threshold, reported rather than hidden).

**Outputs**: `physics_validation_rajasthan.csv`, `spearman_rho_by_cluster_rajasthan.csv`,
`outputs/qc_calibration_check_rajasthan.html`, `physics_validation_summary_rajasthan.txt`.

## Phase 8 — Recommendation Cards (as specified, not yet built)

**Purpose**: pure aggregation — pulls together `cluster_profile_cards_rajasthan.md`,
`mcdm_rankings_rajasthan.csv`, and `physics_validation_rajasthan.csv` into the actual results-section
content, one card per Level-A cluster.

**Required fields per card** (per `phases.md` PROMPT 7, matching framework doc §11 Table 18): cluster
identity (medoid, member count, population, mean max membership probability), full two-tier signature
(mean±std), derived targets (with system-configuration assumption stated), feasibility screening
summary (entered vs. survived, relaxation applied, per-constraint exclusion breakdown), Top-3 with
per-method ranks and Monte Carlo inclusion probability, a **signed criterion-contribution
decomposition** for each Top-3 pick (e.g. "ranked #1 primarily on melting-point fitness (+0.31) and
latent heat (+0.22), partially offset by below-median thermal conductivity (−0.08)"), simulated
performance from Phase 7, and an explicit caveats section (imputed properties among the Top-3,
relaxed feasibility window, membership ambiguity, Kendall's W agreement level — **Cluster 0 would
need this caveat given its W=0.4375**).

**Consistency requirement, worth restating**: the spec explicitly requires the cross-cluster summary
table at the top of the file to use the *same computed numbers* as the individual cards below it
("compute once, reuse, don't recompute independently in two places") — a good practice this audit
recommends enforcing with a shared helper function, consistent with how `signature_lib.py` already
avoids duplicating the Tier-1 formula between Level A and Level B.

## This audit's one addition to the existing plan

Given the confirmed provisional status of the current PCM database and survivor pool, **Phase 8's
cards should carry an explicit, prominent "provisional pending database expansion" caveat** if they
are ever generated against the current 18–25-row database, mirroring the caveat already present in
`mcdm_rankings_rajasthan.csv`'s `pcm_database_status` column — this is a small addition to the spec's
existing caveats list (imputed properties, relaxed window, membership ambiguity, Kendall's W) that
follows directly from this audit's own headline finding and should not be treated as optional once
the database work is not yet complete.
