# 18 — Research Gap Mapping

## Important disambiguation (repeated from `01_PROJECT_CONTEXT.md` because it matters here specifically)

Two distinct gap/novelty systems exist in this project's documents, and this audit keeps them
separate rather than merging them into one table, because merging would misattribute a claim neither
document actually makes:

- **N1–N6** (framework doc §3, Table 3): Objective 1's own novelty positioning, specific to the
  climate-signature/clustering/MCDM/validation pipeline this audit covers.
- **RG1–RG5** (`prompt for extraction.txt`, the literature-summary template): research gaps for the
  **broader, multi-objective project** (this objective plus downstream DRL-control and
  hardware-prototype objectives). RG1–RG5 do not appear anywhere in
  `Objective1_PCM_Climate_Framework_Plan_v3.docx` itself.

## Phase → N (novelty claim) mapping

| Phase | Primary N-claim(s) | How it contributes |
|---|---|---|
| 1 — Data Collection | N6 | Population-weighted, sun-event-aligned sampling — not a uniform grid or arbitrary city list |
| 2 — Preprocessing & Validation | (supports all) | The deaccumulation-bug catch and QUANTILE_MAP decision are the evidentiary basis for claiming the climate backbone (Phases 3+) is trustworthy — without this phase, none of N1–N5 would be defensible |
| 3 — Climate Signature | N2, N3 | Two-tier signature (not a single temperature); Tm_target/L_required corrected to the 42–70°C SWH band (not the 18–28°C comfort band a naive approach might reuse) |
| 4 — Regime Clustering | N1 | GMM-discovered regimes (k=3, statistically selected, not hand-picked); external validation now PARTIALLY wired in (Köppen-Geiger, ARI=0.19/NMI=0.32) — **N1's "discovered, not hand-picked" claim is now supported by internal statistical measures PLUS one external classification cross-check (NBC/ECBC still open)** |
| 5 — Feasibility Filtering | N3 (partial) | Enforces the corrected 42–70°C band and SWH-specific constraints; **currently undermined by database size** — N3's practical value depends on having enough real candidates in-band to filter, which is not yet true (18–25 of 40–60 target rows) |
| 6 — MCDM Ranking | N4 | Four-method consensus + Monte Carlo, not a single TOPSIS winner; Kendall's W explicitly reports when consensus is *not* strong (Cluster 0, W=0.4375) rather than hiding disagreement — this honest reporting is itself part of N4's value proposition |
| 7 — Physics Validation (COMPLETE, 2026-08-11) | N5 | Independently validated the MCDM ranking against simulated solar fraction — **the result is a genuine NEGATIVE validation (Spearman rho ≤0.4, all 3 clusters)**, not a confirmation. This is itself evidence for N5 as a methodology (the validation was performed rigorously and reported honestly, exactly per the framework doc's own "write it out plainly" instruction) even though it does not currently confirm the MCDM ranking's output — N5's claim should read "the ranking WAS physics-tested, honestly, with a negative result attributable in part to the still-undersized PCM database" not "the ranking IS physics-validated." See `19_PHASE_7_ONWARD.md`. |
| 8 — Recommendation Cards (COMPLETE) | (packaging) | Aggregates N1–N5's evidence, including Phase 7's negative result and its caveats, into the final deliverable format — `10_recommendation_cards_rajasthan.py`'s own caveats section surfaces the physics-validation band per cluster, not just the MCDM Top-3 |

## Phase → RG (broader project research gap) mapping — explicitly marked as indirect

Since RG1–RG5 belong to the *broader* multi-objective project rather than Objective 1 itself, this
mapping describes how Objective 1's output **feeds** the later objectives that directly address
RG1–RG4, and how Objective 1 itself directly addresses RG5:

| Phase | Related RG | Nature of contribution |
|---|---|---|
| 1–2 (Data Collection, Validation) | RG5 | Supplies the validated, uncertainty-characterized climate data a later predictive-optimization-under-uncertainty component (RG5, "no predictive optimization under climatic uncertainty") would need as its own input |
| 3–4 (Signature, Clustering) | RG5 | Climate regimes are themselves a climatic-uncertainty-aware framing (population-weighted, statistically validated) — a direct, not merely feeding, contribution to RG5 |
| 5–6 (Feasibility, MCDM) | RG5 | Monte Carlo uncertainty propagation over PCM property/weight perturbation is Objective 1's own predictive-optimization-under-uncertainty contribution |
| 7 (Physics Validation) | RG4 (indirect) | A grey-box simulation is not a real-world experiment, but it is Objective 1's step toward the experimental-validation direction RG4 (limited real-world experimental validation) ultimately calls for — the framework doc itself frames Phase 7 as "what makes the result publishable, not skippable" |
| 8 (Recommendation Cards) | RG2, RG3 (feeding, not addressing) | The per-regime PCM recommendation is the direct input a later hardware-prototype objective (RG2) and demand-alignment objective (RG3) would consume — Objective 1 does not itself build a prototype or model household demand |
| — | RG1 | **Not addressed by Objective 1 at all** — real-time adaptive control is explicitly out of this objective's scope (framework doc §1.2) |

## What this table is not claiming

This mapping does not assert that Objective 1 "solves" RG1–RG4 — only RG5 is directly addressed by
this objective's own methodology (Monte Carlo uncertainty propagation, regime-level rather than
single-point climate targets). RG1–RG4 are gaps the *broader* project addresses across multiple
objectives, and Objective 1's role there is to produce a validated, regime-aware PCM recommendation
that the later objectives can build on — not to close those gaps itself. Presenting this table with
that distinction intact is more defensible in a viva than claiming Objective 1 single-handedly
addresses all five research gaps.
