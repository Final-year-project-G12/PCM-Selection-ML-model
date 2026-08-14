# 08 — Phase 6 Audit: Multi-Criteria Ranking Engine

Script: `08_mcdm_ranking_rajasthan.py` (984 lines). **Updated 2026-08-11 — Phases 7 and 8
(`09_physics_validation_rajasthan.py`, `10_recommendation_cards_rajasthan.py`) are now also
implemented and run; this script is no longer the implementation frontier. It now also stamps a
cross-phase provenance fingerprint and hard-fails if its input doesn't match — see the new section
below.**

## Purpose

Rank each cluster's feasibility survivors using four independent MCDM methods (not one), aggregate
via two independent consensus mechanisms, and quantify ranking stability via Monte Carlo — so the
final recommendation is not an artifact of any single method's assumptions or of fixed-point
property values.

## Inputs

`feasibility_survivors_rajasthan_kappa_calibrated.csv` (or equivalent survivor set),
`cluster_profiles_rajasthan.csv`, `PCM_Properties_cleaned_mice_pmm_detailed.csv` (read directly, a
second time, for the "rich" properties — density/TC/Cp/corrosion-proxy/cost — not passed through
from Phase 5).

## Criteria (8, exact) and weights

| Criterion | Direction | AHP prior (Table 13) | Notes |
|---|---|---|---|
| Tm_fitness | benefit | 0.24 | Gaussian target-fitness transform |
| latent_heat | benefit | 0.20 | |
| vol_latent_heat (ρL) | benefit | 0.12 | |
| thermal_conductivity | benefit | 0.13 | |
| cycling | benefit | 0.11 | |
| supercooling | cost | 0.08 | |
| corrosion | cost | 0.06 (cluster-rescaled 1×–2× by HSI) | **structural proxy**: `2.0` if `pcm_type=="Inorganic"` else `1.0` — not a measured corrosion rating |
| cost | cost | 0.06 | **always NaN** — "no cost field exists anywhere in the source data" (in-code comment) |

Blend: `w_j = 0.5·w_entropy_j + 0.5·w_AHP_j`, computed **per cluster** from that cluster's own
filtered decision matrix.

## Target-based Tm handling — the part most PCM-MCDM papers get wrong, per the project's own framing

```
f_Tm = exp(-(Tm - Tm_target)² / (2σ²)),  σ = 4K
```
`σ=4K` is **explicitly sourced to the framework doc §9.2** ("justify σ=4K from the heat-exchanger
approach temperature"), not independently literature-calibrated — the code's own docstring says so
plainly. An **asymmetric** Gaussian (penalizing Tm-too-high more than Tm-too-low, physically better
motivated per the framework doc) is flagged as a documented, not-yet-implemented extension.
PROMETHEE II additionally handles Tm **natively** on raw `|Tm−Tm_target|` distance with a linear
V-shape preference function (q=2K indifference, p=8K preference threshold) — the stated reason to
keep PROMETHEE in the stack alongside the Gaussian-fitness methods.

## The four ranking methods (exact, as coded)

- **TOPSIS**: vector-normalized weighted-sum distance to ideal-best/ideal-worst, closeness
  coefficient `Ci ∈ [0,1]`; missing values excluded via `skipna=True` (not zero-filled).
- **PROMETHEE II**: net outranking flow, linear preference functions (q=0, p=criterion range) for
  all criteria except Tm (native V-shape as above); net flow normalized by `(n−1)`.
- **VIKOR**: compromise index `Q = v·(S−Sb)/(Sw−Sb) + (1−v)·(R−Rb)/(Rw−Rb)`, v=0.5.
- **GRA**: grey relational grade via ideal-reference distance with distinguishing coefficient ρ=0.5.
- **CoCoSo**: fully implemented but **gated off by default** (`RUN_COCOSO = False`) — "optional 5th
  ranker...never a replacement for the 4 core methods," per framework doc §9.4.

### Three documented, dated bug fixes (all 2026-08-11) — direct evidence of active self-auditing

1. **VIKOR sign inversion**: the compromise-index formula previously read
   `(Sb−Sw)/(Rb−Rw)` — best-minus-worst, the wrong sign — which silently *inverted the entire Q
   ranking*. Caught via a pairwise method-agreement diagnostic showing VIKOR near-totally inverted
   against TOPSIS/PROMETHEE (rho as low as −0.86) in every cluster.
2. **Entropy-weight inflation for near-empty criteria**: a criterion with too few (or zero) real
   values used to receive the *highest possible* entropy weight as an artifact of `np.nansum`
   behavior — inflating `cost`'s weight (always NaN in this database) to 64–75% across every
   Rajasthan cluster in the first run. Fixed: criteria with `<2` real values get weight `0.0`
   directly, bypassing the entropy formula.
3. **Kappa-calibration inequality inversion** (Phase 5, but caught by this script's diagnostics) —
   see `07_PHASE_5_AUDIT.md`.

All three were caught through **pairwise method-agreement or contradiction diagnostics that the
project itself built and ran** — this is exactly the kind of self-verification a methodology
write-up should cite as evidence of rigor, not omit.

## Rank aggregation

- **Borda count**: `Borda(i) = Σ_methods (n − rank_m(i))`.
- **Copeland**: pairwise win/loss majority across methods, `+1`/`−1`/`0` per pair, summed.
- **Kendall's W**: `W = 12S / (m²(n³−n))`; thresholds `W>0.8` strong, `W<0.6` ambiguous, both sourced
  to the framework doc §9.5, not the script author's own judgment.
- Where Borda and Copeland disagree on Top-3 membership/order, the design intent (per `phases.md`)
  is to flag it explicitly — confirmed present as a reported field, not silently resolved one way.

## AHP — the honest gap

`AHP_PAIRWISE_MATRIX = None` — a clearly-marked TODO stub. The eigenvector-method AHP weight
derivation with consistency-ratio check (`CR = CI/RI`, threshold `<0.10`, per framework doc §9.3)
**exists as working code but is never invoked** — the run falls through to the framework doc's
Table 13 indicative weights unmodified (except for the corrosion cluster-rescaling). **Any claim
that this pipeline performs "real AHP elicitation" would currently be inaccurate** — it uses the
framework doc's stated priors, not a project-specific pairwise comparison.

## Monte Carlo — exact numbers, and a documented deviation from the spec

```
N_DRAWS = 1000            (framework doc specifies 5000 — deviation is documented in-code:
                            a 5000-draw run took 606s wall-clock, "impractical for iteration";
                            the framework doc itself names 1000 as a safe, commonly-used fallback)
DIRICHLET_CONCENTRATION = 25.0   (chosen for ≈±20% weight variation around nominal weights)
Gaussian noise: latent_heat ±5%, thermal_conductivity ±10%, Tm ±1K (absolute), cost ±30% (moot, always NaN)
RANDOM_STATE = 42, re-seeded fresh per cluster (not a continuing stream)
```
Imputed-property handling: for a candidate flagged `any_property_imputed`, the Monte Carlo draw is
sampled from `Normal(mean, std)` of real, non-imputed values **within the same PCM family**
(falling back to all non-imputed candidates in the cluster if the family has <2 real donors) —
applied only to `latent_heat` and `thermal_conductivity`; `Tm` always uses plain ±1K noise
regardless of imputation status.

## Actual Rajasthan result — RE-RUN 2026-08-14 against the expanded 55-row database (current)

`mcdm_rankings_rajasthan.csv`: **39 rows across 3 clusters (n=9/14/16 survivors)**, up from the
pre-expansion 20 rows (n=5/8/7). Two bugs (`is_rt_line` column removed by the rewritten
`01_preprocess.py`, and a `PCM_data/PCM_data/` path-nesting mismatch) had to be fixed first to make
this re-run possible at all — see `07_PHASE_5_AUDIT.md` for the full writeup; the `family` field this
script uses for Monte Carlo same-family donor fallback now derives from the real `manufacturer` column
(6 values) rather than the old binary Rubitherm/Pluss flag.

**The dominant entropy criterion changed for every cluster**: `supercooling` now dominates all three
(Cluster 0 = 63.8%, Cluster 1 = 48.6%, Cluster 2 = 57.0%) — all three exceed the script's own 40%
"near-total-domination" flag threshold (previously it was `Tm_fitness` dominating Clusters 0/1 at
48.2%/49.4%, with `supercooling` only dominant in Cluster 2). **Kendall's W**: Cluster 0 = 0.388
(down from 0.4375, still below the 0.6 ambiguous threshold — but **no longer tagged undersized**,
n=9 now within the healthy 8–20 band, so low agreement here can no longer be attributed to sample
size), Cluster 1 = 0.635 (up from 0.536, now crosses into the "moderate" band), Cluster 2 = 0.634 (up
from 0.589, also now "moderate") — **no cluster reaches the "strong agreement" (W>0.8) band**, and
Cluster 0's persistently low W despite a healthy sample size is a new finding worth its own scrutiny
(possible genuine method disagreement on this cluster's ranking, not a data-sparsity artifact). GRA is
newly flagged by the script's own diagnostic as the "structural outlier" method (lowest mean pairwise
rho vs. the other three) in all three clusters — not previously called out by name in this file.

## Literature support

Oluah (2020) is cited by name for the TOPSIS unit-test fixture and as the domination-threshold
comparator (framework doc §13.1 names this as the project's own regression-test anchor — matches to
3 decimal places after refactoring, per `phases.md` PROMPT 5's stated verification requirement).
TOPSIS/PROMETHEE/VIKOR/GRA/CoCoSo are standard, well-established MCDM methods; no dedicated MCDM
methodology paper (e.g., for VIKOR's original formulation) was found cross-referenced in
`references.bib`/`.claude/references.md` during this audit — see `17_LITERATURE_MAPPING.md` for the
full gap analysis.

## Validation

Monte Carlo inclusion probability, Top-1 retention, rank-reversal frequency, Spearman ρ vs. baseline
— all computed and persisted per candidate per cluster. Kendall's W as a per-cluster
cross-method-agreement check. No external/physics validation yet (that is Phase 7).

## Outputs

`mcdm_rankings_rajasthan.csv`, `mcdm_method_agreement_rajasthan.csv`,
`outputs/qc_montecarlo_inclusion_rajasthan.html`.

## Cross-phase provenance stamping and hard-fail check (added 2026-08-11)

Before doing anything else, `load_survivors()` now fingerprints the CURRENT on-disk
`cluster_profiles_rajasthan.csv` (`provenance_lib.file_fingerprint()`/`fingerprint_id()`) and
compares it against the `upstream_cluster_profile_fingerprint` stamp embedded in Phase 5's survivor
file — `assert_fingerprint_match()` raises `SystemExit` (not a warning) on any mismatch. This exists
because Phase 7 caught Phase 5's and Phase 6's outputs disagreeing cluster-by-cluster on which PCMs
belonged to which `cluster_id`, traced to Phase 4's GMM cluster labels not being stable across
separate re-runs (see `06_PHASE_4_AUDIT.md`'s second documented bug and `19_PHASE_7_ONWARD.md`'s full
incident writeup). This script's own output (`mcdm_rankings_rajasthan.csv`) is now stamped with the
same fingerprint, which Phase 7 and Phase 8 each verify in turn.

## Dependencies

Requires Phase 5's κ-calibrated survivor set (itself provisional pending database expansion) and
Phase 4's cluster profiles, now verified via the provenance check above. Feeds Phase 7
(`09_physics_validation_rajasthan.py`, which computes Spearman rho between this script's Borda/
Copeland ranks and simulated solar fraction) and, via Phase 7, Phase 8
(`10_recommendation_cards_rajasthan.py`, which also re-imports this script as a module to recompute
the per-criterion contribution decomposition against its own already-saved weight formula).

## Problems / risks

- **Resolved 2026-08-14**: Phase 6 has now been re-run against the expanded 55-row database (see
  above) — the `pcm_database_status` tag on every output row now reads `"COMPLETE — 55-row
  manufacturer database..."` rather than `"PROVISIONAL — ~25-row..."`. The ranking still runs on a
  κ-relaxed rather than nominal-threshold survivor pool (that policy question remains genuinely
  open, see `19_PHASE_7_ONWARD.md`), and its output has not yet been re-validated by a Phase 7
  re-run — so "provisional pending physics validation" still applies, just not "provisional pending
  database expansion" any more.
- **`cost` and `c
orrosion` are effectively structural placeholders**, not measured criteria — a
  reader could reasonably ask why 12% of the total AHP weight budget (6%+6%) rides on data that
  doesn't exist yet for `cost` and is a binary type-proxy for `corrosion`.
- **AHP is not actually AHP-elicited** — flag this precisely in any write-up; the current weights are
  Table 13's stated priors, not a project-derived pairwise judgment matrix.
- **N_DRAWS=1000 vs the specified 5000** is a defensible, documented engineering tradeoff (the
  framework doc itself sanctions 1000 as a fallback), not a silent shortcut — but should be stated
  explicitly if a reviewer asks why the number differs from the framework doc's primary
  recommendation.

## Status

**COMPLETE as implemented, with three caught-and-fixed bugs (evidence of working self-audit) and
two structural caveats (AHP not elicited, cost/corrosion are placeholders) that should be stated
plainly rather than presented as finished.** **Update, 2026-08-14: this script has now been re-run
against the expanded 55-row database** — 39 survivors across 3 clusters (up from 20), no cluster
undersized, Kendall's W 0.388/0.635/0.634 (Clusters 1–2 now "moderate," Cluster 0 still ambiguous but
no longer explainable by small sample size). Two bugs blocking this re-run (`is_rt_line` column
removed by the rewritten preprocessing script; a `PCM_data/PCM_data/` path mismatch) were found and
fixed — see `07_PHASE_5_AUDIT.md`. **Update, 2026-08-14 (later same day): Phase 7 has now ALSO been
re-run against this fresh ranking** (`09_physics_validation_rajasthan.py`) — the negative validation
result persists (Spearman rho = -0.385/+0.125/-0.097 across the 3 clusters, mean -0.119, all still in
the ≤0.4 "genuine negative" band vs. the pre-expansion -0.900/-0.096/-0.198) — so the larger database
did **not** resolve the MCDM-vs-physics disagreement; if anything Cluster 0's now-healthy sample size
(n=9, no longer undersized) makes its persistently-low Kendall's W a more concerning finding, not a
less concerning one. Phase 8 (`10_recommendation_cards_rajasthan.py`) has also been re-run and
produced new Top-1 picks (RT50 / savE® OM50 / savE® OM50) — see `19_PHASE_7_ONWARD.md` for the full
current-state writeup. Every phase in this chain (5 through 8) is now current as of 2026-08-14; no
further re-run is pending.
