# 08 — Phase 6 Audit: Multi-Criteria Ranking Engine

Script: `08_mcdm_ranking.py`. **Updated 2026-08-11** — provenance fingerprint hard-fail wired in.

> **UNIFIED WITH TAMIL NADU (2026-09-08).** `era5-rajasthan/08_mcdm_ranking.py` and
> `era5-tamilnadu/08_mcdm_ranking.py` are now the same engine byte-for-byte apart from
> `STATE_NAME`, the processed-dir layout and two hint strings. Changes this pass — the
> pipeline has **not** been re-run, so every number below is pre-unification and stale:
> - **Output names** are Tamil Nadu's canonical set for both states, no suffix:
>   `mcdm_full_rankings.csv` (full per-survivor audit trail — was `mcdm_rankings_rajasthan.csv`),
>   `mcdm_topk_by_cluster.csv` (new Top-3 subset), `monte_carlo_stability.csv` (new, MC columns
>   only), `mcdm_method_agreement.csv`, `qc_montecarlo_inclusion.html`. References updated in
>   `08_phase8_supercooling_sweep.py`, `09`, `10`, `11`. `PLOTSV2/*.py` still use old names.
> - **Latent-heat criterion** is now climate-relative (`latent_heat / L_required` per cluster);
>   `vol_latent_heat` (ρ·L) stays separate.
> - **Cycling criterion** is now log-scaled `cycles_confidence` (`log1p(cycles)/log1p(max)`).
> - **Supercooling entropy-weight cap (NEW):** the Shannon-entropy formula overweights
>   near-zero-ideal cost criteria — supercooling reached 48–64% in the first 8-criterion run,
>   which Phase 7/8 diagnosed as the cause of the negative physics-validation correlation
>   (Phase 8's calibrated supercooling-penalty sweep *worsening* agreement confirmed
>   overweighting). Each cost criterion whose ideal ≈ 0 (supercooling; cost/corrosion if they
>   ever get real data) now has its entropy-derived weight clipped to ≤ 2× its Table-13 prior
>   (supercooling ≤ 0.16) *before* the 50/50 entropy-AHP blend, then the entropy vector is
>   renormalised. Supercooling is **not** removed — all 8 Table-13 criteria are retained.
> - **Monte Carlo draws** set to `N_DRAWS = 1000` in both states (one constant; raise both to
>   5000 for the final reported run).

## Purpose

Rank each cluster's feasibility survivors using four independent MCDM methods (not one), aggregate
via two independent consensus mechanisms, and quantify ranking stability via Monte Carlo — so the
final recommendation is not an artifact of any single method's assumptions or of fixed-point
property values.

## CRITICAL UPDATE: L_required Methodology Correction (2026-08-31)

**All Phase 6 outputs from before this date are now STALE.** Phase 3's L_required methodology was corrected 2026-08-31 (SHARE_PCM=0.5), which halves L_required and cascades through Phase 5's feasibility filtering (changing κ calibrations) and into this script's survivor input set. The survivor set fed to Phase 6 is now different; ranking results will change. **Phase 5 and 6 must both be re-run** against updated signatures before these results are valid. See CLAUDE.md §3.1 for full detail.

## Inputs

`feasibility_survivors_by_cluster_kappa_calibrated.csv` (or equivalent survivor set),
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

`mcdm_full_rankings.csv`: **39 rows across 3 clusters (n=9/14/16 survivors)**, up from the
pre-expansion 20 rows (n=5/8/7). Two bugs (`is_rt_line` column removed by the rewritten
`01_preprocess.py`, and a `PCM_data/PCM_data/` path-nesting mismatch) had to be fixed first to make
this re-run possible at all — see `07_PHASE_5_AUDIT.md` for the full writeup; the `family` field this
script uses for Monte Carlo same-family donor fallback now derives from the real `manufacturer` column
(6 values) rather than the old binary Rubitherm/Pluss flag.

**Dominant entropy criterion — PRE- vs POST-unification.** The pre-unification (raw entropy) run had
`supercooling` dominating all three clusters (63.8% / 48.6% / 57.0%) — an artifact of the
entropy formula overweighting a near-zero-ideal cost criterion. The **unified engine (2026-09-08)
caps supercooling's entropy weight at 0.16**, and the actual 2026-09-08 run shows **`Tm_fitness`
dominating** (entropy weight 59.6% / 69.7% / 66.8%, all flagged by the >40%-domination check), with
supercooling *blended* at ≈0.19 / 0.16 / 0.17 (down from ≈0.31–0.47 blended pre-cap). **Kendall's
W** in that run: 0.338 / 0.650 / 0.554 — no cluster reaches "strong" (W>0.8); Cluster 0 stays
"ambiguous". The ported pairwise method-agreement diagnostic flags **GRA as the structural outlier**
(lowest mean pairwise ρ vs the other three) in all three clusters — an independent disagreement
source that the supercooling cap does not address.

## Literature support

Oluah (2020) is cited by name for the TOPSIS unit-test fixture and as the domination-threshold
comparator (framework doc §13.1 names this as the project's own regression-test anchor — matches to
3 decimal places after refactoring, per `phases.md` PROMPT 5's stated verification requirement).
TOPSIS/PROMETHEE/VIKOR/GRA/CoCoSo are standard, well-established MCDM methods; no dedicated MCDM
methodology paper (e.g., for VIKOR's original formulation) was found cross-referenced in
`references.bib`/`.claude/references.md` during this audit — see `13_LITERATURE_MAPPING.md` for the
full gap analysis.

## Validation

Monte Carlo inclusion probability, Top-1 retention, rank-reversal frequency, Spearman ρ vs. baseline
— all computed and persisted per candidate per cluster. Kendall's W as a per-cluster
cross-method-agreement check. No external/physics validation yet (that is Phase 7).

## Outputs

`mcdm_full_rankings.csv` (full per-survivor audit trail), `mcdm_topk_by_cluster.csv` (Top-3 subset),
`monte_carlo_stability.csv` (MC columns), `mcdm_method_agreement.csv` (method-pair ρ/τ),
`outputs/qc_montecarlo_inclusion.html`. All carry `upstream_cluster_profile_fingerprint`. The full
file also carries legacy column aliases (`name`, `topsis_score`, `gra_grade`, `kendall_w`,
`top3_inclusion_probability`, …) so Phase 7/8 / seasonal scripts read it unchanged.

## Cross-phase provenance stamping and hard-fail check (added 2026-08-11)

Before doing anything else, `load_survivors()` now fingerprints the CURRENT on-disk
`cluster_profiles_rajasthan.csv` (`provenance_lib.file_fingerprint()`/`fingerprint_id()`) and
compares it against the `upstream_cluster_profile_fingerprint` stamp embedded in Phase 5's survivor
file — `assert_fingerprint_match()` raises `SystemExit` (not a warning) on any mismatch. This exists
because Phase 7 caught Phase 5's and Phase 6's outputs disagreeing cluster-by-cluster on which PCMs
belonged to which `cluster_id`, traced to Phase 4's GMM cluster labels not being stable across
separate re-runs (see `06_PHASE_4_AUDIT.md`'s second documented bug and `09_PHASE_7_AUDIT.md`'s
"Completion Report" for the full incident writeup). This script's own output
(`mcdm_full_rankings.csv`) is now stamped with the
same fingerprint, which Phase 7 and Phase 8 each verify in turn.

## Dependencies

Requires Phase 5's κ-calibrated survivor set (itself provisional pending database expansion) and
Phase 4's cluster profiles, now verified via the provenance check above. Feeds Phase 7
(`10_physics_validation.py`, which computes Spearman rho between this script's Borda/
Copeland ranks and simulated solar fraction) and, via Phase 7, Phase 8
(`09_recommendation_cards.py`, which also re-imports this script as a module to recompute
the per-criterion contribution decomposition against its own already-saved weight formula).

## Problems / risks

- **Database expansion (2026-08-14)**: Phase 6 was re-run against the expanded 55-row database — the
  `pcm_database_status` tag on every output row moved from `"PROVISIONAL — ~25-row..."` to
  `"COMPLETE — 55-row manufacturer database..."`. The ranking still runs on a κ-relaxed rather than
  nominal-threshold survivor pool (that policy question remains genuinely open, see
  `09_PHASE_7_AUDIT.md`). **Superseding note:** the 2026-08-31 `L_required` methodology correction
  (top banner) changes the Phase 5 survivor set fed to this script, so the 2026-08-14 ranking numbers
  are pre-correction; re-run status for this script post-2026-08-31 is not established here.
- **`cost` and `corrosion` are effectively structural placeholders**, not measured criteria — a
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
fixed — see `07_PHASE_5_AUDIT.md`. **Update, 2026-08-14 (later same day): Phase 7 was also re-run
against that ranking** (`10_physics_validation.py`) — the negative validation result
persisted (Spearman rho = -0.385/+0.125/-0.097 across the 3 clusters, mean -0.119, all in the ≤0.4
"genuine negative" band vs. the pre-expansion -0.900/-0.096/-0.198) — so the larger database did
**not** resolve the MCDM-vs-physics disagreement; if anything Cluster 0's now-healthy sample size
(n=9, no longer undersized) makes its persistently-low Kendall's W a more concerning finding, not a
less concerning one. Phase 8 (`09_recommendation_cards.py`) was also re-run and produced
Top-1 picks RT50 / savE® OM50 / savE® OM50 — see `09_PHASE_7_AUDIT.md` and `10_PHASE_8_AUDIT.md` for
the writeup. **All of the numbers in this "Update, 2026-08-14" paragraph predate the 2026-08-31
`L_required` correction (top banner) and are superseded by it; the post-correction re-run status for
Phases 6–8 is tracked in each phase's own audit, not asserted here.**
