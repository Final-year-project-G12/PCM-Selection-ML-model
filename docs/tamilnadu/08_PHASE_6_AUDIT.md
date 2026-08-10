# 08 — Phase 6 Audit: MCDM Ranking Engine

True script: `08_mcdm_ranking.py` (disk: `09_recommendation_cards (2).py`).

## Status: code complete, never executed. Confirmed a 2-method stack (TOPSIS + GRA), not Rajasthan's
4-method + Monte Carlo stack — this is a real, self-acknowledged scope difference, not a hidden gap.

## Criteria — exactly 5, all benefit-type

```python
AHP_PRIOR = {
    "f_Tm": 0.24/0.80, "latent_heat_kJ_kg": 0.20/0.80, "rho_H_MJ_m3": 0.12/0.80,
    "TC_W_mK": 0.13/0.80, "cycles_confidence": 0.11/0.80,
}
```
(1) Gaussian melting-point fitness `f_Tm`, (2) latent heat, (3) volumetric latent heat, (4) thermal
conductivity, (5) log-scaled cycling confidence. **Corrosion and cost are explicitly excluded**, with
an honest in-code instruction to disclose this rather than silently drop them: *"Corrosion class and
cost are NOT included as ranking criteria — the database doesn't have reliable values for either
yet... Say this explicitly in your methodology rather than silently dropping them."*

## Gaussian Tm-fitness — same σ=4K as Rajasthan, same framework doc §9.2 provenance

```python
SIGMA_TM = 4.0
f_Tm(i) = exp(-((Tm_i - Tm_target)**2) / (2*SIGMA_TM**2))
```
Correctly implemented as the pre-ranking transform, with the same "the one step every PCM-MCDM paper
gets wrong" framing quoted verbatim in-code as Rajasthan's equivalent.

## TOPSIS and GRA — standard, correctly implemented

TOPSIS: vector-normalize → weight → ideal/anti-ideal (all criteria treated as benefit, valid since
`f_Tm` already converts the one target-type criterion to a benefit form) → Euclidean distances →
closeness coefficient. GRA: ideal reference = column max, distinguishing coefficient `ζ=0.5`
(standard value) — **one implementation simplification worth noting**: `delta_min`/`delta_max` are
computed as global scalars over the *entire* delta matrix (`.min()`/`.max()` with no axis argument),
not per-criterion column-wise, which is a simplification relative to some published GRA variants
that compute those bounds per column. This does not make the implementation wrong, but it is a
specific, checkable design choice that a methodology write-up should state explicitly rather than
imply is the only valid GRA formulation.

## Weight blend — 50/50 entropy+AHP, AHP is a labeled proxy prior (more transparent than Rajasthan's None-stub)

```python
ENTROPY_AHP_LAMBDA = 0.5
w_final = 0.5*w_entropy + 0.5*w_ahp
```
`w_ahp` is the framework doc's Table-13 8-criterion weights, **renormalized over just the 5 criteria
this script actually uses** (corrosion/cost/supercooling removed, remainder rescaled to sum to 1) —
`f_Tm=0.24, latent_heat=0.20, rho_H=0.12, TC=0.13, cycles=0.11`, summing to 0.80 of the original
table, each divided by 0.80. This is **not real pairwise AHP elicitation** (same underlying gap as
Rajasthan's `AHP_PAIRWISE_MATRIX=None`), but it is presented with a more honest, specific in-code
label than Rajasthan's bare `None`: *"this is an honest placeholder, not a claimed AHP result...
Replace with a real elicited AHP vector if you get one."* Entropy weights are computed correctly
(standard Shannon-entropy formula) per cluster's own filtered decision matrix — no analog of
Rajasthan's entropy-weight-inflation-for-sparse-criteria bug was found, though this may simply
reflect that TN's 5 criteria (unlike Rajasthan's always-NaN `cost`) likely all have real values for
most candidates, so the failure mode that triggered Rajasthan's bug may not arise here regardless.

## Borda count + Kendall's W — correctly implemented, m=2 rankers

```python
borda(i) = Σ_methods (n - rank_m(i) + 1)
W = 12S / (m²(n³-n))
```
Standard formulas, correctly guarded against `n≤1` (NaN) and division-by-zero. With only 2 rankers
(TOPSIS, GRA), Kendall's W here measures agreement between exactly those two methods — a narrower
concordance signal than Rajasthan's 4-method W, but internally consistent and correctly computed.
Low-W reporting: **threshold `W<0.6`**, printed as an explicit, honest finding — *"per plan v3.0
Section 9.5, this is a genuine, reportable finding (that regime's PCM choice is ambiguous), not a
bug to fix — discuss it rather than hide it."* Same honest-reporting philosophy Rajasthan's Cluster-0
W=0.4375 finding demonstrates.

## Monte Carlo, PROMETHEE, VIKOR — confirmed genuinely absent, not disabled

A full-file read found **zero code** for any of the three — no functions, no unused imports, no dead
branches, no commented-out blocks. They appear **only in prose**, both in the module docstring and in
an explicit end-of-run print block listing them as "genuinely optional... stretch goals, not
required." This is an honest, self-aware scope boundary, not a hidden gap — the script's own final
message states plainly that TOPSIS+GRA alone "already gives you a defensible, falsifiable Top-3 per
cluster, which is the actual headline deliverable of Objective 1," while separately naming the
3 extensions (PROMETHEE II, Monte Carlo, a minimal physics validation) as what would be needed to
match the framework doc's full Phase 6/7 ambition.

## The built-in convergence diagnostic — a genuinely good piece of self-aware engineering

Because `Tm_target_C` is held constant (57°C) across every cluster by design, the script anticipates
that every cluster's #1-ranked PCM could plausibly converge to the same candidate, and — rather than
let that look like a bug — prints an explicit, two-part honest interpretation if it happens:

> *"This is a direct consequence of Tm_target being held constant across all clusters... combined
> with every candidate's latent heat comfortably clearing L_required in every cluster. It is NOT a
> bug."* Then offers two framings: (a) state it as a finding — TN's regimes differ more in solar
> reliability than in delivery-relevant temperature, so real differentiation would need to show up in
> Phase 7 physics simulation, not in the candidate list; or (b) run the optional charging-feasibility
> heuristic to see if a regime-dependent Tm cap changes the outcome.

**Important caveat given this audit's Phase 3 finding**: this diagnostic's second premise ("every
candidate's latent heat comfortably clearing L_required") is currently true *only because*
`L_required` is understated by the Phase 3 bug — once that's fixed, this convergence behavior itself
may no longer hold, and the diagnostic's framing (a) may need re-examination against the corrected
numbers.

## Literature support

Same Table 13 provenance as Rajasthan for the AHP priors. Same standard TOPSIS/GRA method citations
needed (Hwang & Yoon 1981; Deng 1982 — see `12_LITERATURE_MAPPING.md`) but not present in this
project's bibliography files, identical gap to Rajasthan.

## Validation

None possible yet — no `mcdm_topk_by_cluster.csv` exists.

## Outputs (expected)

`mcdm_topk_by_cluster.csv`, `mcdm_full_scores_by_cluster.csv`.

## Dependencies

Requires Phase 5's `feasibility_survivors_by_cluster.csv` (itself blocked on the Phase 3 fix) and
Phase 4's cluster profiles.

## Problems / risks

- **Inherits the Phase 3 `L_required` risk** — any ranking produced today would need re-running after
  the fix.
- **Genuinely narrower method stack than Rajasthan** (2 vs. 4 methods, no Monte Carlo) — self-aware
  and honestly documented, but a reviewer comparing the two state pipelines side-by-side would
  correctly note this asymmetry; if all four states are eventually meant to be directly comparable
  (as multiple docstrings across both pipelines state), this gap should be closed before a
  cross-state comparison is presented, not treated as a permanent TN/Rajasthan difference.
- GRA's global (not per-column) `delta_min`/`delta_max` is a specific implementation choice worth
  disclosing explicitly in a methods section.

## Status

**CODE COMPLETE, NEVER RUN.** A genuinely well-engineered "minimum viable" MCDM stack — every
simplification relative to Rajasthan's more elaborate Phase 6 is explicitly and honestly labeled as
such in the code itself, which is good practice, but the stack is objectively less mature than
Rajasthan's (2 methods vs. 4, no uncertainty quantification) and should not be presented as
equivalent without qualification.
