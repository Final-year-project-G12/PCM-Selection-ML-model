# Phase 6 — MCDM Ranking, Explained

This document explains what `08_mcdm_ranking.py` actually computes, why it's built the way
it is, and how to read every plot `generate_objective1_plots.py` produces from its output.
All numbers below are pulled directly from the current, committed output files
(`data/processed/pcm/mcdm_full_scores_by_cluster.csv` and `mcdm_topk_by_cluster.csv`) —
nothing here is illustrative or approximated.

A companion document, **`CLUSTER_0_BUMP_CHART_EXPLAINED.md`**, walks through Cluster 0's
bump chart candidate-by-candidate in full detail; this file covers the method and all
plots at the whole-pipeline level.

---

## 1. What Phase 6 is doing

**Input:** `feasibility_survivors_by_cluster.csv` (Phase 5's output) — 29 to 30 candidate
PCMs per cluster, already filtered on melting-window, absolute-band, latent-heat-floor,
and corrosion criteria.

**Output:**
- `mcdm_full_scores_by_cluster.csv` — every survivor's full score breakdown (all 4 methods,
  all 5 criteria, both raw and normalized).
- `mcdm_topk_by_cluster.csv` — just the Top-3 consensus candidates per cluster, same columns.

**The job:** take those survivors and rank them by *four independent* multi-criteria
decision-making (MCDM) methods, then combine the four rankings into one consensus Top-3
per cluster. Using four methods instead of one is the point — if all four agree, the pick
is robust; if they disagree, that disagreement is itself a finding worth reporting (via
Kendall's W), not something to hide behind a single number.

---

## 2. The five ranking criteria

| Criterion | Meaning | Type |
|---|---|---|
| `f_Tm` | Melting-point fitness (see below) | benefit (higher = better) |
| `latent_heat_kJ_kg` | Mass-specific stored energy | benefit |
| `rho_H_MJ_m3` | Volumetric stored energy (density × latent heat) | benefit |
| `TC_W_mK` | Thermal conductivity — how fast it charges/discharges | benefit |
| `cycles_confidence` | How well-documented its melt/freeze cycling durability is | benefit |

Corrosion and cost are **not** ranking criteria — the PCM database doesn't have reliable
values for either. Corrosion is instead used as a hard *veto* back in Phase 5
(`07_feasibility_filter.py`), not a ranking weight. This is stated explicitly in the
script rather than silently dropped.

### Why melting point needs special handling

Melting temperature is not simply "higher is better" or "lower is better" — it's a
**target**. A PCM melting exactly at your target temperature is ideal; one melting 10°C
off in either direction is bad, regardless of direction. Feeding raw Tm into TOPSIS/GRA
as an ordinary benefit or cost criterion produces plausible-looking nonsense. So Phase 6
converts Tm to a Gaussian fitness score *before* anything else touches it:

```
f_Tm(i) = exp( -(Tm_i - Tm_target)^2 / (2*sigma^2) ),  sigma = 4 K
```

`f_Tm` is then used as an ordinary benefit criterion downstream, identically to the other
four. `Tm_target` itself is cluster-specific (K = 4 run) — 57.0°C for Clusters 0 and 1, but
56.51°C (Cluster 2) and 55.16°C (Cluster 3), since those two regimes get a climate-driven
downward adjustment from Phase 5's charging-feasibility cap (`07b_charging_feasibility.py`).

---

## 3. Weighting: entropy + AHP blend

Each criterion's weight is **entropy-weighted per cluster** (computed objectively from how
much that criterion actually varies among *that cluster's own* candidates — a criterion
that barely varies contributes little discriminating power and gets down-weighted) —
**blended 50/50** with a fixed AHP-style prior taken from the project plan's literature
table (an honest placeholder until a real pairwise AHP elicitation is done with a domain
expert; documented as such directly in the script).

Actual blended weights, per cluster:

| Cluster | f_Tm | latent_heat | rho_H | TC | cycles_confidence |
|---|---|---|---|---|---|
| 0 | 0.245 | 0.198 | 0.143 | 0.166 | 0.248 |
| 1 | 0.245 | 0.198 | 0.143 | 0.166 | 0.248 |
| 2 | 0.246 | 0.198 | 0.143 | 0.165 | 0.248 |
| 3 | 0.264 | 0.203 | 0.144 | 0.169 | 0.220 |

`f_Tm` and `cycles_confidence` dominate (roughly 22–26% each), latent heat is third
(~20%), and volumetric energy density / conductivity are the smallest (~14–17% each).
Clusters 0 and 1 have identical weights because they have the identical 29-candidate
survivor pool, and Cluster 2's pool is the same set too, so its weights differ only in the
third decimal. Cluster 3 — the only cluster with its own survivor set — shifts weight
toward `f_Tm` (0.264) and away from `cycles_confidence` (0.220, the lowest of any cluster).

---

## 4. The four ranking methods

| Method | Core idea |
|---|---|
| **TOPSIS** | Ranks by Euclidean closeness to an ideal point and distance from an anti-ideal (worst-case) point. Rewards candidates that are good *everywhere*. |
| **GRA** (Grey Relational Analysis) | Ranks by similarity of a candidate's whole criterion *profile shape* to the best-in-class reference — mathematically more sensitive to a candidate's single worst gap than TOPSIS is. |
| **PROMETHEE II** | Pairwise outranking: for every pair of candidates, computes how much one "beats" the other per criterion (with 10%/30% indifference/preference thresholds), nets it into one flow score per candidate. |
| **VIKOR** | Compromise ranking that explicitly balances "best on average across all criteria" (S) against "worst single-criterion regret" (R) into one index Q (v=0.5 weighting between the two). |

These four were chosen specifically because they weight compensation differently — TOPSIS
and GRA are compensatory (a big strength can offset a weakness), VIKOR is partly
regret-based (a bad worst-criterion drags a candidate down even if everything else is
excellent). That's *why* the same candidate can rank very differently across methods —
see Section 7.

### VIKOR's own acceptability check

VIKOR has a formal, two-part check for whether its own top-ranked candidate is a genuinely
*acceptable* single winner:
- **C1 (acceptable advantage):** the Q-gap between rank 1 and rank 2 must exceed
  `1/(n-1)`.
- **C2 (acceptable stability):** the same candidate must also be top-ranked in at least
  one of S or R individually (this check was previously missing from the script and has
  since been added).

Current per-cluster result:

| Cluster | VIKOR compromise status |
|---|---|
| 0 | Single winner acceptable |
| 1 | Single winner acceptable |
| 2 | **C1 fails** (Q gap 0.0266 < 0.0357) — report compromise set: PureTemp 53 / PureTemp 58 |
| 3 | **C1 fails** (Q gap 0.0184 < 0.0345) — report compromise set: PureTemp 53 / Myristic acid (C14) |

This is reported transparently in the output rather than collapsed into a false single
winner — Clusters 2 and 3 genuinely don't have one PCM VIKOR is confident enough to
single out.

---

## 5. Consensus: Borda count + Kendall's W

The final `consensus_rank` is a **Borda count** across all four methods: each candidate
earns points equal to `(n_survivors - rank + 1)` per method, summed, then re-ranked.

**Kendall's W** (coefficient of concordance, 0 = no agreement, 1 = perfect agreement
across all four methods) is reported alongside it as an honesty check on how much to
trust that consensus:

| Cluster | Kendall's W | Reading |
|---|---|---|
| 0 | 0.796 | Fairly strong agreement |
| 1 | 0.796 | Fairly strong agreement (identical ranking to Cluster 0) |
| 2 | **0.782** | Weakest — most inter-method disagreement |
| 3 | **0.842** | Strongest agreement of all four clusters |

No cluster has W below the informal "meaningful ambiguity" threshold of 0.6, so every
cluster's consensus Top-3 is at least moderately trustworthy — but Cluster 2's lower W
(together with its VIKOR compromise set) means its Top-3 should be reported with the most
caveats.

---

## 6. All Phase-6-related plots, explained

All of these are produced by `generate_objective1_plots.py`, reading Phase 6's (and where
noted, Phase 7's) output CSVs. Numbers quoted are the actual current run's output.

### Plot 07 — Bump Chart (`07_bump_chart_ranks.png/.html`, plus one per cluster)
For each cluster's Top-5 consensus candidates, one line per PCM tracks its rank (y-axis,
1=best at top) across the four methods (x-axis: TOPSIS → GRA → PROMETHEE → VIKOR). A flat
line means all methods agree on that PCM's position; a zigzag means the methods
structurally disagree about it. This is the single most direct visual of what Kendall's W
is summarizing as one number. See `CLUSTER_0_BUMP_CHART_EXPLAINED.md` for a full
candidate-by-candidate walkthrough of one cluster's chart.

### Plot 08 — Method Rank Correlation Heatmap (`08_method_rank_correlation_heatmap*`)
A 4×4 Spearman-rho (and Kendall-tau) heatmap between the four methods' ranks, computed
over the pooled Top-3 candidates across all four clusters (`mcdm_topk_by_cluster.csv`).
Current Spearman values:

|  | TOPSIS | GRA | PROMETHEE | VIKOR |
|---|---|---|---|---|
| **TOPSIS** | 1.000 | −0.602 | −0.511 | +0.525 |
| **GRA** | −0.602 | 1.000 | +0.292 | −0.577 |
| **PROMETHEE** | −0.511 | +0.292 | 1.000 | −0.086 |
| **VIKOR** | +0.525 | −0.577 | −0.086 | 1.000 |

TOPSIS and VIKOR agree moderately well with each other (+0.525) — both being at least
partly compensatory/average-based — while TOPSIS and GRA are *anti*-correlated (−0.602)
on this pooled Top-15 set. This is a real, useful diagnostic: it's telling you that among
just the candidates good enough to reach a Top-3, TOPSIS and GRA are tending to disagree
about which end of that narrow, already-strong pool is actually best — exactly the
"profile-shape vs. ideal-distance" tension described in Section 4.

### Plot 09 — Monte Carlo Top-3 Inclusion Probability (`09_monte_carlo_top3_probability*`)
Not part of `08_mcdm_ranking.py` itself, but downstream of it: `09b_monte_carlo_stability.py`
re-runs the ranking thousands of times with the criterion weights Dirichlet-perturbed and
the PCM properties Gaussian-jittered, and records how often each candidate still lands in
the Top-3. High inclusion probability = a robust pick even under weighting uncertainty;
low = a pick that's only in the Top-3 because of this exact weight choice.

### Plot 10 — Rank-Reversal Violin + Bar (`10_rank_reversal_violin*`)
Left panel: a violin plot per method per cluster showing the full rank distribution of
survivors — wider/flatter violins mean that method disagrees more with the others about
ordering within that cluster. Right panel: a horizontal bar chart of the 15 candidates
(across all clusters) with the largest `rank_spread` (max method rank − min method rank).
Currently the single largest spread is **RT64HC in Cluster 2**, spread of 18 (VIKOR rank
25 / TOPSIS rank 23 vs. GRA rank 7) — an extreme case of exactly the TOPSIS/GRA tension Plot 08 shows in
aggregate.

### Plot 11 — Agreement Plot (`11_agreement_plot*`)
Compares Phase 6's `consensus_rank` against Phase 7's *simulated* performance rank (by
annual solar fraction), per cluster, restricted to each cluster's Top-3. This is a
Phase 6-vs-Phase 7 cross-check, not a Phase 6-internal plot — see
`docs/uttarakhand/09_PHASE_7_AUDIT.md` for the physics side of that comparison. Because
it's built from only 3 points per cluster, its per-cluster Spearman values (only ±0.5 or
±1.0 are possible with 3 points) are small-sample artifacts, not the pipeline's real correlation
estimate — that's `physics_validation_spearman.csv`, computed over all ~20 survivors.

### Plot 13 — Recommended PCM Summary (`13_recommended_pcm_summary*`)
The final presentation layer: one consolidated view of each cluster's Top-3 consensus
pick alongside its key properties, meant to be read on its own by someone who hasn't seen
the four-method internals — everything above is what that summary is built from.

---

## 7. Why some PCMs rank higher and others lower — the general pattern

Two structurally different scoring philosophies are running at once across the four
methods, and that's the root cause of most rank disagreement:

- **TOPSIS, GRA, PROMETHEE are compensatory.** A candidate that's mediocre on one
  criterion can still rank well if it's excellent on the others — the criteria trade off
  against each other in a weighted sum or distance metric.
- **VIKOR is partly regret-based.** Its R component specifically tracks a candidate's
  *single worst* criterion gap, so a candidate with one clearly weak criterion gets
  penalized in VIKOR even if its average across all criteria is excellent.

Concretely (see `CLUSTER_0_BUMP_CHART_EXPLAINED.md` for the full numeric walkthrough):
a PCM whose melting point sits noticeably off the cluster's `Tm_target` (low `f_Tm`) can
still place well in TOPSIS/GRA/PROMETHEE if it compensates with high latent heat,
volumetric energy density, and conductivity — but VIKOR's regret term punishes that same
off-target melting point directly, dropping it several ranks. Conversely, a PCM that's
merely "good, not great" on every single criterion — nothing to compensate with, but also
nothing to be punished for — can end up more consistent across all four methods than a
PCM with one standout strength and one clear weakness.

This is also why Kendall's W varies by cluster rather than being a fixed pipeline
property: it depends on how many "one standout strength, one clear weakness" candidates
happen to be in that cluster's particular survivor pool.
