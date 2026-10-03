# 08 — Phase 6 Audit: MCDM Ranking Engine

**Script**: `08_mcdm_ranking.py`

**Status**: **COMPLETE (K = 4).** The Top-3 result for all four clusters, with per-method ranks,
is in `mcdm_topk_by_cluster.csv` and the committed `data/plots/uttarakhand_objective1/` plots.

---

> ## MAJOR UPDATE (2026-09) — this entire file describes a superseded, two-method version
>
> Everything in this file's walkthrough (TOPSIS+GRA only, RT60 as consensus #1 in every
> cluster of the old K = 5 run, no Monte Carlo) reflects `08_mcdm_ranking.py`'s state at an earlier point in the
> project. The script has since been extended to a **four-method stack (TOPSIS + GRA + PROMETHEE II
> + VIKOR)** with a companion `09b_monte_carlo_stability.py` (Dirichlet weight perturbation;
> 2,000 draws/cluster by default, `--draws 5000` for the full spec), and two real bugs in that four-method stack were found and fixed in 2026-09:
> - **VIKOR's compromise check only tested "acceptable advantage," never "acceptable stability"**
>   (the second condition needs S/R, which weren't even passed to the function) — a genuine gap vs.
>   the standard method. Fixed to check both.
> - **TOPSIS applied its own vector normalization on top of a matrix already min-max normalized**
>   by the caller (the same basis GRA/PROMETHEE/VIKOR use) — put TOPSIS on a different effective
>   basis, manufacturing spurious method disagreement. Fixed to use the shared basis directly.
>
> **Current MCDM consensus Top-3 per cluster (K = 4 run, 2026-10)** (not RT60 everywhere as the
> rest of this file describes):
>
> | Cluster | Tm_target | #1 | #2 | #3 | Kendall's W | VIKOR |
> |---|---|---|---|---|---|---|
> | 0 (warm plains, 10 pts) | 57.0 °C | PureTemp 58 | n-Octacosane (C28) | PlusICE A58 | 0.796 | single winner |
> | 1 (mid-hills, 23 pts) | 57.0 °C | PureTemp 58 | n-Octacosane (C28) | PlusICE A58 | 0.796 | single winner |
> | 2 (higher valleys, 9 pts) | 56.51 °C | PureTemp 58 | savE® OM55 | n-Hexacosane (C26) | 0.782 | compromise set PureTemp 53 / PureTemp 58 |
> | 3 (high Himalaya, 3 pts) | 55.16 °C | **PureTemp 53** | n-Hexacosane (C26) | Myristic acid (C14) | 0.842 | compromise set PureTemp 53 / Myristic acid (C14) |
>
> Cluster 3 is genuinely different from the rest because `07b_charging_feasibility.py`'s regime cap
> (a separate bug, see `07_PHASE_5_AUDIT.md`) gives it a real, lower `Tm_target` and a different
> survivor set. **Clusters 0 and 1 produce identical rankings**: same 29 survivors, same
> `Tm_target`, and `L_required` (118 vs 132 kJ/kg) is not a ranking criterion, so the MCDM cannot
> separate them — their difference shows up only in Phase 7 physics (see `09_PHASE_7_AUDIT.md`).
> Cluster 2's slightly lower `Tm_target` changes its #2/#3 but not its #1.
>
> The rest of this file is kept as a historical record of the pipeline's earlier state and the
> (still methodologically sound) reasoning used at the time — read the walkthrough below with that
> in mind, not as the current result.

## Scope — what this script deliberately is and is not

The docstring **used to be** explicit that this was a reduced stack (now superseded, see the update
notice above):

> This is the "minimum viable MCDM stack" from your 4-day sprint plan: TOPSIS + GRA,
> entropy-weighted per cluster, Borda-aggregated to a Top-3. **PROMETHEE II / VIKOR / CoCoSo and
> the 5,000-draw Monte Carlo stability check are NOT implemented here** — they're real, documented
> extensions …, add them if time remains, but this script alone already gives you a defensible,
> falsifiable Top-3 per cluster.

This was true at an earlier project stage. **Now: four methods (TOPSIS + GRA + PROMETHEE II +
VIKOR), plus a companion Monte Carlo script — not "two methods, no Monte Carlo."**

## Inputs

`data/processed/pcm/feasibility_survivors_by_cluster.csv` — `07`'s 220-row output (55 × 4),
filtered to `passes_all == True` per cluster (29/29/29/30 rows).

## Processing

### The Gaussian Tm-fitness transform

The script frames this as the correctness-critical step:

> **THE ONE STEP EVERY PCM-MCDM PAPER GETS WRONG (plan v3.0 Section 9.2).** Melting temperature is
> a TARGET-based criterion, not a benefit or cost — closer to `Tm_target` is better in both
> directions. Feeding raw Tm into TOPSIS/GRA produces plausible-looking nonsense.

```
f_Tm(i) = exp( -(Tm_i - Tm_target)^2 / (2 * sigma^2) ),   sigma = SIGMA_TM = 4.0 K
```

`f_Tm` is then treated as an ordinary benefit criterion. sigma = 4 K is cited to "plan v3.0 Section
9.2 — justified from HX approach temperature."

### Criteria

Five, all benefit-direction after the Tm transform:

| Criterion | Meaning | Source column |
|---|---|---|
| `f_Tm` | Gaussian melting-point fitness | computed from `Tm_C`, `Tm_target_C` |
| `latent_heat_kJ_kg` | gravimetric latent heat | PCM database |
| `rho_H_MJ_m3` | volumetric latent heat | `density * L / 1000` |
| `TC_W_mK` | thermal conductivity | `(TC_liquid + TC_solid)/2` |
| `cycles_confidence` | log-scaled cycling stability | `log1p(cycles)/log1p(max_cycles)` |

Explicitly excluded, and stated as such: "**Corrosion class and cost are NOT included as ranking
criteria** — the database doesn't have reliable values for either yet … Say this explicitly in your
methodology rather than silently dropping them."

`cycles_confidence` NaNs are median-imputed **within each cluster's own candidate set**, with a
`cycles_confidence_imputed` boolean flag retained "(report, don't hide)". With `cycles_tested`
imputed for 48 of 55 database rows already, this flag will rarely fire — but the underlying values
are mostly MICE-RF-PMM estimates regardless (see `07_PHASE_5_AUDIT.md`).

### Weighting

```python
ENTROPY_AHP_LAMBDA = 0.5

AHP_PRIOR = {                       # renormalised over the 5 criteria actually used,
    "f_Tm":              0.24/0.80, # from plan v3.0 Table 13's 8-criterion set with
    "latent_heat_kJ_kg": 0.20/0.80, # corrosion/cost/supercooling removed
    "rho_H_MJ_m3":       0.12/0.80,
    "TC_W_mK":           0.13/0.80,
    "cycles_confidence": 0.11/0.80,
}

w_final = 0.5 * w_entropy + 0.5 * w_ahp      # then renormalised to sum 1
```

Resolved AHP prior: `f_Tm` 0.300, `latent_heat_kJ_kg` 0.250, `TC_W_mK` 0.1625,
`rho_H_MJ_m3` 0.150, `cycles_confidence` 0.1375.

Shannon entropy weights are computed **per cluster from that cluster's own min-max-normalised
decision matrix**. The honesty note is explicit:

> If you get 10 minutes with your guide for a real pairwise AHP matrix, replace `AHP_PRIOR` below
> and rerun — until then this is an **honest placeholder, not a claimed AHP result**.

No pairwise elicitation was performed. There is no `AHP_PAIRWISE_MATRIX` variable in the
Uttarakhand script at all — only the fixed prior above.

### Normalisation and the two methods

Each criterion is min-max normalised to [0, 1] within the cluster's survivor set (constant columns
-> 0.5), then:

**TOPSIS** — `norm = M / sqrt(sum(M^2))` column-wise, weighted; ideal `v+ = max`, anti-ideal
`v- = min`; score `= s-/(s+ + s-)`. All columns treated as benefit criteria, which is correct here
by construction.

**GRA** — reference = column max; `delta = abs(M - ref)`; coefficient
`(delta_min + zeta*delta_max)/(delta + zeta*delta_max)` with `GRA_ZETA = 0.5`; grade = weighted row
sum.

Note: `delta_min`/`delta_max` are taken over the **whole matrix** (`delta.min()`, `delta.max()`),
not per-column. With min-max-normalised inputs `delta_min = 0` and `delta_max = 1` in almost every
case, so the coefficient reduces to `0.5/(delta + 0.5)` — a standard simplification, but worth
stating if GRA's formulation is written up.

### Consensus and agreement

```python
borda = sum over methods of (n - rank + 1)                  # higher = better
consensus_rank = borda.rank(ascending=False, method="min")  # ties share the lower rank

# Kendall's W over m = 2 rankers, n candidates
R = rowwise sum of ranks;  S = sum((R - R_bar)^2)
W = 12*S / (m^2 * (n^3 - n))
```

Kendall's W is written to every row as `kendall_w` and is reported per cluster. The script treats
low agreement as a finding, not a bug:

> `[NOTE] Kendall's W < 0.6 for cluster(s) … — TOPSIS and GRA disagree meaningfully there. Per
> plan v3.0 Section 9.5, this is a genuine, reportable finding (that regime's PCM choice is
> ambiguous), not a bug to fix — discuss it rather than hide it.

### The constant-`Tm_target` diagnostic

`08` contains a purpose-built check for exactly the degeneracy this run exhibits:

```python
top1_sets = topk[topk["consensus_rank"] == 1].groupby("cluster_id")["name"].first()
if top1_sets.nunique() == 1:
    print("[FINDING] Every cluster's #1 PCM is identical …")
```

and then offers two honest reporting options in full text:

> (a) State it as a finding: Uttarakhand's climate regimes differ more in solar reliability/cloud
> persistence than in delivery-relevant temperature, so a single PCM family serves the whole state
> under the corrected `Tm_target` rule — differentiation would need to show up in Phase 7 physics
> simulation (solar fraction per regime), not in the candidate list itself.
>
> (b) Run `07b_charging_feasibility.py` (optional, heuristic regime-dependent upper bound on Tm)
> before 07/08 to see if a real charging-feasibility constraint changes this.

In the old K = 5 two-method run (identical #1 in every cluster) **this diagnostic fired.** In the current K = 4 run it does not: the #1 set is {PureTemp 58, PureTemp 53}.

### Outputs

| File | Contents |
|---|---|
| `data/processed/pcm/mcdm_topk_by_cluster.csv` | Top-3 per cluster = **12 rows** (4 × 3) |
| `data/processed/pcm/mcdm_full_scores_by_cluster.csv` | every survivor's full breakdown, 29 + 29 + 29 + 30 = **117 rows** |

Both are git-ignored. Clusters with fewer than 2 survivors are skipped with a message; that did not
occur here.

---

## Observed results — HISTORICAL (two-method, K = 5 run)

> Everything from here to "What is absent from Phase 6" describes the superseded two-method K = 5
> run. The current K = 4 result is the table in the update notice at the top of this file. The
> `data/plots/objective1/` directory these numbers were recovered from had no generator script and
> has been deleted (2026-10); the `uttarakhand_objective1/` plots now show the K = 4 result.

These numbers were recovered from four artefacts of that run:
`data/plots/objective1/recommended_pcm_summary.html` (consensus ranks),
`data/plots/objective1/consensus_vs_topsis_agreement.html` (consensus vs TOPSIS rank pairs),
`data/plots/uttarakhand_objective1/07_bump_chart_ranks.html` (TOPSIS / GRA / consensus per
cluster), and `data/plots/uttarakhand_objective1/13_recommended_pcm_summary_interactive.html`
(per-candidate properties). **All four agreed.**

### Clusters 0, 2 and 4 — identical Top-3

| Consensus rank | PCM | Family | Tm (°C) | L (kJ/kg) | rho·H (MJ/m³) | TC (W/m·K) | Cycles | TOPSIS rank | GRA rank |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **RT60** | Rubitherm RT | 58.0 | 160 | 140.8 | 0.1695 | 2000 | 4 | 4 |
| **1** (tie) | **PureTemp 58** | PureTemp | 58.0 | 225 | 200.25 | 0.200 | 1620 | **1** | **7** |
| **3** | **n-Hexacosane (C26)** | n-Alkane | 56.5 | 256 | 197.12 | 0.238 | 1404 | **8** | — |

### Clusters 1 and 3 — identical Top-3

| Consensus rank | PCM | Family | Tm (°C) | L (kJ/kg) | rho·H (MJ/m³) | TC (W/m·K) | Cycles | TOPSIS rank | GRA rank |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **RT60** | Rubitherm RT | 58.0 | 160 | 140.8 | 0.1695 | 2000 | 3 | 3 |
| **2** | **savE® OM55** | PLUSS savE | 55.0 | 188 | 175.78 | 0.130 | 2000 | 2 | 5 |
| **2** (tie) | **Palmitic-stearic acid / Expanded graphite** | Composite | 55.2 | 176 | 150.656 | 0.160 | 2000 | **1** | **6** |

All property values above are cross-checked against
`PCM_data/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` and match exactly.

### Frequency across clusters

From `data/plots/objective1/top3_inclusion_probability.html` (a **count** of clusters in which each
PCM reached the Top-3, not a probability — see the note below):

| PCM | Clusters in Top-3 |
|---|---|
| RT60 | **5** |
| PureTemp 58 | 3 |
| n-Hexacosane (C26) | 3 |
| savE® OM55 | 2 |
| Palmitic-stearic acid/Expanded graphite | 2 |

### Method agreement (historical two-method run)

From `data/plots/verify_ranking/06_summary.png` (as captured then) and
`data/plots/uttarakhand_objective1/08_method_rank_correlation_heatmap_interactive.html`
(identical values):

| Pair | Spearman rho |
|---|---|
| TOPSIS vs GRA | **−0.930** |
| TOPSIS vs CONSENSUS | +0.376 |
| GRA vs CONSENSUS | −0.442 |

with `Number of ranked candidates: 15`, `Number of clusters: 5`, `Data completeness: 98.1 %`.

> **Read these correlations carefully.** `verify_04_ranking.py` computes them across the **pooled
> 15 Top-3 rows from all five clusters at once**, not per cluster. They are therefore *not* the
> per-cluster inter-method agreement statistic. The per-cluster statistic the pipeline itself
> computes is Kendall's W, written to `mcdm_topk_by_cluster.csv` — and **that value is not
> available in the source files**, because the CSV is git-ignored and no committed plot renders it.

Even with that caveat, the pattern within a single cluster is unambiguous from the bump chart. In
cluster 0, RT60 ranks 4th on TOPSIS and 4th on GRA, PureTemp 58 ranks **1st on TOPSIS and 7th on
GRA**, and n-Hexacosane C26 ranks **8th on TOPSIS**. In cluster 1, Palmitic-stearic/EG ranks **1st
on TOPSIS and 6th on GRA**. **TOPSIS and GRA disagree strongly, inside every cluster.**

### The consequence of that disagreement

Borda over two strongly anti-correlated rankers produces near-ties. Concretely, in cluster 0 with
29 survivors:

- RT60: `(29 - 4 + 1) + (29 - 4 + 1) = 52`
- PureTemp 58: `(29 - 1 + 1) + (29 - 7 + 1) = 52`

— an exact tie, which is why both are reported at consensus rank 1 (`method="min"`). The same
mechanism produces the rank-2 tie in clusters 1 and 3. **The "winner" in each cluster is decided by
a tie, not by a margin.**

---

## What is absent from Phase 6

| Component | Status in `08_mcdm_ranking.py` (as of this file's original writing vs. now) |
|---|---|
| PROMETHEE II | Was "not implemented" — **now implemented and run** |
| VIKOR | Was "not implemented" — **now implemented and run**; its compromise-check bug (see the update notice) is fixed |
| CoCoSo | Still not implemented |
| Copeland pairwise consensus | Still not implemented (Borda only) |
| Monte Carlo weight/property perturbation | Was "not implemented" — **now implemented in `09b_monte_carlo_stability.py` and run** (2,000 draws/cluster by default; `--draws 5000` for the full spec) |
| Top-3 inclusion probability | Was "not computed" — **now computed**; `09_monte_carlo_top3_probability.png` exists and is populated |
| Analytical criterion contributions | Still not implemented in `08` or `09` |
| AHP pairwise elicitation | Still not performed — a fixed prior is used and labelled a placeholder |

---

## Literature support

**None present in the source files** for TOPSIS, Grey Relational Analysis, Shannon-entropy
weighting, Borda count or Kendall's W. `08` cites plan v3.0 §9, §9.2 (the Gaussian transform and
sigma = 4 K), §9.5 (the low-W interpretation) and Table 13 (the AHP prior) — all internal
references. See `13_LITERATURE_MAPPING.md`.

## Validation

| Check | Result |
|---|---|
| Target-based Tm handled before ranking | **PASS** — Gaussian transform applied first, by design |
| Only `passes_all` rows ranked | **PASS** — `passed = grp[grp["passes_all"]]` |
| Missing `cycles_confidence` flagged, not silently filled | **PASS** — `cycles_confidence_imputed` retained |
| Excluded criteria declared | **PASS** — corrosion and cost named explicitly |
| AHP status declared | **PASS** — labelled "an honest placeholder, not a claimed AHP result" |
| Inter-method agreement reported | **PASS** — Kendall's W per cluster: 0.796 / 0.796 / 0.782 / 0.842 |
| Degenerate-result diagnostic | **PASS** — fired in the old run with two reporting options offered; correctly silent in the K = 4 run |
| Method agreement acceptable | Was **FAIL** (pooled TOPSIS vs GRA rho = −0.930, two-method version); current 4-method Kendall's W is 0.782-0.842 per cluster — a materially healthier agreement picture |
| Per-regime differentiation | **PARTIAL** — Cluster 3 (high Himalaya) gets a different #1 (PureTemp 53) and Cluster 2 a different #2/#3; Clusters 0 and 1 rank identically |
| Rank stability under perturbation | Was **Absent** — **now present**: `09b_monte_carlo_stability.py`, 2,000 draws/cluster (default) |

## Problems / risks

1. **~~RT60 is consensus rank 1 in every cluster.~~ RESOLVED (2026-09), and the root cause was
   different from what this section concluded.** It was downstream of a real bug in
   `07b_charging_feasibility.py`'s regime cap (a normalization step erased its own signal). Fixed;
   in the K = 4 run the high-elevation Cluster 3 gets a real, differentiated #1 (PureTemp 53).
   Clusters 0 and 1 still rank identically (same `Tm_target`, same survivors) — a genuine
   limitation of a near-constant `Tm_target`, reported rather than hidden. See
   `07_PHASE_5_AUDIT.md`.
2. **~~TOPSIS and GRA are strongly anti-correlated~~ (pooled Spearman −0.930, in the two-method
   version).** The current four-method version's Kendall's W (0.782-0.842) is a much healthier
   agreement signal — partly because a genuine TOPSIS normalization bug (see the update notice) was
   also fixed, removing spurious method disagreement that wasn't real multi-criteria disagreement.
3. **Ties in the two-method Borda consensus** were a real concern in the earlier version; the
   four-method version's VIKOR compromise-check (now correctly checking both standard conditions)
   provides a more principled way to flag genuine ambiguity — it currently does so for Clusters 2
   and 3.
4. **RT60's earlier win despite being mid-ranked by both methods** was specific to the two-method
   TOPSIS+GRA version described in this file's walkthrough; the current consensus pick (PureTemp 58
   in Clusters 0-2, PureTemp 53 in Cluster 3) comes from four methods and is not directly
   comparable to this analysis.
5. **~~No uncertainty quantification exists.~~ RESOLVED** — Monte Carlo now quantifies exactly how
   stable these near-tied ranks are under small perturbations of the weights and of the
   substantially-imputed `TC_W_mK` / `cycles_confidence` / `rho_H_MJ_m3` values. Result (K = 4,
   2,000 draws): the best Top-3 inclusion probability in any cluster is 36.4-39.3 % and the best
   Top-1 retention 16.3-18.6 % — much lower than other states (Assam ~95-96%), because
   Uttarakhand's feasible pool (29-30 candidates/cluster) is far larger and more homogeneous than
   Assam's (2-6/cluster). The consensus #1 itself is not the most perturbation-robust candidate:
   n-Octacosane (C28) has the highest Top-3 inclusion in every cluster (36-39 %), while PureTemp 58
   reaches the Top-3 in only 13-14 % of draws in Clusters 0-2. A real finding, not an error.
6. **~~Kendall's W is not recoverable from any committed artefact.~~ RESOLVED — current values
   verified directly (K = 4):** 0.796 (Cluster 0), 0.796 (Cluster 1), 0.782 (Cluster 2), 0.842
   (Cluster 3) — all comfortably above `08`'s own 0.6 "ambiguous regime" threshold, so the
   `[NOTE]` block does not fire for any cluster in the current run.
7. **An earlier generation of this phase is preserved in the plot tree** with a completely
   different Top-3 (RT54HC / RT55 / RT64HC) and a TOPSIS-vs-GRA Spearman of −1.000, from a run with
   a 25-row PCM database. Direct evidence that the recommendation is sensitive to database
   coverage — see `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`.

## Status

**COMPLETE (K = 4) — the fully degenerate result this file's walkthrough describes is fixed.** The
methodology was already sound in its construction — the Gaussian target transform applied before
anything else touches melting temperature, weights half data-driven and half declared-placeholder,
missing values flagged rather than hidden, and the script actively detecting the degeneracy it
produced. The two changes this file's earlier version recommended have both since happened:
PROMETHEE II (and VIKOR) were added as independent methods, and `07b`'s regime cap was fixed so it
actually runs before `07` and produces a real effect. Result: a partly differentiated
recommendation (the high-elevation Cluster 3 genuinely different; Clusters 0 and 1 identical),
Kendall's W of 0.782-0.842 (a much healthier agreement picture than the old pooled −0.930), and
Monte Carlo-quantified stability. The remaining
open items are the ones listed above that were never about the degeneracy (CoCoSo, Copeland
consensus, a real AHP elicitation) — genuine future work, not correctness bugs.
