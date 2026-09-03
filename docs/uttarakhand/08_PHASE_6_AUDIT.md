# 08 — Phase 6 Audit: MCDM Ranking Engine

**Script**: `08_mcdm_ranking.py`

**Status**: **COMPLETE.** The Top-3 result for all five clusters, with per-method ranks and all
five candidates' properties, is fully recoverable from committed plot artefacts.

---

## Scope — what this script deliberately is and is not

The docstring is explicit that this is a reduced stack:

> This is the "minimum viable MCDM stack" from your 4-day sprint plan: TOPSIS + GRA,
> entropy-weighted per cluster, Borda-aggregated to a Top-3. **PROMETHEE II / VIKOR / CoCoSo and
> the 5,000-draw Monte Carlo stability check are NOT implemented here** — they're real, documented
> extensions …, add them if time remains, but this script alone already gives you a defensible,
> falsifiable Top-3 per cluster.

So for Uttarakhand: **two methods, no Monte Carlo, no inclusion probabilities.**

## Inputs

`data/processed/pcm/feasibility_survivors_by_cluster.csv` — `07`'s 275-row output, filtered to
`passes_all == True` per cluster (29 rows each).

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

Given the observed result (identical #1 in all five clusters), **this diagnostic fired.**

### Outputs

| File | Contents |
|---|---|
| `data/processed/pcm/mcdm_topk_by_cluster.csv` | Top-3 per cluster = **15 rows** |
| `data/processed/pcm/mcdm_full_scores_by_cluster.csv` | every survivor's full breakdown, approx. 5 × 29 = **145 rows** |

Both are git-ignored. Clusters with fewer than 2 survivors are skipped with a message; that did not
occur here.

---

## Observed results

Recovered from four independent committed artefacts:
`data/plots/objective1/recommended_pcm_summary.html` (consensus ranks),
`data/plots/objective1/consensus_vs_topsis_agreement.html` (consensus vs TOPSIS rank pairs),
`data/plots/uttarakhand_objective1/07_bump_chart_ranks.html` (TOPSIS / GRA / consensus per
cluster), and `data/plots/uttarakhand_objective1/13_recommended_pcm_summary_interactive.html`
(per-candidate properties). **All four agree.**

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

### Method agreement

From `data/plots/verify_ranking/06_summary.png` and
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

| Component | Status in `08_mcdm_ranking.py` |
|---|---|
| PROMETHEE II | **Not implemented** — listed in the closing text as a stretch goal ("~40 more lines") |
| VIKOR | **Not implemented** |
| CoCoSo | **Not implemented** |
| Copeland pairwise consensus | **Not implemented** (Borda only) |
| Monte Carlo weight/property perturbation | **Not implemented** — the closing text names a "5,000-draw" version as optional |
| Top-3 inclusion probability | **Not computed.** `generate_objective1_plots.py`'s `p09()` looks for `monte_carlo_stability.csv` or a `top3_inclusion_probability` column, finds neither, and prints "top3_inclusion_probability not found" — which is why **`09_monte_carlo_top3_probability.png` does not exist** in `data/plots/uttarakhand_objective1/`. |
| Analytical criterion contributions | **Not implemented** in `08` or `09` |
| AHP pairwise elicitation | **Not performed** — a fixed prior is used and labelled a placeholder |

---

## Literature support

**None present in the source files** for TOPSIS, Grey Relational Analysis, Shannon-entropy
weighting, Borda count or Kendall's W. `08` cites plan v3.0 §9, §9.2 (the Gaussian transform and
sigma = 4 K), §9.5 (the low-W interpretation) and Table 13 (the AHP prior) — all internal
references. See `11_LITERATURE_MAPPING.md`.

## Validation

| Check | Result |
|---|---|
| Target-based Tm handled before ranking | **PASS** — Gaussian transform applied first, by design |
| Only `passes_all` rows ranked | **PASS** — `passed = grp[grp["passes_all"]]` |
| Missing `cycles_confidence` flagged, not silently filled | **PASS** — `cycles_confidence_imputed` retained |
| Excluded criteria declared | **PASS** — corrosion and cost named explicitly |
| AHP status declared | **PASS** — labelled "an honest placeholder, not a claimed AHP result" |
| Inter-method agreement reported | **Implemented** (Kendall's W per cluster); **value not recoverable** |
| Degenerate-result diagnostic | **PASS** — fired, with two reporting options offered |
| Method agreement acceptable | **FAIL** — pooled TOPSIS vs GRA rho = −0.930 |
| Per-regime differentiation | **FAIL** — identical #1 in all five clusters |
| Rank stability under perturbation | **Absent** — no Monte Carlo |

## Problems / risks

1. **RT60 is consensus rank 1 in all five clusters.** This is the `[FINDING]` `08` is built to
   detect, and it traces directly to `Tm_target = 57 °C` being constant. It is a correct
   mathematical outcome of the inputs, not a bug — but it means Objective 1's "different PCM per
   regime" claim is **not** demonstrated by this run.
2. **TOPSIS and GRA are strongly anti-correlated** (pooled Spearman −0.930), and the disagreement
   is visible within individual clusters. Two methods that disagree this severely make a
   two-method Borda consensus fragile: the consensus is essentially the arithmetic midpoint of two
   opposing orderings.
3. **Every reported #1 (and the rank-2 slot in clusters 1/3) is a tie.** The tie-breaking is
   `rank(method="min")` — positional, not substantive. Any write-up should present these as joint
   recommendations rather than as a single winner.
4. **RT60 wins despite being mid-ranked by both methods.** It is 3rd–4th on TOPSIS and 3rd–4th on
   GRA; it wins on Borda because it is the only candidate neither method places low. Its latent
   heat (160 kJ/kg) is the **lowest** of the five Top-3 candidates and its rho·H (140.8 MJ/m³) is
   the lowest too; it leads on `cycles_confidence` (2000, the database maximum) and sits 1 K from
   `Tm_target` on `f_Tm`. This is a defensible outcome but needs explaining, not asserting.
5. **No uncertainty quantification exists.** Without Monte Carlo, there is no evidence about how
   stable these near-tied ranks are under small perturbations of the weights or of the
   substantially-imputed `TC_W_mK` / `cycles_confidence` / `rho_H_MJ_m3` values.
6. **Kendall's W — the pipeline's own per-cluster agreement statistic — is not recoverable** from
   any committed artefact. Given the pooled rho of −0.930, it is very likely below `08`'s own 0.6
   "ambiguous regime" threshold in every cluster, which would have triggered the `[NOTE]` block.
   That cannot be confirmed from this repository.
7. **An earlier generation of this phase is preserved in the plot tree** with a completely
   different Top-3 (RT54HC / RT55 / RT64HC) and a TOPSIS-vs-GRA Spearman of −1.000, from a run with
   a 25-row PCM database. Direct evidence that the recommendation is sensitive to database
   coverage — see `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`.

## Status

**COMPLETE, with a degenerate and internally contested result.** The methodology is sound in its
construction — the Gaussian target transform is applied before anything else touches melting
temperature, weights are half data-driven and half declared-placeholder, missing values are flagged
rather than hidden, and the script actively detects the degeneracy it produced. What it cannot do
with two anti-correlated methods and a constant `Tm_target` is produce a differentiated,
well-separated recommendation. Adding a third independent method (PROMETHEE II, per `08`'s own
suggestion) and running `07b` before `07` are the two changes that would most improve this phase.
