> **⚠️ SUPERSEDED (2026-09-16).** This document describes a run of `08_mcdm_ranking.py` using
> the **5-criterion bespoke Tamil Nadu script** (`f_Tm`, `latent_heat_margin_ratio`, `rho_H_MJ_m3`,
> `TC_W_mK`, `cycles_confidence`), **k=5 clusters**, and an **uncorrected `Tm_target_C=57.0°C`**.
> The pipeline has since moved to the **8-criterion "unified with Rajasthan" engine** (matching
> `08_PHASE_6_AUDIT.md`'s description all along — this doc was written against a stale on-disk
> state), **k=3 clusters** (auto-selected, see `06_PHASE_4_AUDIT.md`), and a **corrected
> `Tm_target_C=67.0°C`** (see `CHANGELOG.md`'s 2026-09-16 entry). None of the specific numbers,
> weights, or plots below reflect the current pipeline. For current results, see
> `08_PHASE_6_AUDIT.md` (now updated with the 2026-09-16 run) and `09_PHASE_7_AUDIT.md` (physics
> validation, including a diagnosed disagreement between MCDM rank and simulated performance that
> this 5-criterion/k=5 run predates entirely). Kept below as an archived snapshot of the analysis
> method, not as current data.

# 14 — Phase 6 MCDM Ranking: Detailed Explanation + Plot-by-Plot Guide

Script: `08_mcdm_ranking.py` (repo root). Plots: `plots/generate_tamilnadu_plots.py`
→ `data/plots/tamilnadu_objective1/`. Data used below is the **actual on-disk output**
(`data/processed/pcm/mcdm_full_scores_by_cluster.csv`, `mcdm_topk_by_cluster.csv`,
`monte_carlo_stability.csv`, all last generated **2026-09-14**).

> **Note on `08_PHASE_6_AUDIT.md`**: that audit document describes a *different*,
> newer 8-criterion "unified with Rajasthan" version of `08_mcdm_ranking.py` (adds
> `supercooling`, `corrosion`, `cost`, splits `vol_latent_heat` out, entropy-cap fix,
> etc.). The script and CSVs actually present in this repo right now are the
> **original 5-criterion Tamil Nadu-specific engine** described in this script's own
> docstring — that is what produced every number and every plot referenced in this
> document and in `15_CLUSTER0_BUMP_CHART_EXPLAINED.md`. If the two docs ever disagree,
> trust this one for what's on disk today.

---

## 1. What Phase 6 does

Phase 5 (`07_feasibility_filter.py`) already threw out every PCM that fails hard
constraints (melting window, latent heat floor, cycling, supercooling, corrosion,
safety) per climate cluster. Phase 6 takes the **survivors** — 9 to 15 candidates
per cluster — and ranks them against each other on five weighted criteria, using
four independent MCDM methods, then merges those four rankings into one consensus
ranking and stress-tests that consensus with a 5,000-draw Monte Carlo simulation.

```
feasibility_survivors_by_cluster.csv (07's output)
        │
        ▼
   rank_cluster()  per cluster_id
        │
        ├─ build 5 criteria (f_Tm, latent_heat_margin_ratio, rho_H, TC, cycles_confidence)
        ├─ min-max normalize each criterion to [0,1] within the cluster
        ├─ blend weights: 50% Shannon entropy (data-driven) + 50% AHP prior (Table 13)
        ├─ score with TOPSIS, GRA, PROMETHEE II, VIKOR
        ├─ rank each method, combine via Borda (primary) + Copeland (cross-check)
        ├─ compute Kendall's W (do the 4 methods agree overall?)
        └─ run 5,000-draw Monte Carlo (perturb weights + properties) → stability metrics
        │
        ▼
mcdm_full_scores_by_cluster.csv   (every survivor, every method, every weight)
mcdm_topk_by_cluster.csv          (consensus_rank ≤ 3 only)
monte_carlo_stability.csv         (top-3 inclusion %, top-1 retention %, per PCM)
```

## 2. The five criteria

| Criterion | Direction | What it measures | Formula |
|---|---|---|---|
| `f_Tm` | benefit | How close the PCM's melting point is to the cluster's Tm target (57.0°C for every Tamil Nadu cluster — see §5) | Gaussian fitness `exp(−(Tm−Tm_target)²/(2σ²))`, σ=4K. **Not** treated as a plain benefit/cost — being *too hot* is scored exactly as badly as being *too cold* by the same margin. |
| `latent_heat_margin_ratio` | benefit | Energy storage capacity **relative to what this cluster actually needs**, not the raw number | `latent_heat_kJ_kg / L_required_kJ_per_kg` (L_required is cluster-specific, 301–326 kJ/kg here). This deliberately replaces the plan's literal "raw latent heat" criterion — raw latent heat is the same 15ish numbers everywhere and carries zero climate signal; the margin ratio does. |
| `rho_H_MJ_m3` | benefit | Volumetric latent heat density (energy per litre of tank volume, not per kg) | precomputed upstream from `density × latent_heat` |
| `TC_W_mK` | benefit | Thermal conductivity — how fast the PCM can charge/discharge | raw W/m·K |
| `cycles_confidence` | benefit | Log-scaled, NaN-safe confidence in the PCM's tested cycle life | `log1p(cycles_tested)/log1p(max_cycles)`, median-imputed where missing (flagged via `cycles_confidence_imputed`) |

All five are min-max normalized to `[0,1]` **within each cluster's own survivor
set** before any method sees them — so "high" or "low" always means relative to
the other candidates in that cluster, not some global scale.

## 3. Weighting: entropy + AHP blend

```
w_final = 0.5 · w_entropy + 0.5 · w_AHP        (renormalized to sum to 1)
```

- **`w_AHP`** — a fixed expert-elicited prior from the project's Table 13 (renormalized
  over just these 5 criteria): f_Tm 0.30, latent-heat-margin 0.25, ρH 0.15, TC 0.163, cycling 0.138.
- **`w_entropy`** — Shannon entropy computed **per cluster** from that cluster's own
  normalized matrix: a criterion where survivors are all bunched together (low
  variance/low information) gets a low weight; a criterion that actually spreads
  the candidates out gets a high weight. This is why the blended weight differs
  cluster to cluster even though `w_AHP` doesn't.

Actual blended weights, this run:

| Cluster | f_Tm | latent_heat_margin | ρH | TC | cycles_confidence | n survivors |
|---|---|---|---|---|---|---|
| 0 | 0.243 | 0.235 | 0.152 | 0.149 | 0.221 | 15 |
| 1 | 0.235 | 0.179 | 0.144 | 0.277 | 0.165 | 9 |
| 2 | 0.214 | 0.198 | 0.134 | 0.173 | 0.281 | 13 |
| 3 | 0.214 | 0.198 | 0.134 | 0.173 | 0.281 | 13 |
| 4 | 0.235 | 0.179 | 0.144 | 0.277 | 0.165 | 9 |

Clusters 1&4 and 2&3 land on **identical** weight vectors — their feasibility
survivor sets are near-duplicates (same climate regime showing up twice in the
5-cluster GMM split), which is exactly the situation the bump-chart generator's
own bug-fix comment flags (see `plots/generate_tamilnadu_plots.py` around Plot 11).
Cluster 1/4 lean harder on thermal conductivity (0.277); Cluster 2/3 lean harder
on cycling confidence (0.281) — those are the two clusters where TC and cycling
values happen to spread the survivor set out the most.

## 4. The four ranking methods

Each method consumes the same normalized matrix and the same weight vector but
aggregates trade-offs differently — which is *why* they don't always agree, and
why a consensus step is needed at all.

| Method | Core idea | What it rewards | What it's sensitive to |
|---|---|---|---|
| **TOPSIS** | Euclidean distance to the ideal point vs. distance from the anti-ideal point; score = `d(anti-ideal) / (d(ideal) + d(anti-ideal))` | Being close to the *best of everything*, balanced across all criteria | Squared distances → one very bad criterion hurts, but is diluted by good ones |
| **GRA** (Grey Relational Analysis) | Grey relational coefficient vs. the ideal (max) row, `(Δmin + ζΔmax)/(Δ + ζΔmax)`, ζ=0.5, then weighted sum | Consistent closeness to the ideal on *every* criterion simultaneously | Punishes candidates that are weak on several criteria even if strong on one — less forgiving of lopsided profiles than TOPSIS |
| **PROMETHEE II** | Pairwise outranking: for every pair of candidates and every criterion, a V-shape preference function (indifference q=0.10, preference p=0.30 of the [0,1] range) accumulates a net flow `φ = φ⁺ − φ⁻` | Being **decisively better** than many peers on many criteria, pairwise | The *distribution* of the whole survivor set, not just the ideal row — a candidate can look fine in isolation but score poorly if many peers narrowly beat it |
| **VIKOR** | Compromise ranking `Q = v·S + (1−v)·R`, v=0.5, where S = weighted-sum gap-to-ideal (group utility) and R = the single worst weighted gap (individual regret) | Candidates with **no catastrophic weakness** on any one criterion, even if unremarkable overall | Rewards balanced/no-outlier profiles; penalizes a single very bad criterion harder than TOPSIS/GRA do, via R |

**VIKOR also runs a formal post-check**: the "acceptable advantage" condition
(is the #1-to-#2 Q gap ≥ `1/(n−1)`?) and reports a compromise-set note instead of
silently declaring a single winner when that condition fails.

## 5. Why the Tm target is *constant* across clusters (important)

Every one of the 5 clusters has `Tm_target_C = 57.0`. This is a deliberate,
documented design choice upstream (the script's own comment: *"Tm_target being
CONSTANT is Section 6.3's explicit design"*) — the storage-tank target temperature
for this objective doesn't vary by climate regime, only the *amount of energy
required* (`L_required_kJ_per_kg`, which ranges 301–326 kJ/kg across clusters) does.

**Consequence**: `f_Tm` scores the *same* physical Tm the same way in every
cluster, and `latent_heat_margin_ratio` only shifts a little (the denominator
changes by ~8% cluster to cluster). With 3 of the 5 criteria (ρH, TC, cycling)
being climate-independent altogether, the survivor rankings end up **very similar
across clusters** — and the consensus #1 PCM is `n-Octacosane (C28)` in **all
five clusters** (confirmed by `mcdm_topk_by_cluster.csv`; the script prints its
own `[FINDING]` flagging exactly this). This is a legitimate, reportable outcome,
not a bug — it falls directly out of a constant Tm target, not out of the ranking
math being broken.

## 6. Consensus: Borda, Copeland, Kendall's W

- **Borda count** (primary): for each method, `points = n − rank + 1`; sum across
  the 4 methods. This is what `consensus_rank` is sorted by.
- **Copeland** (cross-check): for every pair of candidates, +1 if a candidate beats
  the other in more of the 4 methods than it loses, −1 if fewer, 0 if tied; summed.
- **Kendall's W**: how much the 4 methods' rankings agree overall (1 = perfect
  agreement, 0 = no agreement). Actual values this run:

| Cluster | Kendall's W | Interpretation |
|---|---|---|
| 0 | 0.842 | strong agreement |
| 1 | 0.956 | very strong |
| 2 | 0.835 | strong |
| 3 | 0.835 | strong |
| 4 | 0.956 | very strong |

In every cluster this run, `borda_copeland_agree = True` — the two consensus
methods pick the same #1, so there's no "report both" situation to flag here.

## 7. Monte Carlo stability (5,000 draws)

Per draw: weights are redrawn from a Dirichlet distribution centered on the
blended nominal weights (concentration 30 — tighter concentration = more trust
in the nominal weights); Tm is perturbed ±N(0,1K); latent heat, TC and ρH are
perturbed by relative Gaussian noise (5% / 10% / 8% std respectively); TOPSIS
alone re-scores each draw (the standard simplification for MC-MCDM studies —
recomputing all 4 methods 5,000× per cluster is unnecessary). Reports, per PCM:

- **Top-3 inclusion probability** — fraction of 5,000 draws where it lands in the top 3
- **Top-1 retention rate** — fraction where it's specifically #1
- **Mean Spearman ρ vs. the unperturbed baseline ranking** — overall draw-to-draw stability

Consensus #1 in every cluster, `n-Octacosane (C28)`, is Top-3-stable **76.8%**
(cluster 0/2/3) to **90.2%** (cluster 1/4) of draws but Top-1-*retention* is only
**~51–71%** — i.e. it's a robust Top-3 pick but not an unshakeable #1: in roughly
a third to a half of plausible weight/property perturbations, something else
edges it out for the very top spot. See `15_CLUSTER0_BUMP_CHART_EXPLAINED.md` §5
for the full cluster-0 breakdown.

## 8. Why some PCMs rank higher than others (the general mechanism)

**No single criterion decides the winner.** Because every criterion is min-max
normalized to `[0,1]` *within* the cluster before weighting, a candidate wins by
being the best *weighted combination*, not the best on any one axis. Concretely,
for Cluster 0 (weights f_Tm 0.243 / latent-margin 0.235 / ρH 0.152 / TC 0.149 /
cycling 0.221), the normalized scores for the top two finishers are:

| PCM | f_Tm (norm) | latent-margin (norm) | ρH (norm) | TC (norm) | cycling (norm) | weighted sum |
|---|---|---|---|---|---|---|
| **n-Octacosane (C28)** — #1 | 0.387 | 0.927 | **1.000** | 0.840 | 0.675 | **0.738** |
| n-Hexacosane (C26) — #2 | **1.000** | **1.000** | 0.364 | 0.563 | 0.121 | 0.644 |

n-Hexacosane's melting point (56.5°C) is almost exactly on the 57°C target — its
`f_Tm` is essentially perfect (1.000 normalized) and it also has the highest raw
latent-heat margin in the cluster. But it loses the #1 spot because it is
**mediocre-to-weak on the other three criteria**: its volumetric latent heat
density (ρH) is only 36% of the cluster's normalized range, and its cycle
confidence is the second-lowest in the whole survivor set (0.121 normalized —
only 1,404 tested cycles vs. n-Octacosane's 1,581).

n-Octacosane, by contrast, melts at 61.6°C — 4.6K off-target, giving it a much
weaker `f_Tm` (0.387 normalized) — but it has the **highest volumetric energy
density in the cluster** (ρH, 230.2 MJ/m³, normalized to 1.000), near-top thermal
conductivity (0.840 normalized), and strong, well-tested cycling data (0.675
normalized, 1,581 cycles tested). Its weighted sum wins because it is *never
weak*, while n-Hexacosane is *excellent on two criteria and poor on two others*.

**This is the general pattern across the whole ranked list**: candidates that
are simultaneously good-to-great on 4–5 criteria beat candidates that are
outstanding on 1–2 criteria and weak on the rest — because entropy+AHP weighting
spreads real weight across all five axes (no criterion here exceeds ~24% of the
total), so no single strength or single weakness is ever enough to decide the
outcome by itself. The bottom of the ranking (`PlusICE A52`, `RT62HC`,
`n-Nonacosane (C29)`) is occupied by candidates that are weak-to-worst on
*multiple* criteria at once — e.g. `PlusICE A52` has the worst ρH, worst TC,
*and* a below-median `f_Tm` in Cluster 0, with nothing to compensate.

## 9. Plot-by-plot guide (Phase 6 / MCDM-relevant plots)

All in `data/plots/tamilnadu_objective1/` (PNG static + matching `.html` Plotly
interactive, unless noted). Source: `plots/generate_tamilnadu_plots.py`.

| # | File(s) | Reads | Shows |
|---|---|---|---|
| **03** | `03_melting_point_vs_latent_heat*` | `feasibility_survivors_by_cluster.csv` | Scatter of every *feasible* survivor, Tm vs. latent heat, colored by cluster, with the cluster's admissible melting window shaded and the latent-heat floor line drawn. Shows the search space Phase 6 ranks *within* — not the ranking itself. |
| **04** | `04_feasible_candidates_highlighted.png` | feasibility + full PCM database | Same axes but contrasts the full database (grey) against the survivors that passed Phase 5's filter (colored) — visual proof of how much Phase 5 already narrowed the field before Phase 6 even runs. |
| **05 / 06** | `05_pcm_survivors_per_cluster*`, `06_pcm_feasibility_scatter_and_survivors.png` | feasibility | Bar chart of survivor count per cluster (9–15 here) — the `n` that Phase 6 ranks in each cluster. |
| **07** | `07_bump_chart_ranks*` (+ 5 per-cluster variants) | `mcdm_full_scores_by_cluster.csv` | **The core Phase 6 visualization.** One line per (PCM, cluster) pair tracking its rank across TOPSIS → GRA → PROMETHEE → VIKOR → CONSENSUS. Top-5-per-cluster depth (not a global top-12 — see the script's own bug-fix comment). This is what directly shows method disagreement / rank volatility. Cluster 0's version gets a full dedicated breakdown in `15_CLUSTER0_BUMP_CHART_EXPLAINED.md`. |
| **08** | `08_method_rank_correlation_heatmap*` | `mcdm_topk_by_cluster.csv` (top-3 per cluster, n=15 rows) | Spearman ρ / Kendall τ between every pair of methods' ranks. This run: **GRA ↔ PROMETHEE ρ=1.00** (near-identical on the top-3 sets), **TOPSIS ↔ VIKOR ρ=0.91**, but **GRA/PROMETHEE ↔ VIKOR ρ=0.61** — VIKOR is the clear structural outlier among the top-3 candidates statewide, consistent with its "no catastrophic weakness" logic disagreeing with GRA/PROMETHEE's ideal-closeness logic. |
| **09** | `09_monte_carlo_top3_probability*` | `monte_carlo_stability.csv` | Horizontal bar chart, every PCM's Top-3 inclusion probability across all clusters, with 50%/80% confidence reference lines. Read this before trusting any single deterministic Top-3 pick — a pick with <50% MC inclusion is fragile to weight/property uncertainty. |
| **10** | `10_rank_reversal_violin_bar*` | `mcdm_topk_by_cluster.csv` | Left: violin of rank distribution per method per cluster (wide violin = methods disagree a lot in that cluster). Right: bar chart of `rank_spread` (max rank − min rank across the 4 methods) for the 15 most volatile candidates — the ones methods can't agree on. |
| **11** | `11_agreement_plot*` | `mcdm_topk_by_cluster.csv` + `physics_validation_results.csv` | MCDM consensus rank vs. **simulated physical performance rank** (from Phase 7's tank simulation) — the real-world sanity check on whether "MCDM says #1" actually means "performs best in simulation." Points are jittered slightly because Clusters 1&4 / 2&3 land on identical integer coordinates (their survivor sets and weights are near-duplicates, per §3). |
| **11b** | `11b_physics_vs_mcdm_all_clusters*` | `physics_validation_results.csv` | One subplot panel per cluster: simulated annual solar fraction vs. MCDM consensus rank, with the published 54–84% benchmark band shaded — shows whether the MCDM-endorsed Top picks actually land in a physically credible performance range. |
| **13** | `13_recommended_pcm_summary*` | `mcdm_topk_by_cluster.csv` | Final deliverable view: Top-3 PCM per cluster with latent heat, Tm and ρH annotated — the "what do I actually recommend" plot. |

## 10. Outputs reference

| File | Contents |
|---|---|
| `data/processed/pcm/mcdm_full_scores_by_cluster.csv` | Every survivor × every method's score/rank, blended weights, Borda/Copeland/consensus, Kendall's W, MC columns |
| `data/processed/pcm/mcdm_topk_by_cluster.csv` | Rows with `consensus_rank ≤ 3` only |
| `data/processed/pcm/monte_carlo_stability.csv` | Per-PCM Top-3 inclusion %, Top-1 retention %, mean Spearman ρ vs. baseline |

## 11. Next steps in the pipeline

`08_mcdm_ranking.py` → `09_recommendation_cards.py` → `10_physics_validation.py`
(Phase 7, no longer optional per the script's own trailer message).
