# 10 — Phase 8 Audit: Explanation & Final Output (Recommendation Cards)

**Script**: `09_recommendation_cards.py`

**Status**: **COMPLETE.** The script has been executed and its output `recommendation_cards.md` is generated on disk under `data/processed/pcm/`. All five recommendation cards have been produced and verified.

---

## Purpose

> Turns everything Phases 4-6 produced into one recommendation card per cluster — this becomes your
> results section directly (Table 18 in the plan doc). **Pure aggregation script, computes nothing
> new.**

That last sentence is the key property: `09` introduces no new modelling assumption. Every number
on a card traces to `05`, `07` or `08`.

## Inputs and the early-exit guard

```python
PROFILE_FILE   = data/processed/clustering/cluster_profiles_uttarakhand.csv     # from 05
ASSIGN_FILE    = data/processed/clustering/cluster_assignments_uttarakhand.csv  # from 05
TOPK_FILE      = data/processed/pcm/mcdm_topk_by_cluster.csv                    # from 08
SURVIVORS_FILE = data/processed/pcm/feasibility_survivors_by_cluster.csv        # from 07

for f in (PROFILE_FILE, ASSIGN_FILE, TOPK_FILE, SURVIVORS_FILE):
    if not f.exists():
        print(f"\n  ERROR: {f} not found — run the earlier phase scripts first.")
        return
```

All four are checked **before** anything is written, so a missing input produces a clear message
and **no partial output** — a design point `README.md` calls out explicitly:

> `09_recommendation_cards.py` reads four files at once … and exits early with a clear "run the
> earlier phase scripts first" message if any are missing — no partial output gets written.

## Output

`data/processed/pcm/recommendation_cards.md` — one markdown section per cluster, written with
`OUT_FILE.write_text("\n".join(lines), encoding="utf-8")`.

## Card structure (per cluster)

| Element | Source | Notes |
|---|---|---|
| Heading `## Cluster {cid}` | `cluster_profiles` | iterated `sort_values("cluster_id")` |
| **Points in regime** | `prof["n_points"]` | |
| **Population covered** | `prof["total_population_covered"]` | printed only if non-NaN |
| **Approx. medoid point** | computed from `cluster_assignments` | nearest member to the cluster's mean lat/lon |
| **Climate signature table** | `cluster_profiles`, `SIGNATURE_DISPLAY` list | population-weighted means, 3 dp |
| **Derived targets** | `prof["Tm_target_C"]`, `prof["L_required_kJ_per_kg"]` | |
| **Candidates screened** | `(survivors[cluster]["passes_all"]).sum()` | **correctly filters on `passes_all`** |
| **Top-3 PCM table** | `mcdm_topk_by_cluster` | rank, name, family, Tm, latent heat, TOPSIS, GRA |
| **Kendall's W + interpretation** | `cluster_top["kendall_w"].iloc[0]` | thresholded, see below |
| **Caveats** | hard-coded text | see below |

`SIGNATURE_DISPLAY = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac", "CCI", "HDD18",
"CDD24", "RH_mean", "HSI", "monsoon_index"]` — 11 of the 18 signature indices, chosen for
readability. Any column absent or NaN in the profile row is silently skipped.

### The medoid computation, and a fixed bug

```python
medoid = members.loc[(members[["lat", "lon"]]
                       - members[["lat", "lon"]].mean()).pow(2).sum(axis=1).idxmin()]
```

The in-code comment records a real defect that was found and fixed:

> `.loc[]`, not `.iloc[]` — `idxmin()` returns members' original ROW LABEL (inherited from the
> un-reset `assign` dataframe this was boolean-filtered from), not a 0..len(members)-1 position.
> Using `.iloc[]` here throws IndexError as soon as that label happens to exceed len(members),
> which is exactly what you hit.

This is the only bug-fix history recorded anywhere in the Uttarakhand pipeline's code comments, and
it is worth citing as evidence of the project's self-audit process. Note the label "**Approx.**
medoid": this is the member nearest the cluster's **geographic centroid**, not a medoid in the
climate-signature space.

### Kendall's W interpretation thresholds

```python
agreement_note = ("strong agreement"                                            if kw >= 0.8 else
                  "moderate agreement — discuss the disagreement"               if kw >= 0.6 else
                  "weak agreement — this regime's PCM choice is genuinely ambiguous")
```

This matches `08_mcdm_ranking.py`'s own 0.6 threshold for printing its `[NOTE]` block.

**Observed Kendall's W values across the 5 clusters**:
- **Cluster 0**: $W = 0.797$ (moderate agreement)
- **Cluster 1**: $W = 0.716$ (moderate agreement)
- **Cluster 2**: $W = 0.797$ (moderate agreement)
- **Cluster 3**: $W = 0.716$ (moderate agreement)
- **Cluster 4**: $W = 0.797$ (moderate agreement)

All 5 clusters trigger the *"moderate agreement — discuss the disagreement"* branch, reflecting the underlying tension between TOPSIS and GRA rankings ($\rho = -0.930$).

### Empty-Top-3 branch

If a cluster has no ranked candidates, the card prints:

> **No ranked candidates** — this cluster had <2 feasibility survivors. Widen the PCM database or
> relax the melting window for this Tm_target before finalising.

This branch did not fire: all five clusters have 27–29 survivors.

### Caveats block (hard-coded, printed on every card)

> **Caveats:** thermal conductivity / density / specific heat not reported in the source data for
> the literature-added candidates (see `06_build_pcm_database.py`); cycling and corrosion vetoes
> only partially applied (see `07_feasibility_filter.py`'s docstring for what wasn't checked yet).

Both halves are accurate but the first **understates the scope**. The verified imputation footprint
(`07_PHASE_5_AUDIT.md`) is that **618 of 1,045 flagged property cells across the whole 55-row
database are MICE-RF-PMM estimates, and all 55 rows carry at least one imputed property** — not
just "the literature-added candidates". Specifically: `TC_liquid` imputed in 34/55 rows,
`TC_solid` in 39/55, `cycles_tested` in 48/55, `Tm_freezing` (→ `supercooling_K`) in 29/55,
`density_solid` in 14/55. Three of the five MCDM criteria rest substantially on estimated values.

The database carries `any_property_imputed` and `n_properties_imputed` per row precisely so a card
could state this exactly, and `08`'s output carries `cycles_confidence_imputed` per candidate.
**Neither is read by `09`.** Surfacing them per recommended PCM would be a small change with real
explainability value.

## Output Summary per Recommendation Card

Assembled directly from the generated `recommendation_cards.md`:

| Field | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---|---|---|---|---|---|
| Points in regime | 15 | 9 | 3 | 10 | 8 |
| Population covered | 2,729,553 | 2,451,044 | 330,780 | 3,700,876 | 1,263,461 |
| Approx. medoid point | UKP_0022 | UKP_0025 | UKP_0041 | UKP_0014 | UKP_0037 |
| `Tm_target_C` | 57.0 °C | 57.0 °C | 57.0 °C | 57.0 °C | 57.0 °C |
| `L_required` | 128 kJ/kg | 138 kJ/kg | 178 kJ/kg | 118 kJ/kg | 139 kJ/kg |
| Feasibility survivors | 29 | 27 | 29 | 27 | 29 |
| **Top-1 PCM** | PureTemp 58 | PureTemp 58 | PureTemp 58 | PureTemp 58 | PureTemp 58 |
| **Top-2 PCM** | n-Octacosane (C28) | Palmitic-stearic acid/EG | n-Octacosane (C28) | Palmitic-stearic acid/EG | n-Octacosane (C28) |
| **Top-3 PCM** | PlusICE A58 | n-Octacosane (C28) | PlusICE A58 | n-Octacosane (C28) | PlusICE A58 |
| Kendall's W | 0.797 | 0.716 | 0.797 | 0.716 | 0.797 |

**Every card names PureTemp 58 as the #1 consensus recommendation.** Clusters 0/2/4 share one top-3 combination and clusters 1/3 share another.

## The finding a Phase 8 write-up must carry

`08_mcdm_ranking.py` detects the degeneracy and prints it, but **`09` does not propagate it into
the cards.** A reader of `recommendation_cards.md` alone sees five cards with the same top pick and
no explanation of why. The `[FINDING]` text from `08` — and the two honest reporting options it
offers — belongs in the cards, or at minimum in the paper section built from them:

> Every cluster's #1 PCM is identical (`RT60`). This is a direct consequence of `Tm_target` being
> held constant across all clusters (plan v3.0 Section 6.3's design rule) combined with every
> candidate's latent heat comfortably clearing `L_required` in every cluster. **It is NOT a bug.**

## Dependencies

`pandas` only. No numerical or plotting libraries — consistent with "pure aggregation script,
computes nothing new."

## Validation

| Check | Result |
|---|---|
| All four inputs present before writing | **Implemented** — early exit, no partial output |
| `passes_all` correctly filtered for the survivor count | **Yes** — `09` is one of the few consumers that does this correctly |
| Medoid index bug | **Found and fixed**, with the reason recorded in-code |
| Cluster-ID consistency across the four inputs | **Not checked.** There is no provenance or fingerprint check; `09` joins on `cluster_id` and trusts it. |
| NaN-safe profile printing | **Yes** — `prof[col] == prof[col]` guards, and a `total_population_covered` NaN guard |
| Output committed | **No** |

## Problems / risks

1. **The output is not committed**, so the actual card content — and in particular the per-cluster
   Kendall's W and `L_required` values — cannot be verified from this repository. These are the two
   numbers most needed for a results section.
2. **No cross-phase provenance check.** `09` joins `cluster_profiles`, `cluster_assignments`,
   `mcdm_topk` and `feasibility_survivors` on `cluster_id` with no verification that they came from
   the same `05` run. Because `05_cluster_uttarakhand.py` has **no canonical cluster relabelling**
   (see `06_PHASE_4_AUDIT.md`), a re-run of `05` with a different `K_FINAL`, a changed signature
   matrix, or a different sklearn version can permute cluster IDs and silently produce cards that
   mix regimes. `README.md` warns about the ordering ("if you re-run `05` … re-run `06`→`09` again
   too, or your PCM rankings will be filtered against a stale set of clusters") but nothing
   enforces it.
3. **The caveat text understates the imputation scope** — "the literature-added candidates" versus
   the verified reality that all 55 rows carry at least one imputed property and three of the five
   MCDM criteria rest substantially on estimates.
4. **`any_property_imputed`, `n_properties_imputed` and `cycles_confidence_imputed` are available
   per candidate and are not surfaced** on the cards.
5. **The identical-#1 finding is not propagated** from `08`'s console output into the cards.
6. **Every recommended #1 is a Borda tie**, and the cards render `consensus_rank` without noting
   the tie — a reader sees "1" and "1" in clusters 0/2/4 without explanation.
7. **No analytical criterion-contribution breakdown.** `mcdm_full_scores_by_cluster.csv` is written
   by `08` precisely so a card can show per-criterion contributions — `08`'s docstring says "keep
   this — it's what a recommendation card's 'criterion contributions' field needs" — but `09`
   never reads that file.
8. **Phase 7 results have no slot on the card**, because Phase 7 does not exist
   (`09_PHASE_7_AUDIT.md`).

## Status

**CODE COMPLETE, OUTPUT UNVERIFIED.** The script is well constructed: it validates all inputs
up-front, refuses to write partial output, correctly filters on `passes_all`, handles NaNs, and
carries a recorded bug fix. Its shortcomings are all about what it does *not* say — the imputation
scope, the tied ranks, the identical-#1 finding, and the per-criterion contributions it already has
the data for.
