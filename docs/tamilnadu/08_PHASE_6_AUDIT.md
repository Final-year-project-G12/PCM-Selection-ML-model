# 08 — Phase 6 Audit: Multi-Criteria Ranking Engine

Script: `08_mcdm_ranking.py`.

> **UNIFIED WITH RAJASTHAN (2026-09-08).** `era5-tamilnadu/08_mcdm_ranking.py` is now a
> byte-for-byte port of `era5-rajasthan/08_mcdm_ranking.py` — same criteria set, same four
> methods, same aggregation, same Monte Carlo, same provenance hard-fail — differing only in
> `STATE_NAME`, the nested `data/processed/{pcm,clustering}/` layout, one `name → pcm_id`
> column normalisation, and two hint strings. It **replaces** the earlier bespoke Tamil Nadu
> script (5 criteria, PROMETHEE on the Gaussian `f_Tm`, 5000 draws, no Kendall's W /
> pairwise-agreement diagnostics). Bug audit of that old script: **VIKOR sign was correct**
> (denominator `max − min > 0`); its entropy function lacked the `<2 real values → weight 0`
> guard but the bug was **latent** (its 5-criterion set never had an all-NaN column) — the
> 8-criterion set here makes the guard load-bearing and this port carries it. Kappa
> calibration lives in Phase 5 (`07_feasibility_filter.py`), whose ported
> `calibrate_kappa_for_cluster()` uses the correct `>= k` direction.
>
> **RE-RUN 2026-09-16 — clean run against elevation-corrected data, TWO inert fixes, ONE
> criterion added that genuinely helped one of three clusters, investigation closed.** The
> results below supersede the 2026-09-08 numbers. Full step-by-step in `CHANGELOG.md`'s
> 2026-09-16 entries and the full physics-agreement investigation in `09_PHASE_7_AUDIT.md`
> (read that doc for the complete story — this doc summarizes the ranking-engine side of it):
> 1. **`TM_TARGET_C` correction now actually in effect** (57.0°C → **67.0°C**) — a directory-layout
>    path bug (fixed in `config.py`) was silently resolving to a stale, uncorrected
>    `pcm_shared_config.py`; the corrected copy has `T_DELIVERY_C` raised 50→60°C per an
>    Avargani et al. (2021) citation-accuracy fix.
> 2. **`Tm_fitness`/`f_Tm` scored against `Tm_target_capped_C`** (the kt_worst_month
>    achievability ceiling), not the raw `Tm_target_C`. More physically grounded, but **inert**
>    on rank order by itself (every survivor already sits below both numbers).
> 3. **Asymmetric Gaussian** (σ=2K above target, 4K below). **Inert**: Constraint 6 excludes
>    every candidate above the ceiling, so the branch this changes never executes.
> 4. **`thermal_margin` added as a 9th criterion** — `Tm_target_capped_C − Tm`, rewarding
>    headroom below the ceiling. **Genuinely fixed Cluster 0** (physics-agreement Spearman ρ:
>    -0.595 → +0.381) but did **not** fix Clusters 1/2 (+0.176→+0.103, +0.048→+0.024) — root
>    cause there is a dynamic climate-PCM interaction no static criterion can capture (proven
>    with a same-PCM-different-cluster test — see `09_PHASE_7_AUDIT.md` §3). **Investigation
>    closed 2026-09-16**: the criterion is kept (real, cited improvement, harmless where it
>    doesn't apply), and Clusters 1/2's residual disagreement is documented as a genuine finding,
>    not chased further — see `09_PHASE_7_AUDIT.md` §5 for the full reasoning against continuing
>    to tune criteria to match one physics simulation's output.

## Criteria (9, was 8 exact from Table 13 — see `thermal_margin` note) and weights

| Criterion | Direction | AHP prior | Notes |
|---|---|---|---|
| `Tm_fitness` | benefit | 0.14 (was 0.24 — split with `thermal_margin`, see below) | Gaussian target fitness `exp(−(Tm−Tm_target)²/(2σ²))` against **`Tm_target_capped_C`** (achievability ceiling, since 2026-09-16 — was the raw uncapped `Tm_target_C`), **asymmetric σ**: 4K below target, 2K above (since 2026-09-16, confirmed inert on this dataset — see the note at the top of this doc) |
| `latent_heat` | benefit | 0.20 | **climate-relative**: `latent_heat_kJ_kg / L_required` for that cluster |
| `vol_latent_heat` (ρL) | benefit | 0.12 | separate criterion, kept |
| `thermal_conductivity` | benefit | 0.13 | |
| `cycling` | benefit | 0.11 | **log-scaled** `cycles_confidence = log1p(cycles)/log1p(max_cycles)`, NaN-safe |
| `supercooling` | cost | 0.08 | **entropy weight capped at 2× prior (0.16)** — see below |
| `corrosion` | cost | 0.06 (cluster-rescaled 1×–2× by HSI) | structural proxy `2.0 if Inorganic else 1.0` — inert (0 salt hydrates) |
| `cost` | cost | 0.06 | always NaN → entropy weight 0 via the `<2 real values` guard |
| `thermal_margin` | benefit | 0.10 (new, 2026-09-16) | `Tm_target_capped_C − Tm` — headroom below the achievability ceiling. **Not** in the plan doc's literal Table 13 — a documented deviation, same footing as the climate-relative `latent_heat` / log-scaled `cycling` transforms. Its prior was carved out of `Tm_fitness`'s original 0.24 (0.14+0.10=0.24) so the combined "Tm-related" share of the weight pie is unchanged, not silently inflated relative to the other 7 criteria. |

Blend: `w_j = 0.5·w_entropy_j + 0.5·w_AHP_j`, per cluster, from that cluster's own filtered matrix.

### Supercooling entropy-weight cap (the key Phase 6 correction)

The Shannon-entropy formula overweights a near-zero-ideal cost criterion: supercooling has many
candidates reporting ~0 K, so the column is near-degenerate and the formula reads that as low
entropy ⇒ high information ⇒ spuriously large weight (0.48–0.64 in the first 8-criterion run).
Rajasthan's Phase 7/8 diagnosed this as the cause of the negative physics-validation correlation
(the Phase 8 supercooling-penalty sweep *worsening* agreement confirmed the direction:
*over*-weighted, not under-weighted). Fix: any cost criterion whose ideal ≈ 0 (supercooling, plus
corrosion/cost if they ever get real data) has its entropy-derived weight **held at ≤ 2× its
Table-13 prior** (supercooling ≤ 0.16) and the remaining criteria rescaled to fill `1 − Σ caps`,
*before* the 50/50 entropy-AHP blend. Applied identically in both states.

## Methods, aggregation, diagnostics (all ported from Rajasthan)

- **TOPSIS / PROMETHEE II / VIKOR / GRA**, missing values excluded-by-omission (not zero-filled).
- **PROMETHEE II handles Tm natively**: `|Tm − Tm_target|` distance with q=2K / p=8K linear V-shape,
  not the Gaussian `f_Tm` the other three consume — this is why PROMETHEE can be a structural outlier.
- **Borda** (primary) + **Copeland** (cross-check), with a Top-3 disagreement flag.
- **Kendall's W** with plan-doc thresholds (W>0.8 strong, W<0.6 ambiguous).
- **Pairwise method-agreement** (Spearman ρ / Kendall τ per method pair) + outlier summary.
- **λ=0 vs λ=0.5 Top-3 ablation** (is the entropy component load-bearing?).
- **Provenance hard-fail**: `assert_fingerprint_match` on `upstream_cluster_profile_fingerprint`.
- **Monte Carlo**: `N_DRAWS = 5000` (raised from the 1000 fast-iteration fallback 2026-09-16 for
  the final reported run — Rajasthan still at 1000, not yet raised there),
  Dirichlet weight draws + Gaussian property perturbation + family-distribution sampling for
  imputed properties. Reports Top-3 inclusion %, Top-1 retention %, rank-reversal freq, mean ρ vs baseline.

## Outputs

`mcdm_full_rankings.csv` (full per-survivor audit trail — raw `crit_*` values, blended `weight_*`,
all 4 method ranks + raw scores, Borda/consensus/Copeland, Kendall's W, entropy-dominant flag,
λ-ablation flags, MC columns, plus legacy aliases so Phase 7/8 read it unchanged),
`mcdm_topk_by_cluster.csv` (rows with `consensus_rank ≤ 3`), `monte_carlo_stability.csv`,
`mcdm_method_agreement.csv`, `outputs/qc_montecarlo_inclusion.html`. All carry
`upstream_cluster_profile_fingerprint`.

## Results — 2026-09-16 final run (3 clusters, 8/10/8 survivors, n=26; `Tm_target_C=67`, 9 criteria, `N_DRAWS=5000`)

- **`thermal_margin` is now the dominant entropy criterion in all 3 clusters** (weight 0.51–0.64),
  ahead of `Tm_fitness` (0.15–0.20) — the same one-criterion-domination pattern that flagged
  `Tm_fitness` before has simply moved to the new criterion. Blended `weight_thermal_margin` ≈
  0.31–0.37. This is expected (the whole point of the fix was to give this signal real weight)
  and is not itself a problem — see `09_PHASE_7_AUDIT.md` for why it helped Cluster 0 specifically
  and not the other two.
- **Supercooling blends to ≈ 0.118–0.120** — comfortably under its 0.16 cap; the entropy-cap
  guard is present and correct but isn't binding on this run's data.
- **Kendall's W moved**: 0.848 (C0, was 0.786 — up), 0.477 (C1, was 0.839 — **down sharply, now
  "ambiguous"**, below the 0.6 plan-doc threshold), 0.780 (C2, was 0.815 — down slightly). The
  new criterion increased method agreement in Cluster 0 but *reduced* it in Cluster 1 — a
  reminder that this criterion is a genuine trade-off, not a free improvement, consistent with
  §5 of `09_PHASE_7_AUDIT.md`'s conclusion that no single criterion change was going to cleanly
  fix all three clusters at once. All 3 clusters: Borda and Copeland still agree on #1.
- **Consensus Top-1 per cluster is now `RT57HC` in all three** (Tm=56.5°C) — was
  `n-Octacosane (C28)` / `PureTemp 60` / `PureTemp 60` before this criterion. Worth flagging
  explicitly: this reproduces the same "identical PCM across every cluster" pattern seen at the
  very start of this project's Phase 6 history (before the 2026-09-08 unification), though for a
  different reason this time — Tamil Nadu's three achievability ceilings (61.09–61.94°C) are
  close enough to each other that a margin-rewarding criterion converges on the same answer. Not
  necessarily wrong, but worth an independent sanity check rather than taking at face value.
- **Candidate pool**: 8/10/8 survivors (unchanged by the criterion addition — Phase 5 feasibility
  filtering runs before Phase 6 and doesn't see the new criterion).
- **Monte Carlo, at the final `N_DRAWS=5000`**: Top-3 inclusion for each cluster's consensus #1
  (`RT57HC`) is 86.5% (C0) / 75.2% (C1) / 84.2% (C2) — Top-1 retention 49.2% / 40.0% / 42.3%.
  Consistent with the earlier 1000-draw estimates (86.4/75.3/82.4% inclusion) to within ~2
  points, confirming those were already close to converged; the 5000-draw run is the more precise
  number to quote, not a materially different one.

## Status

**Unified with Rajasthan; investigation closed 2026-09-16.** Final 9-criterion engine run
complete against elevation-corrected data, the corrected `TM_TARGET_C=67`, and `N_DRAWS=5000`
(the plan doc's actual default — Rajasthan is still at 1000, not yet raised there). The
physics-vs-MCDM disagreement in Clusters 1/2 is a **closed, documented finding**, not a pending
fix — see `09_PHASE_7_AUDIT.md` §5 for why further criteria-tuning was deliberately not pursued.

## Literature Support
| Component | Reference | Source |
|---|---|---|
| TOPSIS | Hwang & Yoon (1981); Chen et al. (2025) SWH MCDM | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
| GRA | Deng (1982); Chen et al. (2025) | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
| PROMETHEE II | Brans & Mareschal (2005) | Standard MCDM literature |
| VIKOR | Opricovic & Tzeng (2004) | Standard MCDM literature |
| Monte Carlo uncertainty | Chopra et al. (2023) techno-economic MC | `sources/Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md` |
| Entropy+AHP weight blend | Framework doc Table 13 | `13_LITERATURE_MAPPING.md` |
| Entropy pathology on near-zero-ideal cost criteria | project Phase 7/8 diagnostic (Rajasthan `10_PHASE_8_AUDIT.md`) | — |
