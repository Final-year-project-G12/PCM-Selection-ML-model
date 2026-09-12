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
> The pipeline has been run once against the unified engine (2026-09-08); a small residual bug
> in the supercooling entropy cap was fixed after that run, so the numbers below are indicative
> and a fresh run is still pending.

## Criteria (8, exact — Table 13) and weights

| Criterion | Direction | AHP prior (Table 13) | Notes |
|---|---|---|---|
| `Tm_fitness` | benefit | 0.24 | Gaussian target fitness `exp(−(Tm−Tm_target)²/(2σ²))`, σ=4K |
| `latent_heat` | benefit | 0.20 | **climate-relative**: `latent_heat_kJ_kg / L_required` for that cluster |
| `vol_latent_heat` (ρL) | benefit | 0.12 | separate criterion, kept |
| `thermal_conductivity` | benefit | 0.13 | |
| `cycling` | benefit | 0.11 | **log-scaled** `cycles_confidence = log1p(cycles)/log1p(max_cycles)`, NaN-safe |
| `supercooling` | cost | 0.08 | **entropy weight capped at 2× prior (0.16)** — see below |
| `corrosion` | cost | 0.06 (cluster-rescaled 1×–2× by HSI) | structural proxy `2.0 if Inorganic else 1.0` — inert (0 salt hydrates) |
| `cost` | cost | 0.06 | always NaN → entropy weight 0 via the `<2 real values` guard |

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
- **Monte Carlo**: `N_DRAWS = 1000` (both states; raise both to 5000 for the final reported run),
  Dirichlet weight draws + Gaussian property perturbation + family-distribution sampling for
  imputed properties. Reports Top-3 inclusion %, Top-1 retention %, rank-reversal freq, mean ρ vs baseline.

## Outputs

`mcdm_full_rankings.csv` (full per-survivor audit trail — raw `crit_*` values, blended `weight_*`,
all 4 method ranks + raw scores, Borda/consensus/Copeland, Kendall's W, entropy-dominant flag,
λ-ablation flags, MC columns, plus legacy aliases so Phase 7/8 read it unchanged),
`mcdm_topk_by_cluster.csv` (rows with `consensus_rank ≤ 3`), `monte_carlo_stability.csv`,
`mcdm_method_agreement.csv`, `outputs/qc_montecarlo_inclusion.html`. All carry
`upstream_cluster_profile_fingerprint`.

## Results — 2026-09-08 run (3 clusters, 13/13/16 survivors, n=42; INDICATIVE — re-run pending)

- **`Tm_fitness` is the dominant entropy criterion** (weight ≈ 0.54 / 0.71 / 0.70), all flagged by
  the >40%-domination check. Supercooling blends to ≈ 0.16–0.20 (the run used a build with a residual
  cap bug — the fixed cap holds it at 0.16).
- **Kendall's W** ≈ 0.55 / 0.57 / 0.60 — no cluster "strong"; clusters 0–1 "ambiguous", cluster 2 "moderate".
- **Structural outlier**: PROMETHEE II in Cluster 0 (consistent with its native-Tm handling);
  **GRA in Clusters 1 and 2**.
- **Borda ≠ Copeland Top-3** flagged in every cluster.
- Consensus Top-1: `Myristic acid` (C0) / `n-Tetracosane (C24)` (C1) / `Palmitic-Stearic eutectic`
  (C2). Several deterministic Top-3 picks have MC Top-3 inclusion < 50% (flagged).

## Status

**Unified with Rajasthan; fresh run pending** (the 2026-09-08 run predates the final supercooling
cap fix). Re-run `08_mcdm_ranking.py` → `10_physics_validation.py` → `09_recommendation_cards.py`.
Also raise `N_DRAWS` to 5000 for the reported numbers.

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
