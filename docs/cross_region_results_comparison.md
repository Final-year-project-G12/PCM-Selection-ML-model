# Cross-Region Results Comparison: Rajasthan, Tamil Nadu, Uttarakhand, Assam

Objective 1 results for the four states, side by side. Every number below is
read from the state's pipeline outputs dated 2026-10-25 (the latest full run);
the source file is given under each table so any value can be traced. Values
are rounded for reading; use the CSVs for anything that goes in the paper.

Updated 2026-10-28. The previous version of this file predated the
`T_DELIVERY_C`, `M_W_KG` and `SHARE_PCM` corrections and the Assam K=3 final
chain, and did not cover Uttarakhand.

## 1. Sampling and climate regimes

| | Rajasthan | Tamil Nadu | Uttarakhand | Assam |
|---|---|---|---|---|
| Population-weighted points | 320 | 133 | 45 | 80 |
| Population covered (sum over clusters) | 70.3 M | 71.2 M | 10.5 M | 30.8 M |
| Regimes (K) | 3 | 3 | 5 | 3 |
| Points per regime | 109 / 83 / 128 | 46 / 41 / 46 | 7 / 3 / 9 / 10 / 16 | 27 / 28 / 25 |
| Bootstrap ARI, chosen K | 0.81 | 0.66 | not in profile file | 0.69 (mean of 500 resamples) |
| Daily GHI by regime (kWh/m²/day) | 5.03–5.38 | 5.23–5.32 | 4.57–4.90 | 3.86–4.04 (estimate) |

Sources: `cluster_profiles_<state>.csv` in each state's `data/processed/`
(Tamil Nadu, Uttarakhand, Assam under `clustering/`); Assam ARI from
`data/preprocessed/clustering_report.txt`.

Uttarakhand's cluster 1 holds 3 points (mean elevation about 2,200 m), so its
results rest on very few locations.

## 2. Storage design targets

| | Rajasthan | Tamil Nadu | Uttarakhand | Assam |
|---|---|---|---|---|
| `Tm_target` (uncapped) | 67 °C | 67 °C | 57 °C | 56 °C |
| `Tm_target` used by the filter, per regime | 55.5 / 61.1 / 59.2 °C (capped) | 61.9 / 61.1 / 61.4 °C (capped) | 57.0 / 55.2 / 56.5 / 57.0 / 57.0 °C | 50.3 / 49.4 / 49.4 °C (capped) |
| `L_required` (kJ/kg) | 427–443 | 406–438 | 118–178 | 126–131 |

Sources: the cluster-profile files above; Uttarakhand per-regime targets from
`data/processed/pcm/feasibility_survivors_by_cluster.csv`.

**The states do not share design constants.** Rajasthan and Tamil Nadu take
`T_DELIVERY_C`, `SHARE_PCM` and the other sizing constants from the root
`pcm_shared_config.py`; Uttarakhand and Assam use their own `config.py`. That
is why `L_required` is about three times higher in the first pair. Before
comparing feasibility or rankings across all four states in the paper, either
bring Uttarakhand and Assam onto `pcm_shared_config.py` or state the
difference explicitly.

## 3. Feasibility filter

| | Rajasthan | Tamil Nadu | Uttarakhand | Assam (historical chain) | Assam (final K=3 chain) |
|---|---|---|---|---|---|
| Candidates per regime | 62 | 62 | 55 | 62 | 58 |
| Survivors, strict filter | 0 / 0 / 0 | 0 / 0 / 0 | 29 / 30 / 29 / 27 / 29 | 35 / 33 / 33 | 0 confirmed (33 / 32 / 32 conditional) |
| Survivors after κ calibration | 4 / 8 / 11 | 8 / 10 / 8 | n/a | n/a | n/a |

Sources: `feasibility_survivors_by_cluster.csv` and
`feasibility_survivors_by_cluster_kappa_calibrated.csv` (Rajasthan, Tamil
Nadu), `pcm/feasibility_survivors_by_cluster.csv` (Uttarakhand),
`pcm/feasibility_survivors_assam.csv` and
`pcm/mcdm_cluster_eligibility_summary.csv` (Assam).

With the higher `L_required`, no PCM passes the strict latent-heat check in
Rajasthan or Tamil Nadu; the κ-calibrated pool is what MCDM ranks there.
Rajasthan cluster 0 has only 4 survivors and is flagged `undersized`.

## 4. MCDM ranking and stability

| | Rajasthan | Tamil Nadu | Uttarakhand | Assam (historical chain) |
|---|---|---|---|---|
| Top-1 PCM per regime | savE OM55 / PureTemp 60 / Palmitic-stearic acid + EG | RT57HC in all three | PureTemp 58 (regimes 0, 2, 3, 4); PureTemp 53 (regime 1) | savE OM49 in all three |
| Kendall's W across methods | 1.00 / 0.77 / 0.42 | 0.85 / 0.48 / 0.78 | 0.80 / 0.84 / 0.78 / 0.71 / 0.80 | 0.94 / 0.94 / 0.94 |
| Monte Carlo draws per regime | 5,000 | 5,000 | 5,000 | 5,000 (historical chain) |
| Monte Carlo top-1 retention of the #1 PCM | 90.6% / 57.9% / 27.0% | 49.2% / 40.0% / 42.3% | 4.2% / 5.3% / 3.6% / 5.2% / 4.4% | top-3 inclusion of the #1 PCM: 86% / 89% / 90% |

Sources: `mcdm_topk_by_cluster.csv` and `monte_carlo_stability.csv`
(Rajasthan, Tamil Nadu), `pcm/mcdm_topk_by_cluster.csv` and
`pcm/monte_carlo_stability.csv` (Uttarakhand, run 2026-10-28),
`pcm/mcdm_topk_assam.csv` (Assam). Draw counts: `N_DRAWS` in each state's
`08_mcdm_ranking.py`; Uttarakhand `09b_monte_carlo_stability.py --draws 5000`.

- Rajasthan cluster 0's W = 1.00 and 90% retention come from a 4-candidate
  pool, so they show little.
- **Uttarakhand's top-1 picks are not robust.** Under weight and property
  perturbation, the baseline #1 (PureTemp 58; PureTemp 53 in regime 1) stays
  first in only 4–5% of draws and reaches the top 3 in 12–16%. Its mean rank
  across draws is about 10–12 out of 27–30 candidates. n-Octacosane (C28) is
  the most stable candidate in every regime (top-3 inclusion 38–39%), but no
  PCM is a clear winner. This is not a script mismatch: re-ranking with
  `09b`'s functions at the nominal weights and no noise reproduces `08`'s
  top 3 exactly in all five regimes. The large near-equivalent candidate
  pool is the likely cause.
- **Assam final chain:** no ranking was performed, because no regime has two
  or more confirmed-feasible candidates, and Monte Carlo was skipped
  (`n_draws = 0`). The Assam columns above come from the historical chain.
  Its validation report describes that chain as run under an earlier K=4
  partition, but `mcdm_topk_assam.csv` now carries cluster ids 0–2.
  Reconcile this before citing these Assam figures.

## 5. Physics check: MCDM rank vs simulated performance

Spearman ρ between the MCDM consensus rank and the grey-box simulation rank,
per regime:

| | Rajasthan | Tamil Nadu | Uttarakhand |
|---|---|---|---|
| ρ per regime | −0.20 / −0.17 / +0.57 | −0.33 / +0.10 / +0.02 | +0.02 / −0.17 / −0.34 / +0.17 / −0.14 |
| Candidates per regime | 4 / 8 / 11 | 8 / 10 / 8 | 20 each |
| Significant at p < 0.05 | none against Borda (cluster 2 against Copeland: ρ = 0.63, p = 0.036) | none | none |

**Assam** compares the historical ranking with physics across 8 common PCMs,
pooled over regimes:

| Physics metric | ρ | p |
|---|---|---|
| Delivery rate | −0.30 | 0.083 |
| Solar fraction | +0.62 | 0.0001 |
| Cycling | −0.39 | 0.021 |

The report's verdict for Assam is "partially supported / inconclusive".

Sources: `spearman_rho_by_cluster_rajasthan.csv`,
`pcm/physics_validation_spearman.csv` (Tamil Nadu, Uttarakhand), and Assam
`data/preprocessed/validation_comparison_report.txt` §7 and §11.

**Overall:** in no state does the MCDM ranking reliably predict simulated
performance. Only one test reaches p < 0.05 on a per-regime ranking:
Rajasthan cluster 2, against the Copeland ranking. Assam's pooled comparison
also has significant correlations, but they point in opposite directions:
positive for solar fraction, negative for cycling.

This agrees with the project's existing framing: MCDM picks a defensible
shortlist, and the physics model (Objective 2) decides between candidates.

## 6. Known inconsistencies to resolve

- **Design constants differ** between the Rajasthan/Tamil Nadu pair and
  Uttarakhand/Assam (§2).
- **Assam's historical-chain cluster ids** need reconciling before its MCDM
  numbers are cited (§4).

Resolved on 2026-10-28:
- Rajasthan's Monte Carlo now uses 5,000 draws like the other states. Moving
  from 1,000 changed top-3 inclusion by at most 2.3 points and top-1
  retention by at most 1.6; the rankings themselves are unchanged.
- The Phase 10 ρ and verdict in `era5-assam/run_all_assam.py`'s docstring now
  match the validation report.
- `05_cluster_assam.py` no longer hard-codes "129 points" or numbers from an
  earlier run. The bootstrap always resampled the 80 points actually loaded,
  so the 0.69 ARI stands.
- Uttarakhand's Monte Carlo output now exists (§4).
