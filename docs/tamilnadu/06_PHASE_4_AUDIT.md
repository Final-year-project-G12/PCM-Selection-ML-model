# 06 — Phase 4 Audit: Climate Regime Clustering

Scripts: `cluster_lib.py`, `05_cluster_tamilnadu.py`, `05a_level_b_regime_shift_tamilnadu.py`, `05b_cluster_interactive.py`, `11_seasonal_pcm_sensitivity.py`.

**Unified with Rajasthan (2026-09-08).** Tamil Nadu's Phase 4 is now a state-parameterised mirror of Rajasthan's. Key changes from the previous Tamil Nadu version:

- **K-selection**: the hardcoded `K_FINAL = 5` hand-pick is **replaced** by Rajasthan's documented 3-tier `suggest_k()` cascade (now in shared `cluster_lib.py`): (1) k in [2,4] AND silhouette in [0.15, 0.35] → highest bootstrap-ARI among those; (2) any k in the silhouette band → highest bootstrap-ARI; (3) fallback to lowest-BIC k with a printed warning. k-scan range widened 2–10 → **2–12**.
- **Bootstrap-ARI**: `bootstrap_ari_stability()` (50 resamples, GMM fit on the resample, predict on the original data, ARI vs `base_labels`) now runs at **every** scanned k and is the actual tiebreaker.
- **External validation**: the real Köppen-Geiger per-point lookup (Beck et al. 2018, 1-km raster) — copied into `data/raw/koppen/` — with ARI/NMI vs the GMM clusters and a `koppen_validation_tamilnadu.csv` contingency table.
- **Canonical relabeling**: after the final GMM fit, hard cluster IDs are relabeled 0..k-1 by ascending mean latitude **before any output is written**, so "cluster 0" means the same regime across re-runs (`cluster_lib.canonical_relabel_by_latitude`).
- **Provenance hard-fail**: `provenance_lib.py` fingerprint checks are wired into the Phase 5→6→7→8 handoffs (07/08/10/09), identical to Rajasthan — a downstream phase `SystemExit`s if `cluster_profiles_tamilnadu.csv` was regenerated underneath it.
- **The "Level B" name collision is resolved.** The two analyses that were both called "Level B" now have distinct names and files (see below).

<<<<<<< HEAD
## Level B: Seasonal Sensitivity (v3.1 corrected)
- Recomputes `L_required_season` per season using 300 L/day draw (matching `04b`).
- Single-method TOPSIS re-rank per (cluster, season); reports #1 PCM flips.
- NE monsoon out-of-phase cycle provides physical basis for seasonal variation.
=======
## Level A — spatial clustering (`05_cluster_tamilnadu.py`)
>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637

- GMM `covariance_type="diag"` (kept — same reasoning as Rajasthan's 2026-08-10 fix, now documented once in `cluster_lib.py`), `n_init` 5 (scan) / 10 (final), `random_state=42`. K-Means fit in parallel as a **reported comparison baseline only**.
- **Completed-run result (2026-09-08 re-run, k=3):**

| Cluster | Points | Population | GHI_daily_kWh | Ta_mean | bootstrap-ARI (k=3) |
|---|---|---|---|---|---|
| 0 | 46 | 20,468,850 | 5.32 | 29.7 °C | **0.6277** (50/50 resamples) |
| 1 | 41 | 20,422,159 | 5.24 | 27.1 °C | |
| 2 | 46 | 30,335,761 | 5.23 | 29.7 °C | |

  k=3 was selected by the tier-1 rule (k ∈ {2,3,4} all clear the silhouette band + expected single-state range; k=3 has the highest bootstrap-ARI among them). **Not** a hand-pick.
- **Köppen-Geiger external validation**: the 133 points classify as Aw=98, BSh=30, plus 5 coastal points sampling an ocean/no-data pixel (`UNKNOWN_0`). ARI(GMM cluster, Köppen class) = **0.0672**, NMI = **0.1499** — low agreement, read the same way as Rajasthan's: the GMM finds structure at a finer resolution than Köppen's two broad classes capture within one state (a legitimate finding, not a failure). NBC/ECBC still stubbed; state-identity check not meaningful for a single state.
- **QC plots** (inline, no separate script): `outputs/qc_cluster_map_tamilnadu.html` (colour by cluster, opacity by membership confidence, **marker size by population** — new), `qc_k_selection_curve_tamilnadu.html` (BIC + silhouette + **bootstrap-ARI** overlaid, chosen k marked — new), `qc_cluster_profile_bars_tamilnadu.html`, `qc_cluster_population_share_tamilnadu.html`, plus `data/processed/clustering/cluster_map_tamilnadu.png` kept for continuity.

## Level B — regime shift (`05a_level_b_regime_shift_tamilnadu.py`)

A genuine Phase-4 **clustering** step, ported unchanged from Rajasthan's `05a`. Rebuilds a per-point-per-season Tier-1 signature via `signature_lib.build_tier1_signature(group_keys=["point_id", "season"])` (same formulas as Level A, different grouping — no Tier 2, PCA or interaction terms), fits a fresh GMM (k-scan 2–8, same `suggest_k` cascade), and reports:

- **Regime-shift fraction**: **120 / 133 points (90.2%)** have a different seasonal cluster label in at least one season.
- **Season-tautology check**: ARI(cluster, season) = **0.501**, NMI = 0.602 — *moderate* agreement. Some season-tracking, but with real cross-season structure (unlike Rajasthan, whose k=8 Level B recovers "season" almost outright at ARI 0.691). ANOVA feature importance: all top-5 drivers are temperature/GHI (season-defining) variables.
- k chosen = **4**, bootstrap-ARI 0.8157 (50/50).
- **New plot**: `outputs/qc_level_b_regime_shift_sankey_tamilnadu.html` — an alluvial of every point's cluster membership across the 4 seasons, shifting flows drawn orange against grey non-shifting flows.
- Outputs: `data/processed/clustering/{bic_selection_tamilnadu_levelB, cluster_assignments_tamilnadu_levelB, level_b_feature_importance_tamilnadu, level_b_season_tautology_tamilnadu, level_b_season_contingency_tamilnadu}.csv`.

## Seasonal PCM sensitivity (`11_seasonal_pcm_sensitivity.py`)

**Renamed** from `11_level_b_seasonal_analysis.py` — it is a *post-Phase-6* re-ranking, not a clustering step. Holds the Level-A clusters and each cluster's annual MCDM weights fixed, recomputes only `L_required` per season (300 L/day draw, shared `SHARE_PCM`), and re-ranks with TOPSIS.

- **Completed-run result (2026-09-08):** **3 / 12** (cluster, season) combinations flip the #1 PCM:
  - Cluster 0, **Summer**: `PureTemp 58` replaces `n-Octacosane (C28)`.
  - Cluster 2, **Summer** and **Monsoon**: `savE® OM55` replaces `n-Octacosane (C28)`.
  - Cluster 1 is stable across all four seasons.
- **New plot**: `outputs/qc_seasonal_pcm_flip_heatmap_tamilnadu.html` — a (cluster × season) grid coloured by #1-PCM identity, with flipped cells red-outlined against the annual baseline.
- Execution order: runs **after Phase 6** (reads `08`'s `mcdm_full_rankings.csv` — renamed 2026-09-08 from `mcdm_full_scores_by_cluster.csv` — and `06`'s PCM database); `run_all_tamilnadu.py` sequences it last, non-blocking. `05a` (Level B regime shift) runs in Phase-4 order, also non-blocking.

## Status
<<<<<<< HEAD
**COMPLETE (v3.1 fixes applied — re-run `05` and `11` after Phase 3 re-run)**
=======

**COMPLETE (unified pipeline, re-run 2026-09-08).** Level A k=3; Level B regime-shift k=4 (90.2% shift, season-tautology ARI 0.501); seasonal PCM sensitivity 3/12 flips. Downstream Phase 5→6 re-run on the new `cluster_profiles_tamilnadu.csv` (fingerprint stamped through). Phase 7 (`10_physics_validation.py`) and Phase 8 (`09_recommendation_cards.py`) not re-run in this pass — they will hard-fail the provenance check until re-run against the current clustering.
>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637

## Literature Support

| Component | Reference | Source |
|---|---|---|
| GMM climate regime discovery | Liu et al. (2025) — AI PCM TES | `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
| Population-weighted clustering | Novelty N1 (framework doc) | `01_PROJECT_CONTEXT.md` |
| Silhouette expectation band [0.15, 0.35] | Building & Environment (2024) India climate-classification; 2026 thermal-comfort clustering | code comments in `cluster_lib.py` |
| Köppen-Geiger external validation | Beck et al. (2018), doi:10.1038/sdata.2018.214 | `data/raw/koppen/` |
| Seasonal PCM sensitivity | Singh et al. (2025) — monsoon SWH | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Diagonal GMM regularization | Standard small-n practice | `cluster_lib.py` docstring, `METHODS.md` §05 |
