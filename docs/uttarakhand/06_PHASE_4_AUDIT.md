# 06 — Phase 4 Audit: Climate Regime Clustering

**Scripts**: `05_cluster_uttarakhand.py` (single-state, **run**),
`05b_cluster_interactive.py` (explorer), `05_cluster_regions.py` (multi-state, **not run**)

**Status**: **COMPLETE at K = 4** (changed from K = 5 in 2026-10 — see "Choice of K" below).
Cluster assignments for all 45 points are recoverable from
`data/plots/uttarakhand_objective1/02_climate_regime_map_folium.html`.

---

## Purpose and why a single-state script exists

`05_cluster_uttarakhand.py`'s docstring:

> `05_cluster_regions.py` was written for the ORIGINAL v3.0 scope: combine signature matrices from
> FOUR states … and cluster across all of them together. … You are working on Uttarakhand only
> right now. That cross-state comparison isn't required for Objective 1 to stand on its own: the
> objective is "cluster meteorological data and identify Top-2/Top-3 PCM candidates per climatic
> regime" — nothing in the objective statement requires those regimes to span multiple states.

The docstring names the regimes it expects to find within Uttarakhand: "the high-altitude
Himalayan belt around Chamoli/Pithoragarh vs. the Doon Valley around Dehradun vs. the Terai plains
around Udham Singh Nagar/Haridwar … elevation alone spans roughly 200-2000m of populated terrain
here." These are the script author's expectations, stated in prose — the pipeline does **not**
assign district names to clusters, and no committed artefact labels a cluster geographically.

## Inputs

`data/processed/signatures/climate_signature_uttarakhand.csv` — `04b`'s output, one row per point.

## Processing

### Algorithm choice: Gaussian Mixture, diagonal covariance — RESOLVED (was full, before this session)

```python
GaussianMixture(n_components=k, covariance_type="diag", random_state=42, n_init=5)   # selection
GaussianMixture(n_components=k, covariance_type="diag", random_state=42, n_init=10)  # final fit
```

The justification (repeated in `05_cluster_regions.py` and `README_PREPROCESSING.md`) is that
climate is a continuous gradient:

> the boundary between "high-hill" and "valley/plains" Uttarakhand is not a hard line, and a point
> near that boundary genuinely has partial membership in both. Soft membership probabilities are
> kept and are what Phase 5/6 should read for boundary points.

**RESOLVED — `covariance_type` changed from `"full"` to `"diag"`, WITH a documented justification
now present in the script.** The code's own comment: a full covariance matrix needs
`D*(D+1)/2` parameters per component; with the number of standardized signature dimensions and only
45 points total, full covariance is severely overdetermined — "exactly what caused every point's
`max_membership_prob` to saturate at 1.000 in the original run (soft clustering silently degenerating
to hard clustering)," per the in-code comment. Diagonal covariance assumes feature independence
after PCA and needs only `D` parameters per component — "the standard fix for high-dimensional,
low-sample-count GMM, and Tamil Nadu's own 133-point run made the same correction for the same
reason." See "Soft membership" below for what this fixes in practice.

### Model-selection configuration

```python
K_CANDIDATES = list(range(2, 11))                       # K = 2 … 10
K_FINAL      = 4                                        # set manually after review (was 5 until 2026-10)
SILHOUETTE_ACCEPT_LO, SILHOUETTE_ACCEPT_HI = 0.15, 0.40
RANDOM_STATE = 42
```

The 0.15–0.40 band is explicitly wider than the 0.15–0.35 band used by the four-state script, with
the reason given inline: "no artificial between-state gaps inflating it here."

`README_PREPROCESSING.md` sets the expectation and the warning:

> Expected K for one state, and with only 45 points to work with: probably smaller than …
> realistically 2-4 (e.g. high-Himalaya vs. Doon Valley vs. Terai plains). With 45 points, be
> conservative about K: each additional cluster shrinks the average points-per-cluster fast, and a
> GMM fit on very few points per component gets unstable.

**The current run uses K = 4, the top of that recommended range** (the previous run used K = 5,
one above it). With 45 points that is an average of ~11 points per component; the smallest
component still has only 3.

### Feature matrix

`X = sig[[c for c in sig.columns if c.endswith("_z")]].fillna(median).values`

Only the `_z` (standardised) columns from `04b` are used. `lat`/`lon` are absent by construction —
`04b` dropped them from the clustering column list, and `05` re-prints the reason at run time:
"(lat/lon are NOT among these — never cluster on geography, plan v3.0 Section 6.2)."

There are **24** `_z` columns in the current `climate_signature_uttarakhand.csv` (git-ignored, read
locally). From `04b`'s `DROP_FROM_CLUSTERING` logic they comprise: the non-PCA canonical indices (`GHI_mean`, `kt_mean`, `kt_std`, `SAI`, `CCI`,
`cloudy_frac`, `DTR`, `GHI_daily_kWh`, `seasonality`, `HSI`, `wind_mean`, `monsoon_index`), the
constant `Tm_target_C`, `L_required_kJ_per_kg`, the 5 interaction terms, and `PC1…PCn`.

> **Note:** `Tm_target_C` is constant (57.0) across all 45 points, so `StandardScaler` emits a
> zero-variance column. It contributes nothing to the clustering but is not excluded.

### Model-selection outputs

Four metrics per K, written to `bic_selection_uttarakhand.csv`: `BIC`, `silhouette`,
`davies_bouldin`, `calinski_harabasz`, plus an `in_accept_band` boolean.

A K-Means comparison (`KMeans(n_clusters=k, random_state=42, n_init=10)`, silhouette only) is
written to `kmeans_comparison_uttarakhand.csv`, to answer "the 'why not K-Means' question with a
number instead of an assertion."

Both CSVs live under the git-ignored `data/processed/clustering/`; their current contents (read
locally, 2026-10) are reproduced in the next section. `data/plots/verify_clustering/01_elbow_curves.png`
now renders the same four curves in the same feature space (see "Choice of K").

### Choice of K — K = 4 (2026-10)

`bic_selection_uttarakhand.csv` (GMM, `diag`, `n_init=5`) and `kmeans_comparison_uttarakhand.csv`:

| K | BIC | Silhouette | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | K-Means silhouette |
|---|---|---|---|---|---|
| 2 | 308.5 | 0.262 | 1.064 | 12.3 | 0.254 |
| 3 | 31.6 | 0.301 | 1.284 | 16.0 | 0.340 |
| **4** | **−899.8** | **0.362** | **0.935** | **28.9** | 0.377 |
| 5 | −1047.7 | 0.279 | 1.351 | 24.8 | 0.380 |
| 6 | −2904.9 | 0.303 | 1.198 | 23.9 | 0.322 |
| 7 | −1341.7 | 0.331 | 1.021 | 23.4 | 0.342 |
| 8 | −3746.6 | 0.317 | 0.999 | 20.6 | 0.362 |
| 9 | −3689.3 | 0.366 | 0.949 | 23.3 | 0.375 |
| 10 | −4228.1 | 0.363 | 0.969 | 21.7 | 0.366 |

Every K is inside the 0.15–0.40 silhouette band, so the band does not discriminate. The decision
rests on the other evidence:

- **K = 4 has the best Davies-Bouldin and Calinski-Harabasz of K = 2…10** and a silhouette tied
  with the best (0.362 vs 0.366 at K = 9, which would leave ~5 points per component).
- **K = 5 (the previous choice) is worse than K = 4 on all three** internal metrics.
- **BIC is not used**: with diagonal covariance and only 45 points it keeps falling to K = 10 and
  never reaches a minimum — the usual over-fitting behaviour of BIC at small N.
- **K = 3 vs K = 4 is close and was checked explicitly.** With `n_init=10` (as in the final fit)
  K = 3 reaches silhouette 0.362 and DB 0.80, so on those two metrics it ties or beats K = 4.
  K = 4 was kept because (a) K = 3 is exactly K = 4 with Clusters 0 and 1 merged into one 33-point
  cluster (Ta_mean ~24.7 °C plains vs ~20.5 °C mid-hills, ~770 m mean elevation apart);
  (b) K = 4 is more stable across seeds — over 20 seeds silhouette 0.364 ± 0.004 vs 0.353 ± 0.020
  for K = 3, and DB 0.935 ± 0.001 vs 0.877 ± 0.193; (c) on 100 refits of random 80 % subsamples the
  adjusted Rand index against the full-data labels is mean 0.91 / 10th percentile 0.69 for K = 4,
  vs 0.85 / 0.35 for K = 3 and 0.66 / 0.54 for K = 5.

So K = 4 is the most internally consistent and the most stable choice for this data. That
Uttarakhand gets one more regime than the other states (K = 3) is consistent with its far larger
climatic range (~25 °C plains to ~14 °C high Himalaya, ~320 m to ~2,200 m mean elevation); K is
chosen per state from its own data, not fixed across states.

### Final fit and outputs

```python
k_final_safe = min(K_FINAL, len(X) - 1)      # = 4
gmm_final    = GaussianMixture(4, covariance_type="diag", random_state=42, n_init=10)
hard_labels  = gmm_final.fit_predict(X)
soft_probs   = gmm_final.predict_proba(X)
```

| Output file | Contents |
|---|---|
| `bic_selection_uttarakhand.csv` | K = 2…10 × {BIC, silhouette, DB, CH, in_accept_band} |
| `kmeans_comparison_uttarakhand.csv` | K = 2…10 × K-Means silhouette |
| `cluster_assignments_uttarakhand.csv` | `point_id, lat, lon, population, cluster_id, max_membership_prob, prob_cluster0…3` |
| `cluster_profiles_uttarakhand.csv` | one row per cluster: `cluster_id, n_points, total_population_covered`, plus the **population-weighted mean** of every non-`_z` numeric signature column |
| `cluster_map_uttarakhand.png` | lon/lat scatter coloured by `cluster_id`, annotated `C0…C3` |

Population weighting uses `np.average(g[col], weights=g["population"])`, falling back to an
unweighted mean if the weight sum is zero.

`cluster_profiles_uttarakhand.csv` is what `07_feasibility_filter.py` and
`09_recommendation_cards.py` read. Because `profile_cols` is "everything not
`point_id`/`cluster_id` and not ending `_z`", it carries `Tm_target_C` and `L_required_kJ_per_kg`
through — which is exactly what `07` checks for and errors on if absent.

---

## Observed results (current run, K = 4)

Read directly from the current `cluster_assignments_uttarakhand.csv` and
`cluster_profiles_uttarakhand.csv` (2026-10). This replaces the earlier K = 5 tables (sizes
7/3/9/10/16, and before that 12/9/3/7/14), which are superseded.

### Cluster assignments (all 45 points)

| Cluster | n_points | Member `point_id`s (UKP_…) |
|---|---|---|
| **0** | **10** | 0001, 0003, 0009, 0010, 0012, 0013, 0014, 0016, 0017, 0034 |
| **1** | **23** | 0004, 0005, 0006, 0007, 0015, 0018, 0019, 0020, 0022, 0027, 0028, 0029, 0030, 0031, 0032, 0035, 0037, 0038, 0039, 0042, 0043, 0044, 0045 |
| **2** | **9** | 0002, 0008, 0011, 0021, 0024, 0025, 0026, 0033, 0036 |
| **3** | **3** | 0023, 0040, 0041 |

Clusters 2 and 3 have exactly the same members as the corresponding clusters of the earlier runs —
they are robust to the choice of K. Moving from K = 5 only re-partitions the remaining 33 points
into a warm-plains cluster (0) and a mid-hill cluster (1).
`data/plots/verify_clustering/06_cluster_sizes.png` shows 10 / 23 / 9 / 3 (max/min ratio 7.67).

### Population and geographic extent per cluster

| Cluster | n | Population covered | Share | Latitude range (mean) | Longitude range (mean) | Mean elevation |
|---|---|---|---|---|---|---|
| 0 | 10 | **3,700,876** | 35.3 % | 28.875 – 29.875 (29.400) | 77.875 – 79.875 (78.700) | ~323 m |
| 1 | 23 | **3,993,013** | 38.1 % | 29.125 – 30.625 (29.625) | 77.875 – 80.125 (79.375) | ~1,090 m |
| 2 | 9 | **2,451,044** | 23.4 % | 30.125 – 30.625 (30.292) | 78.125 – 78.875 (78.486) | ~1,299 m |
| 3 | **3** | **330,780** | 3.2 % | 30.125 – 30.375 (30.292) | 79.125 – 79.375 (79.292) | ~2,219 m |
| **Total** | **45** | **10,475,713** | 100 % | | | |

(Mean elevation is the population-weighted `elevation_m` from the cluster profiles.)
Cluster 3 is the smallest by both point count (3) and population (3.2 %), and is the most spatially
compact — a 0.25° × 0.25° neighbourhood around 30.25° N, 79.25° E.

### Climate profile per cluster (population-weighted means)

From `cluster_profiles_uttarakhand.csv` (exact values, not read off a chart):

| Index (Tier-1 proxy) | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| `Ta_mean_proxy` (°C) | **24.7** | 20.5 | 19.2 | **13.8** |
| `Ta_p95_proxy` (°C) | 32.4 | 27.3 | 25.8 | 20.7 |
| `Ta_p05_proxy` (°C) | 13.4 | 10.6 | 9.3 | 4.4 |
| `DTR_proxy` (K) | 8.0 | 7.5 | 7.8 | 7.2 |
| `GHI_mean` (W/m², noon) | 724.8 | 702.6 | 705.5 | 683.2 |
| `GHI_daily_kWh_proxy` (kWh/m²/day) | 5.65 | 5.47 | 5.49 | 5.30 |
| `L_required_kJ_per_kg` | 118 | 132 | 138 | 178 |
| `HSI` | 19.0 | 21.7 | 19.2 | 15.0 |

The temperature ordering is monotone and coherent: **C0 (warmest) > C1 > C2 > C3 (coldest)**,
spanning ~11 K of mean-temperature separation, reproduced in `Ta_p95` and `Ta_p05` and mirrored by
a monotone rise in elevation (~323 → 1,090 → 1,299 → 2,219 m). Combined with the geographic
extents above — C0 southernmost, C3 a compact high-elevation pocket — the partition is internally
consistent with an elevation/latitude gradient, even though lat/lon/elevation are not clustering
inputs.

> **The source files do not assign geographic names to the clusters.** No committed artefact in
> `era5-uttarakhand/` labels a cluster as "Terai", "Doon Valley" or "high Himalaya". Descriptions
> such as "warm plains" (C0) or "high Himalaya" (C3) in a write-up are interpretation added on top
> of the pipeline, based on the elevation and temperature columns above.

> **`GHI_mean` enters the clustering matrix carrying the ERA5 GHI anomaly** documented in
> `04_PHASE_2_AUDIT.md` Part A.3. The *relative* ordering across clusters is informative; treat
> absolute magnitudes with that caveat.

### Soft membership — effectively hard at K = 4

Under the old `covariance_type="full"` fit every point's `max_membership_prob` was 1.000. After
switching to `"diag"`, the K = 5 run showed a range of 0.9978–1.0000 (2 points below 1.000).
**At K = 4, every one of the 45 points has `max_membership_prob` = 1.000** — the four regimes are
well separated relative to only 45 points, so the partition is effectively hard. The
soft-clustering rationale in the docstring ("a point near that boundary genuinely has partial
membership in both") does not materialise for this run: `prob_cluster0…3` carries no usable
boundary information, and `05b_cluster_interactive.py`'s boundary-ring feature
(`max prob < 1.5/K`) highlights no points.

### Silhouette

`data/plots/verify_clustering/02_silhouette_plot.png` for the saved K = 4 labels. Since 2026-10
`verify_02_clustering.py` uses the same `_z`-only feature matrix and `diag` covariance as
`05_cluster_uttarakhand.py`, so its numbers now match `bic_selection_uttarakhand.csv`. (Before
that it re-standardised every numeric column — raw, `_proxy`, `_true` and `_z` duplicates — and
used `full` covariance, which produced curves that appeared to favour K = 5.)

| Metric | Value |
|---|---|
| Average silhouette (saved labels) | **0.362** |
| Davies-Bouldin / Calinski-Harabasz | 0.935 / 28.9 |
| Reference threshold drawn on the plot | 0.400 |
| Per-cluster avg/min silhouette | C0: avg 0.33, min 0.12; C1: avg 0.27, min −0.01; C2: avg 0.55, min 0.46; C3: avg 0.65, min 0.50 |

0.362 falls inside `05_cluster_uttarakhand.py`'s accept band of **0.15–0.40** — the script's own
guidance is that a HIGHER silhouette at only 45 points would suggest an over-simplified
signature, not a better result. Only Cluster 1 (the large mid-hill cluster) has a point with a
(marginally) negative silhouette; Clusters 2 and 3 are cleanly separated.

---

## What is absent from Phase 4

| Component | Status |
|---|---|
| Bootstrap / ARI cluster-stability analysis | **Not in the pipeline.** No resampling appears in `05_cluster_uttarakhand.py`; a one-off subsample check (100 × 80 %) was run by hand for the K choice — ARI mean 0.91 at K = 4 — but it is not a committed script. |
| Fitted-model persistence (`joblib` scaler + GMM) | **Not implemented.** Neither `04b`'s `StandardScaler` nor `05`'s fitted `GaussianMixture` is saved; re-running Phases 5–8 requires re-fitting. |
| `sklearn_version` recorded in outputs | **Not implemented.** |
| Canonical cluster relabelling (e.g. by ascending latitude) | **Not implemented.** Cluster IDs come straight from `GaussianMixture.fit_predict` and are stable only because `random_state=42` is fixed. |
| External climate classification (Köppen-Geiger, NBC/ECBC) | **Not implemented.** The K = 4 partition rests entirely on internal statistics. |
| Automatic K selection | **Not implemented by design** — `K_FINAL` is a manually edited constant, and the script prints "update after reviewing this table, then re-run." |

---

## `05_cluster_regions.py` — multi-state, not run

Present but inert. `REGION_FILES` maps `"Uttarakhand"` to this pipeline's own signature file and
`"Rajasthan"` to `../era5-rajasthan/data/processed/signatures/climate_signature_rajasthan.csv`.
`main()` returns early with "Fewer than 2 regions available yet" unless at least two files load.

Its own settings differ from the single-state script: `K_CANDIDATES = range(3, 13)`,
`K_FINAL = 6`, silhouette band 0.15–0.35, and it **re-standardises across the combined matrix**
before fitting. Its output filenames (`point_fingerprints.csv`, `bic_selection.csv`,
`cluster_assignments.csv`, `cluster_profiles.csv`) are un-suffixed, and its `cluster_profiles.csv`
is **not** the file `07`/`09` read.

The docstring cites plan **v2.0 §7** while every other Phase 2–8 script cites v3.0 — a visible
version lag in an unrun file.

---

## `05b_cluster_interactive.py` — explorer

Reads `cluster_assignments_uttarakhand.csv`, `cluster_profiles_uttarakhand.csv` and
`bic_selection_uttarakhand.csv`; writes Folium/Plotly HTML to
`data/processed/clustering/interactive/`. Features per the docstring: a cluster map whose popups
show the full soft-membership probability vector with boundary points (max membership below
`1.5/K`) drawn with a faint ring, a grouped-bar comparison of population-weighted profiles, a
population-share pie per regime, and BIC/silhouette K-selection curves.

**Its output directory is under the git-ignored `data/processed/` tree, so none of it is present in
this repository.**

---

## Literature support

**None present in the source files.** `05_cluster_uttarakhand.py` cites plan v3.0 §6.2;
`05_cluster_regions.py` cites plan v2.0 §7 for the GMM-over-K-Means rationale and the silhouette
band. No external reference for Gaussian Mixture models, BIC model selection, silhouette,
Davies-Bouldin or Calinski-Harabasz appears anywhere in `era5-uttarakhand/`. See
`13_LITERATURE_MAPPING.md`.

## Validation

| Check | Result |
|---|---|
| lat/lon excluded from the clustering matrix | **Confirmed** — dropped by `04b`, re-announced by `05` at run time |
| K selected from a four-metric table | **Implemented** — K = 4 best on DB and CH, silhouette tied-best (table above) |
| K-Means reported as a comparison | **Implemented** — K-Means silhouette 0.377 at K = 4 |
| Silhouette inside the stated accept band | **PASS** — 0.362 in [0.15, 0.40] |
| Clusters spatially coherent | **PASS** — geographically contiguous despite geography being excluded |
| Cluster profiles population-weighted | **Confirmed** — `np.average(..., weights=population)` |
| Bootstrap stability | **Checked once by hand** (ARI mean 0.91, 10th pct 0.69); not part of the pipeline |
| External classification agreement | **Absent** |

## Problems / risks

1. **K = 4 is at the top of the source files' own recommendation.** `README_PREPROCESSING.md`
   says "realistically 2-4" for a 45-point single-state fit. K = 4 is inside that range (the
   previous K = 5 was not), but Cluster 3 still has only 3 points.
2. **Soft membership is 1.000 for every point**, so the stated methodological reason for choosing
   GMM over K-Means is not realised in this run. This should be reported, not left implicit.
3. **Stability evidence is not part of the pipeline.** The K = 4 choice is backed by the four-metric
   table, a 20-seed spread check and a one-off subsample ARI check (all reported above), but none
   of these is a committed, re-runnable script and there is no external classification.
4. **Cluster ID stability depends solely on `random_state=42`.** There is no canonical relabelling
   step, so any change to the signature matrix, sklearn version, or seed can permute cluster IDs
   and silently invalidate the `cluster_id`-keyed joins in `07`, `08` and `09` — none of which
   verify provenance.
5. **Cluster 3 is a 3-point regime carrying 3.2 % of population.** Every per-cluster statistic for
   it — profile means, survivor counts, MCDM ranks — rests on three sampling points.
6. **`Tm_target_C` enters the clustering matrix as a zero-variance column.** Harmless but untidy.
7. **`GHI_mean` enters the clustering matrix carrying the ERA5 GHI anomaly** — the one solar column
   the Tier-2 repair does not cover.

## Status

**COMPLETE.** The clustering is methodologically well argued (soft clustering for a continuous
gradient, geography excluded, four selection metrics plus a K-Means control, population-weighted
profiles) and the result is spatially coherent with a monotone temperature ordering — a genuine
positive finding given that latitude and longitude were excluded from the fit. The open items are
the 3-point Cluster 3, stability evidence that is not yet a committed script, and the unrealised
soft membership.
