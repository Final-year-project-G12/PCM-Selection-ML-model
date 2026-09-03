# 06 — Phase 4 Audit: Climate Regime Clustering

**Scripts**: `05_cluster_uttarakhand.py` (single-state, **run**),
`05b_cluster_interactive.py` (explorer), `05_cluster_regions.py` (multi-state, **not run**)

**Status**: **COMPLETE at K = 5.** Cluster assignments for all 45 points are recoverable from
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

### Algorithm choice: Gaussian Mixture, full covariance

```python
GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)   # selection
GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=10)  # final fit
```

The justification (repeated in `05_cluster_regions.py` and `README_PREPROCESSING.md`) is that
climate is a continuous gradient:

> the boundary between "high-hill" and "valley/plains" Uttarakhand is not a hard line, and a point
> near that boundary genuinely has partial membership in both. Soft membership probabilities are
> kept and are what Phase 5/6 should read for boundary points.

`covariance_type="full"` is used without a separate justification in the Uttarakhand script.

### Model-selection configuration

```python
K_CANDIDATES = list(range(2, 11))                       # K = 2 … 10
K_FINAL      = 5                                        # line 73 — set manually after review
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

**The run used K = 5, one above the top of that recommended range.** With 45 points that is an
average of 9 points per component, and the smallest component has only 3.

### Feature matrix

`X = sig[[c for c in sig.columns if c.endswith("_z")]].fillna(median).values`

Only the `_z` (standardised) columns from `04b` are used. `lat`/`lon` are absent by construction —
`04b` dropped them from the clustering column list, and `05` re-prints the reason at run time:
"(lat/lon are NOT among these — never cluster on geography, plan v3.0 Section 6.2)."

The exact number of `_z` columns is **not available in the source files**
(`climate_signature_uttarakhand.csv` is git-ignored). From `04b`'s `DROP_FROM_CLUSTERING` logic it
comprises: the non-PCA canonical indices (`GHI_mean`, `kt_mean`, `kt_std`, `SAI`, `CCI`,
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

**The contents of both CSVs are not available in the source files** — `data/processed/clustering/`
is git-ignored, and no committed plot renders the BIC or K-Means selection curves for the actual
run. (`05b_cluster_interactive.py` would render them, but its output directory is git-ignored too.)

### Final fit and outputs

```python
k_final_safe = min(K_FINAL, len(X) - 1)      # = 5
gmm_final    = GaussianMixture(5, covariance_type="full", random_state=42, n_init=10)
hard_labels  = gmm_final.fit_predict(X)
soft_probs   = gmm_final.predict_proba(X)
```

| Output file | Contents |
|---|---|
| `bic_selection_uttarakhand.csv` | K = 2…10 × {BIC, silhouette, DB, CH, in_accept_band} |
| `kmeans_comparison_uttarakhand.csv` | K = 2…10 × K-Means silhouette |
| `cluster_assignments_uttarakhand.csv` | `point_id, lat, lon, population, cluster_id, max_membership_prob, prob_cluster0…4` |
| `cluster_profiles_uttarakhand.csv` | one row per cluster: `cluster_id, n_points, total_population_covered`, plus the **population-weighted mean** of every non-`_z` numeric signature column |
| `cluster_map_uttarakhand.png` | lon/lat scatter coloured by `cluster_id`, annotated `C0…C4` |

Population weighting uses `np.average(g[col], weights=g["population"])`, falling back to an
unweighted mean if the weight sum is zero.

`cluster_profiles_uttarakhand.csv` is what `07_feasibility_filter.py` and
`09_recommendation_cards.py` read. Because `profile_cols` is "everything not
`point_id`/`cluster_id` and not ending `_z`", it carries `Tm_target_C` and `L_required_kJ_per_kg`
through — which is exactly what `07` checks for and errors on if absent.

---

## Observed results

### Cluster assignments (all 45 points)

Recovered from the popups in `data/plots/uttarakhand_objective1/02_climate_regime_map_folium.html`:

| Cluster | n_points | Member `point_id`s |
|---|---|---|
| **0** | **12** | 0003, 0004, 0005, 0006, 0007, 0014, 0019, 0020, 0031, 0034, 0037, 0044 |
| **1** | **9** | 0002, 0008, 0011, 0021, 0024, 0025, 0026, 0033, 0036 |
| **2** | **3** | 0023, 0040, 0041 |
| **3** | **7** | 0001, 0009, 0010, 0012, 0013, 0016, 0017 |
| **4** | **14** | 0015, 0018, 0022, 0027, 0028, 0029, 0030, 0032, 0035, 0038, 0039, 0042, 0043, 0045 |

Independently corroborated by `data/plots/verify_clustering/06_cluster_sizes.png`, which prints
12 / 9 / 3 / 7 / 14. Total 45. Max/min size ratio = 14 / 3 = **4.67**.

### Population and geographic extent per cluster

Computed by joining the cluster assignments to the per-point populations and coordinates embedded
in `data/plots/comprehensive/maps/A2_population_map.html`:

| Cluster | n | Population covered | Share | Latitude range (mean) | Longitude range (mean) |
|---|---|---|---|---|---|
| 0 | 12 | **3,432,283** | 32.8 % | 29.125 – 30.375 (29.562) | 77.875 – 79.875 (78.854) |
| 1 | 9 | **2,451,043** | 23.4 % | 30.125 – 30.625 (30.292) | 78.125 – 78.875 (78.486) |
| 2 | **3** | **330,779** | 3.2 % | 30.125 – 30.375 (30.292) | 79.125 – 79.375 (79.292) |
| 3 | 7 | **2,541,919** | 24.3 % | 28.875 – 29.875 (29.268) | 77.875 – 79.875 (78.804) |
| 4 | 14 | **1,719,687** | 16.4 % | 29.125 – 30.625 (29.696) | 77.875 – 80.125 (79.625) |
| **Total** | **45** | **10,475,711** | 100 % | | |

Cluster 2 is the smallest by both point count (3) and population (3.2 %), and is the most spatially
compact — a 0.25° × 0.25° neighbourhood around 30.25° N, 79.25° E.

### Climate profile per cluster (observed medians)

From the boxplots in `data/plots/verify_clustering/05_cluster_profiles.png`, which plot the first
six numeric feature columns of the signature matrix. Values are read from the rendered chart and
are therefore **approximate to the plotting resolution**:

| Index (Tier-1 proxy) | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| `Ta_mean_proxy` (°C) | ~22.8 | ~19.0 | **~13.4** | **~25.0** | ~18.2 |
| `Ta_p95_proxy` (°C) | ~29.8 | ~25.6 | ~20.4 | ~32.8 | ~24.3 |
| `Ta_p05_proxy` (°C) | ~12.1 | ~9.1 | ~4.2 | ~13.8 | ~9.4 |
| `DTR_proxy` (K) | ~7.9 | ~7.8 | ~7.1 | ~7.9 | ~7.2 |
| `GHI_mean` (W/m², noon) | ~52.9 | ~44.5 | ~44.7 | ~55.1 | ~50.0 |
| `GHI_daily_kWh_proxy` (kWh/m²/day) | ~0.404 | ~0.342 | ~0.335 | ~0.428 | ~0.380 |

The temperature ordering is monotone and coherent: **C3 (warmest) > C0 > C1 > C4 > C2 (coldest)**,
spanning ~11.6 K of mean-temperature separation, with the same ordering reproduced in `Ta_p95` and
`Ta_p05`. Combined with the geographic extents above — C3 southernmost, C2 a compact
high-longitude/high-latitude pocket — the partition is internally consistent with an
elevation/latitude gradient.

> **The source files do not assign geographic names to the clusters.** No committed artefact in
> `era5-uttarakhand/` labels a cluster as "Terai", "Doon Valley" or "high Himalaya". Any such
> labelling in a write-up is interpretation added on top of the pipeline, not a pipeline output.

> **The `GHI_mean` and `GHI_daily_kWh_proxy` values above are affected by the ERA5 GHI magnitude
> anomaly** documented in `04_PHASE_2_AUDIT.md` Part A.3. Their *relative* ordering across clusters
> is still informative; their absolute magnitudes are not usable.

### Soft membership

Every one of the 45 popups reports `Prob: 1.000` — `max_membership_prob` rounds to 1.000 at three
decimal places for **every point**. The soft-clustering rationale in the docstring ("a point near
that boundary genuinely has partial membership in both") therefore did **not** materialise in
practice.

This is the expected behaviour of a full-covariance GMM fitted to 45 samples in a high-dimensional
standardised space — each component can shape itself tightly around its members. It means the
`prob_cluster0…4` columns carry no usable boundary information for this run, and
`05b_cluster_interactive.py`'s boundary-point feature (a faint ring where `max prob < 1.5/K`) would
have highlighted nothing.

### Silhouette

`data/plots/verify_clustering/02_silhouette_plot.png` reports, for the **saved K = 5 labels**:

| Metric | Value |
|---|---|
| Average silhouette | **0.279** |
| Reference threshold drawn on the plot | 0.400 |
| Per-cluster spread (approximate) | C0 0 – 0.35, C1 0 – 0.41, C2 0 – 0.61, C3 0 – 0.47, C4 −0.15 – 0.37 |

0.279 falls inside `05_cluster_uttarakhand.py`'s stated accept band of **0.15–0.40** and below the
0.4 "good" threshold used by `VERIFICATION_METHODOLOGY.md`. Cluster 4 (the largest, n = 14)
contains the only points with **negative** silhouette values, indicating a few points closer to a
neighbouring cluster's centroid than to their own.

> **Caveat on this number.** `verify_02_clustering.py` computes silhouette on **its own** feature
> matrix — every numeric column of `climate_signature_uttarakhand.csv` except
> `point_id/cluster_id/lat/lon/population`, re-standardised — which includes the raw indices, the
> `_proxy` and `_true` duplicates, the PCA-block members **and** the `_z` columns. That is a
> different and much larger space than the `_z`-only matrix the GMM was fitted in. The 0.279 figure
> is a valid independent diagnostic but is **not** the silhouette that `05_cluster_uttarakhand.py`
> wrote to `bic_selection_uttarakhand.csv` at K = 5. That value is not available in the source
> files.

---

## What is absent from Phase 4

| Component | Status |
|---|---|
| Bootstrap / ARI cluster-stability analysis | **Not implemented.** No resampling of any kind appears in `05_cluster_uttarakhand.py`. |
| Fitted-model persistence (`joblib` scaler + GMM) | **Not implemented.** Neither `04b`'s `StandardScaler` nor `05`'s fitted `GaussianMixture` is saved; re-running Phases 5–8 requires re-fitting. |
| `sklearn_version` recorded in outputs | **Not implemented.** |
| Canonical cluster relabelling (e.g. by ascending latitude) | **Not implemented.** Cluster IDs come straight from `GaussianMixture.fit_predict` and are stable only because `random_state=42` is fixed. |
| External climate classification (Köppen-Geiger, NBC/ECBC) | **Not implemented.** The K = 5 partition rests entirely on internal statistics. |
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
`11_LITERATURE_MAPPING.md`.

## Validation

| Check | Result |
|---|---|
| lat/lon excluded from the clustering matrix | **Confirmed** — dropped by `04b`, re-announced by `05` at run time |
| K selected from a four-metric table | **Implemented**; table contents not available in the source files |
| K-Means reported as a comparison | **Implemented**; contents not available |
| Silhouette inside the stated accept band | **PASS** — 0.279 in [0.15, 0.40] (verify-suite feature space) |
| Clusters spatially coherent | **PASS** — geographically contiguous despite geography being excluded |
| Cluster profiles population-weighted | **Confirmed** — `np.average(..., weights=population)` |
| Bootstrap stability | **Absent** |
| External classification agreement | **Absent** |

## Problems / risks

1. **K = 5 exceeds the source files' own recommendation.** `README_PREPROCESSING.md` says
   "realistically 2-4" for a 45-point single-state fit and warns that "a GMM fit on very few points
   per component gets unstable." Cluster 2 has 3 points and cluster 3 has 7.
2. **Soft membership collapsed to 1.000 everywhere**, so the stated methodological reason for
   choosing GMM over K-Means is not realised in this run. This should be reported, not left
   implicit.
3. **No stability evidence exists.** With no bootstrap ARI, no model persistence and no external
   classification, the only evidence for K = 5 is the (uncommitted) BIC/silhouette table and the
   verification suite's 0.279 silhouette.
4. **Cluster ID stability depends solely on `random_state=42`.** There is no canonical relabelling
   step, so any change to the signature matrix, sklearn version, or seed can permute cluster IDs
   and silently invalidate the `cluster_id`-keyed joins in `07`, `08` and `09` — none of which
   verify provenance.
5. **Cluster 2 is a 3-point regime carrying 3.2 % of population.** Every per-cluster statistic for
   it — profile means, survivor counts, MCDM ranks — rests on three sampling points.
6. **`Tm_target_C` enters the clustering matrix as a zero-variance column.** Harmless but untidy.
7. **`GHI_mean` enters the clustering matrix carrying the ERA5 GHI anomaly** — the one solar column
   the Tier-2 repair does not cover.

## Status

**COMPLETE.** The clustering is methodologically well argued (soft clustering for a continuous
gradient, geography excluded, four selection metrics plus a K-Means control, population-weighted
profiles) and the result is spatially coherent with a monotone temperature ordering — a genuine
positive finding given that latitude and longitude were excluded from the fit. The open items are
the aggressive K for N = 45, the total absence of stability evidence, and the unrealised soft
membership.
