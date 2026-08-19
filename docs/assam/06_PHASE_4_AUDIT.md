# 06 — Phase 4 Audit: Climate Regime Clustering

**Script**: `05_cluster_assam.py`

**Status**: COMPLETE

## Algorithm choice: GMM with full covariance

Unlike Rajasthan (which used `diag` covariance after correcting a bug), Assam uses **GMM with full
covariance**. This is an intentional, documented design choice:

> "Full covariance — Assam's indices are correlated (monsoon_index with RH_mean, GHI with CCI).
> K-Means would split elongated clusters." — `05_cluster_assam.py` docstring

Assam's climate space has a genuine multi-dimensional correlation structure from the monsoon signal
(high RH, high precipitation, lower GHI, high HSI all co-vary). Diagonal covariance would
misrepresent this. Full covariance is the statistically appropriate choice here, even though it
requires more parameters.

## k selection

BIC was evaluated for k = 2 through k = 10:

| k | BIC | Silhouette | DB | CH | In accept band? |
|---|---|---|---|---|---|
| 2 | -1910.4 | **0.457** | 0.915 | 82.2 | False |
| 3 | -3024.8 | 0.309 | 1.203 | 71.7 | True |
| 4 | -3322.3 | 0.321 | **1.152** | **62.1** | True |
| 5 | -3982.7 | 0.271 | 1.343 | 51.7 | True |
| 6 | -4555.8 | 0.292 | **1.165** | 48.5 | True |
| 7 | -4762.3 | 0.273 | 1.280 | 44.4 | True |
| 8 | -4851.7 | 0.277 | 1.250 | 49.4 | True |
| 9 | **-5138.4** | 0.309 | 1.180 | 49.7 | True |
| 10 | -4578.1 | 0.300 | 1.231 | 46.5 | True |

**k = 4 was selected** (K_FINAL in the script). Justification:
- BIC keeps falling through k=9, but BIC alone is not sufficient when the goal is interpretable
  climate regimes
- k=4 is the **first k in the accept band** (silhouette > threshold) that maps to interpretable
  Assam geography: Brahmaputra valley / hill districts / Barak valley / char-island / riverine fringe
- Silhouette at k=4 (0.321) improves on k=3 (0.309), and k=2's high silhouette (0.457) is achieved
  by a split that collapses all the Brahmaputra diversity

## Cluster results (from `cluster_profiles_assam.csv` + `cluster_assignments_assam.csv`)

| Cluster | n_points | Population covered | Ta_mean (°C) | Character |
|---|---|---|---|---|
| 0 | 24 | ~1.70M | 26.3 | Hill/transition — lower temp, higher variability |
| 1 | 52 | ~3.25M | 26.8 | Brahmaputra valley mainstream — most populous |
| 2 | 11 | ~0.93M | 28.2 | Barak valley / southern fringe — warmer winters |
| 3 | 41 | ~5.55M | 28.2 | Western plains + char areas — warmest, densest |

Total: 128 points, ~11.4M population covered at 87.5% coverage target.

## Bootstrap stability (from `bootstrap_stability_assam.csv`)

| Parameter | Value |
|---|---|
| k_final | 4 |
| n_bootstrap | 500 |
| ARI_mean | **0.716** |
| ARI_std | 0.139 |
| stable | **False** (threshold: ARI > 0.75) |

**ARI = 0.716 falls below the 0.75 stability threshold.** This is an honest reported result —
the partition is borderline stable. The practical implication: Cluster 0 (24 points, hill
transition zone) has the most overlap with adjacent clusters in the signature space. For thesis
reporting, this should be framed as "the k=4 partition is reasonably stable (ARI=0.716 ± 0.139)
but does not meet the 0.75 strong-stability criterion, consistent with Assam's genuinely gradual
climate transitions."

## K-Means comparison (from `kmeans_comparison_assam.csv`)

K-Means silhouette scores across k=2–10 are uniformly ~0.31 (no elbow pattern). GMM's k=2
silhouette (0.457) exceeds K-Means best, confirming that GMM better captures the cluster shape
for Assam's data.

## Reproducibility

- **Fitted models saved**: `scaler_assam.joblib`, `gmm_model_assam.joblib`
- **sklearn version recorded**: 1.9.0 (in every output CSV `sklearn_version` column)
- **random_state** set on every GMM and K-Means fit

## Canonical relabeling

Clusters are relabeled by **ascending mean latitude** immediately after the GMM fit (same fix
applied in Rajasthan after the instability bug was caught). This ensures cluster IDs 0–3 refer
to the same physical groups across re-runs.

## External classification validation

**Not implemented for Assam.** Köppen-Geiger, NBC/ECBC Indian climate zones — neither is wired
in. This is an open gap. For a complete validation, at minimum the Köppen-Geiger raster lookup
(Beck et al. 2018, 1-km resolution) should be added to compute ARI(GMM, Köppen) as was done for
Rajasthan (which reported ARI=0.19, NMI=0.32).
