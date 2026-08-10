# 06 — Phase 4 Audit: Climate Regime Clustering

Scripts: `05_cluster_tamilnadu.py`, `05b_cluster_interactive.py`, `11_level_b_seasonal_analysis.py`.

## Purpose
Group the 133 population points into distinct climatic regimes using GMM clustering (Level A) and evaluate whether these regimes experience seasonal shifts that change the recommended PCM (Level B).

## Level A: GMM Clustering
- Fits K components from 2 to 10. Computes BIC and silhouette scores.
- **Tamil Nadu Choice**: Selects **K_FINAL = 5** regimes.
- Fits a GMM with `covariance_type="full"`.
- **Tamil Nadu Profiles**:
  - *Cluster 0*: 12 points, GHI ~ 5.10, Ta_mean ~ 28.0°C.
  - *Cluster 1*: 43 points, GHI ~ 5.25, Ta_mean ~ 28.0°C.
  - *Cluster 2*: 39 points, GHI ~ 5.10, Ta_mean ~ 28.3°C.
  - *Cluster 3*: 22 points, GHI ~ 5.11, Ta_mean ~ 27.4°C.
  - *Cluster 4*: 17 points, GHI ~ 5.35, Ta_mean ~ 27.2°C.

## Level B: Seasonal Sensitivity
- For each of the 5 clusters, recomputes `L_required_season` per season (Winter, Summer, Monsoon, Retreat) and re-runs a single-method TOPSIS ranking.
- Checks if the #1 ranked PCM flips between seasons.
- **Tamil Nadu Results**:
  - Re-ranks the 7 survivors. Since the latent-heat floor was bypassed (due to the 1000x flow-rate bug), candidates are identical across seasons.
  - Out-of-phase monsoon rainfall (NE monsoon vs SW monsoon) provides strong physical basis for seasonal fluctuations, which is captured in the database assignments.

## Critical Audit Findings (GMM Overfitting)
- Unlike Rajasthan where the covariance type was corrected to `diag` (diagonal), the Tamil Nadu script uses `covariance_type="full"`.
- For 133 samples and 27 dimensions, a full covariance matrix requires fitting `(27 * 28 / 2) = 378` covariance parameters *per cluster*. With 5 clusters, this is `378 * 5 = 1890` parameters, which severely overdetermines the model on only 133 samples. This leads to membership probability saturation (probs ≈ 1.0) and poor generalization.

## Status
**NEEDS CORRECTION** (Change covariance type from `full` to `diag` and resolve input features bias).
