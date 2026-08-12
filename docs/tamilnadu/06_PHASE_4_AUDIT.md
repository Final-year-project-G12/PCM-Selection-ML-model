# 06 — Phase 4 Audit: Climate Regime Clustering

Scripts: `05_cluster_tamilnadu.py`, `05b_cluster_interactive.py`, `11_level_b_seasonal_analysis.py`.

## Purpose
Group the 133 population points into distinct climatic regimes using GMM clustering (Level A) and evaluate whether these regimes experience seasonal shifts that change the recommended PCM (Level B).

## Level A: GMM Clustering (v3.1 corrected)
- Fits K components from 2 to 10; computes BIC and silhouette scores.
- **Tamil Nadu Choice**: K_FINAL = 5 regimes.
- **v3.1 fix**: `covariance_type="diag"` (was `"full"`, which overfit 133×27 features).
- Pre-fix profiles (will change after re-run with corrected GHI features):
  - Cluster 0: 12 pts; Cluster 1: 43 pts; Cluster 2: 39 pts; Cluster 3: 22 pts; Cluster 4: 17 pts.

## Level B: Seasonal Sensitivity (v3.1 corrected)
- Recomputes `L_required_season` per season using 300 L/day draw (matching `04b`).
- Single-method TOPSIS re-rank per (cluster, season); reports #1 PCM flips.
- NE monsoon out-of-phase cycle provides physical basis for seasonal variation.

## Corrected Finding (v3.1 — GMM Overfitting)
- **Was**: `covariance_type="full"` → 1890 covariance parameters on 133 samples → membership saturation.
- **Fixed**: `covariance_type="diag"` in `05_cluster_tamilnadu.py`.

## Status
**COMPLETE (v3.1 fixes applied — re-run `05` and `11` after Phase 3 re-run)**

## Literature Support
| Component | Reference | Source |
|---|---|---|
| GMM climate regime discovery | Liu et al. (2025) — AI PCM TES | `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
| Population-weighted clustering | Novelty N1 (framework doc) | `01_PROJECT_CONTEXT.md` |
| Seasonal PCM sensitivity | Singh et al. (2025) — monsoon SWH | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Diagonal GMM regularization | Standard small-n practice | `METHODS.md` §05 |
