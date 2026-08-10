# 08 — Phase 6 Audit: Multi-Criteria Ranking Engine

Script: `08_mcdm_ranking.py`.

## Purpose
Rank the surviving PCM candidates in each cluster using four independent multi-criteria decision-making (MCDM) methods under weight and property uncertainty.

## Processing Details
1. **Target-Based Fitness**:
   - Converts melting temperature to a Gaussian fitness score:
     `f_Tm = exp( - (Tm - Tm_target)² / (2 * σ²) )`, where `σ = 4.0` K.
2. **Criteria Evaluated**:
   - `f_Tm` (melting point fitness) - benefit.
   - `latent_heat_margin_ratio = latent_heat / L_required` (climate-relative benefit).
   - `rho_H_MJ_m3` (volumetric latent heat) - benefit.
   - `TC_W_mK` (thermal conductivity) - benefit.
   - `cycles_confidence` (log-scaled cycling reliability) - benefit.
3. **Four MCDM Methods**:
   - **TOPSIS**: Closeness to Euclidean ideal/anti-ideal.
   - **GRA**: Grey relational grade vs max reference.
   - **PROMETHEE II**: Net outranking flow (V-shape, q=0.10, p=0.30).
   - **VIKOR**: Compromise index Q (v=0.5) with acceptable-advantage check.
4. **Weights**:
   - Entropy weights (data-driven) blended with AHP prior weights (Table 13 priors) at `λ = 0.5`.
5. **Consensus & Uncertainty**:
   - Primary rank: Borda count across the 4 methods.
   - Cross-check: Copeland pairwise majority.
   - **Monte Carlo**: 5,000 Dirichlet weight draws + Gaussian property perturbations (Tm ±1K, latent heat ±5%, conductivity ±10%). Calculates Top-3 inclusion probability and Top-1 retention.

## Results
- Ranks the 7 survivors for each cluster.
- **Identical Rankings Across Clusters**:
  Since all clusters had the same 7 survivors and very similar weights (due to similar `L_required` and GHI indices), the ranked candidates are almost identical. `RT54HC` or `RT55` are ranked high across all regimes.
- MC stability indicates high confidence for the top candidates.

## Status
**COMPLETE**
