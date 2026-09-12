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
- Ranks the current feasibility survivors for each cluster. The generated `mcdm_topk_by_cluster.csv` contains the Top-3 for each of the five clusters (15 rows total).
- The current climate-relative latent-heat criterion is `latent_heat / L_required`, so the score retains cluster-specific demand information rather than treating raw latent heat as equally useful everywhere.
- Do not describe the ranking as seven survivors per cluster; the feasibility file contains all 62 audited candidates per cluster and the number passing all filters varies by cluster.
- Monte Carlo stability reports run-specific Top-3 inclusion and Top-1 retention probabilities; quote values from `monte_carlo_stability.csv` for the particular run being reported.
- In the current run, `n-Octacosane (C28)` is the consensus rank-1 PCM in all five clusters. This is a statewide consensus result; it does not imply that all alternatives have equal stability or physical performance.

## Status
**COMPLETE**

## Literature Support
| Component | Reference | Source |
|---|---|---|
| TOPSIS | Hwang & Yoon (1981); Chen et al. (2025) SWH MCDM | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
| GRA | Deng (1982); Chen et al. (2025) | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
| PROMETHEE II | Brans & Mareschal (2005) | Standard MCDM literature |
| VIKOR | Opricovic & Tzeng (2004) | Standard MCDM literature |
| Monte Carlo uncertainty | Chopra et al. (2023) techno-economic MC | `sources/Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md` |
| Entropy+AHP weight blend | Framework doc Table 13 | `17_LITERATURE_MAPPING.md` |
