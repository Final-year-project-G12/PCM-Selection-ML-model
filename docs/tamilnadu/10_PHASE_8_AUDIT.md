# 10 — Phase 8 Audit: Recommendation Cards

Script: `09_recommendation_cards.py`.

## Purpose
Synthesize the multi-phase outputs — climate signatures, GMM cluster profiles, feasibility screening, multi-criteria MCDM rankings, Monte Carlo uncertainty bounds, and grey-box physics validation results — into standalone, human-readable **Recommendation Cards** (`recommendation_cards.md`) for each climate regime.

---

## Card Generation Workflow

### 1. Data Aggregation Input Stack
`09_recommendation_cards.py` reads and synthesizes outputs from across the pipeline:
- **Phase 4 Clusters**: `cluster_profiles_tamilnadu.csv` (cluster medoids, coordinates, representative population).
- **Phase 3 Signature**: `climate_signature_tamilnadu.csv` (regime $T_{\text{amb}}$, GHI, HSI, DTR, $T_{m,\text{target}}$, $L_{\text{required}}$).
- **Phase 5 Screening**: `feasibility_survivors_by_cluster.csv` (audited survivor candidates).
- **Phase 6 MCDM Engine**: `mcdm_topk_by_cluster.csv` and `monte_carlo_stability.csv` (Top-3 Borda ranks, Copeland checks, Top-1 retention, Top-3 inclusion probabilities).
- **Phase 7 Physics Validation**: `physics_validation_results.csv` and `physics_validation_spearman.csv` (simulated annual solar fraction, Spearman $\rho$, annual cycles).

### 2. Card Content Architecture
Each regime card in `recommendation_cards.md` provides:
1. **Regime Profile**: Cluster ID, medoid coordinates, member point count, population coverage, baseline climatology ($T_{a,\text{mean}}$, $\text{GHI}_{\text{daily}}$, HSI).
2. **Thermal Performance Targets**: Target melting point ($T_{m,\text{target}}$) and latent heat demand ($L_{\text{required}}$ with `SHARE_PCM = 0.5`).
3. **Screening Summary**: Total candidates audited (62), count of survivors passing Table 12 constraints.
4. **Ranked Recommendations (Top-3)**: Recommended PCM trade names, chemical formulas, melting points, latent heats, thermal conductivities, MCDM Borda ranks, and Monte Carlo confidence intervals.
5. **Physics Validation Verdict**: Simulated 10-year solar fraction, annual cycle counts, and cluster Spearman rank concordance ($\rho$).

---

## Output Status (Completed 62-PCM Run)

- **Generated File**: `recommendation_cards.md` (located in `data/processed/processed/pcm/recommendation_cards.md`).
- **Cluster Recommendations**: Produces 5 dedicated cluster cards covering all 133 population-weighted points in Tamil Nadu.
- **Statewide Winner**: `n-Octacosane (C28)` emerges as the Top-1 consensus recommendation across all five clusters, supported by Monte Carlo stability metrics.

---

## Status
**Analysis COMPLETE (62-PCM run)** — Re-run `09_recommendation_cards.py` (after `10_physics_validation.py`) to update cards in the canonical location whenever upstream rankings or physics outputs change.

---

## Literature Support

| Component | Reference / Method | Source File |
|---|---|---|
| Recommendation Cards Format | Odoi & Yorke (2025) AI SWH review | `sources/OdoiYorke2025AI_SWH_Review_summary.md` |
| Demand-Aligned Output | Objective 1 Deliverable D7 | `01_PROJECT_CONTEXT.md` |
| Multi-Criteria Consensus | Chen et al. (2025) Taguchi+GRA | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
