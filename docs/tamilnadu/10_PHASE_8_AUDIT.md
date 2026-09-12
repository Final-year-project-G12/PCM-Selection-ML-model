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
- **Phase 5 Screening**: `feasibility_survivors_by_cluster_kappa_calibrated.csv` (κ-calibrated audited survivor candidates — the file `09` reads after the 2026-09-08 unification).
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

## Output Status (pre-unification run — STALE)

- **Generated File**: `recommendation_cards.md`. The old copy lived in the
  now-deleted `data/processed/processed/pcm/` tree; the canonical location is
  `data/processed/pcm/recommendation_cards.md`.
- **2026-09-08 unified run (3 clusters, INDICATIVE — re-run pending):** consensus Top-1 is
  `Myristic acid` (Cluster 0) / `n-Tetracosane (C24)` (Clusters 1 & 2) — no longer a single
  statewide winner, and no longer `n-Octacosane` (which is not in the k=3 survivor pool). The
  seasonal-sensitivity check finds Cluster 0's annual #1 flips to `n-Tetracosane (C24)` in all
  four seasons. Regenerate after the fresh Phase 6 run.

---

## Status
**Clean re-run PENDING (Phase 5 unified 2026-09-08).** Re-run `06 → 07 → 08 →
10 → 09` after the unified upstream chain to regenerate the cards in the
canonical `data/processed/pcm/` location. `09` now reads
`feasibility_survivors_by_cluster_kappa_calibrated.csv`.

---

## Literature Support

| Component | Reference / Method | Source File |
|---|---|---|
| Recommendation Cards Format | Odoi & Yorke (2025) AI SWH review | `sources/OdoiYorke2025AI_SWH_Review_summary.md` |
| Demand-Aligned Output | Objective 1 Deliverable D7 | `01_PROJECT_CONTEXT.md` |
| Multi-Criteria Consensus | Chen et al. (2025) Taguchi+GRA | `sources/Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` |
