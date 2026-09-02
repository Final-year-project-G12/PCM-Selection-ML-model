# 08 — Phase 6 Audit: MCDM Ranking Engine

**Script**: `08_mcdm_ranking.py`

**Status**: COMPLETE

## Method stack

All four MCDM methods from plan §9 are implemented:

| Method | Score type | Range |
|---|---|---|
| TOPSIS | Closeness coefficient C_i | [0, 1] (higher = better) |
| GRA | Grey relational grade | [0, 1] (higher = better) |
| PROMETHEE II | Net outranking flow φ | [−1, +1] (higher = better) |
| VIKOR | Compromise index Q_i | [0, 1] (lower = better — correct sign) |

### Criteria (all benefit after transformation)

| Criterion | Derivation | Physical rationale |
|---|---|---|
| `f_Tm` | Gaussian fitness: exp(−(Tm − 44)² / 2×4²) | Melting point proximity to target (σ=4K) |
| `latent_heat_margin_ratio` | L / L_required | Climate-relative latent heat (avoids constant winner when Tm_target is uniform) |
| `rho_H_MJ_m3` | Volumetric latent heat (MJ/m³) | Tank size implication |
| `TC_W_mK` | Thermal conductivity (W/m·K) | Charge/discharge rate |
| `cycles_confidence` | log-scaled cycling stability | Long-term durability |

**Latent heat margin ratio**: Because Tm_target = 44°C is **uniform across all Assam clusters**,
raw latent heat alone provides zero cluster-specific information — the same 6 or 8 PCMs would
receive identical L scores in every cluster. Using L / L_required makes the criterion
**climate-relative**: a PCM that exceeds the Assam-regime's specific energy demand earns more
credit. This is the correct formulation and is documented in `08_mcdm_ranking.py`.

### Weighting: entropy + AHP blend

- **Entropy weights**: Shannon entropy from the normalised decision matrix (objective)
- **AHP weights**: Framework doc's Table 13 priors (subjective; no pairwise elicitation performed —
  same status as Rajasthan: `AHP_PAIRWISE_MATRIX = None`)
- **Blended weight**: `ENTROPY_AHP_LAMBDA = 0.5` (equal blend)

### Consensus layer

- **Borda count** (primary): rank sum across 4 methods
- **Copeland pairwise** (cross-check): pairwise dominance matrix
- **Kendall's W** per cluster: inter-method concordance

### Monte Carlo (5,000 draws)

- `N_MONTE_CARLO_DRAWS = 5000` — **matches the plan spec §9.6 exactly** (Rajasthan used 1,000
  as a documented deviation; Assam corrects this)
- Per-draw: Dirichlet weight perturbation (concentration=30) + Gaussian property noise
  (Tm ±1K, L ±5%, TC ±10%, rhoH ±8%)
- Outputs: Top-3 inclusion probability per PCM per cluster

## Results (from `mcdm_topk_assam.csv`)

### Cluster 0 (24 pts, hill/transition, L_required ~249 kJ/kg)

| Rank | PCM | Family | Tm (°C) | L (kJ/kg) | Kendall's W |
|---|---|---|---|---|---|
| 1 | **RT44HC** | Rubitherm RT | 43.0 | 250 | **0.807 (strong)** |
| 2 | RT45HC | Rubitherm RT | 47.0 | 230 | |
| 3 | C22H46 (docosane-class paraffin) | Paraffin | 44.5 | 249 | |

MC inclusion probability: RT44HC 96.2%, RT45HC 62.2%, C22H46 39.3%

### Cluster 1 (52 pts, Brahmaputra valley, L_required ~244 kJ/kg)

| Rank | PCM | Family | Tm (°C) | L (kJ/kg) | Kendall's W |
|---|---|---|---|---|---|
| 1 | **RT44HC** | Rubitherm RT | 43.0 | 250 | **0.807 (strong)** |
| 2 | RT45HC | Rubitherm RT | 47.0 | 230 | |
| 3 | C22H46 (docosane-class paraffin) | Paraffin | 44.5 | 249 | |

MC inclusion probability: RT44HC 96.2%, RT45HC 62.2%, C22H46 39.3%

### Cluster 2 (11 pts, Barak valley/south, L_required ~232 kJ/kg)

| Rank | PCM | Family | Tm (°C) | L (kJ/kg) | Kendall's W |
|---|---|---|---|---|---|
| 1 | **RT44HC** | Rubitherm RT | 43.0 | 250 | **0.845 (strong)** |
| 2 | RT45HC | Rubitherm RT | 47.0 | 230 | |
| 3 | C22H46 (docosane-class paraffin) | Paraffin | 44.5 | 249 | |

MC inclusion probability: RT44HC 95.2%, RT45HC 67.7%, C22H46 26.5%

### Cluster 3 (41 pts, western plains/char, L_required ~233 kJ/kg)

| Rank | PCM | Family | Tm (°C) | L (kJ/kg) | Kendall's W |
|---|---|---|---|---|---|
| 1 | **RT44HC** | Rubitherm RT | 43.0 | 250 | **0.845 (strong)** |
| 2 | RT45HC | Rubitherm RT | 47.0 | 230 | |
| 3 | C22H46 (docosane-class paraffin) | Paraffin | 44.5 | 249 | |

MC inclusion probability: RT44HC 95.2%, RT45HC 67.7%, C22H46 26.5%

## Key observation: identical #1 across all clusters

RT44HC is the unanimous #1 candidate across all 4 Assam clusters. This is expected given the
uniform Tm_target = 44°C: RT44HC at Tm=43°C has the best Gaussian fitness (1K from target) plus
the highest latent heat in the survivor pool (250 kJ/kg). The identical top-3 ranking in clusters
0/1 and the identical ranking in clusters 2/3 reflects the similar L_required values and the same
corrosion-veto-driven survivor pool.

This uniformity is not a bug — it is the correct mathematical outcome given:
1. Uniform Tm_target = 44°C → same f_Tm for all clusters
2. Similar L_required (232–249 kJ/kg) across clusters → similar margin ratios
3. Small database (25 rows, 6–8 survivors per cluster after veto)

## Known issues

1. **AHP not elicited**: AHP_PAIRWISE_MATRIX = None. Framework doc's Table 13 priors used unmodified.

2. **Literature PCM property gaps**: Criterion contributions for C22H46 and other Singh2025 rows
   are incomplete due to NaN thermal conductivity/density (see Phase 5 audit).

3. **All results are provisional**: Tagged as provisional pending PCM database expansion (25 → 40–60
   rows). The uniform top-3 result is likely to change when more candidates with Tm ≈ 44°C are added.
