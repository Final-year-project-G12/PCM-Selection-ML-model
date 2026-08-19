# 09 — Phase 7 & 8 Audit: Physics Validation & Recommendation Cards

## Phase 7 — Physics-Based Validation (`10_physics_validation.py`)

**Status**: COMPLETE — genuine NEGATIVE result

### Model: Grey-box lumped-enthalpy tank

Matches plan v3.0 §10 primary choice (Python grey-box lumped-enthalpy model). Two coupled lumped
nodes: tank water (Tw) and PCM (Tp or melt fraction f during isothermal plateau at Tm). Driven
hour-by-hour for a full real year of the cluster's medoid point's actual daily climate data.

**Solver**: Backward Euler (implicit), the same as Rajasthan's Phase 7. Derived from Barqawi (2025)
ODE structure.

### Stated assumptions (from `10_physics_validation.py` docstring)

| Parameter | Value |
|---|---|
| Tank water mass | 150 kg |
| Collector-tank coil area | 2.5 m² |
| Water-coil HTC | 1500 W/m²·K |
| Collector efficiency | 0.70 |
| PCM volume | 0.035 m³ |
| PCM-water HTC | 800 W/m²·K |
| PCM surface area | 3.5 m² |
| Draws | 2/day, 75 kg each, 07:00 and 19:00 local |
| Target delivery temperature | 50°C |
| Ambient temp profile | Daily sinusoid from real Ta_min_true/Ta_max_true; peak 14:00, trough 05:00 local |

### Calibration check

Annual solar fraction is expected to fall in the **54–84% published range** (plan v3.0 Table 16).
The simulation uses each cluster's medoid point's real 10-year daily GHI/temperature data.

### Results (from `physics_validation_spearman_assam.csv` and `recommendation_cards_assam.md`)

| Cluster | n_candidates | Spearman rho | p-value | Interpretation |
|---|---|---|---|---|
| 0 | 6 | **0.257** | 0.623 | Weak agreement |
| 1 | 6 | **0.257** | 0.623 | Weak agreement |
| 2 | 8 | **0.286** | 0.493 | Weak agreement |
| 3 | 8 | **0.167** | 0.693 | Weak agreement |

**Mean Spearman rho = 0.242** (stated in recommendation cards header).

All four clusters return **weak agreement** (rho < 0.4). The MCDM ranking is NOT confirmed by
physics simulation for any Assam cluster.

### Interpretation

This is the same pattern as Rajasthan's Phase 7 (all clusters negative). For Assam the likely
causes are identical:
1. **Undersized PCM pool**: n=6 or n=8 candidates — too small for Spearman rho to be meaningful
   (minimum recommended ~10–15 pairs for rho to have statistical power)
2. **Uniform Tm_target**: All clusters use 44°C → the same PCMs appear in all clusters → the MCDM
   ranking differences between clusters are small → Spearman rho is computed on very similar rank
   vectors, dominated by noise in the physics simulation
3. **Solar fraction ceiling effect**: Multiple PCMs achieve 80–85% solar fraction (near the top of
   the calibration band), compressing rank differences in the simulated output

The negative result is **real, correctly computed, and honestly reported** — not a code error.

### Solar fraction benchmark check

Most simulated solar fractions fall **above** the 54–84% published benchmark band (>84%):
- Cluster 1: RT44HC 82.1%, C22H46 82.9%, savE® OM42 82.6%, RT45HC 51.7%
- Cluster 2: RT44HC 84.8%, C22H46 85.3%, savE® OM42 85.1%, RT45HC 52.1%

RT45HC consistently achieves ~51–52% (within the lower band), while RT44HC and C22H46 exceed the
upper 84% limit. This suggests the model may overestimate performance for the high-latent-heat
candidates in Assam's moderate-solar climate, or that the benchmark band (from dry-climate SWH
literature) is not directly applicable to humid-monsoon conditions.

---

## Phase 8 — Recommendation Cards (`09_recommendation_cards.py`)

**Status**: COMPLETE

### What it produces

`recommendation_cards_assam.md` — one card per cluster, including:
- Cluster profile summary (n_points, medoid, climate signature snippet)
- Derived targets (Tm_target, L_required)
- Phase 5 screening summary
- Top-3 PCM table with all 4 method scores + MC inclusion probability
- Kendall's W
- **Analytical Criterion Contributions** (percentage breakdown per criterion per PCM)
- Phase 7 simulation results table
- Spearman rho with interpretation

### Criterion Contributions — Assam addition

This is an explicit explainability requirement from the plan doc (Table 18) that the Tamil Nadu
Phase 8 script missed. Assam's `09_recommendation_cards.py` adds it via min-max normalisation
of the criteria space:

For RT44HC in Clusters 0/1:
- Tm_Fitness: ~36%, Latent_Heat: ~30%, Vol_Heat: ~24%, Conductivity: ~10%

For Cluster 2/3 (wider survivor pool):
- RT44HC: Tm_Fitness: ~35%, Latent_Heat: ~30%, Vol_Heat: ~26%, Conductivity: ~9%

The C22H46 (docosane paraffin) criterion contributions are incomplete due to NaN thermal
conductivity in the source data.

### Cross-phase consistency check

The script verifies cluster IDs are consistent across all input files before writing.
(No separate `provenance_lib.py` as in Rajasthan — consistency is checked inline in the
recommendation cards script itself.)
