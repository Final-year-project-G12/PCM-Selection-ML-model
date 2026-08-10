# 19 — Phase 7 & 8 Audit: Physics-Based Validation and Output

Unlike Rajasthan, where Phase 7 and 8 were only planned, the **Tamil Nadu pipeline has fully implemented both phases**.

## Phase 7: Grey-Box Physics Validation (`10_physics_validation.py`)
1. **Model Structure**:
   - Implements a 3-phase lumped-enthalpy tank model (adapted from Barqawi 2025):
     - *Phase 1 (Sensible Solid)*: Tp < Tm (PCM is solid; heat transfer changes temperature sensibly).
     - *Phase 2 (Isothermal Melting)*: Tp = Tm (isothermal plateau; heat input increases melt fraction f from 0 to 1).
     - *Phase 3 (Sensible Liquid)*: Tp = Tm (PCM is liquid; sensible heating).
   - Driven by the medoid point of each cluster using a full year of real 10-year daily aggregates (`02b` output).
2. **Numerical Method**:
   - Solved with **backward Euler (implicit)** at hourly time steps (`dt = 3600.0` s) to ensure numerical stability for the tightly-coupled differential equations.
3. **Validation Outcome**:
   - Calculates Spearman rank correlation (consensus rank vs simulated solar fraction) to make the MCDM ranking falsifiable:
     - **Cluster 0**: `r = 0.179` (weak agreement)
     - **Cluster 1**: `r = 0.324` (weak agreement)
     - **Cluster 2**: `r = 0.360` (weak agreement)
     - **Cluster 3**: `r = 0.360` (weak agreement)
     - **Cluster 4**: `r = 0.536` (partial agreement)
   - Simulated solar fractions are systematically high (**90–99%**), which is outside the published target band of 54–84%.
   - **Cycles/Year**: Almost all simulated PCMs have **0 or 1 complete cycles per year**.

## Root Cause of Weak Validation Agreement
1. **Disabled Latent Heat Constraint**: Due to the 1000x underestimation of `L_required` in Phase 3, all 25 candidates cleared the latent heat floor, bypassing screening.
2. **Missing Tank Heat Loss**: The simulation does not model ambient tank heat loss. The tank stays hot overnight, resulting in high solar fractions (90-99%) and preventing the PCM from freezing, leading to 0-1 cycles/year.
3. **GHI Feature Contamination**: GMM regimes were clustered using the corrupted, near-zero GHI (`GHI_mean_z`), leading to distorted cluster boundaries.

## Phase 8: Recommendation Cards (`09_recommendation_cards.py`)
- Reads the cluster profiles, MCDM rankings, and physics validation results, and outputs a formatted Markdown file `recommendation_cards.md`.
- Successfully lists the Top-3 candidates, Borda/Copeland agreement, Monte Carlo inclusion probabilities, and Spearman correlations.
