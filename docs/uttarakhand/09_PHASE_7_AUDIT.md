# 09 — Phase 7 Audit: Physics-Based Validation

**Script**: `10_physics_validation.py`

**Status**: **COMPLETE.** Fully implemented, executed, and verified end-to-end against 10-year daily weather data across all 5 Uttarakhand climate regimes.

---

## Purpose of Phase 7

Phase 7 independently validates the Phase 6 MCDM preference ordering against a physical domain simulation. While the MCDM stack ranks PCMs based on static thermophysical properties, the grey-box physics model ranks them based on **simulated delivered thermal performance** (annual solar fraction and delivery target hours met) under realistic weather driving conditions.

Per plan v3.0 Section 10:
> "Everything up to MCDM produces a preference ordering. Nothing in it establishes that a higher-ranked PCM actually performs better... this phase makes the claim falsifiable."

---

## Architecture & Numerical Implementation

`10_physics_validation.py` implements a **two-node lumped-enthalpy tank simulation** adapted from Barqawi et al. (2025):

1. **Physical Nodes**:
   - Node 1: Tank Water Temperature ($T_w$)
   - Node 2: PCM Temperature ($T_p$) and Melt Fraction ($f$) during the isothermal phase change at $T_m$.
2. **Three-Phase Thermal Physics**:
   - Phase 1: Solid sensible heating ($T_p < T_m$)
   - Phase 2: Isothermal phase change ($T_p = T_m, 0 \le f \le 1$)
   - Phase 3: Liquid sensible heating ($T_p > T_m$)
3. **Solver & Numerical Stability**:
   - Solved hour-by-hour over 8,760 steps (365 days) for each cluster medoid point.
   - Uses **Implicit Backward Euler** integration ($2 \times 2$ linear system per step). Backward Euler guarantees unconditional numerical stability given the short thermal time constant of domestic water tanks relative to 1-hour time steps.
4. **Ambient Heat Loss Modeling**:
   - Includes shell-to-ambient thermal loss: $UA_{tank} = 2.0 \text{ W/K}$ (accounting for tank insulation and piping losses).

---

## Stated Simulation Parameters & Assumptions

| Parameter | Value | Description / Source |
|---|---|---|
| Tank Water Mass ($M_w$) | 150 kg | Standard domestic storage tank |
| Collector Area ($A_c$) | 2.5 m² | Mid-size solar collector array |
| Collector Efficiency ($\eta$) | 0.70 | Mid-range flat-plate collector efficiency |
| PCM Volume ($V_{pcm}$) | 0.035 m³ | Internal PCM encapsulation volume |
| Water-Coil HTC ($h_c$) | 1500 W/(m²·K) | Forced convection heat transfer coefficient |
| PCM-Water HTC ($h_p$) | 800 W/(m²·K) | Tank fluid-to-capsule coupling |
| PCM Surface Area ($A_p$) | 3.5 m² | Total heat exchanger surface |
| Domestic Water Draws | 75 kg at 07:00 IST<br>75 kg at 19:00 IST | 150 L/day total draw at target $T_{delivery} = 50\text{ }^\circ\text{C}$ |
| Ambient Heat Loss ($UA_{tank}$) | 2.0 W/K | Tank insulation shell loss to ambient air |

---

## Datasets Consumed & Generated

### Inputs Consumed:
- `data/processed/daily_aggregates_uttarakhand.csv` (from Phase 2 `02b`: real daily NASA POWER solar radiation $GHI_{daily}$ and diurnal temperature bounds $T_{a,\min}, T_{a,\max}$)
- `data/processed/suntimes.csv` (from Phase 1 `00b`)
- `data/processed/clustering/cluster_assignments_uttarakhand.csv` (from Phase 4 `05`)
- `data/processed/pcm/mcdm_full_scores_by_cluster.csv` (from Phase 6 `08`)

### Outputs Generated:
- `data/processed/pcm/physics_validation_results.csv`: Per-cluster, per-PCM annual solar fraction (SF), hours delivery target met, and completed annual melt/freeze cycles.
- `data/processed/pcm/physics_validation_spearman.csv`: Rank correlation ($\rho$) and $p$-value comparing MCDM rank vs. simulated solar fraction.

---

## Validation Results & Findings

### 1. Benchmark Calibration Check (Plan v3.0 Table 16)
**92% of all simulated PCM-cluster pairs** land within the published **54%–84%** annual solar fraction benchmark band for domestic solar water heating systems in India.

### 2. Cluster-by-Cluster Physics vs. MCDM Rank Concordance

| Cluster | Medoid Point | Typical Solar Fraction Band | Spearman $\rho$ | $p$-value | Interpretation |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Cluster 0** | UKP_0019 | 68.7% – 75.2% | **−0.454** | 0.044 | Statistically significant inverse rank correlation |
| **Cluster 1** | UKP_0036 | 68.7% – 75.2% | **+0.555** | 0.011 | Statistically significant positive agreement |
| **Cluster 2** | UKP_0023 | 51.1% – 62.6% | **+0.228** | 0.334 | Weak correlation (high elevation / low solar gains) |
| **Cluster 3** | UKP_0001 | 78.2% – 81.3% | **+0.138** | 0.561 | Weak correlation (high insolation plateau) |
| **Cluster 4** | UKP_0007 | 71.0% – 75.8% | **+0.155** | 0.514 | Weak correlation |
| **Mean** | — | — | **+0.124** | — | Overall weak positive correlation across regimes |

---

## Key Physical Insights & Diagnostics

1. **Delivered Solar Fraction Differentiation**:
   While Phase 5 and Phase 6 returned identical top candidates across clusters (due to uniform $T_{m,target} = 57\text{ }^\circ\text{C}$), Phase 7 demonstrates **clear regional performance differentiation**:
   - **Cluster 3 (Plains/Interior)** achieves the highest solar fractions (~78–81%), driven by strong daily solar insolation.
   - **Cluster 2 (High Himalayan)** yields lower solar fractions (~51–63%), directly reflecting mountain cloud cover and lower ambient temperatures.

2. **Explanation of Low Rank Correlation ($\rho = 0.124$)**:
   - In Phase 6, the TOPSIS and GRA methods showed strong anti-correlation ($\rho = -0.930$), making the MCDM consensus rank a positionally averaged compromise.
   - The physical simulation shows that among top feasibility survivors, thermal storage capacity and melting point ($T_m$) have non-linear interactions with daily draw schedules that static MCDM property weighting cannot capture.

---

## Verification Status

**COMPLETE & VERIFIED.** `10_physics_validation.py` runs cleanly, produces valid physics outputs, and satisfies all requirements of Section 10 of the Objective 1 framework plan.
