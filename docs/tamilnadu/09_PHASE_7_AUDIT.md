# 09 — Phase 7 Audit: Physics-Based Validation

Script: `10_physics_validation.py`.

## Purpose
Validate the MCDM material rankings by solving a grey-box transient energy balance simulation of a PCM-integrated solar water heating tank driven by 10 years of real daily weather for each regime's medoid point, evaluating rank concordance against simulated annual solar fraction.

---

## Model Architecture & Governing Physics

### 1. 3-Phase Lumped-Enthalpy Tank Model (Barqawi 2025)
The thermal storage tank is modeled as a lumped-enthalpy system undergoing three distinct phase states:
1. **Sensible Solid Phase** ($T_{\text{tank}} < T_m$): Heat addition increases solid PCM temperature ($c_{p,\text{solid}}$).
2. **Isothermal Phase Change Phase** ($T_{\text{tank}} = T_m$): Energy absorbed at constant temperature melts the PCM ($L_{\text{fusion}}$).
3. **Sensible Liquid Phase** ($T_{\text{tank}} > T_m$): Heat addition increases liquid PCM temperature ($c_{p,\text{liquid}}$).

### 2. Governing Energy Balance ODE
The system temperature $T_{\text{tank}}$ evolves according to:
$$C_{\text{tank}} \frac{d T_{\text{tank}}}{dt} = Q_{\text{solar}} - Q_{\text{draw}} - U A_{\text{tank}} (T_{\text{tank}} - T_{\text{amb}})$$

- $Q_{\text{solar}} = \eta_{\text{coll}} \cdot A_{\text{coll}} \cdot \text{GHI}$: Collector solar thermal input.
- $Q_{\text{draw}} = \dot{m}_{\text{draw}} \cdot c_{p,\text{water}} \cdot (T_{\text{tank}} - T_{\text{mains}})$: Hot water draw thermal load (300 L/day profile).
- **Ambient Heat Loss Fix (v3.1)**: $U A_{\text{tank}} = 2.0\text{ W/K}$ added to represent tank envelope heat loss to surrounding ambient air ($T_{\text{amb}}$).

### 3. Numerical Solver
- **Scheme**: Backward Euler (implicit 1st order) to guarantee numerical stability across stiff phase-transition boundaries.
- **Time Step**: Hourly step $dt = 3600\text{ s}$ over the full 10-year simulation span (87,660 time steps per candidate).

---

## Validation Findings (Completed 62-PCM Run)

Read from `data/processed/processed/pcm/physics_validation_results.csv` and `physics_validation_spearman.csv` (59 survivor simulations across 5 clusters):

### 1. Spearman Rank Concordance ($\rho$)
Evaluates rank agreement between MCDM Borda consensus order and simulated 10-year solar fraction:
- **Cluster 0**: $\rho = -0.016$ (near-zero agreement)
- **Cluster 1**: $\rho = +0.717$ ($p \approx 0.03$, strong rank concordance)
- **Cluster 2**: $\rho = +0.355$ (moderate positive agreement)
- **Cluster 3**: $\rho = -0.171$ (weak inverse correlation)
- **Cluster 4**: $\rho = -0.000$ (no correlation)
- **Statewide Mean Concordance**: **Mean Spearman $\rho = +0.177$** across all five clusters.

### 2. Benchmark Band & Thermal Performance
- **Solar Fraction Range**: Annual solar fraction spans **31% to 80%**.
- **Benchmark Inclusion**: **24 out of 59 simulations** fall within the published 54%–84% literature benchmark band (Singh et al. 2025) (per cluster: 6/15, 6/9, 5/13, 4/13, 3/9).
- **Statewide Winner (`n-Octacosane (C28)`)**: Simulated solar fraction is 0.71 / 0.65 / 0.69 / 0.51 / 0.31 for clusters 0–4 (in band for clusters 0–2, below band for 3–4).
- **Annual Cycling Stability**: Thermal cycling spans **3 to 260 complete cycles/year** (rank-1: 47 / 65 / 61 / 71 / 24 cycles/yr), confirming that adding $U A_{\text{tank}} = 2.0\text{ W/K}$ enabled realistic daily charge/discharge cycling.

---

## Status
**Analysis COMPLETE (62-PCM run)** — Re-run `10_physics_validation.py` (after `08_mcdm_ranking.py`) to regenerate results in the canonical location. Interpret findings cluster-by-cluster rather than assuming a single global pass/fail.

---

## Literature Support

| Component | Reference / Model | Source File |
|---|---|---|
| Lumped-Enthalpy Tank Model | Barqawi (2025) dynamic simulation | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
| Solar Fraction Benchmark (54–84%) | Singh et al. (2025) review | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Spearman Rank Validation | Framework doc §10 (N5 Novelty) | `13_LITERATURE_MAPPING.md` |
| Backward Euler Numerical Scheme | Ghodusinejad (2026) physics models | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
