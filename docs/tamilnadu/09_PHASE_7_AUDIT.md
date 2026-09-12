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

## Validation Findings — 2026-09-08 unified run (3 clusters, INDICATIVE — re-run pending)

> Ran against the unified Phase 5/6 outputs. Tamil Nadu's Phase 4 now yields **k=3** clusters
> (13/13/16 survivors, n=42), not the earlier 5. The Phase 6 build used had a residual
> supercooling-cap bug (fixed after); a fresh run is pending. Read cluster-by-cluster.

### 1. Spearman Rank Concordance ($\rho$) — MCDM consensus rank vs simulated annual solar fraction
- **Cluster 0**: $\rho = +0.791$ (partial-to-strong agreement)
- **Cluster 1**: $\rho = +0.680$ (partial agreement)
- **Cluster 2**: $\rho = +0.478$ (partial agreement)
- **Statewide mean**: $\rho \approx +0.65$ — the highest concordance the pipeline has produced.

### 2. Benchmark Band — ⚠️ TANK/COLLECTOR CALIBRATION DIVERGES FROM RAJASTHAN

- **Solar Fraction Range**: **30%–53%** across all simulations. **0 of 42 simulations land in the
  published 54–84% benchmark band.** The script itself prints the warning: a systematically off
  solar fraction usually traces to the tank/collector assumptions (`M_W_KG`, `A_C_M2`,
  `COLLECTOR_EFF`, draw schedule), not the PCM choice.
- **Rajasthan's Phase 7 sits at SF ≈ 63–66%, 100% in-band.** Tamil Nadu's `10_physics_validation.py`
  is a **standalone model** (`M_W_KG = 150`, `DRAW_MASS_KG = 75 × 2 = 150 kg/day`); Rajasthan's
  uses the shared `physics_lib.py` (`M_W_KG = 300`, `DRAW_TOTAL_KG_PER_DAY = 300`). Until these two
  Phase 7 models are reconciled and Tamil Nadu is re-calibrated into the benchmark band, **the
  positive $\rho$ above and Rajasthan's $\rho$ are not directly comparable** — this is the next
  unification (Phase 7), separate from the Phase 5/6 unification.
- **Annual cycling**: 144–329 complete cycles/year (rank-1 picks 167/144/206) — realistic daily
  charge/discharge.

---

## Status
**Ran against the unified Phase 5/6 engine 2026-09-08; fresh run pending** (Phase 6 supercooling
cap had a residual bug fixed after). **Phase 7 itself is NOT yet unified** — Tamil Nadu uses a
standalone tank model, Rajasthan uses `physics_lib.py`; they disagree on tank mass and daily draw
and produce SF in different bands (TN 30–53%, RJ 63–66%). Reconciling them is the next unification
step. Re-run `08_mcdm_ranking.py` → `10_physics_validation.py` → `09_recommendation_cards.py`.

---

## Literature Support

| Component | Reference / Model | Source File |
|---|---|---|
| Lumped-Enthalpy Tank Model | Barqawi (2025) dynamic simulation | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
| Solar Fraction Benchmark (54–84%) | Singh et al. (2025) review | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Spearman Rank Validation | Framework doc §10 (N5 Novelty) | `13_LITERATURE_MAPPING.md` |
| Backward Euler Numerical Scheme | Ghodusinejad (2026) physics models | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
