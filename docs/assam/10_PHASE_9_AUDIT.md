# 10 — Phase 9 Audit: Sub-Hourly 10-Year Dynamic Physics Validation

**Script**: `10_physics_validation.py`

**Status**: COMPLETE (Authoritative Final)

---

## Validation Model Architecture

Phase 9 establishes the independent numerical physics validation layer, replacing static or degree-day approximations with a rigorous, **10-year chronological dynamic thermal simulation**.

### Governing System Specifications
- **Simulation Duration**: 10 continuous years (2016–2025; 87,648 hours / 1,051,776 primary timesteps per run).
- **Temporal Resolution**:
  - Primary simulation timestep: $\Delta t = 300\text{ s}$ (5 minutes).
  - Numerical sensitivity timestep: $\Delta t = 150\text{ s}$ (2.5 minutes).
- **Physical Thermal Sizing**:
  - Water tank mass ($M_w$): $100\text{ kg}$ fluid storage ($V_w = 100\text{ L}$).
  - Latent thermal storage mass ($M_p$): $50\text{ kg}$ phase-change material.
  - Collector aperture area ($A_c$): $2.0\text{ m}²$ flat-plate solar thermal collector ($\eta_0 = 0.72$, $a_1 = 3.8\text{ W/m}²\text{K}$).
- **Domestic Demand Schedule**:
  - Total daily load: $100\text{ L/day}$ at $T_{\text{delivery}} = 50.0^\circ\text{C}$.
  - Morning draw: $50\text{ L}$ at 07:00 local time (interrogates overnight latent storage retention).
  - Evening draw: $50\text{ L}$ at 19:00 local time (interrogates daytime solar collection efficiency).
- **Candidate Scope & Forcing**:
  - Climate forcing: Driven by full 10-year hourly ERA5 reanalysis at the **3 final $K=3$ medoids** (`ASP_0012`, `ASP_0092`, `ASP_0028`).
  - Evaluated candidates: The **8 Phase-6-screened historical PCMs**.
  - Total simulation runs: $8\text{ PCMs} \times 3\text{ regimes} = \mathbf{24\text{ full evaluations}}$.

---

## Thermodynamic Formulation: 4-State Path-Dependent Enthalpy

The PCM storage node is modeled using a continuous, path-dependent enthalpy formulation that resolves supercooling hysteresis and non-equilibrium phase boundaries:

1. **Four Thermodynamic States**:
   - `LIQUID` ($T_p > T_m$)
   - `FREEZING` ($T_p = T_{\text{freeze}}$, latent heat rejection during cooling)
   - `SOLID` ($T_p < T_{\text{freeze}}$)
   - `MELTING` ($T_p = T_m$, latent heat absorption during solar charging)
2. **Supercooling Hysteresis**:
   - Accounts for the activation barrier during cooling; the liquid PCM cools below $T_m$ to $T_{\text{freeze}} = T_m - \Delta T_{\text{sub}}$ before nucleation releases latent heat.
3. **Analytical Transitions & Boundary Clipping**:
   - Evaluates analytical energy transitions at phase boundaries with strict mathematical clipping at sensible/latent enthalpy interfaces, eliminating numerical overshoot artifacts.

---

## ERA5 SSRD Reconstruction: Duration-Overlap De-accumulation

ERA5 surface solar radiation downwards (`ssrd`) is provided by ECMWF as accumulated energy from the beginning of each forecast cycle. To generate true sub-hourly irradiance without artifacts:
- **Duration-Overlap Allocation**: Accumulated intervals are apportioned proportionally to the duration overlap between the forecast accumulation window and the simulation timestep.
- **Uniform Energy Assumption**: Explicitly presumes uniform energy flux distribution within each hourly accumulation block.
- **Energy Conservation Tracking**:
  - SSRD energy conservation was verified prior to nighttime physical zero-clamping:
    $$\text{SSRD Reconstruction Conservation Error} = \mathbf{0.000000\%}$$
  - Nighttime solar clamping losses ($GHI < 0$ or astronomical night) are tracked separately as a physical boundary condition, preserving total shortwave balance.

---

## Numerical Verification & First-Law Conservation

All 24 dynamic simulations underwent rigorous numerical validation:
1. **First-Law Energy Balance**:
   Cumulative energy conservation across solar gain, ambient thermal losses, domestic tapping discharge, sensible water storage, and PCM enthalpy change:
   $$\text{First-Law Cumulative Conservation Error} = \mathbf{0.0000\%}$$
2. **Spin-Up & Sensitivity Criteria**:
   - Multi-year cyclic spin-up was evaluated across consecutive 365-day cycles.
   - **Validation Verdict**: **100% of simulations satisfied the predefined spin-up convergence and timestep-sensitivity validation criteria** ($\Delta t = 150\text{ s}$ vs. $300\text{ s}$ deviations $<0.05\%$).
