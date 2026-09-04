# 16 — Climate Signature: Feature-to-PCM-Property Mapping (Assam)

## Governing Design Principle (Framework Plan §6.1)

> "Every index must answer the question 'which PCM property does this constrain, and by what
> physical mechanism?'. If that sentence cannot be completed, the index is removed."

All 18 Assam signature indices across the **129 spatial coordinates** satisfy this criterion.

---

## Feature → Thermal Behavior → PCM Property Map

### Tier 1 — Sun-Event Statistics (ERA5 Hourly)

| Feature | Physical Mechanism | PCM Property Constrained |
|---|---|---|
| `Ta_mean` | Annual mean ambient temperature → baseline storage tank loss | $T_m^{\text{target}}$ sizing ($T_{\text{del}} - \Delta T_{\text{approach}}$) |
| `Ta_p95` | Extreme hot design-day temperature → maximum charging environment | Upper boundary of melting window; thermal safety |
| `Ta_p05` | Extreme cold design-day temperature → nocturnal discharge deficit | Storage loss coefficient; cold-side thermal cycling |
| `HDD18` | Heating degree days → seasonal domestic heating demand | Feeds PCA thermodynamic block; regime separation |
| `CDD24` | Cooling degree days → summer thermal demand baseline | Feeds PCA thermodynamic block; regime separation |
| `RH_mean` | Annual mean relative humidity → condensation and corrosion risk | Corrosion resistance requirement; container material |
| `GHI_daily_kWh` | Mean daily global horizontal irradiation | $L_{\text{required}}$ latent heat capacity sizing |
| `DTR` | Diurnal Temperature Range → daily thermal expansion/contraction | Thermal cycling durability ($\ge 300$ cycles) |
| `HSI` | Humidity-Solar Interaction ($RH_{\text{mean}} \times GHI_{\text{daily}}$) | **Corrosion-veto trigger for Assam.** Excludes inorganic PCMs when $HSI > \text{global } p_{75}$ |

### Tier 2 — Daily-Integral Indices (NASA POWER Daily Integrals)

| Feature | Physical Mechanism | PCM Property Constrained |
|---|---|---|
| `kt_mean` | Annual mean clearness index → solar resource reliability | Storage capacity scaling factor |
| `cloudy_frac` | Fraction of days with $k_t < 0.4$ → intermittent solar charging | Autonomy sizing; partial-cycle charging stress |
| `monsoon_index` | Fraction of annual rainfall occurring in Jun–Sep | Seasonal storage shortfall sizing |
| `CCI` | Cloud Cover Index → daily solar intermittency | Combined charging reliability assessment |
| `SAI` | Solar Availability Index → fraction of usable solar charging days | Latent-heat reserve margin |
| `precipitation_annual` | Total annual precipitation (mm/yr) | Macro-climate regime characterization |
| `Ta_min_true` | True annual minimum daily temperature (°C) | Melting window lower threshold boundary |
| `Ta_max_true` | True annual maximum daily temperature (°C) | Thermal degradation & safety threshold |
| `elev_proxy` | Surface atmospheric pressure elevation proxy | Ineichen clear-sky atmospheric modeling |

---

## Four Climate Representations in the Pipeline

1. **Raw 18-Index Signature**: Dimensions preserved in physical units across all 129 points (`climate_signatures_raw.csv`).
2. **Standardized Matrix**: Normalized to zero mean and unit variance across the 129 coordinates (`climate_signatures_matrix.csv`).
3. **PCA Thermodynamic Block**: Principal component reduction applied strictly to the 7 correlated thermodynamic indices (`Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `RH_mean`, `elev_proxy`). Solar and variability indices are held separate to preserve physical interpretability.
4. **Final Locked GMM Input Representation (5 Features)**:
   - To prevent full-covariance over-parameterization on $N=129$ points, the final locked Phase 3 GMM model clusters on **5 core physical features**:
     $$\{ GHI_{\text{mean}}, Ta_{\text{mean}}, DTR, RH_{\text{mean}}, wind_{\text{mean}} \}$$
   - This formulation captures solar energy, thermal baseline, diurnal cycling, monsoon humidity, and convective cooling, producing an unambiguous global BIC minimum at $K=3$ ($\text{BIC} = 1574.94$).

---

## Uniform Melting Target: $T_m^{\text{target}} = 44.0^\circ\text{C}$

Assam uses a uniform target across all 129 points:
- $T_{\text{delivery}} = 50.0^\circ\text{C}$ (Indian domestic SWH standard)
- $\Delta T_{\text{approach}} = 6.0\text{ K}$ (Heat exchanger approach)
- $T_m^{\text{target}} = 50.0 - 6.0 = \mathbf{44.0^\circ\text{C}}$

Because $T_{a,\text{mean}}$ varies moderately across Assam ($22.6^\circ\text{C}$ to $25.9^\circ\text{C}$ across regimes), the $44.0^\circ\text{C}$ target is applicable state-wide without requiring regional capping adjustments.
