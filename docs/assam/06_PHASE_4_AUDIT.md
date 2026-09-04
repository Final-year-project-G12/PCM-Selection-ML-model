# 06 — Phase 4 Audit: Solar Water Heating (SWH) Design Specification

**Script**: `05_cluster_assam.py`

**Status**: COMPLETE (Authoritative Final)

---

## SWH System Sizing & Operational Requirements

Phase 4 defines the engineering design parameters and operational boundary conditions for the climate-adaptive solar water heating (SWH) system with latent thermal energy storage (LTES).

The downstream physics validation and screening criteria are parameterized as follows:

| System Parameter | Specification / Value | Engineering Justification |
|---|---|---|
| **Target Delivery Temperature ($T_{\text{delivery}}$)** | **50.0 °C** | Standard Indian domestic hot water specification (BIS / MNRE guidelines) |
| **Approach Temperature Difference ($\Delta T_{\text{approach}}$)** | **6.0 K** | Effectiveness of internal tank-to-PCM coil heat exchanger |
| **PCM Target Melting Temperature ($T_m^{\text{target}}$)** | **44.0 °C** | Derived directly: $T_m^{\text{target}} = T_{\text{delivery}} - \Delta T_{\text{approach}} = 50.0 - 6.0 = 44.0^\circ\text{C}$ |
| **Daily Domestic Demand** | **100 L/day** | Baseline domestic hot water consumption for a standard household |
| **Morning Consumption Draw** | **50 L at 07:00 local** | Peak morning usage (showering/domestic use), testing overnight storage retention |
| **Evening Consumption Draw** | **50 L at 19:00 local** | Peak evening domestic usage, testing diurnal daytime solar charging |
| **Tank Water Mass ($M_w$)** | **100 kg** | Coupled fluid sensible storage medium ($V_w = 100\text{ L}$) |
| **PCM Mass ($M_p$)** | **50 kg** | Latent heat thermal energy storage buffer mass |
| **Collector Surface Area** | **2.0 m²** | Standard flat-plate solar collector sizing for 100 L systems |

---

## Final Climate Forcing: Phase 3 Locked $K=3$ GMM Model

The thermal energy demand for the SWH system is driven by the climate forcing established in **Phase 3**. The final locked climate model is a **$K=3$ Gaussian Mixture Model (full covariance)** trained on 5 core physical features (`GHI_mean`, `Ta_mean`, `DTR`, `RH_mean`, `wind_mean`).

### Authoritative $K=3$ Regime Profiles (`table04_cluster_profiles_k3.csv`)

| Regime ID | Spatial Points | Pct of Sites | Covered Population | Annual Mean GHI | Est. Daily GHI | Mean Ambient $T_a$ | Mean DTR | Mean RH | Medoid Station ID |
|---|---|---|---|---|---|---|---|---|---|
| **Cluster 0** | 33 | 25.6% | 4,757,890 | 348.4 W/m² | 3.95 kWh/m²/day | 25.89 °C | 6.34 K | 75.84% | **`ASP_0012`** |
| **Cluster 1** | 61 | 47.3% | 4,271,199 | 373.0 W/m² | 4.08 kWh/m²/day | 25.10 °C | 6.27 K | 78.99% | **`ASP_0092`** |
| **Cluster 2** | 35 | 27.1% | 2,466,324 | 330.3 W/m² | 3.68 kWh/m²/day | 22.59 °C | 6.24 K | 77.86% | **`ASP_0028`** |

*Important Interpretation*: These three regimes represent **climate-feature similarity groupings**, reflecting variations in solar insolation, thermal baseline, and monsoon humidity across Assam. They are **not necessarily contiguous geographic territories**, but rather macro-climatic operating environments.

---

## Regime-Specific Thermal Requirements

Because the domestic delivery temperature ($50.0^\circ\text{C}$) and approach temperature ($6.0\text{ K}$) are uniform across Assam, $T_m^{\text{target}} = 44.0^\circ\text{C}$ applies state-wide. However, differences in ambient thermal baselines ($T_{a,\text{mean}}$) and estimated mains water temperature ($T_{\text{mains}} \approx T_{a,\text{mean}} - 2.0\text{ K}$) yield regime-specific night-discharge thermal deficits:

$$Q_{\text{deficit}} = M_w \cdot C_{p,w} \cdot (T_{\text{delivery}} - T_{\text{mains}})$$

$$L_{\text{required}} = \frac{Q_{\text{deficit}}}{M_p}$$

- **Historical $K=4$ baseline values**: In the historical 4-cluster screening, $L_{\text{required}}$ ranged from $232.3\text{ kJ/kg}$ (warmer southern fringe) to $248.9\text{ kJ/kg}$ (cooler hill/transition zone).
- **Final $K=3$ baseline values**: Under the final 3-regime climate forcing, Cluster 2 (cooler highland ambient baseline, $T_a = 22.59^\circ\text{C}$) establishes the highest latent heat demand ($L_{\text{required}} \approx 252\text{ kJ/kg}$), whereas Cluster 0 and Cluster 1 require approximately $230–240\text{ kJ/kg}$.
