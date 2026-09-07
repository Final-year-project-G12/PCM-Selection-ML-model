# 05 — Phase 3 Audit: Climate Signature Construction

Scripts: `04b_climate_signature.py`, `04d_signature_interactive.py`.

## Purpose
Collapse each point's 10-year hourly and daily weather data into a single, physically grounded climate signature vector. This vector defines the climatology of each location, maps meteorological stress directly to PCM performance requirements, and computes climate-adaptive PCM thermal targets (`Tm_target`, `L_required`).

---

## Processing Details

### 1. Two-Tier Signature Design
- **Tier 1 (Sun-Event Instantaneous Statistics)**: Means, standard deviations, and 5th/95th percentiles of sunrise, solar noon, and sunset ambient temperature ($T_{\text{amb}}$), GHI, relative humidity ($\text{RHum}$), wind speed ($W_{\text{spd}}$), and Thom's (1959) Discomfort Index (HSI).
- **Tier 2 (Daily-Integral Merge)**: True daily integrals computed from full hourly NASA POWER series in `02b_build_daily_aggregates.py`: GHI ($\text{kWh/m}^2/\text{day}$), Solar Anomaly Index (SAI), cloudy fraction, Cloud Continuity Index (CCI), degree days (HDD18, CDD24), and Diurnal Temperature Range (DTR).

### 2. Climate Feature → PCM Property Mapping (formerly `16_CLIMATE_SIGNATURE.md`)
Governing design principle: *"Every climate feature must answer which PCM property it constrains and by what physical mechanism."*

| Climate Feature | SWH Thermal Behavior | PCM Target / Constraint | PCM Property Impacted |
|---|---|---|---|
| **$T_{a,\text{mean}}$** (mean air temp) | Governs baseline hot water heat loss to ambient environment. | Derived target $T_{m,\text{target}}$ and $L_{\text{required}}$. | Selection of optimal melting point. |
| **DTR** (Diurnal Temp Range) | Larger range implies lower night temperatures and higher cooling loads. | Determines thermal demand swing. | Volumetric latent heat capacity ($\rho H$). |
| **$\text{GHI}_{\text{daily\_kWh}}$** | Controls total solar thermal energy available for collector charging. | Governs sizing and autonomy requirements. | Latent heat storage capacity ($L$). |
| **$\text{cloudy\_frac}$** | High fraction implies frequent consecutive low-radiation days. | Restricts charging window and capacity. | Latent heat capacity and thermal conductivity. |
| **$\text{RH}_{\text{mean}}$** | High humidity increases convective and condensation losses. | Corrosion veto trigger. | Material compatibility & encapsulation. |
| **$\text{wind}_{\text{mean}}$** | Strong winds cause convective losses from collector glass cover. | Autonomy margin. | Latent heat storage margin. |
| **$\text{monsoon\_index}$** | Concentrated monsoon rainfall reduces solar fractions for specific months. | Seasonal PCM suitability. | Melting point window and target width. |

### 3. Five Compound Interaction Terms
- `int_GHI_x_ktstd`: Flags erratic solar resources where daily integrals are large but highly variable.
- `int_DTR_x_cloudyfrac`: Captures thermal cycling stress under high weather intermittency.
- `int_RH_x_TaMinusTm`: Measures ambient condensation risk on cold storage boundaries.
- `int_wind_x_TaMinusTsoil`: Estimates evening heat loss from tank to surroundings.
- `int_CCI_x_1minusSAI`: Quantifies combined cloudy autonomy requirement.

### 4. Derived Targets & Sizing Methodology (v3.1 & 2026-08-31 Correction)
- **Melting Point Target**: $T_{m,\text{target}} = T_{\text{delivery}} + \Delta T_{\text{margin}} = 50.0 + 7.0 = 57.0^\circ\text{C}$.
- **Latent Heat Target ($L_{\text{required}}$)**:
  $$L_{\text{required}} = \frac{\text{SHARE\_PCM} \times m_{\text{water}} \times c_{p,\text{water}} \times \Delta T}{m_{\text{PCM\_assumed}}}$$
  - $V_{\text{draw}} = 300\text{ L/day}$ ($m_{\text{water}} = 300\text{ kg}$, Avargani et al. 2021 domestic baseline).
  - $c_{p,\text{water}} = 4.186\text{ kJ/(kg}\cdot\text{K)}$.
  - $\Delta T = T_{\text{delivery}} - T_{\text{mains\_est}}$.
  - $m_{\text{PCM\_assumed}} = 50.0\text{ kg}$.
  - **`SHARE_PCM = 0.5`** (defined in `config.py`): Literature-anchored fractional-share model (Zhao 2022, Huang 2020) where PCM supplies 50% of delivery energy while tank sensible heat and concurrent charging supply the remainder.
  - *Completed Run Targets*: Cluster $L_{\text{required}}$ values are approximately **301–326 kJ/kg** (read from `data/processed/processed/signatures/`).

### 5. Dimensionality Reduction & Normalization
- **PCA Reduction**: Performs PCA on the temperature/climate block, retaining 4 principal components capturing $>95\%$ variance.
- **z-Score Normalization**: Standardizes features for GMM clustering input.

---

## Status
**Analysis COMPLETE (62-PCM run)** — Re-run `04b_climate_signature.py` (after `02b`) to regenerate `climate_signature_tamilnadu.csv` in the canonical location.

---

## Literature Support

| Component | Reference / Method | Source File |
|---|---|---|
| 300 L/day Domestic Draw | Avargani et al. (2021) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Fractional PCM Share (0.5) | Zhao (2022); Huang (2020); Abdelsalam (2020) | `13_LITERATURE_MAPPING.md` |
| Discomfort Index (HSI) | Thom (1959) Discomfort Index | Standard meteorological literature |
| Feature-to-Property Mapping | Liu et al. (2025); Singh et al. (2025) Table 2 | `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
