# 05 — Phase 3 Audit: Climate Signature Construction

Script: `04b_climate_signature.py`, `04d_signature_interactive.py`.

## Purpose
Collapse each point's 10-year hourly/daily weather into a single climate signature vector, which defines the location's climatology and determines the PCM performance targets.

## Processing Details
1. **Tier 1 (Sun-Event Statistics)**:
   - Computes means and percentiles of sun-event temperatures, GHI, humidity, and wind.
   - Calculates `HSI` (Humidity-Stress Index, representing Thom's 1959 Discomfort/THI Index):
     `HSI = T_sunrise - 0.55 * (1 - RH_sunrise/100) * (T_sunrise - 14.5)`
2. **Tier 2 (Daily-Integral Merge)**:
   - Joins true daily integrals from `02b`: GHI, SAI, cloudy fraction, CCI, seasonality, HDD18, CDD24, DTR.
   - Sets canonical values: prefers true Tier-2 values from POWER, keeping Tier-1 proxies alongside.
3. **Derived targets**:
   - `Tm_target = T_delivery + ΔT_approach = 50.0 + 7.0 = 57.0°C` (constant indirect system target).
   - `L_required = (q_night_kw * 3600 * 7) / ASSUMED_PCM_MASS_KG`
4. **Five Interaction Terms**:
   - `int_GHI_x_ktstd = GHI_daily_kWh * kt_std`
   - `int_DTR_x_cloudyfrac = DTR * cloudy_frac`
   - `int_RH_x_TaMinusTm = RH_mean * (Ta_mean - Tm_target)`
   - `int_wind_x_TaMinusTsoil = wind_mean * (Ta_mean - Tsoil_proxy)`
   - `int_CCI_x_1minusSAI = CCI * (1 - SAI)`
5. **PCA Reduction**:
   - Standardizes and reduces `["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]` to **4 components** explaining >95% variance.
6. **Standardization**:
   - Excludes lat/lon, population, and raw PCA variables, z-scoring the rest to build the final clustering matrix.

## Critical Audit Findings (1000x Flow Rate Bug)
In the calculation of `L_required`:
```python
DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60   # evaluates to 0.001 kg/s
```
- This is a units/density confusion error. A draw rate of 60 L/min corresponds to **1.0 kg/s**. By dividing by 1000 and 60, the code converts Liters/minute to m³/s, but fails to multiply by the density of water (1000 kg/m³) to recover kg/s.
- This understates the water draw rate by **1000x** (resulting in a total night draw of only 25.2 kg of water over 7 hours, instead of the 25,200 kg of a 60 L/min draw, or the 300 kg of a standard domestic draw).
- As a result, the derived `L_required` target is only **51–54 kJ/kg** instead of a realistic magnitude. This low floor makes the downstream latent-heat screening filter completely ineffective.

## Status
**NEEDS CORRECTION** (Due to the 1000x flow-rate unit error).
