# 19 — Climate Signature: Feature-to-PCM-Property Mapping (Assam)

## Governing design principle (framework doc §6.1)

> "Every index must answer the question 'which PCM property does this constrain, and by what
> physical mechanism?'. If that sentence cannot be completed, the index is removed."

All 18 Assam signature indices satisfy this criterion. This file maps them explicitly.

## Feature → thermal behavior → PCM property map

### Tier 1 — Sun-event statistics

| Feature | Physical mechanism | PCM property constrained |
|---|---|---|
| `Ta_mean` | Annual mean ambient → average thermal load on tank | Tm_target sizing (delivery T − approach ΔT) |
| `Ta_p95` | Design-day peak temperature → maximum charging environment | Upper end of melting window; safety in extreme heat |
| `Ta_p05` | Design-day cold extreme → night-discharge environment | Thermal loss coefficient sensitivity; cycling stress at low T |
| `HDD18` | Heating degree days → seasonal demand context | Feeds PCA thermodynamic block; regime separation |
| `CDD24` | Cooling degree days → summer demand context | Feeds PCA thermodynamic block; regime separation |
| `RH_mean` | Annual mean relative humidity → condensation risk at PCM container | Corrosion resistance requirement; **load-bearing for Assam** (HSI > p75 triggers corrosion veto in Phase 5) |
| `GHI_daily_kWh` | Mean daily charging energy available | L_required sizing (latent heat floor) |
| `DTR` | Diurnal temperature range → daily thermal cycling magnitude | Cycling-stability requirement (cycles ≥ 300) |
| `HSI` | Humidity-Solar Interaction (RH × GHI) → combined condensation+charge-rate signal | **Primary corrosion-veto trigger for Assam.** High HSI = humid + solar → condensation risk during charge cycles. Clusters where HSI > global p75 get inorganic PCM exclusion in Phase 5. |

### Tier 2 — Daily-integral indices (from NASA POWER + `02b`)

| Feature | Physical mechanism | PCM property constrained |
|---|---|---|
| `kt_mean` | Annual mean clearness index → solar resource quality | L_required scaling; charging reliability |
| `cloudy_frac` | Fraction of days with kt < 0.4 → charging intermittency | Cycling-stress-under-intermittency; autonomy sizing |
| `monsoon_index` | Fraction of precipitation in Jun–Sep → seasonal charging gap | PCM system sizing for monsoon-period under-charging |
| `CCI` | Cloud Cover Index → daily variability measure | Combined with SAI for charging reliability assessment |
| `SAI` | Solar Availability Index → fraction of usable solar days | Latent-heat margin requirement |
| `precipitation_annual` | Annual total precipitation → climate character | Feeds regime separation; corrosion proxy |
| `Ta_min_true` | True annual minimum → cold-side design extreme | Phase 5 melting-window lower bound |
| `Ta_max_true` | True annual maximum → hot-side design extreme | Phase 5 melting-window upper bound |
| `elev_proxy` | Elevation proxy (pressure-derived) → atmospheric context | PCA thermodynamic block; Ineichen model input |

## HSI as load-bearing corrosion signal — Assam distinction

In Rajasthan (dry climate), `HSI` was present in the signature but did not activate the corrosion
veto for most clusters — HSI values were below the global p75 threshold. In Assam, the combination
of high `RH_mean` (~75–85% in valley clusters) and moderate `GHI_daily_kWh` produces `HSI` values
that **do exceed the global p75 threshold** in humid valley clusters (0 and 1), triggering the
inorganic PCM corrosion veto in Phase 5.

This is a real climate-discriminating result: the same 18-index signature, same HSI formula,
same Phase 5 veto rule — but the climate data drives a **different practical outcome** in Assam vs
Rajasthan. This is exactly the kind of climate-adaptive differentiation the framework claims (N2,
N3) and Assam's implementation delivers it.

## Two-tier design rationale

Tier 1 (sun-event statistics) captures conditions at physically meaningful charge/discharge instants
but underestimates true diurnal range. Tier 2 (true daily integrals from NASA POWER 24h data)
captures accurate energy totals and variability. Keeping both avoids each one's blind spot.

## PCA scope: thermodynamic block only

PCA applied to: `Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `RH_mean`, `elev_proxy`
(7 features). Solar block (GHI, kt, SAI, CCI) and variability block (DTR, cloudy_frac,
monsoon_index, HSI, precipitation_annual) are kept out of PCA — they carry the actual discriminating
signal for regime separation and PCM target derivation. Compressing them would be a Type 2 mistake
(reducing the signal most needed for the downstream PCM recommendation).

## Uniform Tm_target = 44°C for Assam

Unlike Rajasthan (where Tm_target was regime-specific with a worst-month cap), Assam uses a uniform
44°C for all 128 points:
- T_delivery = 50°C (Indian domestic standard, from framework doc Table 8)
- ΔT_approach = 6°C (approach temperature for tank-coil heat exchanger)
- Tm_target = 50 − 6 = **44°C**, same for all Assam regimes

This is appropriate because Assam's Ta_mean range (26.3–28.2°C across clusters) is narrow enough
that the worst-month capping mechanism (which for Rajasthan reduced Tm_target in cooler northern
clusters) does not produce materially different values between Assam's clusters. Documenting this
explicitly avoids the inference that Tm_target was arbitrarily fixed.

## T_mains_est_C approximation

`T_mains_est_C = Ta_mean − 2.0` (same formula as Rajasthan and Tamil Nadu). This offset is
**unsourced** in-code — no published correlation for mains-water temperature to ambient air
temperature was found. This directly feeds `L_required_kJ_per_kg` and is a persistent caveat
across all four states. A proper citation (e.g., from ASHRAE, CIBSE, or a published India-specific
pipe temperature study) would strengthen the methodology.

## Literature support

Thom (1959) for HSI formula (Temperature-Humidity Index, repurposed as corrosion-risk proxy).
Avargani et al. (2021) for `L_required_kJ_per_kg` night-discharge energy sizing basis.
Framework doc §6.1 for the governing "answer which PCM property" design principle.
