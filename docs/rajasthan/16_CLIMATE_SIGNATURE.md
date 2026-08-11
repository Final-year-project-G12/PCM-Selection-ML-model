# 16 — Climate Signature: Feature-to-PCM-Property Mapping

## The governing design principle (framework doc §6.1, verbatim)

> "Every index must answer the question 'which PCM property does this constrain, and by what
> physical mechanism?'. If that sentence cannot be completed, the index is removed."

This section maps each retained feature to that justification explicitly — see `05_PHASE_3_AUDIT.md`
for the exact formulas. (This mapping itself is unaffected by the 2026-08-11 changes elsewhere in the
pipeline — see `05_PHASE_3_AUDIT.md` for the two new signature-level QC plots and the corrected
Phase 2.5 input file.)

## Feature → thermal behavior → PCM requirement → PCM property map

```
T_sunrise_mean, RH_sunrise_mean  →  HSI_sunrise (Thom's THI)  →  pre-dawn condensation/corrosion
    risk at the store surface   →  corrosion resistance requirement (feeds Phase 5's corrosion veto)

T_noon_mean, GHI_noon_mean, kt_noon_mean/std  →  charging-window heat availability and reliability
    →  required melting-window achievability, charging feasibility  →  Tm_target_capped_C (Phase 5
    constraint 6)

T_sunset_mean, wind_sunset_mean  →  evening heat-loss potential during discharge onset  →
    interaction term int_wind_x_TsunsetMinusTdelivery  →  discharge-window thermal-loss sensitivity

diurnal_gradient (Tier 1, noon−sunrise) + DTR_true (Tier 2, true daily max−min)  →  daily thermal
    swing magnitude  →  cycling stress on the PCM  →  cycling-stability requirement (Phase 5
    constraint 4, cycles≥300)

GHI_daily_kWh, kt_daily_mean/std, SAI, CCI  →  total charging energy and its day-to-day reliability
    →  latent-heat sizing and autonomy requirement  →  L_required_kJ_per_kg (Phase 5 constraint 3)
    and CCI×(1−SAI) interaction term

HDD18, CDD24  →  heating/cooling demand-day counts (standard degree-day indices, base 18°C/24°C)  →
    seasonal thermal-load context  →  feeds the PCA temperature block (Phase 3), indirectly informs
    regime characterization (Phase 4)

cloudy_frac, seasonality, monsoon_index  →  charging intermittency and seasonal charging-resource
    variability  →  int_DTR_x_cloudyfrac interaction term, int_GHI_x_ktstd interaction term  →
    cycling-stress-under-intermittent-charging proxy

elevation_m  →  (via PCA block only, not standalone)  →  atmospheric/airmass context already baked
    into the pvlib solar-geometry computation upstream  →  indirectly informs regime separation
    (a documented "altitude" PCA component is expected per the framework doc)

daylength_mean, daylength_amplitude  →  seasonal charging-window-length variation  →  charging
    duration context  →  Level-B ablation candidate (flagged as possibly climatically-tautological
    since daylength is deterministic-by-construction from latitude/day-of-year, not a weather outcome)
```

## HSI_sunrise — a named-but-not-original index

Presented in the pipeline's own naming as a "humidity stress index," `HSI_sunrise` is, formula for
formula, **Thom's (1959) Temperature-Humidity Index (THI)** applied to sunrise-mean T/RH:
```
HSI_sunrise = T_sunrise_mean − 0.55·(1 − RH_sunrise_mean/100)·(T_sunrise_mean − 14.5)
```
This should be cited as Thom's THI in any write-up — presenting it as an original derivation would
overstate its novelty. The physical reinterpretation (as a condensation/corrosion-risk proxy at the
coldest instant of the day, rather than THI's original human-comfort framing) is a legitimate,
correctly-reasoned repurposing, but the formula itself is not new.

## Two-tier design — why both tiers are kept, not collapsed into one

Tier 1 (sun-event instantaneous statistics) captures conditions at the physically meaningful
charge/discharge instants but is a sparse 3-samples/day estimator, systematically underestimating
true diurnal range (`diurnal_gradient` vs. `DTR_true`). Tier 2 (true daily integrals from full
hourly NASA-POWER data) captures accurate daily energy totals and variability but cannot be computed
from ERA5 alone within this pipeline's download scope (would require a full 24h/day ERA5 request,
explicitly out of scope). Keeping both, rather than picking one, is the correct choice given each
one's distinct blind spot — and is explicitly justified in-code, not an unexamined default.

## Interaction terms — physical justification restated

Each of the 5 interaction terms (exact formulas in `05_PHASE_3_AUDIT.md`) targets a specific
compound risk that no single index captures alone: resource magnitude × unreliability (charging
risk), thermal swing × cloudiness (cycling stress under intermittency), humidity × approach-to-target
(condensation risk), wind × approach-to-delivery (evening convective loss), and autonomy-runs ×
inverse-steadiness (combined autonomy requirement). All five are named and justified in code
comments, consistent with the "answer which PCM property, by what mechanism" design principle.

## PCA — what it is actually compressing

Only the temperature/elevation block (`Ta_mean, Ta_p95, Ta_p05, T_sunrise_mean, T_noon_mean, HDD18,
CDD24, elevation_m`) goes through PCA — deliberately excluding solar-variability, humidity, and
cycling-relevant indices, which the framework doc identifies as carrying the actual discriminating
signal for regime separation and which must not be compressed away. 4 components retained (95%
variance) for Rajasthan. This is a correctly-scoped dimensionality reduction — it removes redundancy
specifically within the temperature-correlated block, not across the whole feature set.

## What the correlation heatmap actually validates

The `|r|>0.9` flagging pass operates on the **final** feature set (post-PCA-block-removal) — any
flagged pair represents genuinely new collinearity the PCA step did not already absorb, not a
re-discovery of the same temperature collinearity PCA was built to handle. This is the correct scope
for the check (validating that PCA didn't leave a *different* collinearity problem unaddressed), and
it is correctly distinguished in-code from "collinearity PCA already handled."

## Literature support

Thom (1959) for HSI_sunrise/THI. Avargani et al. (2021) and Durin et al. (2018) for the
PCM-facing-quantity derivations (see `05_PHASE_3_AUDIT.md`). No dedicated citation exists for the
specific interaction-term formulas or the PCA-block scoping choice — these read as original,
correctly-reasoned engineering design within this project, appropriately presented as such (not
over-cited to a source that doesn't specifically support them).
