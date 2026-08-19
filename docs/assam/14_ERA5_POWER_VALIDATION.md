# 17 — ERA5 vs NASA POWER Validation (Assam)

## Status: NOT IMPLEMENTED

Unlike Rajasthan (which had `03b_agreement_analysis.py` producing a documented bias-correction
decision), **Assam has no dedicated ERA5-POWER agreement analysis script.** There is no
`bias_decision_assam.txt`. No quantile mapping was computed or applied for Assam.

Phase 3 onward consumes ERA5 GHI directly from `climate_assam_points.csv` without a documented
cross-source bias-correction step.

## What is known (from the Rajasthan validation)

For context, Rajasthan's Phase 2 agreement analysis found:
- **Solar noon overall**: MBE = 10.95 W/m², RMSE = 113.8 W/m², Pearson r = 0.810
- **Decision**: QUANTILE_MAP (not BACKBONE — ERA5 had systematic seasonal bias relative to POWER)
- **Before/after**: RMSE improved in all 4 seasons after quantile-map correction

The same `accum_to_flux()` fix was applied to Assam from the start (correct), so Assam's ERA5 GHI
should be in the same corrected ballpark. However, the degree to which ERA5 and POWER agree for
Assam's specific climate (heavy monsoon, high aerosol loading in pre-monsoon) is **unknown without
running the analysis**.

## Why this matters for Assam specifically

Assam's climate has characteristics that may make ERA5-POWER disagreement **larger** than Rajasthan:
1. **Monsoon cloud cover**: Heavy optically-thick cloud in Jun–Sep may cause larger ERA5/POWER
   discrepancies because cloud-radiation parameterization differs between the two models
2. **Pre-monsoon aerosol loading**: Biomass burning in Mar–May creates aerosol plumes not well
   captured by ERA5's default aerosol climatology
3. **High humidity**: Atmospheric water vapor absorption is a significant factor; ERA5's Linke
   turbidity approximation may diverge from POWER's satellite-based estimate

Given that `kt_mean = 0.696–0.789` for Assam clusters (compared to Rajasthan's ~0.85+), any
GHI bias in the cloudy-day regime could meaningfully affect the `kt_mean`, `cloudy_frac`, and
`monsoon_index` signature indices that are central to cluster separation.

## Recommended action

Before treating Assam's Phase 3+ results as having the same rigor as Rajasthan's:

1. **Create `03b_agreement_analysis_assam.py`**: replicate Rajasthan's three-branch decision logic
   (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW) for Assam's 128 points.
2. **Run the agreement analysis** and record the decision in `bias_decision_assam.txt`.
3. **If QUANTILE_MAP**: apply the fitted correction upstream of Phase 3 (unlike Rajasthan, where
   the correction was computed but not applied back).
4. **Compute the same stratified MBE/RMSE/r statistics** (by season × event) for thesis reporting.

## What can be stated in the thesis now

"Assam's ERA5 GHI was processed using the corrected `accum_to_flux()` function validated against
NASA POWER during the Rajasthan phase. A formal cross-source agreement analysis specific to Assam
has not yet been completed; Phase 3+ results should be treated as provisional pending that
analysis." — This is the correct, honest framing.
