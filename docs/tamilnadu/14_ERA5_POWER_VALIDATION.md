# 14 — ERA5 vs NASA POWER Validation

Scripts: `03_plots_raw.py`, `03b_agreement_analysis.py`, `03b_interactive_raw_qa.py`.

## Pre-Fix Verification Statistics (reference only)
Cross-source agreement on 1,457,547 matched events **before v3.1 deaccumulation fix**:

| Variable | n | MBE (ERA5 − POWER) | RMSE | Pearson r | Status |
|---|---|---|---|---|---|
| **GHI (W/m²)** | 1,457,547 | **−231.89 W/m²** | **404.69 W/m²** | **0.3963** | Bug active (pre-fix) |
| **Clear-sky GHI (W/m²)** | 1,457,547 | −7.04 W/m² | 53.57 W/m² | 0.9947 | Good |
| **T_amb (°C)** | 1,457,547 | +1.08°C | 2.78°C | 0.8454 | Moderate |
| **RHum (%)** | 1,457,547 | −2.93% | 12.52% | 0.8192 | Moderate |
| **Wind speed (m/s)** | 1,457,547 | −1.14 m/s | 1.67 m/s | 0.7332 | Moderate |

## Post-Fix Expected Improvement
After `accum_to_flux()` + Step 2b quantile mapping:
- Noon GHI r expected > 0.80 (Rajasthan reference: 0.8102 after fix alone).
- MBE expected within ±5% of mean POWER GHI after quantile mapping.
- Run `03b_agreement_analysis.py` after re-combine to verify.

## Analysis
- Clear-sky GHI matched perfectly (r = 0.9947) because it is pvlib-derived, not from buggy ERA5 radiation.
- Temperature, humidity, wind show moderate agreement — typical for reanalysis vs satellite products.

## Status
**Validation scripts ready — re-run `02_combine` → `03b` → `04` for post-fix stats**

## Literature Support
| Topic | Reference | Source |
|---|---|---|
| GHI validation metrics (MBE, RMSE, r) | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Quantile mapping decision gate | Mansouri et al. (2025) | `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
| NASA POWER as reference | NASA POWER documentation | `02_DATA_SOURCES_AND_VARIABLES.md` |
