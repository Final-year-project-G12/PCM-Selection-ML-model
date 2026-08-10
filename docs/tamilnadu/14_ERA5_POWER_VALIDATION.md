# 14 — ERA5 vs NASA POWER Validation

Script: `03_plots_raw.py`, `03b_interactive_raw_qa.py`.

## Verification Statistics (Tamil Nadu Raw Data)
Cross-source agreement statistics are evaluated on `1,457,547` matched events:

| Variable | n | MBE (ERA5 - POWER) | RMSE | Pearson r | Status |
|---|---|---|---|---|---|
| **GHI (W/m²)** | 1,457,547 | **-231.89 W/m²** | **404.69 W/m²** | **0.3963** | **UNACCEPTABLE (Bug Active)** |
| **Clear-sky GHI (W/m²)** | 1,457,547 | -7.04 W/m² | 53.57 W/m² | 0.9947 | Good |
| **T_amb (°C)** | 1,457,547 | +1.08°C | 2.78°C | 0.8454 | Moderate |
| **RHum (%)** | 1,457,547 | -2.93% | 12.52% | 0.8192 | Moderate |
| **Wind speed (m/s)** | 1,457,547 | -1.14 m/s | 1.67 m/s | 0.7332 | Moderate |

## Analysis of GHI Disagreement
- The low Pearson correlation (**r = 0.396**) and extreme negative bias (**MBE = -231.89 W/m²**) for GHI are direct symptoms of the deaccumulation bug in `02_combine_tamilnadu.py`.
- The diff-based deaccumulation subtracts consecutive hourly fluxes, driving the GHI to near-zero.
- Clear-sky GHI matches almost perfectly (r = 0.9947, MBE = -7.04 W/m²) because it is derived mathematically from coordinates/timestamps using `pvlib`, which does not use the buggy reanalysis radiation data.
- Wind, temperature, and humidity show moderate agreement, consistent with typical differences between reanalysis grid averages and satellite models.

## Impact
- Because the deaccumulation bug is active, the raw ERA5 GHI is corrupted.
- In Phase 3, this corrupted GHI was used to compute the GMM clustering feature `GHI_mean_z`.
