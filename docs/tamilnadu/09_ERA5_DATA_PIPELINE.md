# 09 — ERA5 Data Pipeline: Deep Audit (Deaccumulation)

## The Core Concept
Accumulated variables in ERA5 (like surface solar radiation downwards `ssrd`, total precipitation `tp`) are stored as running totals.
- **MARS/Classic Convention**: Accumulations reset to 0 at hours 1 and 13 UTC (forecast start times). Diffing consecutive hours recovers the hourly flux, with the raw value used at hours 1 and 13.
- **CDS API Point Request Convention**: When downloading point NetCDF files via the CDS API, the server pre-processes the data, and the downloaded variable is already an hourly flux (not cumulative since reset).

## The Bug in Tamil Nadu
In `02_combine_tamilnadu.py`, the code implements the classic deaccumulation logic:
```python
def deaccumulate(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    diff = s.diff()
    reset_mask = s.index.hour.isin([1, 13])
    diff[reset_mask] = s[reset_mask]
    return diff.clip(lower=0)
```
- Since the downloaded files for Tamil Nadu already contain hourly fluxes (not running totals), this diffing step subtracts the current hour's flux from the previous hour's flux.
- This results in near-zero values for solar radiation (`era5_GHI`, `era5_LW_down`) and precipitation.
- The average noon GHI from this buggy calculation is only **43–59 W/m²** (should be 700–900 W/m²).
- This is a critical data-quality issue that went undetected because the true GHI in the climate signature fell back to NASA POWER (`GHI_daily_kWh_mean`), which is correct. However, `GHI_mean` (noon GHI) was clustered using the buggy ERA5 value.

## The Fix
Replace `deaccumulate()` with a stateless clip:
```python
def accum_to_flux(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    return s.clip(lower=0)
```
In Rajasthan, this fix resolved the noon GHI correlation with NASA POWER (r increased from **r ≈ 0.01** to **r = 0.8102**). This same fix must be applied to Tamil Nadu.
