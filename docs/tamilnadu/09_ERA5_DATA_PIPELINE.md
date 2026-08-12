# 09 — ERA5 Data Pipeline: Deep Audit (Deaccumulation)

## The Core Concept
Accumulated variables in ERA5 (`ssrd`, `strd`, `tp`) can be stored as running totals (MARS convention, reset at hours 1 and 13 UTC) or as pre-processed hourly values (CDS point-download convention).

## The Bug (v3.0 — now fixed)
In `02_combine_tamilnadu.py`, the old code used diff-based deaccumulation:
```python
# OLD (buggy):
def deaccumulate(s):
    diff = s.diff()
    reset_mask = s.index.hour.isin([1, 13])
    diff[reset_mask] = s[reset_mask]
    return diff.clip(lower=0)
```
Since CDS point downloads already contain hourly fluxes, diffing subtracted consecutive fluxes → near-zero GHI (noon r ≈ 0.40, MBE ≈ −232 W/m²).

## The Fix (v3.1 — applied)
```python
def accum_to_flux(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    return s.clip(lower=0)
```
Rajasthan reference: r increased from ≈0.01 to **0.8102** after this fix. Tamil Nadu expected similar improvement after re-run.

## Status
**CORRECTED in `02_combine_tamilnadu.py` — re-run required for updated `climate_tamilnadu_points.csv`**

## Literature Support
| Topic | Reference | Source |
|---|---|---|
| ERA5 accumulation conventions | ECMWF ERA5 documentation | Standard reanalysis practice |
| Reanalysis vs satellite GHI | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Cross-source validation thresholds | Mansouri et al. (2025) | `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
