# 15 — Quality Control Rules and Gates

## Quality Control Thresholds
The 13-step preprocessing script (`04_preprocess_tamilnadu.py`) applies the following physical range gates (`BOUNDS`):

- `era5_GHI`: `[0, 1400]` W/m² (values outside NaN'd, night-masked to 0.0 when SZA >= 90).
- `era5_T_amb`: `[-30, 55]`°C (values outside NaN'd).
- `era5_RHum`: `[0, 100]`% (clipped).
- `era5_W_spd`: `[0, 50]` m/s (values outside NaN'd).
- `era5_P_atm`: `[850, 1060]` hPa (values outside NaN'd).
- `era5_cloud_cover`: `[0, 1]` (values outside NaN'd).
- `era5_precipitation`: `[0, 200]` mm (values outside NaN'd).

## Imputation Steps
1. **Interpolation**: Linear interpolation along the time index (for gaps <= 3 days).
2. **ffill/bfill**: Forward-fill and backward-fill for small edge gaps.
3. **Spatial Median**: Imputes remaining NaNs using the median of that point, or the spatial `impute_zone` median (determined by K-Means clustering on coordinates).
4. **MICE Fallback**: Run `IterativeImputer` (scikit-learn) as a final step.

## Hard PASS/FAIL Gate
Writes a report to `qc_report.txt`.
- Checks that the final missingness rate is **<0.1%** for all features.
- Checks that duplicate rows are **0**.
- Fails the pipeline run if these checks are violated.
