# 04 — Phase 2 Audit: Preprocessing & Cross-Source Validation

**Script(s)**: `02_combine_assam.py`, `02b_build_daily_aggregates_assam.py`

**Status**: COMPLETE

## `02_combine_assam.py` — ERA5 + POWER merge

### What it does
Reads ERA5 NetCDF files (per point, per year-month, instant+accum variable sets) and NASA POWER JSON
files, aligns them in time, applies unit conversions and solar geometry, and writes the combined
`climate_assam_points.csv`.

### ERA5 handling
- **Solar radiation**: ERA5 returns per-hour flux values from CDS. The script applies a stateless
  non-negative clip (no differencing) — the same `accum_to_flux()` fix that was discovered as a
  critical bug in the Rajasthan pipeline. Assam's pipeline inherits the **fixed** version, not the
  broken version.
- **Temperature**: Kelvin → Celsius conversion
- **Wind**: u10 + v10 → speed (m/s) + direction (degrees)
- **Pressure**: Pa → hPa

### Solar geometry (pvlib)
- GHI_clearsky from Ineichen model; CSI = GHI / GHI_clearsky
- Solar zenith, azimuth at each event time
- pvlib `get_solarposition()` called per-point, per-event

### Season mapping
```
Dec, Jan, Feb  → Winter (1)
Mar, Apr, May  → Pre-Monsoon (2)
Jun, Jul, Aug, Sep → Monsoon (3)
Oct, Nov       → Post-Monsoon (4)
```
Monsoon is 4 months (Jun–Sep), reflecting Assam's longer monsoon compared to Rajasthan's Jun–Aug.

### Cross-source matching
- ERA5 and POWER are time-matched by point; `MAX_MATCH_HOURS = 3` tolerance for timestamp alignment
- Both sources' GHI columns are preserved (prefixed `era5_` and `power_`) for downstream agreement analysis

### Output
- `climate_assam_points.csv`: ~1.4 million rows (128 pts × 3653 days × 3 events per day)
- Columns include: `point_id`, `lat`, `lon`, `date`, `event`, `season`, ERA5 variables, POWER variables,
  derived solar geometry

## `02b_build_daily_aggregates_assam.py` — Daily integrals

### What it does
Reads `climate_assam_points.csv` and builds true daily-integral indices from the NASA POWER daily data
(not the event-sampled ERA5 data). Produces:

- `daily_aggregates_assam.csv` (~467k rows: 128 × 3653 days) — daily kt, GHI_daily_kWh, T2M_MAX,
  T2M_MIN, precipitation, etc.
- `tier2_signature_assam.csv` — per-point aggregated Tier 2 indices (CCI, SAI, kt_mean, monsoon_index,
  cloudy_frac, annual precipitation) consumed by `04b_climate_signature.py`

## What is absent (vs Rajasthan)

Rajasthan had explicit cross-source agreement analysis scripts:
- `03_verify_climate_csv.py` (schema/coverage/nulls/range/agreement QA report)
- `03_qc_plots.py` (spatial + distributional QC HTML plots)
- `03b_agreement_analysis.py` (ERA5 vs POWER agreement with decision rule: BACKBONE / QUANTILE_MAP)

**These dedicated agreement-analysis scripts do not exist in the Assam pipeline.** Cross-source
agreement is implicitly validated through the merge logic in `02_combine_assam.py` (both sources
kept as columns, enabling downstream comparison), but there is **no `bias_decision_assam.txt`**
documenting whether BACKBONE or QUANTILE_MAP was applied. This is an open gap relative to Rajasthan.

## Known issues

1. **No explicit ERA5-POWER agreement analysis**: The formal Phase 2 agreement layer (stratified by
   season × event × variable) is not implemented for Assam. Downstream phases consume ERA5 GHI
   without a documented cross-source correction decision. This should be added if Assam's Phase 3+
   results are to be treated with the same rigor as Rajasthan's.

2. **Quantile-mapping correction status**: Not known for Assam — the agreement analysis that would
   trigger a quantile-map correction branch was never run.

3. **Monsoon-month definition**: Assam uses Jun–Sep (4 months) for Monsoon. This is internally
   consistent within all Assam scripts but differs from Rajasthan's Jun–Aug definition. This is the
   correct choice for Assam climatologically.
