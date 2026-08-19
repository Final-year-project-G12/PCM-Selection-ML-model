# 18 — Quality Control Audit (Assam)

**Script**: `04_preprocess_assam.py`

**Status**: COMPLETE — IsolationForest-based QC with imputation

## Part 1 — Physical bounds checking

Applied to `climate_assam_points.csv` before any statistical QC. Out-of-range values are flagged
(not deleted):

| Variable | Lower | Upper |
|---|---|---|
| `era5_GHI` | 0 W/m² | 1400 W/m² |
| `era5_T_amb` | −30 °C | 55 °C |
| `era5_RHum` | 0 % | 100 % |
| `era5_T_dew` | −30 °C | 40 °C |
| `era5_W_spd` | 0 m/s | 50 m/s |
| `era5_P_atm` | 850 hPa | 1060 hPa |
| `era5_cloud_cover` | 0 | 1 |
| `era5_precipitation` | 0 mm | 200 mm |

**Note on `era5_precipitation` upper bound**: 200 mm/day is a reasonable physical ceiling for daily
precipitation, but Assam experiences extreme rainfall events (e.g., Mawsynram at the Meghalaya border
records 10,000+ mm/year with extreme daily events). For ERA5 3-event hourly data, 200 mm/3-hour
would only be flagged if the raw field reaches this threshold. Given that ERA5's precipitation is
a spatially-smoothed 0.25° grid estimate, values exceeding 200 mm/event are very unlikely in the
ERA5 output even for extreme events. This bound is appropriate.

## Part 2 — Outlier detection: IsolationForest

`04_preprocess_assam.py` uses **scikit-learn IsolationForest** (multivariate ensemble tree-based
anomaly detection). This is a **different approach from Rajasthan**, which used a Hampel filter
(univariate, per-column, median ± n_sigma × MAD).

**Why IsolationForest is better-suited for Assam**:
- Rajasthan's Hampel filter had to **exclude GHI/CSI** after a discovered bug (it was winsorizing
  genuine cloud-driven GHI variability). IsolationForest handles multivariate distributions including
  heavy-tailed solar radiation naturally — it scores anomaly-ness in the full feature space, not
  column-by-column.
- Assam's monsoon precipitation produces genuinely heavy-tailed distributions in precipitation and
  cloud cover that a univariate Hampel filter would aggressively flag as outliers. IsolationForest's
  ensemble tree splits better handle asymmetric, multimodal distributions.

**Policy**: Outliers are **flagged but never deleted** — they receive an outlier flag column and are
carried through to downstream phases. This matches the Rajasthan policy.

## Part 3 — Missing data imputation

Three-step fallback chain (same logic as Rajasthan):
1. Linear interpolation for gaps ≤ 3 consecutive events
2. Point-seasonal mean substitution for larger gaps
3. Point-event fallback mean (across all dates for that event) for any remaining gaps

Imputed values receive an `_imputed` boolean flag column.

## Output: `preprocessed/parquet/{point_id}.parquet`

- **128 files** (one per point)
- Columns: physical units, QC-passed, outlier-flagged, imputed, no scaling
- **Why parquet**: Efficient columnar storage; preserves dtypes; faster to read than CSV for
  downstream phases. Same convention as Rajasthan.

## What Assam QC lacks vs Rajasthan

| Rajasthan QC component | Assam status |
|---|---|
| `03_verify_climate_csv.py` (schema/coverage/nulls/range gate) | **Not implemented** |
| `03_qc_plots.py` (8 HTML spatial/distributional visualizations) | **Not implemented** |
| `03b_agreement_analysis.py` (ERA5 vs POWER formal comparison) | **Not implemented** |
| `03c_plots_raw_rajasthan.py` (pre-QC visual diagnostics) | **Not implemented** |
| `03b_quality_check_plots_rajasthan.py` (post-QC visual diagnostics) | **Not implemented** |
| Phase 2.5 IsolationForest outlier detection + imputation | **Implemented** (`04_preprocess_assam.py`) |

The Assam pipeline has the **statistical QC step** (outlier detection + imputation, Phase 2.5) but
lacks the **schema/coverage/agreement validation scripts** (Phase 2 QA gate). This means there is
no automated check that:
- All 128 expected points are present in the combined CSV
- Duplicate (point_id, date, event) combinations don't exist
- Null rates are within acceptable thresholds
- ERA5 and POWER agree within a documented tolerance

These validations are currently only implicit (done manually by inspection if at all).

## Overall QC assessment

The IsolationForest-based Phase 2.5 is a sound, defensible approach that avoids Rajasthan's
Hampel-filter GHI exclusion problem. However, the absence of a formal Phase 2 QA gate and
ERA5-POWER agreement analysis means the Assam pipeline's data quality is less rigorously
documented than Rajasthan's. This is the main methodology-completeness gap for Assam's
Phase 1–2 work.
