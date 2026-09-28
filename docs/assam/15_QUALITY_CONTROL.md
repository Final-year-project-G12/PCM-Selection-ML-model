# 18 — Quality Control Audit (Assam)

**Script**: `04_preprocess_assam.py`

**Status**: COMPLETE — IsolationForest-based QC with imputation

## Part 1 — Physical bounds checking

Applied to `climate_assam_points.csv` before any statistical QC. **Correction:** out-of-range
values are **set to `NaN`** (`df.loc[out_of_bounds, col] = np.nan`), not merely flagged — this
phrasing belongs to the outlier step (Part 2), not the bounds step:

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

## Part 2 — Outlier detection: 3σ screening, then IsolationForest

**Correction:** `04_preprocess_assam.py` applies **two** stages, not IsolationForest alone:
1. Per `(point_id, month, event)` **3σ** test on `era5_T_amb`, `era5_GHI`, `era5_W_spd`.
2. **scikit-learn IsolationForest** (`contamination=0.01`, `random_state=42`) on
   `[era5_T_amb, era5_GHI, era5_RHum, era5_W_spd]` with missing values filled as 0 before fitting.

Both stages set the same `is_outlier` flag. This is a **different approach from Rajasthan**, which used a Hampel filter
(univariate, per-column, median ± n_sigma × MAD).

**Why IsolationForest is better-suited for Assam**:
- Rajasthan's Hampel filter had to **exclude GHI/CSI** after a discovered bug (it was winsorizing
  genuine cloud-driven GHI variability). IsolationForest handles multivariate distributions including
  heavy-tailed solar radiation naturally — it scores anomaly-ness in the full feature space, not
  column-by-column.
- Assam's monsoon precipitation produces genuinely heavy-tailed distributions in precipitation and
  cloud cover that a univariate Hampel filter would aggressively flag as outliers. IsolationForest's
  ensemble tree splits better handle asymmetric, multimodal distributions.

**Policy**: Outlier-flagged rows are **flagged but never deleted** — they receive an outlier flag column and are
carried through to downstream phases. This matches the Rajasthan policy. Note this is distinct
from the site-year deletion in Part 3, which does remove rows (for a different reason: excessive
missingness, not outlier status).

## Part 3 — Missing data imputation

**Correction:** the current script performs a **two-step** fallback (its own step-[5] log message
says "≤ 6h", but the code passes `limit=1`, i.e. bridges only a single consecutive step), followed
by a separate deletion step, not a three-step chain:
1. `interpolate(method="linear", limit=1)` per point — bridges single-step gaps only.
2. Fill remaining gaps with the `(point_id, month, event)` group mean — a climatological
   same-month-same-event value. There is no third, separate "point-event fallback" stage.
3. **Site-years where any imputed column still exceeds 5% missing are dropped entirely** — a
   genuine deletion this document did not previously mention. It can silently shorten a point's
   record length and should be logged/reported (how many site-years were dropped).

No `_imputed` boolean flag column is written by the current script; the only status flag produced
in this step is via the site-year drop, not a per-cell provenance flag.

## Output: `preprocessed/parquet/{point_id}.parquet`

- **129 files** (one per point: `ASP_0001.parquet` through `ASP_0129.parquet`)
- Columns: physical units, QC-passed, outlier-flagged, imputed, no scaling
- **Why parquet**: Efficient columnar storage; preserves dtypes; faster to read than CSV for
  downstream phases. Same convention as Rajasthan.

## What Assam QC Implements vs Rajasthan

| Component | Assam Implementation Status |
|---|---|
| `03_verify_climate_csv.py` (schema/coverage/nulls/range gate) | **Implicit** (validated via pipeline checks) |
| `03_qc_plots.py` (spatial/distributional visualizations) | **Implemented in Phase 11 visuals** |
| `03b_agreement_analysis_assam.py` (ERA5 vs POWER formal comparison) | **Implemented** (`BACKBONE` decision, 1.1% GHI MBE) |
| Phase 2.5 IsolationForest outlier detection + imputation | **Implemented** (`04_preprocess_assam.py`) |

The Assam pipeline has the **statistical QC step** (outlier detection + imputation, Phase 2.5) and the
**cross-source agreement analysis** (`03b_agreement_analysis_assam.py`), validating that:
- All 129 expected points are present and accounted for
- Duplicate (point_id, date, event) combinations don't exist
- Null rates are within acceptable thresholds
- ERA5 and POWER agree within a documented tolerance (1.1% MBE)

These validations are currently only implicit (done manually by inspection if at all).

## Overall QC assessment

The IsolationForest-based Phase 2.5 is a sound, defensible approach that avoids Rajasthan's
Hampel-filter GHI exclusion problem. However, the absence of a formal Phase 2 QA gate and
ERA5-POWER agreement analysis means the Assam pipeline's data quality is less rigorously
documented than Rajasthan's. This is the main methodology-completeness gap for Assam's
Phase 1–2 work.
