# 04 — Phase 2 Audit: Preprocessing and Cross-Source Validation

Scripts: `02_combine_tamilnadu.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`, `03b_agreement_analysis.py`, `03b_interactive_raw_qa.py`, `04_preprocess_tamilnadu.py`, `04c_postprocess_plots.py`, `04c_interactive_postprocess_qc.py`.

## Purpose
Combine ERA5 and NASA POWER weather variables at the sun-event instants, compute true daily averages/integrals, perform quality control, and impute missing values.

## Processing Details
1. **Combine Script (`02_combine_tamilnadu.py`)** — **v3.1 corrected**:
   - Snaps coordinates to the nearest ERA5 grid node, concatenates NetCDFs, applies `accum_to_flux()` (stateless clip — NOT diff-based deaccumulation), computes solar geometry via `pvlib`, and merges with NASA POWER within a 3-hour match window.
2. **Daily Aggregates (`02b_build_daily_aggregates.py`)**:
   - Reads full hourly NASA POWER series. Integrates GHI trapezoidally to daily kWh/m²/day; calculates DTR, HDD18, CDD24, cloudy fraction, CCI.
3. **Cross-Source Agreement (`03b_agreement_analysis.py`)** — **NEW v3.1**:
   - Stratified MBE/RMSE/Pearson-r table; decision gate (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW); GHI scatter by season.
4. **13-Step Preprocessing (`04_preprocess_tamilnadu.py`)** — **v3.1 corrected**:
   - Steps 1–13 unchanged (inspection, physical validation, Hampel, imputation, features, lags, scaling, QC gate).
   - **Step 2b (NEW)**: Per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution; saves `ghi_quantile_mapping_report.csv`.

<<<<<<< HEAD
## Corrected Audit Findings (v3.1)
1. **Deaccumulation Bug — FIXED**:
   - `02_combine_tamilnadu.py` now uses `accum_to_flux(s) = s.clip(lower=0)`.
   - Pre-fix stats (for reference): noon GHI r = 0.3963, MBE = −231.89 W/m². Post-fix expected: r > 0.80 (Rajasthan reference: r = 0.8102).
2. **Quantile-Mapping — FIXED**:
   - Step 2b in `04_preprocess_tamilnadu.py` applies per-season QM after physical validation.
   - `03b_agreement_analysis.py` documents the cross-source decision branch.

## Status
**COMPLETE (v3.1 fixes applied — re-run `02_combine` → `04_preprocess` for updated outputs)**
=======
## Part A: Combining, Solar Geometry, & Deaccumulation

### 1. The ERA5 Deaccumulation Bug & Fix (formerly `09_ERA5_DATA_PIPELINE.md`)
- **Background**: ERA5 radiation fields (`ssrd`, `strd`, `tp`) are stored in MARS as running accumulations reset at 01:00 and 13:00 UTC. However, CDS point-download requests return pre-processed hourly values.
- **The Bug (v3.0)**: `02_combine_tamilnadu.py` previously applied a diff-based deaccumulation function:
  ```python
  # OLD (buggy v3.0):
  def deaccumulate(s):
      diff = s.diff()
      reset_mask = s.index.hour.isin([1, 13])
      diff[reset_mask] = s[reset_mask]
      return diff.clip(lower=0)
  ```
  Since CDS point downloads were already hourly fluxes, taking `.diff()` subtracted consecutive fluxes, reducing GHI near zero (noon Pearson $r \approx 0.3963$, MBE $\approx -231.89\text{ W/m}^2$).
- **The Fix (v3.1)**: Replaced with `accum_to_flux(s)`:
  ```python
  # NEW (v3.1 corrected):
  def accum_to_flux(s):
      s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
      return s.clip(lower=0)
  ```
  *Rajasthan reference*: Pearson $r$ increased from $0.01$ to $0.8102$ after this fix alone. Tamil Nadu achieves a similar correlation recovery after re-running `02_combine_tamilnadu.py`.

### 2. Solar Geometry Calculations (formerly `12_SOLAR_GEOMETRY.md`)
For every matched timestamp in `02_combine_tamilnadu.py`, solar position is computed using `pvlib.location.Location.get_solarposition()`:
- **Solar Zenith Angle (SZA)**: Zenith angle of the sun relative to vertical.
- **Solar Azimuth Angle**: Solar direction angle along the horizon.
- **Extraterrestrial Radiation (ETR)**: Top-of-atmosphere solar flux.
- **Clear-Sky GHI**: Maximum horizontal solar irradiance under clear skies, computed via the **Ineichen clear-sky model** (`pvlib.location.Location.get_clearsky(model="ineichen")`).
- **Nighttime Suppression**: When $\text{SZA} \ge 90.0^\circ$ (sun below horizon), solar variables (`GHI`, `DNI`, `DHI`, `GHI_clearsky`, `CSI`) are strictly forced to $0.0$.
- **Clearness Index**: $\text{CSI} = \text{GHI} / \text{GHI\_clearsky}$, capped at $1.5$ to handle horizon refraction edge cases.

### 3. Solar Derived Variables (formerly `13_SOLAR_DERIVED_VARIABLES.md`)
- **GHI (Global Horizontal Irradiance)**: Direct output from `ssrd` after `accum_to_flux()`.
- **DNI (Direct Normal Irradiance)**: If direct radiation `avg_sdirswrf` is present, it is used directly; otherwise falls back to beam closure:
  $$\text{DNI} = \frac{\text{GHI}}{\cos(\text{SZA})}, \quad \text{clipped to } [0, 1400]\text{ W/m}^2$$
- **DHI (Diffuse Horizontal Irradiance)**: Derived as the closure residual:
  $$\text{DHI} = \max\left(0, \text{GHI} - \text{DNI} \cdot \cos(\text{SZA})\right)$$
- **Physical Bounds**: All solar variables are hard-bounded to $[0, 1400]\text{ W/m}^2$.

### 4. Temporal Matching & Daily Aggregates (formerly `10_TEMPORAL_PROCESSING.md`)
- **Nearest-in-Time Matching**: ERA5 and NASA POWER hourly series are matched within a 3-hour rejection window.
- **Lag Features Date-Gap Check**: Lags (`lag1d`, `lag7d`, `lag30d`) in `04_preprocess_tamilnadu.py` are grouped by `(point_id, event)` and check for contiguous dates to prevent bridging over missing days.
- **Daily Integration Threshold (`02b_build_daily_aggregates.py`)**: A calendar day is integrated into daily $\text{kWh/m}^2/\text{day}$ only if it has **$\ge 20$ valid hours** of NASA POWER data.

---

## Part B: Cross-Source Agreement & Bias Correction (formerly `14_ERA5_POWER_VALIDATION.md`)

### 1. Pre-Fix Verification Statistics (Reference Baseline)
Cross-source agreement on 1,457,547 matched events **before v3.1 deaccumulation fix**:

| Variable | $n$ | MBE (ERA5 − POWER) | RMSE | Pearson $r$ | Status |
|---|---|---|---|---|---|
| **GHI (W/m²)** | 1,457,547 | **−231.89 W/m²** | **404.69 W/m²** | **0.3963** | Bug active (pre-fix) |
| **Clear-sky GHI (W/m²)** | 1,457,547 | −7.04 W/m² | 53.57 W/m² | 0.9947 | Excellent |
| **T_amb (°C)** | 1,457,547 | +1.08°C | 2.78°C | 0.8454 | Good |
| **RHum (%)** | 1,457,547 | −2.93% | 12.52% | 0.8192 | Good |
| **Wind speed (m/s)** | 1,457,547 | −1.14 m/s | 1.67 m/s | 0.7332 | Moderate |

### 2. Post-Fix Cross-Source Decision Gate (`03b_agreement_analysis.py`)
- **Decision Logic** (thresholds as coded — `CORR_GOOD=0.90`, `CORR_SEVERE=0.70`, `MBE_SMALL_FRAC=0.05`, `SEASON_SPREAD_FRAC=0.05`; evaluated on the **noon** GHI row only):
  - `BACKBONE`: noon Pearson $r \ge 0.90$ **and** $|\text{MBE}|$ $\le 5\%$ of mean noon POWER GHI **and** the max−min season-to-season noon MBE spread $\le 5\%$ of that mean $\rightarrow$ ERA5 used directly, POWER kept as a reported cross-check.
  - `QUANTILE_MAP`: noon $r \ge 0.70$ but one of the BACKBONE sub-conditions fails $\rightarrow$ per-season empirical quantile mapping of daytime ERA5 GHI onto the POWER distribution.
  - `MANUAL_REVIEW`: noon $r < 0.70$ or undefined $\rightarrow$ flag, run merge-bug diagnostics, stop.
  - Fixed-weight blending (e.g. `0.6·ERA5 + 0.4·POWER`) is explicitly rejected — no principled derivation for a fixed weight.
- **This gate is advisory.** `03b_agreement_analysis.py` is read-only and never persists a correction. `04_preprocess_tamilnadu.py`'s Step 2b applies the per-season quantile map **unconditionally** (every season, every run) and writes the corrected `era5_GHI` + recomputed `era5_CSI` into `tamilnadu_cleaned_physical.csv` — it does not consult which branch `03b_agreement_analysis.py` reported.
- **Tamil Nadu branch**: post-fix noon GHI clears the $\ge 0.70$ floor but not the full BACKBONE gate, i.e. `QUANTILE_MAP` (numbers in `bias_decision_tamilnadu.txt` / `era5_power_agreement_tamilnadu.csv`).

> **Cross-state note (2026-09-08):** Rajasthan converged onto this same `04_preprocess` contract — its Phase 2.5 is now `04_preprocess_rajasthan.py` (was the leaner `03b_quality_check_rajasthan.py`), so **both** states now feed per-season quantile-mapped ERA5 GHI/CSI into Phase 3, with the same `BOUNDS` screen, `SZA ≥ 90°` night-mask, and 4-stage/MICE imputation. Uttarakhand already used this pattern; Assam uses a partial variant. See `docs/rajasthan/04_PHASE_2_AUDIT.md` Part B.

---

## Part C: Quality Control & Data Cleaning (formerly `15_QUALITY_CONTROL.md`)

### 1. 13-Step Preprocessing Pipeline (`04_preprocess_tamilnadu.py`)
1. **Schema Verification**: Ensures required columns exist.
2. **Physical Bounds Screening**: Applies hard bounds (`BOUNDS` dictionary):
   - `era5_GHI`: $[0, 1400]\text{ W/m}^2$ (night-masked to 0.0 when $\text{SZA} \ge 90^\circ$).
   - `era5_T_amb`: $[-30, 55]^\circ\text{C}$.
   - `era5_RHum`: $[0, 100]\%$.
   - `era5_W_spd`: $[0, 50]\text{ m/s}$.
   - `era5_P_atm`: $[850, 1060]\text{ hPa}$.
   - `era5_cloud_cover`: $[0, 1]$ fraction.
   - `era5_precipitation`: $[0, 200]\text{ mm}$.
   - *Step 2b (Quantile Mapping)*: Applies per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution.
3. **Hampel Filter (Outlier Detection)** — as coded in `04_preprocess_tamilnadu.py` (`HAMPEL_COLS`):
   - Applied to `era5_GHI`, `era5_T_amb`, `era5_RHum`, `era5_W_spd`, **and** `era5_cloud_cover`, over a rolling MAD window of **15 occurrences each side** (31-wide, in event-occurrences of the same `(point_id, event)` series — not hours) at a $3\sigma$ threshold; flagged values → `NaN` → imputation (step 4).
   - **Divergence from Rajasthan (worth resolving):** `03b_quality_check_rajasthan.py` deliberately **excludes** `era5_GHI` / `era5_CSI` from Hampel filtering after three documented empirical corrections (filtering them produced a uniform `GHI_noon_mean`↑ / `kt_noon_std`↓ shift across all points — real cloud-driven low-clearness signal, not sensor noise, and exactly what `cloudy_frac` / `CCI` / `kt_std` / `monsoon_index` are built to measure). `04_preprocess_tamilnadu.py` does **not** carry that exclusion. Now that Rajasthan runs `04_preprocess_rajasthan.py`, this exclusion is currently lost for it too — flagged as a regression risk to check against the regenerated signature.
4. **Imputation Cascade** (`IMPUTE_COLS`, within `(point_id, event)` groups sorted by date):
   - *Stage a*: Linear interpolation, interior gaps $\le 3$ occurrences (`limit_area="inside"`).
   - *Stage b*: `ffill(limit=3)` then `bfill(limit=3)` for edge gaps.
   - *Stage c*: `point_id` median → `impute_zone` median (`impute_zone` = an 8-way KMeans grouping of point lat/lon built here **only** as an imputation fallback, not the Phase 4 climate clustering) → global column median.
   - *Stage d*: `sklearn.impute.IterativeImputer` (MICE) on any cells still missing after a–c (fit on up to a 300k-row sample, `random_state=42`).
5. **Feature Engineering** (steps 6, 9 in-script): wind-direction sin/cos × speed, `cloud_opacity`, `T_depression`, `is_daytime`, IST decimal hour, solar hour angle. *(Not consumed by Phase 3 — `04b_climate_signature.py` rebuilds its own Tier-1/2 signature from the base physical columns.)*
6. **Lag / Rolling / Delta** (steps 7–9): 1/7/30-occurrence lags, 7/30-occurrence rolling mean/std, 1-occurrence delta on `LAG_COLS`; step 9c drops the first 30 occurrences per `(point_id, event)` as lag warm-up. *(Also not read by Phase 3.)*
7. **Scaling** (step 12): per-column `MinMaxScaler` fit on the first 70% of chronologically-sorted rows only (leakage-safe), written to `scalers.pkl` + `<state>_cleaned_scaled.csv`. **Phase 3 reads the *physical* file, never the scaled one.**
8. **Hard PASS/FAIL Gate**:
   - Writes `qc_report.txt`.
   - Checks final missingness rate **$< 0.1\%$** for all features.
   - Checks duplicate rows **$= 0$**.
   - Fails the pipeline execution if any gate is violated.

---

## Status
**COMPLETE (v3.1 fixes applied)** — regenerate outputs by re-running `02_combine_tamilnadu.py` → `04_preprocess_tamilnadu.py` → `04b_climate_signature.py` → downstream. `03b_agreement_analysis.py` is an advisory read-only cross-check and can be run any time; it does not feed `04_preprocess`.

---
>>>>>>> 935afa34a2c58bf28d0e38fac953d563fa476637

## Literature Support
| Method | Reference | Source |
|---|---|---|
| ERA5 vs satellite GHI validation | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Quantile mapping / bias correction | Mansouri et al. (2025) | `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
| Hampel MAD outlier detection | Standard QC practice | `15_QUALITY_CONTROL.md` |
| MICE imputation | Rubin (1987); sklearn IterativeImputer | `15_QUALITY_CONTROL.md` |
