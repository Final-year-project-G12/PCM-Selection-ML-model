# 04 — Phase 2 Audit: Preprocessing, Quality Control, and Cross-Source Validation

Scripts: `02_combine_tamilnadu.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`, `03b_agreement_analysis.py`, `03b_interactive_raw_qa.py`, `04_preprocess_tamilnadu.py`, `04c_postprocess_plots.py`, `04c_interactive_postprocess_qc.py`.

## Purpose
Combine ERA5 reanalysis and NASA POWER satellite weather variables at sun-event instants, compute solar geometry and derived solar components, perform 13-step physical quality control and data cleaning, execute cross-source agreement validation with per-season quantile mapping, and generate daily integral aggregates.

---

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
- **Decision Logic**:
  - `BACKBONE`: Pearson $r \ge 0.85$ and $|\text{MBE}| \le 20\text{ W/m}^2$ $\rightarrow$ use ERA5 directly.
  - `QUANTILE_MAP`: Pearson $r \ge 0.70$ but fails MBE threshold $\rightarrow$ apply Step 2b per-season quantile mapping.
  - `MANUAL_REVIEW`: Pearson $r < 0.70$ $\rightarrow$ flag for review.
- **Tamil Nadu Branch**: Post-fix GHI achieves $r > 0.80$, routing into **Step 2b Quantile Mapping** (`ghi_quantile_mapping_report.csv`).

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
3. **Hampel Filter (Outlier Detection)**:
   - Applied to `era5_T_amb`, `era5_RHum`, and `era5_W_spd` using rolling MAD (Median Absolute Deviation) window of 7 with $3\sigma$ threshold.
   - **Crucial Rule**: **GHI and CSI are deliberately excluded from Hampel filtering** because cloud transients produce real physical spikes; filtering GHI corrupts solar intermittency statistics.
4. **Imputation Cascade**:
   - *Stage 1*: Linear interpolation along time index (gaps $\le 3$ days).
   - *Stage 2*: Forward-fill (`ffill`) and backward-fill (`bfill`) for edge gaps.
   - *Stage 3*: Spatial median imputation using `impute_zone` (K-Means spatial clusters).
   - *Stage 4*: MICE fallback via `sklearn.impute.IterativeImputer`.
5. **Feature Engineering**: Derives HSI, degree days (HDD18, CDD24), and wind power density.
6. **Lag Generation**: Computes 1-day, 7-day, 30-day lags within point-event groups.
7. **Scaling**: z-score scaling saved to `scaler_params.json`.
8. **Hard PASS/FAIL Gate**:
   - Writes `qc_report.txt`.
   - Checks final missingness rate **$< 0.1\%$** for all features.
   - Checks duplicate rows **$= 0$**.
   - Fails the pipeline execution if any gate is violated.

---

## Status
**COMPLETE (v3.1 fixes applied)** — Re-run `02_combine` $\rightarrow$ `03b` $\rightarrow$ `04_preprocess` for updated outputs.

---

## Literature Support

| Component | Reference / Method | Source File |
|---|---|---|
| ERA5 vs Satellite Validation | Ghodusinejad et al. (2026) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Quantile Mapping | Mansouri et al. (2025) | `sources/Mansouri2025MultimodalRenewableForecasting_summary.md` |
| Hampel MAD Filter | Hampel (1974) MAD outlier filter | Standard time-series QA practice |
| MICE Imputation | Rubin (1987); Van Buuren (2018) | `sklearn.impute.IterativeImputer` |
| Ineichen Clear-Sky | Ineichen & Perez (2002) via pvlib | `pvlib.location.Location.get_clearsky()` |
