# What 03 / 04 / 04b / 05 Actually Do

One record per script, in run order. "Physical" = real-unit values (°C,
W/m², %, etc). "Standardized"/"scaled" = transformed for ML, never for
reading physical meaning off directly.

---

## `03_plots_raw.py` — raw QA, before any cleaning

Runs directly on `climate_tamilnadu_points.csv` (02's output). Nothing here
writes anything back — it's a read-only sanity pass, meant to catch pipeline
bugs (timezone, units, source disagreement) while they're still cheap to fix.

| Plot | What it checks |
|---|---|
| A. Point map | Are the 133 points actually covering Tamil Nadu, sized/colored by population as intended |
| B. Event profile | Does GHI/T_amb actually peak at "noon", not sunrise/sunset — the fastest way to catch a timezone bug |
| C. ERA5 vs NASA POWER | MBE/RMSE/correlation per variable — quantifies exactly how much the two independent sources disagree, before Phase 2 decides what to do about it |
| D. Missing-data heatmap | Which points/columns have real gaps vs physically-expected zeros |
| E. Seasonal boxplots | Sanity check against known Tamil Nadu climatology (hot Apr–Jun, NE monsoon Oct–Dec) |
| F. Multi-year trend | Catches a step-change in one specific year that would suggest a download/unit problem |

**If B shows noon isn't the peak, or C shows a large MBE, stop and fix that before running 04** — those two errors are exactly what the plan doc calls "most silent failures at this stage."

---

## `04_preprocess_tamilnadu.py` — Phase 2, Preprocessing & QC

13 steps, run in this order, on the long `(point_id, date, event)` table.

1. **Dataset inspection.** Shape, dtypes, duplicate `(point_id, date, event)` rows, missing % per column — the baseline everything else is measured against.
2. **Physical validation.** Every column checked against a hard physical range (e.g. GHI in [0, 1400] W/m^2, T in [-30, 55] C, RH in [0, 100] %). Out-of-range -> `NaN`, never silently clipped (clipping hides how much data was actually bad). Solar fields (GHI/DNI/DHI/GHI_clearsky/CSI) are additionally forced to 0 whenever `era5_SZA >= 90 deg` — this is the night-masking check, applied per-row since every row here is already a discrete sun-event, not an hour in a continuous series.
3. **Hampel filter (MAD-based outlier flagging).** Per `(point_id, event)` series, a +/-15-occurrence rolling median/MAD — a point flagged as an outlier is set to `NaN`, never deleted, so imputation (step 4) handles it. *Occurrences, not hours* — the window is 15 prior/following instances of that same sun-event, i.e. roughly a month either side.
4. **Hierarchical imputation.** (a) linear interpolation for gaps <=3 occurrences within a series -> (b) forward/backward fill -> (c) point -> spatial-zone -> global median (the "zone" here is a throwaway KMeans grouping on lat/lon, built only as an imputation fallback — **not** the real Phase 4 climate clustering, kept clearly separate and labeled `impute_zone`) -> (d) MICE (`IterativeImputer`) for whatever's still missing after (a)-(c), fit on a sample.
5. **Temporal validation.** Confirms every `(point_id, event)` series has close to the expected number of daily rows, zero duplicate keys remain.
6. **Feature engineering.** Wind vector components (`W_dir_sin/cos`), `cloud_opacity = 1 - CSI`, `T_depression = T_amb - T_dew`, `is_daytime`, and a local-time (IST) decimal hour + solar hour angle — computed per-row since there's no fixed "hour" column in this schema (each sun-event happens at a different UTC hour depending on point and date).
7. **Lag features.** 1/7/30-occurrence lags of GHI, T_amb, RHum, W_spd, cloud_cover, CSI — computed within each `(point_id, event)` group, so "lag 7" means *the same sun-event 7 days earlier*, not 7 hours earlier.
8. **Rolling stats.** 7/30-occurrence trailing mean+std of the same columns.
9. **Delta features.** 1-occurrence (1-day) rate of change for T_amb, GHI, cloud_cover.
9c. **Lag-warmup drop.** The first 30 occurrences of every `(point_id, event)` series have no 30-days-prior lag to draw from, so those columns are legitimately `NaN` there — dropped now (about 1% of rows), *before* scaling sees them, rather than let step 4's imputation quietly paper over "too early in the series" as if it were a real gap.
9b. **Savitzky-Golay diagnostic.** One sample point/year, raw vs. polynomial-smoothed noon GHI — visual QA only (preserves the noon peak shape, unlike a moving average), doesn't touch the dataframe.
10. **Correlation analysis.** Pearson + Spearman over the daytime rows, saved as CSV + heatmap.
11. **VIF.** Variance Inflation Factor per feature — catches multivariate collinearity a pairwise correlation matrix can miss. Expect near-infinite VIF among GHI/DNI/DHI/CSI — they're algebraically related by construction (DNI and DHI are both derived from GHI), so this is structural, not a data problem; say so if you report it.
12. **Scaling.** `MinMaxScaler` fit **only on the first 70% of chronologically-sorted rows**, applied to the whole file — standard leakage prevention. Per-column scalers saved to `scalers.pkl`. This produces a **separate** file (`tamilnadu_cleaned_scaled.csv`) — Phase 3 must never read this one; it reads the physical-units file instead, because the signature indices (kWh/day, HDD18, etc.) are non-linear functions of physical values and would be silently corrupted by pre-scaling.
13. **Final validation (hard gate).** Zero NaN/Inf in the physical file, the *training portion* of the scaled file within [0,1] (val/test rows are allowed to exceed [0,1] if they contain a more extreme value than training ever saw — that's expected, not a bug), zero duplicate keys, all required columns present. Reports PASS/FAIL per check, writes everything to `qc_report.txt`.

**Outputs:** `tamilnadu_cleaned_physical.csv` (feeds 04b), `tamilnadu_cleaned_scaled.csv` + `scalers.pkl` (for later ML/DRL use), `qc_report.txt`, correlation/VIF/Yeo-Johnson CSVs, two diagnostic PNGs.

---

## `04b_climate_signature.py` — Phase 3, Climate Signature Construction

Reads the **physical-units** file only. Collapses each point's entire
10-year, 3x-daily record into one row — this is the object Phase 4 actually
clusters, not the raw data.

- **18 named indices** (Ta_mean, Ta_p95/p05, DTR, GHI_mean, GHI_daily_kWh, kt_mean, kt_std, SAI, CCI, cloudy_frac, HDD18, CDD24, RH_mean, HSI, wind_mean, seasonality, monsoon_index, elev_proxy) — each one line of physical justification (see the script's Table-8-equivalent docstring). Every index is computed from what the 3-events/day sampling actually supports; two are explicitly flagged as proxies (DTR = noon-sunrise, not true max-min; monsoon_index = a JJAS *fraction*, not an absolute rainfall total, since precipitation is only sampled 3x/day).
- **Tm_target and L_required** — the corrected v2.0 rule: `Tm_target = T_delivery + delta_T_approach` (PCM sits *above* delivery temperature so heat flows PCM->water during discharge; the earlier subtract-based rule had the sign backwards). Comes out to a constant 57 C here (50 + 7, indirect-system assumption) — held constant across all points by design, not tuned per cluster.
- **5 interaction terms** (GHI x kt_std, DTR x cloudy_frac, RH x (Ta-Tm), wind x (Ta-Tsoil), CCI x (1-SAI)).
- **PCA on the correlated block only** (Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy) — retained to 95% variance (typically 2-3 components). Solar/variability indices are deliberately kept *out* of PCA since they carry the discriminating signal.
- **Standardization** of the full signature matrix (z-scores), saved alongside the raw values.

**Outputs:** `climate_signature_tamilnadu.csv` (one row per point), `pca_loadings.csv`, correlation heatmap, per-index distribution plot, a Tamil-Nadu map colored by GHI_mean/monsoon_index.

---

## `05_cluster_regions.py` — Phase 4, Climate Regime Clustering

**Not run yet** — needs `climate_signature_{region}.csv` from at least 2
regions. When it runs: combines all regions' signature matrices, re-standardizes
across the combined set (04b's z-scores were only within-region), fits a
**Gaussian Mixture Model** (soft/probabilistic clustering — not K-Means; the
plan doc explicitly rejects K-Means here because climate is a continuous
gradient, not hard-boundaried), selects K by BIC with a stated realistic
silhouette acceptance band of 0.15-0.35 (a very high silhouette here would
mean the signature collapsed to 1-2 dominant variables, not that regimes are
unusually crisp). Outputs both hard cluster labels and full soft membership
probabilities per point, plus population-weighted cluster profiles.

---

## Quick numbering recap

```
02_combine_tamilnadu.py       -> climate_tamilnadu_points.csv         (Phase 1, done)
03_plots_raw.py                -> QA plots, read-only                  (run before 04)
04_preprocess_tamilnadu.py     -> Phase 2                               (this doc)
04b_climate_signature.py       -> Phase 3                               (this doc)
05_cluster_regions.py          -> Phase 4                               (waits on other regions)
```
