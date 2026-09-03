# What 02b / 03 / 04 / 04b / 04c / 05 Actually Do — Tamil Nadu, Objective 1

Scope for this document: **Tamil Nadu only.** No other state's data exists
yet, so nothing here clusters or compares across states. Where the
original 4-state plan (v3.0) called for something state-spanning, that
step is either skipped, deferred, or reduced to a single-state version —
each case is called out explicitly below.

Run order:
```
02_combine_tamilnadu.py         (Phase 1 — already done, confirmed correct)
02b_build_daily_aggregates.py   (Phase 2, Repair 1 — NEW, run before 04)
03_plots_raw.py                 (raw QA — read-only, run before 04)
04_preprocess_tamilnadu.py      (Phase 2 — preprocessing & QC)
04c_postprocess_plots.py        (post-clean QA — NEW, run after 04)
04b_climate_signature.py        (Phase 3 — climate signature, Tier1+Tier2)
05_cluster_tamilnadu.py         (Phase 4 — clustering, TN only)
```

---

## `02_combine_tamilnadu.py` — Phase 1 (already run)

Confirmed correct: 133 population-weighted points x 10 years x 3 sun-events
x ~34 columns -> ~1.46M rows -> ~650MB CSV matches your actual output size
exactly, and the header schema matches the script's `ERA5_OUTPUT_VARS` +
`POWER_VARS` lists. Row 1 (`sunrise`, GHI=0) is physically correct, not a
bug.

---

## `02b_build_daily_aggregates.py` — Phase 2, Repair 1 (NEW — you had not built this)

This is the single most important gap identified in plan v3.0 (Section 4.3,
"the repair that cannot be skipped"). `climate_tamilnadu_points.csv` keeps
only 3 hours out of every 24 — sunrise, noon, sunset. Several signature
indices are mathematically impossible to compute from three instants:

- **True daily GHI integral** (kWh/m^2/day) — a half-sine approximation
  from noon GHI alone (what the old `04b` did) is a proxy, not a
  measurement.
- **True DTR** (Tmax-Tmin) — `noon T - sunrise T` is a lower bound, not
  the real diurnal range (true Tmax usually lags solar noon by 1-3h).
- **True HDD18 / CDD24** — degree-days need a real daily mean temperature.
- **cloudy_frac / CCI** (consecutive-cloudy-day run length) — needs a
  whole-day clearness index, not just the noon instant.

The fix costs **zero new downloads**. `01b_download_nasapower.py` already
cached the FULL hourly series for every point/year at
`data/raw/nasapower/power_{point_id}_{year}.json`; `02_combine` just never
read past the 3 sun-event hours. This script re-reads those same JSON
files, integrates them properly per day, and writes:

- `daily_aggregates_tamilnadu.csv` — one row per (point_id, date), true
  daily-integral values.
- `tier2_signature_tamilnadu.csv` — one row per point_id, the point-level
  Tier-2 indices, ready for `04b` to merge in.

**Known limitation, stated honestly:** `01b`'s NASA POWER request never
included precipitation (`PRECTOTCORR`), so `monsoon_index` still comes
from ERA5's 3x/day precipitation-fraction proxy — it is **not** upgraded
to a true Tier-2 index by this script. If you want that fixed, add
`PRECTOTCORR` to `POWER_PARAMETERS` in `01b_download_nasapower.py` and
re-run just that one script (no CDS queue involved). This is optional,
not required for Objective 1 to be defensible — say so in the paper.

Days with fewer than 20/24 hours reported by NASA POWER are dropped, not
averaged over a short day, so a partial-coverage day can't quietly bias a
mean.

---

## `03_plots_raw.py` — raw QA, before any cleaning (unchanged, already correct)

Runs on `02`'s output, read-only. Six checks: point map (A), sun-event
timing sanity via noon-peaks-GHI (B), ERA5-vs-NASA-POWER agreement/MBE/RMSE
(C), missing-data heatmap (D), seasonal boxplots (E), multi-year trend for
step-changes (F). If B shows noon isn't the peak, or C shows a large
systematic MBE, stop and fix that before running `04` — these are exactly
the "most silent failures at this stage" the plan doc warns about.

---

## `04_preprocess_tamilnadu.py` — Phase 2, Preprocessing & QC (unchanged, already correct)

13 steps on the long `(point_id, date, event)` table — see the script's
own docstrings for exact mechanics per step; the summary is unchanged from
before:

1. Dataset inspection — shape, dtypes, duplicates, missing % baseline.
2. Physical validation — hard-range check per column -> NaN (not clip);
   solar fields forced to 0 when `SZA >= 90 deg`.
3. Hampel filter (MAD) — per (point_id, event) series, +/-15-occurrence
   rolling median/MAD outlier flag -> NaN.
4. Hierarchical imputation — interpolate (<=3 gap) -> ffill/bfill ->
   point/zone/global median -> MICE for whatever's still missing.
5. Temporal validation — coverage and duplicate-key checks.
6. Feature engineering — wind vector components, cloud_opacity,
   T_depression, is_daytime, IST decimal hour + solar hour angle.
7. Lag features — 1/7/30-occurrence lags (days, not hours — see script).
8. Rolling stats — 7/30-occurrence trailing mean+std.
9. Delta features — 1-occurrence rate of change.
9c. Lag-warmup row drop (~1% of rows, first 30 occurrences per series).
9b. Savitzky-Golay diagnostic plot (visual QA only).
10. Correlation analysis (Pearson + Spearman).
11. VIF (multivariate collinearity; expect near-infinite among GHI/DNI/DHI/
    CSI — structural, not a bug, since DNI/DHI are algebraically derived
    from GHI).
12. Scaling — MinMax fit on first 70% of chronologically-sorted rows only
    (leakage-safe); separate output file, never used by Phase 3.
13. Final validation (hard gate) — PASS/FAIL per check, written to
    `qc_report.txt`.

**Outputs:** `tamilnadu_cleaned_physical.csv` (feeds `04b` and `04c`),
`tamilnadu_cleaned_scaled.csv` + `scalers.pkl` (later ML/DRL use only),
`qc_report.txt`, correlation/VIF/Yeo-Johnson CSVs, two diagnostic PNGs.

---

## `04c_postprocess_plots.py` — post-cleaning QA (NEW)

Runs on `04`'s output. `03` checked the raw data; this checks what the 13
cleaning steps actually did to it, before you trust it for Phase 3.

- **A.** Missing-data heatmap post-clean — should be ~0 everywhere (if
  not, step 13's hard gate in `04` should already have failed).
- **B.** Distribution histograms post-clean — watch for a suspicious spike
  exactly at one value, which usually means imputation dominated a column.
- **C.** Parses `qc_report.txt` and plots how many values physical-bounds
  (step 2) vs. Hampel/MAD (step 3) each flagged, per column.
- **D.** GHI vs. GHI-7-days-prior scatter — sanity-checks the lag features
  from step 7 actually carry structure, not noise.
- **E.** One point's cleaned noon-GHI time series for one year with the
  7d/30d rolling mean overlaid — the seasonal shape should look smooth,
  not flattened by imputation.
- **F.** Correlation heatmap including the step-6 engineered features
  (cloud_opacity, T_depression, solar_hour_angle, delta features).

---

## `04b_climate_signature.py` — Phase 3, Climate Signature Construction (UPDATED)

Reads **both** `04`'s physical-units output (Tier 1: sun-event indices)
**and** `02b`'s `tier2_signature_tamilnadu.csv` (Tier 2: true daily-integral
indices), and merges them. Wherever a true Tier-2 value exists it becomes
the canonical signature column (`GHI_daily_kWh`, `DTR`, `kt_mean`,
`kt_std`, `SAI`, `cloudy_frac`, `CCI`, `HDD18`, `CDD24`, `Ta_mean`,
`Ta_p95`, `Ta_p05`, `seasonality`); the old sun-event-only proxy is kept
alongside with a `_proxy` suffix purely so you can report "proxy vs. true
agreement" in your methodology if you want to.

- **Tm_target and L_required** — corrected v2.0/v3.0 rule:
  `Tm_target = T_delivery + delta_T_approach` = 50 + 7 = **57 C**
  (PCM must sit *above* delivery temperature to discharge heat into the
  water — the earlier subtract-based rule had the sign backwards). Held
  constant across all points by design.
- **5 interaction terms** (GHI x kt_std, DTR x cloudy_frac, RH x (Ta-Tm),
  wind x (Ta-Tsoil), CCI x (1-SAI)) — now computed on the canonical
  (true-where-available) columns.
- **PCA on the correlated block only** (Ta_mean, Ta_p95, Ta_p05, HDD18,
  CDD24, RH_mean, elev_proxy) — retained to 95% variance.
- **Clustering matrix** explicitly excludes lat/lon (never cluster on
  geography — it would guarantee the "result" is just a map of Tamil
  Nadu's shape), the raw PCA_BLOCK columns (redundant with PC1..PCn), and
  every `_proxy`/`_true` column (only the canonical version clusters).
- **Standardization** (z-scores) of the final clustering matrix.

**Outputs:** `climate_signature_tamilnadu.csv` (one row per point — this is
what `05` clusters), `pca_loadings.csv`, correlation heatmap, per-index
distribution plot, a Tamil Nadu map colored by true daily GHI / monsoon
index.

**elevation note:** still the flat 150m proxy from `02_combine`, not real
per-point elevation (that's plan v3.0's "Repair 2", written for
Uttarakhand's 200m-7000m range). Tamil Nadu's elevation spread is much
smaller (coastal plain to the Nilgiris at ~2,600m), so this matters less
here — note it as a stated limitation rather than fixing it under a
4-day deadline, unless the Nilgiris points are important to your story.

---

## `05_cluster_tamilnadu.py` — Phase 4, Climate Regime Clustering, TN-only (NEW, replaces 05_cluster_regions.py for now)

**You do not need to cluster across other states to finish Objective 1.**
The objective statement is "cluster meteorological data and identify
Top-2/Top-3 PCM candidates per climatic regime" — nothing requires those
regimes to span state boundaries. Clustering the 133 TN points into their
own regimes (Gaussian Mixture, BIC-selected K, K-Means reported as
comparison) already gives you "different PCMs for different locations"
within Tamil Nadu.

`05_cluster_regions.py` (the original 4-state script) still exists and is
already state-parameterised — if you add another state's Phase 3 output
later, switch to it. Until then, this single-state version produces the
same output shapes (`cluster_assignments_tamilnadu.csv` with soft
membership probabilities, `cluster_profiles_tamilnadu.csv` population-
weighted per regime) so nothing downstream (Phase 5/6) needs to change
when/if you extend to more states later.

Expected K for one state: probably 3-6 (coastal humid, interior dry,
Nilgiris hills, maybe a north/south split) — much smaller than the 6-10
expected across all four states. Realistic silhouette band widened
slightly to 0.15-0.40 versus the 4-state script's 0.15-0.35, since a
single state doesn't have the artificially clean between-state gaps that
inflate silhouette in the multi-state version — say this in your
methodology if your silhouette comes out on the higher side.

---

## Quick numbering recap

```
02_combine_tamilnadu.py          -> climate_tamilnadu_points.csv          (Phase 1, done)
02b_build_daily_aggregates.py    -> tier2_signature_tamilnadu.csv         (Phase 2 Repair 1)
03_plots_raw.py                  -> QA plots, read-only                    (run before 04)
04_preprocess_tamilnadu.py       -> tamilnadu_cleaned_physical.csv         (Phase 2)
04c_postprocess_plots.py         -> QA plots, read-only                    (run after 04)
04b_climate_signature.py         -> climate_signature_tamilnadu.csv        (Phase 3)
05_cluster_tamilnadu.py          -> cluster_profiles_tamilnadu.csv         (Phase 4, TN only)
```
