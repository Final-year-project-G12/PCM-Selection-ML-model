# What 02b / 03 / 04 / 04b / 04c / 05 Actually Do — Uttarakhand, Objective 1

Scope for this document: **Uttarakhand only.** No other state's data
exists in this repo yet, so nothing here clusters or compares across
states. Where the original 4-state plan (v3.0) called for something
state-spanning, that step is either skipped, deferred, or reduced to a
single-state version — each case is called out explicitly below.

Run order:
```
02_combine_uttarakhand.py       (Phase 1)
02b_build_daily_aggregates.py   (Phase 2, Repair 1 — run before 04)
03_plots_raw.py                 (raw QA — read-only, run before 04)
04_preprocess_uttarakhand.py    (Phase 2 — preprocessing & QC)
04c_postprocess_plots.py        (post-clean QA — run after 04)
04b_climate_signature.py        (Phase 3 — climate signature, Tier1+Tier2)
05_cluster_uttarakhand.py       (Phase 4 — clustering, Uttarakhand only)
```

---

## `02_combine_uttarakhand.py` — Phase 1

Merges ERA5 + NASA POWER for every `(point_id, date, event)` combination.
Expected shape: **45 population-weighted points x 10 years x 3 sun-events**
= 45 x 3,653 days x 3 events ≈ **493,000 rows** (a small number fewer, since
some point/day/event combinations get dropped if both sources fall outside
the 3-hour matching window — check your own printed row count against this
estimate rather than trusting the arithmetic blindly; unlike the Tamil
Nadu version of this pipeline, this figure hasn't been independently
confirmed against your actual output file here). Row 1 (`sunrise`, GHI=0)
being physically correct, not a bug, still applies.

The 45-point count itself **is** confirmed from your own `00a_build_population_grid.py`
run (population points covering ~87.5% of state population) and from
`02b_build_daily_aggregates.py`'s printed output (`Points: 45`,
`usable_days=3653` per point, `164,385` point-days total — that number is
`45 x 3653`, i.e. every point had a usable day for essentially the entire
2016-2025 span).

---

## `02b_build_daily_aggregates.py` — Phase 2, Repair 1

This is the single most important gap identified in plan v3.0 (Section 4.3,
"the repair that cannot be skipped"). `climate_uttarakhand_points.csv` keeps
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

- `daily_aggregates_uttarakhand.csv` — one row per (point_id, date), true
  daily-integral values.
- `tier2_signature_uttarakhand.csv` — one row per point_id, the point-level
  Tier-2 indices, ready for `04b` to merge in.

**Confirmed run** (your own terminal output): `Points: 45`, all 45
processed with 0 skipped, `usable_days=3653` for every sampled point shown
in the log, `164,385` total point-days aggregated. That coverage number is
a good sign — it means the "≥20 of 24 hours present" threshold this script
uses wasn't a real bottleneck for your NASA POWER data.

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

## `04_preprocess_uttarakhand.py` — Phase 2, Preprocessing & QC (unchanged, already correct)

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

**Outputs:** `uttarakhand_cleaned_physical.csv` (feeds `04b` and `04c`),
`uttarakhand_cleaned_scaled.csv` + `scalers.pkl` (later ML/DRL use only),
`qc_report.txt`, correlation/VIF/Yeo-Johnson CSVs, two diagnostic PNGs.

**With only 45 points**, a couple of the QC steps deserve a slightly more
skeptical read than they would with Tamil Nadu's 133: step 4's spatial-zone
imputation fallback (a throwaway KMeans on lat/lon) will produce noticeably
coarser zones with 45 points to group, and step 11's VIF report is
computed over fewer independent spatial samples — neither invalidates the
pipeline, but both are worth a sentence in your methodology acknowledging
the smaller point count.

---

## `04c_postprocess_plots.py` — post-cleaning QA (unchanged, already correct)

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

## `04b_climate_signature.py` — Phase 3, Climate Signature Construction

Reads **both** `04`'s physical-units output (Tier 1: sun-event indices)
**and** `02b`'s `tier2_signature_uttarakhand.csv` (Tier 2: true daily-integral
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
  geography — it would guarantee the "result" is just a map of
  Uttarakhand's shape), the raw PCA_BLOCK columns (redundant with PC1..PCn),
  and every `_proxy`/`_true` column (only the canonical version clusters).
- **Standardization** (z-scores) of the final clustering matrix.

**Outputs:** `climate_signature_uttarakhand.csv` (one row per point — this
is what `05` clusters), `pca_loadings.csv`, correlation heatmap, per-index
distribution plot, an Uttarakhand map colored by true daily GHI / monsoon
index.

**elevation note — this is a real limitation here, not a footnote:**
`02_combine_uttarakhand.py` uses a flat **1200m** proxy for every point's
solar-geometry calculations, not real per-point elevation. Unlike Tamil
Nadu (coastal plain to the Nilgiris at ~2,600m, a limitation the parallel
Tamil Nadu doc explicitly downgrades to "matters less here"), Uttarakhand's
populated terrain genuinely spans roughly 200m (Terai plains near
Udham Singh Nagar/Haridwar) to 2000m (hill towns), and elevation drives
both solar-geometry inputs (air mass, clear-sky irradiance) and the
temperature-based indices (HDD18/CDD24, Ta_mean) directly. This is
plan v3.0's "Repair 2," written with Uttarakhand specifically in mind —
if `elev_proxy`/`HSI` show up carrying real weight in `pca_loadings.csv`
or the correlation heatmap, that's a signal this proxy is doing real work
and worth replacing with actual per-point elevation (e.g. from the SRTM
tile you likely already have cached locally, or a quick lookup against
the GADM/WorldPop rasters `00a_build_population_grid.py` already
downloaded) before finalizing clusters, rather than after.

---

## `05_cluster_uttarakhand.py` — Phase 4, Climate Regime Clustering, Uttarakhand-only

**You do not need to cluster across other states to finish Objective 1.**
The objective statement is "cluster meteorological data and identify
Top-2/Top-3 PCM candidates per climatic regime" — nothing requires those
regimes to span state boundaries. Clustering the 45 Uttarakhand points into
their own regimes (Gaussian Mixture, BIC-selected K, K-Means reported as
comparison) already gives you "different PCMs for different locations"
within Uttarakhand.

`05_cluster_regions.py` (the original multi-state script) still exists and
is already state-parameterised — if you add Rajasthan/Assam/Tamil Nadu's
Phase 3 output later, switch to it. Until then, this single-state version
produces the same output shapes (`cluster_assignments_uttarakhand.csv`
with soft membership probabilities, `cluster_profiles_uttarakhand.csv`
population-weighted per regime) so nothing downstream (Phase 5/6) needs to
change when/if you extend to more states later.

Expected K for one state, and with only 45 points to work with: probably
smaller than Tamil Nadu's 3-6 — realistically 2-4 (e.g. high-Himalaya vs.
Doon Valley vs. Terai plains). With 45 points, be conservative about K:
each additional cluster shrinks the average points-per-cluster fast, and a
GMM fit on very few points per component gets unstable. Realistic
silhouette band 0.15-0.40, same reasoning as the parallel Tamil Nadu doc —
say so in your methodology if yours comes out on the higher side, which is
more likely here given the smaller N.

---

## Quick numbering recap

```
02_combine_uttarakhand.py         -> climate_uttarakhand_points.csv          (Phase 1)
02b_build_daily_aggregates.py     -> tier2_signature_uttarakhand.csv         (Phase 2 Repair 1)
03_plots_raw.py                   -> QA plots, read-only                     (run before 04)
04_preprocess_uttarakhand.py      -> uttarakhand_cleaned_physical.csv        (Phase 2)
04c_postprocess_plots.py          -> QA plots, read-only                     (run after 04)
04b_climate_signature.py          -> climate_signature_uttarakhand.csv       (Phase 3)
05_cluster_uttarakhand.py         -> cluster_profiles_uttarakhand.csv        (Phase 4, Uttarakhand only)
```
