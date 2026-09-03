# 04 — Phase 2 Audit: Combine, Cross-Source Validation, and Quality Control

**Scripts**: `02_combine_uttarakhand.py`, `02b_build_daily_aggregates.py`, `03_plots_raw.py`,
`03b_interactive_raw_qa.py`, `04_preprocess_uttarakhand.py`, `04c_postprocess_plots.py`,
`04c_interactive_postprocess_qc.py`

**Status**: **COMPLETE.** `climate_uttarakhand_points.csv` is confirmed at **493,155 rows**;
`uttarakhand_cleaned_physical.csv` at **489,105 rows × 89 columns** with zero residual missing
values.

This file contains everything Phase 2: the merge, the Tier-2 daily-integral repair, the ERA5
de-accumulation analysis, solar geometry and derived variables, the cross-source validation result,
and the 13-step quality-control sequence.

---

# PART A — Combine and Cross-Source Validation

## A.1 Purpose

Merge two independent climate products at the same 45 points and the same sun-event instants, then
repair the one thing a 3-samples-per-day schema cannot express: true daily integrals.

## A.2 Inputs

```
data/raw/era5/points/era5_UK_points_{year}_{month}_{instant,accum}.nc
data/raw/nasapower/power_{point_id}_{year}.json
data/processed/population_grid_points.csv
data/processed/suntimes.csv
```

## A.3 Processing

### ERA5 Accumulated Fields & De-accumulation — the load-bearing assumption

```python
def deaccumulate(s):
    """
    ERA5 hourly reanalysis: accumulated values reset every 12 h.
    Resets happen at hours 1 and 13 UTC (start of each forecast run).
    diff() gives increments between consecutive downloaded hours; at reset
    hours the raw value is used directly since there's no valid predecessor.
    """
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    diff = s.diff()
    reset_mask = s.index.hour.isin([1, 13])
    diff[reset_mask] = s[reset_mask]
    return diff.clip(lower=0)
```

Applied to three fields:

```python
df["GHI"]           = (deaccumulate(df["ssrd"]) / 3600).clip(0)
df["LW_down"]       = (deaccumulate(df["strd"]) / 3600).clip(0)
df["precipitation"] = (deaccumulate(df["tp"])   * 1000).clip(0)
```

**The stated model.** The docstrings in `01_download_era5_uttarakhand.py` and in `deaccumulate()`
assume the MARS convention: `ssrd` is cumulative since the last forecast reset (00 Z or 12 Z), so
the true one-hour flux is `value(h) − value(h−1)`, except at h ∈ {1, 13} where the raw value *is*
the first hour of a new cycle. The special case is argued as mathematically required, not an
optimisation: "hour 13's predecessor (hour 12) belongs to a *different* 12-hour accumulation cycle,
so diffing against it would produce garbage."

`avg_sdirswrf` **bypasses `deaccumulate()` entirely** — only `clip(0)` — consistent with it being a
mean-rate field.

**This assumption is not verified anywhere in `era5-uttarakhand/`, and three independent committed
artefacts indicate the resulting `era5_GHI` is far below physical expectation.**

#### Evidence 1 — raw event profile, before any cleaning

`data/plots/raw/B_event_profile.png` (from `03_plots_raw.py`, run directly on the merged CSV):

| Sun event | Mean `era5_GHI` (W/m²) | Mean `era5_T_amb` (°C) |
|---|---|---|
| sunrise | ≈ 1 | ≈ 15.3 |
| **noon** | **≈ 61** | ≈ 22.8 |
| sunset | ≈ 19 | ≈ 21.7 |

The **timezone check passes** — noon is the peak, which is exactly what check B exists to verify.
But a mean solar-noon GHI of ≈ 61 W/m² at 28.9–30.6 °N is roughly an order of magnitude below any
clear-sky-plus-cloud climatology.

#### Evidence 2 — cross-source disagreement, with two clean controls

`data/plots/raw/C_era5_vs_power_stats.csv`, computed over every row:

| Variable | n | MBE (ERA5 − POWER) | RMSE | Pearson *r* |
|---|---|---|---|---|
| **GHI (W/m²)** | 493,155 | **−211.406** | **369.323** | **0.4321** |
| Clear-sky GHI (W/m²) | 493,155 | +5.314 | 66.360 | **0.9923** |
| T_amb (°C) | 492,936 | −0.089 | 3.695 | 0.9020 |
| RHum (%) | 493,155 | +11.383 | 20.362 | 0.7399 |
| Wind speed (m/s) | 493,155 | −1.141 | 1.703 | 0.5396 |

The diagnostic value is in the **contrast between rows 1 and 2**. `era5_GHI_clearsky` is not an
ERA5 field at all — it is pvlib's Ineichen model evaluated locally at the same coordinates, the
same instants and `altitude = 1200 m` — and it agrees with NASA POWER's independently modelled
clear-sky product to within **+5.3 W/m² at r = 0.9923**. The coordinates, the sun-event time
matching, the nearest-hour lookup, the ERA5 grid snapping and the altitude assumption's effect on
the clear-sky model are therefore **all confirmed correct**. `era5_T_amb`, an instantaneous field
that never touches `deaccumulate()`, agrees to within 0.09 °C at r = 0.902.

Only the **all-sky ERA5 GHI** — the one solar quantity that passes through `deaccumulate()` —
disagrees, and it disagrees by −211 W/m² at r = 0.432. MBE −211.4 W/m² is ten times ERA5's own
whole-file mean of 21.03 W/m². An r of 0.432 says the two series differ in *shape*, not merely by
an offset.

#### Evidence 3 — downstream magnitudes

- Cleaned whole-file `era5_GHI`: mean **21.03 W/m²**, std 37.94, max **702.74 W/m²**.
- Per-cluster noon `GHI_mean`: **≈ 44.5 – 55.1 W/m²**.
- Per-cluster `GHI_daily_kWh_proxy` (half-sine daily integral from noon GHI):
  **≈ 0.33 – 0.43 kWh/m²/day**, against a physically expected several kWh/m²/day.

#### Evidence 4 — a second fingerprint on the other de-accumulated field

`era5_LW_down` = `deaccumulate(strd)/3600`. `04`'s physical bound is `[50, 600]` W/m², and
`data/plots/post_preprocess/C_qc_flag_counts.csv` records **363,525 values (73.7 % of all rows)**
below it. A 50 W/m² floor is far below any plausible surface downwelling longwave value — clear
cold nights are still ~150–250 W/m² — so values falling under it is itself evidence that this
column is depressed in the same way `era5_GHI` is.

#### What can and cannot be concluded

**Can be concluded from `era5-uttarakhand/` alone:**
- `era5_GHI` disagrees with NASA POWER by MBE −211.4 W/m² at r = 0.432, while every
  non-accumulated and locally-computed field agrees well.
- The anomaly is present in the **raw** merged data, so it originates in
  `02_combine_uttarakhand.py` or upstream — **not** in `04_preprocess_uttarakhand.py`.
- The only transformation applied to `ssrd` and `strd` but not to the agreeing fields is
  `deaccumulate()`.
- The pipeline **detected** the disagreement and **never acted on it**.

**Cannot be concluded:** the exact mechanism. Determining whether the CDS request configuration
used here returns cumulative-since-reset values (in which case `diff()` is right) or per-hour
accumulations (in which case `diff()` destroys most of the signal) requires opening one of the
`data/raw/era5/points/*_accum.nc` files. **Those are git-ignored and not in this repository**, so
that check could not be performed as part of this audit.

**Recommended verification, one command, no re-download.** Open any local
`era5_UK_points_2020_06_accum.nc`, extract `ssrd` for a single grid node across the downloaded
hours of one day, and check whether consecutive values **increase monotonically within each 12-hour
window** (→ cumulative, `diff()` correct) or whether each value is independently of order
10⁵–10⁶ J/m² (→ per-hour accumulation, `diff()` wrong, and the fix is a stateless non-negative clip
with no differencing). Compare the resulting W/m² against `power_ALLSKY_SFC_SW_DWN` for the same
instant.

### `02_combine_uttarakhand.py` — the merge/physics script

Four steps, from the docstring:

1. Per point: nearest-neighbour snap to the ERA5 grid, concatenate the full instant+accum hourly
   series across all years, de-accumulate, compute solar geometry.
2. For each `(point_id, date, event)` row in `suntimes.csv`, pick the ERA5 hourly value nearest in
   time to that event's exact UTC timestamp.
3. Same nearest-hour lookup against that point's cached NASA POWER hourly series.
4. Merge into one row per point/date/event and stream-write the output CSV.

Configuration:

```python
DEFAULT_ALT_M   = 1200    # "Uttarakhand is mountainous; populated zones range roughly 200-2000m"
MAX_MATCH_HOURS = 3       # reject a nearest-hour match farther than 3 h from the event
```

**NetCDF handling.** `open_nc()` tries engines `netcdf4` → `scipy` → `h5netcdf`, then falls back to
`mask_and_scale=False, decode_cf=False, decode_times=False` ("Python 3.14 safe"). `safe_values()`
re-applies CF `scale_factor` / `add_offset` / `_FillValue` manually when that fallback was used.
`decode_time()` handles both `valid_time` and `time` coordinates.

**Unit conversions:**

| ERA5 field | Raw unit | Operation | Output |
|---|---|---|---|
| `ssrd` | J/m² (accum) | `deaccumulate() / 3600` | `era5_GHI` (W/m²) |
| `strd` | J/m² (accum) | `deaccumulate() / 3600` | `era5_LW_down` (W/m²) |
| `tp` | m (accum) | `deaccumulate() × 1000` | `era5_precipitation` (mm) |
| `t2m`, `d2m` | K | `− 273.15` | `era5_T_amb`, `era5_T_dew` (°C) |
| `t2m` + `d2m` | — | Magnus, clipped 0–100 | `era5_RHum` (%) |
| `u10`, `v10` | m/s | `√(u²+v²)`, `(deg(atan2(u,v))+360) mod 360` | `era5_W_spd`, `era5_W_dir` |
| `sp` | Pa | `/ 100` | `era5_P_atm` (hPa) |
| `tcc` | 0–1 | pass-through | `era5_cloud_cover` |
| `msdwswrf`/`fdir`/`msdrswrf` | see A.7 | `clip(0)` only | `avg_sdirswrf` → `era5_DNI` |

**In-script bounds applied before Phase 2 QC ever sees the data:**

| Rule | Note |
|---|---|
| `GHI < 0 → 0` | redundant with `deaccumulate`'s own clip |
| `GHI > 1400 → NaN` | never fires — observed max is 702.74 |
| `T_amb < −5 → NaN`, `T_amb > 60 → NaN` | **narrower than `04`'s `BOUNDS` (−30…55 °C)** |
| `RHum.clip(0, 100)` | silent clip |

The `T_amb < −5 °C` cut is Uttarakhand-relevant: sub-−5 °C high-altitude winter sunrise
temperatures are physically real for this state and are discarded at the merge step, **before any
QC accounting sees them**. The cleaned file's `era5_T_amb` minimum is exactly **−5.00 °C** — the
fingerprint of this rule.

**Temporal matching.** `nearest_row()` rejects any match farther than 3 h and is applied
independently to each source; a row is written if *either* matched. **The actually-matched
timestamp is not persisted** — only the requested `time_utc` — so per-row match quality is
unauditable from the output.

### `02b_build_daily_aggregates.py` — Tier-2 daily integrals (NASA POWER only)

The docstring is explicit that this is not optional polish:

> `climate_uttarakhand_points.csv` keeps only 3 rows/day … Several signature indices genuinely
> cannot be computed from three instantaneous samples: the true daily GHI energy integral, the
> true diurnal temperature range (Tmax-Tmin, not noon-sunrise), heating/cooling degree-days from a
> true daily mean, cloudy-day fraction, and the longest consecutive-cloudy-day run.

`README_PREPROCESSING.md` calls it "the single most important gap identified in plan v3.0 (Section
4.3, 'the repair that cannot be skipped')."

**Cost: zero new downloads.** It re-reads the full hourly POWER cache `01b` already wrote.

```python
KT_CLOUDY_THRESHOLD = 0.35     # same threshold 04b uses for CCI/cloudy_frac
MIN_HOURS_PER_DAY   = 20       # else the day is dropped, not averaged short
```

Daily outputs: `GHI_daily_kWh`, `GHIcs_daily_kWh`, `kt_daily` (clipped [0, 1.5], guarded at
`GHIcs > 0.05`), `Ta_mean_true`/`Ta_max_true`/`Ta_min_true`, **`DTR_true` = true Tmax − Tmin**,
`RH_mean_true`, `wind_mean_true`.

Point-level Tier-2 outputs: `n_days_used`, `GHI_daily_kWh_mean`, `kt_daily_mean`, `kt_daily_std`,
`SAI_true`, `cloudy_frac_true`, `CCI_true`, `DTR_true_mean`, `Ta_mean_true`, `Ta_p95_true`,
`Ta_p05_true`, `HDD18_true`, `CDD24_true`, `RH_mean_true`, `wind_mean_true`, `seasonality_true`.

Two details for a methodology write-up:

- **`Ta_p95_true` / `Ta_p05_true` are percentiles of the daily *mean* temperature**, not of the
  true daily maxima/minima — even though `Ta_max_true`/`Ta_min_true` are already computed in the
  daily table. They are therefore not "design-day extremes" in the usual sense.
- **`CCI_true` is a run length in days**, not an index on [0, 1] — the longest consecutive cloudy
  run, via a shift-cumsum run-ID and `transform("sum")`.

**Recorded run result**, quoted in `README_PREPROCESSING.md` from the author's own terminal output:

> **Confirmed run**: `Points: 45`, all 45 processed with 0 skipped, `usable_days=3653` for every
> sampled point shown in the log, `164,385` total point-days aggregated.

45 × 3,653 = 164,385 exactly. The ≥ 20-of-24-hours threshold excluded **no** days — "a good sign —
it means the … threshold this script uses wasn't a real bottleneck for your NASA POWER data."

**Stated limitation**: `01b`'s `POWER_PARAMETERS` never included `PRECTOTCORR`, so `monsoon_index`
is **not** upgraded to a true Tier-2 index and remains an ERA5 3×/day precipitation-fraction proxy
permanently. The docstring gives the one-line fix and flags it as "optional, not required for
Objective 1 to stand up." `NEXT_STEPS.md` explicitly instructs *not* to fix it now.

### `03_plots_raw.py` / `03b_interactive_raw_qa.py` — raw QA, before any cleaning

Read-only, run directly on `02`'s output. Six checks, mapped to plan Table 9:

| Check | Purpose | Output |
|---|---|---|
| A | Point map — is the sample actually population-weighted and covering the state? | `A_point_map.png` / `.html` |
| B | Event profile — **Table 9 check #2 (timezone)**: GHI/T_amb must peak at "noon" | `B_event_profile.png` / `.html` |
| C | ERA5 vs NASA POWER — **Table 9 check #7**: MBE/RMSE/*r* per variable | `C_era5_vs_power.png` / `.html` **+ `C_era5_vs_power_stats.csv`** |
| D | Missing-data heatmap per point × variable | `D_missing_heatmap.png` / `.html` |
| E | Seasonal boxplots against known climatology | `E_seasonal_boxplots.png` / `.html` |
| F | Multi-year trend — a step-change in one year would flag a download/unit bug | `F_yearly_trend.png` / `.html` |

All twelve outputs are committed. `C_era5_vs_power_stats.csv` is committed in both the static and
interactive variants with identical values, confirming both scripts ran on the same data.

## A.4 Code mapping

| Concern | Function / constant | File |
|---|---|---|
| NetCDF opening with engine fallbacks | `open_nc()`, `safe_values()`, `decode_time()` | `02` |
| Spatial snapping | `extract_nearest()` | `02` |
| Kelvin → °C | `kelvin_to_c()` | `02` |
| Relative humidity | `compute_rh()` (Magnus, a = 17.625, b = 243.04) | `02` |
| Accumulated → flux | `deaccumulate()` | `02` |
| Solar geometry | `compute_solar()` | `02` |
| Unit conversions + in-script bounds | `apply_unit_conversions()` | `02` |
| POWER cache load, `-999 → NaN` | `load_power_series()` | `02` |
| Nearest-in-time match | `nearest_row()`, `MAX_MATCH_HOURS = 3` | `02` |
| Per-point orchestration | `process_point_era5()`, `process_point()` | `02` |
| Daily integrals | `daily_from_hourly()` | `02b` |
| Point-level Tier 2 | `build_tier2_row()` | `02b` |

## A.5 Temporal Processing in the Merge

`nearest_row()` is applied independently to the ERA5 series and the POWER series with a 3-hour
rejection window. **Observed: zero rows lost** — the output has exactly 493,155 rows against a
theoretical maximum of 45 × 3,653 × 3 = 493,155. No `(point, date, event)` combination failed to
find both an ERA5 and a POWER reading within 3 hours across all 45 points and all 3,653 days.

That is a genuinely good coverage result. It also means the tolerance was never exercised as a
filter, so it provides no evidence about *typical* match quality — and because the matched
timestamp is not written, the offsets cannot be recovered. **A low-cost fix**: persist
`era5_matched_time_utc` and `power_matched_time_utc` alongside the requested `time_utc`.

Rows with a valid `era5_T_amb` are 492,936, i.e. **219 rows (0.044 %)** lost `era5_T_amb` to the
`< −5 °C` / `> 60 °C` cut or to a missing match.

Duplicate handling: `df[~df.index.duplicated(keep="first")]` is applied to the instant frame, the
accum frame, the joined frame, and the POWER frame — four separate de-duplications before any row
is written.

## A.6 Solar Geometry (why it's computed this way)

```python
def compute_solar(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt, tz="UTC")
    sp = loc.get_solarposition(times)                    # no explicit method=
    cs = loc.get_clearsky(times, model="ineichen")
    df["SZA"], df["solar_azimuth"] = sp["zenith"], sp["azimuth"]
    df["ETR"]          = pvlib.irradiance.get_extra_radiation(times)
    df["GHI_clearsky"] = cs["ghi"]
    ...
```

**Solar-position method is not pinned.** `get_solarposition(times)` relies on the installed pvlib
version's default, whereas `00b_build_suntimes.py` *does* pin `method="spa"`. This is an
inconsistency inside one pipeline and a reproducibility gap: a pvlib version change could shift
`SZA`, `solar_azimuth`, `is_daytime`, and the `SZA ≥ 90` night-masking threshold in `04`'s step 2.
Pinning `method="spa"` in `compute_solar()` closes it at zero analytical cost.

**Clear-sky model: Ineichen with pvlib's default Linke-turbidity climatology**, no site-specific
turbidity. This choice is **independently validated by the pipeline's own statistics** — see the
+5.3 W/m² / r = 0.9923 result in A.3, which is the pipeline's strongest positive finding.

*Uttarakhand caveat:* the default Linke climatology is a coarse global lookup, and this state's
aerosol environment is strongly elevation-dependent — Indo-Gangetic-plain haze in the foothills
versus clean air above the boundary layer — while one 1200 m altitude and one climatological
turbidity are applied to all 45 points. The r = 0.9923 agreement is against another *model*, so it
confirms mutual consistency rather than absolute accuracy.

**Altitude: 1200 m for all 45 points**, feeding the Ineichen air-mass/turbidity correction. A Terai
point at ~200 m and a hill point at ~2000 m receive identical clear-sky curves. The value is **not
written to the output rows**, so the assumption is invisible in the data.

**Night-time handling and division-by-zero protection:**

| Guard | Rule |
|---|---|
| Clear-sky floor | `CSI = 0` wherever `GHI_clearsky ≤ 10 W/m²` |
| CSI ceiling | `clip(0, 1.5)` |
| Zenith clip for DNI division | `cos_z = cos(radians(SZA.clip(0, 89.9)))` — **only** for the DNI/DHI arithmetic; the unclipped `SZA` is what is written |
| Fallback DNI guard | `np.where(cos_z > 0.05, GHI/cos_z, 0)` |
| Night masking (in `04`) | all five solar fields forced to `0.0` where `era5_SZA ≥ 90°` |

A `CSI` of exactly 0 is **three-way ambiguous** in the output: true darkness, clear-sky below the
10 W/m² floor, or night-masked in `04`.

`04`'s night-masking rationale is schema-specific and well reasoned: "even though every row IS a
sun-event, the NEAREST-HOUR match … can land a few hours off true sun position (see 02_combine's
`MAX_MATCH_HOURS=3`), so a 'sunrise' row can occasionally have SZA > 90."

**`ETR` is computed and discarded** — it is not in `ERA5_OUTPUT_VARS`, so it never reaches the
combined CSV and no downstream script uses it.

**Latitude context** (arithmetic, not stated in the source files): the 45 points span
28.875–30.625 °N, so solar-noon zenith ranges from ~6–8° at the June solstice to ~52–54° at the
December solstice, and day length swings from ~10 h to ~14 h. `04b`'s half-sine daily-integral
proxy uses the actual `sunset − sunrise` interval rather than a nominal 12 h, which is the right
choice for a swing that large.

## A.7 Solar-Derived Variables (construction & assumptions)

### GHI

`GHI = deaccumulate(ssrd)/3600`, clipped ≥ 0, `> 1400 → NaN`. The pipeline's most consequential
derived variable — it feeds `CSI`, `DHI`, `cloud_opacity` and every Tier-1 solar signature index.
Observed magnitudes and the anomaly analysis are in A.3.

The `> 1400 → NaN` guard never fires: the observed maximum across 493,155 rows is 702.74 W/m².

### DNI — two-branch derivation

```python
if "avg_sdirswrf" in df.columns:
    df["DNI"] = df["avg_sdirswrf"].clip(0, 1400)                              # primary
else:
    df["DNI"] = np.where(cos_z > 0.05, df["GHI"] / cos_z, 0).clip(0, 1400)    # fallback
```

**Branch 1 (primary) — and its unit-consistency caveat.** `avg_sdirswrf` is set upstream by a
three-name matcher:

```python
fdir_col = next((c for c in df.columns if c in ("msdwswrf", "fdir", "msdrswrf")), None)
if fdir_col:
    df["avg_sdirswrf"] = df[fdir_col].astype(float).clip(0)
```

| Short name | ERA5 convention | Correct handling | What the code does |
|---|---|---|---|
| `msdwswrf` | mean-rate, W/m² | pass through | `clip(0)` ✓ |
| `msdrswrf` | mean-rate, W/m² | pass through | `clip(0)` ✓ |
| `fdir` | accumulated, J/m² | `/3600` | `clip(0)` ✗ — **would over-estimate by 3600×** |

`01_download_era5_uttarakhand.py` requests `mean_surface_direct_short_wave_radiation_flux`, which
maps to `msdwswrf` — a mean-rate field — so the no-conversion branch is almost certainly correct in
practice. **This audit could not verify the actual short name present**, because the NetCDF files
are git-ignored. Before presenting DNI as unit-validated, open one `*_accum.nc` and inspect
`ds.data_vars`.

**Branch 2 (fallback) — explicitly NOT a decomposition model.** `DNI = GHI / cos(SZA)` assumes a
zero diffuse component; it attributes all global horizontal irradiance to the beam. In cloudy
conditions it would over-estimate DNI substantially, and near the horizon it is numerically
unstable (hence the `cos_z > 0.05` guard). It should only execute if the ERA5 direct field is
absent, which `01` requests in every accum call — but **the pipeline does not record which branch
ran**, so this cannot be confirmed from the outputs.

Correct framing for a write-up: "DNI taken from ERA5's mean direct short-wave radiation flux where
available; a `GHI/cos(SZA)` closure fallback exists but is not expected to have been used" — not a
claim of decomposition-model provenance.

### DHI — closure residual

```python
df["DHI"] = (df["GHI"] - df["DNI"] * cos_z).clip(0)
```

**Not independently derived.** The closure equation `GHI = DNI·cos(SZA) + DHI` is satisfied *by
construction*, so agreement with it is evidence of nothing, and any error in GHI or DNI propagates
entirely into DHI. Because GHI is anomalously low while DNI comes from a separate un-deaccumulated
field, the residual will frequently be negative and clipped to zero. `04`'s `BOUNDS` records **no**
`era5_DHI` flags, consistent with the column being dominated by clipped zeros.

**`DHI` is not used by `04b_climate_signature.py`.** Its only appearance downstream is in `04`'s
correlation and VIF reports. It should not be presented as a measured or modelled diffuse quantity.

### Clearness index (CSI / kt)

`CSI = GHI / GHI_clearsky` clipped [0, 1.5], forced to 0 below a 10 W/m² clear-sky floor and again
where `SZA ≥ 90°`.

Because GHI is anomalously low while `GHI_clearsky` is validated as correct, **`era5_CSI` is
correspondingly depressed**, and with it `kt_mean_proxy`, `kt_std_proxy`, `cloudy_frac_proxy`,
`CCI_proxy`, `SAI_proxy` and `era5_cloud_opacity`.

**The two-tier design contains this.** The canonical `kt_mean`, `kt_std`, `SAI`, `cloudy_frac`,
`CCI` and `GHI_daily_kWh` columns that reach the clustering matrix come from **NASA POWER via
`02b`**, and all the `_proxy` variants are excluded from the clustering matrix by `04b`'s suffix
rule. This is a real architectural benefit of the Tier-2 repair and should be reported as such.

**The one exception is `GHI_mean`** (mean noon `era5_GHI`), which carries no `_proxy` suffix and has
no Tier-2 override, so it enters the clustering matrix carrying the anomaly.

### Cloud cover, precipitation, longwave

| Variable | Derivation | Bound in `04` | Flags |
|---|---|---|---|
| `era5_cloud_cover` | `tcc` pass-through | [0, 1] | 0 |
| `era5_precipitation` | `deaccumulate(tp) × 1000` mm | [0, 200] | 0 |
| `era5_LW_down` | `deaccumulate(strd)/3600` | **[50, 600]** | **363,525 (73.7 %)** |

Cleaned `era5_precipitation`: mean **0.08 mm**, std 0.44, max **45.85 mm** — per-instant values at
three sun-event samples per day, not daily totals. It also passes through `deaccumulate()`, so it
carries the same unverified assumption; its only downstream use is `monsoon_index`, a **ratio** in
which a uniform multiplicative error would cancel (a non-uniform one would not).

`era5_LW_down` is not used by `04b`; its impact is limited to the correlation/VIF reports. It is
retained here as the second independent fingerprint of the de-accumulation problem.

### Physical bounds applied to derived solar variables

| Variable | Bound | Where | Flags observed |
|---|---|---|---|
| `GHI` | `<0 → 0`; `>1400 → NaN` | `02` | not counted |
| `GHI`, `DNI`, `GHI_clearsky` | `[0, 1400]` | `04` | 0 each |
| `DHI` | `[0, 900]` | `04` | 0 |
| `CSI` | `[0, 1.5]` | `04` | 0 |
| `LW_down` | `[50, 600]` | `04` | **363,525** |
| all five solar fields | forced `0.0` where `SZA ≥ 90°` | `04` | logged to `qc_report.txt`, not committed |

Not one of the five directly-solar columns triggered a physical-bounds flag. Read with the observed
magnitudes, that says the values sit comfortably *inside* their ranges because they are too small,
not because they are correct.

## A.8 Cross-Source Validation Decision — there isn't one

| Component | Status in `era5-uttarakhand/` |
|---|---|
| Cross-source statistics computed | **Yes** — `03` check C and its interactive twin |
| Statistics persisted | **Yes** — `C_era5_vs_power_stats.csv`, committed in both variants |
| Dedicated agreement-analysis script | **No.** No `03b_agreement_analysis*.py` of any name exists. |
| Bias decision file | **No.** No file records a BACKBONE / quantile-map decision. |
| Threshold-based decision logic | **No.** |
| Bias-correction / quantile-mapping step in `04` | **No.** The 13-step sequence contains no such step. |

**What the pipeline says it will do**, in three separate places:

`03_plots_raw.py`'s docstring: "quantifies exactly how much the two sources disagree, per variable,
**before you decide how (or whether) to bias-correct in 04.**"

`README.md`'s run-order block: "STOP AND LOOK at 03's output before continuing. … **check C: large
ERA5-vs-POWER MBE is expected and gets addressed in 04**."

`README_PREPROCESSING.md`: "If B shows noon isn't the peak, or C shows a large systematic MBE,
**stop and fix that before running `04`** — these are exactly the 'most silent failures at this
stage' the plan doc warns about."

**Check C shows a large systematic MBE. Nothing in `04` addresses it. The gate the source files
describe was not enforced.**

### Variable pairs compared

| ERA5 column | NASA POWER column |
|---|---|
| `era5_GHI` | `power_ALLSKY_SFC_SW_DWN` |
| `era5_GHI_clearsky` | `power_CLRSKY_SFC_SW_DWN` |
| `era5_T_amb` | `power_T2M` |
| `era5_RHum` | `power_RH2M` |
| `era5_W_spd` | `power_WS10M` |

Statistics are pooled over all rows — **no stratification by season, event, point or year**. There
is therefore no evidence about whether the GHI disagreement is uniform across the year or
concentrated in particular months. (Check F, the multi-year trend plot, is committed as a figure
but the statistics CSV carries no per-year breakdown.)

### Reading the other three rows

**RHum (+11.4 %, r = 0.740).** ERA5's relative humidity here is *derived* — the Magnus formula on
`t2m` and `d2m` — while POWER's `RH2M` is its own product. An 11-point offset between the two is a
plausible model-versus-model disagreement rather than a processing artefact, but it is
**unaddressed and reaches the clustering matrix**: `04b` takes `RH_mean` from the ERA5 side (there
is no `CANON_MAP` entry for it), and `RH_mean` is a `PCA_BLOCK` member and the basis of `HSI`.
`02b` computes an unused `RH_mean_true` from POWER.

**Wind (−1.14 m/s, r = 0.540).** Both products are nominally 10 m winds, so the heights match; the
disagreement reflects differing surface-roughness and orographic treatments over complex terrain,
which is exactly where 45 Himalayan-foothill points would show it. The mean disagreement is ~80 % of
the cleaned ERA5 mean of 1.43 m/s. `wind_mean` reaches the clustering matrix from the ERA5 side, and
`02b`'s `wind_mean_true` is likewise unused.

### Which side each clustering-matrix column comes from

| Signature column | Source | Affected by a measured disagreement? |
|---|---|---|
| `GHI_daily_kWh`, `kt_mean`, `kt_std`, `SAI`, `cloudy_frac`, `CCI` | NASA POWER (Tier 2) | No |
| `DTR`, `Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `seasonality` | NASA POWER (Tier 2) | No |
| `GHI_mean` | ERA5 noon GHI, no override | **Yes — the −211 W/m² problem** |
| `RH_mean` | ERA5 (Magnus-derived), no override | **Yes — +11.4 %** |
| `HSI` | ERA5 (`RH_mean` × dew-point-depression fraction) | **Yes** |
| `wind_mean` | ERA5, no override | **Yes — −1.14 m/s** |
| `monsoon_index` | ERA5 precipitation ratio, permanently proxy | Unquantified (no POWER precipitation) |
| `elev_proxy` | ERA5 `P_atm`, 37 % imputed | Not compared (POWER pressure not downloaded) |

**A concrete, cheap improvement:** adding `RH_mean` and `wind_mean` to `04b`'s `CANON_MAP` would
swap two ERA5-side columns for already-computed NASA POWER Tier-2 values at the cost of two
dictionary entries.

## A.9 Mathematical operations

Magnus RH · vector wind magnitude/direction · first-difference de-accumulation with 12-hourly reset
handling · pvlib SPA solar position · Ineichen clear-sky · clearness-index ratio with a low-light
floor · beam/diffuse closure · nearest-neighbour 1-D `argmin` snapping · nearest-in-time index
lookup with a tolerance · daily summation and min/max/mean aggregation · degree-day accumulation ·
run-length encoding for consecutive cloudy days · coefficient of variation for seasonality.

## A.10 Literature support

**None present in the source files for Phase 2.** `02_combine_uttarakhand.py` names `pvlib` and the
string `"ineichen"` but cites no paper; there is no ERA5 product citation, no NASA POWER citation,
no SPA citation, no clear-sky-model citation, and no decomposition-model reference anywhere in
`era5-uttarakhand/`. See `11_LITERATURE_MAPPING.md`.

## A.11 Validation

| Check | Result |
|---|---|
| Noon peaks GHI and T_amb (timezone) | **PASS** — 61 vs 1 vs 19 W/m²; 22.8 vs 15.3 vs 21.7 °C |
| Full point/day/event coverage | **PASS** — 493,155 = 45 × 3,653 × 3 exactly |
| Clear-sky cross-source agreement | **PASS** — MBE +5.3 W/m², r = 0.9923 |
| Temperature cross-source agreement | **PASS** — MBE −0.089 °C, r = 0.902 |
| All-sky GHI cross-source agreement | **FAIL** — MBE −211.4 W/m², r = 0.432, unaddressed |
| Humidity cross-source agreement | **MARGINAL** — MBE +11.4 %, r = 0.740, unaddressed |
| Wind cross-source agreement | **MARGINAL** — MBE −1.14 m/s, r = 0.540, unaddressed |
| Tier-2 daily coverage | **PASS** — 45/45 points, 0 skipped, 164,385 point-days |

## A.12 Outputs

| File | Rows | Committed? |
|---|---|---|
| `data/processed/climate_uttarakhand_points.csv` | **493,155** × 36 cols | No (git-ignored) |
| `data/processed/daily_aggregates_uttarakhand.csv` | ≤ 164,385 | No |
| `data/processed/tier2_signature_uttarakhand.csv` | ≤ 45 | No |
| `data/plots/raw/*.png` + `C_era5_vs_power_stats.csv` | 6 + 1 | **Yes** |
| `data/plots/raw_interactive/*.html` + `C_era5_vs_power_stats.csv` | 6 + 1 | **Yes** |

## A.13 Dependencies

`xarray`, `netCDF4` (and optionally `scipy`/`h5netcdf` as engine fallbacks), `pvlib`, `pandas`,
`numpy`; `matplotlib` + `seaborn` for `03`; `plotly` + `folium` + `branca` for `03b`.

---

# PART B — Phase 2: Preprocessing & Quality Control

**Script**: `04_preprocess_uttarakhand.py` (13 steps), with post-hoc QA by
`04c_postprocess_plots.py` and `04c_interactive_postprocess_qc.py`.

## B.1 Purpose and the schema note it leads with

> The old `04_preprocess_uttarakhand.py` assumed one row per `(city, hour)` — a continuous 24 h/day
> series. This dataset is one row per `(point_id, date, event)` with `event ∈ {sunrise, noon,
> sunset}` — **3 samples/day, not 24**. Every "rolling"/"lag"/"delta" concept below is therefore
> redefined over EVENT OCCURRENCES within a `(point_id, event)` group, sorted by date — e.g.
> "lag7" = the same sun-event 7 days earlier, not 7 hours earlier.

This is the single most important fact to carry into any methodology write-up about this script.

## B.2 Steps 1–3b — inspection, physical validation, outlier flagging

**Step 1 — Dataset inspection.** Shape, dtype counts, duplicate `(point_id, date, event)` count
(dropping any), and the top-15 missing-% columns. Everything is appended to `report_lines` and
written to `qc_report.txt`.

**Step 2 — Physical validation.** Out-of-range values become **`NaN`, never clipped** — the in-code
comment: "matches your 'safer than clipping' rule."

| Column | Lower | Upper | Values flagged (observed) |
|---|---|---|---|
| `era5_LW_down` | **50 W/m²** | 600 W/m² | **363,525 (73.7 %)** |
| `era5_P_atm` | **850 hPa** | 1060 hPa | **182,899 (37.1 %)** |
| `era5_GHI`, `era5_DNI`, `era5_GHI_clearsky` | 0 | 1400 W/m² | 0 |
| `era5_DHI` | 0 | 900 W/m² | 0 |
| `era5_CSI` | 0 | 1.5 | 0 |
| `era5_T_amb` | −30 °C | 55 °C | 0 |
| `era5_T_dew` | −30 °C | 40 °C | 0 |
| `era5_RHum` | 0 % | 100 % | 0 |
| `era5_W_spd` | 0 m/s | 50 m/s | 0 |
| `era5_cloud_cover` | 0 | 1 | 0 |
| `era5_precipitation` | 0 mm | 200 mm | 0 |
| `era5_SZA` | 0° | 180° | 0 |
| `power_*` (5 columns) | see `02_DATA_SOURCES_AND_VARIABLES.md` | | 0 |

Counts are from `data/plots/post_preprocess/C_qc_flag_counts.csv`, which `04c` parses out of
`qc_report.txt`. Columns absent from that CSV had zero flags.

Then night masking: where `era5_SZA ≥ 90°`, all of `era5_GHI, era5_DNI, era5_DHI,
era5_GHI_clearsky, era5_CSI` are forced to `0.0`. Finally `era5_RHum` and `power_RH2M` are
hard-clipped to [0, 100].

**Step 3 — Hampel / MAD outlier flagging.**

```python
HAMPEL_WINDOW = 15;  HAMPEL_N_SIGMA = 3.0
HAMPEL_COLS   = era5_GHI, era5_T_amb, era5_RHum, era5_W_spd, era5_cloud_cover
threshold     = 3.0 × 1.4826 × rolling_MAD          # 31-occurrence centred window, min_periods=5
is_outlier   &= roll_mad > 1e-6                     # skip flat/constant stretches
```

Per `(point_id, event)` series sorted by date — the window is ±15 occurrences of the *same sun
event*, roughly ±15 days. **Policy: flag → `NaN`, never delete.**

| Column | Flagged | % of 493,155 |
|---|---|---|
| `era5_cloud_cover` | **49,519** | 10.04 % |
| `era5_GHI` | **35,559** | 7.21 % |
| `era5_W_spd` | 11,350 | 2.30 % |
| `era5_T_amb` | 9,762 | 1.98 % |
| `era5_RHum` | 8,814 | 1.79 % |
| **Total** | **114,004** | |

**Both high rates have the same cause: univariate MAD filtering misapplied to variables whose
variance is the signal.** `era5_cloud_cover` is a bounded [0, 1] strongly bimodal variable (clear or
overcast), so a rolling median sits near an extreme, the MAD is small, the 3σ threshold is tight,
and genuine clear↔overcast transitions get flagged. `era5_GHI`'s day-to-day variability at a fixed
sun event is genuinely large in the monsoon, so real cloud-driven variation is winsorised as
outliers. Excluding those two columns from `HAMPEL_COLS`, or widening `HAMPEL_N_SIGMA` for them,
is the targeted fix — clouds are weather, not errors.

**Step 3b — Yeo-Johnson skew diagnostic**, report-only, on `era5_GHI, era5_W_spd,
era5_precipitation, era5_cloud_cover, era5_T_amb`. Writes `yeo_johnson_skew.csv`. **No column is
transformed.** Values for this run are **not available in the source files**.

## B.3 Step 4 — Hierarchical imputation

Four tiers, in order, over every column in `IMPUTE_COLS` (all numerics except `lat, lon,
population, weight, grid_lat, grid_lon, month, DOY, year, season_code`):

| Tier | Method | Scope |
|---|---|---|
| (a) | `interpolate(method="linear", limit=3, limit_area="inside")` | within each `(point_id, event)` series |
| (b) | `ffill(limit=3).bfill(limit=3)` | same |
| (c) | point median → `impute_zone` median → global median | progressively coarser |
| (d) | MICE (`IterativeImputer`, `max_iter=10`, `random_state=42`, `sample_posterior=False`) | fit on a ≤ 300,000-row sample |

**The `impute_zone` grouping** is a throwaway `KMeans(n_clusters=min(8, 45), random_state=42,
n_init=10)` on `lat`/`lon` only. The script is emphatic: "this is **NOT** the Phase 4 climate
clustering, just named `impute_zone` to avoid confusion with it." With 45 points and 8 zones, each
averages 5–6 points; `README_PREPROCESSING.md` warns this "will produce noticeably coarser zones
with 45 points to group."

**How many values reached each tier is not available in the source files** — `04` logs the
`Remaining after …` counts to `qc_report.txt`, which is git-ignored, and `04c`'s parser extracts
only the `physical_bounds` and `hampel_MAD` categories.

## B.4 Steps 5–9c — validation, feature engineering, occurrence-based features

**Step 5 — Temporal validation.** Warns for any `(point_id, event)` series with fewer than
`0.99 × expected_days` rows, and re-checks for duplicate keys. Per-series warnings for this run are
**not available in the source files**.

**Step 6 — Feature engineering** (7 features):

| Feature | Definition |
|---|---|
| `era5_W_dir_sin` / `era5_W_dir_cos` | `sin/cos(radians(W_dir)) × W_spd` |
| `era5_cloud_opacity` | `1 − CSI.clip(0,1)` |
| `era5_T_depression` | `T_amb − T_dew` |
| `is_daytime` | `(SZA < 90).astype(int)` |
| `ist_hour_decimal` | `time_utc + 5 h 30 m` as a decimal hour |
| `solar_hour_angle` | `(ist_hour_decimal − 12) × 15` degrees |

Computed per row "since there's no fixed 'hour' column in this schema — each sun-event happens at a
different UTC hour depending on point and date."

**Step 7 — Lag features.** `LAG_COLS` = `era5_GHI, era5_T_amb, era5_RHum, era5_W_spd,
era5_cloud_cover, era5_CSI`; `LAG_OCCURRENCES = [1, 7, 30]`, shifted within `(point_id, event)`
groups → **18 features**.

**Step 8 — Rolling stats.** `ROLL_OCCURRENCES = [7, 30]`, trailing mean + std (`min_periods=3`, std
`fillna(0)`) → **24 features**.

**Step 9 — Delta features.** 1-occurrence `diff()` (`fillna(0)`) for `era5_T_amb`, `era5_GHI`,
`era5_cloud_cover` → **3 features**.

18 + 24 + 3 = 45, matching the "Engineered features: 45" figure in
`data/plots/verify_preprocessing/07_preprocessing_summary.png` exactly.

**Step 9c — Lag-warm-up row drop.** Rows where `era5_GHI_lag30d` is `NaN` are dropped, "before
imputation/scaling see them, rather than let step 4's imputation quietly paper over what is
actually 'this occurrence is too early in this point's series to have a 30-days-prior lag'."

**Observed: 493,155 → 489,105 rows, i.e. exactly 4,050 dropped = 45 points × 3 events × 30
occurrences.** Every group lost precisely its 30-row warm-up, with none lost anywhere else.
Retention **99.2 %**.

**Step 9b — Savitzky-Golay diagnostic.** One sample point, `event == "noon"`, the median year; raw
vs `savgol_filter(polyorder=3)` with a window of up to 31. Visual QA only — the dataframe is
untouched. Not committed.

## B.5 Steps 10–11 — correlation and VIF

Pearson + Spearman on a ≤ 50,000-row sample of daytime rows over 15 columns, and
`variance_inflation_factor` on the same sample after dropping constant columns. **Nothing is
dropped on the basis of VIF** — it is reported only. Outputs `correlation_pearson.csv`,
`correlation_spearman.csv`, `correlation_heatmaps.png`, `vif_report.csv`, none committed, so the
actual values for this run are **not available in the source files**.

Both `README_PREPROCESSING.md` and `PREPROCESSING_STEPS.md` pre-empt the VIF result: near-infinite
VIF among `GHI/DNI/DHI/CSI` is expected and **structural**, because DNI and DHI are algebraically
derived from GHI. `README_PREPROCESSING.md` adds a small-N caveat: "step 11's VIF report is
computed over fewer independent spatial samples."

## B.6 Steps 12–13 — scaling and the hard gate

**Step 12 — leakage-safe MinMax scaling.** A separate `MinMaxScaler` **per column**, fitted on the
first `TRAIN_FRAC = 0.70` of the globally date-sorted rows, applied to the whole file. Scalers
pickled to `scalers.pkl`; output to `uttarakhand_cleaned_scaled.csv`. `SKIP_SCALE` excludes
identifiers, coordinates, calendar columns, `impute_zone` and `is_daytime`.

Because the sort is date-primary and the panel is balanced, this is a true chronological cut —
training is roughly 2016-02 to 2023-01.

The physical/scaled separation is enforced by design and `04b` reads only the physical file.
`PREPROCESSING_STEPS.md` gives the reason: "the signature indices (kWh/day, HDD18, etc.) are
non-linear functions of physical values and would be silently corrupted by pre-scaling."

**Step 13 — the hard gate.**

| Check | Criterion | Verifiable from committed artefacts? |
|---|---|---|
| Physical file: zero NaN in `IMPUTE_COLS` | `== 0` | **Yes — PASS** |
| Physical file: zero Inf | `== 0` | No |
| Scaled file: **train portion** within [0, 1] | `min ≥ −1e−6`, `max ≤ 1+1e−6` | No |
| Zero duplicate `(point_id, date, event)` | `== 0` | Indirect — the exact 489,105 = 493,155 − 4,050 arithmetic is only consistent with zero duplicates |
| All 8 required columns present | present | **Yes** — all 8 appear downstream |

The gate deliberately checks only the *training* portion of the scaled file, reporting the full-file
out-of-range fraction as **informational**: val/test rows may legitimately exceed [0, 1] "if the
val/test period contains a value more extreme than anything seen in training (e.g. a record hot day
in 2024 that wasn't in the 2016-2022 training window) — that's expected, not a bug."

**The final `RESULT: n/5 checks passed` line and `qc_report.txt` itself are not committed**, so the
gate's own verdict cannot be read directly. The two checks that *can* be corroborated both pass.

## B.7 Verified Phase 2 outcome

From `data/plots/verify_preprocessing/07_preprocessing_summary.png`:

| Metric | Value |
|---|---|
| Input records | 493,155 |
| Output records | **489,105** |
| Data retention | **99.2 %** |
| Input dimensions | 36 |
| Output dimensions | **89** |
| Core climate variables | 6 |
| Engineered features | 45 |
| Completeness of all 6 core variables | **100.0 % each** |
| Rows with no missing data | **489,105 (100.0 %)** |

**Zero residual missing data** — this independently confirms step 13's first gate condition passed,
and satisfies `04c`'s check A ("should be essentially all-zero; if not, step 4's imputation didn't
cover something and step 13's hard gate should already have failed").

## B.8 Cleaned-file distributions (observed)

From `data/plots/verify_preprocessing/01_climate_distributions.png`, over all 489,105 rows:

| Column | Mean | Std | Min | Max |
|---|---|---|---|---|
| `era5_T_amb` (°C) | 20.07 | 7.69 | **−5.00** | 42.22 |
| `era5_RHum` (%) | 68.99 | 19.00 | 8.96 | 100.00 |
| `era5_W_spd` (m/s) | 1.43 | 0.75 | 0.00 | 9.08 |
| `era5_P_atm` (hPa) | 901.90 | 47.66 | **850.00** | 1001.65 |
| `era5_GHI` (W/m²) | **21.03** | 37.94 | 0.00 | **702.74** |
| `era5_precipitation` (mm) | 0.08 | 0.44 | 0.00 | 45.85 |

Three observations for a write-up:

1. **`era5_T_amb` minimum is exactly −5.00 °C** — precisely the `02_combine` cut boundary, not a
   climatological floor. Sub-−5 °C high-altitude winter sunrise values were removed at the merge
   step and then imputed.
2. **`era5_P_atm` minimum is exactly 850.00 hPa** — precisely the `BOUNDS` lower limit. 850 hPa is
   ≈ 1,450 m in a standard atmosphere, so **37.1 % of readings from the pipeline's
   higher-elevation points were destroyed and replaced by imputed values pulled toward
   lower-elevation medians.** Only *low* values were removed, so the imputation is directionally
   biased upward. The histogram is visibly multi-modal (peaks near 850–860, ~895, ~910 and
   ~965–980 hPa) — that is the real elevation stratification of the 45 points, truncated at its low
   end with a large spike on the boundary. **`elev_proxy = mean(era5_P_atm)/1013.25` is a
   `PCA_BLOCK` member and therefore feeds the clustering matrix**: the one signature index that
   encodes elevation is computed from the column this bound compresses. Of every issue in this
   pipeline, this is the one most specific to Uttarakhand.
3. **`era5_GHI` is anomalously low** — see Part A.3.

## B.9 Post-cleaning QA (`04c_postprocess_plots.py`)

Six checks, run **after** `04` on `uttarakhand_cleaned_physical.csv`. All six outputs are committed
to `data/plots/post_preprocess/`, five with interactive twins.

| Check | Purpose (from the docstring) | Output |
|---|---|---|
| A | Missing-data heatmap post-clean — "should be essentially all-zero" | `A_missing_post.png` / `.html` |
| B | Distribution sanity — "watch for imputation spikes (a suspicious mode exactly at the point/zone/global median)" | `B_distributions_post.png` / `.html` (43 MB) |
| C | Physical-bounds vs Hampel flag counts, parsed from `qc_report.txt` | `C_qc_flag_counts.png` **+ `C_qc_flag_counts.csv`** |
| D | Lag-feature sanity — GHI vs GHI-7-days-prior, "should be positive and clearly structured … not noise" | `D_lag_sanity.png` / `.html` |
| E | One point's cleaned noon-GHI series for one year with 7 d/30 d rolling means — "seasonal shape should look smooth, not flattened" | `E_point_timeseries.png` / `.html` |
| F | Post-clean correlation heatmap including the step-6 engineered features | `F_correlation_post.png` / `.html` |

`C_qc_flag_counts.csv` is the **only committed artefact anywhere in the repository that carries QC
counts**, and it carries the evidentiary weight of this entire Part B.
`04c_interactive_postprocess_qc.py` implements A, B, D, E, F only — "the qc_report.txt bar chart C
is trivial enough to leave as-is in the PNG script."

## B.10 Inputs, outputs, dependencies

**Inputs**: `data/processed/climate_uttarakhand_points.csv`.

**Outputs** (all under the git-ignored `data/preprocessed/`, none committed):
`uttarakhand_cleaned_physical.csv` (→ Phase 3), `uttarakhand_cleaned_scaled.csv`, `scalers.pkl`,
`qc_report.txt`, `correlation_pearson.csv`, `correlation_spearman.csv`, `correlation_heatmaps.png`,
`vif_report.csv`, `yeo_johnson_skew.csv`, `savitzky_golay_diagnostic.png`.

**Dependencies**: `pandas`, `numpy`, `scipy` (`stats`, `signal.savgol_filter`), `scikit-learn`
(`MinMaxScaler`, `KMeans`, `IterativeImputer`), `statsmodels` (VIF), `matplotlib`, `seaborn`;
`plotly` for `04c_interactive`.

---

# PART C — Combined Problems / Risks

Ranked by severity.

1. **`deaccumulate()`'s assumption is unverified and is associated with an order-of-magnitude GHI
   deficit.** Highest-severity open item in the pipeline. `era5_GHI` feeds `era5_CSI`, `era5_DHI`,
   `era5_cloud_opacity`, every Tier-1 solar index, and `GHI_mean` — which is in the clustering
   matrix. Three independent artefacts corroborate the anomaly; two clean controls (clear-sky GHI
   at r = 0.9923, T_amb at r = 0.902) isolate it to the de-accumulated fields.
2. **The cross-source disagreement was measured and never acted upon.** Three separate source files
   state that a large MBE must be addressed before or in `04`; no such step exists. This is the
   clearest process gap in the pipeline.
3. **`era5_P_atm`'s 850 hPa lower bound is mis-specified for Uttarakhand** and destroyed 37.1 % of
   the column one-sidedly, in the exact variable `elev_proxy` is built from. State-specific, and
   the highest-priority QC fix.
4. **`era5_LW_down`'s 50 W/m² bound destroyed 73.7 %** of that column. Harmless downstream, but a
   second independent fingerprint of the same de-accumulation issue.
5. **The Hampel filter flagged 10.0 % of `era5_cloud_cover` and 7.2 % of `era5_GHI`** — a known
   weakness of univariate MAD filtering on bounded bimodal and high-variance-by-nature variables.
   114,004 values across five columns were replaced by imputation.
6. **Imputed and flagged cells are unmarked in the output.** With 114,004 Hampel-NaN'd values plus
   546,424 bounds-NaN'd values all imputed and unlabelled, a consumer of
   `uttarakhand_cleaned_physical.csv` cannot distinguish measured from reconstructed values. Adding
   `{col}_imputed` booleans would cost little and would let `09`'s caveat text be specific. (The
   *PCM* database does carry `*_imputed` flags; the climate data does not.)
7. **RHum's +11.4 % and wind's −1.14 m/s offsets reach the clustering matrix** while `02b`'s
   already-computed `RH_mean_true` and `wind_mean_true` sit unused. A two-line `CANON_MAP` fix.
8. **`avg_sdirswrf`'s three-name matcher applies one unit convention to three fields** — a latent
   3600× hazard, low-probability given what `01` requests, but unverified.
9. **`get_solarposition()`'s method is not pinned** in `compute_solar()` while it *is* pinned in
   `00b`. One-line reproducibility fix.
10. **Bounds applied in `02` are narrower than those in `04` and are counted nowhere** — the
    `T_amb < −5 °C` cut in particular is state-inappropriate.
11. **`CSI = 0` is three-way ambiguous** (true darkness / clear-sky floor / night mask).
12. **DHI is a closure residual with no independent basis** and should not be presented as a
    modelled diffuse quantity.
13. **Matched timestamps are not persisted**, so per-row temporal match quality is unauditable.
14. **Cross-source statistics are pooled, not stratified** by season, event, point or year.
15. **`ETR` is computed and discarded.**
16. **None of the QC report artefacts are committed** — `qc_report.txt`, `vif_report.csv`,
    `yeo_johnson_skew.csv`, the correlation CSVs and `pca_loadings.csv` are all git-ignored, so the
    step-13 verdict and every diagnostic table are uncheckable from this repository.
    `C_qc_flag_counts.csv` is the sole exception.

---

# PART D — Combined Status

**Phase 2 is COMPLETE and its structural results are strong.**

What went right, and is worth reporting positively:

- **Full coverage with zero loss at the merge**: 493,155 rows = 45 × 3,653 × 3 exactly. No
  `(point, date, event)` failed the 3-hour match on either source.
- **The timezone/sun-event design works**: check B confirms noon peaks both GHI and T_amb.
- **Clear-sky modelling is independently corroborated** at r = 0.9923 / MBE +5.3 W/m² against NASA
  POWER, validating coordinates, timing, grid snapping and the altitude assumption's effect on the
  Ineichen model in one number.
- **The Tier-2 repair delivered**: 45/45 points, 0 skipped, 164,385 point-days, and it insulated
  the clustering matrix's entire temperature and solar block from the ERA5 GHI problem.
- **Cleaning is surgical**: 99.2 % retention, with the only losses being exactly the 4,050-row
  structural lag warm-up, and zero residual missing values afterwards.

What is not right, and blocks a final claim on any solar-derived quantity:

- **The all-sky ERA5 GHI is roughly an order of magnitude low**, the pipeline measured it, and
  nothing corrected it. Verification requires one inspection of a raw `*_accum.nc` file.
- **The 850 hPa pressure bound compresses the one elevation-encoding signature index** for a state
  whose entire methodological weak point is elevation.

Neither of these invalidates the Phase 3–6 chain — the two-tier design routed around the first, and
the second degrades rather than destroys `elev_proxy` — but both must be stated plainly wherever a
solar magnitude or an elevation-derived index is reported.
