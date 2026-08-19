# 13 — Derived Solar Variables Audit

## GHI

`GHI = accum_to_flux(ssrd)/3600`, clipped ≥0 (redundant second clip after `accum_to_flux()`'s own
clip). See `09_ERA5_DATA_PIPELINE.md` for the full deaccumulation story — this is the pipeline's most
consequential derived variable and the one that surfaced the deaccumulation bug.

## DNI — two-branch derivation, neither branch is a true decomposition model

```python
if "avg_sdirswrf" in df.columns:
    df["DNI"] = df["avg_sdirswrf"].clip(0, 1400)
else:
    df["DNI"] = np.where(cos_z > 0.05, df["GHI"] / cos_z, 0).clip(0, 1400)
```

**Branch 1 (primary, used whenever the direct-radiation column matched)**: DNI is taken directly
from whichever ERA5 field matched `msdwswrf`/`fdir`/`msdrswrf` — not decomposed from GHI at all. See
the unit-consistency caveat below; this branch's correctness depends on that field's unit convention
matching the code's assumption.

**Branch 2 (fallback)**: `DNI = GHI / cos(SZA)` where `cos(SZA) > 0.05`, else `0`. This is a crude
algebraic closure (essentially "how much direct beam would be needed, at this sun angle, to account
for all of GHI, if there were zero diffuse component") — **not** a genuine DNI decomposition model.
Established decomposition models (DISC, Erbs, DIRINT) use empirical clearness-index-dependent
relationships calibrated against measured DNI/DHI pairs; this pipeline's fallback branch does none
of that. Since Branch 1 is available whenever the direct-radiation ERA5 field is present (which
should be essentially always, given it's requested unconditionally in `ACCUM_VARS`), Branch 2 is
likely rarely exercised in practice for this dataset — but this was not independently confirmed by
checking how often `avg_sdirswrf` is actually present/non-null in the real output.

## DHI — a closure residual, not independently modeled or observed

```python
df["DHI"] = (df["GHI"] - df["DNI"] * cos_z).clip(0)
```
By construction, `DHI` always exactly satisfies the closure equation `GHI = DHI + DNI·cos(SZA)` (up
to the lower-bound clip) — it is never independently modeled or observed. This means any error in
either `GHI` or `DNI` propagates directly and entirely into `DHI`; `DHI` cannot be used as an
independent cross-check on the other two variables, since it is definitionally derived from them.
State this plainly in any methodology write-up that presents DHI values — it is not a third,
independently-validated quantity.

## Clearness index (CSI)

```python
df["CSI"] = np.where(df["GHI_clearsky"] > 10, (df["GHI"]/df["GHI_clearsky"]).clip(0, 1.5), 0)
```
Forced to `0` (not `NaN`) below the 10 W/m² clear-sky threshold — see `12_SOLAR_GEOMETRY.md` for the
nighttime-handling discussion. Clipped to `[0, 1.5]` in the pipeline itself — note this is **tighter**
than `03_verify_climate_csv.py`'s own declared plausibility bound of `[0, 2]` for `era5_CSI`, meaning
that particular QC range check can structurally never fire (a "dead check" — not incorrect, just
redundant given the upstream clip already enforces a narrower range). See `15_QUALITY_CONTROL.md`.

## The one open unit-consistency question: `avg_sdirswrf`

Restated from `09_ERA5_DATA_PIPELINE.md` because it directly affects DNI: the column-matching logic
(`next((c for c in df.columns if c in ("msdwswrf","fdir","msdrswrf")), None)`) treats these three ERA5
field names as interchangeable and applies identical treatment (`.clip(0)` only, never `/3600`)
regardless of which one actually matched. `fdir` (total sky direct solar radiation at surface) is an
**accumulated** field in ERA5's catalogue and would need the same `accum_to_flux()/3600` treatment
`ssrd`/`strd` receive; `msdwswrf`/`msdrswrf` (mean rate fields) are already W/m² and correctly need no
conversion. **This audit did not independently verify which of the three names is actually present
in the downloaded NetCDF files** — `01_download_era5_rajasthan.py`'s `ACCUM_VARS` list requests
`"mean_surface_direct_short_wave_radiation_flux"` specifically (which maps to `msdwswrf`, a mean-rate
field), so in practice the code path is very likely always hitting the already-correct no-conversion
case — but the code's own generality (accepting `fdir` as an equally-valid match) means this is a
latent risk if the requested variable list ever changes, not a currently-confirmed bug. **Recommend
verifying directly** (open one `_accum.nc` file, check `ds.data_vars`) before this variable is
presented as fully unit-validated in a methodology write-up.

## Physical bounds applied to derived solar variables

| Variable | Applied bound | Where |
|---|---|---|
| GHI | `<0→0`, `>1400→NaN` (asymmetric: low clipped, high dropped) | `02_combine_rajasthan.py` |
| DNI | `clip(0, 1400)` both branches | same |
| DHI | `clip(0)`, no upper bound | same |
| CSI | `clip(0, 1.5)`, forced 0 below GHI_clearsky=10 threshold | same |
| GHI_clearsky | Ineichen model output, range-checked `[0,1400]` downstream | `03_verify_climate_csv.py` |

1400 W/m² as an upper physical bound is a reasonable, standard solar-constant-adjacent ceiling
(extraterrestrial irradiance at 1 AU is ~1361 W/m²; surface GHI/DNI essentially never exceeds this
even under highly-reflective-cloud edge enhancement events) — not independently cited to a specific
source in-code, but a defensible engineering threshold, not an arbitrary one.

## Literature support

No dedicated DNI/DHI decomposition-model citation applies here, precisely because neither branch
implements a published decomposition model (this is itself the finding to report, not omit). If a
future revision adopts a real decomposition model, the standard citations would be Erbs, Klein &
Duffie (1982) or Perez et al. (1990, DISC). As currently implemented, the correct framing for a
write-up is: "DNI is taken directly from ERA5's own direct-radiation field where available, with a
closure-equation fallback; DHI is computed as the closure residual, not independently modeled" — an
accurate, defensible description of what the code does, without overclaiming a decomposition-model
provenance it doesn't have.
