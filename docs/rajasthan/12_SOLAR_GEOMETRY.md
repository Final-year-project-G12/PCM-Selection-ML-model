# 12 — Solar Geometry Audit

Core function: `compute_solar()` in `02_combine_rajasthan.py`.

```python
def compute_solar(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt, tz="UTC")
    times = pd.DatetimeIndex(df.index)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    df["SZA"] = sp["zenith"].values
    df["solar_azimuth"] = sp["azimuth"].values
    df["ETR"] = pvlib.irradiance.get_extra_radiation(times).values
    df["GHI_clearsky"] = cs["ghi"].values
    ...
```

## Solar position algorithm — not explicitly pinned

`get_solarposition(times)` is called **without an explicit `method=` argument**. This means the
algorithm used is whatever pvlib's installed-version default is, not a value pinned in this
pipeline's own code. (`00b_build_suntimes.py`'s sunrise/sunset computation, by contrast, *does*
explicitly pass `method="spa"`.) In current pvlib releases the default for `get_solarposition` is
NREL's SPA-based method, so this is very likely equivalent in practice to an explicit SPA pin — but
"very likely equivalent" is not the same as "verified pinned," and a pvlib version upgrade could
silently change the algorithm used for `SZA`/`solar_azimuth` without any code change or warning in
this pipeline. **Recommendation**: pin `method=` explicitly (or at minimum record the installed
pvlib version in the environment/reproducibility record — see `21_REPRODUCIBILITY.md`) before
treating solar-position results as fully reproducible across environments/time.

## Clear-sky model — Ineichen, default Linke turbidity

`get_clearsky(times, model="ineichen")` — the **Ineichen** clear-sky model, called without an
explicit `linke_turbidity=` override, so it relies on pvlib's bundled default Linke-turbidity
climatology lookup table (interpolated by location/month). This is a standard, defensible choice for
a project of this scope (a location-specific measured turbidity record would be a substantial
additional data-acquisition burden), but it is a real assumption: Linke turbidity captures
atmospheric aerosol/water-vapor loading, and Rajasthan's actual aerosol loading (dust storms are a
real regional phenomenon) may deviate from the climatological default on specific days. This affects
`GHI_clearsky` and therefore `CSI` (clearness index) — worth a one-sentence caveat in a methodology
write-up, not a required fix given the project's scope.

## Extraterrestrial radiation — computed but discarded

`pvlib.irradiance.get_extra_radiation(times)` is computed and assigned to `df["ETR"]` inside
`compute_solar()`, but `ETR` is **not** in `ERA5_OUTPUT_VARS` and is therefore never written to the
final `climate_rajasthan_points.csv`. This is wasted (harmless) computation, not a bug — but if any
downstream analysis ever wants ETR (e.g., for an independent clearness-index cross-check), it would
currently need to be recomputed rather than read from the existing output.

## Altitude usage and the 300 m fallback

`alt_m = point_row.elevation_m` if present and non-NaN, else `DEFAULT_ALT_M = 300`. Passed into both
`pvlib.location.Location(altitude=alt_m)` (feeds atmospheric-pressure/airmass assumptions in the
Ineichen model, and a small refraction correction in `get_solarposition`) and written to the output
CSV per-row as `elevation_m`. Since `00c_attach_elevation.py` now populates real elevation for all
320 points, the 300 m fallback is not currently exercised in practice for Rajasthan — it remains as a
defensive default for any future point that might lack an elevation value.

## Nighttime handling and division-by-zero protection

`CSI` (clearness index) is forced to exactly `0` (not `NaN`) wherever `GHI_clearsky ≤ 10` W/m²
(covers nighttime and near-sunrise/sunset conditions where the ratio would otherwise be numerically
unstable or undefined) — this suppresses what should arguably be an "undefined" ratio at low sun
angles into a defined zero, which is a defensible practical choice (keeps the column always numeric)
but does mean a `CSI=0` value in the output could mean either "genuinely clear-sky-radiation-free" or
"ratio was numerically unstable and suppressed" — these are not distinguishable from the output alone.

`cos_z = cos(deg2rad(SZA.clip(0, 89.9)))` — the SZA is clipped to 89.9° **only for this specific
division**, preventing a divide-by-near-zero in the GHI/cos(SZA) DNI fallback; the clipped value is
never written back to the `SZA` output column itself (the real, unclipped SZA is what's reported).

## Physically impossible values

`SZA` output range-checked to `[0,180]`°, `solar_azimuth` to `[0,360]`° in `03_verify_climate_csv.py`
— see `15_QUALITY_CONTROL.md` for the full bound table. No dedicated check exists for SZA being
*inconsistent* with the sun-event label (e.g., a "noon" row with SZA near 90° would indicate a
matching problem, not a solar-geometry computation error) — `03b_agreement_analysis.py`'s
MANUAL_REVIEW-branch diagnostics include exactly this kind of check (median noon SZA flagged if
>45°) but only when the decision branch resolves to MANUAL_REVIEW, which did not occur on the actual
Rajasthan run (it resolved to QUANTILE_MAP) — so this particular diagnostic was never actually
exercised on this dataset.

## Literature support

Reda & Andreas (2004), *Solar Energy* 76(5) — SPA algorithm (pvlib's likely default for
`get_solarposition`, and the explicit method for `get_sun_rise_set_transit`). Ineichen & Perez
(2002), "A new airmass independent formulation for the Linke turbidity coefficient," *Solar Energy*
73(3) — the standard citation for the Ineichen clear-sky model pvlib implements; not independently
verified as present in `references.bib`/`.claude/references.md` during this audit (see
`17_LITERATURE_MAPPING.md`) — should be added before final write-up, since the clear-sky model is a
load-bearing component of `GHI_clearsky`, `CSI`, and therefore `kt_daily_mean`/`SAI`/`CCI` downstream.
Holmgren, Hansen & Mikofski (2018), "pvlib python: a python package for modeling solar energy
systems," *JOSS* 3(29) — the pvlib software citation itself.
