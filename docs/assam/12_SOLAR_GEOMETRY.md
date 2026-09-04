# 15 — Solar Geometry Audit (Assam)

## Core function: `compute_solar()` in `02_combine_assam.py`

Same structure as Rajasthan — pvlib-based solar position and clear-sky calculation applied per
point, per event, using the point's lat/lon/alt.

```python
def compute_solar(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt, tz="UTC")
    times = pd.DatetimeIndex(df.index)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    df["SZA"] = sp["zenith"].values
    df["solar_azimuth"] = sp["azimuth"].values
    df["GHI_clearsky"] = cs["ghi"].values
    ...
```

## Solar position algorithm — method not explicitly pinned

`get_solarposition(times)` called without an explicit `method=` argument — relies on pvlib's
installed-version default (currently NREL SPA in modern pvlib releases, equivalent to Reda &
Andreas 2004). `00b_build_suntimes.py` *does* pin `method="spa"` explicitly. Recommendation:
pin `method=` in `compute_solar()` too for complete reproducibility across pvlib versions.

## Clear-sky model: Ineichen with default Linke turbidity

`get_clearsky(times, model="ineichen")` — Ineichen clear-sky model with pvlib's bundled default
Linke-turbidity climatology lookup. This is a defensible choice for a project of this scope.

**Assam-specific caveat**: Assam's Linke turbidity is dominated by the monsoon season — heavy
aerosol loading from biomass burning in the pre-monsoon (Mar–May), high humidity and cloud
interaction in the monsoon (Jun–Sep). The default Linke-turbidity climatology may not capture
Assam's actual seasonal aerosol pattern accurately. This affects `GHI_clearsky` and therefore
`CSI` (clearness index). A one-sentence caveat in the methodology write-up is appropriate.

## Altitude: 100m fixed for all 129 points

`alt_m = DEFAULT_ALT_M = 100` for all Assam points (no per-point elevation from ERA5 geopotential).
Passed into both `pvlib.location.Location(altitude=alt_m)` (feeds Ineichen model) and written
to output per-row. For hill district points (Cluster 0), the 100m default underestimates real
elevation, slightly affecting the clear-sky irradiance model. This is a documented approximation.

## Nighttime handling and division-by-zero protection

- `CSI` forced to `0` (not NaN) wherever `GHI_clearsky ≤ 10 W/m²` — covers nighttime and
  near-horizon conditions. A `CSI=0` value could mean "genuinely no solar radiation" or "ratio
  suppressed at low sun angles" — indistinguishable from output alone.
- `cos_z = cos(deg2rad(SZA.clip(0, 89.9)))` — clipped to 89.9° for DNI division only; real
  unclipped SZA is written to output.

## Assam latitude range and solar angles

Assam spans ~24.1–27.8°N. Solar noon zenith angles range from ~43° (winter solstice, northernmost
point) to ~0° (summer solstice, near tropic). The event-aligned sampling correctly captures
these physically meaningful instants. Monsoon-season GHI at noon is substantially reduced by cloud
cover — the `cloudy_frac` and `monsoon_index` signature indices specifically capture this.

## Literature support

Reda & Andreas (2004), *Solar Energy* 76(5) — SPA algorithm. Ineichen & Perez (2002), *Solar
Energy* 73(3) — Ineichen clear-sky model. Holmgren, Hansen & Mikofski (2018), *JOSS* 3(29) —
pvlib software citation.
