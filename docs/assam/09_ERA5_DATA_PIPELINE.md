# 12 — ERA5 Data Pipeline: Deep Audit (Assam)

## The deaccumulation story — inherited fix

The most consequential preprocessing decision in the pipeline concerns how ERA5's accumulated
solar radiation fields are converted to per-hour fluxes. The Rajasthan pipeline discovered this:

**Original assumption**: `ssrd` (solar radiation) is cumulative-since-forecast-reset (MARS convention),
requiring `diff()` against the previous hour to recover an hourly flux.

**What was actually found**: The specific CDS request configuration used by this pipeline returns
each hour as its own ~1-hour accumulated value, NOT a running cumulative total. Applying `diff()` to
this produced near-zero GHI (noon Pearson r ≈ 0.01 vs NASA POWER — physically implausible).

**Fix** — `accum_to_flux()` in `02_combine_assam.py`:
```python
def accum_to_flux(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    return s.clip(lower=0)
```
No differencing. Stateless clip to non-negative. The Assam pipeline was built **after** this fix
was established in Rajasthan — `02_combine_assam.py` inherits the correct version from the start.
This is a direct benefit of developing Assam after the Rajasthan audit.

## Unit conversions applied in `02_combine_assam.py`

| ERA5 field | Raw unit | Operation | Output unit | Output column |
|---|---|---|---|---|
| `ssrd` | J/m² (per-hour accumulation) | `accum_to_flux() / 3600` | W/m² | `era5_GHI` |
| `strd` | J/m² (per-hour accumulation) | `accum_to_flux() / 3600` | W/m² | `era5_LW_down` |
| `t2m` | K | `− 273.15` | °C | `era5_T_amb` |
| `d2m` | K | `− 273.15` | °C | `era5_T_dew` |
| `tp` | m | `× 1000` | mm | `era5_precipitation` |
| `msl` | Pa | `/ 100` | hPa | `era5_P_atm` |
| `u10`, `v10` | m/s | `sqrt(u²+v²)`, `atan2(u,v)` | m/s, degrees | `era5_W_spd`, `era5_W_dir` |
| `avg_sdirswrf` | W/m² (mean-rate) or J/m² (accum) | `clip(0)` only — **no /3600** | W/m² if mean-rate | `era5_DNI` (primary branch) |

## The `avg_sdirswrf` / DNI unit-consistency caveat (inherited from Rajasthan)

The column-matching logic for DNI source accepts `msdwswrf`, `fdir`, or `msdrswrf` — three ERA5
fields with different unit conventions. `fdir` is accumulated (would need `/3600`); `msdwswrf`/
`msdrswrf` are mean-rate (already W/m²). The code applies `clip(0)` uniformly without `/3600`
regardless of which field matched. If the actual downloaded variable is `fdir`, DNI is
over-estimated by 3600×.

This audit did not independently verify which field name is present in the Assam NetCDF files.
The download script requests `mean_surface_direct_short_wave_radiation_flux` (which maps to
`msdwswrf`, a mean-rate field), so the no-conversion branch is **very likely correct in
practice** — but this should be verified by opening a sample `.nc` file and inspecting
`ds.data_vars` before presenting DNI as unit-validated.

## Default elevation: 100m for Assam

Unlike Rajasthan (which ran `00c_attach_elevation.py` to attach per-point ERA5 geopotential
elevation), Assam uses `DEFAULT_ALT_M = 100` (Assam valley/plains baseline) for all 128 points.
This is documented in `02_combine_assam.py`. The 100m default is appropriate for the Brahmaputra
plains (where most of the population lives) but underestimates elevation for Karbi Anglong and
Dima Hasao hill districts (actual elevation 300–900m+). This affects atmospheric pressure
estimation and the `elev_proxy` signature index.

## Cross-source agreement

A formal cross-source validation step via `03b_agreement_analysis_assam.py` compares ERA5 vs NASA POWER GHI. The script automatically determines whether empirical quantile mapping is necessary.

For Assam, the analysis evaluated the daytime GHI and found a Mean Bias Error of **1.1%**. Because this is well below the 10% threshold, it automatically selected the `BACKBONE` decision (meaning the structurally correct ERA5 data is passed through unmodified without the need for synthetic bias correction).

## Literature support

Hersbach et al. (2020), *QJRMS* 146(730) — ERA5 product citation. The `accum_to_flux()` fix is an
empirically-determined pipeline-specific finding (not a general CDS API claim); describe it as such
in any write-up.
