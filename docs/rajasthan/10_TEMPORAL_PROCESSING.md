# 10 — Temporal Processing Audit

## UTC as the sole time reference

Every timestamp in the pipeline is UTC — `time_utc` in `suntimes.csv`, ERA5's native UTC timestamps,
NASA POWER requested with `time-standard=UTC`. There is no IST (India Standard Time, UTC+5:30)
conversion anywhere in the pipeline. This is a reasonable, internally-consistent choice (UTC avoids
daylight-saving/timezone-drift issues entirely, and India has no DST), but it means any figure
intended for a general audience (e.g., "sunrise at 6 AM") needs an explicit UTC→IST conversion at
presentation time, not before — worth stating explicitly in a methodology write-up so a reviewer
doesn't assume local-time sampling.

## Sunrise/noon/sunset via pvlib SPA

`pvlib.location.Location.get_sun_rise_set_transit(dates, method="spa")` — Reda & Andreas (2004)
Solar Position Algorithm. No manual equation-of-time code. `altitude=0` hardcoded for this specific
call (see `03_PHASE_1_AUDIT.md` — a minor inconsistency with the elevation-aware geometry used
downstream, negligible practical effect on sunrise/sunset clock time specifically).

## Cross-midnight UTC handling — the circular-window algorithm

Real, documented case: an eastern Rajasthan point's summer sunrise can land at 23:55 UTC of the
*previous* UTC calendar date (e.g. Dholpur, 2020-06-21 sunrise at 2020-06-20 23:55:54 UTC — quoted
exactly from the code). A naive `min()`/`max()` over raw hour-of-day integers breaks here (observed
hours like `{23,0,1,2}` would numerically span the whole day). The fix,
`circular_hour_window()` in `01_download_era5_rajasthan.py`: find the largest **unobserved** circular
gap in the sorted, deduplicated hour set, take everything else as the "arc," pad by `HOUR_MARGIN=1`
on each side, wrap with modulo-24 arithmetic. This is a correct, general solution to a genuine edge
case, not a hack — cross-midnight cases are common enough (documented, not hypothetical) that a
naive min/max would have silently corrupted the download-hour selection for real points.

`time_utc` (in `suntimes.csv`) always reflects the true instant of the event; `date` is pvlib's
nominal calendar-date assignment, which can differ from `time_utc`'s own UTC date for events near
midnight — this distinction is preserved explicitly in the schema, not collapsed.

## Leap years and date range

`2016-01-01` through `2025-12-31` inclusive — 3653 days (correctly includes leap years 2016, 2020,
2024: 10×365 + 3 = 3653). Ground-truthed directly against `suntimes.csv`'s row count
(320 × 3653 × 3 = 3,506,880, exact match).

## Missing/duplicated timestamps

Handled at the merge stage, not the download stage: `nearest_row()`'s `MAX_MATCH_HOURS=3` rejection
window (see below) is the mechanism that turns a missing/sparse ERA5 or POWER reading into a `NaN`
row rather than a wrong pairing. `03_verify_climate_csv.py` Check 3 independently flags duplicate
`(point_id, date, event)` combinations as a hard **FAIL** (not just a warning) — genuine duplicate
protection exists.

## The "2016-01-01 edge case" — a documentation/code mismatch worth resolving

Referenced by name in three places (module-level framing in `02_combine_rajasthan.py`'s conceptual
docstring, `03_verify_climate_csv.py`'s docstring/comments, `03b_agreement_analysis.py`'s docstring)
as: "2016-01-01 has no prior day to supply an accumulation-deaccumulation predecessor hour... the
affected columns for that one day come out as a natural NaN." **This made sense under the original
`deaccumulate()` design** (which genuinely needed a predecessor hour via `diff()`), but after the
`accum_to_flux()` fix (stateless, no predecessor dependency — see `09_ERA5_DATA_PIPELINE.md`), **no
code in `02_combine_rajasthan.py` actually implements this special case anymore.** This is either (a)
a stale comment left over from before the deaccumulation fix, or (b) a genuinely distinct edge case
this audit did not independently trace. **Recommendation**: before citing "the documented 2016-01-01
edge case" in a methodology write-up, re-verify whether `climate_rajasthan_points.csv` actually shows
any anomalous null pattern on that specific date under the current (fixed) code, since the
originally-cited mechanism for it may no longer apply.

## Nearest-in-time matching / the 3-hour rejection window

`nearest_row(series_df, target_time, max_hours=MAX_MATCH_HOURS=3)`: rejects a match farther than 3
hours from the true sun-event instant. Applied **independently** to the ERA5 series and the POWER
series against the same target — there is no requirement that both sources match the *same* actual
timestamp, so in principle an ERA5 reading up to 3h before the event and a POWER reading up to 3h
after could be paired in the same output row. **The actual matched timestamp is never recorded** —
only the requested `time_utc` is written to `climate_rajasthan_points.csv`. This is a genuine,
acknowledged gap (the code's own `03_qc_plots.py` rejection-window diagnostic is permanently skipped
because of it, with an explicit in-code message saying so). **Low-cost fix**: add
`era5_matched_time_utc`/`power_matched_time_utc` output columns in `02_combine_rajasthan.py`, which
would both enable the already-written rejection-window QC plot and let a reviewer directly verify how
often ERA5/POWER are paired from meaningfully different actual instants.

## Sun-event-aligned vs. fixed-clock-hour sampling — why this matters for solar-thermal validity

Sampling at astronomically-computed sunrise/noon/sunset (rather than fixed 02:00/08:00/14:00 UTC)
ensures the sampled instants are the physically meaningful ones for a solar-thermal system across
all 320 points, all seasons, all 10 years — a fixed-clock-hour scheme would sample "sunrise" at a
genuinely different solar elevation angle depending on season and longitude, contaminating any
sunrise-indexed climate index (like `HSI_sunrise`, `T_sunrise_mean`) with a seasonal/spatial
sampling artifact unrelated to actual climate.

## Seasonal definitions — an inconsistency worth fixing

`02_combine_rajasthan.py`'s `SEASON_MAP` (Winter=Dec-Feb, Summer=Mar-May, **Monsoon=Jun-Aug**,
Retreat=Sep-Nov) disagrees with `02b_build_daily_aggregates.py`'s `MONSOON_MONTHS = [6,7,8,9]`
(**Jun-Sep**). `signature_lib.py`'s own `SEASON_MAP` matches `02_combine_rajasthan.py`'s definition
(Jun-Aug) exactly, by explicit design ("matches 02_combine_rajasthan.py's SEASON_MAP exactly"). So
the *season column* used throughout Tier 1 and clustering is consistently Jun-Aug, but the *Tier-2
`monsoon_index`* feature specifically is computed against a different, wider Jun-Sep window. This is
a real, minor inconsistency — not a bug that corrupts either individual computation, but two
different "monsoon" definitions feeding two different downstream features that a reader might
reasonably assume are consistent. Reconcile before final write-up (either both to Jun-Aug or both to
Jun-Sep, with the choice justified against India Meteorological Department convention, which
typically treats the Indian monsoon as Jun-Sep for Rajasthan).

## Literature support

Reda & Andreas (2004), *Solar Energy* 76(5) — SPA algorithm. No additional dedicated citation was
found or needed for the circular-hour-window algorithm itself (an original, correctly-reasoned
engineering solution to a real edge case, not drawn from a specific published method).
