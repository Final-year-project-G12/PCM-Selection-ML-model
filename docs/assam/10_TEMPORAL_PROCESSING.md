# 13 — Temporal Processing Audit (Assam)

## UTC as sole time reference

All timestamps are UTC — `time_utc` in `suntimes.csv`, ERA5's native UTC, NASA POWER requested
with UTC. No IST (India Standard Time, UTC+5:30) conversion exists in the pipeline. Assam sits at
approximately UTC+5:30 (IST), so solar noon in UTC is approximately 06:00–07:00 UTC depending on
longitude (~89–97°E). Any figures presented to a general audience need explicit UTC→IST conversion
at presentation time.

## Sunrise/noon/sunset via pvlib SPA

`pvlib.location.Location.get_sun_rise_set_transit(dates, method="spa")` — Reda & Andreas (2004)
Solar Position Algorithm, explicitly pinned in `00b_build_suntimes.py`. This provides the three
"events" at which ERA5 is downloaded, ensuring sun-geometry-aligned samples across all 128 points,
all seasons, all 10 years.

## Cross-midnight UTC handling

Assam spans ~89.7–96.0°E longitude. Eastern Assam points (high longitude) will have earlier UTC
sunrise times. The pipeline inherits the `circular_hour_window()` algorithm from the Rajasthan
implementation — finds the largest unobserved circular gap in the sorted event-hour set, pads by
`HOUR_MARGIN=1`, wraps with modulo-24 arithmetic. This correctly handles cases where sunrise falls
near the UTC midnight boundary.

## Leap years and date range

2016-01-01 through 2025-12-31 inclusive — 3653 days (10×365 + 3 leap-year days: 2016, 2020, 2024).
Expected `suntimes.csv` rows: 128 × 3653 × 3 = ~1,402,752.

## Nearest-in-time matching

`nearest_row(series_df, target_time, max_hours=3)` — rejects any ERA5 or POWER reading farther
than 3 hours from the true sun-event instant. Applied independently to each source. The actual
matched timestamp is not persisted in the output (only the requested `time_utc` is written) — this
is a known gap inherited from the Rajasthan design; see `10_IMPLEMENTATION_ISSUES.md`.

## Season mapping (Assam)

```
Dec, Jan, Feb  → Winter (1)
Mar, Apr, May  → Pre-Monsoon (2)
Jun, Jul, Aug, Sep → Monsoon (3)
Oct, Nov       → Post-Monsoon (4)
```

Assam uses a **4-month Monsoon** (Jun–Sep), which is the meteorologically correct choice for Northeast
India per India Meteorological Department convention. This is **internally consistent** within all
Assam scripts — unlike Rajasthan, which had an inconsistency between `02_combine_rajasthan.py`
(Jun–Aug) and `02b` (Jun–Sep). Assam avoids that inconsistency by using Jun–Sep in both `02_combine_assam.py`
and `02b_build_daily_aggregates_assam.py`.

## Monsoon index consistency

`monsoon_index` in the Tier 2 signature (fraction of annual precipitation in Jun–Sep) matches the
4-month Monsoon season definition used in the `season` column throughout the pipeline. This is
**correctly consistent** — a deliberate improvement over Rajasthan's documented monsoon-month mismatch.

## Literature support

Reda & Andreas (2004), *Solar Energy* 76(5) — SPA algorithm. IMD (India Meteorological Department)
monsoon-season convention for Northeast India (Jun–Sep) is standard and well-established; no specific
citation is needed for this widely-used definition.
