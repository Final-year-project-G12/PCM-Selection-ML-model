# 10 — Temporal Processing Audit

## Timezone Alignment
- **Reanalysis Data**: ERA5 is stored in Coordinated Universal Time (UTC).
- **Satellite Data**: NASA POWER is retrieved in UTC.
- **Local Time**: Tamil Nadu runs on Indian Standard Time (IST = UTC + 5:30).
- **Sun-Event Windows**: The hours downloaded are aligned to local solar events (sunrise, noon, sunset) computed from the solar position algorithm (SPA).
- **Circular Window matching**: Matches the nearest hourly NetCDF reading to the exact sun-event time. Uses circular matching to prevent midnight UTC boundary errors.

## Missing Timestamps and Duplicates
- Chronological sorting is applied before Hampel outlier detection and feature lags.
- Lag features (`lag1d`, `lag7d`, `lag30d`) are shifted within `(point_id, event)` groups. A date-gap check is implemented to prevent lag bridging over missing periods.
- In `02b_build_daily_aggregates.py`, a calendar day is only integrated if it has **>=20 valid hours** of NASA POWER data.

## Literature Support
| Topic | Reference | Source |
|---|---|---|
| UTC/IST alignment | Reda & Andreas (2004) SPA sun events | `12_SOLAR_GEOMETRY.md` |
| Daily GHI integration | Ghodusinejad et al. (2026) — temporal aggregation | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| HDD18/CDD24 degree-days | Standard building energy practice | `15_QUALITY_CONTROL.md` |
