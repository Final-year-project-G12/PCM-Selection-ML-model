"""
signature_lib.py
=============================================================================
Shared Tier-1 climate-signature construction logic, factored out of
04_climate_signature_rajasthan.py so it has exactly one implementation.
04_climate_signature_rajasthan.py (Level A / whole-year, one row per point)
and 05_cluster_rajasthan.py's Level B (one row per point PER SEASON) both
call build_tier1_signature() with a different `group_keys` list rather than
each carrying their own copy of the index formulas — if a formula changes,
it changes in one place for both.

Not state-specific despite the "rajasthan" folder it lives in: nothing
here reads a state name, a config path, or a fixed point/column list beyond
the standard era5_* column names both pipelines already share. Safe to
import unmodified from a future era5-assam/era5-tamilnadu/era5-uttarakhand
folder, or to move up a level and import from all of them, when that
cross-state step happens.

HOW TO RUN: not runnable directly — import from here.
"""

import numpy as np
import pandas as pd

EVENT_ORDER = ["sunrise", "noon", "sunset"]

# Matches 02_combine_rajasthan.py's SEASON_MAP exactly — used here only to
# derive a "season" column on dataframes that don't already carry one
# (suntimes.csv has no season column; climate_*_points.csv already does).
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
    9: "Retreat", 10: "Retreat", 11: "Retreat",
}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]


def attach_season(df, date_col="date"):
    """Adds a 'season' column derived from date_col's month, via SEASON_MAP.
    Use for dataframes (e.g. suntimes.csv) that don't already carry season —
    climate_*_points.csv already has one; prefer that column directly there
    rather than re-deriving it, so both stay from the same source of truth."""
    df = df.copy()
    df["season"] = pd.to_datetime(df[date_col]).dt.month.map(SEASON_MAP)
    return df


def compute_hsi_sunrise(t_sunrise_mean, rh_sunrise_mean):
    """Humidity stress index at the coldest/condensation-critical instant
    of the day (sunrise), as a function of T_sunrise_mean and RH_sunrise_
    mean directly. Uses Thom's discomfort/Temperature-Humidity Index
    (Thom, E.C., "The Discomfort Index", Weatherwise 12(2), 1959):
        THI = T - 0.55*(1 - RH/100)*(T - 14.5)      [T in C, RH in %]
    Higher HSI_sunrise means less evaporative relief at the coolest instant
    of the day — a proxy for how "muggy" pre-dawn conditions are, correlated
    with condensation/corrosion risk at the store surface."""
    return t_sunrise_mean - 0.55 * (1 - rh_sunrise_mean / 100.0) * (t_sunrise_mean - 14.5)


def _event_agg(df, group_keys, event, col, stat):
    sub = df[df["event"] == event]
    g = sub.groupby(group_keys, observed=True)[col]
    if stat == "mean":
        return g.mean()
    if stat == "std":
        return g.std()
    if stat.startswith("p"):
        q = int(stat[1:]) / 100.0
        return g.quantile(q)
    raise ValueError(stat)


def build_tier1_signature(events_df, sun_df, group_keys):
    """
    Collapses sun-event samples (events_df) and sunrise/sunset timestamps
    (sun_df) into one Tier-1 climate-signature row per unique combination
    of `group_keys`.

    events_df must carry: group_keys + ["event", "date", "era5_T_amb",
        "era5_RHum", "era5_GHI", "era5_CSI", "era5_W_spd"]
    sun_df must carry: group_keys + ["event", "date", "time_utc"]

    group_keys is the list of columns to build one signature row per
    unique combination of — e.g. ["point_id"] for one row per point
    (04_climate_signature_rajasthan.py's Level A / whole-year use), or
    ["point_id", "season"] for one row per point PER SEASON
    (05_cluster_rajasthan.py's Level B). If group_keys includes a column
    that isn't already on one of the input dataframes (e.g. "season" isn't
    on suntimes.csv), the CALLER must attach it first — attach_season()
    above does this from a date column. events_df's own "season" column
    (already present in climate_*_points.csv) should be used directly
    rather than re-derived, so both stay from the same source of truth.

    Returns a DataFrame indexed by group_keys (a plain Index if
    len(group_keys)==1, else a MultiIndex), with columns:
      T_sunrise_mean, T_sunrise_p05, T_noon_mean, T_sunset_mean,
      T_sunset_p95, diurnal_gradient, kt_noon_mean, kt_noon_std,
      GHI_noon_mean, GHI_sunset_mean, RH_sunrise_mean, HSI_sunrise,
      wind_noon_mean, wind_sunset_mean, daylength_mean,
      daylength_amplitude, Ta_mean, Ta_p95, Ta_p05
    """
    out = pd.DataFrame(index=events_df.groupby(group_keys, observed=True).size().index)

    out["T_sunrise_mean"] = _event_agg(events_df, group_keys, "sunrise", "era5_T_amb", "mean")
    out["T_sunrise_p05"] = _event_agg(events_df, group_keys, "sunrise", "era5_T_amb", "p05")
    out["T_noon_mean"] = _event_agg(events_df, group_keys, "noon", "era5_T_amb", "mean")
    out["T_sunset_mean"] = _event_agg(events_df, group_keys, "sunset", "era5_T_amb", "mean")
    out["T_sunset_p95"] = _event_agg(events_df, group_keys, "sunset", "era5_T_amb", "p95")

    # diurnal_gradient: noon-minus-sunrise proxy. Understates true DTR
    # because peak air temperature typically lags solar noon by 2-3 hours
    # (true daily Tmax occurs mid-to-late afternoon, not solar noon) — this
    # is exactly why Tier 2's DTR_true (true T2M max-min, whole-year only)
    # is kept as a separate, more accurate companion where it's available.
    out["diurnal_gradient"] = out["T_noon_mean"] - out["T_sunrise_mean"]

    out["kt_noon_mean"] = _event_agg(events_df, group_keys, "noon", "era5_CSI", "mean")
    out["kt_noon_std"] = _event_agg(events_df, group_keys, "noon", "era5_CSI", "std")
    out["GHI_noon_mean"] = _event_agg(events_df, group_keys, "noon", "era5_GHI", "mean")
    out["GHI_sunset_mean"] = _event_agg(events_df, group_keys, "sunset", "era5_GHI", "mean")
    out["RH_sunrise_mean"] = _event_agg(events_df, group_keys, "sunrise", "era5_RHum", "mean")
    out["wind_noon_mean"] = _event_agg(events_df, group_keys, "noon", "era5_W_spd", "mean")
    out["wind_sunset_mean"] = _event_agg(events_df, group_keys, "sunset", "era5_W_spd", "mean")

    out["HSI_sunrise"] = compute_hsi_sunrise(out["T_sunrise_mean"], out["RH_sunrise_mean"])

    # Ta_mean / Ta_p95 / Ta_p05 — daily mean of the 3 sun-events (a standard
    # sparse-sampling estimator), then mean/p95/p05 of that daily series
    # within each group. Required by the plan doc's §6.4 PCA block
    # (Level A only reads these three onward; Level B does not run PCA —
    # see 05_cluster_rajasthan.py — but they're computed here regardless
    # since they're cheap and part of the same daily-collapse step as
    # daylength below).
    daily_ta = events_df.groupby(group_keys + ["date"], observed=True)["era5_T_amb"].mean()
    ta_group = daily_ta.groupby(level=group_keys)
    out["Ta_mean"] = ta_group.mean()
    out["Ta_p95"] = ta_group.quantile(0.95)
    out["Ta_p05"] = ta_group.quantile(0.05)

    # Daylength — from suntimes.csv's real datetimes, so cross-midnight UTC
    # wraparound is handled automatically by datetime subtraction, no
    # manual hour-of-day arithmetic needed.
    sun_wide = sun_df.pivot_table(index=group_keys + ["date"], columns="event",
                                   values="time_utc", aggfunc="first")
    daylen_hours = (sun_wide["sunset"] - sun_wide["sunrise"]).dt.total_seconds() / 3600.0
    daylen_hours = daylen_hours.dropna()
    dl_group = daylen_hours.groupby(level=group_keys)
    out["daylength_mean"] = dl_group.mean()
    # "amplitude" = half the seasonal swing (max - min around the mean),
    # the standard oscillation-amplitude convention, not the full
    # peak-to-trough range. For Level B (already split by season) this is
    # computed WITHIN each season's dates, so it reflects day-to-day
    # daylength drift within that season, not the whole year's swing —
    # expect it to be much smaller than Level A's for the same point.
    out["daylength_amplitude"] = (dl_group.max() - dl_group.min()) / 2.0

    return out
