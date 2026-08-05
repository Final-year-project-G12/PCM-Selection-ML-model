"""
VERIFY — climate_rajasthan_points.csv completeness & sanity check
=============================================================================
Checks the final combined output (data/processed/climate_rajasthan_points.csv,
produced by 02_combine_rajasthan.py) against what's actually expected from
population_grid_points.csv (00a) and suntimes.csv (00b):

  1. Schema      — every expected era5_*/power_*/metadata column is present.
  2. Point coverage   — every point_id in population_grid_points.csv shows
                        up in the output; flags missing/extra point_ids.
  3. Row coverage     — each point has one row per (date, event) it should
                        have per suntimes.csv; flags points that are
                        missing, partial, or have duplicate rows.
  4. Null rates       — per-column % missing; flags columns above a
                        threshold (real gaps are expected — e.g. the
                        documented 2016-01-01 deaccumulation edge case —
                        but a column that's mostly empty signals a real bug).
  5. Physical sanity  — value-range checks per variable (temperature,
                        humidity, irradiance, pressure, wind, etc.) using
                        the same bounds the pipeline itself enforces, plus a
                        few this script owns (pressure, POWER irradiance).
  6. Cross-check      — era5_GHI vs power_ALLSKY_SFC_SW_DWN and era5_T_amb
                        vs power_T2M correlation, as a sanity check that the
                        two independent sources broadly agree (this is the
                        entire point of pulling both).

This is a read-only QA tool — it never modifies climate_rajasthan_points.csv.
Safe to re-run at any point, including while 02_combine_rajasthan.py is
still running (it will just report partial coverage accurately).

HOW TO RUN:
  python 03_verify_climate_csv.py
"""

import sys

import numpy as np
import pandas as pd

from config import (
    COMBINED_POINTS_FILE,
    POPULATION_GRID_FILE,
    SUNTIMES_FILE,
)

# ═══════════════════════════════════════════════════════════
# EXPECTED SCHEMA
# ═══════════════════════════════════════════════════════════

METADATA_COLS = [
    "point_id", "lat", "lon", "population", "weight",
    "date", "event", "time_utc", "grid_lat", "grid_lon",
    "month", "DOY", "year", "season", "season_code",
]

ERA5_COLS = [
    "era5_T_amb", "era5_T_dew", "era5_RHum", "era5_W_spd", "era5_W_dir",
    "era5_GHI", "era5_DNI", "era5_DHI", "era5_LW_down", "era5_cloud_cover",
    "era5_precipitation", "era5_P_atm", "era5_SZA", "era5_solar_azimuth",
    "era5_GHI_clearsky", "era5_CSI",
]

POWER_COLS = [
    "power_ALLSKY_SFC_SW_DWN", "power_CLRSKY_SFC_SW_DWN",
    "power_T2M", "power_RH2M", "power_WS10M",
]

REQUIRED_COLUMNS = METADATA_COLS + ERA5_COLS + POWER_COLS

VALID_EVENTS = {"sunrise", "noon", "sunset"}

# Physical plausibility bounds: (column, min, max). Mirrors the bounds the
# pipeline itself enforces in 02_combine_rajasthan.py's apply_unit_conversions
# (T_amb, GHI, RHum) plus a few additional sanity bounds for columns that
# aren't clipped there (pressure, wind, POWER irradiance).
RANGE_CHECKS = [
    ("era5_T_amb",            -5,    60),
    ("era5_T_dew",            -30,   40),
    ("era5_RHum",              0,   100),
    ("era5_W_spd",              0,   40),
    ("era5_GHI",                0,  1400),
    ("era5_DNI",                0,  1400),
    ("era5_DHI",                0,  1400),
    ("era5_GHI_clearsky",       0,  1400),
    ("era5_LW_down",            0,   700),
    ("era5_cloud_cover",        0,     1),
    ("era5_precipitation",      0,   200),
    ("era5_P_atm",            800,  1050),
    ("era5_SZA",                0,   180),
    ("era5_solar_azimuth",      0,   360),
    ("era5_CSI",                0,     2),
    ("power_ALLSKY_SFC_SW_DWN", 0,  1400),
    ("power_CLRSKY_SFC_SW_DWN", 0,  1400),
    ("power_T2M",              -5,    60),
    ("power_RH2M",               0,   100),
    ("power_WS10M",               0,    40),
]

# Columns allowed a higher null rate before it's flagged as a likely bug
# (rather than an expected/occasional gap).
NULL_WARN_THRESHOLD = 0.05    # 5%
NULL_FAIL_THRESHOLD = 0.30    # 30%


# ═══════════════════════════════════════════════════════════
# REPORT HELPERS
# ═══════════════════════════════════════════════════════════

class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def fail(self, msg):
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def ok(self, msg):
        print(f"  [OK]   {msg}")

    def section(self, title):
        print(f"\n{'─'*68}\n  {title}\n{'─'*68}")


# ═══════════════════════════════════════════════════════════
# CHECKS
# ═══════════════════════════════════════════════════════════

def check_schema(df, report):
    report.section("1. SCHEMA")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]

    if missing:
        report.fail(f"Missing {len(missing)} required column(s): {missing}")
    else:
        report.ok(f"All {len(REQUIRED_COLUMNS)} required columns present")

    if extra:
        report.warn(f"{len(extra)} unexpected extra column(s): {extra}")


def check_point_coverage(df, points_df, report):
    report.section("2. POINT COVERAGE")
    expected_ids = set(points_df["point_id"])
    actual_ids = set(df["point_id"].unique()) if "point_id" in df.columns else set()

    missing_points = sorted(expected_ids - actual_ids)
    extra_points = sorted(actual_ids - expected_ids)

    pct = 100 * len(actual_ids) / len(expected_ids) if expected_ids else 0
    print(f"  Points expected : {len(expected_ids)}")
    print(f"  Points present  : {len(actual_ids)}  ({pct:.1f}%)")

    if missing_points:
        preview = missing_points[:10]
        more = f"  ... and {len(missing_points)-10} more" if len(missing_points) > 10 else ""
        if pct < 100:
            report.warn(f"{len(missing_points)} point(s) not yet in the output "
                        f"(expected if 02_combine_rajasthan.py hasn't finished): "
                        f"{preview}{more}")
    else:
        report.ok("Every expected point_id is present")

    if extra_points:
        report.fail(f"{len(extra_points)} point_id(s) in output but not in "
                    f"population_grid_points.csv: {extra_points[:10]}")

    return actual_ids


def check_row_coverage(df, sun_df, actual_ids, report):
    report.section("3. ROW COVERAGE (per point: expected rows from suntimes.csv)")

    expected_counts = sun_df.groupby("point_id").size()
    actual_counts = df.groupby(["point_id", "date", "event"]).size()
    dup_keys = actual_counts[actual_counts > 1]

    if len(dup_keys) > 0:
        report.fail(f"{len(dup_keys)} duplicate (point_id, date, event) combination(s) — "
                    f"first few: {dup_keys.index[:5].tolist()}")
    else:
        report.ok("No duplicate (point_id, date, event) rows")

    row_counts = df.groupby("point_id").size()
    incomplete = []
    for pid in sorted(actual_ids):
        expected = int(expected_counts.get(pid, 0))
        actual = int(row_counts.get(pid, 0))
        if actual != expected:
            incomplete.append((pid, actual, expected))

    if incomplete:
        preview = incomplete[:10]
        more = f"  ... and {len(incomplete)-10} more" if len(incomplete) > 10 else ""
        report.warn(f"{len(incomplete)} present point(s) have row counts that don't match "
                    f"suntimes.csv (partial write, gaps beyond MAX_MATCH_HOURS, or the "
                    f"documented 2016-01-01 edge case): "
                    f"{[f'{p} ({a}/{e})' for p, a, e in preview]}{more}")
    else:
        report.ok("Every present point has the exact expected row count")

    bad_events = set(df["event"].unique()) - VALID_EVENTS
    if bad_events:
        report.fail(f"Unexpected event value(s): {bad_events}")
    else:
        report.ok(f"event column only contains {VALID_EVENTS}")

    try:
        dates = pd.to_datetime(df["date"])
        if dates.min() < pd.Timestamp("2016-01-01") or dates.max() > pd.Timestamp("2025-12-31"):
            report.warn(f"date range outside 2016-01-01..2025-12-31: "
                        f"{dates.min().date()} .. {dates.max().date()}")
        else:
            report.ok(f"date range within bounds: {dates.min().date()} .. {dates.max().date()}")
    except Exception as exc:
        report.fail(f"Could not parse `date` column as dates: {exc}")


def check_null_rates(df, report):
    report.section("4. NULL RATES  (era5_* / power_* columns)")
    n = len(df)
    if n == 0:
        report.warn("No rows to check null rates on.")
        return

    for col in ERA5_COLS + POWER_COLS:
        if col not in df.columns:
            continue
        null_frac = df[col].isna().mean()
        if null_frac >= NULL_FAIL_THRESHOLD:
            report.fail(f"{col}: {null_frac*100:.1f}% null (>= {NULL_FAIL_THRESHOLD*100:.0f}% threshold)")
        elif null_frac >= NULL_WARN_THRESHOLD:
            report.warn(f"{col}: {null_frac*100:.1f}% null (>= {NULL_WARN_THRESHOLD*100:.0f}% threshold)")
    ok_cols = [c for c in ERA5_COLS + POWER_COLS
               if c in df.columns and df[c].isna().mean() < NULL_WARN_THRESHOLD]
    report.ok(f"{len(ok_cols)}/{len(ERA5_COLS)+len(POWER_COLS)} data columns "
              f"under the {NULL_WARN_THRESHOLD*100:.0f}% null threshold")


def check_ranges(df, report):
    report.section("5. PHYSICAL SANITY (value-range checks)")
    for col, lo, hi in RANGE_CHECKS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        out_of_range = ((series < lo) | (series > hi)).sum()
        if out_of_range:
            pct = 100 * out_of_range / len(series)
            (report.fail if pct > 1 else report.warn)(
                f"{col}: {out_of_range:,} value(s) ({pct:.2f}%) outside "
                f"[{lo}, {hi}]  (actual range: {series.min():.1f} .. {series.max():.1f})")
        else:
            report.ok(f"{col}: all values within [{lo}, {hi}]")


def check_cross_source_agreement(df, report):
    report.section("6. ERA5 vs NASA POWER CROSS-CHECK")
    pairs = [
        ("era5_GHI", "power_ALLSKY_SFC_SW_DWN", "GHI"),
        ("era5_T_amb", "power_T2M", "temperature"),
    ]
    for era5_col, power_col, label in pairs:
        if era5_col not in df.columns or power_col not in df.columns:
            continue
        both = df[[era5_col, power_col]].dropna()
        if len(both) < 30:
            report.warn(f"{label}: too few paired rows ({len(both)}) to check correlation yet")
            continue
        corr = both[era5_col].corr(both[power_col])
        mean_abs_diff = (both[era5_col] - both[power_col]).abs().mean()
        msg = (f"{label}: correlation={corr:.3f}, mean_abs_diff={mean_abs_diff:.2f}  "
               f"(n={len(both):,})")
        if corr < 0.5:
            report.warn(f"{msg} — unexpectedly low correlation, worth spot-checking")
        else:
            report.ok(msg)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  VERIFY climate_rajasthan_points.csv")
    print(f"  File: {COMBINED_POINTS_FILE}")
    print("=" * 68)

    if not COMBINED_POINTS_FILE.exists():
        print(f"\n  ERROR: {COMBINED_POINTS_FILE} not found — "
              "run 02_combine_rajasthan.py first.")
        sys.exit(1)
    if not POPULATION_GRID_FILE.exists() or not SUNTIMES_FILE.exists():
        print("\n  ERROR: population_grid_points.csv and/or suntimes.csv not found — "
              "run 00a/00b first (needed as the source of truth for expected coverage).")
        sys.exit(1)

    df = pd.read_csv(COMBINED_POINTS_FILE)
    points_df = pd.read_csv(POPULATION_GRID_FILE)
    sun_df = pd.read_csv(SUNTIMES_FILE)

    print(f"\n  Rows in output   : {len(df):,}")
    print(f"  Points expected  : {len(points_df)}")

    report = Report()

    check_schema(df, report)
    if "point_id" in df.columns:
        actual_ids = check_point_coverage(df, points_df, report)
        check_row_coverage(df, sun_df, actual_ids, report)
    check_null_rates(df, report)
    check_ranges(df, report)
    check_cross_source_agreement(df, report)

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print(f"  Failures : {len(report.failures)}")
    print(f"  Warnings : {len(report.warnings)}")
    if report.failures:
        print("\n  ❌  FAILED — real problems found, see [FAIL] lines above.")
        sys.exit(1)
    elif report.warnings:
        print("\n  ⚠️   PASSED WITH WARNINGS — likely just an in-progress/partial run;"
              " see [WARN] lines above.")
    else:
        print("\n  ✅  ALL CHECKS PASSED.")
    print("=" * 68)


if __name__ == "__main__":
    main()
