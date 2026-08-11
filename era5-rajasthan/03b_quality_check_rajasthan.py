"""
03b_quality_check_rajasthan.py
=============================================================================
SCOPED DATA-QUALITY STEP — outlier flagging (Hampel filter) + missing-data
handling and reporting, on climate_rajasthan_points.csv, run once between
02_combine_rajasthan.py and 04_climate_signature_rajasthan.py.

WHY THIS EXISTS (and why it's narrow, not a port of Tamil Nadu's 13-step
04_preprocess_tamilnadu.py)
---------------------------------------------------------------------------
Rajasthan's pipeline has no equivalent to Tamil Nadu's Phase 2 preprocessing
script. Most of what that script does is either already covered elsewhere
in this pipeline or not needed by it:
  - physical range clipping        -> already inline in 02_combine_rajasthan.py
  - collinearity / VIF-equivalent  -> already built into 04_climate_signature_
                                       rajasthan.py's |r|>0.9 flag
  - standardization                -> already done at the signature stage
                                       (z-score), not MinMax — left unchanged
  - lag/rolling/delta features     -> built in Tamil Nadu for a downstream
                                       ML/DRL model that doesn't exist in
                                       this MCDM-based pipeline; nothing here
                                       consumes lag features, so NOT ported
Two things were genuinely missing and are what THIS script adds, nothing
else: outlier detection on the raw per-event data (silent outliers would
flow straight into Tier 1/2 mean/percentile aggregates with no flag), and
explicit, reported missing-data handling (the alternative — pandas' default
skipna=True scattered through downstream aggregation — never reports HOW
MUCH was missing or WHERE, which is exactly the kind of silent-until-
specifically-checked issue this pipeline has hit before: the ERA5
deaccumulation bug, the kappa-stepping bug, the VIKOR sign bug were all
silent until someone looked).

VARIABLES CHECKED — confirmed against the actual file, not assumed
---------------------------------------------------------------------------
climate_rajasthan_points.csv has 36 columns (era5_*, power_*, metadata).
This script checks exactly the 5 continuous climate variables that
04_climate_signature_rajasthan.py's Tier 1 construction actually reads
(its own pts_cols usecols list): era5_T_amb, era5_RHum, era5_GHI, era5_CSI,
era5_W_spd. Checking the full era5_*/power_* column set (16 + 5 more
columns) would be scope creep beyond this script's stated purpose — "does
the raw data going into Rajasthan's climate signatures have silent quality
issues" — since those extra columns never reach the signature construction
at all. power_* columns are POWER's own independent measurement (used only
for the already-completed 03b_agreement_analysis.py cross-check), not fed
into Tier 1/2 either, so also out of scope here.

HAMPEL WINDOW — corrected TWICE, both times empirically, not by assumption
---------------------------------------------------------------------------
The brief suggested "window=145 for hourly ~6-day windows." This dataset is
NOT continuous hourly data — it's 3 discrete sun-events/day (sunrise/noon/
sunset), exactly the same structural fact that governs every other
rolling/lag concept in this pipeline (see signature_lib.py, 04's Tier 1
construction). A window of 145 would span ~145 DAYS per (point_id, event)
series, not 6 — a >20x error if applied literally. First correction: window
redefined in OCCURRENCES of the same (point_id, event) series (matching
Tamil Nadu's own Hampel filter convention), sized to HAMPEL_WINDOW_EACH_
SIDE=3 (7-occurrence window) to match "roughly 5-7 days" literally.

SECOND CORRECTION (2026-08-11, after inspecting actual flagged values, not
just the aggregate flag rate): the 7-day window was too narrow for GHI's
real synoptic-scale day-to-day variability. Inspecting individual flagged
rows directly (point RJP_0171: raw GHI=304.6 W/m^2 on 2016-02-19, a
physically ordinary partly-cloudy/hazy noon reading, "corrected" to
779.6 W/m^2 — a 2.5x change) showed the filter was systematically
misclassifying ordinary cloud/dust-driven variability as anomalies, not
catching genuine data glitches. Signature evidence this was a real problem,
not a false alarm: GHI_noon_mean increased and kt_noon_std decreased at
ALL 320 points after the first run — a uniform, same-direction shift across
every independent point is not what genuine scattered-anomaly correction
looks like; it's what systematically eroding the low tail of a real
distribution looks like. HAMPEL_WINDOW_EACH_SIDE widened to 15 occurrences
(matching Tamil Nadu's own precedent script exactly — "~a month either
side") to give the local median/MAD baseline enough data to distinguish a
genuinely anomalous single day from an ordinary multi-day cloudy/hazy
stretch. Re-validate (03b_validate_quality_fix_rajasthan.py) after this
change and confirm the uniform-shift signature is gone before trusting the
output — this is a check to actually run, not a change to assume fixed it.

METHODOLOGY
---------------------------------------------------------------------------
1. Outlier detection: per (point_id, event) series, sorted by date, a
   rolling median/MAD (window = HAMPEL_WINDOW_EACH_SIDE occurrences each
   side). A value is flagged if |x - rolling_median| > HAMPEL_N_SIGMA *
   MAD_SCALE * rolling_MAD. Flags are recorded in {var}_outlier_flag
   columns in BOTH outputs; only in climate_rajasthan_points_clean.csv are
   flagged values replaced with the rolling median (winsorize-style) —
   never silently, always alongside the flag that makes it auditable.
   Missing values are never flagged as outliers (can't be both).
2. Missing-data handling, applied AFTER outlier winsorizing (so gap-fill
   never has to reason about a value that's about to be overwritten as an
   outlier): (a) linear interpolation for gaps of SHORT_GAP_MAX_OCCURRENCES
   (3) or fewer consecutive occurrences within a (point_id, event) series;
   (b) remaining gaps filled with that SAME POINT's own seasonal mean for
   that (event, season) combination — deliberately NOT a global or
   cross-point mean, to avoid leaking other points' climate into this
   one's gaps; (c) final fallback (should affect ~0 rows) — that point's
   own (event)-level mean, for the pathological case where an entire
   season is missing for one point/event. No step here uses pandas'
   default skipna=True implicitly; every fill is one of these three named,
   counted operations.
3. Missingness >MISSINGNESS_THRESHOLD_PCT (5%) in a CRITICAL_VAR (GHI or
   T_amb) for a point, BEFORE imputation, adds that point to the "review"
   list — imputation still runs (so downstream scripts get a complete
   file), but the report says explicitly that point's fill is doing
   proportionally more work than a few-percent-missing point's.

OUTPUTS:
  data/processed/climate_rajasthan_points_clean.csv  — same schema as the
      input plus 5 new {var}_outlier_flag boolean columns; row count
      identical to input (nothing dropped).
  data/processed/quality_report_rajasthan.md   — human-readable summary.
  data/processed/quality_report_rajasthan.json — machine-readable version
      of the same summary (full per-point/per-variable detail), for
      reproducibility checks.

WHAT THIS SCRIPT DOES NOT TOUCH:
  02_combine_rajasthan.py (upstream, already has range clipping)
  Lag/rolling/delta feature engineering (not needed downstream)
  MinMax scaling (Rajasthan uses z-score at the signature stage already)

HOW TO RUN:
  python 03b_quality_check_rajasthan.py

VALIDATION (run separately, AFTER this script and AFTER
04_climate_signature_rajasthan.py's input path is updated to read
climate_rajasthan_points_clean.csv):
  python 03b_validate_quality_fix_rajasthan.py
  — confirms row counts match, prints before/after GHI/T_amb stats,
  flags any point with >20% of a variable marked outlier (likely a
  systematic issue, not genuine outliers), and diffs the climate
  signature CSV built from clean data against the pre-fix version.
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import COMBINED_POINTS_FILE, CLEANED_POINTS_FILE, \
    QUALITY_REPORT_MD_FILE, QUALITY_REPORT_JSON_FILE, ensure_data_dirs

ensure_data_dirs()

EVENT_ORDER = ["sunrise", "noon", "sunset"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]

# Exactly the 5 variables 04_climate_signature_rajasthan.py's Tier 1
# construction reads — see module docstring for why this list, not the
# full era5_*/power_* column set. All 5 get missingness reporting/
# imputation; only HAMPEL_VARS (a subset) get outlier detection.
QUALITY_VARS = ["era5_T_amb", "era5_RHum", "era5_GHI", "era5_CSI", "era5_W_spd"]
CRITICAL_VARS = ["era5_GHI", "era5_T_amb"]   # missingness threshold applies to these

# THIRD CORRECTION (2026-08-11, same day, after widening the window made
# GHI/CSI's uniform-shift signature WORSE, not better — see module
# docstring). era5_GHI and era5_CSI are EXCLUDED from Hampel outlier
# detection/winsorizing entirely, at any window width. Their low-tail
# values (cloudy/hazy days) are real, wanted climate signal that
# cloudy_frac, CCI, kt_daily_std, and monsoon_index (Tier 2 indices this
# whole pipeline's clustering depends on) exist specifically to measure —
# not sensor noise. A MAD-based local-median filter structurally cannot
# tell "rare glitch" apart from "this point has a real, recurring
# low-clearness tail because Rajasthan has real weather" for a variable
# whose legitimate distribution is dominated by clear days with a genuine
# heavy low tail; widening the window made this WORSE because a longer
# clear-sky-dominated baseline makes real cloudy stretches look even more
# like deviations, not less. T_amb/RHum/W_spd do NOT have this problem —
# they showed clean, dramatic improvement from the same window widening
# (7.1%->1.6%, 7.2%->2.2%, 6.7%->2.5%), consistent with THEIR flagged
# values being genuine isolated glitches, not legitimate distributional
# tails. GHI/CSI's existing quality gate remains 02_combine_rajasthan.py's
# inline physical-range clipping ([0,1400] W/m^2, SZA-based night zeroing)
# — deliberately not supplemented with a statistical outlier filter here.
HAMPEL_VARS = ["era5_T_amb", "era5_RHum", "era5_W_spd"]

# --- Hampel filter (outlier detection) --------------------------------
# WIDENED 2026-08-11 from 3 to 15 (occurrences each side) after the narrow
# 7-day window was empirically shown to winsorize genuine cloud/dust-driven
# GHI variability, not real anomalies — see module docstring "SECOND
# CORRECTION" for the evidence (a uniform, same-direction GHI_noon_mean/
# kt_noon_std shift at all 320 points is not what real outlier correction
# looks like). Matches Tamil Nadu's own precedent script exactly.
HAMPEL_WINDOW_EACH_SIDE = 15      # occurrences each side -> 31-occurrence window
                                   # (~a month), in OCCURRENCES not hours —
                                   # see module docstring.
HAMPEL_N_SIGMA = 3.0
MAD_SCALE = 1.4826                 # MAD -> sigma consistency constant

# --- Missingness / imputation -----------------------------------------
MISSINGNESS_THRESHOLD_PCT = 5.0    # X% — points above this in a CRITICAL_VAR go on the review list
SHORT_GAP_MAX_OCCURRENCES = 3      # linear-interpolation cutoff, in occurrences

# --- Validation ----------------------------------------------------------
SYSTEMATIC_OUTLIER_PCT_FLAG = 20.0   # >this% flagged for one point/var = likely systematic, not genuine outliers


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def file_fingerprint(path):
    """mtime + size + row count, not a full SHA256 of a 1.5GB file — this
    is a deliberate speed/reproducibility tradeoff, not an oversight: for
    this purpose (detecting whether the INPUT changed between runs),
    mtime+size+row_count is already highly discriminating, and hashing
    1.5GB adds real runtime for marginal benefit over that. Documented
    here rather than silently picked."""
    stat = path.stat()
    with open(path, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1   # minus header
    return {
        "path": str(path), "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "size_bytes": stat.st_size, "row_count": row_count,
        "fingerprint_method": "mtime+size+row_count (not full-file hash — see file_fingerprint() docstring)",
    }


# ═══════════════════════════════════════════════════════════
# 1. HAMPEL OUTLIER DETECTION + WINSORIZING
# ═══════════════════════════════════════════════════════════

def hampel_flag_and_winsorize(df, col):
    """Returns (is_outlier: bool Series, winsorized: float Series). df must
    already be sorted by [point_id, event, date]. Missing values are never
    flagged (can't be both missing and an outlier) and pass through
    unchanged into the winsorized series."""
    window = HAMPEL_WINDOW_EACH_SIDE * 2 + 1
    grp = df.groupby(["point_id", "event"], observed=True)[col]

    roll_median = grp.transform(
        lambda s: s.rolling(window, center=True, min_periods=3).median())
    roll_mad = grp.transform(
        lambda s: (s - s.rolling(window, center=True, min_periods=3).median())
        .abs().rolling(window, center=True, min_periods=3).median())

    threshold = HAMPEL_N_SIGMA * MAD_SCALE * roll_mad
    is_outlier = (df[col] - roll_median).abs() > threshold
    is_outlier &= roll_mad > 1e-9        # skip flat/constant stretches (mad=0 would flag everything)
    is_outlier &= df[col].notna()        # a missing value is not an outlier

    winsorized = df[col].where(~is_outlier, roll_median)
    return is_outlier.fillna(False), winsorized


# ═══════════════════════════════════════════════════════════
# 2. MISSING-DATA IMPUTATION  (three named, counted operations — no
#    implicit skipna=True fallback anywhere)
# ═══════════════════════════════════════════════════════════

def impute_missing(df, col):
    """Returns (imputed: float Series, counts: dict of how many cells each
    of the three named operations filled). Operates on df already sorted
    by [point_id, event, date], AFTER outlier winsorizing (see module
    docstring for why that order)."""
    n_missing_start = int(df[col].isna().sum())

    # (a) Linear interpolation, short interior gaps only (<=3 occurrences)
    s_interp = df.groupby(["point_id", "event"], observed=True)[col].transform(
        lambda s: s.interpolate(method="linear", limit=SHORT_GAP_MAX_OCCURRENCES, limit_area="inside"))
    n_after_interp = int(s_interp.isna().sum())
    n_interp_filled = n_missing_start - n_after_interp

    # (b) That SAME point's own seasonal mean for that (event, season) —
    # never a global or cross-point fallback.
    season_mean = df.assign(**{col: s_interp}).groupby(
        ["point_id", "event", "season"], observed=True)[col].transform("mean")
    still_missing_b = s_interp.isna()
    s_seasonal = s_interp.where(~still_missing_b, season_mean)
    n_after_seasonal = int(s_seasonal.isna().sum())
    n_seasonal_filled = n_after_interp - n_after_seasonal

    # (c) Final fallback — that point's own (event)-level mean, for the
    # pathological case where an entire season is missing for one
    # point/event (should affect ~0 rows; counted, not assumed zero).
    point_event_mean = df.assign(**{col: s_seasonal}).groupby(
        ["point_id", "event"], observed=True)[col].transform("mean")
    still_missing_c = s_seasonal.isna()
    s_final = s_seasonal.where(~still_missing_c, point_event_mean)
    n_after_fallback = int(s_final.isna().sum())
    n_fallback_filled = n_after_seasonal - n_after_fallback

    counts = {
        "missing_before": n_missing_start,
        "filled_by_interpolation": n_interp_filled,
        "filled_by_point_seasonal_mean": n_seasonal_filled,
        "filled_by_point_event_fallback_mean": n_fallback_filled,
        "still_missing_after_all_steps": n_after_fallback,
    }
    return s_final, counts


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    log_header("PHASE 2.5 (SCOPED) — DATA QUALITY CHECK — Rajasthan")
    print(f"  Input : {COMBINED_POINTS_FILE}")
    print(f"  Vars  : {QUALITY_VARS}")
    print(f"  Hampel window: +/-{HAMPEL_WINDOW_EACH_SIDE} occurrences "
          f"({HAMPEL_WINDOW_EACH_SIDE*2+1} total), n_sigma={HAMPEL_N_SIGMA}")

    if not COMBINED_POINTS_FILE.exists():
        raise SystemExit(f"ERROR: {COMBINED_POINTS_FILE} not found — run 02_combine_rajasthan.py first.")

    print("\n[1/6] Fingerprinting input file (for reproducibility) ...")
    fingerprint = file_fingerprint(COMBINED_POINTS_FILE)
    print(f"  {fingerprint}")

    print("\n[2/6] Loading full file (all columns — output schema must match input) ...")
    df = pd.read_csv(COMBINED_POINTS_FILE, parse_dates=["date"])
    df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
    df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
    df = df.sort_values(["point_id", "event", "date"]).reset_index(drop=True)
    n_rows_in = len(df)
    n_points = df["point_id"].nunique()
    print(f"  {n_rows_in:,} rows, {n_points} points, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    print("\n[3/6] Missingness (BEFORE any change) ...")
    missing_before_pct_global = {}
    missing_before_pct_by_point = {}
    for col in QUALITY_VARS:
        pct_global = float(df[col].isna().mean() * 100)
        missing_before_pct_global[col] = pct_global
        by_point = df.groupby("point_id", observed=True)[col].apply(lambda s: float(s.isna().mean() * 100))
        missing_before_pct_by_point[col] = by_point.to_dict()
        print(f"  {col:14s}: {pct_global:.3f}% missing overall")

    high_missingness_points = set()
    for col in CRITICAL_VARS:
        for pid, pct in missing_before_pct_by_point[col].items():
            if pct > MISSINGNESS_THRESHOLD_PCT:
                high_missingness_points.add((pid, col, round(pct, 3)))
    if high_missingness_points:
        print(f"  [REVIEW] {len(high_missingness_points)} (point, critical-var) pairs exceed "
              f"{MISSINGNESS_THRESHOLD_PCT}% missingness:")
        for pid, col, pct in sorted(high_missingness_points, key=lambda x: -x[2])[:10]:
            print(f"    {pid}  {col}  {pct}%")
    else:
        print(f"  No point exceeds {MISSINGNESS_THRESHOLD_PCT}% missingness in a critical variable.")

    print("\n[4/6] Hampel outlier detection + winsorizing "
          f"(HAMPEL_VARS only: {HAMPEL_VARS} — era5_GHI/era5_CSI deliberately excluded, see module docstring) ...")
    outlier_counts_global = {}
    outlier_counts_by_point = {}
    pre_stats = {col: df.groupby("point_id", observed=True)[col].agg(["mean", "std"]).to_dict("index")
                 for col in QUALITY_VARS}

    for col in HAMPEL_VARS:
        is_outlier, winsorized = hampel_flag_and_winsorize(df, col)
        df[f"{col}_outlier_flag"] = is_outlier
        df[col] = winsorized   # winsorized in place — flag column preserves the audit trail

        n_flagged = int(is_outlier.sum())
        outlier_counts_global[col] = n_flagged
        by_point_n = df.groupby("point_id", observed=True)[f"{col}_outlier_flag"].sum()
        by_point_total = df.groupby("point_id", observed=True)[col].size()
        by_point_pct = (by_point_n / by_point_total * 100)
        outlier_counts_by_point[col] = {
            pid: {"n_flagged": int(by_point_n[pid]), "pct_flagged": float(by_point_pct[pid])}
            for pid in by_point_n.index
        }
        print(f"  {col:14s}: {n_flagged:,} values flagged/winsorized "
              f"({100*n_flagged/len(df):.3f}% of rows)")

    for col in QUALITY_VARS:
        if col not in HAMPEL_VARS:
            outlier_counts_global[col] = None
            outlier_counts_by_point[col] = {
                pid: {"n_flagged": None, "pct_flagged": None} for pid in df["point_id"].unique()
            }
            print(f"  {col:14s}: NOT CHECKED (excluded from Hampel detection — see module docstring)")

    # post_stats computed AFTER Hampel (HAMPEL_VARS reflect winsorizing;
    # non-Hampel vars are identical to pre_stats since untouched here —
    # confirms "not checked" means "not changed", not silently different).
    post_stats = {col: df.groupby("point_id", observed=True)[col].agg(["mean", "std"]).to_dict("index")
                  for col in QUALITY_VARS}

    systematic_flags = []
    for col in HAMPEL_VARS:
        for pid, stats in outlier_counts_by_point[col].items():
            if stats["pct_flagged"] > SYSTEMATIC_OUTLIER_PCT_FLAG:
                systematic_flags.append((pid, col, round(stats["pct_flagged"], 2)))
    if systematic_flags:
        print(f"\n  [SYSTEMATIC ISSUE] {len(systematic_flags)} (point, var) pairs have "
              f">{SYSTEMATIC_OUTLIER_PCT_FLAG}% of values flagged as outliers — likely a "
              f"bad sensor/merge glitch, NOT genuine scattered outliers:")
        for pid, col, pct in sorted(systematic_flags, key=lambda x: -x[2]):
            print(f"    {pid}  {col}  {pct}% flagged")

    print("\n[5/6] Missing-data imputation ...")
    imputation_counts = {}
    for col in QUALITY_VARS:
        imputed, counts = impute_missing(df, col)
        df[col] = imputed
        imputation_counts[col] = counts
        print(f"  {col:14s}: {counts}")
        if counts["still_missing_after_all_steps"] > 0:
            print(f"    [WARNING] {counts['still_missing_after_all_steps']} cells still missing "
                  f"after all imputation steps for {col} — investigate before trusting downstream aggregates.")

    n_rows_out = len(df)
    assert n_rows_out == n_rows_in, \
        f"Row count changed ({n_rows_in} -> {n_rows_out}) — this script must never drop rows."
    print(f"\n  Row count preserved: {n_rows_out:,} == {n_rows_in:,}")

    CLEANED_POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_POINTS_FILE, index=False)
    print(f"  Saved: {CLEANED_POINTS_FILE}")

    print("\n[6/6] Writing quality reports ...")
    run_timestamp = datetime.now().isoformat()

    per_point_summary = {}
    for pid in df["point_id"].unique():
        per_point_summary[pid] = {}
        for col in QUALITY_VARS:
            outlier_stats = outlier_counts_by_point[col][pid]
            per_point_summary[pid][col] = {
                "missingness_pct_before": round(missing_before_pct_by_point[col].get(pid, 0.0), 3),
                "hampel_checked": col in HAMPEL_VARS,
                "outlier_n_flagged": outlier_stats["n_flagged"],
                "outlier_pct_flagged": round(outlier_stats["pct_flagged"], 3)
                                       if outlier_stats["pct_flagged"] is not None else None,
                "mean_before": round(pre_stats[col][pid]["mean"], 4) if pd.notna(pre_stats[col][pid]["mean"]) else None,
                "std_before": round(pre_stats[col][pid]["std"], 4) if pd.notna(pre_stats[col][pid]["std"]) else None,
                "mean_after": round(post_stats[col][pid]["mean"], 4) if pd.notna(post_stats[col][pid]["mean"]) else None,
                "std_after": round(post_stats[col][pid]["std"], 4) if pd.notna(post_stats[col][pid]["std"]) else None,
            }

    review_list = sorted(
        {pid for pid, _, _ in high_missingness_points} | {pid for pid, _, _ in systematic_flags})

    json_report = {
        "run_timestamp": run_timestamp,
        "input_fingerprint": fingerprint,
        "config": {
            "quality_vars": QUALITY_VARS, "critical_vars": CRITICAL_VARS,
            "hampel_vars": HAMPEL_VARS,
            "hampel_excluded_vars": [c for c in QUALITY_VARS if c not in HAMPEL_VARS],
            "hampel_excluded_reason": "real cloud-driven low-tail variability, not sensor noise — see module docstring THIRD CORRECTION",
            "hampel_window_each_side_occurrences": HAMPEL_WINDOW_EACH_SIDE,
            "hampel_n_sigma": HAMPEL_N_SIGMA, "mad_scale": MAD_SCALE,
            "missingness_threshold_pct": MISSINGNESS_THRESHOLD_PCT,
            "short_gap_max_occurrences": SHORT_GAP_MAX_OCCURRENCES,
            "systematic_outlier_pct_flag": SYSTEMATIC_OUTLIER_PCT_FLAG,
        },
        "totals": {
            "rows": n_rows_in, "points": n_points,
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "missingness_pct_global_before": {k: round(v, 4) for k, v in missing_before_pct_global.items()},
            "outliers_flagged_global": outlier_counts_global,
            "imputation_counts_global": imputation_counts,
        },
        "high_missingness_points": [
            {"point_id": pid, "variable": col, "missingness_pct": pct}
            for pid, col, pct in sorted(high_missingness_points, key=lambda x: -x[2])
        ],
        "systematic_outlier_points": [
            {"point_id": pid, "variable": col, "outlier_pct": pct}
            for pid, col, pct in sorted(systematic_flags, key=lambda x: -x[2])
        ],
        "review_list": review_list,
        "per_point_detail": per_point_summary,
    }
    with open(QUALITY_REPORT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"  Saved: {QUALITY_REPORT_JSON_FILE}")

    md_lines = [
        "# Rajasthan Climate Data — Quality Report",
        f"\nRun: {run_timestamp}",
        f"\nInput: `{COMBINED_POINTS_FILE.name}`  "
        f"(mtime={fingerprint['mtime']}, {fingerprint['row_count']:,} rows, "
        f"{fingerprint['size_bytes']:,} bytes)",
        f"\nOutput: `{CLEANED_POINTS_FILE.name}` ({n_rows_out:,} rows — row count preserved, nothing dropped)",
        "\n## Configuration",
        f"- Variables checked (missingness + imputation): {', '.join(QUALITY_VARS)}",
        f"- Variables Hampel-checked for outliers: {', '.join(HAMPEL_VARS)}",
        f"- Variables EXCLUDED from Hampel outlier detection: "
        f"{', '.join(c for c in QUALITY_VARS if c not in HAMPEL_VARS)} — "
        f"real cloud-driven low-tail variability that cloudy_frac/CCI/kt_daily_std/monsoon_index "
        f"are designed to measure, not sensor noise; see script docstring THIRD CORRECTION for "
        f"the empirical evidence (widening the window made these variables' uniform-shift signature "
        f"WORSE, not better, ruling out a simple window-tuning fix).",
        f"- Critical variables (missingness threshold applies): {', '.join(CRITICAL_VARS)}",
        f"- Hampel window: +/-{HAMPEL_WINDOW_EACH_SIDE} occurrences "
        f"({HAMPEL_WINDOW_EACH_SIDE*2+1} total, ~{HAMPEL_WINDOW_EACH_SIDE*2+1} days), "
        f"n_sigma={HAMPEL_N_SIGMA}, MAD scale={MAD_SCALE}",
        f"- Missingness review threshold: {MISSINGNESS_THRESHOLD_PCT}% in a critical variable",
        f"- Short-gap linear interpolation cutoff: {SHORT_GAP_MAX_OCCURRENCES} occurrences",
        f"- Systematic-issue outlier threshold: {SYSTEMATIC_OUTLIER_PCT_FLAG}% of a point's values",
        "\n## Global summary",
        "\n| Variable | Missing % (before) | Outliers flagged | Outlier % |",
        "|---|---|---|---|",
    ]
    for col in QUALITY_VARS:
        if col in HAMPEL_VARS:
            md_lines.append(f"| {col} | {missing_before_pct_global[col]:.3f}% | "
                             f"{outlier_counts_global[col]:,} | "
                             f"{100*outlier_counts_global[col]/n_rows_in:.3f}% |")
        else:
            md_lines.append(f"| {col} | {missing_before_pct_global[col]:.3f}% | "
                             f"NOT CHECKED | NOT CHECKED (see Configuration note above) |")

    md_lines.append("\n## Imputation breakdown (global)")
    md_lines.append("\n| Variable | Missing (before) | By interpolation | By point-seasonal mean | By fallback mean | Still missing |")
    md_lines.append("|---|---|---|---|---|---|")
    for col in QUALITY_VARS:
        c = imputation_counts[col]
        md_lines.append(f"| {col} | {c['missing_before']:,} | {c['filled_by_interpolation']:,} | "
                         f"{c['filled_by_point_seasonal_mean']:,} | "
                         f"{c['filled_by_point_event_fallback_mean']:,} | "
                         f"{c['still_missing_after_all_steps']:,} |")

    md_lines.append(f"\n## Points exceeding {MISSINGNESS_THRESHOLD_PCT}% missingness in a critical variable")
    if high_missingness_points:
        md_lines.append("\n| Point | Variable | Missing % |")
        md_lines.append("|---|---|---|")
        for pid, col, pct in sorted(high_missingness_points, key=lambda x: -x[2]):
            md_lines.append(f"| {pid} | {col} | {pct}% |")
    else:
        md_lines.append(f"\nNone — every point is under {MISSINGNESS_THRESHOLD_PCT}% missingness "
                         f"in both critical variables.")

    md_lines.append(f"\n## Points with >{SYSTEMATIC_OUTLIER_PCT_FLAG}% of a variable flagged as outliers "
                     f"(likely systematic, not genuine outliers)")
    if systematic_flags:
        md_lines.append("\n| Point | Variable | Outlier % |")
        md_lines.append("|---|---|---|")
        for pid, col, pct in sorted(systematic_flags, key=lambda x: -x[2]):
            md_lines.append(f"| {pid} | {col} | {pct}% |")
    else:
        md_lines.append(f"\nNone — no point/variable exceeds {SYSTEMATIC_OUTLIER_PCT_FLAG}% flagged.")

    md_lines.append("\n## Review list (union of the two tables above)")
    if review_list:
        md_lines.append(f"\n{len(review_list)} point(s) warrant a manual look before trusting their "
                         f"downstream Tier 1/2 aggregates: {', '.join(review_list)}")
    else:
        md_lines.append("\nNo points flagged for review.")

    md_lines.append("\n## Methodology notes")
    md_lines.append(
        "\n- Outlier detection: per (point_id, event) series (sunrise/noon/sunset kept separate — "
        "never compared across event types), a rolling median/MAD Hampel filter. Flagged values are "
        "winsorized (replaced with the local rolling median) ONLY in the _clean.csv output; the "
        "{var}_outlier_flag columns record every flag so this is auditable, not silent.\n"
        "- Missing-data imputation, in order: (1) linear interpolation for gaps of "
        f"{SHORT_GAP_MAX_OCCURRENCES} or fewer consecutive occurrences; (2) that same point's own "
        "seasonal mean (per event x season) for longer gaps; (3) that point's own event-level mean "
        "as a final fallback for the rare case an entire season is missing. No global or cross-point "
        "fill is used at any step.\n"
        "- Hampel window is sized in OCCURRENCES of the same (point_id, event) series (roughly "
        f"{HAMPEL_WINDOW_EACH_SIDE*2+1} days), not hours — this dataset is 3 discrete sun-events/day, "
        "not continuous hourly data. See the script's module docstring for why an hourly-sized window "
        "would have been a >20x error here.\n"
    )

    with open(QUALITY_REPORT_MD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {QUALITY_REPORT_MD_FILE}")

    log_header("DONE")
    print(f"  {n_rows_out:,} rows, {n_points} points")
    print(f"  Review list: {review_list or 'none'}")
    print(f"\nNext: update 04_climate_signature_rajasthan.py to read "
          f"{CLEANED_POINTS_FILE.name} (already done if you're re-running this), "
          f"then re-run it, then run 03b_validate_quality_fix_rajasthan.py.")
    print("=" * 68)


if __name__ == "__main__":
    main()
