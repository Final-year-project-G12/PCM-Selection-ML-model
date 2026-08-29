"""
03b_validate_quality_fix_rajasthan.py
=============================================================================
VALIDATION for 03b_quality_check_rajasthan.py — run AFTER that script and
AFTER 04_climate_signature_rajasthan.py has been updated to read
climate_rajasthan_points_clean.csv (already done as part of introducing
that quality-check step; see 04's own docstring note).

This is a SEPARATE script, not folded into 03b_quality_check_rajasthan.py
itself, because it re-runs 04_climate_signature_rajasthan.py as a
subprocess and diffs its output against a pre-fix backup — a genuinely
different kind of operation (downstream re-run + diff) from the quality
check's own job (detect + fix + report), and keeping them separate means
03b_quality_check_rajasthan.py stays a pure, independently-rerunnable
data-quality step with no dependency on the signature-construction script.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Independently re-verifies climate_rajasthan_points_clean.csv's row
     count against climate_rajasthan_points.csv's — does NOT just trust
     the number 03b_quality_check_rajasthan.py already printed; recomputes
     it directly from both files, since a validation step that only
     re-reads the thing-under-test's own self-report isn't independent
     verification.
  2. Prints per-point GHI/T_amb mean/std before vs after, reading
     quality_report_rajasthan.json (already computed there — no need to
     recompute what the quality-check script already measured directly
     from the same before/after arrays).
  3. Re-prints the >20%-flagged (systematic issue) points prominently.
  4. Backs up the CURRENT (pre-fix) climate_signature_rajasthan.csv if a
     backup doesn't already exist, re-runs 04_climate_signature_
     rajasthan.py (now reading the clean file), then diffs the two
     climate signature CSVs column-by-column, point-by-point, reporting
     which (point, feature) pairs shifted by more than
     SIGNATURE_DIFF_THRESHOLD_STD (a fraction of that feature's own
     cross-point std, so the threshold is scale-free across very
     differently-scaled columns like GHI_daily_kWh vs HSI_sunrise).

HOW TO RUN:
  python 03b_validate_quality_fix_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from config import (
    COMBINED_POINTS_FILE, CLEANED_POINTS_FILE, QUALITY_REPORT_JSON_FILE,
    CLIMATE_SIGNATURE_FILE, BASE_DIR,
)

SIGNATURE_BACKUP_FILE = CLIMATE_SIGNATURE_FILE.with_name(
    CLIMATE_SIGNATURE_FILE.stem + "_prefix_qualitycheck_backup.csv")

# A (point, feature) pair is reported as "shifted meaningfully" if the
# absolute change exceeds this fraction of that feature's own cross-point
# standard deviation (in the OLD signature) — scale-free across columns
# with very different natural units (GHI_daily_kWh vs HSI_sunrise etc.).
# Own choice, not a plan-doc value — documented as such.
SIGNATURE_DIFF_THRESHOLD_STD = 0.05


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    log_header("VALIDATION — Rajasthan quality-check fix")

    # --- 1. Independent row-count check ------------------------------------
    print("\n[1/4] Independent row-count check ...")
    if not CLEANED_POINTS_FILE.exists():
        raise SystemExit(f"ERROR: {CLEANED_POINTS_FILE} not found — run "
                          f"03b_quality_check_rajasthan.py first.")
    with open(COMBINED_POINTS_FILE, "r", encoding="utf-8") as f:
        n_raw = sum(1 for _ in f) - 1
    with open(CLEANED_POINTS_FILE, "r", encoding="utf-8") as f:
        n_clean = sum(1 for _ in f) - 1
    print(f"  Raw   : {n_raw:,} rows")
    print(f"  Clean : {n_clean:,} rows")
    if n_raw != n_clean:
        print(f"  [FAIL] Row counts differ by {abs(n_raw - n_clean):,} — "
              f"03b_quality_check_rajasthan.py must never drop rows. Investigate before proceeding.")
    else:
        print("  [PASS] Row counts match — no rows dropped.")

    # --- 2. Per-point before/after GHI/T_amb (from the quality report) -----
    print("\n[2/4] GHI / T_amb mean/std, before vs after (from quality_report_rajasthan.json) ...")
    if not QUALITY_REPORT_JSON_FILE.exists():
        print(f"  [SKIP] {QUALITY_REPORT_JSON_FILE} not found.")
        report = None
    else:
        with open(QUALITY_REPORT_JSON_FILE, "r", encoding="utf-8") as f:
            report = json.load(f)
        rows = []
        for pid, detail in report["per_point_detail"].items():
            for col in ("era5_GHI", "era5_T_amb"):
                d = detail.get(col, {})
                rows.append({
                    "point_id": pid, "variable": col,
                    "mean_before": d.get("mean_before"), "mean_after": d.get("mean_after"),
                    "std_before": d.get("std_before"), "std_after": d.get("std_after"),
                })
        diff_df = pd.DataFrame(rows)
        diff_df["mean_abs_change"] = (diff_df["mean_after"] - diff_df["mean_before"]).abs()
        diff_df["std_abs_change"] = (diff_df["std_after"] - diff_df["std_before"]).abs()
        print(f"  Aggregate (across all {diff_df['point_id'].nunique()} points):")
        print(diff_df.groupby("variable")[["mean_abs_change", "std_abs_change"]].mean().to_string())
        print(f"\n  Largest 5 individual point/variable shifts (mean_abs_change):")
        print(diff_df.sort_values("mean_abs_change", ascending=False).head(5)
              [["point_id", "variable", "mean_before", "mean_after", "mean_abs_change"]].to_string(index=False))

    # --- 3. Systematic-issue points ------------------------------------------
    print(f"\n[3/4] Points with >20% of a variable flagged as outlier (systematic issue) ...")
    if report is not None:
        sysflags = report.get("systematic_outlier_points", [])
        if sysflags:
            for row in sysflags:
                print(f"  [SYSTEMATIC] {row['point_id']}  {row['variable']}  {row['outlier_pct']}% flagged")
        else:
            print("  None.")

    # --- 4. Re-run 04 on clean data, diff signatures ------------------------
    log_header("[4/4] Re-running 04_climate_signature_rajasthan.py and diffing signatures")

    if CLIMATE_SIGNATURE_FILE.exists() and not SIGNATURE_BACKUP_FILE.exists():
        shutil.copy(CLIMATE_SIGNATURE_FILE, SIGNATURE_BACKUP_FILE)
        print(f"  Backed up pre-fix signature: {SIGNATURE_BACKUP_FILE}")
    elif SIGNATURE_BACKUP_FILE.exists():
        print(f"  Pre-fix backup already exists, not overwriting: {SIGNATURE_BACKUP_FILE}")
    else:
        print(f"  [WARNING] No existing {CLIMATE_SIGNATURE_FILE.name} to back up — "
              f"nothing to diff against. Run 04 once on raw data first if you want a real before/after.")

    if not SIGNATURE_BACKUP_FILE.exists():
        print("  Skipping re-run + diff (no baseline to compare against).")
        return

    old_sig = pd.read_csv(SIGNATURE_BACKUP_FILE)
    old_sig.rename(columns={old_sig.columns[0]: "point_id"}, inplace=True)

    print("\n  Running 04_climate_signature_rajasthan.py (reading clean data) ...")
    result = subprocess.run([sys.executable, "04_climate_signature_rajasthan.py"],
                             cwd=str(BASE_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print("  [FAIL] 04_climate_signature_rajasthan.py exited non-zero:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(1)
    print("  04_climate_signature_rajasthan.py completed.")

    new_sig = pd.read_csv(CLIMATE_SIGNATURE_FILE)
    new_sig.rename(columns={new_sig.columns[0]: "point_id"}, inplace=True)

    common_cols = [c for c in old_sig.columns if c in new_sig.columns and c != "point_id"
                   and pd.api.types.is_numeric_dtype(old_sig[c])]
    merged = old_sig[["point_id"] + common_cols].merge(
        new_sig[["point_id"] + common_cols], on="point_id", suffixes=("_old", "_new"))

    shifted_rows = []
    for col in common_cols:
        old_col, new_col = f"{col}_old", f"{col}_new"
        col_std = old_sig[col].std()
        if not col_std or col_std == 0 or pd.isna(col_std):
            continue
        diff = (merged[new_col] - merged[old_col]).abs()
        threshold = SIGNATURE_DIFF_THRESHOLD_STD * col_std
        flagged = merged[diff > threshold]
        for _, r in flagged.iterrows():
            shifted_rows.append({
                "point_id": r["point_id"], "feature": col,
                "old_value": r[old_col], "new_value": r[new_col],
                "abs_change": abs(r[new_col] - r[old_col]),
                "change_in_stds": abs(r[new_col] - r[old_col]) / col_std,
            })

    shifted_df = pd.DataFrame(shifted_rows)
    print(f"\n  Compared {len(common_cols)} numeric features x {len(merged)} points "
          f"= {len(common_cols)*len(merged):,} cells")
    if len(shifted_df):
        shifted_df = shifted_df.sort_values("change_in_stds", ascending=False)
        n_points_affected = shifted_df["point_id"].nunique()
        n_features_affected = shifted_df["feature"].nunique()
        print(f"  {len(shifted_df):,} (point, feature) pairs shifted by more than "
              f"{SIGNATURE_DIFF_THRESHOLD_STD} std ({n_points_affected} points, "
              f"{n_features_affected} features affected)")
        print(f"\n  Top 15 largest shifts (by std-normalized change):")
        print(shifted_df.head(15).to_string(index=False))
        print(f"\n  Features most affected (count of points shifted per feature):")
        print(shifted_df["feature"].value_counts().head(10).to_string())
    else:
        print(f"  No (point, feature) pair shifted by more than {SIGNATURE_DIFF_THRESHOLD_STD} std — "
              f"the quality-check fix was LOW-RISK for this dataset (confirmed, not assumed).")

    print("\n" + "=" * 68)
    print("  VALIDATION COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    main()
