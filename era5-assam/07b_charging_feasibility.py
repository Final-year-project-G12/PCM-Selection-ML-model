"""
07b_charging_feasibility.py  -- Assam  (OPTIONAL)
=====================================================
Heuristic regime cap on Tm_target per cluster using kt_mean/kt_std as
a poor-day solar reliability proxy. Adds Tm_target_C_regime_capped to
cluster_profiles_assam.csv. 07_feasibility_filter.py prefers this column
when present.

Assam uses REFERENCE_GOOD_DAY_TEMP_C=65 (vs Tamil Nadu=70) reflecting
higher cloud fraction. This is a stated assumption, not a measured value.

HOW TO RUN (optional — run BEFORE 07_feasibility_filter.py):
  python 07b_charging_feasibility.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

PROFILE_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_assam.csv"

REFERENCE_GOOD_DAY_TEMP_C = 65.0   # FPC delivery on a good Assam clear-sky day
MIN_ACHIEVABLE_TEMP_C     = 42.0   # absolute floor (plan Table 12)
POOR_DAY_Z                = 1.28   # ~5th percentile under normal approximation


def main():
    print("=" * 68)
    print("  OPTIONAL -- Heuristic Charging-Feasibility Regime Cap -- Assam")
    print("=" * 68)

    if not PROFILE_FILE.exists():
        print(f"  ERROR: {PROFILE_FILE} not found -- run 05_cluster_assam.py first.")
        return

    profiles = pd.read_csv(PROFILE_FILE)

    # Remap Assam column names (suffixed _mean/_std from profiling step)
    col_map = {
        "kt_mean_mean": "kt_mean",
        "kt_std_mean":  "kt_std",
        "Tm_target_mean": "Tm_target_C",
    }
    profiles.rename(columns=col_map, inplace=True)

    if "kt_mean" not in profiles.columns or "kt_std" not in profiles.columns:
        print("  ERROR: kt_mean/kt_std columns missing from cluster profiles.")
        return

    poor_day_kt  = (profiles["kt_mean"] - POOR_DAY_Z * profiles["kt_std"]).clip(lower=0.05)
    reliability  = (poor_day_kt / profiles["kt_mean"]).clip(0, 1)
    achievable_t = MIN_ACHIEVABLE_TEMP_C + reliability * (
        REFERENCE_GOOD_DAY_TEMP_C - MIN_ACHIEVABLE_TEMP_C
    )

    profiles["poor_day_kt_estimate"]      = poor_day_kt
    profiles["Tm_target_C_regime_capped"] = np.minimum(profiles["Tm_target_C"], achievable_t)

    changed = profiles["Tm_target_C_regime_capped"] < profiles["Tm_target_C"] - 0.05
    n_changed = int(changed.sum())
    n_total = len(profiles)
    print(f"  Clusters where regime cap lowers Tm_target: {n_changed}/{n_total}")
    cols_show = ["cluster_id", "kt_mean", "kt_std", "Tm_target_C",
                 "poor_day_kt_estimate", "Tm_target_C_regime_capped"]
    print(profiles[cols_show].to_string(index=False))

    profiles.to_csv(PROFILE_FILE, index=False)
    print(f"\n  Saved (column added): {PROFILE_FILE}")
    print("=" * 68)
    print("\nNext: python 07_feasibility_filter.py")


if __name__ == "__main__":
    main()
