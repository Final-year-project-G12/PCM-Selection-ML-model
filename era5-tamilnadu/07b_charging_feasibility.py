"""
07b_charging_feasibility.py   (OPTIONAL)
===========================================
Implements the ONE filter from plan v3.0 Table 12 that 07_feasibility_filter.py
explicitly does NOT apply: "Tm must lie below the collector delivery
temperature achievable on a poor day in that regime." This is the filter
the plan doc says "makes the target regime-dependent" — without it, every
cluster shares the same constant Tm_target and the same feasibility
window, so every cluster gets an identical survivor list (see 07's
output — you'll have seen this if you ran it before this script).

IMPORTANT HONESTY NOTE — read before using this
----------------------------------------------------
This is a HEURISTIC PROXY, not a real collector thermal model. A rigorous
version needs the cluster's 5th-percentile daily insolation fed through an
actual collector efficiency curve (eta_th = F_R[S - U*(T_in-T_amb)/G], as
several of your literature summaries already have) — that's Phase 7
territory, not something to improvise here under deadline pressure.

What this script does instead: it estimates each cluster's "poor-day"
solar reliability from kt_mean and kt_std already in
climate_signature_tamilnadu.csv (assuming roughly-normal day-to-day
clearness), and scales a single reference "good-day" achievable collector
delivery temperature down proportionally for less reliable clusters. The
scaling constants (REFERENCE_GOOD_DAY_TEMP, MIN_ACHIEVABLE_TEMP) are
stated assumptions, not measured values — say so explicitly if you use
this in your paper, exactly the same way you'd flag any other assumption.

If you decide the heuristic is defensible enough to use: run this BEFORE
07_feasibility_filter.py, then re-run 07 and 08. If you decide it's too
speculative for a final-year paper without more validation: don't run it,
and report the constant-Tm_target convergence honestly instead (08's
diagnostic message tells you how to phrase that).

INPUT  : data/processed/signatures/climate_signature_tamilnadu.csv
         data/processed/clustering/cluster_profiles_tamilnadu.csv
OUTPUT : data/processed/clustering/cluster_profiles_tamilnadu.csv
           (adds one new column: Tm_target_C_regime_capped — 07 will use
           this column INSTEAD of Tm_target_C if it exists; the original
           Tm_target_C column is left untouched)

HOW TO RUN (optional):
  python 07b_charging_feasibility.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

PROFILE_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_tamilnadu.csv"

# Stated assumptions — not measured. A flat-plate collector under good
# clear-sky Tamil Nadu insolation can plausibly deliver up to ~70C
# (roughly consistent with Al-Mamun2023's cited FPC 25-100C operating
# band); on a genuinely poor day, delivery drops toward something closer
# to the absolute floor this project already uses (42C).
REFERENCE_GOOD_DAY_TEMP_C = 70.0
MIN_ACHIEVABLE_TEMP_C = 42.0
POOR_DAY_Z = 1.28   # ~5th percentile under a normal approximation


def main():
    print("=" * 68)
    print("  OPTIONAL — Heuristic Charging-Feasibility Regime Cap")
    print("=" * 68)

    if not PROFILE_FILE.exists():
        print(f"\n  ERROR: {PROFILE_FILE} not found — run 05_cluster_tamilnadu.py first.")
        return

    profiles = pd.read_csv(PROFILE_FILE)
    if "kt_mean" not in profiles.columns or "kt_std" not in profiles.columns:
        print("\n  ERROR: cluster_profiles_tamilnadu.csv is missing kt_mean/kt_std — "
              "these should be carried through from the signature matrix by "
              "05_cluster_tamilnadu.py's population-weighted profile step.")
        return

    poor_day_kt = (profiles["kt_mean"] - POOR_DAY_Z * profiles["kt_std"]).clip(lower=0.05)
    reliability_ratio = (poor_day_kt / profiles["kt_mean"]).clip(0, 1)

    achievable_temp = (MIN_ACHIEVABLE_TEMP_C +
                        reliability_ratio * (REFERENCE_GOOD_DAY_TEMP_C - MIN_ACHIEVABLE_TEMP_C))

    profiles["poor_day_kt_estimate"] = poor_day_kt
    profiles["Tm_target_C_regime_capped"] = np.minimum(profiles["Tm_target_C"], achievable_temp)

    changed = (profiles["Tm_target_C_regime_capped"] < profiles["Tm_target_C"] - 0.05)
    print(f"\n  Clusters where the regime cap actually lowers Tm_target: "
          f"{changed.sum()}/{len(profiles)}")
    print(profiles[["cluster_id", "kt_mean", "kt_std", "Tm_target_C",
                     "Tm_target_C_regime_capped"]].to_string(index=False))

    profiles.to_csv(PROFILE_FILE, index=False)
    print(f"\n  Saved (column added): {PROFILE_FILE}")
    print("=" * 68)
    print("\nNext: run 07_feasibility_filter.py — it will automatically prefer")
    print("Tm_target_C_regime_capped over Tm_target_C now that the column exists.")
    print("(If you want to go back to the constant rule, just delete this")
    print("column from cluster_profiles_tamilnadu.csv or re-run 05 to regenerate it.)")


if __name__ == "__main__":
    main()
