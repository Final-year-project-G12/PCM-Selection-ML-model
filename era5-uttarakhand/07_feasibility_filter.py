"""
07_feasibility_filter.py
===========================
PHASE 5 — FEASIBILITY FILTERING (Objective 1 plan v3.0, Section 8, Table 12)

Hard-filters the PCM database against EACH cluster's Tm_target/L_required
before any MCDM ranking runs. This matters because MCDM is compensatory —
a PCM with an unreachable melting point but great latent heat can still
score well in TOPSIS and be physically useless. Filtering first prevents
that (plan v3.0 Section 8, opening paragraph).

Filters applied (Table 12; two are noted as NOT applied — see below):
  1. Melting window     : Tm in [Tm_target-5, Tm_target+8] C
  2. Absolute band       : Tm in [42, 70] C regardless of cluster
  3. Latent heat floor   : L >= 0.7 x L_required for that cluster
  4. Cycling stability   : >=300 cycles where reported; RETAINED and
                            FLAGGED (not excluded) where not reported —
                            per the plan doc, absence of data is not
                            evidence of failure.
  5. Supercooling veto    : exclude if supercooling > 8K (only applies
                            where the value is known; NaN passes through
                            flagged, not excluded)
  NOT applied (need data this project doesn't have yet — flagged as
  future work, not silently skipped):
    - Charging feasibility at the cluster's 5th-percentile insolation day
      (needs a full daily GHI percentile per cluster, not just the mean
      in cluster_profiles_uttarakhand.csv)
    - Corrosion veto against cluster HSI 75th percentile (needs a real
      corrosion_class per PCM; the database currently only distinguishes
      "low_organic" vs "check_manually" for the one inorganic PCM)
    - Safety exclusion (no toxicity data in the current database)

If a cluster keeps fewer than 5 candidates, the melting window is
automatically relaxed by 2K and retried (per Section 8's stated rule). If
a cluster keeps more than 25, that's reported but not narrowed further —
Phase 6's ranking is what should separate them.

INPUT  : data/processed/pcm/pcm_database_uttarakhand.csv        (06's output)
         data/processed/clustering/cluster_profiles_uttarakhand.csv (05's output)
OUTPUT : data/processed/pcm/feasibility_survivors_by_cluster.csv
           one row per (cluster_id, pcm_name) that survived, with the
           per-filter pass/fail detail kept alongside for your methodology
           section's survivor-count table.

HOW TO RUN:
  python 07_feasibility_filter.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

PCM_FILE = PROCESSED_DIR / "pcm" / "pcm_database_uttarakhand.csv"
PROFILE_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_uttarakhand.csv"
OUT_FILE = PROCESSED_DIR / "pcm" / "feasibility_survivors_by_cluster.csv"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7
CYCLES_FLOOR = 300
SUPERCOOLING_MAX_K = 8.0
MIN_SURVIVORS, MAX_RELAX_STEPS, RELAX_STEP_K = 5, 4, 2.0


def filter_cluster(pcm_db, tm_target, l_required, window_relax=0.0):
    lo = tm_target - WINDOW_LOWER_OFFSET - window_relax
    hi = tm_target + WINDOW_UPPER_OFFSET + window_relax

    df = pcm_db.copy()
    df["pass_melting_window"] = df["Tm_C"].between(lo, hi)
    df["pass_absolute_band"] = df["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX)
    l_floor = LATENT_HEAT_FRACTION * l_required
    df["pass_latent_heat"] = df["latent_heat_kJ_kg"] >= l_floor
    df["latent_heat_floor_used"] = l_floor

    df["cycling_known"] = df["cycles_tested"].notna()
    df["pass_cycling"] = np.where(df["cycling_known"], df["cycles_tested"] >= CYCLES_FLOOR, True)
    df["cycling_flag"] = np.where(~df["cycling_known"], "not_reported", "")

    df["supercooling_known"] = df["supercooling_K"].notna()
    df["pass_supercooling"] = np.where(
        df["supercooling_known"], df["supercooling_K"].abs() <= SUPERCOOLING_MAX_K, True)

    df["passes_all"] = (df["pass_melting_window"] & df["pass_absolute_band"] &
                         df["pass_latent_heat"] & df["pass_cycling"] & df["pass_supercooling"])
    df["window_lo"], df["window_hi"], df["window_relax_applied"] = lo, hi, window_relax
    return df


def main():
    print("=" * 68)
    print("  Phase 5 — Feasibility Filtering — Uttarakhand")
    print("=" * 68)

    for f in (PCM_FILE, PROFILE_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found.")
            return

    pcm_db = pd.read_csv(PCM_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    print(f"\n  PCM candidates : {len(pcm_db)}")
    print(f"  Clusters       : {len(profiles)}")

    if "Tm_target_C" not in profiles.columns or "L_required_kJ_per_kg" not in profiles.columns:
        print("\n  ERROR: cluster_profiles_uttarakhand.csv is missing Tm_target_C / "
              "L_required_kJ_per_kg — these come from 04b_climate_signature.py's "
              "output and should already be columns in the signature matrix that "
              "05_cluster_uttarakhand.py's population-weighted mean carries through. "
              "Check 05's profile_cols list includes them.")
        return

    all_rows = []
    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])
        # Uses the regime-capped Tm_target if 07b_charging_feasibility.py was
        # run (optional); otherwise falls back to the constant plan v3.0 rule.
        tm_target = (prof["Tm_target_C_regime_capped"]
                     if "Tm_target_C_regime_capped" in prof.index else prof["Tm_target_C"])
        l_required = prof["L_required_kJ_per_kg"]

        relax = 0.0
        for step in range(MAX_RELAX_STEPS + 1):
            result = filter_cluster(pcm_db, tm_target, l_required, window_relax=relax)
            n_survivors = int(result["passes_all"].sum())
            if n_survivors >= MIN_SURVIVORS or step == MAX_RELAX_STEPS:
                break
            relax += RELAX_STEP_K

        result.insert(0, "cluster_id", cid)
        result.insert(1, "Tm_target_C", tm_target)
        result.insert(2, "L_required_kJ_per_kg", l_required)
        all_rows.append(result)

        survivors = result[result["passes_all"]]
        flag = f" [RELAXED +{relax:.0f}K]" if relax > 0 else ""
        status = "OK" if 5 <= n_survivors <= 25 else ("LOW" if n_survivors < 5 else "HIGH")
        print(f"  Cluster {cid}: Tm_target={tm_target:.1f}C  L_required={l_required:.0f} kJ/kg  "
              f"-> {n_survivors} survivors{flag}  [{status}]")
        if n_survivors > 0:
            print("     " + ", ".join(survivors["name"].tolist()))

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved: {OUT_FILE}")

    print("\n" + "=" * 68)
    print("  DONE")
    n_low = sum(1 for r in all_rows if r["passes_all"].sum() < 5)
    n_high = sum(1 for r in all_rows if r["passes_all"].sum() > 25)
    if n_low:
        print(f"  [NOTE] {n_low} cluster(s) still under 5 survivors after max relaxation "
              f"({MAX_RELAX_STEPS * RELAX_STEP_K:.0f}K) — your database (25 rows) is "
              f"thin for this; add more candidates in the affected Tm range (06's "
              f"'still outstanding' list) if time allows.")
    if n_high:
        print(f"  [NOTE] {n_high} cluster(s) over 25 survivors — expected to narrow "
              f"in Phase 6 ranking, not a problem here.")
    print("=" * 68)
    print("\nNext: python 08_mcdm_ranking.py")


if __name__ == "__main__":
    main()
