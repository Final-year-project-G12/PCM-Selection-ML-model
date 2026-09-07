"""
07_feasibility_filter.py  -- Assam
=====================================
PHASE 5 -- FEASIBILITY FILTERING (plan v3.0 Section 8, Table 12/13)

Hard-filters pcm_database_assam.csv against each cluster's Tm_target
and L_required before any MCDM ranking. Prevents compensatory MCDM
scores from selecting physically infeasible PCMs.

All 7 constraints from Table 13 applied:
  1. Melting window     : Tm in [Tm_target-6, Tm_target+8] C
     (spec says -6 lower; auto-relaxed +2K per step if < 5 survive)
  2. Absolute band       : Tm in [42, 70] C
  3. Latent heat floor   : L >= 0.7 * L_required
  4. Cycling stability   : >=300 cycles if known; retained+flagged if not
  5. Corrosion veto      : exclude inorganic PCMs in clusters where
                           HSI > global p75  *** LOAD-BEARING for Assam ***
  6. Supercooling veto   : exclude supercooling > 8K (known values only)
  7. Safety              : keyword veto -- highly/extremely flammable, toxic

INPUT  : data/processed/pcm/pcm_database_assam.csv
         data/processed/clustering/cluster_profiles_assam.csv
OUTPUT : data/processed/pcm/feasibility_survivors_assam.csv

HOW TO RUN:
  python 07_feasibility_filter.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

PCM_FILE     = PROCESSED_DIR / "pcm" / "pcm_database_assam.csv"
PROFILE_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_assam.csv"
OUT_FILE     = PROCESSED_DIR / "pcm" / "feasibility_survivors_assam.csv"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0
WINDOW_LOWER_OFFSET  = 6.0   # plan Table 12/13: 6K lower bound
WINDOW_UPPER_OFFSET  = 8.0   # plan Table 12/13: 8K upper bound
LATENT_HEAT_FRACTION = 0.7
CYCLES_FLOOR         = 300
SUPERCOOLING_MAX_K   = 8.0
MIN_SURVIVORS        = 5
MAX_RELAX_STEPS      = 4
RELAX_STEP_K         = 2.0
SAFETY_EXCLUDE_KEYWORDS = ("highly flammable", "extremely flammable", "toxic")


def filter_cluster(pcm_db, tm_target, l_required,
                   cluster_hsi, hsi_p75_global, window_relax=0.0):
    lo = tm_target - WINDOW_LOWER_OFFSET - window_relax
    hi = tm_target + WINDOW_UPPER_OFFSET + window_relax

    df = pcm_db.copy()

    # 1. Melting window
    df["pass_melting_window"] = df["Tm_C"].between(lo, hi)
    df["window_lo"] = lo
    df["window_hi"] = hi
    df["window_relax_applied"] = window_relax

    # 2. Absolute band
    df["pass_absolute_band"] = df["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX)

    # 3. Latent heat floor
    l_floor = LATENT_HEAT_FRACTION * l_required
    df["pass_latent_heat"]  = df["latent_heat_kJ_kg"] >= l_floor
    df["latent_heat_floor"] = l_floor

    # 4. Cycling stability
    df["cycling_known"] = df["cycles_tested"].notna()
    df["pass_cycling"]  = np.where(
        df["cycling_known"], df["cycles_tested"] >= CYCLES_FLOOR, True
    )
    df["cycling_flag"] = np.where(~df["cycling_known"], "not_reported", "")

    # 5. Supercooling veto
    df["supercooling_known"] = df["supercooling_K"].notna()
    df["pass_supercooling"]  = np.where(
        df["supercooling_known"],
        df["supercooling_K"].abs() <= SUPERCOOLING_MAX_K,
        True
    )

    # 6. Corrosion veto  *** load-bearing for Assam's high humidity ***
    cluster_is_high_hsi = cluster_hsi > hsi_p75_global
    df["pass_corrosion"] = ~(
        (df["corrosion_class"] == "check_manually") & cluster_is_high_hsi
    )

    # 7. Safety keyword veto
    flam_lower = df["flammable"].astype(str).str.lower()
    df["pass_safety"] = ~flam_lower.str.contains(
        "|".join(SAFETY_EXCLUDE_KEYWORDS), na=False
    )

    df["passes_all"] = (
        df["pass_melting_window"] &
        df["pass_absolute_band"]  &
        df["pass_latent_heat"]    &
        df["pass_cycling"]        &
        df["pass_supercooling"]   &
        df["pass_corrosion"]      &
        df["pass_safety"]
    )
    return df


def main():
    print("=" * 68)
    print("  Phase 5 -- Feasibility Filtering -- Assam")
    print("=" * 68)

    for f in (PCM_FILE, PROFILE_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found.")
            return

    pcm_db   = pd.read_csv(PCM_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    print(f"\n  PCM candidates : {len(pcm_db)}")
    print(f"  Clusters       : {len(profiles)}")

    # Ensure Tm_target_C, L_required_kJ_per_kg, and HSI are present
    if "Tm_target_C" not in profiles.columns:
        if "Tm_target_mean" in profiles.columns:
            profiles["Tm_target_C"] = profiles["Tm_target_mean"]
        else:
            profiles["Tm_target_C"] = 44.0

    if "L_required_kJ_per_kg" not in profiles.columns:
        if "L_required_kWh_mean" in profiles.columns:
            profiles["L_required_kJ_per_kg"] = profiles["L_required_kWh_mean"] * 3600.0 / 50.0
        else:
            t_mains = np.maximum(5.0, profiles.get("Ta_mean_mean", 25.0) - 6.0)
            q_req = (100.0 * 4186.0 * (50.0 - t_mains)) / 3_600_000.0
            profiles["L_required_kJ_per_kg"] = (q_req * 3600.0) / 50.0

    if "HSI" not in profiles.columns:
        profiles["HSI"] = profiles.get("HSI_mean", profiles.get("RH_mean_mean", 75.0))

    hsi_p75_global = profiles["HSI"].quantile(0.75) if "HSI" in profiles.columns else np.inf
    print(f"  Global HSI p75 (corrosion threshold): {hsi_p75_global:.2f}")
    print(f"  NOTE: Corrosion veto IS load-bearing for Assam's high-humidity clusters")

    all_rows = []
    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])

        tm_target = (
            prof["Tm_target_C_regime_capped"]
            if "Tm_target_C_regime_capped" in prof.index
            else prof["Tm_target_C"]
        )
        l_required  = prof["L_required_kJ_per_kg"]
        cluster_hsi = prof.get("HSI", -np.inf)

        relax = 0.0
        for step in range(MAX_RELAX_STEPS + 1):
            result = filter_cluster(
                pcm_db, tm_target, l_required,
                cluster_hsi, hsi_p75_global, window_relax=relax
            )
            n_survivors = int(result["passes_all"].sum())
            if n_survivors >= MIN_SURVIVORS or step == MAX_RELAX_STEPS:
                break
            relax += RELAX_STEP_K

        result.insert(0, "cluster_id",          cid)
        result.insert(1, "Tm_target_C",          tm_target)
        result.insert(2, "L_required_kJ_per_kg", l_required)
        all_rows.append(result)

        survivors = result[result["passes_all"]]
        flag   = f" [RELAXED +{relax:.0f}K]" if relax > 0 else ""
        status = "OK" if 5 <= n_survivors <= 25 else ("LOW" if n_survivors < 5 else "HIGH")
        print(
            f"\n  Cluster {cid}: Tm_target={tm_target:.1f}C  "
            f"L_required={l_required:.0f} kJ/kg  -> {n_survivors} survivors{flag}  [{status}]"
        )
        print(
            f"  HSI={cluster_hsi:.2f}  "
            f"(p75={hsi_p75_global:.2f}  corrosion_veto_active={cluster_hsi > hsi_p75_global})"
        )
        if n_survivors > 0:
            print("     " + ", ".join(survivors["name"].tolist()))

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved: {OUT_FILE}")

    print("\n" + "=" * 68)
    print("  DONE — Phase 5 Feasibility Filtering")

    # Survivor count table (key result for methodology section)
    print("\n  Survivor count by cluster (report in paper Table 13):")
    for r in all_rows:
        cid = int(r["cluster_id"].iloc[0])
        n = int(r["passes_all"].sum())
        relax = float(r["window_relax_applied"].iloc[0])
        note = f" (relaxed +{relax:.0f}K)" if relax > 0 else ""
        print(f"    Cluster {cid}: {n} candidates{note}")

    n_low  = sum(1 for r in all_rows if r["passes_all"].sum() < 5)
    n_high = sum(1 for r in all_rows if r["passes_all"].sum() > 25)
    if n_low:
        print(f"\n  [NOTE] {n_low} cluster(s) under 5 survivors after max relaxation "
              f"({MAX_RELAX_STEPS * RELAX_STEP_K:.0f}K) -- database is thin. "
              "Add more PCMs in the affected Tm range.")
    if n_high:
        print(f"\n  [NOTE] {n_high} cluster(s) over 25 survivors -- will narrow in Phase 6 ranking.")

    print("=" * 68)
    print("\nNext: python 08_mcdm_ranking.py")


if __name__ == "__main__":
    main()
