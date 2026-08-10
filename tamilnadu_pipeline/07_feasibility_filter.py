"""
07_feasibility_filter.py
===========================
PHASE 5 — FEASIBILITY FILTERING (Objective 1 plan v3.0, Section 8, Table 12)

Hard-filters the PCM database against EACH cluster's Tm_target/L_required
before any MCDM ranking runs. This matters because MCDM is compensatory —
a PCM with an unreachable melting point but great latent heat can still
score well in TOPSIS and be physically useless. Filtering first prevents
that (plan v3.0 Section 8, opening paragraph).

Filters applied (all eight from Table 12 now implemented):
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
  6. Corrosion veto        : exclude "check_manually"-class PCMs in any
                            cluster whose HSI is above the 75th percentile
                            ACROSS ALL CLUSTERS. Currently a near-no-op —
                            your 25-row database is almost entirely organic
                            (nothing flagged "check_manually" except one
                            inorganic hydrate) — becomes load-bearing once
                            you add real salt hydrates or extend to Assam.
  7. Safety exclusion      : keyword veto against the flammability field
                            ("highly/extremely flammable", "toxic"). Also
                            currently a no-op given your data — paraffins/
                            fatty acids are "combustible", not "highly
                            flammable" in standard hazard classification.
  NOT applied (still needs data this project doesn't have):
    - Charging feasibility at the cluster's true 5th-percentile insolation
      day (07b_charging_feasibility.py gives a HEURISTIC proxy for this via
      kt_mean/kt_std, not the literal Table-12 mechanism, which needs a
      real daily-GHI percentile per cluster plus a collector efficiency
      curve — that level of rigor is what Phase 7's grey-box model now
      provides instead, per-PCM, per-cluster, via simulated performance).

If a cluster keeps fewer than 5 candidates, the melting window is
automatically relaxed by 2K and retried (per Section 8's stated rule). If
a cluster keeps more than 25, that's reported but not narrowed further —
Phase 6's ranking is what should separate them.

INPUT  : data/processed/pcm/pcm_database_tamilnadu.csv        (06's output)
         data/processed/clustering/cluster_profiles_tamilnadu.csv (05's output)
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

PCM_FILE = PROCESSED_DIR / "pcm" / "pcm_database_tamilnadu.csv"
PROFILE_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_tamilnadu.csv"
OUT_FILE = PROCESSED_DIR / "pcm" / "feasibility_survivors_by_cluster.csv"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7
CYCLES_FLOOR = 300
SUPERCOOLING_MAX_K = 8.0
MIN_SURVIVORS, MAX_RELAX_STEPS, RELAX_STEP_K = 5, 4, 2.0
SAFETY_EXCLUDE_KEYWORDS = ("highly flammable", "extremely flammable", "toxic")


def filter_cluster(pcm_db, tm_target, l_required, cluster_hsi, hsi_p75_global, window_relax=0.0):
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

    # Corrosion veto (Table 12): only bites for PCMs flagged "check_manually"
    # (currently just the one inorganic salt-hydrate-class candidate) AND
    # only in a cluster whose humidity-stress index sits above the 75th
    # percentile ACROSS ALL CLUSTERS (Table 12's literal wording). With a
    # database of mostly organic PCMs this will mostly be a no-op right
    # now — it becomes load-bearing once you add real salt hydrates or
    # extend to a more humid state (Assam).
    cluster_is_high_hsi = cluster_hsi > hsi_p75_global
    df["pass_corrosion"] = ~((df["corrosion_class"] == "check_manually") & cluster_is_high_hsi)

    # Safety exclusion (Table 12): keyword check against the flammability
    # field. Your current 25-row database has no candidate flagged this way
    # (paraffins/fatty acids are "combustible", not "highly flammable" in
    # standard hazard classification) — this is a real filter, just
    # currently a no-op given your data, not a fake one.
    flam_text = df["flammable"].astype(str).str.lower()
    df["pass_safety"] = ~flam_text.str.contains("|".join(SAFETY_EXCLUDE_KEYWORDS), na=False)

    df["passes_all"] = (df["pass_melting_window"] & df["pass_absolute_band"] &
                         df["pass_latent_heat"] & df["pass_cycling"] & df["pass_supercooling"] &
                         df["pass_corrosion"] & df["pass_safety"])
    df["window_lo"], df["window_hi"], df["window_relax_applied"] = lo, hi, window_relax
    return df


def main():
    print("=" * 68)
    print("  Phase 5 — Feasibility Filtering — Tamil Nadu")
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
        print("\n  ERROR: cluster_profiles_tamilnadu.csv is missing Tm_target_C / "
              "L_required_kJ_per_kg — these come from 04b_climate_signature.py's "
              "output and should already be columns in the signature matrix that "
              "05_cluster_tamilnadu.py's population-weighted mean carries through. "
              "Check 05's profile_cols list includes them.")
        return

    all_rows = []
    hsi_p75_global = profiles["HSI"].quantile(0.75) if "HSI" in profiles.columns else np.inf
    if "HSI" not in profiles.columns:
        print("\n  [NOTE] cluster_profiles_tamilnadu.csv has no HSI column — "
              "corrosion veto will be a no-op (never triggers) rather than error out.")

    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])
        # Uses the regime-capped Tm_target if 07b_charging_feasibility.py was
        # run (optional); otherwise falls back to the constant plan v3.0 rule.
        tm_target = (prof["Tm_target_C_regime_capped"]
                     if "Tm_target_C_regime_capped" in prof.index else prof["Tm_target_C"])
        l_required = prof["L_required_kJ_per_kg"]
        cluster_hsi = prof.get("HSI", -np.inf)

        relax = 0.0
        for step in range(MAX_RELAX_STEPS + 1):
            result = filter_cluster(pcm_db, tm_target, l_required, cluster_hsi,
                                     hsi_p75_global, window_relax=relax)
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
