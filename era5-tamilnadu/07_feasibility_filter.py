"""
07_feasibility_filter.py
=============================================================================
PHASE 5 — PCM FEASIBILITY FILTERING (Objective1_PCM_Climate_Framework_Plan_v3,
Table 12), Tamil Nadu.

UNIFIED 2026-09-08 with era5-rajasthan/07_feasibility_filter.py. The
constraint set, constraint ORDER, missing-value semantics, kappa-calibration
procedure and cross-phase provenance stamping are now identical to
Rajasthan's. Two things are deliberately different, and only these two:

  1. STATE NAME + cluster-profile path (data/processed/clustering/
     cluster_profiles_tamilnadu.csv).
  2. OUTPUT FILENAMES. Tamil Nadu's existing convention —
     feasibility_survivors_by_cluster.csv and
     feasibility_survivors_by_cluster_kappa_calibrated.csv, under
     data/processed/pcm/ — is the CANONICAL naming for BOTH states.
     Rajasthan's feasibility_survivors_rajasthan{,_kappa_calibrated}.csv
     were renamed to match these, not the other way round.

DATA SOURCE — single canonical imputation, no second loop
--------------------------------------------------------------
This script reads the candidate pool from 06_build_pcm_database.py's
output, pcm_database_tamilnadu.csv. 06 is a THIN database-building step:
it consumes the ONE canonical MICE+RF+PMM preprocessing output
(PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv, produced by
PCM_data/PCM_data/01_preprocess.py) and only adds derived columns
(rho_H_MJ_m3, Cp_avg, cycles_confidence, ...). It does NOT re-impute
anything. The 62-row pool (55 manufacturer + 7 Singh2025 literature) is
therefore row-for-row identical to Rajasthan's on every shared column —
Rajasthan builds the same pool inline from the same canonical CSV +
literature_rows(); this pipeline routes it through 06 instead. Both are
sanctioned by the plan doc ("a thin database-building step, or an inline
read"); the imputation pipeline itself is shared and singular.

CONSTRAINTS (exact order, matching Rajasthan):
  1. Melting window   : Tm in [Tm_target-5, Tm_target+8] C, auto-relaxed by
                         2K/round up to 4 rounds if a cluster has <5 survivors.
  2. Absolute band     : Tm in [42, 70] C (v3.0-corrected band, Table 12).
  3. Latent heat floor : L >= kappa * L_required, kappa=0.7 FIXED for the
                         primary run. pass / fail / flag_unreported.
  4. Cycling           : >=300 cycles; FLAG (flag_unreported), never
                         excludes, if the cycling value is unreported.
  5. Supercooling      : <=8K; FLAG (flag_unknown), never excludes, if the
                         supercooling value is unknown.
  6. Charging feasibility: Tm <= that cluster's Tm_target_capped_C (from
     Phase 3's kt_worst_month-derived ceiling — referenced DIRECTLY from
     cluster_profiles_tamilnadu.csv, NOT re-derived). This is the ONE
     charging-feasibility path. The former heuristic
     07b_charging_feasibility.py (REFERENCE_GOOD_DAY_TEMP /
     MIN_ACHIEVABLE_TEMP proxy, writing Tm_target_C_regime_capped) has
     been RETIRED — this script no longer reads Tm_target_C_regime_capped.
  7. Corrosion veto : exclude bare (non-encapsulated) salt hydrates where
     the cluster's HSI_sunrise exceeds the 75th percentile across this
     state's clusters, unless the row specifies encapsulation. IMPLEMENTED
     BUT CURRENTLY STRUCTURALLY INERT — the shared 62-row PCM database
     contains ZERO salt-hydrate candidates (every manufacturer and
     literature row is Organic-typed), so is_salt_hydrate() matches
     nothing and every candidate gets c7 = "not_applicable". It becomes
     load-bearing once real salt-hydrate rows / the Assam pipeline exist.
  8. Safety exclusion : flag toxicity / flammability. The `flammable`
     field is an unqualified Yes/No, not a severity grade, and there is
     no toxicity column, so this constraint FLAGS only — it never
     actually excludes on the current data. Matches Rajasthan's semantics.

PRIMARY vs COMPANION run — report BOTH:
  * feasibility_survivors_by_cluster.csv — PRIMARY, fixed kappa=0.7.
  * feasibility_survivors_by_cluster_kappa_calibrated.csv — COMPANION.
    kappa stepped down per cluster (0.7 -> 0.0 in 0.1 increments) until
    the cluster retains 8-20 survivors, evaluated at each cluster's
    melting window AT THE PRIMARY RUN'S FINAL RELAXATION ROUND. Carries a
    `breakeven_kappa` column per candidate.
  The eventual Phase 5 report must state BOTH the nominal kappa=0.7
  survivor count per cluster AND the final calibrated-kappa survivor count
  per cluster.

INPUTS:
  data/processed/pcm/pcm_database_tamilnadu.csv               (06's output)
  data/processed/clustering/cluster_profiles_tamilnadu.csv    (05's output;
      must carry Tm_target_C, Tm_target_capped_C, L_required_kJ_per_kg,
      HSI_sunrise)

OUTPUTS:
  data/processed/pcm/feasibility_survivors_by_cluster.csv
  data/processed/pcm/feasibility_survivors_by_cluster_kappa_calibrated.csv
      one row per (cluster_id, pcm) audited (non-survivors included), one
      column per constraint, plus `survives_all` (and `passes_all` as a
      backward-compatible alias). Both files carry the
      upstream_cluster_profile_fingerprint stamp so Phase 6/7/8 hard-fail
      on a Phase 4/5 clustering mismatch.

HOW TO RUN:
  python 07_feasibility_filter.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
# Cross-phase provenance stamping (mirrors Rajasthan). Phase 5 is the
# PRODUCER end of the chain: it fingerprints the exact cluster_profiles
# file it read and stamps that fingerprint into its own output, so Phase
# 6/7/8 can hard-fail if run against a different clustering run.
from provenance_lib import file_fingerprint, fingerprint_id

STATE_NAME = "tamilnadu"

PCM_FILE = PROCESSED_DIR / "pcm" / f"pcm_database_{STATE_NAME}.csv"
PROFILE_FILE = PROCESSED_DIR / "clustering" / f"cluster_profiles_{STATE_NAME}.csv"

# Tamil Nadu naming convention — canonical for both states. Rajasthan's
# feasibility_survivors_rajasthan{,_kappa_calibrated}.csv were renamed to
# these during the 2026-09-08 unification.
OUT_DIR = PROCESSED_DIR / "pcm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "feasibility_survivors_by_cluster.csv"
OUT_FILE_KAPPA_CALIBRATED = OUT_DIR / "feasibility_survivors_by_cluster_kappa_calibrated.csv"

# --- Constraints 1/2 (melting window + absolute band) ----------------------
MELT_WINDOW_LOW_OFFSET = -5.0     # Tm_target + this
MELT_WINDOW_HIGH_OFFSET = 8.0     # Tm_target + this
RELAX_STEP_K = 2.0
MAX_RELAX_ROUNDS = 4
MIN_SURVIVORS_TARGET = 5
ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0   # v3.0-corrected band, Table 12

# --- Constraint 3 (latent heat floor) --------------------------------------
LATENT_HEAT_KAPPA = 0.7          # FIXED for the primary run

# --- Constraints 4/5 (flag, not exclude) ---------------------------------
CYCLING_MIN = 300
SUPERCOOLING_MAX_K = 8.0

# --- Constraint 7 (corrosion veto) --------------------------------------
CORROSION_HSI_PERCENTILE = 0.75

# --- Kappa-calibration companion pass ----------------------------------
KAPPA_STEPS = [round(0.7 - 0.1 * i, 2) for i in range(8)]   # 0.7, 0.6, ..., 0.0
CALIBRATED_SURVIVOR_LO, CALIBRATED_SURVIVOR_HI = 8, 20


def is_salt_hydrate(row):
    """Structural detection, general per the brief (not tuned to this
    run's data — nothing in the current 62-row database matches: every
    row is Organic-typed and no family string contains "hydrate"). Kept
    for when real salt-hydrate rows / the Assam pipeline need it. Substring
    check on pcm_type / corrosion_class rather than an exact match, because
    pcm_type is a descriptive string."""
    fam = str(row.get("family", "")).lower()
    ptype = str(row.get("pcm_type", "")).lower()
    cclass = str(row.get("corrosion_class", "")).lower()
    return ("inorganic" in ptype) or ("hydrate" in fam) or (cclass == "check_manually")


# ═══════════════════════════════════════════════════════════
# PER-CANDIDATE CONSTRAINT EVALUATION  (constraints 1-8)
# ═══════════════════════════════════════════════════════════

def evaluate(pcm, tm_target, tm_capped, l_required, hsi_p75_cutoff, window_lo, window_hi,
             kappa=LATENT_HEAT_KAPPA):
    """pcm is a dict (row of pcm_database_tamilnadu.csv). Every original
    column is carried through untouched; the constraint verdict columns
    (c1..c8, survives_all, breakeven_kappa, rescuable_by_kappa) are added."""
    # carry through all rich columns (rho_H_MJ_m3, TC_W_mK, cycles_confidence, ...)
    # but drop the private _cluster_hsi the caller injects.
    r = {k: v for k, v in pcm.items() if not str(k).startswith("_")}

    tm = pcm["Tm_C"]
    latent = pcm.get("latent_heat_kJ_kg", np.nan)
    cycles = pcm.get("cycles_tested", np.nan)
    supercool = pcm.get("supercooling_K", np.nan)
    encapsulated = pcm.get("encapsulated", np.nan)   # no such column in the DB -> NaN
    flammable_raw = pcm.get("flammable", pcm.get("flammable_raw", np.nan))

    # 1. Melting window (cluster- and relaxation-round-dependent)
    r["c1_melting_window"] = "pass" if window_lo <= tm <= window_hi else "fail"

    # 2. Absolute band
    r["c2_absolute_band"] = "pass" if ABSOLUTE_TM_MIN <= tm <= ABSOLUTE_TM_MAX else "fail"

    # 3. Latent heat floor. breakeven_kappa = L_actual / L_required: the
    # exact kappa at which THIS candidate would clear the floor on its own.
    r["breakeven_kappa"] = (latent / l_required
                             if pd.notna(latent) and l_required else np.nan)
    threshold = kappa * l_required
    if pd.isna(latent):
        r["c3_latent_heat"] = "flag_unreported"
    else:
        r["c3_latent_heat"] = "pass" if latent >= threshold else "fail"

    # 4. Cycling — flag_unreported, never excludes, if unreported
    if pd.isna(cycles):
        r["c4_cycling"] = "flag_unreported"
    else:
        r["c4_cycling"] = "pass" if cycles >= CYCLING_MIN else "fail"

    # 5. Supercooling — flag_unknown, never excludes, if unknown
    if pd.isna(supercool):
        r["c5_supercooling"] = "flag_unknown"
    else:
        r["c5_supercooling"] = "pass" if supercool <= SUPERCOOLING_MAX_K else "fail"

    # 6. Charging feasibility — Tm must not exceed the cluster's poor-day
    # achievable ceiling (Tm_target_capped_C, referenced directly).
    r["c6_charging_feasibility"] = "pass" if tm <= tm_capped else "fail"

    # 7. Corrosion veto — inert on the current database (0 salt hydrates).
    if not is_salt_hydrate(pcm):
        r["c7_corrosion_veto"] = "not_applicable"
    elif hsi_p75_cutoff == hsi_p75_cutoff and pcm.get("_cluster_hsi", np.nan) <= hsi_p75_cutoff:
        r["c7_corrosion_veto"] = "not_applicable_low_hsi_cluster"
    elif encapsulated is True:
        r["c7_corrosion_veto"] = "pass_encapsulated"
    elif pd.isna(encapsulated):
        # Conservative default: cannot verify encapsulation -> excluded,
        # not silently passed.
        r["c7_corrosion_veto"] = "excluded_unverified_encapsulation"
    else:
        r["c7_corrosion_veto"] = "excluded_bare_high_hsi"

    # 8. Safety exclusion — flag only, never hard-excludes on current data.
    if flammable_raw in ("not_reported", None) or pd.isna(flammable_raw):
        r["c8_safety"] = "flag_unverified_flammability_and_toxicity"
    else:
        r["c8_safety"] = f"flag_unverified_toxicity_flammable_raw={flammable_raw}"

    blocking = [r["c1_melting_window"], r["c2_absolute_band"], r["c3_latent_heat"],
                r["c6_charging_feasibility"]]
    r["survives_c1_c2_c3_c6"] = (all(v == "pass" or v.startswith("flag_") for v in blocking)
                                  and all(v != "fail" for v in blocking))
    r["survives_c7_corrosion"] = not r["c7_corrosion_veto"].startswith("excluded_")
    r["survives_c8_safety"] = not r["c8_safety"].startswith("excluded_")
    r["survives_all"] = (r["survives_c1_c2_c3_c6"] and r["survives_c7_corrosion"]
                          and r["survives_c8_safety"])
    # Backward-compatible alias for downstream scripts that still read
    # `passes_all` (08_mcdm_ranking.py, 09_recommendation_cards.py).
    r["passes_all"] = r["survives_all"]

    non_c3_blocking = [r["c1_melting_window"], r["c2_absolute_band"], r["c6_charging_feasibility"]]
    r["rescuable_by_kappa"] = (all(v != "fail" for v in non_c3_blocking)
                                and r["survives_c7_corrosion"] and r["survives_c8_safety"])
    return r


# ═══════════════════════════════════════════════════════════
# KAPPA-CALIBRATION COMPANION PASS
# ═══════════════════════════════════════════════════════════

def calibrate_kappa_for_cluster(pcm_db, cid, tm_target, tm_capped, l_required,
                                 hsi_p75_cutoff, cluster_hsi, window_lo, window_hi):
    """Steps kappa down from 0.7 in 0.1 increments until this cluster
    retains CALIBRATED_SURVIVOR_LO-HI (8-20) candidates.

    window_lo/window_hi are the cluster's melting window AT THE PRIMARY
    RUN'S FINAL RELAXATION ROUND (passed in by main(), not recomputed
    here): exhaust melting-window relaxation first, exactly as the primary
    run does; THEN calibrate kappa on top of that already-widened window.
    Keeps this a single interpretable 1D search and avoids the degenerate
    result the base-window version produced.

    Returns (calibrated_kappa, rows_at_calibrated_kappa, status,
    already_in_at_07).
    """
    base_rows = []
    for pcm in pcm_db.to_dict("records"):
        pcm["_cluster_hsi"] = cluster_hsi
        base_rows.append(evaluate(pcm, tm_target, tm_capped, l_required,
                                   hsi_p75_cutoff, window_lo, window_hi, kappa=0.7))

    # A candidate passes c3 at threshold k when breakeven_kappa >= k.
    rescuable = [r for r in base_rows if r["rescuable_by_kappa"]]
    already_in_at_07 = sum(1 for r in rescuable
                            if r["breakeven_kappa"] == r["breakeven_kappa"]
                            and r["breakeven_kappa"] >= 0.7)

    chosen_kappa, status = None, None
    for k in KAPPA_STEPS:
        n = sum(1 for r in rescuable if r["breakeven_kappa"] == r["breakeven_kappa"]
                and r["breakeven_kappa"] >= k)
        if CALIBRATED_SURVIVOR_LO <= n <= CALIBRATED_SURVIVOR_HI:
            chosen_kappa, status = k, "in_band"
            break
    if chosen_kappa is None:
        for k in KAPPA_STEPS:
            n = sum(1 for r in rescuable if r["breakeven_kappa"] == r["breakeven_kappa"]
                    and r["breakeven_kappa"] >= k)
            if n >= CALIBRATED_SURVIVOR_LO:
                chosen_kappa, status = k, "exceeds_20_no_clean_landing"
                break
    if chosen_kappa is None:
        chosen_kappa, status = 0.0, "insufficient_even_at_kappa_0"

    final_rows = []
    for pcm in pcm_db.to_dict("records"):
        pcm["_cluster_hsi"] = cluster_hsi
        final_rows.append(evaluate(pcm, tm_target, tm_capped, l_required,
                                    hsi_p75_cutoff, window_lo, window_hi, kappa=chosen_kappa))

    return chosen_kappa, final_rows, status, already_in_at_07


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print(f"  PHASE 5 — FEASIBILITY FILTERING — {STATE_NAME.title()}")
    print("=" * 68)

    for f in (PCM_FILE, PROFILE_FILE):
        if not f.exists():
            raise SystemExit(f"ERROR: {f} not found.")

    pcm_db = pd.read_csv(PCM_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    profile_fp_id = fingerprint_id(file_fingerprint(PROFILE_FILE))
    print(f"  {PROFILE_FILE.name} fingerprint: {profile_fp_id}")
    print(f"\n  PCM candidates : {len(pcm_db)}")
    print(f"  Clusters       : {len(profiles)}")

    for col in ("Tm_target_C", "Tm_target_capped_C", "L_required_kJ_per_kg"):
        if col not in profiles.columns:
            raise SystemExit(
                f"ERROR: cluster_profiles_{STATE_NAME}.csv is missing '{col}'. "
                f"Tm_target_capped_C is Phase 3's kt_worst_month-derived charging-feasibility "
                f"ceiling (Constraint 6); Tm_target_C / L_required_kJ_per_kg come from the "
                f"climate signature. Re-run the unified Phase 3 + 05_cluster_{STATE_NAME}.py.")

    HSI_COL = "HSI_sunrise"
    hsi_p75_cutoff = profiles[HSI_COL].quantile(CORROSION_HSI_PERCENTILE) \
        if HSI_COL in profiles.columns else np.nan
    if HSI_COL not in profiles.columns:
        print(f"\n  [WARN] cluster_profiles_{STATE_NAME}.csv has no {HSI_COL} column — the "
              f"corrosion veto will be a NO-OP. Re-run 04b_climate_signature.py + "
              f"05_cluster_{STATE_NAME}.py to restore it.")
    else:
        print(f"  Corrosion-veto HSI cutoff (75th percentile across {len(profiles)} "
              f"clusters): {hsi_p75_cutoff:.3f}")

    n_salt_hydrate = int(pcm_db.apply(is_salt_hydrate, axis=1).sum())
    print(f"  Salt-hydrate-typed candidates in the database: {n_salt_hydrate} "
          f"(corrosion veto is expected to be inert regardless — see module docstring)")

    all_rows, summary_rows = [], []
    cluster_final_windows = {}

    for prof in profiles.itertuples():
        cid = int(prof.cluster_id)
        tm_target = float(prof.Tm_target_C)
        tm_capped = float(prof.Tm_target_capped_C)
        l_required = float(prof.L_required_kJ_per_kg)
        cluster_hsi = float(getattr(prof, HSI_COL)) if hasattr(prof, HSI_COL) else np.nan

        print(f"\n  --- Cluster {cid} "
              f"(Tm_target={tm_target:.1f}C, Tm_target_capped={tm_capped:.1f}C, "
              f"L_required={l_required:.0f} kJ/kg [combined S+L basis, Phase 3 OPTION A], "
              f"HSI={cluster_hsi:.2f}) ---")

        final_rows, final_relax = None, None
        window_lo = window_hi = None
        for relax_round in range(MAX_RELAX_ROUNDS + 1):
            widen = RELAX_STEP_K * relax_round
            window_lo = tm_target + MELT_WINDOW_LOW_OFFSET - widen
            window_hi = tm_target + MELT_WINDOW_HIGH_OFFSET + widen

            round_rows = []
            for pcm in pcm_db.to_dict("records"):
                pcm["_cluster_hsi"] = cluster_hsi
                round_rows.append(evaluate(pcm, tm_target, tm_capped, l_required,
                                            hsi_p75_cutoff, window_lo, window_hi))
            n_survive = sum(r["survives_all"] for r in round_rows)
            print(f"    Relax round {relax_round} (widen +/-{widen:.0f}K, window "
                  f"[{window_lo:.1f}, {window_hi:.1f}]C): {n_survive} survivors")

            final_rows, final_relax = round_rows, relax_round
            if n_survive >= MIN_SURVIVORS_TARGET:
                break

        cluster_final_windows[cid] = (window_lo, window_hi)
        n_final_survive = sum(r["survives_all"] for r in final_rows)
        if n_final_survive < MIN_SURVIVORS_TARGET:
            print(f"    [WARNING] Cluster {cid} still has only {n_final_survive} survivor(s) "
                  f"after {final_relax} relaxation round(s) (max {MAX_RELAX_ROUNDS}). "
                  f"Constraint 3 (latent heat, kappa=0.7 fixed) is the likeliest remaining "
                  f"cause — check the c3 exclusion count.")

        exclude_counts = {
            "c1_melting_window": sum(r["c1_melting_window"] == "fail" for r in final_rows),
            "c2_absolute_band": sum(r["c2_absolute_band"] == "fail" for r in final_rows),
            "c3_latent_heat": sum(r["c3_latent_heat"] == "fail" for r in final_rows),
            "c4_cycling_fail": sum(r["c4_cycling"] == "fail" for r in final_rows),
            "c4_cycling_flagged_unreported": sum(r["c4_cycling"] == "flag_unreported" for r in final_rows),
            "c5_supercooling_fail": sum(r["c5_supercooling"] == "fail" for r in final_rows),
            "c5_supercooling_flagged_unknown": sum(r["c5_supercooling"] == "flag_unknown" for r in final_rows),
            "c6_charging_feasibility": sum(r["c6_charging_feasibility"] == "fail" for r in final_rows),
            "c7_corrosion_veto": sum(r["c7_corrosion_veto"].startswith("excluded_") for r in final_rows),
            "c8_safety": sum(r["c8_safety"].startswith("excluded_") for r in final_rows),
        }
        print(f"    Entered: {len(final_rows)}  |  Survived (all 8 constraints): {n_final_survive}")
        print(f"    Excluded by: " + ", ".join(f"{k}={v}" for k, v in exclude_counts.items()))

        for r in final_rows:
            r["cluster_id"] = cid
            r["Tm_target_C"] = tm_target
            r["Tm_target_capped_C"] = tm_capped
            r["L_required_kJ_per_kg"] = l_required
            r["relax_round_used"] = final_relax
            r["melting_window_widen_K"] = RELAX_STEP_K * final_relax
        all_rows.extend(final_rows)

        summary_rows.append({"cluster_id": cid, "n_entered": len(final_rows),
                              "n_survived": n_final_survive, "relax_round_used": final_relax,
                              "melting_window_widen_K": RELAX_STEP_K * final_relax,
                              **{f"excluded_{k}": v for k, v in exclude_counts.items()}})

    out_df = pd.DataFrame(all_rows)
    out_df["upstream_cluster_profile_fingerprint"] = profile_fp_id
    # Move the constraint-verdict columns to the front for readability;
    # everything else (rich PCM props) trails after.
    lead = ["cluster_id", "pcm_id", "name", "family", "pcm_type", "Tm_C", "Tm_target_C",
            "Tm_target_capped_C", "L_required_kJ_per_kg", "latent_heat_kJ_kg", "cycles_tested",
            "supercooling_K", "relax_round_used", "melting_window_widen_K",
            "c1_melting_window", "c2_absolute_band", "c3_latent_heat", "c4_cycling",
            "c5_supercooling", "c6_charging_feasibility", "c7_corrosion_veto", "c8_safety",
            "survives_all", "passes_all", "upstream_cluster_profile_fingerprint"]
    ordered = [c for c in lead if c in out_df.columns] + \
              [c for c in out_df.columns if c not in lead]
    out_df = out_df[ordered]
    out_df.to_csv(OUT_FILE, index=False)

    print("\n" + "=" * 68)
    print("  SUMMARY (primary run, fixed kappa=0.7)")
    print("=" * 68)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n  Saved: {OUT_FILE}  ({len(out_df)} (cluster, pcm) rows, "
          f"{int(out_df['survives_all'].sum())} surviving all 8 constraints)")
    print("=" * 68)

    # ═══════════════════════════════════════════════════════════
    # KAPPA-CALIBRATION COMPANION PASS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print(f"  KAPPA-CALIBRATION COMPANION PASS — {STATE_NAME.title()}")
    print(f"  Steps: {KAPPA_STEPS}  |  Target band: "
          f"{CALIBRATED_SURVIVOR_LO}-{CALIBRATED_SURVIVOR_HI} survivors/cluster")
    print(f"  Evaluated at each cluster's melting window from the primary run's FINAL "
          f"relaxation round.")
    print("=" * 68)

    calib_all_rows, calib_summary_rows = [], []
    for prof in profiles.itertuples():
        cid = int(prof.cluster_id)
        cluster_hsi = float(getattr(prof, HSI_COL)) if hasattr(prof, HSI_COL) else np.nan
        window_lo, window_hi = cluster_final_windows[cid]

        chosen_kappa, final_rows, status, n_already_at_07 = calibrate_kappa_for_cluster(
            pcm_db, cid, float(prof.Tm_target_C), float(prof.Tm_target_capped_C),
            float(prof.L_required_kJ_per_kg), hsi_p75_cutoff, cluster_hsi, window_lo, window_hi)

        n_survive = sum(r["survives_all"] for r in final_rows)
        print(f"\n  --- Cluster {cid}: calibrated kappa = {chosen_kappa}  "
              f"({n_survive} survivors, status={status}) ---")
        if status != "in_band":
            print(f"    [NOTE] Did not land cleanly in the "
                  f"{CALIBRATED_SURVIVOR_LO}-{CALIBRATED_SURVIVOR_HI} band at any 0.1 step "
                  f"— see status above.")

        rescuable_sorted = sorted(
            (r for r in final_rows if r["rescuable_by_kappa"]
             and r["breakeven_kappa"] == r["breakeven_kappa"]),
            key=lambda r: -r["breakeven_kappa"])
        print(f"    Admission order as kappa falls from 0.7 (first 5 of "
              f"{len(rescuable_sorted)} rescuable-by-kappa candidates):")
        for r in rescuable_sorted[:5]:
            nm = r.get("pcm_id", r.get("name", "?"))
            print(f"      breakeven_kappa={r['breakeven_kappa']:.3f}  {nm}  "
                  f"(Tm={r['Tm_C']:.1f}C, L={r['latent_heat_kJ_kg']:.0f} kJ/kg)")

        for r in final_rows:
            r["cluster_id"] = cid
            r["Tm_target_C"] = float(prof.Tm_target_C)
            r["Tm_target_capped_C"] = float(prof.Tm_target_capped_C)
            r["L_required_kJ_per_kg"] = float(prof.L_required_kJ_per_kg)
            r["calibrated_kappa"] = chosen_kappa
            r["calibration_status"] = status
        calib_all_rows.extend(final_rows)
        calib_summary_rows.append({"cluster_id": cid, "calibrated_kappa": chosen_kappa,
                                    "n_survived": n_survive, "status": status,
                                    "n_rescuable_by_kappa_total": len(rescuable_sorted)})

    calib_out_df = pd.DataFrame(calib_all_rows)
    calib_out_df["upstream_cluster_profile_fingerprint"] = profile_fp_id
    calib_lead = ["cluster_id", "pcm_id", "name", "family", "pcm_type", "Tm_C", "Tm_target_C",
                  "Tm_target_capped_C", "L_required_kJ_per_kg", "latent_heat_kJ_kg",
                  "cycles_tested", "supercooling_K", "breakeven_kappa", "rescuable_by_kappa",
                  "calibrated_kappa", "calibration_status",
                  "c1_melting_window", "c2_absolute_band", "c3_latent_heat", "c4_cycling",
                  "c5_supercooling", "c6_charging_feasibility", "c7_corrosion_veto", "c8_safety",
                  "survives_all", "passes_all", "upstream_cluster_profile_fingerprint"]
    calib_ordered = [c for c in calib_lead if c in calib_out_df.columns] + \
                    [c for c in calib_out_df.columns if c not in calib_lead]
    calib_out_df = calib_out_df[calib_ordered]
    calib_out_df.to_csv(OUT_FILE_KAPPA_CALIBRATED, index=False)

    print("\n" + "=" * 68)
    print("  KAPPA-CALIBRATION SUMMARY")
    print("=" * 68)
    print(pd.DataFrame(calib_summary_rows).to_string(index=False))
    print(f"\n  Saved: {OUT_FILE_KAPPA_CALIBRATED}  ({len(calib_out_df)} (cluster, pcm) rows, "
          f"{int(calib_out_df['survives_all'].sum())} surviving at each cluster's calibrated kappa)")
    print(f"\n  Report BOTH {OUT_FILE.name} (fixed kappa=0.7 baseline) AND "
          f"{OUT_FILE_KAPPA_CALIBRATED.name} (calibrated) together.")
    print("=" * 68)
    print("\nNext: python 08_mcdm_ranking.py")


if __name__ == "__main__":
    main()
