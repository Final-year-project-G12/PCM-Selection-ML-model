
"""
07_feasibility_filter_rajasthan.py
=============================================================================
PHASE 5 — PCM FEASIBILITY FILTERING (Objective1_PCM_Climate_Framework_Plan_v3,
Table 12), Rajasthan.

Hard-filters the shared PCM candidate database against each of Rajasthan's
Level-A climate clusters (cluster_profiles_rajasthan.csv, from
05_cluster_rajasthan.py) across 8 constraints. STATE_NAME is the only
hardcoded state reference — reusing this for another state means copying
this file and pointing it at that state's own cluster_profiles_{state}.csv
(the PCM database itself is NOT state-specific and needs no changes).

CONSTRAINTS (1-5 pre-existing design, kept as specified; 6-8 new this run):
  1. Melting window   : Tm in [Tm_target-5, Tm_target+8] C, auto-relaxed by
                         2K/round up to 4 rounds if a cluster has <5 survivors.
  2. Absolute band     : Tm in [42, 70] C (v3.0-corrected band, Table 12).
  3. Latent heat floor : L >= kappa * L_required, kappa=0.7 FIXED (kept as
                         specified — see "READ THIS FIRST" below for why
                         this is expected to zero out every cluster).
  4. Cycling           : >=300 cycles; FLAG (not exclude) if unreported.
  5. Supercooling      : <=8K; FLAG (not exclude) if unknown.
  6. Charging feasibility (NEW): Tm <= that cluster's Tm_target_capped_C
     (from 04_climate_signature_rajasthan.py's kt_p05-derived ceiling —
     referenced directly from cluster_profiles_rajasthan.csv, not re-derived).
  7. Corrosion veto (NEW): exclude bare (non-encapsulated) salt hydrates
     where the cluster's HSI_sunrise exceeds the 75th percentile across
     this state's clusters, unless the database row specifies encapsulation.
  8. Safety exclusion (NEW): exclude candidates flagged toxic or highly
     flammable. See "READ THIS FIRST" — the database can't actually
     support this as specified; implemented as designed, inert on current data.

READ THIS FIRST — three data-completeness facts that determine what this
run can actually show, verified against the real files before writing this
script (not assumed):

  A. THE PCM DATABASE IS STILL ~25 ROWS, NOT 40-60. 18 manufacturer rows
     (PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv) + 7
     literature rows (Singh2025 Table 2, reused verbatim from the
     Tamil-Nadu-pipeline reference script that already sourced them — see
     literature_rows() below) = 25 total. The 40-60 row target
     (RT58/60/62HC, OM55/65, a sourced salt hydrate) was never completed.
  B. NO ENCAPSULATION COLUMN EXISTS ANYWHERE, and only ONE row in the
     entire 25-candidate database is even salt-hydrate-typed (savE(R)
     HS36, Inorganic, Tm=35C) — and it's already excluded by constraints
     1/2 regardless (Tm=35C is below both the absolute 42C floor and any
     realistic relaxed melting window for this state's ~57C Tm_target).
     Constraint 7 is implemented correctly and generically (per the brief,
     for when Assam's humid clusters + real salt hydrate rows exist) but
     WILL NOT exclude anything in this run — not because Rajasthan is arid
     (though it is), but because there's structurally nothing for it to
     apply to yet.
  C. NO TOXICITY FIELD EXISTS AT ALL, and the database's `flammability`
     column is an unqualified Yes/No (does the candidate burn at all?),
     not a severity grade ("highly flammable" vs "mildly"). Hard-excluding
     every flammability=Yes row would incorrectly reject most organic PCMs
     (nearly all of which are combustible to some degree) on a field that
     was never designed to support that distinction. Constraint 8 is
     therefore implemented as: flag every candidate's flammability status
     for the record, but only actively EXCLUDE on an explicit high-severity
     signal if the data ever provides one (it currently doesn't) — treat
     `safety_flag` in the output as "needs a human safety review," not as
     a resolved pass/fail, and toxicity is ALWAYS flagged unverified since
     no such column exists to check.

  Net effect: constraints 7 and 8 are correctly built but inert on today's
  data (0 exclusions each, by data-completeness, not by design failure).
  Constraint 3, by contrast, is expected to be MAXIMALLY active — L_required
  is an idealized ceiling (~600-650 kJ/kg for this state's clusters, see
  04_climate_signature_rajasthan.py's docstring correction #4), and 0.7x
  that (~420-455 kJ/kg) exceeds every candidate in the database (best case
  ~252 kJ/kg). Expect EVERY cluster to fail to reach 5 survivors even after
  4 melting-window relaxation rounds, SOLELY because of constraint 3 — the
  melting-window relaxation this script performs (per spec, kept as-is)
  cannot rescue that, since it doesn't touch the latent-heat threshold.
  This is the exact "kappa needs per-cluster calibration, not a fixed 0.7"
  finding already documented in 04's docstring (correction #4, kappa
  calibration procedure) — this run is expected to reproduce it directly.

INPUTS:
  data/processed/cluster_profiles_rajasthan.csv        (05_cluster_rajasthan.py)
  PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv  (shared, not
      state-specific — path resolved relative to this file's grandparent)

OUTPUTS:
  data/processed/feasibility_survivors_rajasthan.csv — PRIMARY run, fixed
      kappa=0.7 (kept exactly as specified). One row per (cluster_id,
      pcm_id) evaluated at that cluster's FINAL melting-window relaxation
      round, with one column per constraint (pass/fail/flag/not_applicable/
      excluded) plus a `survives_all` boolean. Deliberately includes
      non-survivors too, not just the survivor subset — a per-constraint
      audit trail that only ever shows survivors (who trivially pass
      everything) isn't actually an audit trail. Filter to
      `survives_all == True` for the plain Top-N input.
  data/processed/feasibility_survivors_rajasthan_kappa_calibrated.csv —
      COMPANION run, not a replacement. Same audit-trail shape, but kappa
      is stepped down per cluster (0.7 -> 0.0 in 0.1 increments) until
      that cluster retains 8-20 survivors, per the calibration procedure
      already documented in 04_climate_signature_rajasthan.py's docstring
      (correction #4). Evaluated at each cluster's melting window AT THE
      PRIMARY RUN'S FINAL RELAXATION ROUND (melting-window relaxation is
      exhausted first, exactly as the primary run already does; kappa is
      then calibrated on top of that already-widened window) — see
      calibrate_kappa_for_cluster()'s docstring for why an earlier version
      of this that reset to the base window produced a degenerate result.
      Carries a `breakeven_kappa` column (= latent_heat_kJ_kg / L_required,
      computed once, no stepping needed) for every candidate that passes
      every OTHER constraint (`rescuable_by_kappa`) — sorting this
      descending within a cluster IS the admission order as kappa steps
      down from 0.7, without re-running the filter at each discrete step.
      The discrete calibrated kappa is still the headline number (matches
      the plan doc's 0.1-increment stopping-rule granularity); breakeven_
      kappa is the continuous backing evidence for how close each
      admitted candidate actually sits to the cutoff.

  Report both files together, not just the calibrated one — the fixed-
  kappa=0.7 result (0 survivors, fully diagnosed) is what demonstrates a
  literal Avargani-derived latent-heat floor is untenable as a hard
  cutoff; the calibrated result is meaningless without that baseline to
  compare against. Same shape as the ERA5 deaccumulation writeup: broken
  assumption -> diagnosis -> correction -> verification.

HOW TO RUN:
  python 07_feasibility_filter_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, BASE_DIR, ensure_data_dirs
from provenance_lib import file_fingerprint, fingerprint_id

ensure_data_dirs()

STATE_NAME = "rajasthan"

PROFILE_FILE = PROCESSED_DIR / f"cluster_profiles_{STATE_NAME}.csv"
OUT_FILE = PROCESSED_DIR / f"feasibility_survivors_{STATE_NAME}.csv"
OUT_FILE_KAPPA_CALIBRATED = PROCESSED_DIR / f"feasibility_survivors_{STATE_NAME}_kappa_calibrated.csv"

# Shared PCM database — NOT state-specific. Resolved relative to this
# file's grandparent (PCM-Selection-ML-model/), matching where PCM_data/
# actually sits alongside era5-rajasthan/. Edit if PCM_data/ ever moves.
PCM_MANUFACTURER_CSV = BASE_DIR.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"

# --- Constraints 1/2 (melting window + absolute band) ----------------------
MELT_WINDOW_LOW_OFFSET = -5.0     # Tm_target + this
MELT_WINDOW_HIGH_OFFSET = 8.0     # Tm_target + this
RELAX_STEP_K = 2.0
MAX_RELAX_ROUNDS = 4
MIN_SURVIVORS_TARGET = 5
ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0   # v3.0-corrected band, Table 12

# --- Constraint 3 (latent heat floor) --------------------------------------
# FIXED kappa=0.7, kept exactly as specified ("keep all of that") — see
# module docstring's "READ THIS FIRST" point re: why this is expected to
# exclude every candidate given L_required's ceiling framing.
LATENT_HEAT_KAPPA = 0.7

# --- Constraints 4/5 (flag, not exclude) -----------------------------------
CYCLING_MIN = 300
SUPERCOOLING_MAX_K = 8.0

# --- Constraint 7 (corrosion veto) ------------------------------------------
CORROSION_HSI_PERCENTILE = 0.75

# --- Kappa-calibration companion pass (see module docstring) ---------------
KAPPA_STEPS = [round(0.7 - 0.1 * i, 2) for i in range(8)]   # 0.7, 0.6, ..., 0.0
CALIBRATED_SURVIVOR_LO, CALIBRATED_SURVIVOR_HI = 8, 20


# ═══════════════════════════════════════════════════════════
# LOAD + BUILD PCM CANDIDATE DATABASE
# ═══════════════════════════════════════════════════════════

def load_manufacturer_rows(csv_path):
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out["pcm_id"] = df["product"]
    # Was np.where(df["is_rt_line"] == 1, "Rubitherm RT", "PLUSS savE") — that
    # column belonged to the old 2-manufacturer (Rubitherm/Pluss) database
    # schema and no longer exists; the current 01_preprocess.py keeps a real
    # `manufacturer` column instead (Rubitherm/Pluss/PCM Products Ltd/
    # PureTemp/CrodaTherm/Literature), which is a strictly more informative
    # family label now that the database spans 6 manufacturers, not 2. Not
    # used in any constraint logic (is_salt_hydrate() checks pcm_type only).
    out["family"] = df["manufacturer"]
    out["pcm_type"] = df["pcm_type"]
    out["Tm_C"] = df["Tm_melting"]
    out["Tm_freezing_C"] = df["Tm_freezing"]
    out["latent_heat_kJ_kg"] = df["latent_heat_melting"]
    out["cycles_tested"] = df["cycles_tested"]
    out["cycles_tested_status"] = df["cycles_tested_status"]
    out["flammable_raw"] = df["flammability"]
    out["supercooling_K"] = out["Tm_C"] - out["Tm_freezing_C"]
    # No encapsulation column exists anywhere in the source data — see
    # module docstring point B. Explicit NaN (not a guessed default),
    # handled downstream as "cannot verify" per the corrosion-veto logic.
    out["encapsulated"] = np.nan
    out["source"] = "manufacturer_datasheet_MICE_RF_PMM_completed"
    return out


def literature_rows():
    """Singh2025PCM_SWH_ComprehensiveReview Table 2 — same 7 rows already
    sourced and verified in the Tamil Nadu pipeline's PCM database script
    (read there, reused verbatim here with the same citation; not
    re-derived or altered). All Organic; no salt hydrate / no
    toxicity/encapsulation info reported for any of these — left NaN
    rather than guessed, matching that script's own convention."""
    rows = [
        {"pcm_id": "Myristic acid", "family": "Fatty acid", "pcm_type": "Organic",
         "Tm_C": 53.0, "latent_heat_kJ_kg": 190.0},
        {"pcm_id": "Palmitic acid", "family": "Fatty acid", "pcm_type": "Organic",
         "Tm_C": 63.0, "latent_heat_kJ_kg": 185.4},
        {"pcm_id": "Myristic-Palmitic eutectic (58/42)", "family": "Eutectic",
         "pcm_type": "Organic", "Tm_C": 42.6, "latent_heat_kJ_kg": 169.7},
        {"pcm_id": "Palmitic-Stearic eutectic (64.2/35.8)", "family": "Eutectic",
         "pcm_type": "Organic", "Tm_C": 52.3, "latent_heat_kJ_kg": 181.7},
        {"pcm_id": "Paraffin wax (generic)", "family": "Paraffin", "pcm_type": "Organic",
         "Tm_C": 64.0, "latent_heat_kJ_kg": 173.6},
        {"pcm_id": "C22H46 (docosane-class paraffin)", "family": "Paraffin",
         "pcm_type": "Organic", "Tm_C": 44.5, "latent_heat_kJ_kg": 249.0},
        {"pcm_id": "C30H62 (triacontane-class paraffin)", "family": "Paraffin",
         "pcm_type": "Organic", "Tm_C": 65.5, "latent_heat_kJ_kg": 252.0},
    ]
    df = pd.DataFrame(rows)
    df["cycles_tested"] = np.nan
    df["cycles_tested_status"] = "not_reported"
    df["supercooling_K"] = np.nan
    df["flammable_raw"] = "not_reported"
    df["encapsulated"] = np.nan
    df["source"] = "Singh2025_Table2_literature"
    return df


def is_salt_hydrate(row):
    """Structural detection, general per the brief (not tuned to this
    run's data — only savE(R) HS36 currently matches, and it's already
    excluded by constraints 1/2 regardless of this check)."""
    fam = str(row["family"]).lower()
    return row["pcm_type"] == "Inorganic" or "hydrate" in fam


# ═══════════════════════════════════════════════════════════
# PER-CANDIDATE CONSTRAINT EVALUATION  (constraints 1-8)
# ═══════════════════════════════════════════════════════════

def evaluate(pcm, tm_target, tm_capped, l_required, hsi_p75_cutoff, window_lo, window_hi,
             kappa=LATENT_HEAT_KAPPA):
    r = {"pcm_id": pcm["pcm_id"], "family": pcm["family"], "pcm_type": pcm["pcm_type"],
         "Tm_C": pcm["Tm_C"], "latent_heat_kJ_kg": pcm["latent_heat_kJ_kg"],
         "cycles_tested": pcm["cycles_tested"], "supercooling_K": pcm["supercooling_K"],
         "flammable_raw": pcm["flammable_raw"], "encapsulated_raw": pcm["encapsulated"],
         "source": pcm["source"]}

    # 1. Melting window (cluster- and relaxation-round-dependent)
    r["c1_melting_window"] = "pass" if window_lo <= pcm["Tm_C"] <= window_hi else "fail"

    # 2. Absolute band
    r["c2_absolute_band"] = "pass" if ABSOLUTE_TM_MIN <= pcm["Tm_C"] <= ABSOLUTE_TM_MAX else "fail"

    # 3. Latent heat floor — kappa defaults to the FIXED 0.7 used by the
    # primary run, but is a parameter so the kappa-calibration companion
    # pass (see calibrate_kappa() below) can re-evaluate at other kappa
    # values without duplicating this function.
    # breakeven_kappa = L_actual / L_required: the exact kappa at which
    # THIS candidate's latent heat would clear the floor on its own,
    # independent of whatever kappa was actually used for c3's pass/fail
    # above. Computed regardless of pass/fail so the calibration pass can
    # sort on it directly rather than re-running evaluate() at every step.
    r["breakeven_kappa"] = (pcm["latent_heat_kJ_kg"] / l_required
                             if pd.notna(pcm["latent_heat_kJ_kg"]) and l_required else np.nan)
    threshold = kappa * l_required
    if pd.isna(pcm["latent_heat_kJ_kg"]):
        r["c3_latent_heat"] = "flag_unreported"
    else:
        r["c3_latent_heat"] = "pass" if pcm["latent_heat_kJ_kg"] >= threshold else "fail"

    # 4. Cycling — flag, don't exclude, if unreported
    if pd.isna(pcm["cycles_tested"]):
        r["c4_cycling"] = "flag_unreported"
    else:
        r["c4_cycling"] = "pass" if pcm["cycles_tested"] >= CYCLING_MIN else "fail"

    # 5. Supercooling — flag, don't exclude, if unknown
    if pd.isna(pcm["supercooling_K"]):
        r["c5_supercooling"] = "flag_unknown"
    else:
        r["c5_supercooling"] = "pass" if pcm["supercooling_K"] <= SUPERCOOLING_MAX_K else "fail"

    # 6. Charging feasibility (NEW) — Tm must not exceed the cluster's
    # poor-day achievable ceiling (Tm_target_capped_C, referenced directly,
    # not re-derived).
    r["c6_charging_feasibility"] = "pass" if pcm["Tm_C"] <= tm_capped else "fail"

    # 7. Corrosion veto (NEW) — see module docstring point B for why this
    # is inert on the current database (only 1 salt-hydrate-typed row,
    # already excluded by 1/2 regardless).
    if not is_salt_hydrate(pcm):
        r["c7_corrosion_veto"] = "not_applicable"
    elif hsi_p75_cutoff == hsi_p75_cutoff and pcm.get("_cluster_hsi", np.nan) <= hsi_p75_cutoff:
        r["c7_corrosion_veto"] = "not_applicable_low_hsi_cluster"
    elif pcm["encapsulated"] is True:
        r["c7_corrosion_veto"] = "pass_encapsulated"
    elif pd.isna(pcm["encapsulated"]):
        # Conservative default: cannot verify encapsulation -> treated as
        # excluded, not silently passed. See module docstring point B.
        r["c7_corrosion_veto"] = "excluded_unverified_encapsulation"
    else:
        r["c7_corrosion_veto"] = "excluded_bare_high_hsi"

    # 8. Safety exclusion (NEW) — see module docstring point C for why
    # this can only flag, not hard-exclude, on the current database.
    if pcm["flammable_raw"] in ("not_reported", None) or pd.isna(pcm["flammable_raw"]):
        r["c8_safety"] = "flag_unverified_flammability_and_toxicity"
    else:
        # Field is an unqualified Yes/No, not a severity grade — cannot
        # distinguish "highly flammable" (spec's exclusion trigger) from
        # ordinary combustibility. Recorded, not excluded on. Toxicity has
        # no column at all -> always flagged unverified regardless.
        r["c8_safety"] = f"flag_unverified_toxicity_flammable_raw={pcm['flammable_raw']}"

    blocking = [r["c1_melting_window"], r["c2_absolute_band"], r["c3_latent_heat"],
                r["c6_charging_feasibility"]]
    r["survives_c1_c2_c3_c6"] = all(v == "pass" or v.startswith("flag_") for v in blocking) and \
        all(v != "fail" for v in blocking)
    r["survives_c7_corrosion"] = not r["c7_corrosion_veto"].startswith("excluded_")
    r["survives_c8_safety"] = not r["c8_safety"].startswith("excluded_")
    r["survives_all"] = r["survives_c1_c2_c3_c6"] and r["survives_c7_corrosion"] and r["survives_c8_safety"]

    # rescuable_by_kappa: passes every constraint EXCEPT (possibly) c3 —
    # i.e. the set kappa-calibration can actually rescue. A candidate
    # failing on melting window or charging feasibility stays excluded no
    # matter how far kappa drops, so its breakeven_kappa isn't meaningful
    # to report even though it's still computed above.
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
    retains CALIBRATED_SURVIVOR_LO-HI (8-20) candidates, per the
    calibration procedure already documented in
    04_climate_signature_rajasthan.py's docstring (correction #4).

    window_lo/window_hi are the cluster's melting window AT THE PRIMARY
    RUN'S FINAL RELAXATION ROUND (passed in by main(), not recomputed at
    round 0 here) — a corrected design choice, not the original one. A
    first version of this function re-derived the BASE (un-relaxed) window
    independently of the primary run's relaxation, on the reasoning that
    kappa-calibration and melting-window relaxation are two independent
    constraints deserving independent procedures. That produced a
    degenerate, misleading result: constraint 1 at the base window
    requires Tm in [52,65] (Tm_target=57 +5/+8), while constraint 6
    requires Tm <= Tm_target_capped (~43-47C for this state's clusters) —
    two essentially disjoint ranges regardless of kappa, so
    rescuable_by_kappa came out 0 in every cluster for a reason that had
    nothing to do with latent heat at all. Sequencing the two relaxations
    instead (exhaust melting-window relaxation first, as the primary run
    already does; THEN calibrate kappa on top of that already-widened
    window) avoids that artifact while still keeping this a single,
    interpretable 1D search rather than a joint 2D one.

    Returns (calibrated_kappa, rows_at_calibrated_kappa, status_string).
    rows_at_calibrated_kappa carries breakeven_kappa for every candidate
    (computed once, kappa-independent) — sorting the rescuable subset by
    breakeven_kappa descending gives the admission order as kappa fell
    from 0.7, without re-evaluating at every step.
    """
    # breakeven_kappa doesn't depend on which kappa we evaluate at, so
    # compute the base rows (kappa=0.7 is an arbitrary choice here, purely
    # to get c1/c2/c6/c7/c8/breakeven_kappa/rescuable_by_kappa populated —
    # those five constraints and breakeven_kappa are identical at any kappa).
    base_rows = []
    for pcm in pcm_db.to_dict("records"):
        pcm["_cluster_hsi"] = cluster_hsi
        base_rows.append(evaluate(pcm, tm_target, tm_capped, l_required,
                                   hsi_p75_cutoff, window_lo, window_hi, kappa=0.7))

    # A candidate passes c3 at threshold k when latent_heat >= k*L_required,
    # i.e. breakeven_kappa >= k (its own ratio clears the bar) — NOT
    # breakeven_kappa <= k. FIXED 2026-08-11: an earlier version of this
    # loop had the inequality backwards, which counted candidates as
    # "admitted" at kappa values far above their actual breakeven — caught
    # by a direct contradiction in the output (a cluster reporting
    # status="in_band" at kappa=0.7 while its own re-evaluated n_survived
    # came out 0, which is only possible if the stepping search and the
    # final evaluate() call disagreed on who passes at that kappa).
    rescuable = [r for r in base_rows if r["rescuable_by_kappa"]]
    already_in_at_07 = sum(1 for r in rescuable if r["breakeven_kappa"] >= 0.7)

    chosen_kappa, status = None, None
    for k in KAPPA_STEPS:
        n = sum(1 for r in rescuable if r["breakeven_kappa"] == r["breakeven_kappa"]
                and r["breakeven_kappa"] >= k)
        if CALIBRATED_SURVIVOR_LO <= n <= CALIBRATED_SURVIVOR_HI:
            chosen_kappa, status = k, "in_band"
            break
    if chosen_kappa is None:
        # No step landed cleanly in [8,20] — take the first step reaching
        # >=8 (even if it overshoots 20), else report insufficient.
        for k in KAPPA_STEPS:
            n = sum(1 for r in rescuable if r["breakeven_kappa"] == r["breakeven_kappa"]
                    and r["breakeven_kappa"] >= k)
            if n >= CALIBRATED_SURVIVOR_LO:
                chosen_kappa, status = k, "exceeds_20_no_clean_landing"
                break
    if chosen_kappa is None:
        chosen_kappa, status = 0.0, "insufficient_even_at_kappa_0"

    # Re-evaluate c3/survives_all at the chosen kappa for the final output.
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

    if not PROFILE_FILE.exists():
        raise SystemExit(f"ERROR: {PROFILE_FILE} not found — run 05_cluster_{STATE_NAME}.py first.")
    if not PCM_MANUFACTURER_CSV.exists():
        raise SystemExit(f"ERROR: {PCM_MANUFACTURER_CSV} not found.")

    profiles = pd.read_csv(PROFILE_FILE)
    # Provenance stamp — added 2026-08-11 after Phase 7 caught Phase 5's
    # and Phase 6's outputs disagreeing on cluster_id/pcm_id pairing
    # because they'd been run against two different on-disk versions of
    # this exact file. See provenance_lib.py's module docstring.
    profile_fp_id = fingerprint_id(file_fingerprint(PROFILE_FILE))
    print(f"  {PROFILE_FILE.name} fingerprint: {profile_fp_id}")
    manuf = load_manufacturer_rows(PCM_MANUFACTURER_CSV)
    lit = literature_rows()
    pcm_db = pd.concat([manuf, lit], ignore_index=True, sort=False)
    print(f"\n  PCM candidates loaded: {len(manuf)} manufacturer + {len(lit)} literature "
          f"= {len(pcm_db)} total")
    print(f"  Clusters: {len(profiles)}")

    hsi_p75_cutoff = profiles["HSI_sunrise"].quantile(CORROSION_HSI_PERCENTILE) \
        if "HSI_sunrise" in profiles.columns else np.nan
    print(f"  Corrosion-veto HSI cutoff (75th percentile across this state's "
          f"{len(profiles)} clusters): {hsi_p75_cutoff:.3f}")

    n_salt_hydrate = pcm_db.apply(is_salt_hydrate, axis=1).sum()
    print(f"  Salt-hydrate-typed candidates in the database: {n_salt_hydrate} "
          f"(see module docstring point B for why the corrosion veto is expected "
          f"to be inert regardless of this count)")

    all_rows = []
    summary_rows = []
    cluster_final_windows = {}   # cid -> (window_lo, window_hi) at the primary run's final
                                  # relaxation round — reused by the kappa-calibration pass below

    for prof in profiles.itertuples():
        cid = prof.cluster_id
        tm_target = prof.Tm_target_C
        tm_capped = prof.Tm_target_capped_C
        l_required = prof.L_required_kJ_per_kg
        cluster_hsi = prof.HSI_sunrise if hasattr(prof, "HSI_sunrise") else np.nan

        print(f"\n  --- Cluster {cid} "
              f"(Tm_target={tm_target:.1f}C, Tm_target_capped={tm_capped:.1f}C, "
              f"L_required={l_required:.0f} kJ/kg [CEILING], HSI={cluster_hsi:.2f}) ---")

        final_rows = None
        final_relax = None
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
                  f"This cluster's Top-3 result needs an explicit caveat in the paper — "
                  f"see module docstring: constraint 3 (latent heat, kappa=0.7 fixed "
                  f"against a ceiling-framed L_required) is the near-certain cause, not "
                  f"the melting window this relaxation loop actually widens.")

        # Per-constraint exclusion counts for this cluster, for "N excluded
        # by X" reporting in the results section.
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
        print(f"    Entered: {len(final_rows)}  |  Survived (all constraints): {n_final_survive}")
        print(f"    Excluded by: " + ", ".join(f"{k}={v}" for k, v in exclude_counts.items()))

        for r in final_rows:
            r["cluster_id"] = cid
            r["relax_round_used"] = final_relax
            r["melting_window_widen_K"] = RELAX_STEP_K * final_relax
        all_rows.extend(final_rows)

        summary_rows.append({"cluster_id": cid, "n_entered": len(final_rows),
                              "n_survived": n_final_survive, "relax_round_used": final_relax,
                              "melting_window_widen_K": RELAX_STEP_K * final_relax,
                              **{f"excluded_{k}": v for k, v in exclude_counts.items()}})

    out_df = pd.DataFrame(all_rows)
    out_df["upstream_cluster_profile_fingerprint"] = profile_fp_id
    col_order = ["cluster_id", "pcm_id", "family", "pcm_type", "Tm_C", "latent_heat_kJ_kg",
                 "cycles_tested", "supercooling_K", "flammable_raw", "encapsulated_raw", "source",
                 "relax_round_used", "melting_window_widen_K",
                 "c1_melting_window", "c2_absolute_band", "c3_latent_heat", "c4_cycling",
                 "c5_supercooling", "c6_charging_feasibility", "c7_corrosion_veto", "c8_safety",
                 "survives_all", "upstream_cluster_profile_fingerprint"]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]
    out_df.to_csv(OUT_FILE, index=False)

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print(f"\n  Saved: {OUT_FILE}  ({len(out_df)} (cluster, pcm) rows total, "
          f"{int(out_df['survives_all'].sum())} surviving all 8 constraints)")

    if summary_df["n_survived"].lt(MIN_SURVIVORS_TARGET).all():
        print("\n  [WARNING] EVERY cluster failed to reach the 5-survivor target. Per this "
              "run's audit trail, constraint 3 (latent heat, kappa=0.7 fixed) is doing that "
              "on its own in every cluster — check the c3_latent_heat exclusion counts above "
              "before treating constraints 6/7/8 as the cause. See module docstring for the "
              "documented kappa-calibration alternative (04_climate_signature_rajasthan.py's "
              "correction #4) if you want a non-empty result from this state's data as-is.")
    print("=" * 68)

    # ═══════════════════════════════════════════════════════════
    # KAPPA-CALIBRATION COMPANION PASS — companion to, not a replacement
    # for, the fixed-kappa=0.7 result above. See module docstring.
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print(f"  KAPPA-CALIBRATION COMPANION PASS — {STATE_NAME.title()}")
    print(f"  Steps: {KAPPA_STEPS}  |  Target band: "
          f"{CALIBRATED_SURVIVOR_LO}-{CALIBRATED_SURVIVOR_HI} survivors/cluster")
    print(f"  Evaluated at each cluster's melting window from the primary run's FINAL "
          f"relaxation round (not the base window) — see calibrate_kappa_for_cluster()'s "
          f"docstring for why: sequencing (exhaust melting-window relaxation first, then "
          f"calibrate kappa) avoids a degenerate result the base-window version produced.")
    print("=" * 68)

    calib_all_rows, calib_summary_rows = [], []
    for prof in profiles.itertuples():
        cid = prof.cluster_id
        cluster_hsi = prof.HSI_sunrise if hasattr(prof, "HSI_sunrise") else np.nan
        window_lo, window_hi = cluster_final_windows[cid]

        chosen_kappa, final_rows, status, n_already_at_07 = calibrate_kappa_for_cluster(
            pcm_db, cid, prof.Tm_target_C, prof.Tm_target_capped_C,
            prof.L_required_kJ_per_kg, hsi_p75_cutoff, cluster_hsi, window_lo, window_hi)

        n_survive = sum(r["survives_all"] for r in final_rows)
        print(f"\n  --- Cluster {cid}: calibrated kappa = {chosen_kappa}  "
              f"({n_survive} survivors, status={status}) ---")
        if status != "in_band":
            print(f"    [NOTE] Did not land cleanly in the "
                  f"{CALIBRATED_SURVIVOR_LO}-{CALIBRATED_SURVIVOR_HI} band at any 0.1 step "
                  f"— see status above. {'Report this as a finding about the cluster demand ' 'profile, not a latent-heat failure to paper over.' if status == 'insufficient_even_at_kappa_0' else ''}")

        # Sorted descending by breakeven_kappa = admission ORDER as kappa
        # steps down from 0.7: the candidate with the highest breakeven_
        # kappa is admitted first (needs the smallest drop from 0.7), not
        # "already admitted at 0.7" — every rescuable candidate here still
        # FAILS c3 at the literal kappa=0.7 used by the primary run (that's
        # exactly why the primary run shows 0 survivors); each is admitted
        # only once kappa drops to AT OR BELOW its own breakeven_kappa.
        rescuable_sorted = sorted(
            (r for r in final_rows if r["rescuable_by_kappa"] and r["breakeven_kappa"] == r["breakeven_kappa"]),
            key=lambda r: -r["breakeven_kappa"])
        print(f"    Admission order as kappa falls from 0.7 (first 5 shown, "
              f"of {len(rescuable_sorted)} rescuable-by-kappa candidates total — "
              f"each is admitted once kappa drops to <= its own breakeven_kappa, "
              f"NOT already admitted at 0.7):")
        for r in rescuable_sorted[:5]:
            print(f"      breakeven_kappa={r['breakeven_kappa']:.3f}  {r['pcm_id']}  "
                  f"({r['family']}, Tm={r['Tm_C']:.1f}C, L={r['latent_heat_kJ_kg']:.0f} kJ/kg)")

        for r in final_rows:
            r["cluster_id"] = cid
            r["calibrated_kappa"] = chosen_kappa
            r["calibration_status"] = status
        calib_all_rows.extend(final_rows)
        calib_summary_rows.append({"cluster_id": cid, "calibrated_kappa": chosen_kappa,
                                    "n_survived": n_survive, "status": status,
                                    "n_rescuable_by_kappa_total": len(rescuable_sorted)})

    calib_out_df = pd.DataFrame(calib_all_rows)
    calib_out_df["upstream_cluster_profile_fingerprint"] = profile_fp_id
    calib_col_order = ["cluster_id", "pcm_id", "family", "pcm_type", "Tm_C", "latent_heat_kJ_kg",
                        "cycles_tested", "supercooling_K", "flammable_raw", "encapsulated_raw",
                        "source", "breakeven_kappa", "rescuable_by_kappa", "calibrated_kappa",
                        "calibration_status", "c1_melting_window", "c2_absolute_band",
                        "c3_latent_heat", "c4_cycling", "c5_supercooling", "c6_charging_feasibility",
                        "c7_corrosion_veto", "c8_safety", "survives_all",
                        "upstream_cluster_profile_fingerprint"]
    calib_out_df = calib_out_df[[c for c in calib_col_order if c in calib_out_df.columns]]
    calib_out_df.to_csv(OUT_FILE_KAPPA_CALIBRATED, index=False)

    print("\n" + "=" * 68)
    print("  KAPPA-CALIBRATION SUMMARY")
    print("=" * 68)
    print(pd.DataFrame(calib_summary_rows).to_string(index=False))
    print(f"\n  Saved: {OUT_FILE_KAPPA_CALIBRATED}  ({len(calib_out_df)} (cluster, pcm) rows, "
          f"{int(calib_out_df['survives_all'].sum())} surviving at each cluster's calibrated kappa)")
    print(f"\n  Report BOTH {OUT_FILE.name} (fixed kappa=0.7, the diagnostic baseline) AND "
          f"{OUT_FILE_KAPPA_CALIBRATED.name} (calibrated, the usable result) together — "
          f"see module docstring for why the pair is the point, not either file alone.")
    print("=" * 68)


if __name__ == "__main__":
    main()
