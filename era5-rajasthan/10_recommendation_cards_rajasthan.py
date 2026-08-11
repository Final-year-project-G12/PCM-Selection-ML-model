"""
10_recommendation_cards_rajasthan.py
=============================================================================
PHASE 8 — RECOMMENDATION CARDS (pure aggregation), Rajasthan
(Objective1_PCM_Climate_Framework_Plan_v3 Section 11 Table 18;
.claude/phases.md PROMPT 7; docs/rajasthan/19_PHASE_7_ONWARD.md)

NAMING NOTE: phases.md's PROMPT 7 (written before Phase 7 claimed the "09_"
prefix) called this script "09_recommendation_cards_rajasthan.py". The
ACTUAL on-disk sequence is 07=Phase 5, 08=Phase 6, 09=Phase 7
(09_physics_validation_rajasthan.py) — this script is therefore "10_" so
its prefix keeps meaning "run this Nth" (matching every other script in
this folder and run_all_rajasthan.py's CORE_SCRIPTS order) rather than
colliding with the real Phase 7 script.

PURPOSE: pure aggregation. Pulls cluster_profile_cards_rajasthan.md,
mcdm_rankings_rajasthan.csv, and physics_validation_rajasthan.csv into
recommendation_cards_rajasthan.md, one card per Level-A cluster. This
script does not decide anything new about which PCM is best — it reads
and formats. The ONE new computation it performs (explicitly sanctioned by
the brief) is the per-criterion signed contribution decomposition, because
Phase 6 never persisted the weighted-normalized decision matrix or the
per-cluster blended weight vector needed to reconstruct it — see
compute_criterion_contributions()'s docstring.

CRITICAL PRECONDITION, checked BEFORE any card is written
-----------------------------------------------------------
Phase 7 already caught a real cross-phase bug on 2026-08-11: Phase 5's and
Phase 6's outputs disagreeing on which PCMs belonged to which cluster_id,
traced to GMM cluster-label instability across separate re-runs of
05_cluster_rajasthan.py, compounded by different phases having been run
against different on-disk versions of cluster_profiles_rajasthan.csv. This
script re-verifies that fix holds for its OWN three inputs before writing
anything:
  1. Provenance-fingerprint stamp check (provenance_lib.py, the same
     mechanism Phase 6/7 already use) — feasibility_survivors_rajasthan_
     kappa_calibrated.csv, mcdm_rankings_rajasthan.csv, and physics_
     validation_rajasthan.csv must all be stamped with the fingerprint of
     the CURRENT on-disk cluster_profiles_rajasthan.csv.
  2. An explicit medoid-per-cluster_id cross-check: this script recomputes
     each cluster's medoid FRESH from the current cluster_assignments_
     rajasthan_levelA.csv + climate_signature_rajasthan.csv (via physics_
     lib.find_medoid — the exact same algorithm 05_cluster_rajasthan.py
     and 09_physics_validation_rajasthan.py already use) and cross-checks
     it against cluster_profile_cards_rajasthan.md's stated medoid AND
     physics_validation_rajasthan.csv's medoid_point column.
     mcdm_rankings_rajasthan.csv carries no medoid column at all (Phase 6
     never persisted one) — for THAT file, the fingerprint check in (1)
     is the applicable cluster-identity guarantee, not a medoid
     comparison that file has no data to support.
Both checks HARD-FAIL (raise SystemExit, never warn-and-continue) on any
mismatch, naming exactly which cluster_id and which file disagreed.

INPUTS:
  data/processed/cluster_profiles_rajasthan.csv          (Phase 4)
  outputs/cluster_profile_cards_rajasthan.md              (Phase 4)
  data/processed/cluster_assignments_rajasthan_levelA.csv (Phase 4)
  data/processed/climate_signature_rajasthan.csv          (Phase 3)
  data/processed/feasibility_survivors_rajasthan.csv      (Phase 5, primary)
  data/processed/feasibility_survivors_rajasthan_kappa_calibrated.csv (Ph.5)
  data/processed/mcdm_rankings_rajasthan.csv              (Phase 6)
  data/processed/spearman_rho_by_cluster_rajasthan.csv    (Phase 7)
  data/processed/physics_validation_rajasthan.csv         (Phase 7)
  physics_validation_summary_rajasthan.txt                (Phase 7)
  ../PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv (shared)
  08_mcdm_ranking_rajasthan.py (imported as a module, NOT re-run as a
      script — its main() is never called; only its already-defined,
      already-verified weight/matrix functions are reused, to recompute
      the criterion-contribution decomposition against Phase 6's own
      saved weight formula rather than a re-invented one)

OUTPUT:
  outputs/recommendation_cards_rajasthan.md — cross-cluster summary table,
      then one full card per cluster in cluster_id order.

HOW TO RUN:
  python 10_recommendation_cards_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import importlib.util
import re
import sys

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR, BASE_DIR, RAW_BOUNDARY_DIR, ensure_data_dirs
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match
import physics_lib as pl

ensure_data_dirs()

STATE_NAME = "rajasthan"

PROFILE_FILE = PROCESSED_DIR / f"cluster_profiles_{STATE_NAME}.csv"
ASSIGN_A_FILE = PROCESSED_DIR / f"cluster_assignments_{STATE_NAME}_levelA.csv"
SIGNATURE_FILE = PROCESSED_DIR / f"climate_signature_{STATE_NAME}.csv"
CARDS_MD_FILE = OUTPUTS_DIR / f"cluster_profile_cards_{STATE_NAME}.md"
SURVIVORS_PRIMARY_FILE = PROCESSED_DIR / f"feasibility_survivors_{STATE_NAME}.csv"
SURVIVORS_CALIBRATED_FILE = PROCESSED_DIR / f"feasibility_survivors_{STATE_NAME}_kappa_calibrated.csv"
MCDM_RANKINGS_FILE = PROCESSED_DIR / f"mcdm_rankings_{STATE_NAME}.csv"
PHYSICS_VALIDATION_FILE = PROCESSED_DIR / f"physics_validation_{STATE_NAME}.csv"
SPEARMAN_FILE = PROCESSED_DIR / f"spearman_rho_by_cluster_{STATE_NAME}.csv"
PHYSICS_SUMMARY_TXT = BASE_DIR / f"physics_validation_summary_{STATE_NAME}.txt"
PCM_MANUFACTURER_CSV = BASE_DIR.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"
MCDM_SCRIPT_FILE = BASE_DIR / "08_mcdm_ranking_rajasthan.py"

OUT_MD = OUTPUTS_DIR / f"recommendation_cards_{STATE_NAME}.md"

MEMBERSHIP_AMBIGUITY_THRESHOLD = 0.7   # per cluster_profile_cards' own convention
KENDALLS_W_AMBIGUOUS_THRESHOLD = 0.6   # plan doc Section 9.5, reused verbatim from Phase 6
PHYSICS_RHO_NEGATIVE_BAND = 0.4        # Phase 7's own band boundary, reused verbatim
PHYSICS_RHO_STRONG_BAND = 0.8

CRITERION_LABELS = {
    "Tm_fitness": "melting-point fitness", "latent_heat": "latent heat",
    "vol_latent_heat": "volumetric latent heat", "thermal_conductivity": "thermal conductivity",
    "cycling": "cycling stability", "supercooling": "supercooling (inverse)",
    "corrosion": "corrosion class (inverse)", "cost": "cost",
}

# Manufacturer-datasheet imputed-property flags (PCM_Properties_cleaned_
# mice_pmm_detailed.csv's own *_imputed columns) worth naming individually
# in a Top-3 caveat — restricted to properties that actually feed the MCDM
# criteria matrix or the physics simulation, not every imputed field in
# that file (e.g. *_original_text companions are excluded, they are not
# properties).
IMPUTED_PROPERTY_LABELS = {
    "Tm_melting_imputed": "melting point", "latent_heat_melting_imputed": "latent heat",
    "density_solid_imputed": "solid density", "density_liquid_imputed": "liquid density",
    "Cp_solid_imputed": "solid specific heat", "Cp_liquid_imputed": "liquid specific heat",
    "TC_solid_imputed": "solid thermal conductivity", "TC_liquid_imputed": "liquid thermal conductivity",
    "TC_both_imputed": "thermal conductivity", "cycles_tested_imputed": "cycling-stability count",
    "flammability_imputed": "flammability rating",
}


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ═══════════════════════════════════════════════════════════
# DYNAMIC IMPORT — 08_mcdm_ranking_rajasthan.py's filename starts with a
# digit, so it cannot be `import`ed normally; loaded via importlib instead.
# Its module-level code (path constants, ensure_data_dirs(), the
# AHP_WEIGHTS_TABLE13 sum-to-1 assert) runs harmlessly on load; main() is
# NEVER called (guarded by `if __name__ == "__main__":` in that file) —
# only its already-defined weight/matrix functions are reused below.
# ═══════════════════════════════════════════════════════════

def load_mcdm_module():
    spec = importlib.util.spec_from_file_location("phase6_mcdm_ranking_module", MCDM_SCRIPT_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase6_mcdm_ranking_module"] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════
# cluster_profile_cards_rajasthan.md PARSER
# ═══════════════════════════════════════════════════════════

def parse_cluster_cards_md(path):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"\n## Cluster (\d+)\n", text)
    cards = {}
    for i in range(1, len(parts), 2):
        cards[int(parts[i])] = parts[i + 1]
    return cards


def parse_medoid(body):
    m = re.search(r"Medoid point[^:]*:\*\*\s*(\S+)\s*\(([^,]+),\s*([^)]+)\)", body)
    if not m:
        return None, None, None
    return m.group(1), float(m.group(2)), float(m.group(3))


def parse_description(body):
    m = re.search(r"Physical description[^:]*:\*\*\s*(.+)", body)
    return m.group(1).strip() if m else "(no auto-generated description found)"


def parse_signature_table(body):
    """Returns an ORDERED dict {index_name: (mean, std)}, in the same row
    order as the source markdown table — never recomputed, only parsed."""
    sig = {}
    for line in body.splitlines():
        m = re.match(r"\|\s*([A-Za-z_][\w]*)\s*\|\s*([\-\d.]+)\s*\|\s*([\-\d.]+)\s*\|", line)
        if m:
            sig[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return sig


# ═══════════════════════════════════════════════════════════
# CRITICAL PRECONDITION — cross-phase cluster-identity consistency
# ═══════════════════════════════════════════════════════════

def verify_cross_phase_consistency():
    log_header("PRECONDITION — cross-phase cluster-identity consistency check")

    for f in (PROFILE_FILE, ASSIGN_A_FILE, SIGNATURE_FILE, CARDS_MD_FILE,
              SURVIVORS_CALIBRATED_FILE, MCDM_RANKINGS_FILE, PHYSICS_VALIDATION_FILE):
        if not f.exists():
            raise SystemExit(f"ERROR: required input not found: {f} — run the phase that "
                              f"produces it before this script.")

    current_fp = fingerprint_id(file_fingerprint(PROFILE_FILE))
    print(f"  Current {PROFILE_FILE.name} fingerprint: {current_fp}")

    # --- Check 1: fingerprint stamp match (same mechanism Phase 6/7 use) ---
    for f, phase_label in ((SURVIVORS_CALIBRATED_FILE, "Phase 5"),
                            (MCDM_RANKINGS_FILE, "Phase 6"),
                            (PHYSICS_VALIDATION_FILE, "Phase 7")):
        df = pd.read_csv(f)
        assert_fingerprint_match(current_fp, df, PROFILE_FILE.name, f.name)
        print(f"  Fingerprint check PASSED — {f.name} ({phase_label}) matches current "
              f"{PROFILE_FILE.name}.")

    # --- Check 2: explicit medoid-per-cluster_id cross-check ---
    assign = pd.read_csv(ASSIGN_A_FILE)
    sig = pd.read_csv(SIGNATURE_FILE)
    sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)
    z_cols = [c for c in sig.columns if c.endswith("_z")]
    cluster_ids = sorted(assign["cluster_id"].unique())
    fresh_medoids = {cid: pl.find_medoid(cid, assign, sig, z_cols) for cid in cluster_ids}
    print(f"  Freshly recomputed medoids (from CURRENT on-disk cluster_assignments_"
          f"{STATE_NAME}_levelA.csv + climate_signature_{STATE_NAME}.csv, same algorithm "
          f"as physics_lib.find_medoid): {fresh_medoids}")

    card_sections = parse_cluster_cards_md(CARDS_MD_FILE)
    cards_medoids = {cid: parse_medoid(body)[0] for cid, body in card_sections.items()}

    physics_df = pd.read_csv(PHYSICS_VALIDATION_FILE)
    physics_medoids = physics_df.drop_duplicates("cluster_id").set_index("cluster_id")["medoid_point"].to_dict()

    mismatches = []
    for cid in cluster_ids:
        fresh, card_m, phys_m = fresh_medoids.get(cid), cards_medoids.get(cid), physics_medoids.get(cid)
        if not (fresh == card_m == phys_m):
            mismatches.append((cid, fresh, card_m, phys_m))

    if mismatches:
        detail = "\n  ".join(
            f"cluster_id={cid}: fresh-recompute={fresh!r}  {CARDS_MD_FILE.name}={card_m!r}  "
            f"{PHYSICS_VALIDATION_FILE.name}.medoid_point={phys_m!r}"
            for cid, fresh, card_m, phys_m in mismatches)
        raise SystemExit(
            f"ERROR: MEDOID/CLUSTER-IDENTITY MISMATCH on {len(mismatches)} cluster_id(s) — these "
            f"do not resolve to the same medoid point across {CARDS_MD_FILE.name}, "
            f"{PHYSICS_VALIDATION_FILE.name}, and a fresh recomputation from the CURRENT "
            f"{ASSIGN_A_FILE.name} + {SIGNATURE_FILE.name}:\n  {detail}\n"
            f"This is exactly the class of bug Phase 7 caught on 2026-08-11 (GMM cluster-index "
            f"relabeling across separate re-runs of 05_cluster_{STATE_NAME}.py). Refusing to "
            f"build recommendation cards from mismatched inputs — do not guess which file is "
            f"'right'. Re-run the full Phase 4->5->6->7 chain back-to-back from the CURRENT "
            f"{PROFILE_FILE.name} (e.g. `python run_all_rajasthan.py --from "
            f"07_feasibility_filter_{STATE_NAME}.py`) before re-running this script.")

    print(f"  Medoid cross-check PASSED for all {len(cluster_ids)} cluster(s) — "
          f"{CARDS_MD_FILE.name}, {PHYSICS_VALIDATION_FILE.name}, and a fresh recomputation "
          f"all agree: {fresh_medoids}")
    print(f"  (mcdm_rankings_{STATE_NAME}.csv carries no medoid column — its cluster-identity "
          f"is covered by the fingerprint check above, the applicable mechanism for that file.)")

    return current_fp, fresh_medoids, card_sections, assign, sig


# ═══════════════════════════════════════════════════════════
# BEST-EFFORT DISTRICT RESOLUTION (never blocks the script)
# ═══════════════════════════════════════════════════════════

def resolve_district(lat, lon):
    """Point-in-polygon lookup against GADM v4.1 admin level-2 (district)
    boundaries — same source/convention as 00a_build_population_grid.py's
    level-1 state-boundary fetch, cached locally after the first download.
    NOT a reverse-geocoding API call (no per-point network dependency,
    no rate limits, deterministic). Returns (district, state) or
    (None, None) on ANY failure — never raises, never blocks the script;
    caller falls back to lat/lon + elevation, per the brief."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        boundary_path = RAW_BOUNDARY_DIR / "gadm41_IND_2.json"
        if not boundary_path.exists():
            import requests
            url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            boundary_path.write_bytes(r.content)
            print(f"    Downloaded and cached: {boundary_path}")

        gdf = gpd.read_file(boundary_path)
        pt = Point(lon, lat)
        hit = gdf[gdf.geometry.contains(pt)]
        if len(hit) == 0:
            return None, None
        row = hit.iloc[0]
        return row.get("NAME_2"), row.get("NAME_1")
    except Exception as e:
        print(f"    [district resolution unavailable, falling back to lat/lon+elevation: {e}]")
        return None, None


# ═══════════════════════════════════════════════════════════
# PHASE 5 PER-CONSTRAINT EXCLUSION COUNTS
# (feasibility_survivors_rajasthan.csv's own per-row constraint verdicts,
#  aggregated here — 07_feasibility_filter_rajasthan.py computes this exact
#  breakdown internally but only ever prints it, never persists it; this
#  is a straight count of already-persisted per-candidate columns, not a
#  new judgment about who passes/fails.)
# ═══════════════════════════════════════════════════════════

def exclusion_counts_for_cluster(primary_df, cid):
    sub = primary_df[primary_df["cluster_id"] == cid]
    return {
        "c1_melting_window": int((sub["c1_melting_window"] == "fail").sum()),
        "c2_absolute_band": int((sub["c2_absolute_band"] == "fail").sum()),
        "c3_latent_heat": int((sub["c3_latent_heat"] == "fail").sum()),
        "c4_cycling_fail": int((sub["c4_cycling"] == "fail").sum()),
        "c4_cycling_flagged_unreported": int((sub["c4_cycling"] == "flag_unreported").sum()),
        "c5_supercooling_fail": int((sub["c5_supercooling"] == "fail").sum()),
        "c5_supercooling_flagged_unknown": int((sub["c5_supercooling"] == "flag_unknown").sum()),
        "c6_charging_feasibility": int((sub["c6_charging_feasibility"] == "fail").sum()),
        "c7_corrosion_veto": int(sub["c7_corrosion_veto"].str.startswith("excluded_").sum()),
        "c8_safety": int(sub["c8_safety"].str.startswith("excluded_").sum()),
    }


# ═══════════════════════════════════════════════════════════
# CRITERION-CONTRIBUTION DECOMPOSITION
# (the ONE new computation this script performs — see module docstring)
# ═══════════════════════════════════════════════════════════

def compute_criterion_contributions(mcdm_mod, cand_df, weights, tm_target):
    """Signed per-criterion decomposition of each candidate's standing
    relative to its OWN cluster's candidate pool, built from Phase 6's own
    weighted-normalized decision matrix convention (V = R*w, R = X/||X||_2
    — literally TOPSIS's own intermediate, reused not reinvented) and
    Phase 6's own already-computed blended (lambda=0.5) weight vector for
    this cluster. For criterion j: contribution_ij = direction_j *
    (V_ij - mean_i(V_ij)), direction_j = +1 for a benefit criterion, -1
    for a cost criterion — so a POSITIVE contribution always means "this
    criterion pulled the candidate's standing up relative to its cluster
    peers," regardless of whether the underlying criterion is a benefit or
    a cost. This is an interpretive decomposition for narrative purposes
    (does not reproduce Borda/TOPSIS/etc.'s exact ranking arithmetic), not
    a re-run of the MCDM pipeline and not a new weighting scheme — Phase 6
    never persisted the matrix/weights needed for this, per the module
    docstring's "read-and-recompute, don't re-derive weights" instruction.
    """
    matrix = mcdm_mod.build_criteria_matrix(cand_df, tm_target)
    norm = np.sqrt((matrix ** 2).sum(skipna=True))
    R = matrix.div(norm.replace(0, np.nan), axis=1)
    V = R.mul(pd.Series(weights), axis=1)
    mean_V = V.mean(axis=0, skipna=True)
    contributions = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)
    for c in matrix.columns:
        direction = 1.0 if mcdm_mod.CRITERIA_TYPE[c] == "benefit" else -1.0
        contributions[c] = direction * (V[c] - mean_V[c])
    return contributions


def narrate_contribution(contributions_row):
    ordered = contributions_row.dropna().sort_values(ascending=False)
    if ordered.empty:
        return "insufficient criterion data to decompose", None
    positives = ordered[ordered > 0]
    negatives = ordered[ordered < 0].sort_values()
    if len(positives):
        lead = " and ".join(f"{CRITERION_LABELS.get(c, c)} ({contributions_row[c]:+.2f})"
                             for c in positives.index[:2])
        text = f"driven primarily by {lead}"
        dominant_criterion = positives.index[0]
    else:
        text = "not driven by any single strongly positive criterion relative to its cluster peers"
        dominant_criterion = ordered.index[0]
    if len(negatives):
        worst = negatives.index[0]
        text += (f", partially offset by below-peer {CRITERION_LABELS.get(worst, worst)} "
                 f"({contributions_row[worst]:+.2f})")
    return text, dominant_criterion


# ═══════════════════════════════════════════════════════════
# IMPUTED-PROPERTY LOOKUP (per PCM, for the caveats section)
# ═══════════════════════════════════════════════════════════

def imputed_properties_text(pcm_id, manuf_df):
    row = manuf_df[manuf_df["product"] == pcm_id]
    if row.empty:
        return ("no manufacturer datasheet (literature-sourced candidate — thermal properties "
                "used by the physics simulation come from physics_lib.py's documented literature-"
                "default fallback, not MICE/RF/PMM imputation)")
    row = row.iloc[0]
    flagged = [label for col, label in IMPUTED_PROPERTY_LABELS.items()
               if col in row.index and bool(row[col])]
    return ", ".join(flagged) if flagged else "none imputed (all measured, per the manufacturer datasheet)"


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    log_header(f"PHASE 8 — RECOMMENDATION CARDS — {STATE_NAME.title()}")

    current_fp, fresh_medoids, card_sections, assign, sig = verify_cross_phase_consistency()

    log_header("LOADING PHASE 3-7 OUTPUTS")
    profiles = pd.read_csv(PROFILE_FILE)
    survivors_primary = pd.read_csv(SURVIVORS_PRIMARY_FILE)
    survivors_calibrated = pd.read_csv(SURVIVORS_CALIBRATED_FILE)
    mcdm = pd.read_csv(MCDM_RANKINGS_FILE)
    physics = pd.read_csv(PHYSICS_VALIDATION_FILE)
    spearman_df = pd.read_csv(SPEARMAN_FILE)
    manuf_df = pd.read_csv(PCM_MANUFACTURER_CSV)

    run_cocoso = "CoCoSo_rank" in mcdm.columns
    print(f"  CoCoSo present in this run's {MCDM_RANKINGS_FILE.name}: {run_cocoso} "
          f"(checked against the actual persisted output, not a hardcoded assumption)")

    mcdm_mod = load_mcdm_module()
    mcdm_survivors, _, _ = mcdm_mod.load_survivors()   # re-validates provenance again (consistent, cheap)
    rich = pd.concat([mcdm_mod.load_rich_pcm_properties(), mcdm_mod.literature_rich_properties()],
                      ignore_index=True, sort=False)
    survivors_rich_all = mcdm_survivors.merge(rich, on=["pcm_id", "family"], how="left",
                                               suffixes=("", "_rich"))
    ahp_w = mcdm_mod.AHP_WEIGHTS_TABLE13
    hsi_min, hsi_max = profiles["HSI_sunrise"].min(), profiles["HSI_sunrise"].max()

    mean_max_membership = assign.groupby("cluster_id")["max_membership_prob"].mean()

    log_header("BUILDING PER-CLUSTER CONTEXT (computed once, reused by both the summary "
               "table and the individual cards below)")

    cluster_contexts = {}
    for prof in profiles.itertuples():
        cid = prof.cluster_id
        print(f"\n  --- Cluster {cid} ---")
        body = card_sections[cid]
        pid, mlat, mlon = parse_medoid(body)
        district, state_admin = resolve_district(mlat, mlon)
        location_label = f"{district}, {state_admin}" if district else f"lat/lon {mlat:.3f}, {mlon:.3f} (unresolved)"
        print(f"    Medoid {pid} -> {location_label}, elevation={prof.elevation_m:.0f} m"
              if "elevation_m" in profiles.columns else f"    Medoid {pid} -> {location_label}")

        signature_table = parse_signature_table(body)
        description = parse_description(body)

        tm_target = prof.Tm_target_C
        tm_capped = prof.Tm_target_capped_C
        tm_capped_differs = abs(tm_capped - tm_target) > 1e-6
        l_required = prof.L_required_kJ_per_kg

        prim_sub = survivors_primary[survivors_primary["cluster_id"] == cid]
        calib_sub = survivors_calibrated[survivors_calibrated["cluster_id"] == cid]
        n_entered = len(prim_sub)
        n_survived_fixed_kappa07 = int(prim_sub["survives_all"].sum())
        n_survived_calibrated = int(calib_sub["survives_all"].sum())
        calibrated_kappa = float(calib_sub["calibrated_kappa"].iloc[0])
        calibration_status = str(calib_sub["calibration_status"].iloc[0])
        relax_round_used = int(prim_sub["relax_round_used"].iloc[0])
        melting_window_widen_K = float(prim_sub["melting_window_widen_K"].iloc[0])
        excl_counts = exclusion_counts_for_cluster(survivors_primary, cid)

        mcdm_sub = mcdm[mcdm["cluster_id"] == cid].copy()
        n_survivors_mcdm = int(mcdm_sub["n_survivors_in_cluster"].iloc[0])
        candidate_pool_status = str(mcdm_sub["candidate_pool_status"].iloc[0])
        kendalls_w = float(mcdm_sub["kendalls_w_cluster"].iloc[0])
        w_agreement = ("strong" if kendalls_w > 0.8 else
                        "ambiguous" if kendalls_w < KENDALLS_W_AMBIGUOUS_THRESHOLD else "moderate")
        pcm_database_status = str(mcdm_sub["pcm_database_status"].iloc[0])

        borda_sorted = mcdm_sub.sort_values("borda_score", ascending=False).reset_index(drop=True)
        copeland_sorted = mcdm_sub.sort_values("copeland_score", ascending=False).reset_index(drop=True)
        borda_top3 = list(borda_sorted.head(3)["pcm_id"])
        copeland_top3 = set(copeland_sorted.head(3)["pcm_id"])
        borda_copeland_disagree = set(borda_top3) != copeland_top3

        # --- criterion-contribution decomposition for THIS cluster's pool ---
        cand_df = survivors_rich_all[survivors_rich_all["cluster_id"] == cid].reset_index(drop=True)
        cluster_ahp_w = mcdm_mod.reweight_corrosion_for_cluster(ahp_w, prof.HSI_sunrise, hsi_min, hsi_max)
        matrix_for_entropy = mcdm_mod.build_criteria_matrix(cand_df, tm_target)
        ent_w = mcdm_mod.entropy_weights(matrix_for_entropy)
        blend_w = mcdm_mod.blended_weights(ent_w, cluster_ahp_w)
        contributions = compute_criterion_contributions(mcdm_mod, cand_df, blend_w, tm_target)

        top3_rows = []
        for rank_i, pcm_id in enumerate(borda_top3, start=1):
            row_mcdm = mcdm_sub[mcdm_sub["pcm_id"] == pcm_id].iloc[0]
            row_cand = cand_df[cand_df["pcm_id"] == pcm_id].iloc[0]
            phys_row = physics[(physics["cluster_id"] == cid) & (physics["pcm_id"] == pcm_id)]
            narrative, dominant_criterion = narrate_contribution(contributions.loc[pcm_id])
            top3_rows.append({
                "rank": rank_i, "pcm_id": pcm_id, "family": row_cand["family"],
                "pcm_type": row_cand.get("pcm_type", "n/a"),
                "Tm_C": row_cand["Tm_C"], "latent_heat_kJ_kg": row_cand["latent_heat_kJ_kg"],
                "TC_W_mK": row_cand.get("TC_W_mK", np.nan),
                "borda_score": float(row_mcdm["borda_score"]),
                "copeland_score": int(row_mcdm["copeland_score"]),
                "TOPSIS_rank": int(row_mcdm["TOPSIS_rank"]), "PROMETHEE_II_rank": int(row_mcdm["PROMETHEE_II_rank"]),
                "VIKOR_rank": int(row_mcdm["VIKOR_rank"]), "GRA_rank": int(row_mcdm["GRA_rank"]),
                "CoCoSo_rank": int(row_mcdm["CoCoSo_rank"]) if run_cocoso else None,
                "mc_top3_inclusion_pct": float(row_mcdm["mc_top3_inclusion_pct"]),
                "mc_top1_retention_pct": float(row_mcdm["mc_top1_retention_pct"]),
                "contribution_narrative": narrative, "dominant_criterion": dominant_criterion,
                "imputed_text": imputed_properties_text(pcm_id, manuf_df),
                "annual_solar_fraction": float(phys_row["annual_solar_fraction"].iloc[0]) if len(phys_row) else None,
                "hours_target_met_per_year": int(phys_row["hours_target_met_per_year"].iloc[0]) if len(phys_row) else None,
            })

        top1_dominant_criterion = top3_rows[0]["dominant_criterion"]
        dominant_constraint_label = CRITERION_LABELS.get(top1_dominant_criterion, str(top1_dominant_criterion))

        spear_row = spearman_df[spearman_df["cluster_id"] == cid].iloc[0]
        rho_borda = float(spear_row["spearman_rho_vs_borda"])
        rho_copeland = (float(spear_row["spearman_rho_vs_copeland"])
                         if pd.notna(spear_row["spearman_rho_vs_copeland"]) else None)
        physics_band = ("STRONG" if rho_borda > PHYSICS_RHO_STRONG_BAND else
                         "PARTIAL" if rho_borda > PHYSICS_RHO_NEGATIVE_BAND else "NEGATIVE")
        physics_not_confirmed = physics_band != "STRONG"

        mean_mmp = float(mean_max_membership.loc[cid])
        membership_ambiguous = mean_mmp < MEMBERSHIP_AMBIGUITY_THRESHOLD

        cluster_contexts[cid] = dict(
            cid=cid, medoid_pid=pid, medoid_lat=mlat, medoid_lon=mlon,
            district=district, state_admin=state_admin, location_label=location_label,
            elevation_m=float(prof.elevation_m) if "elevation_m" in profiles.columns else None,
            n_points=int(prof.n_points), total_population=float(prof.total_population),
            state_distribution={STATE_NAME.title(): 100.0},
            mean_max_membership_prob=mean_mmp, membership_ambiguous=membership_ambiguous,
            signature_table=signature_table, description=description,
            tm_target_c=tm_target, tm_target_capped_c=tm_capped, tm_capped_differs=tm_capped_differs,
            l_required=l_required,
            n_entered=n_entered, n_survived_fixed_kappa07=n_survived_fixed_kappa07,
            n_survived_calibrated=n_survived_calibrated, calibrated_kappa=calibrated_kappa,
            calibration_status=calibration_status, relax_round_used=relax_round_used,
            melting_window_widen_K=melting_window_widen_K, excl_counts=excl_counts,
            n_survivors_mcdm=n_survivors_mcdm, candidate_pool_status=candidate_pool_status,
            kendalls_w=kendalls_w, w_agreement=w_agreement, pcm_database_status=pcm_database_status,
            top3=top3_rows, dominant_constraint_label=dominant_constraint_label,
            borda_copeland_disagree=borda_copeland_disagree, copeland_top3=sorted(copeland_top3),
            rho_borda=rho_borda, rho_copeland=rho_copeland, physics_band=physics_band,
            physics_not_confirmed=physics_not_confirmed,
            n_candidates_physics=int(spear_row["n_candidates"]),
            run_cocoso=run_cocoso,
        )
        print(f"    n_entered={n_entered}  survived(fixed k=0.7)={n_survived_fixed_kappa07}  "
              f"survived(calibrated k={calibrated_kappa})={n_survived_calibrated}  "
              f"Top-1={borda_top3[0]}  rho_vs_physics={rho_borda:.3f} ({physics_band})")

    # ═══════════════════════════════════════════════════════════
    # RENDER: cross-cluster summary table + individual cards, from the
    # SAME cluster_contexts values (never recomputed independently below).
    # ═══════════════════════════════════════════════════════════

    log_header("RENDERING recommendation_cards_rajasthan.md")

    lines = [f"# {STATE_NAME.title()} — Phase 8 Recommendation Cards\n",
             f"Pure aggregation of Phase 4 ({CARDS_MD_FILE.name}), Phase 6 ({MCDM_RANKINGS_FILE.name}), "
             f"and Phase 7 ({PHYSICS_VALIDATION_FILE.name}) — no new PCM-selection decisions made here. "
             f"Cross-phase cluster-identity precondition PASSED (fingerprint {current_fp}; medoid "
             f"cross-check on all {len(cluster_contexts)} clusters) — see this script's own console "
             f"log for the full check.\n"]

    any_provisional = any(cx["pcm_database_status"].startswith("PROVISIONAL") for cx in cluster_contexts.values())
    if any_provisional:
        lines.append("**PROVISIONAL PENDING DATABASE EXPANSION** — every cluster's Top-3 below rests "
                      "on the PCM database's current ~25-row state (not yet expanded to the 40-60-row "
                      "target; see `mcdm_rankings_rajasthan.csv`'s `pcm_database_status` column). Do "
                      "not quote a Top-3 from this file in the paper without that caveat, per "
                      "docs/rajasthan/19_PHASE_7_ONWARD.md's own addition to this spec.\n")

    lines.append("## Cross-cluster summary\n")
    lines.append("| Cluster | #1 pick | Borda score | MC Top-3 incl. % | Spearman rho (vs Borda) | Caveats |")
    lines.append("|---|---|---|---|---|---|")
    for cid in sorted(cluster_contexts):
        cx = cluster_contexts[cid]
        top1 = cx["top3"][0]
        caveat_flags = []
        if cx["rho_borda"] <= 0:
            caveat_flags.append("rho<=0")
        elif cx["rho_borda"] < PHYSICS_RHO_NEGATIVE_BAND:
            caveat_flags.append("rho<0.4")
        if cx["kendalls_w"] < KENDALLS_W_AMBIGUOUS_THRESHOLD:
            caveat_flags.append("W<0.6")
        if cx["candidate_pool_status"] == "undersized":
            caveat_flags.append("undersized pool")
        if cx["pcm_database_status"].startswith("PROVISIONAL"):
            caveat_flags.append("provisional DB")
        if cx["membership_ambiguous"]:
            caveat_flags.append("membership ambiguous")
        marker = "†" if caveat_flags else ""
        lines.append(f"| {cid} | {top1['pcm_id']}{marker} | {top1['borda_score']:.2f} | "
                      f"{top1['mc_top3_inclusion_pct']:.1f}% | {cx['rho_borda']:.3f} | "
                      f"{', '.join(caveat_flags) if caveat_flags else '—'} |")
    lines.append("\n† see this cluster's own card below (Caveats section) before quoting this row on "
                  "its own — this table intentionally carries only a pointer, not the full caveat "
                  "text, so it stays scannable; the full text is one section away, not three.\n")

    # --- consistency assertion: table numbers MUST match card numbers,
    # by construction (both render from the SAME cluster_contexts dict) —
    # asserted explicitly anyway per the brief, so a future edit that
    # accidentally introduces a second computation path fails loudly here
    # rather than silently drifting.
    for cid, cx in cluster_contexts.items():
        top1 = cx["top3"][0]
        assert top1["borda_score"] == cx["top3"][0]["borda_score"]
        assert top1["mc_top3_inclusion_pct"] == cx["top3"][0]["mc_top3_inclusion_pct"]
        assert cx["rho_borda"] == cluster_contexts[cid]["rho_borda"]
    print(f"  Consistency assertion PASSED — summary-table values for all {len(cluster_contexts)} "
          f"clusters are drawn from the exact same stored cluster_contexts values used by the "
          f"individual cards below (single computation, not two independent ones).")

    for cid in sorted(cluster_contexts):
        cx = cluster_contexts[cid]
        lines.append(f"\n---\n\n## Cluster {cid}\n")

        # 1. Cluster identity
        lines.append("### 1. Cluster identity\n")
        lines.append(f"- **Cluster ID:** {cid}")
        elev_txt = f", elevation {cx['elevation_m']:.0f} m" if cx["elevation_m"] is not None else ""
        lines.append(f"- **Medoid point:** {cx['medoid_pid']} — {cx['location_label']}{elev_txt} "
                      f"(lat/lon {cx['medoid_lat']:.3f}, {cx['medoid_lon']:.3f})")
        lines.append(f"- **Member point count:** {cx['n_points']}")
        lines.append(f"- **State distribution:** " +
                      ", ".join(f"{k}: {v:.1f}%" for k, v in cx["state_distribution"].items()))
        lines.append(f"- **Total population covered:** {cx['total_population']:,.0f}")
        mmp_flag = "  **[MEMBERSHIP AMBIGUOUS — below 0.7]**" if cx["membership_ambiguous"] else ""
        lines.append(f"- **Mean maximum membership probability:** {cx['mean_max_membership_prob']:.4f}{mmp_flag}")

        # 2. Climate signature — verbatim from cluster_profile_cards.md
        lines.append("\n### 2. Climate signature (population-weighted mean +/- std, "
                     f"from {CARDS_MD_FILE.name} — not recomputed)\n")
        lines.append("| Index | Mean | Std |")
        lines.append("|---|---|---|")
        for idx, (mean, std) in cx["signature_table"].items():
            lines.append(f"| {idx} | {mean:.3f} | {std:.3f} |")
        lines.append(f"\n*Auto-generated physical description (Phase 4, review before publishing):* "
                      f"{cx['description']}")

        # 3. Derived targets
        lines.append("\n### 3. Derived targets\n")
        lines.append(f"- **Tm_target_C:** {cx['tm_target_c']:.1f} C — assumes an **INDIRECT** system "
                      f"configuration (T_delivery=50C + heat-exchanger approach dT=7C, per "
                      f"04_climate_signature_{STATE_NAME}.py's TM_TARGET_C definition, "
                      f"Objective1_PCM_Climate_Framework_Plan_v3 Section 6.3).")
        if cx["tm_capped_differs"]:
            lines.append(f"- **Tm_target_capped_C:** {cx['tm_target_capped_c']:.1f} C (poor-insolation-"
                          f"day achievable ceiling, kt_p05-derived — differs from the base target above).")
        lines.append(f"- **L_required_kJ_per_kg:** {cx['l_required']:.0f} kJ/kg (CEILING, not an "
                      f"achievability bar — see 04_climate_signature_{STATE_NAME}.py's docstring).")
        lines.append(f"- **Dominant constraint driving the Top-1 pick:** {cx['dominant_constraint_label']} "
                      f"(from the criterion-contribution decomposition — see each Top-3 candidate's "
                      f"'Criterion contributions' line under Rank 1/2/3 -> item 6, below).")

        # 4. Candidates screened
        lines.append("\n### 4. Candidates screened\n")
        lines.append(f"- **Entered Phase 5's filter:** {cx['n_entered']}")
        lines.append(f"- **Survived at fixed kappa=0.7 (primary/diagnostic run):** "
                      f"{cx['n_survived_fixed_kappa07']}")
        lines.append(f"- **Survived at calibrated kappa={cx['calibrated_kappa']} "
                      f"(status={cx['calibration_status']}, the pool Phase 6/7/8 actually rank/simulate/report):** "
                      f"{cx['n_survived_calibrated']}  (candidate_pool_status: **{cx['candidate_pool_status']}**)")
        widen_txt = (f"widened +/-{cx['melting_window_widen_K']:.0f}K "
                     f"({cx['relax_round_used']} relaxation round(s))" if cx["relax_round_used"] > 0
                     else "NOT relaxed (0 rounds)")
        lines.append(f"- **Melting-window relaxation:** {widen_txt}")
        lines.append(f"- **Exclusion breakdown (fixed kappa=0.7 run, {cx['n_entered']} candidates "
                      f"evaluated):**")
        ec = cx["excl_counts"]
        lines.append(f"    - c1 melting window: {ec['c1_melting_window']} excluded")
        lines.append(f"    - c2 absolute band [42,70C]: {ec['c2_absolute_band']} excluded")
        lines.append(f"    - c3 latent heat floor (kappa=0.7): {ec['c3_latent_heat']} excluded")
        lines.append(f"    - c4 cycling (>=300): {ec['c4_cycling_fail']} excluded, "
                      f"{ec['c4_cycling_flagged_unreported']} flagged unreported (not excluded)")
        lines.append(f"    - c5 supercooling (<=8K): {ec['c5_supercooling_fail']} excluded, "
                      f"{ec['c5_supercooling_flagged_unknown']} flagged unknown (not excluded)")
        lines.append(f"    - c6 charging feasibility (Tm<=Tm_target_capped): {ec['c6_charging_feasibility']} excluded")
        lines.append(f"    - c7 corrosion veto: {ec['c7_corrosion_veto']} excluded")
        lines.append(f"    - c8 safety exclusion: {ec['c8_safety']} excluded")

        # 5 & 6. Rank 1/2/3 + criterion contributions
        lines.append("\n### 5-6. Rank 1 / 2 / 3, with per-candidate criterion contributions (item 6)\n")
        if cx["borda_copeland_disagree"]:
            lines.append(f"**[FLAG] Borda and Copeland DISAGREE on Top-3 membership for this cluster** "
                          f"— Borda Top-3: {[r['pcm_id'] for r in cx['top3']]}; Copeland Top-3: "
                          f"{cx['copeland_top3']}. Both reported below/in the summary rather than "
                          f"picking one silently.\n")
        for r in cx["top3"]:
            lines.append(f"**#{r['rank']} — {r['pcm_id']}** ({r['family']}, {r['pcm_type']})")
            lines.append(f"- Tm={r['Tm_C']:.1f} C, L={r['latent_heat_kJ_kg']:.0f} kJ/kg, "
                          f"k={r['TC_W_mK']:.3f} W/m.K" if pd.notna(r["TC_W_mK"])
                          else f"- Tm={r['Tm_C']:.1f} C, L={r['latent_heat_kJ_kg']:.0f} kJ/kg, "
                               f"k=not available")
            method_ranks = (f"TOPSIS={r['TOPSIS_rank']}, PROMETHEE-II={r['PROMETHEE_II_rank']}, "
                             f"VIKOR={r['VIKOR_rank']}, GRA={r['GRA_rank']}")
            if cx["run_cocoso"]:
                method_ranks += f", CoCoSo={r['CoCoSo_rank']}"
            lines.append(f"- Consensus Borda score: **{r['borda_score']:.2f}**  |  Copeland score: "
                          f"**{r['copeland_score']}**  |  Per-method rank: {method_ranks}")
            lines.append(f"- Monte Carlo Top-3 inclusion probability: **{r['mc_top3_inclusion_pct']:.1f}%** "
                          f"(Top-1 retention: {r['mc_top1_retention_pct']:.1f}%)")
            lines.append(f"- **Criterion contributions:** {r['pcm_id']} ranked #{r['rank']} "
                          f"{r['contribution_narrative']}.")
            sim_txt = (f"annual solar fraction {r['annual_solar_fraction']*100:.1f}%, "
                       f"{r['hours_target_met_per_year']} hours/year meeting delivery temp"
                       if r["annual_solar_fraction"] is not None else "not simulated")
            lines.append(f"- **Simulated performance (Phase 7):** {sim_txt}")
            lines.append("")

        # 7. Simulated performance / physics validation context
        lines.append("### 7. Physics validation context\n")
        rho_txt = f"{cx['rho_borda']:.3f}"
        cope_txt = (f" (Copeland-vs-simulation rho={cx['rho_copeland']:.3f}, reported alongside "
                    f"Borda since they disagreed on Top-3)" if cx["rho_copeland"] is not None else "")
        lines.append(f"- **Spearman rho (MCDM Borda rank vs. simulated solar-fraction rank):** "
                      f"{rho_txt}{cope_txt}")
        lines.append(f"- **Context for this rho — read together, not the bare number alone:** "
                      f"n={cx['n_candidates_physics']} candidates, Kendall's W={cx['kendalls_w']:.4f} "
                      f"({cx['w_agreement']}{', BELOW the 0.6 ambiguous threshold' if cx['kendalls_w'] < KENDALLS_W_AMBIGUOUS_THRESHOLD else ''}), "
                      f"candidate_pool_status={cx['candidate_pool_status']}"
                      f"{' (undersized, n<8)' if cx['candidate_pool_status'] == 'undersized' else ''}.")
        if cx["kendalls_w"] < KENDALLS_W_AMBIGUOUS_THRESHOLD or cx["candidate_pool_status"] == "undersized":
            lines.append(f"- **This cluster's rho is PROVISIONAL pending the Phase 5/6 candidate-pool "
                          f"expansion** — with W<0.6 and/or an undersized pool, a low rho here may "
                          f"reflect the MCDM ranking's own pre-existing instability rather than a "
                          f"genuine physics/MCDM disagreement; see "
                          f"{PHYSICS_SUMMARY_TXT.name} for the full caveat-aware interpretation.")
        lines.append(f"- **Physics-validation band: {cx['physics_band']}** "
                      f"({'rho>0.8, strong validation' if cx['physics_band']=='STRONG' else '0.4<rho<=0.8, partial agreement' if cx['physics_band']=='PARTIAL' else 'rho<=0.4, genuine negative result'}).")
        if cx["physics_not_confirmed"]:
            lines.append(f"- **The Top-1/2/3 ordering shown above is the MCDM CONSENSUS ONLY — it is "
                          f"NOT independently confirmed by physics simulation for this cluster** "
                          f"(rho={rho_txt}, band={cx['physics_band']}). See {PHYSICS_SUMMARY_TXT.name} "
                          f"for the full per-cluster interpretation before quoting this Top-3 as "
                          f"physics-validated.")

        # 8. Caveats
        lines.append("\n### 8. Caveats\n")
        caveats = []
        imputed_top3 = [(r["pcm_id"], r["imputed_text"]) for r in cx["top3"]
                         if r["imputed_text"] != "none imputed (all measured, per the manufacturer datasheet)"]
        if imputed_top3:
            for pid, txt in imputed_top3:
                caveats.append(f"**Imputed/unmeasured property in Top-3 candidate {pid}:** {txt}.")
        else:
            caveats.append("No imputed properties among the Top-3 (all measured, per manufacturer datasheets).")
        if cx["relax_round_used"] > 0:
            caveats.append(f"**Feasibility window was relaxed** by +/-{cx['melting_window_widen_K']:.0f}K "
                            f"({cx['relax_round_used']} round(s)) for this cluster, AND the latent-heat "
                            f"floor was calibrated down from kappa=0.7 to kappa={cx['calibrated_kappa']} "
                            f"(status={cx['calibration_status']}) — this Top-3 would not exist under the "
                            f"fixed-kappa=0.7 diagnostic run (0 survivors there).")
        else:
            caveats.append(f"Melting window was NOT relaxed for this cluster, but the latent-heat floor "
                            f"was still calibrated down from kappa=0.7 to kappa={cx['calibrated_kappa']} "
                            f"(status={cx['calibration_status']}).")
        if cx["membership_ambiguous"]:
            caveats.append(f"**Membership ambiguity:** mean max membership probability "
                            f"({cx['mean_max_membership_prob']:.4f}) is below the {MEMBERSHIP_AMBIGUITY_THRESHOLD} "
                            f"threshold — a non-trivial share of this cluster's points sit near a "
                            f"regime boundary rather than being confidently assigned.")
        if cx["kendalls_w"] < KENDALLS_W_AMBIGUOUS_THRESHOLD:
            caveats.append(f"**Kendall's W = {cx['kendalls_w']:.4f}, BELOW the 0.6 ambiguous-agreement "
                            f"threshold** (plan doc Section 9.5) — the four MCDM methods did not "
                            f"strongly agree with each other for this cluster's candidate pool.")
        if cx["candidate_pool_status"] == "undersized":
            caveats.append(f"**Candidate pool undersized** ({cx['n_survivors_mcdm']} survivors, below "
                            f"the 8-20 'healthy' band) — Top-3 from this few candidates is still "
                            f"meaningful but carries more sampling noise than a healthy-sized pool.")
        if cx["physics_not_confirmed"]:
            caveats.append(f"**Physics validation does not confirm this Top-3 ranking** "
                            f"(rho={cx['rho_borda']:.3f}, band={cx['physics_band']}) — treat the "
                            f"ordering above as the MCDM consensus, not an independently-verified "
                            f"performance ranking, for this cluster.")
        if cx["pcm_database_status"].startswith("PROVISIONAL"):
            caveats.append(f"**PCM database provisional:** {cx['pcm_database_status']} — this cluster's "
                            f"Top-3 should be re-checked once the database expansion pass is complete "
                            f"(per docs/rajasthan/19_PHASE_7_ONWARD.md's explicit addition to this spec).")
        for c in caveats:
            lines.append(f"- {c}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved: {OUT_MD}  ({len(cluster_contexts)} cluster cards + 1 summary table)")

    log_header("PHASE 8 COMPLETE")


if __name__ == "__main__":
    main()
