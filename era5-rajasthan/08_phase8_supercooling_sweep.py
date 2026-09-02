"""
PHASE 8 — SUPERCOOLING PENALTY SENSITIVITY SWEEP
=============================================================================
Extends Phase 7 (09_physics_validation_rajasthan.py) to test whether
accounting for supercooling brings the physics ranking closer to the MCDM
ranking. Runs the full Phase 7 experiment across 4 sensitivity-sweep values
of SUPERCOOLING_PENALTY_K: [0.0, 0.1, 0.2, 0.3].

For each k value:
  1. Confirm energy conservation and draw-profile self-tests pass
  2. Run calibration against the 3 medoids
  3. Run full Phase 7 experiment (all survivors x all clusters)
  4. Compute per-cluster Spearman rho

At the end, report a 3×4 table: (cluster_id × k_value) → rho,
plus honest interpretation of whether rho improved and why.

HOW TO RUN:
  python 08_phase8_supercooling_sweep.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

import physics_lib as pl
from config import PROCESSED_DIR, OUTPUTS_DIR, RAW_POWER_DIR, BASE_DIR, ensure_data_dirs
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match

ensure_data_dirs()

STATE_NAME = "rajasthan"

MCDM_RANKINGS_FILE = PROCESSED_DIR / f"mcdm_rankings_{STATE_NAME}.csv"
SURVIVORS_FILE = PROCESSED_DIR / f"feasibility_survivors_{STATE_NAME}_kappa_calibrated.csv"
PROFILE_FILE = PROCESSED_DIR / f"cluster_profiles_{STATE_NAME}.csv"
ASSIGN_A_FILE = PROCESSED_DIR / f"cluster_assignments_{STATE_NAME}_levelA.csv"
SIGNATURE_FILE = PROCESSED_DIR / f"climate_signature_{STATE_NAME}.csv"
PCM_MANUFACTURER_CSV = BASE_DIR.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"

OUT_SWEEP_RESULTS = PROCESSED_DIR / f"phase8_supercooling_sweep_{STATE_NAME}.csv"
OUT_SWEEP_SUMMARY = BASE_DIR / f"phase8_supercooling_sweep_summary_{STATE_NAME}.txt"

BENCHMARK_SF_LOW, BENCHMARK_SF_HIGH = 0.54, 0.84
CALIBRATION_PCM = {"pcm_id": "RT47", "Tm_C": 46.0, "latent_heat_kJ_kg": 160.0,
                    "density_solid_kg_m3": 880.0, "Cp_solid_JkgK": 2000.0,
                    "Cp_liquid_JkgK": 2000.0, "TC_solid_WmK": 0.200}

# Sensitivity sweep k values: 0.0 (baseline), 0.1, 0.2, 0.3
K_VALUES = [0.0, 0.1, 0.2, 0.3]


def log_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TESTS (run once at the beginning, k-independent)
# ═══════════════════════════════════════════════════════════════════════════

def run_self_tests():
    log_header("[SELF-TESTS] Energy conservation & draw-profile validation")

    print("\n  Energy conservation (constant solar, no draw) ...")
    r1 = pl.self_test_energy_conservation()
    print(f"    residual = {r1['energy_balance_residual_fraction']:.3e}  ->  "
          f"{'PASS' if r1['pass'] else 'FAIL'}")
    if not r1["pass"]:
        raise SystemExit("Energy-conservation self-test FAILED with supercooling penalty enabled.")

    print("\n  Draw-profile volume integration (300 kg/day expected) ...")
    r2 = pl.self_test_draw_profile_integration()
    print(f"    daily total = {r2['daily_total_kg']:.6f} kg  ->  "
          f"{'PASS' if r2['pass'] else 'FAIL'}")
    if not r2["pass"]:
        raise SystemExit("Draw-profile self-test FAILED.")

    print("\n  Both self-tests PASS.")
    return r1, r2


# ═══════════════════════════════════════════════════════════════════════════
# LOAD INPUTS (k-independent)
# ═══════════════════════════════════════════════════════════════════════════

def load_inputs():
    log_header("[LOADING INPUTS]")

    for f in (MCDM_RANKINGS_FILE, SURVIVORS_FILE, PROFILE_FILE, ASSIGN_A_FILE, SIGNATURE_FILE, PCM_MANUFACTURER_CSV):
        if not f.exists():
            raise SystemExit(f"ERROR: required input not found: {f}")

    mcdm = pd.read_csv(MCDM_RANKINGS_FILE)
    survivors = pd.read_csv(SURVIVORS_FILE)
    survivors = survivors[survivors["survives_all"] == True]   # noqa: E712
    assign = pd.read_csv(ASSIGN_A_FILE)
    sig = pd.read_csv(SIGNATURE_FILE)
    sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)
    manuf = pd.read_csv(PCM_MANUFACTURER_CSV)

    print(f"  mcdm_rankings: {len(mcdm)} rows, {mcdm['cluster_id'].nunique()} clusters")
    print(f"  feasibility survivors: {len(survivors)} rows")

    current_profile_fp_id = fingerprint_id(file_fingerprint(PROFILE_FILE))
    assert_fingerprint_match(current_profile_fp_id, survivors, PROFILE_FILE.name, SURVIVORS_FILE.name)
    assert_fingerprint_match(current_profile_fp_id, mcdm, PROFILE_FILE.name, MCDM_RANKINGS_FILE.name)
    print(f"  Provenance check PASSED (fingerprint {current_profile_fp_id}).")

    prop_cols = {"product": "pcm_id", "density_solid": "density_solid_kg_m3",
                 "density_liquid": "density_liquid_kg_m3", "Cp_solid": "Cp_solid_kJkgK",
                 "Cp_liquid": "Cp_liquid_kJkgK", "TC_solid": "TC_solid_WmK", "TC_liquid": "TC_liquid_WmK"}
    manuf_props = manuf[list(prop_cols.keys())].rename(columns=prop_cols)
    manuf_props["Cp_solid_JkgK"] = manuf_props["Cp_solid_kJkgK"] * 1000.0
    manuf_props["Cp_liquid_JkgK"] = manuf_props["Cp_liquid_kJkgK"] * 1000.0

    pcm_intrinsic = survivors.drop_duplicates(subset="pcm_id")[
        ["pcm_id", "family", "pcm_type", "Tm_C", "latent_heat_kJ_kg", "source", "supercooling_K"]]
    pcm_intrinsic = pcm_intrinsic.merge(manuf_props, on="pcm_id", how="left")
    pcm_intrinsic["any_thermal_property_imputed"] = pcm_intrinsic["density_solid_kg_m3"].isna()

    pcm_table = mcdm[["cluster_id", "pcm_id"]].merge(pcm_intrinsic, on="pcm_id", how="left")

    return mcdm, pcm_table, assign, sig


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION (k-dependent, but run once with k=0.0, k=0.1, etc.)
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration_at_k(assign, sig, k_value):
    """Run calibration for a specific k value. Return medoids and weather cache."""
    z_cols = [c for c in sig.columns if c.endswith("_z")]
    cluster_ids = sorted(assign["cluster_id"].unique())
    medoids = {cid: pl.find_medoid(cid, assign, sig, z_cols) for cid in cluster_ids}

    calib_rows = []
    weather_cache = {}
    for cid, pid in medoids.items():
        df = pl.load_nasapower_hourly_year(pid, RAW_POWER_DIR)
        weather_cache[pid] = df
        r = pl.simulate_pcm_swh_year(df, CALIBRATION_PCM)
        sf = r["annual_solar_fraction"]
        in_band = BENCHMARK_SF_LOW <= sf <= BENCHMARK_SF_HIGH
        calib_rows.append({"cluster_id": cid, "medoid_point": pid,
                           "annual_solar_fraction": sf, "in_band_54_84pct": in_band})

    calib_df = pd.DataFrame(calib_rows)
    pct_in_band = calib_df["in_band_54_84pct"].mean() * 100
    return calib_df, medoids, weather_cache, pct_in_band


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT (k-dependent)
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment_at_k(pcm_table, medoids, weather_cache):
    """Run full Phase 7 experiment (all survivors x all clusters) for current k."""
    results = []
    for cid, pid in medoids.items():
        cluster_pcms = pcm_table[pcm_table["cluster_id"] == cid]
        df = weather_cache[pid]
        for _, row in cluster_pcms.iterrows():
            r = pl.simulate_pcm_swh_year(df, row)
            results.append({
                "cluster_id": cid, "pcm_id": row["pcm_id"],
                "annual_solar_fraction": r["annual_solar_fraction"],
                "hours_target_met_per_year": r["hours_target_met_per_year"],
                "mean_melt_fraction": r["mean_melt_fraction"],
            })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# SPEARMAN RHO CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_spearman_at_k(mcdm, results):
    """Compute Spearman rho for each cluster. Return rho values."""
    merged = results.merge(mcdm, on=["cluster_id", "pcm_id"], how="left")
    rho_by_cluster = {}

    for cid in sorted(merged["cluster_id"].unique()):
        sub = merged[merged["cluster_id"] == cid].copy()
        rho, p_val = spearmanr(sub["borda_score"], sub["annual_solar_fraction"])
        rho_by_cluster[cid] = rho

    return rho_by_cluster


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SWEEP LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log_header("PHASE 8: SUPERCOOLING PENALTY SENSITIVITY SWEEP")

    print(f"\n  Sweep range: k = {K_VALUES}")
    print(f"  Penalty formula: h_p_eff = h_p × (1 - k × supercooling_K / 10)")
    print(f"  supercooling_K = Tm_C - Tm_freezing_C (K, from Phase 5 feasibility filter)")

    # Run self-tests once (k-independent)
    run_self_tests()

    # Load inputs once (k-independent)
    mcdm, pcm_table, assign, sig = load_inputs()

    # Sweep across k values
    sweep_results = []
    for k in K_VALUES:
        log_header(f"[SWEEP] k = {k}")

        # Set the penalty parameter
        pl.SUPERCOOLING_PENALTY_K = k
        print(f"\n  SUPERCOOLING_PENALTY_K set to {k}")

        # Calibration
        print(f"\n  Running calibration at k={k} ...")
        calib_df, medoids, weather_cache, pct_in_band = run_calibration_at_k(assign, sig, k)
        print(f"    {pct_in_band:.0f}% of medoids in-band (target ~100%)")

        if pct_in_band < 100:
            print(f"    [WARNING] Not all medoids calibrated in-band at k={k}")

        # Experiment
        print(f"\n  Running Phase 7 experiment at k={k} ({len(pcm_table)} candidates) ...")
        results = run_experiment_at_k(pcm_table, medoids, weather_cache)
        print(f"    Completed {len(results)} simulations")

        # Compute Spearman rho for each cluster
        rho_dict = compute_spearman_at_k(mcdm, results)

        for cid, rho in rho_dict.items():
            sweep_results.append({
                "k_value": k,
                "cluster_id": cid,
                "spearman_rho": rho,
                "pct_medoids_in_band": pct_in_band,
                "n_candidates": len(pcm_table[pcm_table["cluster_id"] == cid])
            })
            print(f"    Cluster {cid}: rho = {rho:.3f}")

    # Save sweep results
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(OUT_SWEEP_RESULTS, index=False)
    print(f"\n  Saved: {OUT_SWEEP_RESULTS}")

    # Write summary
    write_summary(sweep_df)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY WRITER
# ═══════════════════════════════════════════════════════════════════════════

def write_summary(sweep_df):
    """Write interpretation summary."""
    lines = []
    lines.append("=" * 76)
    lines.append("  PHASE 8 SUPERCOOLING PENALTY SENSITIVITY SWEEP SUMMARY")
    lines.append("=" * 76)
    lines.append("")

    # Table of rho values
    lines.append("SPEARMAN RHO BY CLUSTER AND K VALUE:")
    lines.append("")
    pivot = sweep_df.pivot(index="cluster_id", columns="k_value", values="spearman_rho")
    lines.append(pivot.to_string())
    lines.append("")

    # Original Phase 7 rho values for comparison
    lines.append("ORIGINAL PHASE 7 RHO VALUES (k=0, baseline from earlier run):")
    lines.append("  Cluster 0: -0.385")
    lines.append("  Cluster 1: +0.125")
    lines.append("  Cluster 2: -0.097")
    lines.append("")

    # Analysis
    lines.append("INTERPRETATION:")
    for k in K_VALUES[1:]:  # Skip k=0 (baseline control)
        k_data = sweep_df[sweep_df["k_value"] == k]
        print("\n")
        for cid in sorted(k_data["cluster_id"].unique()):
            rho_k = k_data[k_data["cluster_id"] == cid]["spearman_rho"].values[0]
            rho_baseline = sweep_df[(sweep_df["k_value"] == 0.0) & (sweep_df["cluster_id"] == cid)]["spearman_rho"].values[0]
            delta = rho_k - rho_baseline
            direction = "improved" if delta > 0.05 else ("worsened" if delta < -0.05 else "unchanged")
            lines.append(f"  k={k}, Cluster {cid}: rho={rho_k:.3f} (Δ={delta:+.3f}) — {direction}")

    lines.append("")
    lines.append("CONCLUSION:")
    lines.append("  [To be filled after inspection of the rho table above]")
    lines.append("")

    with open(OUT_SWEEP_SUMMARY, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {OUT_SWEEP_SUMMARY}")


if __name__ == "__main__":
    main()
