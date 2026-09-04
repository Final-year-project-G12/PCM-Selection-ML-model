"""
verify_phase9.py
================
PHASE 9 VERIFICATION SCRIPT (Assam SWH PCM Project)

Automated test suite verifying Phase 9 physical implementation integrity:
1. 6 Automated Thermodynamic Unit Tests (Latent heat melting/freezing, boundary clipping,
   energy balance, direction reversal continuity, supercooling hysteresis).
2. True Cluster Medoid derivation verification.
3. Validation CSV schema and non-MCDM candidate labeling verification.
4. Convergence and energy balance criteria verification (<0.1% error).
5. Sub-hourly timestep sensitivity benchmark verification (dt=300 s vs dt=150 s).
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, PREPROCESSED_DIR
import importlib
phys_val = importlib.import_module("10_physics_validation")
PCMStateNode = phys_val.PCMStateNode
derive_true_medoids = phys_val.derive_true_medoids
ASSIGN_FILE = phys_val.ASSIGN_FILE
SIG_RAW_FILE = phys_val.SIG_RAW_FILE
OUT_RESULTS_CSV = phys_val.OUT_RESULTS_CSV

def log(msg=""):
    print(msg, flush=True)

# -----------------------------------------------------------------------------
# 1. AUTOMATED THERMODYNAMIC UNIT TESTS
# -----------------------------------------------------------------------------
def test_1_latent_energy_absorbed_melting():
    Mp, L, Cp_s, Cp_l, sc, Tm = 50.0, 200000.0, 2000.0, 2200.0, 3.0, 44.0
    node = PCMStateNode({"Tm_C": Tm, "latent_heat_kJ_kg": L/1000.0, "Cp_solid_kJ_kgK": Cp_s/1000.0, "Cp_liquid_kJ_kgK": Cp_l/1000.0, "supercooling_K": sc}, Mp=Mp)
    node.reset_state(T_initial=Tm, f_melt_initial=0.0)
    H_start = node.Hp
    dH_melt = node.H_melt_end - node.H_solid_at_Tm
    node.update_enthalpy(dH_melt)
    assert abs(node.Hp - H_start - dH_melt) < 1e-6, "Melting enthalpy increment mismatch!"
    assert abs(node.f_melt - 1.0) < 1e-6, "Melt fraction should be 1.0!"
    assert node.mode == "LIQUID", "Mode should transition to LIQUID!"
    log("  [PASS] Unit Test 1: Latent energy absorbed during melting verified")

def test_2_latent_energy_released_freezing():
    Mp, L, Cp_s, Cp_l, sc, Tm = 50.0, 200000.0, 2000.0, 2200.0, 3.0, 44.0
    node = PCMStateNode({"Tm_C": Tm, "latent_heat_kJ_kg": L/1000.0, "Cp_solid_kJ_kgK": Cp_s/1000.0, "Cp_liquid_kJ_kgK": Cp_l/1000.0, "supercooling_K": sc}, Mp=Mp)
    node.Hp = node.H_freeze_start
    node.mode = "FREEZING"
    node.f_melt = 1.0
    node.Tp = node.T_freeze
    H_start = node.Hp
    dH_freeze = -(node.H_freeze_start - node.H_freeze_end)
    node.update_enthalpy(dH_freeze)
    assert abs(node.Hp - H_start - dH_freeze) < 1e-6, "Freezing enthalpy decrement mismatch!"
    assert abs(node.f_melt - 0.0) < 1e-6, "Melt fraction should be 0.0!"
    assert node.mode == "SOLID", "Mode should transition to SOLID!"
    log("  [PASS] Unit Test 2: Latent energy released during freezing verified")

def test_3_no_latent_plateau_skipped():
    Mp, L, Cp_s, Cp_l, sc, Tm = 50.0, 200000.0, 2000.0, 2200.0, 3.0, 44.0
    node = PCMStateNode({"Tm_C": Tm, "latent_heat_kJ_kg": L/1000.0, "Cp_solid_kJ_kgK": Cp_s/1000.0, "Cp_liquid_kJ_kgK": Cp_l/1000.0, "supercooling_K": sc}, Mp=Mp)
    node.reset_state(T_initial=10.0)
    H_initial = node.Hp
    H_target = node.H_melt_end + Mp * Cp_l * (55.0 - Tm)
    dH_large = H_target - H_initial
    node.update_enthalpy(dH_large)
    assert abs(node.Hp - (H_initial + dH_large)) < 1e-6, "Enthalpy mismatch on multi-boundary jump!"
    assert abs(node.Tp - 55.0) < 1e-6, f"Expected 55.0 °C, got {node.Tp}"
    assert node.f_melt == 1.0 and node.mode == "LIQUID"
    log("  [PASS] Unit Test 3: No latent plateau skipped under multi-boundary jump")

def test_4_no_energy_created_or_destroyed():
    Mp, L, Cp_s, Cp_l, sc, Tm = 50.0, 200000.0, 2000.0, 2200.0, 3.0, 44.0
    node = PCMStateNode({"Tm_C": Tm, "latent_heat_kJ_kg": L/1000.0, "Cp_solid_kJ_kgK": Cp_s/1000.0, "Cp_liquid_kJ_kgK": Cp_l/1000.0, "supercooling_K": sc}, Mp=Mp)
    node.reset_state(T_initial=20.0)
    H_start = node.Hp
    np.random.seed(12345)
    random_dHs = np.random.uniform(-400000.0, 400000.0, 500)
    sum_dH = 0.0
    for dH in random_dHs:
        node.update_enthalpy(dH)
        sum_dH += dH
    assert abs(node.Hp - (H_start + sum_dH)) < 1e-6, "Strict energy balance violated across random sequence!"
    log("  [PASS] Unit Test 4: Strict energy conservation across 500 random heat steps")

def test_5_direction_reversal_continuity():
    Mp, L, Cp_s, Cp_l, sc, Tm = 50.0, 200000.0, 2000.0, 2200.0, 3.0, 44.0
    node = PCMStateNode({"Tm_C": Tm, "latent_heat_kJ_kg": L/1000.0, "Cp_solid_kJ_kgK": Cp_s/1000.0, "Cp_liquid_kJ_kgK": Cp_l/1000.0, "supercooling_K": sc}, Mp=Mp)
    node.reset_state(T_initial=Tm, f_melt_initial=0.5)
    H_start = node.Hp
    T_start = node.Tp
    node.update_enthalpy(5000.0)
    node.update_enthalpy(-5000.0)
    assert abs(node.Hp - H_start) < 1e-6, "Enthalpy jump on direction reversal!"
    assert abs(node.Tp - T_start) < 1e-6, "Temperature jump on direction reversal!"
    log("  [PASS] Unit Test 5: Reversal immediately after boundary remains continuous")

def test_6_supercooling_hysteresis():
    Mp, L, Cp_s, Cp_l, sc, Tm = 50.0, 200000.0, 2000.0, 2200.0, 3.0, 44.0
    node = PCMStateNode({"Tm_C": Tm, "latent_heat_kJ_kg": L/1000.0, "Cp_solid_kJ_kgK": Cp_s/1000.0, "Cp_liquid_kJ_kgK": Cp_l/1000.0, "supercooling_K": sc}, Mp=Mp)
    node.reset_state(T_initial=50.0)
    dH_cool = node.H_freeze_start - node.Hp
    node.update_enthalpy(dH_cool)
    assert abs(node.Tp - (Tm - sc)) < 1e-6, f"Expected T_freeze={Tm - sc}, got {node.Tp}"
    assert node.f_melt == 1.0 and node.mode == "FREEZING"
    log("  [PASS] Unit Test 6: Supercooling hysteresis loop correctly executed")


# -----------------------------------------------------------------------------
# MAIN VERIFICATION ROUTINE
# -----------------------------------------------------------------------------
def main():
    log("=" * 76)
    log("  PHASE 9 — AUTOMATED VERIFICATION SUITE")
    log("=" * 76)
    
    log("\n[SECTION 1] Executing 6 Automated Thermodynamic Unit Tests...")
    test_1_latent_energy_absorbed_melting()
    test_2_latent_energy_released_freezing()
    test_3_no_latent_plateau_skipped()
    test_4_no_energy_created_or_destroyed()
    test_5_direction_reversal_continuity()
    test_6_supercooling_hysteresis()
    log("  => ALL 6 THERMODYNAMIC UNIT TESTS PASSED SUCCESSFULLY.")
    
    log("\n[SECTION 2] Verifying Programmatic True Medoid Derivation...")
    assign_df = pd.read_csv(ASSIGN_FILE)
    sig_raw_df = pd.read_csv(SIG_RAW_FILE)
    medoids = derive_true_medoids(assign_df, sig_raw_df)
    expected_medoids = {0: "ASP_0012", 1: "ASP_0092", 2: "ASP_0028"}
    assert medoids == expected_medoids, f"Medoid mismatch! Expected {expected_medoids}, got {medoids}"
    log(f"  [PASS] True Cluster Medoids verified: {medoids}")
    
    log("\n[SECTION 3] Verifying Output CSV Schema & Non-MCDM Candidate Labels...")
    assert OUT_RESULTS_CSV.exists(), f"Results CSV missing: {OUT_RESULTS_CSV}"
    res_df = pd.read_csv(OUT_RESULTS_CSV)
    assert len(res_df) == 24, f"Expected 24 simulation results (8 PCMs x 3 clusters), got {len(res_df)}"
    
    req_cols = [
        "cluster_id", "medoid_point_id", "pcm_name", "candidate_status_label",
        "pcm_mass_kg", "melting_temp_degC", "latent_heat_kJ_kg", "supercooling_degC",
        "morning_delivery_success_rate", "evening_delivery_success_rate", "overall_delivery_success_rate",
        "hours_Tw_ge_50C_per_year", "solar_fraction", "complete_pcm_cycles_per_year",
        "spinup_converged", "spinup_cycles_run", "max_step_residual_J", "cum_rel_energy_error_pct",
        "dt_sensitivity_passed", "dt_sens_sf_rel_diff_pct", "dt_sens_delivery_abs_diff_pp",
        "dt_sens_cycles_abs_diff", "dt_sens_cum_err_300s_pct", "dt_sens_cum_err_150s_pct",
        "ssrd_raw_energy_J_per_m2", "ssrd_recon_conservation_err_pct", "ssrd_nightclamp_loss_J_per_m2",
        "validation_status"
    ]
    for col in req_cols:
        assert col in res_df.columns, f"Missing column in CSV: {col}"
        
    expected_label = (
        "Phase 6-screened candidate evaluated independently under the "
        "final Phase 3 K=3 climate forcing; not an MCDM-ranked PCM."
    )
    assert (res_df["candidate_status_label"] == expected_label).all(), "Candidate label mismatch!"
    log(f"  [PASS] CSV Schema & non-MCDM candidate labels verified across {len(res_df)} rows.")
    
    log("\n[SECTION 4] Verifying Convergence, First-Law Energy Balance & Sensitivity Criteria...")
    assert (res_df["spinup_converged"] == True).all(), "Spin-up loop failed to converge for some PCMs!"
    assert (res_df["cum_rel_energy_error_pct"] < 0.1).all(), "Cumulative energy relative error >= 0.1%!"
    assert (res_df["dt_sensitivity_passed"] == True).all(), "Sub-hourly timestep sensitivity check failed!"
    assert (res_df["dt_sens_sf_rel_diff_pct"] < 1.0).all(), "dt sensitivity SF relative difference >= 1%!"
    assert (res_df["dt_sens_delivery_abs_diff_pp"] <= 1.0).all(), "dt sensitivity delivery rate difference > 1 pp!"
    assert (res_df["dt_sens_cycles_abs_diff"] <= 1.0).all(), "dt sensitivity cycle count difference > 1 cycle/yr!"
    assert (res_df["dt_sens_cum_err_300s_pct"] < 0.1).all(), "dt 300s cumulative energy error >= 0.1%!"
    assert (res_df["dt_sens_cum_err_150s_pct"] < 0.1).all(), "dt 150s cumulative energy error >= 0.1%!"
    assert (res_df["ssrd_recon_conservation_err_pct"] < 0.001).all(), "SSRD reconstruction conservation error >= 0.001%!"
    assert (res_df["validation_status"] == "PASSED").all(), "Validation status not PASSED for all PCMs!"
    log("  [PASS] 100% convergence, <0.1% cumulative error, SSRD <0.001% conservation, and timestep sensitivity verified.")
    
    log("\n" + "=" * 76)
    log("  PHASE 9 VERIFICATION PASSED PERFECTLY (100% INTEGRITY & COMPLIANCE)")
    log("=" * 76)

if __name__ == "__main__":
    main()

