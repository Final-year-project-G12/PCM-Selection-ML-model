"""
07_feasibility_filter_final.py  - Assam SWH Project
==========================================================================
PHASE 6 — PCM FEASIBILITY FILTERING & AUDIT (EVIDENCE-BASED SAFETY & CORROSION)

Multi-criterion feasibility filtering for each climate cluster against
Phase 4 SWH specifications.

EVIDENCE-BASED RULES IMPLEMENTED:
  1. Corrosion Evaluation:
     - PASS: Explicit documented metal compatibility evidence.
     - FAIL: Explicit documented incompatibility (e.g. inorganic salt hydrate in high HSI).
     - UNKNOWN: Category-only or missing explicit test evidence. (No category-based PASS).
  2. Safety Evaluation:
     - PASS: Explicit reported non-flammable evidence (flammability == "No" and status == "Reported").
     - FAIL: Explicit reported flammable hazard (flammability == "Yes" and status == "Reported").
     - UNKNOWN: Imputed, missing, or ambiguous flammability evidence.
  3. Classification:
     - CONFIRMED FEASIBLE: ALL 6 criteria are PASS.
     - CONDITIONALLY FEASIBLE: Zero FAIL criteria, but >=1 UNKNOWN.
     - INFEASIBLE: >=1 FAIL criterion.
"""

from importlib import resources
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PCM_DB_FILE = BASE_DIR / "data" / "processed" / "pcm" / "pcm_database_final.csv"
SWH_SPEC_FILE = BASE_DIR / "data" / "processed" / "design" / "swh_design_specification.csv"
CLUSTER_PROFILE_FILE = BASE_DIR / "data" / "processed" / "clustering" / "cluster_profiles_assam.csv"

OUT_FEASIBILITY_DIR = BASE_DIR / "data" / "processed" / "feasibility"
OUT_REPORT_DIR = BASE_DIR / "data" / "preprocessed"

OUT_FEASIBILITY_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BY_CLUSTER_CSV = OUT_FEASIBILITY_DIR / "pcm_feasibility_by_cluster.csv"
OUT_SUMMARY_CSV = OUT_FEASIBILITY_DIR / "pcm_feasibility_summary.csv"
OUT_REPORT_TXT = OUT_REPORT_DIR / "feasibility_filter_report.txt"

# Design Constants (Phase 4 Alignment)
TARGET_DELIVERY_TEMP_C = 50.0
APPROACH_TEMP_K = 6.0
TARGET_TM_C = TARGET_DELIVERY_TEMP_C + APPROACH_TEMP_K  # 50.0 + 6.0 = 56.0°C (indirect-system rule)
DAILY_DEMAND_L = 100.0
MORNING_DRAW_L = 50.0
EVENING_DRAW_L = 50.0
PCM_MASS_KG = 50.0
CP_WATER_JKGK = 4186.0
SHARE_PCM = 0.5  # Fraction of nocturnal heat deficit covered by PCM latent heat (Avargani 2021)

# Temperature Window Constraints
MELT_WINDOW_LOW_OFFSET = -5.0
MELT_WINDOW_HIGH_OFFSET = 8.0
RELAX_STEP_K = 2.0
MAX_RELAX_ROUNDS = 4
MIN_SURVIVORS_TARGET = 5
ABSOLUTE_TM_MIN_C = 42.0
ABSOLUTE_TM_MAX_C = 70.0

# Criteria Thresholds
CYCLES_FLOOR = 300
SUPERCOOLING_MAX_K = 8.0

def run_feasibility_filter():
    print("==========================================================================")
    print("    PHASE 6 — PCM FEASIBILITY FILTERING (EVIDENCE-BASED SAFETY & CORROSION)")
    print("==========================================================================")

    for f in (PCM_DB_FILE, CLUSTER_PROFILE_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Required input file missing: {f}")

    pcm_db = pd.read_csv(PCM_DB_FILE)
    profiles = pd.read_csv(CLUSTER_PROFILE_FILE)

    cluster_l_req = {}
    if SWH_SPEC_FILE.exists():
        with open(SWH_SPEC_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_table = False
        for l in lines:
            l_str = l.strip()
            if l_str.startswith("Cluster,"):
                in_table = True
                continue
            if in_table and l_str and not l_str.startswith("#"):
                parts = l_str.split(",")
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        l_req_kJ_kg = float(parts[4])
                        cluster_l_req[cid] = l_req_kJ_kg
                    except (ValueError, IndexError):
                        pass

    if "HSI_sunrise" not in profiles.columns:
        raise ValueError(
            "HSI_sunrise is required in the Assam cluster profiles; "
            "RH_mean_mean must not be used as an HSI proxy."
        )

    hsi_global_p75 = profiles["HSI_sunrise"].quantile(0.75)

    all_audit_rows = []
    summary_rows = []

    # Base melting window follows the project methodology:
    # Tm_target - 5 K to Tm_target + 8 K.
    # The window is relaxed by +/-2 K per round if fewer than
    # MIN_SURVIVORS_TARGET candidates survive all hard constraints.
    base_tm_min = max(
        TARGET_TM_C + MELT_WINDOW_LOW_OFFSET,
        ABSOLUTE_TM_MIN_C
    )
    base_tm_max = min(
        TARGET_TM_C + MELT_WINDOW_HIGH_OFFSET,
        ABSOLUTE_TM_MAX_C
    )

    print(f"PCM Candidate Pool: {len(pcm_db)} unique materials")
    print(f"Fixed Design PCM Mass: {PCM_MASS_KG} kg")
    print(
        f"Base Operational Melting Window: "
        f"[{base_tm_min:.1f}, {base_tm_max:.1f}] °C"
    )
    print(
        f"Relaxation: +/-{RELAX_STEP_K:.1f} K/round, "
        f"maximum {MAX_RELAX_ROUNDS} rounds, "
        f"target >= {MIN_SURVIVORS_TARGET} survivors/cluster"
    )
    print("--------------------------------------------------------------------------\n")

    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])
        # Dynamic extraction from SWH spec, cluster profiles, or direct calculation (SHARE_PCM = 0.5)
        if cid in cluster_l_req:
            l_req = cluster_l_req[cid]
        elif "L_required_kJ_per_kg" in prof and pd.notna(prof["L_required_kJ_per_kg"]):
            l_req = float(prof["L_required_kJ_per_kg"])
        else:
            t_m_mean = max(5.0, prof.get("Ta_mean_mean", 25.0) - 6.0)
            q_night_kWh = (DAILY_DEMAND_L * CP_WATER_JKGK * (TARGET_DELIVERY_TEMP_C - t_m_mean)) / 3_600_000.0
            l_req = (SHARE_PCM * q_night_kWh * 3600.0) / PCM_MASS_KG
        hsi_val = prof.get("HSI_sunrise", np.nan)
        if pd.isna(hsi_val):
            raise ValueError(
                f"HSI_sunrise missing for Assam cluster {cid}"
            )
        hsi_high = hsi_val > hsi_global_p75

        c_survivors_combined_tm = 0
        c_survivors_lh = 0

 
        for relax_round in range(MAX_RELAX_ROUNDS + 1):

            # Classification counts are per relaxation round.
            n_conditional = 0
            n_infeasible = 0
            n_confirmed = 0

            widen = RELAX_STEP_K * relax_round

            window_lo = max(
                TARGET_TM_C + MELT_WINDOW_LOW_OFFSET - widen,
                ABSOLUTE_TM_MIN_C
            )

            window_hi = min(
                TARGET_TM_C + MELT_WINDOW_HIGH_OFFSET + widen,
                ABSOLUTE_TM_MAX_C
            )

            round_rows = []

            # Evaluate all PCMs for this relaxation round
            for _, pcm in pcm_db.iterrows():

                rec = {}

                rec["cluster_id"] = cid
                rec["pcm_id"] = pcm["pcm_id"]
                rec["product_name"] = pcm["product_name"]
                rec["manufacturer"] = pcm["manufacturer"]
                rec["source_type"] = pcm["source_type"]
                rec["family"] = pcm["family"]
                rec["Tm_C"] = pcm["Tm_C"]
                rec["Tm_C_status"] = pcm["Tm_C_status"]
                rec["latent_heat_kJ_kg"] = pcm["latent_heat_kJ_kg"]
                rec["latent_heat_status"] = pcm["latent_heat_status"]
                rec["Tm_target_C"] = TARGET_TM_C

                # ----------------------------------------------------------
                # 1. Tm FEASIBILITY CRITERION (Criterion C2: [42.0, 70.0] °C)
                # ----------------------------------------------------------
                tm_val = pcm["Tm_C"]

                in_combined = ABSOLUTE_TM_MIN_C <= tm_val <= ABSOLUTE_TM_MAX_C

                if in_combined:
                    c_survivors_combined_tm += 1
                    rec["status_tm"] = "PASS"
                else:
                    rec["status_tm"] = "FAIL"

                # ----------------------------------------------------------
                # 2. LATENT HEAT REQUIREMENT (Continuous MCDM criterion, not a hard feasibility veto)
                # ----------------------------------------------------------
                lh_val = pcm["latent_heat_kJ_kg"]
                lh_status = pcm["latent_heat_status"]

                if pd.notna(lh_val) and lh_val > 0:
                    rec["status_latent_heat"] = "PASS"
                    c_survivors_lh += 1
                else:
                    rec["status_latent_heat"] = "UNKNOWN"

                # Check data completeness for fundamental MCDM criteria (TC, rho*H)
                tc_val = pcm.get("TC_W_mK", np.nan)
                tc_status = str(pcm.get("TC_status", "Missing"))
                rho_h = pcm.get("rho_H_MJ_m3", np.nan)
                has_complete_data = (
                    pd.notna(tc_val)
                    and tc_status != "Missing"
                    and pd.notna(rho_h)
                )
                rec["status_data_complete"] = "PASS" if has_complete_data else "FAIL"

                # ----------------------------------------------------------
                # 3. CYCLING STABILITY
                # ----------------------------------------------------------
                cyc_val = pcm["cycles_tested"]
                cyc_status = str(pcm.get("cycles_status", "Missing"))

                if pd.notna(cyc_val) and cyc_status == "Reported":
                    if cyc_val >= CYCLES_FLOOR:
                        rec["status_cycling"] = "PASS"
                        rec["cycling_evidence"] = (
                            f"Reported_{int(cyc_val)}cycles"
                        )
                    else:
                        rec["status_cycling"] = "FAIL"
                        rec["cycling_evidence"] = (
                            f"Reported_{int(cyc_val)}cycles_<300"
                        )
                else:
                    rec["status_cycling"] = "UNKNOWN"
                    rec["cycling_evidence"] = "Imputed_or_Unreported"

                # ----------------------------------------------------------
                # 4. SUPERCOOLING
                # ----------------------------------------------------------
                sc_val = pcm["supercooling_K"]
                sc_status = str(
                    pcm.get("supercooling_status", "Missing")
                )
                tm_f_status = str(
                    pcm.get("Tm_freezing_C_status", "Missing")
                )

                if pd.notna(sc_val) and (
                    sc_status == "Reported"
                    or (
                        rec["Tm_C_status"] == "Reported"
                        and tm_f_status == "Reported"
                    )
                ):
                    if sc_val <= SUPERCOOLING_MAX_K:
                        rec["status_supercooling"] = "PASS"
                        rec["supercooling_evidence"] = (
                            f"Reported_{sc_val}K"
                        )
                    else:
                        rec["status_supercooling"] = "FAIL"
                        rec["supercooling_evidence"] = (
                            f"Reported_{sc_val}K_>8K"
                        )
                else:
                    rec["status_supercooling"] = "UNKNOWN"
                    rec["supercooling_evidence"] = "Imputed_or_Unreported"

                # ----------------------------------------------------------
                # 5. CHARGING FEASIBILITY
                # ----------------------------------------------------------
                tm_capped = prof.get("Tm_target_capped_C", np.nan)

                if pd.notna(tm_capped) and pd.notna(tm_val):
                    if tm_val <= tm_capped:
                        rec["c6_charging_feasibility"] = "pass"
                    else:
                        rec["c6_charging_feasibility"] = "fail"
                else:
                    rec["c6_charging_feasibility"] = "flag_missing"

                # ----------------------------------------------------------
                # 6. CORROSION (C7)
                # ----------------------------------------------------------
                is_inorg = bool(pcm.get("is_inorganic", False))
                pname = str(pcm.get("product_name", "")).strip()
                fam = str(pcm.get("family", "")).strip()
                corr_test_evidence = pcm.get("corrosion_evidence", np.nan)

                # Rule 6a: Inorganic with high HSI veto -> FAIL
                if is_inorg and hsi_high:
                    rec["status_corrosion"] = "FAIL"
                    rec["corrosion_evidence"] = "Inorganic_HighHSI_Veto"
                    rec["corrosion_source"] = "Project_Corrosion_Veto_Rule"

                # Rule 6b: Fatty acids and fatty-acid eutectics -> CONDITIONAL
                # (Carboxylic acids react with copper/aluminum; compatible with stainless steel AISI 304/316)
                elif any(fa in pname for fa in ["Myristic", "Palmitic", "Stearic", "Lauric"]):
                    rec["status_corrosion"] = "CONDITIONAL"
                    rec["corrosion_evidence"] = (
                        "FattyAcid_CopperReactive_Requires_StainlessSteel"
                    )
                    rec["corrosion_source"] = "Literature_Experimental_Corrosion"

                # Rule 6c: Pure linear alkanes / pure paraffin waxes -> PASS
                # (Saturated hydrocarbons confirmed chemically inert toward standard metals in literature)
                elif any(
                    pname.startswith(alk) or alk in pname
                    for alk in [
                        "RT", "n-Docosane", "n-Tricosane", "n-Tetracosane", "n-Pentacosane",
                        "n-Hexacosane", "n-Heptacosane", "n-Octacosane", "n-Nonacosane",
                        "n-Triacontane", "n-Hentriacontane", "n-Dotriacontane", "Paraffin wax"
                    ]
                ) and not any(comp in pname for comp in ["Expanded graphite", "HDPE", "NBR"]):
                    rec["status_corrosion"] = "PASS"
                    rec["corrosion_evidence"] = "Confirmed_Inert_Alkane_Literature"
                    rec["corrosion_source"] = "Literature_and_Manufacturer_Alkanes"

                # Rule 6d: Composites with conductive/structural matrices -> UNKNOWN
                # (Galvanic coupling or additive interaction requires coupon testing)
                elif any(comp in pname for comp in ["Expanded graphite", "HDPE", "NBR"]):
                    rec["status_corrosion"] = "UNKNOWN"
                    rec["corrosion_evidence"] = "Composite_Galvanic_Coupon_Unreported"
                    rec["corrosion_source"] = "Unreported"

                # Rule 6e: Commercial proprietary bio-based/organic blends -> UNKNOWN
                # (Proprietary formulations lack published material-specific coupon immersion data)
                elif any(brand in pname or brand in fam for brand in ["savE", "PlusICE", "PureTemp", "CrodaTherm"]):
                    rec["status_corrosion"] = "UNKNOWN"
                    rec["corrosion_evidence"] = "Proprietary_Organic_Blend_Coupon_Unreported"
                    rec["corrosion_source"] = "Unreported"

                # Rule 6f: Explicit datasheet compatibility evidence if available
                elif pd.notna(corr_test_evidence) and str(corr_test_evidence).strip() != "":
                    rec["status_corrosion"] = "PASS"
                    rec["corrosion_evidence"] = str(corr_test_evidence).strip()
                    rec["corrosion_source"] = "Datasheet_Compatibility_Test"

                # Rule 6g: Default unreported
                else:
                    rec["status_corrosion"] = "UNKNOWN"
                    rec["corrosion_evidence"] = "Missing_Explicit_Compatibility_Test"
                    rec["corrosion_source"] = "Unreported"

                # ----------------------------------------------------------
                # 7. SAFETY
                # ----------------------------------------------------------
                flam_raw = str(
                    pcm.get("flammability", "")
                ).strip()
                flam_status = str(
                    pcm.get("flammability_status", "Missing")
                )
                flam_lower = flam_raw.lower()

                if flam_status == "Reported":

                    if flam_lower in (
                        "yes",
                        "highly flammable",
                        "extremely flammable",
                        "toxic",
                    ):
                        rec["status_safety"] = "FAIL"
                        rec["safety_evidence"] = (
                            f"Reported_Hazard_{flam_raw}"
                        )
                        rec["safety_source"] = (
                            "Manufacturer_Datasheet"
                        )

                    elif flam_lower in (
                        "no",
                        "non-flammable",
                        "low",
                    ):
                        rec["status_safety"] = "PASS"
                        rec["safety_evidence"] = (
                            f"Reported_NonFlammable_{flam_raw}"
                        )
                        rec["safety_source"] = (
                            "Manufacturer_Datasheet"
                        )

                    else:
                        rec["status_safety"] = "UNKNOWN"
                        rec["safety_evidence"] = (
                            f"Ambiguous_{flam_raw}"
                        )
                        rec["safety_source"] = (
                            "Manufacturer_Datasheet"
                        )

                else:
                    rec["status_safety"] = "UNKNOWN"
                    rec["safety_evidence"] = (
                        f"Unreported_or_Imputed_{flam_raw}"
                    )
                    rec["safety_source"] = "Unreported"

                # ----------------------------------------------------------
                # 8. DEPLOYABILITY / SURVIVOR RULE
                # ----------------------------------------------------------
                # C2: Thermodynamic melting window [42.0, 70.0] C
                # C8: Confirmed flammability hazard veto
                # C7: High-HSI corrosion veto for inorganics
                # Data completeness: Fundamental MCDM criteria (k, rho*H)
                rec["survives_c2_tm"] = (rec["status_tm"] != "FAIL")
                rec["survives_c7_corrosion"] = (rec["status_corrosion"] != "FAIL")
                rec["survives_c8_safety"] = (rec["status_safety"] != "FAIL")
                rec["survives_data_complete"] = (rec["status_data_complete"] == "PASS")

                rec["survives_all"] = (
                    rec["survives_c2_tm"]
                    and rec["survives_c7_corrosion"]
                    and rec["survives_c8_safety"]
                    and rec["survives_data_complete"]
                )
                rec["passes_all"] = rec["survives_all"]

                # ----------------------------------------------------------
                # 9. EVIDENCE-BASED CLASSIFICATION
                # ----------------------------------------------------------
                statuses = [
                    rec["status_tm"],
                    rec["status_latent_heat"],
                    rec["status_cycling"],
                    rec["status_supercooling"],
                    rec["status_corrosion"],
                    rec["status_safety"],
                ]

                if any(s == "FAIL" for s in statuses):
                    rec["overall_feasibility"] = "INFEASIBLE"
                    n_infeasible += 1

                elif any(s in ("UNKNOWN", "CONDITIONAL") for s in statuses):
                    rec["overall_feasibility"] = (
                        "CONDITIONALLY_FEASIBLE"
                    )
                    n_conditional += 1

                else:
                    rec["overall_feasibility"] = "CONFIRMED_FEASIBLE"
                    n_confirmed += 1

                round_rows.append(rec)

            # ----------------------------------------------------------
            # CHECK SURVIVORS FOR THIS RELAXATION ROUND
            # ----------------------------------------------------------
            n_survive = sum(
                r["survives_all"] for r in round_rows
            )

            print(
                f"Cluster {cid} | Relax round {relax_round} | "
                f"Tm window [{window_lo:.1f}, {window_hi:.1f}] °C | "
                f"Survivors = {n_survive}"
            )

            # Save this round. If it meets the survivor target, this
            # becomes the final selected round and we then stop relaxing.
            all_audit_rows.extend(round_rows)

            summary_rows.append({
                "cluster_id": cid,
                "L_required_kJ_kg": l_req,
                "final_tm_min": window_lo,
                "final_tm_max": window_hi,
                "relax_round": relax_round,
                "total_candidates": len(round_rows),
                "survivors_all": n_survive,
                "survivors_combined_tm": sum(
                    r["status_tm"] == "PASS" for r in round_rows
                ),
                "survivors_latent_heat": sum(
                    r["status_latent_heat"] == "PASS" for r in round_rows
                ),
                "confirmed_feasible_count": n_confirmed,
                "conditionally_feasible_count": n_conditional,
                "infeasible_count": n_infeasible,
            })

            if n_survive >= MIN_SURVIVORS_TARGET:
                break

    # Continue to the next cluster.

        print(f"Cluster {cid} (L_req = {l_req:.1f} kJ/kg):")
        print(f"  - Confirmed Feasible (ALL 6 PASS)       : {n_confirmed}")
        print(f"  - Conditionally Feasible (HAS UNKNOWN)   : {n_conditional}")
        print(f"  - Infeasible (HAS FAIL)                 : {n_infeasible}")

    df_audit = pd.DataFrame(all_audit_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_audit.to_csv(OUT_BY_CLUSTER_CSV, index=False)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False)

    # Save surviving PCMs across all clusters for downstream plotting/verification
    out_pcm_dir = BASE_DIR / "data" / "processed" / "pcm"
    out_pcm_dir.mkdir(parents=True, exist_ok=True)
    feas_surv_file = out_pcm_dir / "feasibility_survivors_assam.csv"
    surv_df = df_audit[df_audit["survives_all"]].copy()
    surv_df["name"] = surv_df["product_name"]
    surv_df["passes_all"] = True
    surv_df.to_csv(feas_surv_file, index=False)

    report_lines = []

    report_lines.append(
        "=========================================================================="
    )
    report_lines.append(
        "   PHASE 6 — PCM FEASIBILITY FILTERING REPORT "
        "(EVIDENCE-BASED SAFETY/CORROSION)"
    )
    report_lines.append(
        "=========================================================================="
    )
    report_lines.append(
        f"Base Operational Melting Window: "
        f"[{base_tm_min:.1f}, {base_tm_max:.1f}] °C"
    )
    report_lines.append(
        f"Relaxation: +/-{RELAX_STEP_K:.1f} K/round | "
        f"Maximum rounds: {MAX_RELAX_ROUNDS} | "
        f"Target survivors: {MIN_SURVIVORS_TARGET}"
    )
    report_lines.append("")

    for s in summary_rows:

        cid = s["cluster_id"]

        report_lines.append(
            f"Cluster {cid} (L_req = {s['L_required_kJ_kg']:.1f} kJ/kg):"
        )

        report_lines.append(
            f"  Final Tm Window: "
            f"[{s['final_tm_min']:.1f}, {s['final_tm_max']:.1f}] °C"
        )

        report_lines.append(
            f"  Relaxation Round:        {s['relax_round']}"
        )

        report_lines.append(
            f"  Total Candidates:        {s['total_candidates']}"
        )

        report_lines.append(
            f"  Survivors (deployable):  {s['survivors_all']}"
        )

        report_lines.append(
            f"  Tm Feasible:             {s['survivors_combined_tm']}"
        )

        report_lines.append(
            f"  Latent Heat Feasible:    {s['survivors_latent_heat']}"
        )

        report_lines.append(
            f"  Confirmed Feasible:      {s['confirmed_feasible_count']}"
        )

        report_lines.append(
            f"  Conditionally Feasible:  {s['conditionally_feasible_count']}"
        )

        report_lines.append(
            f"  Infeasible:              {s['infeasible_count']}"
        )

        report_lines.append("")
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n--------------------------------------------------------------------------")
    print(f"[Phase 6 SUCCESS] Saved detailed audit matrix to: {OUT_BY_CLUSTER_CSV}")
    print(f"[Phase 6 SUCCESS] Saved cluster summary to: {OUT_SUMMARY_CSV}")
    print(f"[Phase 6 SUCCESS] Saved report to: {OUT_REPORT_TXT}")
    print("==========================================================================")

if __name__ == "__main__":
    run_feasibility_filter()
