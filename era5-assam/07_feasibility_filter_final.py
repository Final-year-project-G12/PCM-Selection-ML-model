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
TARGET_TM_C = 44.0
DAILY_DEMAND_L = 100.0
MORNING_DRAW_L = 50.0
EVENING_DRAW_L = 50.0
PCM_MASS_KG = 50.0

# Temperature Window Constraints
TM_TARGET_LOWER_K = 6.0
TM_TARGET_UPPER_K = 8.0
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
                    cid = int(parts[0])
                    l_req_kJ_kg = float(parts[4])
                    cluster_l_req[cid] = l_req_kJ_kg
    
    if not cluster_l_req:
        cluster_l_req = {0: 252.09, 1: 258.69, 2: 279.70}

    hsi_global_p75 = profiles["RH_mean_mean"].quantile(0.75) if "RH_mean_mean" in profiles.columns else 78.5

    all_audit_rows = []
    summary_rows = []

    broad_tm_min = TARGET_TM_C - TM_TARGET_LOWER_K  # 38.0°C
    broad_tm_max = TARGET_TM_C + TM_TARGET_UPPER_K  # 52.0°C
    combined_tm_min = max(broad_tm_min, ABSOLUTE_TM_MIN_C) # 42.0°C
    combined_tm_max = broad_tm_max                         # 52.0°C

    print(f"PCM Candidate Pool: {len(pcm_db)} unique materials")
    print(f"Fixed Design PCM Mass: {PCM_MASS_KG} kg")
    print(f"Operational Feasibility Window: [{combined_tm_min:.1f}, {combined_tm_max:.1f}] °C")
    print("--------------------------------------------------------------------------\n")

    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])
        l_req = cluster_l_req.get(cid, 250.0)
        hsi_val = prof.get("RH_mean_mean", 75.0)
        hsi_high = hsi_val > hsi_global_p75

        n_confirmed = 0
        n_conditional = 0
        n_infeasible = 0

        c_survivors_combined_tm = 0
        c_survivors_lh = 0

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

            # 1. Tm Feasibility Criterion (42°C to 52°C)
            tm_val = pcm["Tm_C"]
            in_combined = combined_tm_min <= tm_val <= combined_tm_max
            if in_combined:
                c_survivors_combined_tm += 1
                rec["status_tm"] = "PASS"
            else:
                rec["status_tm"] = "FAIL"

            # 2. Latent Heat Requirement Criterion (L_PCM >= L_required)
            lh_val = pcm["latent_heat_kJ_kg"]
            lh_status = pcm["latent_heat_status"]
            if pd.notna(lh_val) and lh_val >= l_req:
                rec["status_latent_heat"] = "PASS"
                c_survivors_lh += 1
            elif pd.notna(lh_val) and lh_val < l_req:
                rec["status_latent_heat"] = "FAIL"
            else:
                rec["status_latent_heat"] = "UNKNOWN"

            # 3. Cycling Stability Criterion (Reported evidence strictly required for PASS)
            cyc_val = pcm["cycles_tested"]
            cyc_status = str(pcm.get("cycles_status", "Missing"))
            if pd.notna(cyc_val) and cyc_status == "Reported":
                if cyc_val >= CYCLES_FLOOR:
                    rec["status_cycling"] = "PASS"
                    rec["cycling_evidence"] = f"Reported_{int(cyc_val)}cycles"
                else:
                    rec["status_cycling"] = "FAIL"
                    rec["cycling_evidence"] = f"Reported_{int(cyc_val)}cycles_<300"
            else:
                rec["status_cycling"] = "UNKNOWN"
                rec["cycling_evidence"] = "Imputed_or_Unreported"

            # 4. Supercooling Criterion (Reported evidence strictly required for PASS)
            sc_val = pcm["supercooling_K"]
            sc_status = str(pcm.get("supercooling_status", "Missing"))
            tm_f_status = str(pcm.get("Tm_freezing_C_status", "Missing"))
            
            if pd.notna(sc_val) and (sc_status == "Reported" or (rec["Tm_C_status"] == "Reported" and tm_f_status == "Reported")):
                if sc_val <= SUPERCOOLING_MAX_K:
                    rec["status_supercooling"] = "PASS"
                    rec["supercooling_evidence"] = f"Reported_{sc_val}K"
                else:
                    rec["status_supercooling"] = "FAIL"
                    rec["supercooling_evidence"] = f"Reported_{sc_val}K_>8K"
            else:
                rec["status_supercooling"] = "UNKNOWN"
                rec["supercooling_evidence"] = "Imputed_or_Unreported"

            # 5. CORROSION CRITERION (Evidence-Based Rule: No Category-Only PASS)
            is_inorg = pcm.get("is_inorganic", False)
            corr_test_evidence = pcm.get("corrosion_evidence", np.nan)
            
            if is_inorg and hsi_high:
                rec["status_corrosion"] = "FAIL"
                rec["corrosion_evidence"] = "Inorganic_HighHSI_Veto"
                rec["corrosion_source"] = "Project_Corrosion_Veto_Rule"
            elif pd.notna(corr_test_evidence) and str(corr_test_evidence).strip() != "":
                rec["status_corrosion"] = "PASS"
                rec["corrosion_evidence"] = str(corr_test_evidence).strip()
                rec["corrosion_source"] = "Datasheet_Compatibility_Test"
            else:
                # Category alone MUST NOT produce PASS -> UNKNOWN
                rec["status_corrosion"] = "UNKNOWN"
                rec["corrosion_evidence"] = "Missing_Explicit_Compatibility_Test"
                rec["corrosion_source"] = "Unreported"

            # 6. SAFETY CRITERION (Evidence-Based Rule)
            flam_raw = str(pcm.get("flammability", "")).strip()
            flam_status = str(pcm.get("flammability_status", "Missing"))
            flam_lower = flam_raw.lower()

            if flam_status == "Reported":
                if flam_lower in ("yes", "highly flammable", "extremely flammable", "toxic"):
                    rec["status_safety"] = "FAIL"
                    rec["safety_evidence"] = f"Reported_Hazard_{flam_raw}"
                    rec["safety_source"] = "Manufacturer_Datasheet"
                elif flam_lower in ("no", "non-flammable", "low"):
                    rec["status_safety"] = "PASS"
                    rec["safety_evidence"] = f"Reported_NonFlammable_{flam_raw}"
                    rec["safety_source"] = "Manufacturer_Datasheet"
                else:
                    rec["status_safety"] = "UNKNOWN"
                    rec["safety_evidence"] = f"Ambiguous_{flam_raw}"
                    rec["safety_source"] = "Manufacturer_Datasheet"
            else:
                rec["status_safety"] = "UNKNOWN"
                rec["safety_evidence"] = f"Unreported_or_Imputed_{flam_raw}"
                rec["safety_source"] = "Unreported"

            # Overall Feasibility Assignment
            statuses = [
                rec["status_tm"], rec["status_latent_heat"], rec["status_cycling"],
                rec["status_supercooling"], rec["status_corrosion"], rec["status_safety"]
            ]

            if any(s == "FAIL" for s in statuses):
                rec["overall_feasibility"] = "INFEASIBLE"
                n_infeasible += 1
            elif any(s == "UNKNOWN" for s in statuses):
                rec["overall_feasibility"] = "CONDITIONALLY_FEASIBLE"
                n_conditional += 1
            else:
                rec["overall_feasibility"] = "CONFIRMED_FEASIBLE"
                n_confirmed += 1

            all_audit_rows.append(rec)

        summary_rows.append({
            "cluster_id": cid,
            "L_required_kJ_kg": l_req,
            "total_candidates": len(pcm_db),
            "survivors_combined_tm": c_survivors_combined_tm,
            "survivors_latent_heat": c_survivors_lh,
            "confirmed_feasible_count": n_confirmed,
            "conditionally_feasible_count": n_conditional,
            "infeasible_count": n_infeasible
        })

        print(f"Cluster {cid} (L_req = {l_req:.1f} kJ/kg):")
        print(f"  - Confirmed Feasible (ALL 6 PASS)       : {n_confirmed}")
        print(f"  - Conditionally Feasible (HAS UNKNOWN)   : {n_conditional}")
        print(f"  - Infeasible (HAS FAIL)                 : {n_infeasible}")

    df_audit = pd.DataFrame(all_audit_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_audit.to_csv(OUT_BY_CLUSTER_CSV, index=False)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False)

    report_lines = []
    report_lines.append("==========================================================================")
    report_lines.append("   PHASE 6 — PCM FEASIBILITY FILTERING REPORT (EVIDENCE-BASED SAFETY/CORROSION)")
    report_lines.append("==========================================================================")
    report_lines.append(f"Operational Feasibility Window: [{combined_tm_min:.1f}, {combined_tm_max:.1f}] °C")
    report_lines.append("")
    for s in summary_rows:
        cid = s["cluster_id"]
        report_lines.append(f"Cluster {cid} (L_req = {s['L_required_kJ_kg']:.1f} kJ/kg):")
        report_lines.append(f"  Total Candidates:         {s['total_candidates']}")
        report_lines.append(f"  Tm Feasible (42-52 °C):   {s['survivors_combined_tm']}")
        report_lines.append(f"  Latent Heat Feasible:    {s['survivors_latent_heat']}")
        report_lines.append(f"  Confirmed Feasible:      {s['confirmed_feasible_count']}")
        report_lines.append(f"  Conditionally Feasible:  {s['conditionally_feasible_count']}")
        report_lines.append(f"  Infeasible:              {s['infeasible_count']}")
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
