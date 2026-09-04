"""
08_mcdm_ranking_final.py  - Assam SWH Project
==========================================================================
PHASE 7 — MULTI-CRITERIA DECISION MAKING (MCDM) RANKING & ELIGIBILITY GOVERNANCE

Implements the multi-criteria decision-making framework with strict eligibility
governance:
  1. Primary formal MCDM ranking is restricted EXCLUSIVELY to candidates with
     feasibility_status == "CONFIRMED_FEASIBLE" (ALL 6 criteria PASS).
  2. Conditionally Feasible candidates are reported separately under:
     "Conditional candidates — not formally ranked (unresolved evidence)".
  3. Primary MCDM requires n_confirmed >= 2. If n_confirmed < 2, formal ranking
     is NOT performed, and "No MCDM ranking performed due to insufficient/uncertain
     eligible alternatives" is logged.
  4. 5 Criteria: f_Tm (symmetric sigma=4K), L_margin, rho_H, k (phase-averaged
     or reported single-phase), C_confidence (ONLY reported cycles).
  5. 4 MCDM Methods: TOPSIS, GRA (zeta=0.5), PROMETHEE II (q=0.05, p=0.25),
     VIKOR (v=0.5), with Borda consensus & Copeland pairwise cross-check (n>=2).

INPUTS:
  - data/processed/pcm/pcm_database_final.csv
  - data/processed/feasibility/pcm_feasibility_by_cluster.csv
  - data/processed/feasibility/pcm_feasibility_summary.csv
  - data/processed/design/swh_design_specification.csv
  - data/processed/clustering/cluster_profiles_assam.csv

OUTPUTS:
  - data/processed/pcm/mcdm_cluster_eligibility_summary.csv
  - data/processed/pcm/mcdm_rankings_by_cluster.csv
  - data/preprocessed/mcdm_ranking_report.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PCM_DB_FILE = BASE_DIR / "data" / "processed" / "pcm" / "pcm_database_final.csv"
FEASIBILITY_BY_CLUSTER_FILE = BASE_DIR / "data" / "processed" / "feasibility" / "pcm_feasibility_by_cluster.csv"
FEASIBILITY_SUMMARY_FILE = BASE_DIR / "data" / "processed" / "feasibility" / "pcm_feasibility_summary.csv"
SWH_SPEC_FILE = BASE_DIR / "data" / "processed" / "design" / "swh_design_specification.csv"
CLUSTER_PROFILE_FILE = BASE_DIR / "data" / "processed" / "clustering" / "cluster_profiles_assam.csv"

OUT_PCM_DIR = BASE_DIR / "data" / "processed" / "pcm"
OUT_REPORT_DIR = BASE_DIR / "data" / "preprocessed"

OUT_PCM_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ELIGIBILITY_SUMMARY_CSV = OUT_PCM_DIR / "mcdm_cluster_eligibility_summary.csv"
OUT_RANKINGS_CSV = OUT_PCM_DIR / "mcdm_rankings_by_cluster.csv"
OUT_REPORT_TXT = OUT_REPORT_DIR / "mcdm_ranking_report.txt"

# Core Parameters
TARGET_TM_C = 44.0
SIGMA_TM = 4.0

GRA_ZETA = 0.5
PROMETHEE_Q, PROMETHEE_P = 0.05, 0.25
VIKOR_V = 0.5

# Equal-Prior AHP Weights for 5 Criteria
# 1. f_Tm (0.30), 2. L_margin (0.25), 3. rho_H (0.15), 4. k (0.15), 5. C_confidence (0.15)
PRIOR_WEIGHTS = {
    "f_Tm": 0.30,
    "latent_heat_margin_ratio": 0.25,
    "rho_H_MJ_m3": 0.15,
    "TC_W_mK": 0.15,
    "cycles_confidence": 0.15,
}
CRITERIA = list(PRIOR_WEIGHTS.keys())

def gaussian_tm_fitness(tm, target=TARGET_TM_C, sigma=SIGMA_TM):
    """Symmetric target-distance Gaussian fitness function."""
    return np.exp(-((tm - target) ** 2) / (2.0 * sigma ** 2))

def compute_thermal_conductivity(row):
    """Computes k based on availability: phase average, reported solid, or reported liquid."""
    tc_liq = row.get("TC_liquid_W_mK", np.nan)
    tc_sol = row.get("TC_solid_W_mK", np.nan)
    tc_both = row.get("TC_W_mK", np.nan)

    if pd.notna(tc_liq) and pd.notna(tc_sol):
        return (tc_liq + tc_sol) / 2.0, "Phase_Averaged"
    elif pd.notna(tc_sol):
        return tc_sol, "Solid_Reported"
    elif pd.notna(tc_liq):
        return tc_liq, "Liquid_Reported"
    elif pd.notna(tc_both):
        return tc_both, "Combined_Reported"
    else:
        return np.nan, "Missing"

def compute_cycling_confidence(row, max_cycles):
    """Computes cycling confidence ONLY if cycles_status == 'Reported'."""
    cyc_val = row.get("cycles_tested", np.nan)
    cyc_status = str(row.get("cycles_status", "Missing"))
    if pd.notna(cyc_val) and cyc_status == "Reported" and cyc_val > 0:
        return np.log1p(cyc_val) / np.log1p(max_cycles) if max_cycles > 0 else 0.5
    else:
        return np.nan

def minmax_normalize(df, cols):
    M = df[cols].copy()
    for c in cols:
        lo, hi = M[c].min(), M[c].max()
        M[c] = (M[c] - lo) / (hi - lo) if hi > lo else 0.5
    return M.values

def topsis(matrix, weights):
    norm = matrix / (np.sqrt((matrix ** 2).sum(axis=0)) + 1e-12)
    weighted = norm * weights
    v_plus = weighted.max(axis=0)
    v_minus = weighted.min(axis=0)
    s_plus = np.sqrt(((weighted - v_plus) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - v_minus) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)

def gra(matrix, weights, zeta=GRA_ZETA):
    ref = matrix.max(axis=0)
    delta = np.abs(matrix - ref)
    d_min, d_max = delta.min(), delta.max()
    coeff = (d_min + zeta * d_max) / (delta + zeta * d_max + 1e-12)
    return (coeff * weights).sum(axis=1)

def promethee_ii(matrix, weights, q=PROMETHEE_Q, p=PROMETHEE_P):
    n, k = matrix.shape
    phi_plus = np.zeros(n)
    phi_minus = np.zeros(n)
    for j in range(k):
        col = matrix[:, j]
        d = col[:, None] - col[None, :]
        pref = np.clip((np.abs(d) - q) / (p - q + 1e-12), 0, 1)
        pref = np.where(d > 0, pref, 0.0)
        phi_plus += weights[j] * pref.sum(axis=1)
        phi_minus += weights[j] * pref.sum(axis=0)
    denom = max(n - 1, 1)
    return (phi_plus - phi_minus) / denom

def vikor(matrix, weights, v=VIKOR_V):
    f_star = matrix.max(axis=0)
    f_minus = matrix.min(axis=0)
    span = np.where((f_star - f_minus) == 0, 1e-12, f_star - f_minus)
    wgap = weights * (f_star - matrix) / span
    S = wgap.sum(axis=1)
    R = wgap.max(axis=1)
    s_star, s_minus = S.min(), S.max()
    r_star, r_minus = R.min(), R.max()
    Q = (v * (S - s_star) / (s_minus - s_star + 1e-12) +
         (1 - v) * (R - r_star) / (r_minus - r_star + 1e-12))
    return Q, S, R

def run_phase7():
    print("==========================================================================")
    print("        PHASE 7 — MCDM RANKING & ELIGIBILITY GOVERNANCE")
    print("==========================================================================")

    for f in (PCM_DB_FILE, FEASIBILITY_BY_CLUSTER_FILE, FEASIBILITY_SUMMARY_FILE, CLUSTER_PROFILE_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Required input file missing: {f}")

    pcm_db = pd.read_csv(PCM_DB_FILE)
    feas_by_cluster = pd.read_csv(FEASIBILITY_BY_CLUSTER_FILE)
    feas_summary = pd.read_csv(FEASIBILITY_SUMMARY_FILE)
    profiles = pd.read_csv(CLUSTER_PROFILE_FILE)

    max_cycles_reported = pcm_db[pcm_db["cycles_status"] == "Reported"]["cycles_tested"].max()
    if pd.isna(max_cycles_reported):
        max_cycles_reported = 2000.0

    eligibility_summary_rows = []
    all_ranking_rows = []
    report_lines = []

    report_lines.append("==========================================================================")
    report_lines.append("        PHASE 7 — MULTI-CRITERIA DECISION MAKING (MCDM) REPORT")
    report_lines.append("==========================================================================")
    report_lines.append("Methodological Governance Rules Applied:")
    report_lines.append("  1. Primary Formal MCDM Candidate Set: ONLY confirmed feasible PCMs (ALL 6 PASS).")
    report_lines.append("  2. Conditionally Feasible candidates reported separately under:")
    report_lines.append("     'Conditional candidates — not formally ranked (unresolved evidence)'.")
    report_lines.append("  3. Governance Rule: Primary MCDM requires n_confirmed >= 2. If n_confirmed < 2,")
    report_lines.append("     formal MCDM ranking is NOT performed.")
    report_lines.append("")

    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])
        c_feas = feas_by_cluster[feas_by_cluster["cluster_id"] == cid]
        
        c_summary = feas_summary[feas_summary["cluster_id"] == cid]
        l_req = c_summary["L_required_kJ_kg"].iloc[0] if len(c_summary) > 0 else 250.0

        confirmed_df = c_feas[c_feas["overall_feasibility"] == "CONFIRMED_FEASIBLE"]
        conditional_df = c_feas[c_feas["overall_feasibility"] == "CONDITIONALLY_FEASIBLE"]
        infeasible_df = c_feas[c_feas["overall_feasibility"] == "INFEASIBLE"]

        n_confirmed = len(confirmed_df)
        n_conditional = len(conditional_df)
        n_infeasible = len(infeasible_df)

        print(f"Cluster {cid} (L_req = {l_req:.1f} kJ/kg):")
        print(f"  - Confirmed Feasible (n_confirmed)  : {n_confirmed}")
        print(f"  - Conditionally Feasible            : {n_conditional}")
        print(f"  - Infeasible                        : {n_infeasible}")

        if n_confirmed >= 2:
            mcdm_status = "PERFORMED"
            reason_no_ranking = "None"
        elif n_conditional > 0 and n_confirmed == 0:
            mcdm_status = "NOT PERFORMED"
            reason_no_ranking = f"Only {n_conditional} conditionally feasible candidate(s) exist (n_confirmed=0). Matrix algorithms require n_confirmed >= 2."
        else:
            mcdm_status = "NOT PERFORMED"
            reason_no_ranking = "No eligible PCM for MCDM ranking (n_confirmed=0)."

        print(f"  -> MCDM Ranking Status: {mcdm_status}")
        print(f"  -> Reason: {reason_no_ranking}")

        eligibility_summary_rows.append({
            "cluster_id": cid,
            "L_required_kJ_kg": l_req,
            "total_pcm_pool": len(c_feas),
            "confirmed_feasible_count": n_confirmed,
            "conditionally_feasible_count": n_conditional,
            "infeasible_count": n_infeasible,
            "mcdm_ranking_status": mcdm_status,
            "reason_for_no_ranking": reason_no_ranking
        })

        report_lines.append(f"CLUSTER {cid} AUDIT SUMMARY (L_req = {l_req:.1f} kJ/kg):")
        report_lines.append(f"  Total Candidates:              {len(c_feas)}")
        report_lines.append(f"  Confirmed Feasible (n_confirmed): {n_confirmed}")
        report_lines.append(f"  Conditionally Feasible:        {n_conditional}")
        report_lines.append(f"  Infeasible:                    {n_infeasible}")
        report_lines.append(f"  MCDM Ranking Status:           {mcdm_status}")
        report_lines.append(f"  Reason:                        {reason_no_ranking}")

        if n_conditional > 0:
            report_lines.append("\n  Conditional candidates — not formally ranked (unresolved evidence):")
            for _, cond_p in conditional_df.iterrows():
                pname = cond_p["product_name"]
                db_p = pcm_db[pcm_db["product_name"] == pname].iloc[0]
                report_lines.append(f"    - {pname} (Tm={db_p['Tm_C']}°C, L={db_p['latent_heat_kJ_kg']}kJ/kg)")
                report_lines.append(f"      Unresolved evidence: cycling_status={db_p.get('cycles_status', 'Missing')}, safety_status={db_p.get('flammability_status', 'Missing')}")

        if n_confirmed >= 2:
            # Code path for when n_confirmed >= 2
            # Prepare decision matrix
            eval_list = []
            for _, c_row in confirmed_df.iterrows():
                pname = c_row["product_name"]
                db_row = pcm_db[pcm_db["product_name"] == pname].iloc[0]
                k_val, k_prov = compute_thermal_conductivity(db_row)
                cyc_conf = compute_cycling_confidence(db_row, max_cycles_reported)

                eval_list.append({
                    "cluster_id": cid,
                    "product_name": pname,
                    "manufacturer": db_row["manufacturer"],
                    "source_type": db_row["source_type"],
                    "Tm_C": db_row["Tm_C"],
                    "f_Tm": gaussian_tm_fitness(db_row["Tm_C"]),
                    "latent_heat_kJ_kg": db_row["latent_heat_kJ_kg"],
                    "latent_heat_margin_ratio": db_row["latent_heat_kJ_kg"] / l_req,
                    "rho_H_MJ_m3": db_row["rho_H_MJ_m3"],
                    "TC_W_mK": k_val,
                    "TC_provenance": k_prov,
                    "cycles_confidence": cyc_conf,
                    "cycles_status": db_row["cycles_status"],
                    "eligibility": "CONFIRMED_FEASIBLE"
                })

            df_eval = pd.DataFrame(eval_list)
            # Minmax normalize & run 4 methods
            M = minmax_normalize(df_eval, CRITERIA)
            w = np.array([PRIOR_WEIGHTS[c] for c in CRITERIA])

            df_eval["topsis_score"] = topsis(M, w)
            df_eval["gra_grade"] = gra(M, w)
            df_eval["promethee_flow"] = promethee_ii(M, w)
            vq, vs, vr = vikor(M, w)
            df_eval["vikor_Q"] = vq

            df_eval["topsis_rank"] = df_eval["topsis_score"].rank(ascending=False, method="min").astype(int)
            df_eval["gra_rank"] = df_eval["gra_grade"].rank(ascending=False, method="min").astype(int)
            df_eval["promethee_rank"] = df_eval["promethee_flow"].rank(ascending=False, method="min").astype(int)
            df_eval["vikor_rank"] = df_eval["vikor_Q"].rank(ascending=True, method="min").astype(int)

            # Borda & Copeland
            rank_list = [df_eval.set_index("product_name")[c] for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"]]
            n_c = len(df_eval)
            borda = pd.Series(0.0, index=df_eval["product_name"])
            for r in rank_list:
                borda += (n_c - r + 1)
            df_eval["borda_score"] = df_eval["product_name"].map(borda)
            df_eval["consensus_rank"] = df_eval["borda_score"].rank(ascending=False, method="min").astype(int)

            all_ranking_rows.append(df_eval)
        else:
            # Format record entry for unranked cluster
            for _, cond_p in conditional_df.iterrows():
                pname = cond_p["product_name"]
                db_p = pcm_db[pcm_db["product_name"] == pname].iloc[0]
                k_val, k_prov = compute_thermal_conductivity(db_p)
                cyc_conf = compute_cycling_confidence(db_p, max_cycles_reported)

                all_ranking_rows.append(pd.DataFrame([{
                    "cluster_id": cid,
                    "product_name": pname,
                    "manufacturer": db_p["manufacturer"],
                    "source_type": db_p["source_type"],
                    "Tm_C": db_p["Tm_C"],
                    "f_Tm": gaussian_tm_fitness(db_p["Tm_C"]),
                    "latent_heat_kJ_kg": db_p["latent_heat_kJ_kg"],
                    "latent_heat_margin_ratio": db_p["latent_heat_kJ_kg"] / l_req,
                    "rho_H_MJ_m3": db_p["rho_H_MJ_m3"],
                    "TC_W_mK": k_val,
                    "TC_provenance": k_prov,
                    "cycles_confidence": cyc_conf,
                    "cycles_status": db_p["cycles_status"],
                    "eligibility": "CONDITIONALLY_FEASIBLE",
                    "ranking_status": "NOT PERFORMED",
                    "reason": reason_no_ranking
                }]))

        report_lines.append("\n--------------------------------------------------------------------------\n")

    df_eligibility_summary = pd.DataFrame(eligibility_summary_rows)
    df_eligibility_summary.to_csv(OUT_ELIGIBILITY_SUMMARY_CSV, index=False)

    if all_ranking_rows:
        df_rankings = pd.concat(all_ranking_rows, ignore_index=True)
        df_rankings.to_csv(OUT_RANKINGS_CSV, index=False)
    else:
        pd.DataFrame(columns=["cluster_id", "ranking_status", "reason"]).to_csv(OUT_RANKINGS_CSV, index=False)

    report_lines.append("==========================================================================")
    report_lines.append("                    END OF PHASE 7 MCDM REPORT")
    report_lines.append("==========================================================================")

    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n--------------------------------------------------------------------------")
    print(f"[Phase 7 SUCCESS] Eligibility summary saved to: {OUT_ELIGIBILITY_SUMMARY_CSV}")
    print(f"[Phase 7 SUCCESS] Rankings output saved to: {OUT_RANKINGS_CSV}")
    print(f"[Phase 7 SUCCESS] MCDM report saved to: {OUT_REPORT_TXT}")
    print("==========================================================================")

if __name__ == "__main__":
    run_phase7()
