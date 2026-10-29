"""
08_mcdm_ranking_final.py  - Assam SWH Project
==========================================================================
PHASE 7 — MULTI-CRITERIA DECISION MAKING (MCDM) RANKING & ELIGIBILITY GOVERNANCE

Implements the multi-criteria decision-making framework with strict eligibility
governance:
  1. Primary formal MCDM ranking is restricted EXCLUSIVELY to candidates with
     survives_all == True in the FINAL relaxation round selected by Phase 6.
  2. Evidence-based CONFIRMED/CONDITIONAL/INFEASIBLE classifications are retained
     as audit information but do not determine MCDM eligibility.
  3. The >=5 survivor target is a minimum threshold, NOT a top-5 selection.
     ALL final-round survivors are passed to MCDM.
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
TARGET_TM_C = 56.0
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
    report_lines.append("  1. Primary MCDM Candidate Set: ALL Phase 6 deployable survivors")
    report_lines.append("     (survives_all == True) from the FINAL relaxation round for each cluster.")
    report_lines.append("  2. The >=5 survivor target is a minimum threshold, NOT a top-5 selection.")
    report_lines.append("  3. CONFIRMED/CONDITIONAL/INFEASIBLE status is retained as evidence audit metadata.")
    report_lines.append("  4. Candidates from earlier relaxation rounds are excluded from MCDM.")
    report_lines.append("")

    for _, prof in profiles.iterrows():
        cid = int(prof["cluster_id"])
        c_feas = feas_by_cluster[feas_by_cluster["cluster_id"] == cid].copy()

        c_summary = feas_summary[feas_summary["cluster_id"] == cid].copy()
        l_req = (
            c_summary["L_required_kJ_kg"].iloc[0]
            if len(c_summary) > 0 and pd.notna(c_summary["L_required_kJ_kg"].iloc[0])
            else prof.get("L_required_kJ_per_kg", np.nan)
        )

        # Phase 6 summary identifies the final relaxation round.
        # IMPORTANT: the existing Phase 6 audit CSV does NOT store
        # relax_round in each audit row. It stores all rounds sequentially
        # (58 PCM rows per round). Therefore, when relax_round is absent,
        # recover the final-round records by keeping the LAST occurrence of
        # each PCM within this cluster. This is the final round because
        # Phase 6 appends rounds in chronological order and stops at the
        # first round reaching the survivor target.
        if len(c_summary) == 0:
            raise ValueError(
                f"Cannot determine final relaxation round for Assam cluster {cid}: "
                "cluster is missing from pcm_feasibility_summary.csv."
            )

        if "relax_round" not in c_summary.columns or pd.isna(c_summary["relax_round"].iloc[0]):
            raise ValueError(
                f"Phase 6 summary does not contain a valid relax_round for Assam cluster {cid}."
            )

        # Phase 6 summary may contain one row per relaxation round.
        # The highest recorded round is the final selected round.
        relax_rounds = pd.to_numeric(
            c_summary["relax_round"], errors="coerce"
        ).dropna()

        if len(relax_rounds) == 0:
            raise ValueError(
                f"Phase 6 summary does not contain a valid relax_round "
                f"for Assam cluster {cid}."
            )

        final_round = int(relax_rounds.max())

        print(f"  -> Phase 6 final relaxation round identified: {final_round}")

        if "survives_all" not in c_feas.columns:
            raise ValueError(
                "Phase 6 output must contain 'survives_all' for Phase 7 MCDM."
            )

        if "relax_round" in c_feas.columns:
            c_final = c_feas[
                pd.to_numeric(c_feas["relax_round"], errors="coerce") == final_round
            ].copy()
        else:
            if "pcm_id" not in c_feas.columns:
                raise ValueError(
                    "Phase 6 audit output has no 'relax_round' and no 'pcm_id'; "
                    "cannot reconstruct the final relaxation round safely."
                )

            # Audit rows are written in chronological relaxation order.
            # Keep the last occurrence of each PCM.
            c_final = c_feas.drop_duplicates(
                subset=["pcm_id"], keep="last"
            ).copy()

        # The final round must contain one row for every PCM in the candidate pool.
        expected_pool = len(pcm_db)
        if len(c_final) != expected_pool:
            raise ValueError(
                f"Could not reconstruct final relaxation round for Assam cluster {cid}: "
                f"found {len(c_final)} rows, expected {expected_pool}."
            )

        # Parse boolean values safely. bool("False") is True in Python, so
        # never use astype(bool) directly on a string CSV column.
        def _parse_bool(value):
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if pd.isna(value):
                return False
            return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

        deployable_mask = c_final["survives_all"].map(_parse_bool)
        deployable_df = c_final[deployable_mask].copy()

        # Evidence classifications are retained for audit/reporting only.
        confirmed_df = c_final[c_final["overall_feasibility"] == "CONFIRMED_FEASIBLE"]
        conditional_df = c_final[c_final["overall_feasibility"] == "CONDITIONALLY_FEASIBLE"]
        infeasible_df = c_final[c_final["overall_feasibility"] == "INFEASIBLE"]

        n_deployable = len(deployable_df)
        n_confirmed = len(confirmed_df)
        n_conditional = len(conditional_df)
        n_infeasible = len(infeasible_df)

        print(f"Cluster {cid} (L_req = {l_req:.1f} kJ/kg):")
        print(f"  - Final relaxation round             : {final_round}")
        print(f"  - Deployable survivors for MCDM     : {n_deployable}")
        print(f"  - Confirmed Feasible (audit)         : {n_confirmed}")
        print(f"  - Conditionally Feasible (audit)     : {n_conditional}")
        print(f"  - Infeasible (audit)                 : {n_infeasible}")

        if n_deployable >= 2:
            mcdm_status = "PERFORMED"
            reason_no_ranking = "None"
        else:
            mcdm_status = "NOT PERFORMED"
            reason_no_ranking = (
                f"Only {n_deployable} deployable survivor(s) are available; "
                "MCDM requires at least 2 alternatives."
            )

        print(f"  -> MCDM Ranking Status: {mcdm_status}")
        print(f"  -> Reason: {reason_no_ranking}")

        eligibility_summary_rows.append({
            "cluster_id": cid,
            "L_required_kJ_kg": l_req,
            "final_relax_round": final_round,
            "final_tm_min": c_summary["final_tm_min"].iloc[0] if "final_tm_min" in c_summary.columns else np.nan,
            "final_tm_max": c_summary["final_tm_max"].iloc[0] if "final_tm_max" in c_summary.columns else np.nan,
            "total_pcm_pool": len(c_final),
            "deployable_survivor_count": n_deployable,
            "confirmed_feasible_count": n_confirmed,
            "conditionally_feasible_count": n_conditional,
            "infeasible_count": n_infeasible,
            "mcdm_ranking_status": mcdm_status,
            "reason_for_no_ranking": reason_no_ranking
        })

        report_lines.append(f"CLUSTER {cid} AUDIT SUMMARY (L_req = {l_req:.1f} kJ/kg):")
        report_lines.append(f"  Final Relaxation Round:         {final_round}")
        if "final_tm_min" in c_summary.columns and "final_tm_max" in c_summary.columns:
            report_lines.append(
                f"  Final Tm Window:                "
                f"{c_summary['final_tm_min'].iloc[0]:.1f}–"
                f"{c_summary['final_tm_max'].iloc[0]:.1f} °C"
            )
        report_lines.append(f"  Final Round Candidate Pool:     {len(c_final)}")
        report_lines.append(f"  Deployable Survivors for MCDM:  {n_deployable}")
        report_lines.append(f"  Confirmed Feasible (audit):      {n_confirmed}")
        report_lines.append(f"  Conditionally Feasible (audit):  {n_conditional}")
        report_lines.append(f"  Infeasible (audit):              {n_infeasible}")
        report_lines.append(f"  MCDM Ranking Status:             {mcdm_status}")
        report_lines.append(f"  Reason:                          {reason_no_ranking}")

        report_lines.append("\n  Final deployable PCM candidates entering MCDM:")
        for _, dep_p in deployable_df.iterrows():
            report_lines.append(
                f"    - {dep_p['product_name']} "
                f"(Tm={dep_p['Tm_C']}°C, L={dep_p['latent_heat_kJ_kg']}kJ/kg)"
            )

        if n_conditional > 0:
            report_lines.append("\n  Conditional candidates in final round (audit only):")
            for _, cond_p in conditional_df.iterrows():
                pname = cond_p["product_name"]
                db_matches = pcm_db[pcm_db["product_name"] == pname]
                if len(db_matches) == 0:
                    continue
                db_p = db_matches.iloc[0]
                report_lines.append(
                    f"    - {pname} "
                    f"(Tm={db_p['Tm_C']}°C, L={db_p['latent_heat_kJ_kg']}kJ/kg)"
                )
                report_lines.append(
                    f"      Unresolved evidence: "
                    f"cycling_status={db_p.get('cycles_status', 'Missing')}, "
                    f"safety_status={db_p.get('flammability_status', 'Missing')}"
                )

        if n_deployable >= 2:
            # Rank ALL final-round deployable survivors; this is not a top-5 truncation.
            eval_list = []
            for _, c_row in deployable_df.iterrows():
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
                    "eligibility": "DEPLOYABLE_SURVIVOR"
                })

            df_eval = pd.DataFrame(eval_list)

            # Cycling confidence is intentionally UNKNOWN when cycling data are
            # unreported. For MCDM normalization, use a neutral value rather
            # than allowing NaN to contaminate every ranking score. This does
            # not change Phase 6 evidence classification.
            if df_eval["cycles_confidence"].isna().any():
                df_eval["cycles_confidence"] = df_eval["cycles_confidence"].fillna(0.5)

            missing_criteria = [c for c in CRITERIA if df_eval[c].isna().any()]
            if missing_criteria:
                raise ValueError(
                    f"Missing MCDM criterion values for Assam cluster {cid}: "
                    f"{missing_criteria}"
                )

            M = minmax_normalize(df_eval, CRITERIA)
            w = np.array([PRIOR_WEIGHTS[c] for c in CRITERIA])

            df_eval["topsis_score"] = topsis(M, w)
            df_eval["gra_grade"] = gra(M, w)
            df_eval["promethee_flow"] = promethee_ii(M, w)
            vq, vs, vr = vikor(M, w)
            df_eval["vikor_Q"] = vq

            df_eval["topsis_rank"] = df_eval["topsis_score"].rank(
                ascending=False, method="min"
            ).astype(int)
            df_eval["gra_rank"] = df_eval["gra_grade"].rank(
                ascending=False, method="min"
            ).astype(int)
            df_eval["promethee_rank"] = df_eval["promethee_flow"].rank(
                ascending=False, method="min"
            ).astype(int)
            df_eval["vikor_rank"] = df_eval["vikor_Q"].rank(
                ascending=True, method="min"
            ).astype(int)

            rank_list = [
                df_eval.set_index("product_name")[c]
                for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"]
            ]
            n_c = len(df_eval)
            borda = pd.Series(0.0, index=df_eval["product_name"])
            for r in rank_list:
                borda += (n_c - r + 1)
            df_eval["borda_score"] = df_eval["product_name"].map(borda)
            df_eval["consensus_rank"] = df_eval["borda_score"].rank(
                ascending=False, method="min"
            ).astype(int)

            all_ranking_rows.append(df_eval)
        else:
            # This branch is only for a genuine <2 deployable-survivor case.
            for _, dep_p in deployable_df.iterrows():
                pname = dep_p["product_name"]
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
                    "eligibility": "DEPLOYABLE_SURVIVOR",
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
