"""
08b_monte_carlo_stability_final.py  - Assam SWH Project
==========================================================================
PHASE 8 — REUSABLE MONTE CARLO STABILITY FRAMEWORK

Performs 5,000-draw Monte Carlo sensitivity & stability analysis on MCDM
rankings for eligible clusters (n_confirmed >= 2).

RULES IMPLEMENTED:
  1. Checks Phase 7 eligibility output. Performs 5,000 draws ONLY if
     n_confirmed >= 2 and baseline MCDM ranking exists.
  2. If n_confirmed < 2, logs:
     "Monte Carlo stability analysis skipped due to insufficient eligible candidates (n < 2)."
  3. Conditional or Infeasible PCMs are NEVER included.
  4. For eligible clusters (n >= 2), perturbs:
     - MCDM criterion weights (Dirichlet concentration alpha=30)
     - Tm (+/- 1 K Gaussian noise)
     - Latent heat (+/- 5% relative Gaussian noise)
     - Thermal conductivity (+/- 10% relative Gaussian noise)
     - Volumetric latent heat (+/- 8% relative Gaussian noise)
     Calculates: Top-1 retention rate, Top-3 inclusion probability,
     and Spearman rank correlation (rho) vs baseline.
  5. Decoupled from downstream physics validation.

INPUTS:
  - data/processed/pcm/mcdm_cluster_eligibility_summary.csv
  - data/processed/pcm/mcdm_rankings_by_cluster.csv

OUTPUTS:
  - data/processed/pcm/monte_carlo_stability_assam.csv
  - data/preprocessed/monte_carlo_stability_report.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ELIGIBILITY_CSV = BASE_DIR / "data" / "processed" / "pcm" / "mcdm_cluster_eligibility_summary.csv"
RANKINGS_CSV = BASE_DIR / "data" / "processed" / "pcm" / "mcdm_rankings_by_cluster.csv"

OUT_PCM_DIR = BASE_DIR / "data" / "processed" / "pcm"
OUT_REPORT_DIR = BASE_DIR / "data" / "preprocessed"

OUT_PCM_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MC_CSV = OUT_PCM_DIR / "monte_carlo_stability_assam.csv"
OUT_REPORT_TXT = OUT_REPORT_DIR / "monte_carlo_stability_report.txt"

N_MONTE_CARLO_DRAWS = 5000
MC_DIRICHLET_CONCENTRATION = 30.0
MC_TM_STD_K = 1.0
MC_RELATIVE_STD = {"latent_heat_kJ_kg": 0.05, "TC_W_mK": 0.10, "rho_H_MJ_m3": 0.08}
RANDOM_SEED = 42

TARGET_TM_C = 44.0
SIGMA_TM = 4.0

PRIOR_WEIGHTS = {
    "f_Tm": 0.30,
    "latent_heat_margin_ratio": 0.25,
    "rho_H_MJ_m3": 0.15,
    "TC_W_mK": 0.15,
    "cycles_confidence": 0.15,
}
CRITERIA = list(PRIOR_WEIGHTS.keys())

def gaussian_tm_fitness(tm, target=TARGET_TM_C, sigma=SIGMA_TM):
    return np.exp(-((tm - target) ** 2) / (2.0 * sigma ** 2))

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

def run_monte_carlo_for_cluster(df_eval, l_req, n_draws=N_MONTE_CARLO_DRAWS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    names = df_eval["product_name"].tolist()
    n_cand = len(names)

    w_baseline = np.array([PRIOR_WEIGHTS[c] for c in CRITERIA])
    base_matrix = minmax_normalize(df_eval, CRITERIA)
    baseline_scores = topsis(base_matrix, w_baseline)
    baseline_rank = pd.Series(baseline_scores, index=names).rank(ascending=False, method="min")

    top3_count = {n: 0 for n in names}
    top1_count = {n: 0 for n in names}
    spearman_rhos = []

    alpha = np.clip(w_baseline, 1e-6, None) * MC_DIRICHLET_CONCENTRATION

    for _ in range(n_draws):
        w_draw = rng.dirichlet(alpha)
        tm_draw = df_eval["Tm_C"].values + rng.normal(0, MC_TM_STD_K, n_cand)
        l_draw = df_eval["latent_heat_kJ_kg"].values * (1 + rng.normal(0, MC_RELATIVE_STD["latent_heat_kJ_kg"], n_cand))
        tc_draw = df_eval["TC_W_mK"].values * (1 + rng.normal(0, MC_RELATIVE_STD["TC_W_mK"], n_cand))
        rho_draw = df_eval["rho_H_MJ_m3"].values * (1 + rng.normal(0, MC_RELATIVE_STD["rho_H_MJ_m3"], n_cand))

        draw_df = pd.DataFrame({
            "f_Tm": gaussian_tm_fitness(tm_draw),
            "latent_heat_margin_ratio": l_draw / l_req,
            "rho_H_MJ_m3": rho_draw,
            "TC_W_mK": tc_draw,
            "cycles_confidence": df_eval["cycles_confidence"].values,
        })

        draw_matrix = minmax_normalize(draw_df, CRITERIA)
        draw_scores = topsis(draw_matrix, w_draw)
        draw_rank = pd.Series(draw_scores, index=names).rank(ascending=False, method="min")

        for name in draw_rank[draw_rank <= 3].index:
            top3_count[name] += 1
        for name in draw_rank[draw_rank == 1].index:
            top1_count[name] += 1

        rho, _ = spearmanr(baseline_rank.values, draw_rank.values)
        spearman_rhos.append(rho if pd.notna(rho) else 0.0)

    res = pd.DataFrame({
        "product_name": names,
        "top1_retention_rate": [top1_count[n] / n_draws for n in names],
        "top3_inclusion_probability": [top3_count[n] / n_draws for n in names],
        "mean_spearman_rho_vs_baseline": [float(np.mean(spearman_rhos))] * n_cand,
        "n_draws": [n_draws] * n_cand,
    })
    return res.sort_values("top1_retention_rate", ascending=False)

def run_phase8_framework():
    print("==========================================================================")
    print("        PHASE 8 — MONTE CARLO STABILITY FRAMEWORK")
    print("==========================================================================")

    if not ELIGIBILITY_CSV.exists() or not RANKINGS_CSV.exists():
        raise FileNotFoundError(f"Phase 7 outputs missing ({ELIGIBILITY_CSV} or {RANKINGS_CSV})")

    df_elig = pd.read_csv(ELIGIBILITY_CSV)
    df_rank = pd.read_csv(RANKINGS_CSV)

    mc_csv_rows = []
    report_lines = []

    report_lines.append("==========================================================================")
    report_lines.append("        PHASE 8 — MONTE CARLO STABILITY ANALYSIS REPORT")
    report_lines.append("==========================================================================")
    report_lines.append("Framework Rules Enforced:")
    report_lines.append("  1. Performed ONLY for clusters with n_confirmed >= 2 and baseline MCDM ranking.")
    report_lines.append("  2. If n_confirmed < 2, stability analysis is skipped with explicit statement.")
    report_lines.append("  3. Conditional or Infeasible PCMs are NEVER included.")
    report_lines.append("  4. 5,000 draws perturbing weights (Dirichlet alpha=30), Tm (+/-1K), L (+/-5%),")
    report_lines.append("     k (+/-10%), and rho_H (+/-8%).")
    report_lines.append("")

    for _, elig_row in df_elig.iterrows():
        cid = int(elig_row["cluster_id"])
        l_req = elig_row["L_required_kJ_kg"]
        n_confirmed = int(elig_row["confirmed_feasible_count"])
        mcdm_status = elig_row["mcdm_ranking_status"]

        print(f"Cluster {cid} (L_req = {l_req:.1f} kJ/kg):")
        print(f"  - Confirmed Feasible candidates (n_confirmed): {n_confirmed}")
        print(f"  - Baseline MCDM Status                       : {mcdm_status}")

        if n_confirmed >= 2 and mcdm_status == "PERFORMED":
            print(f"  -> Executing 5,000 Monte Carlo draws ...")
            c_rank = df_rank[(df_rank["cluster_id"] == cid) & (df_rank["eligibility"] == "CONFIRMED_FEASIBLE")]
            mc_res = run_monte_carlo_for_cluster(c_rank, l_req)
            mc_res.insert(0, "cluster_id", cid)
            mc_res["status"] = "COMPLETED"
            mc_res["skip_reason"] = "None"
            mc_csv_rows.append(mc_res)

            report_lines.append(f"CLUSTER {cid} MONTE CARLO STABILITY RESULTS (5,000 Draws):")
            report_lines.append(f"  Confirmed Candidates Sized: {n_confirmed}")
            report_lines.append(f"  Mean Spearman Rank Correlation (rho): {mc_res['mean_spearman_rho_vs_baseline'].iloc[0]:.4f}")
            for _, r in mc_res.iterrows():
                report_lines.append(f"    - {r['product_name']}: Top-1 Retention = {r['top1_retention_rate']*100:.1f}%, Top-3 Inclusion = {r['top3_inclusion_probability']*100:.1f}%")
        else:
            skip_msg = "Monte Carlo stability analysis skipped due to insufficient eligible candidates (n < 2)."
            print(f"  -> Status: SKIPPED")
            print(f"  -> Statement: {skip_msg}")

            mc_csv_rows.append(pd.DataFrame([{
                "cluster_id": cid,
                "product_name": "None",
                "top1_retention_rate": np.nan,
                "top3_inclusion_probability": np.nan,
                "mean_spearman_rho_vs_baseline": np.nan,
                "n_draws": 0,
                "status": "SKIPPED",
                "skip_reason": skip_msg
            }]))

            report_lines.append(f"CLUSTER {cid} MONTE CARLO STABILITY ANALYSIS:")
            report_lines.append(f"  Confirmed Candidates: {n_confirmed}")
            report_lines.append(f"  Status:               SKIPPED")
            report_lines.append(f"  Statement:            {skip_msg}")

        report_lines.append("\n--------------------------------------------------------------------------\n")

    df_mc_out = pd.concat(mc_csv_rows, ignore_index=True)
    df_mc_out.to_csv(OUT_MC_CSV, index=False)

    report_lines.append("==========================================================================")
    report_lines.append("                 END OF PHASE 8 MONTE CARLO REPORT")
    report_lines.append("==========================================================================")

    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("--------------------------------------------------------------------------")
    print(f"[Phase 8 SUCCESS] Stability results saved to: {OUT_MC_CSV}")
    print(f"[Phase 8 SUCCESS] Stability report saved to: {OUT_REPORT_TXT}")
    print("==========================================================================")

if __name__ == "__main__":
    run_phase8_framework()
