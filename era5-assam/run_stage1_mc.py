"""
run_stage1_mc.py  -- Assam Stage-1 Monte Carlo Stability Recovery
==========================================================================
PURPOSE
-------
Recover the Objective 1 Stage-1 Monte Carlo robustness results for all
3 Assam climate regimes using the ORIGINAL Stage-1 MCDM candidate pool
(Tm_target ~ 40 C; RT44HC / RT45HC / C22H46 / n-Docosane baseline).

IMPORTANT - DO NOT OVERWRITE:
  - feasibility_survivors_assam.csv   (Stage-2 Tm=56 C file)
  - mcdm_full_scores_assam.csv        (Stage-2 Tm=56 C file)
  - monte_carlo_stability_assam.csv   (Stage-2 Tm=56 C file)

OUTPUT (new file, does not overwrite anything):
  data/processed/pcm/monte_carlo_stability_assam_stage1.csv

STAGE-1 TARGET TEMPERATURE RECONCILIATION
------------------------------------------
table07_historical_mcdm_rankings_k4.csv records Tm_target_C = 39.94 C
for the Stage-1 Cluster 2 result.

The Stage-1 pipeline used a CLIMATE-BASED Tm_target formula:
    Tm_target = Ta_p95_cluster_mean + 10 K

rather than the fixed indirect-system rule (50 + 6 = 56 C).

Feasibility window: [Tm_target - 6, Tm_target + 8]
  -> RT44HC (43 C), RT45HC (47 C), C22H46 (44.5 C) all survive.
  -> RT54HC (54 C) does NOT survive -> confirms Stage-2 pool separation.

MONTE CARLO PROCEDURE (identical to 08_mcdm_ranking.py):
  - 5,000 Dirichlet weight draws (alpha=30 x w_final)
  - Property perturbations: Tm +/-1K, L +/-5%, TC +/-10%, rho_H +/-8%
  - TOPSIS re-ranking per draw
  - Top-1 retention rate and Top-3 inclusion probability computed
  - Spearman rho vs baseline recorded
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# -- Paths --
BASE_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PCM_DB_FILE       = PROCESSED_DIR / "pcm" / "pcm_database_final.csv"
CLUSTER_PROF_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_assam.csv"
TABLE07_FILE      = BASE_DIR / "final_outputs" / "tables" / "table07_historical_mcdm_rankings_k4.csv"
CLIMATE_SIG_FILE  = PROCESSED_DIR / "climate_signatures_raw.csv"
ASSIGN_FILE       = PROCESSED_DIR / "clustering" / "gmm_cluster_assignments.csv"

OUT_MC_STAGE1 = PROCESSED_DIR / "pcm" / "monte_carlo_stability_assam_stage1.csv"

# -- Stage-1 Constants (matching 08_mcdm_ranking.py exactly) --
N_MONTE_CARLO_DRAWS        = 5_000
MC_DIRICHLET_CONCENTRATION = 30.0
MC_TM_STD_K                = 1.0
MC_RELATIVE_STD            = {"latent_heat_kJ_kg": 0.05, "TC_W_mK": 0.10, "rho_H_MJ_m3": 0.08}
MC_RANDOM_SEED             = 42

SIGMA_LOWER = 4.0
SIGMA_UPPER = 1.5

WINDOW_LOWER_OFFSET  = 6.0
WINDOW_UPPER_OFFSET  = 8.0
LATENT_HEAT_FRACTION = 0.7
ABSOLUTE_TM_MIN      = 42.0
ABSOLUTE_TM_MAX      = 70.0
CYCLES_FLOOR         = 300
SUPERCOOLING_MAX_K   = 8.0
DT_APPROACH_STAGE1   = 10.0

AHP_PRIOR_BASE = {
    "f_Tm":              0.24 / 0.80,
    "latent_heat_ratio": 0.20 / 0.80,
    "rho_H_MJ_m3":       0.12 / 0.80,
    "TC_W_mK":           0.13 / 0.80,
    "cycles_confidence": 0.11 / 0.80,
}
CRITERIA = list(AHP_PRIOR_BASE.keys())


def gaussian_tm_fitness(tm, tm_target):
    sigma = np.where(tm <= tm_target, SIGMA_LOWER, SIGMA_UPPER)
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


def minmax_normalize(df_sub, cols):
    M = df_sub[cols].copy()
    for c in cols:
        lo, hi = M[c].min(), M[c].max()
        M[c] = (M[c] - lo) / (hi - lo) if hi > lo else 0.5
    return M.fillna(0.0).values


def entropy_weights(matrix):
    X = matrix.copy()
    col_sums = X.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1e-12, col_sums)
    P = X / col_sums
    n = X.shape[0]
    k = 1.0 / np.log(n) if n > 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -k * np.nansum(np.where(P > 0, P * np.log(P), 0), axis=0)
    d = 1 - e
    return d / d.sum() if d.sum() > 0 else np.ones(len(d)) / len(d)


def topsis(matrix, weights):
    norm     = matrix / (np.sqrt((matrix ** 2).sum(axis=0)) + 1e-12)
    weighted = norm * weights
    v_plus   = weighted.max(axis=0)
    v_minus  = weighted.min(axis=0)
    s_plus   = np.sqrt(((weighted - v_plus)  ** 2).sum(axis=1))
    s_minus  = np.sqrt(((weighted - v_minus) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)


def compute_final_weights(base_matrix, ahp_prior):
    ew = entropy_weights(base_matrix)
    keys = list(ahp_prior.keys())
    ahp = np.array([ahp_prior[k] for k in keys])
    ahp = ahp / ahp.sum()
    combined = 0.5 * ahp + 0.5 * ew
    return combined / combined.sum()


def run_monte_carlo_stage1(df_cluster, tm_target, l_required, rng):
    names   = df_cluster["name"].tolist()
    n_cand  = len(names)
    df_cluster = df_cluster.copy().reset_index(drop=True)

    df_cluster["f_Tm"] = gaussian_tm_fitness(df_cluster["Tm_C"].values, tm_target)
    df_cluster["latent_heat_ratio"] = df_cluster["latent_heat_kJ_kg"] / l_required

    base_matrix = minmax_normalize(df_cluster, CRITERIA)
    w_final     = compute_final_weights(base_matrix, AHP_PRIOR_BASE)

    baseline_scores = topsis(base_matrix, w_final)
    baseline_rank   = pd.Series(baseline_scores, index=names).rank(
                          ascending=False, method="min")

    top3_count    = {n: 0 for n in names}
    top1_count    = {n: 0 for n in names}
    spearman_rhos = []

    alpha = np.clip(w_final, 1e-6, None) * MC_DIRICHLET_CONCENTRATION

    for _ in range(N_MONTE_CARLO_DRAWS):
        w_draw   = rng.dirichlet(alpha)
        tm_draw  = df_cluster["Tm_C"].values + rng.normal(0, MC_TM_STD_K, n_cand)
        l_draw   = df_cluster["latent_heat_kJ_kg"].values * (
                   1 + rng.normal(0, MC_RELATIVE_STD["latent_heat_kJ_kg"], n_cand))
        tc_draw  = df_cluster["TC_W_mK"].values * (
                   1 + rng.normal(0, MC_RELATIVE_STD["TC_W_mK"], n_cand))
        rho_draw = df_cluster["rho_H_MJ_m3"].values * (
                   1 + rng.normal(0, MC_RELATIVE_STD["rho_H_MJ_m3"], n_cand))

        draw_df = pd.DataFrame({
            "f_Tm":               gaussian_tm_fitness(tm_draw, tm_target),
            "latent_heat_ratio":  l_draw / l_required,
            "rho_H_MJ_m3":        rho_draw,
            "TC_W_mK":            tc_draw,
            "cycles_confidence":  df_cluster["cycles_confidence"].values,
        })
        draw_matrix = minmax_normalize(draw_df, CRITERIA)
        draw_scores = topsis(draw_matrix, w_draw)
        draw_rank   = pd.Series(draw_scores, index=names).rank(
                          ascending=False, method="min")

        for name in draw_rank[draw_rank <= 3].index:
            top3_count[name] += 1
        for name in draw_rank[draw_rank == 1].index:
            top1_count[name] += 1

        rho, _ = spearmanr(baseline_rank.values, draw_rank.values)
        spearman_rhos.append(rho if rho == rho else 0.0)

    result = pd.DataFrame({
        "name":                          names,
        "top3_inclusion_probability":    [top3_count[n] / N_MONTE_CARLO_DRAWS for n in names],
        "top1_retention_rate":           [top1_count[n] / N_MONTE_CARLO_DRAWS for n in names],
        "mean_spearman_rho_vs_baseline": float(np.mean(spearman_rhos)),
        "n_draws":                       N_MONTE_CARLO_DRAWS,
    })
    return result.sort_values("top3_inclusion_probability", ascending=False).reset_index(drop=True)


def main():
    print("=" * 72)
    print("  Assam Stage-1 Monte Carlo Recovery")
    print("  Objective 1 / Historical MCDM baseline / Tm~40 C")
    print("=" * 72)

    pcm_db   = pd.read_csv(PCM_DB_FILE)
    profiles = pd.read_csv(CLUSTER_PROF_FILE)
    raw_sigs = pd.read_csv(CLIMATE_SIG_FILE)
    assigns  = pd.read_csv(ASSIGN_FILE)
    table07  = pd.read_csv(TABLE07_FILE)

    merged = pd.merge(assigns, raw_sigs, on="point_id")

    if "product_name" in pcm_db.columns and "name" not in pcm_db.columns:
        pcm_db = pcm_db.rename(columns={"product_name": "name"})

    CP_WATER    = 4186.0
    T_DELIVERY  = 50.0
    M_DRAW_KG   = 100.0
    PCM_MASS_KG = 50.0
    SHARE_PCM   = 0.5

    cluster_specs = {}
    for cid in sorted(merged["cluster"].unique()):
        sub         = merged[merged["cluster"] == cid]
        ta_p95_mean = sub["Ta_p95"].mean()
        ta_mean     = sub["Ta_mean"].mean()
        tm_target   = ta_p95_mean + DT_APPROACH_STAGE1
        t_mains     = max(5.0, ta_mean - 6.0)
        q_kWh       = (M_DRAW_KG * CP_WATER * (T_DELIVERY - t_mains)) / 3_600_000.0
        l_req       = (SHARE_PCM * q_kWh * 3600.0) / PCM_MASS_KG

        cluster_specs[cid] = {
            "tm_target": tm_target, "l_required": l_req,
            "ta_p95_mean": ta_p95_mean, "ta_mean": ta_mean,
        }
        print(f"\n  Cluster {cid}: Ta_p95_mean={ta_p95_mean:.3f} C  "
              f"Tm_target(Stage-1)={tm_target:.4f} C  "
              f"L_required={l_req:.2f} kJ/kg")

    c2_target  = cluster_specs[2]["tm_target"]
    t07_target = table07["Tm_target_C"].iloc[0]
    delta      = abs(c2_target - t07_target)
    print(f"\n  VERIFICATION -- Cluster 2 Tm_target:")
    print(f"    Computed (Ta_p95+10): {c2_target:.4f} C")
    print(f"    table07 stored:      {t07_target:.4f} C")
    print(f"    Difference:          {delta:.4f} C  ({'OK' if delta < 0.5 else 'WARNING'})")

    # Feasibility
    print("\n" + "-" * 72)
    print("  Stage-1 Feasibility Filtering")
    print("-" * 72)

    hsi_values = [merged[merged["cluster"] == cid]["HSI"].mean()
                  for cid in sorted(merged["cluster"].unique())]
    hsi_p75    = np.percentile(hsi_values, 75)

    stage1_candidates = {}
    for cid in sorted(cluster_specs.keys()):
        spec = cluster_specs[cid]
        tm   = spec["tm_target"]
        lreq = spec["l_required"]
        lo   = tm - WINDOW_LOWER_OFFSET
        hi   = tm + WINDOW_UPPER_OFFSET

        sub         = merged[merged["cluster"] == cid]
        hsi_cluster = sub["HSI"].mean()

        df = pcm_db.copy()

        if "cycles_tested" in df.columns:
            df["cycles_confidence"] = np.where(
                df["cycles_tested"].notna(),
                np.where(df["cycles_tested"] >= CYCLES_FLOOR, 1.0,
                         df["cycles_tested"] / CYCLES_FLOOR),
                0.8)
        else:
            df["cycles_confidence"] = 0.9

        if "rho_H_MJ_m3" not in df.columns:
            if "density_liquid_kg_m3" in df.columns:
                df["rho_H_MJ_m3"] = (df["density_liquid_kg_m3"] *
                                      df["latent_heat_kJ_kg"]) / 1e3
            else:
                df["rho_H_MJ_m3"] = np.nan

        if "TC_W_mK" not in df.columns:
            tc_col = [c for c in df.columns if "TC" in c or "conductivity" in c.lower()]
            if tc_col:
                df["TC_W_mK"] = df[tc_col[0]]
            else:
                df["TC_W_mK"] = 0.20

        pass_window  = df["Tm_C"].between(lo, hi)
        pass_abs     = df["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX)
        pass_latent  = df["latent_heat_kJ_kg"] >= LATENT_HEAT_FRACTION * lreq

        if "cycles_tested" in df.columns:
            cyc_known    = df["cycles_tested"].notna()
            pass_cycling = np.where(cyc_known, df["cycles_tested"] >= CYCLES_FLOOR, True)
        else:
            pass_cycling = np.ones(len(df), dtype=bool)

        if "supercooling_K" in df.columns:
            sc_known = df["supercooling_K"].notna()
            pass_sc  = np.where(sc_known,
                                df["supercooling_K"].abs() <= SUPERCOOLING_MAX_K, True)
        else:
            pass_sc = np.ones(len(df), dtype=bool)

        if "flammable" in df.columns:
            flam = df["flammable"].astype(str).str.lower()
            pass_safety = ~flam.str.contains(
                "highly flammable|extremely flammable|toxic", na=False)
        else:
            pass_safety = np.ones(len(df), dtype=bool)

        if "corrosion_class" in df.columns:
            pass_corr = ~((df["corrosion_class"] == "check_manually") &
                           (hsi_cluster > hsi_p75))
        else:
            pass_corr = np.ones(len(df), dtype=bool)

        passes_all = (pass_window & pass_abs & pass_latent &
                      pass_cycling & pass_sc & pass_safety & pass_corr)
        survivors = df[passes_all].copy()
        survivors["cluster_id"]  = cid
        survivors["Tm_target_C"] = tm
        survivors["L_required"]  = lreq

        survivors["rho_H_MJ_m3"] = survivors["rho_H_MJ_m3"].fillna(
            survivors["latent_heat_kJ_kg"] * 0.8 / 1e3)
        survivors["TC_W_mK"] = survivors["TC_W_mK"].fillna(0.20)

        stage1_candidates[cid] = survivors
        print(f"\n  Cluster {cid}: Tm_window=[{lo:.2f}, {hi:.2f}] C  "
              f"->  {len(survivors)} survivors")
        if len(survivors) > 0:
            print("     " + ", ".join(survivors["name"].tolist()))

    # Monte Carlo
    print("\n" + "-" * 72)
    print("  Running 5,000-draw Monte Carlo per cluster")
    print("-" * 72)

    rng = np.random.default_rng(MC_RANDOM_SEED)
    all_mc_rows = []

    for cid in sorted(stage1_candidates.keys()):
        survivors = stage1_candidates[cid]
        spec      = cluster_specs[cid]
        tm        = spec["tm_target"]
        lreq      = spec["l_required"]

        if len(survivors) < 2:
            print(f"\n  Cluster {cid}: SKIPPED ({len(survivors)} survivor(s))")
            continue

        print(f"\n  Cluster {cid}: {len(survivors)} candidates -> running MC ...")
        mc_result = run_monte_carlo_stage1(survivors, tm, lreq, rng)
        mc_result.insert(0, "cluster_id", cid)
        mc_result["Tm_target_stage1_C"] = round(tm, 4)
        mc_result["L_required_kJ_kg"]   = round(lreq, 4)
        mc_result["stage_label"] = "Objective 1 Stage-1 / historical MCDM baseline"
        all_mc_rows.append(mc_result)

        print(f"    Rank-1 PCM: {mc_result.iloc[0]['name']}")
        print(f"    Top-1 retention : {mc_result.iloc[0]['top1_retention_rate']:.4f}")
        print(f"    Top-3 inclusion : {mc_result.iloc[0]['top3_inclusion_probability']:.4f}")
        print(f"    Mean Spearman   : {mc_result.iloc[0]['mean_spearman_rho_vs_baseline']:.4f}")
        print()
        for _, row in mc_result.head(5).iterrows():
            print(f"      {row['name']:38s}  "
                  f"Top3={row['top3_inclusion_probability']*100:.1f}%  "
                  f"Top1={row['top1_retention_rate']*100:.1f}%")

    if not all_mc_rows:
        print("\n  ERROR: No clusters produced MC output.")
        return

    final_df = pd.concat(all_mc_rows, ignore_index=True)
    final_df.to_csv(OUT_MC_STAGE1, index=False)
    print(f"\n  Saved: {OUT_MC_STAGE1}")

    # Cross-check with table07
    print("\n" + "-" * 72)
    print("  CROSS-CHECK with table07 (Cluster 2 only)")
    print("-" * 72)
    t07_cols = ["name", "consensus_rank",
                "top3_inclusion_probability", "top1_retention_rate"]
    print("\n  table07 (Stage-1 historical Cluster 2):")
    print(table07[t07_cols].to_string(index=False))

    c2_mc = final_df[final_df["cluster_id"] == 2][
        ["name", "top3_inclusion_probability", "top1_retention_rate"]].head(7)
    print("\n  New MC (Cluster 2):")
    print(c2_mc.to_string(index=False))

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY FOR PAPER TABLE")
    print("=" * 72)
    for cid in sorted(final_df["cluster_id"].unique()):
        sub = final_df[final_df["cluster_id"] == cid]
        r1  = sub.iloc[0]
        print(f"\n  Regime {cid} (Tm_target={r1['Tm_target_stage1_C']:.2f} C):")
        for rank_i, (_, row) in enumerate(sub.head(3).iterrows(), 1):
            print(f"    Rank-{rank_i}: {row['name']:35s}  "
                  f"Top3={row['top3_inclusion_probability']*100:.1f}%  "
                  f"Top1={row['top1_retention_rate']*100:.1f}%")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
