"""
11_level_b_seasonal_analysis.py
===================================
PHASE 4, LEVEL B — TEMPORAL/SEASONAL CLUSTERING (plan v3.0 Section 7.2)

Level A (05_cluster_tamilnadu.py) clusters points across all 10 years
combined — "which PCM for a system installed at this location?" Level B
asks the other question the plan explicitly calls out as likely to
produce your single most interesting result: "does this location need a
DIFFERENT PCM in July than in March?" — Tamil Nadu's north-east monsoon
(Oct-Dec) is out of phase with the south-west monsoon most of India runs
on, which is exactly the situation where a seasonal flip is plausible.

The plan calls this "nearly free" because season/season_code are already
columns in your cleaned data from 04_preprocess_tamilnadu.py. This script
does NOT re-run full GMM clustering per season (that's the literal Level
B spec and is a bigger undertaking) — instead it does the cheaper,
still-genuinely-informative version: for each EXISTING Level-A cluster,
recompute the climate-dependent MCDM inputs (Ta_mean -> L_required; the
same Tm_target) SEPARATELY per season, re-run feasibility + a single-method
TOPSIS ranking (using the SAME entropy+AHP-blended weights already
computed for the annual case, for a fair comparison) per (cluster, season),
and report whether the Top-3 changes.

If you want the literal Level-B spec (full per-point-per-season signature
vectors, independently GMM-clustered) — that's a bigger addition; this
script is the "nearly free" version the plan explicitly permits as a
starting point, not a substitute for a genuinely separate seasonal
clustering. Say which version you did in your methodology.

INPUT  : data/preprocessed/tamilnadu_cleaned_physical.csv   (04's output;
           already has a 'season' column: Winter/Summer/Monsoon/Retreat)
         data/processed/clustering/cluster_assignments_tamilnadu.csv
         data/processed/clustering/cluster_profiles_tamilnadu.csv
         data/processed/pcm/pcm_database_tamilnadu.csv
         data/processed/pcm/mcdm_full_scores_by_cluster.csv  (for the
           annual/Level-A weights, so seasonal comparison uses the same
           weight vector as your headline result)
OUTPUT : data/processed/pcm/level_b_seasonal_topk.csv
         data/processed/pcm/level_b_seasonal_summary.md

HOW TO RUN:
  python 11_level_b_seasonal_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PREPROCESSED_DIR, PROCESSED_DIR

PHYSICAL_FILE = PREPROCESSED_DIR / "tamilnadu_cleaned_physical.csv"
ASSIGN_FILE = PROCESSED_DIR / "clustering" / "cluster_assignments_tamilnadu.csv"
PROFILE_FILE = PROCESSED_DIR / "clustering" / "cluster_profiles_tamilnadu.csv"
PCM_FILE = PROCESSED_DIR / "pcm" / "pcm_database_tamilnadu.csv"
SCORES_FILE = PROCESSED_DIR / "pcm" / "mcdm_full_scores_by_cluster.csv"
OUT_CSV = PROCESSED_DIR / "pcm" / "level_b_seasonal_topk.csv"
OUT_MD = PROCESSED_DIR / "pcm" / "level_b_seasonal_summary.md"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7
SIGMA_TM = 4.0
T_DELIVERY_C = 50.0
# BUG FIX v3.1 — match 04b_climate_signature.py (Avargani et al. 2021: 300 L/day)
DRAW_VOLUME_L = 300.0
DRAW_MASS_KG = DRAW_VOLUME_L * 1.0
CP_WATER = 4.186
ASSUMED_PCM_MASS_KG = 50.0
USE_CLIMATE_RELATIVE_LATENT_HEAT = True   # must match 08_mcdm_ranking.py's setting —
                                            # otherwise the weight_ column lookup below
                                            # from 08's output silently mismatches.
LATENT_CRITERION_NAME = ("latent_heat_margin_ratio" if USE_CLIMATE_RELATIVE_LATENT_HEAT
                          else "latent_heat_kJ_kg")
CRITERIA = ["f_Tm", LATENT_CRITERION_NAME, "rho_H_MJ_m3", "TC_W_mK", "cycles_confidence"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]


def gaussian_tm_fitness(tm, tm_target, sigma=SIGMA_TM):
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


def topsis(matrix, weights):
    norm = matrix / (np.sqrt((matrix ** 2).sum(axis=0)) + 1e-12)
    weighted = norm * weights
    v_plus, v_minus = weighted.max(axis=0), weighted.min(axis=0)
    s_plus = np.sqrt(((weighted - v_plus) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - v_minus) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)


def rank_seasonal(pcm_db, tm_target, l_required, weights):
    df = pcm_db.copy()
    lo, hi = tm_target - WINDOW_LOWER_OFFSET, tm_target + WINDOW_UPPER_OFFSET
    survivors = df[df["Tm_C"].between(lo, hi) & df["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX) &
                    (df["latent_heat_kJ_kg"] >= LATENT_HEAT_FRACTION * l_required)].copy()
    if len(survivors) < 2:
        return None

    survivors["f_Tm"] = gaussian_tm_fitness(survivors["Tm_C"], tm_target)
    if USE_CLIMATE_RELATIVE_LATENT_HEAT:
        survivors["latent_heat_margin_ratio"] = survivors["latent_heat_kJ_kg"] / l_required
    survivors["cycles_confidence"] = survivors["cycles_confidence"].fillna(
        survivors["cycles_confidence"].median())

    M = survivors[CRITERIA].copy()
    for c in CRITERIA:
        lo_c, hi_c = M[c].min(), M[c].max()
        M[c] = (M[c] - lo_c) / (hi_c - lo_c) if hi_c > lo_c else 0.5
    M = M.fillna(0.0).values

    survivors["topsis_score"] = topsis(M, weights)
    survivors["seasonal_rank"] = survivors["topsis_score"].rank(ascending=False, method="min").astype(int)
    return survivors.sort_values("seasonal_rank")


def main():
    print("=" * 68)
    print("  Phase 4, Level B — Seasonal PCM Sensitivity — Tamil Nadu")
    print("=" * 68)

    for f in (PHYSICAL_FILE, ASSIGN_FILE, PROFILE_FILE, PCM_FILE, SCORES_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found.")
            return

    physical = pd.read_csv(PHYSICAL_FILE, parse_dates=["date"])
    assign = pd.read_csv(ASSIGN_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    pcm_db = pd.read_csv(PCM_FILE)
    scores = pd.read_csv(SCORES_FILE)

    all_rows = []
    md_lines = ["# Level B — Seasonal PCM Sensitivity (Tamil Nadu)\n"]

    for cid in sorted(assign["cluster_id"].unique()):
        member_points = assign[assign["cluster_id"] == cid]["point_id"].unique()
        cluster_physical = physical[physical["point_id"].isin(member_points)]

        prof = profiles[profiles["cluster_id"] == cid].iloc[0]
        tm_target = (prof["Tm_target_C_regime_capped"]
                     if "Tm_target_C_regime_capped" in prof.index else prof["Tm_target_C"])

        cluster_scores = scores[scores["cluster_id"] == cid]
        if not len(cluster_scores):
            continue
        weight_cols = [f"weight_{c}" for c in CRITERIA if f"weight_{c}" in cluster_scores.columns]
        weights = cluster_scores[weight_cols].iloc[0].values if len(weight_cols) == len(CRITERIA) \
            else np.ones(len(CRITERIA)) / len(CRITERIA)

        annual_top1 = cluster_scores.sort_values("consensus_rank")["name"].iloc[0]
        md_lines.append(f"\n## Cluster {int(cid)}  (annual/Level-A #1: **{annual_top1}**)\n")
        md_lines.append("| Season | #1 PCM | #2 PCM | #3 PCM | Flips from annual? |")
        md_lines.append("|---|---|---|---|---|")

        for season in SEASON_ORDER:
            season_rows = cluster_physical[cluster_physical["season"] == season]
            if season_rows.empty:
                continue
            ta_mean_season = season_rows["era5_T_amb"].mean()
            t_mains_season = ta_mean_season - 2.0
            q_total_kj = DRAW_MASS_KG * CP_WATER * (T_DELIVERY_C - t_mains_season)
            l_required_season = q_total_kj / ASSUMED_PCM_MASS_KG

            ranked = rank_seasonal(pcm_db, tm_target, l_required_season, weights)
            if ranked is None:
                md_lines.append(f"| {season} | (< 2 survivors) | - | - | - |")
                continue

            top3 = ranked.head(3)["name"].tolist()
            while len(top3) < 3:
                top3.append("-")
            flips = "**YES**" if top3[0] != annual_top1 else "No"

            all_rows.append({"cluster_id": cid, "season": season,
                              "Ta_mean_season": ta_mean_season,
                              "L_required_season": l_required_season,
                              "top1": top3[0], "top2": top3[1], "top3": top3[2],
                              "flips_from_annual": top3[0] != annual_top1})
            md_lines.append(f"| {season} | {top3[0]} | {top3[1]} | {top3[2]} | {flips} |")

        print(f"\n  Cluster {int(cid)} (annual #1: {annual_top1}):")
        for r in [r for r in all_rows if r["cluster_id"] == cid]:
            flag = "  <-- FLIPS" if r["flips_from_annual"] else ""
            print(f"    {r['season']:8s}  Ta_mean={r['Ta_mean_season']:.1f}C  "
                  f"L_required={r['L_required_season']:.0f}  #1={r['top1']}{flag}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_MD}")
    if len(result_df):
        n_flips = result_df["flips_from_annual"].sum()
        print(f"\n  {n_flips}/{len(result_df)} (cluster, season) combinations show a "
              f"#1 PCM different from that cluster's annual pick.")
        if n_flips > 0:
            print("  [FINDING] Seasonal flips detected — per plan v3.0 Section 7.2, this "
                  "is direct empirical motivation for the adaptive control objective "
                  "(Objective 3's DRL controller), generated from your own data. Worth "
                  "a dedicated paragraph/figure in your paper.")
        else:
            print("  No flips detected — also a valid finding: it means the corrected "
                  "Tm_target rule (delivery-temperature-anchored, not ambient-anchored) "
                  "is robust to Tamil Nadu's seasonal swings, which is itself worth "
                  "stating as evidence the rule generalizes across your monsoon cycle.")
    print("=" * 68)


if __name__ == "__main__":
    main()
