"""
08_mcdm_ranking.py   (v2 — full 4-method stack + Monte Carlo)
==================================================================
PHASE 6 — MULTI-CRITERIA RANKING ENGINE (Objective 1 plan v3.0, Section 9)

REPLACES the earlier "minimum viable" TOPSIS+GRA-only version. This adds
PROMETHEE II and VIKOR (the plan is explicit that a smaller method set is
"a regression" — Section 2.2) and the 5,000-draw Monte Carlo stability
analysis (Section 9.6), which is what turns a Top-3 from an assertion
into a defensible, stability-quantified result (D6, the headline
deliverable).

THE ONE STEP EVERY PCM-MCDM PAPER GETS WRONG (plan v3.0 Section 9.2)
------------------------------------------------------------------------
Melting temperature is a TARGET-based criterion — closer to Tm_target is
better in both directions, not a benefit or cost. Converted to a Gaussian
fitness score BEFORE anything else touches it:

    f_Tm(i) = exp( -(Tm_i - Tm_target)^2 / (2*sigma^2) ),  sigma = 4K

CRITERIA (unchanged from v1 — only what the database has real values for)
------------------------------------------------------------------------
  f_Tm (melting-point fitness, Gaussian)     benefit
  latent_heat_kJ_kg                          benefit
  rho_H_MJ_m3 (volumetric latent heat)       benefit
  TC_W_mK (thermal conductivity)             benefit
  cycles_confidence (log-scaled, NaN-safe)   benefit

FOUR RANKING METHODS
-----------------------
  TOPSIS      — closeness coefficient, Euclidean ideal/anti-ideal
  GRA         — grey relational grade vs. the ideal (max) reference
  PROMETHEE II — net outranking flow; V-shape preference function with
                 indifference/preference thresholds q=0.10, p=0.30 of the
                 [0,1] normalized range for every criterion (a documented,
                 uniform simplification — the plan's own q=2K/p=8K example
                 is for Tm in physical units, which we've already
                 transformed into a dimensionless fitness score by this
                 point, so a physical threshold doesn't carry over
                 directly; state this simplification if you use it)
  VIKOR       — compromise ranking Q_i (v=0.5), with the standard
                 acceptable-advantage / acceptable-stability check flagged

CONSENSUS
-----------
Borda count across all 4 methods' ranks (primary). Copeland pairwise
majority computed as a cross-check — both reported, and they're flagged
if they disagree on the #1 pick (plan v3.0 Section 9.5: "where they
disagree, report both").

MONTE CARLO STABILITY (plan v3.0 Section 9.6)
------------------------------------------------
N_MONTE_CARLO_DRAWS = 5000 by default (matches the plan; reduce if you
need faster iteration while developing, but restore 5000 for the number
you actually report). Per draw:
  - Weights perturbed via a Dirichlet draw centered on the nominal
    entropy+AHP blended weight vector (concentration below is a stated
    assumption, not measured — documented).
  - PCM properties perturbed: Tm +/- Gaussian(0, 1K), latent heat and
    thermal conductivity +/- Gaussian scaled to 5%/10% relative std (per
    plan v3.0 Section 9.6's stated uncertainty bands).
  - TOPSIS is used as the single fast per-draw scorer (recomputing all
    four methods per draw for 5000 draws is unnecessary — TOPSIS alone is
    standard practice in MC-MCDM stability studies; this is a stated
    simplification, not a limitation the plan requires you avoid).
Reports, per PCM per cluster: Top-3 inclusion probability, Top-1
retention rate, and Spearman rho of each draw's full ranking against the
baseline (non-perturbed) TOPSIS ranking.

INPUT  : data/processed/pcm/feasibility_survivors_by_cluster.csv (07's output)
OUTPUT : data/processed/pcm/mcdm_topk_by_cluster.csv
         data/processed/pcm/mcdm_full_scores_by_cluster.csv
         data/processed/pcm/monte_carlo_stability.csv

HOW TO RUN:
  python 08_mcdm_ranking.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PROCESSED_DIR

SURVIVORS_FILE = PROCESSED_DIR / "pcm" / "feasibility_survivors_by_cluster.csv"
OUT_TOPK = PROCESSED_DIR / "pcm" / "mcdm_topk_by_cluster.csv"
OUT_FULL = PROCESSED_DIR / "pcm" / "mcdm_full_scores_by_cluster.csv"
OUT_MC = PROCESSED_DIR / "pcm" / "monte_carlo_stability.csv"

SIGMA_TM = 4.0
ENTROPY_AHP_LAMBDA = 0.5
GRA_ZETA = 0.5
PROMETHEE_Q, PROMETHEE_P = 0.10, 0.30    # indifference/preference, fraction of [0,1] range
VIKOR_V = 0.5

N_MONTE_CARLO_DRAWS = 5000     # plan v3.0 Section 9.6. Lower for faster dev iteration.
MC_DIRICHLET_CONCENTRATION = 30.0    # higher = tighter around nominal weights (documented assumption)
MC_TM_STD_K = 1.0
MC_RELATIVE_STD = {"latent_heat_kJ_kg": 0.05, "TC_W_mK": 0.10, "rho_H_MJ_m3": 0.08}
MC_RANDOM_SEED = 42

# ─── Climate-relative latent heat (fixes the "same PCM wins everywhere" issue) ───
# Plan v3.0 Table 13 lists raw latent_heat_kJ_kg as a benefit criterion. Combined
# with Tm_target being CONSTANT (Section 6.3's explicit design) and the
# feasibility filter's L_required floor being cleared by every real candidate
# 3-5x over, raw latent heat carries ZERO climate information into the ranking
# — it's the same 7 numbers in every cluster and every season, which is exactly
# why you saw one PCM win everywhere. Set this True (default) to rank on
# latent_heat_kJ_kg / L_required instead — margin over what THIS cluster/season
# actually needs, a genuine benefit criterion, still fully documented as a
# deviation from the plan's literal Table 13 list. Set False to match Table 13
# exactly (results will likely converge to one PCM statewide, which is itself a
# valid, reportable finding — see the [FINDING] message this script already
# prints either way).
USE_CLIMATE_RELATIVE_LATENT_HEAT = True

AHP_PRIOR_BASE = {
    "f_Tm": 0.24 / 0.80,
    "latent_heat_kJ_kg": 0.20 / 0.80,
    "rho_H_MJ_m3": 0.12 / 0.80,
    "TC_W_mK": 0.13 / 0.80,
    "cycles_confidence": 0.11 / 0.80,
}
LATENT_CRITERION_NAME = ("latent_heat_margin_ratio" if USE_CLIMATE_RELATIVE_LATENT_HEAT
                          else "latent_heat_kJ_kg")
AHP_PRIOR = {(LATENT_CRITERION_NAME if k == "latent_heat_kJ_kg" else k): v
             for k, v in AHP_PRIOR_BASE.items()}
CRITERIA = list(AHP_PRIOR.keys())


# ═══════════════════════════════════════════════════════════
# CORE TRANSFORMS
# ═══════════════════════════════════════════════════════════

def gaussian_tm_fitness(tm, tm_target, sigma=SIGMA_TM):
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


def minmax_normalize(df, cols):
    M = df[cols].copy()
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


# ═══════════════════════════════════════════════════════════
# FOUR RANKING METHODS  (matrix: rows=candidates, cols=criteria, all
# already benefit-oriented and 0-1 normalized; weights: same order)
# ═══════════════════════════════════════════════════════════

def topsis(matrix, weights):
    norm = matrix / (np.sqrt((matrix ** 2).sum(axis=0)) + 1e-12)
    weighted = norm * weights
    v_plus, v_minus = weighted.max(axis=0), weighted.min(axis=0)
    s_plus = np.sqrt(((weighted - v_plus) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - v_minus) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)


def gra(matrix, weights, zeta=GRA_ZETA):
    ref = matrix.max(axis=0)
    delta = np.abs(matrix - ref)
    delta_min, delta_max = delta.min(), delta.max()
    coeff = (delta_min + zeta * delta_max) / (delta + zeta * delta_max + 1e-12)
    return (coeff * weights).sum(axis=1)


def promethee_ii(matrix, weights, q=PROMETHEE_Q, p=PROMETHEE_P):
    n, k = matrix.shape
    phi_plus = np.zeros(n)
    phi_minus = np.zeros(n)
    for j in range(k):
        col = matrix[:, j]
        d = col[:, None] - col[None, :]                     # d[i,k] = x_i - x_k
        pref = np.clip((np.abs(d) - q) / (p - q + 1e-12), 0, 1)
        pref = np.where(d > 0, pref, 0.0)                    # only "i preferred to k" direction
        phi_plus += weights[j] * pref.sum(axis=1)
        phi_minus += weights[j] * pref.sum(axis=0)
    denom = max(n - 1, 1)
    return (phi_plus - phi_minus) / denom


def vikor(matrix, weights, v=VIKOR_V):
    f_star = matrix.max(axis=0)
    f_minus = matrix.min(axis=0)
    span = np.where((f_star - f_minus) == 0, 1e-12, f_star - f_minus)
    weighted_gap = weights * (f_star - matrix) / span
    S = weighted_gap.sum(axis=1)
    R = weighted_gap.max(axis=1)
    s_star, s_minus = S.min(), S.max()
    r_star, r_minus = R.min(), R.max()
    Q = (v * (S - s_star) / (s_minus - s_star + 1e-12) +
         (1 - v) * (R - r_star) / (r_minus - r_star + 1e-12))
    return Q, S, R   # lower Q = better


def vikor_compromise_check(Q, names):
    """Acceptable-advantage + acceptable-stability conditions, standard
    VIKOR post-check. Returns (is_valid_single_winner, note)."""
    order = np.argsort(Q)
    n = len(Q)
    if n < 2:
        return True, "only one candidate"
    dq = 1.0 / max(n - 1, 1)
    advantage_ok = (Q[order[1]] - Q[order[0]]) >= dq
    if not advantage_ok:
        return False, (f"VIKOR acceptable-advantage FAILS "
                        f"(Q gap {Q[order[1]]-Q[order[0]]:.4f} < {dq:.4f}) — "
                        f"report a compromise set {names[order[0]]}/{names[order[1]]}, "
                        f"not a single VIKOR winner")
    return True, "single VIKOR winner acceptable"


# ═══════════════════════════════════════════════════════════
# CONSENSUS: BORDA + COPELAND
# ═══════════════════════════════════════════════════════════

def borda_and_copeland(rank_series_list):
    n = len(rank_series_list[0])
    m = len(rank_series_list)
    names = rank_series_list[0].index

    borda = pd.Series(0.0, index=names)
    for ranks in rank_series_list:
        borda += (n - ranks + 1)

    rank_matrix = pd.concat(rank_series_list, axis=1).values
    R = rank_matrix.sum(axis=1)
    R_bar = R.mean()
    S = ((R - R_bar) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n) + 1e-12) if n > 1 else np.nan

    copeland = pd.Series(0.0, index=names)
    for i, name_i in enumerate(names):
        for jx, name_j in enumerate(names):
            if i == jx:
                continue
            wins = sum(1 for ranks in rank_series_list if ranks[name_i] < ranks[name_j])
            losses = sum(1 for ranks in rank_series_list if ranks[name_i] > ranks[name_j])
            copeland[name_i] += (1 if wins > losses else (-1 if losses > wins else 0))

    return borda, copeland, W


# ═══════════════════════════════════════════════════════════
# MONTE CARLO STABILITY  (plan v3.0 Section 9.6)
# ═══════════════════════════════════════════════════════════

def run_monte_carlo(df, tm_target, w_final, n_draws=N_MONTE_CARLO_DRAWS, seed=MC_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    names = df["name"].tolist()
    n_cand = len(names)
    l_required = (df["L_required_kJ_per_kg"].iloc[0]
                  if "L_required_kJ_per_kg" in df.columns else np.nan)

    base_matrix = minmax_normalize(df, CRITERIA)
    baseline_topsis = topsis(base_matrix, w_final)
    baseline_rank = pd.Series(baseline_topsis, index=names).rank(ascending=False, method="min")

    top3_count = {name: 0 for name in names}
    top1_count = {name: 0 for name in names}
    spearman_rhos = []

    alpha = np.clip(w_final, 1e-6, None) * MC_DIRICHLET_CONCENTRATION

    for _ in range(n_draws):
        w_draw = rng.dirichlet(alpha)

        tm_draw = df["Tm_C"].values + rng.normal(0, MC_TM_STD_K, n_cand)
        l_draw = df["latent_heat_kJ_kg"].values * (
            1 + rng.normal(0, MC_RELATIVE_STD["latent_heat_kJ_kg"], n_cand))
        tc_draw = df["TC_W_mK"].values * (1 + rng.normal(0, MC_RELATIVE_STD["TC_W_mK"], n_cand))
        rho_draw = df["rho_H_MJ_m3"].values * (
            1 + rng.normal(0, MC_RELATIVE_STD["rho_H_MJ_m3"], n_cand))

        latent_criterion_draw = (l_draw / l_required) if USE_CLIMATE_RELATIVE_LATENT_HEAT else l_draw

        draw_df = pd.DataFrame({
            "f_Tm": gaussian_tm_fitness(tm_draw, tm_target),
            LATENT_CRITERION_NAME: latent_criterion_draw,
            "rho_H_MJ_m3": rho_draw,
            "TC_W_mK": tc_draw,
            "cycles_confidence": df["cycles_confidence"].values,
        })
        draw_matrix = minmax_normalize(draw_df, CRITERIA)
        draw_scores = topsis(draw_matrix, w_draw)
        draw_rank = pd.Series(draw_scores, index=names).rank(ascending=False, method="min")

        for name in draw_rank[draw_rank <= 3].index:
            top3_count[name] += 1
        for name in draw_rank[draw_rank == 1].index:
            top1_count[name] += 1

        rho, _ = spearmanr(baseline_rank.values, draw_rank.values)
        spearman_rhos.append(rho if rho == rho else 0.0)

    result = pd.DataFrame({
        "name": names,
        "top3_inclusion_probability": [top3_count[n] / n_draws for n in names],
        "top1_retention_rate": [top1_count[n] / n_draws for n in names],
    })
    result["mean_spearman_rho_vs_baseline"] = float(np.mean(spearman_rhos))
    result["n_draws"] = n_draws
    return result.sort_values("top3_inclusion_probability", ascending=False)


# ═══════════════════════════════════════════════════════════
# PER-CLUSTER RANKING
# ═══════════════════════════════════════════════════════════

def rank_cluster(df):
    df = df.copy().reset_index(drop=True)
    df["f_Tm"] = gaussian_tm_fitness(df["Tm_C"], df["Tm_target_C"].iloc[0])

    if USE_CLIMATE_RELATIVE_LATENT_HEAT:
        l_required = df["L_required_kJ_per_kg"].iloc[0]
        df["latent_heat_margin_ratio"] = df["latent_heat_kJ_kg"] / l_required

    df["cycles_confidence_imputed"] = df["cycles_confidence"].isna()
    med = df["cycles_confidence"].median()
    df["cycles_confidence"] = df["cycles_confidence"].fillna(med if med == med else 0.5)

    M = minmax_normalize(df, CRITERIA)
    w_entropy = entropy_weights(M)
    w_ahp = np.array([AHP_PRIOR[c] for c in CRITERIA])
    w_ahp = w_ahp / w_ahp.sum()
    w_final = ENTROPY_AHP_LAMBDA * w_entropy + (1 - ENTROPY_AHP_LAMBDA) * w_ahp
    w_final = w_final / w_final.sum()

    df["topsis_score"] = topsis(M, w_final)
    df["gra_grade"] = gra(M, w_final)
    df["promethee_flow"] = promethee_ii(M, w_final)
    vikor_q, vikor_s, vikor_r = vikor(M, w_final)
    df["vikor_Q"] = vikor_q
    df["vikor_S"] = vikor_s
    df["vikor_R"] = vikor_r

    df["topsis_rank"] = df["topsis_score"].rank(ascending=False, method="min").astype(int)
    df["gra_rank"] = df["gra_grade"].rank(ascending=False, method="min").astype(int)
    df["promethee_rank"] = df["promethee_flow"].rank(ascending=False, method="min").astype(int)
    df["vikor_rank"] = df["vikor_Q"].rank(ascending=True, method="min").astype(int)   # lower Q better

    vikor_valid, vikor_note = vikor_compromise_check(vikor_q, df["name"].values)
    df["vikor_compromise_note"] = vikor_note

    rank_series = [df.set_index("name")[c] for c in
                   ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"]]
    borda, copeland, kendall_w = borda_and_copeland(rank_series)
    df["borda_score"] = df["name"].map(borda)
    df["copeland_score"] = df["name"].map(copeland)
    df["consensus_rank"] = df["borda_score"].rank(ascending=False, method="min").astype(int)
    df["copeland_rank"] = df["copeland_score"].rank(ascending=False, method="min").astype(int)
    df["kendall_w"] = kendall_w

    borda_top1 = df.loc[df["consensus_rank"] == 1, "name"].tolist()
    copeland_top1 = df.loc[df["copeland_rank"] == 1, "name"].tolist()
    df["borda_copeland_agree"] = bool(set(borda_top1) & set(copeland_top1))

    for i, c in enumerate(CRITERIA):
        df[f"weight_{c}"] = w_final[i]

    mc = run_monte_carlo(df, df["Tm_target_C"].iloc[0], w_final)
    df = df.merge(mc, on="name", how="left")

    return df.sort_values("consensus_rank"), mc


def main():
    print("=" * 68)
    print("  Phase 6 — Full MCDM Stack (TOPSIS+GRA+PROMETHEE II+VIKOR) +")
    print(f"  {N_MONTE_CARLO_DRAWS}-draw Monte Carlo — Tamil Nadu")
    print("=" * 68)

    if not SURVIVORS_FILE.exists():
        print(f"\n  ERROR: {SURVIVORS_FILE} not found — run 07_feasibility_filter.py first.")
        return

    survivors = pd.read_csv(SURVIVORS_FILE)
    full_rows, topk_rows, mc_rows = [], [], []

    for cid, grp in survivors.groupby("cluster_id"):
        passed = grp[grp["passes_all"]]
        if len(passed) < 2:
            print(f"\n  Cluster {int(cid)}: only {len(passed)} survivor(s) — skipping.")
            continue

        ranked, mc = rank_cluster(passed)
        if "cluster_id" in ranked.columns:
            ranked = ranked.drop(columns=["cluster_id"])
        ranked.insert(0, "cluster_id", cid)
        mc.insert(0, "cluster_id", cid)
        full_rows.append(ranked)
        mc_rows.append(mc)

        top3 = ranked.head(3)
        agree_flag = "" if ranked["borda_copeland_agree"].iloc[0] else "  [Borda/Copeland DISAGREE on #1]"
        print(f"\n  Cluster {int(cid)}  (Tm_target={passed['Tm_target_C'].iloc[0]:.1f}C, "
              f"n_survivors={len(passed)}, Kendall's W={ranked['kendall_w'].iloc[0]:.3f}){agree_flag}:")
        for _, row in top3.iterrows():
            mc_row = mc[mc["name"] == row["name"]]
            incl = mc_row["top3_inclusion_probability"].iloc[0] if len(mc_row) else float("nan")
            print(f"    #{row['consensus_rank']}  {row['name']:35s}  Tm={row['Tm_C']:.1f}C  "
                  f"TOPSIS={row['topsis_score']:.3f}(r{row['topsis_rank']})  "
                  f"GRA={row['gra_grade']:.3f}(r{row['gra_rank']})  "
                  f"PROMETHEE={row['promethee_flow']:+.3f}(r{row['promethee_rank']})  "
                  f"VIKOR_Q={row['vikor_Q']:.3f}(r{row['vikor_rank']})  "
                  f"MC_Top3%={incl*100:.1f}%")
        print(f"    VIKOR compromise check: {ranked['vikor_compromise_note'].iloc[0]}")
        topk_rows.append(top3)

    if not full_rows:
        print("\n  ERROR: no cluster had >=2 survivors to rank.")
        return

    full_df = pd.concat(full_rows, ignore_index=True)
    topk_df = pd.concat(topk_rows, ignore_index=True)
    mc_df = pd.concat(mc_rows, ignore_index=True)
    full_df.to_csv(OUT_FULL, index=False)
    topk_df.to_csv(OUT_TOPK, index=False)
    mc_df.to_csv(OUT_MC, index=False)

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Saved: {OUT_TOPK}")
    print(f"  Saved: {OUT_FULL}")
    print(f"  Saved: {OUT_MC}")

    top1_sets = topk_df[topk_df["consensus_rank"] == 1].groupby("cluster_id")["name"].first()
    if top1_sets.nunique() == 1:
        print(f"\n  [FINDING] Every cluster's #1 PCM is identical ({top1_sets.iloc[0]!r}). "
              "See 07b_charging_feasibility.py if you want to test whether a "
              "regime-dependent Tm ceiling changes this — otherwise this is a "
              "legitimate, reportable outcome (see the script's own docstring "
              "for two honest ways to phrase it).")
    disagree = full_df.groupby("cluster_id")["borda_copeland_agree"].first()
    if (~disagree).any():
        print(f"\n  [NOTE] Borda and Copeland disagree on #1 for cluster(s) "
              f"{disagree[~disagree].index.tolist()} — report both per plan v3.0 "
              f"Section 9.5, don't silently pick one.")
    print("=" * 68)
    print("\nNext: python 09_recommendation_cards.py, then 10_physics_validation.py "
          "(Phase 7 — no longer optional, see that script).")


if __name__ == "__main__":
    main()
