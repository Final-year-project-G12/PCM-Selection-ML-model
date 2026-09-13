"""
08_mcdm_ranking.py
=====================
PHASE 6 — MULTI-CRITERIA RANKING ENGINE (Objective 1 plan v3.0, Section 9)

Full 4-method stack: TOPSIS, GRA, PROMETHEE II, and VIKOR, entropy+AHP
weighted per cluster, Borda-aggregated to a Top-3. (Earlier version of
this script only ran TOPSIS+GRA; PROMETHEE II and VIKOR are added below
so every method the bump-chart plot expects actually gets computed.)

THE ONE STEP EVERY PCM-MCDM PAPER GETS WRONG (plan v3.0 Section 9.2)
------------------------------------------------------------------------
Melting temperature is a TARGET-based criterion, not a benefit or cost —
closer to Tm_target is better in both directions. Feeding raw Tm into
TOPSIS/GRA produces plausible-looking nonsense. This script converts Tm
to a Gaussian fitness score BEFORE anything else touches it:

    f_Tm(i) = exp( -(Tm_i - Tm_target)^2 / (2*sigma^2) ),  sigma = 4K

f_Tm is then used as an ordinary benefit criterion downstream.

CRITERIA USED (only what your database actually has values for)
------------------------------------------------------------------
  f_Tm (melting-point fitness, Gaussian)     benefit
  latent_heat_kJ_kg                          benefit
  rho_H_MJ_m3 (volumetric latent heat)       benefit
  TC_W_mK (thermal conductivity)             benefit
  cycles_confidence (log-scaled, NaN-safe)   benefit, missing -> median-imputed
                                              with a flag column so you can
                                              report how many candidates per
                                              cluster had unreported cycling
Corrosion class and cost are NOT included as ranking criteria — the
database doesn't have reliable values for either yet (see 06's docstring
for what to add). Say this explicitly in your methodology rather than
silently dropping them.

FOUR RANKING METHODS
-----------------------
  TOPSIS       — closeness coefficient, Euclidean ideal/anti-ideal
  GRA          — grey relational grade vs. the ideal (max) reference
  PROMETHEE II — net outranking flow; V-shape preference function with
                 indifference/preference thresholds q=0.10, p=0.30 of the
                 [0,1] normalized range for every criterion (a documented,
                 uniform simplification)
  VIKOR        — compromise ranking Q_i (v=0.5), with the standard
                 acceptable-advantage / acceptable-stability check flagged

WEIGHTS
---------
Entropy weights computed per cluster from that cluster's own filtered
decision matrix (objective, data-driven). Blended 0.5/0.5 with a fixed
AHP-style prior drawn from plan v3.0 Table 13 (renormalised over just the
5 criteria actually used here). If you get 10 minutes with your guide for
a real pairwise AHP matrix, replace AHP_PRIOR below and rerun — until
then this is an honest placeholder, not a claimed AHP result.

CONSENSUS
-----------
Borda count across all 4 methods' ranks.

INPUT  : data/processed/pcm/feasibility_survivors_by_cluster.csv (07's output)
OUTPUT : data/processed/pcm/mcdm_topk_by_cluster.csv
           per-cluster Top-3 with individual TOPSIS/GRA/PROMETHEE/VIKOR
           ranks, Borda consensus rank, and Kendall's W (4-method
           agreement) per cluster
         data/processed/pcm/mcdm_full_scores_by_cluster.csv
           every surviving candidate's full score breakdown, not just Top-3
           (keep this — it's what a recommendation card's "criterion
           contributions" field needs)

HOW TO RUN:
  python 08_mcdm_ranking.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

SURVIVORS_FILE = PROCESSED_DIR / "pcm" / "feasibility_survivors_by_cluster.csv"
OUT_TOPK = PROCESSED_DIR / "pcm" / "mcdm_topk_by_cluster.csv"
OUT_FULL = PROCESSED_DIR / "pcm" / "mcdm_full_scores_by_cluster.csv"

SIGMA_TM = 4.0          # K, plan v3.0 Section 9.2 — justified from HX approach temperature
ENTROPY_AHP_LAMBDA = 0.5
GRA_ZETA = 0.5           # distinguishing coefficient, standard value
PROMETHEE_Q, PROMETHEE_P = 0.10, 0.30    # indifference/preference, fraction of [0,1] range
VIKOR_V = 0.5

# Renormalised AHP-style prior over the 5 criteria this script actually
# uses (Tm fitness, latent heat, volumetric latent heat, conductivity,
# cycling) — drawn proportionally from plan v3.0 Table 13's 8-criterion
# set with corrosion/cost/supercooling removed and the rest rescaled to
# sum to 1. Replace with a real elicited AHP vector if you get one.
AHP_PRIOR = {
    "f_Tm": 0.24 / 0.80,
    "latent_heat_kJ_kg": 0.20 / 0.80,
    "rho_H_MJ_m3": 0.12 / 0.80,
    "TC_W_mK": 0.13 / 0.80,
    "cycles_confidence": 0.11 / 0.80,
}
CRITERIA = list(AHP_PRIOR.keys())


def gaussian_tm_fitness(tm, tm_target, sigma=SIGMA_TM):
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


def entropy_weights(matrix):
    """Standard Shannon-entropy weighting. matrix: rows=candidates,
    cols=criteria, already non-negative (benefit-normalised)."""
    X = matrix.copy()
    col_sums = X.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1e-12, col_sums)
    P = X / col_sums
    n = X.shape[0]
    k = 1.0 / np.log(n) if n > 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -k * np.nansum(np.where(P > 0, P * np.log(P), 0), axis=0)
    d = 1 - e   # degree of diversification
    w = d / d.sum() if d.sum() > 0 else np.ones(len(d)) / len(d)
    return w


def topsis(matrix, weights):
    """matrix already benefit-normalised (higher=better) AND already
    min-max scaled to [0,1] per column by the caller (rank_cluster()) —
    same basis gra()/promethee_ii()/vikor() consume directly, and the
    same basis w_final (entropy+AHP) was computed on. Deliberately does
    NOT re-apply TOPSIS's classic vector (Euclidean) normalization on top
    of that: doing so would rescale each column by a second,
    data-dependent factor after w_final was already fixed, putting TOPSIS
    on a different effective normalization basis than the other three
    methods and manufacturing method disagreement (lower Kendall's W)
    that isn't a real multi-criteria disagreement. All columns are
    treated as benefit criteria (true here since f_Tm/L/rho_H/TC/cycles
    are all benefit after the Gaussian transform)."""
    weighted = matrix * weights
    v_plus = weighted.max(axis=0)
    v_minus = weighted.min(axis=0)
    s_plus = np.sqrt(((weighted - v_plus) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - v_minus) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)


def gra(matrix, weights, zeta=GRA_ZETA):
    """Grey Relational Analysis. matrix columns already 0-1 normalised
    benefit criteria; ideal reference = column max (best case)."""
    ref = matrix.max(axis=0)
    delta = np.abs(matrix - ref)
    delta_min, delta_max = delta.min(), delta.max()
    coeff = (delta_min + zeta * delta_max) / (delta + zeta * delta_max + 1e-12)
    grade = (coeff * weights).sum(axis=1)
    return grade


def promethee_ii(matrix, weights, q=PROMETHEE_Q, p=PROMETHEE_P):
    """Net outranking flow. matrix/weights same shape convention as
    topsis()/gra() — 0-1 normalised benefit criteria. V-shape preference
    function with indifference threshold q and preference threshold p,
    both expressed as a fraction of the already-[0,1]-normalised range
    (a documented simplification vs. a physical-units threshold)."""
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
    """Compromise ranking. Returns (Q, S, R) — lower Q is better."""
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
    return Q, S, R


def vikor_compromise_check(Q, S, R, names):
    """Acceptable-advantage (C1) + acceptable-stability (C2) conditions,
    standard VIKOR post-check (Opricovic & Tzeng). Returns
    (is_valid_single_winner, note).

    C1: Q(2nd) - Q(1st) >= DQ = 1/(n-1).
    C2: the Q-best alternative must ALSO be best-ranked by S alone or by
    R alone — otherwise the "winner" is an artifact of the v-weighted
    blend, not a genuinely stable compromise. Both conditions must hold
    for a single winner to be reported; if either fails, VIKOR itself
    prescribes reporting a compromise set instead.
    """
    order = np.argsort(Q)
    n = len(Q)
    if n < 2:
        return True, "only one candidate"
    dq = 1.0 / max(n - 1, 1)
    winner = order[0]

    advantage_ok = (Q[order[1]] - Q[order[0]]) >= dq
    if not advantage_ok:
        return False, (f"VIKOR acceptable-advantage (C1) FAILS "
                        f"(Q gap {Q[order[1]]-Q[order[0]]:.4f} < {dq:.4f}) — "
                        f"report a compromise set {names[order[0]]}/{names[order[1]]}, "
                        f"not a single VIKOR winner")

    stability_ok = (winner == np.argmin(S)) or (winner == np.argmin(R))
    if not stability_ok:
        return False, (f"VIKOR acceptable-stability (C2) FAILS "
                        f"({names[winner]} is Q-best but not best-ranked by S "
                        f"({names[np.argmin(S)]}) or by R ({names[np.argmin(R)]}) alone) — "
                        f"report {names[winner]} and {names[np.argmin(S)]} as a compromise set, "
                        f"not a single VIKOR winner")

    return True, "single VIKOR winner acceptable"


def borda_from_ranks(rank_series_list):
    """rank_series_list: list of pandas Series (index=candidate, values=rank,
    1=best). Returns Borda score (higher=better) and Kendall's W."""
    n = len(rank_series_list[0])
    m = len(rank_series_list)
    borda = pd.Series(0.0, index=rank_series_list[0].index)
    for ranks in rank_series_list:
        borda += (n - ranks + 1)

    # Kendall's W (coefficient of concordance) across the m rankers
    rank_matrix = pd.concat(rank_series_list, axis=1).values  # n x m
    R = rank_matrix.sum(axis=1)  # sum of ranks per candidate
    R_bar = R.mean()
    S = ((R - R_bar) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n) + 1e-12) if n > 1 else np.nan
    return borda, W


def rank_cluster(df):
    """df: survivors for one cluster, passes_all==True rows only."""
    df = df.copy().reset_index(drop=True)
    df["f_Tm"] = gaussian_tm_fitness(df["Tm_C"], df["Tm_target_C"].iloc[0])

    # NaN-safe cycles_confidence: median-impute within this cluster's
    # candidate set, flag which rows were imputed (report, don't hide).
    df["cycles_confidence_imputed"] = df["cycles_confidence"].isna()
    med = df["cycles_confidence"].median()
    df["cycles_confidence"] = df["cycles_confidence"].fillna(med if med == med else 0.5)

    # Min-max normalise each criterion to [0,1] (benefit direction, all
    # five criteria here are "higher is better" post Gaussian-transform).
    M = df[CRITERIA].copy()
    for c in CRITERIA:
        lo, hi = M[c].min(), M[c].max()
        M[c] = (M[c] - lo) / (hi - lo) if hi > lo else 0.5
    M = M.fillna(0.0).values

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

    vikor_valid, vikor_note = vikor_compromise_check(vikor_q, vikor_s, vikor_r, df["name"].values)
    df["vikor_compromise_note"] = vikor_note

    borda, kendall_w = borda_from_ranks([df.set_index("name")["topsis_rank"],
                                          df.set_index("name")["gra_rank"],
                                          df.set_index("name")["promethee_rank"],
                                          df.set_index("name")["vikor_rank"]])
    df["borda_score"] = df["name"].map(borda)
    df["consensus_rank"] = df["borda_score"].rank(ascending=False, method="min").astype(int)
    df["kendall_w"] = kendall_w

    for i, c in enumerate(CRITERIA):
        df[f"weight_{c}"] = w_final[i]

    return df.sort_values("consensus_rank")


def main():
    print("=" * 68)
    print("  Phase 6 — MCDM Ranking (TOPSIS+GRA+PROMETHEE II+VIKOR, entropy+AHP weights) — Uttarakhand")
    print("=" * 68)

    if not SURVIVORS_FILE.exists():
        print(f"\n  ERROR: {SURVIVORS_FILE} not found — run 07_feasibility_filter.py first.")
        return

    survivors = pd.read_csv(SURVIVORS_FILE)
    full_rows, topk_rows = [], []

    for cid, grp in survivors.groupby("cluster_id"):
        passed = grp[grp["passes_all"]]
        if len(passed) < 2:
            print(f"\n  Cluster {int(cid)}: only {len(passed)} survivor(s) — "
                  f"cannot rank with <2 candidates, skipping. Widen the "
                  f"feasibility window or database for this cluster.")
            continue

        ranked = rank_cluster(passed)
        if "cluster_id" in ranked.columns:   # already carried through from 07's output
            ranked = ranked.drop(columns=["cluster_id"])
        ranked.insert(0, "cluster_id", cid)
        full_rows.append(ranked)

        top3 = ranked.head(3)
        print(f"\n  Cluster {int(cid)}  (Tm_target={passed['Tm_target_C'].iloc[0]:.1f}C, "
              f"n_survivors={len(passed)}, Kendall's W={ranked['kendall_w'].iloc[0]:.3f}):")
        for _, row in top3.iterrows():
            print(f"    #{row['consensus_rank']}  {row['name']:35s}  "
                  f"Tm={row['Tm_C']:.1f}C  TOPSIS={row['topsis_score']:.3f}(r{row['topsis_rank']})  "
                  f"GRA={row['gra_grade']:.3f}(r{row['gra_rank']})  "
                  f"PROMETHEE={row['promethee_flow']:+.3f}(r{row['promethee_rank']})  "
                  f"VIKOR_Q={row['vikor_Q']:.3f}(r{row['vikor_rank']})")
        print(f"    VIKOR compromise check: {ranked['vikor_compromise_note'].iloc[0]}")
        topk_rows.append(top3)

    if not full_rows:
        print("\n  ERROR: no cluster had >=2 survivors to rank. Check 07's output "
              "and widen the PCM database (06) or feasibility window (07) as needed.")
        return

    full_df = pd.concat(full_rows, ignore_index=True)
    topk_df = pd.concat(topk_rows, ignore_index=True)
    full_df.to_csv(OUT_FULL, index=False)
    topk_df.to_csv(OUT_TOPK, index=False)

    print("\n" + "=" * 68)
    print("  DONE")

    # Diagnostic: if Tm_target didn't vary across clusters (plan v3.0's
    # "constant by design" rule), the Top-3 can legitimately converge to
    # the same PCMs everywhere — report it explicitly rather than let it
    # pass silently, since it directly affects whether Objective 1's
    # "different PCM per regime" claim actually holds.
    top1_sets = topk_df[topk_df["consensus_rank"] == 1].groupby("cluster_id")["name"].first()
    if top1_sets.nunique() == 1:
        print("\n  [FINDING] Every cluster's #1 PCM is identical "
              f"({top1_sets.iloc[0]!r}). This is a direct consequence of "
              "Tm_target being held constant across all clusters (plan v3.0 "
              "Section 6.3's design rule) combined with every candidate's "
              "latent heat comfortably clearing L_required in every cluster. "
              "It is NOT a bug. Two honest ways to report it:")
        print("    (a) State it as a finding: Uttarakhand's climate regimes differ more "
              "in solar reliability/cloud persistence than in delivery-relevant "
              "temperature, so a single PCM family serves the whole state under "
              "the corrected Tm_target rule — differentiation would need to show "
              "up in Phase 7 physics simulation (solar fraction per regime), "
              "not in the candidate list itself.")
        print("    (b) Run 07b_charging_feasibility.py (optional, heuristic "
              "regime-dependent upper bound on Tm) before 07/08 to see if a "
              "real charging-feasibility constraint changes this.")
    print(f"  Saved: {OUT_TOPK}   (Top-3 per cluster — your headline results table)")
    print(f"  Saved: {OUT_FULL}   (every survivor's full score, for recommendation cards)")
    low_w = full_df.groupby("cluster_id")["kendall_w"].first()
    ambiguous = low_w[low_w < 0.6]
    if len(ambiguous):
        print(f"\n  [NOTE] Kendall's W < 0.6 for cluster(s) {list(ambiguous.index)} — "
              f"the 4 methods disagree meaningfully there. Per plan v3.0 Section 9.5, "
              f"this is a genuine, reportable finding (that regime's PCM choice is "
              f"ambiguous), not a bug to fix — discuss it rather than hide it.")
    print("=" * 68)
    print("\nStill genuinely optional beyond this:")
    print("  - 5,000-draw Monte Carlo weight/property perturbation for a")
    print("    Top-3 inclusion-probability confidence figure")
    print("  - A minimal grey-box physics validation run per cluster's Top-1")
    print("    (see 10_physics_validation.py — already implemented)")
    print("\nWithout Monte Carlo, you already have a defensible, falsifiable Top-3 per")
    print("cluster from 4 independent ranking methods — write the recommendation cards")
    print("from mcdm_topk_by_cluster.csv + cluster_profiles_uttarakhand.csv (Phase 8)")
    print("and you have a complete Objective 1.")


if __name__ == "__main__":
    main()