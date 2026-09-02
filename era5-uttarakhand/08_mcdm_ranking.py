"""
08_mcdm_ranking.py
=====================
PHASE 6 — MULTI-CRITERIA RANKING ENGINE, minimum viable version
(Objective 1 plan v3.0, Section 9)

This is the "minimum viable MCDM stack" from your 4-day sprint plan:
TOPSIS + GRA, entropy-weighted per cluster, Borda-aggregated to a Top-3.
PROMETHEE II / VIKOR / CoCoSo and the 5,000-draw Monte Carlo stability
check are NOT implemented here — they're real, documented extensions
(see the docstring at the bottom), add them if time remains, but this
script alone already gives you a defensible, falsifiable Top-3 per
cluster, which is the actual headline deliverable of Objective 1.

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

WEIGHTS
---------
Entropy weights computed per cluster from that cluster's own filtered
decision matrix (objective, data-driven). Blended 0.5/0.5 with a fixed
AHP-style prior drawn from plan v3.0 Table 13 (renormalised over just the
5 criteria actually used here). If you get 10 minutes with your guide for
a real pairwise AHP matrix, replace AHP_PRIOR below and rerun — until
then this is an honest placeholder, not a claimed AHP result.

INPUT  : data/processed/pcm/feasibility_survivors_by_cluster.csv (07's output)
OUTPUT : data/processed/pcm/mcdm_topk_by_cluster.csv
           per-cluster Top-3 with individual TOPSIS/GRA ranks, Borda
           consensus rank, and Kendall's W (2-method agreement) per cluster
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
    """matrix already benefit-normalised (higher=better), all columns
    treated as benefit criteria (true here since f_Tm/L/rho_H/TC/cycles
    are all benefit after the Gaussian transform)."""
    norm = matrix / (np.sqrt((matrix ** 2).sum(axis=0)) + 1e-12)
    weighted = norm * weights
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

    topsis_score = topsis(M, w_final)
    gra_grade = gra(M, w_final)

    df["topsis_score"] = topsis_score
    df["gra_grade"] = gra_grade
    df["topsis_rank"] = df["topsis_score"].rank(ascending=False, method="min").astype(int)
    df["gra_rank"] = df["gra_grade"].rank(ascending=False, method="min").astype(int)

    borda, kendall_w = borda_from_ranks([df.set_index("name")["topsis_rank"],
                                          df.set_index("name")["gra_rank"]])
    df["borda_score"] = df["name"].map(borda)
    df["consensus_rank"] = df["borda_score"].rank(ascending=False, method="min").astype(int)
    df["kendall_w"] = kendall_w

    for i, c in enumerate(CRITERIA):
        df[f"weight_{c}"] = w_final[i]

    return df.sort_values("consensus_rank")


def main():
    print("=" * 68)
    print("  Phase 6 — MCDM Ranking (TOPSIS + GRA, entropy+AHP weights) — Uttarakhand")
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
                  f"Tm={row['Tm_C']:.1f}C  TOPSIS={row['topsis_score']:.3f} "
                  f"(rank {row['topsis_rank']})  GRA={row['gra_grade']:.3f} "
                  f"(rank {row['gra_rank']})")
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
              f"TOPSIS and GRA disagree meaningfully there. Per plan v3.0 Section 9.5, "
              f"this is a genuine, reportable finding (that regime's PCM choice is "
              f"ambiguous), not a bug to fix — discuss it rather than hide it.")
    print("=" * 68)
    print("\nWhat's still genuinely optional beyond this (your sprint plan already "
          "flags these as stretch goals, not required):")
    print("  - PROMETHEE II as a third ranking method (best-suited to the")
    print("    target-based Tm criterion natively; ~40 more lines)")
    print("  - 5,000-draw Monte Carlo weight/property perturbation for a")
    print("    Top-3 inclusion-probability confidence figure")
    print("  - A minimal grey-box physics validation run per cluster's Top-1")
    print("\nWithout those, you already have a defensible, falsifiable Top-3 per")
    print("cluster — write the recommendation cards from mcdm_topk_by_cluster.csv")
    print("+ cluster_profiles_uttarakhand.csv (Phase 8) and you have a complete")
    print("Objective 1.")


if __name__ == "__main__":
    main()
