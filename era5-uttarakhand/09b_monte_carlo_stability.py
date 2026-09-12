"""
09b_monte_carlo_stability.py
=====================================
PHASE 6, OPTIONAL — Monte Carlo weight/property perturbation stability check

This script was NOT provided in any upload — neither Tamil Nadu's nor your
own Uttarakhand folder had a working copy of it, only Tamil Nadu's already-
computed OUTPUT file (monte_carlo_stability.csv) existed anywhere I could
see. This is a fresh implementation, built to match the exact methodology
08_mcdm_ranking.py's own docstring and print statements describe: "5,000-
draw Monte Carlo weight/property perturbation for a Top-3 inclusion-
probability confidence figure" (also matches the project's stated
methodology: "Uncertainty: 5000-draw Monte Carlo (Dirichlet weights x
Gaussian property noise)").

WHAT THIS ANSWERS
--------------------
08_mcdm_ranking.py gives you ONE ranking per cluster, computed from ONE
specific set of weights (entropy+AHP blend) and ONE specific set of PCM
property values (point estimates from the database, some of them MICE/PMM-
imputed). But those weights are a modelling choice, not a physical law, and
several PCM properties (especially cycles_confidence, and any
any_property_imputed=True row) carry real uncertainty. This script asks:
if you perturbed the weights and the properties within a defensible range,
5000 different times, how often does each candidate still land in the
Top-3? A PCM with a high Top-3 inclusion probability is a robust pick; one
that only makes Top-3 under the exact nominal weights is a fragile pick,
worth flagging in your recommendation cards.

METHOD (matches 08_mcdm_ranking.py's rank_cluster() EXACTLY, only the
weights and the property matrix are perturbed per draw — the four ranking
methods, Borda consensus, and all constants below are copy-identical to
08_mcdm_ranking.py so results are directly comparable. If you ever edit
08_mcdm_ranking.py's constants or algorithms, mirror the change here too.)
--------------------------------------------------------------------------
Per draw, per cluster:
  1. WEIGHTS: draw from a Dirichlet distribution centered on that cluster's
     actual w_final (entropy+AHP blend from 08's own run), concentration
     DIRICHLET_CONCENTRATION=50 (higher = tighter around nominal; 50 keeps
     draws plausible while still exploring +-15-20% relative shifts on each
     weight — a documented, not derived, choice).
  2. PROPERTIES: add independent Gaussian noise to each of the 5 criteria's
     RAW values before re-normalising:
       Tm_C                +- 1.0 K            (typical datasheet/measurement tolerance)
       latent_heat_kJ_kg    +- 5% of value      (typical manufacturer datasheet tolerance)
       rho_H_MJ_m3          +- 5% of value
       TC_W_mK              +- 5% of value
       cycles_confidence    +- 0.10 (absolute, already 0-1 scaled)     ; rows flagged
                              any_property_imputed=True get 2x this noise,
                              reflecting the extra uncertainty from MICE/PMM
                              imputation rather than a real measurement.
  3. Re-run gaussian_tm_fitness, re-normalise, re-run TOPSIS+GRA+PROMETHEE+
     VIKOR with the perturbed weights, re-compute Borda consensus_rank.
  4. Record which candidates land in consensus_rank <= 3 this draw.

OUTPUT
--------
  data/processed/pcm/monte_carlo_stability.csv
    columns: cluster_id, name, top3_inclusion_probability, mean_consensus_rank,
             std_consensus_rank, n_draws

HOW TO RUN:
  python 09b_monte_carlo_stability.py             # default 2000 draws/cluster (~1-2 min)
  python 09b_monte_carlo_stability.py --draws 5000 # full spec, slower
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

SURVIVORS_FILE = PROCESSED_DIR / "pcm" / "feasibility_survivors_by_cluster.csv"
OUT_FILE = PROCESSED_DIR / "pcm" / "monte_carlo_stability.csv"

# ── Copied EXACTLY from 08_mcdm_ranking.py — keep these two files in sync ──
SIGMA_TM = 4.0
GRA_ZETA = 0.5
PROMETHEE_Q, PROMETHEE_P = 0.10, 0.30
VIKOR_V = 0.5
AHP_PRIOR = {
    "f_Tm": 0.24 / 0.80,
    "latent_heat_kJ_kg": 0.20 / 0.80,
    "rho_H_MJ_m3": 0.12 / 0.80,
    "TC_W_mK": 0.13 / 0.80,
    "cycles_confidence": 0.11 / 0.80,
}
CRITERIA = list(AHP_PRIOR.keys())
ENTROPY_AHP_LAMBDA = 0.5

# ── Monte Carlo specific settings (new, documented here) ──────────────────
DIRICHLET_CONCENTRATION = 50.0
NOISE_TM_K = 1.0
NOISE_REL_FRAC = 0.05          # +-5% relative noise, latent_heat/rho_H/TC
NOISE_CYCLES_ABS = 0.10        # +-0.10 absolute, cycles_confidence (already 0-1)
IMPUTED_NOISE_MULTIPLIER = 2.0
RANDOM_SEED = 42


def gaussian_tm_fitness(tm, tm_target, sigma=SIGMA_TM):
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


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
    w = d / d.sum() if d.sum() > 0 else np.ones(len(d)) / len(d)
    return w


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
    delta_min, delta_max = delta.min(), delta.max()
    coeff = (delta_min + zeta * delta_max) / (delta + zeta * delta_max + 1e-12)
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
    weighted_gap = weights * (f_star - matrix) / span
    S = weighted_gap.sum(axis=1)
    R = weighted_gap.max(axis=1)
    s_star, s_minus = S.min(), S.max()
    r_star, r_minus = R.min(), R.max()
    Q = (v * (S - s_star) / (s_minus - s_star + 1e-12) +
         (1 - v) * (R - r_star) / (r_minus - r_star + 1e-12))
    return Q


def borda_consensus_rank(topsis_score, gra_grade, promethee_flow, vikor_q):
    n = len(topsis_score)
    topsis_rank = pd.Series(topsis_score).rank(ascending=False, method="min").values
    gra_rank = pd.Series(gra_grade).rank(ascending=False, method="min").values
    promethee_rank = pd.Series(promethee_flow).rank(ascending=False, method="min").values
    vikor_rank = pd.Series(vikor_q).rank(ascending=True, method="min").values
    borda = (n - topsis_rank + 1) + (n - gra_rank + 1) + (n - promethee_rank + 1) + (n - vikor_rank + 1)
    return pd.Series(borda).rank(ascending=False, method="min").values


def one_draw(base_matrix_raw, w_nominal, rng):
    """base_matrix_raw: dict of criterion -> raw value array (pre-normalise).
    Returns consensus_rank array for this single perturbed draw."""
    n = len(next(iter(base_matrix_raw.values()))["value"])
    perturbed = {}
    for c in CRITERIA:
        raw = base_matrix_raw[c]["value"]
        noise_scale = base_matrix_raw[c]["noise_scale"]
        perturbed[c] = raw + rng.normal(0, noise_scale, size=n)

    M = np.column_stack([perturbed[c] for c in CRITERIA])
    for j in range(M.shape[1]):
        lo, hi = M[:, j].min(), M[:, j].max()
        M[:, j] = (M[:, j] - lo) / (hi - lo) if hi > lo else 0.5
    M = np.nan_to_num(M, nan=0.0)

    w_draw = rng.dirichlet(w_nominal * DIRICHLET_CONCENTRATION)

    ts = topsis(M, w_draw)
    gr = gra(M, w_draw)
    pm = promethee_ii(M, w_draw)
    vk = vikor(M, w_draw)
    return borda_consensus_rank(ts, gr, pm, vk)


def run_cluster(cid, df, n_draws, rng):
    df = df.copy().reset_index(drop=True)
    tm_target = df["Tm_target_C"].iloc[0]

    df["cycles_confidence"] = df["cycles_confidence"].fillna(df["cycles_confidence"].median())
    imputed_flag = df.get("any_property_imputed", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    f_tm_nominal = gaussian_tm_fitness(df["Tm_C"], tm_target)

    base_matrix_raw = {
        "f_Tm": {"value": f_tm_nominal.values,
                 "noise_scale": np.full(len(df), NOISE_TM_K)},  # approximated via Tm noise below
        "latent_heat_kJ_kg": {"value": df["latent_heat_kJ_kg"].values,
                               "noise_scale": (NOISE_REL_FRAC * df["latent_heat_kJ_kg"].values)},
        "rho_H_MJ_m3": {"value": df["rho_H_MJ_m3"].values,
                        "noise_scale": (NOISE_REL_FRAC * df["rho_H_MJ_m3"].values)},
        "TC_W_mK": {"value": df["TC_W_mK"].values,
                    "noise_scale": (NOISE_REL_FRAC * df["TC_W_mK"].values)},
        "cycles_confidence": {"value": df["cycles_confidence"].values,
                              "noise_scale": np.full(len(df), NOISE_CYCLES_ABS)},
    }
    # Tm noise is applied on the RAW Tm_C before the Gaussian-fitness transform
    # (perturbing Tm directly is more physically meaningful than perturbing
    # f_Tm's output value), then f_Tm is recomputed per draw from the
    # perturbed Tm — override the f_Tm entry accordingly:
    tm_raw = df["Tm_C"].values

    # Imputed rows get doubled noise on every perturbed criterion
    for c in ["latent_heat_kJ_kg", "rho_H_MJ_m3", "TC_W_mK", "cycles_confidence"]:
        base_matrix_raw[c]["noise_scale"] = np.where(
            imputed_flag, base_matrix_raw[c]["noise_scale"] * IMPUTED_NOISE_MULTIPLIER,
            base_matrix_raw[c]["noise_scale"])

    # Compute nominal weights (identical to 08_mcdm_ranking.py's rank_cluster)
    M_nominal = np.column_stack([
        (lambda v: (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.full(len(v), 0.5))(
            base_matrix_raw[c]["value"]) for c in CRITERIA
    ])
    w_entropy = entropy_weights(M_nominal)
    w_ahp = np.array([AHP_PRIOR[c] for c in CRITERIA])
    w_ahp = w_ahp / w_ahp.sum()
    w_nominal = ENTROPY_AHP_LAMBDA * w_entropy + (1 - ENTROPY_AHP_LAMBDA) * w_ahp
    w_nominal = w_nominal / w_nominal.sum()
    w_nominal = np.clip(w_nominal, 1e-6, None)  # Dirichlet needs strictly positive alpha

    n = len(df)
    rank_draws = np.zeros((n_draws, n))
    for d in range(n_draws):
        tm_perturbed = tm_raw + rng.normal(0, NOISE_TM_K, size=n)
        base_matrix_raw["f_Tm"]["value"] = gaussian_tm_fitness(tm_perturbed, tm_target)
        rank_draws[d, :] = one_draw(base_matrix_raw, w_nominal, rng)

    top3_prob = (rank_draws <= 3).mean(axis=0)
    mean_rank = rank_draws.mean(axis=0)
    std_rank = rank_draws.std(axis=0)

    out = pd.DataFrame({
        "cluster_id": cid,
        "name": df["name"].values,
        "top3_inclusion_probability": top3_prob,
        "mean_consensus_rank": mean_rank,
        "std_consensus_rank": std_rank,
        "n_draws": n_draws,
    }).sort_values("top3_inclusion_probability", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000,
                     help="Monte Carlo draws per cluster (paper/spec default is 5000; "
                          "2000 is a faster default that still gives a stable estimate "
                          "within about +-1-2 percentage points)")
    args = ap.parse_args()

    print("=" * 68)
    print(f"  Phase 6, optional — Monte Carlo stability ({args.draws} draws/cluster) — Uttarakhand")
    print("=" * 68)

    if not SURVIVORS_FILE.exists():
        print(f"\n  ERROR: {SURVIVORS_FILE} not found — run 07_feasibility_filter.py first.")
        return

    survivors = pd.read_csv(SURVIVORS_FILE)
    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for cid, grp in survivors.groupby("cluster_id"):
        passed = grp[grp["passes_all"]]
        if len(passed) < 2:
            print(f"\n  Cluster {int(cid)}: <2 survivors, skipping (same rule as 08_mcdm_ranking.py).")
            continue
        print(f"\n  Cluster {int(cid)}: {len(passed)} candidates, running {args.draws} draws ...")
        out = run_cluster(cid, passed, args.draws, rng)
        results.append(out)
        top5 = out.head(5)
        for _, row in top5.iterrows():
            print(f"    {row['name']:35s}  Top-3 prob={row['top3_inclusion_probability']*100:5.1f}%  "
                  f"mean_rank={row['mean_consensus_rank']:.1f}  std={row['std_consensus_rank']:.1f}")

    if not results:
        print("\n  ERROR: no cluster had >=2 survivors. Nothing to save.")
        return

    final = pd.concat(results, ignore_index=True)
    final.to_csv(OUT_FILE, index=False)
    print("\n" + "=" * 68)
    print(f"  Saved: {OUT_FILE}")
    print("  This unblocks build_input_package.py's 'monte_carlo_stability.csv' "
          "[SKIP-MISSING] warning — re-run it to pick this file up.")
    print("=" * 68)


if __name__ == "__main__":
    main()
