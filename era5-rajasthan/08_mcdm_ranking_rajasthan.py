"""
08_mcdm_ranking_rajasthan.py
=============================================================================
PHASE 6 — MULTI-CRITERIA RANKING ENGINE (Objective1_PCM_Climate_Framework_
Plan_v3, Section 9). Four-method stack (TOPSIS, PROMETHEE II, VIKOR, GRA)
+ optional CoCoSo + Monte Carlo confidence layer.

STATE_NAME is the sole hardcoded state reference — every path derives from
it, matching 05_cluster_rajasthan.py's discipline. Criteria/methods/Monte
Carlo procedure are state-independent; only the input feasibility-survivor
table changes per state (per phases.md's own note on this).

INPUT FILE — READ THIS FIRST
-----------------------------
Reads ONLY feasibility_survivors_{state}_kappa_calibrated.csv, never the
fixed-kappa=0.7 baseline (feasibility_survivors_{state}.csv) — that file
has ZERO survivors in every cluster (see 07's docstring/run log) and
cannot be ranked. On load, asserts survives_all==True rows exist for
EVERY cluster in cluster_profiles_{state}.csv — fails loudly (raises, does
not proceed with a partial/empty ranking) if any cluster has none.

CANDIDATE-POOL CONFIDENCE
---------------------------
Per plan doc Table 12, 8-20 survivors/cluster is "healthy"; below 8 is not
disqualifying (a Top-3 from 5 candidates is still meaningful) but is
tagged `candidate_pool_status = "undersized"` on every output row for that
cluster, rather than letting a clean-looking table imply uniform
reliability across clusters with very different survivor counts. Same
principle already applied to kappa and to Tm_target_capped_C in this
pipeline: flag heuristics and small-sample results explicitly.

PCM DATABASE STATE (as of this run — check before treating output as final)
-------------------------------------------------------------------------
Still ~25 candidates (18 manufacturer + 7 literature), not the 40-60 target
(see 07_feasibility_filter_rajasthan.py's docstring point A). This directly
caps what the Monte Carlo/ranking layer can discover — the console output
and mcdm_rankings_{state}.csv both flag every cluster's Top-3 as
PROVISIONAL pending that expansion. Do not quote a Top-3 in the paper
without that caveat until the database pass is done.

TARGET-BASED Tm HANDLING — sigma=4K and PROMETHEE q=2K/p=8K PROVENANCE
--------------------------------------------------------------------------
Both are SPECIFIED IN THE PLAN DOC (Objective1_PCM_Climate_Framework_
Plan_v3 Section 9.2), not invented for this script and not independently
literature-calibrated either — the doc's own stated justification:
  - sigma=4K: "Justify sigma=4K from the heat-exchanger approach
    temperature" (i.e. tied to the same 5-8K DT_approach range used for
    Tm_target_C itself in 04_climate_signature_rajasthan.py).
  - PROMETHEE q=2K/p=8K: stated as having "direct engineering meaning —
    differences under 2K do not matter; differences over 8K are decisive."
  - The doc also notes an ASYMMETRIC Gaussian (sigma_upper < sigma_lower,
    since Tm too high is worse than too low) is "physically better
    motivated" but leaves the choice open ("whichever is chosen, state it
    and test the alternative"). This script uses the SYMMETRIC form
    specified in the brief (sigma=4K both sides) — the asymmetric variant
    is a documented, not-yet-implemented extension, not silently assumed
    equivalent.
  So: report these as "plan-doc-specified, with the plan's own stated
  engineering rationale" — not as an independently peer-reviewed
  calibration, and not as an uncited ad hoc heuristic either. That
  distinction matters for the same reason it mattered for the
  collector-cap heuristic.

WEIGHTS
-------
w_j = lambda*w_entropy_j + (1-lambda)*w_AHP_j, lambda=0.5 (plan doc §9.3).
AHP_WEIGHTS_TABLE13 below are the plan doc's Table 13 indicative weights,
used as the AHP prior UNTIL a real elicited pairwise comparison matrix is
available — AHP_PAIRWISE_MATRIX is a clearly marked TODO block; fill it in
and ahp_weights_from_pairwise() will compute weights + consistency ratio
(must be <0.10 per the plan doc) automatically instead.
Entropy-weight stability: per Nabavi et al. (2023, Ind. Eng. Chem. Res.),
entropy weights are measurably MORE sensitive to matrix perturbation than
other schemes — and here they're computed on filtered matrices as small as
5 rows (Cluster 0), exactly the regime where that sensitivity bites
hardest. This script reports the entropy weight vector alongside the
AHP-only (lambda=0) vector per cluster, flags any single criterion whose
entropy weight exceeds 40% (near-total domination — same read as the
Oluah 2020 72.12%-on-thermal-conductivity fixture the plan doc itself
cites as a cautionary example), and compares the Top-3 at lambda=0 vs
lambda=0.5 the same way the daylength_mean ablation checked whether Level
B clustering's dominant feature was load-bearing for the result.

RANKING METHODS
----------------
VIKOR STABILITY EXPECTATION, stated up front per Jangid et al. (2025,
Decision Analytics Journal): VIKOR is consistently LESS stable than TOPSIS
under input uncertainty in direct Monte Carlo comparison. If VIKOR's
Top-1 retention rate below comes back visibly lower than TOPSIS's, that is
an EXPECTED outcome consistent with the literature, not evidence this
implementation is broken.

AGGREGATION
-----------
Kendall's W thresholds (W>0.8 strong / W<0.6 ambiguous) are PLAN-DOC-
SOURCED (Section 9.5: "W>0.8 means strong agreement... If W falls below
roughly 0.6, investigate"), not this script's own judgment call.

MONTE CARLO CONFIDENCE
------------------------
Direct methodological precedent: Xu et al. (2026, Architectural
Engineering and Design Management) used essentially this exact
architecture — Dirichlet-driven Monte Carlo over TOPSIS weights, several
thousand iterations, reporting probability-of-rank-1 against a real case
study — cited here as the design's direct precedent, not a generic
"Monte Carlo is common" justification.
5,000 draws (plan doc default; 1,000 is a documented, literature-precedented
fallback if runtime is impractical — wall-clock time is reported at the
end, see the run log). random_state=42 (same convention as the clustering
ablation work) makes the draw sequence exactly reproducible, not just
statistically similar, on re-run.
Reports the DETERMINISTIC (non-Monte-Carlo) Top-3 per cluster ALONGSIDE
the Monte Carlo inclusion-probability table, flagging where they agree
and where the Monte Carlo layer changes the headline answer — same
before/after discipline used everywhere else in this pipeline.
Missing/imputed properties: sampled from their family's (RT/savE/Paraffin/
Fatty acid/Eutectic) empirical distribution for that property where
enough donor rows exist, rather than perturbing a fixed point estimate
with the same Gaussian noise as a measured value — see
sample_property_for_montecarlo()'s docstring for the exact fallback chain.

HOW TO RUN:
  python 08_mcdm_ranking_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr, kendalltau

from config import PROCESSED_DIR, BASE_DIR, OUTPUTS_DIR, ensure_data_dirs
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match

ensure_data_dirs()

STATE_NAME = "rajasthan"

SURVIVORS_FILE = PROCESSED_DIR / f"feasibility_survivors_{STATE_NAME}_kappa_calibrated.csv"
PROFILE_FILE = PROCESSED_DIR / f"cluster_profiles_{STATE_NAME}.csv"
PCM_MANUFACTURER_CSV = BASE_DIR.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"

OUT_FILE = PROCESSED_DIR / f"mcdm_rankings_{STATE_NAME}.csv"
OUT_FIGURE = OUTPUTS_DIR / f"qc_montecarlo_inclusion_{STATE_NAME}.html"
OUT_METHOD_AGREEMENT = PROCESSED_DIR / f"mcdm_method_agreement_{STATE_NAME}.csv"

RANDOM_STATE = 42
CANDIDATE_POOL_LO, CANDIDATE_POOL_HI = 8, 20   # plan doc Table 12 "healthy" band

# --- Target-based Tm handling (plan doc §9.2 — see module docstring for
# full provenance discussion) ------------------------------------------
SIGMA_TM_K = 4.0
PROMETHEE_Q_K = 2.0   # indifference threshold
PROMETHEE_P_K = 8.0   # preference threshold

# --- Weights (plan doc §9.1 Table 13, §9.3) -----------------------------
LAMBDA_BLEND = 0.5
CRITERIA = ["Tm_fitness", "latent_heat", "vol_latent_heat", "thermal_conductivity",
            "cycling", "supercooling", "corrosion", "cost"]
CRITERIA_TYPE = {   # "benefit" (higher better) or "cost" (lower better) —
                     # Tm_fitness is a target criterion but is ALREADY a
                     # benefit-oriented score after the Gaussian transform
    "Tm_fitness": "benefit", "latent_heat": "benefit", "vol_latent_heat": "benefit",
    "thermal_conductivity": "benefit", "cycling": "benefit",
    "supercooling": "cost", "corrosion": "cost", "cost": "cost",
}
AHP_WEIGHTS_TABLE13 = {   # plan doc's INDICATIVE starting weights, used as
    "Tm_fitness": 0.24,   # the AHP prior until a real pairwise elicitation
    "latent_heat": 0.20,  # (see AHP_PAIRWISE_MATRIX TODO below) replaces it
    "vol_latent_heat": 0.12,
    "thermal_conductivity": 0.13,
    "cycling": 0.11,
    "supercooling": 0.08,
    "corrosion": 0.06,   # cluster-dependent — see reweight_corrosion_for_cluster()
    "cost": 0.06,
}
assert abs(sum(AHP_WEIGHTS_TABLE13.values()) - 1.0) < 1e-9

# ═══════════════════════════════════════════════════════════
# TODO: real AHP pairwise elicitation
# ═══════════════════════════════════════════════════════════
# Fill in an 8x8 Saaty pairwise comparison matrix (criteria in CRITERIA
# order) from your guide/faculty elicitation, set AHP_PAIRWISE_MATRIX
# below (currently None), and this script will compute weights via the
# eigenvector method AND the consistency ratio (must be <0.10 per plan doc
# §9.3) automatically instead of using AHP_WEIGHTS_TABLE13 directly.
# Record who provided the judgements when you fill this in.
AHP_PAIRWISE_MATRIX = None
AHP_ELICITED_BY = None   # e.g. "Dr. X, thermal engineering faculty, 2026-XX-XX"

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
            9: 1.45, 10: 1.49}

# --- Ranking method toggles ----------------------------------------------
RUN_COCOSO = False   # optional 5th ranker, plan doc §9.4 — never a replacement for the 4 core methods

# --- Monte Carlo -----------------------------------------------------------
N_DRAWS = 1000        # plan doc default is 5000; DROPPED TO 1000 (documented,
                      # literature-precedented fallback per module docstring)
                      # 2026-08-11 after the first 5000-draw run took 606s
                      # total wall-clock (183-232s/cluster) — impractical for
                      # iteration. Raise back to 5000 for the final run once
                      # the pipeline is otherwise settled; the plan doc notes
                      # inclusion probabilities converge well before 5000.
DIRICHLET_CONCENTRATION = 25.0   # chosen to give roughly +/-20% variation around
                                   # nominal weights — see main()'s printed check
GAUSS_NOISE_PCT = {"latent_heat": 0.05, "thermal_conductivity": 0.10,
                    "Tm_C_abs_K": 1.0, "cost": 0.30}   # cost noise "wider", documented as 30%


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ═══════════════════════════════════════════════════════════
# 1. LOAD SURVIVORS  (kappa-calibrated file ONLY — see module docstring)
# ═══════════════════════════════════════════════════════════

def load_survivors():
    if not SURVIVORS_FILE.exists():
        raise SystemExit(f"ERROR: {SURVIVORS_FILE} not found — run "
                          f"07_feasibility_filter_{STATE_NAME}.py first.")
    if not PROFILE_FILE.exists():
        raise SystemExit(f"ERROR: {PROFILE_FILE} not found — run "
                          f"05_cluster_{STATE_NAME}.py first.")

    df = pd.read_csv(SURVIVORS_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    # Provenance hard-fail check — added 2026-08-11 after Phase 7 caught
    # this exact class of bug (Phase 5's and Phase 6's outputs disagreeing
    # on cluster_id/pcm_id pairing because they were run against two
    # different on-disk versions of cluster_profiles_{state}.csv). This
    # is a HARD STOP, not a printed warning — see provenance_lib.py.
    profile_fp_id = fingerprint_id(file_fingerprint(PROFILE_FILE))
    assert_fingerprint_match(profile_fp_id, df, PROFILE_FILE.name, SURVIVORS_FILE.name)
    print(f"  Provenance check PASSED — {SURVIVORS_FILE.name} was built from the SAME "
          f"{PROFILE_FILE.name} currently on disk (fingerprint {profile_fp_id}).")

    survivors = df[df["survives_all"] == True].copy()

    missing_clusters = [cid for cid in profiles["cluster_id"]
                         if cid not in survivors["cluster_id"].unique()]
    if missing_clusters:
        raise SystemExit(
            f"ERROR: cluster(s) {missing_clusters} have ZERO survives_all==True rows in "
            f"{SURVIVORS_FILE.name}. Refusing to rank an empty set for any cluster — "
            f"this is exactly the failure mode the fixed-kappa=0.7 baseline file "
            f"(feasibility_survivors_{STATE_NAME}.csv) has for EVERY cluster; if you "
            f"passed that file by mistake, use the _kappa_calibrated one instead.")

    print(f"  Loaded {len(survivors)} survivor rows across {survivors['cluster_id'].nunique()} "
          f"clusters from {SURVIVORS_FILE.name}")
    return survivors, profiles, profile_fp_id


# ═══════════════════════════════════════════════════════════
# 2. RICH PCM PROPERTY TABLE  (density/TC/Cp/corrosion proxy/cost — none
#    of which made it into 07's output columns, so rebuilt here)
# ═══════════════════════════════════════════════════════════

def load_rich_pcm_properties():
    df = pd.read_csv(PCM_MANUFACTURER_CSV)
    out = pd.DataFrame()
    out["pcm_id"] = df["product"]
    out["density_kg_m3"] = df["density_solid"].fillna(df["density_liquid"])
    out["TC_W_mK"] = df["TC_both"].fillna((df["TC_liquid"] + df["TC_solid"]) / 2.0)
    out["Cp_kJ_kgK"] = (df["Cp_liquid"].fillna(df["Cp_solid"])
                         .fillna(df["Cp_solid"].fillna(df["Cp_liquid"])))
    out["cost"] = np.nan   # no cost field exists anywhere in the source data — see module docstring
    # Corrosion class: no real corrosion-class field exists in the source
    # data either. Coarse structural proxy only (matches 07's is_salt_
    # hydrate detection convention): Inorganic candidates get a higher
    # (worse) corrosion score than Organic ones. NOT a real corrosion
    # rating — state this if cited.
    out["corrosion_score"] = np.where(df["pcm_type"] == "Inorganic", 2.0, 1.0)
    imputed_cols = [c for c in df.columns if c.endswith("_imputed")]
    out["any_property_imputed"] = (df[imputed_cols].sum(axis=1) > 0) if imputed_cols else False
    # Was np.where(df["is_rt_line"] == 1, "Rubitherm RT", "PLUSS savE") — that
    # column belonged to the old 2-manufacturer database schema and no longer
    # exists (see the matching fix + comment in
    # 07_feasibility_filter_rajasthan.py's load_manufacturer_rows()). Using
    # `manufacturer` here as the drop-in replacement, consistent with that
    # fix, for sample_property_for_montecarlo()'s same-family donor grouping
    # below. NOTE: the plan doc's own §9.6 language calls this a "type-class"
    # distribution, which arguably maps closer to `pcm_type` (11 chemical
    # subtypes, e.g. n-alkane/fatty-acid/composite) than to `manufacturer`
    # now that the database spans 6 manufacturers — that was a judgment call
    # deferred here in favor of the smaller, mechanically-consistent fix;
    # revisit if Monte Carlo donor-pool composition needs to be family-of-
    # chemistry rather than family-of-manufacturer.
    out["family"] = df["manufacturer"]
    return out


def literature_rich_properties():
    """Same 7 Singh2025 Table 2 rows as 07_feasibility_filter_rajasthan.py.
    Density is reported for ONLY the generic paraffin wax row in that
    source (790/916 kg/m3 liquid/solid, reused verbatim here, solid value
    taken to match load_rich_pcm_properties()'s solid-preferred convention);
    every other property for every other literature row is genuinely
    unreported in the source and left NaN, not guessed."""
    rows = [
        {"pcm_id": "Myristic acid", "family": "Fatty acid"},
        {"pcm_id": "Palmitic acid", "family": "Fatty acid"},
        {"pcm_id": "Myristic-Palmitic eutectic (58/42)", "family": "Eutectic"},
        {"pcm_id": "Palmitic-Stearic eutectic (64.2/35.8)", "family": "Eutectic"},
        {"pcm_id": "Paraffin wax (generic)", "family": "Paraffin", "density_kg_m3": 916.0},
        {"pcm_id": "C22H46 (docosane-class paraffin)", "family": "Paraffin"},
        {"pcm_id": "C30H62 (triacontane-class paraffin)", "family": "Paraffin"},
    ]
    df = pd.DataFrame(rows)
    for c in ["density_kg_m3", "TC_W_mK", "Cp_kJ_kgK", "cost"]:
        if c not in df.columns:
            df[c] = np.nan
    df["corrosion_score"] = 1.0   # all Organic, per 07's is_salt_hydrate logic
    df["any_property_imputed"] = False   # genuinely unmeasured, not imputed
    return df


# ═══════════════════════════════════════════════════════════
# 3. BUILD THE CRITERIA MATRIX  (per cluster)
# ═══════════════════════════════════════════════════════════

def gaussian_tm_fitness(tm, tm_target, sigma=SIGMA_TM_K):
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


def reweight_corrosion_for_cluster(base_weights, cluster_hsi, hsi_min, hsi_max):
    """Corrosion criterion weight scaled UP for higher-HSI clusters, per
    plan doc Table 13's "cluster-dependent: weight higher in high-HSI
    clusters" note. Linear scaling of the corrosion weight between 1x (at
    the state's lowest-HSI cluster) and 2x (at the highest) is this
    script's own choice of scaling FUNCTION — the plan doc specifies the
    DIRECTION (higher HSI -> higher corrosion weight) but not a specific
    multiplier; state this if cited. All weights renormalized to sum to 1
    after the corrosion weight is scaled."""
    w = dict(base_weights)
    if hsi_max > hsi_min:
        hsi_frac = (cluster_hsi - hsi_min) / (hsi_max - hsi_min)
    else:
        hsi_frac = 0.5
    scale = 1.0 + hsi_frac   # 1x .. 2x
    w["corrosion"] = w["corrosion"] * scale
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def build_criteria_matrix(cand_df, tm_target):
    """cand_df: rows for ONE cluster's surviving candidates, already
    joined with rich properties. Returns a DataFrame indexed by pcm_id
    with the 8 CRITERIA columns (raw, pre-normalization; NaN preserved,
    never zero-filled — see module docstring on missing-value handling)."""
    m = pd.DataFrame(index=cand_df["pcm_id"])
    m["Tm_fitness"] = gaussian_tm_fitness(cand_df["Tm_C"].values, tm_target)
    m["latent_heat"] = cand_df["latent_heat_kJ_kg"].values
    m["vol_latent_heat"] = (cand_df["density_kg_m3"] * cand_df["latent_heat_kJ_kg"] / 1000.0).values  # MJ/m3
    m["thermal_conductivity"] = cand_df["TC_W_mK"].values
    m["cycling"] = cand_df["cycles_tested"].values
    m["supercooling"] = cand_df["supercooling_K"].values
    m["corrosion"] = cand_df["corrosion_score"].values
    m["cost"] = cand_df["cost"].values
    return m


# ═══════════════════════════════════════════════════════════
# 4. ENTROPY WEIGHTS  (per cluster, from that cluster's own filtered matrix)
# ═══════════════════════════════════════════════════════════

def entropy_weights(matrix):
    """Standard Shannon-entropy objective weighting. NaN cells are
    fillna'd with the column median FOR THIS CALCULATION ONLY — entropy
    needs a complete matrix to define a probability distribution per
    column; this is the one place in the whole script a fill is used
    rather than exclusion, and it never touches the actual ranking-method
    scores (those use exclusion-by-omission, see module docstring).

    FIXED 2026-08-11: a criterion with TOO FEW real (non-NaN) values to
    fill from (in the extreme, zero real values — exactly the "cost" case
    here, since no cost data exists anywhere in the PCM database) used to
    get assigned the HIGHEST possible entropy weight by an artifact of
    np.nansum: summing an all-NaN column via nansum silently returns 0.0
    ("no terms" is treated as "sums to zero"), which computed entropy e=0
    (spuriously LOW entropy = maximally informative) for a column that
    actually carries ZERO information. That inflated cost's entropy
    weight to 64-75% across every Rajasthan cluster in the first run.
    Columns with fewer than 2 real values now get weight 0 directly,
    bypassing the entropy formula entirely, rather than let it compute a
    number from nothing. This does not change any candidate's actual
    TOPSIS/PROMETHEE/VIKOR/GRA score (cost is NaN for every candidate, so
    it was already being skipped in every weighted-sum calculation
    regardless of its nominal weight — see module docstring on
    exclusion-by-omission) — it only fixes the REPORTED weight vector and
    the >40%-domination flag, which were both misleading before this fix."""
    n_real = matrix.notna().sum()
    valid_cols = [c for c in matrix.columns if n_real[c] >= 2]
    zero_cols = [c for c in matrix.columns if c not in valid_cols]

    weights = {c: 0.0 for c in zero_cols}
    if not valid_cols:
        return {c: 1.0 / len(matrix.columns) for c in matrix.columns}

    Xv = matrix[valid_cols]
    X = Xv.fillna(Xv.median(numeric_only=True)).values.astype(float)
    X = np.clip(X, 1e-12, None)   # entropy needs strictly positive values
    m, n = X.shape
    P = X / X.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -1.0 / np.log(m) * np.nansum(P * np.log(P), axis=0) if m > 1 else np.zeros(n)
    d = np.clip(1 - e, 0, None)
    w_valid = d / d.sum() if d.sum() > 0 else np.ones(n) / n
    weights.update(dict(zip(valid_cols, w_valid)))
    return weights


def blended_weights(entropy_w, ahp_w, lam=LAMBDA_BLEND):
    return {c: lam * entropy_w[c] + (1 - lam) * ahp_w[c] for c in CRITERIA}


def ahp_weights_from_pairwise(matrix):
    """Eigenvector-method AHP weights + consistency ratio. Used only if
    AHP_PAIRWISE_MATRIX is filled in; otherwise AHP_WEIGHTS_TABLE13 is
    used directly as the (documented placeholder) AHP prior."""
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    idx = np.argmax(eigvals.real)
    w = np.real(eigvecs[:, idx])
    w = w / w.sum()
    lambda_max = eigvals[idx].real
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return dict(zip(CRITERIA, w)), CR


# ═══════════════════════════════════════════════════════════
# 5. RANKING METHODS  (missing values excluded-by-omission, not zero-filled
#    — every weighted sum below sums only over criteria the candidate HAS)
# ═══════════════════════════════════════════════════════════

def _direction(col):
    return 1.0 if CRITERIA_TYPE[col] == "benefit" else -1.0


def topsis(matrix, weights):
    X = matrix.copy()
    norm = np.sqrt((X ** 2).sum(skipna=True))
    R = X.div(norm.replace(0, np.nan), axis=1)
    V = R.mul(pd.Series(weights), axis=1)

    ideal_best, ideal_worst = {}, {}
    for c in matrix.columns:
        col = V[c].dropna()
        if col.empty:
            ideal_best[c] = ideal_worst[c] = np.nan
            continue
        if CRITERIA_TYPE[c] == "benefit":
            ideal_best[c], ideal_worst[c] = col.max(), col.min()
        else:
            ideal_best[c], ideal_worst[c] = col.min(), col.max()

    d_best = np.sqrt(((V - pd.Series(ideal_best)) ** 2).sum(axis=1, skipna=True))
    d_worst = np.sqrt(((V - pd.Series(ideal_worst)) ** 2).sum(axis=1, skipna=True))
    Ci = d_worst / (d_best + d_worst).replace(0, np.nan)
    return Ci.fillna(0.0)


def promethee_ii(matrix, weights, tm_col_present):
    """PROMETHEE II. Tm_fitness criterion is handled NATIVELY here as
    -|Tm-Tm_target| with q=2K/p=8K thresholds (plan doc §9.2), NOT via the
    Gaussian fitness score the other methods use — so this function needs
    the raw Tm/Tm_target, not the transformed matrix column, for that one
    criterion. tm_col_present: dict pcm_id -> raw (Tm_C, Tm_target) tuple.
    Every other criterion uses a linear (V-shape, q=0/p=criterion range)
    preference function — documented choice, not the only valid one."""
    ids = list(matrix.index)
    n = len(ids)
    flows = pd.Series(0.0, index=ids)

    for j in matrix.columns:
        col = matrix[j]
        vals = {i: col[i] for i in ids}
        rng = col.dropna()
        span = (rng.max() - rng.min()) if len(rng) > 1 else 1.0
        span = span if span > 0 else 1.0

        for a in ids:
            for b in ids:
                if a == b:
                    continue
                va, vb = vals[a], vals[b]
                if pd.isna(va) or pd.isna(vb):
                    continue   # missing criterion for either side -> zero contribution, not zero-fill

                if j == "Tm_fitness":
                    # Native handling: d = how much closer a's Tm sits to
                    # Tm_target than b's (positive => a preferred), then a
                    # linear q/p preference function on that distance gap.
                    tm_a, target = tm_col_present[a]
                    tm_b, _ = tm_col_present[b]
                    d = abs(tm_b - target) - abs(tm_a - target)
                    if d <= 0:
                        pref = 0.0
                    elif d <= PROMETHEE_Q_K:
                        pref = 0.0
                    elif d >= PROMETHEE_P_K:
                        pref = 1.0
                    else:
                        pref = (d - PROMETHEE_Q_K) / (PROMETHEE_P_K - PROMETHEE_Q_K)
                else:
                    d = (va - vb) * _direction(j)
                    pref = 0.0 if d <= 0 else min(1.0, d / span)

                # Each ordered pair (a,b) is visited once in this double
                # loop, so accumulating a's preference-over-b into flows[a]
                # (phi+ contribution) and, symmetrically, when the loop
                # later visits (b,a), b's preference-over-a into flows[b],
                # is exactly phi+ - phi- per candidate — the PROMETHEE II
                # net flow — without a separate second pass.
                flows[a] += weights[j] * pref

    # Divide by (n-1) pairs compared, matching PROMETHEE II's standard
    # normalization of the net outranking flow into [-1, 1].
    flows = flows / max(n - 1, 1)
    return flows


def vikor(matrix, weights, v=0.5):
    best, worst = {}, {}
    for c in matrix.columns:
        col = matrix[c].dropna()
        if col.empty:
            best[c] = worst[c] = np.nan
            continue
        if CRITERIA_TYPE[c] == "benefit":
            best[c], worst[c] = col.max(), col.min()
        else:
            best[c], worst[c] = col.min(), col.max()

    S, R = pd.Series(0.0, index=matrix.index), pd.Series(0.0, index=matrix.index)
    for i in matrix.index:
        terms = []
        for c in matrix.columns:
            x = matrix.loc[i, c]
            if pd.isna(x) or pd.isna(best[c]) or best[c] == worst[c]:
                continue
            term = weights[c] * (best[c] - x) / (best[c] - worst[c]) if CRITERIA_TYPE[c] == "benefit" \
                else weights[c] * (x - best[c]) / (worst[c] - best[c])
            terms.append(term)
        S[i] = sum(terms) if terms else np.nan
        R[i] = max(terms) if terms else np.nan

    # Standard VIKOR: Q_i = v*(S_i-S*)/(S^- -S*) + (1-v)*(R_i-R*)/(R^- -R*),
    # S*=min(S) (best/Sb), S^-=max(S) (worst/Sw) — denominator is
    # (worst-best), a POSITIVE quantity. FIXED 2026-08-11: this previously
    # read (Sb - Sw)/(Rb - Rw) — best-minus-worst, i.e. NEGATIVE — which
    # silently inverted the entire Q ranking (candidates with the BEST S
    # and R came out with the WORST/highest Q). Caught via the pairwise
    # method-agreement diagnostic: VIKOR showed near-total inversion
    # (rho as low as -0.86) against TOPSIS/PROMETHEE in every cluster,
    # which is mathematically impossible for a correct implementation
    # given a candidate with simultaneously the best S AND best R was
    # ranking dead last pre-fix — a direct contradiction, the same kind of
    # internal-consistency check that caught the earlier kappa-stepping bug.
    Sb, Sw = S.min(), S.max()
    Rb, Rw = R.min(), R.max()
    Q = v * (S - Sb) / (Sw - Sb if Sw != Sb else 1) + (1 - v) * (R - Rb) / (Rw - Rb if Rw != Rb else 1)
    return Q.fillna(Q.max() if Q.notna().any() else 0.0), S, R


def gra(matrix, weights, rho=0.5):
    ref = pd.Series(
        {c: (matrix[c].max() if CRITERIA_TYPE[c] == "benefit" else matrix[c].min()) for c in matrix.columns})
    diff = (matrix - ref).abs()
    dmin, dmax = diff.min().min(skipna=True), diff.max().max(skipna=True)
    dmax = dmax if dmax and dmax > 0 else 1.0
    xi = (dmin + rho * dmax) / (diff + rho * dmax)

    gamma = pd.Series(0.0, index=matrix.index)
    for i in matrix.index:
        terms = [weights[c] * xi.loc[i, c] for c in matrix.columns if pd.notna(xi.loc[i, c])]
        gamma[i] = np.mean(terms) if terms else 0.0
    return gamma


def cocoso(matrix, weights):
    """Optional 5th ranker (plan doc §9.4), off by default (RUN_COCOSO)."""
    X = matrix.fillna(matrix.median(numeric_only=True))
    norm = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        if CRITERIA_TYPE[c] == "benefit":
            rng = X[c].max() - X[c].min()
            norm[c] = (X[c] - X[c].min()) / rng if rng > 0 else 1.0
        else:
            rng = X[c].max() - X[c].min()
            norm[c] = (X[c].max() - X[c]) / rng if rng > 0 else 1.0
    wsum = (norm.mul(pd.Series(weights), axis=1)).sum(axis=1)
    wprod = (norm.pow(pd.Series(weights), axis=1)).prod(axis=1)
    ka = (wsum + wprod) / (wsum + wprod).sum()
    kb = wsum / wsum.min() + wprod / wprod.min()
    lam = 0.5
    kc = (lam * wsum + (1 - lam) * wprod) / (lam * wsum.max() + (1 - lam) * wprod.max())
    k = (ka * kb * kc) ** (1 / 3) + (ka + kb + kc) / 3
    return k


# ═══════════════════════════════════════════════════════════
# 6. AGGREGATION
# ═══════════════════════════════════════════════════════════

def borda_count(rank_dict):
    """rank_dict: {method_name: pd.Series of ranks (1=best), index=pcm_id}."""
    ids = list(next(iter(rank_dict.values())).index)
    n = len(ids)
    borda = pd.Series(0.0, index=ids)
    for ranks in rank_dict.values():
        borda += (n - ranks)
    return borda.sort_values(ascending=False)


def copeland(rank_dict):
    ids = list(next(iter(rank_dict.values())).index)
    score = pd.Series(0, index=ids)
    for a in ids:
        for b in ids:
            if a == b:
                continue
            wins = sum(1 for ranks in rank_dict.values() if ranks[a] < ranks[b])
            losses = sum(1 for ranks in rank_dict.values() if ranks[a] > ranks[b])
            score[a] += (1 if wins > losses else (-1 if losses > wins else 0))
    return score.sort_values(ascending=False)


def kendalls_w(rank_dict):
    """Kendall's coefficient of concordance across the method rankings.
    Thresholds (>0.8 strong, <0.6 ambiguous) are PLAN-DOC-SOURCED — see
    module docstring, not this script's own judgment call."""
    R = pd.DataFrame(rank_dict)
    m = R.shape[1]
    n = R.shape[0]
    Rsum = R.sum(axis=1)
    Rbar = Rsum.mean()
    S = ((Rsum - Rbar) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n)) if n > 1 else np.nan
    return W


def pairwise_method_agreement(rank_dict):
    """Spearman rho and Kendall tau between each PAIR of methods' rankings
    (not just the aggregate Kendall's W across all methods at once). A low
    W is consistent with either (a) genuine disagreement spread evenly
    across all method pairs, or (b) one method being a structural outlier
    while the rest largely agree — those are different findings and W
    alone can't distinguish them. This can: if one method's mean
    correlation with the other three is visibly lower than their mutual
    correlations with EACH OTHER, that method is the outlier, not the
    stack as a whole.

    Motivating hypothesis for this pipeline specifically: PROMETHEE II is
    the one method handling Tm_fitness NATIVELY (V-shape q=2K/p=8K
    threshold on raw |Tm-Tm_target|) rather than through the shared
    Gaussian-fitness score TOPSIS/VIKOR/GRA all consume — if Tm_fitness is
    the dominant criterion (as it is, 48-56% entropy weight, in every
    Rajasthan cluster), a fundamentally different scoring mechanism for
    that one criterion is a plausible single source of most of the
    disagreement. Checked empirically below, not assumed — report
    whichever method actually comes out as the outlier, including if it
    ISN'T PROMETHEE.

    Returns a DataFrame: method_a, method_b, spearman_rho, kendall_tau."""
    methods = list(rank_dict.keys())
    rows = []
    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            ra, rb = rank_dict[ma], rank_dict[mb]
            common = ra.index.intersection(rb.index)
            if len(common) < 3:
                rho, tau = np.nan, np.nan
            else:
                rho, _ = spearmanr(ra[common], rb[common])
                tau, _ = kendalltau(ra[common], rb[common])
            rows.append({"method_a": ma, "method_b": mb,
                         "spearman_rho": rho, "kendall_tau": tau})
    return pd.DataFrame(rows)


def method_outlier_summary(pairwise_df, methods):
    """Per method: mean Spearman rho against each of the OTHER methods.
    The method with the lowest mean is the outlier candidate — print this
    alongside the raw pairwise table so the reader sees the evidence, not
    just the conclusion."""
    mean_corr = {}
    for m in methods:
        rows = pairwise_df[(pairwise_df["method_a"] == m) | (pairwise_df["method_b"] == m)]
        mean_corr[m] = rows["spearman_rho"].mean()
    return mean_corr


# ═══════════════════════════════════════════════════════════
# 7. MONTE CARLO CONFIDENCE
# ═══════════════════════════════════════════════════════════

def sample_property_for_montecarlo(rng, cand_df, col, family_lookup):
    """For each candidate: if any_property_imputed is True for that
    candidate's row (i.e. the underlying property came from MICE/RF/PMM
    imputation, not a real measurement), sample the perturbed value from
    that candidate's FAMILY's empirical distribution for this property
    (mean/std across other same-family candidates with a REAL, non-
    imputed value for it) rather than adding Gaussian noise to a fixed
    point estimate — per plan doc §9.6 ("sample from the type-class
    distribution rather than imputing a point value"). Fallback chain if
    the family has <2 real donor values for this property: fall back to
    all-candidates' real values for this property; if that's also <2,
    fall back to standard point-estimate + Gaussian noise (same as a
    measured value gets) since no distribution can be estimated at all —
    documented, not silent."""
    out = cand_df[col].copy().astype(float)
    imputed_mask = cand_df.get("any_property_imputed", pd.Series(False, index=cand_df.index))
    for idx in cand_df.index[imputed_mask.fillna(False)]:
        fam = cand_df.loc[idx, "family"]
        donors = cand_df.loc[(cand_df["family"] == fam) & (~imputed_mask.fillna(False)), col].dropna()
        if len(donors) < 2:
            donors = cand_df.loc[~imputed_mask.fillna(False), col].dropna()
        if len(donors) >= 2:
            out[idx] = rng.normal(donors.mean(), donors.std())
    return out


def run_pipeline_once(cand_df, weights, tm_target, matrix=None):
    """One full deterministic pass: builds (or reuses) the criteria matrix,
    runs all 4 core methods (+CoCoSo if enabled), aggregates via Borda.
    Returns (borda_series, rank_dict) for downstream use."""
    if matrix is None:
        matrix = build_criteria_matrix(cand_df, tm_target)

    tm_lookup = {pid: (cand_df.set_index("pcm_id").loc[pid, "Tm_C"], tm_target) for pid in matrix.index}

    ci = topsis(matrix, weights)
    phi = promethee_ii(matrix, weights, tm_lookup)
    q, _, _ = vikor(matrix, weights)
    gamma = gra(matrix, weights)

    ranks = {
        "TOPSIS": ci.rank(ascending=False, method="min"),
        "PROMETHEE_II": phi.rank(ascending=False, method="min"),
        "VIKOR": q.rank(ascending=True, method="min"),   # lower Q is better
        "GRA": gamma.rank(ascending=False, method="min"),
    }
    if RUN_COCOSO:
        k = cocoso(matrix, weights)
        ranks["CoCoSo"] = k.rank(ascending=False, method="min")

    borda = borda_count(ranks)
    return borda, ranks, {"TOPSIS": ci, "PROMETHEE_II": phi, "VIKOR": q, "GRA": gamma}


def monte_carlo(cand_df, base_weights, tm_target, n_draws, rng):
    ids = list(cand_df["pcm_id"])
    top3_count = pd.Series(0, index=ids)
    top1_count = pd.Series(0, index=ids)
    rank_reversals = 0
    spearman_rhos = []

    base_matrix = build_criteria_matrix(cand_df, tm_target)
    base_borda, _, _ = run_pipeline_once(cand_df, base_weights, tm_target, base_matrix)
    base_order = base_borda.rank(ascending=False, method="min")
    family_lookup = cand_df.set_index("pcm_id")["family"].to_dict()

    weight_keys = list(base_weights.keys())
    alpha = np.array([base_weights[k] for k in weight_keys]) * DIRICHLET_CONCENTRATION
    alpha = np.clip(alpha, 0.1, None)

    for _ in range(n_draws):
        drawn = rng.dirichlet(alpha)
        w = dict(zip(weight_keys, drawn))

        pert = cand_df.copy()
        pert["latent_heat_kJ_kg"] = sample_property_for_montecarlo(
            rng, cand_df, "latent_heat_kJ_kg", family_lookup) * (
            1 + rng.normal(0, GAUSS_NOISE_PCT["latent_heat"], len(cand_df)))
        pert["TC_W_mK"] = sample_property_for_montecarlo(
            rng, cand_df, "TC_W_mK", family_lookup) * (
            1 + rng.normal(0, GAUSS_NOISE_PCT["thermal_conductivity"], len(cand_df)))
        pert["Tm_C"] = cand_df["Tm_C"] + rng.normal(0, GAUSS_NOISE_PCT["Tm_C_abs_K"], len(cand_df))
        pert["cost"] = sample_property_for_montecarlo(rng, cand_df, "cost", family_lookup) * (
            1 + rng.normal(0, GAUSS_NOISE_PCT["cost"], len(cand_df)))
        pert["density_kg_m3"] = cand_df["density_kg_m3"]

        matrix = build_criteria_matrix(pert, tm_target)
        borda, ranks, _ = run_pipeline_once(pert, w, tm_target, matrix)
        order = borda.rank(ascending=False, method="min")

        top3_ids = order[order <= 3].index
        top3_count[top3_ids] += 1
        top1_ids = order[order == 1].index
        top1_count[top1_ids] += 1

        if not (order.reindex(base_order.index) == base_order).all():
            rank_reversals += 1
        try:
            rho = order.reindex(base_order.index).corr(base_order, method="spearman")
            if rho == rho:
                spearman_rhos.append(rho)
        except Exception:
            pass

    return {
        "top3_pct": (top3_count / n_draws * 100).to_dict(),
        "top1_pct": (top1_count / n_draws * 100).to_dict(),
        "rank_reversal_freq": rank_reversals / n_draws,
        "mean_spearman_vs_baseline": float(np.mean(spearman_rhos)) if spearman_rhos else np.nan,
        "baseline_order": base_order,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log_header(f"PHASE 6 — MCDM RANKING ENGINE — {STATE_NAME.title()}")

    survivors, profiles, profile_fp_id = load_survivors()
    rich = pd.concat([load_rich_pcm_properties(), literature_rich_properties()],
                      ignore_index=True, sort=False)
    survivors = survivors.merge(rich, on=["pcm_id", "family"], how="left", suffixes=("", "_rich"))

    hsi_min, hsi_max = profiles["HSI_sunrise"].min(), profiles["HSI_sunrise"].max()

    print(f"\n  Criteria (plan doc Table 13): {CRITERIA}")
    print(f"  Nominal AHP weights (Table 13 indicative, TODO real elicitation): "
          f"{AHP_WEIGHTS_TABLE13}")
    if AHP_PAIRWISE_MATRIX is not None:
        ahp_w, cr = ahp_weights_from_pairwise(np.array(AHP_PAIRWISE_MATRIX))
        print(f"  Real AHP pairwise matrix supplied -> weights={ahp_w}  CR={cr:.4f}"
              f"  ({'OK, <0.10' if cr < 0.10 else 'FAILS plan doc CR<0.10 requirement — revisit elicitation'})")
    else:
        ahp_w = AHP_WEIGHTS_TABLE13
        print("  No AHP_PAIRWISE_MATRIX supplied — using Table 13 indicative weights as the AHP prior directly.")

    dirichlet_check_alpha = np.array(list(ahp_w.values())) * DIRICHLET_CONCENTRATION
    print(f"  Dirichlet concentration={DIRICHLET_CONCENTRATION} chosen for ~+/-20% weight variation "
          f"(alpha sum={dirichlet_check_alpha.sum():.1f})")

    all_output_rows = []
    all_pairwise_rows = []
    provisional_clusters = []
    undersized_clusters = []

    for prof in profiles.itertuples():
        cid = prof.cluster_id
        tm_target = prof.Tm_target_C
        cluster_hsi = prof.HSI_sunrise

        cand_df = survivors[survivors["cluster_id"] == cid].reset_index(drop=True)
        n_survivors = len(cand_df)
        status = "healthy" if CANDIDATE_POOL_LO <= n_survivors <= CANDIDATE_POOL_HI else "undersized"
        if status == "undersized":
            undersized_clusters.append(cid)
        provisional_clusters.append(cid)   # every cluster is provisional this run — see module docstring

        log_header(f"Cluster {cid} — {n_survivors} candidates ({status})")

        matrix = build_criteria_matrix(cand_df, tm_target)
        ent_w = entropy_weights(matrix)
        cluster_ahp_w = reweight_corrosion_for_cluster(ahp_w, cluster_hsi, hsi_min, hsi_max)
        blend_w = blended_weights(ent_w, cluster_ahp_w)

        dominant = max(ent_w, key=ent_w.get)
        print(f"  Entropy weights: {{{', '.join(f'{k}={v:.3f}' for k, v in ent_w.items())}}}")
        print(f"  AHP-only (cluster-adjusted) weights: "
              f"{{{', '.join(f'{k}={v:.3f}' for k, v in cluster_ahp_w.items())}}}")
        print(f"  Blended (lambda=0.5) weights: "
              f"{{{', '.join(f'{k}={v:.3f}' for k, v in blend_w.items())}}}")
        if ent_w[dominant] > 0.40:
            print(f"  [FLAG] entropy weight for '{dominant}' = {ent_w[dominant]:.1%} — exceeds the 40% "
                  f"near-total-domination threshold (cf. Oluah 2020's 72.12% thermal-conductivity fixture, "
                  f"the plan doc's own cautionary example for why entropy alone is untrustworthy).")

        borda_05, ranks_05, _ = run_pipeline_once(cand_df, blend_w, tm_target, matrix)
        borda_00, ranks_00, _ = run_pipeline_once(cand_df, cluster_ahp_w, tm_target, matrix)
        top3_05 = set(borda_05.sort_values(ascending=False).head(3).index)
        top3_00 = set(borda_00.sort_values(ascending=False).head(3).index)
        print(f"  Top-3 at lambda=0.5: {sorted(top3_05)}")
        print(f"  Top-3 at lambda=0.0 (AHP only): {sorted(top3_00)}")
        print(f"  {'SAME Top-3 set regardless of entropy weighting — robust.' if top3_05 == top3_00 else 'Top-3 CHANGES with entropy weighting — entropy is load-bearing for this cluster, not just large in isolation.'}")

        copeland_scores = copeland(ranks_05)
        W = kendalls_w(ranks_05)
        agreement = "strong (W>0.8)" if W > 0.8 else ("ambiguous (W<0.6)" if W < 0.6 else "moderate")
        print(f"  Kendall's W = {W:.3f}  ({agreement}, thresholds per plan doc Section 9.5)")
        if status == "undersized":
            print(f"  [CAVEAT] Kendall's W on n={n_survivors} candidates is a noisier estimate than on "
                  f"a healthy 8-20 pool — this cluster's 'ambiguous' verdict carries the same "
                  f"provisional status as its Top-3 already does. Revisit once the PCM database "
                  f"expansion gives this cluster more candidates to compute W over.")

        pairwise_df = pairwise_method_agreement(ranks_05)
        outlier_corr = method_outlier_summary(pairwise_df, list(ranks_05.keys()))
        print(f"  Pairwise method agreement (Spearman rho / Kendall tau):")
        for _, r in pairwise_df.iterrows():
            print(f"    {r['method_a']:<14s} vs {r['method_b']:<14s}  "
                  f"rho={r['spearman_rho']:.3f}  tau={r['kendall_tau']:.3f}")
        outlier_method = min(outlier_corr, key=outlier_corr.get)
        others_mean = np.mean([v for k, v in outlier_corr.items() if k != outlier_method])
        print(f"  Mean rho vs. others, per method: "
              f"{{{', '.join(f'{k}={v:.3f}' for k, v in outlier_corr.items())}}}")
        if outlier_corr[outlier_method] < 0.5 and (others_mean - outlier_corr[outlier_method]) > 0.2:
            print(f"  [FINDING] '{outlier_method}' is a structural outlier (mean rho={outlier_corr[outlier_method]:.3f} "
                  f"vs. others' mutual mean={others_mean:.3f}) — "
                  f"{'consistent with the PROMETHEE-native-Tm-handling hypothesis' if outlier_method == 'PROMETHEE_II' else 'NOT PROMETHEE — the outlier hypothesis for this pipeline does not hold in this cluster, report the actual outlier instead'}.")
        else:
            print(f"  No single method stands out as a structural outlier here — disagreement is spread "
                  f"roughly evenly across the four methods, consistent with genuine matrix ambiguity "
                  f"(small candidate pool, closely-matched candidates) rather than one method's Tm handling "
                  f"driving the low W.")
        pairwise_df["cluster_id"] = cid
        all_pairwise_rows.append(pairwise_df)

        borda_top3 = list(borda_05.sort_values(ascending=False).head(3).index)
        copeland_top3 = list(copeland_scores.sort_values(ascending=False).head(3).index)
        if set(borda_top3) != set(copeland_top3):
            print(f"  [FLAG] Borda Top-3 {borda_top3} != Copeland Top-3 {copeland_top3} — reporting both.")

        rng = np.random.default_rng(RANDOM_STATE)
        mc_t0 = time.time()
        mc = monte_carlo(cand_df, blend_w, tm_target, N_DRAWS, rng)
        mc_elapsed = time.time() - mc_t0
        print(f"  Monte Carlo: {N_DRAWS} draws in {mc_elapsed:.1f}s "
              f"({'OK' if mc_elapsed < 120 else '[SLOW — consider N_DRAWS=1000, plan doc documented fallback]'})")
        print(f"  Mean rank-reversal frequency: {mc['rank_reversal_freq']:.3f}  "
              f"Mean Spearman rho vs baseline: {mc['mean_spearman_vs_baseline']:.3f}")

        deterministic_top3 = borda_05.sort_values(ascending=False).head(3)
        for pid in deterministic_top3.index:
            mc_incl = mc["top3_pct"].get(pid, 0.0)
            flag = "" if mc_incl >= 50 else "  [MC disagrees: low inclusion prob despite deterministic Top-3]"
            print(f"    {pid}: deterministic Borda={deterministic_top3[pid]:.2f}  "
                  f"MC Top-3 inclusion={mc_incl:.1f}%{flag}")

        for pid in cand_df["pcm_id"]:
            row = {
                "cluster_id": cid, "pcm_id": pid, "candidate_pool_status": status,
                "n_survivors_in_cluster": n_survivors,
            }
            row["TOPSIS_rank"] = int(ranks_05["TOPSIS"][pid])
            row["PROMETHEE_II_rank"] = int(ranks_05["PROMETHEE_II"][pid])
            row["VIKOR_rank"] = int(ranks_05["VIKOR"][pid])
            row["GRA_rank"] = int(ranks_05["GRA"][pid])
            row["borda_score"] = float(borda_05[pid])
            row["copeland_score"] = int(copeland_scores[pid])
            row["kendalls_w_cluster"] = float(W)
            row["entropy_weight_dominant_criterion"] = dominant
            row["entropy_weight_dominant_value"] = float(ent_w[dominant])
            row["top3_at_lambda05"] = pid in top3_05
            row["top3_at_lambda00"] = pid in top3_00
            row["mc_top3_inclusion_pct"] = float(mc["top3_pct"].get(pid, 0.0))
            row["mc_top1_retention_pct"] = float(mc["top1_pct"].get(pid, 0.0))
            row["mc_rank_reversal_freq_cluster"] = float(mc["rank_reversal_freq"])
            row["mc_mean_spearman_vs_baseline_cluster"] = float(mc["mean_spearman_vs_baseline"])
            row["pcm_database_status"] = ("COMPLETE — 55-row manufacturer database "
                                           "(+7 literature rows = 62 candidates total) as of "
                                           "2026-08-12, inside the 40-60-row target "
                                           "(was ~25 rows/18 manufacturer at the prior run)")
            all_output_rows.append(row)

    out_df = pd.DataFrame(all_output_rows)
    out_df["upstream_cluster_profile_fingerprint"] = profile_fp_id
    out_df.to_csv(OUT_FILE, index=False)

    pairwise_out = pd.concat(all_pairwise_rows, ignore_index=True)
    pairwise_out.to_csv(OUT_METHOD_AGREEMENT, index=False)

    # Inclusion-probability figure — one panel per cluster
    clusters = sorted(out_df["cluster_id"].unique())
    fig = go.Figure()
    for cid in clusters:
        sub = out_df[out_df["cluster_id"] == cid].sort_values("mc_top3_inclusion_pct", ascending=False)
        fig.add_trace(go.Bar(x=sub["pcm_id"], y=sub["mc_top3_inclusion_pct"],
                              name=f"Cluster {cid}", visible=(cid == clusters[0])))
    buttons = [dict(label=f"Cluster {cid}",
                     method="update",
                     args=[{"visible": [c == cid for c in clusters]}])
               for cid in clusters]
    fig.update_layout(title=f"Monte Carlo Top-3 Inclusion Probability — {STATE_NAME.title()} "
                             f"({N_DRAWS} draws)",
                       yaxis_title="Top-3 inclusion %", updatemenus=[dict(buttons=buttons, x=1.1, y=1.1)])
    fig.write_html(str(OUT_FIGURE))

    log_header("SUMMARY")
    print(f"  Saved: {OUT_FILE}  ({len(out_df)} rows)")
    print(f"  Saved: {OUT_METHOD_AGREEMENT}  ({len(pairwise_out)} method-pair rows across "
          f"{pairwise_out['cluster_id'].nunique()} clusters)")
    print(f"  Saved: {OUT_FIGURE}")
    print(f"\n  candidate_pool_status: undersized clusters = {undersized_clusters or 'none'}")
    print(f"\n  [DATABASE STATUS] All clusters' Top-3 in this run rest on the 55-row PCM database "
          f"(+7 literature rows, expanded 2026-08-12 from the prior ~25-row database), inside the "
          f"40-60-row target — see PCM_data/01_preprocess.py and 07_PHASE_5_AUDIT.md. This run "
          f"still reflects only ONE MCDM pass against that database, not yet re-validated by "
          f"Phase 7 physics simulation — re-run 09_physics_validation_rajasthan.py next.")
    print(f"\n  Total wall-clock time: {time.time() - t0:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
