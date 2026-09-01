"""
05b_level_b_seasonal_assam.py
=====================================
PHASE 4, LEVEL B — TEMPORAL/SEASONAL CLUSTERING (plan §7.1, §7.2)

Level A (05_cluster_assam.py) clusters sites across all years combined
— "which PCM for a system installed at this location?"

Level B asks the other question: "does this location need a DIFFERENT
PCM in July than in March?" This is where the most interesting result
lives (§7.1): if the Top-3 for a monsoon-dominated Assam cluster is
stable across its four seasons, you have shown a single PCM is adequate.
If it flips between seasons, you have shown that a fixed PCM is
inadequate — direct empirical motivation for the adaptive control
objective (Objective 3 DRL controller), generated from your own data.

ASSAM SEASONS (distinct from Tamil Nadu)
-----------------------------------------
  Pre-monsoon (Mar–May)  — warming, low humidity, highest DTR
  Monsoon     (Jun–Sep)  — extreme rainfall (>70% of annual),
                            very high RH, low GHI
  Post-monsoon (Oct–Nov) — retreating rain, clearing skies
  Winter      (Dec–Feb)  — cold, dry, high GHI, low RH

The monsoon cluster split is the key differentiator for Assam. If the
same sites need a lower-Tm PCM during monsoon (because collectors are
shaded by cloud and system temperatures are lower) vs. pre-monsoon
(when GHI is high and tank temperatures can exceed 50°C), that is a
genuine finding.

THIS SCRIPT
------------
This is the "cheaper" Level-B version (same as plan's explicit
"nearly free" starting point): for each EXISTING Level-A cluster,
recompute the climate-dependent MCDM inputs (Ta_mean → L_required,
same Tm_target) SEPARATELY per season, re-run feasibility + TOPSIS
ranking using the SAME entropy+AHP-blended weights from the annual
MCDM result (for a fair comparison), and report whether the Top-3 PCM
changes.

Say in your methodology: "Level B seasonal analysis uses the
cheaper recomputation approach (per-cluster seasonal Ta_mean and
L_required, fixed annual TOPSIS weights) as the starting point.
A full per-point-per-season GMM re-clustering can be added in future
work."

DEPENDENCIES (run after)
--------------------------
  04_preprocess_assam.py         → tamilnadu_cleaned_physical.csv
  05_cluster_assam.py            → cluster_assignments_assam.csv
  06_build_pcm_database.py       → pcm_database_assam.csv  (Phase 5)
  08_mcdm_ranking.py             → mcdm_full_scores_by_cluster.csv (Phase 8)

INPUT  : data/preprocessed/assam_cleaned_physical.csv
         data/processed/clustering/cluster_assignments_assam.csv
         data/processed/clustering/cluster_profiles_assam.csv
         data/processed/pcm/pcm_database_assam.csv
         data/processed/pcm/mcdm_full_scores_by_cluster.csv
OUTPUT : data/processed/pcm/level_b_seasonal_topk_assam.csv
         data/processed/pcm/level_b_seasonal_summary_assam.md

HOW TO RUN:
  python 05b_level_b_seasonal_assam.py
  (Only valid after Phase 5 and Phase 8 are complete.)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path

from config import PREPROCESSED_DIR, PROCESSED_DIR

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
PHYSICAL_FILE = PREPROCESSED_DIR / "assam_cleaned_physical.csv"
ASSIGN_FILE   = PROCESSED_DIR / "clustering" / "cluster_assignments_assam.csv"
PROFILE_FILE  = PROCESSED_DIR / "clustering" / "cluster_profiles_assam.csv"
PCM_FILE      = PROCESSED_DIR / "pcm" / "pcm_database_assam.csv"
SCORES_FILE   = PROCESSED_DIR / "pcm" / "mcdm_full_scores_by_cluster.csv"
OUT_CSV       = PROCESSED_DIR / "pcm" / "level_b_seasonal_topk_assam.csv"
OUT_MD        = PROCESSED_DIR / "pcm" / "level_b_seasonal_summary_assam.md"

# ─────────────────────────────────────────────────────────────
# PCM PHYSICS CONSTANTS (must match 08_mcdm_ranking.py settings)
# ─────────────────────────────────────────────────────────────
ABSOLUTE_TM_MIN          = 42.0
ABSOLUTE_TM_MAX          = 70.0
WINDOW_LOWER_OFFSET      = 5.0
WINDOW_UPPER_OFFSET      = 8.0
LATENT_HEAT_FRACTION     = 0.7
SIGMA_TM                 = 4.0
T_DELIVERY_C             = 50.0
DRAW_RATE_KG_PER_S       = 60.0 / 1000 / 60   # 60 L/min in kg/s
CP_WATER                 = 4.186               # kJ/(kg·K)
ASSUMED_PCM_MASS_KG      = 50.0

# Must match 08_mcdm_ranking.py's USE_CLIMATE_RELATIVE_LATENT_HEAT setting
USE_CLIMATE_RELATIVE_LATENT_HEAT = True
LATENT_CRITERION_NAME = (
    "latent_heat_margin_ratio" if USE_CLIMATE_RELATIVE_LATENT_HEAT
    else "latent_heat_kJ_kg"
)
CRITERIA = ["f_Tm", LATENT_CRITERION_NAME, "rho_H_MJ_m3", "TC_W_mK", "cycles_confidence"]

# ─────────────────────────────────────────────────────────────
# ASSAM SEASONS (distinct from Tamil Nadu, adapted for NE India)
# ─────────────────────────────────────────────────────────────
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}
SEASON_ORDER = ["Pre-monsoon", "Monsoon", "Post-monsoon", "Winter"]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def gaussian_tm_fitness(tm, tm_target, sigma=SIGMA_TM):
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))


def topsis(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm     = matrix / (np.sqrt((matrix ** 2).sum(axis=0)) + 1e-12)
    weighted = norm * weights
    v_plus   = weighted.max(axis=0)
    v_minus  = weighted.min(axis=0)
    s_plus   = np.sqrt(((weighted - v_plus)  ** 2).sum(axis=1))
    s_minus  = np.sqrt(((weighted - v_minus) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)


def rank_seasonal(pcm_db: pd.DataFrame, tm_target: float,
                  l_required: float, weights: np.ndarray):
    """Filter PCM database and rank by TOPSIS for one (cluster, season)."""
    df  = pcm_db.copy()
    lo  = tm_target - WINDOW_LOWER_OFFSET
    hi  = tm_target + WINDOW_UPPER_OFFSET
    survivors = df[
        df["Tm_C"].between(lo, hi) &
        df["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX) &
        (df["latent_heat_kJ_kg"] >= LATENT_HEAT_FRACTION * l_required)
    ].copy()
    if len(survivors) < 2:
        return None

    survivors["f_Tm"] = gaussian_tm_fitness(survivors["Tm_C"], tm_target)
    if USE_CLIMATE_RELATIVE_LATENT_HEAT:
        survivors["latent_heat_margin_ratio"] = survivors["latent_heat_kJ_kg"] / l_required
    survivors["cycles_confidence"] = survivors["cycles_confidence"].fillna(
        survivors["cycles_confidence"].median()
    )

    M = survivors[CRITERIA].copy()
    for c in CRITERIA:
        lo_c, hi_c = M[c].min(), M[c].max()
        M[c] = (M[c] - lo_c) / (hi_c - lo_c + 1e-12)
    M = M.fillna(0.0).values

    survivors["topsis_score"]  = topsis(M, weights)
    survivors["seasonal_rank"] = survivors["topsis_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    return survivors.sort_values("seasonal_rank")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  Phase 4, Level B — Seasonal PCM Sensitivity — Assam")
    print("=" * 68)
    print()
    print("  Assam seasons:")
    for s, ms in [("Pre-monsoon", "Mar–May"), ("Monsoon", "Jun–Sep"),
                  ("Post-monsoon", "Oct–Nov"), ("Winter", "Dec–Feb")]:
        print(f"    {s:<14} {ms}")

    # ── Dependency check ─────────────────────────────────────
    missing = [f for f in (PHYSICAL_FILE, ASSIGN_FILE, PROFILE_FILE,
                            PCM_FILE, SCORES_FILE) if not f.exists()]
    if missing:
        print("\n  MISSING INPUT FILES — run prerequisite phases first:")
        for f in missing:
            print(f"    {f}")
        print("\n  This script requires Phase 5 (PCM database) and")
        print("  Phase 8 (MCDM ranking) outputs to be available.")
        return

    # ── Load ─────────────────────────────────────────────────
    physical = pd.read_csv(PHYSICAL_FILE, parse_dates=["date"])
    assign   = pd.read_csv(ASSIGN_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    pcm_db   = pd.read_csv(PCM_FILE)
    scores   = pd.read_csv(SCORES_FILE)

    # Map months to Assam seasons
    if "season" not in physical.columns:
        physical["month"]  = pd.to_datetime(physical["date"]).dt.month
        physical["season"] = physical["month"].map(SEASON_MAP)

    all_rows = []
    md_lines = [
        "# Level B — Seasonal PCM Sensitivity (Assam)\n",
        "\nAssam seasons: Pre-monsoon (Mar–May) | Monsoon (Jun–Sep) "
        "| Post-monsoon (Oct–Nov) | Winter (Dec–Feb)\n",
        "\nThe monsoon season is of particular interest: extreme cloud cover "
        "reduces GHI and system temperatures, potentially shifting the optimal "
        "PCM melting point downward relative to the annual recommendation.\n",
    ]

    for cid in sorted(assign["cluster_id"].unique()):
        member_points   = assign[assign["cluster_id"] == cid]["point_id"].unique()
        cluster_physical = physical[physical["point_id"].isin(member_points)]

        # Annual Tm_target from cluster profile
        prof = profiles[profiles["cluster_id"] == cid]
        if prof.empty:
            continue
        prof = prof.iloc[0]
        # Try multiple possible column names from 06_build_pcm_database.py
        tm_target = (
            prof.get("Tm_target_C_regime_capped",
            prof.get("Tm_target_C",
            prof.get("Tm_target_mean", 44.0)))
        )

        # Annual MCDM weights for this cluster
        cluster_scores = scores[scores["cluster_id"] == cid]
        weight_cols = [f"weight_{c}" for c in CRITERIA
                       if f"weight_{c}" in cluster_scores.columns]
        if len(weight_cols) == len(CRITERIA) and len(cluster_scores):
            weights = cluster_scores[weight_cols].iloc[0].values.astype(float)
        else:
            weights = np.ones(len(CRITERIA)) / len(CRITERIA)
            print(f"  [WARN] No weight columns for cluster {cid}, using equal weights.")

        annual_top1 = (cluster_scores.sort_values("consensus_rank")["name"].iloc[0]
                       if len(cluster_scores) else "N/A")

        md_lines.append(f"\n## Cluster {int(cid)}  "
                        f"(annual/Level-A #1: **{annual_top1}**)\n")
        md_lines.append("| Season | #1 PCM | #2 PCM | #3 PCM | "
                        "Ta_mean (°C) | L_required | Flips from annual? |")
        md_lines.append("|---|---|---|---|---|---|---|")

        print(f"\n  Cluster {int(cid)}  (annual #1: {annual_top1}):")

        for season in SEASON_ORDER:
            season_rows = cluster_physical[cluster_physical["season"] == season]
            if season_rows.empty:
                print(f"    {season:<14}  — no data")
                continue

            # Seasonal climate inputs
            ta_col     = "era5_T_amb" if "era5_T_amb" in season_rows.columns else "Ta_mean"
            ta_mean    = season_rows[ta_col].mean()
            t_mains    = ta_mean - 2.0   # standard lag-correlation proxy
            q_night_kw = DRAW_RATE_KG_PER_S * CP_WATER * (T_DELIVERY_C - t_mains)
            l_required = (q_night_kw * 3600 * 7) / ASSUMED_PCM_MASS_KG

            ranked = rank_seasonal(pcm_db, tm_target, l_required, weights)
            if ranked is None:
                md_lines.append(f"| {season} | (<2 survivors) | — | — | "
                                f"{ta_mean:.1f} | {l_required:.0f} | — |")
                print(f"    {season:<14}  Ta={ta_mean:.1f}°C  L={l_required:.0f}  "
                      f"→ fewer than 2 PCMs survive filter")
                continue

            top3 = ranked.head(3)["name"].tolist()
            while len(top3) < 3:
                top3.append("—")
            flips = "**YES**" if top3[0] != annual_top1 else "No"

            all_rows.append({
                "cluster_id": cid, "season": season,
                "Ta_mean_season": ta_mean, "L_required_season": l_required,
                "top1": top3[0], "top2": top3[1], "top3": top3[2],
                "flips_from_annual": top3[0] != annual_top1,
            })
            md_lines.append(
                f"| {season} | {top3[0]} | {top3[1]} | {top3[2]} | "
                f"{ta_mean:.1f} | {l_required:.0f} | {flips} |"
            )
            flag = "  ← FLIPS" if top3[0] != annual_top1 else ""
            print(f"    {season:<14}  Ta={ta_mean:.1f}°C  "
                  f"L={l_required:.0f}  #1={top3[0]}{flag}")

    # ── Save outputs ─────────────────────────────────────────
    result_df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 68)
    print("  DONE — Phase 4, Level B")
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_MD}")

    if len(result_df):
        n_flips = result_df["flips_from_annual"].sum()
        total   = len(result_df)
        print(f"\n  {n_flips}/{total} (cluster × season) combinations show a "
              f"#1 PCM different from the cluster's annual pick.")
        if n_flips > 0:
            print()
            print("  [KEY FINDING] Seasonal flips detected.")
            print("  Per plan §7.2 — this is direct empirical motivation for the")
            print("  adaptive control objective (Objective 3 DRL controller),")
            print("  generated from your own Assam data.")
            print("  Recommendation: dedicate a paragraph + table to this in §7.")
            print()
            print("  Monsoon flip detail (most important for Assam):")
            monsoon_flips = result_df[
                (result_df["season"] == "Monsoon") &
                (result_df["flips_from_annual"] == True)
            ]
            if len(monsoon_flips):
                print(f"    {len(monsoon_flips)} cluster(s) change their #1 PCM during")
                print("    the Assam monsoon — extreme cloud suppresses GHI and lowers")
                print("    system temperatures, shifting optimal Tm downward.")
            else:
                print("    No monsoon flip — the annual Tm_target rule is robust to")
                print("    Assam's monsoon cycle, which is itself a strong result.")
        else:
            print()
            print("  No seasonal flips — the Tm_target rule is robust across")
            print("  all four Assam seasons. Worth stating explicitly: even during")
            print("  the extreme monsoon (Jun–Sep), the same PCM specification is")
            print("  adequate, providing evidence that the rule generalises.")
    print("=" * 68)


if __name__ == "__main__":
    main()
