"""
11_seasonal_pcm_sensitivity.py
=============================================================================
SEASONAL PCM SENSITIVITY — TAMIL NADU  (post-Phase-6 analysis)

RENAMED 2026-09-08 from `11_level_b_seasonal_analysis.py`. The old name was
part of a genuine name collision: TWO different analyses were both being
called "Level B".

  * "Level B — Regime Shift" is a Phase-4 CLUSTERING step — rebuild a Tier-1
    signature per point per season, fit a fresh GMM, and ask whether a
    point's climate regime changes across the year. That now lives in
    `05a_level_b_regime_shift_tamilnadu.py`, immediately after Level A.
  * THIS script is not a clustering step at all. It runs AFTER Phase 6: it
    takes the existing Level-A clusters and the annual MCDM weights that
    08_mcdm_ranking.py already computed, recomputes only the
    climate-dependent MCDM input (Ta_mean -> L_required) per season, re-ranks
    the PCM shortlist with TOPSIS, and reports how often the #1 PCM flips.

Keeping both under one "level_b" name made it impossible to tell from a
filename which result a number came from. The two are now named for what
they measure, and Rajasthan has an identical counterpart
(`11_seasonal_pcm_sensitivity_rajasthan.py`).

WHAT THIS DOES NOT DO: it does NOT re-run full GMM clustering per season
(that is 05a's job). It reuses the SAME entropy+AHP-blended weights computed
for the annual case, so the seasonal comparison is like-for-like — only
L_required moves. Say which version you did in your methodology.

Tamil Nadu's north-east monsoon (Oct-Dec) is out of phase with the
south-west monsoon most of India runs on, which is exactly the situation
where a seasonal flip is physically plausible rather than an artefact.

INPUT  : data/preprocessed/tamilnadu_cleaned_physical.csv   (has a 'season'
           column: Winter/Summer/Monsoon/Retreat)
         data/processed/clustering/cluster_assignments_tamilnadu.csv
         data/processed/clustering/cluster_profiles_tamilnadu.csv
         data/processed/pcm/pcm_database_tamilnadu.csv
         data/processed/pcm/mcdm_full_scores_by_cluster.csv  (annual/Level-A
           weights, so the seasonal comparison uses the same weight vector
           as the headline result)
OUTPUT : data/processed/pcm/seasonal_pcm_sensitivity_topk.csv
         data/processed/pcm/seasonal_pcm_sensitivity_summary.md
         outputs/qc_seasonal_pcm_flip_heatmap_tamilnadu.html   (NEW — the
           (cluster x season) grid of #1-ranked PCM identity, with flipped
           cells outlined against the annual baseline)

HOW TO RUN:
  python 11_seasonal_pcm_sensitivity.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    PREPROCESSED_DIR, PROCESSED_DIR, OUTPUTS_DIR, SHARE_PCM,
    ASSUMED_PCM_MASS_KG, T_DELIVERY_C, latent_heat_floor_kj_kg,
)
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match

STATE_NAME = "tamilnadu"

PHYSICAL_FILE = PREPROCESSED_DIR / f"{STATE_NAME}_cleaned_physical.csv"
ASSIGN_FILE = PROCESSED_DIR / "clustering" / f"cluster_assignments_{STATE_NAME}.csv"
PROFILE_FILE = PROCESSED_DIR / "clustering" / f"cluster_profiles_{STATE_NAME}.csv"
PCM_FILE = PROCESSED_DIR / "pcm" / f"pcm_database_{STATE_NAME}.csv"
SCORES_FILE = PROCESSED_DIR / "pcm" / "mcdm_full_rankings.csv"   # renamed 2026-09-08 Phase 6 unification (was mcdm_full_scores_by_cluster.csv)
OUT_CSV = PROCESSED_DIR / "pcm" / "seasonal_pcm_sensitivity_topk.csv"
OUT_MD = PROCESSED_DIR / "pcm" / "seasonal_pcm_sensitivity_summary.md"
HEATMAP_FILE = OUTPUTS_DIR / f"qc_seasonal_pcm_flip_heatmap_{STATE_NAME}.html"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7
SIGMA_TM = 4.0
# T_DELIVERY_C, SHARE_PCM and ASSUMED_PCM_MASS_KG come from the shared
# pcm_shared_config.py (via config.py) so this script sizes L_required
# identically to Phase 3's 04b_climate_signature.py. Values unchanged.
DRAW_VOLUME_L = 300.0                 # Avargani et al. 2021 domestic baseline
DRAW_MASS_KG = DRAW_VOLUME_L * 1.0
CP_WATER = 4.186
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
    l_floor = latent_heat_floor_kj_kg(l_required, LATENT_HEAT_FRACTION)
    survivors = df[df["Tm_C"].between(lo, hi) & df["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX) &
                    (df["latent_heat_kJ_kg"] >= l_floor)].copy()
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


def build_flip_heatmap(result_df, annual_top1_by_cluster, path):
    """(cluster x season) grid of the #1-ranked PCM per cell.

    Colour encodes PCM IDENTITY (a discrete palette, one colour per distinct
    winning PCM), so a row that is all one colour is a cluster whose choice
    is season-insensitive and a row that changes colour is one that flips.
    Cells whose winner differs from that cluster's ANNUAL #1 are outlined in
    red and prefixed with a marker, so the flip count is legible directly
    off the figure rather than only from the summary line.
    """
    if result_df.empty:
        print("  [SKIP] no (cluster, season) results to plot.")
        return

    clusters = sorted(result_df["cluster_id"].unique())
    seasons = [s for s in SEASON_ORDER if s in set(result_df["season"])]

    pivot = (result_df.pivot_table(index="cluster_id", columns="season",
                                    values="top1", aggfunc="first")
             .reindex(index=clusters, columns=seasons))

    # Discrete colour code per distinct winning PCM.
    names = sorted({v for v in pivot.values.ravel() if isinstance(v, str)})
    code = {n: i for i, n in enumerate(names)}
    z = [[code.get(pivot.loc[c, s], np.nan) for s in seasons] for c in clusters]

    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
               "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]
    if len(names) == 1:
        colorscale = [[0.0, palette[0]], [1.0, palette[0]]]
    else:
        colorscale = []
        for i, n in enumerate(names):
            lo_f, hi_f = i / len(names), (i + 1) / len(names)
            colorscale += [[lo_f, palette[i % len(palette)]],
                           [hi_f, palette[i % len(palette)]]]

    text, shapes = [], []
    n_flips = 0
    for ri, c in enumerate(clusters):
        row_text = []
        annual = annual_top1_by_cluster.get(c, None)
        for si, s in enumerate(seasons):
            val = pivot.loc[c, s]
            if not isinstance(val, str):
                row_text.append("")
                continue
            flipped = (annual is not None and val != annual)
            if flipped:
                n_flips += 1
                shapes.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=si - 0.5, x1=si + 0.5, y0=ri - 0.5, y1=ri + 0.5,
                    line=dict(color="#d62728", width=4), fillcolor="rgba(0,0,0,0)",
                ))
            row_text.append(("▲ " if flipped else "") + val)
        text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=seasons, y=[f"Cluster {c}" for c in clusters],
        colorscale=colorscale, showscale=False,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="%{y} · %{x}<br>#1 PCM: %{text}<extra></extra>",
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        title=(f"Seasonal PCM Sensitivity — {STATE_NAME.title()}<br>"
               f"<sub>#1-ranked PCM per (cluster, season). Colour = PCM identity; "
               f"red outline + ▲ = differs from that cluster's ANNUAL #1 "
               f"({n_flips}/{len(result_df)} cells flip).</sub>"),
        shapes=shapes, height=120 + 90 * len(clusters),
        xaxis=dict(side="top"), yaxis=dict(autorange="reversed"),
    )
    fig.write_html(str(path))
    print(f"  Saved: {path}")


def main():
    print("=" * 68)
    print(f"  Seasonal PCM Sensitivity (post-Phase-6) — {STATE_NAME.title()}")
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

    # Provenance: the annual weights and cluster ids used below must come
    # from the same clustering run as the profiles on disk.
    current_profile_fp_id = fingerprint_id(file_fingerprint(PROFILE_FILE))
    assert_fingerprint_match(current_profile_fp_id, scores,
                              PROFILE_FILE.name, SCORES_FILE.name)
    print(f"  Provenance check PASSED (fingerprint {current_profile_fp_id}).")

    all_rows = []
    annual_top1_by_cluster = {}
    md_lines = [f"# Seasonal PCM Sensitivity — {STATE_NAME.title()}\n",
                "Post-Phase-6 analysis: Level-A clusters and the annual "
                "entropy+AHP MCDM weights are held fixed; only the "
                "climate-dependent `L_required` is recomputed per season, then "
                "the shortlist is re-ranked with TOPSIS. This is NOT a "
                "re-clustering — for that see "
                "`05a_level_b_regime_shift_tamilnadu.py`.\n"]

    for cid in sorted(assign["cluster_id"].unique()):
        member_points = assign[assign["cluster_id"] == cid]["point_id"].unique()
        cluster_physical = physical[physical["point_id"].isin(member_points)]

        prof = profiles[profiles["cluster_id"] == cid].iloc[0]
        # 07b_charging_feasibility.py (which wrote Tm_target_C_regime_capped)
        # was retired 2026-09-08; the charging-feasibility ceiling is now
        # Phase 3's Tm_target_capped_C. Fall back to the legacy column, then
        # to the uncapped Tm_target_C, for older cluster_profiles files.
        if "Tm_target_capped_C" in prof.index and prof["Tm_target_capped_C"] == prof["Tm_target_capped_C"]:
            tm_target = prof["Tm_target_capped_C"]
        elif "Tm_target_C_regime_capped" in prof.index:
            tm_target = prof["Tm_target_C_regime_capped"]
        else:
            tm_target = prof["Tm_target_C"]

        cluster_scores = scores[scores["cluster_id"] == cid]
        if not len(cluster_scores):
            continue
        weight_cols = [f"weight_{c}" for c in CRITERIA if f"weight_{c}" in cluster_scores.columns]
        weights = cluster_scores[weight_cols].iloc[0].values if len(weight_cols) == len(CRITERIA) \
            else np.ones(len(CRITERIA)) / len(CRITERIA)

        annual_top1 = cluster_scores.sort_values("consensus_rank")["name"].iloc[0]
        annual_top1_by_cluster[int(cid)] = annual_top1
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
            l_required_season = (q_total_kj * SHARE_PCM) / ASSUMED_PCM_MASS_KG

            ranked = rank_seasonal(pcm_db, tm_target, l_required_season, weights)
            if ranked is None:
                md_lines.append(f"| {season} | (< 2 survivors) | - | - | - |")
                continue

            top3 = ranked.head(3)["name"].tolist()
            while len(top3) < 3:
                top3.append("-")
            flips = "**YES**" if top3[0] != annual_top1 else "No"

            all_rows.append({"cluster_id": int(cid), "season": season,
                              "Ta_mean_season": ta_mean_season,
                              "L_required_season": l_required_season,
                              "annual_top1": annual_top1,
                              "top1": top3[0], "top2": top3[1], "top3": top3[2],
                              "flips_from_annual": top3[0] != annual_top1})
            md_lines.append(f"| {season} | {top3[0]} | {top3[1]} | {top3[2]} | {flips} |")

        print(f"\n  Cluster {int(cid)} (annual #1: {annual_top1}):")
        for r in [r for r in all_rows if r["cluster_id"] == int(cid)]:
            flag = "  <-- FLIPS" if r["flips_from_annual"] else ""
            print(f"    {r['season']:8s}  Ta_mean={r['Ta_mean_season']:.1f}C  "
                  f"L_required={r['L_required_season']:.0f}  #1={r['top1']}{flag}")

    result_df = pd.DataFrame(all_rows)
    if len(result_df):
        result_df["upstream_cluster_profile_fingerprint"] = current_profile_fp_id
    result_df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("\nSeasonal PCM flip heatmap ...")
    build_flip_heatmap(result_df, annual_top1_by_cluster, HEATMAP_FILE)

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_MD}")
    if len(result_df):
        n_flips = int(result_df["flips_from_annual"].sum())
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
                  "is robust to this state's seasonal swings, which is itself worth "
                  "stating as evidence the rule generalizes.")
    print("=" * 68)


if __name__ == "__main__":
    main()
