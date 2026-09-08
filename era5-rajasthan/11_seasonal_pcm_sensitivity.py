"""
11_seasonal_pcm_sensitivity.py
=============================================================================
SEASONAL PCM SENSITIVITY — RAJASTHAN  (post-Phase-6 analysis)

NEW 2026-09-08. Ported from Tamil Nadu's equivalent (which was renamed from
`11_level_b_seasonal_analysis.py` to `11_seasonal_pcm_sensitivity.py` in the
same pass) so both states run the same seasonal check. Rajasthan previously
had no equivalent at all.

WHY THE NAME CHANGED ON THE TAMIL NADU SIDE: two different analyses were
both being called "Level B", which made it impossible to tell from a
filename which result a number came from.

  * "Level B — Regime Shift" is a Phase-4 CLUSTERING step — rebuild a Tier-1
    signature per point per season, fit a fresh GMM, and ask whether a
    point's climate regime changes across the year. That lives in
    `05a_level_b_regime_shift_rajasthan.py`.
  * THIS script is not a clustering step. It runs AFTER Phase 6: it holds
    the Level-A clusters AND the annual blended MCDM weight vector fixed,
    recomputes only the climate-dependent input (Ta_mean -> L_required) per
    season, re-ranks each cluster's candidate pool with TOPSIS, and reports
    how often the #1 PCM flips.

METHOD NOTE — what is held fixed and what moves. The weight vector is
computed ONCE per cluster from that cluster's annual candidate pool, using
Phase 6's own functions (literature Table-13 priors, corrosion reweighted by
the cluster's HSI_sunrise, blended lambda=0.5 with Shannon-entropy weights).
It is then held CONSTANT across all four seasons, so any rank change is
attributable to the seasonal L_required alone and not to the weights moving
underneath the comparison. Only the latent-heat floor (and hence which
candidates survive) and the latent-heat criterion column change per season.
Tm_target_C is the constant 57 C design rule — the same value Phase 6 ranks
against — so the melting-window criterion does not move either.

INPUTS:
  data/processed/climate_rajasthan_points.csv                   (season +
      era5_T_amb per point, for the per-season Ta_mean)
  data/processed/cluster_assignments_rajasthan_levelA.csv       (Phase 4)
  data/processed/cluster_profiles_rajasthan.csv                 (Phase 4)
  data/processed/feasibility_survivors_by_cluster_kappa_calibrated.csv (Ph.5)
  data/processed/mcdm_full_rankings.csv                    (Phase 6 —
      annual #1 per cluster, by borda_score, same convention Phase 8 uses)

OUTPUTS:
  data/processed/seasonal_pcm_sensitivity_rajasthan.csv
  outputs/seasonal_pcm_sensitivity_rajasthan.md
  outputs/qc_seasonal_pcm_flip_heatmap_rajasthan.html   (the (cluster x
      season) grid of #1-ranked PCM identity, flipped cells outlined
      against the annual baseline)

HOW TO RUN:
  python 11_seasonal_pcm_sensitivity.py
"""

import warnings
warnings.filterwarnings("ignore")

import importlib.util
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    BASE_DIR, COMBINED_POINTS_FILE, PROCESSED_DIR, OUTPUTS_DIR,
    SHARE_PCM, ASSUMED_PCM_MASS_KG, T_DELIVERY_C, ensure_data_dirs,
)
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match

ensure_data_dirs()

STATE_NAME = "rajasthan"

MCDM_SCRIPT_FILE = BASE_DIR / "08_mcdm_ranking.py"
ASSIGN_FILE = PROCESSED_DIR / f"cluster_assignments_{STATE_NAME}_levelA.csv"
PROFILE_FILE = PROCESSED_DIR / f"cluster_profiles_{STATE_NAME}.csv"
SURVIVORS_FILE = PROCESSED_DIR / "feasibility_survivors_by_cluster_kappa_calibrated.csv"
MCDM_RANKINGS_FILE = PROCESSED_DIR / "mcdm_full_rankings.csv"   # renamed 2026-09-08 (was mcdm_rankings_rajasthan.csv)
OUT_CSV = PROCESSED_DIR / f"seasonal_pcm_sensitivity_{STATE_NAME}.csv"
OUT_MD = OUTPUTS_DIR / f"seasonal_pcm_sensitivity_{STATE_NAME}.md"
HEATMAP_FILE = OUTPUTS_DIR / f"qc_seasonal_pcm_flip_heatmap_{STATE_NAME}.html"

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]

# Night-discharge sizing basis — identical to Phase 3's 04b (Avargani et al.
# 2021: 300 L total over the discharge window). T_DELIVERY_C, SHARE_PCM and
# ASSUMED_PCM_MASS_KG come from the shared pcm_shared_config.py via config.py.
DRAW_VOLUME_L = 300.0
DRAW_MASS_KG = DRAW_VOLUME_L * 1.0
CP_WATER = 4.186

# Table 12 latent-heat floor: L >= max(100 kJ/kg, kappa * L_required), the
# same rule Phase 5 applies. Tamil Nadu keeps this in config.py as
# latent_heat_floor_kj_kg(); Rajasthan's Phase 5 calibrates kappa per cluster,
# but for a like-for-like SEASONAL comparison the nominal kappa is held fixed
# so that only L_required moves between seasons.
LATENT_HEAT_FRACTION = 0.7
LATENT_HEAT_ABSOLUTE_MIN_KJ_KG = 100.0


def latent_heat_floor(l_required, fraction=LATENT_HEAT_FRACTION,
                       absolute_min=LATENT_HEAT_ABSOLUTE_MIN_KJ_KG):
    return max(absolute_min, fraction * l_required)


def load_mcdm_module():
    """08_mcdm_ranking.py's filename starts with a digit, so it cannot be
    `import`ed normally; loaded via importlib instead — the same pattern
    09_recommendation_cards.py already uses. Its module-level code
    (path constants, ensure_data_dirs(), the Table-13 sum-to-1 assert) runs
    harmlessly on load; main() is NEVER called (guarded by __main__), only
    its already-defined weight/matrix functions are reused."""
    spec = importlib.util.spec_from_file_location("phase6_mcdm_ranking_module", MCDM_SCRIPT_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase6_mcdm_ranking_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def topsis_scores(matrix, weights, criteria_type):
    """TOPSIS on Phase 6's own weighted-normalized decision matrix
    convention (R = X/||X||_2, V = R*w) — reused, not reinvented. Cost
    criteria have their ideal/anti-ideal swapped."""
    X = matrix.fillna(matrix.median())
    norm = np.sqrt((X ** 2).sum())
    R = X.div(norm.replace(0, np.nan), axis=1).fillna(0.0)
    V = R.mul(pd.Series(weights), axis=1)

    v_best, v_worst = {}, {}
    for c in V.columns:
        if criteria_type.get(c, "benefit") == "benefit":
            v_best[c], v_worst[c] = V[c].max(), V[c].min()
        else:
            v_best[c], v_worst[c] = V[c].min(), V[c].max()
    s_plus = np.sqrt(((V - pd.Series(v_best)) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - pd.Series(v_worst)) ** 2).sum(axis=1))
    return s_minus / (s_plus + s_minus + 1e-12)


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

    for f in (COMBINED_POINTS_FILE, ASSIGN_FILE, PROFILE_FILE,
              SURVIVORS_FILE, MCDM_RANKINGS_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found — run the earlier phases first.")
            return

    assign = pd.read_csv(ASSIGN_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    survivors = pd.read_csv(SURVIVORS_FILE)
    mcdm = pd.read_csv(MCDM_RANKINGS_FILE)

    # PROVENANCE HARD-FAIL — the annual weights, cluster ids and candidate
    # pools below must all come from the clustering run currently on disk.
    current_profile_fp_id = fingerprint_id(file_fingerprint(PROFILE_FILE))
    assert_fingerprint_match(current_profile_fp_id, survivors,
                              PROFILE_FILE.name, SURVIVORS_FILE.name)
    assert_fingerprint_match(current_profile_fp_id, mcdm,
                              PROFILE_FILE.name, MCDM_RANKINGS_FILE.name)
    print(f"  Provenance check PASSED (fingerprint {current_profile_fp_id}).")

    print("\n  Loading per-season ambient temperature ...")
    pts = pd.read_csv(COMBINED_POINTS_FILE,
                       usecols=["point_id", "season", "era5_T_amb"])
    ta_by_point_season = pts.groupby(["point_id", "season"])["era5_T_amb"].mean()

    mcdm_mod = load_mcdm_module()
    rich = pd.concat([mcdm_mod.load_rich_pcm_properties(),
                      mcdm_mod.literature_rich_properties()],
                     ignore_index=True, sort=False)
    survivors_rich = survivors.merge(rich, on=["pcm_id", "family"], how="left",
                                      suffixes=("", "_rich"))
    prior_w = mcdm_mod.LITERATURE_WEIGHTS_TABLE13
    hsi_min, hsi_max = profiles["HSI_sunrise"].min(), profiles["HSI_sunrise"].max()

    all_rows = []
    annual_top1_by_cluster = {}
    md_lines = [f"# Seasonal PCM Sensitivity — {STATE_NAME.title()}\n",
                "Post-Phase-6 analysis: Level-A clusters and each cluster's "
                "annual blended (entropy + Table-13 prior, lambda=0.5, "
                "HSI-reweighted corrosion) MCDM weight vector are held FIXED; "
                "only the climate-dependent `L_required` is recomputed per "
                "season, then the candidate pool is re-ranked with TOPSIS. "
                "This is NOT a re-clustering — for that see "
                "`05a_level_b_regime_shift_rajasthan.py`.\n"]

    for cid in sorted(assign["cluster_id"].unique()):
        cid = int(cid)
        prof_rows = profiles[profiles["cluster_id"] == cid]
        if prof_rows.empty:
            continue
        prof = prof_rows.iloc[0]
        tm_target = prof["Tm_target_C"]   # constant 57 C rule — same as Phase 6

        mcdm_sub = mcdm[mcdm["cluster_id"] == cid]
        if not len(mcdm_sub):
            continue
        # Annual #1 by borda_score — the same convention Phase 8's cards use.
        annual_top1 = mcdm_sub.sort_values("borda_score", ascending=False)["pcm_id"].iloc[0]
        annual_top1_by_cluster[cid] = annual_top1

        cand_df = survivors_rich[(survivors_rich["cluster_id"] == cid) &
                                  (survivors_rich["survives_all"])].reset_index(drop=True)
        if len(cand_df) < 2:
            print(f"\n  Cluster {cid}: only {len(cand_df)} survivor(s) — skipping.")
            continue

        # Weight vector: computed ONCE from the annual pool and held constant
        # across seasons, so a rank flip is attributable to L_required alone.
        cluster_prior_w = mcdm_mod.reweight_corrosion_for_cluster(
            prior_w, prof["HSI_sunrise"], hsi_min, hsi_max)
        annual_matrix = mcdm_mod.build_criteria_matrix(
            cand_df, tm_target, float(prof["L_required_kJ_per_kg"]))
        ent_w = mcdm_mod.entropy_weights(annual_matrix)
        blend_w = mcdm_mod.blended_weights(ent_w, cluster_prior_w)

        member_points = assign[assign["cluster_id"] == cid]["point_id"].unique()

        md_lines.append(f"\n## Cluster {cid}  (annual/Phase-6 #1: **{annual_top1}**)\n")
        md_lines.append("| Season | Ta_mean | L_required | #1 PCM | #2 PCM | #3 PCM | Flips? |")
        md_lines.append("|---|---|---|---|---|---|---|")

        for season in SEASON_ORDER:
            keys = [(p, season) for p in member_points
                    if (p, season) in ta_by_point_season.index]
            if not keys:
                continue
            ta_mean_season = float(np.mean([ta_by_point_season.loc[k] for k in keys]))
            t_mains_season = ta_mean_season - 2.0
            q_total_kj = DRAW_MASS_KG * CP_WATER * (T_DELIVERY_C - t_mains_season)
            l_required_season = (q_total_kj * SHARE_PCM) / ASSUMED_PCM_MASS_KG

            floor = latent_heat_floor(l_required_season)
            seasonal = cand_df[cand_df["latent_heat_kJ_kg"] >= floor].reset_index(drop=True)
            if len(seasonal) < 2:
                md_lines.append(f"| {season} | {ta_mean_season:.1f} C | "
                                f"{l_required_season:.0f} | (< 2 survivors) | - | - | - |")
                continue

            # Climate-relative latent-heat criterion now uses the SEASONAL
            # L_required, so the re-ranking reflects the seasonal demand
            # shift itself, not just the seasonal feasibility floor.
            matrix = mcdm_mod.build_criteria_matrix(seasonal, tm_target, l_required_season)
            scores = topsis_scores(matrix, blend_w, mcdm_mod.CRITERIA_TYPE)
            seasonal = seasonal.assign(topsis_score=scores.values)
            ranked = seasonal.sort_values("topsis_score", ascending=False)

            top3 = ranked.head(3)["pcm_id"].tolist()
            while len(top3) < 3:
                top3.append("-")
            flipped = top3[0] != annual_top1
            flips = "**YES**" if flipped else "No"

            all_rows.append({"cluster_id": cid, "season": season,
                              "Ta_mean_season": ta_mean_season,
                              "L_required_season": l_required_season,
                              "latent_heat_floor_used": floor,
                              "n_seasonal_survivors": len(seasonal),
                              "annual_top1": annual_top1,
                              "top1": top3[0], "top2": top3[1], "top3": top3[2],
                              "flips_from_annual": flipped})
            md_lines.append(f"| {season} | {ta_mean_season:.1f} C | "
                            f"{l_required_season:.0f} | {top3[0]} | {top3[1]} | "
                            f"{top3[2]} | {flips} |")

        print(f"\n  Cluster {cid} (annual #1: {annual_top1}):")
        for r in [r for r in all_rows if r["cluster_id"] == cid]:
            flag = "  <-- FLIPS" if r["flips_from_annual"] else ""
            print(f"    {r['season']:8s}  Ta_mean={r['Ta_mean_season']:.1f}C  "
                  f"L_required={r['L_required_season']:.0f}  "
                  f"n={r['n_seasonal_survivors']:2d}  #1={r['top1']}{flag}")

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
                  "(Objective 3's DRL controller), generated from your own data.")
        else:
            print("  No flips detected — also a valid finding: the delivery-temperature-"
                  "anchored Tm_target rule is robust to this state's seasonal swings.")
    print("=" * 68)


if __name__ == "__main__":
    main()
