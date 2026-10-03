"""
weight_tilt_sweep.py — EXPLORATORY SENSITIVITY ANALYSIS, NOT PRODUCTION METHODOLOGY
=====================================================================================
Purpose (thesis "before vs after" comparison, per student request 2026-09-29):
show, as a transparent sensitivity/threshold exercise, how far Phase 6's MCDM
criterion weights would need to be tilted away from their current entropy-derived
values before Phase 7's Spearman rho (MCDM Borda rank vs. simulated solar
fraction) flips sign in each Rajasthan cluster.

THIS IS NOT AN ADOPTED METHODOLOGY CHANGE. It does not modify, overwrite, or
re-run any production script (08_mcdm_ranking.py, 10_physics_validation.py) or
production output (mcdm_full_rankings.csv, physics_validation_rajasthan.csv,
recommendation_cards_rajasthan.md, physics_validation_summary_rajasthan.txt).
It reads the existing physics_validation_rajasthan.csv (already-simulated,
ground-truth annual_solar_fraction per candidate; already-computed per-criterion
normalized scores crit_*; already-computed entropy-derived baseline weights
weight_*) and recomputes, OFFLINE, a simplified weighted-sum composite score at
a sequence of weight tilts. All output goes to this sensitivity_sweep/ folder
only.

METHOD
------
1. Baseline weights = the actual entropy-derived weight_* values Phase 6 used
   for each cluster (read straight from the file — not re-estimated).
2. For each cluster, identify the single criterion whose crit_* values are most
   positively correlated (Spearman) with annual_solar_fraction — i.e. the one
   criterion that, weighted up, would most directly push the composite ranking
   toward agreement with the physics simulation. This is the single "tilt knob"
   swept — same one-parameter-at-a-time style as the existing Phase 8
   supercooling-penalty k-sweep (08_phase8_supercooling_sweep.py).
3. Sweep tilt alpha in [0, 1] (step 0.05): new_weight_i = (1-alpha)*baseline_i
   for every OTHER criterion, and the removed mass is added to the best-aligned
   criterion's own weight. Weights always sum to 1. alpha=0 reproduces the
   current baseline; alpha=1 puts all weight on the single best-aligned
   criterion.
4. At each alpha: composite_score = sum(weight_i * crit_i) across the 7 real
   criteria (Tm_fitness, latent_heat, vol_latent_heat, thermal_conductivity,
   cycling, supercooling, corrosion — 'cost' excluded, always NaN in this
   database). Rank candidates by composite_score (descending). Spearman rho of
   this rank vs. annual_solar_fraction is recorded, together with the p-value
   and n.
5. NOTE — simplified proxy, not the full 4-method engine: this composite score
   is a single weighted sum, not the full TOPSIS+PROMETHEE-II+VIKOR+GRA
   Borda-consensus Phase 6 actually uses. Re-running all four methods at every
   sweep step was out of scope for an illustrative sensitivity exercise. As a
   sanity check, this script also reports the weighted-sum rho AT alpha=0
   (baseline weights) next to the production Phase 7 rho for the same cluster,
   so the reader can see how closely the simplified proxy tracks the real
   pipeline before trusting the swept trajectory.

OUTPUT
------
sensitivity_sweep/weight_tilt_trajectory.csv — full alpha -> rho trajectory,
    every cluster, every step (not just the flip point).
sensitivity_sweep/weight_tilt_summary.md — human-readable summary: baseline
    rho, swept tilt criterion, flip-point alpha (if any) per cluster.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

IN_FILE = "../data/processed/physics_validation_rajasthan.csv"
OUT_TRAJECTORY = "weight_tilt_trajectory.csv"
OUT_SUMMARY = "weight_tilt_summary.md"

# The 7 real criteria this database actually populates (cost is always NaN;
# f_Tm/latent_heat_margin_ratio/rho_H_MJ_m3/cycles_confidence are duplicate/
# alternate-form columns for Tm_fitness/latent_heat/vol_latent_heat/cycling
# respectively — not independent criteria, excluded to avoid double-counting).
CRITERIA = [
    "Tm_fitness", "latent_heat", "vol_latent_heat",
    "thermal_conductivity", "cycling", "supercooling", "corrosion",
]

# Current on-disk production Phase 7 rho (Borda vs. simulated SF), for the
# baseline sanity-check comparison. Hardcoded from
# outputs/recommendation_cards_rajasthan.md / physics_validation_summary_rajasthan.txt
# (2026-09-19 Tm_fitness-scoring-fix run) — NOT recomputed by this script.
PRODUCTION_RHO = {0: -0.200, 1: -0.168, 2: 0.569}

ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def composite_rank(df_cluster, weights):
    score = np.zeros(len(df_cluster))
    for c in CRITERIA:
        score += weights[c] * df_cluster[f"crit_{c}"].fillna(0.0).to_numpy()
    # rank 1 = best (highest composite score), matching MCDM convention
    return pd.Series(score, index=df_cluster.index).rank(ascending=False, method="average")


def main():
    df = pd.read_csv(IN_FILE)

    trajectory_rows = []
    summary_lines = [
        "# Weight-Tilt Sensitivity Sweep — Rajasthan (EXPLORATORY, not adopted)\n",
        "Generated by `weight_tilt_sweep.py`. See that script's module docstring for full",
        "method and caveats. **This does not replace or supersede the production Phase 7",
        "result reported in `outputs/recommendation_cards_rajasthan.md` and",
        "`docs/rajasthan/09_PHASE_7_AUDIT.md`.**\n",
    ]

    for cluster_id, g in df.groupby("cluster_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        baseline_weights = {c: g[f"weight_{c}"].iloc[0] for c in CRITERIA}
        # renormalize (defensive — should already sum ~1 minus cost/TC_W_mK slack)
        wsum = sum(baseline_weights.values())
        baseline_weights = {c: w / wsum for c, w in baseline_weights.items()}

        # Identify the criterion most positively aligned with physics ground truth
        alignment = {}
        for c in CRITERIA:
            crit_vals = g[f"crit_{c}"].fillna(0.0)
            if crit_vals.nunique() <= 1:
                alignment[c] = -np.inf
                continue
            r, _ = spearmanr(crit_vals, g["annual_solar_fraction"])
            alignment[c] = r if not np.isnan(r) else -np.inf
        best_criterion = max(alignment, key=alignment.get)

        # Baseline sanity check (alpha=0)
        base_rank = composite_rank(g, baseline_weights)
        base_rho, base_p = spearmanr(base_rank, g["annual_solar_fraction"])
        # composite rank is ascending-is-better; solar fraction ascending-is-better too,
        # so a GOOD composite rank (rank 1 = best PCM) should correlate NEGATIVELY with
        # rank-of-solar-fraction-ascending, but POSITIVELY with raw solar fraction values
        # when a low rank number = high composite score = (hopefully) high solar fraction.
        # We correlate rank directly against the raw solar fraction (continuous), consistent
        # with spearmanr's own internal rank-transform, and flip sign so that "higher rank
        # number (worse PCM) tracking lower solar fraction" reads as POSITIVE agreement,
        # matching the production Phase 7 rho convention (Borda RANK vs. solar fraction,
        # both ascending = better, i.e. rank 1/highest SF should co-occur).
        base_rho = -base_rho

        flip_alpha = None
        for alpha in ALPHAS:
            weights = {}
            removed_mass = 0.0
            for c in CRITERIA:
                if c == best_criterion:
                    continue
                w = (1 - alpha) * baseline_weights[c]
                removed_mass += baseline_weights[c] - w
                weights[c] = w
            weights[best_criterion] = baseline_weights[best_criterion] + removed_mass

            rank = composite_rank(g, weights)
            rho, p = spearmanr(rank, g["annual_solar_fraction"])
            rho = -rho  # same sign convention as above

            trajectory_rows.append({
                "cluster_id": cluster_id,
                "n_candidates": n,
                "tilt_criterion": best_criterion,
                "alpha": alpha,
                "weight_on_tilt_criterion": weights[best_criterion],
                "spearman_rho": rho,
                "p_value": p,
            })

            if flip_alpha is None and np.sign(rho) != np.sign(base_rho) and rho > 0:
                flip_alpha = alpha

        summary_lines.append(f"## Cluster {cluster_id} (n={n})\n")
        summary_lines.append(
            f"- Production Phase 7 rho (2026-09-19 on-disk, full 4-method Borda): "
            f"**{PRODUCTION_RHO[cluster_id]:+.3f}**"
        )
        tracks_production = (
            np.sign(base_rho) == np.sign(PRODUCTION_RHO[cluster_id])
            or PRODUCTION_RHO[cluster_id] == 0
        )
        track_note = (
            "tracks the production sign"
            if tracks_production
            else "DOES NOT track the production sign; treat this cluster's sweep with extra caution, the proxy itself already diverges from the real 4-method engine at baseline"
        )
        summary_lines.append(
            f"- This script's simplified weighted-sum proxy at baseline (alpha=0) weights: "
            f"**{base_rho:+.3f}** (p={base_p:.3f}) — {track_note}"
        )
        summary_lines.append(f"- Criterion swept (most physics-aligned by raw Spearman corr.): `{best_criterion}` (r={alignment[best_criterion]:+.3f})")
        if flip_alpha is not None:
            summary_lines.append(
                f"- **Sign flips to positive at alpha={flip_alpha:.2f}** "
                f"(weight on `{best_criterion}` rises from "
                f"{baseline_weights[best_criterion]:.1%} at baseline to "
                f"{baseline_weights[best_criterion] + flip_alpha*(1-baseline_weights[best_criterion]):.1%} "
                f"at the flip point)."
            )
        else:
            summary_lines.append(
                "- **Sign never flips positive across the full sweep (alpha=0 to 1.0)** "
                "— i.e. even putting 100% of the weight on the single most physics-aligned "
                "criterion does not produce a positive correlation for this cluster."
            )
        summary_lines.append("")

    traj_df = pd.DataFrame(trajectory_rows)
    traj_df.to_csv(OUT_TRAJECTORY, index=False)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Wrote {OUT_TRAJECTORY} ({len(traj_df)} rows) and {OUT_SUMMARY}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
