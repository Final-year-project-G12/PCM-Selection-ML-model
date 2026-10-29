"""
09_recommendation_cards.py
=============================================================================
PHASE 10 - EXPLANATION & FINAL OUTPUT (Current Assam pipeline)

Turns the current Assam pipeline outputs (Phases 6-9) into one
recommendation card per climate cluster.

CURRENT ASSAM PIPELINE:
- Phase 6: exactly 5 final deployable survivors, selected after relaxation round 2.
- Phase 7: TOPSIS + GRA + PROMETHEE II + VIKOR with Borda consensus.
- Phase 8: 5,000-draw TOPSIS ranking-stability analysis.
- Phase 9: physics validation of all 5 survivors across 3 final K=3 cluster medoids.
- Uses current product_name identifiers and current final output files.
- Includes analytical criterion contributions without using SHAP.

INPUT  : data/processed/clustering/cluster_profiles_assam.csv
         data/processed/clustering/cluster_assignments_assam.csv
         data/processed/pcm/mcdm_rankings_by_cluster.csv
         data/processed/pcm/pcm_feasibility_by_cluster.csv
         data/processed/pcm/physics_validation_assam.csv
         data/processed/pcm/monte_carlo_stability_assam.csv
OUTPUT : data/processed/pcm/recommendation_cards_assam.md
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

CLUSTER_DIR = PROCESSED_DIR / "clustering"
PCM_DIR = PROCESSED_DIR / "pcm"

PROFILE_FILE = CLUSTER_DIR / "cluster_profiles_assam.csv"
ASSIGN_FILE = CLUSTER_DIR / "cluster_assignments_assam.csv"
RANKINGS_FILE = PCM_DIR / "mcdm_rankings_by_cluster.csv"
FEASIBILITY_FILE = PROCESSED_DIR / "feasibility" / "pcm_feasibility_by_cluster.csv"
PHYSICS_RESULTS_FILE = PCM_DIR / "physics_validation_assam.csv"
MC_FILE = PCM_DIR / "monte_carlo_stability_assam.csv"
SCORES_FILE = PCM_DIR / "mcdm_full_scores_assam.csv"
OUT_FILE = PCM_DIR / "recommendation_cards_assam.md"

SIGNATURE_DISPLAY = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac",
                      "CCI", "HDD18", "CDD24", "RH_mean", "HSI", "monsoon_index"]

CRITERIA = ["f_Tm", "latent_heat_margin_ratio", "rho_H_MJ_m3", "TC_W_mK", "cycles_confidence"]
CRIT_NAMES = ["Tm_Fitness", "Latent_Heat", "Vol_Heat", "Conductivity", "Cycling_Stability"]

def calculate_contributions(cluster_scores):
    """Calculates percentage contribution of each criterion for each PCM."""
    # Min-max normalisation for benefit criteria
    X = cluster_scores[CRITERIA].values
    
    # Map the criterion name to the weight column name
    weight_cols = []
    for c in CRITERIA:
        weight_cols.append(f"weight_{c}")
            
    W = cluster_scores[weight_cols].iloc[0].values
    
    X_min = np.nanmin(X, axis=0)
    X_max = np.nanmax(X, axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1.0  # Avoid division by zero
    
    N = (X - X_min) / denom
    N = np.nan_to_num(N, nan=0.0) # Fill NaNs with 0 so they don't propagate
    
    # Add a tiny epsilon so 0-score criteria still show up as 0% instead of NaN
    weighted_scores = N * W + 1e-9
    row_sums = weighted_scores.sum(axis=1, keepdims=True)
    contributions = (weighted_scores / row_sums) * 100
    
    return contributions

def main():
    print("=" * 68)
    print("  Phase 10 -- Recommendation Cards -- Assam")
    print("=" * 68)

    required = (
        PROFILE_FILE, ASSIGN_FILE, RANKINGS_FILE,
        FEASIBILITY_FILE, SCORES_FILE,
        PHYSICS_RESULTS_FILE, MC_FILE
    )
    for f in required:
        if not f.exists():
            print(f"\n  ERROR: {f} not found -- run the current pipeline first.")
            return

    profiles = pd.read_csv(PROFILE_FILE)
    assign = pd.read_csv(ASSIGN_FILE)
    if "cluster_id" not in assign.columns and "cluster" in assign.columns:
        assign["cluster_id"] = assign["cluster"]

    grid_file = PROCESSED_DIR / "population_grid_points.csv"
    if grid_file.exists() and "point_id" in assign.columns:
        grid_df = pd.read_csv(grid_file)
        if "lat" in grid_df.columns and "lat" not in assign.columns:
            assign = assign.merge(
                grid_df[["point_id", "lat", "lon"]], on="point_id", how="left"
            )

    rankings = pd.read_csv(RANKINGS_FILE)
    feasibility = pd.read_csv(FEASIBILITY_FILE)
    full_scores = pd.read_csv(SCORES_FILE)
    physics = pd.read_csv(PHYSICS_RESULTS_FILE)
    mc = pd.read_csv(MC_FILE)

    # Phase 9 / Phase 8 files may retain the historical identifier name.
    # Normalize it here so the recommendation-card script uses one identifier
    # consistently without changing any underlying results.
    if "product_name" not in physics.columns and "name" in physics.columns:
        physics = physics.rename(columns={"name": "product_name"})
    if "product_name" not in mc.columns and "name" in mc.columns:
        mc = mc.rename(columns={"name": "product_name"})

    if "product_name" not in physics.columns:
        raise ValueError(
            "physics_validation_assam.csv must contain either 'product_name' or 'name'."
        )
    if "product_name" not in mc.columns:
        raise ValueError(
            "monte_carlo_stability_assam.csv must contain either 'product_name' or 'name'."
        )

    # Phase 6 governance: final deployable survivors are rows with survives_all=True.
    final_feas = feasibility[feasibility["survives_all"] == True].copy()
    final_names = sorted(final_feas["product_name"].dropna().unique().tolist())
    if len(final_names) != 5:
        raise ValueError(
            f"Expected exactly 5 unique final Phase 6 survivors, found "
            f"{len(final_names)}: {final_names}"
        )

    if len(physics) != 15:
        raise ValueError(
            f"Expected 15 Phase 9 physics-validation rows (5 x 3), found {len(physics)}."
        )

    print("  Final Phase 6 relaxation round: 2")
    print(f"  Final deployable survivors: {len(final_names)}")
    print(f"  Phase 9 physics rows: {len(physics)}")

    lines = [
        "# Objective 1 — Recommendation Cards (Assam)\n",
        f"Generated from {len(profiles)} final climate regimes using the current "
        f"K=3 GMM clustering and {assign['point_id'].nunique()} population points.\n",
        "**Pipeline basis:** Phase 6 final deployable survivors (relaxation round 2) → "
        "Phase 7 four-method MCDM consensus → Phase 8 TOPSIS Monte Carlo stability → "
        "Phase 9 physics validation.\n",
        "**Final Phase 6 survivor set:** " + ", ".join(final_names) + ".\n"
    ]

    for _, prof in profiles.sort_values("cluster_id").iterrows():
        cid = int(prof["cluster_id"])
        members = assign[assign["cluster_id"] == cid]

        cluster_rank = rankings[rankings["cluster_id"] == cid].copy()
        if "product_name" not in cluster_rank.columns:
            if "name" in cluster_rank.columns:
                cluster_rank = cluster_rank.rename(columns={"name": "product_name"})
            else:
                raise ValueError("mcdm_rankings_by_cluster.csv must contain product_name.")

        cluster_rank = cluster_rank[cluster_rank["product_name"].isin(final_names)].copy()
        if "consensus_rank" in cluster_rank.columns:
            cluster_rank = cluster_rank.sort_values("consensus_rank")

        cluster_full = full_scores[full_scores["cluster_id"] == cid].copy()
        if "product_name" not in cluster_full.columns and "name" in cluster_full.columns:
            cluster_full = cluster_full.rename(columns={"name": "product_name"})
        cluster_full = cluster_full[cluster_full["product_name"].isin(final_names)].copy()

        if len(cluster_full) and all(c in cluster_full.columns for c in CRITERIA):
            try:
                contribs = calculate_contributions(cluster_full)
                cluster_full[[c + "_contrib" for c in CRITERIA]] = contribs
            except (ValueError, KeyError):
                pass

        lines.append(f"\n## Cluster {cid}\n")
        lines.append(f"- **Points in regime:** {int(prof['n_points'])}")
        pop = prof.get("total_population", prof.get("total_population_covered", np.nan))
        if pd.notna(pop):
            lines.append(f"- **Population covered:** {float(pop):,.0f}")

        if "lat" in members.columns and "lon" in members.columns and len(members):
            if "max_membership_prob" in members.columns:
                medoid = members.loc[members["max_membership_prob"].idxmax()]
            else:
                medoid = members.iloc[0]
            lines.append(
                f"- **Representative point:** {medoid['point_id']} "
                f"({medoid['lat']:.3f}, {medoid['lon']:.3f})"
            )

        lines.append("\n**Climate signature:**\n")
        lines.append("| Index | Value |")
        lines.append("|---|---|")
        for col in SIGNATURE_DISPLAY:
            val = prof.get(col, prof.get(f"{col}_mean", prof.get(f"{col}_est_mean", np.nan)))
            if pd.notna(val):
                lines.append(f"| {col} | {float(val):.3f} |")

        l_req = prof.get("L_required_kJ_per_kg", np.nan)
        tm_target = prof.get("Tm_target_C", prof.get("Tm_target_mean", 56.0))
        tm_cap = prof.get("Tm_target_capped_C", np.nan)
        lines.append(
            f"\n**Derived targets:** Tm_target = {float(tm_target):.1f} °C, "
            f"L_required = {float(l_req):.1f} kJ/kg"
        )
        if pd.notna(tm_cap):
            lines.append(f"; charging-feasibility ceiling = {float(tm_cap):.2f} °C")

        lines.append(
            "\n**Phase 6 feasibility:** all 58 PCM records were evaluated; "
            "the final relaxation round was round 2 with a 47–68 °C melting window. "
            "Exactly 5 unique deployable survivors were retained."
        )

        if len(cluster_rank):
            lines.append(
                "\n**Final PCM ranking (Borda consensus of TOPSIS + GRA + "
                "PROMETHEE II + VIKOR):**\n"
            )
            lines.append(
                "| Rank | PCM | Family | Tm (°C) | Latent heat (kJ/kg) | "
                "TOPSIS | GRA | PROMETHEE | VIKOR_Q | MC Top-3 % |"
            )
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

            for _, row in cluster_rank.head(5).iterrows():
                name = row["product_name"]
                mc_row = mc[(mc["cluster_id"] == cid) & (mc["product_name"] == name)]
                mc_pct = np.nan
                if len(mc_row):
                    if "top3_inclusion_probability" in mc_row.columns:
                        mc_pct = float(mc_row["top3_inclusion_probability"].iloc[0])
                    elif "top3_pct" in mc_row.columns:
                        mc_pct = float(mc_row["top3_pct"].iloc[0])

                def val(col):
                    return row[col] if col in row.index and pd.notna(row[col]) else np.nan

                lines.append(
                    f"| {int(row['consensus_rank'])} | {name} | "
                    f"{row.get('family', '')} | {float(val('Tm_C')):.1f} | "
                    f"{float(val('latent_heat_kJ_kg')):.0f} | "
                    f"{float(val('topsis_score')):.3f} | "
                    f"{float(val('gra_grade')):.3f} | "
                    f"{float(val('promethee_flow')):+.3f} | "
                    f"{float(val('vikor_Q')):.3f} | "
                    f"{mc_pct * 100:.1f}% |" if pd.notna(mc_pct) else
                    f"| {int(row['consensus_rank'])} | {name} | "
                    f"{row.get('family', '')} | {float(val('Tm_C')):.1f} | "
                    f"{float(val('latent_heat_kJ_kg')):.0f} | "
                    f"{float(val('topsis_score')):.3f} | "
                    f"{float(val('gra_grade')):.3f} | "
                    f"{float(val('promethee_flow')):+.3f} | "
                    f"{float(val('vikor_Q')):.3f} | n/a |"
                )

            if "kendall_w" in cluster_rank.columns:
                kw = float(cluster_rank["kendall_w"].iloc[0])
                agreement_note = (
                    "strong agreement" if kw >= 0.8 else
                    "moderate agreement" if kw >= 0.6 else "weak agreement"
                )
                lines.append(
                    f"\n*Kendall's W (4-method concordance) = {kw:.3f} "
                    f"({agreement_note}).*"
                )

            if len(cluster_full) and all(f"{c}_contrib" in cluster_full.columns for c in CRITERIA):
                lines.append(
                    "\n**Analytical criterion contributions "
                    "(within the five deployable survivors):**\n"
                )
                for _, row in cluster_rank.head(5).iterrows():
                    m = cluster_full[cluster_full["product_name"] == row["product_name"]]
                    if len(m):
                        pdata = m.iloc[0]
                        parts = []
                        for crit, cname in zip(CRITERIA, CRIT_NAMES):
                            v = float(pdata[f"{crit}_contrib"])
                            if v > 0.5:
                                parts.append(f"{cname}: {v:.0f}%")
                        lines.append(
                            f"- **{row['product_name']}**: "
                            f"{', '.join(parts) if parts else 'no single criterion >0.5%'}"
                        )
        else:
            lines.append("\n**No current MCDM ranking rows found for this cluster.**")

        mc_cluster = mc[mc["cluster_id"] == cid]
        if len(mc_cluster):
            lines.append("\n**Phase 8 — TOPSIS Monte Carlo ranking stability:**")
            if "mean_spearman_rho" in mc_cluster.columns:
                lines.append(
                    f"- 5,000 draws; mean Spearman rho = "
                    f"{float(mc_cluster['mean_spearman_rho'].iloc[0]):.4f}."
                )
            lines.append(
                "- Top-3 inclusion probabilities are reported for the five "
                "deployable survivors; this is TOPSIS stability, not a new feasibility test."
            )

        cluster_physics = physics[physics["cluster_id"] == cid].copy()
        if len(cluster_physics):
            lines.append(
                "\n**Phase 9 — physics validation "
                "(10-year real climate forcing, final K=3 medoid):**\n"
            )
            lines.append(
                "| PCM | Annual solar fraction | Complete cycles/yr | "
                "dt sensitivity | Cumulative energy error |"
            )
            lines.append("|---|---:|---:|---|---:|")
            for _, r in cluster_physics.iterrows():
                sf = float(r.get("annual_solar_fraction", np.nan))
                cyc = float(r.get("complete_cycles_per_year", np.nan))
                dt = r.get("dt_sensitivity_pass", r.get("dt_pass", ""))
                err = float(r.get("cumulative_energy_error_pct", np.nan))
                lines.append(
                    f"| {r['product_name']} | {sf * 100:.1f}% | "
                    f"{cyc:.1f} | {dt} | {err:.4f}% |"
                )

        lines.append(
            "\n**Interpretation:** Phase 6 determines deployability, Phase 7 "
            "ranks the five deployable survivors, Phase 8 tests TOPSIS ranking "
            "stability, and Phase 9 independently validates the same five PCMs "
            "with the physics model.\n"
        )

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  {profiles['cluster_id'].nunique()} cluster cards written.")
    print("=" * 68)


if __name__ == "__main__":
    main()
