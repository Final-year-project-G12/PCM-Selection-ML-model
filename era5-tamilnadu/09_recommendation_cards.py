"""
09_recommendation_cards.py   (v2 — includes Phase 7 physics validation)
=============================================================================
PHASE 8 — EXPLANATION & FINAL OUTPUT (Objective 1 plan v3.0, Section 11)

Turns everything Phases 4-7 produced into one recommendation card per
cluster — this becomes your results section directly (Table 18 in the
plan doc). Pure aggregation script, computes nothing new.

CHANGE FROM v1: now includes 10_physics_validation.py's simulated annual
solar fraction and per-cluster Spearman rho (MCDM rank vs. simulated
performance) — Phase 7 is no longer future work, so the cards should
report it, not omit it. If you haven't run 10_physics_validation.py yet,
this script still works and just omits that section with a note.

INPUT  : data/processed/clustering/cluster_profiles_tamilnadu.csv
         data/processed/clustering/cluster_assignments_tamilnadu.csv
         data/processed/pcm/mcdm_topk_by_cluster.csv
         data/processed/pcm/feasibility_survivors_by_cluster.csv
         data/processed/pcm/physics_validation_results.csv    (optional)
         data/processed/pcm/physics_validation_spearman.csv   (optional)
OUTPUT : data/processed/pcm/recommendation_cards.md

HOW TO RUN:
  python 09_recommendation_cards.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import PROCESSED_DIR

CLUSTER_DIR = PROCESSED_DIR / "clustering"
PCM_DIR = PROCESSED_DIR / "pcm"

PROFILE_FILE = CLUSTER_DIR / "cluster_profiles_tamilnadu.csv"
ASSIGN_FILE = CLUSTER_DIR / "cluster_assignments_tamilnadu.csv"
TOPK_FILE = PCM_DIR / "mcdm_topk_by_cluster.csv"
SURVIVORS_FILE = PCM_DIR / "feasibility_survivors_by_cluster.csv"
PHYSICS_RESULTS_FILE = PCM_DIR / "physics_validation_results.csv"
PHYSICS_SPEARMAN_FILE = PCM_DIR / "physics_validation_spearman.csv"
OUT_FILE = PCM_DIR / "recommendation_cards.md"

SIGNATURE_DISPLAY = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac",
                      "CCI", "HDD18", "CDD24", "RH_mean", "HSI", "monsoon_index"]


def main():
    print("=" * 68)
    print("  Phase 8 — Recommendation Cards — Tamil Nadu")
    print("=" * 68)

    for f in (PROFILE_FILE, ASSIGN_FILE, TOPK_FILE, SURVIVORS_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found — run the earlier phase scripts first.")
            return

    profiles = pd.read_csv(PROFILE_FILE)
    assign = pd.read_csv(ASSIGN_FILE)
    topk = pd.read_csv(TOPK_FILE)
    survivors = pd.read_csv(SURVIVORS_FILE)

    physics_available = PHYSICS_RESULTS_FILE.exists() and PHYSICS_SPEARMAN_FILE.exists()
    if physics_available:
        physics_results = pd.read_csv(PHYSICS_RESULTS_FILE)
        physics_spearman = pd.read_csv(PHYSICS_SPEARMAN_FILE)
        print("  Phase 7 physics validation results found — including in cards.")
    else:
        print("  [NOTE] Phase 7 outputs not found — run 10_physics_validation.py "
              "for a complete Objective 1 (physics validation is no longer "
              "treated as optional future work in this pipeline). Cards will "
              "still generate without it.")

    lines = ["# Objective 1 — Recommendation Cards (Tamil Nadu)\n",
             f"Generated from {len(profiles)} climate regimes "
             f"(GMM clustering, {assign['point_id'].nunique()} population points).\n"]

    if physics_available:
        overall_rho = physics_spearman["spearman_rho"].mean()
        lines.append(f"**Physics validation summary (Phase 7):** mean Spearman rho "
                      f"across clusters = {overall_rho:.3f} (MCDM consensus rank vs. "
                      f"simulated annual solar fraction, grey-box lumped-enthalpy "
                      f"tank model driven by each cluster's medoid point's real "
                      f"10-year daily climate data). See "
                      f"`10_physics_validation.py`'s docstring for the full stated "
                      f"assumption list (tank size, collector efficiency, draw "
                      f"schedule) before quoting this number without qualification.\n")

    for _, prof in profiles.sort_values("cluster_id").iterrows():
        cid = int(prof["cluster_id"])
        members = assign[assign["cluster_id"] == cid]
        n_survivors = int((survivors[survivors["cluster_id"] == cid]["passes_all"]).sum())
        cluster_top = topk[topk["cluster_id"] == cid].sort_values("consensus_rank")

        lines.append(f"\n## Cluster {cid}\n")
        lines.append(f"- **Points in regime:** {int(prof['n_points'])}")
        if "total_population_covered" in prof and prof["total_population_covered"] == prof["total_population_covered"]:
            lines.append(f"- **Population covered:** {prof['total_population_covered']:,.0f}")
        if "lat" in members.columns and "lon" in members.columns and len(members):
            medoid = members.loc[members["max_membership_prob"].idxmax()] \
                if "max_membership_prob" in members.columns else members.iloc[0]
            lines.append(f"- **Medoid point (highest membership confidence):** "
                          f"{medoid['point_id']} ({medoid['lat']:.3f}, {medoid['lon']:.3f})")

        lines.append("\n**Climate signature (population-weighted mean):**\n")
        lines.append("| Index | Value |")
        lines.append("|---|---|")
        for col in SIGNATURE_DISPLAY:
            if col in prof and prof[col] == prof[col]:
                lines.append(f"| {col} | {prof[col]:.3f} |")

        lines.append(f"\n**Derived targets:** Tm_target = {prof.get('Tm_target_C', float('nan')):.1f} C, "
                      f"L_required = {prof.get('L_required_kJ_per_kg', float('nan')):.0f} kJ/kg")
        lines.append(f"\n**Candidates screened:** {n_survivors} survived Phase 5 feasibility filtering "
                      f"(melting window, absolute band, latent-heat floor, cycling, supercooling, "
                      f"corrosion veto, safety exclusion)")

        if len(cluster_top):
            lines.append("\n**Top-3 PCM candidates (Borda consensus of TOPSIS + GRA + "
                          "PROMETHEE II + VIKOR):**\n")
            lines.append("| Rank | PCM | Family | Tm (C) | Latent heat (kJ/kg) | "
                          "TOPSIS | GRA | PROMETHEE | VIKOR_Q | MC Top-3 % |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for _, row in cluster_top.iterrows():
                mc_pct = row.get("top3_inclusion_probability", float("nan"))
                mc_str = f"{mc_pct*100:.1f}%" if mc_pct == mc_pct else "n/a"
                lines.append(f"| {int(row['consensus_rank'])} | {row['name']} | "
                              f"{row.get('family', '')} | {row['Tm_C']:.1f} | "
                              f"{row['latent_heat_kJ_kg']:.0f} | {row['topsis_score']:.3f} | "
                              f"{row['gra_grade']:.3f} | {row.get('promethee_flow', float('nan')):+.3f} | "
                              f"{row.get('vikor_Q', float('nan')):.3f} | {mc_str} |")
            kw = cluster_top["kendall_w"].iloc[0]
            agreement_note = ("strong agreement" if kw >= 0.8 else
                               "moderate agreement — discuss the disagreement" if kw >= 0.6 else
                               "weak agreement — this regime's PCM choice is genuinely ambiguous")
            lines.append(f"\n*Kendall's W (4-method concordance) = {kw:.3f} ({agreement_note})*")
            if "borda_copeland_agree" in cluster_top.columns and not cluster_top["borda_copeland_agree"].iloc[0]:
                lines.append("\n*Borda and Copeland consensus disagree on #1 for this cluster "
                              "— report both, per plan v3.0 Section 9.5.*")
        else:
            lines.append("\n**No ranked candidates** — this cluster had <2 feasibility "
                          "survivors. Widen the PCM database or relax the melting window "
                          "for this Tm_target before finalising.")

        if physics_available:
            cluster_physics = physics_results[physics_results["cluster_id"] == cid].sort_values("consensus_rank")
            cluster_rho_row = physics_spearman[physics_spearman["cluster_id"] == cid]
            if len(cluster_physics):
                lines.append("\n**Phase 7 — simulated annual performance "
                              "(grey-box lumped-enthalpy tank, real climate data):**\n")
                lines.append("| PCM | Consensus rank | Simulated solar fraction | "
                              "In 54-84% benchmark band? | Complete cycles/yr |")
                lines.append("|---|---|---|---|---|")
                for _, r in cluster_physics.head(5).iterrows():
                    lines.append(f"| {r['name']} | {int(r['consensus_rank'])} | "
                                  f"{r['annual_solar_fraction']*100:.1f}% | "
                                  f"{'Yes' if r['in_benchmark_band_54_84pct'] else 'No'} | "
                                  f"{int(r['complete_cycles_per_year'])} |")
            if len(cluster_rho_row):
                rho = cluster_rho_row["spearman_rho"].iloc[0]
                interp = cluster_rho_row["interpretation"].iloc[0]
                lines.append(f"\n*Spearman rho (MCDM rank vs. simulated solar fraction) "
                              f"for this cluster: {rho:.3f} — {interp}*")

        lines.append("\n**Caveats:** thermal conductivity / density / specific heat "
                      "not reported in the source data for the literature-added candidates "
                      "(see 06_build_pcm_database.py); Phase 7's tank/collector parameters "
                      "are stated assumptions, not measurements (see "
                      "10_physics_validation.py's docstring).\n")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  {profiles['cluster_id'].nunique()} cluster cards written.")
    print("=" * 68)
    print("\nThis is your Objective 1 results section. Paste recommendation_cards.md")
    print("content into your IEEE draft and reformat the tables to IEEE style.")


if __name__ == "__main__":
    main()
