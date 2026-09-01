"""
09_recommendation_cards.py
=============================
PHASE 8 — EXPLANATION & FINAL OUTPUT (Objective 1 plan v3.0, Section 11)

Turns everything Phases 4-6 produced into one recommendation card per
cluster — this becomes your results section directly (Table 18 in the
plan doc). Pure aggregation script, computes nothing new.

INPUT  : data/processed/clustering/cluster_profiles_uttarakhand.csv
         data/processed/clustering/cluster_assignments_uttarakhand.csv
         data/processed/pcm/mcdm_topk_by_cluster.csv
         data/processed/pcm/feasibility_survivors_by_cluster.csv
OUTPUT : data/processed/pcm/recommendation_cards.md
           one markdown section per cluster, ready to paste into your
           IEEE paper's results section (reformat to IEEE table style
           when you paste it in — this gives you the content, not the
           final typesetting)

HOW TO RUN:
  python 09_recommendation_cards.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import PROCESSED_DIR

CLUSTER_DIR = PROCESSED_DIR / "clustering"
PCM_DIR = PROCESSED_DIR / "pcm"

PROFILE_FILE = CLUSTER_DIR / "cluster_profiles_uttarakhand.csv"
ASSIGN_FILE = CLUSTER_DIR / "cluster_assignments_uttarakhand.csv"
TOPK_FILE = PCM_DIR / "mcdm_topk_by_cluster.csv"
SURVIVORS_FILE = PCM_DIR / "feasibility_survivors_by_cluster.csv"
OUT_FILE = PCM_DIR / "recommendation_cards.md"

SIGNATURE_DISPLAY = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac",
                      "CCI", "HDD18", "CDD24", "RH_mean", "HSI", "monsoon_index"]


def main():
    print("=" * 68)
    print("  Phase 8 — Recommendation Cards — Uttarakhand")
    print("=" * 68)

    for f in (PROFILE_FILE, ASSIGN_FILE, TOPK_FILE, SURVIVORS_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found — run the earlier phase scripts first.")
            return

    profiles = pd.read_csv(PROFILE_FILE)
    assign = pd.read_csv(ASSIGN_FILE)
    topk = pd.read_csv(TOPK_FILE)
    survivors = pd.read_csv(SURVIVORS_FILE)

    lines = ["# Objective 1 — Recommendation Cards (Uttarakhand)\n",
             f"Generated from {len(profiles)} climate regimes "
             f"(GMM clustering, {assign['point_id'].nunique()} population points).\n"]

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
            # .loc[], not .iloc[] — idxmin() returns members' original ROW
            # LABEL (inherited from the un-reset `assign` dataframe this was
            # boolean-filtered from), not a 0..len(members)-1 position. Using
            # .iloc[] here throws IndexError as soon as that label happens to
            # exceed len(members), which is exactly what you hit.
            medoid = members.loc[(members[["lat", "lon"]]
                                   - members[["lat", "lon"]].mean()).pow(2).sum(axis=1).idxmin()]
            lines.append(f"- **Approx. medoid point:** {medoid['point_id']} "
                          f"({medoid['lat']:.3f}, {medoid['lon']:.3f})")

        lines.append("\n**Climate signature (population-weighted mean):**\n")
        lines.append("| Index | Value |")
        lines.append("|---|---|")
        for col in SIGNATURE_DISPLAY:
            if col in prof and prof[col] == prof[col]:
                lines.append(f"| {col} | {prof[col]:.3f} |")

        lines.append(f"\n**Derived targets:** Tm_target = {prof.get('Tm_target_C', float('nan')):.1f} C, "
                      f"L_required = {prof.get('L_required_kJ_per_kg', float('nan')):.0f} kJ/kg")
        lines.append(f"\n**Candidates screened:** {n_survivors} survived Phase 5 feasibility filtering")

        if len(cluster_top):
            lines.append("\n**Top-3 PCM candidates (consensus of TOPSIS + GRA, Borda-aggregated):**\n")
            lines.append("| Rank | PCM | Family | Tm (C) | Latent heat (kJ/kg) | TOPSIS | GRA |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, row in cluster_top.iterrows():
                lines.append(f"| {int(row['consensus_rank'])} | {row['name']} | "
                              f"{row.get('family', '')} | {row['Tm_C']:.1f} | "
                              f"{row['latent_heat_kJ_kg']:.0f} | {row['topsis_score']:.3f} | "
                              f"{row['gra_grade']:.3f} |")
            kw = cluster_top["kendall_w"].iloc[0]
            agreement_note = ("strong agreement" if kw >= 0.8 else
                               "moderate agreement — discuss the disagreement" if kw >= 0.6 else
                               "weak agreement — this regime's PCM choice is genuinely ambiguous")
            lines.append(f"\n*Kendall's W = {kw:.3f} ({agreement_note})*")
        else:
            lines.append("\n**No ranked candidates** — this cluster had <2 feasibility "
                          "survivors. Widen the PCM database or relax the melting window "
                          "for this Tm_target before finalising.")

        lines.append("\n**Caveats:** thermal conductivity / density / specific heat "
                      "not reported in the source data for the literature-added candidates "
                      "(see 06_build_pcm_database.py); cycling and corrosion vetoes only "
                      "partially applied (see 07_feasibility_filter.py's docstring for what "
                      "wasn't checked yet).\n")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  {profiles['cluster_id'].nunique()} cluster cards written.")
    print("=" * 68)
    print("\nThis is your Objective 1 results section. Paste recommendation_cards.md")
    print("content into your IEEE draft and reformat the tables to IEEE style.")


if __name__ == "__main__":
    main()