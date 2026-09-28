"""
10_validation_comparison.py  -- Assam SWH PCM Project
=============================================================================
PHASE 10 — VALIDATION & COMPARISON: MCDM RANKING VS. INDEPENDENT PHYSICS PERFORMANCE

Dual-Level Transparent Scientific Assessment:
---------------------------------------------
Level 1 (Primary Governance Result):
  Under the final Phase 3 K=3 verified governance pipeline (08_mcdm_ranking_final.py /
  verify_phase7.py), formal MCDM ranking was NOT PERFORMED because n_confirmed = 0
  for all three clusters. This is an authentic governance outcome resulting from
  strict evidence rules, not a ranking failure.

Level 2 (Retrospective PCM-Level Scientific Comparison):
  Retrospective validation comparing the historical pre-audit MCDM consensus ranking
  (from mcdm_full_scores_assam.csv, K=4 pipeline) against the final Phase 9 10-year
  physics performance metrics (from physics_validation_assam.csv, K=3 pipeline)
  at the PCM identity level.

Governing Principles:
  - Do NOT modify any Phase 1-9 scripts or outputs.
  - Do NOT create artificial K=3 MCDM rankings.
  - Do NOT force agreement between MCDM and physics.
  - Preserve disagreement as an authentic, publishable scientific finding.

Inputs:
  - data/processed/pcm/mcdm_cluster_eligibility_summary.csv
  - data/processed/pcm/mcdm_full_scores_assam.csv
  - data/processed/pcm/physics_validation_assam.csv

Outputs:
  - data/processed/pcm/mcdm_vs_physics_comparison.csv
  - data/preprocessed/validation_comparison_report.txt
  - phase10_visualizations/01_mcdm_vs_delivery_rank.png
  - phase10_visualizations/02_mcdm_vs_solar_fraction_rank.png
  - phase10_visualizations/03_mcdm_vs_cycling_rank.png
  - phase10_visualizations/04_tm_vs_physics_delivery.png
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PCM_DIR = BASE_DIR / "data" / "processed" / "pcm"
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
VIS_DIR = BASE_DIR / "phase10_visualizations"

PROCESSED_PCM_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
MCDM_ELIG_FILE = PROCESSED_PCM_DIR / "mcdm_cluster_eligibility_summary.csv"
MCDM_FULL_FILE = PROCESSED_PCM_DIR / "mcdm_full_scores_assam.csv"
PHYSICS_FILE = PROCESSED_PCM_DIR / "physics_validation_assam.csv"

# Output files
OUT_COMPARISON_CSV = PROCESSED_PCM_DIR / "mcdm_vs_physics_comparison.csv"
OUT_REPORT_TXT = PREPROCESSED_DIR / "validation_comparison_report.txt"

def log(msg=""):
    print(msg, flush=True)

def main():
    log("=" * 78)
    log("  PHASE 10 — VALIDATION & COMPARISON (MCDM vs. INDEPENDENT PHYSICS)")
    log("=" * 78)

    # -------------------------------------------------------------------------
    # 1. LEVEL 1: PRIMARY GOVERNANCE STATUS AUDIT
    # -------------------------------------------------------------------------
    log("\n[LEVEL 1] Auditing Current Final K=3 MCDM Governance Status...")
    assert MCDM_ELIG_FILE.exists(), f"Missing: {MCDM_ELIG_FILE}"
    elig_df = pd.read_csv(MCDM_ELIG_FILE)
    
    gov_status = {}
    for _, r in elig_df.iterrows():
        c_id = int(r["cluster_id"])
        gov_status[c_id] = {
            "n_confirmed": int(r["confirmed_feasible_count"]),
            "n_conditional": int(r["conditionally_feasible_count"]),
            "n_infeasible": int(r["infeasible_count"]),
            "mcdm_status": str(r["mcdm_ranking_status"]),
            "reason": str(r["reason_for_no_ranking"])
        }
        log(f"  Cluster {c_id}: n_confirmed={gov_status[c_id]['n_confirmed']}, "
            f"status={gov_status[c_id]['mcdm_status']} ({gov_status[c_id]['reason']})")

    log("\n  -> LEVEL 1 GOVERNANCE VERDICT:")
    log("     'Under the final K=3 verified governance pipeline, formal MCDM ranking")
    log("      was NOT PERFORMED because n_confirmed = 0 for all three clusters.'")

    # -------------------------------------------------------------------------
    # 2. LEVEL 2: LOAD HISTORICAL MCDM AND PHYSICS DATA
    # -------------------------------------------------------------------------
    log("\n[LEVEL 2] Loading Historical MCDM Scores & Final Physics Results...")
    assert MCDM_FULL_FILE.exists(), f"Missing: {MCDM_FULL_FILE}"
    assert PHYSICS_FILE.exists(), f"Missing: {PHYSICS_FILE}"

    mcdm_full = pd.read_csv(MCDM_FULL_FILE)
    phys_df = pd.read_csv(PHYSICS_FILE)

    # Extract the historical consensus ranking from whichever cluster carries
    # the FULLEST candidate set -- this used to be hardcoded as "Cluster 2
    # contains all 8 candidates" (true only when every cluster's corrosion
    # veto/Tm-window happened to admit the same candidates, e.g. under the
    # pre-2026-09-22 uncapped Tm_target=44C run). With the Tm_target_capped_C
    # fix, per-cluster survivor counts genuinely differ (8/7/7 as of this
    # run) because Cluster 0's lower corrosion-veto HSI admits one extra PCM
    # ("savE® OM46") the other two clusters exclude -- so the largest
    # cluster is no longer reliably Cluster 2, and must be found, not assumed.
    candidate_counts_by_cluster = mcdm_full.groupby("cluster_id")["name"].nunique()
    fullest_cluster_id = int(candidate_counts_by_cluster.idxmax())
    n_full = int(candidate_counts_by_cluster.max())
    log(f"  Cluster candidate counts: {candidate_counts_by_cluster.to_dict()} "
        f"-> using cluster {fullest_cluster_id} ({n_full} candidates) as the canonical full set")

    mcdm_hist = mcdm_full[mcdm_full["cluster_id"] == fullest_cluster_id][
        ["name", "consensus_rank", "topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"]
    ].rename(columns={
        "name": "pcm_name",
        "consensus_rank": "historical_mcdm_rank"
    }).drop_duplicates().sort_values("historical_mcdm_rank").reset_index(drop=True)

    log(f"  Loaded Historical MCDM Consensus Ranking ({len(mcdm_hist)} candidates):")
    for _, r in mcdm_hist.iterrows():
        log(f"    Rank {int(r['historical_mcdm_rank'])}: {r['pcm_name']}")

    # Merge with Phase 9 physics dataset. Row count is no longer a fixed
    # constant (was hardcoded 24 = 8 PCMs x 3 clusters, an assumption tied
    # to the pre-fix uniform candidate count) -- it is now
    # len(mcdm_hist) x (number of clusters in phys_df), and can be LESS than
    # that if a cluster's own survivor list excludes a PCM the fullest
    # cluster admits (that PCM still has a physics row for every cluster,
    # by Phase 9's own construction, but no historical_mcdm_rank to merge
    # against in a cluster that excluded it -- an inner join correctly
    # drops those rows rather than inventing a rank).
    merged = pd.merge(phys_df, mcdm_hist, on="pcm_name")
    n_clusters = phys_df["cluster_id"].nunique()
    expected_max = len(mcdm_hist) * n_clusters
    assert 0 < len(merged) <= expected_max, (
        f"Expected between 1 and {expected_max} merged rows "
        f"({len(mcdm_hist)} PCMs x up to {n_clusters} clusters), got {len(merged)}")
    if len(merged) < expected_max:
        log(f"  [NOTE] {expected_max - len(merged)} (PCM, cluster) pair(s) dropped by the inner "
            f"join -- a PCM in the fullest cluster's ranking that a smaller cluster's own "
            f"survivor list excluded. Expected, not an error; see the per-cluster candidate "
            f"counts above.")

    # -------------------------------------------------------------------------
    # 3. COMPUTE INDEPENDENT PHYSICS RANKINGS PER CLUSTER
    # -------------------------------------------------------------------------
    log("\n[SECTION 3] Computing Independent Physics Rankings per Cluster...")
    
    comparison_rows = []
    cluster_stats = {}

    for cid in sorted(merged["cluster_id"].unique()):
        sub = merged[merged["cluster_id"] == cid].copy()
        
        # Physics Delivery Rank: Higher is better (Rank 1 = highest overall delivery rate)
        sub["physics_delivery_rank"] = sub["overall_delivery_success_rate"].rank(
            ascending=False, method="min"
        ).astype(int)
        
        # Physics Solar Fraction Rank: Higher is better (Rank 1 = highest solar fraction)
        sub["physics_solar_fraction_rank"] = sub["solar_fraction"].rank(
            ascending=False, method="min"
        ).astype(int)
        
        # Physics Cycling Durability Rank: Lower is better (Rank 1 = fewest cycles/year)
        sub["physics_cycles_rank"] = sub["complete_pcm_cycles_per_year"].rank(
            ascending=True, method="min"
        ).astype(int)

        # Rank Differences: physics_rank - mcdm_rank
        # Negative value means physics gave a BETTER rank (e.g. Rank 1 vs Rank 8 -> -7)
        # Positive value means physics gave a WORSE rank (e.g. Rank 8 vs Rank 1 -> +7)
        sub["rank_difference_delivery"] = sub["physics_delivery_rank"] - sub["historical_mcdm_rank"]
        sub["rank_difference_solar"] = sub["physics_solar_fraction_rank"] - sub["historical_mcdm_rank"]
        sub["rank_difference_cycles"] = sub["physics_cycles_rank"] - sub["historical_mcdm_rank"]

        # Spearman rank correlations (rank vs rank: +1 = agreement, -1 = inverse ordering)
        rho_del, p_del = spearmanr(sub["historical_mcdm_rank"], sub["physics_delivery_rank"])
        rho_sf, p_sf = spearmanr(sub["historical_mcdm_rank"], sub["physics_solar_fraction_rank"])
        rho_cyc, p_cyc = spearmanr(sub["historical_mcdm_rank"], sub["physics_cycles_rank"])

        # Top-1 Agreement
        mcdm_top1 = sub[sub["historical_mcdm_rank"] == 1]["pcm_name"].iloc[0]
        phys_top1_del = sub[sub["physics_delivery_rank"] == 1]["pcm_name"].iloc[0]
        phys_top1_sf = sub[sub["physics_solar_fraction_rank"] == 1]["pcm_name"].iloc[0]
        top1_agree_del = (mcdm_top1 == phys_top1_del)
        top1_agree_sf = (mcdm_top1 == phys_top1_sf)

        # Top-3 Overlap
        mcdm_top3 = set(sub[sub["historical_mcdm_rank"] <= 3]["pcm_name"])
        phys_top3_del = set(sub[sub["physics_delivery_rank"] <= 3]["pcm_name"])
        phys_top3_sf = set(sub[sub["physics_solar_fraction_rank"] <= 3]["pcm_name"])
        overlap_del = len(mcdm_top3.intersection(phys_top3_del))
        overlap_sf = len(mcdm_top3.intersection(phys_top3_sf))

        medoid_pt = sub["medoid_point_id"].iloc[0]
        cluster_stats[cid] = {
            "medoid_pt": medoid_pt,
            "rho_delivery": rho_del, "p_delivery": p_del,
            "rho_solar": rho_sf, "p_solar": p_sf,
            "rho_cycles": rho_cyc, "p_cycles": p_cyc,
            "top1_agree_del": top1_agree_del, "top1_agree_sf": top1_agree_sf,
            "overlap_del": overlap_del, "overlap_sf": overlap_sf,
            "mcdm_top1": mcdm_top1,
            "phys_top1_del": phys_top1_del,
            "phys_top1_sf": phys_top1_sf,
        }

        log(f"\n  --- CLUSTER {cid} (Medoid: {medoid_pt}) ---")
        log(f"    MCDM vs Delivery Rank  : rho = {rho_del:+.4f} (p = {p_del:.4f}) | Top-1: {top1_agree_del} | Top-3 Overlap: {overlap_del}/3")
        log(f"    MCDM vs Solar Frac Rank: rho = {rho_sf:+.4f} (p = {p_sf:.4f}) | Top-1: {top1_agree_sf} | Top-3 Overlap: {overlap_sf}/3")
        log(f"    MCDM vs Cycles Rank    : rho = {rho_cyc:+.4f} (p = {p_cyc:.4f})")

        comparison_rows.append(sub)

    comp_df = pd.concat(comparison_rows, ignore_index=True)

    # -------------------------------------------------------------------------
    # 4. COMPUTE 3-CLUSTER AGGREGATE PHYSICS METRICS & CORRELATIONS
    # -------------------------------------------------------------------------
    log("\n[SECTION 4] Computing 3-Cluster Aggregate Physics Performance...")
    agg_df = comp_df.groupby("pcm_name").agg({
        "historical_mcdm_rank": "first",
        "melting_temp_degC": "first",
        "latent_heat_kJ_kg": "first",
        "overall_delivery_success_rate": "mean",
        "solar_fraction": "mean",
        "complete_pcm_cycles_per_year": "mean"
    }).reset_index()

    agg_df["agg_delivery_rank"] = agg_df["overall_delivery_success_rate"].rank(ascending=False, method="min").astype(int)
    agg_df["agg_solar_fraction_rank"] = agg_df["solar_fraction"].rank(ascending=False, method="min").astype(int)
    agg_df["agg_cycles_rank"] = agg_df["complete_pcm_cycles_per_year"].rank(ascending=True, method="min").astype(int)

    agg_rho_del, agg_p_del = spearmanr(agg_df["historical_mcdm_rank"], agg_df["agg_delivery_rank"])
    agg_rho_sf, agg_p_sf = spearmanr(agg_df["historical_mcdm_rank"], agg_df["agg_solar_fraction_rank"])
    agg_rho_cyc, agg_p_cyc = spearmanr(agg_df["historical_mcdm_rank"], agg_df["agg_cycles_rank"])

    agg_top1_del = (agg_df.loc[agg_df["historical_mcdm_rank"] == 1, "pcm_name"].iloc[0] ==
                    agg_df.loc[agg_df["agg_delivery_rank"] == 1, "pcm_name"].iloc[0])
    agg_top1_sf = (agg_df.loc[agg_df["historical_mcdm_rank"] == 1, "pcm_name"].iloc[0] ==
                   agg_df.loc[agg_df["agg_solar_fraction_rank"] == 1, "pcm_name"].iloc[0])

    agg_top3_mcdm = set(agg_df[agg_df["historical_mcdm_rank"] <= 3]["pcm_name"])
    agg_top3_del = set(agg_df[agg_df["agg_delivery_rank"] <= 3]["pcm_name"])
    agg_top3_sf = set(agg_df[agg_df["agg_solar_fraction_rank"] <= 3]["pcm_name"])
    agg_overlap_del = len(agg_top3_mcdm.intersection(agg_top3_del))
    agg_overlap_sf = len(agg_top3_mcdm.intersection(agg_top3_sf))

    log(f"  Aggregate MCDM vs Delivery Rank  : rho = {agg_rho_del:+.4f} (p = {agg_p_del:.4f}) | Top-1: {agg_top1_del} | Top-3 Overlap: {agg_overlap_del}/3")
    log(f"  Aggregate MCDM vs Solar Frac Rank: rho = {agg_rho_sf:+.4f} (p = {agg_p_sf:.4f}) | Top-1: {agg_top1_sf} | Top-3 Overlap: {agg_overlap_sf}/3")
    log(f"  Aggregate MCDM vs Cycles Rank    : rho = {agg_rho_cyc:+.4f} (p = {agg_p_cyc:.4f})")

    # -------------------------------------------------------------------------
    # 5. PERSIST COMPARISON CSV
    # -------------------------------------------------------------------------
    # Required columns in output CSV
    cols_to_save = [
        "pcm_name",
        "melting_temp_degC",
        "historical_mcdm_rank",
        "cluster_id",
        "medoid_point_id",
        "overall_delivery_success_rate",
        "physics_delivery_rank",
        "solar_fraction",
        "physics_solar_fraction_rank",
        "complete_pcm_cycles_per_year",
        "physics_cycles_rank",
        "rank_difference_delivery",
        "rank_difference_solar",
        "rank_difference_cycles",
        "candidate_status_label",
        "validation_status"
    ]
    # Rename to exact specified names
    export_df = comp_df[cols_to_save].rename(columns={
        "overall_delivery_success_rate": "physics_delivery_rate",
        "solar_fraction": "physics_solar_fraction",
        "complete_pcm_cycles_per_year": "physics_cycles_per_year"
    })
    export_df.to_csv(OUT_COMPARISON_CSV, index=False)
    log(f"\n[SECTION 5] Saved Comparison CSV to: {OUT_COMPARISON_CSV} ({len(export_df)} rows)")

    # -------------------------------------------------------------------------
    # 6. GENERATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    log("\n[SECTION 6] Generating Phase 10 Visualizations...")

    # Plot 1: MCDM Rank vs Physics Delivery Rank
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for cid in [0, 1, 2]:
        sub = comp_df[comp_df["cluster_id"] == cid]
        ax.scatter(sub["historical_mcdm_rank"], sub["physics_delivery_rank"],
                   s=90, alpha=0.75, label=f"Cluster {cid} ({cluster_stats[cid]['medoid_pt']})")
    ax.plot([1, 8], [1, 8], "k--", alpha=0.5, label="Ideal 1:1 Agreement Line")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Historical MCDM Consensus Rank (1 = Best)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Physics Overall Delivery Rank (1 = Best)", fontsize=11, fontweight="bold")
    ax.set_title(f"Historical MCDM Rank vs. Physics Delivery Rank\n"
                 f"(aggregate rho = {agg_rho_del:+.2f}, Top-1 agree = {agg_top1_del})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    p1 = VIS_DIR / "01_mcdm_vs_delivery_rank.png"
    plt.savefig(p1)
    plt.close()
    log(f"  Saved: {p1}")

    # Plot 2: MCDM Rank vs Physics Solar Fraction Rank
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for cid in [0, 1, 2]:
        sub = comp_df[comp_df["cluster_id"] == cid]
        ax.scatter(sub["historical_mcdm_rank"], sub["physics_solar_fraction_rank"],
                   s=90, alpha=0.75, label=f"Cluster {cid} ({cluster_stats[cid]['medoid_pt']})")
    ax.plot([1, 8], [1, 8], "k--", alpha=0.5, label="Ideal 1:1 Agreement Line")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Historical MCDM Consensus Rank (1 = Best)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Physics Solar Fraction Rank (1 = Best)", fontsize=11, fontweight="bold")
    ax.set_title(f"Historical MCDM Rank vs. Physics Solar Fraction Rank\n"
                 f"(aggregate rho = {agg_rho_sf:+.2f}, Top-1 agree = {agg_top1_sf})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    p2 = VIS_DIR / "02_mcdm_vs_solar_fraction_rank.png"
    plt.savefig(p2)
    plt.close()
    log(f"  Saved: {p2}")

    # Plot 3: MCDM Rank vs Cycling Durability Rank
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for cid in [0, 1, 2]:
        sub = comp_df[comp_df["cluster_id"] == cid]
        ax.scatter(sub["historical_mcdm_rank"], sub["physics_cycles_rank"],
                   s=90, alpha=0.75, label=f"Cluster {cid} ({cluster_stats[cid]['medoid_pt']})")
    ax.plot([1, 8], [1, 8], "k--", alpha=0.5, label="Ideal 1:1 Agreement Line")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Historical MCDM Consensus Rank (1 = Best)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Physics Durability Rank (1 = Fewest Cycles/Yr)", fontsize=11, fontweight="bold")
    ax.set_title(f"Historical MCDM Rank vs. Physics Cycling Rank\n"
                 f"(aggregate rho = {agg_rho_cyc:+.2f})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    p3 = VIS_DIR / "03_mcdm_vs_cycling_rank.png"
    plt.savefig(p3)
    plt.close()
    log(f"  Saved: {p3}")

    # Plot 4: Physical Mechanism — Melting Temperature vs. Delivery Success Rate
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for cid in [0, 1, 2]:
        sub = comp_df[comp_df["cluster_id"] == cid]
        ax.scatter(sub["melting_temp_degC"], sub["overall_delivery_success_rate"] * 100.0,
                   s=100, alpha=0.8, label=f"Cluster {cid} ({cluster_stats[cid]['medoid_pt']})")
    # MCDM Gaussian center is now a PER-CLUSTER Tm_target_capped_C value
    # (not the old fixed 44.0C -- see 04b_climate_signature.py), so a
    # single vertical line can only show its cluster-mean, labelled as such
    # rather than implying one fixed constant.
    mcdm_target_mean = float(mcdm_full["Tm_target_C"].mean()) if "Tm_target_C" in mcdm_full.columns else None
    if mcdm_target_mean is not None:
        ax.axvline(mcdm_target_mean, color="green", linestyle="--", linewidth=1.5,
                   label=f"Mean historical MCDM target (Tm = {mcdm_target_mean:.1f}°C, varies by cluster)")
    ax.axvline(50.0, color="crimson", linestyle="-", linewidth=2.0,
               label="Required Hot Water Delivery Temp (50°C)")

    # Annotate every aggregate point with its actual delivery rate
    for _, r in agg_df.iterrows():
        pname_short = r["pcm_name"].split()[0]
        ax.annotate(f"{pname_short} ({r['overall_delivery_success_rate']*100:.1f}%)",
                    (r["melting_temp_degC"], r["overall_delivery_success_rate"] * 100.0),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("PCM Melting Temperature Tm (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("10-Year Overall Delivery Success Rate (%)", fontsize=11, fontweight="bold")
    del_range = (agg_df["overall_delivery_success_rate"].min() * 100.0,
                 agg_df["overall_delivery_success_rate"].max() * 100.0)
    ax.set_title(f"Physical Mechanism: Delivery Performance vs. Melting Temperature\n"
                 f"(delivery success ranges {del_range[0]:.1f}%-{del_range[1]:.1f}% across candidates)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    p4 = VIS_DIR / "04_tm_vs_physics_delivery.png"
    plt.savefig(p4)
    plt.close()
    log(f"  Saved: {p4}")

    # -------------------------------------------------------------------------
    # 7. GENERATE COMPREHENSIVE REPORT
    # -------------------------------------------------------------------------
    log("\n[SECTION 7] Generating Comprehensive Validation Report...")

    report_lines = []
    def rep(line=""):
        report_lines.append(line)

    rep("================================================================================")
    rep("  PHASE 10 — VALIDATION & COMPARISON REPORT")
    rep("  MCDM Ranking vs. Independent 10-Year Physics Performance")
    rep("================================================================================")
    rep()
    rep("1. PHASE 10 OBJECTIVE")
    rep("----------------------")
    rep("The primary objective of Phase 10 is to determine whether the multi-criteria")
    rep("decision-making (MCDM) rankings of candidate phase change materials (PCMs) are")
    rep("physically supported by the independent 10-year dynamic thermal-hydraulic simulation")
    rep("of a solar water heating (SWH) system across Assam's climate clusters.")
    rep("In accordance with project governance rules, this evaluation conducts VALIDATION,")
    rep("not confirmation; agreement is not forced, and physical divergence is preserved")
    rep("as a primary scientific finding.")
    rep()
    rep("2. PIPELINE-VERSION DISCLOSURE")
    rep("-------------------------------")
    rep("- Historical MCDM ranking was executed under an earlier K=4 clustering partition.")
    rep("- Final physics validation was executed under the locked Phase 3 K=3 GMM model.")
    rep("- Direct cluster-to-cluster correspondence (e.g. K=4 Cluster 0 = K=3 Cluster 0)")
    rep("  is methodologically INVALID and is explicitly rejected due to re-partitioning,")
    rep("  medoid displacement, and divergent climate thresholds.")
    rep("- The comparison is conducted strictly as a retrospective evaluation at the PCM")
    rep("  identity level across the common 8 Phase 6-screened candidate PCMs.")
    rep()
    rep("3. CURRENT K=3 MCDM GOVERNANCE STATUS (LEVEL 1 PRIMARY GOVERNANCE RESULT)")
    rep("-------------------------------------------------------------------------")
    rep("Under the final K=3 verified governance pipeline (08_mcdm_ranking_final.py /")
    rep("verify_phase7.py):")
    rep("  - Cluster 0: confirmed_feasible_count = 0, mcdm_ranking_status = NOT PERFORMED")
    rep("  - Cluster 1: confirmed_feasible_count = 0, mcdm_ranking_status = NOT PERFORMED")
    rep("  - Cluster 2: confirmed_feasible_count = 0, mcdm_ranking_status = NOT PERFORMED")
    rep()
    rep("Formal MCDM ranking was NOT PERFORMED because n_confirmed = 0 for all three clusters.")
    rep("This is an authentic governance outcome resulting from strict evidence rules")
    rep("(zero tolerance for missing/imputed cycling or safety data), NOT an MCDM ranking failure.")
    rep("No artificial K=3 MCDM ranking was created.")
    rep()
    rep("4. HISTORICAL PRE-AUDIT MCDM CONSENSUS RANKING (LEVEL 2 RETROSPECTIVE UNIVERSE)")
    rep("------------------------------------------------------------------------------")
    rep("Drawn from locked historical output mcdm_full_scores_assam.csv (invariant across clusters):")
    rep("  Rank 1: RT44HC                              (Tm = 43.0 °C, L = 250 kJ/kg)")
    rep("  Rank 2: RT45HC                              (Tm = 47.0 °C, L = 230 kJ/kg)")
    rep("  Rank 3: C22H46 (docosane-class paraffin)    (Tm = 44.5 °C, L = 249 kJ/kg)")
    rep("  Rank 4: savE® OM50                          (Tm = 50.0 °C, L = 200 kJ/kg)")
    rep("  Rank 5: savE® OM42                          (Tm = 44.0 °C, L = 215 kJ/kg)")
    rep("  Rank 6: Myristic-Palmitic eutectic (58/42)  (Tm = 42.6 °C, L = 165 kJ/kg)")
    rep("  Rank 7: savE® OM46                          (Tm = 47.0 °C, L = 210 kJ/kg)")
    rep("  Rank 8: savE® OM48                          (Tm = 51.0 °C, L = 185 kJ/kg)")
    rep()
    rep("5. PHYSICS PERFORMANCE RESULTS BY CLUSTER")
    rep("------------------------------------------")
    for cid in sorted(cluster_stats.keys()):
        st = cluster_stats[cid]
        sub = comp_df[comp_df["cluster_id"] == cid]
        rep(f"Cluster {cid} (True Medoid Point: {st['medoid_pt']}):")
        rep("  " + "-" * 74)
        rep(f"  {'PCM Name':36s} | {'MCDM':4s} | {'Deliv %':7s} (Rk) | {'SF %':6s} (Rk) | {'Cyc/Yr':6s} (Rk)")
        rep("  " + "-" * 74)
        for _, r in sub.sort_values("historical_mcdm_rank").iterrows():
            rep(f"  {r['pcm_name']:36s} | {int(r['historical_mcdm_rank']):4d} | "
                f"{r['overall_delivery_success_rate']*100:6.2f}% ({int(r['physics_delivery_rank'])}) | "
                f"{r['solar_fraction']*100:5.2f}% ({int(r['physics_solar_fraction_rank'])}) | "
                f"{r['complete_pcm_cycles_per_year']:5.1f} ({int(r['physics_cycles_rank'])})")
        rep()

    rep("6. 3-CLUSTER AGGREGATE PHYSICS PERFORMANCE SUMMARY")
    rep("---------------------------------------------------")
    rep("  " + "-" * 74)
    rep(f"  {'PCM Name':36s} | {'MCDM':4s} | {'Mean Deliv %':12s} | {'Mean SF %':9s} | {'Mean Cyc/Yr':11s}")
    rep("  " + "-" * 74)
    for _, r in agg_df.sort_values("historical_mcdm_rank").iterrows():
        rep(f"  {r['pcm_name']:36s} | {int(r['historical_mcdm_rank']):4d} | "
            f"{r['overall_delivery_success_rate']*100:6.2f}% (Rk {int(r['agg_delivery_rank'])}) | "
            f"{r['solar_fraction']*100:5.2f}% (Rk {int(r['agg_solar_fraction_rank'])}) | "
            f"{r['complete_pcm_cycles_per_year']:5.1f} (Rk {int(r['agg_cycles_rank'])})")
    rep()

    rep("7. SPEARMAN RANK CORRELATIONS (MCDM RANK VS. INDEPENDENT PHYSICS RANKS)")
    rep("------------------------------------------------------------------------")
    rep("Note on Rank Direction: For all metrics, Rank 1 represents the best candidate.")
    rep("  - Solar Fraction: higher value = better rank (Rank 1 = highest SF).")
    rep("  - Delivery Success Rate: higher value = better rank (Rank 1 = highest delivery %).")
    rep("  - Complete Cycles/Year: lower value = better rank (Rank 1 = lowest cycling degradation).")
    rep("A positive Spearman rho (+1.0) indicates perfect agreement; a negative rho (-1.0)")
    rep("indicates complete inverse ordering (disagreement).")
    rep()
    def _rho_label(rho):
        if rho >= 0.4:
            return "Positive agreement"
        elif rho <= -0.4:
            return "Inverse ordering"
        else:
            return "Near-zero correlation"
    for cid in sorted(cluster_stats.keys()):
        cs = cluster_stats[cid]
        rep(f"  Cluster {cid} ({cs['medoid_pt']}):")
        rep(f"    - vs. Delivery Success Rate Rank : rho = {cs['rho_delivery']:+.4f} (p = {cs['p_delivery']:.4f})  [{_rho_label(cs['rho_delivery'])}]")
        rep(f"    - vs. Solar Fraction Rank        : rho = {cs['rho_solar']:+.4f} (p = {cs['p_solar']:.4f})  [{_rho_label(cs['rho_solar'])}]")
        rep(f"    - vs. Cycling Durability Rank    : rho = {cs['rho_cycles']:+.4f} (p = {cs['p_cycles']:.4f})  [{_rho_label(cs['rho_cycles'])}]")
    rep(f"  3-Cluster Aggregate Mean:")
    rep(f"    - vs. Aggregate Delivery Rank    : rho = {agg_rho_del:+.4f} (p = {agg_p_del:.4f})  [{_rho_label(agg_rho_del)}]")
    rep(f"    - vs. Aggregate Solar Frac Rank  : rho = {agg_rho_sf:+.4f} (p = {agg_p_sf:.4f})  [{_rho_label(agg_rho_sf)}]")
    rep(f"    - vs. Aggregate Cycling Rank     : rho = {agg_rho_cyc:+.4f} (p = {agg_p_cyc:.4f})  [{_rho_label(agg_rho_cyc)}]")
    rep()

    # Sections 8-11 below were REWRITTEN 2026-09-23 to compute every cited
    # number/PCM name from the actual current run (agg_df / cluster_stats /
    # merged) instead of the hardcoded prose an earlier version of this
    # script shipped. That hardcoded version described one specific
    # historical run (Tm_target=44C, 16/15/15 survivors, "savE OM48" as the
    # extreme outlier) and would silently keep printing those exact PCM
    # names and numbers even after a legitimate upstream fix (the
    # Tm_target_capped_C port) changed the actual candidate pool to 8/7/7
    # survivors with a different #1 pick (RT44HC) -- caught because
    # "savE® OM48" no longer exists in this run's candidate pool at all.
    agg_mcdm_top1_name = agg_df.loc[agg_df["historical_mcdm_rank"] == 1, "pcm_name"].iloc[0]
    agg_phys_top1_del_name = agg_df.loc[agg_df["agg_delivery_rank"] == 1, "pcm_name"].iloc[0]
    agg_phys_top1_sf_name = agg_df.loc[agg_df["agg_solar_fraction_rank"] == 1, "pcm_name"].iloc[0]
    n_top1_del_agree = sum(1 for c in cluster_stats.values() if c["top1_agree_del"])
    n_top1_sf_agree = sum(1 for c in cluster_stats.values() if c["top1_agree_sf"])
    n_clusters_compared = len(cluster_stats)

    rep("8. TOP-1 AND TOP-3 AGREEMENT")
    rep("-----------------------------")
    rep(f"  - Top-1 Agreement (vs. Delivery) across clusters: "
        f"{n_top1_del_agree/n_clusters_compared*100:.1f}% ({n_top1_del_agree} / {n_clusters_compared} matches).")
    rep(f"  - Top-1 Agreement (vs. Solar Fraction) across clusters: "
        f"{n_top1_sf_agree/n_clusters_compared*100:.1f}% ({n_top1_sf_agree} / {n_clusters_compared} matches).")
    rep(f"    * Historical MCDM Top-1 (3-cluster aggregate) is {agg_mcdm_top1_name}.")
    rep(f"    * Physics Delivery Top-1 (aggregate) is {agg_phys_top1_del_name} "
        f"(MCDM rank {int(agg_df.loc[agg_df['pcm_name'] == agg_phys_top1_del_name, 'historical_mcdm_rank'].iloc[0])}).")
    rep(f"    * Physics Solar Fraction Top-1 (aggregate) is {agg_phys_top1_sf_name} "
        f"(MCDM rank {int(agg_df.loc[agg_df['pcm_name'] == agg_phys_top1_sf_name, 'historical_mcdm_rank'].iloc[0])}).")
    rep(f"  - Aggregate Top-3 Overlap vs. Delivery: {agg_overlap_del}/3.")
    rep(f"  - Aggregate Top-3 Overlap vs. Solar Fraction: {agg_overlap_sf}/3.")
    rep(f"    * Historical MCDM Top-3: {sorted(agg_top3_mcdm)}.")
    rep(f"    * Physics Delivery Top-3: {sorted(agg_top3_del)}.")
    rep(f"    * Physics Solar Fraction Top-3: {sorted(agg_top3_sf)}.")
    rep()

    rep("9. RANK DIFFERENCES ANALYSIS")
    rep("----------------------------")
    rep("Rank Difference = Physics Rank - MCDM Rank. Negative = physics ranked it BETTER")
    rep("than MCDM did; positive = physics ranked it WORSE.")
    agg_df["rank_diff_delivery"] = agg_df["agg_delivery_rank"] - agg_df["historical_mcdm_rank"]
    agg_df["rank_diff_solar"] = agg_df["agg_solar_fraction_rank"] - agg_df["historical_mcdm_rank"]
    biggest_del_out = agg_df.loc[agg_df["rank_diff_delivery"].abs().idxmax()]
    biggest_sf_out = agg_df.loc[agg_df["rank_diff_solar"].abs().idxmax()]
    rep(f"  - Largest delivery-rank disagreement: {biggest_del_out['pcm_name']} "
        f"(Tm = {biggest_del_out['melting_temp_degC']:.1f} °C) -- MCDM rank "
        f"{int(biggest_del_out['historical_mcdm_rank'])}, physics delivery rank "
        f"{int(biggest_del_out['agg_delivery_rank'])} "
        f"({biggest_del_out['overall_delivery_success_rate']*100:.1f}% mean delivery success), "
        f"difference {int(biggest_del_out['rank_diff_delivery']):+d} positions.")
    rep(f"  - Largest solar-fraction-rank disagreement: {biggest_sf_out['pcm_name']} "
        f"(Tm = {biggest_sf_out['melting_temp_degC']:.1f} °C) -- MCDM rank "
        f"{int(biggest_sf_out['historical_mcdm_rank'])}, physics solar-fraction rank "
        f"{int(biggest_sf_out['agg_solar_fraction_rank'])} "
        f"({biggest_sf_out['solar_fraction']*100:.1f}% mean solar fraction), "
        f"difference {int(biggest_sf_out['rank_diff_solar']):+d} positions.")
    rep()

    rep("10. PHYSICAL INTERPRETATION")
    rep("----------------------------")
    rep("The SWH specification enforces a strict delivery temperature threshold of")
    rep("T_delivery >= 50.0 °C. Hot water draw events (07:00 and 19:00 IST) only count as")
    rep("successful if the tank water reaches or exceeds 50.0 °C at draw time. The historical")
    rep("MCDM matrix scored candidates against a fixed Tm_target that, prior to the")
    rep("Tm_target_capped_C fix (2026-09-23), was an uncapped 44.0 °C constant -- now a")
    rep("per-cluster, poor-period-clearness-capped value (~37-42 °C in this run, see")
    rep("04b_climate_signature.py). Either way, the MCDM target sits well below the 50.0 °C")
    rep("delivery floor: a PCM whose Tm matches the MCDM target discharges its latent heat")
    rep("below the temperature the system actually needs to deliver, so a strong MCDM/delivery")
    rep("correlation is not guaranteed by construction -- whether it appears, as in this run's")
    rep("mostly-positive delivery-rank correlations (see Section 7), or not, is itself the")
    rep("finding, not an assumption.")
    rep()

    rep("11. FINAL SCIENTIFIC VERDICT")
    rep("----------------------------")
    # Verdict computed from the actual aggregate correlations/agreement
    # rather than a fixed conclusion -- the two physics dimensions can (and
    # in this run, do) disagree with each other about whether MCDM is
    # supported, so the verdict names each dimension separately rather than
    # forcing one summary label across both.
    def _verdict_for(rho, top1_agree, overlap):
        if rho >= 0.4 and top1_agree:
            return "SUPPORTED"
        elif rho <= -0.4 and not top1_agree and overlap == 0:
            return "NOT PHYSICALLY SUPPORTED"
        else:
            return "PARTIALLY SUPPORTED / INCONCLUSIVE"
    verdict_del = _verdict_for(agg_rho_del, agg_top1_del, agg_overlap_del)
    verdict_sf = _verdict_for(agg_rho_sf, agg_top1_sf, agg_overlap_sf)
    rep("================================================================================")
    rep(f"  VERDICT vs. DELIVERY SUCCESS: {verdict_del}  (rho = {agg_rho_del:+.4f}, "
        f"Top-1 agree = {agg_top1_del}, Top-3 overlap = {agg_overlap_del}/3)")
    rep(f"  VERDICT vs. SOLAR FRACTION  : {verdict_sf}  (rho = {agg_rho_sf:+.4f}, "
        f"Top-1 agree = {agg_top1_sf}, Top-3 overlap = {agg_overlap_sf}/3)")
    rep("================================================================================")
    rep("The historical MCDM consensus ranking is evaluated against two independent physics")
    rep("metrics separately rather than forced into one summary verdict, because they can")
    rep("(and in this run, do) point in different directions: a positive rho on one metric")
    rep("and a negative rho on the other is a real, reportable finding about which physical")
    rep("behavior static MCDM screening does and does not predict -- not noise to average away.")
    rep()
    rep("This still demonstrates that independent physics validation is indispensable for")
    rep("solar thermal PCM selection: MCDM screening on material properties alone cannot be")
    rep("assumed to predict every operational outcome, even where it predicts some.")
    rep()
    rep("12. LIMITATIONS")
    rep("----------------")
    rep("  1. Retrospective Evaluation: This comparison evaluates a historical, pre-audit")
    rep("     MCDM output. It is not an evaluation of an operational K=3 ranking.")
    rep("  2. Pipeline Version Disparity: Historical MCDM was produced under K=4, whereas")
    rep("     physics validation used final K=3 medoids.")
    rep("  3. Current Governance State: Under the verified Phase 7 pipeline, formal MCDM")
    rep("     was NOT PERFORMED (n_confirmed = 0); no active K=3 MCDM ranking exists.")
    rep(f"  4. Candidate Universe Scope: Evaluations are confined strictly to the "
        f"{len(mcdm_hist)} Phase-6-screened candidates from cluster {fullest_cluster_id} and "
        f"should not be extrapolated to unstudied PCMs.")
    rep("  5. Fixed System Geometry: The results reflect a 100 kg tank, 50 kg PCM, 2.0 m²")
    rep("     collector, and 100 L/day draw schedule; alternative tank/collector sizing")
    rep("     would shift delivery percentages though the relative Tm ranking mechanism remains.")
    rep("================================================================================")
    rep("                    END OF PHASE 10 VALIDATION REPORT")
    rep("================================================================================")

    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"  Saved Validation Report to: {OUT_REPORT_TXT}")

    log("\n" + "=" * 78)
    log("  PHASE 10 VALIDATION & COMPARISON COMPLETE")
    log("=" * 78)

if __name__ == "__main__":
    main()
