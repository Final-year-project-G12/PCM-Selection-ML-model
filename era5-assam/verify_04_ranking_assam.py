"""
Verification Script 04: MCDM Multi-Criteria Ranking & Stability (Assam)
========================================================================
Validate MCDM ranking results & Monte Carlo simulation:
- Method rank correlation (TOPSIS, GRA, PROMETHEE, VIKOR)
- Top-3 inclusion probability distribution
- Monte Carlo rank variance
- Consensus Borda rank alignment

Output folder: data/plots/verify_ranking/
"""

import os, warnings, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
TOPK = os.path.join(BASE, "data", "processed", "pcm", "mcdm_topk_assam.csv")
FULL = os.path.join(BASE, "data", "processed", "pcm", "mcdm_full_scores_assam.csv")
MC_CSV = os.path.join(BASE, "data", "processed", "pcm", "monte_carlo_stability_assam.csv")
OUT  = os.path.join(BASE, "data", "plots", "verify_ranking")
os.makedirs(OUT, exist_ok=True)

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

def ensure_ranks(df):
    for sc, rc, asc in [
        ("topsis_score", "topsis_rank", False),
        ("gra_grade", "gra_rank", False),
        ("promethee_flow", "promethee_rank", False),
        ("vikor_Q", "vikor_rank", True),
        ("borda_score", "consensus_rank", False)
    ]:
        if rc not in df.columns and sc in df.columns and "cluster_id" in df.columns:
            df[rc] = df.groupby("cluster_id")[sc].rank(ascending=asc, method="min").astype(int)
    return df

def sfig(name):
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {name}")

print("=== [Verify 04] MCDM Ranking Verification (Assam) ===")

topk = pd.read_csv(TOPK) if os.path.exists(TOPK) else None
mc = pd.read_csv(MC_CSV) if os.path.exists(MC_CSV) else None

if topk is not None:
    topk = ensure_ranks(topk)

# 1. Method Spearman Correlation
print("[1/5] Spearman Correlation between Methods")
if topk is not None:
    ranks = [c for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"] if c in topk.columns]
    if len(ranks) >= 2:
        corr = topk[ranks].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                    xticklabels=[r.replace("_rank", "").upper() for r in ranks],
                    yticklabels=[r.replace("_rank", "").upper() for r in ranks])
        ax.set_title("Verify Ranking 01: Inter-Method Spearman Correlation\n[Historical K=4 Pre-Audit Reference; Phase 10 Baseline]", fontsize=10)
        plt.tight_layout(); sfig("01_method_correlation.png")

# 2. Top-3 Inclusion Probability
print("[2/5] Top-3 Inclusion Probability Distribution")
mc_df = mc if (mc is not None and "top3_inclusion_probability" in mc.columns) else (topk if (topk is not None and "top3_inclusion_probability" in topk.columns) else None)
if mc_df is not None and "top3_inclusion_probability" in mc_df.columns:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    scale = 100 if mc_df["top3_inclusion_probability"].max() <= 1.0 else 1
    sns.histplot(mc_df["top3_inclusion_probability"] * scale, bins=20, kde=True, color="#911eb4", ax=ax)
    ax.set(title="Verify Ranking 02: Historical Top-3 Inclusion Probability (%)\n[Pre-Audit MC Reference; Final K=3 Monte Carlo SKIPPED]",
           xlabel="Historical Top-3 Probability (%)", ylabel="Count")
    ax.grid(alpha=0.3); sfig("02_top3_inclusion_probability.png")

# 3. Rank Variance across Clusters
print("[3/5] Rank Variance across Clusters")
if topk is not None and "consensus_rank" in topk.columns:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.boxplot(data=topk, x="cluster_id", y="consensus_rank", palette="Set2", ax=ax)
    ax.set(title="Verify Ranking 03: Historical Consensus Rank Spread per Cluster\n[Pre-Audit K=4 Reference; Evaluated against Final Physics in Phase 10]",
           xlabel="Historical Cluster ID", ylabel="Historical Consensus Rank")
    ax.grid(alpha=0.3, axis="y"); sfig("03_rank_distributions.png")

# 4. Method Agreement Plot
print("[4/5] Method Agreement Heatmap for Top Candidates")
if topk is not None:
    top1 = topk[topk.get("consensus_rank", 1) == 1]
    if not top1.empty:
        ranks = [c for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"] if c in top1.columns]
        if ranks:
            name_col = "name" if "name" in top1.columns else "PCM_Name"
            labels = top1[name_col].astype(str) if name_col in top1.columns else top1.index.astype(str)
            mat = top1[ranks].values
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            sns.heatmap(mat, annot=True, fmt="d", cmap="Blues_r", ax=ax, yticklabels=labels, xticklabels=[r.replace("_rank", "").upper() for r in ranks])
            ax.set_title("Verify Ranking 05: Method Agreement on Top-1 Candidates\n[Historical Pre-Audit K=4 Reference; Refuted by Final Physics in Phase 10]", fontsize=9.5)
            plt.tight_layout(); sfig("05_method_agreement.png")

# 5. Ranking Summary Text Card
print("[5/5] Ranking Summary Text Card")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.axis("off")
summary_text = (
    "HISTORICAL K=4 MCDM & MONTE CARLO QA (PRE-AUDIT ARTIFACT)\n"
    "---------------------------------------------------------\n"
    f"Historical Ranked Rows   : {len(topk) if topk is not None else 'N/A'} (Top-3 x 4 clusters)\n"
    "Historical MCDM Methods  : TOPSIS, GRA, PROMETHEE, VIKOR, Consensus Borda\n"
    "Historical Monte Carlo   : 1,000 draws (Pre-audit historical run)\n"
    "Final K=3 Governance     : MCDM NOT PERFORMED, MC SKIPPED (n_confirmed=[0,0,0])\n"
    "Preservation Purpose     : Retrospective baseline for Phase 10 physics audit\n"
)
ax.text(0.05, 0.5, summary_text, fontsize=10.5, family="monospace", va="center")
sfig("06_ranking_summary.png")

print(f"Verify 04 complete! Outputs saved in: {OUT}")
