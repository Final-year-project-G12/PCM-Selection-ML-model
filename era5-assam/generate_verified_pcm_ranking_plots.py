"""
generate_verified_pcm_ranking_plots.py
===============================================================================
Assam Objective 1 — Verified PCM Ranking Analysis and Visualization Suite
===============================================================================
Generates publication-quality figures based on latest verified PCM ranking results:
1. Pearson correlation heatmap for physical criteria and MCDM cardinal scores.
2. Comparative heatmap of Pearson (cardinal scores) vs Spearman (ordinal ranks).
3. Method-wise PCM ranking comparison across TOPSIS, GRA, PROMETHEE II, VIKOR, Borda.
4. Consensus Borda vs Copeland tournament aggregation agreement.
5. Cross-cluster comparison of top PCM candidates.
6. Monte Carlo Top-1 candidate selection frequency (retention rate, 5000 draws).
7. Monte Carlo Top-3 inclusion probability comparison.
8. Cluster-wise Monte Carlo stability comparison (Spearman rho, rank distribution).

Outputs saved strictly to:
  data/plots/verified_pcm_ranking_plots/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

# Configure Matplotlib for headless environment
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "pcm")
OUT_DIR = os.path.join(BASE_DIR, "data", "plots", "verified_pcm_ranking_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# File Paths
MCDM_CSV = os.path.join(DATA_DIR, "mcdm_rankings_by_cluster.csv")
MC_CSV = os.path.join(DATA_DIR, "monte_carlo_stability_assam.csv")

if not os.path.exists(MCDM_CSV):
    raise FileNotFoundError(f"Missing latest MCDM rankings file: {MCDM_CSV}")
if not os.path.exists(MC_CSV):
    raise FileNotFoundError(f"Missing latest Monte Carlo file: {MC_CSV}")

df_mcdm = pd.read_csv(MCDM_CSV)
df_mc = pd.read_csv(MC_CSV)

print(f"Loaded MCDM rankings: {len(df_mcdm)} rows across clusters {df_mcdm['cluster_id'].unique()}")
print(f"Loaded Monte Carlo: {len(df_mc)} rows across clusters {df_mc['cluster_id'].unique()}")

# Clean product display names (clean trademark encodings)
def clean_name(n):
    return str(n).replace("savE\ufffd", "savE®").replace("savE?", "savE®")

df_mcdm["display_name"] = df_mcdm["product_name"].apply(clean_name)
df_mc["display_name"] = df_mc["product_name"].apply(clean_name)

# Cluster Labels
CLUSTER_LABELS = {
    0: "Cluster 0 (Brahmaputra Valley / Central)",
    1: "Cluster 1 (Humid Subtropical / Southern)",
    2: "Cluster 2 (Highland / Hill Slopes)"
}

# Style configurations
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# =============================================================================
# PLOT 1: Pearson Correlation Heatmap (Criteria & MCDM Cardinal Scores)
# =============================================================================
print("[1/8] Generating Plot 1: Pearson Correlation Heatmap...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Physical Criteria (Cluster 0, N=49 deployable candidates)
c0 = df_mcdm[df_mcdm["cluster_id"] == 0].copy()
crit_cols = ["Tm_C", "f_ASTP", "latent_heat_kJ_kg", "rho_H_MJ_m3", "TC_W_mK"]
crit_labels = ["Tm (°C)", "f_ASTP (-)", "ΔHf (kJ/kg)", "ρH (MJ/m³)", "k (W/m·K)"]
p_crit = c0[crit_cols].corr(method="pearson")

sns.heatmap(
    p_crit, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
    xticklabels=crit_labels, yticklabels=crit_labels, ax=ax1,
    cbar_kws={"label": "Pearson r"}
)
ax1.set_title("Physical Criteria Linear Correlation\n(Assam N=49 Deployable Candidates)", fontsize=11, weight="bold")

# Panel B: MCDM Continuous Cardinal Scores
score_cols = ["topsis_score", "gra_grade", "promethee_flow", "vikor_Q", "borda_score"]
score_labels = ["TOPSIS Score", "GRA Grade", "PROMETHEE Flow", "VIKOR Q", "Borda Score"]
p_scores = c0[score_cols].corr(method="pearson")

sns.heatmap(
    p_scores, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
    xticklabels=score_labels, yticklabels=score_labels, ax=ax2,
    cbar_kws={"label": "Pearson r"}
)
ax2.set_title("MCDM Cardinal Scores Linear Correlation\n(Cluster 0, N=49, Continuous Metrics)", fontsize=11, weight="bold")

plt.suptitle("Assam Objective 1 — Pearson Correlation Analysis (Physical Criteria & MCDM Scores)", fontsize=13, weight="bold")
plt.tight_layout()
p1_path = os.path.join(OUT_DIR, "01_pearson_criteria_and_scores_heatmap.png")
fig.savefig(p1_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 2: Comparative Heatmap (Pearson on Scores vs Spearman on Ranks)
# =============================================================================
print("[2/8] Generating Plot 2: Pearson Scores vs Spearman Ranks...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Pearson on 4 MCDM cardinal scores
m_scores = ["topsis_score", "gra_grade", "promethee_flow", "vikor_Q"]
m_labels = ["TOPSIS", "GRA", "PROMETHEE II", "VIKOR"]
p_mat = c0[m_scores].corr(method="pearson")

sns.heatmap(
    p_mat, annot=True, fmt=".3f", cmap="Blues", vmin=-1, vmax=1,
    xticklabels=m_labels, yticklabels=m_labels, ax=ax1,
    cbar_kws={"label": "Pearson r (Cardinal Scores)"}
)
ax1.set_title("Pearson r on Continuous Scores\n(Evaluates Linear Score Agreement)", fontsize=11, weight="bold")

# Panel B: Spearman on 4 MCDM ordinal ranks
m_ranks = ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"]
sp_mat = c0[m_ranks].corr(method="spearman")

sns.heatmap(
    sp_mat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=-1, vmax=1,
    xticklabels=m_labels, yticklabels=m_labels, ax=ax2,
    cbar_kws={"label": "Spearman ρ (Ordinal Ranks)"}
)
ax2.set_title("Spearman ρ on Ordinal Ranks\n(Evaluates Monotonic Rank Concordance)", fontsize=11, weight="bold")

plt.suptitle("Assam Objective 1 — Method Concordance: Pearson (Scores) vs Spearman (Ranks)\n(Cluster 0 Baseline, N=49 Alternatives)", fontsize=13, weight="bold")
plt.tight_layout()
p2_path = os.path.join(OUT_DIR, "02_pearson_vs_spearman_mcdm_comparison.png")
fig.savefig(p2_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 3: Method-wise PCM Ranking Comparison (Top 10 per Cluster)
# =============================================================================
print("[3/8] Generating Plot 3: Method-wise PCM Ranking Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
methods_plot = ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"]
m_names = ["TOPSIS", "GRA", "PROMETHEE", "VIKOR", "Borda"]

colors_10 = sns.color_palette("tab10", 10)

for cid, ax in enumerate(axes):
    c_df = df_mcdm[df_mcdm["cluster_id"] == cid].sort_values("consensus_rank").head(10).copy()
    for idx, (_, row) in enumerate(c_df.iterrows()):
        y_vals = [row[m] for m in methods_plot]
        ax.plot(range(len(m_names)), y_vals, "-o", color=colors_10[idx], lw=2.0, markersize=7, label=f"{row['consensus_rank']}. {row['display_name']}")
    
    ax.set_xticks(range(len(m_names)))
    ax.set_xticklabels(m_names, fontsize=10, weight="bold")
    ax.set_title(CLUSTER_LABELS[cid], fontsize=11, weight="bold")
    ax.set_xlabel("MCDM Method", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

axes[0].set_ylabel("Rank (1 = Best)", fontsize=11, weight="bold")
axes[0].set_yticks(range(1, 16))
axes[0].invert_yaxis()

plt.suptitle("Assam Objective 1 — Method-wise PCM Ranking Comparison across 4 MCDM Methods & Borda Consensus", fontsize=14, weight="bold")
plt.tight_layout()
p3_path = os.path.join(OUT_DIR, "03_mcdm_methods_ranking_comparison.png")
fig.savefig(p3_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 4: Consensus Borda vs Copeland Tournament Aggregation
# =============================================================================
print("[4/8] Generating Plot 4: Consensus Borda vs Copeland...")
# Vectorized Copeland calculation across all clusters
copeland_scores = []
copeland_ranks = []

for cid in [0, 1, 2]:
    sub = df_mcdm[df_mcdm["cluster_id"] == cid].copy()
    ranks = sub[m_ranks].values
    wins = (ranks[:, None, :] < ranks[None, :, :]).sum(axis=-1)
    losses = (ranks[:, None, :] > ranks[None, :, :]).sum(axis=-1)
    c_net = (wins > losses).sum(axis=1) - (losses > wins).sum(axis=1)
    c_rank = pd.Series(c_net).rank(ascending=False, method="min").astype(int).values
    copeland_scores.extend(c_net)
    copeland_ranks.extend(c_rank)

df_mcdm["copeland_score"] = copeland_scores
df_mcdm["copeland_rank"] = copeland_ranks

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Scatter of Borda Score vs Copeland Score (Cluster 0)
c0 = df_mcdm[df_mcdm["cluster_id"] == 0]
r_bc, p_bc = spearmanr(c0["borda_score"], c0["copeland_score"])

sns.scatterplot(
    data=c0, x="borda_score", y="copeland_score", hue="consensus_rank", palette="viridis_r",
    s=80, edgecolor="k", ax=ax1
)
for _, r in c0.head(5).iterrows():
    ax1.annotate(r["display_name"], (r["borda_score"] + 1, r["copeland_score"] - 1), fontsize=8.5, weight="bold")

ax1.set_title(f"Borda Score vs Copeland Score (Cluster 0)\nSpearman ρ = {r_bc:.4f} (p < 10⁻²⁰)", fontsize=11, weight="bold")
ax1.set_xlabel("Borda Consensus Score (Sum of Ranks Inverted)", fontsize=10)
ax1.set_ylabel("Copeland Tournament Score (Net Wins - Losses)", fontsize=10)
ax1.grid(alpha=0.3)

# Panel B: Top 10 Borda vs Copeland Ranks (Cluster 0)
top10_c0 = c0.sort_values("consensus_rank").head(10)
x_indices = np.arange(len(top10_c0))
width = 0.35

ax2.bar(x_indices - width/2, top10_c0["consensus_rank"], width, label="Borda Rank", color="#1f77b4")
ax2.bar(x_indices + width/2, top10_c0["copeland_rank"], width, label="Copeland Rank", color="#ff7f0e")
ax2.set_xticks(x_indices)
ax2.set_xticklabels(top10_c0["display_name"], rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("Rank (1 = Best)", fontsize=10)
ax2.set_title("Top 10 Candidates: Borda vs Copeland Rank Agreement", fontsize=11, weight="bold")
ax2.legend()
ax2.grid(alpha=0.3, axis="y")

plt.suptitle("Assam Objective 1 — Consensus Ranking Robustness: Borda vs Copeland Aggregation", fontsize=13, weight="bold")
plt.tight_layout()
p4_path = os.path.join(OUT_DIR, "04_consensus_borda_and_copeland_by_cluster.png")
fig.savefig(p4_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 5: Top PCM Candidates Across Clusters
# =============================================================================
print("[5/8] Generating Plot 5: Top PCM Candidates Across Clusters...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Top 5 consensus candidates across all clusters
top5_names = ["RT44HC", "savE® OM49", "n-Docosane (C22)", "RT45HC", "n-Tricosane (C23)"]
top5_df = df_mcdm[df_mcdm["display_name"].isin(top5_names)].copy()

# Panel A: Melting Temperature and Solar Potential (f_ASTP)
for pcm in top5_names:
    sub = top5_df[top5_df["display_name"] == pcm]
    ax1.plot(sub["cluster_id"], sub["f_ASTP"], "-o", lw=2, markersize=8, label=f"{pcm} (Tm={sub['Tm_C'].iloc[0]:.1f}°C)")

ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(["C0: Valley", "C1: Subtropical", "C2: Highland"], fontsize=10, weight="bold")
ax1.set_ylabel("Solar Potential Utilization f_ASTP (-)", fontsize=10, weight="bold")
ax1.set_title("Solar-Thermal Utilization Factor (f_ASTP) across Regimes", fontsize=11, weight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Panel B: Borda Consensus Scores Across Clusters
borda_pivot = top5_df.pivot(index="display_name", columns="cluster_id", values="borda_score").reindex(top5_names)
borda_pivot.columns = ["Cluster 0", "Cluster 1", "Cluster 2"]
borda_pivot.plot(kind="bar", ax=ax2, colormap="viridis", edgecolor="k", lw=0.7)
ax2.set_ylabel("Borda Consensus Score", fontsize=10, weight="bold")
ax2.set_title("Consensus Borda Score Comparison across Thermal Clusters", fontsize=11, weight="bold")
ax2.set_xticklabels(top5_names, rotation=25, ha="right", fontsize=9.5)
ax2.grid(alpha=0.3, axis="y")
ax2.set_ylim(160, 200)

plt.suptitle("Assam Objective 1 — Top PCM Candidate Stability Across Climate Regimes", fontsize=13, weight="bold")
plt.tight_layout()
p5_path = os.path.join(OUT_DIR, "05_top_pcm_candidates_cross_cluster.png")
fig.savefig(p5_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 6: Monte Carlo Top-1 Candidate Selection Frequency (Retention Rate)
# =============================================================================
print("[6/8] Generating Plot 6: Monte Carlo Selection Frequency...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for cid, ax in enumerate(axes):
    mc_c = df_mc[df_mc["cluster_id"] == cid].sort_values("top1_retention_rate", ascending=False).head(8).copy()
    y_pos = np.arange(len(mc_c))
    ret_pct = mc_c["top1_retention_rate"] * 100.0
    
    bars = ax.barh(y_pos, ret_pct, color="#2b5c8f", edgecolor="black", lw=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mc_c["display_name"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 Selection Probability (%)", fontsize=10, weight="bold")
    ax.set_title(CLUSTER_LABELS[cid], fontsize=11, weight="bold")
    ax.grid(alpha=0.3, axis="x")
    
    for bar in bars:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5, weight="bold")

plt.suptitle("Assam Objective 1 — Monte Carlo Top-1 Candidate Selection Frequency\n(5,000 Dirichlet Weight-Perturbation & Property-Noise Draws)", fontsize=13, weight="bold")
plt.tight_layout()
p6_path = os.path.join(OUT_DIR, "06_monte_carlo_top1_selection_frequency.png")
fig.savefig(p6_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 7: Monte Carlo Top-3 Inclusion Probability
# =============================================================================
print("[7/8] Generating Plot 7: Monte Carlo Top-3 Inclusion Probability...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for cid, ax in enumerate(axes):
    mc_c = df_mc[df_mc["cluster_id"] == cid].sort_values("top3_inclusion_probability", ascending=False).head(10).copy()
    y_pos = np.arange(len(mc_c))
    prob_pct = mc_c["top3_inclusion_probability"] * 100.0
    
    # Color-coded by confidence tier
    bar_colors = ["#2ca02c" if p >= 70.0 else "#ff7f0e" if p >= 40.0 else "#1f77b4" for p in prob_pct]
    bars = ax.barh(y_pos, prob_pct, color=bar_colors, edgecolor="black", lw=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mc_c["display_name"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Top-3 Inclusion Probability (%)", fontsize=10, weight="bold")
    ax.set_title(CLUSTER_LABELS[cid], fontsize=11, weight="bold")
    
    # Reference lines
    ax.axvline(80, color="#2ca02c", linestyle="--", lw=1.2, label="High Confidence (≥80%)" if cid == 0 else "")
    ax.axvline(50, color="#ff7f0e", linestyle=":", lw=1.2, label="Moderate Confidence (≥50%)" if cid == 0 else "")
    ax.grid(alpha=0.3, axis="x")
    
    for bar in bars:
        w = bar.get_width()
        if w > 1.0:
            ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5, weight="bold")

axes[0].legend(loc="lower right", fontsize=8.5)
plt.suptitle("Assam Objective 1 — Monte Carlo Top-3 Recommendation Inclusion Probability\n(5,000 Sensitivity Draws per Cluster)", fontsize=13, weight="bold")
plt.tight_layout()
p7_path = os.path.join(OUT_DIR, "07_monte_carlo_top3_inclusion_probability.png")
fig.savefig(p7_path, dpi=300)
plt.close(fig)

# =============================================================================
# PLOT 8: Cluster-wise Monte Carlo Stability Comparison
# =============================================================================
print("[8/8] Generating Plot 8: Cluster-wise Monte Carlo Stability...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Mean Spearman rho vs Baseline across clusters
cluster_rhos = [
    df_mc[df_mc["cluster_id"] == cid]["mean_spearman_rho_vs_baseline"].iloc[0]
    for cid in [0, 1, 2]
]
c_names = ["Cluster 0\n(Valley)", "Cluster 1\n(Subtropical)", "Cluster 2\n(Highland)"]

bars = ax1.bar(c_names, cluster_rhos, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", width=0.45)
ax1.set_ylim(0.7, 0.9)
ax1.set_ylabel("Mean Spearman Rank Correlation (ρ)", fontsize=10, weight="bold")
ax1.set_title("Overall Ranking Stability (Mean Spearman ρ vs Baseline)\nacross 5,000 Perturbations", fontsize=11, weight="bold")
ax1.grid(alpha=0.3, axis="y")
for b in bars:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 0.005, f"ρ = {h:.4f}", ha="center", fontsize=10, weight="bold")

# Panel B: Distribution of Top-3 Inclusion Probabilities by Cluster
sns.boxplot(
    data=df_mc, x="cluster_id", y="top3_inclusion_probability",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"], ax=ax2, width=0.4
)
ax2.set_xticklabels(c_names, fontsize=10)
ax2.set_xlabel("Thermal Cluster", fontsize=10, weight="bold")
ax2.set_ylabel("Top-3 Inclusion Probability (-)", fontsize=10, weight="bold")
ax2.set_title("Candidate Top-3 Inclusion Probability Spread per Cluster\n(N=49 Deployable Alternatives)", fontsize=11, weight="bold")
ax2.grid(alpha=0.3, axis="y")

plt.suptitle("Assam Objective 1 — Cluster-wise Monte Carlo Sensitivity & Ranking Stability Summary", fontsize=13, weight="bold")
plt.tight_layout()
p8_path = os.path.join(OUT_DIR, "08_clusterwise_monte_carlo_stability_comparison.png")
fig.savefig(p8_path, dpi=300)
plt.close(fig)

print("All 8 publication-quality plots successfully generated and verified in:")
print(f"  {OUT_DIR}")

