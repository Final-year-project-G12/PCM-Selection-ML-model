"""
plot_top3_agreement.py  — Assam
=================================
Generates the agreement plot (MCDM Consensus Rank vs Physics Simulation Rank)
restricted to only the TOP-3 MCDM-ranked PCMs per cluster.

Matches the exact visual style of 11_agreement_plot.png:
  - White background, red dashed 1:1 line
  - Per-cluster marker shapes with small horizontal dodge
  - No annotation boxes or arrows
  - Absolute physics rank (within each cluster, across ALL PCMs)
  - MCDM consensus rank as-is (1, 2, 3)

Data sources:
  - data/processed/pcm/mcdm_topk_assam.csv
  - data/processed/pcm/physics_validation_results_assam.csv

Outputs:
  - plots_assam_ppt/6 PCM Recommendation and Output/Assam/11_agreement_plot_top3.png
  - plots_assam_ppt/6 PCM Recommendation and Output/Assam/11_agreement_plot_top3_interactive.html
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TOPK_CSV  = os.path.join(BASE_DIR, "data", "processed", "pcm", "mcdm_topk_assam.csv")
PHYS_CSV  = os.path.join(BASE_DIR, "data", "processed", "pcm", "physics_validation_results_assam.csv")
OUT_DIR   = os.path.join(BASE_DIR, "plots_assam_ppt",
                          "6 PCM Recommendation and Output", "Assam")
os.makedirs(OUT_DIR, exist_ok=True)

# ── style constants matching original 11_agreement_plot.png ──────────────────
PAL = ["#e6194b", "#3cb44b", "#4363d8"]
CLUSTER_MARKERS = {0: "o", 1: "s", 2: "^"}
CLUSTER_OFFSETS = {0: -0.12, 1: 0.0, 2: 0.12}
CLUSTER_LABELS  = {
    0: "Cluster 0 (Circle)",
    1: "Cluster 1 (Square)",
    2: "Cluster 2 (Triangle)",
}

# ── load data ─────────────────────────────────────────────────────────────────
topk = pd.read_csv(TOPK_CSV)
phys = pd.read_csv(PHYS_CSV)

# Keep only the MCDM-top-3 per cluster
top3 = topk[topk["consensus_rank"] <= 3].copy()

# ABSOLUTE physics rank within each cluster (across ALL PCMs in that cluster)
phys = phys.copy()
phys["sim_rank"] = phys.groupby("cluster_id")["hours_target_met_per_year"].rank(
    ascending=False, method="min"
)

# Merge — only top-3 MCDM rows, but carrying their absolute physics rank
mg = top3.merge(
    phys[["cluster_id", "name", "hours_target_met_per_year", "sim_rank",
          "annual_solar_fraction"]],
    on=["cluster_id", "name"], how="left"
)
mg["cluster_id"] = mg["cluster_id"].astype(int)

print("Top-3 MCDM PCMs with absolute physics ranks:")
print(mg[["cluster_id", "name", "consensus_rank", "sim_rank",
          "hours_target_met_per_year"]].to_string(index=False))

# Spearman correlation
valid = mg.dropna(subset=["sim_rank", "consensus_rank"])
rho, pval = spearmanr(valid["sim_rank"], valid["consensus_rank"])
print(f"\nSpearman rho = {rho:.3f}  (p = {pval:.4f})  n = {len(valid)}")

# ═══════════════════════════════════════════════════════════════════════
# MATPLOTLIB STATIC PNG  — exact style of original 11_agreement_plot.png
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.5, 7.2))

# axis limits: x goes up to the max absolute physics rank among top-3 set
# y always 1–3 (MCDM top-3)
x_max = int(mg["sim_rank"].max())
y_max = 3

# 1:1 reference line (extend from 0.5 to the max of both axes)
ref_max = max(x_max, y_max)
ax.plot([0.5, ref_max + 0.5], [0.5, ref_max + 0.5],
        "r--", lw=1.5, alpha=0.8, label="Perfect agreement (1:1)")

# Scatter per cluster with horizontal dodge
for cid, grp in mg.groupby("cluster_id"):
    cid = int(cid)
    v = grp[["sim_rank", "consensus_rank"]].notna().all(axis=1)
    marker = CLUSTER_MARKERS.get(cid, "o")
    offset = CLUSTER_OFFSETS.get(cid, 0.0)
    color  = PAL[cid % len(PAL)]

    x_vals = grp.loc[v, "sim_rank"] + offset
    y_vals = grp.loc[v, "consensus_rank"]

    ax.scatter(
        x_vals, y_vals,
        color=color, marker=marker, s=120, alpha=0.9,
        edgecolors="white", linewidths=1.2,
        label=CLUSTER_LABELS.get(cid, f"Cluster {cid}"), zorder=4
    )

# Axes ticks and limits
ax.set_xticks(range(1, x_max + 1))
ax.set_yticks(range(1, y_max + 1))
ax.set_xlim(0.5, x_max + 0.5)
ax.set_ylim(0.5, y_max + 0.5)

ax.set_xlabel("Simulated Performance Rank (Annual Solar Hours)",
              fontsize=11, fontweight="bold")
ax.set_ylabel("MCDM Consensus Rank (Borda)",
              fontsize=11, fontweight="bold")
ax.set_title(
    "Assam - Physics Simulation vs MCDM Consensus Rank\n"
    "(Top 3 PCMs per Climate Regime - Distinct Shapes & Dodged Ranks)",
    fontsize=12, fontweight="bold", pad=10
)

ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.35, zorder=1)
plt.tight_layout()

out_png = os.path.join(OUT_DIR, "11_agreement_plot_top3.png")
fig.savefig(out_png, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] PNG saved: {out_png}")

# ═══════════════════════════════════════════════════════════════════════
# PLOTLY INTERACTIVE HTML
# ═══════════════════════════════════════════════════════════════════════
SYMBOL_MAP = {"Cluster 0 (Circle)": "circle",
              "Cluster 1 (Square)": "square",
              "Cluster 2 (Triangle)": "triangle-up"}
COLOR_MAP  = {"Cluster 0 (Circle)": PAL[0],
              "Cluster 1 (Square)": PAL[1],
              "Cluster 2 (Triangle)": PAL[2]}

mg_px = mg.copy()
mg_px["offset"]         = mg_px["cluster_id"].map(CLUSTER_OFFSETS).fillna(0)
mg_px["sim_rank_dodged"] = mg_px["sim_rank"] + mg_px["offset"]
mg_px["Cluster"]         = mg_px["cluster_id"].map(CLUSTER_LABELS).fillna("Cluster")

fig_px = go.Figure()

# 1:1 reference line
rng = [0.5, ref_max + 0.5]
fig_px.add_trace(go.Scatter(
    x=rng, y=rng, mode="lines",
    line=dict(dash="dash", color="red", width=1.5),
    name="Perfect agreement (1:1)", hoverinfo="skip"
))

for cl_label, grp in mg_px.groupby("Cluster"):
    v = grp[["sim_rank", "consensus_rank"]].notna().all(axis=1)
    grp_v = grp[v]

    hover_texts = []
    for _, row in grp_v.iterrows():
        hover_texts.append(
            f"<b>{row['name']}</b><br>"
            f"MCDM Rank: #{int(row['consensus_rank'])}<br>"
            f"Physics Rank (absolute): #{int(row['sim_rank'])}<br>"
            f"Hours Target Met: {int(row['hours_target_met_per_year'])} hrs/yr<br>"
            f"Solar Fraction: {row['annual_solar_fraction']:.4f}<br>"
            f"{cl_label}"
        )

    fig_px.add_trace(go.Scatter(
        x=grp_v["sim_rank_dodged"].values,
        y=grp_v["consensus_rank"].values,
        mode="markers",
        name=cl_label,
        marker=dict(
            symbol=SYMBOL_MAP.get(cl_label, "circle"),
            size=12, color=COLOR_MAP.get(cl_label, "#333"),
            line=dict(width=1, color="white")
        ),
        hovertext=hover_texts, hoverinfo="text"
    ))

fig_px.update_layout(
    title=dict(
        text=(
            "Assam - Physics Simulation vs MCDM Consensus Rank<br>"
            "<sup>(Top 3 PCMs per Climate Regime - Distinct Shapes & Dodged Ranks)</sup>"
        ),
        font=dict(size=14)
    ),
    xaxis=dict(
        title="Simulated Performance Rank (Annual Solar Hours)",
        tickmode="linear", tick0=1, dtick=1,
        range=[0.5, x_max + 0.5],
        tickfont=dict(size=12), zeroline=False, gridcolor="#e0e0e0"
    ),
    yaxis=dict(
        title="MCDM Consensus Rank (Borda)",
        tickmode="linear", tick0=1, dtick=1,
        range=[0.5, y_max + 0.5],
        tickfont=dict(size=12), zeroline=False, gridcolor="#e0e0e0"
    ),
    template="plotly_white",
    height=600, width=780,
    legend=dict(x=0.02, y=0.97, bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cccccc", borderwidth=1),
    font=dict(family="Arial, sans-serif")
)

out_html = os.path.join(OUT_DIR, "11_agreement_plot_top3_interactive.html")
fig_px.write_html(out_html, include_plotlyjs="inline")
print(f"[OK] HTML saved: {out_html}")
print("\nDone.")
