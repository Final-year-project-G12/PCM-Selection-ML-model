"""
plot_top3_agreement.py  — Assam (v2)
=================================
Generates the agreement plot (MCDM Consensus Rank vs Physics Simulation Rank)
restricted to only the TOP-3 MCDM-ranked PCMs per cluster.

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
import matplotlib.patheffects as pe
import plotly.graph_objects as go
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TOPK_CSV  = os.path.join(BASE_DIR, "data", "processed", "pcm", "mcdm_topk_assam.csv")
PHYS_CSV  = os.path.join(BASE_DIR, "data", "processed", "pcm", "physics_validation_results_assam.csv")
OUT_DIR   = os.path.join(BASE_DIR, "plots_assam_ppt",
                          "6 PCM Recommendation and Output", "Assam")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colours & markers per cluster ────────────────────────────────────────────
PALETTE  = ["#e6194b", "#3cb44b", "#4363d8"]
MARKERS  = {0: "o",  1: "s",  2: "^"}
JITTER   = {0: -0.16, 1: 0.0, 2: 0.16}   # horizontal dodge per cluster
CL_LABEL = {
    0: "Cluster 0 – Lowland Valley",
    1: "Cluster 1 – Floodplain",
    2: "Cluster 2 – Foothill",
}

# short name for labels
def shorten(name):
    return (name.replace("savE\u00ae ", "OM")
                .replace(" (docosane-class paraffin)", " C22H46"))

# ── load data ─────────────────────────────────────────────────────────────────
topk  = pd.read_csv(TOPK_CSV)
phys  = pd.read_csv(PHYS_CSV)

# Keep only the MCDM-top-3 per cluster (consensus_rank 1,2,3)
top3 = topk[topk["consensus_rank"] <= 3].copy()

# Absolute cluster-scoped physics rank (across ALL PCMs in that cluster)
phys = phys.copy()
phys["sim_rank_abs"] = phys.groupby("cluster_id")["hours_target_met_per_year"].rank(
    ascending=False, method="min")

mg = top3.merge(
    phys[["cluster_id", "name", "hours_target_met_per_year", "sim_rank_abs",
          "annual_solar_fraction", "complete_cycles_per_year"]],
    on=["cluster_id", "name"], how="left"
)

# Relative physics rank within the top-3 MCDM subset (1=best among these 3)
mg["sim_rank"] = mg.groupby("cluster_id")["hours_target_met_per_year"].rank(
    ascending=False, method="min")
mg["short_name"] = mg["name"].apply(shorten)
mg["cluster_id"] = mg["cluster_id"].astype(int)

print("Merged top-3 data:")
print(mg[["cluster_id", "name", "consensus_rank", "sim_rank",
          "hours_target_met_per_year"]].to_string(index=False))

# Spearman
valid = mg.dropna(subset=["sim_rank", "consensus_rank"])
rho, pval = spearmanr(valid["sim_rank"], valid["consensus_rank"])
print(f"\nSpearman rho = {rho:.3f}  (p = {pval:.4f})  n = {len(valid)}")

# ─────────────────────────────────────────────────────────────────────
# MATPLOTLIB STATIC PNG   (3 x 3 rank grid with jitter per cluster)
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8.5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

# light grey grid background cells (rank 1–3 x rank 1–3)
for xi in [1, 2, 3]:
    for yi in [1, 2, 3]:
        color = "#e8f5e9" if xi == yi else "#ffffff"
        ax.add_patch(plt.Rectangle((xi-0.45, yi-0.45), 0.9, 0.9,
                                    fc=color, ec="#dde", lw=0.7, zorder=0, alpha=0.7))

# 1:1 reference line
ax.plot([0.55, 3.45], [0.55, 3.45],
        color="#c0392b", ls="--", lw=2.0, alpha=0.65, label="Perfect agreement (1:1)", zorder=2)

# ── per-PCM label offset strategy ────────────────────────────────────
# We'll annotate with arrows so labels never overlap
label_anchors = {
    0: (+0.38, +0.28),   # Cluster 0 – upper right
    1: (+0.38, -0.28),   # Cluster 1 – lower right
    2: (-0.38, +0.28),   # Cluster 2 – upper left
}

for cid, grp in mg.groupby("cluster_id"):
    valid_g = grp.dropna(subset=["sim_rank", "consensus_rank"])
    if valid_g.empty:
        continue
    col = PALETTE[cid % len(PALETTE)]
    mkr = MARKERS.get(cid, "o")
    jit = JITTER.get(cid, 0)
    loff = label_anchors.get(cid, (0.3, 0.3))

    x_vals = valid_g["sim_rank"].values + jit
    y_vals = valid_g["consensus_rank"].values

    # Scatter
    ax.scatter(x_vals, y_vals,
               color=col, marker=mkr, s=220, alpha=0.93,
               edgecolors="white", linewidths=1.8,
               label=CL_LABEL.get(cid, f"Cluster {cid}"), zorder=5)

    # Annotate each point with PCM name, absolute physics rank, hours
    for (_, row), xv, yv in zip(valid_g.iterrows(), x_vals, y_vals):
        sn = row["short_name"]
        hrs = int(row["hours_target_met_per_year"])
        abs_r = int(row["sim_rank_abs"]) if pd.notna(row.get("sim_rank_abs")) else "?"
        xt = xv + loff[0]
        yt = yv + loff[1]

        ax.annotate(
            f"{sn}\n{hrs} hrs/yr (abs #{abs_r})",
            xy=(xv, yv),
            xytext=(xt, yt),
            fontsize=8.5,
            color=col,
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1,
                            connectionstyle="arc3,rad=0.1"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=0.9, alpha=0.88),
            zorder=8
        )

# ── Spearman annotation box ───────────────────────────────────────────────────
sign_str = "negative" if rho < 0 else "positive (weak)"
ax.text(
    0.97, 0.05,
    f"Spearman rho = {rho:.3f}  (p = {pval:.3f})\n{sign_str} correlation | Top-3 PCMs only",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.55", fc="#fff3cd", ec="#e0a800", lw=1.3, alpha=0.93)
)

# ── axes styling ─────────────────────────────────────────────────────────────
ax.set_xticks([1, 2, 3])
ax.set_yticks([1, 2, 3])
ax.set_xticklabels(["#1\n(Best)", "#2", "#3\n(Worst)"], fontsize=11)
ax.set_yticklabels(["#1\n(Best)", "#2", "#3\n(Worst)"], fontsize=11)
ax.set_xlim(0.4, 3.8)
ax.set_ylim(0.4, 3.8)
ax.set_xlabel("Physics Simulation Rank (Annual Hours Target Met)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("MCDM Consensus Rank (Borda Score)\n(Historical K=4 Pre-Audit)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title(
    "Assam - MCDM vs Physics Agreement: Top 3 PCMs per Climate Cluster\n"
    "Historical K=4 MCDM Rankings vs 10-Year Sub-Hourly Physics Simulation",
    fontsize=12.5, fontweight="bold", pad=14
)

ax.grid(False)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")

legend = ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92,
                   edgecolor="#cccccc", frameon=True, title="Climate Cluster",
                   title_fontsize=9)
plt.tight_layout(pad=1.8)

out_png = os.path.join(OUT_DIR, "11_agreement_plot_top3.png")
fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n[OK] PNG saved: {out_png}")

# ═══════════════════════════════════════════════════════════════════════
# PLOTLY INTERACTIVE HTML
# ═══════════════════════════════════════════════════════════════════════
SYMBOL_MAP = {0: "circle", 1: "square", 2: "triangle-up"}
COLOR_MAP  = {0: PALETTE[0], 1: PALETTE[1], 2: PALETTE[2]}

fig_px = go.Figure()

# 1:1 diagonal line
fig_px.add_trace(go.Scatter(
    x=[0.55, 3.45], y=[0.55, 3.45], mode="lines",
    line=dict(dash="dash", color="#c0392b", width=2),
    name="Perfect agreement (1:1)", hoverinfo="skip"
))

for cid, grp in mg.groupby("cluster_id"):
    cid = int(cid)
    valid_g = grp.dropna(subset=["sim_rank", "consensus_rank"])
    if valid_g.empty:
        continue

    hover_texts = []
    for _, row in valid_g.iterrows():
        abs_r = int(row['sim_rank_abs']) if pd.notna(row.get('sim_rank_abs')) else '?'
        hover_texts.append(
            f"<b>{row['name']}</b><br>"
            f"MCDM Rank (K=4): #{int(row['consensus_rank'])}<br>"
            f"Physics Rank (vs top-3): #{int(row['sim_rank'])}<br>"
            f"Physics Rank (vs ALL PCMs): #{abs_r}<br>"
            f"Hours Target Met: {int(row['hours_target_met_per_year'])} hrs/yr<br>"
            f"Solar Fraction: {row['annual_solar_fraction']:.4f}<br>"
            f"Cluster: {cid} | {CL_LABEL.get(cid, '')}"
        )

    fig_px.add_trace(go.Scatter(
        x=valid_g["sim_rank"].values,
        y=valid_g["consensus_rank"].values,
        mode="markers+text",
        name=CL_LABEL.get(cid, f"Cluster {cid}"),
        marker=dict(
            symbol=SYMBOL_MAP.get(cid, "circle"),
            size=20, color=COLOR_MAP.get(cid, "#333"),
            line=dict(width=2, color="white")
        ),
        text=valid_g["short_name"].tolist(),
        textposition=["top right", "bottom left", "top left"][: len(valid_g)],
        textfont=dict(size=11, color=COLOR_MAP.get(cid, "#333")),
        hovertext=hover_texts, hoverinfo="text"
    ))

fig_px.update_layout(
    title=dict(
        text=(
            "Assam - MCDM vs Physics Agreement: Top 3 PCMs per Climate Cluster<br>"
            f"<sup>Spearman rho = {rho:.3f} (p = {pval:.3f}) | Historical K=4 MCDM vs 10-Year Physics</sup>"
        ),
        font=dict(size=15)
    ),
    xaxis=dict(
        title="Physics Simulation Rank (Annual Hours Target Met)",
        tickmode="array", tickvals=[1, 2, 3],
        ticktext=["#1 (Best)", "#2", "#3 (Worst)"],
        range=[0.4, 3.8], tickfont=dict(size=13), zeroline=False, gridcolor="#e5e5e5"
    ),
    yaxis=dict(
        title="MCDM Consensus Rank (Borda)",
        tickmode="array", tickvals=[1, 2, 3],
        ticktext=["#1 (Best)", "#2", "#3 (Worst)"],
        range=[0.4, 3.8], tickfont=dict(size=13), zeroline=False, gridcolor="#e5e5e5"
    ),
    template="plotly_white", height=620, width=800,
    legend=dict(x=0.02, y=0.97, bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#cccccc", borderwidth=1),
    font=dict(family="Inter, Arial, sans-serif"),
    annotations=[dict(
        text=(
            f"<b>Spearman rho = {rho:.3f}</b><br>"
            f"p = {pval:.3f} | Top-3 PCMs only<br>"
            f"n = {len(valid)} data points"
        ),
        xref="paper", yref="paper", x=0.98, y=0.04,
        showarrow=False, align="right",
        bgcolor="#fff3cd", bordercolor="#e0a800", borderwidth=1.5,
        font=dict(size=11)
    )]
)

out_html = os.path.join(OUT_DIR, "11_agreement_plot_top3_interactive.html")
fig_px.write_html(out_html, include_plotlyjs="inline")
print(f"[OK] HTML saved: {out_html}")
print("\nDone.")
