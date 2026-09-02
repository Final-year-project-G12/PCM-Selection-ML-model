"""
Assam PCM Pipeline - Objective 1 Plot Generator
Generates all 13 required plots (static PNG + interactive Plotly/Folium)
Output: data/plots/assam_objective1/
"""

import os, sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import folium
    from folium.plugins import MarkerCluster
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

warnings.filterwarnings("ignore")

BASE         = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV      = os.path.join(BASE, "data", "processed", "climate_assam_points.csv")
PHYSICAL_CSV = os.path.join(BASE, "data", "preprocessed", "assam_cleaned_physical.csv")
CLUSTERS     = os.path.join(BASE, "data", "processed", "clustering", "cluster_assignments_assam.csv")
PCM_DB       = os.path.join(BASE, "data", "processed", "pcm", "pcm_database_assam.csv")
FEASIBILITY  = os.path.join(BASE, "data", "processed", "pcm", "feasibility_survivors_assam.csv")
TOPK         = os.path.join(BASE, "data", "processed", "pcm", "mcdm_topk_assam.csv")
MC_STABILITY = os.path.join(BASE, "data", "processed", "pcm", "monte_carlo_stability_assam.csv")
PHYS_VAL     = os.path.join(BASE, "data", "processed", "pcm", "physics_validation_results_assam.csv")
OUT          = os.path.join(BASE, "data", "plots", "assam_objective1")
os.makedirs(OUT, exist_ok=True)

METHODS = ["topsis", "gra", "promethee", "vikor"]
MRANK   = [f"{m}_rank" for m in METHODS] + ["consensus_rank"]
PAL     = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

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

def load(path, label="", **kw):
    if not os.path.exists(path):
        print(f"  skip {label}: not found ({path})")
        return None
    return pd.read_csv(path, **kw)

def sfig(n, dpi=150):
    plt.savefig(os.path.join(OUT, n), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  {n}")

def shtml(fig, n):
    fig.write_html(os.path.join(OUT, n), include_plotlyjs="cdn")
    print(f"  {n}")

# ---- Plot 1: Raw vs Preprocessed GHI ----
def p01():
    print("[1/13] Raw vs Preprocessed Radiation")
    raw = load(RAW_CSV, "raw", nrows=200000)
    pre = load(PHYSICAL_CSV, "preprocessed", nrows=200000)
    if raw is None or pre is None:
        raw_sig = load(os.path.join(BASE, "data", "processed", "climate_signatures_raw.csv"), "raw_sig")
        if raw_sig is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(raw_sig["GHI_mean"].dropna(), bins=30, color="#e07b39", alpha=0.8, label="GHI Mean (Signature)")
            ax.set(title="Assam - Solar Radiation (GHI) Distribution across Points", xlabel="GHI Mean (W/m2)", ylabel="Point Count")
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); sfig("01_raw_vs_preprocessed_radiation.png")
        return
    pt_col = "point_id" if "point_id" in raw.columns else raw.columns[0]
    pt = raw[pt_col].iloc[0]
    r = raw[raw[pt_col] == pt]
    p = pre[pre[pt_col] == pt] if pt_col in pre.columns else pre.head(len(r))
    ghi_r = "era5_GHI" if "era5_GHI" in r.columns else ([c for c in r.columns if "GHI" in c.upper()] or [None])[0]
    ghi_p = "era5_GHI" if "era5_GHI" in p.columns else ([c for c in p.columns if "GHI" in c.upper()] or [None])[0]
    if not ghi_r: return
    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    ax[0].plot(r[ghi_r].values[:500], color="#e07b39", lw=0.8, alpha=0.8, label="Raw GHI")
    ax[0].set(title=f"Raw GHI - Point {pt}", ylabel="GHI (W/m2)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if ghi_p:
        ax[1].plot(p[ghi_p].values[:500], color="#3b7dd8", lw=0.9, label="Preprocessed GHI")
    ax[1].set(title="Preprocessed GHI", ylabel="GHI (W/m2)", xlabel="Record index"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.suptitle("Assam - Raw vs Preprocessed Solar Radiation (GHI)", fontsize=13)
    plt.tight_layout(); sfig("01_raw_vs_preprocessed_radiation.png")

# ---- Plot 2: Climate Regime Map ----
def p02():
    print("[2/13] Climate-Regime Map")
    df = load(CLUSTERS, "clusters")
    if df is None or not {"lat", "lon", "cluster_id"}.issubset(df.columns): return
    nc = df["cluster_id"].nunique()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 8))
    for cid, g in df.groupby("cluster_id"):
        ax.scatter(g["lon"], g["lat"], color=cmap(int(cid)), s=80, alpha=0.85, edgecolors="white", lw=0.4, label=f"Cluster {cid}")
    ax.set(title="Assam - Climate Regime Map\n(GMM clusters per grid point)", xlabel="Longitude E", ylabel="Latitude N")
    ax.legend(title="Regime", loc="lower right", fontsize=9); ax.grid(alpha=0.25, ls="--")
    plt.tight_layout(); sfig("02_climate_regime_map.png")
    
    if HAS_PLOTLY:
        fig_px = px.scatter(df, x="lon", y="lat", color=df["cluster_id"].astype(str),
                            hover_data=["point_id", "cluster_id"] if "point_id" in df.columns else ["cluster_id"],
                            title="Assam - Climate Regime Map", labels={"color": "Cluster"}, template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Set1)
        fig_px.update_traces(marker=dict(size=9, opacity=0.8)); fig_px.update_layout(height=600)
        shtml(fig_px, "02_climate_regime_map_interactive.html")

    if HAS_FOLIUM:
        m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=7, tiles="CartoDB positron")
        mc = MarkerCluster().add_to(m)
        cf = ["red", "green", "blue", "purple", "orange", "darkred", "lightred", "beige"]
        for _, r in df.iterrows():
            cid = int(r["cluster_id"])
            folium.CircleMarker(location=[r["lat"], r["lon"]], radius=6, color=cf[cid % len(cf)], fill=True, fill_opacity=0.75,
                popup=folium.Popup(f"<b>Point:</b> {r.get('point_id', '')}<br><b>Cluster:</b> {cid}", max_width=220)
            ).add_to(mc)
        m.save(os.path.join(OUT, "02_climate_regime_map_folium.html"))
        print("  02_climate_regime_map_folium.html")

# ---- Plot 3: Candidate Space Scatter ----
def p03():
    print("[3/13] PCM Candidate Space (Melting Point vs Latent Heat)")
    db = load(PCM_DB, "pcm_db")
    feas = load(FEASIBILITY, "feasibility")
    if db is None: return
    tm_col = "Tm_C" if "Tm_C" in db.columns else ([c for c in db.columns if "Tm" in c or "melt" in c.lower()] or [None])[0]
    lh_col = "latent_heat_kJ_kg" if "latent_heat_kJ_kg" in db.columns else ([c for c in db.columns if "latent" in c.lower()] or [None])[0]
    if not tm_col or not lh_col: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(db[tm_col], db[lh_col], color="gray", alpha=0.5, s=50, label=f"All Candidates (n={len(db)})")
    if feas is not None and tm_col in feas.columns and lh_col in feas.columns:
        ax.scatter(feas[tm_col], feas[lh_col], color="#3b7dd8", alpha=0.85, s=80, edgecolors="white", lw=0.5, label=f"Feasible Survivors (n={len(feas)})")
    ax.set(xlabel="Melting Temperature (°C)", ylabel="Latent Heat (kJ/kg)", title="Assam - PCM Candidate Property Space")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("03_melting_point_vs_latent_heat.png")

# ---- Plot 4 & 5: Feasible Candidates & Survivors per Cluster ----
def p04_05():
    print("[4-5/13] Feasible Candidates and Survivors per Cluster")
    feas = load(FEASIBILITY, "feasibility")
    if feas is None or "cluster_id" not in feas.columns: return
    counts = feas.groupby("cluster_id").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color=[PAL[int(c) % len(PAL)] for c in counts.index], edgecolor="white")
    ax.set(xlabel="Cluster ID", ylabel="Number of Feasible PCM Candidates", title="Assam - PCM Feasible Survivors per Cluster")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(1, v * 0.02), str(v), ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); sfig("05_pcm_survivors_per_cluster.png")

# ---- Plot 7: Bump Chart Ranks ----
def p07():
    print("[7/13] Bump Chart for Rank Stability across MCDM Methods")
    topk = load(TOPK, "topk")
    if topk is None or "cluster_id" not in topk.columns: return
    topk = ensure_ranks(topk)
    name_col = "name" if "name" in topk.columns else "PCM_Name"
    ranks = [c for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"] if c in topk.columns]
    if len(ranks) < 2: return
    clus = sorted(topk["cluster_id"].unique())
    sub = topk[topk["cluster_id"] == clus[0]].head(8)
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, r in sub.iterrows():
        y = [r[rk] for rk in ranks]
        ax.plot([rk.replace("_rank", "").upper() for rk in ranks], y, "-o", lw=2, label=str(r[name_col])[:18] if name_col in sub.columns else str(r.name))
    ax.invert_yaxis()
    ax.set(title=f"Assam - MCDM Rank Bump Chart (Cluster {clus[0]})", ylabel="Rank (lower=better)", xlabel="MCDM Method")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left"); ax.grid(alpha=0.3); sfig("07_bump_chart_ranks.png")

# ---- Plot 8: Method Rank Correlation Heatmap ----
def p08():
    print("[8/13] Method Rank Correlation Heatmap")
    topk = load(TOPK, "topk")
    if topk is None: return
    topk = ensure_ranks(topk)
    ranks = [c for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"] if c in topk.columns]
    if len(ranks) < 2: return
    corr = topk[ranks].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, xticklabels=[r.replace("_rank", "").upper() for r in ranks], yticklabels=[r.replace("_rank", "").upper() for r in ranks])
    ax.set_title("Assam - Spearman Rank Correlation across MCDM Methods", fontsize=12)
    plt.tight_layout(); sfig("08_method_rank_correlation_heatmap.png")

# ---- Plot 10: Rank Reversal Distribution ----
def p10():
    print("[10/13] Rank Reversal Violin Plot")
    topk = load(TOPK, "topk")
    if topk is None: return
    topk = ensure_ranks(topk)
    ranks = [c for c in ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank"] if c in topk.columns]
    if len(ranks) < 2: return
    topk["rank_std"] = topk[ranks].std(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=topk, x="cluster_id", y="rank_std", palette="Set2", ax=ax)
    ax.set(title="Assam - MCDM Rank Standard Deviation across Methods per Cluster", xlabel="Cluster ID", ylabel="Rank Std Dev")
    ax.grid(alpha=0.3, axis="y"); sfig("10_rank_reversal_violin_bar.png")

# ---- Plot 11: Consensus Agreement Plot ----
def p11():
    print("[11/13] Consensus Agreement Plot")
    topk = load(TOPK, "topk")
    if topk is None or "consensus_rank" not in topk.columns: return
    fig, ax = plt.subplots(figsize=(9, 5))
    name_col = "name" if "name" in topk.columns else "PCM_Name"
    top1 = topk[topk["consensus_rank"] == 1]
    ax.bar(top1["cluster_id"].astype(str), top1.get("borda_score", pd.Series([1]*len(top1))), color="#3cb44b", edgecolor="white")
    for idx, row in top1.iterrows():
        ax.text(str(row["cluster_id"]), 0.5, str(row.get(name_col, ""))[:12], ha="center", color="white", fontweight="bold", rotation=90 if len(str(row.get(name_col, ""))) > 10 else 0)
    ax.set(title="Assam - #1 Ranked PCM per Cluster (Consensus Borda)", xlabel="Cluster ID", ylabel="Consensus Score")
    ax.grid(alpha=0.3, axis="y"); sfig("11_agreement_plot.png")

# ---- Plot 13: Recommended PCM Summary ----
def p13():
    print("[13/13] Recommended PCM Summary")
    topk = load(TOPK, "topk")
    if topk is None: return
    topk = ensure_ranks(topk)
    top1 = topk[topk["consensus_rank"] == 1]
    if top1.empty: return
    name_col = "name" if "name" in top1.columns else "PCM_Name"
    fig, ax = plt.subplots(figsize=(10, 4 + len(top1)*0.5))
    ax.axis("off")
    tbl_data = []
    for _, r in top1.iterrows():
        tbl_data.append([
            f"Cluster {r['cluster_id']}",
            str(r.get(name_col, "")),
            f"{r.get('Tm_C', r.get('Tm_target_C', 0)):.1f} °C",
            f"{r.get('latent_heat_kJ_kg', 0):.0f} kJ/kg",
            f"{r.get('TC_W_mK', 0):.2f} W/mK"
        ])
    tbl = ax.table(cellText=tbl_data, colLabels=["Cluster", "Top Candidate", "Melting Temp", "Latent Heat", "Thermal Cond"], loc="center", cellLoc="center")
    tbl.scale(1, 1.8); tbl.set_fontsize(10)
    ax.set_title("Assam PCM Selection - Top Recommendation Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout(); sfig("13_recommended_pcm_summary.png")

def main():
    print(f"=== Generating Assam Objective 1 Plots -> {OUT} ===")
    p01()
    p02()
    p03()
    p04_05()
    p07()
    p08()
    p10()
    p11()
    p13()
    print("=== All Objective 1 Assam plots complete! ===")

if __name__ == "__main__":
    main()
