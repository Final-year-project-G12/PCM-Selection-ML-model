"""
Comparison Plots - Assam PCM Pipeline
======================================
Generates cross-step comparison plots to help verify results make sense.
Output: data/plots/comparison/

Plots:
  1. Cluster GHI profiles: mean GHI by cluster over months
  2. PCM temperature target vs cluster mean temperature
  3. All MCDM method rankings side-by-side per cluster (top 5)
  4. Monte Carlo stability: top3 prob vs consensus rank scatter
  5. Latent heat distribution: feasible survivors vs all candidates
  6. Physics validation: hours_target_met vs MCDM rank
  7. Cross-cluster summary: key properties of top PCM per cluster
  8. Sensitivity: how rank changes if weights shift +/- 20% per criterion
"""

import os, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

BASE     = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
CLUSTERS = os.path.join(BASE, "data", "processed", "clustering", "cluster_assignments_assam.csv")
SIG_CSV  = os.path.join(BASE, "data", "processed", "climate_signatures_raw.csv")
FEAS     = os.path.join(BASE, "data", "processed", "pcm", "feasibility_survivors_assam.csv")
PCM_DB   = os.path.join(BASE, "data", "processed", "pcm", "pcm_database_assam.csv")
TOPK     = os.path.join(BASE, "data", "processed", "pcm", "mcdm_topk_assam.csv")
MC_CSV   = os.path.join(BASE, "data", "processed", "pcm", "monte_carlo_stability_assam.csv")
PHYS     = os.path.join(BASE, "data", "processed", "pcm", "physics_validation_assam.csv")
CMP_PHYS = os.path.join(BASE, "data", "processed", "pcm", "mcdm_vs_physics_comparison.csv")
CPROF    = os.path.join(BASE, "data", "processed", "clustering", "cluster_profiles_assam.csv")
OUT      = os.path.join(BASE, "data", "plots", "comparison")
os.makedirs(OUT, exist_ok=True)

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
MEDOID_MAP = {0: "ASP_0012", 1: "ASP_0092", 2: "ASP_0028"}

def load(p, label=""):
    if not os.path.exists(p):
        print(f"  skip {label}: not found ({p})")
        return None
    return pd.read_csv(p)

def sfig(n):
    plt.savefig(os.path.join(OUT, n), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {n}")

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

# ── Comparison 1: Cluster Mean GHI from Signature ────────────────────────
print("[1/8] Cluster GHI profiles from signature")
sig = load(SIG_CSV, "signature")
clu = load(CLUSTERS, "clusters")
if clu is not None and "cluster_id" not in clu.columns and "cluster" in clu.columns:
    clu["cluster_id"] = clu["cluster"]
VALID_CLUSTERS = sorted(clu["cluster_id"].unique()) if clu is not None else [0, 1, 2]

if sig is not None and clu is not None:
    merge_col = "point_id" if "point_id" in sig.columns and "point_id" in clu.columns else None
    if merge_col:
        mg = sig.merge(clu[[merge_col, "cluster_id"]], on=merge_col, how="inner")
    elif len(sig) == len(clu):
        mg = sig.copy()
        mg["cluster_id"] = clu["cluster_id"].values
    else:
        mg = None

    if mg is not None:
        ghi_col = "GHI_mean" if "GHI_mean" in mg.columns else ([c for c in mg.columns if "GHI" in c.upper()] or [None])[0]
        if ghi_col:
            fig, ax = plt.subplots(figsize=(9, 5))
            for cid in sorted(mg["cluster_id"].unique()):
                g = mg[mg["cluster_id"] == cid]
                ax.bar(str(cid), g[ghi_col].mean(), color=PAL[int(cid) % len(PAL)], edgecolor="white", lw=1, alpha=0.9, label=f"Cluster {cid}")
                ax.errorbar(str(cid), g[ghi_col].mean(), yerr=g[ghi_col].std(), fmt="none", color="black", capsize=5, lw=1.5)
            ax.set(xlabel="Cluster", ylabel=f"{ghi_col} (mean +/- std)", title="Comparison 1: Mean GHI by Climate Regime (Assam)")
            ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y"); sfig("01_comparison_cluster_ghi.png")
        else:
            print("  No GHI column in signature")

# ── Comparison 2: PCM Tm_target vs Cluster Mean Temp ────────────────────
print("[2/8] PCM Tm_target vs cluster mean temperature")
if sig is not None and clu is not None:
    if "point_id" in sig.columns and "point_id" in clu.columns:
        mg2 = sig.merge(clu[["point_id", "cluster_id"]], on="point_id", how="inner")
    elif len(sig) == len(clu):
        mg2 = sig.copy()
        mg2["cluster_id"] = clu["cluster_id"].values
    else:
        mg2 = None

    feas = load(FEAS, "feasibility")
    if mg2 is not None and feas is not None:
        t_col = "Ta_mean" if "Ta_mean" in mg2.columns else ("Ta_mean_proxy" if "Ta_mean_proxy" in mg2.columns else ([c for c in mg2.columns if "T_" in c or "temp" in c.lower()] or [None])[0])
        tm_col = "Tm_target_C" if "Tm_target_C" in feas.columns else ([c for c in feas.columns if "target" in c.lower() or "Tm" in c] or [None])[0]
        if t_col and tm_col:
            clust_T = mg2.groupby("cluster_id")[t_col].mean()
            tm_target = feas.groupby("cluster_id")[tm_col].first()
            comp = pd.DataFrame({"ClusterMeanT_C": clust_T, "PCM_Tm_target": tm_target}).dropna()
            if not comp.empty:
                fig, ax = plt.subplots(figsize=(9, 6))
                for cid, row in comp.iterrows():
                    ax.scatter(row["ClusterMeanT_C"], row["PCM_Tm_target"], color=PAL[int(cid) % len(PAL)], s=150, zorder=3, label=f"Cluster {cid}")
                    ax.annotate(f"C{cid}", (row["ClusterMeanT_C"], row["PCM_Tm_target"]), textcoords="offset points", xytext=(5, 5), fontsize=9)
                xlim = ax.get_xlim()
                ax.plot(xlim, [x + 25 for x in xlim], "r--", lw=1, label="+25C offset")
                ax.plot(xlim, [x + 35 for x in xlim], "g--", lw=1, label="+35C offset")
                ax.set(xlabel=f"Cluster Mean Temperature ({t_col}) (C)", ylabel="PCM Target Melting Point (C)", title="Comparison 2: Cluster Temperature vs PCM Tm Target (Assam)")
                ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("02_comparison_temp_vs_tm_target.png")

# ── Comparison 3: All MCDM Rankings Side-by-Side per Cluster ───────────
print("[3/8] MCDM method comparison: top 5 per cluster")
topk = load(TOPK, "topk")
if topk is not None:
    if "cluster_id" in topk.columns:
        topk = topk[topk["cluster_id"].isin(VALID_CLUSTERS)]
    topk = ensure_ranks(topk)
    methods = ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"]
    methods = [m for m in methods if m in topk.columns]
    name_col = "name" if "name" in topk.columns else ("PCM_Name" if "PCM_Name" in topk.columns else "name")
    if len(methods) >= 2 and "cluster_id" in topk.columns:
        clus_ids = sorted(topk["cluster_id"].unique())
        fig, axes = plt.subplots(len(clus_ids), 1, figsize=(13, 4.5 * len(clus_ids)), squeeze=False)
        for idx, cid in enumerate(clus_ids):
            sub = topk[topk["cluster_id"] == cid].sort_values("consensus_rank" if "consensus_rank" in methods else methods[0]).head(5)
            x = np.arange(len(sub)); w = 0.15; ax = axes[idx, 0]
            for mi, m in enumerate(methods):
                if m in sub.columns:
                    ax.bar(x + mi * w, sub[m].values, width=w, label=m.replace("_rank", "").upper(), color=sns.color_palette("Set2", len(methods))[mi], edgecolor="white")
            ax.set_xticks(x + (len(methods) - 1) * w / 2)
            ax.set_xticklabels(sub[name_col].tolist() if name_col in sub.columns else sub.index.astype(str), rotation=20, ha="right", fontsize=9)
            med = MEDOID_MAP.get(int(cid), f"C{cid}")
            ax.set(title=f"Cluster {cid} ({med}) - Historical K=4 Top 5 Ranks (Phase 10 Reference)", ylabel="Rank (lower=better)")
            ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
        plt.suptitle("Comparison 3: Historical K=4 MCDM Methods (Pre-Audit Reference for Phase 10)\n[Note: Current K=3 MCDM NOT PERFORMED (n_confirmed=[0,0,0]). Historical K=4 ranks shown for Phase 10 comparison.]", fontsize=11, fontweight="bold")
        plt.tight_layout(); sfig("03_comparison_mcdm_methods.png")

# Marker mapping per cluster: Cluster 0 -> circle, Cluster 1 -> square, Cluster 2 -> triangle
MARKERS = {0: 'o', 1: 's', 2: '^'}

# ── Comparison 4: Monte Carlo Top3-Prob vs Consensus Rank ────────────────
print("[4/8] Monte Carlo stability vs consensus rank")
mc = load(MC_CSV, "monte_carlo")
if mc is not None and "cluster_id" in mc.columns:
    mc = mc[mc["cluster_id"].isin(VALID_CLUSTERS)]
if mc is not None and "product_name" in mc.columns and "name" not in mc.columns:
    mc = mc.rename(columns={"product_name": "name"})
if mc is not None and topk is not None and "top3_inclusion_probability" in mc.columns and mc["top3_inclusion_probability"].notna().any():
    name_col = "name" if "name" in topk.columns else "PCM_Name"
    mc[name_col] = mc[name_col].astype(str)
    topk[name_col] = topk[name_col].astype(str)
    mg4 = mc.merge(topk[["cluster_id", name_col, "consensus_rank"]].drop_duplicates(subset=["cluster_id", name_col]), on=["cluster_id", name_col], how="inner").dropna(subset=["consensus_rank"])
    if not mg4.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        scale = 100 if mg4["top3_inclusion_probability"].max() <= 1.0 else 1
        for cid, g in mg4.groupby("cluster_id"):
            m = MARKERS.get(int(cid) % len(MARKERS), 'o')
            ax.scatter(g["consensus_rank"], g["top3_inclusion_probability"] * scale, color=PAL[int(cid) % len(PAL)], marker=m, s=100, alpha=0.85, edgecolors="black", lw=0.6, label=f"Cluster {cid}")
            for _, row in g.iterrows():
                ax.annotate(str(row[name_col]), (row["consensus_rank"], row["top3_inclusion_probability"] * scale), fontsize=6, alpha=0.7)
        ax.set(xlabel="Historical Consensus Rank (Pre-Audit)", ylabel="Top-3 Inclusion Probability (%)",
               title="Comparison 4: Historical K=4 Monte Carlo Stability (Pre-Audit Reference)\n[Note: Current K=3 Monte Carlo SKIPPED. Historical pre-audit data shown for Phase 10 audit.]")
        ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")
    elif topk is not None and "top3_inclusion_probability" in topk.columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        scale = 100 if topk["top3_inclusion_probability"].max() <= 1.0 else 1
        for cid, g in topk.groupby("cluster_id"):
            m = MARKERS.get(int(cid) % len(MARKERS), 'o')
            ax.scatter(g["consensus_rank"], g["top3_inclusion_probability"] * scale, color=PAL[int(cid) % len(PAL)], marker=m, s=100, alpha=0.85, edgecolors="black", lw=0.6, label=f"Cluster {cid}")
        ax.set(xlabel="Historical Consensus Rank (Pre-Audit)", ylabel="Top-3 Prob (%)",
               title="Comparison 4: Historical K=4 Monte Carlo Stability (Pre-Audit Reference)\n[Note: Current K=3 Monte Carlo SKIPPED. Historical pre-audit data shown for Phase 10 audit.]")
        ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")
    else:
        print("  skip MC comparison: no active Monte Carlo draws")
elif topk is not None and "top3_inclusion_probability" in topk.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    scale = 100 if topk["top3_inclusion_probability"].max() <= 1.0 else 1
    for cid, g in topk.groupby("cluster_id"):
        m = MARKERS.get(int(cid) % len(MARKERS), 'o')
        ax.scatter(g["consensus_rank"], g["top3_inclusion_probability"] * scale, color=PAL[int(cid) % len(PAL)], marker=m, s=100, alpha=0.85, edgecolors="black", lw=0.6, label=f"Cluster {cid}")
    ax.set(xlabel="Historical Consensus Rank (Pre-Audit)", ylabel="Top-3 Prob (%)",
           title="Comparison 4: Historical K=4 Monte Carlo Stability (Pre-Audit Reference)\n[Note: Current K=3 Monte Carlo SKIPPED. Historical pre-audit data shown for Phase 10 audit.]")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")

# ── Comparison 5: Latent Heat Distribution - Feasible vs All ─────────────
print("[5/8] Latent heat distribution comparison")
feas = load(FEAS, "feasibility")
db = load(PCM_DB, "pcm_db")
if feas is not None and "latent_heat_kJ_kg" in feas.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    if db is not None and "latent_heat_kJ_kg" in db.columns:
        ax.hist(db["latent_heat_kJ_kg"].dropna(), bins=40, alpha=0.5, color="gray", label=f"All candidates (n={len(db)})", density=True)
    ax.hist(feas["latent_heat_kJ_kg"].dropna(), bins=30, alpha=0.8, color="#3b7dd8", label=f"Feasible survivors (n={len(feas)})", density=True)
    ax.axvline(feas["latent_heat_kJ_kg"].median(), color="#3b7dd8", ls="--", lw=2, label=f"Feasible median: {feas['latent_heat_kJ_kg'].median():.0f} kJ/kg")
    ax.set(xlabel="Latent Heat (kJ/kg)", ylabel="Density", title="Comparison 5: Latent Heat Distribution - All vs Historical Screened Candidates (Assam)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("05_comparison_latent_heat_distribution.png")

# ── Comparison 6: Physics Validation - Solar Fraction & Hours Target Met vs MCDM Rank ────
print("[6/8] Physics validation vs MCDM rank")
phys = load(PHYS, "physics_val")
cmp_phys = load(CMP_PHYS, "mcdm_vs_physics")

if phys is not None:
    if "cluster_id" in phys.columns:
        phys = phys[phys["cluster_id"].isin(VALID_CLUSTERS)]
    
    h_col = "hours_Tw_ge_50C_per_year" if "hours_Tw_ge_50C_per_year" in phys.columns else ("hours_target_met_per_year" if "hours_target_met_per_year" in phys.columns else "hours_target_met")
    sf_col = "solar_fraction" if "solar_fraction" in phys.columns else ("annual_solar_fraction" if "annual_solar_fraction" in phys.columns else None)
    name_col = "pcm_name" if "pcm_name" in phys.columns else ("name" if "name" in phys.columns else "PCM_Name")

    if cmp_phys is not None and "historical_mcdm_rank" in cmp_phys.columns:
        cmp_name = "pcm_name" if "pcm_name" in cmp_phys.columns else "name"
        mg6 = phys.merge(
            cmp_phys[["cluster_id", cmp_name, "historical_mcdm_rank"]].drop_duplicates(),
            left_on=["cluster_id", name_col],
            right_on=["cluster_id", cmp_name],
            how="inner"
        )
        mg6["consensus_rank"] = mg6["historical_mcdm_rank"]
    elif topk is not None:
        topk_name = "name" if "name" in topk.columns else "PCM_Name"
        cols_to_merge = ["cluster_id", topk_name, "consensus_rank"]
        mg6 = phys.merge(
            topk[cols_to_merge].drop_duplicates(subset=["cluster_id", topk_name]),
            left_on=["cluster_id", name_col],
            right_on=["cluster_id", topk_name],
            how="inner"
        )
    else:
        mg6 = None

    if mg6 is not None and "consensus_rank" in mg6.columns and not mg6.empty:
        mg6 = mg6[mg6["cluster_id"].isin(VALID_CLUSTERS)]
        
        fig, axes = plt.subplots(1, 2 if sf_col else 1, figsize=(14 if sf_col else 9, 6))
        ax1 = axes[0] if sf_col else axes
        for cid in sorted(mg6["cluster_id"].unique()):
            g = mg6[mg6["cluster_id"] == cid]
            m = MARKERS.get(int(cid), 'o')
            med = MEDOID_MAP.get(int(cid), f"C{cid}")
            lbl = f"Cluster {cid} ({med})"
            v = g[["consensus_rank", h_col]].notna().all(axis=1)
            ax1.scatter(g.loc[v, "consensus_rank"], g.loc[v, h_col],
                        color=PAL[int(cid) % len(PAL)], marker=m, s=110, alpha=0.85,
                        edgecolors="black", lw=0.7, label=lbl)
            
        ax1.set_xticks(sorted(mg6["consensus_rank"].unique()))
        ax1.set(xlabel="Historical MCDM Consensus Rank (1 = Best)",
                ylabel="Hours Target Met per Year (Tw >= 50C)",
                title="Physics Validation: Hours Target Met vs Historical MCDM Rank")
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(alpha=0.25)
        
        # Annotate key materials for physical interpretability
        rank1_row = mg6[mg6["consensus_rank"] == 1]
        rank8_row = mg6[mg6["consensus_rank"] == mg6["consensus_rank"].max()]
        if not rank1_row.empty:
            r1_name = str(rank1_row.iloc[0][name_col]).split()[0]
            ax1.annotate(f"{r1_name} (Hist MCDM #1)", (1, rank1_row[h_col].mean()),
                         textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
        if not rank8_row.empty:
            max_r = int(rank8_row["consensus_rank"].iloc[0])
            ax1.annotate(f"savE OM48 (Hist MCDM #{max_r})", (max_r, rank8_row[h_col].mean()),
                         textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, fontweight="bold")
        
        if sf_col:
            ax2 = axes[1]
            scale = 100 if mg6[sf_col].max() <= 1.0 else 1
            for cid in sorted(mg6["cluster_id"].unique()):
                g = mg6[mg6["cluster_id"] == cid]
                m = MARKERS.get(int(cid), 'o')
                med = MEDOID_MAP.get(int(cid), f"C{cid}")
                lbl = f"Cluster {cid} ({med})"
                v = g[["consensus_rank", sf_col]].notna().all(axis=1)
                ax2.scatter(g.loc[v, "consensus_rank"], g.loc[v, sf_col] * scale,
                            color=PAL[int(cid) % len(PAL)], marker=m, s=110, alpha=0.85,
                            edgecolors="black", lw=0.7, label=lbl)
            ax2.set_xticks(sorted(mg6["consensus_rank"].unique()))
            ax2.set(xlabel="Historical MCDM Consensus Rank (1 = Best)",
                    ylabel="Annual Solar Thermal Fraction (%)",
                    title="Physics Validation: Solar Fraction (%) vs Historical MCDM Rank")
            ax2.legend(fontsize=9, loc="upper left")
            ax2.grid(alpha=0.25)
            
            if not rank1_row.empty:
                r1_name = str(rank1_row.iloc[0][name_col]).split()[0]
                ax2.annotate(f"{r1_name} (Hist MCDM #1)", (1, rank1_row[sf_col].mean() * scale),
                             textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, fontweight="bold")
            if not rank8_row.empty:
                max_r = int(rank8_row["consensus_rank"].iloc[0])
                ax2.annotate(f"savE OM48 (Hist MCDM #{max_r})", (max_r, rank8_row[sf_col].mean() * scale),
                             textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
        
        plt.suptitle("Comparison 6: Final K=3 Physics Validation vs Historical K=4 MCDM Ranking (Phase 10 Audit)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        sfig("06_comparison_physics_vs_rank.png")

# ── Comparison 7: Cross-Cluster Top PCM Key Properties ───────────────────
print("[7/8] Cross-cluster summary: top PCM properties")
if topk is not None and "consensus_rank" in topk.columns:
    top1 = topk[topk["consensus_rank"] == 1].drop_duplicates(subset="cluster_id").copy()
    name_col = "name" if "name" in top1.columns else "PCM_Name"
    props = [c for c in ["Tm_C", "latent_heat_kJ_kg", "rho_H_MJ_m3", "TC_W_mK", "cycles_tested"] if c in top1.columns]
    if len(props) >= 2:
        fig, axes = plt.subplots(1, len(props), figsize=(4 * len(props), 5))
        if len(props) == 1: axes = [axes]
        for i, p in enumerate(props):
            axes[i].bar(top1["cluster_id"].astype(str), top1[p], color=[PAL[int(c) % len(PAL)] for c in top1["cluster_id"]], edgecolor="white")
            axes[i].set(title=p, xlabel="Cluster"); axes[i].grid(alpha=0.3, axis="y")
            for j, (_, row) in enumerate(top1.iterrows()):
                if pd.notna(row.get(p)) and name_col in top1.columns:
                    axes[i].text(j, row[p] * 1.01, str(row.get(name_col, ""))[:10], ha="center", fontsize=7, rotation=25)
        plt.suptitle("Comparison 7: Historical K=4 MCDM Rank #1 Candidate Properties (Phase 10 Reference)\n[Note: RT44HC was historical MCDM #1; Phase 10 proved it physically inferior (#8 delivery). Not a K=3 recommendation.]", fontsize=10, fontweight="bold")
        plt.tight_layout(); sfig("07_comparison_cross_cluster_top_pcm.png")

# ── Comparison 8: Weight Sensitivity ─────────────────────────────────────
print("[8/8] Rank sensitivity to weight perturbation")
if topk is not None:
    topk = ensure_ranks(topk)
    score_cols = [c for c in ["topsis_score", "gra_grade", "promethee_flow"] if c in topk.columns]
    name_col = "name" if "name" in topk.columns else "PCM_Name"
    if len(score_cols) >= 2:
        c1, c2 = score_cols[0], score_cols[1]
        results = []
        for w1 in [0.3, 0.5, 0.7]:
            w2 = 1 - w1
            for cid in topk.groupby("cluster_id"):
                cid_val = cid[0]
                g = cid[1]
                v = g[[c1, c2]].notna().all(axis=1)
                if v.sum() > 0:
                    comb = w1 * g.loc[v, c1] / max(1, g.loc[v, c1].max()) + w2 * g.loc[v, c2] / max(1, g.loc[v, c2].max())
                    rk = comb.rank(ascending=False, method="min")
                    for idx2 in g.loc[v].index:
                        results.append({"w1": w1, "Cluster": cid_val, "Name": g.loc[idx2, name_col] if name_col in g.columns else str(idx2), "ComboRank": rk.loc[idx2]})
        rdf = pd.DataFrame(results)
        if not rdf.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_names = topk[topk.get("consensus_rank", pd.Series(dtype=float)) <= 3][name_col].unique() if "consensus_rank" in topk.columns and name_col in topk.columns else rdf["Name"].value_counts().head(6).index
            for nm in top_names[:6]:
                sub = rdf[rdf["Name"] == nm]
                if not sub.empty:
                    ax.plot(sub["w1"], sub["ComboRank"], "-o", label=str(nm)[:20], lw=1.5, ms=6)
            ax.set(xlabel=f"Weight on {c1.replace('_score','').replace('_grade','').upper()} (remainder on {c2.replace('_score','').replace('_grade','').upper()})",
                   ylabel="Historical Combined Rank",
                   title="Comparison 8: Historical K=4 MCDM Rank Sensitivity to Weight Perturbation (Pre-Audit Reference)\n[Note: Current K=3 MCDM NOT PERFORMED. Historical pre-audit sensitivity shown for Phase 10.]")
            ax.invert_yaxis(); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25); sfig("08_comparison_rank_sensitivity.png")

print("\nAll comparison plots saved to:", OUT)
