"""
Comparison Plots - Uttarakhand PCM Pipeline
============================================
Generates cross-step comparison plots to help verify results make sense.
Output: data/plots/comparison/

Plots:
  1. Cluster GHI profiles: mean GHI by cluster over months
  2. PCM temperature target vs cluster mean temperature
  3. All 4 MCDM method rankings side-by-side per cluster (top 5)
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

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CLUSTERS= os.path.join(BASE,"data","processed","clustering","cluster_assignments_uttarakhand.csv")
SIG_CSV = os.path.join(BASE,"data","processed","signatures","climate_signature_uttarakhand.csv")
FEAS    = os.path.join(BASE,"data","processed","pcm","feasibility_survivors_by_cluster.csv")
PCM_DB  = os.path.join(BASE,"data","processed","pcm","pcm_database_uttarakhand.csv")
TOPK    = os.path.join(BASE,"data","processed","pcm","mcdm_topk_by_cluster.csv")
MC_CSV  = os.path.join(BASE,"data","processed","pcm","monte_carlo_stability.csv")
PHYS    = os.path.join(BASE,"data","processed","pcm","physics_validation_results.csv")
CPROF   = os.path.join(BASE,"data","processed","clustering","cluster_profiles_uttarakhand.csv")
OUT     = os.path.join(BASE,"data","plots","comparison")
os.makedirs(OUT, exist_ok=True)

PAL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45"]

def load(p,label=""):
    if not os.path.exists(p): print(f"  skip {label}: not found"); return None
    return pd.read_csv(p)

def sfig(n):
    plt.savefig(os.path.join(OUT,n),dpi=150,bbox_inches="tight"); plt.close(); print(f"  {n}")

def ensure_ranks(df):
    for sc,rc,asc in [("topsis_score","topsis_rank",False),("gra_grade","gra_rank",False),
                      ("promethee_flow","promethee_rank",False),("vikor_Q","vikor_rank",True),
                      ("borda_score","consensus_rank",False)]:
        if rc not in df.columns and sc in df.columns:
            df[rc]=df.groupby("cluster_id")[sc].rank(ascending=asc,method="min").astype(int)
    return df

# ── Comparison 1: Cluster Mean GHI from Signature ────────────────────────
print("[1/8] Cluster GHI profiles from signature")
sig=load(SIG_CSV,"signature"); clu=load(CLUSTERS,"clusters")
if sig is not None and clu is not None:
    mg=sig.merge(clu[["point_id","cluster_id"]],on="point_id",how="inner")
    ghi_col="GHI_mean" if "GHI_mean" in mg.columns else ([c for c in mg.columns if "GHI" in c.upper()] or [None])[0]
    if ghi_col:
        fig,ax=plt.subplots(figsize=(9,5))
        for cid,g in mg.groupby("cluster_id"):
            ax.bar(str(cid),g[ghi_col].mean(),color=PAL[int(cid)%len(PAL)],edgecolor="white",lw=1,alpha=0.9,label=f"Cluster {cid}")
            ax.errorbar(str(cid),g[ghi_col].mean(),yerr=g[ghi_col].std(),fmt="none",color="black",capsize=5,lw=1.5)
        ax.set(xlabel="Cluster",ylabel=f"{ghi_col} (mean +/- std)",title="Comparison 1: Mean GHI by Climate Regime (Uttarakhand)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3,axis="y"); sfig("01_comparison_cluster_ghi.png")
    else: print("  No GHI column in signature")

# ── Comparison 2: PCM Tm_target vs Cluster Mean Temp ────────────────────
print("[2/8] PCM Tm_target vs cluster mean temperature")
if sig is not None and clu is not None:
    mg2=sig.merge(clu[["point_id","cluster_id"]],on="point_id",how="inner")
    t_col="Ta_mean_proxy" if "Ta_mean_proxy" in mg2.columns else ([c for c in mg2.columns if "T_" in c or "temp" in c.lower()] or [None])[0]
    feas=load(FEAS,"feasibility")
    if t_col and feas is not None and "Tm_target_C" in feas.columns:
        clust_T=mg2.groupby("cluster_id")[t_col].mean()
        tm_target=feas.groupby("cluster_id")["Tm_target_C"].first()
        comp=pd.DataFrame({"ClusterMeanT_C":clust_T,"PCM_Tm_target":tm_target}).dropna()
        fig,ax=plt.subplots(figsize=(9,6))
        for cid,row in comp.iterrows():
            ax.scatter(row["ClusterMeanT_C"],row["PCM_Tm_target"],color=PAL[int(cid)%len(PAL)],s=150,zorder=3,label=f"Cluster {cid}")
            ax.annotate(f"C{cid}",(row["ClusterMeanT_C"],row["PCM_Tm_target"]),textcoords="offset points",xytext=(5,5),fontsize=9)
        xlim=ax.get_xlim(); ax.plot(xlim,[x+25 for x in xlim],"r--",lw=1,label="+25C offset"); ax.plot(xlim,[x+35 for x in xlim],"g--",lw=1,label="+35C offset")
        ax.set(xlabel=f"Cluster Mean Temperature ({t_col}) (C)",ylabel="PCM Target Melting Point (C)",title="Comparison 2: Cluster Temperature vs PCM Tm Target (Uttarakhand)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("02_comparison_temp_vs_tm_target.png")

# ── Comparison 3: All 4 MCDM Rankings Side-by-Side per Cluster ───────────
print("[3/8] MCDM method comparison: top 5 per cluster")
topk=load(TOPK,"topk")
if topk is not None:
    topk=ensure_ranks(topk)
    methods=["topsis_rank","gra_rank","promethee_rank","vikor_rank","consensus_rank"]
    methods=[m for m in methods if m in topk.columns]
    if len(methods)>=2:
        clus_ids=sorted(topk["cluster_id"].unique())
        fig,axes=plt.subplots(len(clus_ids),1,figsize=(13,4.5*len(clus_ids)),squeeze=False)
        for idx,cid in enumerate(clus_ids):
            sub=topk[topk["cluster_id"]==cid].sort_values("consensus_rank" if "consensus_rank" in methods else methods[0]).head(5)
            x=np.arange(len(sub)); w=0.15; ax=axes[idx,0]
            for mi,m in enumerate(methods):
                if m in sub.columns:
                    ax.bar(x+mi*w,sub[m].values,width=w,label=m.replace("_rank","").upper(),color=sns.color_palette("Set2",len(methods))[mi],edgecolor="white")
            ax.set_xticks(x+(len(methods)-1)*w/2); ax.set_xticklabels(sub["name"].tolist(),rotation=20,ha="right",fontsize=9)
            ax.set(title=f"Cluster {cid} - Top 5 PCM Ranks by Method",ylabel="Rank (lower=better)"); ax.legend(fontsize=8); ax.grid(alpha=0.25,axis="y")
        plt.suptitle("Comparison 3: MCDM Methods Side-by-Side (Uttarakhand)",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("03_comparison_mcdm_methods.png")

# ── Comparison 4: Monte Carlo Top3-Prob vs Consensus Rank ────────────────
print("[4/8] Monte Carlo stability vs consensus rank")
mc=load(MC_CSV,"monte_carlo")
if mc is not None and topk is not None and "top3_inclusion_probability" in mc.columns:
    mg4=mc.merge(topk[["cluster_id","name","consensus_rank"]].drop_duplicates(subset=["cluster_id","name"]),on=["cluster_id","name"],how="inner").dropna(subset=["consensus_rank"])
    fig,ax=plt.subplots(figsize=(10,7))
    scale=100 if mg4["top3_inclusion_probability"].max()<=1.0 else 1
    for cid,g in mg4.groupby("cluster_id"):
        ax.scatter(g["consensus_rank"],g["top3_inclusion_probability"]*scale,color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,edgecolors="white",lw=0.5,label=f"Cluster {cid}")
        for _,row in g.iterrows():
            ax.annotate(row["name"],(row["consensus_rank"],row["top3_inclusion_probability"]*scale),fontsize=6,alpha=0.7)
    ax.set(xlabel="Consensus Rank",ylabel="Top-3 Inclusion Probability (%)",title="Comparison 4: Monte Carlo Stability vs MCDM Consensus Rank (Uttarakhand)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")
elif topk is not None and "top3_inclusion_probability" in topk.columns:
    fig,ax=plt.subplots(figsize=(10,7))
    scale=100 if topk["top3_inclusion_probability"].max()<=1.0 else 1
    for cid,g in topk.groupby("cluster_id"):
        ax.scatter(g["consensus_rank"],g["top3_inclusion_probability"]*scale,color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,label=f"Cluster {cid}")
    ax.set(xlabel="Consensus Rank",ylabel="Top-3 Prob (%)",title="Comparison 4: Monte Carlo Stability vs Consensus Rank (Uttarakhand)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")

# ── Comparison 5: Latent heat distribution - feasible vs all ─────────────
print("[5/8] Latent heat distribution comparison")
feas=load(FEAS,"feasibility"); db=load(PCM_DB,"pcm_db")
if feas is not None and "latent_heat_kJ_kg" in feas.columns:
    fig,ax=plt.subplots(figsize=(10,6))
    if db is not None and "latent_heat_kJ_kg" in db.columns:
        ax.hist(db["latent_heat_kJ_kg"].dropna(),bins=40,alpha=0.5,color="gray",label=f"All candidates (n={len(db)})",density=True)
    ax.hist(feas["latent_heat_kJ_kg"].dropna(),bins=30,alpha=0.8,color="#3b7dd8",label=f"Feasible survivors (n={len(feas)})",density=True)
    ax.axvline(feas["latent_heat_kJ_kg"].median(),color="#3b7dd8",ls="--",lw=2,label=f"Feasible median: {feas['latent_heat_kJ_kg'].median():.0f} kJ/kg")
    ax.set(xlabel="Latent Heat (kJ/kg)",ylabel="Density",title="Comparison 5: Latent Heat Distribution - All vs Feasible Survivors (Uttarakhand)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("05_comparison_latent_heat_distribution.png")

# ── Comparison 6: Physics validation - hours_met vs MCDM rank ────────────
print("[6/8] Physics validation vs MCDM rank")
phys=load(PHYS,"physics_val")
if phys is not None and topk is not None and "hours_target_met_per_year" in phys.columns:
    mg6=topk.merge(phys[["cluster_id","name","hours_target_met_per_year","complete_cycles_per_year"]].drop_duplicates(subset=["cluster_id","name"]),on=["cluster_id","name"],how="inner")
    if "consensus_rank" in mg6.columns and not mg6.empty:
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for cid,g in mg6.groupby("cluster_id"):
            v=g[["consensus_rank","hours_target_met_per_year"]].notna().all(axis=1)
            axes[0].scatter(g.loc[v,"consensus_rank"],g.loc[v,"hours_target_met_per_year"],color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,label=f"Cluster {cid}")
        axes[0].set(xlabel="Consensus Rank",ylabel="Hours Target Met per Year",title="MCDM Rank vs Hours Target Met")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)
        if "complete_cycles_per_year" in mg6.columns:
            for cid,g in mg6.groupby("cluster_id"):
                v=g[["consensus_rank","complete_cycles_per_year"]].notna().all(axis=1)
                axes[1].scatter(g.loc[v,"consensus_rank"],g.loc[v,"complete_cycles_per_year"],color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,label=f"Cluster {cid}")
            axes[1].set(xlabel="Consensus Rank",ylabel="Complete Cycles per Year",title="MCDM Rank vs Complete Cycles")
            axes[1].legend(fontsize=9); axes[1].grid(alpha=0.25)
        plt.suptitle("Comparison 6: Physics Validation vs MCDM Ranking (Uttarakhand)",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("06_comparison_physics_vs_rank.png")

# ── Comparison 7: Cross-cluster top PCM key properties ───────────────────
print("[7/8] Cross-cluster summary: top PCM properties")
if topk is not None and "consensus_rank" in topk.columns:
    top1=topk[topk["consensus_rank"]==1].drop_duplicates(subset="cluster_id").copy()
    props=[c for c in ["Tm_C","latent_heat_kJ_kg","rho_H_MJ_m3","TC_W_mK","cycles_tested"] if c in top1.columns]
    if len(props)>=2:
        fig,axes=plt.subplots(1,len(props),figsize=(4*len(props),5))
        if len(props)==1: axes=[axes]
        for i,p in enumerate(props):
            axes[i].bar(top1["cluster_id"].astype(str),top1[p],color=[PAL[int(c)%len(PAL)] for c in top1["cluster_id"]],edgecolor="white")
            axes[i].set(title=p,xlabel="Cluster"); axes[i].grid(alpha=0.3,axis="y")
            for j,(_,row) in enumerate(top1.iterrows()):
                if pd.notna(row.get(p)):
                    axes[i].text(j,row[p]*1.01,row.get("name","")[:10],ha="center",fontsize=7,rotation=25)
        plt.suptitle("Comparison 7: Top PCM Properties per Cluster (Uttarakhand) - Rank #1",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("07_comparison_cross_cluster_top_pcm.png")

# ── Comparison 8: Weight sensitivity ─────────────────────────────────────
print("[8/8] Rank sensitivity to weight perturbation")
if topk is not None:
    topk=ensure_ranks(topk)
    score_cols=[c for c in ["topsis_score","gra_grade","promethee_flow"] if c in topk.columns]
    if len(score_cols)>=2:
        # Simulate +/-20% weight shift on topsis vs gra
        c1,c2=score_cols[0],score_cols[1]
        results=[]
        for w1 in [0.3,0.5,0.7]:
            w2=1-w1
            for cid,g in topk.groupby("cluster_id"):
                v=g[[c1,c2]].notna().all(axis=1)
                if v.sum()>0:
                    comb=w1*g.loc[v,c1]/max(1,g.loc[v,c1].max())+w2*g.loc[v,c2]/max(1,g.loc[v,c2].max())
                    rk=comb.rank(ascending=False,method="min")
                    for idx2 in g.loc[v].index:
                        results.append({"w1":w1,"Cluster":cid,"Name":g.loc[idx2,"name"] if "name" in g.columns else str(idx2),"ComboRank":rk.loc[idx2]})
        rdf=pd.DataFrame(results)
        if not rdf.empty:
            fig,ax=plt.subplots(figsize=(12,6))
            top_names=topk[topk.get("consensus_rank",pd.Series(dtype=float))<=3]["name"].unique() if "consensus_rank" in topk.columns and "name" in topk.columns else rdf["Name"].value_counts().head(6).index
            for nm in top_names[:6]:
                sub=rdf[rdf["Name"]==nm]
                if not sub.empty:
                    ax.plot(sub["w1"],sub["ComboRank"],"-o",label=nm[:20],lw=1.5,ms=6)
            ax.set(xlabel=f"Weight on {c1.replace('_score','').replace('_grade','').upper()} (remainder on {c2.replace('_score','').replace('_grade','').upper()})",
                   ylabel="Combined Rank",title="Comparison 8: Rank Sensitivity to Weight Perturbation (Uttarakhand)")
            ax.invert_yaxis(); ax.legend(fontsize=8,loc="upper right"); ax.grid(alpha=0.25); sfig("08_comparison_rank_sensitivity.png")
    else: print("  Insufficient score columns for sensitivity")

print("\nAll comparison plots saved to:", OUT)
