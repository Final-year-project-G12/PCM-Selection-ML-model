"""
Comparison Plots - Rajasthan PCM Pipeline (PLOTSV2)
===================================================
Generates cross-step comparison plots to help verify results make sense.
Output: PLOTSV2/comparison_plots/

Direct port of era5-tamilnadu/plots/comparison_plots_tamilnadu.py - same 8
comparisons, same figure sizes, same palette, same filenames. Only the input
schema differs, so everything Rajasthan-specific is isolated in the SCHEMA
ADAPTER block below (Rajasthan names its columns pcm_id / TOPSIS_rank where
Tamil Nadu names them name / topsis_rank).

Two comparisons needed real adaptation, not just renaming:
  - Comparison 7 reads PCM properties (Tm_C, latent_heat_kJ_kg, cycles_tested)
    from the feasibility table; Rajasthan's MCDM table carries ranks only.
  - Comparison 8 perturbs method *ranks*, because Rajasthan's MCDM output
    stores an integer rank per method and a Borda score - it has no
    topsis_score / gra_grade / promethee_flow columns to blend.

Plots:
  1. Cluster GHI profiles: mean GHI by cluster
  2. PCM temperature target vs cluster mean temperature
  3. All 4 MCDM method rankings side-by-side per cluster (top 5)
  4. Monte Carlo stability: top3 prob vs consensus rank scatter
  5. Latent heat distribution: feasible survivors vs all candidates
  6. Physics validation: annual solar fraction / hours met vs MCDM rank
  7. Cross-cluster summary: key properties of top PCM per cluster
  8. Sensitivity: how rank changes as weight shifts between two methods
"""
import os, sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# PCM product names carry non-ASCII characters (e.g. "savE(R) OM42"), which
# blow up on the cp1252 Windows console. Keep prints safe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# Level A = one row per grid point, which is what the signature merges against.
# (Level B is seasonal and would duplicate every point once per season.)
CLUSTERS= os.path.join(BASE,"data","processed","cluster_assignments_rajasthan_levelA.csv")
SIG_CSV = os.path.join(BASE,"data","processed","climate_signature_rajasthan.csv")
FEAS    = os.path.join(BASE,"data","processed","feasibility_survivors_rajasthan_kappa_calibrated.csv")
PCM_DB  = os.path.join(BASE,"..","PCM_data","data","PCM_Properties_cleaned_mice_pmm_detailed.csv")
TOPK    = os.path.join(BASE,"data","processed","mcdm_rankings_rajasthan.csv")
MC_CSV  = os.path.join(BASE,"data","processed","mcdm_rankings_rajasthan.csv")
PHYS    = os.path.join(BASE,"data","processed","physics_validation_rajasthan.csv")
CPROF   = os.path.join(BASE,"data","processed","cluster_profiles_rajasthan.csv")
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)),"comparison_plots")
os.makedirs(OUT, exist_ok=True)

PAL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45"]

# ---------------------------------------------------------------- SCHEMA ADAPTER
# Rajasthan -> canonical (Tamil Nadu) column names.
RENAME = {
    "pcm_id": "name",
    "TOPSIS_rank": "topsis_rank",
    "GRA_rank": "gra_rank",
    "PROMETHEE_II_rank": "promethee_rank",
    "VIKOR_rank": "vikor_rank",
    "mc_top3_inclusion_pct": "top3_inclusion_probability",
    "latent_heat_melting": "latent_heat_kJ_kg",
}

def load(p,label=""):
    if not os.path.exists(p):
        print(f"  skip {label}: not found at {p}"); return None
    df = pd.read_csv(p)
    return df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns})

def sfig(n):
    plt.savefig(os.path.join(OUT,n),dpi=150,bbox_inches="tight"); plt.close(); print(f"  {n}")

def ensure_ranks(df):
    """Rajasthan ships an integer rank per method plus a Borda score; only the
    consensus rank has to be derived (higher Borda = better = rank 1)."""
    for sc,rc,asc in [("topsis_score","topsis_rank",False),("gra_grade","gra_rank",False),
                      ("promethee_flow","promethee_rank",False),("vikor_Q","vikor_rank",True),
                      ("borda_score","consensus_rank",False)]:
        if rc not in df.columns and sc in df.columns:
            df[rc]=df.groupby("cluster_id")[sc].rank(ascending=asc,method="min").astype(int)
    return df

print("\n" + "="*70)
print("RAJASTHAN COMPARISON PLOTS (PLOTSV2)")
print("="*70 + "\n")

# ── Comparison 1: Cluster Mean GHI from Signature ────────────────────────
print("[1/8] Cluster GHI profiles from signature")
sig=load(SIG_CSV,"signature"); clu=load(CLUSTERS,"clusters")
if sig is not None and clu is not None:
    mg=sig.merge(clu[["point_id","cluster_id"]],on="point_id",how="inner")
    ghi_col="GHI_daily_kWh" if "GHI_daily_kWh" in mg.columns else ([c for c in mg.columns if "GHI" in c.upper()] or [None])[0]
    if ghi_col:
        fig,ax=plt.subplots(figsize=(9,5))
        for cid,g in mg.groupby("cluster_id"):
            ax.bar(str(cid),g[ghi_col].mean(),color=PAL[int(cid)%len(PAL)],edgecolor="white",lw=1,alpha=0.9,label=f"Cluster {cid}")
            ax.errorbar(str(cid),g[ghi_col].mean(),yerr=g[ghi_col].std(),fmt="none",color="black",capsize=5,lw=1.5)
        ax.set(xlabel="Cluster",ylabel=f"{ghi_col} (mean +/- std)",title="Comparison 1: Mean GHI by Climate Regime (Rajasthan)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3,axis="y"); sfig("01_comparison_cluster_ghi.png")
    else: print("  No GHI column in signature")

# ── Comparison 2: PCM Tm_target vs Cluster Mean Temp ────────────────────
print("[2/8] PCM Tm_target vs cluster mean temperature")
if sig is not None and clu is not None:
    mg2=sig.merge(clu[["point_id","cluster_id"]],on="point_id",how="inner")
    t_col="Ta_mean" if "Ta_mean" in mg2.columns else ([c for c in mg2.columns if "T_" in c or "temp" in c.lower()] or [None])[0]
    # Rajasthan carries Tm_target_C on the signature (per point), not on the
    # feasibility table the way Tamil Nadu does.
    tm_col="Tm_target_capped_C" if "Tm_target_capped_C" in mg2.columns else ("Tm_target_C" if "Tm_target_C" in mg2.columns else None)
    if t_col and tm_col:
        comp=pd.DataFrame({"ClusterMeanT_C":mg2.groupby("cluster_id")[t_col].mean(),
                           "PCM_Tm_target":mg2.groupby("cluster_id")[tm_col].mean()}).dropna()
        fig,ax=plt.subplots(figsize=(9,6))
        for cid,row in comp.iterrows():
            ax.scatter(row["ClusterMeanT_C"],row["PCM_Tm_target"],color=PAL[int(cid)%len(PAL)],s=150,zorder=3,label=f"Cluster {cid}")
            ax.annotate(f"C{cid}",(row["ClusterMeanT_C"],row["PCM_Tm_target"]),textcoords="offset points",xytext=(5,5),fontsize=9)
        xlim=ax.get_xlim(); ax.plot(xlim,[x+25 for x in xlim],"r--",lw=1,label="+25C offset"); ax.plot(xlim,[x+35 for x in xlim],"g--",lw=1,label="+35C offset")
        ax.set(xlabel=f"Cluster Mean Temperature ({t_col}) (C)",ylabel=f"PCM Target Melting Point ({tm_col}) (C)",title="Comparison 2: Cluster Temperature vs PCM Tm Target (Rajasthan)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("02_comparison_temp_vs_tm_target.png")
    else: print("  Missing temperature or Tm_target column in signature")

# ── Comparison 3: All 4 MCDM Rankings Side-by-Side per Cluster ───────────
print("[3/8] MCDM method comparison: top 5 per cluster")
topk=load(TOPK,"topk")
if topk is not None:
    topk=ensure_ranks(topk)
    methods=[m for m in ["topsis_rank","gra_rank","promethee_rank","vikor_rank","consensus_rank"] if m in topk.columns]
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
        plt.suptitle("Comparison 3: MCDM Methods Side-by-Side (Rajasthan)",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("03_comparison_mcdm_methods.png")

# ── Comparison 4: Monte Carlo Top3-Prob vs Consensus Rank ────────────────
print("[4/8] Monte Carlo stability vs consensus rank")
mc=load(MC_CSV,"monte_carlo")
if mc is not None: mc=ensure_ranks(mc)
if mc is not None and topk is not None and "top3_inclusion_probability" in mc.columns and MC_CSV!=TOPK:
    mg4=mc.merge(topk[["cluster_id","name","consensus_rank"]].drop_duplicates(subset=["cluster_id","name"]),on=["cluster_id","name"],how="inner").dropna(subset=["consensus_rank"])
elif topk is not None and "top3_inclusion_probability" in topk.columns:
    # Rajasthan folds the Monte Carlo columns into the MCDM ranking table, so
    # there is nothing to merge - the stability numbers are already alongside
    # the consensus rank.
    mg4=topk.dropna(subset=["consensus_rank","top3_inclusion_probability"])
else:
    mg4=None
if mg4 is not None and not mg4.empty:
    fig,ax=plt.subplots(figsize=(10,7))
    scale=100 if mg4["top3_inclusion_probability"].max()<=1.0 else 1
    for cid,g in mg4.groupby("cluster_id"):
        ax.scatter(g["consensus_rank"],g["top3_inclusion_probability"]*scale,color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,edgecolors="white",lw=0.5,label=f"Cluster {cid}")
        for _,row in g.iterrows():
            ax.annotate(str(row["name"])[:18],(row["consensus_rank"],row["top3_inclusion_probability"]*scale),fontsize=6,alpha=0.7)
    ax.set(xlabel="Consensus Rank",ylabel="Top-3 Inclusion Probability (%)",title="Comparison 4: Monte Carlo Stability vs MCDM Consensus Rank (Rajasthan)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")

# ── Comparison 5: Latent heat distribution - feasible vs all ─────────────
print("[5/8] Latent heat distribution comparison")
feas=load(FEAS,"feasibility"); db=load(PCM_DB,"pcm_db")
if feas is not None and "latent_heat_kJ_kg" in feas.columns:
    # The kappa-calibrated table lists every candidate with a survives_all flag;
    # the "survivors" curve is that subset, not the whole file.
    surv=feas[feas["survives_all"]==True] if "survives_all" in feas.columns else feas
    fig,ax=plt.subplots(figsize=(10,6))
    if db is not None and "latent_heat_kJ_kg" in db.columns:
        ax.hist(db["latent_heat_kJ_kg"].dropna(),bins=40,alpha=0.5,color="gray",label=f"All candidates (n={len(db)})",density=True)
    if len(surv):
        ax.hist(surv["latent_heat_kJ_kg"].dropna(),bins=30,alpha=0.8,color="#3b7dd8",label=f"Feasible survivors (n={len(surv)})",density=True)
        ax.axvline(surv["latent_heat_kJ_kg"].median(),color="#3b7dd8",ls="--",lw=2,label=f"Feasible median: {surv['latent_heat_kJ_kg'].median():.0f} kJ/kg")
    ax.set(xlabel="Latent Heat (kJ/kg)",ylabel="Density",title="Comparison 5: Latent Heat Distribution - All vs Feasible Survivors (Rajasthan)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("05_comparison_latent_heat_distribution.png")

# ── Comparison 6: Physics validation vs MCDM rank ───────────────────────
print("[6/8] Physics validation vs MCDM rank")
phys=load(PHYS,"physics_val")
if phys is not None: phys=ensure_ranks(phys)
if phys is not None and "consensus_rank" in phys.columns:
    # Rajasthan's physics table already carries the MCDM columns, so it is
    # self-contained - no merge against topk needed.
    panels=[(c,l) for c,l in [("annual_solar_fraction","Annual Solar Fraction"),
                              ("hours_target_met_per_year","Hours Target Met per Year")] if c in phys.columns]
    if panels:
        fig,axes=plt.subplots(1,len(panels),figsize=(7*len(panels),6),squeeze=False)
        for i,(col,lab) in enumerate(panels):
            ax=axes[0,i]
            for cid,g in phys.groupby("cluster_id"):
                v=g[["consensus_rank",col]].notna().all(axis=1)
                ax.scatter(g.loc[v,"consensus_rank"],g.loc[v,col],color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,edgecolors="white",lw=0.5,label=f"Cluster {cid}")
            v=phys[["consensus_rank",col]].notna().all(axis=1)
            if v.sum()>2:
                rho,p=spearmanr(phys.loc[v,"consensus_rank"],phys.loc[v,col])
                ax.set_title(f"MCDM Rank vs {lab}\n(pooled Spearman rho={rho:+.3f}, p={p:.3f})")
            else:
                ax.set_title(f"MCDM Rank vs {lab}")
            ax.set(xlabel="Consensus Rank (lower=better)",ylabel=lab)
            ax.legend(fontsize=9); ax.grid(alpha=0.25)
        plt.suptitle("Comparison 6: Physics Validation vs MCDM Ranking (Rajasthan)",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("06_comparison_physics_vs_rank.png")

# ── Comparison 7: Cross-cluster top PCM key properties ───────────────────
print("[7/8] Cross-cluster summary: top PCM properties")
if topk is not None and "consensus_rank" in topk.columns:
    top1=topk[topk["consensus_rank"]==1].drop_duplicates(subset="cluster_id").copy()
    # Rajasthan's MCDM table holds ranks only; the thermophysical properties
    # live on the feasibility table, so pull them across.
    if feas is not None:
        prop_cols=[c for c in ["Tm_C","latent_heat_kJ_kg","cycles_tested","supercooling_K"] if c in feas.columns]
        if prop_cols:
            top1=top1.merge(feas[["cluster_id","name"]+prop_cols].drop_duplicates(subset=["cluster_id","name"]),
                            on=["cluster_id","name"],how="left")
    props=[c for c in ["Tm_C","latent_heat_kJ_kg","cycles_tested","supercooling_K"] if c in top1.columns and top1[c].notna().any()]
    if len(props)>=2:
        fig,axes=plt.subplots(1,len(props),figsize=(4*len(props),5),squeeze=False)
        for i,p in enumerate(props):
            ax=axes[0,i]
            ax.bar(top1["cluster_id"].astype(str),top1[p],color=[PAL[int(c)%len(PAL)] for c in top1["cluster_id"]],edgecolor="white")
            ax.set(title=p,xlabel="Cluster"); ax.grid(alpha=0.3,axis="y")
            # Headroom for the PCM labels, so they never ride up into the title.
            lo,hi=ax.get_ylim(); pad=(hi-lo)*0.14; ax.set_ylim(lo-(pad if lo<0 else 0),hi+pad)
            off=(ax.get_ylim()[1]-ax.get_ylim()[0])*0.02
            for j,(_,row) in enumerate(top1.iterrows()):
                if pd.notna(row.get(p)):
                    # Negative bars (e.g. supercooling_K) label below the bar.
                    y=row[p]-off*2 if row[p]<0 else row[p]+off
                    va="top" if row[p]<0 else "bottom"
                    ax.text(j,y,str(row.get("name",""))[:10],ha="center",va=va,fontsize=7,rotation=25)
        plt.suptitle("Comparison 7: Top PCM Properties per Cluster (Rajasthan) - Rank #1",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("07_comparison_cross_cluster_top_pcm.png")
    else: print("  Insufficient property columns for cross-cluster summary")

# ── Comparison 8: Weight sensitivity ─────────────────────────────────────
print("[8/8] Rank sensitivity to weight perturbation")
if topk is not None:
    topk=ensure_ranks(topk)
    # Tamil Nadu blends two MCDM *scores*. Rajasthan stores ranks, so blend the
    # ranks instead: normalise each to [0,1] within its cluster (0 = best) and
    # take a weighted sum. Same question, same shape of answer.
    rank_cols=[c for c in ["topsis_rank","gra_rank","promethee_rank","vikor_rank"] if c in topk.columns]
    if len(rank_cols)>=2:
        c1,c2=rank_cols[0],rank_cols[1]
        results=[]
        for w1 in [0.3,0.5,0.7]:
            w2=1-w1
            for cid,g in topk.groupby("cluster_id"):
                v=g[[c1,c2]].notna().all(axis=1)
                if v.sum()>0:
                    n1=g.loc[v,c1]/max(1,g.loc[v,c1].max()); n2=g.loc[v,c2]/max(1,g.loc[v,c2].max())
                    rk=(w1*n1+w2*n2).rank(ascending=True,method="min")  # lower blended rank = better
                    for idx2 in g.loc[v].index:
                        results.append({"w1":w1,"Cluster":cid,"Name":g.loc[idx2,"name"] if "name" in g.columns else str(idx2),"ComboRank":rk.loc[idx2]})
        rdf=pd.DataFrame(results)
        if not rdf.empty:
            fig,ax=plt.subplots(figsize=(12,6))
            top_names=topk[topk["consensus_rank"]<=3]["name"].unique() if "consensus_rank" in topk.columns and "name" in topk.columns else rdf["Name"].value_counts().head(6).index
            for nm in list(top_names)[:6]:
                sub=rdf[rdf["Name"]==nm]
                if not sub.empty:
                    ax.plot(sub["w1"],sub["ComboRank"],"-o",label=str(nm)[:20],lw=1.5,ms=6)
            ax.set(xlabel=f"Weight on {c1.replace('_rank','').upper()} (remainder on {c2.replace('_rank','').upper()})",
                   ylabel="Combined Rank",title="Comparison 8: Rank Sensitivity to Weight Perturbation (Rajasthan)")
            ax.invert_yaxis(); ax.legend(fontsize=8,loc="upper right"); ax.grid(alpha=0.25); sfig("08_comparison_rank_sensitivity.png")
    else: print("  Insufficient rank columns for sensitivity")

print("\nAll comparison plots saved to:", OUT)
