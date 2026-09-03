"""
Verification Script 4: MCDM Ranking Validation - Rajasthan
Output: PLOTSV2/verify_ranking/
Run AFTER: 08_mcdm_ranking_rajasthan.py

Port of tamilnadu_pipeline/plots/verify_04_ranking_tamilnadu.py and
era5-uttarakhand/verify_04_ranking.py - same 6 plots, same style.

Correction against plotting/verify_04_ranking_rajasthan.py: that script keeps
`borda_score` in its list of "rank columns", so the correlation heatmap
compares a score against ranks (giving a spurious -1 row) and the top-3
inclusion count in plot 2 is divided by a method count that includes a column
where "<= 3" is almost never true. Here `borda_score` is used only to derive
`consensus_rank` and is then dropped from the method list.

Ranks are per-cluster, so correlations are computed within each cluster and
averaged (pooling ranks from a 9-candidate cluster with a 16-candidate one
would mix two different scales).
"""
import os, sys, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOPK    = os.path.join(BASE,"data","processed","mcdm_rankings_rajasthan.csv")
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)),"verify_ranking")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
try:
    topk=pd.read_csv(TOPK)
    print(f"  MCDM rankings: {topk.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}"); raise SystemExit(1)

# Consensus rank = Borda score ranked high-to-low within each cluster.
if "consensus_rank" not in topk.columns and "borda_score" in topk.columns:
    topk["consensus_rank"]=topk.groupby("cluster_id")["borda_score"].rank(ascending=False,method="min").astype(int)

rc=[c for c in ["TOPSIS_rank","GRA_rank","PROMETHEE_II_rank","VIKOR_rank","consensus_rank"] if c in topk.columns]
labs=[c.replace("_rank","").upper() for c in rc]
print(f"  Rank cols: {rc}")

def mean_within_cluster_rho(c1,c2):
    """Spearman rho between two rank columns, averaged over clusters."""
    vals=[]
    for _,g in topk.groupby("cluster_id"):
        v=g[c1].notna()&g[c2].notna()
        if v.sum()>1:
            r,_=spearmanr(g.loc[v,c1],g.loc[v,c2])
            if not np.isnan(r): vals.append(r)
    return float(np.mean(vals)) if vals else np.nan

# 1. Rank correlations heatmap
print("[1/6] Rank correlations...")
fig,ax=plt.subplots(figsize=(10,8))
if len(rc)>=2:
    mt=np.eye(len(rc))
    for i,c1 in enumerate(rc):
        for j,c2 in enumerate(rc):
            if i<j:
                r=mean_within_cluster_rho(c1,c2); mt[i,j]=mt[j,i]=r
    sns.heatmap(mt,annot=True,fmt=".2f",cmap="RdYlGn",center=0.0,xticklabels=labs,yticklabels=labs,ax=ax,vmin=-1,vmax=1,cbar_kws={"label":"Spearman rho"})
    ax.set_title("MCDM Method Rank Correlation - Rajasthan\n(mean of within-cluster rho)")
else:
    ax.text(0.5,0.5,"Insufficient rank methods",ha="center",va="center",transform=ax.transAxes,fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"01_method_correlation.png"),dpi=150); plt.close(); print("  01_method_correlation.png")

# 2. Top-3 inclusion probability
print("[2/6] Top-3 inclusion...")
fig,ax=plt.subplots(figsize=(12,6))
pcm_col = "pcm_id" if "pcm_id" in topk.columns else "name" if "name" in topk.columns else None

if pcm_col and rc:
    rows=[]
    for _,row in topk.iterrows():
        present=[c for c in rc if pd.notna(row.get(c))]
        if not present: continue
        n3=sum(1 for c in present if row[c]<=3)
        rows.append({"Candidate":f"{row[pcm_col]} (C{int(row['cluster_id'])})","Prob":100*n3/len(present)})
    if rows:
        tc=pd.DataFrame(rows).sort_values("Prob",ascending=False).head(15)
        bars=ax.barh(range(len(tc)),tc["Prob"].values,color="steelblue",edgecolor="black")
        ax.set_yticks(range(len(tc))); ax.set_yticklabels(tc["Candidate"].tolist(),fontsize=10)
        ax.set(xlabel="Share of MCDM methods placing it in the top 3 (%)",title="PCM Frequency in Top-3 - Rajasthan")
        ax.axvline(80,color="green",ls="--",lw=2,label="Good (>=80%)"); ax.axvline(50,color="orange",ls="--",lw=2,label="Fair (>=50%)")
        ax.legend(); ax.grid(alpha=0.3,axis="x"); ax.invert_yaxis()
        for b in bars:
            w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,f"{w:.0f}%",ha="left",va="center",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"02_top3_inclusion_probability.png"),dpi=150); plt.close(); print("  02_top3_inclusion_probability.png")

# 3. Rank distributions
print("[3/6] Rank distributions...")
fig,ax=plt.subplots(figsize=(10,6))
if len(rc)>=2:
    rows=[]
    for c in rc:
        for r in topk[c].dropna(): rows.append({"Method":c.replace("_rank","").upper(),"Rank":float(r)})
    rdf=pd.DataFrame(rows)
    sns.boxplot(data=rdf,x="Method",y="Rank",hue="Method",palette="Set2",legend=False,ax=ax)
    ax.set(ylabel="Rank (1=best)",title="Rank Distribution Across Methods - Rajasthan"); ax.invert_yaxis(); ax.grid(alpha=0.3,axis="y")
else:
    ax.text(0.5,0.5,"Insufficient data",ha="center",va="center",transform=ax.transAxes,fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"03_rank_distributions.png"),dpi=150); plt.close(); print("  03_rank_distributions.png")

# 4. Rank reversal
print("[4/6] Rank reversal...")
fig,ax=plt.subplots(figsize=(10,6))
if pcm_col and len(rc)>=2:
    data=[]
    for _,row in topk.iterrows():
        rnks=[row[c] for c in rc if pd.notna(row.get(c))]
        if len(rnks)>1: data.append({"Candidate":f"{row[pcm_col]} (C{int(row['cluster_id'])})","Spread":max(rnks)-min(rnks)})
    if data:
        rf=pd.DataFrame(data).sort_values("Spread",ascending=False).head(15)
        bars=ax.barh(range(len(rf)),rf["Spread"],color="coral",edgecolor="black")
        ax.set_yticks(range(len(rf))); ax.set_yticklabels(rf["Candidate"].tolist(),fontsize=10)
        ax.set(xlabel="Absolute rank spread (max-min across methods)",title="Rank Instability Across Methods - Rajasthan")
        ax.grid(alpha=0.3,axis="x"); ax.invert_yaxis()
        for b in bars:
            w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,str(int(w)),ha="left",va="center",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"04_rank_reversal_frequency.png"),dpi=150); plt.close(); print("  04_rank_reversal_frequency.png")

# 5. Agreement
print("[5/6] Agreement analysis...")
fig,ax=plt.subplots(figsize=(10,8))
topsis_col = "TOPSIS_rank" if "TOPSIS_rank" in topk.columns else None
consensus_col = "consensus_rank" if "consensus_rank" in topk.columns else None

if consensus_col and topsis_col:
    v=topk[[topsis_col,consensus_col]].notna().all(axis=1)
    if v.sum()>0:
        x=topk.loc[v,topsis_col]; y=topk.loc[v,consensus_col]
        sc=ax.scatter(x,y,c=topk.loc[v,"cluster_id"],cmap="tab10",s=90,alpha=0.7,edgecolors="black")
        mx=max(x.max(),y.max()); ax.plot([1,mx],[1,mx],"r--",label="Perfect agreement")
        ax.set(xlabel="TOPSIS rank",ylabel="Consensus (Borda) rank",title="Consensus vs TOPSIS Rank - Rajasthan")
        ax.legend(); ax.grid(alpha=0.3); plt.colorbar(sc,ax=ax,label="Cluster")
else:
    ax.text(0.5,0.5,"Required columns not found",ha="center",va="center",transform=ax.transAxes,fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"05_method_agreement.png"),dpi=150); plt.close(); print("  05_method_agreement.png")

# 6. Summary
print("[6/6] Summary report...")
fig,ax=plt.subplots(figsize=(12,8))
txt=(f"MCDM RANKING VALIDATION SUMMARY (Rajasthan)\n{'='*60}\n"
     f"Methods: {len(rc)}  Ranked candidates: {len(topk)}  Clusters: {topk['cluster_id'].nunique()}\n\n"
     f"Spearman rho (mean of within-cluster):\n")
pairs=[("TOPSIS_rank","GRA_rank"),("TOPSIS_rank","consensus_rank"),("GRA_rank","consensus_rank"),
       ("PROMETHEE_II_rank","consensus_rank"),("VIKOR_rank","consensus_rank")]
for c1,c2 in pairs:
    if c1 in topk.columns and c2 in topk.columns:
        r=mean_within_cluster_rho(c1,c2)
        txt+=f"  {c1.replace('_rank','').upper()} vs {c2.replace('_rank','').upper()}: {r:.3f}\n"

if "kendalls_w_cluster" in topk.columns:
    txt+="\nKendall's W (method concordance) per cluster:\n"
    for c,g in topk.groupby("cluster_id"):
        txt+=f"  Cluster {c}: {g['kendalls_w_cluster'].iloc[0]:.3f}\n"

txt+="\nTop-3 by consensus (Borda) rank:\n"
if pcm_col and "consensus_rank" in topk.columns:
    for c,g in topk.groupby("cluster_id"):
        t3=g[g["consensus_rank"]<=3].sort_values("consensus_rank")
        txt+=f"  Cluster {c}:\n"
        for _,row in t3.iterrows():
            txt+=f"    {int(row['consensus_rank'])}. {row[pcm_col]}\n"

ax.text(0.05,0.98,txt,transform=ax.transAxes,va="top",ha="left",fontsize=10,family="monospace"); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"06_summary.png"),dpi=150); plt.close(); print("  06_summary.png")

print("\nRanking verification done. Plots:", OUT)
