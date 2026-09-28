"""
Verification Script 4: MCDM Ranking Validation - Tamil Nadu
Output: data/plots/verify_ranking/
Run AFTER: 08_mcdm_ranking.py
"""
import os, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOPK    = os.path.join(BASE,"data","processed","pcm","mcdm_topk_by_cluster.csv")
FULL    = os.path.join(BASE,"data","processed","pcm","mcdm_full_scores_by_cluster.csv")
OUT     = os.path.join(BASE,"data","plots","verify_ranking")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
try:
    topk=pd.read_csv(TOPK); full=pd.read_csv(FULL); print(f"  topk: {topk.shape}  full: {full.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}"); raise SystemExit(1)

for sc,rc,asc in [("topsis_score","topsis_rank",False),("gra_grade","gra_rank",False),
                  ("promethee_flow","promethee_rank",False),("vikor_Q","vikor_rank",True),
                  ("borda_score","consensus_rank",False)]:
    if rc not in topk.columns and sc in topk.columns:
        topk[rc]=topk.groupby("cluster_id")[sc].rank(ascending=asc,method="min").astype(int)

rc=[c for c in ["topsis_rank","gra_rank","promethee_rank","vikor_rank","consensus_rank"] if c in topk.columns]
print(f"  Rank cols: {rc}")

# 1. Rank correlations heatmap
print("[1/6] Rank correlations")
fig,ax=plt.subplots(figsize=(10,8))
if len(rc)>=2:
    mt=np.eye(len(rc))
    for i,c1 in enumerate(rc):
        for j,c2 in enumerate(rc):
            if i<j:
                v=topk[c1].notna()&topk[c2].notna()
                if v.sum()>1:
                    r,_=spearmanr(topk.loc[v,c1],topk.loc[v,c2]); mt[i,j]=mt[j,i]=r
    labs=[c.replace("_rank","").upper() for c in rc]
    sns.heatmap(mt,annot=True,fmt=".2f",cmap="RdYlGn",center=0.7,xticklabels=labs,yticklabels=labs,ax=ax,vmin=-1,vmax=1,cbar_kws={"label":"Spearman rho"})
    ax.set_title("MCDM Method Rank Correlation - Tamil Nadu")
else:
    ax.text(0.5,0.5,"Insufficient rank methods",ha="center",va="center",transform=ax.transAxes,fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"01_method_correlation.png"),dpi=150); plt.close(); print("  01_method_correlation.png")

# 2. Top-3 inclusion probability
print("[2/6] Top-3 inclusion")
fig,ax=plt.subplots(figsize=(12,6))
if "name" in topk.columns and rc:
    inc={}
    for _,row in topk.iterrows():
        cand=row["name"]; n3=sum(1 for c in rc if pd.notna(row.get(c)) and row[c]<=3)
        inc[cand]=inc.get(cand,0)+n3
    if inc:
        prb={k:100*v/len(rc) for k,v in inc.items()}
        tc=sorted(prb.items(),key=lambda kv:kv[1],reverse=True)[:15]
        nm,vl=zip(*tc)
        bars=ax.barh(range(len(nm)),vl,color="steelblue",edgecolor="black")
        ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm,fontsize=10)
        ax.set(xlabel="Top-3 inclusion prob (%)",title="PCM Frequency in Top-3 - Tamil Nadu")
        ax.axvline(80,color="green",ls="--",lw=2,label="Good (>=80%)"); ax.axvline(50,color="orange",ls="--",lw=2,label="Fair (>=50%)")
        ax.legend(); ax.grid(alpha=0.3,axis="x")
        for i,b in enumerate(bars):
            w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,f"{w:.0f}%",ha="left",va="center",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"02_top3_inclusion_probability.png"),dpi=150); plt.close(); print("  02_top3_inclusion_probability.png")

# 3. Rank distributions
print("[3/6] Rank distributions")
fig,ax=plt.subplots(figsize=(10,6))
if len(rc)>=2:
    rows=[]
    for c in rc:
        for r in topk[c].dropna(): rows.append({"Method":c.replace("_rank","").upper(),"Rank":float(r)})
    rdf=pd.DataFrame(rows); sns.boxplot(data=rdf,x="Method",y="Rank",ax=ax,palette="Set2")
    ax.set(ylabel="Rank",title="Rank Distribution Across Methods - Tamil Nadu"); ax.invert_yaxis(); ax.grid(alpha=0.3,axis="y")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"03_rank_distributions.png"),dpi=150); plt.close(); print("  03_rank_distributions.png")

# 4. Rank reversal
print("[4/6] Rank reversal")
fig,ax=plt.subplots(figsize=(10,6))
if "name" in topk.columns and len(rc)>=2:
    data=[]
    for _,row in topk.iterrows():
        rnks=[row[c] for c in rc if pd.notna(row.get(c))]
        if len(rnks)>1: data.append({"Candidate":row["name"],"Spread":max(rnks)-min(rnks)})
    if data:
        rf=pd.DataFrame(data).sort_values("Spread",ascending=False).head(15)
        bars=ax.barh(range(len(rf)),rf["Spread"],color="coral",edgecolor="black")
        ax.set_yticks(range(len(rf))); ax.set_yticklabels(rf["Candidate"].tolist(),fontsize=10)
        ax.set(xlabel="Absolute rank spread",title="Rank Instability Across Methods - Tamil Nadu"); ax.grid(alpha=0.3,axis="x")
        for i,b in enumerate(bars):
            w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,str(int(w)),ha="left",va="center",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"04_rank_reversal_frequency.png"),dpi=150); plt.close(); print("  04_rank_reversal_frequency.png")

# 5. Agreement
print("[5/6] Agreement analysis")
fig,ax=plt.subplots(figsize=(10,8))
if "consensus_rank" in topk.columns and "topsis_rank" in topk.columns:
    v=topk[["topsis_rank","consensus_rank"]].notna().all(axis=1)
    if v.sum()>0:
        x=topk.loc[v,"topsis_rank"]; y=topk.loc[v,"consensus_rank"]
        sc=ax.scatter(x,y,c=topk.loc[v,"cluster_id"],cmap="tab10",s=90,alpha=0.7,edgecolors="black")
        mx=max(x.max(),y.max()); ax.plot([1,mx],[1,mx],"r--",label="Perfect agreement")
        ax.set(xlabel="TOPSIS rank",ylabel="Consensus rank",title="Consensus vs TOPSIS Rank - Tamil Nadu")
        ax.legend(); ax.grid(alpha=0.3); plt.colorbar(sc,ax=ax,label="Cluster")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"05_method_agreement.png"),dpi=150); plt.close(); print("  05_method_agreement.png")

# 6. Summary
print("[6/6] Summary")
fig,ax=plt.subplots(figsize=(12,8))
txt=f"MCDM RANKING VALIDATION SUMMARY (Tamil Nadu)\n{'='*60}\nMethods: {len(rc)}  Ranked: {len(topk)}  Clusters: {topk['cluster_id'].nunique()}\n\nSpearman rho:\n"
pairs=[("topsis_rank","gra_rank"),("topsis_rank","consensus_rank"),("gra_rank","consensus_rank"),("promethee_rank","consensus_rank"),("vikor_rank","consensus_rank")]
for c1,c2 in pairs:
    if c1 in topk.columns and c2 in topk.columns:
        v=topk[c1].notna()&topk[c2].notna(); r,_=spearmanr(topk.loc[v,c1],topk.loc[v,c2]) if v.sum()>1 else (float("nan"),None)
        txt+=f"  {c1.replace('_rank','').upper()} vs {c2.replace('_rank','').upper()}: {r:.3f}\n"
txt+="\nTop-3 consensus:\n"
if "name" in topk.columns and "consensus_rank" in topk.columns:
    t3=topk[topk["consensus_rank"]<=3][["name","consensus_rank"]].drop_duplicates().sort_values("consensus_rank").head(5)
    for _,row in t3.iterrows(): txt+=f"  {int(row['consensus_rank'])}. {row['name']}\n"
ax.text(0.05,0.95,txt,transform=ax.transAxes,va="top",ha="left",fontsize=11,family="monospace"); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"06_summary.png"),dpi=150); plt.close(); print("  06_summary.png")
print("\nRanking verification done. Plots:", OUT)
