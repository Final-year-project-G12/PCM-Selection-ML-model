"""
Verification Script 3: PCM Feasibility - Tamil Nadu
Output: data/plots/verify_feasibility/
Run AFTER: 07_feasibility_filter.py
"""
import os, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALL_CSV = os.path.join(BASE,"data","processed","pcm","pcm_database_tamilnadu.csv")
SUR_CSV = os.path.join(BASE,"data","processed","pcm","feasibility_survivors_by_cluster.csv")
OUT     = os.path.join(BASE,"data","plots","verify_feasibility")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
try:
    sur=pd.read_csv(SUR_CSV); print(f"  Survivors: {sur.shape}")
except FileNotFoundError:
    print("ERROR: run 07_feasibility_filter.py first"); raise SystemExit(1)
try:
    alc=pd.read_csv(ALL_CSV); hf=True; print(f"  Full DB: {alc.shape}")
except FileNotFoundError:
    hf=False; print("  Full DB not found, skipping ratios")

PAL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]

# 1. Survival rate per cluster
print("[1/6] Survival rates")
fig,ax=plt.subplots(figsize=(10,6)); cids=sorted(sur["cluster_id"].unique())
if hf and "cluster_id" in alc.columns:
    sp=[100*len(sur[sur["cluster_id"]==c])/max(1,len(alc[alc["cluster_id"]==c])) for c in cids]
    bars=ax.bar(cids,sp,color=[PAL[int(c)%len(PAL)] for c in cids],edgecolor="black"); ax.set_ylabel("Survival rate (%)")
    ax.axhline(10,color="green",ls="--",lw=2); ax.axhline(50,color="green",ls="--",lw=2)
else:
    cnt=sur.groupby("cluster_id").size()
    bars=ax.bar(cnt.index,cnt.values,color=[PAL[int(c)%len(PAL)] for c in cnt.index],edgecolor="black"); ax.set_ylabel("Survivors")
for b in bars:
    h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h,f"{h:.1f}",ha="center",va="bottom",fontsize=10)
ax.set(xlabel="Cluster ID",title="PCM Survival Rate per Climate Regime - Tamil Nadu"); ax.grid(alpha=0.3,axis="y")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"01_survival_rate_by_cluster.png"),dpi=150); plt.close(); print("  01_survival_rate_by_cluster.png")

# 2. Property space
print("[2/6] Feasible property space")
fig,ax=plt.subplots(figsize=(10,8))
if {"Tm_C","latent_heat_kJ_kg"}.issubset(sur.columns):
    sc=ax.scatter(sur["Tm_C"],sur["latent_heat_kJ_kg"],c=sur["cluster_id"],cmap="tab10",s=100,alpha=0.6,edgecolors="black",lw=0.5)
    ax.set(xlabel="Melting Point (C)",ylabel="Latent Heat (kJ/kg)",title="Feasible PCM Property Space - Tamil Nadu")
    ax.grid(alpha=0.3)
    if {"window_lo","window_hi"}.issubset(sur.columns):
        lo=sur["window_lo"].dropna().min(); hi=sur["window_hi"].dropna().max()
        ax.axvspan(lo,hi,alpha=0.12,color="green",label=f"Accepted Tm band: {lo:.0f}-{hi:.0f}C")
    ax.axhline(100,color="gray",ls=":",alpha=0.6,label="Latent heat floor"); ax.legend(); plt.colorbar(sc,ax=ax,label="Cluster")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"02_feasible_property_space.png"),dpi=150); plt.close(); print("  02_feasible_property_space.png")

# 3. Top candidates per cluster
print("[3/6] Top candidates per cluster")
fig,ax=plt.subplots(figsize=(12,6))
if "latent_heat_kJ_kg" in sur.columns and "name" in sur.columns:
    tp=sur.groupby("cluster_id").apply(lambda x: x.nlargest(3,"latent_heat_kJ_kg") if len(x)>=3 else x).reset_index(drop=True)
    tc=tp["name"].value_counts().head(12)
    bars=ax.barh(range(len(tc)),tc.values,color="coral",edgecolor="black")
    ax.set_yticks(range(len(tc))); ax.set_yticklabels(tc.index,fontsize=10)
    ax.set(xlabel="Frequency in top-3 by cluster",title="Highest-Latent-Heat Survivors per Cluster - Tamil Nadu")
    ax.grid(alpha=0.3,axis="x")
    for i,b in enumerate(bars):
        w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,str(int(w)),ha="left",va="center",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"03_top_candidates_per_cluster.png"),dpi=150); plt.close(); print("  03_top_candidates_per_cluster.png")

# 4. Constraint analysis
print("[4/6] Constraint analysis")
cc=[c for c in sur.columns if "pass_" in c.lower()]
fig,ax=plt.subplots(figsize=(10,6))
if cc:
    sm=[]
    for c in cc:
        p=(sur[c]==True).sum() if pd.api.types.is_bool_dtype(sur[c]) else int(sur[c].sum())
        sm.append({"Constraint":c,"Pass":p,"Fail":len(sur)-p})
    if sm:
        pd.DataFrame(sm).set_index("Constraint")[["Pass","Fail"]].plot(kind="barh",stacked=True,ax=ax,color=["green","red"])
        ax.set(xlabel="Candidates",title="Constraint Pass/Fail - Tamil Nadu"); ax.grid(alpha=0.3,axis="x")
else:
    ax.text(0.5,0.5,"No constraint columns available",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"04_constraint_analysis.png"),dpi=150); plt.close(); print("  04_constraint_analysis.png")

# 5. Property distributions by cluster
print("[5/6] Property distributions")
pp=[p for p in ["Tm_C","latent_heat_kJ_kg","density_liquid_kg_m3","Cp_liquid_kJ_kgK"] if p in sur.columns][:3]
if pp:
    fig,axes=plt.subplots(1,len(pp),figsize=(5*len(pp),5))
    if len(pp)==1: axes=[axes]
    for i,p in enumerate(pp):
        sns.boxplot(data=sur,x="cluster_id",y=p,ax=axes[i],palette="Set2"); axes[i].set_title(f"{p} by Cluster"); axes[i].grid(alpha=0.3,axis="y")
    plt.suptitle("PCM Property Distributions (Tamil Nadu survivors)",fontsize=14,y=1.02); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"05_property_distributions.png"),dpi=150); plt.close(); print("  05_property_distributions.png")

# 6. Summary
print("[6/6] Summary")
fig,ax=plt.subplots(figsize=(10,6))
ts=len(sur); uc=sur["cluster_id"].nunique()
txt=f"PCM FEASIBILITY FILTERING SUMMARY (Tamil Nadu)\n{'='*50}\nTotal Survivors: {ts}\nClusters: {uc}\nAvg Survivors/Cluster: {ts/uc:.1f}\n\nBy Cluster:\n"
for c in sorted(sur["cluster_id"].unique()): txt+=f"  Cluster {c}: {len(sur[sur['cluster_id']==c])} PCMs\n"
if hf and "cluster_id" in alc.columns:
    osr=100*ts/len(alc); q="OK (10-50%)" if 10<=osr<=50 else ("TOO STRICT (<10%)" if osr<10 else "TOO LOOSE (>50%)")
    txt+=f"\nOverall survival: {osr:.1f}%  [{q}]"
ax.text(0.05,0.95,txt,transform=ax.transAxes,va="top",ha="left",fontsize=12,family="monospace"); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"06_summary.png"),dpi=150); plt.close(); print("  06_summary.png")

print("\nFeasibility verification done. Plots:", OUT)
