"""
Verification Script 3: PCM Feasibility - Rajasthan
Output: outputs/objective1_plots_rajasthan/verify_feasibility/
Run AFTER: Phase 5 feasibility screening
Checks survival rates, property space, top candidates, constraints.
"""
import os, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALL_CSV = os.path.join(BASE,"../../PCM_data/data","PCM_Properties_cleaned_mice_pmm_detailed.csv")
SUR_CSV = os.path.join(BASE,"data","processed","feasibility_survivors_rajasthan_kappa_calibrated.csv")
OUT     = os.path.join(BASE,"outputs","objective1_plots_rajasthan","verify_feasibility")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
try:
    sur=pd.read_csv(SUR_CSV); print(f"  Survivors: {sur.shape}")
except FileNotFoundError:
    print("ERROR: run Phase 5 feasibility screening first"); raise SystemExit(1)
try:
    alc=pd.read_csv(ALL_CSV); hf=True; print(f"  Full DB: {alc.shape}")
except FileNotFoundError:
    hf=False; print("  [WARN] Full DB not found, skipping ratios")

PAL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]

# 1. Survival rate per cluster
print("[1/6] Survival rates...")
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
ax.set(xlabel="Cluster ID",title="PCM Survival Rate per Climate Regime - Rajasthan"); ax.grid(alpha=0.3,axis="y")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"01_survival_rate_by_cluster.png"),dpi=150); plt.close(); print("  ✓ 01_survival_rate_by_cluster.png")

# 2. Property space
print("[2/6] Feasible property space...")
fig,ax=plt.subplots(figsize=(10,8))
# Try different column name variants
tm_col = "Tm_C" if "Tm_C" in sur.columns else "Tm_melting" if "Tm_melting" in sur.columns else None
lh_col = "latent_heat_kJ_kg" if "latent_heat_kJ_kg" in sur.columns else "latent_heat_melting" if "latent_heat_melting" in sur.columns else None

if tm_col and lh_col:
    sc=ax.scatter(sur[tm_col],sur[lh_col],c=sur["cluster_id"],cmap="tab10",s=100,alpha=0.6,edgecolors="black",lw=0.5)
    ax.set(xlabel="Melting Point (°C)",ylabel="Latent Heat (kJ/kg)",title="Feasible PCM Property Space - Rajasthan")
    ax.grid(alpha=0.3)
    # Look for temperature window columns
    window_cols = [c for c in sur.columns if "window" in c.lower()]
    if len(window_cols) >= 2:
        lo=sur[window_cols[0]].dropna().min(); hi=sur[window_cols[1]].dropna().max()
        ax.axvspan(lo,hi,alpha=0.12,color="green",label=f"Accepted Tm band: {lo:.0f}-{hi:.0f}°C")
    ax.axhline(100,color="gray",ls=":",alpha=0.6,label="Latent heat floor"); ax.legend(); plt.colorbar(sc,ax=ax,label="Cluster")
else:
    ax.text(0.5,0.5,"Required columns not found",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"02_feasible_property_space.png"),dpi=150); plt.close(); print("  ✓ 02_feasible_property_space.png")

# 3. Top candidates per cluster
print("[3/6] Top candidates per cluster...")
fig,ax=plt.subplots(figsize=(12,6))
pcm_col = "pcm_id" if "pcm_id" in sur.columns else "name" if "name" in sur.columns else None
lh_col_check = "latent_heat_kJ_kg" if "latent_heat_kJ_kg" in sur.columns else "latent_heat_melting" if "latent_heat_melting" in sur.columns else None

if lh_col_check and pcm_col:
    tp=sur.groupby("cluster_id").apply(lambda x: x.nlargest(3,lh_col_check) if len(x)>=3 else x).reset_index(drop=True)
    tc=tp[pcm_col].value_counts().head(12)
    bars=ax.barh(range(len(tc)),tc.values,color="coral",edgecolor="black")
    ax.set_yticks(range(len(tc))); ax.set_yticklabels(tc.index,fontsize=10)
    ax.set(xlabel="Frequency in top-3 by cluster",title="Highest-Latent-Heat Survivors per Cluster - Rajasthan")
    ax.grid(alpha=0.3,axis="x")
    for i,b in enumerate(bars):
        w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,str(int(w)),ha="left",va="center",fontsize=9)
else:
    ax.text(0.5,0.5,"Required columns not found",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"03_top_candidates_per_cluster.png"),dpi=150); plt.close(); print("  ✓ 03_top_candidates_per_cluster.png")

# 4. Constraint analysis
print("[4/6] Constraint analysis...")
cc=[c for c in sur.columns if "pass_" in c.lower()]
fig,ax=plt.subplots(figsize=(10,6))
if cc:
    sm=[]
    for c in cc:
        p=(sur[c]==True).sum() if pd.api.types.is_bool_dtype(sur[c]) else int(sur[c].sum())
        sm.append({"Constraint":c,"Pass":p,"Fail":len(sur)-p})
    if sm:
        pd.DataFrame(sm).set_index("Constraint")[["Pass","Fail"]].plot(kind="barh",stacked=True,ax=ax,color=["green","red"])
        ax.set(xlabel="Candidates",title="Constraint Pass/Fail - Rajasthan"); ax.grid(alpha=0.3,axis="x")
else:
    ax.text(0.5,0.5,"No constraint columns available",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"04_constraint_analysis.png"),dpi=150); plt.close(); print("  ✓ 04_constraint_analysis.png")

# 5. Property distributions by cluster
print("[5/6] Property distributions...")
pp=[p for p in ["Tm_C","Tm_melting","latent_heat_kJ_kg","latent_heat_melting","density_liquid_kg_m3","Cp_liquid_kJ_kgK"] if p in sur.columns][:3]
if pp:
    fig,axes=plt.subplots(1,len(pp),figsize=(5*len(pp),5))
    if len(pp)==1: axes=[axes]
    for i,p in enumerate(pp):
        sns.boxplot(data=sur,x="cluster_id",y=p,ax=axes[i],palette="Set2"); axes[i].set_title(f"{p} by Cluster"); axes[i].grid(alpha=0.3,axis="y")
    plt.suptitle("PCM Property Distributions (Rajasthan survivors)",fontsize=14,y=1.02); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"05_property_distributions.png"),dpi=150); plt.close(); print("  ✓ 05_property_distributions.png")
else:
    print("  [WARN] Not enough property columns available")

# 6. Summary
print("[6/6] Summary report...")
fig,ax=plt.subplots(figsize=(10,6))
ts=len(sur); uc=sur["cluster_id"].nunique()
txt=f"PCM FEASIBILITY FILTERING SUMMARY (Rajasthan)\n{'='*50}\nTotal Survivors: {ts}\nClusters: {uc}\nAvg Survivors/Cluster: {ts/uc:.1f}\n\nBy Cluster:\n"
for c in sorted(sur["cluster_id"].unique()): txt+=f"  Cluster {c}: {len(sur[sur['cluster_id']==c])} PCMs\n"
if hf and "cluster_id" in alc.columns:
    osr=100*ts/len(alc); q="OK (10-50%)" if 10<=osr<=50 else ("TOO STRICT (<10%)" if osr<10 else "TOO LOOSE (>50%)")
    txt+=f"\nOverall survival: {osr:.1f}%  [{q}]"
ax.text(0.05,0.95,txt,transform=ax.transAxes,va="top",ha="left",fontsize=12,family="monospace"); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"06_summary.png"),dpi=150); plt.close(); print("  ✓ 06_summary.png")

print("\n✓ Feasibility verification done. Plots:", OUT)
