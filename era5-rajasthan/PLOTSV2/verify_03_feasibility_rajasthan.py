"""
Verification Script 3: PCM Feasibility - Rajasthan
Output: PLOTSV2/verify_feasibility/
Run AFTER: 07_feasibility_filter_rajasthan.py

Port of tamilnadu_pipeline/plots/verify_03_feasibility_tamilnadu.py and
era5-uttarakhand/verify_03_feasibility.py - same 6 plots, same style.

Two corrections against plotting/verify_03_feasibility_rajasthan.py, both
schema mismatches that made the old figures wrong rather than merely ugly:

  * Rajasthan's survivors CSV holds EVERY candidate x cluster pair with a
    boolean `survives_all` column, not just the survivors. The old script
    plotted all 186 rows, so plot 1 reported "62 survivors" in each of the
    three clusters instead of the real 9 / 14 / 16.
  * Constraints are named c1_melting_window ... c8_safety and hold the
    strings "pass"/"fail"/"not_applicable"/"flag_*", not booleans in
    "pass_*" columns. The old script found no matching columns, so plot 4
    rendered an empty "No constraint columns available" placeholder.
"""
import os, sys, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALL_CSV = os.path.join(BASE,"..","PCM_data","data","PCM_Properties_cleaned_mice_pmm_detailed.csv")
SUR_CSV = os.path.join(BASE,"data","processed","feasibility_survivors_rajasthan_kappa_calibrated.csv")
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)),"verify_feasibility")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
try:
    evald=pd.read_csv(SUR_CSV); print(f"  Evaluated candidate x cluster rows: {evald.shape}")
except FileNotFoundError:
    print("ERROR: run 07_feasibility_filter_rajasthan.py first"); raise SystemExit(1)

# The file is the full evaluation table; survivors are the flagged subset.
if "survives_all" in evald.columns:
    sur=evald[evald["survives_all"].astype(str).str.lower()=="true"].copy()
else:
    sur=evald.copy()
n_pool=evald["pcm_id"].nunique() if "pcm_id" in evald.columns else len(evald)
print(f"  Candidate pool: {n_pool}   Survivors: {len(sur)}")

try:
    alc=pd.read_csv(ALL_CSV); hf=True; print(f"  Manufacturer DB: {alc.shape}")
except FileNotFoundError:
    hf=False; print("  [WARN] Full DB not found, using evaluated pool as denominator")

PAL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]

# 1. Survival rate per cluster
print("[1/6] Survival rates...")
fig,ax=plt.subplots(figsize=(10,6)); cids=sorted(evald["cluster_id"].unique())
denom={c:max(1,len(evald[evald["cluster_id"]==c])) for c in cids}
sp=[100*len(sur[sur["cluster_id"]==c])/denom[c] for c in cids]
bars=ax.bar(cids,sp,color=[PAL[int(c)%len(PAL)] for c in cids],edgecolor="black")
ax.set_ylabel("Survival rate (%)")
ax.axhline(10,color="green",ls="--",lw=2,label="Selectivity band 10-50%"); ax.axhline(50,color="green",ls="--",lw=2)
for b,c in zip(bars,cids):
    h=b.get_height(); n=len(sur[sur["cluster_id"]==c])
    ax.text(b.get_x()+b.get_width()/2,h,f"{h:.1f}%\n({n}/{denom[c]})",ha="center",va="bottom",fontsize=10)
ax.set(xlabel="Cluster ID",title="PCM Survival Rate per Climate Regime - Rajasthan")
ax.set_xticks(cids); ax.set_ylim(0,max(sp)*1.35); ax.legend(); ax.grid(alpha=0.3,axis="y")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"01_survival_rate_by_cluster.png"),dpi=150); plt.close(); print("  01_survival_rate_by_cluster.png")

# 2. Property space
print("[2/6] Feasible property space...")
fig,ax=plt.subplots(figsize=(10,8))
tm_col = "Tm_C" if "Tm_C" in sur.columns else "Tm_melting" if "Tm_melting" in sur.columns else None
lh_col = "latent_heat_kJ_kg" if "latent_heat_kJ_kg" in sur.columns else "latent_heat_melting" if "latent_heat_melting" in sur.columns else None

if tm_col and lh_col:
    ax.scatter(evald[tm_col],evald[lh_col],color="#cccccc",s=45,alpha=0.6,label=f"Evaluated pool ({n_pool} PCMs)",zorder=2)
    sc=ax.scatter(sur[tm_col],sur[lh_col],c=sur["cluster_id"],cmap="tab10",s=100,alpha=0.75,edgecolors="black",lw=0.5,zorder=3)
    ax.set(xlabel="Melting Point (degC)",ylabel="Latent Heat (kJ/kg)",title="Feasible PCM Property Space - Rajasthan")
    ax.grid(alpha=0.3)
    # Accepted Tm band = span of the survivors' melting points per the
    # filter's final (relaxed) melting window.
    lo=sur[tm_col].min(); hi=sur[tm_col].max()
    ax.axvspan(lo,hi,alpha=0.12,color="green",label=f"Survivor Tm span: {lo:.0f}-{hi:.0f}degC")
    ax.axhline(100,color="gray",ls=":",alpha=0.6,label="Latent heat floor 100 kJ/kg")
    ax.legend(fontsize=9); plt.colorbar(sc,ax=ax,label="Cluster")
else:
    ax.text(0.5,0.5,"Required columns not found",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"02_feasible_property_space.png"),dpi=150); plt.close(); print("  02_feasible_property_space.png")

# 3. Top candidates per cluster
print("[3/6] Top candidates per cluster...")
fig,ax=plt.subplots(figsize=(12,6))
pcm_col = "pcm_id" if "pcm_id" in sur.columns else "name" if "name" in sur.columns else None

if lh_col and pcm_col and len(sur):
    tp=pd.concat([g.nlargest(3,lh_col) for _,g in sur.groupby("cluster_id")])
    tc=tp[pcm_col].value_counts().head(12)
    bars=ax.barh(range(len(tc)),tc.values,color="coral",edgecolor="black")
    ax.set_yticks(range(len(tc))); ax.set_yticklabels(tc.index,fontsize=10)
    ax.set(xlabel="Frequency in top-3 by cluster",title="Highest-Latent-Heat Survivors per Cluster - Rajasthan")
    ax.grid(alpha=0.3,axis="x"); ax.invert_yaxis()
    for b in bars:
        w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,str(int(w)),ha="left",va="center",fontsize=9)
else:
    ax.text(0.5,0.5,"Required columns not found",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"03_top_candidates_per_cluster.png"),dpi=150); plt.close(); print("  03_top_candidates_per_cluster.png")

# 4. Constraint analysis
print("[4/6] Constraint analysis...")
cc=[c for c in evald.columns if c.startswith("c") and c[1:2].isdigit() and "_" in c]
fig,ax=plt.subplots(figsize=(11,6))
if cc:
    sm=[]
    for c in cc:
        vals=evald[c].astype(str).str.lower()
        sm.append({"Constraint":c,
                   "Pass":int((vals=="pass").sum()),
                   "Fail":int((vals=="fail").sum()),
                   "N/A or flagged":int((~vals.isin(["pass","fail"])).sum())})
    pd.DataFrame(sm).set_index("Constraint")[["Pass","Fail","N/A or flagged"]].plot(
        kind="barh",stacked=True,ax=ax,color=["#3cb44b","#e6194b","#bbbbbb"])
    ax.set(xlabel=f"Candidate x cluster evaluations (n={len(evald)})",
           title="Constraint Pass/Fail - Rajasthan")
    ax.grid(alpha=0.3,axis="x"); ax.invert_yaxis(); ax.legend(loc="lower right",fontsize=9)
else:
    ax.text(0.5,0.5,"No constraint columns available",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"04_constraint_analysis.png"),dpi=150); plt.close(); print("  04_constraint_analysis.png")

# 5. Property distributions by cluster
print("[5/6] Property distributions...")
pp=[p for p in ["Tm_C","latent_heat_kJ_kg","supercooling_K","cycles_tested"] if p in sur.columns][:3]
if pp and len(sur):
    fig,axes=plt.subplots(1,len(pp),figsize=(5*len(pp),5))
    if len(pp)==1: axes=[axes]
    for i,p in enumerate(pp):
        sns.boxplot(data=sur,x="cluster_id",y=p,hue="cluster_id",palette="Set2",legend=False,ax=axes[i])
        axes[i].set_title(f"{p} by Cluster"); axes[i].grid(alpha=0.3,axis="y")
    plt.suptitle("PCM Property Distributions (Rajasthan survivors)",fontsize=14,y=1.02); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"05_property_distributions.png"),dpi=150); plt.close(); print("  05_property_distributions.png")
else:
    print("  [WARN] Not enough property columns available")

# 6. Summary
print("[6/6] Summary report...")
fig,ax=plt.subplots(figsize=(10,6))
ts=len(sur); uc=evald["cluster_id"].nunique()
txt=(f"PCM FEASIBILITY FILTERING SUMMARY (Rajasthan)\n{'='*50}\n"
     f"Candidate pool: {n_pool} PCMs\nEvaluations: {len(evald)} (pool x {uc} clusters)\n"
     f"Total Survivors: {ts}\nClusters: {uc}\nAvg Survivors/Cluster: {ts/uc:.1f}\n\nBy Cluster:\n")
for c in sorted(evald["cluster_id"].unique()):
    n=len(sur[sur["cluster_id"]==c]); d=denom[c]
    txt+=f"  Cluster {c}: {n}/{d} PCMs ({100*n/d:.1f}%)\n"
osr=100*ts/len(evald)
q="OK (10-50%)" if 10<=osr<=50 else ("TOO STRICT (<10%)" if osr<10 else "TOO LOOSE (>50%)")
txt+=f"\nOverall survival: {osr:.1f}%  [{q}]"
ax.text(0.05,0.95,txt,transform=ax.transAxes,va="top",ha="left",fontsize=12,family="monospace"); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"06_summary.png"),dpi=150); plt.close(); print("  06_summary.png")

print("\nFeasibility verification done. Plots:", OUT)
