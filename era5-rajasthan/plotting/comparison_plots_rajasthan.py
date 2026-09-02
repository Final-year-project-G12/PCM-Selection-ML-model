"""
Comparison Plots - Rajasthan PCM Pipeline
==========================================
Generates cross-step comparison plots to help verify results make sense.
Output: outputs/objective1_plots_rajasthan/comparison_plots/

Plots:
  1. Cluster GHI profiles: mean GHI by cluster over months
  2. PCM temperature target vs cluster mean temperature
  3. All 4 MCDM method rankings side-by-side per cluster (top 5)
  4. Monte Carlo stability: top3 prob vs consensus rank scatter
  5. Latent heat distribution: feasible survivors vs all candidates
  6. Physics validation: annual_solar_fraction vs MCDM rank
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
CLUSTERS= os.path.join(BASE,"data","processed","cluster_assignments_rajasthan_levelB.csv")  # Use Level B (final clustering)
SIG_CSV = os.path.join(BASE,"data","processed","climate_signature_rajasthan.csv")
FEAS    = os.path.join(BASE,"data","processed","feasibility_survivors_rajasthan_kappa_calibrated.csv")
PCM_DB  = os.path.join("..","..","PCM_data","data","PCM_Properties_cleaned_mice_pmm_detailed.csv")
TOPK    = os.path.join(BASE,"data","processed","mcdm_rankings_rajasthan.csv")
MC_CSV  = os.path.join(BASE,"data","processed","mcdm_rankings_rajasthan.csv")
PHYS    = os.path.join(BASE,"data","processed","physics_validation_rajasthan.csv")
CPROF   = os.path.join(BASE,"data","processed","cluster_profiles_rajasthan.csv")
OUT     = os.path.join(BASE,"outputs","objective1_plots_rajasthan","comparison_plots")
os.makedirs(OUT, exist_ok=True)

# Print data file paths for debugging
print(f"Looking for data files:")
print(f"  Climate Signature: {SIG_CSV}")
print(f"  Cluster Assignments: {CLUSTERS}")
print(f"  MCDM Rankings: {TOPK}")
print(f"  Physics Validation: {PHYS}")
print(f"  Feasibility: {FEAS}")
print()

PAL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45"]

def load(p,label=""):
    if not os.path.exists(p):
        print(f"  [WARN] skip {label}: not found at {p}");
        return None
    return pd.read_csv(p)

def sfig(n):
    plt.savefig(os.path.join(OUT,n),dpi=150,bbox_inches="tight");
    plt.close();
    print(f"  ✓ {n}")

def ensure_ranks(df):
    """Ensure rank columns exist; compute if missing."""
    for sc,rc,asc in [("topsis_score","topsis_rank",False),
                      ("gra_rank","gra_rank",False),
                      ("promethee_score","promethee_rank",False),
                      ("vikor_rank","vikor_rank",True),
                      ("borda_score","consensus_rank",False)]:
        if rc not in df.columns and sc in df.columns:
            df[rc]=df.groupby("cluster_id")[sc].rank(ascending=asc,method="min").astype(int)
    return df

print("\n" + "="*70)
print("RAJASTHAN COMPARISON PLOTS - Phase 4-9 Verification")
print("="*70 + "\n")

# ── Comparison 1: Cluster Mean GHI ────────────────────────────────────────
print("[1/8] Cluster GHI profiles from climate signature")
sig=load(SIG_CSV,"climate_signature")
clu=load(CLUSTERS,"climate_clusters")
if sig is not None and clu is not None:
    # Merge on point_id or available key
    merge_key = "point_id" if "point_id" in sig.columns and "point_id" in clu.columns else \
                "id" if "id" in sig.columns and "id" in clu.columns else None

    if merge_key:
        mg=sig.merge(clu[[merge_key,"cluster_id"]],on=merge_key,how="inner")
    else:
        # Fallback: assume same row order
        if len(sig)==len(clu):
            mg = sig.copy()
            mg["cluster_id"] = clu["cluster_id"].values
        else:
            print("  [WARN] Cannot merge on available keys"); mg=None

    if mg is not None:
        ghi_col="era5_GHI" if "era5_GHI" in mg.columns else \
                "GHI_mean" if "GHI_mean" in mg.columns else \
                ([c for c in mg.columns if "GHI" in c.upper()] or [None])[0]

        if ghi_col:
            fig,ax=plt.subplots(figsize=(9,5))
            for cid,g in mg.groupby("cluster_id"):
                ax.bar(str(int(cid)),g[ghi_col].mean(),
                      color=PAL[int(cid)%len(PAL)],edgecolor="white",lw=1,alpha=0.9,
                      label=f"Cluster {int(cid)}")
                ax.errorbar(str(int(cid)),g[ghi_col].mean(),
                           yerr=g[ghi_col].std(),fmt="none",color="black",capsize=5,lw=1.5)
            ax.set(xlabel="Cluster",ylabel=f"{ghi_col} (mean ± std)",
                  title="Comparison 1: Mean GHI by Climate Regime (Rajasthan)")
            ax.legend(fontsize=9); ax.grid(alpha=0.3,axis="y"); sfig("01_comparison_cluster_ghi.png")
        else: print("  [WARN] No GHI column found in signature")
else:
    print("  [SKIP] Signature or clusters file missing")

# ── Comparison 2: PCM Tm_target vs Cluster Mean Temp ────────────────────
print("[2/8] PCM Tm_target vs cluster mean temperature")
if sig is not None and clu is not None:
    merge_key = "point_id" if "point_id" in sig.columns and "point_id" in clu.columns else \
                "id" if "id" in sig.columns and "id" in clu.columns else None

    if merge_key:
        mg2=sig.merge(clu[[merge_key,"cluster_id"]],on=merge_key,how="inner")
    else:
        if len(sig)==len(clu):
            mg2 = sig.copy()
            mg2["cluster_id"] = clu["cluster_id"].values
        else:
            mg2 = None

    if mg2 is not None:
        # Look for temperature column
        t_col="era5_T_amb" if "era5_T_amb" in mg2.columns else \
              "Ta_mean_proxy" if "Ta_mean_proxy" in mg2.columns else \
              ([c for c in mg2.columns if "T_" in c or "temp" in c.lower()] or [None])[0]

        feas=load(FEAS,"feasibility")

        if t_col and feas is not None:
            # Get Tm_target: check multiple column name possibilities
            tm_col = "Tm_target_C" if "Tm_target_C" in feas.columns else \
                     "Tm_C" if "Tm_C" in feas.columns else \
                     ([c for c in feas.columns if "Tm" in c] or [None])[0]

            if tm_col:
                clust_T=mg2.groupby("cluster_id")[t_col].mean()
                tm_target=feas.groupby("cluster_id")[tm_col].first()
                comp=pd.DataFrame({"ClusterMeanT_C":clust_T,"PCM_Tm_target":tm_target}).dropna()

                if not comp.empty:
                    fig,ax=plt.subplots(figsize=(9,6))
                    for cid,row in comp.iterrows():
                        ax.scatter(row["ClusterMeanT_C"],row["PCM_Tm_target"],
                                  color=PAL[int(cid)%len(PAL)],s=150,zorder=3,
                                  label=f"Cluster {int(cid)}")
                        ax.annotate(f"C{int(cid)}",(row["ClusterMeanT_C"],row["PCM_Tm_target"]),
                                   textcoords="offset points",xytext=(5,5),fontsize=9)
                    xlim=ax.get_xlim()
                    ax.plot(xlim,[x+25 for x in xlim],"r--",lw=1,label="+25°C offset")
                    ax.plot(xlim,[x+35 for x in xlim],"g--",lw=1,label="+35°C offset")
                    ax.set(xlabel=f"Cluster Mean Temperature ({t_col}) (°C)",
                          ylabel="PCM Target Melting Point (°C)",
                          title="Comparison 2: Cluster Temperature vs PCM Tm Target (Rajasthan)")
                    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("02_comparison_temp_vs_tm_target.png")
else:
    print("  [SKIP] Signature or clusters file missing")

# ── Comparison 3: All MCDM Rankings Side-by-Side per Cluster ───────────────
print("[3/8] MCDM method comparison: top 5 per cluster")
topk=load(TOPK,"mcdm_rankings")
if topk is not None:
    topk=ensure_ranks(topk)

    # Adapt column names for Rajasthan
    methods=["TOPSIS_rank","GRA_rank","PROMETHEE_II_rank","VIKOR_rank","borda_score"]
    methods=[m for m in methods if m in topk.columns]

    # If exact columns not found, try alternatives
    if len(methods)<2:
        potential=[c for c in topk.columns if "rank" in c.lower() or "score" in c.lower()]
        methods = potential[:5]

    if len(methods)>=2:
        # Use pcm_id as fallback for name column
        if "name" not in topk.columns and "pcm_id" in topk.columns:
            topk["name"] = topk["pcm_id"]

        clus_ids=sorted(topk["cluster_id"].unique())
        fig,axes=plt.subplots(len(clus_ids),1,figsize=(13,4.5*len(clus_ids)),squeeze=False)

        for idx,cid in enumerate(clus_ids):
            # Sort by consensus/borda rank if available
            sort_col = "consensus_rank" if "consensus_rank" in topk.columns else \
                       "borda_score" if "borda_score" in topk.columns else methods[0]

            sub=topk[topk["cluster_id"]==cid].sort_values(sort_col,ascending=False).head(5)

            x=np.arange(len(sub))
            w=0.15
            ax=axes[idx,0]

            for mi,m in enumerate(methods):
                if m in sub.columns:
                    ax.bar(x+mi*w, sub[m].values, width=w,
                          label=m.replace("_rank","").replace("_score","").upper(),
                          color=sns.color_palette("Set2",len(methods))[mi],
                          edgecolor="white")

            pcm_names = sub["name"].tolist() if "name" in sub.columns else sub.index.tolist()
            ax.set_xticks(x+(len(methods)-1)*w/2)
            ax.set_xticklabels(pcm_names, rotation=20, ha="right", fontsize=9)
            ax.set(title=f"Cluster {int(cid)} - Top 5 PCM Ranks by Method",
                  ylabel="Rank (lower=better)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25,axis="y")

        plt.suptitle("Comparison 3: MCDM Methods Side-by-Side (Rajasthan)",
                    fontsize=13, fontweight="bold")
        plt.tight_layout()
        sfig("03_comparison_mcdm_methods.png")
else:
    print("  [SKIP] MCDM rankings file missing")

# ── Comparison 4: Monte Carlo Top3-Prob vs Consensus Rank ────────────────
print("[4/8] Monte Carlo stability vs consensus rank")
mc=load(MC_CSV,"monte_carlo")
if mc is not None and topk is not None:
    # Check if MC inclusion probability exists
    mc_prob_col = "mc_top3_inclusion_pct" if "mc_top3_inclusion_pct" in mc.columns else \
                  "top3_inclusion_probability" if "top3_inclusion_probability" in mc.columns else \
                  ([c for c in mc.columns if "inclusion" in c.lower() or "top3" in c.lower()] or [None])[0]

    if mc_prob_col:
        # Merge MC data with rankings
        merge_on = ["cluster_id","pcm_id"] if "pcm_id" in mc.columns and "pcm_id" in topk.columns else \
                   ["cluster_id","name"] if "name" in mc.columns and "name" in topk.columns else None

        if merge_on:
            mg4=mc.merge(topk[["cluster_id"] + merge_on[1:] + ["consensus_rank"]].drop_duplicates(),
                         on=merge_on, how="inner").dropna(subset=["consensus_rank"])
        else:
            # Fallback: topk itself may have inclusion probability
            mg4 = topk.dropna(subset=["consensus_rank", mc_prob_col]) if mc_prob_col in topk.columns else None

        if mg4 is not None and not mg4.empty and "consensus_rank" in mg4.columns:
            fig,ax=plt.subplots(figsize=(10,7))
            scale=100 if mg4[mc_prob_col].max()<=1.0 else 1

            for cid,g in mg4.groupby("cluster_id"):
                pcm_col = "pcm_id" if "pcm_id" in g.columns else "name" if "name" in g.columns else None

                ax.scatter(g["consensus_rank"], g[mc_prob_col]*scale,
                          color=PAL[int(cid)%len(PAL)], s=80, alpha=0.8,
                          edgecolors="white", lw=0.5, label=f"Cluster {int(cid)}")

                if pcm_col:
                    for _,row in g.iterrows():
                        ax.annotate(row[pcm_col], (row["consensus_rank"], row[mc_prob_col]*scale),
                                   fontsize=6, alpha=0.7)

            ax.set(xlabel="Consensus Rank",
                  ylabel="Top-3 Inclusion Probability (%)",
                  title="Comparison 4: Monte Carlo Stability vs MCDM Consensus Rank (Rajasthan)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
            sfig("04_comparison_mc_vs_rank.png")
else:
    print("  [SKIP] Monte Carlo or MCDM rankings file missing")

# ── Comparison 5: Latent heat distribution - feasible vs all ─────────────
print("[5/8] Latent heat distribution comparison")
feas=load(FEAS,"feasibility")
db=load(PCM_DB,"pcm_db")

if feas is not None:
    lh_col = "latent_heat_kJ_kg" if "latent_heat_kJ_kg" in feas.columns else \
             "latent_heat_melting" if "latent_heat_melting" in feas.columns else \
             ([c for c in feas.columns if "latent" in c.lower()] or [None])[0]

    if lh_col:
        fig,ax=plt.subplots(figsize=(10,6))

        if db is not None and lh_col in db.columns:
            ax.hist(db[lh_col].dropna(), bins=40, alpha=0.5, color="gray",
                   label=f"All candidates (n={len(db)})", density=True)

        ax.hist(feas[lh_col].dropna(), bins=30, alpha=0.8, color="#3b7dd8",
               label=f"Feasible survivors (n={len(feas)})", density=True)

        median_lh = feas[lh_col].median()
        ax.axvline(median_lh, color="#3b7dd8", ls="--", lw=2,
                  label=f"Feasible median: {median_lh:.0f} kJ/kg")

        ax.set(xlabel="Latent Heat (kJ/kg)", ylabel="Density",
              title="Comparison 5: Latent Heat Distribution - All vs Feasible Survivors (Rajasthan)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        sfig("05_comparison_latent_heat_distribution.png")
else:
    print("  [SKIP] Feasibility file missing")

# ── Comparison 6: Physics validation - solar_fraction vs MCDM rank ────────
print("[6/8] Physics validation vs MCDM rank")
phys=load(PHYS,"physics_validation")
if phys is not None and topk is not None:
    # Look for physics validation column
    phys_col = "annual_solar_fraction" if "annual_solar_fraction" in phys.columns else \
               "solar_fraction" if "solar_fraction" in phys.columns else \
               "hours_target_met_per_year" if "hours_target_met_per_year" in phys.columns else \
               ([c for c in phys.columns if "solar" in c.lower() or "hours" in c.lower()] or [None])[0]

    if phys_col:
        merge_on = ["cluster_id","pcm_id"] if "pcm_id" in phys.columns and "pcm_id" in topk.columns else \
                   ["cluster_id","name"] if "name" in phys.columns and "name" in topk.columns else None

        if merge_on:
            mg6 = topk.merge(phys[[col for col in ["cluster_id"] + merge_on[1:] + [phys_col] if col in phys.columns]].drop_duplicates(),
                            on=merge_on, how="inner")
        else:
            mg6 = topk.copy()
            for col in [phys_col]:
                if col in phys.columns:
                    mg6[col] = phys[col].values[:len(mg6)]

        if "consensus_rank" in mg6.columns and not mg6.empty:
            fig,axes=plt.subplots(1,2,figsize=(14,6))

            # Plot 1: MCDM rank vs solar fraction
            for cid,g in mg6.groupby("cluster_id"):
                v = g[["consensus_rank",phys_col]].notna().all(axis=1)
                axes[0].scatter(g.loc[v,"consensus_rank"], g.loc[v,phys_col],
                               color=PAL[int(cid)%len(PAL)], s=80, alpha=0.8,
                               label=f"Cluster {int(cid)}")

            axes[0].set(xlabel="Consensus Rank", ylabel=phys_col,
                       title="MCDM Rank vs Physics Performance")
            axes[0].legend(fontsize=9)
            axes[0].grid(alpha=0.25)

            # Plot 2: If complete_cycles or other metric exists
            cycles_col = "complete_cycles_per_year" if "complete_cycles_per_year" in mg6.columns else \
                         "cycles" if "cycles" in mg6.columns else None

            if cycles_col:
                for cid,g in mg6.groupby("cluster_id"):
                    v = g[["consensus_rank",cycles_col]].notna().all(axis=1)
                    axes[1].scatter(g.loc[v,"consensus_rank"], g.loc[v,cycles_col],
                                   color=PAL[int(cid)%len(PAL)], s=80, alpha=0.8,
                                   label=f"Cluster {int(cid)}")
                axes[1].set(xlabel="Consensus Rank", ylabel=cycles_col,
                           title="MCDM Rank vs Operational Cycles")
                axes[1].legend(fontsize=9)
                axes[1].grid(alpha=0.25)

            plt.suptitle("Comparison 6: Physics Validation vs MCDM Ranking (Rajasthan)",
                        fontsize=13, fontweight="bold")
            plt.tight_layout()
            sfig("06_comparison_physics_vs_rank.png")
else:
    print("  [SKIP] Physics validation or MCDM rankings file missing")

# ── Comparison 7: Cross-cluster top PCM key properties ───────────────────
print("[7/8] Cross-cluster summary: top PCM properties")
if topk is not None:
    consensus_col = "consensus_rank" if "consensus_rank" in topk.columns else \
                    "borda_score" if "borda_score" in topk.columns else None

    if consensus_col:
        if "borda_score" in topk.columns and consensus_col == "borda_score":
            # Lower Borda score is better
            top1 = topk[topk["borda_score"]==topk.groupby("cluster_id")["borda_score"].transform("min")].drop_duplicates(subset="cluster_id").copy()
        else:
            top1 = topk[topk[consensus_col]==1].drop_duplicates(subset="cluster_id").copy()

        # Look for property columns
        props=[c for c in ["Tm_C","Tm_melting","latent_heat_kJ_kg","latent_heat_melting",
                          "rho_H_MJ_m3","TC_W_mK","cycles_tested","cycles"] if c in top1.columns]

        if len(props)>=2 and len(top1)>=1:
            fig,axes=plt.subplots(1,len(props),figsize=(4*len(props),5))
            if len(props)==1: axes=[axes]

            for i,p in enumerate(props):
                cluster_ids = top1["cluster_id"].astype(str).values
                values = top1[p].values
                colors = [PAL[int(c)%len(PAL)] for c in top1["cluster_id"]]

                axes[i].bar(cluster_ids, values, color=colors, edgecolor="white")
                axes[i].set(title=p, xlabel="Cluster")
                axes[i].grid(alpha=0.3, axis="y")

                for j,(_,row) in enumerate(top1.iterrows()):
                    if pd.notna(row.get(p)):
                        pcm_name = row.get("name", row.get("pcm_id", ""))[:10]
                        axes[i].text(j, row[p]*1.01, pcm_name, ha="center", fontsize=7, rotation=25)

            plt.suptitle("Comparison 7: Top PCM Properties per Cluster (Rajasthan) - Rank #1",
                        fontsize=13, fontweight="bold")
            plt.tight_layout()
            sfig("07_comparison_cross_cluster_top_pcm.png")
else:
    print("  [SKIP] MCDM rankings file missing")

# ── Comparison 8: Weight sensitivity ─────────────────────────────────────
print("[8/8] Rank sensitivity to weight perturbation")
if topk is not None:
    topk=ensure_ranks(topk)

    # Look for score columns
    score_cols=[c for c in ["topsis_score","TOPSIS_score","gra_score","GRA_rank",
                            "promethee_score","PROMETHEE_II_score"] if c in topk.columns]

    if len(score_cols)<2:
        # Fallback: use first two score/rank columns found
        potential = [c for c in topk.columns if "score" in c.lower() and "consensus" not in c.lower()]
        score_cols = potential[:2]

    if len(score_cols)>=2:
        c1,c2 = score_cols[0], score_cols[1]
        results=[]

        for w1 in [0.3,0.5,0.7]:
            w2 = 1-w1
            for cid,g in topk.groupby("cluster_id"):
                v = g[[c1,c2]].notna().all(axis=1)
                if v.sum()>0:
                    # Normalize and combine scores
                    s1_norm = g.loc[v,c1]/max(1,g.loc[v,c1].max())
                    s2_norm = g.loc[v,c2]/max(1,g.loc[v,c2].max())
                    comb = w1*s1_norm + w2*s2_norm
                    rk = comb.rank(ascending=False, method="min")

                    for idx2 in g.loc[v].index:
                        pcm_name = g.loc[idx2,"name"] if "name" in g.columns else \
                                   g.loc[idx2,"pcm_id"] if "pcm_id" in g.columns else str(idx2)
                        results.append({
                            "w1": w1,
                            "Cluster": cid,
                            "Name": pcm_name,
                            "ComboRank": rk.loc[idx2]
                        })

        rdf = pd.DataFrame(results)
        if not rdf.empty:
            fig,ax=plt.subplots(figsize=(12,6))

            # Get top names to plot
            if "consensus_rank" in topk.columns and "name" in topk.columns:
                top_names = topk[topk["consensus_rank"]<=3]["name"].unique()
            elif "pcm_id" in rdf.columns:
                top_names = rdf["Name"].value_counts().head(6).index
            else:
                top_names = rdf["Name"].unique()[:6]

            for nm in top_names[:6]:
                sub = rdf[rdf["Name"]==nm]
                if not sub.empty:
                    ax.plot(sub["w1"], sub["ComboRank"], "-o", label=str(nm)[:20],
                           lw=1.5, ms=6)

            c1_label = c1.replace("_score","").replace("_rank","").replace("_grade","").upper()
            c2_label = c2.replace("_score","").replace("_rank","").replace("_grade","").upper()

            ax.set(xlabel=f"Weight on {c1_label} (remainder on {c2_label})",
                  ylabel="Combined Rank",
                  title="Comparison 8: Rank Sensitivity to Weight Perturbation (Rajasthan)")
            ax.invert_yaxis()
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
            sfig("08_comparison_rank_sensitivity.png")
    else:
        print("  [WARN] Insufficient score columns for sensitivity analysis")
else:
    print("  [SKIP] MCDM rankings file missing")

print("\n" + "="*70)
print(f"✓ All comparison plots saved to: {OUT}")
print("="*70 + "\n")
