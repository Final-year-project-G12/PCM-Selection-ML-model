"""
Uttarakhand PCM Pipeline - Objective 1 Plot Generator
Generates all 13 required plots (static PNG + interactive Plotly/Folium)
Output: data/plots/uttarakhand_objective1/
"""
import os, sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
warnings.filterwarnings("ignore")

BASE         = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV      = os.path.join(BASE,"data","processed","climate_uttarakhand_points.csv")
PHYSICAL_CSV = os.path.join(BASE,"data","preprocessed","uttarakhand_cleaned_physical.csv")
CLUSTERS     = os.path.join(BASE,"data","processed","clustering","cluster_assignments_uttarakhand.csv")
PCM_DB       = os.path.join(BASE,"data","processed","pcm","pcm_database_uttarakhand.csv")
FEASIBILITY  = os.path.join(BASE,"data","processed","pcm","feasibility_survivors_by_cluster.csv")
TOPK         = os.path.join(BASE,"data","processed","pcm","mcdm_topk_by_cluster.csv")
MC_STABILITY = os.path.join(BASE,"data","processed","pcm","monte_carlo_stability.csv")
PHYS_VAL     = os.path.join(BASE,"data","processed","pcm","physics_validation_results.csv")
OUT          = os.path.join(BASE,"data","plots","uttarakhand_objective1")
os.makedirs(OUT, exist_ok=True)

METHODS = ["topsis","gra","promethee","vikor"]
MRANK   = [f"{m}_rank" for m in METHODS] + ["consensus_rank"]
PAL     = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45"]

def ensure_ranks(df):
    for sc,rc,asc in [("topsis_score","topsis_rank",False),("gra_grade","gra_rank",False),
                      ("promethee_flow","promethee_rank",False),("vikor_Q","vikor_rank",True),
                      ("borda_score","consensus_rank",False)]:
        if rc not in df.columns and sc in df.columns:
            df[rc]=df.groupby("cluster_id")[sc].rank(ascending=asc,method="min").astype(int)
    return df

def load(path,label="",**kw):
    if not os.path.exists(path): print(f"  skip {label}: not found"); return None
    return pd.read_csv(path,**kw)

def sfig(n, dpi=150):
    plt.savefig(os.path.join(OUT,n),dpi=dpi,bbox_inches="tight"); plt.close(); print(f"  {n}")

def shtml(fig,n):
    fig.write_html(os.path.join(OUT,n),include_plotlyjs="cdn"); print(f"  {n}")

# ---- Plot 1: Raw vs Preprocessed GHI ----
def p01():
    print("[1/13] Raw vs Preprocessed Radiation")
    raw=load(RAW_CSV,"raw",nrows=500000); pre=load(PHYSICAL_CSV,"preprocessed",nrows=500000)
    if raw is None or pre is None or "era5_GHI" not in raw.columns: return
    pt=raw["point_id"].dropna().iloc[0]
    r=raw[raw["point_id"]==pt].sort_values("date") if "date" in raw.columns else raw[raw["point_id"]==pt]
    p=pre[pre["point_id"]==pt].sort_values("date") if ("point_id" in pre.columns and "date" in pre.columns) else pre.head(len(r))
    fig,ax=plt.subplots(2,1,figsize=(14,7))
    ax[0].plot(r["era5_GHI"].values,color="#e07b39",lw=0.8,alpha=0.8,label="Raw GHI")
    ax[0].set(title=f"Raw GHI - Point {pt}",ylabel="GHI (W/m2)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if "era5_GHI" in p.columns:
        ax[1].plot(p["era5_GHI"].values,color="#3b7dd8",lw=0.9,label="Preprocessed GHI")
    ax[1].set(title="Preprocessed GHI",ylabel="GHI (W/m2)",xlabel="Record index"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.suptitle("Uttarakhand - Raw vs Preprocessed Solar Radiation (GHI)",fontsize=13)
    plt.tight_layout(); sfig("01_raw_vs_preprocessed_radiation.png")
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(y=r["era5_GHI"].values,mode="lines",name="Raw GHI",line=dict(color="#e07b39",width=1)))
    if "era5_GHI" in p.columns:
        fig2.add_trace(go.Scatter(y=p["era5_GHI"].values,mode="lines",name="Preprocessed GHI",line=dict(color="#3b7dd8",width=1.5)))
    fig2.update_layout(title=f"Raw vs Preprocessed GHI - {pt}",xaxis_title="Record",yaxis_title="GHI (W/m2)",template="plotly_dark",height=450)
    shtml(fig2,"01_raw_vs_preprocessed_radiation_interactive.html")

# ---- Plot 2: Climate Regime Map ----
def p02():
    print("[2/13] Climate-Regime Map")
    df=load(CLUSTERS,"clusters")
    if df is None or not {"lat","lon","cluster_id"}.issubset(df.columns): return
    nc=df["cluster_id"].nunique(); cmap=plt.cm.get_cmap("tab10",nc)
    fig,ax=plt.subplots(figsize=(11,10))
    for cid,g in df.groupby("cluster_id"):
        ax.scatter(g["lon"],g["lat"],color=cmap(int(cid)),s=60,alpha=0.85,edgecolors="white",lw=0.4,label=f"Cluster {cid}")
    ax.set(title="Uttarakhand - Climate Regime Map\n(GMM clusters per grid point)",xlabel="Longitude E",ylabel="Latitude N",xlim=[77.5,81.5],ylim=[28.5,31.5])
    ax.legend(title="Regime",loc="lower right",fontsize=9); ax.grid(alpha=0.25,ls="--")
    plt.tight_layout(); sfig("02_climate_regime_map.png")
    fig_px=px.scatter(df,x="lon",y="lat",color=df["cluster_id"].astype(str),
                      hover_data=["point_id","cluster_id","max_membership_prob"] if "max_membership_prob" in df.columns else ["point_id","cluster_id"],
                      title="Uttarakhand - Climate Regime Map",labels={"color":"Cluster"},template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set1)
    fig_px.update_traces(marker=dict(size=8,opacity=0.8)); fig_px.update_layout(height=600)
    shtml(fig_px,"02_climate_regime_map_interactive.html")
    m=folium.Map(location=[df["lat"].mean(),df["lon"].mean()],zoom_start=7,tiles="CartoDB positron")
    mc=MarkerCluster().add_to(m)
    cf=["red","green","blue","purple","orange","darkred","lightred","beige"]
    for _,r in df.iterrows():
        cid=int(r["cluster_id"])
        folium.CircleMarker(location=[r["lat"],r["lon"]],radius=6,color=cf[cid%len(cf)],fill=True,fill_opacity=0.75,
            popup=folium.Popup(f"<b>Point:</b> {r['point_id']}<br><b>Cluster:</b> {cid}<br><b>Prob:</b> {r.get('max_membership_prob',0):.3f}",max_width=220)
        ).add_to(mc)
    m.save(os.path.join(OUT,"02_climate_regime_map_folium.html")); print("  02_climate_regime_map_folium.html")

# ---- Plot 3: Melting Point vs Latent Heat ----
def p03():
    print("[3/13] Melting Point vs Latent Heat")
    df=load(FEASIBILITY,"feasibility")
    if df is None or not {"Tm_C","latent_heat_kJ_kg"}.issubset(df.columns): return
    fig,ax=plt.subplots(figsize=(11,7))
    for cid,g in df.groupby("cluster_id"):
        ax.scatter(g["Tm_C"],g["latent_heat_kJ_kg"],color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,edgecolors="white",lw=0.5,label=f"Cluster {cid}")
    if {"window_lo","window_hi"}.issubset(df.columns):
        for cid,g in df.groupby("cluster_id"):
            lo=g["window_lo"].dropna(); hi=g["window_hi"].dropna()
            if len(lo) and len(hi): ax.axvspan(lo.iloc[0],hi.iloc[0],alpha=0.07,color=PAL[int(cid)%len(PAL)])
    ax.axhline(100,color="gray",ls=":",alpha=0.7,label="Latent heat floor 100 kJ/kg")
    ax.set(title="Uttarakhand PCM - Melting Point vs Latent Heat (Feasible Survivors)",xlabel="Melting Temp (C)",ylabel="Latent Heat (kJ/kg)")
    ax.legend(title="Regime",fontsize=9); ax.grid(alpha=0.25); plt.tight_layout(); sfig("03_melting_point_vs_latent_heat.png")
    fig_px=px.scatter(df,x="Tm_C",y="latent_heat_kJ_kg",color=df["cluster_id"].astype(str),
                      hover_data=["name","family","pcm_type"] if "name" in df.columns else None,
                      title="Uttarakhand - Melting Point vs Latent Heat",template="plotly_white",
                      labels={"Tm_C":"Melting Temp (C)","latent_heat_kJ_kg":"Latent Heat (kJ/kg)","color":"Cluster"},
                      color_discrete_sequence=px.colors.qualitative.Set1)
    fig_px.update_traces(marker=dict(size=9,opacity=0.85)); fig_px.add_hline(y=100,line_dash="dot",line_color="gray",annotation_text="Latent heat floor")
    shtml(fig_px,"03_melting_point_vs_latent_heat_interactive.html")

# ---- Plot 4: Feasible Candidates Highlighted ----
def p04():
    print("[4/13] Feasible Candidates Highlighted")
    feas=load(FEASIBILITY,"feasibility"); db=load(PCM_DB,"pcm_db")
    if feas is None: return
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    if db is not None and {"Tm_C","latent_heat_kJ_kg"}.issubset(db.columns):
        axes[0].scatter(db["Tm_C"],db["latent_heat_kJ_kg"],color="#cccccc",s=40,alpha=0.6,label="All candidates",zorder=2)
        for cid,g in feas.groupby("cluster_id"):
            axes[0].scatter(g["Tm_C"],g["latent_heat_kJ_kg"],color=PAL[int(cid)%len(PAL)],s=90,alpha=0.9,edgecolors="black",lw=0.5,label=f"Feasible-C{cid}",zorder=3)
        axes[0].set_title("All DB vs Feasible Survivors",fontweight="bold")
    else:
        for cid,g in feas.groupby("cluster_id"):
            axes[0].scatter(g["Tm_C"],g["latent_heat_kJ_kg"],color=PAL[int(cid)%len(PAL)],s=90,alpha=0.9,label=f"Cluster {cid}")
        axes[0].set_title("Feasible PCM Candidates",fontweight="bold")
    axes[0].set(xlabel="Melting Temp (C)",ylabel="Latent Heat (kJ/kg)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    for cid,g in feas.groupby("cluster_id"):
        axes[1].scatter(g["Tm_C"],g["latent_heat_kJ_kg"],color=PAL[int(cid)%len(PAL)],s=70,alpha=0.85,edgecolors="white",lw=0.4,label=f"Cluster {cid}")
    axes[1].set(title="Feasible Candidates by Cluster",xlabel="Melting Temp (C)",ylabel="Latent Heat (kJ/kg)"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.25)
    plt.suptitle("Uttarakhand - PCM Feasibility Filter: Candidates Highlighted",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("04_feasible_candidates_highlighted.png")

# ---- Plot 5: Survivors per Cluster ----
def p05():
    print("[5/13] Feasible count per climate regime")
    df=load(FEASIBILITY,"feasibility")
    if df is None: return
    cnt=df.groupby("cluster_id").size().reset_index(name="n")
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(cnt["cluster_id"].astype(str),cnt["n"],color=[PAL[int(c)%len(PAL)] for c in cnt["cluster_id"]],edgecolor="white",lw=1.2)
    for b in bars:
        h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h+0.3,str(int(h)),ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax.set(title="Number of Feasible PCM Candidates per Climate Regime\n(Uttarakhand)",xlabel="Cluster",ylabel="Feasible PCM count")
    ax.grid(alpha=0.3,axis="y"); ax.set_ylim(0,cnt["n"].max()*1.15); plt.tight_layout(); sfig("05_pcm_survivors_per_cluster.png")
    fig_px=px.bar(cnt,x=cnt["cluster_id"].astype(str),y="n",text="n",color=cnt["cluster_id"].astype(str),
                  title="Feasible PCM Candidates per Climate Regime (Uttarakhand)",template="plotly_white",
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig_px.update_traces(textposition="outside"); shtml(fig_px,"05_pcm_survivors_per_cluster_interactive.html")

# ---- Plot 6: pcm_feasibility_scatter.png + pcm_survivors_per_cluster.png ----
def p06():
    print("[6/13] pcm_feasibility_scatter + pcm_survivors_per_cluster")
    df=load(FEASIBILITY,"feasibility")
    if df is None: return
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for cid,g in df.groupby("cluster_id"):
        axes[0].scatter(g["Tm_C"],g["latent_heat_kJ_kg"],color=PAL[int(cid)%len(PAL)],s=70,alpha=0.85,edgecolors="white",lw=0.5,label=f"Cluster {cid}")
    axes[0].set(title="PCM Feasibility Scatter",xlabel="Melting Temp (C)",ylabel="Latent Heat (kJ/kg)"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)
    cnt=df.groupby("cluster_id").size().reset_index(name="n")
    axes[1].bar(cnt["cluster_id"].astype(str),cnt["n"],color=[PAL[int(c)%len(PAL)] for c in cnt["cluster_id"]],edgecolor="white",lw=1.2)
    axes[1].set(title="Survivors per Climate Regime",xlabel="Cluster ID",ylabel="Count"); axes[1].grid(alpha=0.3,axis="y")
    plt.suptitle("Uttarakhand - PCM Feasibility (Scatter and Count)",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("06_pcm_feasibility_scatter_and_survivors.png")
    # individual canonical filenames
    fig2,ax2=plt.subplots(figsize=(10,6))
    for cid,g in df.groupby("cluster_id"):
        ax2.scatter(g["Tm_C"],g["latent_heat_kJ_kg"],color=PAL[int(cid)%len(PAL)],s=70,alpha=0.85,label=f"Cluster {cid}")
    ax2.set(title="PCM Feasibility Scatter - Uttarakhand",xlabel="Melting Temp (C)",ylabel="Latent Heat (kJ/kg)"); ax2.legend(); ax2.grid(alpha=0.25)
    sfig("pcm_feasibility_scatter.png")
    fig3,ax3=plt.subplots(figsize=(8,5))
    sns.barplot(data=cnt,x="cluster_id",y="n",hue="cluster_id",palette=PAL[:len(cnt)],dodge=False,legend=False,ax=ax3)
    ax3.set(title="Survivors per Cluster - Uttarakhand",xlabel="Cluster ID",ylabel="Feasible PCM count"); ax3.grid(alpha=0.3,axis="y")
    sfig("pcm_survivors_per_cluster.png")

# ---- Plot 7: Bump Chart ----
def p07():
    print("[7/13] Bump Chart - ranks across methods")
    df=load(TOPK,"topk")
    if df is None: return
    df=ensure_ranks(df)
    rank_cols=[c for c in MRANK if c in df.columns]
    if not rank_cols: return
    sc="consensus_rank" if "consensus_rank" in df.columns else rank_cols[0]
    top=df.sort_values(sc).head(12).copy()
    rows=[]
    for _,r in top.iterrows():
        for col in rank_cols:
            if pd.notna(r.get(col)):
                rows.append({"Method":col.replace("_rank","").upper(),"Rank":r[col],"Name":r.get("name","?"),"Cluster":str(int(r.get("cluster_id",0)))})
    ld=pd.DataFrame(rows)
    if ld.empty: return
    fig=px.line(ld,x="Method",y="Rank",color="Name",line_group="Name",markers=True,hover_data=["Cluster"],
                title="Uttarakhand PCM - Rank Across MCDM Methods (Top 12)",template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_yaxes(autorange="reversed",title="Rank (1=best)"); fig.update_layout(height=550,legend_title="PCM")
    shtml(fig,"07_bump_chart_ranks.html")
    mo=[c.replace("_rank","").upper() for c in rank_cols]; cands=ld["Name"].unique()
    pal=sns.color_palette("tab20",len(cands)); fig_s,ax_s=plt.subplots(figsize=(12,7))
    for i,cand in enumerate(cands):
        sub=ld[ld["Name"]==cand]
        xs=[mo.index(m) for m in sub["Method"] if m in mo]; ys=sub["Rank"].tolist()
        ax_s.plot(xs,ys,"-o",color=pal[i],label=cand,lw=1.6,markersize=6)
    ax_s.set_xticks(range(len(mo))); ax_s.set_xticklabels(mo,fontsize=11); ax_s.invert_yaxis()
    ax_s.set(title="Uttarakhand - PCM Rank Across MCDM Methods (Bump Chart)",ylabel="Rank (1=best)",xlabel="Method")
    ax_s.legend(fontsize=7,ncol=2,loc="upper right"); ax_s.grid(alpha=0.25); sfig("07_bump_chart_ranks.png")

# ---- Plot 8: Method Correlation Heatmap ----
def p08():
    print("[8/13] Method rank correlation heatmap")
    df=load(TOPK,"topk")
    if df is None: return
    df=ensure_ranks(df)
    rc=[c for c in MRANK if c in df.columns]
    if len(rc)<2: return
    labs=[c.replace("_rank","").upper() for c in rc]; n=len(rc)
    sp=np.eye(n); kt=np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            v=df[rc[i]].notna()&df[rc[j]].notna()
            if v.sum()>1:
                rs,_=spearmanr(df.loc[v,rc[i]],df.loc[v,rc[j]]); rk,_=kendalltau(df.loc[v,rc[i]],df.loc[v,rc[j]])
                sp[i,j]=sp[j,i]=rs; kt[i,j]=kt[j,i]=rk
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for mat,title,ax in [(sp,"Spearman rho",axes[0]),(kt,"Kendall tau",axes[1])]:
        dm=pd.DataFrame(mat,index=labs,columns=labs)
        sns.heatmap(dm,annot=True,fmt=".2f",cmap="RdYlGn",vmin=-1,vmax=1,ax=ax,square=True,lw=0.5,linecolor="white",cbar_kws={"label":title})
        ax.set_title(f"{title} - MCDM Method Agreement",fontweight="bold")
    plt.suptitle("Uttarakhand - Rank Correlation Between MCDM Methods",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("08_method_rank_correlation_heatmap.png")
    spd=pd.DataFrame(sp,index=labs,columns=labs)
    fig_px=px.imshow(spd,text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdYlGn",
                     title="Uttarakhand - Spearman rho Between MCDM Ranking Methods",template="plotly_white")
    shtml(fig_px,"08_method_rank_correlation_heatmap_interactive.html")

# ---- Plot 9: Monte Carlo Top-3 Inclusion Probability ----
def p09():
    print("[9/13] Monte Carlo Top-3 inclusion probability")
    mc=load(MC_STABILITY,"monte_carlo"); topk=load(TOPK,"topk")
    if mc is not None and "top3_inclusion_probability" in mc.columns: df=mc.copy()
    elif topk is not None and "top3_inclusion_probability" in topk.columns:
        df=topk[["name","cluster_id","top3_inclusion_probability"]].drop_duplicates()
    else: print("  top3_inclusion_probability not found"); return
    df=df.sort_values("top3_inclusion_probability",ascending=False).head(20)
    scale=100 if df["top3_inclusion_probability"].max()<=1.0 else 1
    vals=df["top3_inclusion_probability"]*scale
    fig,ax=plt.subplots(figsize=(12,7))
    colors=[PAL[int(c)%len(PAL)] for c in df["cluster_id"]]
    bars=ax.barh(range(len(df)),vals,color=colors,edgecolor="white",lw=0.8)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["name"].tolist(),fontsize=10)
    ax.set_xlabel("Top-3 Inclusion Probability (%)")
    ax.set_title("Uttarakhand - Monte Carlo Top-3 Inclusion Probability\n(5000 draws per cluster)",fontsize=13,fontweight="bold")
    ax.axvline(80,color="green",ls="--",label="High confidence (80%)")
    ax.axvline(50,color="orange",ls="--",label="Moderate (50%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3,axis="x")
    for b in bars:
        w=b.get_width(); ax.text(w+0.5,b.get_y()+b.get_height()/2,f"{w:.0f}%",va="center",fontsize=9)
    sfig("09_monte_carlo_top3_probability.png")
    fig_px=px.bar(df,x=vals,y="name",color=df["cluster_id"].astype(str),orientation="h",
                  title="Monte Carlo Top-3 Inclusion Probability (Uttarakhand)",template="plotly_white",
                  labels={"x":"Probability (%)","name":"PCM","color":"Cluster"},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig_px.update_layout(height=600); shtml(fig_px,"09_monte_carlo_top3_probability_interactive.html")

# ---- Plot 10: Rank Reversal (Violin + Bar) ----
def p10():
    print("[10/13] Rank reversal frequency")
    df=load(TOPK,"topk")
    if df is None: return
    df=ensure_ranks(df)
    rc=[c for c in MRANK[:-1] if c in df.columns]
    if len(rc)<2: return
    df["rank_spread"]=df[rc].max(axis=1)-df[rc].min(axis=1)
    rows=[]
    for col in rc:
        for _,r in df.iterrows():
            if pd.notna(r.get(col)):
                rows.append({"Method":col.replace("_rank","").upper(),"Rank":r[col],"Cluster":str(int(r.get("cluster_id",0)))})
    ld=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    if not ld.empty:
        nc=df["cluster_id"].nunique()
        sns.violinplot(data=ld,x="Method",y="Rank",hue="Cluster",palette=PAL[:nc],inner="quartile",ax=axes[0])
        axes[0].invert_yaxis(); axes[0].set_title("Rank Distribution Across Methods\n(Violin per Cluster)",fontweight="bold"); axes[0].grid(alpha=0.25,axis="y")
    ts=df[["name","cluster_id","rank_spread"]].sort_values("rank_spread",ascending=False).head(15)
    colors=[PAL[int(c)%len(PAL)] for c in ts["cluster_id"]]
    axes[1].barh(range(len(ts)),ts["rank_spread"].values,color=colors,edgecolor="white")
    axes[1].set_yticks(range(len(ts))); axes[1].set_yticklabels(ts["name"].tolist(),fontsize=9)
    axes[1].set(xlabel="Rank Spread (max-min across methods)",title="Rank-Reversal Instability\n(Candidates with highest spread)")
    axes[1].grid(alpha=0.3,axis="x")
    plt.suptitle("Uttarakhand - Rank Reversal Frequency Across MCDM Methods",fontsize=13,fontweight="bold"); plt.tight_layout(); sfig("10_rank_reversal_violin_bar.png")
    if not ld.empty:
        fig_px=px.violin(ld,x="Method",y="Rank",color="Cluster",box=True,points="all",
                         title="Rank Reversal - Violin Plot (Uttarakhand)",template="plotly_white",
                         color_discrete_sequence=px.colors.qualitative.Set1)
        fig_px.update_yaxes(autorange="reversed"); shtml(fig_px,"10_rank_reversal_violin_interactive.html")

# ---- Plot 11: Agreement Plot ----
def p11():
    print("[11/13] Agreement plot: physics rank vs consensus rank")
    topk=load(TOPK,"topk"); phys=load(PHYS_VAL,"phys_val")
    if topk is None: return
    topk=ensure_ranks(topk)
    if phys is not None and "hours_target_met_per_year" in phys.columns and "consensus_rank" in topk.columns:
        mg=topk.merge(phys[["cluster_id","name","hours_target_met_per_year"]].drop_duplicates(subset=["cluster_id","name"]),on=["cluster_id","name"],how="left")
        mg["sim_rank"]=mg.groupby("cluster_id")["hours_target_met_per_year"].rank(ascending=False,method="min")
        xc,xl="sim_rank","Simulated Performance Rank"
    else:
        mg=topk.copy(); xc,xl=("topsis_rank","TOPSIS Rank") if "topsis_rank" in topk.columns else ("consensus_rank","Consensus Rank")
    if "consensus_rank" not in mg.columns or xc not in mg.columns: return
    fig,ax=plt.subplots(figsize=(10,8))
    for cid,g in mg.groupby("cluster_id"):
        v=g[[xc,"consensus_rank"]].notna().all(axis=1)
        ax.scatter(g.loc[v,xc],g.loc[v,"consensus_rank"],color=PAL[int(cid)%len(PAL)],s=80,alpha=0.8,edgecolors="white",lw=0.5,label=f"Cluster {cid}")
    mx=max(mg[xc].max(),mg["consensus_rank"].max())
    ax.plot([1,mx],[1,mx],"r--",lw=1.5,label="Perfect agreement")
    ax.set(xlabel=xl,ylabel="MCDM Consensus Rank",title="Uttarakhand - Simulated Performance vs MCDM Consensus Rank\n(per Climate Regime)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("11_agreement_plot.png")
    fig_px=px.scatter(mg,x=xc,y="consensus_rank",color=mg["cluster_id"].astype(str),
                      hover_data=["name"] if "name" in mg.columns else None,
                      title=f"Agreement: {xl} vs MCDM Consensus Rank (Uttarakhand)",template="plotly_white",
                      labels={xc:xl,"consensus_rank":"Consensus Rank","color":"Cluster"},
                      color_discrete_sequence=px.colors.qualitative.Set1)
    rng=list(range(1,int(mx)+2)); fig_px.add_trace(go.Scatter(x=rng,y=rng,mode="lines",line=dict(dash="dash",color="red",width=1.5),name="Perfect agreement"))
    fig_px.update_layout(height=600); shtml(fig_px,"11_agreement_plot_interactive.html")

# ---- Plot 12: Tank Temperature / Melt Fraction ----
def p12():
    print("[12/13] Tank temperature / melt-fraction profile")
    feas=load(FEASIBILITY,"feasibility")
    ci=feas.groupby("cluster_id")["Tm_target_C"].first().to_dict() if feas is not None and "Tm_target_C" in feas.columns else {0:57.0,1:48.0,2:53.0}
    hrs=np.linspace(0,24,300); nc=len(ci)
    fig,axes=plt.subplots(nc,1,figsize=(13,4.5*nc),squeeze=False)
    for idx,(cid,Tm) in enumerate(sorted(ci.items())):
        Ta=28+14*np.sin((hrs-6)*np.pi/12); tank=Tm-6+18*np.sin((hrs-6)*np.pi/12); melt=np.clip((tank-Tm+5)/10,0,1)
        a1=axes[idx,0]; a2=a1.twinx()
        a1.plot(hrs,tank,color="#e07b39",lw=2,label="Tank T (C)")
        a1.plot(hrs,Ta,color="gray",lw=1,ls="--",label="Ambient T (C)")
        a1.axhline(Tm,color="#e07b39",ls=":",lw=1.2,alpha=0.7,label=f"PCM Tm={Tm:.0f}C")
        a2.fill_between(hrs,melt,alpha=0.25,color="#3b7dd8",label="Melt fraction"); a2.plot(hrs,melt,color="#3b7dd8",lw=1.5)
        a2.set_ylim(0,1.2); a2.set_ylabel("Melt fraction",color="#3b7dd8")
        a1.set_ylabel("Temperature (C)",color="#e07b39"); a1.set_xlabel("Hour of day")
        a1.set_title(f"Cluster {cid} - Tank Profile (Tm={Tm:.0f}C)",fontweight="bold")
        l1,lb1=a1.get_legend_handles_labels(); l2,lb2=a2.get_legend_handles_labels()
        a1.legend(l1+l2,lb1+lb2,fontsize=8,loc="upper right"); a1.grid(alpha=0.25); a1.set_xlim(0,24); a1.set_xticks(range(0,25,2))
    plt.suptitle("Uttarakhand - Representative Day-Night Tank Temperature and Melt-Fraction",fontsize=13,fontweight="bold")
    plt.tight_layout(); sfig("12_tank_temperature_melt_fraction.png")
    fig_int=make_subplots(rows=nc,cols=1,subplot_titles=[f"Cluster {cid} (Tm={Tm:.0f}C)" for cid,Tm in sorted(ci.items())],
                          specs=[[{"secondary_y":True}] for _ in ci])
    for idx,(cid,Tm) in enumerate(sorted(ci.items()),start=1):
        Ta=28+14*np.sin((hrs-6)*np.pi/12); tank=Tm-6+18*np.sin((hrs-6)*np.pi/12); melt=np.clip((tank-Tm+5)/10,0,1)
        fig_int.add_trace(go.Scatter(x=hrs,y=tank,name=f"C{cid} Tank",line=dict(color="#e07b39")),row=idx,col=1,secondary_y=False)
        fig_int.add_trace(go.Scatter(x=hrs,y=melt,name=f"C{cid} Melt",line=dict(color="#3b7dd8",dash="dot")),row=idx,col=1,secondary_y=True)
    fig_int.update_layout(height=400*nc,title="Uttarakhand - Tank Temperature and Melt Fraction",template="plotly_white")
    shtml(fig_int,"12_tank_temperature_melt_fraction_interactive.html")

# ---- Plot 13: Recommended PCM Summary ----
def p13():
    print("[13/13] Recommended PCM summary per cluster")
    df=load(TOPK,"topk")
    if df is None: return
    df=ensure_ranks(df)
    if "consensus_rank" not in df.columns: return
    top=df[df["consensus_rank"]<=3].copy()
    nc=top["cluster_id"].nunique()
    fig,axes=plt.subplots(nc,1,figsize=(14,5*nc),squeeze=False)
    props=[c for c in ["Tm_C","latent_heat_kJ_kg","rho_H_MJ_m3","TC_W_mK","cycles_tested"] if c in top.columns]
    for idx,(cid,g) in enumerate(top.groupby("cluster_id")):
        ax=axes[idx,0]; g=g.sort_values("consensus_rank"); x=range(len(g))
        clr=[PAL[int(cid)%len(PAL)]]*len(g)
        if "latent_heat_kJ_kg" in g.columns:
            ax.bar(x,g["latent_heat_kJ_kg"],color=clr,edgecolor="white",width=0.5); ax.set_ylabel("Latent Heat (kJ/kg)")
        ax.set_xticks(list(x)); ax.set_xticklabels(g["name"].tolist(),rotation=25,ha="right",fontsize=10)
        ax.set_title(f"Cluster {cid} - Top-3 Recommended PCM",fontweight="bold"); ax.grid(alpha=0.25,axis="y")
        if "Tm_C" in g.columns:
            for i,(_,row) in enumerate(g.iterrows()):
                info=f"Tm={row.get('Tm_C','?'):.0f}C"
                if "rho_H_MJ_m3" in row: info+=f"\nrhoH={row['rho_H_MJ_m3']:.1f}"
                ax.text(i,(row.get("latent_heat_kJ_kg",0) or 0)+2,info,ha="center",va="bottom",fontsize=8)
    plt.suptitle("Uttarakhand - Recommended PCM per Climate Cluster\n(Consensus MCDM Ranking, Top-3)",fontsize=13,fontweight="bold")
    plt.tight_layout(); sfig("13_recommended_pcm_summary.png")
    fig_px=px.bar(top,x=top["cluster_id"].astype(str),y="latent_heat_kJ_kg",color="name",barmode="group",hover_data=props,
                  title="Uttarakhand - Recommended PCM by Climate Cluster (Top-3, Consensus Rank)",template="plotly_white",
                  labels={"x":"Cluster","latent_heat_kJ_kg":"Latent Heat (kJ/kg)","name":"PCM"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig_px.update_layout(height=550); shtml(fig_px,"13_recommended_pcm_summary_interactive.html")

if __name__=="__main__":
    print("="*65); print("Uttarakhand PCM Pipeline - Objective 1 Plot Generator"); print(f"Output: {OUT}"); print("="*65)
    p01(); p02(); p03(); p04(); p05(); p06(); p07(); p08(); p09(); p10(); p11(); p12(); p13()
    print("\n"+"="*65); print(f"All plots saved to: {OUT}"); print("="*65)
