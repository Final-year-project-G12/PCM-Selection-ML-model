"""
Verification Script 2: Clustering Validation - Rajasthan
Output: outputs/objective1_plots_rajasthan/verify_clustering/
Run AFTER: Phase 4 clustering
Checks cluster quality, separation, geographic distribution, profiles.
"""
import os, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score

BASE     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIG_CSV  = os.path.join(BASE,"data","processed","climate_signature_rajasthan.csv")
CLU_CSV  = os.path.join(BASE,"data","processed","cluster_assignments_rajasthan_levelB.csv")
OUT      = os.path.join(BASE,"outputs","objective1_plots_rajasthan","verify_clustering")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
try:
    sig=pd.read_csv(SIG_CSV); clu=pd.read_csv(CLU_CSV)
    print(f"  Signature: {sig.shape}  Clusters: {clu.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}"); raise SystemExit(1)

# Merge on available key
merge_key = "point_id" if "point_id" in sig.columns and "point_id" in clu.columns else \
            "id" if "id" in sig.columns and "id" in clu.columns else None

if merge_key:
    cg=clu[[merge_key,"cluster_id"]].copy()
    if "lat" in clu.columns:
        cg["c_lat"]=clu["lat"]
    if "lon" in clu.columns:
        cg["c_lon"]=clu["lon"]

    mg=sig.merge(cg,on=merge_key,how="inner")

    # Fill lat/lon if available
    if "c_lat" in mg.columns:
        mg["lat"]=mg["lat"].fillna(mg["c_lat"]) if "lat" in mg.columns else mg["c_lat"]
    if "c_lon" in mg.columns:
        mg["lon"]=mg["lon"].fillna(mg["c_lon"]) if "lon" in mg.columns else mg["c_lon"]
else:
    mg = sig.copy()
    mg["cluster_id"] = clu["cluster_id"].values if len(sig)==len(clu) else 0

print(f"  Merged: {mg.shape}")

# Get numeric features (skip non-numeric and metadata columns)
fc=[c for c in mg.columns if c not in {"point_id","cluster_id","lat","lon","c_lat","c_lon","population","id"} and pd.api.types.is_numeric_dtype(mg[c])]
X=mg[fc].fillna(mg[fc].median()).values
Xs=StandardScaler().fit_transform(X)
labs=mg["cluster_id"].astype(int).to_numpy(); k=len(np.unique(labs))
print(f"  Features: {len(fc)}  Clusters: {k}")

# 1. Elbow curves
print("[1/6] Elbow curves...")
k_range=range(2,9); ss=[]; bic=[]; db=[]; ch=[]
for ki in k_range:
    gmm=GaussianMixture(n_components=ki,random_state=42,n_init=10); lbl=gmm.fit_predict(Xs)
    ss.append(silhouette_score(Xs,lbl)); bic.append(gmm.bic(Xs)); db.append(davies_bouldin_score(Xs,lbl)); ch.append(calinski_harabasz_score(Xs,lbl))
fig,axes=plt.subplots(2,2,figsize=(14,10))
for ax,vals,ttl,mk in [(axes[0,0],ss,"Silhouette (higher better)","bo-"),(axes[0,1],bic,"BIC (lower better)","go-"),(axes[1,0],db,"Davies-Bouldin (lower better)","ro-"),(axes[1,1],ch,"Calinski-Harabasz (higher better)","mo-")]:
    ax.plot(k_range,vals,mk,lw=2,ms=8); ax.axvline(k,color="red",ls="--",label=f"Actual k={k}"); ax.set_title(ttl); ax.set_xlabel("k"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"01_elbow_curves.png"),dpi=150); plt.close(); print("  ✓ 01_elbow_curves.png")

# 2. Silhouette
print("[2/6] Silhouette analysis...")
sv=silhouette_samples(Xs,labs); sa=silhouette_score(Xs,labs)
fig,ax=plt.subplots(figsize=(10,8)); y_lo=10; clrs=plt.cm.Set3(np.linspace(0,1,k))
for i in range(k):
    vals=np.sort(sv[labs==i]); y_hi=y_lo+len(vals)
    ax.fill_betweenx(np.arange(y_lo,y_hi),0,vals,facecolor=clrs[i],edgecolor=clrs[i],alpha=0.7,label=f"Cluster {i}"); y_lo=y_hi+10
ax.axvline(sa,color="red",ls="--",lw=2,label=f"Avg={sa:.3f}"); ax.axvline(0.4,color="green",ls=":",lw=2,label="Threshold 0.4")
ax.set(xlabel="Silhouette Coeff",ylabel="Cluster",title=f"Silhouette - Rajasthan clusters (k={k})")
ax.legend(loc="best",fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT,"02_silhouette_plot.png"),dpi=150); plt.close(); print("  ✓ 02_silhouette_plot.png")

# 3. PCA projection
print("[3/6] PCA projection...")
pca=PCA(n_components=2); Xp=pca.fit_transform(Xs)
fig,ax=plt.subplots(figsize=(10,8))
sc=ax.scatter(Xp[:,0],Xp[:,1],c=labs,cmap="tab10",s=50,alpha=0.6,edgecolors="black",lw=0.5)
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",title=f"PCA Projection - Rajasthan clusters (k={k})")
plt.colorbar(sc,ax=ax,label="Cluster"); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT,"03_pca_projection.png"),dpi=150); plt.close(); print("  ✓ 03_pca_projection.png")

# 4. Geographic map
print("[4/6] Geographic distribution...")
cd=mg[["point_id","lat","lon","cluster_id"]].drop_duplicates().dropna(subset=["lat","lon"])
if not cd.empty:
    fig,ax=plt.subplots(figsize=(10,8))
    sc=ax.scatter(cd["lon"],cd["lat"],c=cd["cluster_id"],cmap="tab10",s=60,alpha=0.8,edgecolors="black",lw=0.5)
    ax.set(xlabel="Longitude",ylabel="Latitude",title=f"Geographic Distribution - Rajasthan clusters (k={k})")
    plt.colorbar(sc,ax=ax,label="Cluster"); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"04_geographic_map.png"),dpi=150); plt.close(); print("  ✓ 04_geographic_map.png")
else:
    print("  [WARN] No geographic coordinates available")

# 5. Cluster profiles
print("[5/6] Cluster profiles...")
cm=mg.copy(); cm["cluster_id"]=labs; kf=[c for c in fc if c!="n_days_used"][:6]
fig,axes=plt.subplots(2,3,figsize=(15,10)); axes=axes.flatten()
for i,feat in enumerate(kf):
    sns.boxplot(data=cm,x="cluster_id",y=feat,ax=axes[i],palette="Set2"); axes[i].set_title(f"{feat} by Cluster"); axes[i].grid(alpha=0.3,axis="y")
for j in range(len(kf),len(axes)): axes[j].axis("off")
plt.suptitle("Rajasthan - Cluster Profiles",fontsize=14,y=1.0); plt.tight_layout()
plt.savefig(os.path.join(OUT,"05_cluster_profiles.png"),dpi=150); plt.close(); print("  ✓ 05_cluster_profiles.png")

# 6. Cluster sizes
print("[6/6] Cluster sizes...")
cs=pd.Series(labs).value_counts().sort_index()
fig,ax=plt.subplots(figsize=(10,6))
bars=ax.bar(cs.index,cs.values,color="steelblue",edgecolor="black")
for b in bars:
    h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h,str(int(h)),ha="center",va="bottom",fontsize=10)
ax.set(xlabel="Cluster ID",ylabel="Samples",title=f"Cluster Size Distribution - Rajasthan (k={k})")
ax.grid(alpha=0.3,axis="y"); plt.tight_layout()
plt.savefig(os.path.join(OUT,"06_cluster_sizes.png"),dpi=150); plt.close(); print("  ✓ 06_cluster_sizes.png")

print("\n" + "="*60)
print("CLUSTERING VALIDATION SUMMARY (Rajasthan)")
print("="*60)
print(f"Clusters: {k}  Silhouette: {sa:.3f}  DB: {davies_bouldin_score(Xs,labs):.3f}  CH: {calinski_harabasz_score(Xs,labs):.1f}")
for cid in sorted(np.unique(labs)):
    mv=sv[labs==cid]; print(f"  Cluster {cid}: n={mv.shape[0]}  avg_sil={mv.mean():.3f}")
print("\n✓ All plots saved to:", OUT)
