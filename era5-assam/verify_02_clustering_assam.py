"""
Verification Script 02: Clustering & Regional Climate Zoning (Assam)
=====================================================================
Validate GMM / K-Means clustering quality:
- BIC curve and Silhouette selection
- PCA projection (2D/3D representation)
- Geographic cluster assignment map
- Cluster profiles & feature distributions

Output folder: data/plots/verify_clustering/
"""

import os, warnings, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
CLUSTERS = os.path.join(BASE, "data", "processed", "clustering", "cluster_assignments_assam.csv")
PROFILES = os.path.join(BASE, "data", "processed", "clustering", "cluster_profiles_assam.csv")
BIC_CSV  = os.path.join(BASE, "data", "processed", "clustering", "bic_selection_assam.csv")
SIG_CSV  = os.path.join(BASE, "data", "processed", "climate_signatures_raw.csv")
OUT      = os.path.join(BASE, "data", "plots", "verify_clustering")
os.makedirs(OUT, exist_ok=True)

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

def sfig(name):
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {name}")

print("=== [Verify 02] Clustering Verification (Assam) ===")

clu = pd.read_csv(CLUSTERS) if os.path.exists(CLUSTERS) else None
prof = pd.read_csv(PROFILES) if os.path.exists(PROFILES) else None
bic = pd.read_csv(BIC_CSV) if os.path.exists(BIC_CSV) else None
sig = pd.read_csv(SIG_CSV) if os.path.exists(SIG_CSV) else None

# 1. BIC / Silhouette Selection Curves
print("[1/6] BIC & Model Selection Curves")
if bic is not None and "K" in bic.columns:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "#3b7dd8"
    ax1.set_xlabel("Number of Clusters (k)")
    if "BIC" in bic.columns:
        ax1.plot(bic["K"], bic["BIC"], "-o", color=color, label="BIC")
        ax1.set_ylabel("BIC (lower is better)", color=color)
    if "Silhouette" in bic.columns:
        ax2 = ax1.twinx()
        color2 = "#e6194b"
        ax2.plot(bic["K"], bic["Silhouette"], "-s", color=color2, label="Silhouette")
        ax2.set_ylabel("Silhouette Score (higher is better)", color=color2)
    plt.title("Verify Clustering 01: BIC & Silhouette Curves (Assam)")
    sfig("01_elbow_curves.png")

# 2. Silhouette Analysis
print("[2/6] Silhouette Analysis")
if sig is not None and clu is not None:
    clu_clean = clu.copy()
    c_col = "cluster_id" if "cluster_id" in clu_clean.columns else "cluster"
    mg_sil = sig.merge(clu_clean[["point_id", c_col]], on="point_id", how="inner") if "point_id" in sig.columns and "point_id" in clu_clean.columns else sig.copy()
    if c_col not in mg_sil.columns and c_col in clu_clean.columns:
        mg_sil[c_col] = clu_clean[c_col].values[:len(mg_sil)]
    
    CORE_FEATS = ["GHI_mean", "Ta_mean", "DTR", "RH_mean", "wind_mean"]
    fc = [c for c in CORE_FEATS if c in mg_sil.columns]
    if len(fc) == len(CORE_FEATS):
        Xs = StandardScaler().fit_transform(mg_sil[fc].values)
        labs = mg_sil[c_col].astype(int).to_numpy()
        k = len(np.unique(labs))
        sv = silhouette_samples(Xs, labs)
        sa = silhouette_score(Xs, labs)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_lo = 10
        clrs = plt.cm.Set3(np.linspace(0, 1, k))
        for i in range(k):
            vals = np.sort(sv[labs == i])
            y_hi = y_lo + len(vals)
            ax.fill_betweenx(np.arange(y_lo, y_hi), 0, vals, facecolor=clrs[i], edgecolor=clrs[i], alpha=0.7, label=f"Cluster {i}")
            y_lo = y_hi + 10
        ax.axvline(sa, color="red", ls="--", lw=2, label=f"Avg={sa:.3f}")
        ax.axvline(0.4, color="green", ls=":", lw=2, label="Threshold 0.4")
        ax.set(xlabel="Silhouette Coeff", ylabel="Cluster", title=f"Silhouette - Assam clusters (Level A, k={k})")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        sfig("02_silhouette_plot.png")

# 3. PCA Projection
print("[3/6] PCA Projection")
if sig is not None and clu is not None:
    clu_clean = clu.copy()
    if "cluster_id" not in clu_clean.columns and "cluster" in clu_clean.columns:
        clu_clean["cluster_id"] = clu_clean["cluster"]
    if "point_id" in sig.columns and "point_id" in clu_clean.columns:
        mg = sig.merge(clu_clean[["point_id", "cluster_id"]], on="point_id", how="inner")
    else:
        min_len = min(len(sig), len(clu_clean))
        mg = sig.iloc[:min_len].copy()
        mg["cluster_id"] = clu_clean["cluster_id"].iloc[:min_len].values

    num_cols = mg.select_dtypes(include=[np.number]).drop(columns=["cluster_id"], errors="ignore").columns
    if len(num_cols) >= 2:
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(mg[num_cols].fillna(mg[num_cols].mean()))
        fig, ax = plt.subplots(figsize=(9, 6))
        for cid in sorted(mg["cluster_id"].unique()):
            mask = (mg["cluster_id"] == cid).values
            ax.scatter(pcs[mask, 0], pcs[mask, 1], color=PAL[int(cid) % len(PAL)], s=60, alpha=0.8, label=f"Cluster {cid}")
        ax.set(xlabel=f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} var)",
               ylabel=f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} var)",
               title="Verify Clustering 03: PCA Projection of Assam Climate Regimes")
        ax.legend(); ax.grid(alpha=0.25); sfig("03_pca_projection.png")

# 4. Geographic Map
print("[4/6] Geographic Cluster Map")
if clu is not None:
    clu_map = clu.copy()
    if "cluster_id" not in clu_map.columns and "cluster" in clu_map.columns:
        clu_map["cluster_id"] = clu_map["cluster"]
    if not {"lat", "lon"}.issubset(clu_map.columns):
        pop_file = os.path.join(BASE, "data", "processed", "population_grid_points.csv")
        if os.path.exists(pop_file):
            pop = pd.read_csv(pop_file)
            clu_map = clu_map.merge(pop[["point_id", "lat", "lon"]], on="point_id")

    if {"lat", "lon", "cluster_id"}.issubset(clu_map.columns):
        fig, ax = plt.subplots(figsize=(10, 7))
        for cid in sorted(clu_map["cluster_id"].unique()):
            sub = clu_map[clu_map["cluster_id"] == cid]
            ax.scatter(sub["lon"], sub["lat"], color=PAL[int(cid) % len(PAL)], s=70, alpha=0.85, label=f"Cluster {cid}")
        ax.set(xlabel="Longitude E", ylabel="Latitude N", title="Verify Clustering 04: Geographic Cluster Map (Assam)")
        ax.legend(); ax.grid(alpha=0.3); sfig("04_geographic_map.png")

# 5. Cluster Profile Comparison
print("[5/6] Cluster Profiles Bar Chart")
if prof is not None and ("cluster_id" in prof.columns or "cluster" in prof.columns):
    prof_clean = prof.copy()
    c_col = "cluster_id" if "cluster_id" in prof_clean.columns else "cluster"
    cols = [c for c in prof_clean.columns if c not in ["cluster_id", "cluster", "count", "n_points", "pct_points", "total_population"]]
    if cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        prof_clean.set_index(c_col)[cols[:5]].plot(kind="bar", ax=ax, colormap="tab10")
        ax.set(title="Verify Clustering 05: Mean Climate Features by Cluster (Assam)", ylabel="Normalized Feature Value", xlabel="Cluster ID")
        ax.grid(alpha=0.3, axis="y"); plt.tight_layout(); sfig("05_cluster_profiles.png")

# 6. Cluster Sizes
print("[6/6] Cluster Sizes")
if clu is not None:
    c_col = "cluster_id" if "cluster_id" in clu.columns else "cluster"
    sizes = clu[c_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sizes.index.astype(str), sizes.values, color=[PAL[int(c) % len(PAL)] for c in sizes.index], edgecolor="white")
    ax.set(title="Verify Clustering 06: Point Distribution per Cluster (Assam)", xlabel="Cluster ID", ylabel="Number of Grid Points")
    for i, v in enumerate(sizes.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); sfig("06_cluster_sizes.png")

print(f"Verify 02 complete! Outputs saved in: {OUT}")
