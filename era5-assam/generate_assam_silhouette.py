"""
generate_assam_silhouette.py
============================
Generates the exact publication-quality Silhouette Plot for Assam (k=3),
matching the exact design, colormap (Set3), line thresholds, and typography as
Rajasthan's silhouette verification figure.
"""

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLU_PATH = os.path.join(BASE_DIR, "data", "processed", "clustering", "cluster_assignments_assam.csv")
SIG_PATH = os.path.join(BASE_DIR, "data", "processed", "climate_signatures_raw.csv")

OUT_VERIFY = os.path.join(BASE_DIR, "data", "plots", "verify_clustering")
OUT_VISUALS = os.path.join(BASE_DIR, "final_outputs", "visuals")
OUT_PPT = os.path.join(BASE_DIR, "plots_assam_ppt", "4 Climate Region Discovery (Clustering)", "Assam")
ARTIFACT_DIR = r"C:\Users\hp\.gemini\antigravity-ide\brain\20a73da3-105f-40a6-a2ea-5ba94cae2fd3"

os.makedirs(OUT_VERIFY, exist_ok=True)
os.makedirs(OUT_VISUALS, exist_ok=True)
os.makedirs(OUT_PPT, exist_ok=True)
if os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Loading Assam datasets...")
clu = pd.read_csv(CLU_PATH)
sig = pd.read_csv(SIG_PATH)

# Merge on point_id
mg = sig.merge(clu[["point_id", "cluster"]], on="point_id", how="inner")
print(f"Merged samples: {len(mg)}")

# 5 Core physical climate features established in Phase 3 GMM
CORE_FEATS = ["GHI_mean", "Ta_mean", "DTR", "RH_mean", "wind_mean"]
X = mg[CORE_FEATS].values
Xs = StandardScaler().fit_transform(X)
labs = mg["cluster"].astype(int).to_numpy()
k = len(np.unique(labs))

sv = silhouette_samples(Xs, labs)
sa = silhouette_score(Xs, labs)
print(f"Assam Silhouette Score: {sa:.3f} across {k} clusters")
for i in range(k):
    v = sv[labs == i]
    print(f"  Cluster {i}: N={len(v)}, min={v.min():.3f}, mean={v.mean():.3f}, max={v.max():.3f}")

clrs = plt.cm.Set3(np.linspace(0, 1, k))

# =========================================================================
# 1. Clean Standalone Plot (02_silhouette_plot.png)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
y_lo = 10
for i in range(k):
    vals = np.sort(sv[labs == i])
    y_hi = y_lo + len(vals)
    ax.fill_betweenx(
        np.arange(y_lo, y_hi),
        0,
        vals,
        facecolor=clrs[i],
        edgecolor=clrs[i],
        alpha=0.7,
        label=f"Cluster {i}"
    )
    y_lo = y_hi + 10

ax.axvline(sa, color="red", ls="--", lw=2, label=f"Avg={sa:.3f}")
ax.axvline(0.4, color="green", ls=":", lw=2, label="Threshold 0.4")
ax.set_xlabel("Silhouette Coeff", fontsize=11)
ax.set_ylabel("Cluster", fontsize=11)
ax.set_title(f"Silhouette - Assam clusters (Level A, k={k})", fontsize=13)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()

path_verify = os.path.join(OUT_VERIFY, "02_silhouette_plot.png")
path_visuals = os.path.join(OUT_VISUALS, "02_silhouette_plot_assam.png")
path_ppt = os.path.join(OUT_PPT, "02_silhouette_plot.png")
fig.savefig(path_verify, dpi=300, bbox_inches="tight")
fig.savefig(path_visuals, dpi=300, bbox_inches="tight")
fig.savefig(path_ppt, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path_verify}")

# =========================================================================
# 2. Captioned Figures matching user's uploaded Rajasthan image exactly
# =========================================================================
def make_captioned_image(subcap_text, out_filename):
    # Same aspect ratio as uploaded Rajasthan image
    fig, ax = plt.subplots(figsize=(7.5, 7.8), dpi=300)
    plt.subplots_adjust(bottom=0.20, top=0.94, left=0.12, right=0.92)

    y_lo = 10
    for i in range(k):
        vals = np.sort(sv[labs == i])
        y_hi = y_lo + len(vals)
        ax.fill_betweenx(
            np.arange(y_lo, y_hi),
            0,
            vals,
            facecolor=clrs[i],
            edgecolor=clrs[i],
            alpha=0.7,
            label=f"Cluster {i}"
        )
        y_lo = y_hi + 10

    ax.axvline(sa, color="red", ls="--", lw=1.8, label=f"Avg={sa:.3f}")
    ax.axvline(0.4, color="green", ls=":", lw=1.8, label="Threshold 0.4")
    ax.set_xlabel("Silhouette Coeff", fontsize=8)
    ax.set_ylabel("Cluster", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"Silhouette - Assam clusters (Level A, k={k})", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=6.8, framealpha=0.9)
    ax.grid(alpha=0.3)

    # LaTeX Computer Modern subcaption
    fig.text(
        0.52,
        0.05,
        subcap_text,
        ha="center",
        va="center",
        fontsize=24,
        fontfamily="DejaVu Serif"
    )

    out_file = os.path.join(OUT_VISUALS, out_filename)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

    if os.path.exists(ARTIFACT_DIR):
        artifact_copy = os.path.join(ARTIFACT_DIR, out_filename)
        shutil.copy2(out_file, artifact_copy)
        print(f"Copied to artifact: {artifact_copy}")

# Generate with (a), (b), and (d)
make_captioned_image("(a) Assam (k=3)", "silhouette_assam_subcap_a.png")
make_captioned_image("(b) Assam (k=3)", "silhouette_assam_subcap_b.png")
make_captioned_image("(d) Assam (k=3)", "silhouette_assam_subcap_d.png")

# Also copy clean version to artifacts
if os.path.exists(ARTIFACT_DIR):
    shutil.copy2(path_visuals, os.path.join(ARTIFACT_DIR, "02_silhouette_plot_assam_clean.png"))

print("All tasks completed successfully!")
