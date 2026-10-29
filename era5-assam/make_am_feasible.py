import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PCM_DB_CSV = os.path.join(BASE_DIR, "data", "processed", "pcm", "pcm_database_assam.csv")
if not os.path.exists(PCM_DB_CSV):
    PCM_DB_CSV = os.path.join(BASE_DIR, "data", "processed", "pcm", "pcm_database_final.csv")

FEAS_CSV = os.path.join(BASE_DIR, "data", "processed", "pcm", "feasibility_survivors_assam.csv")

pcm_db = pd.read_csv(PCM_DB_CSV)
feas_df = pd.read_csv(FEAS_CSV)

if "passes_all" in feas_df.columns:
    feas_surv = feas_df[feas_df["passes_all"] == True].copy()
else:
    feas_surv = feas_df.copy()

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Panel 1: All DB vs Feasible Survivors
axes[0].scatter(
    pcm_db["Tm_C"], pcm_db["latent_heat_kJ_kg"],
    color="#cccccc", s=45, alpha=0.6, label="All candidates", zorder=2
)

for cid, g in feas_surv.groupby("cluster_id"):
    axes[0].scatter(
        g["Tm_C"], g["latent_heat_kJ_kg"],
        color=PAL[int(cid) % len(PAL)], s=85, alpha=0.9,
        edgecolors="black", lw=0.5, label=f"Feasible-C{cid}", zorder=3
    )

axes[0].set_title("All DB vs Feasible Survivors", fontweight="bold", fontsize=12)
axes[0].set_xlabel("Melting Temp (°C)", fontsize=11)
axes[0].set_ylabel("Latent Heat (kJ/kg)", fontsize=11)
axes[0].legend(fontsize=9, loc="lower left")
axes[0].grid(True, alpha=0.25, linestyle="--")

# Panel 2: Feasible Candidates by Cluster
for cid, g in feas_surv.groupby("cluster_id"):
    axes[1].scatter(
        g["Tm_C"], g["latent_heat_kJ_kg"],
        color=PAL[int(cid) % len(PAL)], s=75, alpha=0.85,
        edgecolors="white", lw=0.5, label=f"Cluster {cid}", zorder=3
    )

axes[1].set_title("Feasible Candidates by Cluster", fontweight="bold", fontsize=12)
axes[1].set_xlabel("Melting Temp (°C)", fontsize=11)
axes[1].set_ylabel("Latent Heat (kJ/kg)", fontsize=11)
axes[1].legend(fontsize=9, loc="lower left")
axes[1].grid(True, alpha=0.25, linestyle="--")

plt.suptitle("Assam - PCM Feasibility Filter: Candidates Highlighted", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save outputs
out_path_local = os.path.join(BASE_DIR, "am_feasible.png")
plt.savefig(out_path_local, dpi=300, bbox_inches="tight")

artifact_dir = r"C:\Users\hp\.gemini\antigravity-ide\brain\c45636a5-63f2-4e34-9226-da786b9bde58"
out_path_artifact = os.path.join(artifact_dir, "am_feasible.png")
plt.savefig(out_path_artifact, dpi=300, bbox_inches="tight")

print(f"Successfully generated am_feasible.png:")
print(f"  Local path: {out_path_local}")
print(f"  Artifact path: {out_path_artifact}")
