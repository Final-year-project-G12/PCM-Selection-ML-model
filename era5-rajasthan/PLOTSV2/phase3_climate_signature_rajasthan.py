"""
Phase 3 (Climate Feature Engineering / Climate Signature) plots - Rajasthan
Output: PLOTSV2/phase3_climate_signature/

Port of the diagnostic-plot block at the end of
era5-uttarakhand/04b_climate_signature.py (and its Tamil Nadu twin) - same
three figures, same style, same filenames:

    signature_correlation_heatmap.png
    signature_distributions.png
    point_signature_map.png

Rajasthan's 04_climate_signature_rajasthan.py writes the equivalent figures as
interactive Plotly HTML (outputs/signature_*.html), so there was no PNG to put
in the curated Plots folder beside the other two states. This produces them.

Reads the finished signature matrix rather than recomputing it - this script is
purely a plotter and never writes back to climate_signature_rajasthan.csv.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIG_CSV = os.path.join(BASE, "data", "processed", "climate_signature_rajasthan.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_climate_signature")
os.makedirs(OUT, exist_ok=True)

# Uttarakhand's canonical Tier1+Tier2 index list, mapped onto Rajasthan's
# column names (this pipeline spells out which event each index is measured
# at, e.g. RH_sunrise_mean where Uttarakhand writes RH_mean).
INDEX_COLS = [
    "Ta_mean", "Ta_p95", "Ta_p05", "DTR_true", "GHI_daily_kWh",
    "kt_daily_mean", "kt_daily_std", "SAI", "CCI", "cloudy_frac", "HDD18", "CDD24",
    "RH_sunrise_mean", "HSI_sunrise", "wind_noon_mean", "seasonality",
    "monsoon_index", "elevation_m",
]

print("=" * 68)
print("  Phase 3 Climate-Signature Diagnostic Plots - Rajasthan")
print(f"  Input  : {SIG_CSV}")
print(f"  Output : {OUT}")
print("=" * 68)

sig = pd.read_csv(SIG_CSV)
print(f"\n  Signature matrix: {sig.shape[0]} points x {sig.shape[1]} columns")

missing = [c for c in INDEX_COLS if c not in sig.columns]
if missing:
    print(f"  [WARN] not in this signature file, skipped: {missing}")
idx = [c for c in INDEX_COLS if c in sig.columns]
print(f"  Indices plotted: {len(idx)}")

# ------------------------------------------------- 1. Correlation heatmap
print("\n[1/3] Signature correlation heatmap ...")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(sig[idx].corr(), ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, annot_kws={"size": 7})
ax.set_title("Climate Signature Correlation (Tier1+Tier2 canonical) - Rajasthan points")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "signature_correlation_heatmap.png"), dpi=140, bbox_inches="tight")
plt.close()
print("  Saved: signature_correlation_heatmap.png")

# --------------------------------------------------- 2. Index distributions
print("[2/3] Signature distributions ...")
n_idx = len(idx)
ncols = 4
nrows = int(np.ceil(n_idx / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
axes = axes.flatten()
for ax, col in zip(axes, idx):
    vals = sig[col].dropna()
    if len(vals) == 0:
        ax.set_title(f"{col} (no data)", fontsize=10)
    elif np.isclose(vals.min(), vals.max()):
        ax.bar([0], [len(vals)], color="#4c72b0", alpha=0.8)
        ax.set_title(f"{col}\n(constant = {vals.iloc[0]:.3g})", fontsize=9)
    else:
        ax.hist(vals, bins=20, color="#4c72b0", alpha=0.8)
        ax.set_title(col, fontsize=10)
    ax.grid(alpha=0.3)
for ax in axes[n_idx:]:
    ax.axis("off")
plt.suptitle("Rajasthan - Climate Signature Index Distributions", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "signature_distributions.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: signature_distributions.png")

# ------------------------------------------------------ 3. Point signature map
print("[3/3] Point signature map ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, col, title in zip(axes, ["GHI_daily_kWh", "monsoon_index"],
                          ["True daily GHI (kWh/m^2/day)", "Monsoon Index (JJAS fraction, proxy)"]):
    sc = ax.scatter(sig["lon"], sig["lat"], c=sig[col], cmap="viridis", s=40,
                    edgecolors="white", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
plt.suptitle("Rajasthan - Climate Signature Across Sampling Points", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "point_signature_map.png"), dpi=140, bbox_inches="tight")
plt.close()
print("  Saved: point_signature_map.png")

print("\n" + "=" * 68)
print("  DONE - PNGs in", OUT)
print("=" * 68)
