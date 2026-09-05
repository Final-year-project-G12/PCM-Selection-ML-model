"""
04b_climate_signature.py  (v3.0 — Tier 1 + Tier 2 merged)
============================================================
PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION (Objective 1 plan v3.0, Section 6)

CHANGE FROM THE EARLIER VERSION OF THIS SCRIPT
-------------------------------------------------
The earlier version only used the 3-events/day merged CSV and approximated
GHI_daily_kWh with a half-sine formula, and DTR as (noon - sunrise). Those
are proxies, not measurements, and the plan doc (v3.0 Section 4.3, "Repair
1") is explicit that this is the single highest-value remaining data task.

This version REQUIRES 02b_build_daily_aggregates.py to have been run first.
It merges that script's tier2_signature_tamilnadu.csv (true daily GHI
integral, true DTR, true HDD18/CDD24, true cloudy_frac/CCI, computed from
the FULL NASA POWER hourly cache) onto the Tier-1 sun-event indices below.
Wherever a true Tier-2 value exists it is used as the canonical signature
column; the old sun-event-only proxy is KEPT alongside with a `_proxy`
suffix, purely for your methodology write-up ("proxy vs. true, they agree
to within X%" is a good sentence to be able to write).

monsoon_index remains proxy-only (see 02b's docstring: NASA POWER precip
was never downloaded) — say so plainly if you cite it.

INPUT  : data/preprocessed/tamilnadu_cleaned_physical.csv   (04's Phase-2 output)
         data/processed/tier2_signature_tamilnadu.csv       (02b's output)
OUTPUT : data/processed/signatures/
           climate_signature_tamilnadu.csv   <- one row per point_id: Tier 1
                                                  + Tier 2 + interactions +
                                                  PCA components + Tm_target/
                                                  L_required + standardized copy
           pca_loadings.csv
           signature_correlation_heatmap.png
           signature_distributions.png
           point_signature_map.png

HOW TO RUN:
  python 04b_climate_signature.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import PREPROCESSED_DIR, PROCESSED_DIR, SHARE_PCM

SIGNATURE_DIR = PROCESSED_DIR / "signatures"
SIGNATURE_DIR.mkdir(parents=True, exist_ok=True)

PHYSICAL_FILE = PREPROCESSED_DIR / "tamilnadu_cleaned_physical.csv"
TIER2_FILE = PROCESSED_DIR / "tier2_signature_tamilnadu.csv"

# --- Tm_target rule, v2.0/v3.0 CORRECTED (unchanged from before) ---------
T_DELIVERY_C = 50.0
DT_APPROACH_C = 7.0
TM_TARGET_C = T_DELIVERY_C + DT_APPROACH_C   # 57 C, indirect-system assumption

# --- HOT-WATER DRAW SIZING (BUG FIX v3.1) -----------------------------------
# Previous code: DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60  → 0.001 kg/s (WRONG)
#   This converts 60 L/min → m³/s but OMITS the ×1000 kg/m³ density factor,
#   giving a night draw of only 25.2 kg over 7 h instead of a domestic-scale
#   draw.  That makes L_required ≈ 52 kJ/kg, rendering the latent-heat floor
#   (0.7 × 52 = 36 kJ/kg) a no-op — every PCM in the database clears it.
#
# FIX: use a flat domestic daily draw volume (Avargani et al. 2021: 300 L/day)
# distributed over a 7-hour overnight storage window.  Mass = 300 kg (water
# density 1 kg/L).  This is consistent with the Rajasthan pipeline and with
# the 10_physics_validation.py draw schedule (75 kg × 2 draws = 150 kg/day
# for the smaller validation tank — the signature uses a full household draw).
DRAW_VOLUME_L   = 300.0               # litres per day (domestic household)
DRAW_MASS_KG    = DRAW_VOLUME_L * 1.0  # kg (density of water ≈ 1 kg/L)
DRAW_HOURS      = 7.0                  # overnight storage window (hours)
CP_WATER        = 4.186               # kJ/(kg·K)
ASSUMED_PCM_MASS_KG = 50.0

KT_CLOUDY_THRESHOLD = 0.35

print("=" * 68)
print("  PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION (Tier1+Tier2) — Tamil Nadu")
print(f"  Input  : {PHYSICAL_FILE}")
print(f"  Tier 2 : {TIER2_FILE}")
print(f"  Output : {SIGNATURE_DIR}/")
print(f"  Tm_target = {T_DELIVERY_C} + {DT_APPROACH_C} = {TM_TARGET_C:.0f} C")
print("=" * 68)

if not TIER2_FILE.exists():
    raise FileNotFoundError(
        f"{TIER2_FILE} not found. Run 02b_build_daily_aggregates.py first — "
        "it reads the NASA POWER hourly cache already on disk and produces "
        "this file. This script cannot proceed without it (plan v3.0 Repair 1).")

df = pd.read_csv(PHYSICAL_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=["sunrise", "noon", "sunset"], ordered=True)
tier2 = pd.read_csv(TIER2_FILE)
print(f"\n  Loaded: {len(df):,} rows, {df['point_id'].nunique()} points (Tier 1)")
print(f"  Loaded: {len(tier2)} points (Tier 2)")


def daily_frame(point_df):
    piv = point_df.pivot_table(index="date", columns="event",
                                values=["era5_T_amb", "era5_GHI", "era5_CSI",
                                        "era5_RHum", "era5_precipitation",
                                        "era5_T_dew"],
                                observed=True)
    piv.columns = [f"{v}_{e}" for v, e in piv.columns]
    return piv.reset_index()


def build_signature_tier1(point_id, point_df):
    """Sun-event-only (Tier 1) indices — same as before, proxies flagged."""
    d = daily_frame(point_df)
    row = {"point_id": point_id}

    ta_cols = [c for c in ["era5_T_amb_sunrise", "era5_T_amb_noon", "era5_T_amb_sunset"]
               if c in d.columns]
    d["Ta_daily_mean"] = d[ta_cols].mean(axis=1)
    row["Ta_mean_proxy"] = d["Ta_daily_mean"].mean()
    row["Ta_p95_proxy"] = d["Ta_daily_mean"].quantile(0.95)
    row["Ta_p05_proxy"] = d["Ta_daily_mean"].quantile(0.05)

    if {"era5_T_amb_noon", "era5_T_amb_sunrise"}.issubset(d.columns):
        d["DTR_proxy"] = d["era5_T_amb_noon"] - d["era5_T_amb_sunrise"]
        row["DTR_proxy"] = d["DTR_proxy"].mean()
    else:
        row["DTR_proxy"] = np.nan

    noon = point_df[point_df["event"] == "noon"].set_index("date")
    row["GHI_mean"] = noon["era5_GHI"].mean()

    sr = point_df[point_df["event"] == "sunrise"].set_index("date")["time_utc"]
    ss = point_df[point_df["event"] == "sunset"].set_index("date")["time_utc"]
    daylen_hours = (pd.to_datetime(ss) - pd.to_datetime(sr)).dt.total_seconds() / 3600.0
    daylen_hours = daylen_hours.reindex(noon.index)
    ghi_kw = noon["era5_GHI"] / 1000.0
    daily_kwh = (2.0 / np.pi) * ghi_kw * daylen_hours
    row["GHI_daily_kWh_proxy"] = daily_kwh.mean()

    row["kt_mean_proxy"] = noon["era5_CSI"].mean()
    row["kt_std_proxy"] = noon["era5_CSI"].std()

    ghi_sum = point_df["era5_GHI"].sum()
    ghics_sum = point_df["era5_GHI_clearsky"].sum()
    row["SAI_proxy"] = ghi_sum / ghics_sum if ghics_sum > 0 else np.nan

    kt_daily = noon["era5_CSI"].reindex(pd.date_range(noon.index.min(), noon.index.max(), freq="D"))
    is_cloudy = kt_daily < KT_CLOUDY_THRESHOLD
    row["cloudy_frac_proxy"] = is_cloudy.mean()
    run_lengths = is_cloudy.astype(int).groupby(
        (is_cloudy != is_cloudy.shift()).cumsum()).transform("sum")
    row["CCI_proxy"] = int((run_lengths * is_cloudy.astype(int)).max()) if len(run_lengths) else 0

    # Annualise (same fix as 02b_build_daily_aggregates.py's Tier-2 version) —
    # sum-over-10-years would silently be ~10x a real annual HDD/CDD figure.
    n_years_here = pd.to_datetime(d["date"]).dt.year.nunique()
    n_years_here = max(n_years_here, 1)
    row["HDD18_proxy"] = np.maximum(0, 18 - d["Ta_daily_mean"]).sum() / n_years_here
    row["CDD24_proxy"] = np.maximum(0, d["Ta_daily_mean"] - 24).sum() / n_years_here

    row["RH_mean"] = point_df["era5_RHum"].mean()
    t_dep = point_df["era5_T_amb"] - point_df["era5_T_dew"]
    row["HSI"] = row["RH_mean"] * (t_dep < 3).mean()

    row["wind_mean"] = point_df["era5_W_spd"].mean()

    monthly_ghi = noon.groupby(noon.index.month)["era5_GHI"].mean()
    row["seasonality_proxy"] = monthly_ghi.std() / monthly_ghi.mean() if monthly_ghi.mean() > 0 else np.nan

    # monsoon_index stays proxy-only (see module docstring) — ERA5 precip,
    # 3x/day sampled, JJAS fraction not absolute total.
    precip = point_df.set_index("date")["era5_precipitation"]
    jjas = precip[precip.index.month.isin([6, 7, 8, 9])].sum()
    total = precip.sum()
    row["monsoon_index"] = jjas / total if total > 0 else np.nan

    row["elev_proxy"] = point_df["era5_P_atm"].mean() / 1013.25

    row["lat"] = point_df["lat"].iloc[0]
    row["lon"] = point_df["lon"].iloc[0]
    row["population"] = point_df["population"].iloc[0]

    return row


print("\n[1/6] Building Tier-1 (sun-event) signature vectors ...")
rows = []
points = df["point_id"].unique()
for i, pid in enumerate(points, start=1):
    point_df = df[df["point_id"] == pid]
    rows.append(build_signature_tier1(pid, point_df))
    if i % 20 == 0 or i == len(points):
        print(f"  [{i}/{len(points)}] {pid}")

sig = pd.DataFrame(rows).set_index("point_id")
print(f"\n  Tier-1 matrix: {sig.shape[0]} points x {sig.shape[1]} columns")

# ═══════════════════════════════════════════════════════════
print("\n[2/6] Merging Tier-2 (true daily-integral) indices ...")
tier2_indexed = tier2.set_index("point_id")
sig = sig.join(tier2_indexed, how="left")
n_missing_tier2 = sig["GHI_daily_kWh_mean"].isna().sum() if "GHI_daily_kWh_mean" in sig.columns else len(sig)
print(f"  Points with Tier-2 coverage: {len(sig) - n_missing_tier2}/{len(sig)}")
if n_missing_tier2 > 0:
    print(f"  [WARN] {n_missing_tier2} points have no Tier-2 row (missing/short POWER cache) "
          f"— their canonical columns below fall back to the Tier-1 proxy.")

# Canonical columns: true value where available, else the sun-event proxy.
CANON_MAP = {
    "GHI_daily_kWh": "GHI_daily_kWh_mean",
    "DTR": "DTR_true_mean",
    "kt_mean": "kt_daily_mean",
    "kt_std": "kt_daily_std",
    "SAI": "SAI_true",
    "cloudy_frac": "cloudy_frac_true",
    "CCI": "CCI_true",
    "HDD18": "HDD18_true",
    "CDD24": "CDD24_true",
    "Ta_mean": "Ta_mean_true",
    "Ta_p95": "Ta_p95_true",
    "Ta_p05": "Ta_p05_true",
    "seasonality": "seasonality_true",
}
for canon, true_col in CANON_MAP.items():
    proxy_col = f"{canon}_proxy"
    if true_col in sig.columns:
        sig[canon] = sig[true_col].where(sig[true_col].notna(),
                                          sig.get(proxy_col, np.nan))
    else:
        sig[canon] = sig.get(proxy_col, np.nan)

print("  Canonical columns set (true Tier-2 value preferred, sun-event proxy as fallback):")
print(f"    {list(CANON_MAP.keys())}")

# ═══════════════════════════════════════════════════════════
print("\n[3/6] Derived PCM-facing quantities (Tm_target, L_required) ...")

sig["Tm_target_C"] = TM_TARGET_C
sig["T_mains_est_C"] = sig["Ta_mean"] - 2.0

# L_required = PCM-specific latent heat target (kJ/kg PCM).
# OPTION A (2026-08-31): SHARE_PCM = 0.5 — PCM supplies ~50% of overnight
# delivery; tank sensible heat + concurrent charging supply the remainder
# (Zhao 2022; Huang 2020; Abdelsalam 2020; Koželj 2021). See 07_PHASE_5_AUDIT.md.
q_total_kJ = DRAW_MASS_KG * CP_WATER * (T_DELIVERY_C - sig["T_mains_est_C"])
sig["L_required_kJ_per_kg"] = (q_total_kJ * SHARE_PCM) / ASSUMED_PCM_MASS_KG
print(f"  Tm_target: constant {TM_TARGET_C:.0f} C across all points")
print(f"  Draw volume: {DRAW_VOLUME_L:.0f} L ({DRAW_MASS_KG:.0f} kg), "
      f"delivery at {T_DELIVERY_C:.0f} C, PCM mass {ASSUMED_PCM_MASS_KG:.0f} kg, "
      f"SHARE_PCM={SHARE_PCM}")
print(f"  L_required range: {sig['L_required_kJ_per_kg'].min():.0f} - "
      f"{sig['L_required_kJ_per_kg'].max():.0f} kJ/kg")

# ═══════════════════════════════════════════════════════════
print("\n[4/6] Interaction terms ...")

sig["int_GHI_x_ktstd"] = sig["GHI_daily_kWh"] * sig["kt_std"]
sig["int_DTR_x_cloudyfrac"] = sig["DTR"] * sig["cloudy_frac"]
sig["int_RH_x_TaMinusTm"] = sig["RH_mean"] * (sig["Ta_mean"] - sig["Tm_target_C"])
sig["Tsoil_proxy_C"] = sig["Ta_mean"] - 3.0
sig["int_wind_x_TaMinusTsoil"] = sig["wind_mean"] * (sig["Ta_mean"] - sig["Tsoil_proxy_C"])
sig["int_CCI_x_1minusSAI"] = sig["CCI"] * (1 - sig["SAI"])
print("  Added 5 interaction terms")

# ═══════════════════════════════════════════════════════════
print("\n[5/6] PCA on the correlated temperature/pressure block ...")

PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]
pca_input = sig[PCA_BLOCK].fillna(sig[PCA_BLOCK].median())
pca_scaler = StandardScaler()
pca_input_scaled = pca_scaler.fit_transform(pca_input)

pca = PCA(n_components=0.95, random_state=42)
pca_scores = pca.fit_transform(pca_input_scaled)
n_comp = pca_scores.shape[1]
for i in range(n_comp):
    sig[f"PC{i+1}"] = pca_scores[:, i]

loadings = pd.DataFrame(pca.components_.T, index=PCA_BLOCK,
                         columns=[f"PC{i+1}" for i in range(n_comp)])
loadings.to_csv(SIGNATURE_DIR / "pca_loadings.csv")
print(f"  {n_comp} components retained (95% variance)")
print(loadings.round(3).to_string())
print(f"  Explained variance ratio: {np.round(pca.explained_variance_ratio_, 3)}")

# Clustering matrix excludes raw PCA_BLOCK cols (now redundant with PC1..PCn),
# excludes lat/lon (never cluster on geography — plan v3.0 Section 6.2),
# excludes proxy/true duplicate columns (only the canonical version clusters).
DROP_FROM_CLUSTERING = set(PCA_BLOCK) | {"lat", "lon", "population",
                                          "T_mains_est_C", "Tsoil_proxy_C"}
DROP_FROM_CLUSTERING |= {c for c in sig.columns if c.endswith("_proxy")}
DROP_FROM_CLUSTERING |= {c for c in sig.columns if c.endswith("_true") or c.endswith("_true_mean")}
sig_for_clustering_cols = [c for c in sig.columns if c not in DROP_FROM_CLUSTERING]

# ═══════════════════════════════════════════════════════════
print("\n[6/6] Standardizing the site-level signature matrix ...")

std_scaler = StandardScaler()
sig_std = sig[sig_for_clustering_cols].fillna(sig[sig_for_clustering_cols].median())
sig_std_vals = std_scaler.fit_transform(sig_std)
sig_standardized = pd.DataFrame(sig_std_vals, index=sig.index,
                                 columns=[f"{c}_z" for c in sig_for_clustering_cols])

full_out = sig.join(sig_standardized)
out_path = SIGNATURE_DIR / "climate_signature_tamilnadu.csv"
full_out.to_csv(out_path)
print(f"  Final signature matrix: {full_out.shape[0]} points x {full_out.shape[1]} columns")
print(f"  Saved: {out_path}")

# ═══════════════════════════════════════════════════════════
print("\nDiagnostic plots ...")

INDEX_COLS = ["Ta_mean", "Ta_p95", "Ta_p05", "DTR", "GHI_daily_kWh",
              "kt_mean", "kt_std", "SAI", "CCI", "cloudy_frac", "HDD18", "CDD24",
              "RH_mean", "HSI", "wind_mean", "seasonality", "monsoon_index", "elev_proxy"]
INDEX_COLS = [c for c in INDEX_COLS if c in sig.columns]

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(sig[INDEX_COLS].corr(), ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, annot_kws={"size": 7})
ax.set_title("Climate Signature Correlation (Tier1+Tier2 canonical) — Tamil Nadu points")
plt.tight_layout()
plt.savefig(SIGNATURE_DIR / "signature_correlation_heatmap.png", dpi=140, bbox_inches="tight")
plt.close()

n_idx = len(INDEX_COLS)
ncols = 4
nrows = int(np.ceil(n_idx / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
axes = axes.flatten()
for ax, col in zip(axes, INDEX_COLS):
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
plt.tight_layout()
plt.savefig(SIGNATURE_DIR / "signature_distributions.png", dpi=130, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, col, title in zip(axes, ["GHI_daily_kWh", "monsoon_index"],
                           ["True daily GHI (kWh/m^2/day)", "Monsoon Index (JJAS fraction, proxy)"]):
    sc = ax.scatter(sig["lon"], sig["lat"], c=sig[col], cmap="viridis", s=40,
                     edgecolors="white", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(SIGNATURE_DIR / "point_signature_map.png", dpi=140, bbox_inches="tight")
plt.close()

print("  Saved: signature_correlation_heatmap.png, signature_distributions.png, "
      "point_signature_map.png")

print("\n" + "=" * 68)
print("  PHASE 3 COMPLETE (Tier 1 + Tier 2)")
print(f"  Output: {out_path}")
print("=" * 68)
print("\nNext: run 05_cluster_tamilnadu.py (Phase 4, within-TN only).")
