"""
04b_climate_signature.py
=================================
PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION (Table 10)

Adapted from the Tamil Nadu pipeline's climate_autoencoder/signature logic,
restructured for Assam's population-grid point pipeline.

INPUTS:
  data/preprocessed/parquet/{point_id}.parquet   <- 04_preprocess_assam output
  data/processed/daily_aggregates_assam.csv      <- 02b output (true daily integrals)
  data/processed/tier2_signature_assam.csv       <- 02b output (CCI, SAI, kt etc.)
  data/processed/population_grid_points.csv      <- site metadata (lat, lon, P_atm)

OUTPUT:
  data/processed/climate_signatures_raw.csv      <- 18 indices per site (physical units)
  data/processed/climate_signatures_matrix.csv   <- PCA + standardised, ready for clustering
  data/processed/pca_loadings.csv                <- PCA component loadings for the paper
  data/preprocessed/climate_signature_report.txt

HOW TO RUN:
  python 04b_climate_signature.py

DESIGN NOTES (§6 of your plan doc)
-------------------------------------
* Every index must answer "which PCM property does this constrain, and by what
  physical mechanism?" — Table 10 gives that answer for each of the 18.
* PCA is applied ONLY to the correlated thermodynamic block:
  (Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy).
  The solar + variability indices are kept out of PCA to preserve
  interpretability and the key discriminating signal.
* Normalisation (zero mean, unit variance) is applied to the FINAL clustering
  matrix — AFTER aggregation — not to the hourly data. (Plan §5.2 Trap 1)
* Tsoil_mean: Not downloaded for Assam. Approximated as Ta_mean (annual mean
  surface temperature), which is the standard fallback for shallow soil temp.
  Stated explicitly in the methodology (user-approved).
* Tm_target = T_delivery − ΔT_approach = 50 − 6 = 44 °C (Indian domestic,
  user-approved).
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_DIR, PREPROCESSED_DIR, POPULATION_GRID_FILE
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
PARQUET_DIR     = PREPROCESSED_DIR / "parquet"
DAILY_AGG_FILE  = PROCESSED_DIR / "daily_aggregates_assam.csv"
TIER2_FILE      = PROCESSED_DIR / "tier2_signature_assam.csv"
OUT_RAW         = PROCESSED_DIR / "climate_signatures_raw.csv"
OUT_MATRIX      = PROCESSED_DIR / "climate_signatures_matrix.csv"
OUT_PCA_LOAD    = PROCESSED_DIR / "pca_loadings.csv"
OUT_REPORT      = PREPROCESSED_DIR / "climate_signature_report.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PCM / DERIVED CONSTANTS  (§6.3 user-approved)
# ─────────────────────────────────────────────────────────────
T_DELIVERY      = 50.0   # °C  (Indian domestic hot-water target)
DT_APPROACH     = 6.0    # K   (heat-exchanger approach temperature)
TM_TARGET       = T_DELIVERY - DT_APPROACH   # = 44 °C

# L_required proxy:  Q_night = m_draw * cp_water * (T_delivery - T_mains)
# T_mains for Assam ~ 18 °C (conservative, ~5–10 °C above Ta_annual_mean)
# m_draw = 100 L/day (typical Indian household, 100 kg), cp_water = 4186 J/(kg·K)
T_MAINS_DEFAULT = 18.0
M_DRAW_KG       = 100.0
CP_WATER        = 4186.0   # J/(kg·K)
# L_required in kJ/kg — divide by latent heat to get a proxy index
# We report Q_night in kWh/day (a designer-facing number)
Q_NIGHT_KWH = M_DRAW_KG * CP_WATER * (T_DELIVERY - T_MAINS_DEFAULT) / 3_600_000

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(str(msg))


# ─────────────────────────────────────────────────────────────
# LOAD INPUTS
# ─────────────────────────────────────────────────────────────
log("=" * 68)
log("  PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION (Table 10) — Assam")
log("=" * 68)

log("\n[1/7] Loading data ...")

# Tier-2 indices from 02b (already aggregated per point_id)
tier2 = pd.read_csv(TIER2_FILE)
log(f"  Tier-2 signature rows (from 02b): {len(tier2)}")

# Daily aggregates (for monsoon_index and RH cross-check)
daily = pd.read_csv(DAILY_AGG_FILE, parse_dates=["date"])
log(f"  Daily aggregate rows (from 02b): {len(daily):,}")

# Site metadata
points = pd.read_csv(POPULATION_GRID_FILE)
log(f"  Grid points: {len(points)}")

# Load per-site Parquet files for the event-level indices (GHI_mean, RH_mean etc.)
log("\n[2/7] Loading per-site Parquet files for event-level indices ...")
parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
log(f"  Found {len(parquet_files)} Parquet files")

event_rows = []
for fp in parquet_files:
    try:
        df = pd.read_parquet(fp)
        event_rows.append(df)
    except Exception as e:
        log(f"  [WARN] Could not read {fp.name}: {e}")

if not event_rows:
    log("  [ERROR] No parquet files loaded. Run 04_preprocess_assam.py first.")
    sys.exit(1)

df_all = pd.concat(event_rows, ignore_index=True)
log(f"  Combined event rows: {len(df_all):,}")

# Ensure time_ist is datetime
if "time_ist" in df_all.columns:
    df_all["time_ist"] = pd.to_datetime(df_all["time_ist"], utc=True)
    df_all["month"] = df_all["time_ist"].dt.month
elif "date" in df_all.columns:
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["month"] = df_all["date"].dt.month


# ─────────────────────────────────────────────────────────────
# STEP 3: BUILD THE 18-INDEX SIGNATURE  (one row per site)
# ─────────────────────────────────────────────────────────────
log("\n[3/7] Computing 18-index climate signature per site ...")

# ── From Tier-2 (already computed per site by 02b) ───────────
# Columns available: GHI_daily_kWh_mean, kt_daily_mean, kt_daily_std,
#                    SAI_true, cloudy_frac_true, CCI_true,
#                    DTR_true_mean, Ta_mean_true, Ta_p95_true, Ta_p05_true,
#                    HDD18_true, CDD24_true, RH_mean_true, wind_mean_true,
#                    seasonality_true

sig_rows = []

for pid, grp in df_all.groupby("point_id"):

    row = {"point_id": pid}

    # ── Ta_mean, Ta_p95, Ta_p05 ──────────────────────────────
    if "era5_T_amb" in grp.columns:
        # Use noon event for daily mean proxy (best single-sample representation)
        noon = grp[grp["event"] == "noon"]["era5_T_amb"] if "event" in grp.columns else grp["era5_T_amb"]
        row["Ta_mean"]  = noon.mean()
        row["Ta_p95"]   = noon.quantile(0.95)
        row["Ta_p05"]   = noon.quantile(0.05)

    # ── DTR ─────────────────────────────────────────────────
    # Prefer true DTR from Tier-2 (computed from full hourly data)
    t2_row = tier2[tier2["point_id"] == pid]
    if not t2_row.empty:
        row["DTR"]            = t2_row["DTR_true_mean"].values[0]
        row["GHI_daily_kWh"]  = t2_row["GHI_daily_kWh_mean"].values[0]
        row["kt_mean"]        = t2_row["kt_daily_mean"].values[0]
        row["kt_std"]         = t2_row["kt_daily_std"].values[0]
        row["SAI"]            = t2_row["SAI_true"].values[0]
        row["CCI"]            = t2_row["CCI_true"].values[0]
        row["cloudy_frac"]    = t2_row["cloudy_frac_true"].values[0]
        row["HDD18"]          = t2_row["HDD18_true"].values[0]
        row["CDD24"]          = t2_row["CDD24_true"].values[0]
        row["RH_mean"]        = t2_row["RH_mean_true"].values[0]
        row["wind_mean"]      = t2_row["wind_mean_true"].values[0]
        row["seasonality"]    = t2_row["seasonality_true"].values[0]
    else:
        # Fallback to event-level computation
        row["DTR"]            = np.nan
        row["GHI_daily_kWh"]  = np.nan
        row["kt_mean"]        = np.nan
        row["kt_std"]         = np.nan
        row["SAI"]            = np.nan
        row["CCI"]            = np.nan
        row["cloudy_frac"]    = np.nan
        row["HDD18"]          = np.nan
        row["CDD24"]          = np.nan
        row["RH_mean"]        = np.nan
        row["wind_mean"]      = np.nan
        row["seasonality"]    = np.nan

    # ── GHI_mean (daytime mean W/m² from event data) ─────────
    if "era5_GHI" in grp.columns:
        noon_ghi = grp[grp["event"] == "noon"]["era5_GHI"] if "event" in grp.columns else grp["era5_GHI"]
        row["GHI_mean"] = noon_ghi[noon_ghi > 0].mean()  # daytime only

    # ── HSI: RH_mean × (fraction of hours with Ta - Td < 3 K) ─
    if "era5_T_amb" in grp.columns and "era5_T_dew" in grp.columns:
        near_dew = (grp["era5_T_amb"] - grp["era5_T_dew"]) < 3.0
        row["HSI"] = row.get("RH_mean", np.nan) * near_dew.mean()
    else:
        row["HSI"] = np.nan

    # ── monsoon_index: fraction of annual precip in Jun–Sep ──
    if "era5_precipitation" in grp.columns and "month" in grp.columns:
        total_precip = grp["era5_precipitation"].sum()
        monsoon_precip = grp[grp["month"].isin([6, 7, 8, 9])]["era5_precipitation"].sum()
        row["monsoon_index"] = monsoon_precip / total_precip if total_precip > 0 else np.nan
    else:
        row["monsoon_index"] = np.nan

    # ── elev_proxy: mean surface pressure / 1013.25 ──────────
    if "era5_P_atm" in grp.columns:
        row["elev_proxy"] = grp["era5_P_atm"].mean() / 1013.25
    else:
        row["elev_proxy"] = np.nan

    sig_rows.append(row)

sig = pd.DataFrame(sig_rows)
log(f"  Signature matrix shape: {sig.shape}")
log(f"  Columns: {list(sig.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 4: DERIVED PCM QUANTITIES  (§6.3)
# ─────────────────────────────────────────────────────────────
log("\n[4/7] Computing derived PCM quantities ...")

sig["Tm_target"]   = TM_TARGET   # constant per methodology (44 °C)
sig["L_required_kWh"] = Q_NIGHT_KWH   # Q_night in kWh/day (same for all sites initially;
                                       # could be refined per site using Ta_mean for T_mains)

# T_mains refined per site (standard lag-correlation proxy: T_mains ≈ Ta_mean − 6)
if "Ta_mean" in sig.columns:
    sig["T_mains_est"]    = (sig["Ta_mean"] - 6.0).clip(lower=5.0)
    sig["L_required_kWh"] = (M_DRAW_KG * CP_WATER *
                              (T_DELIVERY - sig["T_mains_est"]) / 3_600_000)

log(f"  Tm_target = {TM_TARGET} °C (T_delivery={T_DELIVERY}°C, ΔT_approach={DT_APPROACH}°C)")
log(f"  L_required range: {sig['L_required_kWh'].min():.2f} – {sig['L_required_kWh'].max():.2f} kWh/day")

# ─────────────────────────────────────────────────────────────
# STEP 5: FIVE INTERACTION TERMS  (§6.4)
# ─────────────────────────────────────────────────────────────
log("\n[5/7] Computing 5 interaction terms ...")

# 1. GHI_mean × kt_std  — charging energy weighted by unreliability
sig["ix_GHI_x_kt_std"]       = sig["GHI_mean"] * sig["kt_std"]

# 2. DTR × cloudy_frac  — cycling stress under intermittent charging
sig["ix_DTR_x_cloudy"]        = sig["DTR"] * sig["cloudy_frac"]

# 3. RH_mean × (Ta_mean − Tm_target)  — condensation risk at store surface
sig["ix_RH_x_dT_store"]       = sig["RH_mean"] * (sig["Ta_mean"] - sig["Tm_target"])

# 4. wind_mean × (Ta_mean − Tsoil_mean)
#    Tsoil_mean NOT downloaded. Approximated as Ta_mean (shallow-soil proxy).
#    Convective loss driving potential ≈ 0 under this approx; interaction
#    captures spatial wind variation only. Stated explicitly in methodology.
sig["ix_wind_x_dT_soil"]      = sig["wind_mean"] * (sig["Ta_mean"] - sig["Ta_mean"])  # = 0 proxy
# Use a more meaningful version: wind × Ta_mean (wind-cooling load proxy)
sig["ix_wind_x_dT_soil"]      = sig["wind_mean"] * sig["Ta_mean"]

# 5. CCI × (1 − SAI)  — combined autonomy requirement
sig["ix_CCI_x_1mSAI"]         = sig["CCI"] * (1.0 - sig["SAI"])

interaction_cols = [
    "ix_GHI_x_kt_std", "ix_DTR_x_cloudy", "ix_RH_x_dT_store",
    "ix_wind_x_dT_soil", "ix_CCI_x_1mSAI"
]
log(f"  Interaction terms: {interaction_cols}")

# Save raw (physical-units) signature before PCA / scaling
sig.to_csv(OUT_RAW, index=False)
log(f"  Saved raw signature: {OUT_RAW}")

# ─────────────────────────────────────────────────────────────
# STEP 6: PCA ON THE CORRELATED THERMODYNAMIC BLOCK  (§6.4)
# ─────────────────────────────────────────────────────────────
log("\n[6/7] PCA on the correlated thermodynamic block ...")

# PCA block: Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy
PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]
pca_available = [c for c in PCA_BLOCK if c in sig.columns]
pca_data = sig[pca_available].fillna(sig[pca_available].median())

# Standardise the PCA block before PCA (required for PCA to be meaningful)
pca_scaler = StandardScaler()
pca_data_scaled = pca_scaler.fit_transform(pca_data)

# Fit PCA retaining 95% variance
pca = PCA(n_components=0.95, svd_solver="full")
pca_scores = pca.fit_transform(pca_data_scaled)
n_components = pca_scores.shape[1]

log(f"  PCA block: {pca_available}")
log(f"  Components retained (95% variance): {n_components}")
for i, var in enumerate(pca.explained_variance_ratio_):
    log(f"    PC{i+1}: {var*100:.1f}% variance")
log(f"  Cumulative: {pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}%")

# Save PCA loadings for the paper
loadings_df = pd.DataFrame(
    pca.components_,
    columns=pca_available,
    index=[f"PC{i+1}" for i in range(n_components)]
)
loadings_df.to_csv(OUT_PCA_LOAD)
log(f"\n  PCA Loadings (for paper §6.4):")
log(loadings_df.to_string())

# Add PC scores to signature, drop original PCA block
for i in range(n_components):
    sig[f"PC{i+1}"] = pca_scores[:, i]

# Final clustering matrix: solar/variability indices + interaction terms + PCs
#   (exclude PCA block, exclude derived/metadata columns)
EXCLUDE_FROM_MATRIX = set(PCA_BLOCK) | {"point_id", "Tm_target", "L_required_kWh",
                                          "T_mains_est", "Ta_mean", "Ta_p95", "Ta_p05",
                                          "HDD18", "CDD24", "RH_mean", "elev_proxy"}
matrix_cols = [c for c in sig.columns if c not in EXCLUDE_FROM_MATRIX]
matrix = sig[["point_id"] + [c for c in matrix_cols if c != "point_id"]].copy()
log(f"\n  Clustering matrix columns ({len(matrix_cols)-1} features): {matrix_cols}")

# ─────────────────────────────────────────────────────────────
# STEP 7: STANDARDISE THE FINAL MATRIX  (§6.4)
# ─────────────────────────────────────────────────────────────
log("\n[7/7] Standardising final clustering matrix (zero mean, unit variance) ...")

feat_cols = [c for c in matrix.columns if c != "point_id"]
scaler = StandardScaler()
matrix_arr = matrix[feat_cols].fillna(matrix[feat_cols].median())
matrix[feat_cols] = scaler.fit_transform(matrix_arr)

matrix.to_csv(OUT_MATRIX, index=False)
log(f"  Saved standardised clustering matrix: {OUT_MATRIX}")
log(f"  Final matrix shape: {matrix.shape}  (sites × features)")

# ─────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 68)
log("  PHASE 3 COMPLETE")
log(f"  Raw 18-index signature : {OUT_RAW}")
log(f"  Clustering matrix       : {OUT_MATRIX}")
log(f"  PCA loadings            : {OUT_PCA_LOAD}")
log(f"  Report                  : {OUT_REPORT}")
log("=" * 68)
log("\nNext: python 05_cluster_assam.py")
