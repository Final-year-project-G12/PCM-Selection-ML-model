"""
04b_climate_signature.py
=================================
PHASE 2 — CLIMATE SIGNATURE CONSTRUCTION & PCA (Assam Project)

Constructs ONE climate-signature vector for each of the 129 population grid points in Assam.

PHYSICAL RIGOR & TERMINOLOGY CORRECTIONS:
------------------------------------------
1. Explicitly distinguishes 3-event daytime sample statistics (sunrise, noon, sunset)
   from true 24-hour daily integrals.
2. Applies PCA ONLY to the correlated thermodynamic block:
   (Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy).
   Solar resource and variability features remain uncompressed to preserve key discriminating signals.
3. Guarantees 100% point retention: exactly 129 grid points in raw signature and matrix.

INPUTS:
  data/preprocessed/assam_cleaned_physical.csv
  data/processed/population_grid_points.csv

OUTPUTS:
  data/processed/climate_signatures_raw.csv      (129 rows, physical units)
  data/processed/climate_signatures_matrix.csv   (129 rows, standardized for clustering)
  data/processed/pca_loadings.csv                (PCA component loadings)
  data/preprocessed/climate_signature_report.txt
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent
PHYSICAL_FILE = BASE_DIR / "data" / "preprocessed" / "assam_cleaned_physical.csv"
GRID_POINTS_FILE = BASE_DIR / "data" / "processed" / "population_grid_points.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW = PROCESSED_DIR / "climate_signatures_raw.csv"
OUT_MATRIX = PROCESSED_DIR / "climate_signatures_matrix.csv"
OUT_PCA_LOAD = PROCESSED_DIR / "pca_loadings.csv"
OUT_REPORT = PREPROCESSED_DIR / "climate_signature_report.txt"

# Design Parameters (§6.3)
T_DELIVERY = 50.0       # °C (Indian domestic hot-water target)
DT_APPROACH = 6.0       # K  (Heat exchanger approach temperature)
TM_TARGET = T_DELIVERY - DT_APPROACH # = 44 °C
M_DRAW_KG = 100.0       # kg/day (100 L/day household hot water demand)
CP_WATER = 4186.0       # J/(kg·K)

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(str(msg))

def main():
    log("=" * 72)
    log("  PHASE 2 — CLIMATE SIGNATURE CONSTRUCTION & PCA (Assam)")
    log("=" * 72)

    # 1. Load Site Metadata (Master 129 points)
    points_df = pd.read_csv(GRID_POINTS_FILE)
    expected_pids = list(points_df["point_id"].unique())
    log(f"\n[1] Master population grid points loaded: {len(expected_pids)}")

    # 2. Load Preprocessed Cleaned Physical Dataset (Optimized with target columns)
    log("\n[2] Loading physical dataset (assam_cleaned_physical.csv)...")
    target_cols = [
        "point_id", "date", "event", "era5_T_amb", "era5_T_dew", "era5_RHum",
        "era5_W_spd", "era5_GHI", "era5_GHI_clearsky", "era5_CSI", "era5_P_atm",
        "era5_precipitation", "month"
    ]
    df = pd.read_csv(PHYSICAL_FILE, usecols=lambda c: c in target_cols)
    log(f"  Total records loaded: {len(df):,}")
    log(f"  Unique points in physical dataset: {df['point_id'].nunique()}")

    # Ensure month column exists
    if "date" in df.columns and "month" not in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month

    # 3. Construct 18-Feature Signature for Every Grid Point
    log("\n[3] Constructing 18 physical climate features per site...")

    sig_rows = []

    for pid in expected_pids:
        grp = df[df["point_id"] == pid]

        if len(grp) == 0:
            log(f"  [CRITICAL ERROR] Grid point {pid} missing from dataset!")
            sys.exit(1)

        row = {"point_id": pid}

        # --- A. Temperature Features (°C) ---
        # 3-event daytime ambient temperature statistics
        row["Ta_mean"] = grp["era5_T_amb"].mean()
        row["Ta_p95"] = grp["era5_T_amb"].quantile(0.95)
        row["Ta_p05"] = grp["era5_T_amb"].quantile(0.05)

        # Diurnal Temperature Range (DTR in K): Mean noon vs sunrise/sunset delta
        noon_t = grp[grp["event"] == "noon"]["era5_T_amb"]
        sunrise_t = grp[grp["event"] == "sunrise"]["era5_T_amb"]
        sunset_t = grp[grp["event"] == "sunset"]["era5_T_amb"]
        min_event_t = pd.concat([sunrise_t, sunset_t]).groupby(grp["date"]).min()
        max_event_t = noon_t.groupby(grp["date"]).max()
        dtr_series = (max_event_t - min_event_t).dropna()
        row["DTR"] = dtr_series.mean() if len(dtr_series) > 0 else (row["Ta_p95"] - row["Ta_p05"]) / 2.0

        # Heating / Cooling Degree Days (°C·day)
        # Event-sampled degree days base 18°C and 24°C from 3 daily event samples (sunrise, noon, sunset)
        row["HDD18"] = np.maximum(0, 18.0 - grp["era5_T_amb"]).mean() * 365.25
        row["CDD24"] = np.maximum(0, grp["era5_T_amb"] - 24.0).mean() * 365.25

        # --- B. Solar Resource Features ---
        # Daytime mean GHI (W/m²) when GHI > 0
        daytime_ghi = grp[grp["era5_GHI"] > 0]["era5_GHI"]
        row["GHI_mean"] = daytime_ghi.mean() if len(daytime_ghi) > 0 else 0.0

        # Estimated Daily GHI Integral (kWh/m²/day) [GHI_daily_kWh_est]
        # PROXY ESTIMATE from peak noon GHI using 6.5 equivalent solar hours (not a true 24-hour integral)
        row["GHI_daily_kWh_est"] = (grp[grp["event"] == "noon"]["era5_GHI"].mean() * 6.5) / 1000.0

        # Clear-sky index kt = GHI / GHI_clearsky
        if "era5_CSI" in grp.columns:
            kt_valid = grp[grp["era5_CSI"] > 0]["era5_CSI"]
            row["kt_mean"] = kt_valid.mean() if len(kt_valid) > 0 else 0.5
            row["kt_std"] = kt_valid.std() if len(kt_valid) > 0 else 0.15
        else:
            row["kt_mean"] = 0.60
            row["kt_std"] = 0.15

        # Solar Availability Index (SAI): Fraction of days with estimated daily GHI >= 2.0 kWh/m²/day
        noon_ghi_daily = grp[grp["event"] == "noon"].set_index("date")["era5_GHI"] * 6.5 / 1000.0
        row["SAI"] = (noon_ghi_daily >= 2.0).mean() if len(noon_ghi_daily) > 0 else 0.85

        # Cloudiness & Intermittency
        cloudy_events = (grp["era5_GHI"] / np.maximum(10, grp["era5_GHI_clearsky"])) < 0.35
        row["cloudy_frac"] = cloudy_events.mean()

        # Cloud Continuity Index (CCI): Autonomy persistence proxy (1 - cloudy_frac_std)
        row["CCI"] = 1.0 - (cloudy_events.groupby(grp["date"]).mean().std() if len(cloudy_events) > 0 else 0.2)

        # --- C. Humidity & Condensation Features ---
        row["RH_mean"] = grp["era5_RHum"].mean()

        # Humidity-Storage Interaction (HSI): RH_mean * fraction of event samples where (Ta - Td) < 3 K
        if "era5_T_dew" in grp.columns:
            near_dew = (grp["era5_T_amb"] - grp["era5_T_dew"]) < 3.0
            row["HSI"] = row["RH_mean"] * near_dew.mean()
        else:
            row["HSI"] = row["RH_mean"] * 0.25

        # --- D. Wind & Surface Pressure ---
        row["wind_mean"] = grp["era5_W_spd"].mean()
        row["elev_proxy"] = grp["era5_P_atm"].mean() / 1013.25

        # --- E. Monsoon & Seasonality ---
        if "era5_precipitation" in grp.columns and "month" in grp.columns:
            total_precip = grp["era5_precipitation"].sum()
            monsoon_precip = grp[grp["month"].isin([6, 7, 8, 9])]["era5_precipitation"].sum()
            row["monsoon_index"] = monsoon_precip / total_precip if total_precip > 0 else 0.70
        else:
            row["monsoon_index"] = 0.70

        # Seasonality coefficient of variation of monthly GHI
        monthly_ghi = grp.groupby("month")["era5_GHI"].mean()
        row["seasonality"] = monthly_ghi.std() / monthly_ghi.mean() if monthly_ghi.mean() > 0 else 0.15

        sig_rows.append(row)

    sig_df = pd.DataFrame(sig_rows)
    log(f"  Raw signature DataFrame constructed. Shape: {sig_df.shape}")

    # Verify 129 Point Retention
    log("\n[4] Verifying Grid Point Retention:")
    log(f"  Raw unique points = {df['point_id'].nunique()}")
    log(f"  Climate signature points = {sig_df['point_id'].nunique()}")
    missing_pids = set(expected_pids) - set(sig_df["point_id"].unique())
    log(f"  Missing point IDs = {len(missing_pids)}")
    if len(missing_pids) == 0:
        log("  [PASS] 100% Point Retention Verified (129/129 points present).")

    # Save raw signatures
    sig_df.to_csv(OUT_RAW, index=False)
    log(f"  Saved raw physical signatures to: {OUT_RAW}")

    # 4. Refined Energy Requirements & Targets
    log("\n[5] Derived PCM & System Targets:")
    sig_df["Tm_target"] = TM_TARGET
    # Site-specific mains water temperature estimation: T_mains ≈ max(5.0, Ta_mean - 6.0)
    sig_df["T_mains_est"] = np.maximum(5.0, sig_df["Ta_mean"] - 6.0)
    sig_df["L_required_kWh"] = (M_DRAW_KG * CP_WATER * (T_DELIVERY - sig_df["T_mains_est"])) / 3_600_000

    log(f"  Tm_target = {TM_TARGET}°C (T_delivery={T_DELIVERY}°C, ΔT_approach={DT_APPROACH}K)")
    log(f"  Hot water demand = {M_DRAW_KG} kg/day")
    log(f"  T_mains range across sites: {sig_df['T_mains_est'].min():.2f}°C – {sig_df['T_mains_est'].max():.2f}°C")
    log(f"  L_required range across sites: {sig_df['L_required_kWh'].min():.2f} – {sig_df['L_required_kWh'].max():.2f} kWh/day")

    # 5. PCA Execution on Correlated Thermodynamic Block
    log("\n[6] PCA on Correlated Thermodynamic Block:")
    PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]
    log(f"  Thermodynamic Block Variables ({len(PCA_BLOCK)}): {PCA_BLOCK}")

    pca_data = sig_df[PCA_BLOCK].copy()
    pca_scaler = StandardScaler()
    pca_data_scaled = pca_scaler.fit_transform(pca_data)

    # Fit PCA with 95% variance target or n_components=3
    pca = PCA(n_components=0.95, svd_solver="full")
    pca_scores = pca.fit_transform(pca_data_scaled)
    n_pcs = pca_scores.shape[1]

    log(f"  Retained Components: {n_pcs}")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        log(f"    - PC{i+1}: {var_ratio*100:.2f}% variance")
    log(f"  Cumulative Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Save Loadings Matrix
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=PCA_BLOCK,
        index=[f"PC{i+1}" for i in range(n_pcs)]
    )
    loadings_df.to_csv(OUT_PCA_LOAD)
    log(f"  Saved PCA Loadings Matrix to: {OUT_PCA_LOAD}")
    log("\n  PCA Loadings Table:")
    log(loadings_df.to_string())

    # Add PCs to Signature DataFrame
    for i in range(n_pcs):
        sig_df[f"PC{i+1}"] = pca_scores[:, i]

    # 6. Construct Final Standardized Clustering Matrix
    log("\n[7] Constructing Final Standardized Clustering Matrix...")
    
    # Interaction terms (Pure Climate Features ONLY — No SWH design constants or PCM targets)
    sig_df["ix_GHI_x_kt_std"] = sig_df["GHI_mean"] * sig_df["kt_std"]
    sig_df["ix_DTR_x_cloudy"] = sig_df["DTR"] * sig_df["cloudy_frac"]
    sig_df["ix_RH_x_Ta"] = sig_df["RH_mean"] * sig_df["Ta_mean"]
    sig_df["ix_wind_x_Ta"] = sig_df["wind_mean"] * sig_df["Ta_mean"]
    sig_df["ix_CCI_x_1mSAI"] = sig_df["CCI"] * (1.0 - sig_df["SAI"])

    non_pca_features = [
        "GHI_mean", "GHI_daily_kWh_est", "kt_mean", "kt_std", "SAI", "CCI",
        "cloudy_frac", "DTR", "wind_mean", "HSI", "monsoon_index", "seasonality",
        "ix_GHI_x_kt_std", "ix_DTR_x_cloudy", "ix_RH_x_Ta", "ix_wind_x_Ta", "ix_CCI_x_1mSAI"
    ]
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]

    matrix_features = non_pca_features + pc_cols
    log(f"  Features in clustering matrix ({len(matrix_features)}): {matrix_features}")

    # Standardize final matrix
    matrix_df = pd.DataFrame()
    matrix_df["point_id"] = sig_df["point_id"]

    matrix_scaler = StandardScaler()
    matrix_scaled_vals = matrix_scaler.fit_transform(sig_df[matrix_features])
    for i, col in enumerate(matrix_features):
        matrix_df[col] = matrix_scaled_vals[:, i]

    # Verification of final matrix shape & nulls
    log(f"  Climate signature matrix points = {matrix_df['point_id'].nunique()}")
    log(f"  Matrix shape: {matrix_df.shape} (sites × features)")
    log(f"  Matrix missing values = {matrix_df.isnull().sum().sum()}")

    matrix_df.to_csv(OUT_MATRIX, index=False)
    log(f"  Saved Standardized Clustering Matrix to: {OUT_MATRIX}")

    # Save Full Report
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nSaved Phase 2 Report to: {OUT_REPORT}")

    log("\n" + "=" * 72)
    log("  PHASE 1 & 2 IMPLEMENTATION & VERIFICATION COMPLETE")
    log("=" * 72)

if __name__ == "__main__":
    main()
