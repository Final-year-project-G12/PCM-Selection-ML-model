"""
STEP 4 — DATA FUSION: CLIMATE + PCM → MODEL-READY DATASET
===========================================================
Merges the processed ERA5 climate data with PCM properties and labels
into a single model-ready CSV for the XGBoost PCM classifier.

Two outputs:
  A) climate_pcm_fused.csv  — hourly rows, one row per timestep per city
                               with the optimal PCM label attached
  B) classifier_dataset.csv — aggregated daily rows, one row per day per city,
                               averaged climate features + PCM label
                               (this is what you feed to the classifier directly)

Data fusion strategy:
  - Temporal join: each climate row gets the PCM best suited to its
    current RRTDHS and T_set (computed in Step 2 and 3)
  - PCM label is attached by: city × RRTDHS_bin × season
  - All PCM thermophysical properties are added as static columns
    (they don't change per timestep — they are the classifier TARGET features)

Sources:
  - Barqawi 2025: feature vector X = [GHI, DNI, DHI, Tamb, Wspd, RHum, Hour, Month]
  - Kou 2025: RRTDHS as the climate-to-PCM bridge variable
  - Singh 2025: PCM selection priority (latent heat > TC > Tm > Cp > density)
  - Odoi-Yorke 2025: "solar irradiance, ambient temp, flow rate, demand profile, time-of-day, season"
  - Ghodusinejad 2026: "Month, day, season, time of day, RH, wind, CSI, SZA"
  - Yan 2025: HTF temperature as dominant operational feature (47% importance)

Requirements:
    pip install pandas numpy scikit-learn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
CLIMATE_CSV    = "data/processed/climate_all_cities.csv"
PCM_CSV        = "data/processed/pcm_cleaned.csv"
LABEL_MAP_CSV  = "data/processed/pcm_label_mapping.csv"
OUTPUT_DIR     = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DEMAND PROFILE (Synthetic Indian Household)
# RG3: your novel contribution — no existing paper includes this
# Source: TERI / BEE Indian household data; Odoi-Yorke 2025 notes
#         "demand profile" as a missing feature in all reviewed papers
# ─────────────────────────────────────────────

def add_demand_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic Indian household hot-water demand profile.
    Pattern: morning peak (06:00–08:00), evening peak (18:00–20:00).
    Total daily demand: ~120 L/day (Indian household average, BEE 2022).
    """
    # Demand in litres per hour — triangular peaks
    demand_profile = {
        5: 5, 6: 25, 7: 30, 8: 20, 9: 10,
        10: 5, 11: 5, 12: 5, 13: 5, 14: 5,
        15: 5, 16: 5, 17: 10, 18: 20, 19: 25,
        20: 15, 21: 10, 22: 5, 23: 3,
    }
    df["demand_L_per_hr"] = df["hour"].map(demand_profile).fillna(2.0)

    # Demand level categories
    def demand_level(d):
        if d >= 20: return "high"
        elif d >= 10: return "medium"
        else: return "low"

    df["demand_level"] = df["demand_L_per_hr"].map(demand_level)
    df["demand_code"]  = df["demand_level"].map({"low": 0, "medium": 1, "high": 2})

    # Morning and evening demand flags (for RL reward function later)
    df["is_morning_peak"] = df["hour"].between(6, 8).astype(int)
    df["is_evening_peak"] = df["hour"].between(18, 20).astype(int)

    return df


# ─────────────────────────────────────────────
# RRTDHS BINNING FOR LABEL ASSIGNMENT
# ─────────────────────────────────────────────

def rrtdhs_to_bin(rrtdhs):
    """Bin RRTDHS into: low (<4), medium (4–5.7), high (>5.7)."""
    if rrtdhs < 4.0:    return "low"
    elif rrtdhs < 5.7:  return "medium"
    else:               return "high"


# ─────────────────────────────────────────────
# MAIN FUSION FUNCTION
# ─────────────────────────────────────────────

def fuse_climate_pcm(
    climate_df: pd.DataFrame,
    pcm_df: pd.DataFrame,
    label_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join climate data with PCM label and properties.

    Fusion logic:
      1. For each row in climate_df, find which season it belongs to
      2. Look up the best PCM for that city × season from label_map
      3. Join the full PCM properties for that best PCM from pcm_df
    """
    print("[FUSION] Starting climate × PCM fusion...")

    # Add demand profile
    climate_df = add_demand_profile(climate_df)

    # RRTDHS bin per row
    climate_df["rrtdhs_bin"] = climate_df["RRTDHS"].apply(rrtdhs_to_bin)

    # ── Join label_map on city × season ──────────────────────────────────
    merged = climate_df.merge(
        label_map[["city", "season", "best_pcm_product", "best_pcm_id",
                   "best_pcm_Tm", "optimal_Tm_target", "rrtdhs_approx"]],
        on=["city", "season"],
        how="left",
    )

    # ── Join PCM properties for the selected PCM ─────────────────────────
    # Only keep thermophysical properties (not descriptive text)
    pcm_feature_cols = [
        "product",
        "Tm_melting", "Tm_freezing", "Tm_nucleation",
        "latent_heat_melting", "latent_heat_freezing",
        "heat_storage_Wh_kg", "density_liquid", "density_solid",
        "Cp_liquid", "Cp_solid", "TC_liquid", "TC_solid", "TC_both",
        "volume_expansion", "max_op_temp", "flash_point", "cycles_tested",
        "rho_H_MJ_m3", "TC_ratio", "Cp_avg", "density_avg",
        "pcm_type_code", "pcm_suitability_score", "is_flammable",
        "cycles_confidence",
    ]
    pcm_props = pcm_df[[c for c in pcm_feature_cols if c in pcm_df.columns]].copy()
    pcm_props = pcm_props.rename(columns={"product": "best_pcm_product"})

    merged = merged.merge(pcm_props, on="best_pcm_product", how="left")

    # ── Encode the PCM label numerically ─────────────────────────────────
    # The classifier will predict pcm_label_code → map back to pcm name
    le = LabelEncoder()
    merged["pcm_label"] = merged["best_pcm_product"].fillna("unknown")
    merged["pcm_label_code"] = le.fit_transform(merged["pcm_label"])

    # Save label encoder classes for inference
    label_classes = pd.DataFrame({
        "pcm_label_code": range(len(le.classes_)),
        "pcm_label":      le.classes_,
    })
    label_classes.to_csv(os.path.join(OUTPUT_DIR, "pcm_label_encoder.csv"), index=False)
    print(f"   PCM label classes: {list(le.classes_)}")

    print(f"[FUSION] Result shape: {merged.shape}")
    return merged


def build_classifier_dataset(fused_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly fused data to DAILY rows for the PCM classifier.

    The PCM selector doesn't need hourly resolution —
    it needs to answer: "given today's and tomorrow's forecast,
    which PCM should be active?" → daily aggregation is right.

    Each row = one day × one city, with:
      - Climate features: daily mean, max, min of key variables
      - Temporal features: DOY, month, season
      - PCM label: the optimal PCM for that day (from RRTDHS + season)
    """
    print("[CLASSIFIER DATASET] Aggregating to daily resolution...")

    fused_df["date"] = pd.to_datetime(fused_df["timestamp"]).dt.date

    # Aggregation functions per column type
    agg_dict = {}

    # Climate variables — various aggregations
    for col, func in [
        ("GHI",         ["mean", "max", "sum"]),
        ("DNI",         ["mean", "max"]),
        ("DHI",         ["mean"]),
        ("T_amb",       ["mean", "max", "min"]),
        ("RHum",        ["mean"]),
        ("W_spd",       ["mean", "max"]),
        ("cloud_cover", ["mean"]),
        ("CSI",         ["mean"]),
        ("SZA",         ["mean"]),
        ("ETR",         ["mean", "max"]),
        ("RRTDHS",      ["first"]),       # monthly value, take first
        ("P_atm",       ["mean"]),
        ("precipitation", ["sum"]),
    ]:
        if col in fused_df.columns:
            agg_dict[col] = func

    # Temporal features — take first value of the day
    for col in ["month", "DOY", "year", "season", "season_code",
                "sunrise_hour", "sunset_hour", "high_solar_resource"]:
        if col in fused_df.columns:
            agg_dict[col] = "first"

    # Location metadata — first value
    for col in ["city", "lat", "lon", "altitude_m", "climate_zone", "T_set"]:
        if col in fused_df.columns:
            agg_dict[col] = "first"

    # Demand — daily total
    if "demand_L_per_hr" in fused_df.columns:
        agg_dict["demand_L_per_hr"] = "sum"

    # PCM properties — first (they're constant for a city×season)
    pcm_agg_cols = [
        "Tm_melting", "Tm_freezing", "latent_heat_melting",
        "heat_storage_Wh_kg", "TC_both", "density_solid",
        "Cp_avg", "rho_H_MJ_m3", "pcm_suitability_score",
        "pcm_type_code", "is_flammable", "pcm_label", "pcm_label_code",
        "best_pcm_product", "best_pcm_Tm", "optimal_Tm_target",
    ]
    for col in pcm_agg_cols:
        if col in fused_df.columns:
            agg_dict[col] = "first"

    daily = fused_df.groupby(["city", "date"]).agg(agg_dict).reset_index()

    # Flatten multi-level column names from multiple aggregations
    new_cols = []
    for col in daily.columns:
        if isinstance(col, tuple):
            if col[1] in ("", "first"):
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)
    daily.columns = new_cols

    # Rename aggregated columns for clarity
    rename_map = {
        "demand_L_per_hr_sum": "demand_total_L",
        "GHI_sum": "GHI_daily_kWh_m2",   # note: still in W·h/m², divide by 1000 for kWh
        "RRTDHS_first": "RRTDHS",
    }
    daily = daily.rename(columns={k: v for k, v in rename_map.items() if k in daily.columns})
    if "GHI_daily_kWh_m2" in daily.columns:
        daily["GHI_daily_kWh_m2"] = daily["GHI_daily_kWh_m2"] / 1000

    # ── Lag features (yesterday's values) for forecasting ─────────────────
    # Ghodusinejad 2026: lag features are key for time-series irradiance forecasting
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    for col in ["GHI_mean", "T_amb_mean", "T_amb_max", "CSI_mean", "cloud_cover_mean"]:
        if col in daily.columns:
            daily[f"{col}_lag1"] = daily.groupby("city")[col].shift(1)
            daily[f"{col}_lag2"] = daily.groupby("city")[col].shift(2)
            daily[f"{col}_lag7"] = daily.groupby("city")[col].shift(7)   # same weekday

    # Rolling 7-day mean (for smoothed trend)
    for col in ["GHI_mean", "T_amb_mean"]:
        if col in daily.columns:
            daily[f"{col}_roll7"] = (
                daily.groupby("city")[col]
                     .transform(lambda x: x.rolling(7, min_periods=1).mean())
            )

    print(f"[CLASSIFIER DATASET] Shape: {daily.shape}")
    return daily


def add_forecasting_targets(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-day forecast targets for the LSTM/XGBoost forecasting model.

    Target 1: GHI_next_day  — tomorrow's mean GHI (W/m²)
    Target 2: T_amb_next_day — tomorrow's mean ambient temp (°C)

    These become the INPUT to the PCM classifier after forecasting.
    Ghodusinejad 2026: day-ahead forecasting is the standard horizon
    for solar resource planning.
    """
    for col, target_col in [
        ("GHI_mean",   "GHI_next_day"),
        ("T_amb_mean", "T_amb_next_day"),
    ]:
        if col in daily_df.columns:
            daily_df[target_col] = daily_df.groupby("city")[col].shift(-1)

    return daily_df


if __name__ == "__main__":
    # ── Load inputs ──────────────────────────────────────────────────────
    print("[LOAD] Reading climate data...")
    climate_df = pd.read_csv(CLIMATE_CSV, parse_dates=["timestamp"])
    print(f"  Climate shape: {climate_df.shape}")

    print("[LOAD] Reading PCM data...")
    pcm_df = pd.read_csv(PCM_CSV)
    print(f"  PCM shape: {pcm_df.shape}")

    print("[LOAD] Reading label mapping...")
    label_map = pd.read_csv(LABEL_MAP_CSV)
    print(f"  Label map:\n{label_map.to_string(index=False)}")

    # ── Hourly fused dataset ──────────────────────────────────────────────
    fused = fuse_climate_pcm(climate_df, pcm_df, label_map)
    fused_path = os.path.join(OUTPUT_DIR, "climate_pcm_fused_hourly.csv")
    fused.to_csv(fused_path, index=False)
    print(f"\n✅ Hourly fused dataset saved → {fused_path}")

    # ── Daily classifier dataset ──────────────────────────────────────────
    daily = build_classifier_dataset(fused)
    daily = add_forecasting_targets(daily)

    # Final column audit
    print(f"\n[FINAL DATASET] Columns ({len(daily.columns)}):")
    climate_cols = [c for c in daily.columns if any(
        x in c for x in ["GHI", "DNI", "DHI", "T_amb", "RHum", "W_spd",
                          "cloud", "CSI", "SZA", "RRTDHS", "ETR", "precip",
                          "P_atm", "sunrise", "sunset", "lag", "roll"]
    )]
    pcm_cols = [c for c in daily.columns if any(
        x in c for x in ["Tm_", "latent", "TC_", "density", "Cp_", "pcm",
                          "rho_H", "heat_storage", "max_op", "flash", "cycles"]
    )]
    time_cols = [c for c in daily.columns if any(
        x in c for x in ["month", "DOY", "year", "season", "date", "city",
                          "lat", "lon", "alt", "climate_zone", "T_set",
                          "demand", "high_solar"]
    )]
    target_cols = [c for c in daily.columns if "next_day" in c or "label" in c]

    print(f"  Climate features ({len(climate_cols)}): {climate_cols}")
    print(f"  PCM features ({len(pcm_cols)}):     {pcm_cols}")
    print(f"  Time/meta ({len(time_cols)}):        {time_cols}")
    print(f"  Targets ({len(target_cols)}):         {target_cols}")

    daily_path = os.path.join(OUTPUT_DIR, "classifier_dataset.csv")
    daily.to_csv(daily_path, index=False)
    print(f"\n✅ Classifier dataset saved → {daily_path}")
    print(f"   Shape: {daily.shape}")
    print(f"   NaN summary:\n{daily.isnull().sum()[daily.isnull().sum() > 0]}")
    print("\nNext step: run 05_train_forecaster.py  (LSTM irradiance forecasting)")
    print("           run 06_train_pcm_classifier.py (XGBoost PCM selection)")
