"""
STEP 3 — PCM DATA PROCESSING & LABEL GENERATION
=================================================
Cleans your PCM CSV, computes derived features, and generates
the classification target label (which PCM is optimal for a given climate).

PCM selection logic is sourced from:
  - Kou 2025:    RRTDHS > 5.7 → Tm ≈ T_set + 2°C (high solar resource)
                 RRTDHS < 5.7 → Tm can go below T_set (low solar resource)
  - Singh 2025:  Optimal Tm range for SWH = 40–70°C
                 PCM selection priority: latent heat > TC > Tm > Cp > density
  - Yan 2025:    HTF temperature 60–80°C → PCM Tm should be 5–10°C below HTF temp
  - Barqawi 2025: All 5 PCMs had Tm = 44°C, only geometry varied — your project
                  improves on this by using different Tm values (novel contribution)

Your PCM CSV columns (as provided):
    PCM, Product, Manufacturer, Type, Appearance,
    Melting Temperature (°C), Freezing/Congealing Temperature (°C),
    Nucleation Temperature (°C), Latent Heat - Melting (kJ/kg),
    Latent Heat - Freezing (kJ/kg), Heat Storage Capacity (Wh/kg),
    Density - Liquid (kg/m³), Density - Solid (kg/m³),
    Specific Heat - Liquid (kJ/kgK), Specific Heat - Solid (kJ/kgK),
    Thermal Conductivity - Liquid (W/mK), Thermal Conductivity - Solid (W/mK),
    Thermal Conductivity - Both Phases (W/mK), Volume Expansion (%),
    Max Operating Temperature (°C), Flammability, Flash Point (°C),
    Number of Cycles Tested

Requirements:
    pip install pandas numpy scikit-learn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT_PCM_CSV = r"C:\Users\chiru\OneDrive\Desktop\Final_yr_Proj\sem6_R2\PCM_Properties.csv"
OUTPUT_DIR    = r"C:\Users\chiru\OneDrive\Desktop\Final_yr_Proj\sem6_R2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# COLUMN NAME STANDARDIZATION
# Maps your messy CSV column names → clean snake_case names
# ─────────────────────────────────────────────

COLUMN_MAP = {
    "PCM":                                    "pcm_id",
    "Product":                                "product",
    "Manufacturer":                           "manufacturer",
    "Type":                                   "pcm_type",
    "Appearance":                             "appearance",
    "Melting Temperature (°C)":               "Tm_melting",
    "Melting Temperature (Â°C)":              "Tm_melting",      # encoding variant
    "Freezing/Congealing Temperature (°C)":   "Tm_freezing",
    "Freezing/Congealing Temperature (Â°C)":  "Tm_freezing",
    "Nucleation Temperature (°C)":            "Tm_nucleation",
    "Nucleation Temperature (Â°C)":           "Tm_nucleation",
    "Latent Heat - Melting (kJ/kg)":          "latent_heat_melting",
    "Latent Heat - Freezing (kJ/kg)":         "latent_heat_freezing",
    "Heat Storage Capacity (Wh/kg)":          "heat_storage_Wh_kg",
    "Density - Liquid (kg/m³)":              "density_liquid",
    "Density - Liquid (kg/mÂ³)":             "density_liquid",
    "Density - Solid (kg/m³)":               "density_solid",
    "Density - Solid (kg/mÂ³)":              "density_solid",
    "Specific Heat - Liquid (kJ/kgK)":        "Cp_liquid",
    "Specific Heat - Solid (kJ/kgK)":         "Cp_solid",
    "Thermal Conductivity - Liquid (W/mK)":   "TC_liquid",
    "Thermal Conductivity - Solid (W/mK)":    "TC_solid",
    "Thermal Conductivity - Both Phases (W/mK)": "TC_both",
    "Volume Expansion (%)":                   "volume_expansion",
    "Max Operating Temperature (°C)":         "max_op_temp",
    "Max Operating Temperature (Â°C)":        "max_op_temp",
    "Flammability":                           "flammability",
    "Flash Point (°C)":                       "flash_point",
    "Flash Point (Â°C)":                      "flash_point",
    "Number of Cycles Tested":                "cycles_tested",
}

# ─────────────────────────────────────────────
# PCM FILTERING — SWH-SUITABLE RANGE
# Singh 2025: optimal Tm for SWH = 40–70°C
# Allow slight margin below (35°C) for pre-heating scenarios
# ─────────────────────────────────────────────

SWH_TM_MIN = 35.0   # °C
SWH_TM_MAX = 75.0   # °C


def load_and_clean_pcm(csv_path: str) -> pd.DataFrame:
    """Load PCM CSV, standardize columns, fix encoding, filter SWH range."""
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

    # Rename columns
    df = df.rename(columns=COLUMN_MAP)
    # Drop duplicate columns (encoding variants map to same target name)
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop any remaining unmapped columns that are clearly not useful
    # Use dict.fromkeys to deduplicate while preserving order
    keep = list(dict.fromkeys(COLUMN_MAP.values()))
    df = df[[c for c in keep if c in df.columns]]

    # ── Numeric coercion ──────────────────────
    def extract_numeric(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if not val_str:
            return np.nan
        # 1. Handle "peak:" format explicitly
        if "peak:" in val_str.lower():
            import re
            match = re.search(r'peak:\s*([0-9.]+)', val_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # 2. Find all numeric components
        import re
        numbers = re.findall(r'-?\d+\.?\d*', val_str)
        if numbers:
            # If it's a simple range "x-y" with no peak, return the average
            if '-' in val_str and "peak:" not in val_str.lower() and len(numbers) >= 2 and not val_str.startswith('-'):
                try:
                    return (float(numbers[0]) + float(numbers[1])) / 2.0
                except ValueError:
                    pass
            # Just take the first valid number found (works for "~800", ">180", "90 (provisional)")
            # but if it's "~800" findall returns '800'
            return float(numbers[0])
        return np.nan

    numeric_cols = [
        "Tm_melting", "Tm_freezing", "Tm_nucleation",
        "latent_heat_melting", "latent_heat_freezing", "heat_storage_Wh_kg",
        "density_liquid", "density_solid",
        "Cp_liquid", "Cp_solid",
        "TC_liquid", "TC_solid", "TC_both",
        "volume_expansion", "max_op_temp", "flash_point", "cycles_tested",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Handle Missing Values (Imputation & Physical Fallbacks) ──────────────
    
    # 1. Thermal Conductivity (TC)
    if "TC_both" not in df.columns:
        df["TC_both"] = np.nan
    
    # Fill TC_both if missing using average of solid & liquid
    mask_tc_both = df["TC_both"].isna() & df["TC_liquid"].notna() & df["TC_solid"].notna()
    df.loc[mask_tc_both, "TC_both"] = (df.loc[mask_tc_both, "TC_liquid"] + df.loc[mask_tc_both, "TC_solid"]) / 2
    
    # Fill TC_both using either liquid or solid if the other is also missing
    df["TC_both"] = df["TC_both"].fillna(df["TC_liquid"]).fillna(df["TC_solid"]).fillna(df["TC_both"].median())

    # Backfill TC_liquid and TC_solid from TC_both
    df["TC_liquid"] = df["TC_liquid"].fillna(df["TC_both"])
    df["TC_solid"]  = df["TC_solid"].fillna(df["TC_both"])

    # 2. Specific Heat (Cp)
    # Solid Cp is typically ~90% of liquid Cp for PCMs if unknown
    df["Cp_liquid"] = df["Cp_liquid"].fillna(df["Cp_solid"] * 1.1).fillna(2.2) # fallback 2.2 kJ/kgK
    df["Cp_solid"]  = df["Cp_solid"].fillna(df["Cp_liquid"] * 0.9).fillna(2.0)

    # 3. Latent Heat & Temperatures
    df["latent_heat_freezing"] = df["latent_heat_freezing"].fillna(df["latent_heat_melting"])
    df["Tm_freezing"] = df["Tm_freezing"].fillna(df["Tm_melting"] - 1.0)
    df["Tm_nucleation"] = df["Tm_nucleation"].fillna(df["Tm_freezing"] - 2.0)

    # 4. Density
    df["density_liquid"] = df["density_liquid"].fillna(df["density_solid"] * 0.9).fillna(850.0)
    df["density_solid"]  = df["density_solid"].fillna(df["density_liquid"] * 1.1).fillna(900.0)

    # 5. Other physical properties
    df["heat_storage_Wh_kg"] = df["heat_storage_Wh_kg"].fillna(df["heat_storage_Wh_kg"].median())
    df["volume_expansion"]   = df["volume_expansion"].fillna(df["volume_expansion"].median()).fillna(10.0)
    df["max_op_temp"]        = df["max_op_temp"].fillna(df["Tm_melting"] + 30.0)
    df["flash_point"]        = df["flash_point"].fillna(df["flash_point"].median()).fillna(df["max_op_temp"] + 50.0)
    df["cycles_tested"]      = df["cycles_tested"].fillna(0) # Assume 0 if not reported
    
    # Clean text columns
    for col in ["flammability", "appearance", "manufacturer", "pcm_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ── Filter: only SWH-suitable PCMs ───────────────────────────────────
    # Singh 2025: 40–70°C; we use 35–75°C with margin
    before = len(df)
    df = df[df["Tm_melting"].between(SWH_TM_MIN, SWH_TM_MAX)].copy()
    print(f"[PCM FILTER] {before} -> {len(df)} PCMs in SWH range "
          f"({SWH_TM_MIN}-{SWH_TM_MAX} C)")

    # ── Drop rows with missing critical values ────────────────────────────
    critical = ["Tm_melting", "latent_heat_melting"]
    df = df.dropna(subset=critical)
    print(f"[PCM FILTER] After dropping NaN critical: {len(df)} PCMs remain")

    return df.reset_index(drop=True)


def add_derived_pcm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived PCM features useful for the ML classifier.
    Feature importance order from Singh 2025:
      latent heat > TC > Tm > Cp > density
    """
    # Thermal inertia: latent heat per °C of phase change range
    df["latent_per_degree"] = df["latent_heat_melting"] / (
        df["Tm_melting"] - df["Tm_freezing"].fillna(df["Tm_melting"] - 2) + 1e-6
    )

    # Volumetric energy density (ρH) — Kou 2025 key metric
    # ρH = density_solid (kg/m³) × latent_heat_melting (kJ/kg) / 1000 → MJ/m³
    df["rho_H_MJ_m3"] = (df["density_solid"].fillna(800) *
                          df["latent_heat_melting"]) / 1000

    # TC ratio: liquid/solid (>1 means liquid phase conducts better)
    df["TC_ratio"] = (df["TC_liquid"].fillna(df["TC_both"]) /
                      df["TC_solid"].fillna(df["TC_both"]).replace(0, np.nan))

    # Average specific heat
    df["Cp_avg"] = (df["Cp_liquid"].fillna(2.0) + df["Cp_solid"].fillna(2.0)) / 2

    # Average density
    df["density_avg"] = (df["density_liquid"].fillna(800) +
                         df["density_solid"].fillna(850)) / 2

    # Flammability binary — safety flag for embedded system
    df["is_flammable"] = df["flammability"].str.lower().str.contains(
        "yes|flammable|combustible", na=False
    ).astype(int)

    # Cycles-tested confidence score (log scale, normalized to 0–1)
    max_cycles = df["cycles_tested"].fillna(0).max()
    if max_cycles > 0:
        df["cycles_confidence"] = np.log1p(df["cycles_tested"].fillna(0)) / np.log1p(max_cycles)
    else:
        df["cycles_confidence"] = 0.5

    # PCM type encoding (organic=0, inorganic=1, eutectic=2)
    type_map = {"organic": 0, "inorganic": 1, "eutectic": 2}
    df["pcm_type_code"] = df["pcm_type"].str.lower().map(
        lambda x: next((v for k, v in type_map.items() if k in str(x)), 0)
    )

    return df


def compute_pcm_suitability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite PCM suitability score for SWH.
    Based on Singh 2025 priority order:
      latent heat (40%) > TC (25%) > Tm position (20%) > Cp (10%) > density (5%)

    This score is used to GENERATE the target label for the classifier.
    Higher score = better PCM for SWH.
    """
    scaler = MinMaxScaler()

    score_features = {
        "latent_heat_melting":  0.40,
        "TC_both":              0.25,
        "Cp_avg":               0.10,
        "rho_H_MJ_m3":         0.15,
        "cycles_confidence":    0.10,
    }

    available = {k: v for k, v in score_features.items() if k in df.columns}

    # Normalize each feature to 0–1
    score_df = df[list(available.keys())].fillna(0)
    score_norm = pd.DataFrame(
        scaler.fit_transform(score_df),
        columns=score_df.columns,
        index=df.index
    )

    # Weighted sum
    df["pcm_suitability_score"] = sum(
        score_norm[col] * weight
        for col, weight in available.items()
    )

    return df


if __name__ == "__main__":
    # ── Load and clean PCM data ───────────────────────────────────────────
    if not os.path.exists(INPUT_PCM_CSV):
        print(f"[ERROR] PCM CSV not found at: {INPUT_PCM_CSV}")
        print("  -> Copy your PCM CSV to data/raw/pcm_data.csv and re-run.")
        raise SystemExit(1)

    pcm_df = load_and_clean_pcm(INPUT_PCM_CSV)
    pcm_df = add_derived_pcm_features(pcm_df)
    pcm_df = compute_pcm_suitability_score(pcm_df)

    # Save cleaned PCM dataset
    pcm_out = os.path.join(OUTPUT_DIR, "pcm_cleaned.csv")
    try:
        pcm_df.to_csv(pcm_out, index=False)
        print(f"\n[OK] Cleaned PCM data saved -> {pcm_out}")
    except PermissionError:
        pcm_out = os.path.join(OUTPUT_DIR, "pcm_cleaned_new.csv")
        pcm_df.to_csv(pcm_out, index=False)
        print(f"\n[WARNING] Original file was open in Excel. Saved output instead to -> {pcm_out}")
    print(f"   Shape: {pcm_df.shape}")
    
    # Display full table in terminal
    print("\n" + "=" * 80)
    print("CLEANED PCM DATA (sorted by suitability score)")
    print("=" * 80)

    display_cols = [
        "product", "pcm_type", "Tm_melting", "Tm_freezing",
        "latent_heat_melting", "TC_both", "Cp_avg", "density_avg",
        "rho_H_MJ_m3", "is_flammable", "pcm_suitability_score",
    ]
    display_cols = [c for c in display_cols if c in pcm_df.columns]

    sorted_df = pcm_df[display_cols].sort_values(
        "pcm_suitability_score", ascending=False
    )
    print(sorted_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS (numeric columns)")
    print("-" * 80)
    numeric_summary = pcm_df.select_dtypes(include=[np.number]).describe().T
    numeric_summary = numeric_summary[["count", "mean", "min", "max", "std"]]
    print(numeric_summary.to_string())

    print(f"\nTotal PCMs after cleaning: {len(pcm_df)}")
    print(f"Columns: {list(pcm_df.columns)}")
    print(f"\nOutput file: {pcm_out}")

