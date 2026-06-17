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

Methods:
    1. Handle Missing Values in PCM Table: Imputes structural nulls (Nucleation, Flash Points) and deploys advanced MICE (IterativeImputer with RandomForests) for complex missing thermal metrics.
    2. Encode Categorical PCM Columns: Utilizes LabelEncoder for strings such as flammability and PCM structure type.
    3. Compute Derived PCM Features: Extracts hysteresis (Tm_range), average latent heats, and volumetric density (rhoH).
    4. Feature Selection for PCM Classifier: Drops noisy/granular strings (Appearance, Manufacturer) unhelpful to ML.
    5. Multi-Criteria Decision Making (TOPSIS): Computes highly sophisticated geometric target suitability scores by calculating L2 Euclidean distances to theoretically 'Ideal' PCMs.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PCM_CSV = os.path.join(BASE_DIR, "PCM_Properties.csv")
OUTPUT_DIR    = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COLUMN NAME STANDARDIZATION
# Maps messy CSV column names → clean snake_case names

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

# PCM FILTERING — SWH-SUITABLE RANGE
# Singh 2025: optimal Tm for SWH = 40–70°C
# Allow slight margin below (35°C) for pre-heating scenarios

SWH_TM_MIN = 35.0   # °C
SWH_TM_MAX = 75.0   # °C


def visualize_preprocessing_diff(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """
    Visualizes the differences before and after preprocessing (imputation of missing values).
    Shows missing values heatmaps and distributions of key imputed features.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Missing Values Heatmap Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(df_before.isnull(), cbar=False, cmap='viridis', ax=axes[0])
        axes[0].set_title('Missing Values Before Imputation', fontsize=14)
        
        sns.heatmap(df_after.isnull(), cbar=False, cmap='viridis', ax=axes[1])
        axes[1].set_title('Missing Values After Imputation', fontsize=14)
        
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("[VISUALIZATION ERROR] matplotlib and/or seaborn are not installed.")
        print("Please install them using: pip install matplotlib seaborn")

def load_and_clean_pcm(csv_path: str, visualize: bool = False) -> pd.DataFrame:
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

    df_before_imputation = df.copy() if visualize else None

    # ── Handle Missing Values in PCM Table ────────────────────────────────
    
    # METHOD: Handle Missing Values in PCM Table
    # As per IEEE 11141790 (Multimodal Learning Techniques for Renewable Energy), ML-based 
    # imputation (e.g., KNN, VAEs) is preferred for handling missing modality data points
    # over standard mean/median fallbacks. Here we implement KNN Imputation.
    from sklearn.impute import KNNImputer
    
    # 1. First enforce known physical relationships (domain constraints)
    if "TC_both" not in df.columns:
        df["TC_both"] = np.nan
    mask_tc_both = df["TC_both"].isna() & df["TC_liquid"].notna() & df["TC_solid"].notna()
    df.loc[mask_tc_both, "TC_both"] = (df.loc[mask_tc_both, "TC_liquid"] + df.loc[mask_tc_both, "TC_solid"]) / 2
    
    # Pre-fill easy single-variable derivations
    df["latent_heat_freezing"] = df["latent_heat_freezing"].fillna(df["latent_heat_melting"])
    df["Tm_freezing"] = df["Tm_freezing"].fillna(df["Tm_melting"] - 1.0)
    df["Tm_nucleation"] = df["Tm_nucleation"].fillna(df["Tm_freezing"])
    df["max_op_temp"]        = df["max_op_temp"].fillna(df["Tm_melting"] + 30.0)
    df["cycles_tested"]      = df["cycles_tested"].fillna(0) # Assume 0 if not reported
    
    # METHOD: Missing value handling (imputation)
    # 2. Apply Multivariate Feature Imputation (MICE) via Chained Equations using Random Forests
    # This complex method replaces simple KNN. It models each feature as a function of the others.
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42),
        max_iter=10, 
        random_state=42
    )
    
    cols_to_impute = [
        "TC_both", "TC_liquid", "TC_solid", 
        "Cp_liquid", "Cp_solid",
        "density_liquid", "density_solid", 
        "heat_storage_Wh_kg", "volume_expansion", "flash_point"
    ]
    
    # Ensure columns exist before imputing
    available_cols = [c for c in cols_to_impute if c in df.columns]
    
    if available_cols and len(df) >= 5:
        imputed_array = imputer.fit_transform(df[available_cols])
        df[available_cols] = pd.DataFrame(imputed_array, columns=available_cols, index=df.index)
    
    # 3. Final physical fallback logic for anything KNN missed (e.g. if all values were NaN)
    df["TC_both"] = df["TC_both"].fillna(df["TC_liquid"]).fillna(df["TC_solid"]).fillna(0.5)
    df["TC_liquid"] = df["TC_liquid"].fillna(df["TC_both"])
    df["TC_solid"]  = df["TC_solid"].fillna(df["TC_both"])
    
    df["Cp_liquid"] = df["Cp_liquid"].fillna(df["Cp_solid"] * 1.1).fillna(2.2)
    df["Cp_solid"]  = df["Cp_solid"].fillna(df["Cp_liquid"] * 0.9).fillna(2.0)
    
    df["density_liquid"] = df["density_liquid"].fillna(df["density_solid"] * 0.9).fillna(850.0)
    df["density_solid"]  = df["density_solid"].fillna(df["density_liquid"] * 1.1).fillna(900.0)
    
    df["heat_storage_Wh_kg"] = df["heat_storage_Wh_kg"].fillna(200.0)
    df["volume_expansion"]   = df["volume_expansion"].fillna(10.0)
    # flash_point handled as Flash_Point_known flag down in derived features
    
    # Clean text columns
    for col in ["flammability", "appearance", "manufacturer", "pcm_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            
    if visualize and df_before_imputation is not None:
        visualize_preprocessing_diff(df_before_imputation, df)

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
    # METHOD: Compute Derived PCM Features
    Compute derived PCM features useful for the ML classifier.
    Feature importance order from Singh 2025:
      latent heat > TC > Tm > Cp > density
    """
    # Thermal inertia: latent heat per °C of phase change range
    df["latent_per_degree"] = df["latent_heat_melting"] / (
        df["Tm_melting"] - df["Tm_freezing"].fillna(df["Tm_melting"] - 2) + 1e-6
    )

    df["latent_heat_avg"] = (df["latent_heat_melting"].fillna(0) + df["latent_heat_freezing"].fillna(0)) / 2
    df["Tm_range"] = df["Tm_melting"] - df["Tm_freezing"]
    df["heat_storage_Wh_kg"] = df["heat_storage_Wh_kg"].fillna(df["latent_heat_melting"] / 3.6)

    # Volumetric energy density (ρH) — Kou 2025 key metric
    # ρH = density_solid (kg/m³) × latent_heat_avg (kJ/kg) / 1000 → MJ/m³
    df["rho_H_MJ_m3"] = (df["density_solid"].fillna(800) * df["latent_heat_avg"]) / 1000

    # TC ratio: liquid/solid (>1 means liquid phase conducts better)
    df["TC_ratio"] = (df["TC_liquid"].fillna(df["TC_both"]) /
                      df["TC_solid"].fillna(df["TC_both"]).replace(0, np.nan))

    # Average specific heat
    df["Cp_avg"] = (df["Cp_liquid"].fillna(2.0) + df["Cp_solid"].fillna(2.0)) / 2

    # Average density
    df["density_avg"] = (df["density_liquid"].fillna(800) +
                         df["density_solid"].fillna(850)) / 2

    # Cycles-tested confidence score (log scale, normalized to 0–1)
    max_cycles = df["cycles_tested"].fillna(0).max()
    if max_cycles > 0:
        df["cycles_confidence"] = np.log1p(df["cycles_tested"].fillna(0)) / np.log1p(max_cycles)
    else:
        df["cycles_confidence"] = 0.5

    # Target Flag for unknown Flash Points
    df["flash_point_known"] = df["flash_point"].notna().astype(int)

    # METHOD: Encode Categorical PCM Columns
    # Categorical Encoding with standard LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["pcm_type_code"] = le.fit_transform(df["pcm_type"].fillna("Unknown"))
    df["flammability_enc"] = le.fit_transform(df["flammability"].fillna("Unknown"))

    # METHOD: Feature Selection for PCM Classifier
    # Drop low ML value categorical strings
    df = df.drop(columns=["manufacturer", "appearance"], errors="ignore")

    return df


def compute_pcm_suitability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    # METHOD: Multi-Criteria Decision Making (TOPSIS Algorithm)
    Compute a composite PCM suitability score for SWH using the advanced 
    Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS).
    Replaces basic linear sum with geometric ideal point distance calculations.
    """
    score_features = {
        "latent_heat_melting":  0.40,
        "TC_both":              0.25,
        "Cp_avg":               0.10,
        "rho_H_MJ_m3":          0.15,
        "cycles_confidence":    0.10,
    }

    available_cols = [k for k in score_features.keys() if k in df.columns]
    
    # 1. Vector Normalization
    X = df[available_cols].fillna(0).values
    norm_X = X / (np.sqrt((X**2).sum(axis=0)) + 1e-8)
    
    # 2. Apply Weighting
    weights = np.array([score_features[k] for k in available_cols])
    V = norm_X * weights
    
    # 3. Determine Ideal Best (V+) and Ideal Worst (V-) solutions
    # (Assuming all criteria here are beneficial to maximize)
    V_plus = np.max(V, axis=0)
    V_minus = np.min(V, axis=0)
    
    # 4. Calculate Geometric Separation (Euclidean L2 distances)
    S_plus = np.sqrt(((V - V_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - V_minus)**2).sum(axis=1))
    
    # 5. Calculate Relative Closeness to the Ideal Solution (TOPSIS Score 0.0-1.0)
    df["pcm_suitability_score"] = S_minus / (S_plus + S_minus + 1e-8)

    return df


if __name__ == "__main__":
    # ── Load and clean PCM data ───────────────────────────────────────────
    if not os.path.exists(INPUT_PCM_CSV):
        print(f"[ERROR] PCM CSV not found at: {INPUT_PCM_CSV}")
        print("  -> Copy your PCM CSV to data/raw/pcm_data.csv and re-run.")
        raise SystemExit(1)

    pcm_df = load_and_clean_pcm(INPUT_PCM_CSV, visualize=True)
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
        "product", "pcm_type", "Tm_melting", "Tm_range",
        "latent_heat_avg", "TC_both", "Cp_avg", "density_avg",
        "rho_H_MJ_m3", "flammability_enc", "pcm_suitability_score",
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