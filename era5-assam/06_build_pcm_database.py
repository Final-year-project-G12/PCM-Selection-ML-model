"""
06_build_pcm_database.py  - Assam
==========================================================================
PHASE 5 PREP - PCM PROPERTY DATABASE

Reads the MICE+RF+PMM cleaned manufacturer dataset, appends literature
PCMs from Singh2025 Table 2, computes derived properties, and writes
pcm_database_assam.csv for use by 07_feasibility_filter.py.

INPUT  : ../../PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv
OUTPUT : data/processed/pcm/pcm_database_assam.csv

HOW TO RUN:
  python 06_build_pcm_database.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

INPUT_CSV = PROCESSED_DIR.parent.parent.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"

OUT_DIR = PROCESSED_DIR / "pcm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "pcm_database_assam.csv"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0

IMPUTABLE_PROPS = ["Tm_melting", "Tm_freezing", "latent_heat_melting", "density_liquid",
                   "density_solid", "Cp_liquid", "Cp_solid", "TC_liquid", "TC_solid",
                   "TC_both", "cycles_tested", "flammability"]


def load_manufacturer_rows(csv_path):
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out["name"]                 = df["product"]
    out["family"]               = np.where(df["is_rt_line"] == 1, "Rubitherm RT", "PLUSS savE")
    out["pcm_type"]             = df["pcm_type"]
    out["Tm_C"]                 = df["Tm_melting"]
    out["Tm_freezing_C"]        = df["Tm_freezing"]
    out["latent_heat_kJ_kg"]    = df["latent_heat_melting"]
    out["density_liquid_kg_m3"] = df["density_liquid"]
    out["density_solid_kg_m3"]  = df["density_solid"]
    out["Cp_liquid_kJ_kgK"]     = df["Cp_liquid"]
    out["Cp_solid_kJ_kgK"]      = df["Cp_solid"]
    out["TC_W_mK"]              = (df["TC_liquid"] + df["TC_solid"]) / 2.0
    out["cycles_tested"]        = df["cycles_tested"]
    out["cycles_tested_status"] = df["cycles_tested_status"]
    out["flammable"]            = df["flammability"]
    out["supercooling_K"]       = out["Tm_C"] - out["Tm_freezing_C"]
    imputed_cols = [c + "_imputed" for c in IMPUTABLE_PROPS if c + "_imputed" in df.columns]
    out["n_properties_imputed"] = df[imputed_cols].sum(axis=1)
    out["any_property_imputed"] = out["n_properties_imputed"] > 0
    out["source"] = "manufacturer_datasheet_MICE_RF_PMM_completed"
    return out


def literature_rows():
    rows = [
        {"name": "Myristic acid",           "family": "Fatty acid", "pcm_type": "Organic",
         "Tm_C": 53.0,  "latent_heat_kJ_kg": 190.0},
        {"name": "Palmitic acid",            "family": "Fatty acid", "pcm_type": "Organic",
         "Tm_C": 63.0,  "latent_heat_kJ_kg": 185.4},
        {"name": "Myristic-Palmitic eutectic (58/42)", "family": "Eutectic", "pcm_type": "Organic",
         "Tm_C": 42.6,  "latent_heat_kJ_kg": 169.7},
        {"name": "Palmitic-Stearic eutectic (64.2/35.8)", "family": "Eutectic", "pcm_type": "Organic",
         "Tm_C": 52.3,  "latent_heat_kJ_kg": 181.7},
        {"name": "Paraffin wax (generic)",   "family": "Paraffin",  "pcm_type": "Organic",
         "Tm_C": 64.0,  "latent_heat_kJ_kg": 173.6,
         "density_solid_kg_m3": 916.0, "density_liquid_kg_m3": 790.0},
        {"name": "C22H46 (docosane-class paraffin)",     "family": "Paraffin", "pcm_type": "Organic",
         "Tm_C": 44.5,  "latent_heat_kJ_kg": 249.0},
        {"name": "C30H62 (triacontane-class paraffin)",  "family": "Paraffin", "pcm_type": "Organic",
         "Tm_C": 65.5,  "latent_heat_kJ_kg": 252.0},
    ]
    df = pd.DataFrame(rows)
    for c in ["density_liquid_kg_m3", "density_solid_kg_m3", "Cp_liquid_kJ_kgK",
              "Cp_solid_kJ_kgK", "TC_W_mK", "cycles_tested", "supercooling_K", "Tm_freezing_C"]:
        if c not in df.columns:
            df[c] = np.nan
    df["cycles_tested_status"] = "not_reported"
    df["flammable"]            = "Likely (organic)"
    df["any_property_imputed"] = False
    df["n_properties_imputed"] = 0
    df["source"]               = "Singh2025_Table2_literature"
    return df


def add_derived(db):
    density_for_rho = db["density_solid_kg_m3"].fillna(db["density_liquid_kg_m3"])
    db["rho_H_MJ_m3"] = (density_for_rho * db["latent_heat_kJ_kg"]) / 1000.0
    db["Cp_avg_kJ_kgK"] = (
        db["Cp_liquid_kJ_kgK"].fillna(db["Cp_solid_kJ_kgK"]) +
        db["Cp_solid_kJ_kgK"].fillna(db["Cp_liquid_kJ_kgK"])
    ) / 2.0
    max_cycles = db["cycles_tested"].max()
    db["cycles_confidence"] = np.where(
        db["cycles_tested"].notna(),
        np.log1p(db["cycles_tested"]) / np.log1p(max_cycles) if max_cycles and max_cycles > 0 else np.nan,
        np.nan
    )
    db["in_absolute_band"] = db["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX)
    db["corrosion_class"]  = np.where(
        db["pcm_type"].astype(str).str.contains("Inorganic", na=False),
        "check_manually", "low_organic"
    )
    return db


def main():
    print("=" * 68)
    print("  PCM Property Database Builder - Assam")
    print(f"  Source: {INPUT_CSV}")
    print("=" * 68)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found.")

    manuf = load_manufacturer_rows(INPUT_CSV)
    lit   = literature_rows()
    db    = pd.concat([manuf, lit], ignore_index=True, sort=False)
    db    = add_derived(db)
    db    = db.sort_values("Tm_C").reset_index(drop=True)
    db.to_csv(OUT_FILE, index=False)

    n_in_band   = db["in_absolute_band"].sum()
    n_imputed   = manuf["any_property_imputed"].sum()
    n_inorganic = (db["corrosion_class"] == "check_manually").sum()

    print(f"\n  Total candidates           : {len(db)}")
    print(f"  In 42-70C absolute band    : {n_in_band}")
    print(f"  Tm range                   : {db['Tm_C'].min():.1f} - {db['Tm_C'].max():.1f} C")
    print(f"  MICE-PMM imputed rows      : {n_imputed}/{len(manuf)}")
    print(f"  Inorganic (check_manually) : {n_inorganic}  <- corrosion veto IS active for Assam's high HSI")
    print()
    print(db[["name", "family", "Tm_C", "latent_heat_kJ_kg", "TC_W_mK",
               "cycles_tested", "any_property_imputed", "in_absolute_band",
               "corrosion_class"]].to_string(index=False))
    print(f"\n  Saved: {OUT_FILE}")
    print("=" * 68)
    print("\nNext: python 07b_charging_feasibility.py  (optional)")
    print("      python 07_feasibility_filter.py")


if __name__ == "__main__":
    main()
