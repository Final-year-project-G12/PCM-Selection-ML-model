"""
06_build_pcm_database.py   (v3 — consumes the updated 55-row MICE+RF+PMM CSV)
===============================================================================
PHASE 5 PREP — PCM PROPERTY DATABASE (Objective 1 plan v3.0, Section 4.5 / D2)

This script builds the candidate PCM database from the updated
PCM_Properties_cleaned_mice_pmm_detailed.csv, which now contains **55 rows**
(31 manufacturer + 24 literature) covering 6 manufacturer brands plus a rich
set of literature-sourced PCMs (n-alkanes, fatty acids, composites, blends),
all MICE+RF+PMM-imputed to full coverage across the 42-70 °C band.

The CSV row breakdown:
  Manufacturer           | Rows
  -----------------------|-----
  Literature             |  24
  Rubitherm Technologies |  14
  Pluss Advanced Tech.   |   7
  PureTemp               |   5
  PCM Products Ltd.      |   4
  CrodaTherm             |   1
  TOTAL                  |  55

This script:
1. Reads PCM_Properties_cleaned_mice_pmm_detailed.csv (the *_imputed flag
   columns are what make this useful — the lean version doesn't carry them).
2. Maps the `manufacturer` column to a `family` label and renames columns to
   the schema 07/08/09 already expect (Tm_C, latent_heat_kJ_kg, TC_W_mK, ...).
   NO CHANGES NEEDED to 07_feasibility_filter.py, 08_mcdm_ranking.py, or
   09_recommendation_cards.py.
3. Carries through an `any_property_imputed` flag and `cycles_tested_status`
   per row, so recommendation cards can honestly report which picks rest on
   MICE-PMM-estimated values vs. pure manufacturer datasheet numbers.
4. Sets `source` = "literature_MICE_RF_PMM_completed" for literature rows and
   "manufacturer_datasheet_MICE_RF_PMM_completed" for commercial product rows.

NOTE: The old script appended 7 hardcoded Singh2025 literature rows;
      those PCMs are now fully included in the 55-row CSV with imputed
      properties — the manual append is no longer needed.

INPUT  : PCM_Properties_cleaned_mice_pmm_detailed.csv
         (edit INPUT_CSV below to wherever you extracted it)
OUTPUT : data/processed/pcm/pcm_database_uttarakhand.csv

HOW TO RUN:
  python 06_build_pcm_database.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

# EDIT THIS PATH to wherever PCM_Properties_cleaned_mice_pmm_detailed.csv
# actually sits after you unzip PCM_data (2).zip.
INPUT_CSV = PROCESSED_DIR.parent.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"

OUT_DIR = PROCESSED_DIR / "pcm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "pcm_database_uttarakhand.csv"

ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0   # v3.0-corrected band, Table 12

IMPUTABLE_PROPS = ["Tm_melting", "Tm_freezing", "latent_heat_melting", "density_liquid",
                   "density_solid", "Cp_liquid", "Cp_solid", "TC_liquid", "TC_solid",
                   "TC_both", "cycles_tested", "flammability"]

# Mapping from manufacturer column values to the short family label used
# downstream (recommendation cards, MCDM tables).
_MANUFACTURER_FAMILY_MAP = {
    "Rubitherm Technologies":      "Rubitherm RT",
    "Pluss Advanced Technologies": "PLUSS savE",
    "PCM Products Ltd.":           "PCM Products",
    "PureTemp":                    "PureTemp",
    "CrodaTherm":                  "CrodaTherm",
}

# For literature rows the family is derived from the pcm_type field.
_PCMTYPE_FAMILY_MAP = {
    "Organic n-alkane":            "n-Alkane",
    "Organic fatty acid":          "Fatty acid",
    "Organic/composite blend":     "Composite",
    "Organic blend":               "Blend",
    "Organic/polymer blend":       "Polymer blend",
    "Organic/eutectic composite":  "Eutectic composite",
    "Organic PCM":                 "Organic PCM",
    "Organic bio-based PCM":       "Bio-based PCM",
    "Organic commercial PCM":      "Commercial PCM",
    "Organic":                     "Organic",
    "Organic (RT-line)":           "Rubitherm RT",
}


def _derive_family(manufacturer: str, pcm_type: str) -> str:
    """Return a short family label for one row."""
    if manufacturer in _MANUFACTURER_FAMILY_MAP:
        return _MANUFACTURER_FAMILY_MAP[manufacturer]
    # Literature rows: derive from pcm_type
    return _PCMTYPE_FAMILY_MAP.get(pcm_type, pcm_type)


def load_all_pcm_rows(csv_path):
    """Load all 55 rows from the MICE+RF+PMM-cleaned CSV.

    The CSV already contains both manufacturer product rows and literature
    rows (24 of 55), all with full imputed coverage — no separate manual
    append is needed.
    """
    df = pd.read_csv(csv_path)

    out = pd.DataFrame()
    out["name"] = df["product"]
    out["manufacturer"] = df["manufacturer"]
    out["family"] = [
        _derive_family(m, t)
        for m, t in zip(df["manufacturer"], df["pcm_type"])
    ]
    out["pcm_type"] = df["pcm_type"]
    out["Tm_C"] = df["Tm_melting"]
    out["Tm_freezing_C"] = df["Tm_freezing"]
    out["latent_heat_kJ_kg"] = df["latent_heat_melting"]
    out["density_liquid_kg_m3"] = df["density_liquid"]
    out["density_solid_kg_m3"] = df["density_solid"]
    out["Cp_liquid_kJ_kgK"] = df["Cp_liquid"]
    out["Cp_solid_kJ_kgK"] = df["Cp_solid"]
    # Prefer per-phase average over the imputed-constant TC_both
    out["TC_W_mK"] = (df["TC_liquid"] + df["TC_solid"]) / 2.0
    out["cycles_tested"] = df["cycles_tested"]
    out["cycles_tested_status"] = df["cycles_tested_status"]
    out["flammable"] = df["flammability"]
    out["supercooling_K"] = out["Tm_C"] - out["Tm_freezing_C"]

    imputed_cols = [c + "_imputed" for c in IMPUTABLE_PROPS if c + "_imputed" in df.columns]
    out["n_properties_imputed"] = df[imputed_cols].sum(axis=1)
    out["any_property_imputed"] = out["n_properties_imputed"] > 0

    # Distinguish commercial product rows from literature rows
    is_lit = df["manufacturer"] == "Literature"
    out["source"] = np.where(
        is_lit,
        "literature_MICE_RF_PMM_completed",
        "manufacturer_datasheet_MICE_RF_PMM_completed",
    )
    return out


def add_derived(db):
    density_for_rho = db["density_solid_kg_m3"].fillna(db["density_liquid_kg_m3"])
    db["rho_H_MJ_m3"] = (density_for_rho * db["latent_heat_kJ_kg"]) / 1000.0
    db["Cp_avg_kJ_kgK"] = (db["Cp_liquid_kJ_kgK"].fillna(db["Cp_solid_kJ_kgK"]) +
                            db["Cp_solid_kJ_kgK"].fillna(db["Cp_liquid_kJ_kgK"])) / 2.0
    max_cycles = db["cycles_tested"].max()
    db["cycles_confidence"] = np.where(
        db["cycles_tested"].notna(),
        np.log1p(db["cycles_tested"]) / np.log1p(max_cycles) if max_cycles and max_cycles > 0 else np.nan,
        np.nan)
    db["in_absolute_band"] = db["Tm_C"].between(ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX)
    db["corrosion_class"] = np.where(db["pcm_type"].astype(str).str.contains("Inorganic", na=False),
                                      "check_manually", "low_organic")
    return db


def main():
    print("=" * 68)
    print("  PCM Property Database Builder v3 (55-row MICE+RF+PMM source) — Uttarakhand")
    print(f"  Input CSV: {INPUT_CSV}")
    print("=" * 68)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Edit INPUT_CSV at the top of this script "
            "to point at PCM_Properties_cleaned_mice_pmm_detailed.csv.")

    db = load_all_pcm_rows(INPUT_CSV)
    db = add_derived(db)
    db = db.sort_values("Tm_C").reset_index(drop=True)
    db.to_csv(OUT_FILE, index=False)

    n_in_band = db["in_absolute_band"].sum()
    n_manuf = int((db["source"] == "manufacturer_datasheet_MICE_RF_PMM_completed").sum())
    n_lit = int((db["source"] == "literature_MICE_RF_PMM_completed").sum())
    n_imputed_rows = int(db["any_property_imputed"].sum())
    print(f"\n  Total candidates       : {len(db)}  ({n_manuf} manufacturer + {n_lit} literature)")
    print(f"  In 42-70C absolute band: {n_in_band}")
    print(f"  Tm range               : {db['Tm_C'].min():.1f} - {db['Tm_C'].max():.1f} C")
    print(f"  Rows with >=1 MICE-PMM-imputed property: "
          f"{n_imputed_rows}/{len(db)}")

    print("\n" + db[["name", "family", "Tm_C", "latent_heat_kJ_kg", "TC_W_mK",
                       "cycles_tested", "any_property_imputed", "in_absolute_band"]]
          .to_string(index=False))

    print(f"\n  Saved: {OUT_FILE}")
    print("=" * 68)
    print("\n07_feasibility_filter.py, 08_mcdm_ranking.py, and")
    print("09_recommendation_cards.py need NO changes — run them as before:")
    print("  python 07_feasibility_filter.py")
    print("  python 08_mcdm_ranking.py")
    print("  python 09_recommendation_cards.py")


if __name__ == "__main__":
    main()
