"""
05b_swh_design_specification.py
======================================
PHASE 4 — SWH SYSTEM DESIGN SPECIFICATION (Assam Project)

Establishes the physical solar water heater (SWH) system specifications,
clearly distinguishing constant physical design parameters from climate-dependent
energy requirements across Assam's clusters.

DESIGN CONSTANTS:
  - Target hot-water delivery temp: 50.0 °C
  - PCM target temperature (Tm_target): 44.0 °C
  - Approach temperature (dT_approach): 6.0 K
  - Daily hot-water demand: 100.0 L/day (~100.0 kg/day)
  - Morning draw (07:00 IST): 50.0 L
  - Evening draw (19:00 IST): 50.0 L
  - Tank PCM mass: 50.0 kg

CLIMATE-DEPENDENT QUANTITIES:
  - T_mains_est (°C)
  - Q_required (kWh/day)
  - Required PCM latent energy capacity per kg (kJ/kg for 50 kg PCM mass)

INPUTS:
  data/processed/clustering/gmm_cluster_assignments.csv
  data/processed/climate_signatures_raw.csv
  data/processed/clustering/gmm_cluster_profiles.csv

OUTPUTS:
  data/processed/design/swh_design_specification.csv
  data/preprocessed/swh_design_report.txt
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
ASSIGN_FILE = BASE_DIR / "data" / "processed" / "clustering" / "gmm_cluster_assignments.csv"
RAW_SIG_FILE = BASE_DIR / "data" / "processed" / "climate_signatures_raw.csv"

DESIGN_DIR = BASE_DIR / "data" / "processed" / "design"
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
DESIGN_DIR.mkdir(parents=True, exist_ok=True)

OUT_DESIGN_CSV = DESIGN_DIR / "swh_design_specification.csv"
OUT_DESIGN_REPORT = PREPROCESSED_DIR / "swh_design_report.txt"

# Design Constants (§4)
T_DELIVERY = 50.0       # °C
DT_APPROACH = 6.0       # K
TM_TARGET = 44.0        # °C (50 - 6)
DAILY_DEMAND_L = 100.0  # L/day
M_DRAW_KG = 100.0       # kg/day
MORNING_DRAW_L = 50.0   # L
EVENING_DRAW_L = 50.0   # L
PCM_MASS_KG = 50.0      # kg
CP_WATER_JKGK = 4186.0  # J/(kg·K)

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(str(msg))

def main():
    log("=" * 72)
    log("  PHASE 4 — SWH SYSTEM DESIGN SPECIFICATION (Assam Project)")
    log("=" * 72)

    # 1. Document System Design Constants
    log("\n[1] Physical System Design Constants Table:")
    constants_table = [
        {"Parameter": "Target Hot-Water Delivery Temperature", "Value": f"{T_DELIVERY:.1f} °C", "Type": "Design constant"},
        {"Parameter": "Heat-Exchanger Approach Temperature", "Value": f"{DT_APPROACH:.1f} K", "Type": "Design constant"},
        {"Parameter": "Target PCM Melting Temperature (Tm_target)", "Value": f"{TM_TARGET:.1f} °C", "Type": "Design constant"},
        {"Parameter": "Total Daily Domestic Hot-Water Demand", "Value": f"{DAILY_DEMAND_L:.1f} L/day (~{M_DRAW_KG:.1f} kg/day)", "Type": "Design constant"},
        {"Parameter": "Morning Hot-Water Draw (07:00 IST)", "Value": f"{MORNING_DRAW_L:.1f} L", "Type": "Design constant"},
        {"Parameter": "Evening Hot-Water Draw (19:00 IST)", "Value": f"{EVENING_DRAW_L:.1f} L", "Type": "Design constant"},
        {"Parameter": "Assumed Tank PCM Mass", "Value": f"{PCM_MASS_KG:.1f} kg", "Type": "Design constant"},
        {"Parameter": "Water Specific Heat Capacity", "Value": f"{CP_WATER_JKGK:.1f} J/(kg·K)", "Type": "Physical constant"},
    ]

    const_df = pd.DataFrame(constants_table)
    log(const_df.to_string(index=False))

    # 2. Load Assignments & Physical Signatures
    log("\n[2] Loading Phase 3 cluster assignments and raw signatures...")
    assign_df = pd.read_csv(ASSIGN_FILE)
    raw_sig_df = pd.read_csv(RAW_SIG_FILE)

    merged = pd.merge(assign_df, raw_sig_df, on="point_id")
    log(f"  Merged records count: {len(merged)} (Expected: 129)")

    # 3. Calculate Cluster-Wise Climate-Dependent Energy Requirements
    log("\n[3] Cluster-Wise Climate Energy Requirements:")
    cluster_ids = sorted(merged["cluster"].unique())

    cluster_specs = []

    for c_id in cluster_ids:
        sub = merged[merged["cluster"] == c_id]

        # Mains water estimation: T_mains ≈ max(5.0, Ta_mean - 6.0)
        t_mains_mean = np.maximum(5.0, sub["Ta_mean"] - 6.0).mean()

        # Q_required = m_draw * Cp * (T_delivery - T_mains_est) in kWh/day
        q_required_kWh = (M_DRAW_KG * CP_WATER_JKGK * (T_DELIVERY - t_mains_mean)) / 3_600_000

        # Required latent energy per kg of PCM (for 50 kg PCM design assumption)
        # Q_required (kWh/day) * 3600 (kJ/kWh) / 50 (kg)
        req_energy_kJ_kg = (q_required_kWh * 3600.0) / PCM_MASS_KG

        c_spec = {
            "Cluster": c_id,
            "Grid_Points_Count": len(sub),
            "T_mains_est_degC": round(t_mains_mean, 2),
            "Q_required_kWh_day": round(q_required_kWh, 3),
            "Required_PCM_energy_kJ_kg_for_50kg": round(req_energy_kJ_kg, 2),
            "Type": "Climate-dependent"
        }
        cluster_specs.append(c_spec)

    cluster_spec_df = pd.DataFrame(cluster_specs)
    log(cluster_spec_df.to_string(index=False))

    # Save to CSV
    # Combine constants and cluster-wise specs in structured CSV output
    with open(OUT_DESIGN_CSV, "w", encoding="utf-8") as f:
        f.write("# SYSTEM DESIGN CONSTANTS\n")
        const_df.to_csv(f, index=False)
        f.write("\n# CLUSTER-WISE CLIMATE-DEPENDENT REQUIREMENTS\n")
        cluster_spec_df.to_csv(f, index=False)

    log(f"\n  Saved SWH Design Specification CSV to: {OUT_DESIGN_CSV}")

    # 4. Save Comprehensive Phase 4 Summary Report
    with open(OUT_DESIGN_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"  Saved SWH Design Summary Report to: {OUT_DESIGN_REPORT}")

    log("\n" + "=" * 72)
    log("  PHASE 4 COMPLETE")
    log("=" * 72)

if __name__ == "__main__":
    main()
