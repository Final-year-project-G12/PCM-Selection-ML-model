"""
10_physics_validation.py
============================
PHASE 9 - PHYSICS-BASED VALIDATION (Assam SWH PCM System)

Corrected Solar Water Heater (SWH) + PCM Grey-Box Thermal Model.
Performs 10-year (2016-2025) sub-hourly physics simulation across Assam climate clusters.

KEY PHYSICAL & METHODOLOGICAL NOTES:
--------------------------------------
1. Water Demand: Morning (07:00 IST) = 50 kg, Evening (19:00 IST) = 50 kg. Total = 100 kg/day.
2. Tank Water Mass Mw = 100 kg, PCM Mass Mp = 50 kg (fixed design parameters).
3. Collector Model: Hottel-Whillier-Bliss (Ac = 2.0 m2, FR_tau_alpha = 0.72, FR_UL = 4.5 W/m2K).
4. Tank Heat Loss: Explicit ambient loss (UA_tank = 1.0 W/K).
5. PCM Heat Transfer: (UA)_pcm = 375.0 W/K (h_pcm = 150 W/m2K, A_pcm = 2.5 m2).
6. 4-State Path-Dependent Enthalpy Formulation:
   (PHASE_LIQUID, PHASE_FREEZING, PHASE_SOLID, PHASE_MELTING) with exact boundary clipping,
   supercooling hysteresis (T_freeze = Tm - DeltaT_supercooling), and Cp_l != Cp_s energy continuity.
7. Sub-Hourly Integration: dt = 300 s (5 minutes, 12 sub-steps/hour).
8. Climate Forcing: Chronological 10-year ERA5 forcing (2016-2025) reconstructed from NetCDF files.
9. Spin-Up: 2016 warm-up loop until starting state converges across 4 variables (Tw, Tp, f_melt, H_sys).
10. Complete First-Law System Balance: Cumulative relative error < 0.1%.
11. Delivery Success Evaluation: Evaluated at 07:00 and 19:00 BEFORE mains water replenishment.
12. MCDM Status: Phase 7 n_confirmed = 0 for all clusters; no MCDM ranking performed.

PIPELINE-VERSION INCONSISTENCY (DOCUMENTED):
---------------------------------------------
Phase 6 feasibility screening (07_feasibility_filter.py) and Phase 7 MCDM ranking
(08_mcdm_ranking.py) were executed against the K=4 cluster profile (cluster IDs 0-3).
Phase 3 was subsequently re-locked to a final K=3 GMM model (cluster IDs 0-2).
The current cluster_profiles_assam.csv on disk reflects the K=3 model.

Phase 6 and Phase 7 output files are treated as locked historical artifacts and are
NOT modified. Phase 9 uses:
  - Candidate PCMs: Phase 6 feasibility survivors (passes_all=True) from
    feasibility_survivors_assam.csv / pcm_database_assam.csv. These were screened
    under the K=4 pipeline but the 7-constraint feasibility logic is climate-independent.
    They are labelled "Phase 6-screened candidates".
  - Climate forcing: Phase 3 final K=3 medoids (ASP_0012, ASP_0092, ASP_0028),
    derived programmatically from the 5-feature GMM standardization.
  The two cluster sets originate from different pipeline versions. This is a
  documented pipeline-version inconsistency; Phase 6 files are not rerun.

ERA5 SSRD RECONSTRUCTION:
--------------------------
Accumulated ERA5 ssrd (J/m2) is de-accumulated per interval. ERA5 accumulated forecast
variables represent accumulation over the interval LEADING UP TO each forecast step
(i.e., each timestep t_i carries radiation accumulated from the previous step t_{i-1}).
The de-accumulated interval energy E_i (J/m2) is distributed to overlapping IST hourly
bins proportionally to overlap duration, assuming uniform mean irradiance within each
accumulation interval. This provides an energy-conserving hourly temporal reconstruction
but does NOT reproduce the unknown true sub-hourly solar profile.

UTC+05:30 TIMESTAMP FLOORING APPROXIMATION:
--------------------------------------------
The UTC+05:30 timestamp flooring introduces a 30-minute temporal attribution approximation.
It is retained consistently across the 10-year simulation and documented as a preprocessing
limitation; its impact is not independently quantified.

ASSUMED MODEL PARAMETERS (not site-measured values):
------------------------------------------------------
  FR_TAU_ALPHA = 0.72            Zero-loss optical efficiency (assumed; typical FPC 0.65-0.80)
  FR_UL_WM2K   = 4.5 W/m2K      Collector heat loss coeff (assumed; typical FPC 3.5-6.0)
  UA_TANK_WK   = 1.0 W/K        Tank heat loss conductance (assumed; no measured insulation data)
  UA_PCM_WK    = 375.0 W/K      PCM-water conductance (assumed: h=150 W/m2K * A=2.5 m2)
  Cp_default   = 2000 J/kg/K    Applied when PCM Cp_solid/Cp_liquid is absent in database
  Tmains       = Tdaily_mean - 6 C   Empirical mains water temperature approximation
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc

from config import PROCESSED_DIR, PREPROCESSED_DIR, RAW_POINTS_DIR

# Paths
ASSIGN_FILE    = PROCESSED_DIR / "clustering" / "cluster_assignments_assam.csv"
SIG_RAW_FILE   = PROCESSED_DIR / "climate_signatures_raw.csv"
GRID_FILE      = PROCESSED_DIR / "population_grid_points.csv"
# Phase 6 candidate sources (pcm_database_assam.csv + feasibility_survivors_assam.csv)
PCM_ASSAM_FILE = PROCESSED_DIR / "pcm" / "pcm_database_assam.csv"
FEAS_FILE      = PROCESSED_DIR / "pcm" / "feasibility_survivors_assam.csv"

OUT_RESULTS_CSV = PROCESSED_DIR / "pcm" / "physics_validation_assam.csv"
OUT_REPORT_TXT  = PREPROCESSED_DIR / "physics_validation_report.txt"

# -----------------------------------------------------------------------------
# FIXED PHYSICAL SYSTEM DESIGN CONSTANTS (§1, §3, §4, §5)
# -----------------------------------------------------------------------------
M_W_KG = 100.0           # Tank water mass (kg)
C_W_JKGK = 4186.0        # Water specific heat capacity (J/kg·K)
M_P_KG = 50.0            # PCM mass (kg)
T_DELIVERY_C = 50.0      # Target hot water delivery temperature (°C)
T_M_TARGET_C = 44.0      # Target PCM melting temperature (°C)

# Collector parameters (Hottel-Whillier-Bliss)
A_C_M2 = 2.0             # Solar collector area (m²)
FR_TAU_ALPHA = 0.72      # Zero-loss optical efficiency
FR_UL_WM2K = 4.5         # Overall thermal loss coefficient (W/m²K)

# Heat transfer conductances
UA_TANK_WK = 1.0         # Tank heat loss conductance to ambient (W/K)
UA_PCM_WK = 375.0        # PCM-water thermal conductance (W/K)

# Water draw schedule
DRAW_HOURS = [7, 19]     # IST local draw hours (07:00 AM and 07:00 PM)
DRAW_MASS_KG = 50.0      # Draw mass per event (kg)

# Simulation timestep & spin-up settings
DT_SEC = 300.0           # Sub-hourly integration timestep (seconds = 5 minutes)
T_REF_C = 0.0            # Reference temperature for enthalpy state (0 °C)

GMM_FEATURE_COLS = ["GHI_mean", "Ta_mean", "DTR", "RH_mean", "wind_mean"]

report_lines = []

def log(msg=""):
    print(msg, flush=True)
    report_lines.append(str(msg))


# -----------------------------------------------------------------------------
# 1. PROGRAMMATIC TRUE MEDOID SELECTION (§1)
# -----------------------------------------------------------------------------
def derive_true_medoids(assign_df, sig_raw_df):
    """
    Derives true cluster medoids using pairwise Euclidean distance in the 5
    standardized Phase 3 GMM input features (excluding point_id).
    """
    merged = pd.merge(assign_df[["point_id", "cluster"]], sig_raw_df[["point_id"] + GMM_FEATURE_COLS], on="point_id")
    
    # Standardize 5 GMM features across dataset
    X_raw = merged[GMM_FEATURE_COLS].values
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_raw - X_mean) / X_std
    
    merged_norm = pd.DataFrame(X_norm, columns=GMM_FEATURE_COLS)
    merged_norm["point_id"] = merged["point_id"]
    merged_norm["cluster"] = merged["cluster"]
    
    medoids = {}
    for c_id in sorted(merged["cluster"].unique()):
        sub = merged_norm[merged_norm["cluster"] == c_id]
        X_sub = sub[GMM_FEATURE_COLS].values
        # Compute pairwise Euclidean distance matrix
        dist_mat = np.sqrt(((X_sub[:, np.newaxis, :] - X_sub[np.newaxis, :, :]) ** 2).sum(axis=2))
        sum_dists = dist_mat.sum(axis=1)
        medoid_idx = sum_dists.argmin()
        medoid_pt = sub["point_id"].iloc[medoid_idx]
        medoids[c_id] = medoid_pt
    return medoids


# -----------------------------------------------------------------------------
# 2. ERA5 10-YEAR HOURLY CLIMATE FORCING RECONSTRUCTION (§2)
# -----------------------------------------------------------------------------
IST_OFFSET_S = 5 * 3600 + 30 * 60   # UTC+05:30 in seconds = 19800 s

def _duration_overlap_ssrd(t_sec_arr, ssrd_raw_arr):
    """
    De-accumulates ERA5 ssrd and distributes interval energy to IST hourly bins
    using duration-overlap allocation.

    ERA5 SEMANTICS: accumulated forecast variables represent accumulation over the
    interval LEADING UP TO each forecast step. Each valid_time[i] is the END of the
    accumulation interval; the interval starts at valid_time[i-1] (or at the forecast
    day boundary for the first step of each forecast run).

    For each interval i:
        E_i  = ssrd_diff[i]                  [J/m2]  de-accumulated interval energy
        dt_i = t_end_i - t_start_i           [s]     actual interval duration

    Energy is distributed to each overlapping IST hourly bin h proportionally:
        E_ih = E_i * overlap_h / dt_i
    where overlap_h = duration of [t_start_i, t_end_i) intersecting IST hour h.

    This assumes uniform mean irradiance within each accumulation interval.
    It provides an energy-conserving temporal reconstruction but does NOT reproduce
    the unknown true sub-hourly solar profile.

    Returns:
        hourly_ssrd_J : pd.Series  index=IST Timestamp (floored to hour), values=J/m2
        raw_total_J   : float      sum of all de-accumulated interval energies
        conservation_err_pct : float  |reconstructed - raw| / raw * 100
    """
    n = len(t_sec_arr)
    t_sec = t_sec_arr.astype(np.int64)
    ssrd  = np.asarray(ssrd_raw_arr, dtype=np.float64)

    # --- De-accumulate per-interval energy and duration ---
    ssrd_diff = np.zeros(n, dtype=np.float64)  # J/m2 per interval
    dt_arr    = np.zeros(n, dtype=np.float64)  # duration (s) per interval

    # Interval 0: starts at UTC midnight of the day containing t_sec[0]
    t0_day_start = int(t_sec[0]) // 86400 * 86400  # UTC midnight
    dt_arr[0]    = max(float(t_sec[0] - t0_day_start), 3600.0)
    ssrd_diff[0] = max(float(ssrd[0]), 0.0)

    for i in range(1, n):
        dt_i       = float(t_sec[i] - t_sec[i - 1])
        dt_arr[i]  = dt_i if dt_i > 0 else 3600.0
        if ssrd[i] >= ssrd[i - 1]:
            ssrd_diff[i] = max(float(ssrd[i] - ssrd[i - 1]), 0.0)
        else:
            # Accumulation reset (forecast re-initialization): treat post-reset value
            # as fresh accumulation from the start of the new forecast run.
            ssrd_diff[i] = max(float(ssrd[i]), 0.0)

    raw_total_J = float(ssrd_diff.sum())

    # --- Duration-overlap allocation to IST hourly bins ---
    # Convert interval end/start to IST epoch seconds
    t_end_ist   = t_sec.astype(np.float64) + IST_OFFSET_S
    t_start_ist = t_end_ist - dt_arr

    hourly_J = {}   # key: IST hour bin start (epoch s), value: accumulated J/m2

    for i in range(n):
        if ssrd_diff[i] <= 0.0:
            continue
        E_i  = ssrd_diff[i]
        dt_i = dt_arr[i]
        s    = t_start_ist[i]
        e    = t_end_ist[i]

        # Walk each 3600-s IST bin that overlaps [s, e)
        bin_s = int(s) // 3600 * 3600
        bin_e = bin_s + 3600
        while bin_s < e:
            overlap = min(e, bin_e) - max(s, bin_s)
            if overlap > 0.0:
                key = bin_s
                hourly_J[key] = hourly_J.get(key, 0.0) + E_i * (overlap / dt_i)
            bin_s = bin_e
            bin_e += 3600

    # Convert bin keys (IST epoch seconds) to Timestamps
    keys_sorted = sorted(hourly_J.keys())
    ts_index    = pd.to_datetime(keys_sorted, unit="s")
    hourly_ssrd_J = pd.Series([hourly_J[k] for k in keys_sorted], index=ts_index)

    # --- Energy conservation check (BEFORE nighttime clamping) ---
    reconstructed_J      = float(hourly_ssrd_J.sum())
    conservation_err_pct = abs(reconstructed_J - raw_total_J) / max(raw_total_J, 1.0) * 100.0

    return hourly_ssrd_J, raw_total_J, conservation_err_pct


def load_era5_hourly_forcing(point_id, grid_df):
    """
    Reconstructs 10-year (2016-2025) hourly climate forcing (Tamb, Isolar, Tmains)
    from raw ERA5 NetCDF files for a given grid point.

    SSRD reconstruction uses duration-overlap allocation (see _duration_overlap_ssrd).
    UTC+05:30 flooring introduces a 30-minute temporal attribution approximation;
    retained consistently across 10 years and documented as a preprocessing limitation.

    Returns:
        hourly_df              : pd.DataFrame  columns [Tamb, Isolar, Tmains]
        ssrd_raw_total_J       : float         sum of de-accumulated interval energies
        ssrd_conservation_err  : float         |recon - raw| / raw * 100 (%)
        ssrd_nightclamp_loss_J : float         energy removed by nighttime clamping
    """
    pt_info = grid_df[grid_df["point_id"] == point_id].iloc[0]
    lat, lon = pt_info["lat"], pt_info["lon"]

    inst_records = []
    t_acc_all    = []   # raw UTC timestamps for ssrd
    ssrd_all     = []   # raw accumulated ssrd values

    for y in range(2016, 2026):
        for m in range(1, 13):
            inst_f = RAW_POINTS_DIR / f"era5_AS_points_{y}_{m:02d}_instant.nc"
            acc_f  = RAW_POINTS_DIR / f"era5_AS_points_{y}_{m:02d}_accum.nc"
            if not (inst_f.exists() and acc_f.exists()):
                continue

            ds_inst = nc.Dataset(inst_f)
            ds_acc  = nc.Dataset(acc_f)

            lats    = ds_inst.variables["latitude"][:]
            lons    = ds_inst.variables["longitude"][:]
            lat_idx = int(np.abs(lats - lat).argmin())
            lon_idx = int(np.abs(lons - lon).argmin())

            t_inst = np.array(ds_inst.variables["valid_time"][:], dtype=np.int64)
            t2m_k  = np.array(ds_inst.variables["t2m"][:, lat_idx, lon_idx], dtype=np.float64) - 273.15

            t_acc_m  = np.array(ds_acc.variables["valid_time"][:], dtype=np.int64)
            ssrd_m   = np.array(ds_acc.variables["ssrd"][:, lat_idx, lon_idx], dtype=np.float64)

            ds_inst.close()
            ds_acc.close()

            inst_records.append(pd.DataFrame({"time_sec": t_inst, "Tamb": t2m_k}))
            t_acc_all.append(t_acc_m)
            ssrd_all.append(ssrd_m)

    # --- Ambient temperature: group by floored IST hour, mean ---
    # UTC+05:30 flooring is a preprocessing approximation (30-min attribution shift).
    full_inst = (
        pd.concat(inst_records)
        .drop_duplicates(subset=["time_sec"])
        .sort_values("time_sec")
        .reset_index(drop=True)
    )
    full_inst["dt_ist"] = (
        pd.to_datetime(full_inst["time_sec"], unit="s")
        + pd.Timedelta(seconds=IST_OFFSET_S)
    ).dt.floor("1h")
    tamb_hourly = full_inst.groupby("dt_ist")["Tamb"].mean()

    # --- SSRD: de-duplicate and sort before de-accumulation ---
    t_acc_arr  = np.concatenate(t_acc_all)
    ssrd_arr   = np.concatenate(ssrd_all)
    sort_idx   = np.argsort(t_acc_arr, kind="stable")
    t_acc_arr  = t_acc_arr[sort_idx]
    ssrd_arr   = ssrd_arr[sort_idx]
    # Remove duplicate timestamps (keep first occurrence after sorting)
    _, unique_idx = np.unique(t_acc_arr, return_index=True)
    t_acc_arr  = t_acc_arr[unique_idx]
    ssrd_arr   = ssrd_arr[unique_idx]

    # Duration-overlap SSRD reconstruction
    hourly_ssrd_J, raw_total_J, conservation_err_pct = _duration_overlap_ssrd(
        t_acc_arr, ssrd_arr
    )
    # Convert J/m2 per IST hour -> W/m2 (hourly mean irradiance)
    hourly_isolar = hourly_ssrd_J / 3600.0

    # --- Build complete 10-year hourly index ---
    hourly_index = pd.date_range(
        start="2016-01-01 00:00:00", end="2025-12-31 23:00:00", freq="1h"
    )
    hourly_df = pd.DataFrame(index=hourly_index)
    hourly_df["Tamb"]   = tamb_hourly.reindex(hourly_index).interpolate(method="time").bfill().ffill()
    hourly_df["Isolar"] = (
        hourly_isolar.reindex(hourly_index).interpolate(method="time").fillna(0.0).clip(lower=0.0)
    )

    # --- Nighttime clamping (AFTER conservation check) ---
    # Report energy removed before zeroing.
    pre_clamp_J    = float((hourly_df["Isolar"] * 3600.0).sum())
    hours_col      = hourly_df.index.hour
    night_mask     = (hours_col < 5) | (hours_col > 18)
    hourly_df.loc[night_mask, "Isolar"] = 0.0
    post_clamp_J   = float((hourly_df["Isolar"] * 3600.0).sum())
    nightclamp_loss_J = pre_clamp_J - post_clamp_J

    # --- Mains water temperature (empirical approximation) ---
    daily_ta_mean      = hourly_df["Tamb"].resample("D").transform("mean")
    hourly_df["Tmains"] = np.maximum(5.0, daily_ta_mean - 6.0)

    return hourly_df, raw_total_J, conservation_err_pct, nightclamp_loss_J


# -----------------------------------------------------------------------------
# 3. ENTHALPY-CONTINUOUS PCM STATE UPDATE ENGINE (§6)
# -----------------------------------------------------------------------------
class PCMStateNode:
    """
    Thermodynamically complete PCM thermal node supporting 4-state path-dependent
    supercooling hysteresis (PHASE_LIQUID, PHASE_FREEZING, PHASE_SOLID, PHASE_MELTING)
    with exact boundary clipping and latent energy conservation.
    """
    def __init__(self, pcm_row, Mp=M_P_KG, T_ref=T_REF_C):
        self.Mp = Mp
        self.Tm = float(pcm_row["Tm_C"])
        self.L = float(pcm_row["latent_heat_kJ_kg"]) * 1000.0  # J/kg
        self.T_ref = T_ref
        
        # Specific heats
        cp_s = pcm_row.get("Cp_solid_kJ_kgK", np.nan)
        cp_l = pcm_row.get("Cp_liquid_kJ_kgK", np.nan)
        self.Cp_s = float(cp_s) * 1000.0 if pd.notna(cp_s) else 2000.0  # J/kg·K
        self.Cp_l = float(cp_l) * 1000.0 if pd.notna(cp_l) else 2000.0  # J/kg·K
        
        # Supercooling temperature
        sc = pcm_row.get("supercooling_K", pcm_row.get("supercooling_degC", np.nan))
        if pd.notna(sc) and float(sc) > 0.0:
            self.delta_T_sc = float(sc)
            self.has_supercooling_data = True
        else:
            self.delta_T_sc = 0.0
            self.has_supercooling_data = False
            
        self.T_freeze = self.Tm - self.delta_T_sc
        
        # Base Enthalpy Reference Levels (relative to T_ref = 0 °C solid)
        # Solid at T_freeze (0% melt, end of freezing / start of solid cooling)
        self.H_freeze_end = self.Mp * self.Cp_s * (self.T_freeze - self.T_ref)
        # Solid at Tm (0% melt, start of melting plateau)
        self.H_solid_at_Tm = self.H_freeze_end + self.Mp * self.Cp_s * self.delta_T_sc
        # Liquid at T_freeze (100% melt, start of freezing plateau)
        self.H_freeze_start = self.H_freeze_end + self.Mp * self.L
        # Liquid at Tm (100% melt, end of melting plateau)
        self.H_melt_end = self.H_freeze_start + self.Mp * self.Cp_l * self.delta_T_sc
        
        # Initial State
        self.reset_state(T_initial=20.0)

    def reset_state(self, T_initial=20.0, f_melt_initial=None):
        self.Tp = float(T_initial)
        
        if self.Tp > self.Tm:
            self.mode = "LIQUID"
            self.f_melt = 1.0
            self.Hp = self.H_melt_end + self.Mp * self.Cp_l * (self.Tp - self.Tm)
        elif self.Tp < self.T_freeze:
            self.mode = "SOLID"
            self.f_melt = 0.0
            self.Hp = self.H_freeze_end - self.Mp * self.Cp_s * (self.T_freeze - self.Tp)
        else:
            # T_freeze <= Tp <= Tm
            if f_melt_initial is None:
                f_melt_initial = 1.0 if self.Tp == self.Tm else 0.0
            self.f_melt = float(f_melt_initial)
            
            if self.f_melt >= 1.0:
                self.mode = "LIQUID"
                self.f_melt = 1.0
                self.Hp = self.H_freeze_start + self.Mp * self.Cp_l * (self.Tp - self.T_freeze)
            elif self.f_melt <= 0.0:
                self.mode = "SOLID"
                self.f_melt = 0.0
                self.Hp = self.H_solid_at_Tm - self.Mp * self.Cp_s * (self.Tm - self.Tp)
            else:
                self.mode = "MELTING"
                self.Tp = self.Tm
                self.Hp = self.H_solid_at_Tm + self.f_melt * (self.Mp * self.L)

    def update_enthalpy(self, dH_step):
        """
        Updates PCM enthalpy by dH_step using exact phase boundary clipping
        and 4-state path mode transitions (LIQUID, FREEZING, SOLID, MELTING).
        """
        dH_rem = dH_step
        
        while abs(dH_rem) > 1e-12:
            if self.mode == "LIQUID":
                if dH_rem >= 0:
                    # Heating liquid above T_freeze / Tm
                    self.Hp += dH_rem
                    self.Tp = self.Tm + (self.Hp - self.H_melt_end) / (self.Mp * self.Cp_l)
                    self.f_melt = 1.0
                    dH_rem = 0.0
                else:
                    # Cooling liquid down towards H_freeze_start (T_freeze)
                    dH_avail = self.H_freeze_start - self.Hp # <= 0
                    if dH_rem > dH_avail:
                        self.Hp += dH_rem
                        self.Tp = self.T_freeze + (self.Hp - self.H_freeze_start) / (self.Mp * self.Cp_l)
                        self.f_melt = 1.0
                        dH_rem = 0.0
                    else:
                        self.Hp = self.H_freeze_start
                        self.Tp = self.T_freeze
                        self.f_melt = 1.0
                        dH_rem -= dH_avail
                        self.mode = "FREEZING"

            elif self.mode == "FREEZING":
                if dH_rem <= 0:
                    # Isothermal freezing plateau at T_freeze
                    dH_avail = self.H_freeze_end - self.Hp # <= 0
                    if dH_rem > dH_avail:
                        self.Hp += dH_rem
                        self.f_melt = (self.Hp - self.H_freeze_end) / (self.Mp * self.L)
                        self.Tp = self.T_freeze
                        dH_rem = 0.0
                    else:
                        self.Hp = self.H_freeze_end
                        self.f_melt = 0.0
                        self.Tp = self.T_freeze
                        dH_rem -= dH_avail
                        self.mode = "SOLID"
                else:
                    # Reheating while freezing -> increases f_melt towards H_freeze_start (T_freeze)
                    dH_avail = self.H_freeze_start - self.Hp # >= 0
                    if dH_rem < dH_avail:
                        self.Hp += dH_rem
                        self.f_melt = (self.Hp - self.H_freeze_end) / (self.Mp * self.L)
                        self.Tp = self.T_freeze
                        dH_rem = 0.0
                    else:
                        self.Hp = self.H_freeze_start
                        self.f_melt = 1.0
                        self.Tp = self.T_freeze
                        dH_rem -= dH_avail
                        self.mode = "LIQUID"

            elif self.mode == "SOLID":
                if dH_rem <= 0:
                    # Cooling solid below T_freeze / Tm
                    self.Hp += dH_rem
                    self.Tp = self.T_freeze + (self.Hp - self.H_freeze_end) / (self.Mp * self.Cp_s)
                    self.f_melt = 0.0
                    dH_rem = 0.0
                else:
                    # Heating solid up towards H_solid_at_Tm (Tm)
                    dH_avail = self.H_solid_at_Tm - self.Hp # >= 0
                    if dH_rem < dH_avail:
                        self.Hp += dH_rem
                        self.Tp = self.T_freeze + (self.Hp - self.H_freeze_end) / (self.Mp * self.Cp_s)
                        self.f_melt = 0.0
                        dH_rem = 0.0
                    else:
                        self.Hp = self.H_solid_at_Tm
                        self.Tp = self.Tm
                        self.f_melt = 0.0
                        dH_rem -= dH_avail
                        self.mode = "MELTING"

            elif self.mode == "MELTING":
                if dH_rem >= 0:
                    # Isothermal melting plateau at Tm
                    dH_avail = self.H_melt_end - self.Hp # >= 0
                    if dH_rem < dH_avail:
                        self.Hp += dH_rem
                        self.f_melt = (self.Hp - self.H_solid_at_Tm) / (self.Mp * self.L)
                        self.Tp = self.Tm
                        dH_rem = 0.0
                    else:
                        self.Hp = self.H_melt_end
                        self.f_melt = 1.0
                        self.Tp = self.Tm
                        dH_rem -= dH_avail
                        self.mode = "LIQUID"
                else:
                    # Cooling while melting -> decreases f_melt towards H_solid_at_Tm (Tm)
                    dH_avail = self.H_solid_at_Tm - self.Hp # <= 0
                    if dH_rem > dH_avail:
                        self.Hp += dH_rem
                        self.f_melt = (self.Hp - self.H_solid_at_Tm) / (self.Mp * self.L)
                        self.Tp = self.Tm
                        dH_rem = 0.0
                    else:
                        self.Hp = self.H_solid_at_Tm
                        self.f_melt = 0.0
                        self.Tp = self.Tm
                        dH_rem -= dH_avail
                        self.mode = "SOLID"


# -----------------------------------------------------------------------------
# 4. SUB-HOURLY 10-YEAR THERMAL SIMULATION ENGINE (§7, §8, §9)
# -----------------------------------------------------------------------------
def simulate_pcm_swh_10year(forcing_df, pcm_row, dt_sec=DT_SEC):
    """
    Executes chronological sub-hourly simulation over 10-year climate forcing
    following a 2016 warm-up spin-up convergence period.
    Optimized using pre-extracted numpy arrays for high performance.
    """
    substeps_per_hour = int(3600.0 / dt_sec)
    
    # Pre-extract numpy arrays from forcing_df
    forcing_2016 = forcing_df[forcing_df.index.year == 2016]
    tamb_2016 = forcing_2016["Tamb"].values
    isolar_2016 = forcing_2016["Isolar"].values
    tmains_2016 = forcing_2016["Tmains"].values
    hours_2016 = forcing_2016.index.hour.values
    
    # 1. Warm-Up Spin-Up Loop on 2016 Climate Data
    pcm = PCMStateNode(pcm_row)
    Tw = tmains_2016[0]
    pcm.reset_state(T_initial=Tw, f_melt_initial=0.0)
    
    max_spinup_cycles = 15
    converged = False
    spinup_cycles_run = 0
    
    for cycle in range(1, max_spinup_cycles + 1):
        spinup_cycles_run = cycle
        Tw_start = Tw
        Tp_start = pcm.Tp
        fm_start = pcm.f_melt
        Hsys_start = M_W_KG * C_W_JKGK * Tw + pcm.Hp
        
        for i in range(len(forcing_2016)):
            tamb = tamb_2016[i]
            isolar = isolar_2016[i]
            tmains = tmains_2016[i]
            h_local = hours_2016[i]
            
            for step in range(substeps_per_hour):
                sub_min = step * (60 // substeps_per_hour)
                if sub_min == 0 and h_local in DRAW_HOURS:
                    Tw = 0.5 * Tw + 0.5 * tmains
                    
                Qc = max(0.0, A_C_M2 * FR_TAU_ALPHA * isolar - A_C_M2 * FR_UL_WM2K * max(0.0, Tw - tamb))
                Qloss = UA_TANK_WK * (Tw - tamb)
                Q_pcm_water = UA_PCM_WK * (Tw - pcm.Tp)
                
                pcm.update_enthalpy(Q_pcm_water * dt_sec)
                dTw = ((Qc - Qloss - Q_pcm_water) * dt_sec) / (M_W_KG * C_W_JKGK)
                Tw += dTw
                
        Tw_end = Tw
        Tp_end = pcm.Tp
        fm_end = pcm.f_melt
        Hsys_end = M_W_KG * C_W_JKGK * Tw + pcm.Hp
        
        dT_w_diff = abs(Tw_end - Tw_start)
        dT_p_diff = abs(Tp_end - Tp_start)
        df_m_diff = abs(fm_end - fm_start)
        dH_sys_rel = abs(Hsys_end - Hsys_start) / max(abs(Hsys_start), 1e-6)
        
        if dT_w_diff < 0.05 and dT_p_diff < 0.05 and df_m_diff < 0.001 and dH_sys_rel < 0.0001:
            converged = True
            break
            
    # 2. Chronological 10-Year Validation Simulation (2016-2025)
    tamb_all = forcing_df["Tamb"].values
    isolar_all = forcing_df["Isolar"].values
    tmains_all = forcing_df["Tmains"].values
    hours_all = forcing_df.index.hour.values
    n_hours = len(forcing_df)
    
    E_solar_total = 0.0
    E_refill_total = 0.0
    E_loss_total = 0.0
    E_draw_total = 0.0
    
    H_sys_initial = M_W_KG * C_W_JKGK * Tw + pcm.Hp
    
    hours_tw_ge_50 = 0
    morning_draws_total = 0
    morning_draws_success = 0
    evening_draws_total = 0
    evening_draws_success = 0
    
    complete_pcm_cycles = 0
    reached_full_melt = False
    
    max_step_residual = 0.0
    sum_abs_step_residual = 0.0
    total_substeps = 0
    
    for i in range(n_hours):
        tamb = tamb_all[i]
        isolar = isolar_all[i]
        tmains = tmains_all[i]
        h_local = hours_all[i]
        
        if Tw >= T_DELIVERY_C:
            hours_tw_ge_50 += 1
            
        for step in range(substeps_per_hour):
            total_substeps += 1
            
            E_refill_step = 0.0
            E_draw_step = 0.0
            
            if step == 0 and h_local in DRAW_HOURS:
                is_success = (Tw >= T_DELIVERY_C)
                draw_energy = DRAW_MASS_KG * C_W_JKGK * (Tw - T_REF_C)
                refill_energy = DRAW_MASS_KG * C_W_JKGK * (tmains - T_REF_C)
                
                E_draw_total += draw_energy
                E_refill_total += refill_energy
                
                E_draw_step = draw_energy
                E_refill_step = refill_energy
                
                if h_local == 7:
                    morning_draws_total += 1
                    if is_success:
                        morning_draws_success += 1
                else:
                    evening_draws_total += 1
                    if is_success:
                        evening_draws_success += 1
                        
                E_w_before = M_W_KG * C_W_JKGK * (Tw - T_REF_C)
                H_p_before = pcm.Hp
                
                Tw = 0.5 * Tw + 0.5 * tmains
            else:
                E_w_before = M_W_KG * C_W_JKGK * (Tw - T_REF_C)
                H_p_before = pcm.Hp
                
            Qc = max(0.0, A_C_M2 * FR_TAU_ALPHA * isolar - A_C_M2 * FR_UL_WM2K * max(0.0, Tw - tamb))
            E_solar_step = Qc * dt_sec
            E_solar_total += E_solar_step
            
            Qloss = UA_TANK_WK * (Tw - tamb)
            E_loss_step = Qloss * dt_sec
            E_loss_total += E_loss_step
            
            Q_pcm_water = UA_PCM_WK * (Tw - pcm.Tp)
            pcm.update_enthalpy(Q_pcm_water * dt_sec)
            
            dTw = ((Qc - Qloss - Q_pcm_water) * dt_sec) / (M_W_KG * C_W_JKGK)
            Tw += dTw
            
            E_w_after = M_W_KG * C_W_JKGK * (Tw - T_REF_C)
            H_p_after = pcm.Hp
            
            E_in_step = E_solar_step + E_refill_step
            E_out_step = E_loss_step + E_draw_step
            
            dE_stored_step = (E_w_after - E_w_before) + (H_p_after - H_p_before)
            residual_step = abs(E_in_step - E_out_step - dE_stored_step)
            
            if residual_step > max_step_residual:
                max_step_residual = residual_step
            sum_abs_step_residual += residual_step
            
            if pcm.f_melt >= 0.999:
                reached_full_melt = True
            elif pcm.f_melt <= 0.001 and reached_full_melt:
                complete_pcm_cycles += 1
                reached_full_melt = False

    H_sys_final = M_W_KG * C_W_JKGK * Tw + pcm.Hp
    dE_stored_total = H_sys_final - H_sys_initial
    
    E_in_total = E_solar_total + E_refill_total
    E_out_total = E_loss_total + E_draw_total
    
    cum_abs_residual = abs(E_in_total - E_out_total - dE_stored_total)
    cum_rel_error_pct = (cum_abs_residual / max(E_in_total, 1.0)) * 100.0
    mean_step_residual = sum_abs_step_residual / max(total_substeps, 1)
    
    morning_rate = morning_draws_success / max(morning_draws_total, 1)
    evening_rate = evening_draws_success / max(evening_draws_total, 1)
    overall_rate = (morning_draws_success + evening_draws_success) / max(morning_draws_total + evening_draws_total, 1)
    
    useful_solar_delivered = E_solar_total - E_loss_total
    solar_fraction = min(1.0, max(0.0, useful_solar_delivered / max(E_draw_total, 1.0)))
    
    return {
        "spinup_cycles": spinup_cycles_run,
        "spinup_converged": converged,
        "morning_delivery_success_rate": morning_rate,
        "evening_delivery_success_rate": evening_rate,
        "overall_delivery_success_rate": overall_rate,
        "hours_Tw_ge_50C_per_year": hours_tw_ge_50 / 10.0,
        "solar_fraction": solar_fraction,
        "complete_pcm_cycles_10yr": complete_pcm_cycles,
        "complete_pcm_cycles_per_year": complete_pcm_cycles / 10.0,
        "max_step_residual_J": max_step_residual,
        "mean_step_residual_J": mean_step_residual,
        "cum_abs_residual_J": cum_abs_residual,
        "cum_rel_error_pct": cum_rel_error_pct,
    }


# -----------------------------------------------------------------------------
# MAIN EXECUTION SCRIPT (§13, §14)
# -----------------------------------------------------------------------------
def main():
    log("=" * 76)
    log("  PHASE 9 -- PHYSICS-BASED VALIDATION & 10-YEAR SIMULATION (ASSAM)")
    log("=" * 76)

    log("\n[1] Loading Phase 3 assignments, raw signatures, and Phase 6 PCM candidates...")
    assign_df  = pd.read_csv(ASSIGN_FILE)
    sig_raw_df = pd.read_csv(SIG_RAW_FILE)
    grid_df    = pd.read_csv(GRID_FILE)

    # ---- Phase 6 candidate universe (locked historical screening, K=4 pipeline) ----
    # feasibility_survivors_assam.csv was produced by 07_feasibility_filter.py under
    # the K=4 cluster pipeline. It is used here as the Assam-specific candidate set
    # WITHOUT modification. The 8 unique PCMs that pass any Phase 6 cluster are
    # evaluated at the Phase 3 final K=3 medoids (pipeline-version inconsistency documented).
    feas_df      = pd.read_csv(FEAS_FILE)
    phase6_names = sorted(feas_df[feas_df["passes_all"] == True]["name"].unique().tolist())
    pcm_assam    = pd.read_csv(PCM_ASSAM_FILE)
    # Rename 'name' -> 'product_name' so PCMStateNode and result rows are consistent
    pcm_assam    = pcm_assam.rename(columns={"name": "product_name"})
    candidates   = pcm_assam[pcm_assam["product_name"].isin(phase6_names)].copy().reset_index(drop=True)

    log(f"  Phase 6-screened PCM candidates (K=4 pipeline, N={len(candidates)}):")
    for _, r in candidates[["product_name", "Tm_C"]].iterrows():
        log(f"    {r['product_name']:48s}  Tm = {r['Tm_C']:.1f} C")

    # ---- Phase 3 K=3 medoids ----
    medoid_map = derive_true_medoids(assign_df, sig_raw_df)
    log("\n[2] Programmatically Derived Phase 3 K=3 True Cluster Medoids (5 GMM Features):")
    for c_id, pt_id in medoid_map.items():
        log(f"  Cluster {c_id}: True Medoid Point ID = {pt_id}")

    log("\n[PIPELINE NOTE] Candidate eligibility: Phase 6 K=4 screening.")
    log("                Climate forcing:        Phase 3 K=3 medoids.")
    log("                This is a documented pipeline-version inconsistency.")
    log("                Phase 6/7 files are not modified.")

    results_list = []

    for c_id in sorted(medoid_map.keys()):
        medoid_pt = medoid_map[c_id]
        log("\n" + "-" * 60)
        log(f"  PROCESSING CLUSTER {c_id} (Phase 3 K=3 Medoid: {medoid_pt})")
        log("-" * 60)

        log("  Loading 10-year ERA5 hourly climate forcing (2016-2025)...")
        forcing_df, raw_ssrd_J, ssrd_cons_err, nightclamp_J = load_era5_hourly_forcing(
            medoid_pt, grid_df
        )
        log(f"  Loaded {len(forcing_df)} hourly steps ({len(forcing_df)/8760:.2f} years)")
        log(f"  SSRD raw de-accumulated energy:        {raw_ssrd_J:.6e} J/m2")
        log(f"  SSRD reconstruction conservation error (pre-clamp): {ssrd_cons_err:.6f}%")
        log(f"  SSRD energy removed by night clamping: {nightclamp_J:.6e} J/m2")
        if ssrd_cons_err >= 0.1:
            log(f"  [WARNING] SSRD conservation error {ssrd_cons_err:.4f}% exceeds 0.1% threshold!")

        log(f"\n  Phase 9 Physics Evaluation for {len(candidates)} Phase 6-screened candidates...")

        for _, pcm_row in candidates.iterrows():
            pcm_name = pcm_row["product_name"]
            # Candidate label: Phase 6-screened, independent physics evaluation, non-MCDM
            candidate_label = (
                "Phase 6-screened candidate evaluated independently under the "
                "final Phase 3 K=3 climate forcing; not an MCDM-ranked PCM."
            )

            # Execute 10-Year Sub-Hourly Simulation (dt = 300 s)
            sim_300 = simulate_pcm_swh_10year(forcing_df, pcm_row, dt_sec=300.0)

            # Execute Benchmark Simulation for Timestep Sensitivity Check (dt = 150 s)
            sim_150 = simulate_pcm_swh_10year(forcing_df, pcm_row, dt_sec=150.0)

            # Compute and PERSIST pre-specified dt sensitivity numerical differences
            diff_sf       = (abs(sim_300["solar_fraction"] - sim_150["solar_fraction"])
                             / max(sim_300["solar_fraction"], 1e-4) * 100.0)
            diff_delivery = (abs(sim_300["overall_delivery_success_rate"]
                                 - sim_150["overall_delivery_success_rate"]) * 100.0)
            diff_cycles   = abs(sim_300["complete_pcm_cycles_per_year"]
                                - sim_150["complete_pcm_cycles_per_year"])

            timestep_passed = (
                diff_sf       <  1.0  and
                diff_delivery <= 1.0  and
                diff_cycles   <= 1.0  and
                sim_300["cum_rel_error_pct"] < 0.1 and
                sim_150["cum_rel_error_pct"] < 0.1
            )

            log(f"    {pcm_name:38s} | Overall: {sim_300['overall_delivery_success_rate']*100:5.1f}% "
                f"| SF: {sim_300['solar_fraction']*100:5.1f}% "
                f"| Cyc/yr: {sim_300['complete_pcm_cycles_per_year']:5.1f} "
                f"| CumErr: {sim_300['cum_rel_error_pct']:.4f}% "
                f"| dt: {'[PASS]' if timestep_passed else '[FAIL]'}")

            results_list.append({
                # Identification
                "cluster_id":             c_id,
                "medoid_point_id":        medoid_pt,
                "pcm_name":               pcm_name,
                "candidate_status_label": candidate_label,
                # PCM properties
                "pcm_mass_kg":            M_P_KG,
                "melting_temp_degC":      pcm_row["Tm_C"],
                "latent_heat_kJ_kg":      pcm_row["latent_heat_kJ_kg"],
                "supercooling_degC":      pcm_row.get("supercooling_K",
                                              pcm_row.get("supercooling_degC", 0.0)),
                # Simulation results (dt = 300 s)
                "morning_delivery_success_rate":  round(sim_300["morning_delivery_success_rate"],  4),
                "evening_delivery_success_rate":  round(sim_300["evening_delivery_success_rate"],  4),
                "overall_delivery_success_rate":  round(sim_300["overall_delivery_success_rate"],  4),
                "hours_Tw_ge_50C_per_year":       round(sim_300["hours_Tw_ge_50C_per_year"],        1),
                "solar_fraction":                 round(sim_300["solar_fraction"],                  4),
                "complete_pcm_cycles_per_year":   round(sim_300["complete_pcm_cycles_per_year"],    2),
                "spinup_converged":               sim_300["spinup_converged"],
                "spinup_cycles_run":              sim_300["spinup_cycles"],
                "max_step_residual_J":            round(sim_300["max_step_residual_J"],             6),
                "cum_rel_energy_error_pct":        round(sim_300["cum_rel_error_pct"],               6),
                # Timestep sensitivity (actual numerical differences persisted)
                "dt_sensitivity_passed":           timestep_passed,
                "dt_sens_sf_rel_diff_pct":        round(diff_sf,       4),
                "dt_sens_delivery_abs_diff_pp":   round(diff_delivery, 4),
                "dt_sens_cycles_abs_diff":         round(diff_cycles,   3),
                "dt_sens_cum_err_300s_pct":        round(sim_300["cum_rel_error_pct"], 6),
                "dt_sens_cum_err_150s_pct":        round(sim_150["cum_rel_error_pct"], 6),
                # ERA5 SSRD conservation (per cluster, same for all PCMs in that cluster)
                "ssrd_raw_energy_J_per_m2":        round(raw_ssrd_J,       0),
                "ssrd_recon_conservation_err_pct": round(ssrd_cons_err,    6),
                "ssrd_nightclamp_loss_J_per_m2":   round(nightclamp_J,     0),
                # Overall status
                "validation_status": (
                    "PASSED"
                    if (sim_300["spinup_converged"]
                        and sim_300["cum_rel_error_pct"] < 0.1
                        and timestep_passed
                        and ssrd_cons_err < 0.1)
                    else "FAILED"
                ),
            })

    res_df = pd.DataFrame(results_list)
    res_df.to_csv(OUT_RESULTS_CSV, index=False)
    log(f"\n[3] Saved Physics Validation CSV to: {OUT_RESULTS_CSV}")
    log(f"    Total rows: {len(res_df)} ({res_df['pcm_name'].nunique()} PCMs x "
        f"{res_df['cluster_id'].nunique()} clusters)")

    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"  Saved Physics Validation Report to: {OUT_REPORT_TXT}")

    log("\n" + "=" * 76)
    log("  PHASE 9 PHYSICS VALIDATION COMPLETE")
    log("=" * 76)


if __name__ == "__main__":
    main()
