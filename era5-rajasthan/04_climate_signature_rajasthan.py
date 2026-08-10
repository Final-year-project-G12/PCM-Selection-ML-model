"""
04_climate_signature_rajasthan.py
=============================================================================
PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION  (Objective1_PCM_Climate_Framework
_Plan_v3, §6.2-6.4)

Reduces each point's 10-year record to one climate-signature vector — the
object Phase 4 (climate regime clustering) actually clusters, not the raw
sun-event/daily data. Builds two tiers plus PCM-facing derived quantities,
five interaction terms, and a PCA-compressed correlated block, mirroring
the Tamil Nadu pipeline's 04b_climate_signature.py column conventions so
Phase 4's cross-region 05_cluster_regions.py can concatenate both states'
outputs directly.

INPUTS:
  data/processed/climate_rajasthan_points.csv           (sun-event samples:
      sunrise/noon/sunset, 02_combine_rajasthan.py's output — physical
      units. NOTE: this file's era5_GHI/LW_down/precipitation columns were
      corrupted by a since-fixed deaccumulation bug (see accum_to_flux()
      in 02_combine_rajasthan.py) — if you re-generate this file, re-run
      02_combine_rajasthan.py first; this script does not re-derive GHI.)
  data/processed/daily_aggregates_rajasthan_summary.csv (Tier 2, one row
      per point — GHI_daily_kWh, SAI, kt_daily_mean/std, cloudy_frac, CCI,
      HDD18, CDD24, DTR_true, seasonality, monsoon_index)
  data/processed/daily_aggregates_rajasthan.csv          (per-point/day
      table — read for its date + kt_daily columns, to get both kt_p05
      (kept for reference) and kt_worst_month (the value actually used —
      see correction #5) per point; the summary file only carries
      kt_daily_mean/std, not the per-day detail either of these need)
  data/processed/suntimes.csv                            (sunrise/noon/
      sunset UTC timestamps, for daylength)
  data/processed/population_grid_points.csv              (elevation_m,
      population, weight, lat, lon per point)

OUTPUT:
  data/processed/climate_signature_rajasthan.csv — one row per point_id.

THREE CORRECTIONS TO THE ORIGINAL SPEC, FLAGGED HERE RATHER THAN SILENTLY
APPLIED — see the printed report and the docstrings below for detail:

  1. "7-day autonomy" -> "7-hour night discharge". The brief asked for a
     7-day-autonomy draw basis for L_required "matching whatever basis the
     Tamil Nadu version used." Tamil Nadu's 04b_climate_signature.py (and
     Objective1_PCM_Climate_Framework_Plan_v3.docx §6.3, citing Avargani
     et al. 2021's "300 L at 60+-2 C for 7 h of operation" benchmark) both
     use a 7-HOUR night discharge window, not 7 days — "autonomy" in the
     plan doc is a separate concept (CCI: longest run of low-clearness
     days). This script uses 7 hours, matching Tamil Nadu and the doc's
     actual Avargani citation, for genuine cross-state comparability.
  2. elevation_m IS a clustering feature, via PCA, not excluded entirely.
     The brief's "STATIC ATTRIBUTES ... NOT included in the clustering
     feature matrix" bullet and its own "DIMENSIONALITY REDUCTION" bullet
     (which explicitly lists elevation_m in the PCA block) conflict at
     face value. The plan doc resolves this: population/weight are
     reporting-only, but elevation's own line reads "expected to be the
     dominant splitting variable within Uttarakhand" — i.e. elevation IS
     meant to inform clustering. Resolution used here: elevation_m's raw
     value is carried through for reporting/mapping and is NOT itself
     z-scored as a standalone clustering column, but it DOES feed the PCA
     block, and the resulting PC*_z scores (which subsume it) are what
     enters the clustering matrix. This satisfies both bullets literally.
  3. DRAW_RATE_KG_PER_S unit-conversion bug — SUPERSEDED, see #4. A first
     pass (2026-08-10) fixed `60.0/1000/60` (=0.001 kg/s, an erroneous
     extra L->m^3 division with no matching *1000 water-density step back
     to kg/s, off by 1000x) to `60.0/60` (=1.0 kg/s, i.e. a literal
     60 L/min sustained draw). That was arithmetically correct but still
     rested on an unsourced design basis — "60 L/min" appears nowhere in
     Objective1_PCM_Climate_Framework_Plan_v3.docx. Superseded same-day by
     correction #4 below, which replaces the whole rate-times-duration
     construction with the plan doc's actual cited benchmark.
  4. Night-discharge design basis replaced with Avargani et al. 2021's
     literature figure directly, fixed 2026-08-10 (second pass). §6.3's
     Q_night formula was being fed a 60 L/min rate sustained for 7 hours
     (25,200 L total) — no source for that figure anywhere in the plan
     doc. The plan doc's own §6.3 citation is Avargani et al. 2021,
     J. Energy Storage: "300 L of hot water at 60+-2 C for 7 h of
     operation" — a TOTAL volume over the window, not a per-minute rate.
     Replaced DRAW_RATE_KG_PER_S (a rate) with NIGHT_DRAW_TOTAL_L = 300.0
     (a total), removing the rate*duration step entirely so there is no
     multiplication left to reintroduce a units slip. NIGHT_DISCHARGE_
     HOURS is kept only as a documented constant for provenance/reporting
     (matches the Avargani window), not as a multiplier in the energy
     calc anymore.
     L_required IS A CEILING, NOT AN ACHIEVABILITY BAR. L_required =
     Q_night / m_PCM assumes the PCM bed alone supplies the ENTIRE 300 L/
     7 h Avargani load, with zero contribution from the tank's own
     sensible heat or from collector charging overlapping the discharge
     window. Neither assumption holds in a real system — not even in
     Avargani's own rig, whose ~36-38 kg paraffin bed could not plausibly
     have supplied that whole load from latent heat alone. Treat
     L_required_kJ_per_kg the same way Tm_target_capped_C is already
     treated in this script: an idealized upper-bound design heuristic,
     not a value any single-material PCM is expected to fully reach.
     CONSEQUENCE FOR PHASE 5 (not yet built): Table 12's feasibility
     filter, "L >= kappa * L_required" with kappa=0.7 fixed, will zero out
     every PCM candidate in every cluster if kappa is left fixed at 0.7 —
     no organic paraffin (~150-250 kJ/kg) or salt hydrate (~250-280 kJ/kg)
     in any realistic database gets close to these totals. When Phase 5
     is built, calibrate kappa per cluster instead of fixing it a priori:
     start at kappa=0.7 and step down in increments of 0.1 until that
     cluster retains 8-20 candidates (the same healthy-survivor-count
     band already used elsewhere in Table 12), recording the kappa used
     per cluster alongside the melting-window relaxation flag in the same
     results table — the same escalation pattern Table 12 already uses
     for the melting-window constraint ("if fewer than five candidates,
     relax by 2K and record"), applied to the constraint that's actually
     broken, not a new, differently-justified escape hatch. If even
     kappa->0 cannot retain 8 candidates in some cluster, that is a
     finding about that cluster's demand profile (report it), not a
     latent-heat failure to paper over. Alternative if a hard gate proves
     unworkable even with calibrated kappa: rank candidates by proximity
     to L_required instead of gating on a fraction of it.
  5. Tm_target_capped_C re-derived on a worst-MONTH basis, not worst-DAY,
     fixed 2026-08-11. The original cap used kt_p05 (the 5th-percentile
     SINGLE DAY's clearness) linearly collapsed toward Ta_mean — flagged
     at the time as an unvalidated heuristic pending independent sanity-
     check. That check came back negative for the single-day basis
     specifically: Hottel-Whillier-Bliss itself is well-validated (recent
     evacuated-tube work reports ~0.8C/1.1% error against measured outlet
     temperature) but ONLY at normal-to-good operating conditions — no
     validation study extends it down to near-zero irradiance, which is
     exactly where a linear model is least trustworthy (heat-loss
     coefficients aren't necessarily constant as delta-T shrinks, and real
     systems often have a minimum-irradiance circulation cutoff a linear
     model can't represent). Direct field evidence at the actual location:
     Nahar (2003), "Year round performance and potential of a natural
     circulation type of solar water heater in India," tested at Jodhpur —
     squarely in this state's arid-west cluster territory — and reported
     100 L delivered at an AVERAGE 50-70C across the year, with hot water
     demand and testing concentrated in WINTER, the weaker-solar season.
     The original kt_p05-based cap came out to 40.8-49.5C across all 320
     points — at or below the low end of what a real system in that exact
     city delivered on average, not even worst-case. That gap is evidence
     the single-worst-day linear collapse was too aggressive, not that
     Rajasthan's poor-day performance is genuinely that constrained.
     FIX: kt_p05 (worst SINGLE DAY) replaced with kt_worst_month (the
     lowest of each point's 12 calendar-month MEAN clearness values) as
     the linear formula's input — same interpolation mechanism, still an
     explicit heuristic pending full validation, but anchored to a
     literature-standard "worst month" sizing basis (Durin et al. 2018,
     "'Worst Month' and 'Critical Period' Methods for the Sizing of Solar
     Irrigation Systems"; the same family of method the f-chart approach
     institutionalizes) instead of an unvalidated single-day extrapolation.
     A monthly mean is inherently far less extreme than a single-day p05
     figure, so this raises the cap substantially without asserting a
     specific nonlinear collector model this stage isn't scoped to build
     (that remains explicit Phase 7 territory, same as 07b_charging_
     feasibility.py's own docstring already says). OLD kt_p05-based value
     kept as Tm_target_capped_C_p05day for audit/comparison, NOT read by
     any downstream script; Tm_target_capped_C itself (the name 05/07
     already reference) now means the worst-month version. kt_p05 itself
     is also kept unchanged (still used nowhere else, retained for record).

HOW TO RUN:
  python 04_climate_signature_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import glob
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from config import (
    COMBINED_POINTS_FILE,
    DAILY_AGGREGATES_FILE,
    DAILY_AGGREGATES_SUMMARY_FILE,
    SUNTIMES_FILE,
    POPULATION_GRID_FILE,
    RAW_POWER_DIR,
    CLIMATE_SIGNATURE_FILE,
    OUTPUTS_DIR,
    ensure_data_dirs,
)
from signature_lib import EVENT_ORDER, build_tier1_signature

ensure_data_dirs()

# --- PCM-facing design basis (§6.3, matches Tamil Nadu's 04b for
# cross-state comparability) ------------------------------------------
T_DELIVERY_C = 50.0            # Indian domestic SWH delivery target, §6.3
DT_APPROACH_C = 7.0            # heat-exchanger approach temp, midpoint of the doc's 5-8K range
TM_TARGET_C = T_DELIVERY_C + DT_APPROACH_C   # -> 57 C, indirect-system assumption

"""
Night-discharge design basis and latent-heat floor [Avargani et al. 2021,
J. Energy Storage] -- 300 L sustained at 60 +/- 2 degC over a 7 h window.
NIGHT_DRAW_TOTAL_L is a TOTAL volume delivered over the window, not a
per-minute flow rate. Prior DRAW_RATE_KG_PER_S = 60.0/1000/60 implied
25,200 L delivered over 7h -- ~84x the cited benchmark -- and was corrected
here; see Objective1_Section5_Methodology_Update.docx / plan v3 addendum.
[NOTE: that addendum file was not found in this project tree as of this
edit -- confirm it exists, or update this pointer, before citing it further.]

L_required = Q_night / m_PCM is an IDEALIZED CEILING, not an achievable
target: it assumes the PCM bed alone supplies the entire night draw with
zero contribution from tank sensible heat or overlapping daytime charging.
No single-material PCM in the database approaches it directly -- not even
Avargani's own ~36-38 kg paraffin bed could have met it on this basis.

Table 12's latent-heat floor is therefore L >= kappa * L_required with
kappa CALIBRATED per cluster (start 0.7, step down by 0.1 until 8-20
candidates survive), not a fixed 0.7 hard cutoff -- see plan doc §6.3.1
for the full calibration procedure and rationale. Record the kappa used
per cluster in the Phase 5 results table alongside the melting-window
relaxation flag.
"""
NIGHT_DRAW_TOTAL_L = 300.0        # total litres delivered during the discharge window [Avargani 2021]
NIGHT_DISCHARGE_HOURS = 7.0       # duration of the discharge window [Avargani 2021] — provenance/
                                   # reporting only below, NOT used as a rate multiplier
WATER_DENSITY_KG_PER_L = 1.0      # kg/L

NIGHT_DRAW_TOTAL_KG = NIGHT_DRAW_TOTAL_L * WATER_DENSITY_KG_PER_L   # = 300.0 kg
CP_WATER = 4.186                         # kJ/kg.K
ASSUMED_PCM_MASS_KG = 50.0                # placeholder mass, matches Tamil Nadu

# PCA block — the correlated temperature/pressure columns (§6.4). Kept
# OUT of the final standalone clustering matrix; replaced by their PCA
# component scores instead.
PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "T_sunrise_mean", "T_noon_mean",
             "HDD18", "CDD24", "elevation_m"]

print("=" * 68)
print("  PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION — Rajasthan")
print(f"  Tm_target (base)   = {T_DELIVERY_C:.0f} + {DT_APPROACH_C:.0f} = {TM_TARGET_C:.0f} C")
print(f"  L_required basis   = {NIGHT_DRAW_TOTAL_L:.0f} L over {NIGHT_DISCHARGE_HOURS:.0f}h "
      f"night discharge (Avargani 2021), {ASSUMED_PCM_MASS_KG:.0f} kg PCM — "
      f"a CEILING, not an achievability bar [see module docstring, correction #4]")
print("=" * 68)


# ═══════════════════════════════════════════════════════════
# 1. LOAD  (usecols only — climate_rajasthan_points.csv is ~1.5GB)
# ═══════════════════════════════════════════════════════════

print("\n[1/9] Loading inputs ...")

pts_cols = ["point_id", "date", "event", "era5_T_amb", "era5_RHum",
            "era5_GHI", "era5_CSI", "era5_W_spd"]
events_df = pd.read_csv(COMBINED_POINTS_FILE, usecols=pts_cols, parse_dates=["date"])
events_df["event"] = pd.Categorical(events_df["event"], categories=EVENT_ORDER, ordered=True)
print(f"  climate_rajasthan_points.csv : {len(events_df):,} rows, "
      f"{events_df['point_id'].nunique()} points")

sun_df = pd.read_csv(SUNTIMES_FILE, parse_dates=["date"])
sun_df["time_utc"] = pd.to_datetime(sun_df["time_utc"], utc=True)
print(f"  suntimes.csv                 : {len(sun_df):,} rows")

daily_kt = pd.read_csv(DAILY_AGGREGATES_FILE, usecols=["point_id", "date", "kt_daily"],
                        parse_dates=["date"])
print(f"  daily_aggregates_rajasthan.csv (date + kt_daily) : {len(daily_kt):,} rows")

tier2 = pd.read_csv(DAILY_AGGREGATES_SUMMARY_FILE)
print(f"  daily_aggregates_rajasthan_summary.csv : {len(tier2)} points")

static_df = pd.read_csv(POPULATION_GRID_FILE)
print(f"  population_grid_points.csv   : {len(static_df)} points")

# monsoon_index proxy check — PRECTOTCORR was never requested by
# 01b_download_nasapower.py (POWER_PARAMETERS has no precipitation
# variable), so Tier 2's monsoon_index (built by 02b_build_daily_
# aggregates.py) is always a GHI-fraction proxy here, never a true
# precipitation-based index. Verified dynamically against the cache
# rather than hardcoded, in case the download pipeline is extended later.
sample_power_files = glob.glob(str(RAW_POWER_DIR / "power_*.json"))[:5]
prectot_found = False
for fp in sample_power_files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            params = json.load(f).get("properties", {}).get("parameter", {})
        if "PRECTOTCORR" in params:
            prectot_found = True
            break
    except Exception:
        continue
if not prectot_found:
    print("  [WARN] PRECTOTCORR not found in sampled NASA POWER cache files — "
        "Tier 2's monsoon_index is a GHI-fraction PROXY (Jun-Sep mean GHI / "
        "annual mean GHI, per 02b_build_daily_aggregates.py), not a true "
        "precipitation-based index. State this in the methodology write-up.")


# ═══════════════════════════════════════════════════════════
# 2/3. TIER 1 + DAYLENGTH — via signature_lib.build_tier1_signature()
# ═══════════════════════════════════════════════════════════
# Shared with 05_cluster_rajasthan.py's Level B (one row per point PER
# SEASON) — see signature_lib.py's docstring. Here group_keys=["point_id"]
# reproduces the original whole-year, one-row-per-point behavior exactly.

print("\n[2-3/9] Tier 1 (per-event aggregates) + daylength, via signature_lib ...")

tier1 = build_tier1_signature(events_df, sun_df, group_keys=["point_id"])
tier1.index.name = "point_id"

print(f"  Tier 1: {tier1.shape[1]} columns x {tier1.shape[0]} points")
print(f"  daylength_mean range: {tier1['daylength_mean'].min():.2f} - "
      f"{tier1['daylength_mean'].max():.2f} h")


# ═══════════════════════════════════════════════════════════
# 4. TIER 2 — join daily-integral indices
# ═══════════════════════════════════════════════════════════

print("\n[4/9] Tier 2 — joining daily_aggregates_rajasthan_summary.csv ...")

tier2_cols = ["point_id", "GHI_daily_kWh", "SAI", "kt_daily_mean", "kt_daily_std",
              "cloudy_frac", "CCI", "HDD18", "CDD24", "DTR_true", "seasonality",
              "monsoon_index"]
sig = tier1.join(tier2.set_index("point_id")[tier2_cols[1:]], how="left")
n_missing_tier2 = sig["GHI_daily_kWh"].isna().sum()
if n_missing_tier2:
    print(f"  [WARN] {n_missing_tier2} point(s) have no Tier 2 row "
          f"(daily_aggregates_rajasthan_summary.csv incomplete for those points)")


# ═══════════════════════════════════════════════════════════
# 5. STATIC ATTRIBUTES  (reporting/mapping only — see correction #2)
# ═══════════════════════════════════════════════════════════

print("\n[5/9] Joining static attributes (elevation_m, population, weight, lat, lon) ...")

static_cols = static_df.set_index("point_id")[["lat", "lon", "population", "weight", "elevation_m"]]
sig = sig.join(static_cols, how="left")


# ═══════════════════════════════════════════════════════════
# 6. DERIVED PCM-FACING QUANTITIES  (§6.3)
# ═══════════════════════════════════════════════════════════

print("\n[6/9] Derived PCM-facing quantities (Tm_target, Tm_target_capped, L_required) ...")

sig["Tm_target_C"] = TM_TARGET_C   # constant by design rule — same across all points

# kt_p05 per point — 5th percentile of the DAILY (not per-event) kt series.
# Kept for reference/audit only — NOT used for Tm_target_capped_C anymore,
# see correction #5. Nothing downstream reads this column.
kt_p05 = daily_kt.groupby("point_id")["kt_daily"].quantile(0.05)
sig["kt_p05"] = kt_p05

# kt_worst_month per point — the lowest of that point's 12 calendar-month
# MEAN kt_daily values (2016-2025 pooled per month before taking the min).
# This is the value Tm_target_capped_C actually uses (correction #5): a
# monthly mean is inherently far less extreme than a single-day p05
# figure, matching literature-standard "worst month" solar sizing practice
# (Durin et al. 2018) rather than an unvalidated single-day extrapolation.
daily_kt["month"] = daily_kt["date"].dt.month
monthly_mean_kt = daily_kt.groupby(["point_id", "month"])["kt_daily"].mean()
kt_worst_month = monthly_mean_kt.groupby("point_id").min()
sig["kt_worst_month"] = kt_worst_month

# Upper-bound cap: Tm must lie below the collector delivery temperature
# achievable on a poor-insolation period in that regime (plan doc §6.3:
# "Tm must lie below the collector delivery temperature achievable on a
# poor day ... which is what makes the target regime-dependent").
# Approximated via a Hottel-Whillier-style linear scaling: for a fixed
# heat-loss coefficient, a flat-plate collector's achievable temperature
# rise above ambient scales roughly linearly with incident irradiance/
# clearness. The base Tm_target_C (57C) is treated as achievable above
# Ta_mean on a "typical" day (that point's own kt_daily_mean); on a poor
# period the achievable rise is scaled down by the clearness ratio:
#
#   Tm_cap_C = Ta_mean + (kt_poor / kt_daily_mean) * (Tm_target_C - Ta_mean)
#
# where kt_poor is now kt_worst_month (was kt_p05 — see correction #5).
# Still an explicit approximation/heuristic, not a calibrated collector
# model — state it as such if cited. Flat-plate stagnation temperature is
# a second upper bound the plan doc also names (§6.3) but is out of scope
# for this script (not requested); note it if you extend this later.
"""
Tm_target_capped_C applies a poor-period clearness cap to Tm_target_C
using a linear Hottel-Whillier-style collapse-to-ambient heuristic.
CORRECTED 2026-08-11 (correction #5): the poor-period input is now
kt_worst_month (lowest of the 12 calendar-month mean clearness values),
not kt_p05 (a single day's 5th-percentile clearness). Verified against
independent field data before making this change, not just theorized:
Nahar (2003) field-tested a natural-circulation SWH at Jodhpur -- squarely
in this state's arid-west cluster territory, the SAME city, not a generic
climate analog -- and reported 100 L delivered at an AVERAGE 50-70C across
the year, concentrated in WINTER (the weaker-solar season). The prior
kt_p05-based cap came out to 40.8-49.5C across all 320 points -- at or
below the low end of what a real system in that exact city achieved on
AVERAGE, not even worst-case. That gap was evidence the single-day linear
collapse was too aggressive, not that Rajasthan's poor-period performance
is genuinely that constrained. This remains an explicit heuristic, not a
calibrated collector model -- it is not validated against an independent
estimate for the exact shape of the linear scaling, only re-anchored to a
less extreme, literature-standard input -- and should not feed a hard
per-point filter without further validation. The prior kt_p05-based value
is kept as Tm_target_capped_C_p05day for audit/comparison; nothing
downstream reads it.
"""
kt_ratio_p05day = (sig["kt_p05"] / sig["kt_daily_mean"]).clip(upper=1.0)
tm_cap_p05day = sig["Ta_mean"] + kt_ratio_p05day * (sig["Tm_target_C"] - sig["Ta_mean"])
sig["Tm_target_capped_C_p05day"] = np.minimum(sig["Tm_target_C"], tm_cap_p05day)

kt_ratio = (sig["kt_worst_month"] / sig["kt_daily_mean"]).clip(upper=1.0)
tm_cap = sig["Ta_mean"] + kt_ratio * (sig["Tm_target_C"] - sig["Ta_mean"])

sig["tm_target_capped_flag"] = sig["Tm_target_C"] > tm_cap
sig["Tm_target_capped_C"] = np.minimum(sig["Tm_target_C"], tm_cap)
n_capped = int(sig["tm_target_capped_flag"].sum())
print(f"  Tm_target_C (base)              : constant {TM_TARGET_C:.0f} C across all points")
print(f"  Tm_target_capped_C (worst-month): {sig['Tm_target_capped_C'].min():.1f} - "
      f"{sig['Tm_target_capped_C'].max():.1f} C  "
      f"({n_capped}/{len(sig)} points where the base target exceeds the poor-period cap)")
print(f"  Tm_target_capped_C_p05day (OLD, reference only): "
      f"{sig['Tm_target_capped_C_p05day'].min():.1f} - {sig['Tm_target_capped_C_p05day'].max():.1f} C")

# T_mains estimate — same simple offset form as Tamil Nadu's
# 04b_climate_signature.py (Ta_mean - 2 C). Not derived from a specific
# published correlation; kept identical to the Tamil Nadu precedent for
# cross-state comparability, as the plan doc's §6.3 "standard lag
# correlation" wording is not itself pinned to a specific formula.
sig["T_mains_est_C"] = sig["Ta_mean"] - 2.0

# Q_night = total night-draw mass x cp_water x (T_delivery - T_mains) — a
# TOTAL energy over the whole discharge window, not a rate. See module
# docstring correction #4: L_required is an idealized CEILING (PCM alone
# supplying the entire Avargani load with no sensible-heat or concurrent-
# charging contribution), not a value any single-material PCM should be
# expected to fully reach — do not gate Phase 5 on a fixed fraction of it.
Q_night_kJ = NIGHT_DRAW_TOTAL_KG * CP_WATER * (T_DELIVERY_C - sig["T_mains_est_C"])
sig["L_required_kJ_per_kg"] = Q_night_kJ / ASSUMED_PCM_MASS_KG
print(f"  L_required_kJ_per_kg  : {sig['L_required_kJ_per_kg'].min():.0f} - "
      f"{sig['L_required_kJ_per_kg'].max():.0f} kJ/kg  (CEILING, not an achievability bar — "
      f"assumes {ASSUMED_PCM_MASS_KG:.0f} kg PCM supplies the entire "
      f"{NIGHT_DRAW_TOTAL_L:.0f} L/{NIGHT_DISCHARGE_HOURS:.0f}h Avargani load alone)")


# ═══════════════════════════════════════════════════════════
# 7. INTERACTION TERMS  (§6.4 — justifications quoted from the plan doc)
# ═══════════════════════════════════════════════════════════

print("\n[7/9] Interaction terms ...")

# charging energy weighted by its unreliability; high values mean a large
# but erratic resource
sig["int_GHI_x_ktstd"] = sig["GHI_daily_kWh"] * sig["kt_daily_std"]
# cycling stress under intermittent charging, the worst case for phase
# stability
sig["int_DTR_x_cloudyfrac"] = sig["DTR_true"] * sig["cloudy_frac"]
# condensation risk at the store surface at the condensation-critical
# instant
sig["int_RH_x_TsunriseMinusTm"] = sig["RH_sunrise_mean"] * (sig["T_sunrise_mean"] - sig["Tm_target_C"])
# convective loss driving potential during the evening draw
sig["int_wind_x_TsunsetMinusTdelivery"] = sig["wind_sunset_mean"] * (sig["T_sunset_mean"] - T_DELIVERY_C)
# combined autonomy requirement
sig["int_CCI_x_1minusSAI"] = sig["CCI"] * (1 - sig["SAI"])

print("  Added 5 interaction terms (GHIxkt_std, DTRxcloudy_frac, "
      "RHx(Tsunrise-Tm), windx(Tsunset-Tdelivery), CCIx(1-SAI))")


# ═══════════════════════════════════════════════════════════
# 8. PCA ON THE CORRELATED TEMPERATURE/PRESSURE BLOCK ONLY
# ═══════════════════════════════════════════════════════════

print("\n[8/9] PCA on the correlated block "
      f"({', '.join(PCA_BLOCK)}) ...")

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
print(f"  {n_comp} components retained (95% variance). Loadings:")
print(loadings.round(3).to_string())
print(f"  Explained variance ratio: {np.round(pca.explained_variance_ratio_, 3)}")
print("  -> Read the sign/magnitude pattern per component (plan doc expects roughly "
      "'heat', 'altitude', 'seasonal amplitude' across all four states) — name them "
      "in the write-up.")


# ═══════════════════════════════════════════════════════════
# 9. STANDARDIZE THE CLUSTERING-READY MATRIX  (*_z columns)
# ═══════════════════════════════════════════════════════════

print("\n[9/9] Standardizing the clustering-ready feature matrix ...")

# Explicitly excluded from every standardized/clustering-ready column:
#   - lat, lon                 : plotting only, never fitting (plan doc §6.2:
#                                 "Do not put latitude and longitude in the
#                                 clustering matrix ... clusters geography,
#                                 not climate")
#   - population, weight        : reporting only (plan doc §6.2)
#   - elevation_m                : feeds PCA (see correction #2 above), but
#                                 its raw value is not itself a standalone
#                                 clustering column — the PC*_z scores
#                                 subsume it
#   - PCA_BLOCK raw columns      : replaced by PC1..PCn
#   - point_id, T_mains_est_C, kt_p05, tm_target_capped_flag, n_days-like
#     helper/proxy/diagnostic columns : not independent climate dimensions
NON_CLUSTERING_COLS = set(PCA_BLOCK) | {
    "lat", "lon", "population", "weight", "T_mains_est_C", "kt_p05",
    "kt_worst_month", "Tm_target_capped_C_p05day", "tm_target_capped_flag",
}

clustering_cols = [c for c in sig.columns if c not in NON_CLUSTERING_COLS]

std_scaler = StandardScaler()
clustering_input = sig[clustering_cols].fillna(sig[clustering_cols].median())
clustering_scaled = std_scaler.fit_transform(clustering_input)
clustering_z = pd.DataFrame(clustering_scaled, index=sig.index,
                             columns=[f"{c}_z" for c in clustering_cols])

full_out = sig.join(clustering_z)
full_out.index.name = "point_id"
full_out.to_csv(CLIMATE_SIGNATURE_FILE)
print(f"  Clustering matrix: {len(clustering_cols)} columns "
      f"(includes {n_comp} PCA components)")
print(f"  Final signature matrix: {full_out.shape[0]} points x {full_out.shape[1]} columns")
print(f"  Saved: {CLIMATE_SIGNATURE_FILE}")


# ═══════════════════════════════════════════════════════════
# CORRELATION HEATMAP + |r| > 0.9 FLAG  (final feature set, pre-PCA
# block already removed — any flagged pair here is a NEW collinearity,
# not one already absorbed by the PCA step)
# ═══════════════════════════════════════════════════════════

print("\nCorrelation heatmap of the final feature set ...")

corr_input = sig[clustering_cols]
const_cols = [c for c in clustering_cols if corr_input[c].std(skipna=True) in (0, np.nan) or corr_input[c].nunique(dropna=True) <= 1]
if const_cols:
    print(f"  [NOTE] excluding constant column(s) from the correlation check "
          f"(correlation undefined): {const_cols}")
corr_cols_nonconst = [c for c in clustering_cols if c not in const_cols]
corr_matrix = corr_input[corr_cols_nonconst].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.columns.tolist(),
    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="Pearson r"),
))
fig.update_layout(
    title="Climate Signature — Final Feature Set Correlation (Rajasthan)",
    height=max(500, 28 * len(corr_cols_nonconst)), width=max(600, 28 * len(corr_cols_nonconst)),
    xaxis=dict(tickangle=45),
)
heatmap_path = OUTPUTS_DIR / "signature_correlation_heatmap_rajasthan.html"
fig.write_html(str(heatmap_path))
print(f"  Saved: {heatmap_path}")

flagged_pairs = []
for i, c1 in enumerate(corr_cols_nonconst):
    for c2 in corr_cols_nonconst[i + 1:]:
        r = corr_matrix.loc[c1, c2]
        if pd.notna(r) and abs(r) > 0.9:
            flagged_pairs.append((c1, c2, round(float(r), 3)))

if flagged_pairs:
    print(f"\n  [FLAG] {len(flagged_pairs)} pair(s) with |r| > 0.9 in the FINAL feature "
          f"set (the PCA_BLOCK columns are already removed from this set, so none of "
          f"these are already handled by the PCA step — genuinely new collinearity):")
    for c1, c2, r in sorted(flagged_pairs, key=lambda x: -abs(x[2])):
        print(f"    {c1}  <->  {c2}   r={r}")
else:
    print("\n  No pair with |r| > 0.9 remains in the final feature set "
          "(beyond what the PCA step already absorbed).")

print("\n" + "=" * 68)
print("  PHASE 3 COMPLETE — Rajasthan")
print(f"  Output: {CLIMATE_SIGNATURE_FILE}")
print("=" * 68)
print("\nOnce Tamil Nadu's (or another region's) climate_signature_*.csv also "
      "exists with this same column convention, Phase 4 (05_cluster_regions.py) "
      "can combine them for Gaussian Mixture clustering.")
