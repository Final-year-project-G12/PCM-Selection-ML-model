"""
04b_climate_signature.py
=============================================================================
PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION — TAMIL NADU
(Objective1_PCM_Climate_Framework_Plan_v3, §6.2-6.4)

UNIFIED WITH RAJASTHAN (2026-09-08)
----------------------------------
This script is now a state-parameterised mirror of
era5-rajasthan/04b_climate_signature.py. Both states:
  * build the Tier-1 sun-event signature through the SAME shared code
    (signature_lib.build_tier1_signature, group_keys=["point_id"]) — one
    implementation, no per-state copy of the index formulas;
  * use the SAME curated Tier-1 column list (T_sunrise_mean/p05,
    T_noon_mean, T_sunset_mean/p95, diurnal_gradient, kt_noon_mean/std,
    GHI_noon_mean, GHI_sunset_mean, RH_sunrise_mean, wind_noon_mean/
    sunset_mean, HSI_sunrise, Ta_mean/p95/p05, daylength_mean/amplitude);
  * use the SAME 5 sun-event interaction terms (int_GHI_x_ktstd,
    int_DTR_x_cloudyfrac, int_RH_x_TsunriseMinusTm,
    int_wind_x_TsunsetMinusTdelivery, int_CCI_x_1minusSAI) — Tamil Nadu's
    former int_RH_x_TaMinusTm / int_wind_x_TaMinusTsoil (the latter using
    an undefined T_soil proxy) are REMOVED;
  * use the SAME 8-column PCA block [Ta_mean, Ta_p95, Ta_p05,
    T_sunrise_mean, T_noon_mean, HDD18, CDD24, elevation_m] — Tamil Nadu's
    elevation_m is real per-point ERA5 orography (00c_attach_elevation.py),
    not the old flat 150 m proxy;
  * derive Tm_target_C / Tm_target_capped_C / L_required_kJ_per_kg by the
    SAME formulas, reading SHARE_PCM / ASSUMED_PCM_MASS_KG / T_DELIVERY_C /
    DT_APPROACH_C from pcm_shared_config.py (via config.py);
  * emit the SAME output column schema (names + order + units), so Phase 4's
    cross-region 05_cluster_regions.py can concatenate both directly.

The one genuinely state-specific piece is the INPUT plumbing: Tamil Nadu's
Tier-2 daily-integral table (tier2_signature_tamilnadu.csv, from
02b_build_daily_aggregates.py) carries the same indices under `_true`/
`_mean`-suffixed names, which are renamed to the canonical Rajasthan names
on load; and monsoon_index — absent from Tamil Nadu's Tier-2 table — is
computed here as the same Jun-Sep GHI-fraction proxy Rajasthan's 02b uses,
from the 3x/day ERA5 noon GHI (a coarser sample than Rajasthan's full-
hourly NASA POWER integral — state this as a limitation).

INPUTS:
  data/preprocessed/tamilnadu_cleaned_physical.csv    (04_preprocess_
      tamilnadu.py's sun-event physical-units output — sunrise/noon/sunset)
  data/processed/tier2_signature_tamilnadu.csv         (02b's Tier-2 point
      table — renamed to canonical names on load)
  data/processed/daily_aggregates_tamilnadu.csv        (02b's per-point/day
      table — read for date + kt_daily, for kt_p05 and kt_worst_month)
  data/processed/suntimes.csv                          (sunrise/noon/sunset
      UTC timestamps, for daylength)
  data/processed/population_grid_points.csv            (elevation_m,
      population, weight, lat, lon per point)

OUTPUT:
  data/processed/signatures/climate_signature_tamilnadu.csv  (one row per
      point_id — single "processed/", not the former "processed/processed/")

DERIVED-QUANTITY CORRECTIONS (identical to Rajasthan's 04b — see that file's
docstring for the full provenance of each):
  * L_required_kJ_per_kg = (SHARE_PCM * Q_night) / ASSUMED_PCM_MASS_KG, with
    SHARE_PCM = 0.5 (OPTION A, 2026-08-31, literature-anchored combined
    sensible+latent share — Zhao 2022, Huang 2020, Abdelsalam 2020,
    Koželj 2021). A CEILING, not an achievability bar.
  * Q_night from Avargani et al. (2021)'s 300 L @ 60±2°C over 7 h night-
    discharge benchmark (a TOTAL volume over the window, not a rate).
  * Tm_target_capped_C from a worst-MONTH clearness cap (lowest of the 12
    calendar-month mean kt values), Hottel-Whillier-style linear collapse
    toward Ta_mean (Durin et al. 2018 worst-month sizing basis). The old
    kt_p05 single-day value is retained as Tm_target_capped_C_p05day for
    audit only; nothing downstream reads it.

REQUIRED LIBRARIES:
  pip install pandas numpy scikit-learn plotly

HOW TO RUN:
  python 04b_climate_signature.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    PREPROCESSED_DIR,
    PROCESSED_DIR,
    SUNTIMES_FILE,
    POPULATION_GRID_FILE,
    OUTPUTS_DIR,
    ensure_data_dirs,
    # Cross-state design-basis constants — defined once in
    # PCM-Selection-ML-model/pcm_shared_config.py, re-exported by config.py.
    T_DELIVERY_C,
    DT_APPROACH_C,
    TM_TARGET_C,
    ASSUMED_PCM_MASS_KG,
    SHARE_PCM,
    PCA_N_COMPONENTS,
    T_MAINS_EST_C_TODO,
)
from signature_lib import EVENT_ORDER, build_tier1_signature

ensure_data_dirs()

STATE_NAME = "tamilnadu"

PHYSICAL_FILE = PREPROCESSED_DIR / f"{STATE_NAME}_cleaned_physical.csv"
TIER2_FILE = PROCESSED_DIR / f"tier2_signature_{STATE_NAME}.csv"
DAILY_AGGREGATES_FILE = PROCESSED_DIR / f"daily_aggregates_{STATE_NAME}.csv"

# OUTPUT PATH — single "processed/" (the former "processed/processed/"
# doubling is fixed). Mirrors Rajasthan's data/processed/ ... convention;
# 05_cluster_tamilnadu.py already reads from data/processed/signatures/.
SIGNATURE_DIR = PROCESSED_DIR / "signatures"
SIGNATURE_DIR.mkdir(parents=True, exist_ok=True)
CLIMATE_SIGNATURE_FILE = SIGNATURE_DIR / f"climate_signature_{STATE_NAME}.csv"

# --- PCM-facing design basis (§6.3) -------------------------------------
# T_DELIVERY_C / DT_APPROACH_C / TM_TARGET_C / ASSUMED_PCM_MASS_KG /
# SHARE_PCM come from pcm_shared_config.py (via config.py) so Rajasthan and
# Tamil Nadu's 04b cannot drift apart. Values: 50.0 + 7.0 -> 57.0 C;
# 50.0 kg PCM placeholder; SHARE_PCM = 0.5 (literature-anchored).

# Night-discharge design basis [Avargani et al. 2021, J. Energy Storage]:
# 300 L at 60±2°C over 7 h. A TOTAL volume delivered over the window, not a
# per-minute flow rate. Avargani's own system achieves it via an integrated
# collector + PCM tank + sensible tank, not PCM latent heat alone — hence
# the SHARE_PCM fractional-share model for L_required below.
NIGHT_DRAW_TOTAL_L = 300.0
NIGHT_DISCHARGE_HOURS = 7.0       # provenance/reporting only, NOT a rate multiplier
WATER_DENSITY_KG_PER_L = 1.0
NIGHT_DRAW_TOTAL_KG = NIGHT_DRAW_TOTAL_L * WATER_DENSITY_KG_PER_L   # = 300.0 kg
CP_WATER = 4.186                  # kJ/kg.K

# PCA block — the correlated temperature/elevation columns (§6.4). Kept OUT
# of the final standalone clustering matrix; replaced by their PCA component
# scores. IDENTICAL to Rajasthan's PCA_BLOCK. (Earlier Tamil Nadu revisions
# used a different 7-column block with RH_mean and a pressure-ratio
# elev_proxy and labelled it "temperature/pressure block" — there is no
# pressure variable here; it is a temperature + elevation block, and
# elevation_m is now real per-point orography.)
PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "T_sunrise_mean", "T_noon_mean",
             "HDD18", "CDD24", "elevation_m"]

# Canonical-name map: Tamil Nadu's Tier-2 table (02b_build_daily_
# aggregates.py) carries these indices under `_true`/`_mean`-suffixed names.
# Rename to the same names Rajasthan's daily_aggregates_rajasthan_summary.csv
# uses, so the Tier-2 join and every column downstream is identical.
TIER2_RENAME = {
    "GHI_daily_kWh_mean": "GHI_daily_kWh",
    "SAI_true": "SAI",
    "kt_daily_mean": "kt_daily_mean",
    "kt_daily_std": "kt_daily_std",
    "cloudy_frac_true": "cloudy_frac",
    "CCI_true": "CCI",
    "HDD18_true": "HDD18",
    "CDD24_true": "CDD24",
    "DTR_true_mean": "DTR_true",
    "seasonality_true": "seasonality",
}
# Same order/name as Rajasthan's tier2_cols. monsoon_index is not in Tamil
# Nadu's Tier-2 table — it is computed below and attached before the join so
# it lands in this exact position.
TIER2_COLS = ["point_id", "GHI_daily_kWh", "SAI", "kt_daily_mean", "kt_daily_std",
              "cloudy_frac", "CCI", "HDD18", "CDD24", "DTR_true", "seasonality",
              "monsoon_index"]

print("=" * 68)
print("  PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION — Tamil Nadu")
print(f"  Tm_target (base)   = {T_DELIVERY_C:.0f} + {DT_APPROACH_C:.0f} = {TM_TARGET_C:.0f} C")
print(f"  L_required basis   = {NIGHT_DRAW_TOTAL_L:.0f} L over {NIGHT_DISCHARGE_HOURS:.0f}h "
      f"night discharge (Avargani 2021), {ASSUMED_PCM_MASS_KG:.0f} kg PCM — "
      f"a CEILING, not an achievability bar [see module docstring]")
print("=" * 68)


# ═══════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════

print("\n[1/9] Loading inputs ...")

pts_cols = ["point_id", "date", "event", "era5_T_amb", "era5_RHum",
            "era5_GHI", "era5_CSI", "era5_W_spd"]
events_df = pd.read_csv(PHYSICAL_FILE, usecols=pts_cols, parse_dates=["date"])
events_df["event"] = pd.Categorical(events_df["event"], categories=EVENT_ORDER, ordered=True)
print(f"  tamilnadu_cleaned_physical.csv : {len(events_df):,} rows, "
      f"{events_df['point_id'].nunique()} points")

sun_df = pd.read_csv(SUNTIMES_FILE, parse_dates=["date"])
sun_df["time_utc"] = pd.to_datetime(sun_df["time_utc"], utc=True)
print(f"  suntimes.csv                  : {len(sun_df):,} rows")

daily_kt = pd.read_csv(DAILY_AGGREGATES_FILE, usecols=["point_id", "date", "kt_daily"],
                        parse_dates=["date"])
print(f"  daily_aggregates_tamilnadu.csv (date + kt_daily) : {len(daily_kt):,} rows")

tier2 = pd.read_csv(TIER2_FILE).rename(columns=TIER2_RENAME)
# Force the Tier-2 index columns to float64. Rajasthan's equivalent columns
# arrive as float64 (its summary table carries NaNs / the left-join promotes
# them); Tamil Nadu's CCI in particular is written as int by 02b. Casting
# here keeps the final schema dtype-identical across states.
for _c in ("GHI_daily_kWh", "SAI", "kt_daily_mean", "kt_daily_std",
           "cloudy_frac", "CCI", "HDD18", "CDD24", "DTR_true", "seasonality"):
    if _c in tier2.columns:
        tier2[_c] = tier2[_c].astype("float64")
print(f"  tier2_signature_tamilnadu.csv  : {len(tier2)} points")

static_df = pd.read_csv(POPULATION_GRID_FILE)
print(f"  population_grid_points.csv     : {len(static_df)} points")
if "elevation_m" not in static_df.columns or static_df["elevation_m"].isna().any():
    raise SystemExit(
        "ERROR: population_grid_points.csv is missing real per-point elevation_m. "
        "Run 00c_attach_elevation.py first (it attaches ERA5 orography-derived "
        "elevation — required for the PCA block).")
print(f"  elevation_m range             : {static_df['elevation_m'].min():.0f} - "
      f"{static_df['elevation_m'].max():.0f} m (real ERA5 orography, not a flat proxy)")

# monsoon_index — same definition as Rajasthan's 02b (Jun-Sep mean GHI /
# annual mean GHI), but computed here from the 3x/day ERA5 noon GHI because
# Tamil Nadu's Tier-2 table doesn't carry it and the NASA POWER hourly cache
# needed for a full-hourly integral is not on disk. A coarser sample than
# Rajasthan's — state this in the methodology. (It is still a solar-fraction
# proxy, NOT a precipitation-based index; PRECTOTCORR was never downloaded.)
noon = events_df[events_df["event"] == "noon"].copy()
noon["month"] = noon["date"].dt.month
monthly_noon_ghi = noon.groupby(["point_id", "month"])["era5_GHI"].mean().reset_index()
jjas_ghi = (monthly_noon_ghi[monthly_noon_ghi["month"].isin([6, 7, 8, 9])]
            .groupby("point_id")["era5_GHI"].mean())
annual_ghi = monthly_noon_ghi.groupby("point_id")["era5_GHI"].mean()
monsoon_index = (jjas_ghi / annual_ghi).rename("monsoon_index")
tier2 = tier2.merge(monsoon_index, on="point_id", how="left")
print("  [WARN] monsoon_index is a Jun-Sep GHI-fraction PROXY from 3x/day ERA5 "
      "noon GHI (not a full-hourly integral, not precipitation-based). State "
      "this in the methodology write-up.")


# ═══════════════════════════════════════════════════════════
# 2/3. TIER 1 + DAYLENGTH — via signature_lib.build_tier1_signature()
# ═══════════════════════════════════════════════════════════
# Shared with Rajasthan's 04b and 05_cluster_rajasthan.py's Level B (one row
# per point PER SEASON) — one implementation, different group_keys. Here
# group_keys=["point_id"] gives the whole-year, one-row-per-point signature.

print("\n[2-3/9] Tier 1 (per-event aggregates) + daylength, via signature_lib ...")

tier1 = build_tier1_signature(events_df, sun_df, group_keys=["point_id"])
tier1.index.name = "point_id"

print(f"  Tier 1: {tier1.shape[1]} columns x {tier1.shape[0]} points")
print(f"  daylength_mean range: {tier1['daylength_mean'].min():.2f} - "
      f"{tier1['daylength_mean'].max():.2f} h")


# ═══════════════════════════════════════════════════════════
# 4. TIER 2 — join daily-integral indices
# ═══════════════════════════════════════════════════════════

print("\n[4/9] Tier 2 — joining tier2_signature_tamilnadu.csv (canonical names) ...")

sig = tier1.join(tier2.set_index("point_id")[TIER2_COLS[1:]], how="left")
n_missing_tier2 = sig["GHI_daily_kWh"].isna().sum()
if n_missing_tier2:
    print(f"  [WARN] {n_missing_tier2} point(s) have no Tier 2 row "
          f"(tier2_signature_tamilnadu.csv incomplete for those points)")


# ═══════════════════════════════════════════════════════════
# 5. STATIC ATTRIBUTES  (reporting/mapping only; elevation_m feeds PCA)
# ═══════════════════════════════════════════════════════════

print("\n[5/9] Joining static attributes (elevation_m, population, weight, lat, lon) ...")

static_cols = static_df.set_index("point_id")[["lat", "lon", "population", "weight", "elevation_m"]]
sig = sig.join(static_cols, how="left")


# ═══════════════════════════════════════════════════════════
# 6. DERIVED PCM-FACING QUANTITIES  (§6.3)
# ═══════════════════════════════════════════════════════════

print("\n[6/9] Derived PCM-facing quantities (Tm_target, Tm_target_capped, L_required) ...")

sig["Tm_target_C"] = TM_TARGET_C   # constant by design rule — same across all points

# kt_p05 per point — 5th percentile of the DAILY kt series. Kept for
# reference/audit only; NOT used for Tm_target_capped_C (see the worst-month
# correction in the module docstring). Nothing downstream reads this column.
kt_p05 = daily_kt.groupby("point_id")["kt_daily"].quantile(0.05)
sig["kt_p05"] = kt_p05

# kt_worst_month per point — the lowest of that point's 12 calendar-month
# MEAN kt_daily values. A monthly mean is far less extreme than a single-day
# p05 figure, matching literature-standard "worst month" solar sizing
# practice (Durin et al. 2018).
daily_kt["month"] = daily_kt["date"].dt.month
monthly_mean_kt = daily_kt.groupby(["point_id", "month"])["kt_daily"].mean()
kt_worst_month = monthly_mean_kt.groupby("point_id").min()
sig["kt_worst_month"] = kt_worst_month

# Upper-bound cap: Tm must lie below the collector delivery temperature
# achievable on a poor-insolation period in that regime. Hottel-Whillier-
# style linear scaling: achievable temperature rise above ambient scales
# roughly linearly with clearness.
#   Tm_cap_C = Ta_mean + (kt_poor / kt_daily_mean) * (Tm_target_C - Ta_mean)
# where kt_poor = kt_worst_month. An explicit heuristic, not a calibrated
# collector model — state it as such if cited. The old kt_p05-based value is
# kept as Tm_target_capped_C_p05day for audit/comparison only.
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

# T_mains estimate — flat `Ta_mean - 2 C` offset, identical in both states'
# 04b. TODO (shared, tracked once): this is NOT a published correlation —
# see pcm_shared_config.T_MAINS_EST_C_TODO. Replace with a Kusuda &
# Achenbach-style ground-temperature annual-lag model before L_required (and
# the Phase 5 latent-heat gate it drives) is presented as final.
sig["T_mains_est_C"] = sig["Ta_mean"] - 2.0

# Q_night = total night-draw mass x cp_water x (T_delivery - T_mains) — a
# TOTAL energy over the discharge window. PCM contributes a literature-
# anchored fractional share (SHARE_PCM); tank sensible heat + concurrent
# daytime charging supply the rest.
Q_night_kJ = NIGHT_DRAW_TOTAL_KG * CP_WATER * (T_DELIVERY_C - sig["T_mains_est_C"])
Q_night_pcm_kJ = SHARE_PCM * Q_night_kJ
sig["L_required_kJ_per_kg"] = Q_night_pcm_kJ / ASSUMED_PCM_MASS_KG
print(f"  L_required_kJ_per_kg  : {sig['L_required_kJ_per_kg'].min():.0f} - "
      f"{sig['L_required_kJ_per_kg'].max():.0f} kJ/kg  (literature-anchored, "
      f"PCM {SHARE_PCM*100:.0f}% of total night delivery, with tank sensible heat "
      f"+ concurrent charging supplying the rest)")
print(f"  [TODO] {T_MAINS_EST_C_TODO}")


# ═══════════════════════════════════════════════════════════
# 7. INTERACTION TERMS  (§6.4 — identical to Rajasthan; sun-event based)
# ═══════════════════════════════════════════════════════════

print("\n[7/9] Interaction terms ...")

# charging energy weighted by its unreliability; high values mean a large
# but erratic resource
sig["int_GHI_x_ktstd"] = sig["GHI_daily_kWh"] * sig["kt_daily_std"]
# cycling stress under intermittent charging, the worst case for phase
# stability
sig["int_DTR_x_cloudyfrac"] = sig["DTR_true"] * sig["cloudy_frac"]
# condensation risk at the store surface at the condensation-critical
# instant (sunrise) — replaces Tamil Nadu's former int_RH_x_TaMinusTm
sig["int_RH_x_TsunriseMinusTm"] = sig["RH_sunrise_mean"] * (sig["T_sunrise_mean"] - sig["Tm_target_C"])
# convective loss driving potential during the evening draw — replaces
# Tamil Nadu's former int_wind_x_TaMinusTsoil (T_soil was an undefined proxy)
sig["int_wind_x_TsunsetMinusTdelivery"] = sig["wind_sunset_mean"] * (sig["T_sunset_mean"] - T_DELIVERY_C)
# combined autonomy requirement
sig["int_CCI_x_1minusSAI"] = sig["CCI"] * (1 - sig["SAI"])

print("  Added 5 interaction terms (GHIxkt_std, DTRxcloudy_frac, "
      "RHx(Tsunrise-Tm), windx(Tsunset-Tdelivery), CCIx(1-SAI))")


# ═══════════════════════════════════════════════════════════
# 8. PCA ON THE CORRELATED TEMPERATURE/ELEVATION BLOCK ONLY
# ═══════════════════════════════════════════════════════════

print("\n[8/9] PCA on the correlated block "
      f"({', '.join(PCA_BLOCK)}) ...")

pca_input = sig[PCA_BLOCK].fillna(sig[PCA_BLOCK].median())
pca_scaler = StandardScaler()
pca_input_scaled = pca_scaler.fit_transform(pca_input)

# n_components PINNED (pcm_shared_config.PCA_N_COMPONENTS) rather than a
# data-determined 0.95-variance threshold, so climate_signature_tamilnadu.csv
# and climate_signature_rajasthan.csv carry the SAME PC1..PCn columns and
# Phase 4's 05_cluster_regions.py can concatenate them. Tamil Nadu's
# temp/elevation block reaches 95% variance by PC3, so PC4 here is a
# low-variance (near-noise) component kept only for schema alignment.
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
pca_scores = pca.fit_transform(pca_input_scaled)
n_comp = pca_scores.shape[1]
for i in range(n_comp):
    sig[f"PC{i+1}"] = pca_scores[:, i]

loadings = pd.DataFrame(pca.components_.T, index=PCA_BLOCK,
                         columns=[f"PC{i+1}" for i in range(n_comp)])
print(f"  {n_comp} components retained (pinned; cumulative variance "
      f"{pca.explained_variance_ratio_.sum():.3f}). Loadings:")
print(loadings.round(3).to_string())
print(f"  Explained variance ratio: {np.round(pca.explained_variance_ratio_, 3)}")
print("  -> Read the sign/magnitude pattern per component (plan doc expects roughly "
      "'heat', 'altitude', 'seasonal amplitude' across all four states) — name them "
      "in the write-up.")


# ═══════════════════════════════════════════════════════════
# 9. STANDARDIZE THE CLUSTERING-READY MATRIX  (*_z columns)
# ═══════════════════════════════════════════════════════════

print("\n[9/9] Standardizing the clustering-ready feature matrix ...")

# Explicitly excluded from every standardized/clustering-ready column —
# IDENTICAL to Rajasthan's NON_CLUSTERING_COLS:
#   - lat, lon                 : plotting only, never fitting (plan doc §6.2)
#   - population, weight        : reporting only (plan doc §6.2)
#   - elevation_m                : feeds PCA, not a standalone clustering
#                                 column — the PC*_z scores subsume it
#   - PCA_BLOCK raw columns      : replaced by PC1..PCn
#   - T_mains_est_C, kt_p05, kt_worst_month, Tm_target_capped_C_p05day,
#     tm_target_capped_flag      : helper/proxy/diagnostic, not independent
#                                 climate dimensions
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
# CORRELATION HEATMAP + |r| > 0.9 FLAG  (final feature set, PCA block
# already removed — any flagged pair is NEW collinearity)
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
    title="Climate Signature — Final Feature Set Correlation (Tamil Nadu)",
    height=max(500, 28 * len(corr_cols_nonconst)), width=max(600, 28 * len(corr_cols_nonconst)),
    xaxis=dict(tickangle=45),
)
heatmap_path = OUTPUTS_DIR / "signature_correlation_heatmap_tamilnadu.html"
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


# ═══════════════════════════════════════════════════════════
# SIGNATURE-LEVEL QC PLOTS
# ═══════════════════════════════════════════════════════════

print("\nSignature-level QC plots ...")

n_cols_grid = 4
n_rows_grid = int(np.ceil(len(corr_cols_nonconst) / n_cols_grid))
fig = make_subplots(rows=n_rows_grid, cols=n_cols_grid, subplot_titles=corr_cols_nonconst)
for idx, col in enumerate(corr_cols_nonconst):
    r, c = divmod(idx, n_cols_grid)
    fig.add_trace(go.Histogram(x=sig[col], showlegend=False, marker_color="#4c72b0"),
                  row=r + 1, col=c + 1)
fig.update_layout(title="Climate Signature — Distribution of Every Clustering-Input Column "
                         f"({len(sig)} points, Tamil Nadu)",
                   height=max(600, 220 * n_rows_grid), showlegend=False)
dist_path = OUTPUTS_DIR / "signature_distributions_tamilnadu.html"
fig.write_html(str(dist_path))
print(f"  Saved: {dist_path}")

fig = make_subplots(rows=1, cols=2, subplot_titles=["GHI_daily_kWh (Tier 2 daily mean)", "monsoon_index (Jun-Sep GHI-fraction proxy)"])
fig.add_trace(go.Scatter(
    x=sig["lon"], y=sig["lat"], mode="markers",
    marker=dict(size=7, color=sig["GHI_daily_kWh"], colorscale="YlOrRd",
                colorbar=dict(title="kWh/m^2/day", x=0.44), showscale=True),
    text=sig.index, name="GHI_daily_kWh",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=sig["lon"], y=sig["lat"], mode="markers",
    marker=dict(size=7, color=sig["monsoon_index"], colorscale="Blues",
                colorbar=dict(title="monsoon_index", x=1.0), showscale=True),
    text=sig.index, name="monsoon_index",
), row=1, col=2)
fig.update_layout(title=f"Point-Signature Map — Tamil Nadu ({len(sig)} points)", showlegend=False)
fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
map_path = OUTPUTS_DIR / "signature_point_map_tamilnadu.html"
fig.write_html(str(map_path))
print(f"  Saved: {map_path}")

print("\n" + "=" * 68)
print("  PHASE 3 COMPLETE — Tamil Nadu")
print(f"  Output: {CLIMATE_SIGNATURE_FILE}")
print("=" * 68)
print("\nColumn schema now matches climate_signature_rajasthan.csv exactly — "
      "Phase 4's 05_cluster_regions.py can concatenate both states directly.")
