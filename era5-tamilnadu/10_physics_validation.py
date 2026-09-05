"""
10_physics_validation.py
============================
PHASE 7 — PHYSICS-BASED VALIDATION (Objective 1 plan v3.0, Section 10)

NOT deferred to future work. The plan is explicit about why this can't be
skipped: "Everything up to [MCDM] produces a preference ordering. Nothing
in it establishes that a higher-ranked PCM actually performs better...
this phase makes the claim falsifiable." Four MCDM methods agreeing with
each other is close to a tautology if they're all fed the same matrix —
this script is the independent check.

TOOL CHOICE (matches plan v3.0 Section 10.1 exactly)
--------------------------------------------------------
Python grey-box lumped-enthalpy tank model — the plan's own PRIMARY
choice, explicitly over EnergyPlus (rejected: "no supported path to place
a latent-heat PCM inside the tank node network") and CFD (rejected: "out
of scope and unnecessary... a well-calibrated lumped model validated
against published experiment is appropriate").

MODEL (3-phase, adapted from Barqawi2025's ODE structure — already
extracted in your Sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md
Eqs. 1-16 — combined with the collector-coupling form used there)
--------------------------------------------------------------------------
Two coupled lumped nodes: tank water (Tw) and PCM (Tp or melt fraction f
during the isothermal plateau at Tm). Three phases per the PCM's thermal
state (pre-melt sensible, isothermal melting, post-melt sensible), driven
hour-by-hour for a full REAL year of the cluster's medoid point's actual
daily GHI/temperature data (data/processed/daily_aggregates_tamilnadu.csv
from 02b — this is real measured/reanalysis-derived data, not synthetic).

Each hourly step is solved with BACKWARD EULER (implicit), not forward
Euler — the water-tank time constant here (~minutes, given a
well-coupled coil) is short relative to an hourly step, so forward Euler
would be numerically unstable; backward Euler is unconditionally stable
for this linear system and is a single closed-form 2x2 solve per step, no
iterative solver needed.

STATED ASSUMPTIONS — all of these are documented choices, not measured
values, exactly like every other stated assumption elsewhere in this
pipeline. Report them as such if you cite results from this script.
--------------------------------------------------------------------------
  Tank water mass Mw          150 kg   (mid-range domestic tank; SWH
                                         literature in your Sources/ spans
                                         30-360 L depending on household size)
  Collector-tank coil area Ac  2.5 m^2  (Barqawi2025 Table 1)
  Water-coil HTC hc            1500 W/m^2K  (Barqawi2025)
  Collector efficiency eff     0.70     (mid-range of Al-Mamun2023's cited
                                         45-73% FPC efficiency band)
  PCM volume                   0.035 m^3  (Barqawi2025 mid-configuration)
  PCM-water HTC hp             800 W/m^2K  (Barqawi2025)
  PCM surface area Ap          3.5 m^2   (Barqawi2025)
  Draws                        2/day, 75 kg each, 07:00 and 19:00 local
                                         (IST) — a stated, simple household
                                         schedule, not a measured profile
  Target delivery temperature  50 C      (same T_delivery used throughout
                                         this pipeline's Tm_target rule)
  Ambient ombient ambient temp  daily sinusoid built from that day's real
                                         Ta_min_true/Ta_max_true, peak 14:00
                                         local, trough 05:00 local — a
                                         standard diurnal-cycle assumption

CALIBRATION CHECK (plan v3.0 Table 16)
------------------------------------------
Annual solar fraction is expected to land in the 54-84% published range
for SWH systems (Table 16). This script reports where every simulated
result falls relative to that band — if results are systematically
outside it, that's a signal to revisit the tank/collector parameters
above before trusting the Spearman rho, not something to silently accept.

OUTPUT INTERPRETATION (plan v3.0 Table 17 — all three outcomes are
publishable, decide now that you'll report whichever one you get)
--------------------------------------------------------------------------
  rho > 0.8       : MCDM ranking predicts physical performance (strong)
  0.4 < rho < 0.8 : partial agreement (the most likely outcome, still good —
                    identify which criterion is driving disagreement)
  rho < 0.4       : MCDM ranking doesn't predict performance — diagnose
                    which criterion/weight is wrong, report before/after

INPUT  : data/processed/daily_aggregates_tamilnadu.csv   (02b's output)
         data/processed/suntimes.csv                     (00b's output)
         data/processed/clustering/cluster_assignments_tamilnadu.csv
         data/processed/pcm/mcdm_full_scores_by_cluster.csv
OUTPUT : data/processed/pcm/physics_validation_results.csv
           one row per (cluster_id, pcm_name): simulated annual solar
           fraction, hours/year delivery target met, complete cycles/year
         data/processed/pcm/physics_validation_spearman.csv
           one row per cluster: Spearman rho vs. MCDM consensus rank

HOW TO RUN:
  python 10_physics_validation.py

This can take a few minutes (full year, hourly steps, every feasibility
survivor in every cluster) — that's expected, not a hang.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PROCESSED_DIR, SUNTIMES_FILE

DAILY_FILE = PROCESSED_DIR / "daily_aggregates_tamilnadu.csv"
ASSIGN_FILE = PROCESSED_DIR / "clustering" / "cluster_assignments_tamilnadu.csv"
SCORES_FILE = PROCESSED_DIR / "pcm" / "mcdm_full_scores_by_cluster.csv"
OUT_RESULTS = PROCESSED_DIR / "pcm" / "physics_validation_results.csv"
OUT_SPEARMAN = PROCESSED_DIR / "pcm" / "physics_validation_spearman.csv"

# ─── Stated tank/collector assumptions (see docstring) ──────────────────
M_W_KG = 150.0
C_W_JKGK = 4186.0
A_C_M2 = 2.5
H_C_WM2K = 1500.0
COLLECTOR_EFF = 0.70

V_PCM_M3 = 0.035
H_P_WM2K = 800.0
A_P_M2 = 3.5
DEFAULT_PCM_DENSITY_KG_M3 = 800.0
DEFAULT_CP_JKGK = 2000.0

# ─── Ambient tank heat-loss term (BUG FIX v3.1) ──────────────────────────
# The original model omitted tank-to-ambient losses.  Without this term the
# tank stays at delivery temperature all night, producing solar fractions of
# 90-99% and 0-1 complete PCM freeze-melt cycles per year — both outside the
# published 54-84% SF benchmark band (plan v3.0 Table 16) and physically
# unrealistic for a domestic solar water heater.
#
# UA_TANK_W_K represents the total conductance of the tank shell to ambient
# air.  For a well-insulated 150 L stainless-steel tank with 50 mm mineral
# wool insulation (k ≈ 0.04 W/m·K), the outer area is ≈ 1.5 m² and the
# effective U-value is ≈ 0.8 W/m²·K → UA ≈ 1.2 W/K.  A slightly higher
# value of 2.0 W/K is used here as a conservative (higher loss) estimate
# consistent with real-world installation imperfections and pipe losses.
# Reported as a stated assumption per the framework plan.
UA_TANK_W_K = 2.0   # W/K  tank-to-ambient conductance (stated assumption)

# ─── Night/idle collector-coupling isolation (BUG FIX) ───────────────────
# The base model couples the tank to the collector via a*(Tc-Tw) every hour,
# day and night alike. At night Tc collapses to Tamb (isolar=0), so an
# un-isolated coupling lets the tank drain heat back out through the idle
# collector loop at essentially the same rate it charges during the day —
# physically wrong (real installations use a check valve or a
# controller-gated pump specifically to prevent this) and, on top of the
# separate UA_TANK ambient-loss term above, double-counts overnight losses.
# This is the same bug class documented and fixed in the Rajasthan pipeline
# ("Barqawi's bidirectional a·(Tc−Tw) term let the tank drain heat through
# an idle collector overnight nearly as fast as it charged during the day").
# Fix: gate the collector-coupling coefficient to a small fraction of its
# daytime value whenever the collector is colder than the tank (Tc < Tw).
NIGHT_ISOLATION_FRACTION = 0.05

DRAW_HOURS_LOCAL = [7, 19]
DRAW_MASS_KG = 75.0
T_DELIVERY_C = 50.0

MAX_PCMS_PER_CLUSTER = 20      # safety cap, not expected to bind given ~62-row database
BENCHMARK_SF_LOW, BENCHMARK_SF_HIGH = 0.54, 0.84   # plan v3.0 Table 16


def pick_medoid_point(assign_df, cluster_id):
    sub = assign_df[assign_df["cluster_id"] == cluster_id]
    return sub.loc[sub["max_membership_prob"].idxmax(), "point_id"]


def pick_representative_year(daily_point_df):
    daily_point_df = daily_point_df.copy()
    daily_point_df["year"] = pd.to_datetime(daily_point_df["date"]).dt.year
    counts = daily_point_df.groupby("year").size()
    best_year = counts.idxmax()
    return daily_point_df[daily_point_df["year"] == best_year].sort_values("date").reset_index(drop=True)


def build_hourly_drivers(year_df, sun_df, point_id):
    """Returns hourly arrays (len = 24*n_days):
      Tc        — collector coil temperature driving the tank [C]
      T_mains   — mains water temperature (constant within a day) [C]
      hour_of_day — local hour 0-23
      Tamb      — raw ambient air temperature [C] (needed by tank loss term)
    """
    sun_local = sun_df[sun_df["point_id"] == point_id].copy()
    sun_local["date"] = pd.to_datetime(sun_local["date"]).dt.date
    sun_local["time_utc"] = pd.to_datetime(sun_local["time_utc"], utc=True)
    sun_pivot = sun_local.pivot_table(index="date", columns="event", values="time_utc", aggfunc="first")

    Tc_all, Tmains_all, hour_all, Tamb_all = [], [], [], []

    for _, day in year_df.iterrows():
        d = pd.to_datetime(day["date"]).date()
        ta_mean = day["Ta_mean_true"]
        ta_min = day.get("Ta_min_true", ta_mean - 4)
        ta_max = day.get("Ta_max_true", ta_mean + 4)
        ghi_kwh = day["GHI_daily_kWh"]
        t_mains = ta_mean - 2.0

        if d in sun_pivot.index and pd.notna(sun_pivot.loc[d].get("sunrise", pd.NaT)) \
                and pd.notna(sun_pivot.loc[d].get("sunset", pd.NaT)):
            sr = sun_pivot.loc[d, "sunrise"] + pd.Timedelta(hours=5, minutes=30)
            ss = sun_pivot.loc[d, "sunset"] + pd.Timedelta(hours=5, minutes=30)
            sunrise_hr = sr.hour + sr.minute / 60.0
            sunset_hr = ss.hour + ss.minute / 60.0
        else:
            sunrise_hr, sunset_hr = 6.25, 18.25   # fallback: TN annual-average approx

        daylen_hr = max(sunset_hr - sunrise_hr, 1.0)
        imax_kw = (ghi_kwh * np.pi) / (2.0 * daylen_hr) if ghi_kwh == ghi_kwh else 0.0
        imax_wm2 = max(imax_kw, 0.0) * 1000.0

        for h in range(24):
            if sunrise_hr <= h <= sunset_hr:
                isolar = imax_wm2 * np.sin(np.pi * (h - sunrise_hr) / daylen_hr)
            else:
                isolar = 0.0
            # diurnal ambient sinusoid: peak 14:00, trough 05:00 local
            tamb = (ta_min + ta_max) / 2 + (ta_max - ta_min) / 2 * np.sin(
                2 * np.pi * (h - 5) / 24 - np.pi / 2) if (ta_max == ta_max and ta_min == ta_min) else ta_mean
            tc = tamb + COLLECTOR_EFF * isolar / 20.0    # Barqawi2025 Eq. 3 form
            Tc_all.append(tc)
            Tmains_all.append(t_mains)
            hour_all.append(h)
            Tamb_all.append(tamb)   # <-- raw ambient temp, separate from Tc

    return np.array(Tc_all), np.array(Tmains_all), np.array(hour_all), np.array(Tamb_all)


def simulate_pcm_swh_year(Tc, T_mains, hour_of_day, pcm_row, tamb_arr=None, dt=3600.0):
    Tm = pcm_row["Tm_C"]
    Hf = pcm_row["latent_heat_kJ_kg"] * 1000.0   # J/kg
    density = pcm_row.get("density_solid_kg_m3", np.nan)
    if not (density == density):
        density = DEFAULT_PCM_DENSITY_KG_M3
    Mp = density * V_PCM_M3

    Cp_s = pcm_row.get("Cp_solid_kJ_kgK", np.nan)
    Cp_l = pcm_row.get("Cp_liquid_kJ_kgK", np.nan)
    Cp_s = Cp_s * 1000.0 if Cp_s == Cp_s else DEFAULT_CP_JKGK
    Cp_l = Cp_l * 1000.0 if Cp_l == Cp_l else DEFAULT_CP_JKGK

    a = 1.0 / (M_W_KG * C_W_JKGK / (H_C_WM2K * A_C_M2))     # 1/tau_w
    eta = (H_P_WM2K * A_P_M2) / (H_C_WM2K * A_C_M2)
    b = a * eta
    tau_ps = Mp * Cp_s / (H_P_WM2K * A_P_M2)
    tau_pl = Mp * Cp_l / (H_P_WM2K * A_P_M2)
    Qp_max = Hf * Mp

    Tw = T_mains[0] + 10.0
    Tp = Tw
    phase = 1
    Qp = 0.0
    n_complete_cycles = 0
    was_liquid_this_day = False

    solar_delivered_total = 0.0
    demand_total = 0.0
    hours_target_met = 0

    for i in range(len(Tc)):
        tc, tmains, h = Tc[i], T_mains[i], hour_of_day[i]
        # ambient temperature for this hour (used in tank loss term)
        # build_hourly_drivers stores Ta in the Tc array via tamb + solar gain;
        # we need raw tamb here.  Since Tc = tamb + eff*isolar/20, we cannot
        # easily separate them after the fact.  Instead we track ambient
        # independently via a separate Tamb array built in build_hourly_drivers.
        # IMPLEMENTATION NOTE: This function receives a pre-built Tamb array
        # (tamb_arr) added to the signature in the corrected build_hourly_drivers.
        tamb = tamb_arr[i] if tamb_arr is not None else 30.0   # fallback: Tamil Nadu mean

        # Tank-to-ambient heat loss [J] this step:  Q_loss = UA * (Tw - tamb) * dt
        # Incorporated into the backward-Euler denominator to maintain
        # unconditional stability (treats loss as linear in Tw at step end).
        loss_coeff = UA_TANK_W_K * dt / (M_W_KG * C_W_JKGK)   # dimensionless

        # Night/idle isolation (see NIGHT_ISOLATION_FRACTION docstring above):
        # only the collector-tank coupling is gated, not the PCM-tank coupling
        # b (an internal storage exchange, not an external loop with a valve).
        a_eff = a if tc >= Tw else a * NIGHT_ISOLATION_FRACTION

        if phase == 1:
            c = 1.0 / tau_ps
            denom1 = 1 + dt * a_eff + dt * b + loss_coeff
            # Closed-form backward-Euler solve of the coupled 2x2 implicit system:
            #   Tw_new*denom1 = Tw + dt*a*tc + loss_coeff*tamb + dt*b*Tp_new
            #   Tp_new*(1+dt*c) = Tp + dt*c*Tw_new
            # Substituting the second into the first and solving for Tw_new gives
            # a numerator of dt*b*Tp (the OLD Tp only) — NOT dt*b*(Tp + dt*c*Tw),
            # which was a spurious extra term (the same bug class documented in
            # the Rajasthan pipeline's physics_lib.py: "a wrong closed-form
            # backward-Euler solve... caused unbounded temperature blow-up").
            # Verified: with typical parameters this spurious term pushed Tw_new
            # above the driving collector temperature Tc in a single step, which
            # is thermodynamically impossible for this passive linear coupling.
            Tw_new = ((Tw + dt * a_eff * tc + loss_coeff * tamb) * (1 + dt * c)
                      + dt * b * Tp) / \
                     (denom1 * (1 + dt * c) - dt * b * dt * c)
            Tp_new = (Tp + dt * c * Tw_new) / (1 + dt * c)
            Tw, Tp = Tw_new, Tp_new
            if Tp >= Tm:
                phase, Tp, Qp = 2, Tm, 0.0
        elif phase == 2:
            denom = 1 + dt * a_eff + dt * b + loss_coeff
            Tw_new = (Tw + dt * a_eff * tc + dt * b * Tm + loss_coeff * tamb) / denom
            dQ = H_P_WM2K * A_P_M2 * max(0.0, Tw_new - Tm) * dt
            Qp += dQ
            Tw = Tw_new
            if Qp >= Qp_max:
                phase = 3
                Tp = Tm + max(0.0, Qp - Qp_max) / (Mp * Cp_l + 1e-9)
                was_liquid_this_day = True
        else:  # phase 3
            c = 1.0 / tau_pl
            denom1 = 1 + dt * a_eff + dt * b + loss_coeff
            # Same closed-form fix as phase 1 above (numerator uses old Tp only).
            Tw_new = ((Tw + dt * a_eff * tc + loss_coeff * tamb) * (1 + dt * c)
                      + dt * b * Tp) / \
                     (denom1 * (1 + dt * c) - dt * b * dt * c)
            Tp_new = (Tp + dt * c * Tw_new) / (1 + dt * c)
            Tw, Tp = Tw_new, Tp_new
            if Tp < Tm:
                phase = 1
                if was_liquid_this_day:
                    n_complete_cycles += 1
                    was_liquid_this_day = False

        if Tw >= T_DELIVERY_C:
            hours_target_met += 1

        if h in DRAW_HOURS_LOCAL:
            demand_energy = DRAW_MASS_KG * C_W_JKGK * max(0.0, T_DELIVERY_C - tmains)
            solar_energy = DRAW_MASS_KG * C_W_JKGK * max(0.0, min(Tw, T_DELIVERY_C) - tmains)
            demand_total += demand_energy
            solar_delivered_total += solar_energy
            Tw = (Tw * (M_W_KG - DRAW_MASS_KG) + tmains * DRAW_MASS_KG) / M_W_KG

    solar_fraction = solar_delivered_total / demand_total if demand_total > 0 else np.nan
    return {
        "annual_solar_fraction": solar_fraction,
        "hours_target_met_per_year": hours_target_met,
        "complete_cycles_per_year": n_complete_cycles,
    }


def main():
    print("=" * 68)
    print("  Phase 7 — Grey-Box Physics Validation — Tamil Nadu")
    print("=" * 68)

    for f in (DAILY_FILE, SUNTIMES_FILE, ASSIGN_FILE, SCORES_FILE):
        if not f.exists():
            print(f"\n  ERROR: {f} not found.")
            return

    daily_all = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    sun_df = pd.read_csv(SUNTIMES_FILE)
    assign_df = pd.read_csv(ASSIGN_FILE)
    scores_df = pd.read_csv(SCORES_FILE)

    all_results, spearman_rows = [], []

    for cid in sorted(assign_df["cluster_id"].unique()):
        medoid = pick_medoid_point(assign_df, cid)
        year_df = pick_representative_year(daily_all[daily_all["point_id"] == medoid])
        if len(year_df) < 300:
            print(f"\n  Cluster {int(cid)}: medoid {medoid} has only {len(year_df)} usable days "
                  f"in its best year — skipping (need a more complete year).")
            continue

        Tc, T_mains, hour_of_day, Tamb = build_hourly_drivers(year_df, sun_df, medoid)
        print(f"\n  Cluster {int(cid)}  medoid={medoid}  "
              f"year={pd.to_datetime(year_df['date']).dt.year.iloc[0]}  "
              f"({len(year_df)} days, {len(Tc)} hourly steps)  "
              f"[UA_TANK={UA_TANK_W_K} W/K ambient loss active]")

        candidates = scores_df[scores_df["cluster_id"] == cid].sort_values("consensus_rank")
        candidates = candidates.head(MAX_PCMS_PER_CLUSTER)

        for _, pcm_row in candidates.iterrows():
            sim = simulate_pcm_swh_year(Tc, T_mains, hour_of_day, pcm_row, tamb_arr=Tamb)
            in_band = BENCHMARK_SF_LOW <= sim["annual_solar_fraction"] <= BENCHMARK_SF_HIGH
            all_results.append({
                "cluster_id": cid, "medoid_point": medoid, "name": pcm_row["name"],
                "consensus_rank": pcm_row["consensus_rank"],
                "annual_solar_fraction": sim["annual_solar_fraction"],
                "in_benchmark_band_54_84pct": in_band,
                "hours_target_met_per_year": sim["hours_target_met_per_year"],
                "complete_cycles_per_year": sim["complete_cycles_per_year"],
            })
            print(f"    {pcm_row['name']:35s}  rank={int(pcm_row['consensus_rank'])}  "
                  f"SF={sim['annual_solar_fraction']*100:5.1f}%  "
                  f"{'[in band]' if in_band else '[OUT OF BAND]'}  "
                  f"cycles/yr={sim['complete_cycles_per_year']}")

        if len(candidates) >= 3:
            rho, pval = spearmanr(candidates["consensus_rank"],
                                   [r["annual_solar_fraction"] for r in all_results[-len(candidates):]])
            # consensus_rank: 1=best; solar_fraction: higher=better -> expect NEGATIVE rho
            # for good agreement. Report the sign-corrected rho (positive = agreement).
            rho_agreement = -rho if rho == rho else np.nan
            interp = ("strong agreement" if rho_agreement > 0.8 else
                      "partial agreement" if rho_agreement > 0.4 else
                      "weak agreement — diagnose before trusting the MCDM ranking here")
            spearman_rows.append({"cluster_id": cid, "n_candidates": len(candidates),
                                   "spearman_rho": rho_agreement, "p_value": pval,
                                   "interpretation": interp})
            print(f"    Spearman rho (MCDM rank vs. simulated solar fraction): "
                  f"{rho_agreement:.3f}  ({interp})")
        else:
            print(f"    [SKIP] <3 candidates — Spearman rho not meaningful for this cluster.")

    results_df = pd.DataFrame(all_results)
    spearman_df = pd.DataFrame(spearman_rows)
    results_df.to_csv(OUT_RESULTS, index=False)
    spearman_df.to_csv(OUT_SPEARMAN, index=False)

    print("\n" + "=" * 68)
    print("  DONE")
    print(f"  Saved: {OUT_RESULTS}")
    print(f"  Saved: {OUT_SPEARMAN}")
    if len(results_df):
        pct_in_band = results_df["in_benchmark_band_54_84pct"].mean() * 100
        print(f"\n  {pct_in_band:.0f}% of all simulations landed in the published "
              f"54-84% solar-fraction benchmark band (plan v3.0 Table 16).")
        if pct_in_band < 50:
            print("  [NOTE] Less than half in-band — before trusting the Spearman rho "
                  "results, revisit the stated tank/collector assumptions at the top "
                  "of this script (M_W_KG, A_C_M2, COLLECTOR_EFF, draw schedule) — a "
                  "systematically low or high solar fraction usually traces to one of "
                  "those, not to the PCM choice itself.")
    if len(spearman_df):
        print(f"\n  Mean Spearman rho across clusters: {spearman_df['spearman_rho'].mean():.3f}")
        print("  Report per plan v3.0 Table 17 — ALL THREE outcome bands are publishable")
        print("  if diagnosed; don't chase a specific number.")
    print("=" * 68)


if __name__ == "__main__":
    main()
