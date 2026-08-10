"""
03b_agreement_analysis.py
=============================================================================
CROSS-SOURCE VALIDATION — RAJASTHAN POPULATION POINTS  (ERA5 vs NASA POWER)
=============================================================================
Decides whether ERA5 alone is a defensible backbone for downstream
preprocessing, or needs bias correction against NASA POWER, before any
preprocessing step touches the physical values. Read-only with respect to
climate_rajasthan_points.csv — this script never writes back to it.

INPUT  : data/processed/climate_rajasthan_points.csv  (02_combine_rajasthan.py's output)
OUTPUTS:
  data/processed/era5_power_agreement_rajasthan.csv
      Full variable x season x event MBE/RMSE/Pearson-r table, plus
      "ALL x <event>" and "<season> x ALL" and "ALL x ALL" aggregate rows
      per variable.
  outputs/qc_era5_power_scatter_rajasthan.html
      ERA5-vs-POWER GHI scatter, one panel per season, y=x identity line.
  outputs/bias_decision_rajasthan.txt
      Which of the three decision branches was taken, with the supporting
      numbers, written in prose for direct use in the methodology section.

VARIABLES COMPARED
-------------------
  era5_GHI    vs power_ALLSKY_SFC_SW_DWN   (GHI / irradiance, W/m^2)
  era5_T_amb  vs power_T2M                 (2m temperature, degC)
  era5_RHum   vs power_RH2M                (relative humidity, %)
  era5_W_spd  vs power_WS10M               (10m wind speed, m/s)

DECISION RULE  (see decide_branch() for the exact thresholds)
---------------------------------------------------------------------
Applied to GHI specifically (the variable that matters most downstream),
using the NOON event as the primary signal — sunrise/sunset sit near the
solar-elevation zero-crossing, where both sources are close to zero and
relative disagreement is naturally large but physically inconsequential;
that is expected, not a failure, and is not what drives this decision.

  1. BACKBONE      : high noon correlation, small noon MBE, no meaningful
                      season-to-season MBE spread -> ERA5 alone is used
                      downstream, POWER remains a reported cross-check.
  2. QUANTILE_MAP  : correlation still reasonable but a systematic,
                      season-dependent bias is present -> per-season
                      quantile mapping of daytime ERA5 GHI onto the POWER
                      distribution, fit and reported separately per season.
  3. MANUAL_REVIEW : correlation is low/uninterpretable -> do NOT blend or
                      average the two sources. Print diagnostic checks for
                      likely merge bugs and stop.

Fixed-weight blending (e.g. 0.6*ERA5 + 0.4*POWER) is explicitly rejected —
there is no derivation for a fixed weight — and is not implemented here.

HOW TO RUN:
  python 03b_agreement_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COMBINED_POINTS_FILE, PROCESSED_DIR, OUTPUTS_DIR, ensure_data_dirs

ensure_data_dirs()

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
EVENT_ORDER = ["sunrise", "noon", "sunset"]

VARIABLE_PAIRS = [
    ("era5_GHI", "power_ALLSKY_SFC_SW_DWN", "GHI"),
    ("era5_T_amb", "power_T2M", "T_amb"),
    ("era5_RHum", "power_RH2M", "RHum"),
    ("era5_W_spd", "power_WS10M", "W_spd"),
]

# --- decision-rule thresholds (documented, not tuned to a specific run) ---
CORR_GOOD = 0.90            # noon Pearson r at/above this -> agreement is "high"
CORR_SEVERE = 0.70          # noon Pearson r below this -> "low/uninterpretable"
MBE_SMALL_FRAC = 0.05       # |MBE| as a fraction of mean noon POWER GHI
SEASON_SPREAD_FRAC = 0.05   # (max-min season noon MBE) as a fraction of mean noon POWER GHI
N_QUANTILES = 100           # quantile-mapping resolution


# ═══════════════════════════════════════════════════════════
# LOAD  (usecols only — this file can be large; read-only)
# ═══════════════════════════════════════════════════════════

def load_combined():
    header = pd.read_csv(COMBINED_POINTS_FILE, nrows=0)
    available = set(header.columns)

    pairs = [(e, p, lbl) for e, p, lbl in VARIABLE_PAIRS if e in available and p in available]
    missing = [lbl for e, p, lbl in VARIABLE_PAIRS if (e, p, lbl) not in pairs]
    if missing:
        print(f"  [WARN] variable(s) not found in {COMBINED_POINTS_FILE.name}, skipping: {missing}")

    needed = {"season", "event"}
    for e, p, _ in pairs:
        needed.add(e)
        needed.add(p)
    if "era5_SZA" in available:
        needed.add("era5_SZA")

    usecols = [c for c in header.columns if c in needed]
    df = pd.read_csv(COMBINED_POINTS_FILE, usecols=usecols)
    df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
    df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
    return df, pairs


# ═══════════════════════════════════════════════════════════
# 1/2. STRATIFIED AGREEMENT STATS  (variable x season x event, + ALL rows)
# ═══════════════════════════════════════════════════════════

def compute_stats(era5_vals, power_vals):
    diff = era5_vals - power_vals
    n = len(diff)
    if n == 0:
        return {"n": 0, "MBE": np.nan, "RMSE": np.nan, "pearson_r": np.nan}
    mbe = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    corr = (float(era5_vals.corr(power_vals))
            if n > 1 and era5_vals.std() > 0 and power_vals.std() > 0 else np.nan)
    return {"n": n, "MBE": round(mbe, 4), "RMSE": round(rmse, 4),
            "pearson_r": round(corr, 4) if corr == corr else np.nan}


def build_agreement_table(df, pairs):
    rows = []
    for era5_col, power_col, label in pairs:
        sub = df[[era5_col, power_col, "season", "event"]].dropna(subset=[era5_col, power_col])
        n_rows_before = len(rows)

        rows.append({"variable": label, "season": "ALL", "event": "ALL",
                     **compute_stats(sub[era5_col], sub[power_col])})

        for season in SEASON_ORDER:
            s = sub[sub["season"] == season]
            rows.append({"variable": label, "season": season, "event": "ALL",
                         **compute_stats(s[era5_col], s[power_col])})

        for event in EVENT_ORDER:
            s = sub[sub["event"] == event]
            rows.append({"variable": label, "season": "ALL", "event": event,
                         **compute_stats(s[era5_col], s[power_col])})

        for season in SEASON_ORDER:
            for event in EVENT_ORDER:
                s = sub[(sub["season"] == season) & (sub["event"] == event)]
                rows.append({"variable": label, "season": season, "event": event,
                             **compute_stats(s[era5_col], s[power_col])})

        print(f"  [OK]   {label}: {len(sub):,} paired rows -> "
              f"{len(rows) - n_rows_before} stratified rows")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 3. GHI SCATTER — one panel per season, y=x identity line
# ═══════════════════════════════════════════════════════════

def build_ghi_scatter(df, pairs):
    ghi_pair = next(((e, p, lbl) for e, p, lbl in pairs if lbl == "GHI"), None)
    if ghi_pair is None:
        print("  [SKIP] GHI scatter — GHI columns not available")
        return
    era5_col, power_col, _ = ghi_pair

    sub = df[[era5_col, power_col, "season"]].dropna(subset=[era5_col, power_col])
    global_max = float(max(sub[era5_col].max(), sub[power_col].max())) if len(sub) else 1400.0

    fig = make_subplots(rows=2, cols=2, subplot_titles=SEASON_ORDER,
                         horizontal_spacing=0.08, vertical_spacing=0.12)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for season, (r, c) in zip(SEASON_ORDER, positions):
        s = sub[sub["season"] == season]
        if len(s) > 20_000:
            s = s.sample(20_000, random_state=42)
        fig.add_trace(
            go.Scattergl(x=s[power_col], y=s[era5_col], mode="markers",
                         marker=dict(size=3, opacity=0.25, color="#4c72b0"),
                         name=season, showlegend=False),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(x=[0, global_max], y=[0, global_max], mode="lines",
                       line=dict(color="black", dash="dash", width=1),
                       showlegend=False),
            row=r, col=c,
        )
        fig.update_xaxes(title_text="NASA POWER GHI (W/m²)", row=r, col=c)
        fig.update_yaxes(title_text="ERA5 GHI (W/m²)", row=r, col=c)

    fig.update_layout(
        title="ERA5 vs NASA POWER GHI by season — Rajasthan (dashed = y=x identity line)",
        height=850, width=950,
    )
    out = OUTPUTS_DIR / "qc_era5_power_scatter_rajasthan.html"
    fig.write_html(str(out))
    print(f"  [OK]   GHI scatter -> {out}")


# ═══════════════════════════════════════════════════════════
# QUANTILE MAPPING  (branch 2 only — GHI, daytime rows, per season)
# ═══════════════════════════════════════════════════════════

def fit_quantile_mapping(era5_vals, power_vals, n_quantiles=N_QUANTILES):
    """Empirical quantile mapping: era5 quantiles -> power quantiles.
    Returns a function that maps new ERA5 GHI values into POWER's
    distribution via monotonic interpolation on the quantile pairs."""
    qs = np.linspace(0, 1, n_quantiles + 1)
    era5_q = np.quantile(era5_vals, qs)
    power_q = np.quantile(power_vals, qs)
    era5_q_u, idx = np.unique(era5_q, return_index=True)   # np.interp needs strictly increasing xp
    power_q_u = power_q[idx]

    def mapper(x):
        return np.interp(x, era5_q_u, power_q_u, left=power_q_u[0], right=power_q_u[-1])

    return mapper


def apply_quantile_mapping(df, pairs):
    ghi_pair = next(((e, p, lbl) for e, p, lbl in pairs if lbl == "GHI"), None)
    if ghi_pair is None:
        return pd.DataFrame()
    era5_col, power_col, _ = ghi_pair

    results = []
    print("\n  Per-season quantile mapping (daytime ERA5 GHI > 0 only):")
    for season in SEASON_ORDER:
        mask = (df["season"] == season) & (df[era5_col] > 0) & df[era5_col].notna() & df[power_col].notna()
        paired = df.loc[mask]
        if len(paired) < 30:
            print(f"    [WARN] {season}: only {len(paired)} paired daytime GHI rows — mapping may be unstable")
        if len(paired) < 2:
            continue

        mapper = fit_quantile_mapping(paired[era5_col].values, paired[power_col].values)
        before = compute_stats(paired[era5_col], paired[power_col])
        corrected = pd.Series(mapper(paired[era5_col].values), index=paired.index)
        after = compute_stats(corrected, paired[power_col])

        results.append({
            "season": season, "n": before["n"],
            "MBE_before": before["MBE"], "RMSE_before": before["RMSE"], "r_before": before["pearson_r"],
            "MBE_after": after["MBE"], "RMSE_after": after["RMSE"], "r_after": after["pearson_r"],
        })
        print(f"    {season:<8s} n={before['n']:>7,}   "
              f"MBE {before['MBE']:>7.2f} -> {after['MBE']:>7.2f}   "
              f"RMSE {before['RMSE']:>7.2f} -> {after['RMSE']:>7.2f}   "
              f"r {before['pearson_r']:.3f} -> {after['pearson_r']:.3f}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════
# MERGE-BUG DIAGNOSTICS  (branch 3 only)
# ═══════════════════════════════════════════════════════════

def run_merge_bug_diagnostics(df, pairs):
    print("\n  Running merge-bug diagnostics ...")
    lines = []

    # --- units mismatch: compare median magnitude of each pair -------------
    for era5_col, power_col, label in pairs:
        sub = df[[era5_col, power_col]].dropna()
        if sub.empty:
            continue
        era5_med = float(sub[era5_col].median())
        power_med = float(sub[power_col].median())
        ratio = (era5_med / power_med) if power_med not in (0, np.nan) and power_med == power_med else np.nan
        msg = (f"  {label}: median ERA5={era5_med:.2f}  median POWER={power_med:.2f}"
               + (f"  ratio={ratio:.2f}" if ratio == ratio else "  ratio=undefined (POWER median ~0)"))
        print(msg); lines.append(msg)
        if ratio == ratio and (ratio > 3 or ratio < 1 / 3):
            flag = (f"    [FLAG] {label}: ERA5/POWER median ratio is far from 1 — check for a units "
                    f"mismatch (e.g. an unapplied scale_factor, or one side in different units).")
            print(flag); lines.append(flag)

    # --- accumulation-confusion check (GHI): a deaccumulation bug tends to
    # produce either near-zero GHI everywhere (missed 12h reset) or a heavy
    # excess of implausibly spiky values. -----------------------------------
    ghi_pair = next(((e, p, lbl) for e, p, lbl in pairs if lbl == "GHI"), None)
    if ghi_pair is not None:
        era5_col, _, _ = ghi_pair
        noon_ghi = df.loc[df["event"] == "noon", era5_col].dropna()
        if len(noon_ghi):
            frac_low = float((noon_ghi < 5).mean())
            msg = (f"  GHI: {100*frac_low:.1f}% of NOON-event ERA5 GHI readings are below 5 W/m² "
                   f"(near-zero at solar noon should be rare — a high fraction here suggests a "
                   f"deaccumulation bug, e.g. values landing on/near a 12h accumulation reset hour).")
            print(msg); lines.append(msg)

    # --- nearest-hour-match check via solar geometry ------------------------
    # climate_rajasthan_points.csv does not carry the actual matched ERA5/
    # POWER timestamp (only the requested time_utc — see 03_qc_plots.py's
    # note on this), so this is a proxy: noon-event SZA should be low and
    # sunrise/sunset SZA should sit near ~90 deg. A high noon SZA suggests
    # the nearest-hour match (MAX_MATCH_HOURS=3 in 02_combine_rajasthan.py)
    # is landing far from the true sun-event time.
    if "era5_SZA" in df.columns:
        sza_by_event = df.groupby("event", observed=True)["era5_SZA"].median()
        msg = f"  Median era5_SZA by event: {sza_by_event.round(1).to_dict()}"
        print(msg); lines.append(msg)
        noon_sza = sza_by_event.get("noon", np.nan)
        if noon_sza == noon_sza and noon_sza > 45:
            flag = (f"    [FLAG] median noon-event SZA is {noon_sza:.1f}° (expected well under 45° for "
                    f"most of Rajasthan's latitude range) — the nearest-hour match may be landing far "
                    f"from true solar noon; re-check suntimes.csv and MAX_MATCH_HOURS.")
            print(flag); lines.append(flag)

    lines.append("")
    lines.append("  These are structural checks, not a fix — if any [FLAG] fired above, inspect a "
                 "handful of raw NetCDF/JSON records by hand for the flagged point/date before "
                 "trusting either source further.")
    return lines


# ═══════════════════════════════════════════════════════════
# 4. DECISION ENGINE
# ═══════════════════════════════════════════════════════════

def decide_branch(agreement_df, df, pairs):
    if not any(lbl == "GHI" for _, _, lbl in pairs):
        print("  [SKIP] decision rule — GHI not available, cannot decide")
        return "SKIPPED", []

    noon_all = agreement_df[(agreement_df["variable"] == "GHI")
                             & (agreement_df["event"] == "noon")
                             & (agreement_df["season"] == "ALL")]
    r_noon = float(noon_all.iloc[0]["pearson_r"]) if len(noon_all) else np.nan
    mbe_noon = float(noon_all.iloc[0]["MBE"]) if len(noon_all) else np.nan
    n_noon = int(noon_all.iloc[0]["n"]) if len(noon_all) else 0

    power_col = next(p for e, p, lbl in VARIABLE_PAIRS if lbl == "GHI")
    mean_power_noon_ghi = float(df.loc[df["event"] == "noon", power_col].mean())

    season_noon = agreement_df[(agreement_df["variable"] == "GHI")
                                & (agreement_df["event"] == "noon")
                                & (agreement_df["season"] != "ALL")]
    season_mbes = season_noon["MBE"].dropna()
    season_spread = float(season_mbes.max() - season_mbes.min()) if len(season_mbes) else np.nan

    mbe_frac = abs(mbe_noon) / mean_power_noon_ghi if mean_power_noon_ghi else np.nan
    spread_frac = season_spread / mean_power_noon_ghi if mean_power_noon_ghi else np.nan

    print("\n  GHI, noon event, all seasons combined:")
    print(f"    n={n_noon:,}  MBE={mbe_noon:.2f} W/m² ({100*mbe_frac:.1f}% of mean POWER GHI)  "
          f"r={r_noon:.3f}")
    print(f"    Per-season noon MBE spread (max-min): {season_spread:.2f} W/m² "
          f"({100*spread_frac:.1f}% of mean POWER GHI)")
    if len(season_noon):
        print("    Per-season noon MBE:")
        for _, row in season_noon.iterrows():
            print(f"      {row['season']:<8s} MBE={row['MBE']:>7.2f}  RMSE={row['RMSE']:>7.2f}  r={row['pearson_r']}")

    qmap_results = pd.DataFrame()
    diagnostics = []

    if r_noon == r_noon and r_noon >= CORR_GOOD and mbe_frac == mbe_frac and mbe_frac <= MBE_SMALL_FRAC \
            and spread_frac == spread_frac and spread_frac <= SEASON_SPREAD_FRAC:
        branch = "BACKBONE"
        print(f"\n  DECISION: BACKBONE — ERA5 is the backbone, NASA POWER is the reported "
              f"cross-check, no correction applied.")

    elif r_noon == r_noon and r_noon >= CORR_SEVERE:
        branch = "QUANTILE_MAP"
        direction = "underestimating" if mbe_noon < 0 else "overestimating"
        print(f"\n  DECISION: QUANTILE_MAP — systematic, season-dependent bias detected "
              f"(ERA5 {direction} GHI on average). Applying per-season quantile mapping.")
        qmap_results = apply_quantile_mapping(df, pairs)

    else:
        branch = "MANUAL_REVIEW"
        print(f"\n  DECISION: MANUAL_REVIEW — noon GHI correlation ({r_noon}) is below the "
              f"{CORR_SEVERE} severe-disagreement threshold, or undefined. Not averaging or "
              f"blending the two sources. Running diagnostics instead.")
        diagnostics = run_merge_bug_diagnostics(df, pairs)

    decision_info = {
        "r_noon": r_noon, "mbe_noon": mbe_noon, "n_noon": n_noon,
        "mean_power_noon_ghi": mean_power_noon_ghi,
        "mbe_frac": mbe_frac, "season_spread": season_spread, "spread_frac": spread_frac,
        "season_noon": season_noon,
    }
    return branch, decision_info, qmap_results, diagnostics


# ═══════════════════════════════════════════════════════════
# DECISION TXT  (paste-ready prose for the methodology section)
# ═══════════════════════════════════════════════════════════

def write_decision_txt(branch, info, qmap_results, diagnostics):
    lines = []
    lines.append("RAJASTHAN — ERA5 vs NASA POWER CROSS-SOURCE AGREEMENT DECISION")
    lines.append("=" * 68)
    lines.append("Generated by 03b_agreement_analysis.py from "
                  "data/processed/climate_rajasthan_points.csv")
    lines.append("")
    lines.append("GHI, noon event, all seasons combined:")
    lines.append(f"  n = {info['n_noon']:,}")
    lines.append(f"  Mean Bias Error (ERA5 - POWER): {info['mbe_noon']:.2f} W/m^2 "
                  f"({100*info['mbe_frac']:.1f}% of mean POWER GHI)")
    lines.append(f"  Pearson r: {info['r_noon']}")
    lines.append(f"  Per-season noon MBE spread (max-min): {info['season_spread']:.2f} W/m^2 "
                  f"({100*info['spread_frac']:.1f}% of mean POWER GHI)")
    lines.append("")
    lines.append(f"DECISION: {branch}")
    lines.append("")

    if branch == "BACKBONE":
        lines.append(
            f"ERA5 and NASA POWER GHI show strong agreement at solar noon "
            f"(Pearson r = {info['r_noon']:.3f}, MBE = {info['mbe_noon']:.2f} W/m^2, "
            f"{100*info['mbe_frac']:.1f}% of mean daytime GHI), with no meaningful "
            f"season-to-season bias pattern (spread = {info['season_spread']:.2f} W/m^2, "
            f"{100*info['spread_frac']:.1f}% of mean). ERA5 is therefore used as the primary "
            f"climate backbone for this pipeline; NASA POWER is retained only as an "
            f"independent reported cross-check, with no bias correction applied to ERA5 "
            f"before downstream preprocessing."
        )

    elif branch == "QUANTILE_MAP":
        direction = "underestimating" if info["mbe_noon"] < 0 else "overestimating"
        n_rmse_improved = int((qmap_results["RMSE_after"] < qmap_results["RMSE_before"]).sum()) if len(qmap_results) else 0
        n_r_improved = int((qmap_results["r_after"] > qmap_results["r_before"]).sum()) if len(qmap_results) else 0
        n_seasons = len(qmap_results)
        lines.append(
            f"ERA5 and NASA POWER GHI show a systematic, season-dependent bias at solar noon "
            f"(Pearson r = {info['r_noon']:.3f}; per-season noon MBE spread = "
            f"{info['season_spread']:.2f} W/m^2, {100*info['spread_frac']:.1f}% of mean daytime "
            f"GHI) — consistent with ERA5 {direction} GHI in specific seasons (see the "
            f"per-season breakdown in era5_power_agreement_rajasthan.csv). Rather than average "
            f"or blend the two sources with an undocumented fixed weight, daytime ERA5 GHI is "
            f"empirically quantile-mapped onto the NASA POWER distribution, fit separately for "
            f"each of the four seasons. RMSE improved in {n_rmse_improved}/{n_seasons} seasons "
            f"and Pearson r improved in {n_r_improved}/{n_seasons} seasons after correction "
            f"(see the before/after table below)."
        )
        lines.append("")
        lines.append("Per-season quantile-mapping before/after:")
        if len(qmap_results):
            lines.append(qmap_results.to_string(index=False))

    elif branch == "MANUAL_REVIEW":
        lines.append(
            f"ERA5 and NASA POWER GHI show weak or uninterpretable agreement at solar noon "
            f"(Pearson r = {info['r_noon']}, below the {CORR_SEVERE} severe-disagreement "
            f"threshold), with no simple bias pattern that a season-wise correction could "
            f"safely remove. Rather than average, blend, or otherwise silently reconcile the "
            f"two sources, this run stops here and flags the mismatch for manual review. "
            f"Diagnostic checks below narrow down the likely cause (a unit mismatch, a "
            f"deaccumulation bug, or a nearest-hour match landing far from the true "
            f"sun-event time)."
        )
        lines.append("")
        lines.append("Diagnostics:")
        lines.extend(diagnostics)

    lines.append("")
    lines.append("Note: fixed-weight blending (e.g. 0.6*ERA5 + 0.4*POWER) was not considered as "
                 "an option in this decision tree — there is no principled derivation for a fixed "
                 "weight between two independent reanalysis/satellite-derived products.")

    out = OUTPUTS_DIR / "bias_decision_rajasthan.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK]   decision summary -> {out}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  CROSS-SOURCE AGREEMENT ANALYSIS — Rajasthan (ERA5 vs NASA POWER)")
    print(f"  Input  : {COMBINED_POINTS_FILE}")
    print(f"  Output : {PROCESSED_DIR}/era5_power_agreement_rajasthan.csv, "
          f"{OUTPUTS_DIR}/qc_era5_power_scatter_rajasthan.html, "
          f"{OUTPUTS_DIR}/bias_decision_rajasthan.txt")
    print("=" * 68)

    if not COMBINED_POINTS_FILE.exists():
        print(f"\n  ERROR: {COMBINED_POINTS_FILE} not found — run 02_combine_rajasthan.py first.")
        raise SystemExit(1)

    print("\n[1/4] Loading combined ERA5+POWER file (columns of interest only) ...")
    df, pairs = load_combined()
    print(f"  Loaded: {len(df):,} rows")

    print("\n[2/4] Stratified agreement stats (variable x season x event) ...")
    agreement_df = build_agreement_table(df, pairs)
    agreement_path = PROCESSED_DIR / "era5_power_agreement_rajasthan.csv"
    agreement_df.to_csv(agreement_path, index=False)
    print(f"  Saved: {agreement_path}  ({len(agreement_df)} rows)")

    print("\n[3/4] GHI scatter by season ...")
    build_ghi_scatter(df, pairs)

    print("\n[4/4] Decision rule ...")
    branch, info, qmap_results, diagnostics = decide_branch(agreement_df, df, pairs)
    write_decision_txt(branch, info, qmap_results, diagnostics)

    print("\n" + "=" * 68)
    print(f"  DONE — branch taken: {branch}")
    print("=" * 68)


if __name__ == "__main__":
    main()
