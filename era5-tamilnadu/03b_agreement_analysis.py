"""
03b_agreement_analysis.py
=============================================================================
CROSS-SOURCE VALIDATION — TAMIL NADU POPULATION POINTS  (ERA5 vs NASA POWER)
=============================================================================
Decides whether ERA5 alone is a defensible backbone for downstream
preprocessing, or needs bias correction against NASA POWER, before any
preprocessing step touches the physical values. Read-only with respect to
climate_tamilnadu_points.csv — this script never writes back to it.

INPUT  : data/processed/climate_tamilnadu_points.csv  (02_combine output)
OUTPUTS:
  data/processed/era5_power_agreement_tamilnadu.csv
  outputs/qc_era5_power_scatter_tamilnadu.html
  outputs/bias_decision_tamilnadu.txt

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
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
EVENT_ORDER = ["sunrise", "noon", "sunset"]

VARIABLE_PAIRS = [
    ("era5_GHI", "power_ALLSKY_SFC_SW_DWN", "GHI"),
    ("era5_T_amb", "power_T2M", "T_amb"),
    ("era5_RHum", "power_RH2M", "RHum"),
    ("era5_W_spd", "power_WS10M", "W_spd"),
]

CORR_GOOD = 0.90
CORR_SEVERE = 0.70
MBE_SMALL_FRAC = 0.05
SEASON_SPREAD_FRAC = 0.05
N_QUANTILES = 100


def load_combined():
    header = pd.read_csv(COMBINED_POINTS_FILE, nrows=0)
    available = set(header.columns)
    pairs = [(e, p, lbl) for e, p, lbl in VARIABLE_PAIRS if e in available and p in available]
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
    return pd.DataFrame(rows)


def build_ghi_scatter(df, pairs):
    ghi_pair = next(((e, p, lbl) for e, p, lbl in pairs if lbl == "GHI"), None)
    if ghi_pair is None:
        return
    era5_col, power_col, _ = ghi_pair
    sub = df[[era5_col, power_col, "season"]].dropna(subset=[era5_col, power_col])
    global_max = float(max(sub[era5_col].max(), sub[power_col].max())) if len(sub) else 1400.0
    fig = make_subplots(rows=2, cols=2, subplot_titles=SEASON_ORDER)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for season, (r, c) in zip(SEASON_ORDER, positions):
        s = sub[sub["season"] == season]
        if len(s) > 20_000:
            s = s.sample(20_000, random_state=42)
        fig.add_trace(go.Scattergl(x=s[power_col], y=s[era5_col], mode="markers",
                                   marker=dict(size=3, opacity=0.25)), row=r, col=c)
        fig.add_trace(go.Scatter(x=[0, global_max], y=[0, global_max], mode="lines",
                                 line=dict(color="black", dash="dash", width=1),
                                 showlegend=False), row=r, col=c)
    fig.update_layout(title="ERA5 vs NASA POWER GHI by season — Tamil Nadu", height=850, width=950)
    out = OUTPUTS_DIR / "qc_era5_power_scatter_tamilnadu.html"
    fig.write_html(str(out))
    print(f"  [OK]   GHI scatter -> {out}")


def fit_quantile_mapping(era5_vals, power_vals, n_quantiles=N_QUANTILES):
    qs = np.linspace(0, 1, n_quantiles + 1)
    era5_q = np.quantile(era5_vals, qs)
    power_q = np.quantile(power_vals, qs)
    era5_q_u, idx = np.unique(era5_q, return_index=True)
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
    for season in SEASON_ORDER:
        mask = (df["season"] == season) & (df[era5_col] > 0) & df[era5_col].notna() & df[power_col].notna()
        paired = df.loc[mask]
        if len(paired) < 2:
            continue
        mapper = fit_quantile_mapping(paired[era5_col].values, paired[power_col].values)
        before = compute_stats(paired[era5_col], paired[power_col])
        corrected = pd.Series(mapper(paired[era5_col].values), index=paired.index)
        after = compute_stats(corrected, paired[power_col])
        results.append({"season": season, "n": before["n"],
                        "MBE_before": before["MBE"], "RMSE_before": before["RMSE"], "r_before": before["pearson_r"],
                        "MBE_after": after["MBE"], "RMSE_after": after["RMSE"], "r_after": after["pearson_r"]})
    return pd.DataFrame(results)


def run_merge_bug_diagnostics(df, pairs):
    lines = []
    ghi_pair = next(((e, p, lbl) for e, p, lbl in pairs if lbl == "GHI"), None)
    if ghi_pair is not None:
        era5_col, _, _ = ghi_pair
        noon_ghi = df.loc[df["event"] == "noon", era5_col].dropna()
        if len(noon_ghi):
            frac_low = float((noon_ghi < 5).mean())
            lines.append(f"  GHI: {100*frac_low:.1f}% of NOON ERA5 GHI below 5 W/m² "
                         f"(high fraction suggests deaccumulation bug)")
    if "era5_SZA" in df.columns:
        sza_by_event = df.groupby("event", observed=True)["era5_SZA"].median()
        lines.append(f"  Median era5_SZA by event: {sza_by_event.round(1).to_dict()}")
    return lines


def decide_branch(agreement_df, df, pairs):
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

    print(f"\n  GHI noon: n={n_noon:,}  MBE={mbe_noon:.2f} W/m²  r={r_noon:.3f}")

    qmap_results = pd.DataFrame()
    diagnostics = []
    if r_noon == r_noon and r_noon >= CORR_GOOD and mbe_frac == mbe_frac and mbe_frac <= MBE_SMALL_FRAC \
            and spread_frac == spread_frac and spread_frac <= SEASON_SPREAD_FRAC:
        branch = "BACKBONE"
    elif r_noon == r_noon and r_noon >= CORR_SEVERE:
        branch = "QUANTILE_MAP"
        qmap_results = apply_quantile_mapping(df, pairs)
    else:
        branch = "MANUAL_REVIEW"
        diagnostics = run_merge_bug_diagnostics(df, pairs)

    info = {"r_noon": r_noon, "mbe_noon": mbe_noon, "n_noon": n_noon,
            "mean_power_noon_ghi": mean_power_noon_ghi,
            "mbe_frac": mbe_frac, "season_spread": season_spread, "spread_frac": spread_frac}
    return branch, info, qmap_results, diagnostics


def write_decision_txt(branch, info, qmap_results, diagnostics):
    lines = [
        "TAMIL NADU — ERA5 vs NASA POWER CROSS-SOURCE AGREEMENT DECISION",
        "=" * 68,
        f"GHI noon: n={info['n_noon']:,}  MBE={info['mbe_noon']:.2f} W/m²  r={info['r_noon']}",
        f"DECISION: {branch}",
        "",
    ]
    if branch == "QUANTILE_MAP" and len(qmap_results):
        lines.append("Per-season quantile-mapping before/after:")
        lines.append(qmap_results.to_string(index=False))
    elif branch == "MANUAL_REVIEW":
        lines.append("Diagnostics:")
        lines.extend(diagnostics)
    out = OUTPUTS_DIR / "bias_decision_tamilnadu.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK]   decision summary -> {out}")


def main():
    print("=" * 68)
    print("  CROSS-SOURCE AGREEMENT ANALYSIS — Tamil Nadu (ERA5 vs NASA POWER)")
    if not COMBINED_POINTS_FILE.exists():
        raise SystemExit(f"ERROR: {COMBINED_POINTS_FILE} not found — run 02_combine_tamilnadu.py first.")
    df, pairs = load_combined()
    agreement_df = build_agreement_table(df, pairs)
    agreement_path = PROCESSED_DIR / "era5_power_agreement_tamilnadu.csv"
    agreement_df.to_csv(agreement_path, index=False)
    build_ghi_scatter(df, pairs)
    branch, info, qmap_results, diagnostics = decide_branch(agreement_df, df, pairs)
    write_decision_txt(branch, info, qmap_results, diagnostics)
    print(f"\n  DONE — branch taken: {branch}")


if __name__ == "__main__":
    main()
