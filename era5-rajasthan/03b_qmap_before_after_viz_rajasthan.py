"""
03b_qmap_before_after_viz_rajasthan.py
=============================================================================
BEFORE/AFTER VISUALIZATION — ERA5 GHI QUANTILE-MAPPING CORRECTION (RAJASTHAN)
=============================================================================
03b_agreement_analysis.py's QUANTILE_MAP branch fits a per-season empirical
quantile mapping of daytime ERA5 GHI onto the NASA POWER GHI distribution
(fit_quantile_mapping() / apply_quantile_mapping() there) and reports its
before/after MBE/RMSE/r improvement to the console and to
outputs/bias_decision_rajasthan.txt.

*** That mapping is diagnostic-only: 03b never writes it back into
climate_rajasthan_points.csv or any file a later phase reads. Every
downstream script (03b_quality_check_rajasthan.py onward) still consumes
uncorrected ERA5 GHI. *** This script changes nothing about that — it
re-fits the identical mapping (same function, same N_QUANTILES=100, same
per-season/daytime-only fit domain) purely to visualize what applying it
would look like, side by side with the uncorrected scatter, so the effect
of a correction that currently only exists as printed numbers can actually
be seen. It writes only a standalone HTML plot.

GHI-ONLY: 03b_agreement_analysis.py's apply_quantile_mapping() only ever
looks up the GHI variable pair (see its ghi_pair lookup) — T_amb / RHum /
W_spd are compared for agreement in the same script but never quantile-
mapped. This script is GHI-only for the same reason. It does not add a
variable switcher, so as not to imply the other three variables received
the same treatment.

INPUT  : data/processed/climate_rajasthan_points.csv   (02_combine_rajasthan.py's
         raw, uncorrected output)
         data/processed/era5_power_agreement_rajasthan.csv (03b's stratified
         stats, for the noon-only decision-basis annotation)
         outputs/bias_decision_rajasthan.txt             (branch + per-season n)
OUTPUT : outputs/era5_power_scatter_before_after.html

HOW TO RUN:
  python 03b_qmap_before_after_viz_rajasthan.py
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COMBINED_POINTS_FILE, PROCESSED_DIR, OUTPUTS_DIR, ensure_data_dirs

ensure_data_dirs()

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]

# dataviz-skill categorical slots 1/2/3/7 (blue/orange/aqua/violet) — the
# skill's own palette notes flag slot 4 (yellow) as colliding with slot 2
# (orange) under the "all pairs visible at once" scatter case, so it's
# skipped here rather than used for a 4th season.
SEASON_COLOR = {
    "Winter": "#2a78d6",
    "Summer": "#eb6834",
    "Monsoon": "#1baf7a",
    "Retreat": "#4a3aa7",
}
INSUFFICIENT_COLOR = "#c3c2b7"   # dataviz-skill "muted" ink role

ERA5_GHI_COL = "era5_GHI"
POWER_GHI_COL = "power_ALLSKY_SFC_SW_DWN"
N_QUANTILES = 100   # must match 03b_agreement_analysis.py's N_QUANTILES exactly
MIN_N = 30          # 03b's own apply_quantile_mapping() n>=30 [WARN] gate
SAMPLE_CAP = 15_000  # per-season plotted-point cap (stats/trendline use full data, not the sample)

AGREEMENT_FILE = PROCESSED_DIR / "era5_power_agreement_rajasthan.csv"
DECISION_FILE = OUTPUTS_DIR / "bias_decision_rajasthan.txt"
OUT_FILE = OUTPUTS_DIR / "era5_power_scatter_before_after.html"


# ═══════════════════════════════════════════════════════════
# QUANTILE MAPPING — line-for-line copy of 03b_agreement_analysis.py's
# fit_quantile_mapping(). Duplicated rather than imported: 03b has no
# importable module surface (it's a __main__ script whose functions are
# only ever called from its own main()), and this file is explicitly a
# diagnostic re-fit, not a shared library dependency.
# ═══════════════════════════════════════════════════════════

def fit_quantile_mapping(era5_vals, power_vals, n_quantiles=N_QUANTILES):
    qs = np.linspace(0, 1, n_quantiles + 1)
    era5_q = np.quantile(era5_vals, qs)
    power_q = np.quantile(power_vals, qs)
    era5_q_u, idx = np.unique(era5_q, return_index=True)   # np.interp needs strictly increasing xp
    power_q_u = power_q[idx]

    def mapper(x):
        return np.interp(x, era5_q_u, power_q_u, left=power_q_u[0], right=power_q_u[-1])

    return mapper


def compute_stats(era5_vals, power_vals):
    era5_vals = np.asarray(era5_vals, dtype=float)
    power_vals = np.asarray(power_vals, dtype=float)
    diff = era5_vals - power_vals
    n = len(diff)
    if n == 0:
        return {"n": 0, "MBE": np.nan, "RMSE": np.nan, "pearson_r": np.nan}
    mbe = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    corr = (float(pd.Series(era5_vals).corr(pd.Series(power_vals)))
            if n > 1 and era5_vals.std() > 0 and power_vals.std() > 0 else np.nan)
    return {"n": n, "MBE": mbe, "RMSE": rmse, "pearson_r": corr}


def ols_trendline(x, y):
    a, b = np.polyfit(x, y, 1)
    x_line = np.array([float(np.min(x)), float(np.max(x))])
    return x_line, a * x_line + b


# ═══════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════

def load_daytime_rows():
    usecols = ["season", "event", ERA5_GHI_COL, POWER_GHI_COL]
    df = pd.read_csv(COMBINED_POINTS_FILE, usecols=usecols)
    df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
    # Same daytime mask as 03b_agreement_analysis.py's apply_quantile_mapping():
    # ERA5 GHI > 0, both sides present. Night/near-zero sunrise-sunset rows are
    # excluded from the fit domain there, so they're excluded here too.
    mask = (df[ERA5_GHI_COL] > 0) & df[ERA5_GHI_COL].notna() & df[POWER_GHI_COL].notna()
    return df.loc[mask].reset_index(drop=True)


def parse_decision_txt():
    """Pull the branch name and per-season n straight out of
    bias_decision_rajasthan.txt so the on-figure textbox quotes exactly what
    03b_agreement_analysis.py decided, rather than re-deriving it."""
    if not DECISION_FILE.exists():
        print(f"  [WARN] {DECISION_FILE} not found — branch/n textbox will be omitted")
        return "UNKNOWN", {}
    text = DECISION_FILE.read_text(encoding="utf-8")
    branch_match = re.search(r"^DECISION:\s*(\S+)", text, re.MULTILINE)
    branch = branch_match.group(1) if branch_match else "UNKNOWN"

    n_per_season = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in SEASON_ORDER:
            try:
                n_per_season[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return branch, n_per_season


def load_noon_decision_basis():
    """GHI/noon/ALL-seasons row from era5_power_agreement_rajasthan.csv — the
    actual stats 03b's decide_branch() used to pick QUANTILE_MAP, shown for
    context. Deliberately NOT reused as this figure's before/after numbers:
    it's noon-only + all events pooled differently, this figure's panels use
    the full daytime (GHI>0) fit domain instead."""
    if not AGREEMENT_FILE.exists():
        print(f"  [WARN] {AGREEMENT_FILE} not found — decision-basis annotation will be omitted")
        return None
    agreement_df = pd.read_csv(AGREEMENT_FILE)
    row = agreement_df[(agreement_df["variable"] == "GHI")
                        & (agreement_df["season"] == "ALL")
                        & (agreement_df["event"] == "noon")]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"n": int(r["n"]), "MBE": float(r["MBE"]), "RMSE": float(r["RMSE"]), "pearson_r": float(r["pearson_r"])}


# ═══════════════════════════════════════════════════════════
# APPLY MAPPING (per season, matching 03b's granularity exactly — the
# mapper is fit per season only, pooling all events within the daytime
# mask; there is no per-event mapping in 03b to reproduce)
# ═══════════════════════════════════════════════════════════

def build_before_after(df):
    df = df.copy()
    df["era5_GHI_after"] = df[ERA5_GHI_COL].astype(float)
    season_n = {}
    season_stats = {}

    for season in SEASON_ORDER:
        sub_mask = (df["season"] == season).values
        sub = df.loc[sub_mask]
        n = len(sub)
        season_n[season] = n

        if n < MIN_N:
            print(f"  [WARN] {season}: only {n} paired daytime GHI rows (< {MIN_N}) — "
                  f"mapping NOT applied, flagged insufficient n")
            season_stats[season] = {
                "before": compute_stats(sub[ERA5_GHI_COL].values, sub[POWER_GHI_COL].values),
                "after": None,
                "insufficient": True,
            }
            continue

        mapper = fit_quantile_mapping(sub[ERA5_GHI_COL].values, sub[POWER_GHI_COL].values)
        corrected = mapper(sub[ERA5_GHI_COL].values)
        df.loc[sub_mask, "era5_GHI_after"] = corrected

        before = compute_stats(sub[ERA5_GHI_COL].values, sub[POWER_GHI_COL].values)
        after = compute_stats(corrected, sub[POWER_GHI_COL].values)
        season_stats[season] = {"before": before, "after": after, "insufficient": False}
        print(f"  [OK]   {season}: n={n:,}  MBE {before['MBE']:>7.2f} -> {after['MBE']:>7.2f}   "
              f"RMSE {before['RMSE']:>7.2f} -> {after['RMSE']:>7.2f}   "
              f"r {before['pearson_r']:.3f} -> {after['pearson_r']:.3f}")

    return df, season_n, season_stats


# ═══════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════

def build_figure(df, season_n, season_stats, branch, decision_n_per_season, noon_basis):
    global_max = float(max(df[ERA5_GHI_COL].max(), df[POWER_GHI_COL].max(), df["era5_GHI_after"].max()))
    axis_range = [0, global_max * 1.02]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Before — raw ERA5 GHI vs NASA POWER GHI",
                         "After — quantile-mapped ERA5 GHI vs NASA POWER GHI"],
        horizontal_spacing=0.09,
    )

    pooled_before_x, pooled_before_y = [], []
    pooled_after_x, pooled_after_y = [], []

    for season in SEASON_ORDER:
        sub = df[df["season"] == season]
        n = len(sub)
        if n == 0:
            continue
        insufficient = season_stats[season]["insufficient"]
        color = INSUFFICIENT_COLOR if insufficient else SEASON_COLOR[season]
        opacity = 0.15 if insufficient else 0.35

        plot_sub = sub.sample(SAMPLE_CAP, random_state=42) if n > SAMPLE_CAP else sub
        legend_name = f"{season} (n={n:,})" + (" — insufficient n" if insufficient else "")

        fig.add_trace(
            go.Scattergl(x=plot_sub[POWER_GHI_COL], y=plot_sub[ERA5_GHI_COL],
                         mode="markers", marker=dict(size=3, opacity=opacity, color=color),
                         name=legend_name, legendgroup=season, showlegend=True),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scattergl(x=plot_sub[POWER_GHI_COL], y=plot_sub["era5_GHI_after"],
                         mode="markers", marker=dict(size=3, opacity=opacity, color=color),
                         name=legend_name, legendgroup=season, showlegend=False),
            row=1, col=2,
        )

        pooled_before_x.append(sub[POWER_GHI_COL].values)
        pooled_before_y.append(sub[ERA5_GHI_COL].values)
        pooled_after_x.append(sub[POWER_GHI_COL].values)
        pooled_after_y.append(sub["era5_GHI_after"].values)

        if insufficient:
            for col in (1, 2):
                suffix = "" if col == 1 else " (unmapped)"
                fig.add_annotation(
                    x=float(sub[POWER_GHI_COL].mean()), y=float(sub[ERA5_GHI_COL].mean()),
                    text=f"{season}: insufficient n ({n} < {MIN_N}){suffix}",
                    showarrow=True, arrowhead=1, font=dict(size=10, color="#52514e"),
                    row=1, col=col,
                )

    pooled_before_x = np.concatenate(pooled_before_x)
    pooled_before_y = np.concatenate(pooled_before_y)
    pooled_after_x = np.concatenate(pooled_after_x)
    pooled_after_y = np.concatenate(pooled_after_y)

    before_stats = compute_stats(pooled_before_y, pooled_before_x)
    after_stats = compute_stats(pooled_after_y, pooled_after_x)

    for col, (x, y, stats, label) in enumerate(
        [(pooled_before_x, pooled_before_y, before_stats, "before"),
         (pooled_after_x, pooled_after_y, after_stats, "after")], start=1
    ):
        fig.add_trace(
            go.Scatter(x=axis_range, y=axis_range, mode="lines",
                        line=dict(color="black", dash="dash", width=1),
                        name="1:1 line", showlegend=(col == 1)),
            row=1, col=col,
        )
        xl, yl = ols_trendline(x, y)
        fig.add_trace(
            go.Scatter(x=xl, y=yl, mode="lines", line=dict(color="#0b0b0b", width=2),
                        name="linear fit", showlegend=(col == 1)),
            row=1, col=col,
        )

        axis_suffix = "" if col == 1 else "2"
        fig.add_annotation(
            xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
            x=0.03, y=0.97, xanchor="left", yanchor="top", align="left", showarrow=False,
            text=(f"<b>{label} (recomputed on this panel's data)</b><br>"
                  f"n={stats['n']:,}<br>MBE={stats['MBE']:.2f} W/m²<br>"
                  f"RMSE={stats['RMSE']:.2f} W/m²<br>r={stats['pearson_r']:.4f}"),
            font=dict(size=11, color="#0b0b0b"),
            bgcolor="rgba(252,252,251,0.88)", bordercolor="#c3c2b7", borderwidth=1,
        )

    fig.update_xaxes(title_text="NASA POWER GHI (W/m²)", range=axis_range, row=1, col=1)
    fig.update_xaxes(title_text="NASA POWER GHI (W/m²)", range=axis_range, row=1, col=2)
    fig.update_yaxes(title_text="ERA5 GHI, raw (W/m²)", range=axis_range, row=1, col=1)
    fig.update_yaxes(title_text="ERA5 GHI, quantile-mapped (W/m²)", range=axis_range, row=1, col=2)

    decision_lines = [f"Branch decision (03b_agreement_analysis.py, from bias_decision_rajasthan.txt): <b>{branch}</b>"]
    if decision_n_per_season:
        parts = ", ".join(f"{s}: n={decision_n_per_season.get(s, 0):,}" for s in SEASON_ORDER
                           if s in decision_n_per_season)
        decision_lines.append(f"Per-season n (noon event, from the decision file): {parts}")
    if noon_basis is not None:
        decision_lines.append(
            f"Original decision basis (GHI, noon event, all seasons pooled): "
            f"n={noon_basis['n']:,}, MBE={noon_basis['MBE']:.2f} W/m², r={noon_basis['pearson_r']:.4f} "
            f"— NOT the same fit domain as the panels above (those use all daytime events, GHI&gt;0)."
        )
    decision_lines.append(
        "Diagnostic only — this quantile-mapping correction is <b>not</b> written back into "
        "climate_rajasthan_points.csv or read by any downstream script."
    )
    decision_lines.append(
        "GHI only — T_amb / RHum / W_spd are compared for agreement in 03b_agreement_analysis.py but never quantile-mapped; no variable switcher is offered here for that reason."
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=1.22, xanchor="center", yanchor="bottom",
        align="left", showarrow=False, text="<br>".join(decision_lines),
        font=dict(size=11.5, color="#33322f"),
        bgcolor="#f9f9f7", bordercolor="#c3c2b7", borderwidth=1, borderpad=8,
    )

    fig.update_layout(
        title=dict(text="ERA5 GHI quantile-mapping correction — before vs after (Rajasthan, daytime rows only)", y=0.99),
        height=800, width=1350,
        margin=dict(t=230),
        legend=dict(title="Season (gray = insufficient n, mapping skipped)", orientation="h",
                    yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return fig


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  BEFORE/AFTER VIZ — ERA5 GHI quantile-mapping correction (Rajasthan)")
    print(f"  Input  : {COMBINED_POINTS_FILE}")
    print(f"  Output : {OUT_FILE}")
    print("=" * 68)

    if not COMBINED_POINTS_FILE.exists():
        print(f"\n  ERROR: {COMBINED_POINTS_FILE} not found — run 02_combine_rajasthan.py first.")
        raise SystemExit(1)

    print("\n[1/4] Loading daytime (ERA5 GHI > 0) matched rows ...")
    df = load_daytime_rows()
    print(f"  Loaded: {len(df):,} daytime rows")

    print("\n[2/4] Re-fitting per-season quantile mapping (identical to 03b) ...")
    df, season_n, season_stats = build_before_after(df)

    print("\n[3/4] Reading branch decision + agreement stats for annotation ...")
    branch, decision_n_per_season = parse_decision_txt()
    noon_basis = load_noon_decision_basis()
    print(f"  Branch: {branch}")

    print("\n[4/4] Building before/after figure ...")
    fig = build_figure(df, season_n, season_stats, branch, decision_n_per_season, noon_basis)
    fig.write_html(str(OUT_FILE))
    print(f"  [OK]   before/after scatter -> {OUT_FILE}")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()


'''
Inference :
The correction did what quantile mapping is supposed to do, and only that.

MBE went from 13.37 → -0.01 W/m² — the systematic bias is essentially eliminated (as expected, since quantile mapping is fit specifically to align the marginal distributions per season).
RMSE only dropped ~7% (91.66 → 85.19). That's the tell: quantile mapping corrects the distribution shape, not the pairwise/timing error. A lot of the RMSE is coming from real physical mismatches at matched timestamps (cloud timing, local convective events) that a marginal-distribution correction can't touch.
r barely moved (0.9700 → 0.9737) — it was already high before correction, so there wasn't much room for a marginal-only fix to change it.

The residual scatter tells you where the disagreement actually lives.

Both panels still show visible spread away from the 1:1 line even after correction — most of it concentrated in the green (Monsoon) points, which fan out well below the line at mid-range GHI (roughly 400–800 W/m² NASA POWER, ERA5 reading noticeably lower). That's consistent with monsoon-season cloud representation being where the two reanalysis/satellite-derived products diverge most — a known ERA5-vs-POWER weak point during heavy cloud cover, not something a per-season quantile map can fully absorb since it corrects the aggregate seasonal distribution, not each individual cloudy day's timing.
The near-zero cluster at the origin (low-GHI/twilight-adjacent rows) is essentially unchanged before/after — makes sense, since quantile mapping preserves rank order and low-GHI values are already tightly clustered near zero in both products.

This directly supports the "diagnostic-only, not applied" decision flagged in the Phase 2 audit.
Now that you can see it visually: the correction is real and worth applying if you ever do persist it, but it's fixing bias, not dispersion. If Phase 3+ is currently consuming raw ERA5 GHI, the practical consequence is a small but non-zero systematic overestimate (~13 W/m² before correction) baked into every daily/seasonal aggregate downstream — worth quantifying whether that's "small relative to signal" as the audit doc's own suggested language hedges, or whether it's large enough at the seasonal level to matter for GHI_daily_kWh/kt_daily_mean and therefore Tm_target_capped_C. Given Tm_target_capped_C's formula depends on kt_worst_month/kt_daily_mean ratios rather than raw GHI, a systematic (near-constant-percentage) ERA5 bias might partially cancel in that ratio — but that's worth actually checking against the numbers rather than assuming.
'''