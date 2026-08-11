"""
03c_plots_raw_rajasthan.py
=============================================================================
RAW-DATA QC PLOTS — visual sanity checks on climate_rajasthan_points.csv
(02_combine_rajasthan.py's output), run BEFORE 03b_quality_check_
rajasthan.py, while pipeline bugs are still cheap to fix. Fills a real gap:
Rajasthan previously had no point-map/event-profile/missingness-heatmap/
seasonal/yearly visual QC at all — only the ERA5-vs-POWER agreement piece
(03b_agreement_analysis.py). Modeled on the Tamil Nadu pipeline's
03_plots_raw.py / 03b_interactive_raw_qa.py (same six checks), built in
Plotly/Folium here since that's this pipeline's established convention
(matplotlib is not installed in this project's venv — see 04_climate_
signature_rajasthan.py's own note on the same point) rather than
duplicating a static-matplotlib + interactive-Plotly pair the way Tamil
Nadu does.

READ-ONLY — never writes back to climate_rajasthan_points.csv.

REQUIRED LIBRARIES (install if missing):
  pip install pandas numpy plotly folium branca

  All of these are already used elsewhere in this pipeline (03_qc_plots.py,
  03b_agreement_analysis.py, 05_cluster_rajasthan.py) — if those scripts
  already run for you, you have everything this one needs too.

PLOTS PRODUCED (all interactive HTML, matching this pipeline's convention):
  A. Point map           — points colored by mean noon GHI (climate-signal
                            map; distinct from 03_qc_plots.py's population/
                            elevation/download-status maps).
  B. Event profile        — mean GHI/T_amb by sun-event (sunrise/noon/
                            sunset), with std error bars. Fastest way to
                            catch a timezone bug: noon must be the peak.
  C. Missing-data heatmap — % missing per point x variable, over the same
                            5 variables 03b_quality_check_rajasthan.py
                            checks (era5_T_amb, era5_RHum, era5_GHI,
                            era5_CSI, era5_W_spd). Expect this to render
                            uniformly ~0% for the current Rajasthan data
                            (03b_quality_check_rajasthan.py already found
                            0% missingness) — kept as permanent QC
                            infrastructure for future re-downloads/re-runs
                            and other states, not just this one.
  D. Seasonal boxplots    — noon GHI and T_amb by season, sanity-checked
                            against known Rajasthan climatology (hot dry
                            summer, weak/variable monsoon, mild winter).
  E. Yearly trend         — mean noon GHI/T_amb by year, 2016-2025 — spots
                            a step-change that would suggest a download or
                            unit problem in a specific year.

OUTPUTS: outputs/qc_raw_*.html (5 files)

HOW TO RUN:
  python 03c_plots_raw_rajasthan.py
  (Reads the ~1.5GB climate_rajasthan_points.csv with usecols only — still
  expect this to take a minute or two just to load.)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import COMBINED_POINTS_FILE, OUTPUTS_DIR, ensure_data_dirs

ensure_data_dirs()

EVENT_ORDER = ["sunrise", "noon", "sunset"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
QUALITY_VARS = ["era5_T_amb", "era5_RHum", "era5_GHI", "era5_CSI", "era5_W_spd"]   # matches 03b_quality_check_rajasthan.py's scope

print("=" * 68)
print("  RAW DATA QC PLOTS — Rajasthan Population Points")
print(f"  Input  : {COMBINED_POINTS_FILE}")
print(f"  Output : {OUTPUTS_DIR}/")
print("=" * 68)

print("\nLoading data (usecols only — this is still a large file) ...")
usecols = ["point_id", "lat", "lon", "population", "date", "event", "season", "year"] + QUALITY_VARS
df = pd.read_csv(COMBINED_POINTS_FILE, usecols=usecols, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}  |  "
      f"Years: {df['year'].min()}-{df['year'].max()}")


# ═══════════════════════════════════════════════════════════
# A. POINT MAP — colored by mean noon GHI
# ═══════════════════════════════════════════════════════════
print("\n[A] Point map (mean noon GHI) ...")

noon = df[df["event"] == "noon"]
point_ghi = noon.groupby("point_id", observed=True).agg(
    lat=("lat", "first"), lon=("lon", "first"), ghi_mean=("era5_GHI", "mean"),
).reset_index()

fig = px.scatter(
    point_ghi, x="lon", y="lat", color="ghi_mean", hover_name="point_id",
    color_continuous_scale="YlOrRd",
    title=f"Rajasthan — {len(point_ghi)} Points, Colored by Mean Noon GHI (W/m^2)",
    labels={"ghi_mean": "Mean noon GHI (W/m^2)", "lon": "Longitude", "lat": "Latitude"},
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
out = OUTPUTS_DIR / "qc_raw_point_map_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}  (GHI range: {point_ghi['ghi_mean'].min():.0f} - {point_ghi['ghi_mean'].max():.0f} W/m^2)")


# ═══════════════════════════════════════════════════════════
# B. EVENT PROFILE — timezone / sun-event sanity check
# ═══════════════════════════════════════════════════════════
print("\n[B] Event profile (sunrise/noon/sunset) ...")

event_means = df.groupby("event", observed=True)[["era5_GHI", "era5_T_amb"]].mean()
event_stds = df.groupby("event", observed=True)[["era5_GHI", "era5_T_amb"]].std()
peak_event_ghi = event_means["era5_GHI"].idxmax()
print(f"  Peak GHI at event: {peak_event_ghi}  "
      f"({'OK — noon peaks as expected' if peak_event_ghi == 'noon' else 'CHECK — expected noon'})")

fig = make_subplots(rows=1, cols=2, subplot_titles=["GHI (W/m^2)", "T_amb (degC)"])
for col_idx, (col, label) in enumerate([("era5_GHI", "GHI"), ("era5_T_amb", "T_amb")], start=1):
    fig.add_trace(go.Bar(x=event_means.index.astype(str), y=event_means[col],
                          error_y=dict(type="data", array=event_stds[col]),
                          name=label, showlegend=False),
                  row=1, col=col_idx)
fig.update_layout(title="Mean GHI / T_amb by Sun Event (+/- 1 std) — noon must peak for GHI")
out = OUTPUTS_DIR / "qc_raw_event_profile_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════
# C. MISSING DATA HEATMAP
# ═══════════════════════════════════════════════════════════
print("\n[C] Missing data heatmap ...")

miss_by_point = df.groupby("point_id", observed=True)[QUALITY_VARS].apply(lambda g: g.isna().mean() * 100)

fig = go.Figure(data=go.Heatmap(
    z=miss_by_point.values, x=QUALITY_VARS, y=miss_by_point.index.tolist(),
    colorscale="Reds", zmin=0, zmax=max(5, float(miss_by_point.values.max())),
    colorbar=dict(title="% missing"),
))
fig.update_layout(title="% Missing Data — per Point x Variable", height=max(500, 6 * len(miss_by_point)))
out = OUTPUTS_DIR / "qc_raw_missing_heatmap_rajasthan.html"
fig.write_html(str(out))
overall_missing = df[QUALITY_VARS].isna().mean() * 100
print(f"  Overall % missing: {overall_missing.round(3).to_dict()}")
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════
# D. SEASONAL BOXPLOTS
# ═══════════════════════════════════════════════════════════
print("\n[D] Seasonal boxplots ...")

fig = make_subplots(rows=1, cols=2, subplot_titles=["Noon GHI by Season", "Noon T_amb by Season"])
for col_idx, col in enumerate(["era5_GHI", "era5_T_amb"], start=1):
    for season in SEASON_ORDER:
        sub = noon.loc[noon["season"] == season, col]
        fig.add_trace(go.Box(y=sub, name=season, showlegend=False), row=1, col=col_idx)
fig.update_layout(title="Noon GHI / T_amb by Season (all points, all years)")
out = OUTPUTS_DIR / "qc_raw_seasonal_boxplots_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════
# E. MULTI-YEAR TREND
# ═══════════════════════════════════════════════════════════
print("\n[E] Multi-year trend (discontinuity check) ...")

yearly = noon.groupby("year", observed=True)[["era5_GHI", "era5_T_amb"]].mean()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=["Mean noon GHI by year", "Mean noon T_amb by year"])
fig.add_trace(go.Scatter(x=yearly.index, y=yearly["era5_GHI"], mode="lines+markers", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=yearly.index, y=yearly["era5_T_amb"], mode="lines+markers", showlegend=False), row=2, col=1)
fig.update_layout(title="Year-by-Year Mean (noon event, all points) — should be gently varying, not a step-change")
out = OUTPUTS_DIR / "qc_raw_yearly_trend_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")
print(yearly.round(2).to_string())

print("\n" + "=" * 68)
print("  DONE — inspect the HTML files in", OUTPUTS_DIR)
print("  If B shows noon isn't the peak, or F shows a step-change in one year,")
print("  resolve that BEFORE trusting 03b_quality_check_rajasthan.py's output.")
print("=" * 68)
