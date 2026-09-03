"""
03b_interactive_raw_qa.py
============================
Interactive (Plotly + Folium) version of 03_plots_raw.py's checks — same
data (climate_tamilnadu_points.csv), same six checks (A-F), but as
zoomable/hoverable HTML instead of static PNGs. Nothing here writes back
to the data; purely a read-only QA pass, same as 03.

Requires: pip install plotly folium branca

INPUT  : data/processed/climate_tamilnadu_points.csv
OUTPUT : data/plots/raw_interactive/*.html

HOW TO RUN:
  python 03b_interactive_raw_qa.py
Then open the .html files in a browser (they're standalone — plotly.js is
pulled from CDN, so you need internet the first time each file is opened;
folium map embeds its own tiles/JS and works offline once loaded).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import branca.colormap as cm

from config import COMBINED_POINTS_FILE, PLOTS_DIR

OUT_DIR = PLOTS_DIR / "raw_interactive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ORDER = ["sunrise", "noon", "sunset"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]

print("=" * 68)
print("  Interactive Raw Data QA — Tamil Nadu Population Points")
print(f"  Input  : {COMBINED_POINTS_FILE}")
print(f"  Output : {OUT_DIR}/")
print("=" * 68)

print("\nLoading data ...")
df = pd.read_csv(COMBINED_POINTS_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}")

# ═══════════════════════════════════════════════════════════
# A. FOLIUM POINT MAP
# ═══════════════════════════════════════════════════════════
print("\n[A] Interactive point map (Folium) ...")

point_meta = df.groupby("point_id").agg(
    lat=("lat", "first"), lon=("lon", "first"), population=("population", "first")
).reset_index()

fmap = folium.Map(location=[point_meta["lat"].mean(), point_meta["lon"].mean()],
                   zoom_start=7, tiles="CartoDB positron")
colormap = cm.LinearColormap(
    colors=["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    vmin=point_meta["population"].min(), vmax=point_meta["population"].max(),
    caption="Population (2020, WorldPop)")
for r in point_meta.itertuples():
    folium.CircleMarker(
        location=[r.lat, r.lon],
        radius=4 + 10 * (r.population / point_meta["population"].max()),
        color=colormap(r.population), weight=1, fill=True, fill_opacity=0.75,
        popup=folium.Popup(f"<b>{r.point_id}</b><br>pop: {r.population:,.0f}"
                            f"<br>lat/lon: {r.lat:.3f}, {r.lon:.3f}", max_width=220),
    ).add_to(fmap)
colormap.add_to(fmap)
fmap.save(str(OUT_DIR / "A_point_map.html"))
print(f"  Saved: A_point_map.html  ({len(point_meta)} points)")

# ═══════════════════════════════════════════════════════════
# B. EVENT PROFILE
# ═══════════════════════════════════════════════════════════
print("\n[B] Event profile (Plotly) ...")

fig = make_subplots(rows=1, cols=2, subplot_titles=("GHI (W/m²)", "T_amb (°C)"))
for i, col in enumerate(["era5_GHI", "era5_T_amb"], start=1):
    means = df.groupby("event", observed=True)[col].mean()
    stds = df.groupby("event", observed=True)[col].std()
    fig.add_trace(go.Bar(x=means.index.astype(str), y=means.values,
                          error_y=dict(type="data", array=stds.values),
                          marker_color=["#f9c74f", "#f3722c", "#577590"], showlegend=False),
                  row=1, col=i)
fig.update_layout(title="Mean GHI / T_amb by Sun Event (± 1 std)", height=450)
fig.write_html(str(OUT_DIR / "B_event_profile.html"), include_plotlyjs="cdn")
peak_event = df.groupby("event", observed=True)["era5_GHI"].mean().idxmax()
print(f"  Peak GHI event: {peak_event}  "
      f"({'OK' if peak_event == 'noon' else 'CHECK — expected noon'})")
print("  Saved: B_event_profile.html")

# ═══════════════════════════════════════════════════════════
# C. ERA5 VS NASA POWER
# ═══════════════════════════════════════════════════════════
print("\n[C] ERA5 vs NASA POWER agreement (Plotly) ...")

pairs = [
    ("era5_GHI", "power_ALLSKY_SFC_SW_DWN", "GHI (W/m²)"),
    ("era5_GHI_clearsky", "power_CLRSKY_SFC_SW_DWN", "Clear-sky GHI (W/m²)"),
    ("era5_T_amb", "power_T2M", "T_amb (°C)"),
    ("era5_RHum", "power_RH2M", "RHum (%)"),
    ("era5_W_spd", "power_WS10M", "Wind speed (m/s)"),
]
fig = make_subplots(rows=2, cols=3, subplot_titles=[p[2] for p in pairs])
mbe_rows = []
for i, (era5_col, power_col, label) in enumerate(pairs):
    r, c = divmod(i, 3)
    sub = df[[era5_col, power_col]].dropna()
    if sub.empty:
        continue
    sample = sub.sample(min(15_000, len(sub)), random_state=42)
    diff = sub[era5_col] - sub[power_col]
    mbe, rmse = diff.mean(), np.sqrt((diff ** 2).mean())
    corr = sub[era5_col].corr(sub[power_col])
    mbe_rows.append({"variable": label, "n": len(sub), "MBE": round(mbe, 3),
                      "RMSE": round(rmse, 3), "pearson_r": round(corr, 4)})
    fig.add_trace(go.Scattergl(x=sample[power_col], y=sample[era5_col], mode="markers",
                                marker=dict(size=3, opacity=0.15, color="#4c72b0"),
                                showlegend=False,
                                name=label), row=r + 1, col=c + 1)
    lims = [sample[[power_col, era5_col]].min().min(), sample[[power_col, era5_col]].max().max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="black"),
                              showlegend=False), row=r + 1, col=c + 1)
fig.update_layout(title="ERA5 (y) vs NASA POWER (x) — hover to zoom into any point", height=700)
fig.write_html(str(OUT_DIR / "C_era5_vs_power.html"), include_plotlyjs="cdn")
mbe_df = pd.DataFrame(mbe_rows)
mbe_df.to_csv(OUT_DIR / "C_era5_vs_power_stats.csv", index=False)
print(mbe_df.to_string(index=False))
print("  Saved: C_era5_vs_power.html, C_era5_vs_power_stats.csv")

# ═══════════════════════════════════════════════════════════
# D. MISSING-DATA HEATMAP
# ═══════════════════════════════════════════════════════════
print("\n[D] Missing-data heatmap (Plotly) ...")

check_cols = ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
              "era5_precipitation", "power_ALLSKY_SFC_SW_DWN", "power_T2M"]
check_cols = [c for c in check_cols if c in df.columns]
miss_by_point = df.groupby("point_id")[check_cols].apply(lambda g: g.isna().mean() * 100)
fig = px.imshow(miss_by_point, aspect="auto", color_continuous_scale="Reds",
                 labels=dict(color="% missing"),
                 title="% Missing Data — per Point x Variable (hover for exact value)")
fig.update_layout(height=1400)
fig.write_html(str(OUT_DIR / "D_missing_heatmap.html"), include_plotlyjs="cdn")
print(f"  Overall max % missing: {miss_by_point.values.max():.3f}")
print("  Saved: D_missing_heatmap.html")

# ═══════════════════════════════════════════════════════════
# E. SEASONAL BOXPLOTS
# ═══════════════════════════════════════════════════════════
print("\n[E] Seasonal boxplots (Plotly) ...")

noon = df[df["event"] == "noon"]
fig = make_subplots(rows=1, cols=2, subplot_titles=("Noon GHI by Season", "Noon T_amb by Season"))
fig.add_trace(go.Box(x=noon["season"], y=noon["era5_GHI"], showlegend=False,
                      marker_color="#f3722c"), row=1, col=1)
fig.add_trace(go.Box(x=noon["season"], y=noon["era5_T_amb"], showlegend=False,
                      marker_color="#4cc9f0"), row=1, col=2)
fig.update_layout(height=500)
fig.write_html(str(OUT_DIR / "E_seasonal_boxplots.html"), include_plotlyjs="cdn")
print("  Saved: E_seasonal_boxplots.html")

# ═══════════════════════════════════════════════════════════
# F. MULTI-YEAR TREND
# ═══════════════════════════════════════════════════════════
print("\n[F] Multi-year trend (Plotly) ...")

yearly = noon.groupby("year")[["era5_GHI", "era5_T_amb"]].mean().reset_index()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=("Mean noon GHI (W/m²)", "Mean noon T_amb (°C)"))
fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["era5_GHI"], mode="lines+markers",
                          line=dict(color="#f3722c"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["era5_T_amb"], mode="lines+markers",
                          line=dict(color="#4cc9f0"), showlegend=False), row=2, col=1)
fig.update_layout(title="Year-by-Year Mean (noon event) — should vary gently, no step-change", height=600)
fig.write_html(str(OUT_DIR / "F_yearly_trend.html"), include_plotlyjs="cdn")
print("  Saved: F_yearly_trend.html")

print("\n" + "=" * 68)
print("  DONE — open the .html files in", OUT_DIR)
print("=" * 68)
