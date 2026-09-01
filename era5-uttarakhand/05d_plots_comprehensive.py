"""
05d_plots_comprehensive.py
============================
ERA5 Uttarakhand — Comprehensive Visualization Batch
(the point/event-schema equivalent of the original state-wide
notebook-based plotting pipeline)

Delivered as a plain script rather than a notebook on purpose — it runs
identically via `python 05d_plots_comprehensive.py` or cell-by-cell if you
paste sections into Jupyter; a script is easier to diff, version, and
re-run unattended than a notebook, and every section below is still
clearly delimited so pasting into cells is a copy-paste job if you want
the notebook experience back.

WHAT'S DIFFERENT FROM THE OLD NOTEBOOK
-----------------------------------------
- No `city` / `district` / `climate_zone` columns — those belonged to the
  old 260+ named-city dict. This schema has `point_id` + `lat`/`lon`, so
  every map/plot below groups and colors by point_id or by numeric climate
  signature values, not by administrative zone.
- No continuous hourly series — every "daily"/"diurnal" plot here is
  built from the 3 sun-events/day (sunrise/noon/sunset) actually sampled.
- Reads the PROCESSED file (04's output) by default so plots reflect the
  QC'd backbone, not raw data with its outliers/gaps still in it — flip
  USE_PROCESSED to False if you want raw-data plots instead.

HOW TO RUN
----------
  pip install folium branca matplotlib seaborn plotly
  python 05d_plots_comprehensive.py

OUTPUT
------
  data/plots/comprehensive/
    maps/           — standalone Folium HTML (open directly in a browser)
    timeseries/      — matplotlib PNG
    statistics/       — matplotlib PNG
    solar_resource/   — matplotlib PNG
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm

from config import COMBINED_POINTS_FILE, PREPROCESSED_DIR, PLOTS_DIR

USE_PROCESSED = True   # False -> plots raw climate_uttarakhand_points.csv instead

PLOT_DIR = PLOTS_DIR / "comprehensive"
for sub in ["maps", "timeseries", "statistics", "solar_resource"]:
    (PLOT_DIR / sub).mkdir(parents=True, exist_ok=True)

DATA_FILE = (PREPROCESSED_DIR / "uttarakhand_cleaned_physical.csv") if USE_PROCESSED else COMBINED_POINTS_FILE
DATA_LABEL = "processed" if USE_PROCESSED else "raw"

EVENT_ORDER = ["sunrise", "noon", "sunset"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
SEASON_COLORS = {"Winter": "#4cc9f0", "Summer": "#f9c74f",
                  "Monsoon": "#06d6a0", "Retreat": "#f3722c"}
EVENT_COLORS = {"sunrise": "#f9c74f", "noon": "#4cc9f0", "sunset": "#f3722c"}
TN_CENTER = [10.9, 78.5]

print("=" * 68)
print("  ERA5 Uttarakhand — Comprehensive Visualization Batch")
print(f"  Input  : {DATA_FILE}  ({DATA_LABEL})")
print(f"  Output : {PLOT_DIR}/")
print("=" * 68)

if not os.path.exists(DATA_FILE):
    raise SystemExit(f"\nERROR: {DATA_FILE} not found — run the earlier pipeline steps first.")

print("\nLoading data ...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}  |  "
      f"Years: {df['year'].min()}-{df['year'].max()}")

# ── Per-point summary (used by every map below) ─────────────
noon_only = df[df["event"] == "noon"]
point_summary = df.groupby("point_id").agg(
    lat=("lat", "first"), lon=("lon", "first"),
    population=("population", "first"),
    T_amb_mean=("era5_T_amb", "mean"),
    RHum_mean=("era5_RHum", "mean"),
).reset_index()
noon_ghi = noon_only.groupby("point_id")["era5_GHI"].mean().rename("GHI_noon_mean")
point_summary = point_summary.merge(noon_ghi, on="point_id", how="left")
print(f"  Point summary built: {len(point_summary)} points")


def save_and_report(fmap, filepath, label):
    fmap.save(str(filepath))
    print(f"    Saved: {filepath}")


# ═══════════════════════════════════════════════════════════
# A. FOLIUM INTERACTIVE MAPS
# ═══════════════════════════════════════════════════════════
print("\n[A] Folium interactive maps ...")

# ── A0. All points overview, clustered ──────────────────────
print("  A0. All points overview map ...")
m0 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")
cluster = MarkerCluster(name=f"All {len(point_summary)} points").add_to(m0)
for _, row in point_summary.iterrows():
    popup_html = (
        f"<div style='font-family:Arial;font-size:13px;width:220px;'>"
        f"<b style='font-size:14px;'>{row['point_id']}</b><br>"
        f"<hr style='margin:3px 0'>"
        f"Population   : {row['population']:,.0f}<br>"
        f"Noon GHI mean: <b>{row['GHI_noon_mean']:.1f} W/m²</b><br>"
        f"T_amb mean   : {row['T_amb_mean']:.1f} °C<br>"
        f"RHum mean    : {row['RHum_mean']:.1f} %<br>"
        f"Lat/Lon      : {row['lat']:.3f}°N, {row['lon']:.3f}°E</div>"
    )
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=row["point_id"],
    ).add_to(cluster)
plain_layer = folium.FeatureGroup(name="Individual markers (no cluster)", show=False)
for _, row in point_summary.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=5,
        color="white", weight=0.8, fill=True,
        fill_color="#4cc9f0", fill_opacity=0.85, tooltip=row["point_id"],
    ).add_to(plain_layer)
plain_layer.add_to(m0)
folium.LayerControl().add_to(m0)
save_and_report(m0, PLOT_DIR / "maps" / "A0_all_points_overview.html", "A0")

# ── A1. GHI mean map + heatmap overlay ──────────────────────
print("  A1. GHI mean spatial map ...")
m1 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")
ghi_min, ghi_max = point_summary["GHI_noon_mean"].min(), point_summary["GHI_noon_mean"].max()
colormap_ghi = cm.LinearColormap(
    ["#2d6a4f", "#52b788", "#d9ed92", "#f9c74f", "#f3722c"],
    vmin=ghi_min, vmax=max(ghi_max, ghi_min + 1e-6), caption="Noon GHI mean (W/m²)")
colormap_ghi.add_to(m1)
for _, row in point_summary.iterrows():
    radius = 6 + (row["GHI_noon_mean"] - ghi_min) / max(ghi_max - ghi_min, 1) * 8
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=radius,
        color="white", weight=0.8, fill=True,
        fill_color=colormap_ghi(row["GHI_noon_mean"]), fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['point_id']}</b><br>Noon GHI: {row['GHI_noon_mean']:.1f} W/m²<br>"
            f"T_amb: {row['T_amb_mean']:.1f} °C", max_width=220),
        tooltip=f"{row['point_id']}: {row['GHI_noon_mean']:.1f} W/m²",
    ).add_to(m1)
heat_data = [[r["lat"], r["lon"], r["GHI_noon_mean"]] for _, r in point_summary.iterrows()]
HeatMap(heat_data, radius=30, blur=20, min_opacity=0.4, name="GHI Heatmap").add_to(m1)
folium.LayerControl().add_to(m1)
save_and_report(m1, PLOT_DIR / "maps" / "A1_GHI_mean_map.html", "A1")

# ── A2. Population weight map ────────────────────────────────
print("  A2. Population-weight map ...")
m2 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB dark_matter")
pop_min, pop_max = point_summary["population"].min(), point_summary["population"].max()
colormap_pop = cm.LinearColormap(
    ["#023e8a", "#0096c7", "#ade8f4", "#f9c74f", "#f3722c"],
    vmin=pop_min, vmax=max(pop_max, pop_min + 1e-6), caption="Population (WorldPop 2020)")
colormap_pop.add_to(m2)
for _, row in point_summary.iterrows():
    radius = 5 + (row["population"] - pop_min) / max(pop_max - pop_min, 1) * 12
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=radius,
        color="white", weight=1, fill=True,
        fill_color=colormap_pop(row["population"]), fill_opacity=0.85,
        popup=folium.Popup(f"<b>{row['point_id']}</b><br>Population: {row['population']:,.0f}",
                            max_width=200),
        tooltip=f"{row['point_id']}: {row['population']:,.0f}",
    ).add_to(m2)
save_and_report(m2, PLOT_DIR / "maps" / "A2_population_map.html", "A2")

# ── A3. India context map ────────────────────────────────────
print("  A3. All points on India context map ...")
m3 = folium.Map(location=[22.5, 78.9], zoom_start=5, tiles="CartoDB positron")
for _, row in point_summary.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=4,
        color="#1d3557", weight=0.5, fill=True,
        fill_color="#e63946", fill_opacity=0.85,
        popup=folium.Popup(f"<b>{row['point_id']}</b><br>Noon GHI: {row['GHI_noon_mean']:.1f} W/m²",
                            max_width=200),
        tooltip=row["point_id"],
    ).add_to(m3)
bounds = [[point_summary["lat"].min() - 0.3, point_summary["lon"].min() - 0.3],
          [point_summary["lat"].max() + 0.3, point_summary["lon"].max() + 0.3]]
m3.fit_bounds(bounds)
save_and_report(m3, PLOT_DIR / "maps" / "A3_india_context.html", "A3")


# ═══════════════════════════════════════════════════════════
# B. TIME SERIES PLOTS (event-based, not hourly)
# ═══════════════════════════════════════════════════════════
print("\n[B] Time series plots ...")

# ── B1. Daily noon GHI — sample of points, 7-day rolling ────
print("  B1. Noon GHI by point (sample of 12) ...")
sample_points = sorted(noon_only["point_id"].unique())[:12]
palette = sns.color_palette("tab20", len(sample_points))

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
for i, pid in enumerate(sample_points):
    sub = noon_only[noon_only["point_id"] == pid].sort_values("date")
    ax.plot(sub["date"], sub["era5_GHI"].rolling(7, min_periods=1).mean(),
            label=pid, color=palette[i], linewidth=1.1, alpha=0.85)
ax.set_title(f"Daily Noon GHI — Sample of 12 Points (7-day rolling mean, {DATA_LABEL})",
             color="white", fontsize=13, pad=12)
ax.set_xlabel("Date", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["bottom", "left"]:
    ax.spines[s].set_color("#333333")
ax.legend(loc="upper right", fontsize=7, ncol=2, facecolor="#1a1a2e",
          labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "timeseries" / "B1_noon_GHI_sample_points.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B1 saved")

# ── B2. All points overlay + state mean ─────────────────────
print("  B2. Noon GHI — all points overlay ...")
fig, ax = plt.subplots(figsize=(17, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
for pid in noon_only["point_id"].unique():
    sub = noon_only[noon_only["point_id"] == pid].sort_values("date")
    ax.plot(sub["date"], sub["era5_GHI"].rolling(7, min_periods=1).mean(),
            color="#4cc9f0", linewidth=0.4, alpha=0.10)
state_daily = noon_only.groupby("date")["era5_GHI"].mean().reset_index()
ax.plot(state_daily["date"], state_daily["era5_GHI"].rolling(7, min_periods=1).mean(),
        color="#f9c74f", linewidth=2.6, label="Uttarakhand mean (7-day rolling)")
ax.set_title(f"Daily Noon GHI — All {noon_only['point_id'].nunique()} Points ({DATA_LABEL})",
             color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["bottom", "left"]:
    ax.spines[s].set_color("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "timeseries" / "B2_noon_GHI_all_points.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B2 saved")

# ── B3. T_amb vs GHI scatter, colored by event ───────────────
print("  B3. T_amb vs GHI scatter ...")
df_day = df[df["era5_GHI"] > 10]
df_scat = df_day.sample(min(20_000, len(df_day)), random_state=42)
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for event, col in EVENT_COLORS.items():
    sub = df_scat[df_scat["event"] == event]
    if len(sub) == 0:
        continue
    ax.scatter(sub["era5_T_amb"], sub["era5_GHI"], c=col, label=event,
               alpha=0.35, s=7, edgecolors="none")
ax.set_title(f"Air Temperature vs GHI — by Sun Event ({DATA_LABEL})", color="white", fontsize=13)
ax.set_xlabel("T_amb (°C)", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values():
    sp.set_color("#333333")
ax.legend(fontsize=8, markerscale=3, facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "timeseries" / "B3_Tamb_vs_GHI_scatter.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B3 saved")

# ── B4. Annual cycle — noon GHI by month, all points averaged ─
print("  B4. Annual cycle GHI ...")
monthly = noon_only.groupby("month")["era5_GHI"].agg(["mean", "std"]).reset_index()
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
ax.plot(monthly["month"], monthly["mean"], marker="o", color="#4cc9f0", linewidth=2.2, markersize=5)
ax.fill_between(monthly["month"], monthly["mean"] - monthly["std"], monthly["mean"] + monthly["std"],
                 alpha=0.2, color="#4cc9f0")
ax.axvspan(6, 9, alpha=0.08, color="#06d6a0", label="SW monsoon (Jun-Sep)")
ax.axvspan(10, 12, alpha=0.08, color="#f3722c", label="NE monsoon (Oct-Dec)")
ax.set_xticks(range(1, 13)); ax.set_xticklabels(month_labels, color="#aaaaaa")
ax.set_title(f"Annual Noon GHI Cycle — All Points Averaged, ±1 std ({DATA_LABEL})", color="white", fontsize=13)
ax.set_xlabel("Month", color="#aaaaaa"); ax.set_ylabel("Mean Noon GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values():
    sp.set_color("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333", fontsize=9)
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "timeseries" / "B4_annual_cycle_GHI.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B4 saved")


# ═══════════════════════════════════════════════════════════
# C. STATISTICAL PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[C] Statistical plots ...")

# ── C1. Correlation matrix ───────────────────────────────────
print("  C1. Correlation matrix ...")
corr_cols = [c for c in
             ["era5_GHI", "era5_DNI", "era5_DHI", "era5_CSI", "era5_GHI_clearsky",
              "era5_T_amb", "era5_T_dew", "era5_RHum", "era5_W_spd", "era5_cloud_cover",
              "era5_LW_down", "era5_P_atm", "era5_precipitation", "era5_SZA"]
             if c in df.columns]
day_sub = df[df["era5_SZA"] < 85] if "era5_SZA" in df.columns else df[df["era5_GHI"] > 0]
df_corr = day_sub[corr_cols].sample(min(50_000, len(day_sub)), random_state=42)
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
sns.heatmap(df_corr.corr(), ax=ax, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="#1f2937", annot_kws={"size": 8, "color": "white"},
            cbar_kws={"shrink": 0.8})
ax.set_title(f"Feature Correlation Matrix — daytime rows ({DATA_LABEL})", color="white", fontsize=13, pad=15)
ax.tick_params(colors="#cccccc", labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "statistics" / "C1_correlation_matrix.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C1 saved")

# ── C2. GHI violin by season ──────────────────────────────────
print("  C2. GHI violin by season ...")
df_vio = noon_only[noon_only["era5_GHI"] > 10]
season_present = [s for s in SEASON_ORDER if (df_vio["season"] == s).sum() > 0]
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
if len(season_present) > 0:
    parts = ax.violinplot(
        [df_vio[df_vio["season"] == s]["era5_GHI"].values for s in season_present],
        positions=range(len(season_present)), showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [SEASON_COLORS.get(s, "#888") for s in season_present]):
        pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor("white")
    for k in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[k].set_color("white" if k == "cmedians" else "#aaaaaa")
    ax.set_xticks(range(len(season_present)))
    ax.set_xticklabels(season_present, color="#cccccc", fontsize=10)
ax.set_title(f"Noon GHI Distribution by Season ({DATA_LABEL})", color="white", fontsize=13)
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values():
    sp.set_color("#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "statistics" / "C2_GHI_violin_season.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C2 saved")

# ── C3. Diurnal profile — mean value per event, by season ────
print("  C3. Diurnal (event) profile by season ...")
event_season = df.groupby(["event", "season"], observed=True)["era5_GHI"].mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
x = np.arange(len(EVENT_ORDER))
width = 0.8 / max(len(season_present), 1)
for i, season in enumerate(season_present):
    vals = [event_season[(event_season["event"] == e) & (event_season["season"] == season)]["era5_GHI"].mean()
            for e in EVENT_ORDER]
    ax.bar(x + i * width, vals, width, label=season, color=SEASON_COLORS.get(season, "#888"), alpha=0.85)
ax.set_xticks(x + width * max(len(season_present) - 1, 0) / 2)
ax.set_xticklabels(EVENT_ORDER, color="#cccccc")
ax.set_title(f"Mean GHI by Sun Event and Season ({DATA_LABEL})", color="white", fontsize=13)
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values():
    sp.set_color("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333", fontsize=9)
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "statistics" / "C3_diurnal_profile_season.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C3 saved")

# ── C4. Cloud cover vs GHI 2D density ─────────────────────────
print("  C4. Cloud vs GHI density ...")
if "era5_cloud_cover" in df.columns:
    df_ghi5 = df[df["era5_GHI"] > 5]
    df_samp = df_ghi5.sample(min(30_000, len(df_ghi5)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
    h = ax.hist2d(df_samp["era5_cloud_cover"], df_samp["era5_GHI"],
                  bins=60, cmap="plasma", norm=mcolors.LogNorm())
    plt.colorbar(h[3], ax=ax, label="Count (log scale)")
    ax.set_title(f"Cloud Cover vs GHI — 2D Density ({DATA_LABEL})", color="white", fontsize=13)
    ax.set_xlabel("Cloud Cover (0-1)", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_color("#333333")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "statistics" / "C4_cloud_vs_GHI_density.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    C4 saved")
else:
    print("    [SKIP] era5_cloud_cover column not present")


# ═══════════════════════════════════════════════════════════
# D. SOLAR RESOURCE QUALITY PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[D] Solar resource quality plots ...")

# ── D1. CSI distribution ──────────────────────────────────────
print("  D1. CSI distribution ...")
if "era5_CSI" in df.columns:
    df_csi = df[(df["era5_CSI"] > 0) & (df["era5_CSI"] <= 1.5) & (df["era5_GHI"] > 10)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#111827")
    axes[0].hist(df_csi["era5_CSI"], bins=60, color="#4cc9f0", alpha=0.8, edgecolor="none")
    axes[0].axvline(1.0, color="#f9c74f", linewidth=1.5, linestyle="--", label="Perfect clear sky")
    axes[0].set_title("Clear Sky Index (CSI) Distribution", color="white", fontsize=12)
    axes[0].set_xlabel("CSI (0=cloudy, 1=clear, >1=enhancement)", color="#aaaaaa")
    axes[0].set_ylabel("Count", color="#aaaaaa"); axes[0].tick_params(colors="#aaaaaa")
    for sp in axes[0].spines.values():
        sp.set_color("#333333")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    axes[0].grid(axis="y", color="#1f2937", linewidth=0.5)

    csi_by_season = df_csi.groupby("season", observed=True)["era5_CSI"].mean().reindex(season_present)
    axes[1].bar(csi_by_season.index.astype(str), csi_by_season.values,
                color=[SEASON_COLORS.get(s, "#888") for s in csi_by_season.index], alpha=0.85)
    axes[1].set_title("Mean CSI by Season", color="white", fontsize=12)
    axes[1].set_ylabel("Mean CSI", color="#aaaaaa"); axes[1].tick_params(colors="#aaaaaa")
    for sp in axes[1].spines.values():
        sp.set_color("#333333")
    axes[1].grid(axis="y", color="#1f2937", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "solar_resource" / "D1_CSI_distribution.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    D1 saved")

# ── D2. Top 20 points by noon GHI ─────────────────────────────
print("  D2. Top 20 points by noon GHI ...")
top20 = point_summary.nlargest(20, "GHI_noon_mean").sort_values("GHI_noon_mean")
fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
bars = ax.barh(top20["point_id"], top20["GHI_noon_mean"], color="#4cc9f0", alpha=0.85, edgecolor="none")
for bar, val in zip(bars, top20["GHI_noon_mean"]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
            va="center", ha="left", color="white", fontsize=8)
ax.set_title(f"Top 20 Points — Mean Noon GHI, W/m² ({DATA_LABEL})", color="white", fontsize=13)
ax.set_xlabel("Mean Noon GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#cccccc", labelsize=8)
for sp in ax.spines.values():
    sp.set_color("#333333")
ax.grid(axis="x", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(PLOT_DIR / "solar_resource" / "D2_top20_points_GHI.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    D2 saved")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  ALL PLOTS COMPLETE")
print(f"\n  Saved to: {PLOT_DIR}/")
print("""
  maps/
    A0_all_points_overview.html    <- all points, clustered markers
    A1_GHI_mean_map.html           <- colour + size by noon GHI + heatmap overlay
    A2_population_map.html         <- colour + size by population
    A3_india_context.html          <- national context

  timeseries/
    B1_noon_GHI_sample_points.png
    B2_noon_GHI_all_points.png
    B3_Tamb_vs_GHI_scatter.png
    B4_annual_cycle_GHI.png

  statistics/
    C1_correlation_matrix.png
    C2_GHI_violin_season.png
    C3_diurnal_profile_season.png
    C4_cloud_vs_GHI_density.png

  solar_resource/
    D1_CSI_distribution.png
    D2_top20_points_GHI.png
""")
print("=" * 68)
