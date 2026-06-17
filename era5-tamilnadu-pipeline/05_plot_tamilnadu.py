"""
05_plot_tamilnadu.py
====================
All visualizations for the ERA5 Tamil Nadu solar-climate dataset.
Run AFTER 04_preprocess_tamilnadu.py.

Based on: "Multimodal Learning Techniques for Time Series Forecasting
           in Renewable Energy Systems" (Mansouri et al., IEEE Access 2025)

PLOTS PRODUCED:
  A. Folium interactive maps (HTML):
     A1. GHI mean spatial map (all 222 locations)
     A2. Climate zone map
     A3. District heatmap of solar resource
     A4. Full ERA5 grid point map

  B. Time series plots:
     B1. GHI daily mean — all districts (multi-line)
     B2. GHI vs clearsky GHI for 3 representative cities
     B3. Temperature vs GHI scatter by climate zone
     B4. Annual cycle — monthly mean GHI by climate zone

  C. Statistical / distribution plots:
     C1. Correlation matrix (solar + weather features)
     C2. GHI distribution by climate zone (violin plot)
     C3. Diurnal profile — mean GHI by hour (all seasons)
     C4. Cloud cover vs GHI heatmap (2D density)

  D. Feature engineering verification:
     D1. Lag feature correlation with GHI
     D2. Rolling mean comparison
     D3. Train/val/test split timeline

  E. Solar resource quality:
     E1. RRTDHS score heatmap (city × month)
     E2. CSI distribution (clear sky index)
     E3. Top 20 cities by mean GHI

OUTPUTS: data/plots/ folder

HOW TO RUN IN COLAB:
  COLAB = True below
  All plots saved as PNG + interactive HTML maps
"""

COLAB = False   # ← change to True in Colab

if COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/tamilnadu_era5"
    # Install folium if needed
    import subprocess
    subprocess.run(["pip", "install", "folium", "branca", "-q"])
else:
    from config import (
        BASE_DIR,
        PROCESSED_DIR,
        PREPROCESSED_DIR,
        PLOTS_DIR,
        PROCESSED_NAMED_DIR,
        PROCESSED_GRID_DIR,
        ensure_data_dirs,
    )

    ensure_data_dirs()

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on Colab and servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
if COLAB:
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    PREPROC_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
    PLOT_DIR = os.path.join(BASE_DIR, "data", "plots")
    BY_LOC_DIR = os.path.join(PROCESSED_DIR, "by_location")
else:
    PROCESSED_DIR = str(PROCESSED_DIR)
    PREPROC_DIR = str(PREPROCESSED_DIR)
    PLOT_DIR = str(PLOTS_DIR)
    BY_LOC_DIR = str(PROCESSED_NAMED_DIR)
    GRID_DIR = str(PROCESSED_GRID_DIR)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "maps"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "timeseries"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "statistics"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "solar_resource"), exist_ok=True)

print("=" * 68)
print("  ERA5 Tamil Nadu — Visualization Pipeline")
print(f"  Output : {PLOT_DIR}/")
print("=" * 68)

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
print("\nLoading data ...")

COMBINED_FILE = os.path.join(PROCESSED_DIR, "climate_tamilnadu_all.csv")
df = pd.read_csv(COMBINED_FILE, parse_dates=["timestamp"])
df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

# Load preprocessed if available
PREPROC_FILE = os.path.join(PREPROC_DIR, "full_preprocessed.csv")
if os.path.exists(PREPROC_FILE):
    df_pre = pd.read_csv(PREPROC_FILE, parse_dates=["timestamp"])
    print(f"  Preprocessed file loaded: {len(df_pre):,} rows")
else:
    df_pre = None
    print("  [NOTE] Preprocessed file not found — feature plots will be skipped.")
    print("         Run 04_preprocess_tamilnadu.py first.")

print(f"  Combined file: {len(df):,} rows  |  {df['city'].nunique()} cities")

# City-level summary for maps
city_summary = df.groupby("city").agg(
    lat=("lat", "first"),
    lon=("lon", "first"),
    alt=("altitude_m", "first"),
    district=("district", "first"),
    climate_zone=("climate_zone", "first"),
    GHI_mean=("GHI", "mean"),
    T_amb_mean=("T_amb", "mean"),
    RRTDHS_mean=("RRTDHS", "mean"),
    high_solar_pct=("high_solar_resource", "mean"),
    GHI_max=("GHI", "max"),
).reset_index()

# Climate zone color palette
CZ_COLORS = {
    "hot-humid-coastal":  "#0077b6",
    "hot-humid":          "#48cae4",
    "hot-semi-arid":      "#f77f00",
    "semi-arid":          "#fcbf49",
    "hot-arid-coastal":   "#d62828",
    "hot-arid":           "#9d0208",
    "cool-hilly":         "#2dc653",
    "semi-arid-elevated": "#a8dadc",
    "hot-humid-elevated": "#06d6a0",
}

print("  Data loaded ✓\n")


# ═══════════════════════════════════════════════════════════
# ── A. FOLIUM INTERACTIVE MAPS ────────────────────────────
# ═══════════════════════════════════════════════════════════
print("[A] Creating Folium interactive maps ...")

TN_CENTER = [10.9, 78.5]   # center of Tamil Nadu

# ── A1. GHI Mean Map ──────────────────────────────────────
print("  A1. GHI mean spatial map ...")

m1 = folium.Map(location=TN_CENTER, zoom_start=7,
                tiles="CartoDB positron")

# Color scale for GHI
ghi_min = city_summary["GHI_mean"].min()
ghi_max = city_summary["GHI_mean"].max()
colormap_ghi = cm.LinearColormap(
    colors=["#2d6a4f", "#52b788", "#d9ed92", "#f9c74f", "#f3722c", "#90be6d"],
    vmin=ghi_min, vmax=ghi_max,
    caption="Mean GHI (W/m²)"
)
colormap_ghi.add_to(m1)

for _, row in city_summary.iterrows():
    ghi = row["GHI_mean"]
    color = colormap_ghi(ghi)
    radius = 6 + (ghi - ghi_min) / (ghi_max - ghi_min) * 8

    popup_html = f"""
    <div style="font-family:Arial; width:220px; font-size:13px;">
      <b style="font-size:15px;">{row['city']}</b><br>
      <hr style="margin:4px 0">
      District    : {row['district']}<br>
      Climate zone: {row['climate_zone']}<br>
      Altitude    : {row['alt']:.0f} m<br>
      <hr style="margin:4px 0">
      <b>Mean GHI  : {ghi:.1f} W/m²</b><br>
      Max GHI   : {row['GHI_max']:.0f} W/m²<br>
      Mean Temp : {row['T_amb_mean']:.1f} °C<br>
      High solar: {row['high_solar_pct']*100:.0f}% of hours<br>
    </div>
    """
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color="white",
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['city']}: {ghi:.1f} W/m²",
    ).add_to(m1)

# Add HeatMap layer
heat_data = [[row["lat"], row["lon"], row["GHI_mean"]]
             for _, row in city_summary.iterrows()]
HeatMap(heat_data, radius=30, blur=20,
        min_opacity=0.4, name="GHI Heatmap").add_to(m1)
folium.LayerControl().add_to(m1)

m1_path = os.path.join(PLOT_DIR, "maps", "A1_GHI_mean_map.html")
m1.save(m1_path)
print(f"    Saved: {m1_path}")


# ── A2. Climate Zone Map ───────────────────────────────────
print("  A2. Climate zone map ...")

m2 = folium.Map(location=TN_CENTER, zoom_start=7,
                tiles="CartoDB positron")

for cz, color in CZ_COLORS.items():
    subset = city_summary[city_summary["climate_zone"] == cz]
    for _, row in subset.iterrows():
        popup_html = f"""
        <div style="font-family:Arial; font-size:13px; width:200px;">
          <b>{row['city']}</b><br>
          Climate: <b>{cz}</b><br>
          District: {row['district']}<br>
          Altitude: {row['alt']:.0f} m<br>
          Mean GHI: {row['GHI_mean']:.1f} W/m²
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color="white",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=f"{row['city']} | {cz}",
        ).add_to(m2)

# Legend
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background: white; padding: 12px; border-radius: 8px;
     box-shadow: 2px 2px 8px rgba(0,0,0,0.3); font-family: Arial; font-size: 12px;">
  <b>Climate Zones</b><br>
"""
for cz, color in CZ_COLORS.items():
    legend_html += (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border-radius:50%;margin-right:6px;"></span>'
        f'{cz}<br>'
    )
legend_html += "</div>"
m2.get_root().html.add_child(folium.Element(legend_html))

m2_path = os.path.join(PLOT_DIR, "maps", "A2_climate_zone_map.html")
m2.save(m2_path)
print(f"    Saved: {m2_path}")


# ── A3. District Solar Resource Map ───────────────────────
print("  A3. District solar resource heatmap ...")

m3 = folium.Map(location=TN_CENTER, zoom_start=7,
                tiles="CartoDB dark_matter")

# District-level aggregation
dist_summary = city_summary.groupby("district").agg(
    lat=("lat", "mean"),
    lon=("lon", "mean"),
    GHI_mean=("GHI_mean", "mean"),
    high_solar_pct=("high_solar_pct", "mean"),
    n_cities=("city", "count"),
).reset_index()

colormap_dist = cm.LinearColormap(
    colors=["#023e8a", "#0096c7", "#ade8f4", "#f9c74f", "#f3722c"],
    vmin=dist_summary["GHI_mean"].min(),
    vmax=dist_summary["GHI_mean"].max(),
    caption="District Mean GHI (W/m²)"
)
colormap_dist.add_to(m3)

for _, row in dist_summary.iterrows():
    color = colormap_dist(row["GHI_mean"])
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=15,
        color="white",
        weight=1.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{row['district']}</b><br>"
            f"Mean GHI: {row['GHI_mean']:.1f} W/m²<br>"
            f"High solar: {row['high_solar_pct']*100:.0f}% of hours<br>"
            f"Locations tracked: {row['n_cities']}",
            max_width=200),
        tooltip=f"{row['district']}: {row['GHI_mean']:.1f} W/m²",
    ).add_to(m3)

m3_path = os.path.join(PLOT_DIR, "maps", "A3_district_solar_resource.html")
m3.save(m3_path)
print(f"    Saved: {m3_path}")


# ── A4. Full ERA5 Grid Point Map ───────────────────────────
print("  A4. ERA5 grid points map ...")

GRID_FILE = os.path.join(PROCESSED_DIR, "grid", "era5_TN_grid_all.csv")
if os.path.exists(GRID_FILE):
    df_grid = pd.read_csv(GRID_FILE, usecols=["grid_lat", "grid_lon", "GHI",
                                               "T_amb", "timestamp"],
                          parse_dates=["timestamp"], nrows=200_000)
    grid_summary = df_grid.groupby(["grid_lat", "grid_lon"]).agg(
        GHI_mean=("GHI", "mean"),
        T_mean=("T_amb", "mean"),
    ).reset_index()

    m4 = folium.Map(location=TN_CENTER, zoom_start=7,
                    tiles="CartoDB positron")
    colormap_grid = cm.LinearColormap(
        colors=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
        vmin=grid_summary["GHI_mean"].min(),
        vmax=grid_summary["GHI_mean"].max(),
        caption="ERA5 Grid GHI Mean (W/m²)"
    )
    colormap_grid.add_to(m4)

    for _, row in grid_summary.iterrows():
        folium.Rectangle(
            bounds=[
                [row["grid_lat"] - 0.125, row["grid_lon"] - 0.125],
                [row["grid_lat"] + 0.125, row["grid_lon"] + 0.125],
            ],
            color="white",
            weight=0.3,
            fill=True,
            fill_color=colormap_grid(row["GHI_mean"]),
            fill_opacity=0.7,
            tooltip=f"({row['grid_lat']:.2f}°N, {row['grid_lon']:.2f}°E) "
                    f"GHI={row['GHI_mean']:.1f} W/m²",
        ).add_to(m4)

    m4_path = os.path.join(PLOT_DIR, "maps", "A4_ERA5_grid_map.html")
    m4.save(m4_path)
    print(f"    Saved: {m4_path}")
else:
    print("    [SKIP] era5_TN_grid_all.csv not found")


# ═══════════════════════════════════════════════════════════
# ── B. TIME SERIES PLOTS ──────────────────────────────────
# ═══════════════════════════════════════════════════════════
print("\n[B] Time series plots ...")

# ── B1. Daily mean GHI — all districts ────────────────────
print("  B1. Daily GHI by district ...")

df["date"] = df["timestamp"].dt.date
daily_dist = df.groupby(["date", "district"])["GHI"].mean().reset_index()
daily_dist["date"] = pd.to_datetime(daily_dist["date"])

# Sample 12 districts for readability
districts_plot = daily_dist["district"].unique()[:12]
palette = sns.color_palette("tab20", len(districts_plot))

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

for i, dist in enumerate(districts_plot):
    sub = daily_dist[daily_dist["district"] == dist]
    ax.plot(sub["date"], sub["GHI"].rolling(7).mean(),
            label=dist, color=palette[i], linewidth=1.2, alpha=0.85)

ax.set_title("Daily Mean GHI — Tamil Nadu Districts (7-day rolling mean)",
             color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#aaaaaa")
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=8, ncol=2,
          facecolor="#1a1a2e", labelcolor="white",
          edgecolor="#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B1_daily_GHI_districts.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B1 saved ✓")


# ── B2. GHI vs Clearsky GHI — 3 cities ────────────────────
print("  B2. GHI vs clearsky GHI ...")

cities_sample = ["Chennai", "Coimbatore", "Ooty"]
cities_sample = [c for c in cities_sample if c in df["city"].unique()]

fig, axes = plt.subplots(len(cities_sample), 1,
                         figsize=(16, 4 * len(cities_sample)))
fig.patch.set_facecolor("#0d1117")
if len(cities_sample) == 1:
    axes = [axes]

for ax, city in zip(axes, cities_sample):
    ax.set_facecolor("#111827")
    sub = df[df["city"] == city].copy()
    # One week sample in June (peak solar season)
    week = sub[(sub["timestamp"].dt.month == 6) &
               (sub["timestamp"].dt.day <= 7)]
    ax.fill_between(week["timestamp"], week["GHI_clearsky"],
                    alpha=0.3, color="#f9c74f", label="Clearsky GHI")
    ax.plot(week["timestamp"], week["GHI_clearsky"],
            color="#f9c74f", linewidth=0.8, alpha=0.7)
    ax.fill_between(week["timestamp"], week["GHI"],
                    alpha=0.6, color="#4cc9f0", label="Actual GHI")
    ax.plot(week["timestamp"], week["GHI"],
            color="#4cc9f0", linewidth=1.2)
    ax.set_title(f"{city}  — GHI vs Clearsky GHI (June week 1)",
                 color="white", fontsize=12)
    ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    ax.grid(color="#1f2937", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B2_GHI_vs_clearsky.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B2 saved ✓")


# ── B3. Temperature vs GHI scatter by climate zone ─────────
print("  B3. Temperature vs GHI scatter ...")

# Sample 10% of daytime rows only
df_day = df[df["SZA"] < 85].sample(frac=0.05, random_state=42)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#111827")

for cz, color in CZ_COLORS.items():
    sub = df_day[df_day["climate_zone"] == cz]
    if len(sub) == 0:
        continue
    ax.scatter(sub["T_amb"], sub["GHI"], c=color, label=cz,
               alpha=0.35, s=8, edgecolors="none")

ax.set_title("Air Temperature vs GHI — by Climate Zone (daytime only)",
             color="white", fontsize=13)
ax.set_xlabel("T_amb (°C)", color="#aaaaaa")
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_color("#333333")
ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
          labelcolor="white", edgecolor="#333333", markerscale=3)
ax.grid(color="#1f2937", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B3_Tamb_vs_GHI_scatter.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B3 saved ✓")


# ── B4. Annual cycle — monthly mean GHI by climate zone ───
print("  B4. Annual cycle GHI by climate zone ...")

monthly_cz = df.groupby(["month", "climate_zone"])["GHI"].mean().reset_index()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#111827")

for cz, color in CZ_COLORS.items():
    sub = monthly_cz[monthly_cz["climate_zone"] == cz]
    if len(sub) == 0:
        continue
    sub = sub.sort_values("month")
    ax.plot(sub["month"], sub["GHI"], marker="o", color=color,
            label=cz, linewidth=2, markersize=5)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels, color="#aaaaaa")
ax.set_title("Annual GHI Cycle — by Climate Zone",
             color="white", fontsize=13)
ax.set_xlabel("Month", color="#aaaaaa")
ax.set_ylabel("Mean GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_color("#333333")
ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
          labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)

# Add season bands
ax.axvspan(6, 9, alpha=0.07, color="#4cc9f0", label="Monsoon")
ax.axvspan(3, 5, alpha=0.07, color="#f9c74f", label="Summer")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B4_annual_cycle_GHI.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    B4 saved ✓")


# ═══════════════════════════════════════════════════════════
# ── C. STATISTICAL PLOTS ──────────────────────────────────
# ═══════════════════════════════════════════════════════════
print("\n[C] Statistical plots ...")

# ── C1. Correlation Matrix ─────────────────────────────────
print("  C1. Correlation matrix ...")

corr_cols = [c for c in ["GHI", "DNI", "DHI", "CSI", "GHI_clearsky",
                          "T_amb", "T_dew", "RHum", "W_spd",
                          "cloud_cover", "LW_down", "P_atm",
                          "precipitation", "SZA", "RRTDHS"]
             if c in df.columns]

# Daytime only for meaningful solar correlations
df_corr = df[df["SZA"] < 85][corr_cols].sample(50_000, random_state=42)
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f",
            cmap=cmap, center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="#1f2937",
            annot_kws={"size": 8, "color": "white"},
            cbar_kws={"shrink": 0.8})

ax.set_title("Feature Correlation Matrix (daytime hours only)",
             color="white", fontsize=13, pad=15)
ax.tick_params(colors="#cccccc", labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "statistics", "C1_correlation_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C1 saved ✓")


# ── C2. GHI Distribution by Climate Zone (violin) ─────────
print("  C2. GHI violin plot by climate zone ...")

df_violin = df[df["GHI"] > 10].copy()  # daytime only
cz_order = sorted(df_violin["climate_zone"].unique())

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#111827")

violin_data = [df_violin[df_violin["climate_zone"] == cz]["GHI"].values
               for cz in cz_order]
parts = ax.violinplot(violin_data, positions=range(len(cz_order)),
                      showmedians=True, showextrema=True)

cz_palette = [CZ_COLORS.get(cz, "#888888") for cz in cz_order]
for i, (pc, color) in enumerate(zip(parts["bodies"], cz_palette)):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor("white")

parts["cmedians"].set_color("white")
parts["cmins"].set_color("#aaaaaa")
parts["cmaxes"].set_color("#aaaaaa")
parts["cbars"].set_color("#aaaaaa")

ax.set_xticks(range(len(cz_order)))
ax.set_xticklabels(cz_order, rotation=30, ha="right",
                   color="#cccccc", fontsize=9)
ax.set_title("GHI Distribution by Climate Zone (daytime hours)",
             color="white", fontsize=13)
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_color("#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "statistics", "C2_GHI_violin_climate_zone.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C2 saved ✓")


# ── C3. Diurnal Profile — GHI by Hour and Season ──────────
print("  C3. Diurnal profile by season ...")

season_colors = {
    "Winter": "#4cc9f0", "Summer": "#f9c74f",
    "Monsoon": "#06d6a0", "Retreat": "#f3722c"
}
hourly_season = df.groupby(["hour", "season"])["GHI"].mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#111827")

for season, color in season_colors.items():
    sub = hourly_season[hourly_season["season"] == season]
    if len(sub) == 0:
        continue
    ax.plot(sub["hour"], sub["GHI"], color=color, label=season,
            linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(sub["hour"], sub["GHI"], alpha=0.15, color=color)

ax.set_title("Diurnal GHI Profile — by Season (Tamil Nadu average)",
             color="white", fontsize=13)
ax.set_xlabel("Hour of Day (UTC)", color="#aaaaaa")
ax.set_ylabel("Mean GHI (W/m²)", color="#aaaaaa")
ax.set_xticks(range(0, 24))
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_color("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "statistics", "C3_diurnal_profile.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C3 saved ✓")


# ── C4. Cloud Cover vs GHI — 2D Density ───────────────────
print("  C4. Cloud cover vs GHI density ...")

df_samp = df[df["GHI"] > 5].sample(min(30_000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#111827")

h = ax.hist2d(df_samp["cloud_cover"], df_samp["GHI"],
              bins=60, cmap="plasma", norm=mcolors.LogNorm())
plt.colorbar(h[3], ax=ax, label="Count (log scale)")

ax.set_title("Cloud Cover vs GHI — 2D Density (daytime hours)",
             color="white", fontsize=13)
ax.set_xlabel("Cloud Cover (0–1)", color="#aaaaaa")
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_color("#333333")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "statistics", "C4_cloud_vs_GHI_density.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    C4 saved ✓")


# ═══════════════════════════════════════════════════════════
# ── D. FEATURE ENGINEERING VERIFICATION PLOTS ────────────
# ═══════════════════════════════════════════════════════════
print("\n[D] Feature engineering plots ...")

if df_pre is not None:

    # ── D1. Lag Feature Correlation with GHI ──────────────
    print("  D1. Lag feature correlations ...")

    lag_cols = [c for c in df_pre.columns if "GHI_lag" in c or c == "GHI"]
    if len(lag_cols) > 1:
        lag_corrs = df_pre[lag_cols].corr()["GHI"].drop("GHI").sort_values()

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#111827")

        colors_bar = ["#f3722c" if v < 0 else "#4cc9f0" for v in lag_corrs.values]
        ax.barh(lag_corrs.index, lag_corrs.values, color=colors_bar, alpha=0.8)
        ax.set_title("Lag Feature Correlation with GHI",
                     color="white", fontsize=13)
        ax.set_xlabel("Pearson Correlation", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.axvline(0, color="white", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.grid(axis="x", color="#1f2937", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "features", "D1_lag_correlations.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print("    D1 saved ✓")

    # ── D2. Rolling Mean vs Raw GHI ────────────────────────
    print("  D2. Rolling mean comparison ...")

    roll_cols = [c for c in df_pre.columns if "GHI_roll" in c and "mean" in c]
    if roll_cols and "GHI" in df_pre.columns:
        # One city, one day
        city_ex = df_pre["city"].iloc[0] if "city" in df_pre.columns else None
        if city_ex:
            sub = df_pre[df_pre["city"] == city_ex].head(72)  # 3 days
        else:
            sub = df_pre.head(72)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#111827")

        ax.plot(range(len(sub)), sub["GHI"], color="white",
                linewidth=0.8, alpha=0.5, label="Raw GHI")
        roll_palette = ["#f9c74f", "#f3722c", "#4cc9f0"]
        for i, col in enumerate(roll_cols[:3]):
            if col in sub.columns:
                ax.plot(range(len(sub)), sub[col],
                        color=roll_palette[i], linewidth=1.8,
                        label=col.replace("GHI_", ""))

        ax.set_title("Rolling Mean Smoothing of GHI (72-hour window)",
                     color="white", fontsize=13)
        ax.set_xlabel("Time steps", color="#aaaaaa")
        ax.set_ylabel("GHI (normalized)", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
        ax.grid(color="#1f2937", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "features", "D2_rolling_mean.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print("    D2 saved ✓")

    # ── D3. Train/Val/Test Split Timeline ─────────────────
    print("  D3. Train/val/test split timeline ...")

    df_pre_sorted = df_pre.sort_values("timestamp")
    n = len(df_pre_sorted)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    timestamps = df_pre_sorted["timestamp"].values
    ghi_vals   = df_pre_sorted["GHI"].values if "GHI" in df_pre_sorted.columns else np.zeros(n)

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")

    ax.plot(timestamps[:t_end], ghi_vals[:t_end],
            color="#4cc9f0", linewidth=0.5, alpha=0.6, label="Train (70%)")
    ax.plot(timestamps[t_end:v_end], ghi_vals[t_end:v_end],
            color="#f9c74f", linewidth=0.5, alpha=0.8, label="Validation (15%)")
    ax.plot(timestamps[v_end:], ghi_vals[v_end:],
            color="#f3722c", linewidth=0.5, alpha=0.8, label="Test (15%)")

    ax.axvline(timestamps[t_end], color="#f9c74f", linewidth=1.5, linestyle="--")
    ax.axvline(timestamps[v_end], color="#f3722c", linewidth=1.5, linestyle="--")
    ax.set_title("Train / Validation / Test Temporal Split",
                 color="white", fontsize=13)
    ax.set_ylabel("GHI (normalized)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    ax.grid(color="#1f2937", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "features", "D3_train_val_test_split.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    D3 saved ✓")

else:
    print("  [SKIP] Preprocessed data not found — D plots skipped.")


# ═══════════════════════════════════════════════════════════
# ── E. SOLAR RESOURCE QUALITY PLOTS ──────────────────────
# ═══════════════════════════════════════════════════════════
print("\n[E] Solar resource quality plots ...")

# ── E1. RRTDHS Heatmap (city × month) ─────────────────────
print("  E1. RRTDHS heatmap ...")

# Top 30 cities by mean GHI
top_cities = city_summary.nlargest(30, "GHI_mean")["city"].tolist()
rrtdhs_pivot = (
    df[df["city"].isin(top_cities)]
    .groupby(["city", "month"])["RRTDHS"]
    .mean()
    .unstack(level="month")
)
rrtdhs_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")

sns.heatmap(rrtdhs_pivot, ax=ax, cmap="YlOrRd",
            linewidths=0.3, linecolor="#1f2937",
            annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"label": "RRTDHS Score", "shrink": 0.8})

ax.set_title("RRTDHS Solar Resource Score — Top 30 Cities by Month",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="#cccccc", labelsize=8)
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "solar_resource", "E1_RRTDHS_heatmap.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    E1 saved ✓")


# ── E2. CSI Distribution ──────────────────────────────────
print("  E2. CSI distribution ...")

df_csi = df[(df["CSI"] > 0) & (df["CSI"] <= 1.5) & (df["GHI"] > 10)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#111827")

# Left: overall CSI histogram
axes[0].hist(df_csi["CSI"], bins=60, color="#4cc9f0",
             alpha=0.8, edgecolor="none")
axes[0].set_title("Clear Sky Index (CSI) Distribution",
                  color="white", fontsize=12)
axes[0].set_xlabel("CSI (0=cloudy, 1=clear, >1=enhancement)",
                   color="#aaaaaa")
axes[0].set_ylabel("Count", color="#aaaaaa")
axes[0].tick_params(colors="#aaaaaa")
for spine in axes[0].spines.values():
    spine.set_color("#333333")
axes[0].axvline(1.0, color="#f9c74f", linewidth=1.5,
                linestyle="--", label="Perfect clear sky")
axes[0].grid(axis="y", color="#1f2937", linewidth=0.5)
axes[0].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")

# Right: CSI by season
for season, color in season_colors.items():
    sub = df_csi[df_csi["season"] == season]["CSI"]
    if len(sub) == 0:
        continue
    axes[1].hist(sub, bins=50, alpha=0.5, color=color,
                 label=season, edgecolor="none")

axes[1].set_title("CSI Distribution by Season", color="white", fontsize=12)
axes[1].set_xlabel("CSI", color="#aaaaaa")
axes[1].tick_params(colors="#aaaaaa")
for spine in axes[1].spines.values():
    spine.set_color("#333333")
axes[1].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
axes[1].grid(axis="y", color="#1f2937", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "solar_resource", "E2_CSI_distribution.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    E2 saved ✓")


# ── E3. Top 20 Cities by Mean GHI ─────────────────────────
print("  E3. Top 20 cities by GHI ...")

top20 = city_summary.nlargest(20, "GHI_mean").sort_values("GHI_mean")

fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#111827")

bar_colors = [CZ_COLORS.get(cz, "#888888")
              for cz in top20["climate_zone"]]

bars = ax.barh(top20["city"], top20["GHI_mean"],
               color=bar_colors, alpha=0.85, edgecolor="none")

for bar, val in zip(bars, top20["GHI_mean"]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", ha="left",
            color="white", fontsize=8)

ax.set_title("Top 20 Locations — Mean GHI (W/m²)",
             color="white", fontsize=13)
ax.set_xlabel("Mean GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#cccccc", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#333333")
ax.grid(axis="x", color="#1f2937", linewidth=0.5)

# Add a simple color legend for climate zones shown
shown_cz = top20["climate_zone"].unique()
handles = [plt.Rectangle((0, 0), 1, 1, color=CZ_COLORS.get(cz, "#888"))
           for cz in shown_cz]
ax.legend(handles, shown_cz, loc="lower right",
          fontsize=7, facecolor="#1a1a2e",
          labelcolor="white", edgecolor="#333333")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "solar_resource", "E3_top20_GHI_cities.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("    E3 saved ✓")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  ✅  ALL PLOTS COMPLETE")
print(f"\n  Saved to: {PLOT_DIR}/")
print()
print("  maps/")
print("    A1_GHI_mean_map.html        ← interactive, open in browser")
print("    A2_climate_zone_map.html")
print("    A3_district_solar_resource.html")
print("    A4_ERA5_grid_map.html")
print()
print("  timeseries/")
print("    B1_daily_GHI_districts.png")
print("    B2_GHI_vs_clearsky.png")
print("    B3_Tamb_vs_GHI_scatter.png")
print("    B4_annual_cycle_GHI.png")
print()
print("  statistics/")
print("    C1_correlation_matrix.png")
print("    C2_GHI_violin_climate_zone.png")
print("    C3_diurnal_profile.png")
print("    C4_cloud_vs_GHI_density.png")
print()
print("  features/")
print("    D1_lag_correlations.png      (needs preprocessed data)")
print("    D2_rolling_mean.png          (needs preprocessed data)")
print("    D3_train_val_test_split.png  (needs preprocessed data)")
print()
print("  solar_resource/")
print("    E1_RRTDHS_heatmap.png")
print("    E2_CSI_distribution.png")
print("    E3_top20_GHI_cities.png")
print("=" * 68)
print("\nTo view HTML maps: open the .html files in any browser (Chrome/Firefox).")