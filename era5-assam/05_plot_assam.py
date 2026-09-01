# 05_plot_assam.py
# ERA5 Assam - All Visualizations
#
# Mirror of 05_plot_tamilnadu.py (friend's reference code) adapted for the
# Assam pipeline schema. Key differences from the TN version:
#
#   TN schema                   | Assam schema
#   ----------------------------|-------------------------------------------
#   climate_tamilnadu_all.csv   | climate_assam_points.csv
#   city / district columns     | point_id (ASP_XXXX)
#   climate_zone (named string) | cluster_id (0-3) from GMM
#   full_preprocessed.csv       | preprocessed/parquet/{point_id}.parquet
#   era5_GHI (no prefix)        | era5_GHI / era5_GHI_corrected (prefixed)
#   Seasons: Winter/Summer/     | Seasons: Winter/Pre-Monsoon/
#            Monsoon/Retreat     |          Monsoon/Post-Monsoon
#
# Cluster meanings (from recommendation_cards_assam.md):
#   Cluster-0  24 pts  medoid ASP_0013 (27.375, 94.875)  kt=0.696
#   Cluster-1  52 pts  medoid ASP_0017 (26.875, 94.125)  kt=0.758
#   Cluster-2  11 pts  medoid ASP_0008 (24.875, 92.875)  kt=0.789
#   Cluster-3  41 pts  medoid ASP_0001 (26.125, 91.625)  kt=0.772
#
# HOW TO RUN
# ----------
#   python 05_plot_assam.py
#   Outputs -> era5-assam/data/plots/
#
# PLOTS PRODUCED (same A-E structure as TN friend's code)
# --------------------------------------------------------
#  A. Folium interactive maps (HTML)
#     A0. All grid points overview  (colour = GMM cluster)
#     A1. GHI mean spatial map      (colour + size by GHI, + heatmap layer)
#     A2. Climate cluster map       (colour = GMM cluster)
#     A3. Population-weighted solar resource map (dark basemap)
#     A4. National context map      (Assam inside India)
#  B. Time series plots (PNG)
#     B1. Daily GHI by cluster      (7-day rolling mean)
#     B2. GHI vs clearsky GHI       (3 sample grid points, June week 1)
#     B3. T_amb vs GHI scatter      (by cluster, daytime only)
#     B4. Annual GHI cycle          (by cluster)
#     B5. Daily GHI all points overlay
#  C. Statistical plots (PNG)
#     C1. Feature correlation matrix (daytime)
#     C2. GHI violin by cluster
#     C3. Diurnal profile by season
#     C4. Cloud cover vs GHI 2D density
#  D. Feature engineering verification (from parquet)
#     D1. Lag feature correlations with GHI
#     D2. Rolling mean comparison (72-step window)
#     D3. Train / Val / Test temporal split
#  E. Solar resource quality (PNG)
#     E1. Solar resource score heatmap (top-30 points x 12 months)
#     E2. CSI distribution (overall + by season)
#     E3. Top-20 grid points by mean GHI

# -- 0. Imports ----------------------------------------------------------
import os
import sys
import warnings
import webbrowser

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

# -- 1. Path configuration -----------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

COMBINED_FILE = os.path.join(_HERE, "data", "processed", "climate_assam_points.csv")
CLUSTER_FILE  = os.path.join(_HERE, "data", "processed", "clustering",
                              "cluster_assignments_assam.csv")
PREPROC_DIR   = os.path.join(_HERE, "data", "preprocessed", "parquet")
PLOT_DIR      = os.path.join(_HERE, "data", "plots")

for _sub in ["maps", "timeseries", "statistics", "features", "solar_resource"]:
    os.makedirs(os.path.join(PLOT_DIR, _sub), exist_ok=True)

print("=" * 68)
print("  ERA5 Assam - Visualization Pipeline")
print(f"  Output : {PLOT_DIR}/")
print("=" * 68)


# ==========================================================================
# LOAD & HARMONISE DATA
# (mirrors 05_plot_tamilnadu.py LOAD DATA section)
# ==========================================================================
print("\nLoading data ...")

df = pd.read_csv(
    COMBINED_FILE,
    engine="python",        # handles quoted commas (same as TN friend's code)
    on_bad_lines="warn",
)

# ------------------------------------------------------------------
# Column rename: strip era5_ prefix so all downstream code matches TN
# ------------------------------------------------------------------
rename_map = {
    "era5_GHI":           "GHI",
    "era5_DNI":           "DNI",
    "era5_DHI":           "DHI",
    "era5_CSI":           "CSI",
    "era5_GHI_clearsky":  "GHI_clearsky",
    "era5_T_amb":         "T_amb",
    "era5_T_dew":         "T_dew",
    "era5_RHum":          "RHum",
    "era5_W_spd":         "W_spd",
    "era5_W_dir":         "W_dir",
    "era5_cloud_cover":   "cloud_cover",
    "era5_LW_down":       "LW_down",
    "era5_P_atm":         "P_atm",
    "era5_precipitation": "precipitation",
    "era5_SZA":           "SZA",
    "era5_solar_azimuth": "solar_azimuth",
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# ------------------------------------------------------------------
# Timestamp (Assam combined file uses 'time_utc' not 'timestamp')
# ------------------------------------------------------------------
if "time_utc" in df.columns:
    df["timestamp"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
elif "date" in df.columns:
    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")

df = df[df["timestamp"].notna()].copy()
df = df.sort_values(["point_id", "timestamp"]).reset_index(drop=True)

# Derived time columns
if "hour" not in df.columns:
    df["hour"] = df["timestamp"].dt.hour
if "month" not in df.columns:
    df["month"] = df["timestamp"].dt.month
df["date"] = df["timestamp"].dt.date

# Season (Assam 4-season calendar; friend's TN uses Winter/Summer/Monsoon/Retreat)
if "season" not in df.columns:
    _sm = {
        12: "Winter",       1: "Winter",       2: "Winter",
         3: "Pre-Monsoon",  4: "Pre-Monsoon",  5: "Pre-Monsoon",
         6: "Monsoon",      7: "Monsoon",       8: "Monsoon",   9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    df["season"] = df["month"].map(_sm)

# ------------------------------------------------------------------
# RRTDHS proxy (same formula as friend's 02_combine_tamilnadu.py)
# RRTDHS = GHI / daily_max_GHI, clipped [0,1]
# ------------------------------------------------------------------
if "RRTDHS" not in df.columns:
    _day_max = df.groupby(["point_id", df["timestamp"].dt.date])["GHI"].transform("max")
    df["RRTDHS"] = (df["GHI"] / _day_max.replace(0, np.nan)).fillna(0).clip(0, 1)

if "high_solar_resource" not in df.columns:
    df["high_solar_resource"] = (df["GHI"] > 400).astype(int)

# ------------------------------------------------------------------
# Load GMM cluster assignments (climate zone proxy)
# TN uses a named 'climate_zone' column baked into the CSV;
# Assam derives it from the separate GMM output file.
# ------------------------------------------------------------------
print("  Loading cluster assignments ...")
if os.path.exists(CLUSTER_FILE):
    cl = pd.read_csv(CLUSTER_FILE)[["point_id", "cluster_id", "max_membership_prob"]]
    df = df.merge(cl, on="point_id", how="left")
    df["cluster_id"]   = df["cluster_id"].fillna(-1).astype(int)
    df["climate_zone"] = "Cluster-" + df["cluster_id"].astype(str)
else:
    print("  [WARN] Cluster file not found - assigning 'Unknown'")
    df["climate_zone"] = "Unknown"
    df["cluster_id"]   = -1

# city alias (TN groups by 'city'; Assam groups by 'point_id')
df["city"] = df["point_id"]

print(f"  Combined : {len(df):,} rows  |  {df['point_id'].nunique()} grid points")

# ------------------------------------------------------------------
# Point-level summary (mirrors city_summary in TN friend's code)
# ------------------------------------------------------------------
city_summary = df.groupby("city").agg(
    lat            = ("lat",                "first"),
    lon            = ("lon",                "first"),
    climate_zone   = ("climate_zone",       "first"),
    cluster_id     = ("cluster_id",         "first"),
    GHI_mean       = ("GHI",               "mean"),
    T_amb_mean     = ("T_amb",             "mean"),
    RRTDHS_mean    = ("RRTDHS",            "mean"),
    high_solar_pct = ("high_solar_resource","mean"),
    GHI_max        = ("GHI",               "max"),
).reset_index()

# Attach population from grid file if not already merged
_pop_file = os.path.join(_HERE, "data", "processed", "population_grid_points.csv")
if "population" not in city_summary.columns and os.path.exists(_pop_file):
    _pop = pd.read_csv(_pop_file).rename(columns={"point_id": "city"})
    city_summary = city_summary.merge(_pop[["city", "population"]], on="city", how="left")

# ------------------------------------------------------------------
# Colour palettes
# TN uses CZ_COLORS dict keyed by named climate zone strings;
# Assam uses cluster IDs 0-3 (from GMM, 4-cluster optimal BIC).
# ------------------------------------------------------------------
CLUSTER_COLORS = {
    "Cluster-0": "#0077b6",   # 24 pts — northeast Assam
    "Cluster-1": "#f77f00",   # 52 pts — central/most-common regime
    "Cluster-2": "#2dc653",   # 11 pts — southern tip (Barak valley)
    "Cluster-3": "#e63946",   # 41 pts — Brahmaputra valley / Guwahati belt
    "Unknown":   "#888888",
}
SEASON_COLORS = {
    "Winter":       "#4cc9f0",
    "Pre-Monsoon":  "#f9c74f",
    "Monsoon":      "#06d6a0",
    "Post-Monsoon": "#f3722c",
}

ASSAM_CENTER = [26.2, 92.5]   # geographic centre of Assam
print("  Data loaded\n")


# ==========================================================================
# HELPER (same as TN friend's code)
# ==========================================================================
def _save_and_show(fmap, filepath, label):
    fmap.save(filepath)
    print(f"    Saved: {filepath}")
    if sys.stdout.isatty():
        webbrowser.open("file://" + os.path.abspath(filepath))


# ==========================================================================
# A. FOLIUM INTERACTIVE MAPS
# (same structure as A0-A4 in 05_plot_tamilnadu.py)
# ==========================================================================
print("[A] Creating Folium interactive maps ...")

# -- A0. All grid points overview -----------------------------------------
print("  A0. All grid points overview map ...")

m0 = folium.Map(location=ASSAM_CENTER, zoom_start=7, tiles="CartoDB positron")
cluster_layer = MarkerCluster(name="All grid points").add_to(m0)

for _, row in city_summary.iterrows():
    cz_col = CLUSTER_COLORS.get(row["climate_zone"], "#888888")
    popup_html = (
        "<div style='font-family:Arial;font-size:13px;width:240px;'>"
        f"<b style='font-size:14px;'>{row['city']}</b><br>"
        "<hr style='margin:3px 0'>"
        f"Climate cluster : <b>{row['climate_zone']}</b><br>"
        "<hr style='margin:3px 0'>"
        f"Mean GHI  : <b>{row['GHI_mean']:.1f} W/m2</b><br>"
        f"Max GHI   : {row['GHI_max']:.0f} W/m2<br>"
        f"Mean Temp : {row['T_amb_mean']:.1f} C<br>"
        f"High solar: {row['high_solar_pct']*100:.0f}% of timesteps<br>"
        f"Lat/Lon   : {row['lat']:.3f}N, {row['lon']:.3f}E"
        "</div>"
    )
    folium.CircleMarker(
        location     = [row["lat"], row["lon"]],
        radius       = 7,
        color        = "white",
        weight       = 1.2,
        fill         = True,
        fill_color   = cz_col,
        fill_opacity = 0.9,
        popup        = folium.Popup(popup_html, max_width=270),
        tooltip      = f"{row['city']} | {row['climate_zone']} | GHI {row['GHI_mean']:.0f} W/m2",
    ).add_to(cluster_layer)

plain_layer = folium.FeatureGroup(name="Individual markers (no cluster)", show=False)
for _, row in city_summary.iterrows():
    cz_col = CLUSTER_COLORS.get(row["climate_zone"], "#888888")
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=5,
        color="white", weight=0.8, fill=True,
        fill_color=cz_col, fill_opacity=0.85,
        tooltip=row["city"],
    ).add_to(plain_layer)
plain_layer.add_to(m0)

_legend = (
    "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
    "background:white;padding:14px;border-radius:10px;"
    "box-shadow:2px 2px 10px rgba(0,0,0,0.3);font-family:Arial;font-size:12px;'>"
    "<b>ERA5 Assam</b><br>"
    f"<span style='color:#555;font-size:11px;'>All {len(city_summary)} grid points</span><br>"
    "<hr style='margin:6px 0'>"
    "<b>Climate Clusters (GMM, k=4)</b><br>"
)
for cz, col in CLUSTER_COLORS.items():
    cnt = (city_summary["climate_zone"] == cz).sum()
    if cnt == 0: continue
    _legend += (
        f"<span style='display:inline-block;width:11px;height:11px;"
        f"background:{col};border-radius:50%;margin-right:5px;'></span>"
        f"{cz} <span style='color:#888'>({cnt})</span><br>"
    )
_legend += "</div>"
m0.get_root().html.add_child(folium.Element(_legend))
folium.LayerControl(collapsed=False).add_to(m0)
_save_and_show(m0, os.path.join(PLOT_DIR, "maps", "A0_all_grid_points_overview.html"), "A0")


# -- A1. GHI Mean Map (colour + size by GHI + heatmap layer) --------------
print("  A1. GHI mean spatial map ...")

m1 = folium.Map(location=ASSAM_CENTER, zoom_start=7, tiles="CartoDB positron")
ghi_min, ghi_max = city_summary["GHI_mean"].min(), city_summary["GHI_mean"].max()
colormap_ghi = cm.LinearColormap(
    colors=["#2d6a4f", "#52b788", "#d9ed92", "#f9c74f", "#f3722c"],
    vmin=ghi_min, vmax=ghi_max, caption="Mean GHI (W/m2)"
)
colormap_ghi.add_to(m1)

for _, row in city_summary.iterrows():
    ghi    = row["GHI_mean"]
    radius = 6 + (ghi - ghi_min) / max(ghi_max - ghi_min, 1) * 10
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=radius,
        color="white", weight=1, fill=True,
        fill_color=colormap_ghi(ghi), fill_opacity=0.85,
        popup=folium.Popup(
            "<div style='font-family:Arial;width:220px;font-size:13px;'>"
            f"<b style='font-size:15px;'>{row['city']}</b><br>"
            f"<hr style='margin:4px 0'>Cluster: {row['climate_zone']}<br>"
            f"<b>Mean GHI: {ghi:.1f} W/m2</b><br>"
            f"Max GHI: {row['GHI_max']:.0f} W/m2<br>"
            f"Mean Temp: {row['T_amb_mean']:.1f} C<br>"
            f"High solar: {row['high_solar_pct']*100:.0f}% of timesteps</div>",
            max_width=250),
        tooltip=f"{row['city']}: {ghi:.1f} W/m2",
    ).add_to(m1)

heat_data = [[r["lat"], r["lon"], r["GHI_mean"]] for _, r in city_summary.iterrows()]
HeatMap(heat_data, radius=30, blur=20, min_opacity=0.4, name="GHI Heatmap").add_to(m1)
folium.LayerControl().add_to(m1)
_save_and_show(m1, os.path.join(PLOT_DIR, "maps", "A1_GHI_mean_map.html"), "A1")


# -- A2. Climate Cluster Map ----------------------------------------------
print("  A2. Climate cluster map ...")

m2 = folium.Map(location=ASSAM_CENTER, zoom_start=7, tiles="CartoDB positron")
for cz, color in CLUSTER_COLORS.items():
    sub = city_summary[city_summary["climate_zone"] == cz]
    for _, row in sub.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=7,
            color="white", weight=1, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(
                "<div style='font-family:Arial;font-size:13px;width:200px;'>"
                f"<b>{row['city']}</b><br>Cluster: <b>{cz}</b><br>"
                f"Mean GHI: {row['GHI_mean']:.1f} W/m2</div>",
                max_width=230),
            tooltip=f"{row['city']} | {cz}",
        ).add_to(m2)

_leg2 = (
    "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
    "background:white;padding:12px;border-radius:8px;"
    "box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-family:Arial;font-size:12px;'>"
    "<b>Climate Clusters (GMM k=4)</b><br>"
)
for cz, col in CLUSTER_COLORS.items():
    if (city_summary["climate_zone"] == cz).sum() == 0: continue
    _leg2 += (f"<span style='display:inline-block;width:12px;height:12px;"
               f"background:{col};border-radius:50%;margin-right:6px;'></span>{cz}<br>")
_leg2 += "</div>"
m2.get_root().html.add_child(folium.Element(_leg2))
_save_and_show(m2, os.path.join(PLOT_DIR, "maps", "A2_climate_cluster_map.html"), "A2")


# -- A3. Population-weighted solar resource (dark basemap) ----------------
print("  A3. Population-weighted solar resource map ...")

m3 = folium.Map(location=ASSAM_CENTER, zoom_start=7, tiles="CartoDB dark_matter")
colormap_res = cm.LinearColormap(
    colors=["#023e8a", "#0096c7", "#ade8f4", "#f9c74f", "#f3722c"],
    vmin=city_summary["GHI_mean"].min(),
    vmax=city_summary["GHI_mean"].max(),
    caption="Mean GHI (W/m2)"
)
colormap_res.add_to(m3)

for _, row in city_summary.iterrows():
    pop_val = row.get("population", 0) or 0
    r_size  = 6 + min(pop_val / 200_000, 1) * 14
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=r_size,
        color="white", weight=1.5, fill=True,
        fill_color=colormap_res(row["GHI_mean"]), fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{row['city']}</b><br>"
            f"Mean GHI: {row['GHI_mean']:.1f} W/m2<br>"
            f"High solar: {row['high_solar_pct']*100:.0f}% of timesteps<br>"
            f"Population: {int(pop_val):,}",
            max_width=220),
        tooltip=f"{row['city']}: {row['GHI_mean']:.1f} W/m2  pop={int(pop_val):,}",
    ).add_to(m3)
_save_and_show(m3, os.path.join(PLOT_DIR, "maps", "A3_population_solar_resource.html"), "A3")


# -- A4. National context map ---------------------------------------------
print("  A4. National context map ...")

m4 = folium.Map(location=[22.5, 82.0], zoom_start=5, tiles="CartoDB positron")
for _, row in city_summary.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=4,
        color="#1d3557", weight=0.5, fill=True,
        fill_color="#e63946", fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['city']}</b><br>Cluster: {row['climate_zone']}<br>"
            f"Mean GHI: {row['GHI_mean']:.1f} W/m2",
            max_width=200),
        tooltip=row["city"],
    ).add_to(m4)

m4.fit_bounds([[city_summary["lat"].min() - 0.5, city_summary["lon"].min() - 0.5],
               [city_summary["lat"].max() + 0.5, city_summary["lon"].max() + 0.5]])
_save_and_show(m4, os.path.join(PLOT_DIR, "maps", "A4_assam_india_context.html"), "A4")


# ==========================================================================
# B. TIME SERIES PLOTS
# (mirrors B1-B5 in 05_plot_tamilnadu.py)
# ==========================================================================
print("\n[B] Time series plots ...")

df["date_dt"] = pd.to_datetime(df["date"])

# -- B1. Daily GHI by cluster (7-day rolling mean) ------------------------
print("  B1. Daily GHI by cluster ...")

daily_cz = df.groupby(["date_dt", "climate_zone"])["GHI"].mean().reset_index()
clusters_plot = sorted(daily_cz["climate_zone"].unique())
palette = {cz: CLUSTER_COLORS.get(cz, "#888") for cz in clusters_plot}

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
for cz in clusters_plot:
    sub = daily_cz[daily_cz["climate_zone"] == cz]
    ax.plot(sub["date_dt"], sub["GHI"].rolling(7, min_periods=1).mean(),
            label=cz, color=palette[cz], linewidth=1.4, alpha=0.85)
ax.set_title("Daily Mean GHI - Assam Clusters (7-day rolling mean)",
             color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#aaaaaa"); ax.set_ylabel("GHI (W/m2)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["bottom","left"]: ax.spines[s].set_color("#333333")
ax.legend(loc="upper right", fontsize=9, ncol=2,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B1_daily_GHI_clusters.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B1 saved")


# -- B2. GHI vs clearsky GHI (3 sample grid points, June week 1) ----------
print("  B2. GHI vs clearsky GHI ...")

if "GHI_clearsky" in df.columns:
    _all_pts   = df["point_id"].unique().tolist()
    sample_pts = _all_pts[:3]

    fig, axes = plt.subplots(len(sample_pts), 1,
                             figsize=(16, 4 * len(sample_pts)), squeeze=False)
    fig.patch.set_facecolor("#0d1117")
    for ax, pt in zip(axes[:, 0], sample_pts):
        ax.set_facecolor("#111827")
        sub  = df[df["point_id"] == pt]
        june = sub[sub["timestamp"].dt.month == 6]
        if len(june) >= 3:
            week = june[june["timestamp"] < june["timestamp"].iloc[0] + pd.Timedelta("7d")]
        else:
            week = sub.head(21)
        ax.fill_between(week["timestamp"], week["GHI_clearsky"], alpha=0.3, color="#f9c74f")
        ax.plot(week["timestamp"], week["GHI_clearsky"],
                color="#f9c74f", linewidth=0.8, alpha=0.7, label="Clearsky GHI")
        ax.fill_between(week["timestamp"], week["GHI"], alpha=0.5, color="#4cc9f0")
        ax.plot(week["timestamp"], week["GHI"],
                color="#4cc9f0", linewidth=1.2, label="ERA5 GHI")
        ax.set_title(f"{pt} - GHI vs Clearsky GHI (June, week 1)",
                     color="white", fontsize=12)
        ax.set_ylabel("W/m2", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333", fontsize=9)
        ax.grid(color="#1f2937", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B2_GHI_vs_clearsky.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print("    B2 saved")
else:
    print("    [SKIP] GHI_clearsky not in data")


# -- B3. Temperature vs GHI scatter (by cluster, daytime) -----------------
print("  B3. T_amb vs GHI scatter ...")

_df_day = (df[df["SZA"] < 85] if "SZA" in df.columns else df[df["GHI"] > 10])
df_scat = _df_day.sample(min(20_000, len(_df_day)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for cz, col in CLUSTER_COLORS.items():
    sub = df_scat[df_scat["climate_zone"] == cz]
    if len(sub) == 0: continue
    ax.scatter(sub["T_amb"], sub["GHI"], c=col, label=cz,
               alpha=0.35, s=7, edgecolors="none")
ax.set_title("Air Temperature vs GHI - by Cluster (daytime)",
             color="white", fontsize=13)
ax.set_xlabel("T_amb (deg C)", color="#aaaaaa"); ax.set_ylabel("GHI (W/m2)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.legend(fontsize=8, markerscale=3,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B3_Tamb_vs_GHI_scatter.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B3 saved")


# -- B4. Annual GHI cycle by cluster ---------------------------------------
print("  B4. Annual cycle GHI ...")

monthly_cz = df.groupby(["month", "climate_zone"])["GHI"].mean().reset_index()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for cz, col in CLUSTER_COLORS.items():
    sub = monthly_cz[monthly_cz["climate_zone"] == cz].sort_values("month")
    if len(sub) == 0: continue
    ax.plot(sub["month"], sub["GHI"], marker="o", color=col,
            label=cz, linewidth=2, markersize=5)
ax.axvspan(6, 9, alpha=0.07, color="#4cc9f0")    # Monsoon band
ax.axvspan(3, 5, alpha=0.07, color="#f9c74f")    # Pre-Monsoon band
ax.set_xticks(range(1, 13)); ax.set_xticklabels(month_labels, color="#aaaaaa")
ax.set_title("Annual GHI Cycle - by Climate Cluster (Assam)",
             color="white", fontsize=13)
ax.set_xlabel("Month", color="#aaaaaa"); ax.set_ylabel("Mean GHI (W/m2)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B4_annual_cycle_GHI.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B4 saved")


# -- B5. All points daily overlay ------------------------------------------
print("  B5. Daily GHI - all grid points overlay ...")

daily_city = df.groupby(["date_dt", "point_id"])["GHI"].mean().reset_index()
all_pts    = sorted(daily_city["point_id"].unique())

fig, ax = plt.subplots(figsize=(17, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
for pt in all_pts:
    sub = daily_city[daily_city["point_id"] == pt]
    ax.plot(sub["date_dt"], sub["GHI"].rolling(7, min_periods=1).mean(),
            color="#4cc9f0", linewidth=0.45, alpha=0.12)
state_daily = daily_city.groupby("date_dt")["GHI"].mean().reset_index()
ax.plot(state_daily["date_dt"],
        state_daily["GHI"].rolling(7, min_periods=1).mean(),
        color="#f9c74f", linewidth=2.8, label="Assam mean (7-day rolling)")
ax.set_title(f"Daily Mean GHI - All {len(all_pts)} Assam Grid Points",
             color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#aaaaaa"); ax.set_ylabel("GHI (W/m2)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["bottom","left"]: ax.spines[s].set_color("#333333")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries", "B5_daily_GHI_all_points.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B5 saved")


# ==========================================================================
# C. STATISTICAL PLOTS
# (mirrors C1-C4 in 05_plot_tamilnadu.py)
# ==========================================================================
print("\n[C] Statistical plots ...")

# -- C1. Feature correlation matrix (daytime only) ------------------------
print("  C1. Correlation matrix ...")

corr_cols = [c for c in
             ["GHI","DNI","DHI","CSI","GHI_clearsky","T_amb","T_dew",
              "RHum","W_spd","cloud_cover","LW_down","P_atm",
              "precipitation","SZA","RRTDHS"]
             if c in df.columns]
_day = (df[df["SZA"] < 85] if "SZA" in df.columns else df[df["GHI"] > 0])
df_corr = _day[corr_cols].sample(min(50_000, len(_day)), random_state=42)

fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
sns.heatmap(df_corr.corr(), ax=ax, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="#1f2937",
            annot_kws={"size":8, "color":"white"},
            cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Matrix (daytime only) - Assam",
             color="white", fontsize=13, pad=15)
ax.tick_params(colors="#cccccc", labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "statistics", "C1_correlation_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    C1 saved")


# -- C2. GHI violin by cluster --------------------------------------------
print("  C2. GHI violin by cluster ...")

df_vio   = df[df["GHI"] > 10].copy()
cz_order = sorted(df_vio["climate_zone"].dropna().unique())

if len(cz_order) > 0:
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
    parts = ax.violinplot(
        [df_vio[df_vio["climate_zone"] == cz]["GHI"].values for cz in cz_order],
        positions=range(len(cz_order)), showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"],
                       [CLUSTER_COLORS.get(cz, "#888") for cz in cz_order]):
        pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor("white")
    for k in ["cmedians","cmins","cmaxes","cbars"]:
        parts[k].set_color("white" if k == "cmedians" else "#aaaaaa")
    ax.set_xticks(range(len(cz_order)))
    ax.set_xticklabels(cz_order, rotation=20, ha="right", color="#cccccc", fontsize=10)
    ax.set_title("GHI Distribution by Climate Cluster (daytime) - Assam",
                 color="white", fontsize=13)
    ax.set_ylabel("GHI (W/m2)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values(): sp.set_color("#333333")
    ax.grid(axis="y", color="#1f2937", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "statistics", "C2_GHI_violin_cluster.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print("    C2 saved")


# -- C3. Diurnal GHI profile by season ------------------------------------
print("  C3. Diurnal profile ...")

hourly_s = df.groupby(["hour", "season"])["GHI"].mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for season, col in SEASON_COLORS.items():
    sub = hourly_s[hourly_s["season"] == season]
    if len(sub) == 0: continue
    ax.plot(sub["hour"], sub["GHI"], color=col, label=season,
            linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(sub["hour"], sub["GHI"], alpha=0.15, color=col)
ax.set_title("Diurnal GHI Profile - by Season (Assam average)",
             color="white", fontsize=13)
ax.set_xlabel("Hour of Day (UTC)", color="#aaaaaa")
ax.set_ylabel("Mean GHI (W/m2)", color="#aaaaaa")
ax.set_xticks(range(0, 24)); ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "statistics", "C3_diurnal_profile.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    C3 saved")


# -- C4. Cloud cover vs GHI 2D density ------------------------------------
print("  C4. Cloud vs GHI density ...")

if "cloud_cover" in df.columns:
    df_samp = df[df["GHI"] > 5].sample(min(30_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
    h = ax.hist2d(df_samp["cloud_cover"], df_samp["GHI"],
                  bins=60, cmap="plasma", norm=mcolors.LogNorm())
    plt.colorbar(h[3], ax=ax, label="Count (log scale)")
    ax.set_title("Cloud Cover vs GHI - 2D Density (Assam)",
                 color="white", fontsize=13)
    ax.set_xlabel("Cloud Cover (0-1)", color="#aaaaaa")
    ax.set_ylabel("GHI (W/m2)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values(): sp.set_color("#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "statistics", "C4_cloud_vs_GHI_density.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print("    C4 saved")
else:
    print("    [SKIP] cloud_cover not in data")


# ==========================================================================
# D. FEATURE ENGINEERING VERIFICATION
# (mirrors D1-D3 in 05_plot_tamilnadu.py)
# TN reads full_preprocessed.csv; Assam reads preprocessed/parquet/*.parquet
# ==========================================================================
print("\n[D] Feature engineering plots ...")

df_pre = None
if os.path.isdir(PREPROC_DIR):
    parquet_files = sorted(
        [os.path.join(PREPROC_DIR, f) for f in os.listdir(PREPROC_DIR)
         if f.endswith(".parquet")]
    )
    if parquet_files:
        _frames = [pd.read_parquet(pf) for pf in parquet_files[:5]]
        df_pre  = pd.concat(_frames, ignore_index=True)
        print(f"  Loaded {len(parquet_files)} parquet files ({len(_frames)} used for D-plots)")
    else:
        print("  [NOTE] No parquet files found - D plots will be skipped.")
        print("         Run 04_preprocess_assam.py first.")

if df_pre is not None:
    # Rename era5_* in parquet too
    df_pre.rename(columns={k: v for k, v in rename_map.items()
                            if k in df_pre.columns}, inplace=True)
    # Use bias-corrected GHI if available (Assam-specific: era5_GHI_corrected)
    if "era5_GHI_corrected" in df_pre.columns:
        df_pre.rename(columns={"era5_GHI_corrected": "GHI_corrected"}, inplace=True)
        df_pre["GHI"] = df_pre["GHI_corrected"]
    # Parse timestamp
    if "time_ist" in df_pre.columns:
        df_pre["timestamp"] = pd.to_datetime(df_pre["time_ist"], errors="coerce")
    elif "time_utc" in df_pre.columns:
        df_pre["timestamp"] = pd.to_datetime(df_pre["time_utc"], utc=True, errors="coerce")
    df_pre = df_pre.sort_values("timestamp").reset_index(drop=True)

    # D1. Lag feature correlations with GHI
    # (TN creates GHI_lag1h..24h in 04_preprocess_tamilnadu.py;
    #  Assam parquet doesn't have lags, so we fall back to raw feature corrs)
    print("  D1. Feature-GHI correlations ...")
    lag_cols = [c for c in df_pre.columns if "GHI_lag" in c or "lag" in c.lower()]
    if len(lag_cols) > 1:
        # True lag columns found
        lag_corrs = df_pre[lag_cols + ["GHI"]].corr()["GHI"].drop("GHI").sort_values()
        x_label = "Pearson Correlation"
        title   = "Lag Feature Correlation with GHI - Assam"
        fname   = "D1_lag_correlations.png"
    else:
        # Fall back to raw feature correlations (same as TN C1 but restricted scope)
        feat_cols = [c for c in df_pre.columns if c in
                     ["T_amb","T_dew","RHum","W_spd","cloud_cover",
                      "LW_down","P_atm","precipitation","SZA","GHI_clearsky","DNI","DHI"]]
        _day_pre  = df_pre[df_pre.get("GHI", pd.Series([1])) > 0].copy() if "GHI" in df_pre.columns else df_pre
        lag_corrs = _day_pre[feat_cols + ["GHI"]].corr()["GHI"].drop("GHI").sort_values()
        x_label   = "Pearson Correlation"
        title     = "Feature Correlation with GHI (post-scaling) - Assam"
        fname     = "D1_feature_correlations.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
    ax.barh(lag_corrs.index, lag_corrs.values,
            color=["#f3722c" if v < 0 else "#4cc9f0" for v in lag_corrs.values],
            alpha=0.8)
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel(x_label, color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.axvline(0, color="white", linewidth=0.8)
    for sp in ax.spines.values(): sp.set_color("#333333")
    ax.grid(axis="x", color="#1f2937", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "features", fname),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print(f"    D1 saved ({fname})")

    # D2. Rolling mean comparison
    print("  D2. Rolling mean comparison ...")
    roll_cols = [c for c in df_pre.columns if "GHI_roll" in c and "mean" in c]
    if roll_cols and "GHI" in df_pre.columns:
        city_ex = df_pre["point_id"].iloc[0] if "point_id" in df_pre.columns else None
        sub = (df_pre[df_pre["point_id"] == city_ex].head(72)
               if city_ex else df_pre.head(72))
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
        ax.plot(range(len(sub)), sub["GHI"], color="white",
                linewidth=0.8, alpha=0.5, label="Raw GHI")
        for i, col in enumerate(roll_cols[:3]):
            if col in sub.columns:
                ax.plot(range(len(sub)), sub[col],
                        color=["#f9c74f","#f3722c","#4cc9f0"][i],
                        linewidth=1.8, label=col.replace("GHI_",""))
        ax.set_title("Rolling Mean Smoothing of GHI (72-step window) - Assam",
                     color="white", fontsize=13)
        ax.set_xlabel("Time steps", color="#aaaaaa")
        ax.set_ylabel("GHI (normalized)", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
        ax.grid(color="#1f2937", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "features", "D2_rolling_mean.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(); print("    D2 saved")
    else:
        print("    [SKIP] No rolling-mean columns in parquet (D2)")

    # D3. Train / Val / Test temporal split (70/15/15 - same as TN)
    print("  D3. Train/val/test split timeline ...")
    if "GHI" in df_pre.columns and "timestamp" in df_pre.columns:
        df_ps = df_pre.sort_values("timestamp")
        n     = len(df_ps)
        t_end, v_end = int(n * 0.70), int(n * 0.85)
        ts   = df_ps["timestamp"].values
        gv   = df_ps["GHI"].values
        fig, ax = plt.subplots(figsize=(16, 4))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
        ax.plot(ts[:t_end], gv[:t_end],       color="#4cc9f0", linewidth=0.5, alpha=0.6, label="Train (70%)")
        ax.plot(ts[t_end:v_end], gv[t_end:v_end], color="#f9c74f", linewidth=0.5, alpha=0.8, label="Val (15%)")
        ax.plot(ts[v_end:], gv[v_end:],       color="#f3722c", linewidth=0.5, alpha=0.8, label="Test (15%)")
        ax.axvline(ts[t_end], color="#f9c74f", linewidth=1.5, linestyle="--")
        ax.axvline(ts[v_end], color="#f3722c", linewidth=1.5, linestyle="--")
        ax.set_title("Train / Validation / Test Temporal Split - Assam",
                     color="white", fontsize=13)
        ax.set_ylabel("GHI (W/m2)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
        ax.grid(color="#1f2937", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "features", "D3_train_val_test_split.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(); print("    D3 saved")
else:
    print("  [SKIP] Preprocessed parquet not found - D plots skipped.")
    print("         Run 04_preprocess_assam.py first.")


# ==========================================================================
# E. SOLAR RESOURCE QUALITY PLOTS
# (mirrors E1-E3 in 05_plot_tamilnadu.py)
# ==========================================================================
print("\n[E] Solar resource quality plots ...")

# E1. RRTDHS / Solar resource score heatmap (top 30 x month) -------------
print("  E1. Solar resource score heatmap ...")

top_pts = city_summary.nlargest(30, "GHI_mean")["city"].tolist()
rrtdhs_pivot = (
    df[df["city"].isin(top_pts)]
    .groupby(["city", "month"])["RRTDHS"].mean()
    .unstack(level="month")
)
rrtdhs_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")
sns.heatmap(rrtdhs_pivot, ax=ax, cmap="YlOrRd",
            linewidths=0.3, linecolor="#1f2937",
            annot=True, fmt=".2f", annot_kws={"size":7},
            cbar_kws={"label":"Solar Resource Score","shrink":0.8})
ax.set_title("Solar Resource Score - Top 30 Grid Points by Month (Assam)",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="#cccccc", labelsize=8)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "solar_resource", "E1_solar_resource_heatmap.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    E1 saved")


# E2. CSI distribution (overall + by season) ------------------------------
print("  E2. CSI distribution ...")

if "CSI" in df.columns:
    df_csi = df[(df["CSI"] > 0) & (df["CSI"] <= 1.5) & (df["GHI"] > 10)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.set_facecolor("#111827")

    axes[0].hist(df_csi["CSI"], bins=60, color="#4cc9f0", alpha=0.8, edgecolor="none")
    axes[0].axvline(1.0, color="#f9c74f", linewidth=1.5,
                    linestyle="--", label="Perfect clear sky")
    axes[0].set_title("Clear Sky Index (CSI) Distribution - Assam",
                      color="white", fontsize=12)
    axes[0].set_xlabel("CSI (0=cloudy, 1=clear, >1=enhancement)", color="#aaaaaa")
    axes[0].set_ylabel("Count", color="#aaaaaa"); axes[0].tick_params(colors="#aaaaaa")
    for sp in axes[0].spines.values(): sp.set_color("#333333")
    axes[0].grid(axis="y", color="#1f2937", linewidth=0.5)
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")

    for season, col in SEASON_COLORS.items():
        sub = (df_csi[df_csi["season"] == season]["CSI"]
               if "season" in df_csi.columns else pd.Series(dtype=float))
        if len(sub) == 0: continue
        axes[1].hist(sub, bins=50, alpha=0.5, color=col, label=season, edgecolor="none")
    axes[1].set_title("CSI Distribution by Season - Assam", color="white", fontsize=12)
    axes[1].set_xlabel("CSI", color="#aaaaaa"); axes[1].tick_params(colors="#aaaaaa")
    for sp in axes[1].spines.values(): sp.set_color("#333333")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    axes[1].grid(axis="y", color="#1f2937", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "solar_resource", "E2_CSI_distribution.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print("    E2 saved")
else:
    print("    [SKIP] CSI column not in data")


# E3. Top 20 grid points by mean GHI (colour = cluster) -------------------
print("  E3. Top 20 grid points by GHI ...")

top20 = city_summary.nlargest(20, "GHI_mean").sort_values("GHI_mean")
fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
bars = ax.barh(top20["city"], top20["GHI_mean"],
               color=[CLUSTER_COLORS.get(cz, "#888") for cz in top20["climate_zone"]],
               alpha=0.85, edgecolor="none")
for bar, val in zip(bars, top20["GHI_mean"]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", ha="left", color="white", fontsize=8)
ax.set_title("Top 20 Grid Points - Mean GHI (W/m2) - Assam",
             color="white", fontsize=13)
ax.set_xlabel("Mean GHI (W/m2)", color="#aaaaaa")
ax.tick_params(colors="#cccccc", labelsize=9)
for sp in ax.spines.values(): sp.set_color("#333333")
ax.grid(axis="x", color="#1f2937", linewidth=0.5)
shown_cz = top20["climate_zone"].unique()
ax.legend(
    [plt.Rectangle((0,0),1,1, color=CLUSTER_COLORS.get(cz,"#888")) for cz in shown_cz],
    shown_cz, loc="lower right", fontsize=8,
    facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "solar_resource", "E3_top20_GHI_points.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    E3 saved")


# ==========================================================================
# FINAL SUMMARY
# ==========================================================================
print("\n" + "=" * 68)
print("  ALL PLOTS COMPLETE")
print(f"\n  Saved to: {PLOT_DIR}/")
print("""
  maps/
    A0_all_grid_points_overview.html       <- all 128 points, colour=cluster
    A1_GHI_mean_map.html                   <- colour + size + heatmap by GHI
    A2_climate_cluster_map.html            <- GMM k=4 cluster colours
    A3_population_solar_resource.html      <- population-weighted (dark map)
    A4_assam_india_context.html            <- national context

  timeseries/
    B1_daily_GHI_clusters.png
    B2_GHI_vs_clearsky.png
    B3_Tamb_vs_GHI_scatter.png
    B4_annual_cycle_GHI.png
    B5_daily_GHI_all_points.png

  statistics/
    C1_correlation_matrix.png
    C2_GHI_violin_cluster.png
    C3_diurnal_profile.png
    C4_cloud_vs_GHI_density.png

  features/   (require preprocessed parquet from 04_preprocess_assam.py)
    D1_lag_correlations.png  or  D1_feature_correlations.png
    D2_rolling_mean.png          (if rolling cols present in parquet)
    D3_train_val_test_split.png

  solar_resource/
    E1_solar_resource_heatmap.png
    E2_CSI_distribution.png
    E3_top20_GHI_points.png
""")
print("  HTML maps open automatically in your default browser (local).")
print("=" * 68)
