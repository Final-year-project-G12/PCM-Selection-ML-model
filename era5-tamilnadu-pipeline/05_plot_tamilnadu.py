# ╔══════════════════════════════════════════════════════════════════════╗
# ║  05_plot_tamilnadu.py                                                ║
# ║  ERA5 Tamil Nadu — All Visualizations                                ║
# ║                                                                      ║
# ║  Based on: "Multimodal Learning Techniques for Time Series           ║
# ║  Forecasting in Renewable Energy Systems"                            ║
# ║  Mansouri et al., IEEE Access 2025                                   ║
# ║                                                                      ║
# ║  HOW TO RUN                                                          ║
# ║  ──────────────────────────────────────────────────────────────────  ║
# ║  Option A — VS Code / Jupyter (.ipynb)                               ║
# ║    1. Set COLAB = False (default)                                    ║
# ║    2. Ensure climate_tamilnadu_all.csv is at data/processed/         ║
# ║    3. Run 04_preprocess_tamilnadu.py first (for D-plots)             ║
# ║    4. Run this file                                                  ║
# ║    5. HTML maps open automatically in your browser                   ║
# ║                                                                      ║
# ║  Option B — Google Colab                                             ║
# ║    1. Upload climate_tamilnadu_all.csv to /content/                  ║
# ║    2. Set COLAB = True                                               ║
# ║    3. Run all cells                                                  ║
# ║    4. HTML maps are displayed inline in the notebook                 ║
# ║       PNG plots are saved to /content/data/plots/                   ║
# ║                                                                      ║
# ║  PLOTS PRODUCED                                                      ║
# ║  ──────────────────────────────────────────────────────────────────  ║
# ║  A. Folium interactive maps (HTML — all 222 locations)               ║
# ║     A0. All 222 data locations — plain overview map        [NEW]     ║
# ║     A1. GHI mean spatial map (colour + size by GHI)                  ║
# ║     A2. Climate zone map (colour by climate zone)                    ║
# ║     A3. District solar resource map                                  ║
# ║     A4. All locations on India map (national context)                ║
# ║  B. Time series plots (PNG)                                          ║
# ║     B1. Daily mean GHI — all districts                               ║
# ║     B2. GHI vs clearsky GHI — sample cities                         ║
# ║     B3. Temperature vs GHI scatter by climate zone                  ║
# ║     B4. Annual cycle — monthly mean GHI by climate zone             ║
# ║     B5. Daily mean GHI — ALL 222 cities overlay                      ║
# ║  C. Statistical plots (PNG)                                          ║
# ║     C1. Correlation matrix                                           ║
# ║     C2. GHI violin by climate zone                                   ║
# ║     C3. Diurnal profile by season                                    ║
# ║     C4. Cloud cover vs GHI 2D density                               ║
# ║  D. Feature engineering verification (PNG, needs preprocessed data) ║
# ║     D1. Lag feature correlations                                     ║
# ║     D2. Rolling mean comparison                                      ║
# ║     D3. Train/val/test split timeline                                ║
# ║  E. Solar resource quality (PNG)                                     ║
# ║     E1. RRTDHS heatmap (city × month)                               ║
# ║     E2. CSI distribution                                             ║
# ║     E3. Top 20 cities by mean GHI                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 0. Mode switch ────────────────────────────────────────────────────
COLAB = False   # ← set True when running in Google Colab

# ── 1. Installs (Colab only) ──────────────────────────────────────────
if COLAB:
    import subprocess
    subprocess.run(["pip", "install", "folium", "branca", "-q"], check=False)

# ── 2. Imports ────────────────────────────────────────────────────────
import os, sys, warnings, webbrowser
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
if COLAB:
    matplotlib.use("Agg")           # Colab: save to files
else:
    matplotlib.use("Agg")           # VS Code: also save to files, view via file explorer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm

# ── 3. Path configuration ─────────────────────────────────────────────
if COLAB:
    _SEARCH = [
        "/content/climate_tamilnadu_all.csv",
        "/content/drive/MyDrive/climate_tamilnadu_all.csv",
        "/content/drive/MyDrive/tamilnadu_era5/data/processed/climate_tamilnadu_all.csv",
    ]
    COMBINED_FILE = next((p for p in _SEARCH if os.path.exists(p)), None)
    if COMBINED_FILE is None:
        raise FileNotFoundError(
            "climate_tamilnadu_all.csv not found. "
            "Upload to /content/ or mount Drive."
        )
    PREPROC_FILE = "/content/data/preprocessed/full_preprocessed.csv"
    PLOT_DIR     = "/content/data/plots"
else:
    _HERE         = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    COMBINED_FILE = os.path.join(_HERE, "data", "processed", "climate_tamilnadu_all.csv")
    PREPROC_FILE  = os.path.join(_HERE, "data", "preprocessed", "full_preprocessed.csv")
    PLOT_DIR      = os.path.join(_HERE, "data", "plots")

for _sub in ["maps","timeseries","statistics","features","solar_resource"]:
    os.makedirs(os.path.join(PLOT_DIR, _sub), exist_ok=True)

print("=" * 68)
print("  ERA5 Tamil Nadu — Visualization Pipeline")
print(f"  Output : {PLOT_DIR}/")
print("=" * 68)


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
print("\nLoading data ...")

df = pd.read_csv(
    COMBINED_FILE,
    engine="python",
    on_bad_lines="warn",
    parse_dates=["timestamp"],
)
if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df[df["timestamp"].notna()].copy()
df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

# Derive season if missing
if "season" not in df.columns and "month" in df.columns:
    _sm = {12:"Winter",1:"Winter",2:"Winter",
            3:"Summer",4:"Summer",5:"Summer",
            6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
           10:"Retreat",11:"Retreat"}
    df["season"] = df["month"].map(_sm)

# RRTDHS proxy if column missing
if "RRTDHS" not in df.columns:
    _grp = df.groupby(["city", df["timestamp"].dt.date])["GHI"].transform("max")
    df["RRTDHS"] = (df["GHI"] / _grp.replace(0, np.nan)).fillna(0).clip(0, 1)

# high_solar_resource proxy if missing
if "high_solar_resource" not in df.columns:
    df["high_solar_resource"] = (df["GHI"] > 400).astype(int)

# Load preprocessed if available
df_pre = None
if os.path.exists(PREPROC_FILE):
    df_pre = pd.read_csv(PREPROC_FILE, parse_dates=["timestamp"])
    print(f"  Preprocessed file loaded: {len(df_pre):,} rows")
else:
    print("  [NOTE] Preprocessed file not found — D-plots will be skipped.")
    print("         Run 04_preprocess_tamilnadu.py first.")

print(f"  Combined : {len(df):,} rows  |  {df['city'].nunique()} cities")

# ── City-level summary ────────────────────────────────────
city_summary = df.groupby("city").agg(
    lat           = ("lat",                "first"),
    lon           = ("lon",                "first"),
    alt           = ("altitude_m",         "first"),
    district      = ("district",           "first"),
    climate_zone  = ("climate_zone",       "first"),
    GHI_mean      = ("GHI",               "mean"),
    T_amb_mean    = ("T_amb",             "mean"),
    RRTDHS_mean   = ("RRTDHS",            "mean"),
    high_solar_pct= ("high_solar_resource","mean"),
    GHI_max       = ("GHI",               "max"),
).reset_index()

# ── Palettes ──────────────────────────────────────────────
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
SEASON_COLORS = {
    "Winter":"#4cc9f0","Summer":"#f9c74f",
    "Monsoon":"#06d6a0","Retreat":"#f3722c",
}

TN_CENTER = [10.9, 78.5]
print("  Data loaded ✓\n")


# ═══════════════════════════════════════════════════════════
# HELPER — display map inline in Colab OR open in browser
# ═══════════════════════════════════════════════════════════
def _save_and_show(fmap, filepath, label):
    fmap.save(filepath)
    print(f"    Saved: {filepath}")
    if COLAB:
        # Display inline inside the Colab notebook cell
        from IPython.display import display, IFrame
        display(IFrame(src=filepath, width="100%", height="500px"))
    elif sys.stdout.isatty():
        # Interactive run only — nbconvert has no TTY and webbrowser.open() can hang
        webbrowser.open("file://" + os.path.abspath(filepath))


# ═══════════════════════════════════════════════════════════
# A. FOLIUM INTERACTIVE MAPS
# ═══════════════════════════════════════════════════════════
print("[A] Creating Folium interactive maps ...")


# ── A0. ALL 222 LOCATIONS — plain overview  [NEW MAP] ─────
print("  A0. All 222 data locations overview map ...")

m0 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")

# MarkerCluster so dots don't overlap at low zoom
cluster = MarkerCluster(name="All 222 locations").add_to(m0)

for _, row in city_summary.iterrows():
    cz_col  = CZ_COLORS.get(str(row["climate_zone"]), "#888888")
    popup_html = (
        f"<div style='font-family:Arial;font-size:13px;width:230px;'>"
        f"<b style='font-size:14px;'>{row['city']}</b><br>"
        f"<hr style='margin:3px 0'>"
        f"District     : <b>{row['district']}</b><br>"
        f"Climate zone : {row['climate_zone']}<br>"
        f"Altitude     : {row['alt']:.0f} m<br>"
        f"<hr style='margin:3px 0'>"
        f"Mean GHI     : <b>{row['GHI_mean']:.1f} W/m²</b><br>"
        f"Max GHI      : {row['GHI_max']:.0f} W/m²<br>"
        f"Mean Temp    : {row['T_amb_mean']:.1f} °C<br>"
        f"High solar   : {row['high_solar_pct']*100:.0f}% of hours<br>"
        f"Lat/Lon      : {row['lat']:.3f}°N, {row['lon']:.3f}°E"
        f"</div>"
    )
    folium.CircleMarker(
        location   = [row["lat"], row["lon"]],
        radius     = 7,
        color      = "white",
        weight     = 1.2,
        fill       = True,
        fill_color = cz_col,
        fill_opacity = 0.9,
        popup      = folium.Popup(popup_html, max_width=260),
        tooltip    = f"📍 {row['city']}  |  {row['climate_zone']}  |  GHI {row['GHI_mean']:.0f} W/m²",
    ).add_to(cluster)

# Also add plain dots (non-clustered) as a second layer for full visibility
plain_layer = folium.FeatureGroup(name="Individual markers (no cluster)", show=False)
for _, row in city_summary.iterrows():
    cz_col = CZ_COLORS.get(str(row["climate_zone"]), "#888888")
    folium.CircleMarker(
        location   = [row["lat"], row["lon"]],
        radius     = 5,
        color      = "white",
        weight     = 0.8,
        fill       = True,
        fill_color = cz_col,
        fill_opacity = 0.85,
        tooltip    = row["city"],
    ).add_to(plain_layer)
plain_layer.add_to(m0)

# Legend box
_legend = (
    "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
    "background:white;padding:14px;border-radius:10px;"
    "box-shadow:2px 2px 10px rgba(0,0,0,0.3);font-family:Arial;font-size:12px;'>"
    f"<b>ERA5 Tamil Nadu</b><br>"
    f"<span style='color:#555;font-size:11px;'>All {len(city_summary)} monitoring locations</span><br>"
    "<hr style='margin:6px 0'>"
    "<b>Climate Zones</b><br>"
)
for cz, col in CZ_COLORS.items():
    cnt = (city_summary["climate_zone"] == cz).sum()
    if cnt == 0:
        continue
    _legend += (
        f"<span style='display:inline-block;width:11px;height:11px;"
        f"background:{col};border-radius:50%;margin-right:5px;'></span>"
        f"{cz} <span style='color:#888'>({cnt})</span><br>"
    )
_legend += "</div>"
m0.get_root().html.add_child(folium.Element(_legend))
folium.LayerControl(collapsed=False).add_to(m0)

m0_path = os.path.join(PLOT_DIR, "maps", "A0_all_222_locations_overview.html")
_save_and_show(m0, m0_path, "A0")


# ── A1. GHI Mean Map ──────────────────────────────────────
print("  A1. GHI mean spatial map ...")

m1 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")

ghi_min, ghi_max = city_summary["GHI_mean"].min(), city_summary["GHI_mean"].max()
colormap_ghi = cm.LinearColormap(
    colors=["#2d6a4f","#52b788","#d9ed92","#f9c74f","#f3722c"],
    vmin=ghi_min, vmax=ghi_max, caption="Mean GHI (W/m²)"
)
colormap_ghi.add_to(m1)

for _, row in city_summary.iterrows():
    ghi    = row["GHI_mean"]
    radius = 6 + (ghi - ghi_min) / max(ghi_max - ghi_min, 1) * 8
    popup_html = (
        f"<div style='font-family:Arial;width:220px;font-size:13px;'>"
        f"<b style='font-size:15px;'>{row['city']}</b><br>"
        f"<hr style='margin:4px 0'>"
        f"District: {row['district']}<br>Climate: {row['climate_zone']}<br>"
        f"Altitude: {row['alt']:.0f} m<br><hr style='margin:4px 0'>"
        f"<b>Mean GHI: {ghi:.1f} W/m²</b><br>"
        f"Max GHI: {row['GHI_max']:.0f} W/m²<br>"
        f"Mean Temp: {row['T_amb_mean']:.1f} °C<br>"
        f"High solar: {row['high_solar_pct']*100:.0f}% of hours</div>"
    )
    folium.CircleMarker(
        location=[row["lat"],row["lon"]], radius=radius,
        color="white", weight=1, fill=True,
        fill_color=colormap_ghi(ghi), fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['city']}: {ghi:.1f} W/m²",
    ).add_to(m1)

heat_data = [[r["lat"],r["lon"],r["GHI_mean"]] for _,r in city_summary.iterrows()]
HeatMap(heat_data, radius=30, blur=20, min_opacity=0.4, name="GHI Heatmap").add_to(m1)
folium.LayerControl().add_to(m1)

m1_path = os.path.join(PLOT_DIR, "maps", "A1_GHI_mean_map.html")
_save_and_show(m1, m1_path, "A1")


# ── A2. Climate Zone Map ───────────────────────────────────
print("  A2. Climate zone map ...")

m2 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")

for cz, color in CZ_COLORS.items():
    sub = city_summary[city_summary["climate_zone"] == cz]
    for _, row in sub.iterrows():
        folium.CircleMarker(
            location=[row["lat"],row["lon"]], radius=7,
            color="white", weight=1, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(
                f"<div style='font-family:Arial;font-size:13px;width:200px;'>"
                f"<b>{row['city']}</b><br>Climate: <b>{cz}</b><br>"
                f"District: {row['district']}<br>Mean GHI: {row['GHI_mean']:.1f} W/m²</div>",
                max_width=230),
            tooltip=f"{row['city']} | {cz}",
        ).add_to(m2)

_leg2 = (
    "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
    "background:white;padding:12px;border-radius:8px;"
    "box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-family:Arial;font-size:12px;'>"
    "<b>Climate Zones</b><br>"
)
for cz, col in CZ_COLORS.items():
    if (city_summary["climate_zone"] == cz).sum() == 0:
        continue
    _leg2 += (
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{col};border-radius:50%;margin-right:6px;'></span>{cz}<br>"
    )
_leg2 += "</div>"
m2.get_root().html.add_child(folium.Element(_leg2))

m2_path = os.path.join(PLOT_DIR, "maps", "A2_climate_zone_map.html")
_save_and_show(m2, m2_path, "A2")


# ── A3. District Solar Resource Map ───────────────────────
print("  A3. District solar resource map ...")

dist_summary = city_summary.groupby("district").agg(
    lat           =("lat","mean"),
    lon           =("lon","mean"),
    GHI_mean      =("GHI_mean","mean"),
    high_solar_pct=("high_solar_pct","mean"),
    n_cities      =("city","count"),
).reset_index()

m3 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB dark_matter")
colormap_dist = cm.LinearColormap(
    colors=["#023e8a","#0096c7","#ade8f4","#f9c74f","#f3722c"],
    vmin=dist_summary["GHI_mean"].min(),
    vmax=dist_summary["GHI_mean"].max(),
    caption="District Mean GHI (W/m²)"
)
colormap_dist.add_to(m3)

for _, row in dist_summary.iterrows():
    folium.CircleMarker(
        location=[row["lat"],row["lon"]], radius=15,
        color="white", weight=1.5, fill=True,
        fill_color=colormap_dist(row["GHI_mean"]), fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{row['district']}</b><br>"
            f"Mean GHI: {row['GHI_mean']:.1f} W/m²<br>"
            f"High solar: {row['high_solar_pct']*100:.0f}% of hours<br>"
            f"Locations: {row['n_cities']}",
            max_width=200),
        tooltip=f"{row['district']}: {row['GHI_mean']:.1f} W/m²",
    ).add_to(m3)

m3_path = os.path.join(PLOT_DIR, "maps", "A3_district_solar_resource.html")
_save_and_show(m3, m3_path, "A3")


# ── A4. All locations on India map ────────────────────────
print("  A4. All locations on India context map ...")

m4 = folium.Map(location=[22.5, 78.9], zoom_start=5, tiles="CartoDB positron")

for _, row in city_summary.iterrows():
    folium.CircleMarker(
        location=[row["lat"],row["lon"]], radius=4,
        color="#1d3557", weight=0.5, fill=True,
        fill_color="#e63946", fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['city']}</b><br>District: {row['district']}<br>"
            f"Climate: {row['climate_zone']}<br>Mean GHI: {row['GHI_mean']:.1f} W/m²",
            max_width=220),
        tooltip=row["city"],
    ).add_to(m4)

bounds = [[city_summary["lat"].min()-0.3, city_summary["lon"].min()-0.3],
          [city_summary["lat"].max()+0.3, city_summary["lon"].max()+0.3]]
m4.fit_bounds(bounds)

m4_path = os.path.join(PLOT_DIR, "maps", "A4_all_locations_india_context.html")
_save_and_show(m4, m4_path, "A4")


# ═══════════════════════════════════════════════════════════
# B. TIME SERIES PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[B] Time series plots ...")

df["date"] = df["timestamp"].dt.date

# ── B1. Daily GHI — all districts ─────────────────────────
print("  B1. Daily GHI by district ...")

daily_dist = df.groupby(["date","district"])["GHI"].mean().reset_index()
daily_dist["date"] = pd.to_datetime(daily_dist["date"])
districts_plot = daily_dist["district"].unique()[:12]
palette = sns.color_palette("tab20", len(districts_plot))

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
for i, dist in enumerate(districts_plot):
    sub = daily_dist[daily_dist["district"] == dist]
    ax.plot(sub["date"], sub["GHI"].rolling(7, min_periods=1).mean(),
            label=dist, color=palette[i], linewidth=1.2, alpha=0.85)
ax.set_title("Daily Mean GHI — Tamil Nadu Districts (7-day rolling mean)",
             color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["bottom","left"]: ax.spines[s].set_color("#333333")
ax.legend(loc="upper right", fontsize=8, ncol=2,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"timeseries","B1_daily_GHI_districts.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B1 saved ✓")


# ── B2. GHI vs clearsky GHI ───────────────────────────────
print("  B2. GHI vs clearsky GHI ...")

_all_cities = df["city"].unique().tolist()
_prefer     = ["Chennai","Coimbatore","Ooty","Madurai","Salem"]
cities_s    = [c for c in _prefer if c in _all_cities] or _all_cities[:3]

fig, axes = plt.subplots(len(cities_s), 1,
                         figsize=(16, 4*len(cities_s)), squeeze=False)
fig.patch.set_facecolor("#0d1117")
for ax, city in zip(axes[:,0], cities_s):
    ax.set_facecolor("#111827")
    sub  = df[df["city"] == city]
    june = sub[sub["timestamp"].dt.month == 6]
    week = (june[june["timestamp"] < june["timestamp"].iloc[0] + pd.Timedelta("7d")]
            if len(june) >= 24 else sub.head(168))
    if "GHI_clearsky" in week.columns:
        ax.fill_between(week["timestamp"], week["GHI_clearsky"],
                        alpha=0.3, color="#f9c74f")
        ax.plot(week["timestamp"], week["GHI_clearsky"],
                color="#f9c74f", linewidth=0.8, alpha=0.7, label="Clearsky GHI")
    ax.fill_between(week["timestamp"], week["GHI"],
                    alpha=0.5, color="#4cc9f0")
    ax.plot(week["timestamp"], week["GHI"],
            color="#4cc9f0", linewidth=1.2, label="Actual GHI")
    ax.set_title(f"{city} — GHI vs Clearsky GHI (June, week 1)",
                 color="white", fontsize=12)
    ax.set_ylabel("W/m²", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white",
              edgecolor="#333333", fontsize=9)
    ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"timeseries","B2_GHI_vs_clearsky.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B2 saved ✓")


# ── B3. Temperature vs GHI scatter ────────────────────────
print("  B3. T_amb vs GHI scatter ...")

_df_day = (df[df["SZA"] < 85] if "SZA" in df.columns else df[df["GHI"] > 10])
df_scat = _df_day.sample(min(20_000, len(_df_day)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for cz, col in CZ_COLORS.items():
    sub = df_scat[df_scat["climate_zone"] == cz]
    if len(sub) == 0: continue
    ax.scatter(sub["T_amb"], sub["GHI"], c=col, label=cz,
               alpha=0.35, s=7, edgecolors="none")
ax.set_title("Air Temperature vs GHI — by Climate Zone (daytime)",
             color="white", fontsize=13)
ax.set_xlabel("T_amb (°C)", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.legend(fontsize=7, markerscale=3,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"timeseries","B3_Tamb_vs_GHI_scatter.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B3 saved ✓")


# ── B4. Annual cycle by climate zone ──────────────────────
print("  B4. Annual cycle GHI ...")

monthly_cz  = df.groupby(["month","climate_zone"])["GHI"].mean().reset_index()
month_labels= ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for cz, col in CZ_COLORS.items():
    sub = monthly_cz[monthly_cz["climate_zone"] == cz].sort_values("month")
    if len(sub) == 0: continue
    ax.plot(sub["month"], sub["GHI"], marker="o", color=col,
            label=cz, linewidth=2, markersize=5)
ax.axvspan(6, 9, alpha=0.07, color="#4cc9f0")
ax.axvspan(3, 5, alpha=0.07, color="#f9c74f")
ax.set_xticks(range(1,13)); ax.set_xticklabels(month_labels, color="#aaaaaa")
ax.set_title("Annual GHI Cycle — by Climate Zone",
             color="white", fontsize=13)
ax.set_xlabel("Month", color="#aaaaaa"); ax.set_ylabel("Mean GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.legend(fontsize=8, facecolor="#1a1a2e",
          labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"timeseries","B4_annual_cycle_GHI.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B4 saved ✓")


# ── B5. All 222 cities daily overlay ──────────────────────
print("  B5. Daily GHI — all 222 cities overlay ...")

daily_city = df.groupby(["date","city"])["GHI"].mean().reset_index()
daily_city["date"] = pd.to_datetime(daily_city["date"])
all_cities = sorted(daily_city["city"].unique())

fig, ax = plt.subplots(figsize=(17, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
for city in all_cities:
    sub = daily_city[daily_city["city"] == city]
    ax.plot(sub["date"],
            sub["GHI"].rolling(7, min_periods=1).mean(),
            color="#4cc9f0", linewidth=0.45, alpha=0.12)
state_daily = daily_city.groupby("date")["GHI"].mean().reset_index()
ax.plot(state_daily["date"],
        state_daily["GHI"].rolling(7, min_periods=1).mean(),
        color="#f9c74f", linewidth=2.8,
        label="Tamil Nadu mean (7-day rolling)")
ax.set_title(f"Daily Mean GHI — All {len(all_cities)} Tamil Nadu Locations",
             color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#aaaaaa"); ax.set_ylabel("GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["bottom","left"]: ax.spines[s].set_color("#333333")
ax.legend(fontsize=9, facecolor="#1a1a2e",
          labelcolor="white", edgecolor="#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"timeseries","B5_daily_GHI_all_cities.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    B5 saved ✓")


# ═══════════════════════════════════════════════════════════
# C. STATISTICAL PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[C] Statistical plots ...")

# ── C1. Correlation matrix ────────────────────────────────
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
            annot_kws={"size":8,"color":"white"},
            cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Matrix (daytime only)",
             color="white", fontsize=13, pad=15)
ax.tick_params(colors="#cccccc", labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"statistics","C1_correlation_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    C1 saved ✓")


# ── C2. GHI violin by climate zone ────────────────────────
print("  C2. GHI violin by climate zone ...")

df_vio   = df[df["GHI"] > 10].copy()
cz_order = sorted(df_vio["climate_zone"].dropna().unique())

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
parts = ax.violinplot(
    [df_vio[df_vio["climate_zone"] == cz]["GHI"].values for cz in cz_order],
    positions=range(len(cz_order)), showmedians=True, showextrema=True)
for pc, col in zip(parts["bodies"],
                   [CZ_COLORS.get(cz,"#888") for cz in cz_order]):
    pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor("white")
for k in ["cmedians","cmins","cmaxes","cbars"]:
    parts[k].set_color("white" if k == "cmedians" else "#aaaaaa")
ax.set_xticks(range(len(cz_order)))
ax.set_xticklabels(cz_order, rotation=30, ha="right",
                   color="#cccccc", fontsize=9)
ax.set_title("GHI Distribution by Climate Zone (daytime)",
             color="white", fontsize=13)
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.grid(axis="y", color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"statistics","C2_GHI_violin_climate_zone.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    C2 saved ✓")


# ── C3. Diurnal profile by season ─────────────────────────
print("  C3. Diurnal profile ...")

hourly_s = df.groupby(["hour","season"])["GHI"].mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
for season, col in SEASON_COLORS.items():
    sub = hourly_s[hourly_s["season"] == season]
    if len(sub) == 0: continue
    ax.plot(sub["hour"], sub["GHI"], color=col, label=season,
            linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(sub["hour"], sub["GHI"], alpha=0.15, color=col)
ax.set_title("Diurnal GHI Profile — by Season (Tamil Nadu average)",
             color="white", fontsize=13)
ax.set_xlabel("Hour of Day (UTC)", color="#aaaaaa")
ax.set_ylabel("Mean GHI (W/m²)", color="#aaaaaa")
ax.set_xticks(range(0, 24)); ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
ax.grid(color="#1f2937", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"statistics","C3_diurnal_profile.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    C3 saved ✓")


# ── C4. Cloud cover vs GHI 2D density ─────────────────────
print("  C4. Cloud vs GHI density ...")

df_samp = df[df["GHI"] > 5].sample(min(30_000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
h = ax.hist2d(df_samp["cloud_cover"], df_samp["GHI"],
              bins=60, cmap="plasma", norm=mcolors.LogNorm())
plt.colorbar(h[3], ax=ax, label="Count (log scale)")
ax.set_title("Cloud Cover vs GHI — 2D Density",
             color="white", fontsize=13)
ax.set_xlabel("Cloud Cover (0–1)", color="#aaaaaa")
ax.set_ylabel("GHI (W/m²)", color="#aaaaaa"); ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333333")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"statistics","C4_cloud_vs_GHI_density.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    C4 saved ✓")


# ═══════════════════════════════════════════════════════════
# D. FEATURE ENGINEERING VERIFICATION (needs preprocessed data)
# ═══════════════════════════════════════════════════════════
print("\n[D] Feature engineering plots ...")

if df_pre is not None:

    # D1. Lag correlations
    print("  D1. Lag feature correlations ...")
    lag_cols = [c for c in df_pre.columns if "GHI_lag" in c or c == "GHI"]
    if len(lag_cols) > 1:
        lag_corrs = df_pre[lag_cols].corr()["GHI"].drop("GHI").sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
        ax.barh(lag_corrs.index, lag_corrs.values,
                color=["#f3722c" if v < 0 else "#4cc9f0" for v in lag_corrs.values],
                alpha=0.8)
        ax.set_title("Lag Feature Correlation with GHI",
                     color="white", fontsize=13)
        ax.set_xlabel("Pearson Correlation", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.axvline(0, color="white", linewidth=0.8)
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.grid(axis="x", color="#1f2937", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,"features","D1_lag_correlations.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(); print("    D1 saved ✓")

    # D2. Rolling mean comparison
    print("  D2. Rolling mean comparison ...")
    roll_cols = [c for c in df_pre.columns if "GHI_roll" in c and "mean" in c]
    if roll_cols and "GHI" in df_pre.columns:
        city_ex = df_pre["city"].iloc[0] if "city" in df_pre.columns else None
        sub = (df_pre[df_pre["city"] == city_ex].head(72) if city_ex
               else df_pre.head(72))
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
        ax.plot(range(len(sub)), sub["GHI"], color="white",
                linewidth=0.8, alpha=0.5, label="Raw GHI")
        for i, col in enumerate(roll_cols[:3]):
            if col in sub.columns:
                ax.plot(range(len(sub)), sub[col],
                        color=["#f9c74f","#f3722c","#4cc9f0"][i],
                        linewidth=1.8, label=col.replace("GHI_",""))
        ax.set_title("Rolling Mean Smoothing of GHI (72-hour window)",
                     color="white", fontsize=13)
        ax.set_xlabel("Time steps", color="#aaaaaa")
        ax.set_ylabel("GHI (normalized)", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
        ax.grid(color="#1f2937", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,"features","D2_rolling_mean.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(); print("    D2 saved ✓")

    # D3. Train/val/test timeline
    print("  D3. Train/val/test split timeline ...")
    df_ps = df_pre.sort_values("timestamp")
    n     = len(df_ps)
    t_end, v_end = int(n*0.70), int(n*0.85)
    ts   = df_ps["timestamp"].values
    gv   = df_ps["GHI"].values if "GHI" in df_ps.columns else np.zeros(n)
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
    ax.plot(ts[:t_end], gv[:t_end], color="#4cc9f0", linewidth=0.5, alpha=0.6, label="Train (70%)")
    ax.plot(ts[t_end:v_end], gv[t_end:v_end], color="#f9c74f", linewidth=0.5, alpha=0.8, label="Val (15%)")
    ax.plot(ts[v_end:], gv[v_end:], color="#f3722c", linewidth=0.5, alpha=0.8, label="Test (15%)")
    ax.axvline(ts[t_end], color="#f9c74f", linewidth=1.5, linestyle="--")
    ax.axvline(ts[v_end], color="#f3722c", linewidth=1.5, linestyle="--")
    ax.set_title("Train / Validation / Test Temporal Split",
                 color="white", fontsize=13)
    ax.set_ylabel("GHI (normalized)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values(): sp.set_color("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    ax.grid(color="#1f2937", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"features","D3_train_val_test_split.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print("    D3 saved ✓")

else:
    print("  [SKIP] Preprocessed data not found — D plots skipped.")


# ═══════════════════════════════════════════════════════════
# E. SOLAR RESOURCE QUALITY PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[E] Solar resource quality plots ...")

# E1. RRTDHS heatmap
print("  E1. RRTDHS heatmap ...")

top_cities = city_summary.nlargest(30, "GHI_mean")["city"].tolist()
rrtdhs_pivot = (
    df[df["city"].isin(top_cities)]
    .groupby(["city","month"])["RRTDHS"].mean()
    .unstack(level="month")
)
rrtdhs_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")
sns.heatmap(rrtdhs_pivot, ax=ax, cmap="YlOrRd",
            linewidths=0.3, linecolor="#1f2937",
            annot=True, fmt=".2f", annot_kws={"size":7},
            cbar_kws={"label":"RRTDHS Score","shrink":0.8})
ax.set_title("Solar Resource Score — Top 30 Cities by Month",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="#cccccc", labelsize=8)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"solar_resource","E1_RRTDHS_heatmap.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    E1 saved ✓")


# E2. CSI distribution
print("  E2. CSI distribution ...")

if "CSI" in df.columns:
    df_csi = df[(df["CSI"] > 0) & (df["CSI"] <= 1.5) & (df["GHI"] > 10)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.set_facecolor("#111827")
    axes[0].hist(df_csi["CSI"], bins=60, color="#4cc9f0", alpha=0.8, edgecolor="none")
    axes[0].axvline(1.0, color="#f9c74f", linewidth=1.5,
                    linestyle="--", label="Perfect clear sky")
    axes[0].set_title("Clear Sky Index (CSI) Distribution", color="white", fontsize=12)
    axes[0].set_xlabel("CSI (0=cloudy, 1=clear, >1=enhancement)", color="#aaaaaa")
    axes[0].set_ylabel("Count", color="#aaaaaa"); axes[0].tick_params(colors="#aaaaaa")
    for sp in axes[0].spines.values(): sp.set_color("#333333")
    axes[0].grid(axis="y", color="#1f2937", linewidth=0.5)
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    for season, col in SEASON_COLORS.items():
        sub = df_csi[df_csi["season"] == season]["CSI"] if "season" in df_csi.columns else pd.Series(dtype=float)
        if len(sub) == 0: continue
        axes[1].hist(sub, bins=50, alpha=0.5, color=col, label=season, edgecolor="none")
    axes[1].set_title("CSI Distribution by Season", color="white", fontsize=12)
    axes[1].set_xlabel("CSI", color="#aaaaaa"); axes[1].tick_params(colors="#aaaaaa")
    for sp in axes[1].spines.values(): sp.set_color("#333333")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
    axes[1].grid(axis="y", color="#1f2937", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"solar_resource","E2_CSI_distribution.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(); print("    E2 saved ✓")
else:
    print("    [SKIP] CSI column not in data")


# E3. Top 20 cities by GHI
print("  E3. Top 20 cities by GHI ...")

top20 = city_summary.nlargest(20, "GHI_mean").sort_values("GHI_mean")
fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827")
bars = ax.barh(top20["city"], top20["GHI_mean"],
               color=[CZ_COLORS.get(cz,"#888") for cz in top20["climate_zone"]],
               alpha=0.85, edgecolor="none")
for bar, val in zip(bars, top20["GHI_mean"]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", ha="left", color="white", fontsize=8)
ax.set_title("Top 20 Locations — Mean GHI (W/m²)", color="white", fontsize=13)
ax.set_xlabel("Mean GHI (W/m²)", color="#aaaaaa")
ax.tick_params(colors="#cccccc", labelsize=9)
for sp in ax.spines.values(): sp.set_color("#333333")
ax.grid(axis="x", color="#1f2937", linewidth=0.5)
shown_cz = top20["climate_zone"].unique()
ax.legend([plt.Rectangle((0,0),1,1, color=CZ_COLORS.get(cz,"#888")) for cz in shown_cz],
          shown_cz, loc="lower right", fontsize=7,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#333333")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"solar_resource","E3_top20_GHI_cities.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(); print("    E3 saved ✓")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  ✅  ALL PLOTS COMPLETE")
print(f"\n  Saved to: {PLOT_DIR}/")
print("""
  maps/
    A0_all_222_locations_overview.html   ← ALL cities, colour=climate zone  [NEW]
    A1_GHI_mean_map.html                 ← colour + size by GHI
    A2_climate_zone_map.html
    A3_district_solar_resource.html
    A4_all_locations_india_context.html  ← national context

  timeseries/
    B1_daily_GHI_districts.png
    B2_GHI_vs_clearsky.png
    B3_Tamb_vs_GHI_scatter.png
    B4_annual_cycle_GHI.png
    B5_daily_GHI_all_cities.png          ← all 222 city overlay

  statistics/
    C1_correlation_matrix.png
    C2_GHI_violin_climate_zone.png
    C3_diurnal_profile.png
    C4_cloud_vs_GHI_density.png

  features/   (require preprocessed data from 04_...)
    D1_lag_correlations.png
    D2_rolling_mean.png
    D3_train_val_test_split.png

  solar_resource/
    E1_RRTDHS_heatmap.png
    E2_CSI_distribution.png
    E3_top20_GHI_cities.png
""")
print("  HTML maps open automatically in your default browser (local)")
print("  or display inline in Colab.")
print("=" * 68)