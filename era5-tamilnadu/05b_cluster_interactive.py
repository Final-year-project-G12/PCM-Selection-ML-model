"""
05b_cluster_interactive.py
=============================
Interactive (Folium + Plotly) explorer for 05_cluster_tamilnadu.py's
output — cluster map with hoverable membership probabilities (the soft
GMM assignment a static PNG throws away), plus an interactive cluster
profile comparison.

Requires: pip install plotly folium

INPUT  : data/processed/clustering/cluster_assignments_tamilnadu.csv
         data/processed/clustering/cluster_profiles_tamilnadu.csv
         data/processed/clustering/bic_selection_tamilnadu.csv
OUTPUT : data/processed/clustering/interactive/*.html

HOW TO RUN:
  python 05b_cluster_interactive.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


import folium

from config import PROCESSED_DIR

CLUSTER_DIR = PROCESSED_DIR / "clustering"
OUT_DIR = CLUSTER_DIR / "interactive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSIGN_FILE = CLUSTER_DIR / "cluster_assignments_tamilnadu.csv"
PROFILE_FILE = CLUSTER_DIR / "cluster_profiles_tamilnadu.csv"
BIC_FILE = CLUSTER_DIR / "bic_selection_tamilnadu.csv"

PALETTE = px.colors.qualitative.Set1

print("=" * 68)
print("  Interactive Cluster Explorer — Tamil Nadu")
print(f"  Output : {OUT_DIR}/")
print("=" * 68)

for f in (ASSIGN_FILE, PROFILE_FILE):
    if not f.exists():
        raise FileNotFoundError(f"{f} not found — run 05_cluster_tamilnadu.py first.")

assign = pd.read_csv(ASSIGN_FILE)
profiles = pd.read_csv(PROFILE_FILE)
n_clusters = assign["cluster_id"].nunique()
print(f"  Points: {len(assign)}  |  Clusters: {n_clusters}")

# ═══════════════════════════════════════════════════════════
# A. FOLIUM CLUSTER MAP — colored by hard label, popup shows soft probs
# ═══════════════════════════════════════════════════════════
print("\n[A] Cluster map (Folium, hover for full membership probability vector) ...")

'''fmap = folium.Map(location=[assign["lat"].mean(), assign["lon"].mean()],
                   zoom_start=7, tiles="CartoDB positron")'''

fmap = folium.Map(
    location=[assign["lat"].mean(), assign["lon"].mean()],
    zoom_start=7,
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr='&copy; OpenStreetMap contributors &copy; CARTO'
)
prob_cols = [c for c in assign.columns if c.startswith("prob_cluster")]

for r in assign.itertuples():
    color = PALETTE[int(r.cluster_id) % len(PALETTE)]
    prob_lines = "<br>".join(
        f"{c.replace('prob_cluster','cluster ')}: {getattr(r, c):.2f}" for c in prob_cols)
    popup_html = (f"<b>{r.point_id}</b><br>hard label: cluster {int(r.cluster_id)}"
                  f"<br>max membership: {r.max_membership_prob:.2f}"
                  f"<br><br>{prob_lines}")
    # thinner/paler ring if this point is ambiguous (max prob close to 1/n_clusters)
    is_boundary = r.max_membership_prob < (1.5 / n_clusters)
    folium.CircleMarker(
        location=[r.lat, r.lon], radius=7 if not is_boundary else 5,
        color="black" if is_boundary else color,
        weight=2 if is_boundary else 1,
        fill=True, fill_color=color, fill_opacity=0.85 if not is_boundary else 0.4,
        popup=folium.Popup(popup_html, max_width=260),
    ).add_to(fmap)

legend_html = "".join(
    f'<i style="background:{PALETTE[c % len(PALETTE)]};width:10px;height:10px;'
    f'display:inline-block;margin-right:4px;"></i>Cluster {c}<br>'
    for c in sorted(assign["cluster_id"].unique()))
legend_html = (f'<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; '
               f'background:white; padding:8px 12px; border:1px solid #999; '
               f'border-radius:4px; font-size:12px;">{legend_html}'
               f'<i style="border:2px solid black;width:8px;height:8px;'
               f'display:inline-block;margin-right:4px;"></i>faint ring = boundary point'
               f'</div>')
fmap.get_root().html.add_child(folium.Element(legend_html))
fmap.save(str(OUT_DIR / "A_cluster_map.html"))
print("  Saved: A_cluster_map.html")

# ═══════════════════════════════════════════════════════════
# B. CLUSTER PROFILE COMPARISON
# ═══════════════════════════════════════════════════════════
print("\n[B] Cluster profile comparison (Plotly radar-style bar) ...")

compare_cols = [c for c in ["GHI_daily_kWh", "Ta_mean", "DTR", "HDD18", "CDD24",
                             "cloudy_frac", "RH_mean", "HSI", "L_required_kJ_per_kg"]
                 if c in profiles.columns]
fig = go.Figure()
for cid in profiles["cluster_id"]:
    row = profiles[profiles["cluster_id"] == cid].iloc[0]
    fig.add_trace(go.Bar(name=f"Cluster {int(cid)}", x=compare_cols,
                          y=[row[c] for c in compare_cols]))
fig.update_layout(barmode="group", title="Population-weighted cluster profiles",
                   height=500)
fig.write_html(str(OUT_DIR / "B_cluster_profiles.html"), include_plotlyjs="cdn")
print("  Saved: B_cluster_profiles.html")

# ═══════════════════════════════════════════════════════════
# C. POPULATION PER CLUSTER
# ═══════════════════════════════════════════════════════════
print("\n[C] Population covered per cluster ...")

if "total_population_covered" in profiles.columns:
    fig = px.pie(profiles, names="cluster_id", values="total_population_covered",
                 title="Population coverage per climate regime (novelty claim N6)",
                 hole=0.35)
    fig.write_html(str(OUT_DIR / "C_population_share.html"), include_plotlyjs="cdn")
    print("  Saved: C_population_share.html")

# ═══════════════════════════════════════════════════════════
# D. K-SELECTION CURVES (if bic_selection file exists)
# ═══════════════════════════════════════════════════════════
if BIC_FILE.exists():
    print("\n[D] K-selection curves (BIC / silhouette) ...")
    k_table = pd.read_csv(BIC_FILE)
    fig = make_subplots_bic = go.Figure()
    fig.add_trace(go.Scatter(x=k_table["k"], y=k_table["BIC"], mode="lines+markers",
                              name="BIC (lower better)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=k_table["k"], y=k_table["silhouette"], mode="lines+markers",
                              name="silhouette", yaxis="y2"))
    fig.update_layout(
        title="K selection — BIC vs silhouette (accept band 0.15-0.40)",
        xaxis=dict(title="K"),
        yaxis=dict(title="BIC", side="left"),
        yaxis2=dict(title="silhouette", overlaying="y", side="right"),
        height=500)
    fig.write_html(str(OUT_DIR / "D_k_selection.html"), include_plotlyjs="cdn")
    print("  Saved: D_k_selection.html")

print("\n" + "=" * 68)
print("  DONE — open the .html files in", OUT_DIR)
print("=" * 68)
