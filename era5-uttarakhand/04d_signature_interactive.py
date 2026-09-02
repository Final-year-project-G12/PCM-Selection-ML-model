"""
04d_signature_interactive.py
===============================
Interactive (Folium + Plotly) explorer for 04b_climate_signature.py's
output. New script — there was no PNG equivalent of a variable-toggle
map before; this adds one, plus interactive versions of the correlation
heatmap and distribution plots 04b already saves as PNG.

Requires: pip install plotly folium branca

INPUT  : data/processed/signatures/climate_signature_uttarakhand.csv
OUTPUT : data/processed/signatures/interactive/*.html

HOW TO RUN:
  python 04d_signature_interactive.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import folium
import branca.colormap as cm

from config import PROCESSED_DIR

SIGNATURE_FILE = PROCESSED_DIR / "signatures" / "climate_signature_uttarakhand.csv"
OUT_DIR = PROCESSED_DIR / "signatures" / "interactive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Indices worth toggling through on the map — edit this list freely.
MAP_LAYERS = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac",
              "CCI", "HDD18", "CDD24", "RH_mean", "HSI", "monsoon_index",
              "L_required_kJ_per_kg"]

print("=" * 68)
print("  Interactive Climate Signature Explorer — Uttarakhand")
print(f"  Input  : {SIGNATURE_FILE}")
print(f"  Output : {OUT_DIR}/")
print("=" * 68)

if not SIGNATURE_FILE.exists():
    raise FileNotFoundError(f"{SIGNATURE_FILE} not found — run 04b_climate_signature.py first.")

sig = pd.read_csv(SIGNATURE_FILE)
sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)
print(f"  Points: {len(sig)}")

# ═══════════════════════════════════════════════════════════
# A. FOLIUM MULTI-LAYER MAP — one toggleable layer per index
# ═══════════════════════════════════════════════════════════
print("\n[A] Multi-layer Folium map (toggle indices in the layer control) ...")

fmap = folium.Map(location=[sig["lat"].mean(), sig["lon"].mean()],
                   zoom_start=7, tiles="CartoDB positron")

layers_added = 0
for col in MAP_LAYERS:
    if col not in sig.columns or sig[col].isna().all():
        continue
    vals = sig[col].dropna()
    colormap = cm.LinearColormap(
        colors=["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=vals.min(), vmax=vals.max(), caption=col)
    fg = folium.FeatureGroup(name=col, show=(layers_added == 0))
    for r in sig.itertuples():
        val = getattr(r, col, np.nan)
        if val != val:  # NaN
            continue
        folium.CircleMarker(
            location=[r.lat, r.lon], radius=6,
            color=colormap(val), fill=True, fill_opacity=0.85, weight=1,
            popup=folium.Popup(f"<b>{r.point_id}</b><br>{col}: {val:.3g}"
                                f"<br>population: {getattr(r,'population',float('nan')):,.0f}",
                                max_width=220),
        ).add_to(fg)
    fg.add_to(fmap)
    layers_added += 1

folium.LayerControl(collapsed=False).add_to(fmap)
fmap.save(str(OUT_DIR / "A_signature_layers.html"))
print(f"  Saved: A_signature_layers.html  ({layers_added} toggleable layers)")

# ═══════════════════════════════════════════════════════════
# B. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════
print("\n[B] Signature correlation heatmap (Plotly) ...")

INDEX_COLS = ["Ta_mean", "Ta_p95", "Ta_p05", "DTR", "GHI_daily_kWh",
              "kt_mean", "kt_std", "SAI", "CCI", "cloudy_frac", "HDD18", "CDD24",
              "RH_mean", "HSI", "wind_mean", "seasonality", "monsoon_index", "elev_proxy"]
INDEX_COLS = [c for c in INDEX_COLS if c in sig.columns]
corr = sig[INDEX_COLS].corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                 title="Climate Signature Correlation (Tier1+Tier2 canonical) — Uttarakhand")
fig.update_layout(height=750)
fig.write_html(str(OUT_DIR / "B_correlation.html"), include_plotlyjs="cdn")
print("  Saved: B_correlation.html")

# ═══════════════════════════════════════════════════════════
# C. DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════
print("\n[C] Index distributions (Plotly) ...")

n = len(INDEX_COLS)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=INDEX_COLS)
for i, col in enumerate(INDEX_COLS):
    r, c = divmod(i, ncols)
    fig.add_trace(go.Histogram(x=sig[col].dropna(), nbinsx=20, marker_color="#4c72b0",
                                showlegend=False), row=r + 1, col=c + 1)
fig.update_layout(title=f"Signature index distributions across {len(sig)} points", height=280 * nrows)
fig.write_html(str(OUT_DIR / "C_distributions.html"), include_plotlyjs="cdn")
print("  Saved: C_distributions.html")

# ═══════════════════════════════════════════════════════════
# D. SCATTER MATRIX — quick look at the criteria that will drive MCDM
# ═══════════════════════════════════════════════════════════
print("\n[D] Scatter matrix of key PCM-facing indices ...")

key_cols = [c for c in ["GHI_daily_kWh", "Ta_mean", "HDD18", "L_required_kJ_per_kg",
                         "HSI", "cloudy_frac"] if c in sig.columns]
fig = px.scatter_matrix(sig, dimensions=key_cols, hover_name="point_id",
                         title="Key PCM-facing indices — look for the clusters "
                               "05_cluster_uttarakhand.py should be finding")
fig.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.6))
fig.update_layout(height=800)
fig.write_html(str(OUT_DIR / "D_scatter_matrix.html"), include_plotlyjs="cdn")
print("  Saved: D_scatter_matrix.html")

print("\n" + "=" * 68)
print("  DONE — open the .html files in", OUT_DIR)
print("=" * 68)
