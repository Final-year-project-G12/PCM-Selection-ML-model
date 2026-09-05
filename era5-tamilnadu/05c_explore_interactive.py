"""
06_explore_interactive.py
===========================
ERA5 Tamil Nadu — Interactive Explorer (Streamlit)

Adapted from your old city/hourly explorer (03_explore_raw_tamilnadu.py) to
the current point_id/date/event schema. Same visual language (dark theme,
Plotly + Folium), different data model: instead of one row per (city, hour),
this is one row per (point_id, date, event) with event in
{sunrise, noon, sunset} — so every time-series chart here plots THREE
colored traces (one per event) against date, rather than one continuous
hourly line.

HOW TO RUN
----------
1. Install once:
   pip install streamlit plotly pandas numpy folium branca

2. Run:
   streamlit run 06_explore_interactive.py

3. Browser opens at http://localhost:8501

WHAT YOU GET
------------
Sidebar: 2 dropdowns — Location (point_id) + Property (any era5_*/power_*
variable present in both raw and processed data) — plus a View selector.

View 1 — Raw Time Series
  Selected location + property, one line per sun-event (sunrise/noon/sunset),
  date on X axis, value on Y axis. Out-of-bound points (physical validation
  bounds from 04_preprocess_tamilnadu.py) marked in orange. Range slider.

View 2 — Processed Time Series
  Same chart, same location+property, but reading 04's cleaned output
  (imputed, outlier-flagged-then-filled, physically validated).

View 3 — Raw vs Processed Comparison
  Both sources overlaid for the same location+property+event: raw as a
  faint dashed line, processed as a solid line — the fastest way to SEE
  what Hampel filtering / imputation / physical validation actually did
  at a specific site, not just read it in the QC report.

View 4 — Location Map
  Folium map of all 133 points, colored by the selected Property's mean
  value (processed data, noon event) — selected Location is highlighted
  with a distinct marker so you can see where it sits relative to others.

DATA REQUIRED
-------------
  data/processed/climate_tamilnadu_points.csv          (02's output — raw)
  data/preprocessed/tamilnadu_cleaned_physical.csv      (04's output — processed)
Both must exist; run 02 then 04 first if either is missing.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import folium
import branca.colormap as cm

from config import COMBINED_POINTS_FILE, PREPROCESSED_DIR

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ERA5 Tamil Nadu — Interactive Explorer",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW_FILE = COMBINED_POINTS_FILE
PROCESSED_FILE = PREPROCESSED_DIR / "tamilnadu_cleaned_physical.csv"
OUT_DIR = PREPROCESSED_DIR.parent / "plots" / "interactive_explorer"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(RAW_FILE):
    st.error(f"**File not found:** `{RAW_FILE}`\n\nRun `02_combine_tamilnadu.py` first.")
    st.stop()
if not os.path.exists(PROCESSED_FILE):
    st.warning(
        f"**Processed file not found:** `{PROCESSED_FILE}`\n\n"
        "Run `04_preprocess_tamilnadu.py` first. Raw-only views below still work."
    )

EVENT_ORDER = ["sunrise", "noon", "sunset"]
EVENT_COLORS = {"sunrise": "#f9c74f", "noon": "#4cc9f0", "sunset": "#f3722c"}

# Physical bounds — same dict as 04_preprocess_tamilnadu.py, used only to
# mark out-of-bound RAW points (the processed file has these already fixed).
BOUNDS = {
    "era5_GHI": (0, 1400), "era5_DNI": (0, 1400), "era5_DHI": (0, 900),
    "era5_GHI_clearsky": (0, 1400), "era5_CSI": (0, 1.5),
    "era5_LW_down": (50, 600),
    "era5_T_amb": (-30, 55), "era5_T_dew": (-30, 40), "era5_RHum": (0, 100),
    "era5_W_spd": (0, 50), "era5_P_atm": (850, 1060),
    "era5_cloud_cover": (0, 1), "era5_precipitation": (0, 200),
    "era5_SZA": (0, 180),
    "power_ALLSKY_SFC_SW_DWN": (0, 1400), "power_CLRSKY_SFC_SW_DWN": (0, 1400),
    "power_T2M": (-30, 55), "power_RH2M": (0, 100), "power_WS10M": (0, 50),
}

NON_PLOT = {"lat", "lon", "grid_lat", "grid_lon", "population", "weight",
            "month", "DOY", "year", "season_code", "is_daytime",
            "impute_zone", "ist_hour_decimal", "solar_hour_angle"}


# ═══════════════════════════════════════════════════════════
# LOAD DATA — cached so Streamlit doesn't reload on every click
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading raw data …")
def load_raw(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
    return df.sort_values(["point_id", "event", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading processed data …")
def load_processed(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
    return df.sort_values(["point_id", "event", "date"]).reset_index(drop=True)


df_raw = load_raw(RAW_FILE)
df_proc = load_processed(PROCESSED_FILE)

LOCATIONS = sorted(df_raw["point_id"].unique())

# Property list = numeric era5_*/power_* columns present in RAW (comparison
# view additionally requires presence in processed — checked per-view).
PROPERTIES = sorted([
    c for c in df_raw.columns
    if c not in NON_PLOT and c not in ("point_id", "event", "date", "time_utc",
                                        "season", "grid_lat", "grid_lon")
    and (c.startswith("era5_") or c.startswith("power_"))
])


@st.cache_data(show_spinner=False)
def get_location_data(df, point_id):
    return df[df["point_id"] == point_id].copy()


# ═══════════════════════════════════════════════════════════
# SIDEBAR — the 2 dropdowns + view selector
# ═══════════════════════════════════════════════════════════
st.sidebar.markdown(
    "<h2 style='color:#4cc9f0;margin-bottom:4px'>🌞 ERA5 Tamil Nadu</h2>"
    "<p style='color:#8b949e;font-size:12px;margin-top:0'>"
    "133 population-weighted points · 3 sun-events/day · 2016-2025</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

location = st.sidebar.selectbox("Location (point_id)", options=LOCATIONS, index=0)
prop = st.sidebar.selectbox(
    "Property", options=PROPERTIES,
    index=PROPERTIES.index("era5_GHI") if "era5_GHI" in PROPERTIES else 0,
)

st.sidebar.markdown("---")
view = st.sidebar.radio(
    "View",
    options=["Raw Time Series", "Processed Time Series",
             "Raw vs Processed Comparison", "Location Map"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Raw rows:** {len(df_raw):,}  \n"
    + (f"**Processed rows:** {len(df_proc):,}" if df_proc is not None else "**Processed:** not loaded")
)


# ═══════════════════════════════════════════════════════════
# HELPER — build a per-event time-series figure
# ═══════════════════════════════════════════════════════════
def build_timeseries_fig(grp, prop, title, mark_outliers=False):
    fig = go.Figure()
    lo, hi = BOUNDS.get(prop, (None, None))

    for event in EVENT_ORDER:
        sub = grp[grp["event"] == event].sort_values("date")
        if prop not in sub.columns or sub.empty:
            continue
        series = sub[prop]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=series, mode="lines+markers",
            line=dict(color=EVENT_COLORS[event], width=1.3),
            marker=dict(size=3),
            name=event, connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>" + f"{event}: " + "%{y:.3f}<extra></extra>",
        ))
        if mark_outliers and lo is not None:
            out_mask = (series < lo) | (series > hi)
            if out_mask.any():
                fig.add_trace(go.Scatter(
                    x=sub.loc[out_mask, "date"], y=series[out_mask],
                    mode="markers",
                    marker=dict(color="#ff4d4d", size=7, symbol="x"),
                    name=f"{event} out-of-bound",
                    hovertemplate="%{x}<br>%{y:.3f}  <- OUT OF BOUNDS<extra></extra>",
                ))

    if lo is not None:
        fig.add_hline(y=lo, line_dash="dot", line_color="#888", line_width=0.8,
                       annotation_text=f"min {lo}", annotation_font_size=9)
        fig.add_hline(y=hi, line_dash="dot", line_color="#888", line_width=0.8,
                       annotation_text=f"max {hi}", annotation_font_size=9)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        title=title,
        xaxis=dict(title="Date", gridcolor="#30363d",
                   rangeslider=dict(visible=True, bgcolor="#1a1f27",
                                     bordercolor="#30363d", thickness=0.06)),
        yaxis=dict(title=prop, gridcolor="#30363d"),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
        margin=dict(t=50, b=80),
        height=520,
    )
    return fig


def stats_row(grp, prop):
    series = grp[prop] if prop in grp.columns else pd.Series(dtype=float)
    n = len(series)
    n_miss = int(series.isna().sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{n:,}")
    c2.metric("Missing", f"{n_miss:,}",
              delta=f"{100*n_miss/max(n,1):.2f}%" if n else "—", delta_color="inverse")
    c3.metric("Min", f"{series.min():.2f}" if series.notna().any() else "—")
    c4.metric("Max", f"{series.max():.2f}" if series.notna().any() else "—")
    c5.metric("Mean", f"{series.mean():.2f}" if series.notna().any() else "—")


# ═══════════════════════════════════════════════════════════
# VIEW 1 — RAW TIME SERIES
# ═══════════════════════════════════════════════════════════
if view == "Raw Time Series":
    st.markdown(
        f"<h3 style='color:#4cc9f0;margin-bottom:4px'>{location} — {prop} (raw)</h3>"
        f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        f"Before any cleaning · one line per sun-event · red x = physically out-of-bound</p>",
        unsafe_allow_html=True,
    )
    grp = get_location_data(df_raw, location)
    if prop not in grp.columns:
        st.warning(f"Column `{prop}` not found.")
    else:
        stats_row(grp, prop)
        fig = build_timeseries_fig(grp, prop, f"{location} — {prop} (raw)", mark_outliers=True)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# VIEW 2 — PROCESSED TIME SERIES
# ═══════════════════════════════════════════════════════════
elif view == "Processed Time Series":
    st.markdown(
        f"<h3 style='color:#06d6a0;margin-bottom:4px'>{location} — {prop} (processed)</h3>"
        f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        f"After Phase 2 QC: physical validation, Hampel outlier filtering, "
        f"hierarchical imputation applied</p>",
        unsafe_allow_html=True,
    )
    if df_proc is None:
        st.error("Processed file not loaded — run `04_preprocess_tamilnadu.py` first.")
    else:
        grp = get_location_data(df_proc, location)
        if prop not in grp.columns:
            st.warning(f"Column `{prop}` not found in processed data.")
        else:
            stats_row(grp, prop)
            fig = build_timeseries_fig(grp, prop, f"{location} — {prop} (processed)")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# VIEW 3 — RAW VS PROCESSED COMPARISON
# ═══════════════════════════════════════════════════════════
elif view == "Raw vs Processed Comparison":
    st.markdown(
        f"<h3 style='color:#f9c74f;margin-bottom:4px'>{location} — {prop} — Raw vs Processed</h3>"
        f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        f"Faint dotted = raw · solid = processed · same location, property, and sun-events, "
        f"overlaid so you can SEE what QC changed</p>",
        unsafe_allow_html=True,
    )
    if df_proc is None:
        st.error("Processed file not loaded — run `04_preprocess_tamilnadu.py` first.")
    else:
        grp_raw = get_location_data(df_raw, location)
        grp_proc = get_location_data(df_proc, location)
        if prop not in grp_raw.columns or prop not in grp_proc.columns:
            st.warning(f"Column `{prop}` not found in one of the two datasets.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw**")
                stats_row(grp_raw, prop)
            with col2:
                st.markdown("**Processed**")
                stats_row(grp_proc, prop)

            fig = go.Figure()
            for event in EVENT_ORDER:
                r = grp_raw[grp_raw["event"] == event].sort_values("date")
                p = grp_proc[grp_proc["event"] == event].sort_values("date")
                if prop in r.columns and not r.empty:
                    fig.add_trace(go.Scatter(
                        x=r["date"], y=r[prop], mode="lines",
                        line=dict(color=EVENT_COLORS[event], width=1.0, dash="dot"),
                        opacity=0.45, name=f"{event} (raw)", connectgaps=False,
                        hovertemplate="%{x|%Y-%m-%d}<br>raw: %{y:.3f}<extra></extra>",
                    ))
                if prop in p.columns and not p.empty:
                    fig.add_trace(go.Scatter(
                        x=p["date"], y=p[prop], mode="lines",
                        line=dict(color=EVENT_COLORS[event], width=1.6),
                        name=f"{event} (processed)", connectgaps=False,
                        hovertemplate="%{x|%Y-%m-%d}<br>processed: %{y:.3f}<extra></extra>",
                    ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                title=f"{location} — {prop} — raw (dotted, faint) vs processed (solid)",
                xaxis=dict(title="Date", gridcolor="#30363d",
                           rangeslider=dict(visible=True, bgcolor="#1a1f27",
                                             bordercolor="#30363d", thickness=0.06)),
                yaxis=dict(title=prop, gridcolor="#30363d"),
                legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hovermode="x unified",
                margin=dict(t=60, b=80),
                height=560,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### What changed — by event")
            rows = []
            for event in EVENT_ORDER:
                r = grp_raw[grp_raw["event"] == event][prop] if prop in grp_raw.columns else pd.Series(dtype=float)
                p = grp_proc[grp_proc["event"] == event][prop] if prop in grp_proc.columns else pd.Series(dtype=float)
                rows.append({
                    "event": event,
                    "raw_missing_%": round(100 * r.isna().mean(), 2) if len(r) else None,
                    "processed_missing_%": round(100 * p.isna().mean(), 2) if len(p) else None,
                    "raw_mean": round(r.mean(), 3) if r.notna().any() else None,
                    "processed_mean": round(p.mean(), 3) if p.notna().any() else None,
                    "raw_std": round(r.std(), 3) if r.notna().any() else None,
                    "processed_std": round(p.std(), 3) if p.notna().any() else None,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# VIEW 4 — LOCATION MAP
# ═══════════════════════════════════════════════════════════
elif view == "Location Map":
    st.markdown(
        f"<h3 style='color:#06d6a0;margin-bottom:4px'>All 133 Points — colored by {prop} (noon mean)</h3>"
        f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        f"Selected location (<b>{location}</b>) highlighted in red</p>",
        unsafe_allow_html=True,
    )

    source_df = df_proc if df_proc is not None else df_raw
    if prop not in source_df.columns:
        st.warning(f"Column `{prop}` not found.")
    else:
        noon = source_df[source_df["event"] == "noon"]
        summary = noon.groupby("point_id").agg(
            lat=("lat", "first"), lon=("lon", "first"),
            value=(prop, "mean"),
        ).reset_index()

        @st.cache_data(show_spinner="Building map …")
        def build_map(summary_df, prop_name, highlight_id):
            vmin, vmax = summary_df["value"].min(), summary_df["value"].max()
            vmax = max(vmax, vmin + 1e-6)
            colormap = cm.LinearColormap(
                ["#2d6a4f", "#52b788", "#d9ed92", "#f9c74f", "#f3722c"],
                vmin=vmin, vmax=vmax, caption=f"{prop_name} (noon mean)")
            m = folium.Map(location=[10.9, 78.5], zoom_start=7, tiles="OpenStreetMap")
            colormap.add_to(m)
            for _, row in summary_df.iterrows():
                is_selected = row["point_id"] == highlight_id
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=11 if is_selected else 6,
                    color="#ff0000" if is_selected else "white",
                    weight=2.5 if is_selected else 0.8,
                    fill=True,
                    fill_color="#ff0000" if is_selected else colormap(row["value"]),
                    fill_opacity=0.95 if is_selected else 0.85,
                    popup=folium.Popup(
                        f"<b>{row['point_id']}</b><br>{prop_name}: {row['value']:.2f}<br>"
                        f"Lat/Lon: {row['lat']:.3f}, {row['lon']:.3f}", max_width=220),
                    tooltip=f"{row['point_id']}: {row['value']:.2f}",
                ).add_to(m)
            map_path = os.path.join(OUT_DIR, "location_map.html")
            m.save(map_path)
            return open(map_path).read()

        map_html = build_map(summary, prop, location)
        st.components.v1.html(map_html, height=620, scrolling=False)
