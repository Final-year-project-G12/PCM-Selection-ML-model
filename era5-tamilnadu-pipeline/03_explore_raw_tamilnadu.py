"""
03_explore_raw_tamilnadu.py
===========================
ERA5 Tamil Nadu — Interactive Raw Data Explorer (Streamlit)

Works on Python 3.14+ (no Flask/pkgutil dependency).

HOW TO RUN
----------
1. Install once:
   pip install streamlit plotly pandas numpy folium

2. Run:
   streamlit run 03_explore_raw_tamilnadu.py

3. Browser opens automatically at http://localhost:8501

WHAT YOU GET
------------
Sidebar: pick City (top 100) + Variable (all numeric columns)

Page 1 — Time Series
  Daily mean line with hourly raw toggle
  Gaps shown as visible breaks  (connectgaps=False)
  Out-of-bound points highlighted orange
  Range slider to zoom any date range
  Missing value table for selected city

Page 2 — Outlier Detective
  Timeline with outliers marked
  Box plot showing distribution vs physical bounds
  Full outlier data table

Page 3 — Quality Map
  Folium map embedded — all 222 cities coloured by % missing

OUTPUTS SAVED → data/before_pre/
  quality_audit.csv    — missing % per city x variable
  outlier_report.csv   — out-of-bound counts per city x variable
  quality_map.html     — standalone Folium map you can open separately
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import folium
import branca.colormap as cm

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ERA5 Tamil Nadu — Raw Explorer",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
_HERE         = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
COMBINED_FILE = os.path.join(_HERE, "data", "processed", "climate_tamilnadu_all.csv")
OUT_DIR       = os.path.join(_HERE, "data", "before_pre")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(COMBINED_FILE):
    st.error(
        f"**File not found:** `{COMBINED_FILE}`\n\n"
        "Place `climate_tamilnadu_all.csv` at `data/processed/` next to this script."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════
# PHYSICAL BOUNDS
# ═══════════════════════════════════════════════════════════
BOUNDS = {
    "GHI":          (0,    1400),
    "DNI":          (0,    1400),
    "DHI":          (0,     900),
    "GHI_clearsky": (0,    1400),
    "CSI":          (0,     1.5),
    "ETR":          (0,    1415),
    "LW_down":      (50,    600),
    "T_amb":        (-5,     55),
    "T_dew":        (-20,    40),
    "RHum":         (0,     100),
    "W_spd":        (0,      50),
    "W_dir":        (0,     360),
    "P_atm":        (850,  1060),
    "cloud_cover":  (0,       1),
    "precipitation":(0,     200),
    "SZA":          (0,     180),
}

# ═══════════════════════════════════════════════════════════
# LOAD DATA — cached so Streamlit doesn't reload on every click
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading climate_tamilnadu_all.csv …")
def load_data(path):
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="warn",
        parse_dates=["timestamp"],
    )
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    return df

df_full = load_data(COMBINED_FILE)

# ── Plot variables (all numeric except coords/flags) ──────
NON_PLOT = {"lat","lon","grid_lat","grid_lon","altitude_m",
            "season_code","high_solar_resource","year","hour",
            "month","DOY","T_set"}
PLOT_VARS = sorted([
    c for c in df_full.columns
    if c not in NON_PLOT
    and c != "timestamp"
    and str(df_full[c].dtype).startswith(("float","int"))
])

# ── Top 100 cities by completeness ────────────────────────
@st.cache_data(show_spinner="Ranking cities by data quality …")
def get_top_cities(df, plot_vars, n=100):
    score_cols = [c for c in plot_vars if df[c].isna().mean() < 1.0]
    completeness = (
        df.groupby("city")[score_cols]
        .apply(lambda g: 1 - g.isna().mean().mean())
        .sort_values(ascending=False)
    )
    return completeness.head(n).index.tolist(), completeness

TOP_CITIES, city_completeness = get_top_cities(df_full, PLOT_VARS)

# ── Build & save audit/outlier reports (once) ─────────────
@st.cache_data(show_spinner="Computing quality audit …")
def build_audit(df, top_cities, out_dir):
    audit_cols = [c for c in
        ["GHI","DNI","DHI","T_amb","RHum","W_spd","cloud_cover",
         "precipitation","P_atm","LW_down","CSI","GHI_clearsky"]
        if c in df.columns]
    rows = []
    for city in top_cities:
        grp = df[df["city"] == city]
        n   = len(grp)
        for col in audit_cols:
            nm = int(grp[col].isna().sum())
            rows.append({"city": city, "column": col,
                         "n_missing": nm,
                         "pct_missing": round(100 * nm / n, 3)})
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(os.path.join(out_dir, "quality_audit.csv"), index=False)
    return audit_df

@st.cache_data(show_spinner="Computing outlier report …")
def build_outlier_report(df, top_cities, bounds, out_dir):
    rows = []
    for city in top_cities:
        grp = df[df["city"] == city]
        n   = len(grp)
        for col, (lo, hi) in bounds.items():
            if col not in grp.columns:
                continue
            n_out = int(((grp[col] < lo) | (grp[col] > hi)).sum())
            if n_out > 0:
                rows.append({"city": city, "column": col,
                             "n_outliers": n_out,
                             "pct_outliers": round(100 * n_out / n, 3),
                             "bound_lo": lo, "bound_hi": hi})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(out_dir, "outlier_report.csv"), index=False)
    return out_df

@st.cache_data(show_spinner="Building quality map …")
def build_quality_map(df, city_completeness, out_dir):
    city_meta = df.groupby("city").agg(
        lat=("lat","first"), lon=("lon","first"),
        district=("district","first"),
        climate_zone=("climate_zone","first"),
        n_rows=("timestamp","count"),
    ).reset_index()
    city_meta = city_meta.merge(
        city_completeness.rename("completeness").reset_index(),
        on="city", how="left"
    )
    city_meta["pct_missing"] = (1 - city_meta["completeness"]) * 100

    vmin = city_meta["pct_missing"].min()
    vmax = max(city_meta["pct_missing"].max(), vmin + 0.01)
    cmap_q = cm.LinearColormap(
        ["#2dc653","#fcbf49","#f3722c","#d62828"],
        vmin=vmin, vmax=vmax, caption="% Missing Data (RAW)"
    )
    m = folium.Map(location=[10.9, 78.5], zoom_start=7,
                   tiles="CartoDB positron")
    cmap_q.add_to(m)
    for _, row in city_meta.iterrows():
        frac = (row["pct_missing"] - vmin) / max(vmax - vmin, 1e-9)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5 + frac * 9,
            color="white", weight=0.8,
            fill=True,
            fill_color=cmap_q(row["pct_missing"]),
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<div style='font-family:Arial;font-size:12px;width:200px;'>"
                f"<b>{row['city']}</b><br>"
                f"District: {row['district']}<br>"
                f"Climate: {row['climate_zone']}<br>"
                f"Rows: {row['n_rows']:,}<br>"
                f"<b>Missing: {row['pct_missing']:.3f}%</b></div>",
                max_width=220),
            tooltip=f"{row['city']}: {row['pct_missing']:.2f}% missing",
        ).add_to(m)
    map_path = os.path.join(out_dir, "quality_map.html")
    m.save(map_path)
    return open(map_path).read()

audit_df       = build_audit(df_full, TOP_CITIES, OUT_DIR)
outlier_df     = build_outlier_report(df_full, TOP_CITIES, BOUNDS, OUT_DIR)
map_html       = build_quality_map(df_full, city_completeness, OUT_DIR)

# ── Per-city data fetcher (cached per city to avoid re-reading 3.9M rows) ──
@st.cache_data(show_spinner=False)
def get_city_data(city):
    return df_full[df_full["city"] == city].sort_values("timestamp").copy()

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.markdown(
    "<h2 style='color:#4cc9f0;margin-bottom:4px'>🌞 ERA5 Tamil Nadu</h2>"
    "<p style='color:#8b949e;font-size:12px;margin-top:0'>Raw data explorer<br>"
    "No preprocessing applied</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

city = st.sidebar.selectbox(
    "📍 City (top 100 by completeness)",
    options=sorted(TOP_CITIES),
    index=0,
)

variable = st.sidebar.selectbox(
    "📊 Variable",
    options=PLOT_VARS,
    index=PLOT_VARS.index("GHI") if "GHI" in PLOT_VARS else 0,
)

resolution = st.sidebar.radio(
    "⏱ Resolution",
    options=["Daily mean", "Hourly raw"],
    index=0,
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📄 View",
    options=["📈 Time Series", "⚠️ Outlier Detective", "🗺️ Quality Map"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Loaded:** {len(df_full):,} rows  \n"
    f"**Cities:** {df_full['city'].nunique()}  \n"
    f"**Period:** {df_full['timestamp'].min().date()} →  \n"
    f"{df_full['timestamp'].max().date()}  \n"
    f"**Outputs saved to:**  \n`data/before_pre/`"
)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — TIME SERIES
# ═══════════════════════════════════════════════════════════
if page == "📈 Time Series":

    st.markdown(
        f"<h3 style='color:#4cc9f0;margin-bottom:4px'>"
        f"📈 {city} — {variable}</h3>"
        f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        f"Raw data · gaps = visible breaks in the line · "
        f"orange × = physically out-of-bound values</p>",
        unsafe_allow_html=True,
    )

    grp = get_city_data(city)

    if variable not in grp.columns:
        st.warning(f"Column `{variable}` not found for {city}.")
        st.stop()

    # Resample if daily
    if resolution == "Daily mean":
        plot_df = (grp.set_index("timestamp")[variable]
                     .resample("D").mean()
                     .reset_index())
        plot_df.columns = ["timestamp", variable]
        x_label = "Date (daily mean)"
    else:
        plot_df = grp[["timestamp", variable]].copy()
        x_label = "Timestamp (hourly)"

    series   = plot_df[variable]
    n_total  = len(plot_df)
    n_miss   = int(series.isna().sum())
    pct_miss = round(100 * n_miss / n_total, 3)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{n_total:,}")
    c2.metric("Missing", f"{n_miss:,}", delta=f"{pct_miss}%",
              delta_color="inverse")
    c3.metric("Min", f"{series.min():.2f}" if n_total > 0 else "—")
    c4.metric("Max", f"{series.max():.2f}" if n_total > 0 else "—")
    c5.metric("Mean", f"{series.mean():.2f}" if n_total > 0 else "—")

    # Build figure
    fig = go.Figure()

    # Gap shading
    gap_mask = series.isna()
    gap_start = None
    for i, is_gap in enumerate(gap_mask):
        if is_gap and gap_start is None:
            gap_start = plot_df["timestamp"].iloc[i]
        elif not is_gap and gap_start is not None:
            fig.add_vrect(
                x0=gap_start, x1=plot_df["timestamp"].iloc[i],
                fillcolor="rgba(243,114,44,0.15)",
                layer="below", line_width=0,
            )
            gap_start = None

    # Main line
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=series,
        mode="lines",
        line=dict(color="#4cc9f0", width=1.5 if resolution=="Daily mean" else 0.8),
        connectgaps=False,
        name=variable,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
                      + f"{variable}: " + "%{y:.3f}<extra></extra>",
    ))

    # Out-of-bound markers
    if variable in BOUNDS:
        lo, hi = BOUNDS[variable]
        out_mask = (series < lo) | (series > hi)
        if out_mask.any():
            fig.add_trace(go.Scatter(
                x=plot_df.loc[out_mask, "timestamp"],
                y=series[out_mask],
                mode="markers",
                marker=dict(color="#f3722c", size=8, symbol="x"),
                name=f"Out-of-bounds  (< {lo} or > {hi})",
                hovertemplate="%{x}<br>" + f"{variable}: "
                              + "%{y:.3f}  ← OUT OF BOUNDS<extra></extra>",
            ))
            fig.add_hline(y=lo, line_dash="dot", line_color="#f3722c",
                          line_width=0.8, opacity=0.6,
                          annotation_text=f"min {lo}",
                          annotation_font_size=10)
            fig.add_hline(y=hi, line_dash="dot", line_color="#f3722c",
                          line_width=0.8, opacity=0.6,
                          annotation_text=f"max {hi}",
                          annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        xaxis=dict(
            title=x_label,
            gridcolor="#30363d",
            rangeslider=dict(visible=True, bgcolor="#1a1f27",
                             bordercolor="#30363d", thickness=0.06),
        ),
        yaxis=dict(title=variable, gridcolor="#30363d"),
        legend=dict(orientation="h", y=1.06,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(t=40, b=80, l=60, r=20),
        hovermode="x unified",
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Missing value table
    st.markdown("#### Missing values — this city")
    city_miss = audit_df[(audit_df["city"] == city) &
                         (audit_df["n_missing"] > 0)].copy()
    if len(city_miss) > 0:
        city_miss = city_miss.sort_values("pct_missing", ascending=False)
        st.dataframe(
            city_miss.rename(columns={
                "column": "Variable",
                "n_missing": "Missing rows",
                "pct_missing": "% Missing",
            })[["Variable","Missing rows","% Missing"]],
            use_container_width=True,
            height=200,
        )
    else:
        st.success(f"No missing values found for {city} ✓")


# ═══════════════════════════════════════════════════════════
# PAGE 2 — OUTLIER DETECTIVE
# ═══════════════════════════════════════════════════════════
elif page == "⚠️ Outlier Detective":

    out_vars = [v for v in BOUNDS.keys() if v in PLOT_VARS]
    out_var  = st.selectbox(
        "Variable with physical bounds",
        options=out_vars,
        index=out_vars.index(variable) if variable in out_vars else 0,
    )

    lo, hi = BOUNDS[out_var]

    st.markdown(
        f"<h3 style='color:#f3722c;margin-bottom:4px'>"
        f"⚠️ {city} — {out_var}</h3>"
        f"<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        f"Physical bounds: <b>{lo}</b> ≤ {out_var} ≤ <b>{hi}</b></p>",
        unsafe_allow_html=True,
    )

    grp    = get_city_data(city)
    series = grp[out_var] if out_var in grp.columns else pd.Series(dtype=float)

    out_mask  = (series < lo) | (series > hi)
    norm_mask = ~out_mask & series.notna()

    n_out  = int(out_mask.sum())
    pct_out = round(100 * out_mask.mean(), 4)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(grp):,}")
    c2.metric("Outliers", f"{n_out:,}",
              delta=f"{pct_out}%", delta_color="inverse")
    c3.metric("Bound lo", str(lo))
    c4.metric("Bound hi", str(hi))

    col_left, col_right = st.columns([2, 1])

    # ── Scatter / timeline ────────────────────────────────
    with col_left:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=grp.loc[norm_mask, "timestamp"],
            y=series[norm_mask],
            mode="lines",
            line=dict(color="#4cc9f0", width=0.8),
            connectgaps=False,
            name="Normal",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f}<extra></extra>",
        ))
        if out_mask.any():
            fig_sc.add_trace(go.Scatter(
                x=grp.loc[out_mask, "timestamp"],
                y=series[out_mask],
                mode="markers",
                marker=dict(color="#f3722c", size=9, symbol="x",
                            line=dict(width=2)),
                name=f"Outliers ({n_out:,})",
                hovertemplate="%{x}<br>%{y:.4f}  ← OUTLIER<extra></extra>",
            ))
        fig_sc.add_hline(y=lo, line_dash="dash", line_color="#f3722c",
                         line_width=1.2,
                         annotation_text=f"lo={lo}",
                         annotation_font_size=10)
        fig_sc.add_hline(y=hi, line_dash="dash", line_color="#f3722c",
                         line_width=1.2,
                         annotation_text=f"hi={hi}",
                         annotation_font_size=10)
        fig_sc.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            title=f"{city} — {out_var} with outliers highlighted",
            xaxis=dict(
                title="Timestamp", gridcolor="#30363d",
                rangeslider=dict(visible=True, bgcolor="#1a1f27",
                                 bordercolor="#30363d", thickness=0.06),
            ),
            yaxis=dict(title=out_var, gridcolor="#30363d"),
            legend=dict(orientation="h", y=1.06,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            margin=dict(t=40, b=80),
            hovermode="x unified",
            height=430,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Box plot ──────────────────────────────────────────
    with col_right:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=series.dropna(),
            name=out_var,
            marker_color="#4cc9f0",
            boxmean="sd",
            line=dict(color="#4cc9f0"),
            fillcolor="rgba(76,201,240,0.15)",
        ))
        fig_box.add_hline(y=lo, line_dash="dash",
                          line_color="#f3722c", line_width=1.5,
                          annotation_text=f"lo={lo}",
                          annotation_font_size=10)
        fig_box.add_hline(y=hi, line_dash="dash",
                          line_color="#f3722c", line_width=1.5,
                          annotation_text=f"hi={hi}",
                          annotation_font_size=10)
        fig_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            title="Distribution",
            yaxis=dict(title=out_var, gridcolor="#30363d"),
            margin=dict(t=40, b=60),
            height=430,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Outlier table ─────────────────────────────────────
    st.markdown("#### Out-of-bound values")
    out_data = grp.loc[out_mask, ["timestamp", out_var]].copy()
    if len(out_data) > 0:
        out_data["direction"] = out_data[out_var].apply(
            lambda v: f"< {lo}  (too low)" if v < lo else f"> {hi}  (too high)"
        )
        out_data["timestamp"] = out_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            out_data.rename(columns={"timestamp": "Timestamp",
                                     out_var: "Value",
                                     "direction": "Issue"}),
            use_container_width=True,
            height=min(300, 36 * len(out_data) + 38),
        )
        st.caption(f"{len(out_data):,} outlier rows — "
                   f"full list in `data/before_pre/outlier_report.csv`")
    else:
        st.success(f"No out-of-bound values for {out_var} in {city} ✓")

    # ── Cross-city outlier summary for this variable ───────
    st.markdown(f"#### Outlier count — all cities — {out_var}")
    cross = outlier_df[outlier_df["column"] == out_var].sort_values(
        "n_outliers", ascending=False)
    if len(cross) > 0:
        fig_bar = go.Figure(go.Bar(
            x=cross["city"], y=cross["n_outliers"],
            marker_color="#f3722c", opacity=0.8,
            hovertemplate="%{x}<br>Outliers: %{y}<extra></extra>",
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            xaxis=dict(title="City", gridcolor="#30363d",
                       tickangle=-45),
            yaxis=dict(title="# Outliers", gridcolor="#30363d"),
            margin=dict(t=20, b=120),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.success(f"No outliers found for {out_var} in any city ✓")


# ═══════════════════════════════════════════════════════════
# PAGE 3 — QUALITY MAP
# ═══════════════════════════════════════════════════════════
elif page == "🗺️ Quality Map":

    st.markdown(
        "<h3 style='color:#2dc653;margin-bottom:4px'>🗺️ Data Quality Map</h3>"
        "<p style='color:#8b949e;font-size:12px;margin-top:0'>"
        "All 222 cities · colour = % missing data (raw, before preprocessing) · "
        "green = good, red = most missing · click any dot for details</p>",
        unsafe_allow_html=True,
    )

    # Folium map embedded via iframe
    st.components.v1.html(map_html, height=620, scrolling=False)

    # Also show a quality summary table
    st.markdown("#### Quality summary — top 100 cities used in the explorer")
    summary = (audit_df.groupby("city")["pct_missing"]
               .mean().round(3)
               .reset_index()
               .rename(columns={"pct_missing": "avg_pct_missing_across_cols"})
               .sort_values("avg_pct_missing_across_cols"))
    st.dataframe(summary, use_container_width=True, height=300)
