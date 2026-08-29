"""
00d_population_grid_viz.py
=============================================================================
PHASE 1 — INTERACTIVE VISUALIZATION OF THE POPULATION-WEIGHTED SAMPLING GRID

Plotly-only (folium map generation removed 2026-08-12 — see git history if
the earlier grid_map.html version is ever needed again): a single
self-contained outputs/grid_plot.html, scatter_mapbox-style, one trace per
metric (population / weight / elevation_m) with a dropdown to switch which
metric drives both marker size AND color.

Needs no live Python server or Mapbox token to view — carto-darkmatter is
a free, tokenless basemap.

COLOR-CONTRAST FIX (2026-08-12): the original carto-positron + light
sequential colorscales (Reds/Blues/YlOrBr) at default opacity read as
washed out against the light basemap. Fixed via: perceptually-uniform
high-contrast colorscales (Viridis for elevation_m, Plasma for weight,
Inferno for population), a log10 transform on population's COLOR channel
specifically (population is heavily right-skewed — a linear color scale
collapses most low-population cells into the same pale shade; marker SIZE
still uses the raw value, only color is log-transformed), opacity raised
to 0.88, a thin dark marker border so markers stand out regardless of fill
color, explicit per-trace cmin/cmax (so switching the dropdown never
silently re-normalizes the color scale against a different trace's range,
which would make otherwise-identical-looking colors mean different things
depending on which metric is active), and a dark carto-darkmatter basemap
(bright sequential colorscales pop far more against dark than light tiles).

Standalone-but-consistent: reuses this folder's config.py path constants
(POPULATION_GRID_FILE, OUTPUTS_DIR) rather than hardcoding paths, matching
03_qc_plots.py's own convention — but fails with a clear, explicit error
(not an opaque pandas traceback) if the CSV isn't present yet.

INPUT (columns required): point_id, lat, lon, population, weight,
elevation_m — 00a_build_population_grid.py's own output schema.

HOW TO RUN:
  python 00d_population_grid_viz.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import POPULATION_GRID_FILE, OUTPUTS_DIR, ensure_data_dirs

ensure_data_dirs()

COVERAGE_TARGET_PCT = 87.5   # 00a_build_population_grid.py's COVERAGE_TARGET = 0.875

GRID_PLOT_FILE = OUTPUTS_DIR / "grid_plot.html"

REQUIRED_COLUMNS = ["point_id", "lat", "lon", "population", "weight", "elevation_m"]

MARKER_OPACITY = 0.88
MAPBOX_STYLE = "carto-darkmatter"


def log(msg):
    print(f"  {msg}")


# ═══════════════════════════════════════════════════════════
# LOAD + VALIDATE
# ═══════════════════════════════════════════════════════════

def load_points():
    if not POPULATION_GRID_FILE.exists():
        raise SystemExit(
            f"ERROR: {POPULATION_GRID_FILE} not found. Run 00a_build_population_grid.py "
            f"first to generate the population-weighted sampling grid.")

    df = pd.read_csv(POPULATION_GRID_FILE)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"ERROR: {POPULATION_GRID_FILE} is missing required column(s) {missing_cols} — "
            f"found columns: {list(df.columns)}. Re-run 00a_build_population_grid.py "
            f"(and 00c_attach_elevation.py for elevation_m) if this file predates them.")
    if df.empty:
        raise SystemExit(f"ERROR: {POPULATION_GRID_FILE} exists but has no rows.")

    n_nan_elev = int(df["elevation_m"].isna().sum())
    n_nan_pop = int(df["population"].isna().sum())
    log(f"Loaded {len(df)} points from {POPULATION_GRID_FILE.name} "
        f"({n_nan_pop} with NaN population, {n_nan_elev} with NaN elevation_m).")
    return df


# ═══════════════════════════════════════════════════════════
# PLOTLY SCATTER_MAPBOX WITH METRIC-SWITCHING DROPDOWN
# ═══════════════════════════════════════════════════════════

def pixel_size(values, lo=5.0, hi=28.0):
    """sqrt-area scaling (not raw value) for go.Scattermapbox's marker.size
    (literal pixel values, unlike px.scatter_mapbox's automatic size-ref
    handling) — keeps small points visible without letting the biggest
    cells swamp the map. NaN -> minimum size, never dropped. Always driven
    by the metric's RAW value, even for population (whose color channel
    is log-transformed below) — size and color are independent encodes."""
    v = values.fillna(values.min() if values.notna().any() else 0.0)
    vmin, vmax = float(v.min()), float(v.max())
    span = max(vmax - vmin, 1e-9)
    frac = ((v - vmin) / span).clip(lower=0.0)
    return (lo + (hi - lo) * np.sqrt(frac)).tolist()


def bbox_center_zoom(lats, lons):
    """Heuristic auto-fit: center = bounding-box midpoint; zoom derived
    from the larger of the lat/lon spans via a log-scale approximation
    (not a published formula — a documented heuristic, clamped to a safe
    [3, 12] range so it degrades gracefully rather than over/under-zooming
    for an unusually tight or wide point spread)."""
    lat_span = max(float(lats.max() - lats.min()), 1e-6)
    lon_span = max(float(lons.max() - lons.min()), 1e-6)
    span = max(lat_span, lon_span)
    zoom = 8.5 - np.log2(span / 4.0 + 1e-9)
    zoom = float(np.clip(zoom, 3.0, 12.0))
    center = {"lat": float((lats.max() + lats.min()) / 2), "lon": float((lons.max() + lons.min()) / 2)}
    return center, zoom


def color_channel(vals, log_transform):
    """Returns (color_values, cmin, cmax) for one trace's marker.color.
    log_transform=True (population only) maps color
    on log10(value) so a heavily right-skewed distribution doesn't collapse
    every low-population cell into the same pale shade — cmin/cmax are
    computed on the SAME transformed scale so the colorbar's own range
    matches what's actually being plotted, not the untransformed raw
    range. NaN is filled with the metric's median before any transform
    (consistent with the size channel's own NaN handling)."""
    filled = vals.fillna(vals.median() if vals.notna().any() else 0.0)
    if log_transform:
        transformed = np.log10(filled.clip(lower=1e-3))
        return transformed, float(transformed.min()), float(transformed.max())
    return filled, float(filled.min()), float(filled.max())


def build_plotly_map(df):
    # (column, label, colorscale, log_transform_for_color)
    # Viridis/Plasma (perceptually uniform, high-contrast) for the two
    # genuinely continuous/sequential metrics; Inferno + log10 color
    # transform for population specifically, since it's heavily
    # right-skewed and a linear scale would otherwise wash out every
    # low-population cell into the same pale color.
    metrics = [
        ("population", "Population", "Inferno", True),
        ("weight", "Sampling weight (%)", "Plasma", False),
        ("elevation_m", "Elevation (m)", "Viridis", False),
    ]

    center, zoom = bbox_center_zoom(df["lat"], df["lon"])
    fig = go.Figure()
    customdata = np.stack([
        df["population"].fillna(-1),
        df["weight"].fillna(-1),
        df["elevation_m"].fillna(-9999),
    ], axis=-1)

    # go.Scattermapbox's marker object has NO "line" property at all (Mapbox
    # GL traces don't support a per-marker stroke the way go.Scatter/
    # Scattergeo do — confirmed by Plotly raising ValueError on marker.line
    # here) so "a thin dark marker border" is delivered via the standard
    # workaround instead: a solid-black "halo" trace drawn slightly larger
    # and directly UNDERNEATH each colored trace (added to the figure
    # first), giving the same visual border effect. Each dropdown button
    # must therefore toggle BOTH the halo and the colored trace for its
    # metric together — see trace_pairs below.
    trace_pairs = []
    for i, (col, label, colorscale, log_transform) in enumerate(metrics):
        vals = df[col]
        sizes = pixel_size(vals)
        halo_sizes = [s + 3.0 for s in sizes]
        color_vals, cmin, cmax = color_channel(vals, log_transform)
        colorbar_title = f"log10({label})" if log_transform else label

        halo_idx = len(fig.data)
        fig.add_trace(go.Scattermapbox(
            lat=df["lat"], lon=df["lon"], mode="markers",
            marker=dict(size=halo_sizes, color="black", opacity=1.0),
            hoverinfo="skip", showlegend=False,
            visible=(i == 0), name=f"{label} (border)",
        ))

        color_idx = len(fig.data)
        fig.add_trace(go.Scattermapbox(
            lat=df["lat"], lon=df["lon"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=color_vals,
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                showscale=True,
                opacity=MARKER_OPACITY,
                colorbar=dict(title=dict(text=colorbar_title, font=dict(color="#e8e8e8")),
                               x=1.0, tickfont=dict(size=11, color="#e8e8e8")),
            ),
            text=df["point_id"],
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Population: %{customdata[0]:,.0f}<br>"
                "Weight: %{customdata[1]:.3%}<br>"
                "Elevation: %{customdata[2]:.1f} m"
                "<extra></extra>"
            ),
            visible=(i == 0),
            name=label,
        ))
        trace_pairs.append((halo_idx, color_idx))

    n_traces = len(fig.data)
    buttons = []
    for i, (col, label, _, _) in enumerate(metrics):
        visible_mask = [False] * n_traces
        halo_idx, color_idx = trace_pairs[i]
        visible_mask[halo_idx] = True
        visible_mask[color_idx] = True
        buttons.append(dict(
            label=label,
            method="update",
            args=[{"visible": visible_mask},
                  {"title": (f"Rajasthan population-weighted sampling grid (320 points, "
                             f"{COVERAGE_TARGET_PCT:.1f}% cumulative-population coverage) "
                             f"— sized/colored by {label}")}],
        ))

    fig.update_layout(
        mapbox=dict(style=MAPBOX_STYLE, center=center, zoom=zoom),
        title=(f"Rajasthan population-weighted sampling grid (320 points, "
               f"{COVERAGE_TARGET_PCT:.1f}% cumulative-population coverage) "
               f"— sized/colored by {metrics[0][1]}"),
        updatemenus=[dict(
            buttons=buttons, direction="down", x=0.01, y=0.99,
            xanchor="left", yanchor="top", showactive=True,
        )],
        margin=dict(l=0, r=0, t=60, b=0),
        height=720,
        paper_bgcolor="#1a1a1a", font=dict(color="#e8e8e8"),
    )

    GRID_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(GRID_PLOT_FILE))
    log(f"Saved: {GRID_PLOT_FILE}  (dropdown: {[m[1] for m in metrics]}, "
        f"basemap={MAPBOX_STYLE}, opacity={MARKER_OPACITY})")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  POPULATION GRID VISUALIZATION — Rajasthan")
    print("=" * 68)

    df = load_points()
    build_plotly_map(df)

    print("=" * 68)
    print("  DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
