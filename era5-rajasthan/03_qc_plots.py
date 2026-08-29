"""
QC PLOTS — RAJASTHAN POPULATION POINTS  (sanity-check visualizations)
=============================================================================
Not final results — throwaway sanity-check plots meant to catch problems
during data acquisition (Week 1-2 of Objective 1), before preprocessing
starts. Reads only the processed/status CSVs the earlier scripts already
produce; never touches raw NetCDF/JSON caches (that's 02b's job).

Every plot is independently skippable: if an input file the plot needs
doesn't exist yet (e.g. elevation hasn't been attached, downloads are still
running), this script prints a warning and moves on rather than crashing —
so it stays runnable at any point during acquisition, not just once
everything is finished.

Single-state note: population_grid_points.csv in this pipeline has no
`state` column — this folder (era5-rajasthan) only ever produces Rajasthan
points, unlike the sibling era5-uttarakhand/ and era5-tamilnadu-pipeline/
directories, which are separate pipeline instances with their own separate
outputs. Anything described elsewhere as "per-state" (elevation box plot,
per-state point counts) is therefore a single Rajasthan-only group here.

Folium maps (spatial QC):
  outputs/qc_population_map.html        — points sized by population, colored by weight
  outputs/qc_elevation_map.html         — points colored by elevation_m, flags default/NaN
  outputs/qc_download_status_map.html   — points colored by ERA5+POWER completion

Plotly charts (distributional / time-series QC):
  outputs/qc_population_weight_scatter.html
  outputs/qc_population_histogram.html
  outputs/qc_elevation_histogram.html
  outputs/qc_elevation_boxplot.html
  outputs/qc_suntimes_line.html
  outputs/qc_download_status_by_year.html
  outputs/qc_rejection_window.html      — only if climate_rajasthan_points.csv
                                           carries matched-time columns (see note
                                           in build_rejection_window_plot())

HOW TO RUN:
  python 03_qc_plots.py

Safe to re-run at any time — every output is simply overwritten.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import folium
import branca.colormap as bcm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    POPULATION_GRID_FILE,
    SUNTIMES_FILE,
    COMBINED_POINTS_FILE,
    POINTS_DOWNLOAD_STATUS_FILE,
    POWER_DOWNLOAD_STATUS_FILE,
    RAW_BOUNDARY_DIR,
    OUTPUTS_DIR,
    ensure_data_dirs,
)

ensure_data_dirs()

YEARS = [str(y) for y in range(2016, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
BOUNDARY_FILE = RAW_BOUNDARY_DIR / "gadm41_IND_1.json"
BOUNDARY_STATE_NAME = "Rajasthan"

REJECTION_WINDOW_HOURS = 3  # matches 02_combine_rajasthan.py's MAX_MATCH_HOURS


def warn_skip(label, reason):
    print(f"  [SKIP] {label} — {reason}")


def safe_read_csv(path, label, **kwargs):
    if not path.exists():
        warn_skip(label, f"{path} not found")
        return None
    try:
        df = pd.read_csv(path, **kwargs)
    except Exception as exc:
        warn_skip(label, f"failed to read {path}: {exc}")
        return None
    if df.empty:
        warn_skip(label, f"{path} exists but is empty")
        return None
    return df


# ═══════════════════════════════════════════════════════════
# 1. POPULATION SAMPLING MAP
# ═══════════════════════════════════════════════════════════

def build_population_map(points_df):
    if points_df is None:
        warn_skip("population map", "population_grid_points.csv not available")
        return

    center = [points_df["lat"].mean(), points_df["lon"].mean()]
    m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=7)

    add_boundary_overlay(m)

    colormap = bcm.linear.YlOrRd_09.scale(points_df["weight"].min(), points_df["weight"].max())
    colormap.caption = "Sampling weight"

    pop_min, pop_max = points_df["population"].min(), points_df["population"].max()
    pop_span = max(pop_max - pop_min, 1e-9)

    for row in points_df.itertuples(index=False):
        frac = (row.population - pop_min) / pop_span
        radius = 3 + 12 * np.sqrt(max(frac, 0))
        color = colormap(row.weight)
        popup = folium.Popup(
            f"<b>{row.point_id}</b><br>"
            f"lat: {row.lat:.4f}<br>lon: {row.lon:.4f}<br>"
            f"population: {row.population:,.0f}<br>"
            f"weight: {row.weight:.6f}",
            max_width=250,
        )
        folium.CircleMarker(
            location=[row.lat, row.lon], radius=radius,
            color=color, weight=1, fill=True, fill_color=color, fill_opacity=0.75,
            popup=popup,
        ).add_to(m)

    colormap.add_to(m)
    out = OUTPUTS_DIR / "qc_population_map.html"
    m.save(str(out))
    print(f"  [OK]   population map -> {out}")


def add_boundary_overlay(m):
    """Overlay the Rajasthan boundary from the GADM GeoJSON cached by
    00a_build_population_grid.py, if it's still on disk. Reads the raw
    GeoJSON directly (no geopandas dependency needed for this QC script)."""
    if not BOUNDARY_FILE.exists():
        warn_skip("boundary overlay", f"{BOUNDARY_FILE} not found (run 00a first, "
                                       "or it's fine to skip — map still works without it)")
        return
    try:
        with open(BOUNDARY_FILE, "r", encoding="utf-8") as f:
            geo = json.load(f)
        features = [ft for ft in geo.get("features", [])
                    if ft.get("properties", {}).get("NAME_1") == BOUNDARY_STATE_NAME]
        if not features:
            warn_skip("boundary overlay", f"no NAME_1=={BOUNDARY_STATE_NAME} feature in {BOUNDARY_FILE}")
            return
        rj_geojson = {"type": "FeatureCollection", "features": features}
        folium.GeoJson(
            rj_geojson, name="Rajasthan boundary",
            style_function=lambda _: {"color": "#333333", "weight": 2, "fill": False},
        ).add_to(m)
    except Exception as exc:
        warn_skip("boundary overlay", f"failed to load/parse {BOUNDARY_FILE}: {exc}")


# ═══════════════════════════════════════════════════════════
# 2. ELEVATION MAP
# ═══════════════════════════════════════════════════════════

def build_elevation_map(points_df):
    if points_df is None:
        warn_skip("elevation map", "population_grid_points.csv not available")
        return
    if "elevation_m" not in points_df.columns:
        warn_skip("elevation map", "no elevation_m column — run 00c_attach_elevation.py first")
        return

    center = [points_df["lat"].mean(), points_df["lon"].mean()]
    m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=7)

    valid = points_df["elevation_m"].dropna()
    vmin = float(valid.min()) if not valid.empty else 0.0
    vmax = float(valid.max()) if not valid.empty else 1.0
    terrain_cmap = bcm.LinearColormap(
        colors=["#1a9850", "#66bd63", "#fee08b", "#f46d43", "#a50026"],
        vmin=vmin, vmax=max(vmax, vmin + 1e-9),
    )
    terrain_cmap.caption = "Elevation (m)"

    # NaN is the only real failure signature in this file: 00c either
    # computes a real per-point value from the geopotential field or raises
    # (it never writes a hardcoded fallback here). DEFAULT_ALT_M=300 is only
    # ever applied transiently inside 02_combine_rajasthan.py's merge step,
    # never written back to population_grid_points.csv — so a value merely
    # *close to* 300 is real terrain, not a failed attach, and isn't flagged.
    flagged = 0
    for row in points_df.itertuples(index=False):
        elev = getattr(row, "elevation_m", np.nan)
        is_bad = pd.isna(elev)
        popup = folium.Popup(
            f"<b>{row.point_id}</b><br>elevation: "
            f"{'NaN' if pd.isna(elev) else f'{elev:.1f} m'}"
            + ("<br><b>⚠️ elevation attach missing for this point</b>" if is_bad else ""),
            max_width=250,
        )
        if is_bad:
            flagged += 1
            folium.CircleMarker(
                location=[row.lat, row.lon], radius=6,
                color="black", weight=2, fill=True,
                fill_color="gray", fill_opacity=0.9, popup=popup,
            ).add_to(m)
        else:
            color = terrain_cmap(elev)
            folium.CircleMarker(
                location=[row.lat, row.lon], radius=5,
                color=color, weight=1, fill=True,
                fill_color=color, fill_opacity=0.8, popup=popup,
            ).add_to(m)

    terrain_cmap.add_to(m)
    out = OUTPUTS_DIR / "qc_elevation_map.html"
    m.save(str(out))
    print(f"  [OK]   elevation map -> {out}  "
          f"({flagged} point(s) flagged as NaN elevation)")


# ═══════════════════════════════════════════════════════════
# 3. DOWNLOAD COVERAGE MAP
# ═══════════════════════════════════════════════════════════

def era5_completion():
    """ERA5 downloads are ONE bbox-wide request per (year, month, var_type)
    — not per point — so completion is a single pipeline-wide figure, not a
    per-point one. Every point is affected identically by ERA5 gaps."""
    status_df = safe_read_csv(POINTS_DOWNLOAD_STATUS_FILE, "ERA5 download status", dtype=str)
    expected = len(YEARS) * len(MONTHS) * 2
    if status_df is None:
        return 0.0, 0, expected
    ok_combos = set(
        (r.year, r.month, str(r.var_type).strip())
        for r in status_df[status_df["status"] == "OK"].itertuples(index=False)
    )
    n_ok = len(ok_combos)
    return (100.0 * n_ok / expected if expected else 0.0), n_ok, expected


def power_completion_by_point():
    """Fraction of the 10 expected years OK for each point_id."""
    status_df = safe_read_csv(POWER_DOWNLOAD_STATUS_FILE, "NASA POWER download status", dtype=str)
    if status_df is None:
        return {}, None
    ok = status_df[status_df["status"] == "OK"]
    counts = ok.groupby("point_id")["year"].nunique().to_dict()
    return counts, status_df


def build_download_status_map(points_df):
    if points_df is None:
        warn_skip("download status map", "population_grid_points.csv not available")
        return

    era5_pct, era5_ok, era5_expected = era5_completion()
    era5_label = "complete" if era5_pct >= 99.9 else ("partial" if era5_pct > 0 else "missing")

    power_ok_years, power_status_df = power_completion_by_point()
    n_years_expected = len(YEARS)

    center = [points_df["lat"].mean(), points_df["lon"].mean()]
    m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=7)

    color_map = {"green": "#1a9850", "yellow": "#fdae61", "red": "#d73027"}
    counts = {"green": 0, "yellow": 0, "red": 0}

    for row in points_df.itertuples(index=False):
        years_ok = power_ok_years.get(row.point_id, 0)
        power_frac = years_ok / n_years_expected if n_years_expected else 0.0
        power_label = "complete" if power_frac >= 0.999 else ("partial" if power_frac > 0 else "missing")

        if era5_label == "complete" and power_label == "complete":
            bucket = "green"
        elif era5_label == "missing" and power_label == "missing":
            bucket = "red"
        else:
            bucket = "yellow"
        counts[bucket] += 1

        note = ""
        if power_status_df is not None:
            fails = power_status_df[
                (power_status_df["point_id"] == row.point_id) & (power_status_df["status"] == "FAIL")]
            if not fails.empty:
                last = fails.sort_values("timestamp").iloc[-1]
                note = f"<br>last POWER FAIL ({last['timestamp']}): {last['note']}"

        popup = folium.Popup(
            f"<b>{row.point_id}</b><br>"
            f"ERA5 (pipeline-wide): {era5_label} ({era5_ok}/{era5_expected} combos)<br>"
            f"NASA POWER: {power_label} ({years_ok}/{n_years_expected} years)"
            f"{note}",
            max_width=280,
        )
        folium.CircleMarker(
            location=[row.lat, row.lon], radius=5,
            color=color_map[bucket], weight=1, fill=True,
            fill_color=color_map[bucket], fill_opacity=0.85, popup=popup,
        ).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; line-height: 1.5;">
      <b>Download status</b><br>
      <span style="color:{color_map['green']}">●</span> both sources complete ({counts['green']})<br>
      <span style="color:{color_map['yellow']}">●</span> one source missing/partial ({counts['yellow']})<br>
      <span style="color:{color_map['red']}">●</span> both missing ({counts['red']})<br>
      <span style="font-size:11px;color:#666">ERA5 status is pipeline-wide
      (one bbox download) — identical for every point.</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out = OUTPUTS_DIR / "qc_download_status_map.html"
    m.save(str(out))
    print(f"  [OK]   download status map -> {out}  "
          f"(green={counts['green']} yellow={counts['yellow']} red={counts['red']})")


# ═══════════════════════════════════════════════════════════
# 4/5. PLOTLY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════

def build_population_charts(points_df):
    if points_df is None:
        warn_skip("population charts", "population_grid_points.csv not available")
        return

    fig = px.scatter(
        points_df, x="population", y="weight", hover_name="point_id",
        title="Population vs sampling weight (Rajasthan)",
        labels={"population": "Population", "weight": "Sampling weight"},
    )
    out = OUTPUTS_DIR / "qc_population_weight_scatter.html"
    fig.write_html(str(out))
    print(f"  [OK]   population/weight scatter -> {out}")

    fig = px.histogram(
        points_df, x="population", nbins=40,
        title="Population per retained point (should skew toward high-population cells)",
        labels={"population": "Population"},
    )
    out = OUTPUTS_DIR / "qc_population_histogram.html"
    fig.write_html(str(out))
    print(f"  [OK]   population histogram -> {out}")


def build_elevation_charts(points_df):
    if points_df is None:
        warn_skip("elevation charts", "population_grid_points.csv not available")
        return
    if "elevation_m" not in points_df.columns:
        warn_skip("elevation charts", "no elevation_m column — run 00c_attach_elevation.py first")
        return

    fig = px.histogram(
        points_df, x="elevation_m", nbins=40,
        title="Elevation distribution (Rajasthan)",
        labels={"elevation_m": "Elevation (m)"},
    )
    out = OUTPUTS_DIR / "qc_elevation_histogram.html"
    fig.write_html(str(out))
    print(f"  [OK]   elevation histogram -> {out}")

    df = points_df.copy()
    df["group"] = "Rajasthan"
    fig = px.box(
        df, x="group", y="elevation_m", points="all",
        title="Elevation spread — Rajasthan only "
              "(no cross-state comparison available in this single-state pipeline; "
              "compare against era5-uttarakhand's own output separately if needed)",
        labels={"group": "", "elevation_m": "Elevation (m)"},
    )
    out = OUTPUTS_DIR / "qc_elevation_boxplot.html"
    fig.write_html(str(out))
    print(f"  [OK]   elevation box plot -> {out}")


# ═══════════════════════════════════════════════════════════
# 6. SUN-TIMES SANITY CHECK
# ═══════════════════════════════════════════════════════════

def pick_representative_points(points_df, n=5):
    if points_df is None or len(points_df) == 0:
        return []
    sorted_df = points_df.sort_values("lon").reset_index(drop=True)
    n = min(n, len(sorted_df))
    idx = np.linspace(0, len(sorted_df) - 1, n).round().astype(int)
    idx = sorted(set(idx))
    return sorted_df.loc[idx, "point_id"].tolist()


def build_suntimes_chart(points_df):
    sun_df = safe_read_csv(SUNTIMES_FILE, "suntimes chart")
    if sun_df is None:
        return

    point_ids = pick_representative_points(points_df, n=5)
    if not point_ids:
        point_ids = sun_df["point_id"].drop_duplicates().head(5).tolist()
        print("  [WARN] population_grid_points.csv unavailable — picking arbitrary "
              "points instead of spanning longitude for the suntimes chart")

    sub = sun_df[sun_df["point_id"].isin(point_ids)].copy()
    if sub.empty:
        warn_skip("suntimes chart", "none of the selected points appear in suntimes.csv")
        return

    sub["time_utc"] = pd.to_datetime(sub["time_utc"], utc=True)
    sub["date"] = pd.to_datetime(sub["date"])
    sub["decimal_hour"] = (sub["time_utc"].dt.hour
                            + sub["time_utc"].dt.minute / 60.0
                            + sub["time_utc"].dt.second / 3600.0)

    events = ["sunrise", "noon", "sunset"]
    fig = make_subplots(rows=len(events), cols=1, shared_xaxes=True,
                         subplot_titles=[e.capitalize() for e in events],
                         vertical_spacing=0.06)

    wraparound_seen = False
    for r, event in enumerate(events, start=1):
        ev_df = sub[sub["event"] == event]
        for i, pid in enumerate(point_ids):
            pt_df = ev_df[ev_df["point_id"] == pid].sort_values("date")
            if pt_df.empty:
                continue
            jump = pt_df["decimal_hour"].diff().abs() > 12
            if jump.any():
                wraparound_seen = True
            fig.add_trace(
                go.Scatter(
                    x=pt_df["date"], y=pt_df["decimal_hour"], mode="lines",
                    name=pid, legendgroup=pid, showlegend=(r == 1),
                    line=dict(width=1),
                ),
                row=r, col=1,
            )
        fig.update_yaxes(title_text="UTC hour", range=[0, 24], row=r, col=1)

    if wraparound_seen:
        fig.add_annotation(
            text="Dips/jumps near 0h/24h reflect the documented cross-midnight "
                 "UTC wraparound (see README) — not a data bug.",
            xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
            font=dict(size=11, color="#666"),
        )

    fig.update_layout(
        title="Sun-event UTC times across 2016-2025 (representative points, spanning longitude)",
        height=850,
    )
    out = OUTPUTS_DIR / "qc_suntimes_line.html"
    fig.write_html(str(out))
    print(f"  [OK]   suntimes line chart -> {out}  (points: {', '.join(point_ids)})")


# ═══════════════════════════════════════════════════════════
# 7. DOWNLOAD STATUS OVER TIME
# ═══════════════════════════════════════════════════════════

def bucket_status_by_year(status_df, key_cols, expected_keys_by_year):
    """expected_keys_by_year: {year: set(of expected key-tuples for that year)}.
    Returns {year: {"complete": n, "failed": n, "partial": n}} where
    "partial" here means "not yet attempted" (no row at all) — appropriate
    for a QC pass run mid-acquisition, not a true 3-way per-unit status
    (each atomic combo/point-year is itself only ever OK or FAIL)."""
    ok_keys, fail_keys = set(), set()
    if status_df is not None:
        for r in status_df.itertuples(index=False):
            key = tuple(getattr(r, c) for c in key_cols)
            if r.status == "OK":
                ok_keys.add(key)
            elif r.status == "FAIL":
                fail_keys.add(key)
    fail_keys -= ok_keys  # a later retry that succeeded overrides an earlier FAIL

    result = {}
    for year, expected in expected_keys_by_year.items():
        complete = len(expected & ok_keys)
        failed = len(expected & fail_keys)
        partial = len(expected) - complete - failed
        result[year] = {"complete": complete, "failed": failed, "partial": partial}
    return result


def build_download_status_by_year_chart(points_df):
    era5_status_df = safe_read_csv(POINTS_DOWNLOAD_STATUS_FILE, "ERA5 status-by-year chart", dtype=str)
    power_status_df = safe_read_csv(POWER_DOWNLOAD_STATUS_FILE, "NASA POWER status-by-year chart", dtype=str)

    if era5_status_df is None and power_status_df is None:
        warn_skip("download status by year", "neither status CSV is available")
        return

    era5_expected = {
        y: {(y, m, vt) for m in MONTHS for vt in ("instant", "accum")}
        for y in YEARS
    }
    era5_buckets = bucket_status_by_year(
        era5_status_df.assign(var_type=era5_status_df["var_type"].str.strip())
        if era5_status_df is not None else None,
        ["year", "month", "var_type"], era5_expected,
    )

    point_ids = points_df["point_id"].tolist() if points_df is not None else (
        power_status_df["point_id"].unique().tolist() if power_status_df is not None else []
    )
    power_expected = {y: {(pid, y) for pid in point_ids} for y in YEARS}
    power_buckets = bucket_status_by_year(power_status_df, ["point_id", "year"], power_expected)

    rows = []
    for y in YEARS:
        for status, n in era5_buckets.get(y, {}).items():
            rows.append({"year": y, "source": "ERA5", "status": status, "count": n})
        for status, n in power_buckets.get(y, {}).items():
            rows.append({"year": y, "source": "NASA POWER", "status": status, "count": n})
    chart_df = pd.DataFrame(rows)

    fig = px.bar(
        chart_df, x="year", y="count", color="status", facet_row="source",
        barmode="group",
        category_orders={"status": ["complete", "partial", "failed"]},
        color_discrete_map={"complete": "#1a9850", "partial": "#fdae61", "failed": "#d73027"},
        title="Download completion by year (partial = not yet attempted)",
        labels={"count": "combinations", "year": "Year"},
    )
    fig.update_yaxes(matches=None)
    out = OUTPUTS_DIR / "qc_download_status_by_year.html"
    fig.write_html(str(out))
    print(f"  [OK]   download status by year -> {out}")

    return era5_buckets, power_buckets


# ═══════════════════════════════════════════════════════════
# 8. REJECTION-WINDOW PREVIEW
# ═══════════════════════════════════════════════════════════

def build_rejection_window_plot():
    combined_df = safe_read_csv(COMBINED_POINTS_FILE, "rejection-window preview")
    if combined_df is None:
        return

    matched_cols = {"era5": "era5_matched_time_utc", "power": "power_matched_time_utc"}
    have_any = any(c in combined_df.columns for c in matched_cols.values())
    if not have_any:
        warn_skip(
            "rejection-window preview",
            f"{COMBINED_POINTS_FILE.name} doesn't carry the matched-reading "
            "timestamp columns (era5_matched_time_utc / power_matched_time_utc) — "
            "02_combine_rajasthan.py currently only writes the requested "
            "time_utc, not the actual matched ERA5/POWER timestamp, so the "
            "true offset can't be recovered from this file as-is.",
        )
        return

    fig = go.Figure()
    for source, col in matched_cols.items():
        if col not in combined_df.columns:
            continue
        requested = pd.to_datetime(combined_df["time_utc"], utc=True)
        matched = pd.to_datetime(combined_df[col], utc=True)
        offset_hours = (matched - requested).dt.total_seconds().abs() / 3600.0
        fig.add_trace(go.Histogram(x=offset_hours.dropna(), name=source, opacity=0.6))

    fig.update_layout(barmode="overlay", title="Requested vs matched reading time offset")
    fig.add_vline(x=REJECTION_WINDOW_HOURS, line_dash="dash", line_color="red",
                  annotation_text=f"{REJECTION_WINDOW_HOURS}h rejection threshold")
    out = OUTPUTS_DIR / "qc_rejection_window.html"
    fig.write_html(str(out))
    print(f"  [OK]   rejection-window preview -> {out}")


# ═══════════════════════════════════════════════════════════
# QC SUMMARY (stdout)
# ═══════════════════════════════════════════════════════════

def print_qc_summary(points_df, era5_buckets, power_buckets):
    print("\n" + "═" * 68)
    print("  QC SUMMARY")
    print("═" * 68)

    if points_df is not None:
        print(f"  Points (Rajasthan)     : {len(points_df)}")
        print(f"  Sum of retained population : {points_df['population'].sum():,.0f}")
        print("  (Total state population isn't persisted by 00a, so the exact "
              "realized coverage fraction can't be recomputed here — 00a targets "
              "~87.5% by construction.)")
        if "elevation_m" in points_df.columns:
            elev = points_df["elevation_m"].dropna()
            if not elev.empty:
                print(f"  Elevation (Rajasthan)  : min={elev.min():.1f}m  "
                      f"max={elev.max():.1f}m  mean={elev.mean():.1f}m")
            n_bad = points_df["elevation_m"].isna().sum()
            if n_bad:
                print(f"  ⚠️  {n_bad} point(s) with NaN elevation "
                      "(attach missing) — see qc_elevation_map.html")
        else:
            print("  Elevation              : not attached yet (run 00c_attach_elevation.py)")
    else:
        print("  Points                 : population_grid_points.csv not available")

    if era5_buckets:
        print("\n  ERA5 completion % per year (pipeline-wide, 24 combos/year):")
        for y in YEARS:
            b = era5_buckets.get(y, {"complete": 0, "failed": 0, "partial": 0})
            total = sum(b.values()) or 1
            print(f"    {y}: complete={100*b['complete']/total:5.1f}%  "
                  f"failed={100*b['failed']/total:5.1f}%  partial={100*b['partial']/total:5.1f}%")

    if power_buckets:
        print("\n  NASA POWER completion % per year (point-years):")
        for y in YEARS:
            b = power_buckets.get(y, {"complete": 0, "failed": 0, "partial": 0})
            total = sum(b.values()) or 1
            print(f"    {y}: complete={100*b['complete']/total:5.1f}%  "
                  f"failed={100*b['failed']/total:5.1f}%  partial={100*b['partial']/total:5.1f}%")

    print("═" * 68)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 68)
    print("  QC PLOTS — Rajasthan Population Points")
    print(f"  Output dir: {OUTPUTS_DIR}")
    print("═" * 68)

    points_df = safe_read_csv(POPULATION_GRID_FILE, "population points")

    print("\n[Folium maps]")
    build_population_map(points_df)
    build_elevation_map(points_df)
    build_download_status_map(points_df)

    print("\n[Plotly charts]")
    build_population_charts(points_df)
    build_elevation_charts(points_df)
    build_suntimes_chart(points_df)
    buckets = build_download_status_by_year_chart(points_df)
    era5_buckets, power_buckets = buckets if buckets else (None, None)
    build_rejection_window_plot()

    print_qc_summary(points_df, era5_buckets, power_buckets)

    print(f"\nAll available QC plots written to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
