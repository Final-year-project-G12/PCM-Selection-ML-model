"""
03b_coverage_viz_rajasthan.py
=============================================================================
ERA5 vs NASA POWER COVERAGE + AGREEMENT VISUALIZATIONS — RAJASTHAN
=============================================================================
Two self-contained, standalone HTML visualizations built from
03b_agreement_analysis.py's inputs/output, split into a spatial view and a
temporal view. Read-only with respect to every input file below — this
script never writes back to any of them.

  1. outputs/spatial_coverage_map.html   (folium)
     Population-grid points colored two ways (LayerControl toggle):
       - data completeness % (green->red diverging colormap)
       - mean GHI MBE, ERA5 - POWER (red-blue diverging, centered at 0)
     Points that would independently fail 03b_agreement_analysis.py's own
     BACKBONE/QUANTILE_MAP/MANUAL_REVIEW decision thresholds (applied here
     per-point, see NOTE below) get a thicker black border plus a warning
     marker in a separate togglable layer.

  2. outputs/temporal_coverage_plot.html (plotly)
     - A point_id x month heatmap with a dropdown across THREE views, since
       Phase 1 confirmed 240/240 ERA5 files and 3200/3200 POWER files
       downloaded — meaning plain presence/absence has ~no variance and
       renders as a single flat color:
         (a) Null rate % — fraction of CORE_VALUE_COLS that are NaN among
             present rows (post 02_combine_rajasthan.py's physical-
             plausibility filtering: GHI>1400->NaN, T_amb outside
             [-5,60]->NaN). Sequential Reds, zmin=0, zmax=observed max
             (not an assumed 0-100% range). If this comes out uniformly 0
             (it does, in the current run — see console log), that is
             reported as a finding, not hidden: 0 nulls survived that
             filtering for the entire dataset.
         (b) GHI |MBE| magnitude (W/m^2) — reuses the same per-row ERA5-
             POWER diff as the spatial map's per-point stat, but kept at
             (point_id, month) grain, where the known season-level MBE
             spread (see bias_decision_rajasthan.txt) gives it real
             variance. Sequential Reds, zmin=0, zmax=observed max.
         (c) Availability (source x event coverage) — the original binary
             encoding, but no longer collapsing "one source missing for
             all 3 events" and "one event missing for both sources" into
             the same 0.5: each cell is the mean of 6 (source, event)
             completeness fractions (2 sources x 3 events), with a hover
             breakdown ("ERA5: 3/3 events fully present, POWER: 2/3
             events fully present") so the composition is inspectable,
             not just the collapsed number. Colorbar is labeled "1.0 =
             fully available (expected given 240/240, 3200/3200 download
             completeness)" so a flat-green render reads as confirmation.
       A fourth candidate metric — the +/-3h nearest-hour match-rejection
       rate — is NOT included: 02_combine_rajasthan.py's nearest_row()
       (MAX_MATCH_HOURS=3) simply returns None on rejection and nothing
       downstream logs a per-row or per-stratum rejection count anywhere
       on disk, so this metric is not currently derivable from any existing
       output file. Documented here (not just a code comment) per request;
       skipped rather than faked.
     - A season x event MBE/RMSE/Pearson-r chart, faceted by variable
       (GHI, T_amb, RHum, W_spd), with a dropdown to switch metric and a
       y=0 reference line shown only for MBE.
     - An annotation box pulling the branch decision and n-per-stratum out
       of outputs/bias_decision_rajasthan.txt (annotation only — parsed,
       never recomputed).

NOTE on per-point MANUAL_REVIEW flagging: era5_power_agreement_rajasthan.csv
has NO point_id column — it is stratified only by variable x season x
event, so "which points are MANUAL_REVIEW" is not literally present in
that file. What's flagged here instead is a per-point re-application of
03b_agreement_analysis.py's own decide_branch() thresholds (CORR_GOOD,
CORR_SEVERE, MBE_SMALL_FRAC — mirrored below, not imported, since a
digit-leading filename can't be `import`-ed normally) to each point's own
GHI r / MBE, computed directly from climate_rajasthan_points.csv. This
identifies points that would independently fail the same test the global
decision used, not points literally rows of the agreement CSV. This caveat
is rendered as a static legend caption on the map itself (not just here),
since "0 flagged" reads very differently once you know the flag is
strata-level, not a true per-point classification.

INPUTS:
  data/processed/climate_rajasthan_points.csv        (02_combine_rajasthan.py)
  data/processed/era5_power_agreement_rajasthan.csv  (03b_agreement_analysis.py)
  data/processed/population_grid_points.csv          (00a_build_population_grid.py)
  data/processed/suntimes.csv                        (00b_build_suntimes.py; optional
                                                        — the "expected" grid source)
  outputs/bias_decision_rajasthan.txt                (03b_agreement_analysis.py;
                                                        optional, annotation only)

HOW TO RUN:
  python 03b_coverage_viz_rajasthan.py
"""

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import folium
import branca.colormap as bcm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    COMBINED_POINTS_FILE,
    POPULATION_GRID_FILE,
    SUNTIMES_FILE,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    ensure_data_dirs,
)

ensure_data_dirs()

AGREEMENT_FILE = PROCESSED_DIR / "era5_power_agreement_rajasthan.csv"
BIAS_DECISION_FILE = OUTPUTS_DIR / "bias_decision_rajasthan.txt"

SPATIAL_MAP_FILE = OUTPUTS_DIR / "spatial_coverage_map.html"
TEMPORAL_PLOT_FILE = OUTPUTS_DIR / "temporal_coverage_plot.html"

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
EVENT_ORDER = ["sunrise", "noon", "sunset"]
VARIABLE_ORDER = ["GHI", "T_amb", "RHum", "W_spd"]
VARIABLE_UNITS = {"GHI": "W/m²", "T_amb": "°C", "RHum": "%", "W_spd": "m/s"}

GHI_ERA5_COL = "era5_GHI"
GHI_POWER_COL = "power_ALLSKY_SFC_SW_DWN"
CORE_VALUE_COLS = [
    "era5_GHI", "power_ALLSKY_SFC_SW_DWN",
    "era5_T_amb", "power_T2M",
    "era5_RHum", "power_RH2M",
    "era5_W_spd", "power_WS10M",
]

# Mirrored from 03b_agreement_analysis.py's decide_branch() thresholds —
# see this file's module docstring NOTE for why they're duplicated, not imported.
CORR_GOOD = 0.90
CORR_SEVERE = 0.70
MBE_SMALL_FRAC = 0.05


def log(msg):
    print(f"  {msg}")


# ═══════════════════════════════════════════════════════════
# LOAD + VALIDATE
# ═══════════════════════════════════════════════════════════

def require_file(path, purpose):
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found — needed for {purpose}. "
                          f"Run the script that produces it first.")


def load_population_grid():
    require_file(POPULATION_GRID_FILE, "point lat/lon in the spatial map")
    df = pd.read_csv(POPULATION_GRID_FILE)
    required = ["point_id", "lat", "lon", "population", "weight", "elevation_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {POPULATION_GRID_FILE} is missing required column(s) {missing} — "
                          f"found columns: {list(df.columns)}.")
    if df.empty:
        raise SystemExit(f"ERROR: {POPULATION_GRID_FILE} exists but has no rows.")
    log(f"Loaded {len(df)} points from {POPULATION_GRID_FILE.name}.")
    return df


def load_agreement():
    require_file(AGREEMENT_FILE, "the MBE/RMSE/r faceted chart and stratum counts")
    df = pd.read_csv(AGREEMENT_FILE)
    required = ["variable", "season", "event", "n", "MBE", "RMSE", "pearson_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {AGREEMENT_FILE} is missing required column(s) {missing} — "
                          f"found columns: {list(df.columns)}. Re-run 03b_agreement_analysis.py.")
    if df.empty:
        raise SystemExit(f"ERROR: {AGREEMENT_FILE} exists but has no rows.")
    log(f"Loaded {len(df)} stratified rows from {AGREEMENT_FILE.name}.")
    return df


def check_climate_columns():
    require_file(COMBINED_POINTS_FILE, "per-point completeness/MBE stats and the availability heatmap")
    header = pd.read_csv(COMBINED_POINTS_FILE, nrows=0)
    needed = ["point_id", "date", "event"] + CORE_VALUE_COLS
    missing = [c for c in needed if c not in header.columns]
    if missing:
        raise SystemExit(f"ERROR: {COMBINED_POINTS_FILE} is missing required column(s) {missing} — "
                          f"found columns: {list(header.columns)}.")
    return header.columns


# ═══════════════════════════════════════════════════════════
# 1. PER-POINT AGGREGATION  (climate_rajasthan_points.csv -> one row/point_id)
# ═══════════════════════════════════════════════════════════

def classify_point(r, mbe_frac):
    if r is None or r != r:
        return "MANUAL_REVIEW"
    if r >= CORR_GOOD and mbe_frac == mbe_frac and mbe_frac <= MBE_SMALL_FRAC:
        return "BACKBONE"
    if r >= CORR_SEVERE:
        return "QUANTILE_MAP"
    return "MANUAL_REVIEW"


def compute_point_stats():
    check_climate_columns()
    needed = ["point_id", "date", "event"] + CORE_VALUE_COLS
    df = pd.read_csv(COMBINED_POINTS_FILE, usecols=needed)
    log(f"Loaded {len(df):,} rows from {COMBINED_POINTS_FILE.name} for per-point aggregation.")

    null_frac = df[CORE_VALUE_COLS].isna().mean(axis=1)
    diff = df[GHI_ERA5_COL] - df[GHI_POWER_COL]
    by_point = df["point_id"]

    stats = pd.DataFrame({
        "rows_present": by_point.groupby(by_point).size(),
        "mean_null_rate": null_frac.groupby(by_point).mean(),
        "mbe_ghi": diff.groupby(by_point).mean(),
        "rmse_ghi": diff.groupby(by_point).apply(lambda s: float(np.sqrt(np.nanmean(s ** 2)))),
        "mean_power_ghi": df[GHI_POWER_COL].groupby(by_point).mean(),
    })
    stats["r_ghi"] = df.groupby("point_id").apply(
        lambda g: g[GHI_ERA5_COL].corr(g[GHI_POWER_COL])
        if g[GHI_ERA5_COL].std() > 0 and g[GHI_POWER_COL].std() > 0 else np.nan
    )
    stats["abs_mbe_ghi"] = stats["mbe_ghi"].abs()

    if SUNTIMES_FILE.exists():
        exp_header = pd.read_csv(SUNTIMES_FILE, nrows=0)
        if "point_id" in exp_header.columns:
            expected_counts = pd.read_csv(SUNTIMES_FILE, usecols=["point_id"]).groupby("point_id").size()
            stats = stats.join(expected_counts.rename("expected_rows"), how="left")
        else:
            log(f"[WARN] {SUNTIMES_FILE.name} has no point_id column — completeness %% will fall back "
                f"to each point's own max observed row count as 'expected'.")
            stats["expected_rows"] = stats["rows_present"].max()
    else:
        log(f"[WARN] {SUNTIMES_FILE} not found — completeness %% will fall back to each point's own "
            f"max observed row count as 'expected' (relative, not the true intended grid).")
        stats["expected_rows"] = stats["rows_present"].max()

    stats["expected_rows"] = stats["expected_rows"].fillna(stats["rows_present"])
    stats["pct_rows_present"] = (stats["rows_present"] / stats["expected_rows"] * 100).clip(upper=100)

    mbe_frac = (stats["abs_mbe_ghi"] / stats["mean_power_ghi"]).replace([np.inf, -np.inf], np.nan)
    stats["branch"] = [classify_point(r, f) for r, f in zip(stats["r_ghi"], mbe_frac)]

    stats = stats.reset_index().rename(columns={"index": "point_id"})
    n_flag = int((stats["branch"] == "MANUAL_REVIEW").sum())
    log(f"Per-point stats computed for {len(stats)} points ({n_flag} classify as "
        f"MANUAL_REVIEW-equivalent under CORR_SEVERE={CORR_SEVERE}).")
    return stats


# ═══════════════════════════════════════════════════════════
# 2. SPATIAL MAP  (folium — two togglable colorings + MANUAL_REVIEW flags)
# ═══════════════════════════════════════════════════════════

FLAG_CAVEAT_TEXT = (
    "⚠️ MANUAL_REVIEW flags derive from strata-level (variable × season × event) "
    "thresholds re-applied per point, not a true per-point classification — "
    "\"0 flagged\" means no point's containing stratum failed the threshold, "
    "not that per-point review was individually ruled out."
)


def add_flag_caveat_caption(m):
    """Static, once-rendered legend caption (not repeated in every popup) —
    see this file's module docstring NOTE on per-point MANUAL_REVIEW flagging."""
    caption_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 8px 12px; max-width: 320px;
                font-size: 12px; line-height: 1.4; color: #333;
                border: 1px solid #999; border-radius: 4px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3);">
      {FLAG_CAVEAT_TEXT}
    </div>
    """
    m.get_root().html.add_child(folium.Element(caption_html))


def build_spatial_map(stats_df, pop_df):
    merged = pop_df.merge(stats_df, on="point_id", how="inner")
    n_dropped = len(pop_df) - len(merged)
    if n_dropped:
        log(f"[WARN] {n_dropped} population-grid point(s) had no matching per-point stats "
            f"(not present in {COMBINED_POINTS_FILE.name}) — excluded from the map.")
    if merged.empty:
        raise SystemExit("ERROR: no point_id overlap between population_grid_points.csv and "
                          "climate_rajasthan_points.csv — cannot build the spatial map.")

    center = [merged["lat"].mean(), merged["lon"].mean()]
    m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=7)

    comp_min = float(merged["pct_rows_present"].min())
    comp_max = float(merged["pct_rows_present"].max())
    completeness_cmap = bcm.linear.RdYlGn_11.scale(comp_min, max(comp_max, comp_min + 1e-9))
    completeness_cmap.caption = "Data completeness (% of expected rows present) — green = complete, red = gaps"

    max_abs_mbe = float(merged["mbe_ghi"].abs().max())
    max_abs_mbe = max_abs_mbe if max_abs_mbe > 0 else 1.0
    mbe_cmap = bcm.linear.RdBu_11.scale(-max_abs_mbe, max_abs_mbe)
    mbe_cmap.caption = "Mean GHI MBE, ERA5 - POWER (W/m^2) — red = ERA5 underestimates, blue = ERA5 overestimates"

    completeness_layer = folium.FeatureGroup(name="Colored by: completeness %", show=True)
    mbe_layer = folium.FeatureGroup(name="Colored by: mean GHI MBE", show=False)
    flag_layer = folium.FeatureGroup(name="MANUAL_REVIEW-equivalent flags", show=True)

    n_flagged = 0
    for row in merged.itertuples(index=False):
        r_txt = f"{row.r_ghi:.3f}" if pd.notna(row.r_ghi) else "N/A"
        popup_html = (
            f"<b>{row.point_id}</b><br>"
            f"Completeness: {row.pct_rows_present:.1f}%<br>"
            f"Mean GHI MBE (ERA5-POWER): {row.mbe_ghi:.2f} W/m²<br>"
            f"Mean GHI RMSE: {row.rmse_ghi:.2f} W/m²<br>"
            f"Mean GHI r: {r_txt}<br>"
            f"Null rate (core vars): {100 * row.mean_null_rate:.2f}%<br>"
            f"Per-point branch: <b>{row.branch}</b>"
        )
        is_flagged = row.branch == "MANUAL_REVIEW"
        if is_flagged:
            n_flagged += 1
        border_color = "black" if is_flagged else "#333333"
        border_weight = 3 if is_flagged else 1

        folium.CircleMarker(
            location=[row.lat, row.lon], radius=7, color=border_color, weight=border_weight,
            fill=True, fill_color=completeness_cmap(row.pct_rows_present), fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(completeness_layer)

        mbe_val = row.mbe_ghi if pd.notna(row.mbe_ghi) else 0.0
        folium.CircleMarker(
            location=[row.lat, row.lon], radius=7, color=border_color, weight=border_weight,
            fill=True, fill_color=mbe_cmap(mbe_val), fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(mbe_layer)

        if is_flagged:
            folium.Marker(
                location=[row.lat, row.lon],
                icon=folium.DivIcon(html='<div style="font-size:16px; transform: translate(-6px,-22px);">⚠️</div>'),
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(flag_layer)

    completeness_layer.add_to(m)
    mbe_layer.add_to(m)
    flag_layer.add_to(m)
    completeness_cmap.add_to(m)
    mbe_cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    add_flag_caveat_caption(m)

    SPATIAL_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(SPATIAL_MAP_FILE))
    log(f"Saved: {SPATIAL_MAP_FILE}  ({len(merged)} points, {n_flagged} flagged MANUAL_REVIEW-equivalent)")


# ═══════════════════════════════════════════════════════════
# 3. TEMPORAL — HEATMAP  (point_id x month, 3-view dropdown)
#    null rate % | GHI |MBE| magnitude | source x event availability
#    (see module docstring for why plain 1/0.5/0 availability alone is
#    ~flat given 240/240 ERA5 + 3200/3200 POWER download completeness,
#    and why a +/-3h match-rejection-rate view isn't included at all)
# ═══════════════════════════════════════════════════════════

def load_expected_event_counts():
    """Expected (point_id, year_month, event) row counts from suntimes.csv
    — the true intended grid, so a (point,date,event) combo entirely absent
    from climate_rajasthan_points.csv counts as missing, not just NaN
    values within rows that are present. Returns None (never raises) if
    unavailable — callers fall back to observed counts, which can't detect
    whole-row absence but everything else still works."""
    if not SUNTIMES_FILE.exists():
        log(f"[WARN] {SUNTIMES_FILE} not found — heatmap cells fall back to observed row counts as "
            f"'expected' (can't detect rows absent from {COMBINED_POINTS_FILE.name} entirely).")
        return None
    exp_header = pd.read_csv(SUNTIMES_FILE, nrows=0)
    if not {"point_id", "date", "event"}.issubset(exp_header.columns):
        log(f"[WARN] {SUNTIMES_FILE.name} missing point_id/date/event — same fallback as above.")
        return None
    exp = pd.read_csv(SUNTIMES_FILE, usecols=["point_id", "date", "event"])
    exp["year_month"] = exp["date"].str.slice(0, 7)
    return exp.groupby(["point_id", "year_month", "event"]).size().rename("n_expected")


def compute_temporal_cell_metrics():
    check_climate_columns()
    needed = ["point_id", "date", "event"] + CORE_VALUE_COLS
    df = pd.read_csv(COMBINED_POINTS_FILE, usecols=needed)
    log(f"Loaded {len(df):,} rows from {COMBINED_POINTS_FILE.name} for the temporal heatmap.")
    df["year_month"] = df["date"].str.slice(0, 7)
    df["era5_present"] = df[GHI_ERA5_COL].notna()
    df["power_present"] = df[GHI_POWER_COL].notna()
    df["null_frac_row"] = df[CORE_VALUE_COLS].isna().mean(axis=1)
    df["ghi_diff"] = df[GHI_ERA5_COL] - df[GHI_POWER_COL]

    # --- (c) availability: per (point, month, event) source completeness,
    # then averaged across the 3 events per source -> 6-way mean per cell ---
    event_actual = df.groupby(["point_id", "year_month", "event"]).agg(
        n_rows=("era5_present", "size"),
        era5_n=("era5_present", "sum"),
        power_n=("power_present", "sum"),
    )
    event_expected = load_expected_event_counts()
    if event_expected is None:
        event_expected = event_actual["n_rows"].rename("n_expected")

    event_grid = event_expected.to_frame().join(event_actual[["era5_n", "power_n"]], how="left").fillna(0)
    event_grid["era5_frac"] = (event_grid["era5_n"] / event_grid["n_expected"].replace(0, np.nan)).clip(0, 1)
    event_grid["power_frac"] = (event_grid["power_n"] / event_grid["n_expected"].replace(0, np.nan)).clip(0, 1)
    # "fully present" = this event's completeness fraction is (essentially) 1.0
    event_grid["era5_full"] = (event_grid["era5_frac"] >= 0.999).astype(int)
    event_grid["power_full"] = (event_grid["power_frac"] >= 0.999).astype(int)

    by_point_month = event_grid.groupby(["point_id", "year_month"])
    combined = pd.DataFrame({
        "era5_avg": by_point_month["era5_frac"].mean(),
        "power_avg": by_point_month["power_frac"].mean(),
        "era5_events_full": by_point_month["era5_full"].sum(),
        "power_events_full": by_point_month["power_full"].sum(),
    })
    # mean of 6 (source, event) fractions == mean of the two 3-event averages
    combined["availability"] = (combined["era5_avg"] + combined["power_avg"]) / 2

    # --- (a)/(b) null rate % and GHI |MBE| — (point, month) grain ---
    nm = df.groupby(["point_id", "year_month"]).agg(
        null_rate=("null_frac_row", "mean"),
        mbe_ghi=("ghi_diff", "mean"),
    )
    nm["abs_mbe_ghi"] = nm["mbe_ghi"].abs()

    combined = combined.join(nm, how="outer")
    return combined


def _wide(combined, col, row_order, col_order):
    w = combined[col].unstack("year_month")
    return w.reindex(index=row_order, columns=col_order)


def build_temporal_heatmap_figure(combined):
    row_order = sorted(combined.index.get_level_values(0).unique())
    col_order = sorted(combined.index.get_level_values(1).unique())
    shape = (len(row_order), len(col_order))

    null_wide = _wide(combined, "null_rate", row_order, col_order) * 100.0     # -> %
    abs_mbe_wide = _wide(combined, "abs_mbe_ghi", row_order, col_order)
    avail_wide = _wide(combined, "availability", row_order, col_order)
    era5_full_wide = _wide(combined, "era5_events_full", row_order, col_order)
    power_full_wide = _wide(combined, "power_events_full", row_order, col_order)

    avail_hover = np.empty(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            e, p = era5_full_wide.values[i, j], power_full_wide.values[i, j]
            e_txt = "n/a" if pd.isna(e) else f"{int(e)}/3"
            p_txt = "n/a" if pd.isna(p) else f"{int(p)}/3"
            avail_hover[i, j] = f"ERA5: {e_txt} events fully present<br>POWER: {p_txt} events fully present"

    null_max = float(np.nanmax(null_wide.values)) if np.isfinite(null_wide.values).any() else 0.0
    abs_mbe_max = float(np.nanmax(abs_mbe_wide.values)) if np.isfinite(abs_mbe_wide.values).any() else 0.0

    if null_max <= 1e-9:
        log("[FINDING] Null-rate heatmap is uniformly 0.00% — every row's core ERA5/POWER values "
            "survived 02_combine_rajasthan.py's physical-plausibility filtering (GHI>1400->NaN, "
            "T_amb outside [-5,60]->NaN) with zero rejections. This is a genuine finding given the "
            "confirmed 240/240 ERA5 + 3200/3200 POWER download completeness, not a broken plot — see "
            "the GHI |MBE| magnitude view for a metric with real spread.")
    if abs_mbe_max <= 1e-9:
        log("[FINDING] GHI |MBE| heatmap is uniformly ~0 W/m^2 at (point, month) grain — unexpected "
            "given the season-level MBE spread in bias_decision_rajasthan.txt; worth a second look.")

    # views: (key, dropdown label, z-wide, colorscale, zmin, zmax, hover text array or None,
    #         hovertemplate, colorbar title, page title)
    views = [
        ("null_rate", "Null rate % (post physical-plausibility filtering)",
         null_wide, "Reds", 0.0, max(null_max, 1e-6), None,
         "Point %{y}<br>Month %{x}<br>Null rate=%{z:.2f}%<extra></extra>",
         "Null rate %",
         "ERA5/POWER null rate (post physical-plausibility filtering) — point_id × month"),
        ("abs_mbe", "GHI |MBE| magnitude (secondary lens)",
         abs_mbe_wide, "Reds", 0.0, max(abs_mbe_max, 1e-6), None,
         "Point %{y}<br>Month %{x}<br>|MBE|=%{z:.2f} W/m²<extra></extra>",
         "|MBE| (W/m²)",
         "GHI |MBE| magnitude, ERA5 vs POWER — point_id × month"),
        ("availability", "Availability (source × event coverage)",
         avail_wide, "RdYlGn", 0.0, 1.0, avail_hover.tolist(),
         "Point %{y}<br>Month %{x}<br>Availability=%{z:.2f}<br>%{text}<extra></extra>",
         "Availability<br>1.0 = fully available<br>(expected given 240/240,<br>3200/3200 downloads)",
         "ERA5 + NASA POWER source × event availability — point_id × month"),
    ]

    default = views[0]
    fig = go.Figure(go.Heatmap(
        z=default[2].values, x=col_order, y=row_order,
        colorscale=default[3], zmin=default[4], zmax=default[5],
        text=default[6], hovertemplate=default[7],
        colorbar=dict(title=default[8]),
    ))

    buttons = []
    for key, label, wide, colorscale, zmin, zmax, hover_text, hovertemplate, cbar_title, page_title in views:
        buttons.append(dict(
            label=label, method="update",
            args=[
                {"z": [wide.values], "colorscale": [colorscale], "zmin": [zmin], "zmax": [zmax],
                 "text": [hover_text], "hovertemplate": [hovertemplate],
                 "colorbar.title.text": [cbar_title]},
                {"title": page_title},
            ],
        ))

    fig.update_layout(
        title=default[9],
        xaxis_title="Month", yaxis_title="point_id",
        height=max(500, min(1400, int(12 * len(row_order)))),
        margin=dict(l=90, r=40, t=90, b=60),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.1,
                           xanchor="left", yanchor="top", showactive=True)],
    )
    log(f"Built temporal heatmap ({shape[0]} points x {shape[1]} months), 3 views: "
        f"null_rate (max={null_max:.4f}%), abs_mbe (max={abs_mbe_max:.2f} W/m²), availability.")
    return fig


# ═══════════════════════════════════════════════════════════
# 4. TEMPORAL — FACETED MBE/RMSE/r CHART  (season x event, per variable)
# ═══════════════════════════════════════════════════════════

METRICS = [("MBE", "MBE (ERA5 − POWER)"), ("RMSE", "RMSE"), ("pearson_r", "Pearson r")]
EVENT_COLORS = {"sunrise": "#f4a261", "noon": "#e76f51", "sunset": "#6d597a"}


def build_faceted_chart(agreement_df):
    sub = agreement_df[(agreement_df["season"] != "ALL") & (agreement_df["event"] != "ALL")].copy()
    present_vars = [v for v in VARIABLE_ORDER if v in sub["variable"].unique()]
    if not present_vars:
        log("[SKIP] faceted MBE/RMSE/r chart — none of the expected variables "
            f"{VARIABLE_ORDER} found in {AGREEMENT_FILE.name}.")
        return None

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)][:len(present_vars)]
    n_rows = 2 if len(present_vars) > 2 else 1
    n_cols = 2 if len(present_vars) > 1 else 1
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=present_vars,
                         horizontal_spacing=0.1, vertical_spacing=0.18)

    trace_meta = []  # per-trace (variable, event, season-indexed sub-df), same order as add_trace calls
    for var, (r, c) in zip(present_vars, positions):
        vsub = sub[sub["variable"] == var]
        for event in EVENT_ORDER:
            esub = vsub[vsub["event"] == event].set_index("season").reindex(SEASON_ORDER)
            fig.add_trace(go.Bar(
                x=SEASON_ORDER, y=esub["MBE"].tolist(),
                name=event, legendgroup=event, showlegend=(var == present_vars[0]),
                marker_color=EVENT_COLORS.get(event),
                customdata=np.stack([
                    esub["RMSE"].to_numpy(dtype=float),
                    esub["pearson_r"].to_numpy(dtype=float),
                    esub["n"].fillna(0).to_numpy(dtype=float),
                ], axis=-1),
                hovertemplate=(f"<b>{var}</b> — %{{x}}, {event}<br>MBE=%{{y:.3f}}<br>"
                                "RMSE=%{customdata[0]:.3f}<br>r=%{customdata[1]:.3f}<br>"
                                "n=%{customdata[2]:,.0f}<extra></extra>"),
            ), row=r, col=c)
            trace_meta.append(esub)
        fig.update_xaxes(title_text="Season", row=r, col=c)
        fig.update_yaxes(title_text=f"MBE ({VARIABLE_UNITS.get(var, '')})", row=r, col=c)

    zero_shapes = []
    for i in range(len(present_vars)):
        suffix = "" if i == 0 else str(i + 1)
        zero_shapes.append(dict(type="line", xref=f"x{suffix} domain", x0=0, x1=1,
                                 yref=f"y{suffix}", y0=0, y1=0,
                                 line=dict(color="black", width=1, dash="dot")))

    buttons = []
    for metric_key, metric_label in METRICS:
        y_arrays = [esub[metric_key].tolist() for esub in trace_meta]
        axis_titles = [{f"yaxis{'' if i == 0 else i + 1}.title.text": f"{metric_label}"}
                        for i in range(len(present_vars))]
        layout_update = {"shapes": zero_shapes if metric_key == "MBE" else [],
                          "title": f"ERA5 vs NASA POWER — {metric_label} by season × event"}
        for d in axis_titles:
            layout_update.update(d)
        buttons.append(dict(label=metric_label, method="update",
                             args=[{"y": y_arrays}, layout_update]))

    fig.update_layout(
        title="ERA5 vs NASA POWER — MBE (ERA5 − POWER) by season × event",
        barmode="group",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.15,
                           xanchor="left", yanchor="top", showactive=True)],
        shapes=zero_shapes,
        height=650 if n_rows == 1 else 780,
        legend=dict(title="Event"),
        margin=dict(t=130),
    )
    log(f"Built faceted MBE/RMSE/r chart for variables: {present_vars}.")
    return fig


# ═══════════════════════════════════════════════════════════
# 5. ANNOTATION  (parsed from bias_decision_rajasthan.txt — annotation only)
# ═══════════════════════════════════════════════════════════

def build_annotation_html():
    if not BIAS_DECISION_FILE.exists():
        log(f"[WARN] {BIAS_DECISION_FILE} not found — temporal plot will omit the branch-decision annotation.")
        return "<i>bias_decision_rajasthan.txt not found — branch-decision annotation unavailable.</i>"

    try:
        text = BIAS_DECISION_FILE.read_text(encoding="utf-8")
    except Exception as exc:
        log(f"[WARN] could not read {BIAS_DECISION_FILE}: {exc}")
        return f"<i>Could not read {BIAS_DECISION_FILE.name}: {exc}</i>"

    branch_m = re.search(r"DECISION:\s*(\w+)", text)
    branch = branch_m.group(1) if branch_m else "UNKNOWN"

    n_m = re.search(r"n\s*=\s*([\d,]+)", text)
    n_noon = n_m.group(1) if n_m else "n/a"

    mbe_m = re.search(r"Mean Bias Error.*?:\s*([-\d.]+)\s*W", text)
    mbe_noon = mbe_m.group(1) if mbe_m else "n/a"

    r_m = re.search(r"Pearson r:\s*([\d.]+)", text)
    r_noon = r_m.group(1) if r_m else "n/a"

    season_rows = re.findall(r"^\s*(Winter|Summer|Monsoon|Retreat)\s+(\d+)\s", text, re.MULTILINE)
    season_html = ""
    if season_rows:
        parts = ", ".join(f"{s}: n={int(n):,}" for s, n in season_rows)
        season_html = f"<br><b>Per-season n:</b> {parts}"

    return (
        f"<b>Branch decision:</b> {branch} &nbsp;|&nbsp; "
        f"<b>GHI noon, all seasons:</b> n={n_noon}, MBE={mbe_noon} W/m², r={r_noon}"
        f"{season_html}"
    )


# ═══════════════════════════════════════════════════════════
# 6. ASSEMBLE temporal_coverage_plot.html
# ═══════════════════════════════════════════════════════════

def build_temporal_html(heatmap_fig, faceted_fig, annotation_html):
    heatmap_div = heatmap_fig.to_html(full_html=False, include_plotlyjs="inline", div_id="heatmap-div")
    faceted_div = (faceted_fig.to_html(full_html=False, include_plotlyjs=False, div_id="faceted-div")
                   if faceted_fig is not None
                   else "<p><i>Faceted MBE/RMSE/r chart skipped — see console log.</i></p>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ERA5 vs NASA POWER — Temporal Coverage (Rajasthan)</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 18px; }}
  .annotation {{ background: #eef3fb; border: 1px solid #b9cbe8; border-radius: 6px;
                 padding: 10px 14px; margin-bottom: 26px; font-size: 13.5px; color: #1a3a5c; }}
  .section {{ margin-bottom: 40px; }}
</style>
</head>
<body>
  <h1>ERA5 vs NASA POWER — Temporal Coverage &amp; Agreement (Rajasthan)</h1>
  <div class="subtitle">climate_rajasthan_points.csv vs era5_power_agreement_rajasthan.csv — self-contained, no server required</div>
  <div class="annotation">{annotation_html}</div>
  <div class="section">{heatmap_div}</div>
  <div class="section">{faceted_div}</div>
</body>
</html>"""

    TEMPORAL_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEMPORAL_PLOT_FILE.write_text(html, encoding="utf-8")
    log(f"Saved: {TEMPORAL_PLOT_FILE}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  ERA5 vs NASA POWER — COVERAGE & AGREEMENT VISUALIZATIONS (Rajasthan)")
    print("=" * 68)

    print("\n[1/5] Loading population grid + agreement table ...")
    pop_df = load_population_grid()
    agreement_df = load_agreement()

    print("\n[2/5] Per-point aggregation from climate_rajasthan_points.csv ...")
    stats_df = compute_point_stats()

    print("\n[3/5] Spatial coverage map (folium) ...")
    build_spatial_map(stats_df, pop_df)

    print("\n[4/5] Temporal coverage — heatmap + faceted chart (plotly) ...")
    cell_metrics = compute_temporal_cell_metrics()
    heatmap_fig = build_temporal_heatmap_figure(cell_metrics)
    faceted_fig = build_faceted_chart(agreement_df)
    annotation_html = build_annotation_html()

    print("\n[5/5] Assembling temporal_coverage_plot.html ...")
    build_temporal_html(heatmap_fig, faceted_fig, annotation_html)

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
