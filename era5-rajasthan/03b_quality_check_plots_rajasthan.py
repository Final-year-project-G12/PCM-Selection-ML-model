"""
03b_quality_check_plots_rajasthan.py
=============================================================================
VISUALIZATION for 03b_quality_check_rajasthan.py's output. A separate
script, not folded into 03b_quality_check_rajasthan.py itself, to keep
that script scoped to detection/correction/reporting only (matching its
own "keep this scoped" design) — this one just reads what it already
produced (climate_rajasthan_points.csv, climate_rajasthan_points_clean.csv,
quality_report_rajasthan.json) and plots it. Modeled on the Tamil Nadu
pipeline's 04c_postprocess_plots.py / 04c_interactive_postprocess_qc.py
(same QC-after-cleaning idea), in Plotly since that's this pipeline's
convention.

REQUIRED LIBRARIES (install if missing):
  pip install pandas numpy plotly

READ-ONLY — reads climate_rajasthan_points.csv and climate_rajasthan_
points_clean.csv, writes nothing back to either.

PLOTS PRODUCED:
  A. Missing-data heatmap (post-clean) — % missing per point x variable,
     AFTER imputation. Expect uniformly 0% for current Rajasthan data
     (03b_quality_check_rajasthan.py found 0% missingness to begin with)
     — kept as permanent infrastructure, not just a check for today.
  B. Distribution histograms — raw vs winsorized overlay, per HAMPEL_VAR
     (era5_T_amb, era5_RHum, era5_W_spd — NOT era5_GHI/era5_CSI, which
     03b_quality_check_rajasthan.py deliberately excludes from Hampel
     correction; see that script's docstring THIRD CORRECTION for why).
  C. Outlier flag-count bar chart — per point, how many values were
     flagged/winsorized, per HAMPEL_VAR. The "systematic issue" (>20%)
     threshold is drawn as a reference line.
  D. Sample point annual time series — one point, one year, raw vs
     winsorized overlaid for all 3 HAMPEL_VARs, so a winsorizing
     correction can be checked BY EYE, not just by aggregate stats — this
     is exactly the kind of check that caught the GHI/CSI over-correction
     problem in the first quality-check run.
  E. Correlation heatmap (post-clean) — among the 5 QUALITY_VARS, after
     cleaning. Sanity check only (e.g. GHI/CSI should stay strongly
     correlated; T_amb/RHum should stay negatively correlated).

OUTPUTS: outputs/qc_clean_*.html (5 files)

HOW TO RUN (run 03b_quality_check_rajasthan.py first — this script reads
its output):
  python 03b_quality_check_plots_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    COMBINED_POINTS_FILE, CLEANED_POINTS_FILE, QUALITY_REPORT_JSON_FILE,
    OUTPUTS_DIR, ensure_data_dirs,
)

ensure_data_dirs()

EVENT_ORDER = ["sunrise", "noon", "sunset"]
QUALITY_VARS = ["era5_T_amb", "era5_RHum", "era5_GHI", "era5_CSI", "era5_W_spd"]
HAMPEL_VARS = ["era5_T_amb", "era5_RHum", "era5_W_spd"]   # must match 03b_quality_check_rajasthan.py
SYSTEMATIC_OUTLIER_PCT_FLAG = 20.0   # must match 03b_quality_check_rajasthan.py

print("=" * 68)
print("  QUALITY-CHECK QC PLOTS — Rajasthan")
print(f"  Raw    : {COMBINED_POINTS_FILE}")
print(f"  Clean  : {CLEANED_POINTS_FILE}")
print(f"  Output : {OUTPUTS_DIR}/")
print("=" * 68)

if not CLEANED_POINTS_FILE.exists():
    raise SystemExit(f"ERROR: {CLEANED_POINTS_FILE} not found — run 03b_quality_check_rajasthan.py first.")

flag_cols = [f"{c}_outlier_flag" for c in HAMPEL_VARS]
usecols_raw = ["point_id", "date", "event", "season", "year"] + QUALITY_VARS
usecols_clean = usecols_raw + flag_cols

print("\nLoading raw + clean data (usecols only) ...")
raw = pd.read_csv(COMBINED_POINTS_FILE, usecols=usecols_raw, parse_dates=["date"])
raw["event"] = pd.Categorical(raw["event"], categories=EVENT_ORDER, ordered=True)
clean = pd.read_csv(CLEANED_POINTS_FILE, usecols=usecols_clean, parse_dates=["date"])
clean["event"] = pd.Categorical(clean["event"], categories=EVENT_ORDER, ordered=True)
print(f"  Raw: {len(raw):,} rows   Clean: {len(clean):,} rows")


# ═══════════════════════════════════════════════════════════
# A. MISSING-DATA HEATMAP (POST-CLEAN)
# ═══════════════════════════════════════════════════════════
print("\n[A] Missing-data heatmap (post-clean) ...")

miss_by_point = clean.groupby("point_id", observed=True)[QUALITY_VARS].apply(lambda g: g.isna().mean() * 100)
fig = go.Figure(data=go.Heatmap(
    z=miss_by_point.values, x=QUALITY_VARS, y=miss_by_point.index.tolist(),
    colorscale="Reds", zmin=0, zmax=max(1, float(miss_by_point.values.max())),
    colorbar=dict(title="% missing"),
))
fig.update_layout(title="% Missing Data POST-CLEAN — per Point x Variable "
                         "(expect ~0% — imputation already ran)",
                   height=max(500, 6 * len(miss_by_point)))
out = OUTPUTS_DIR / "qc_clean_missing_heatmap_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════
# B. DISTRIBUTION HISTOGRAMS — raw vs winsorized, HAMPEL_VARS only
# ═══════════════════════════════════════════════════════════
print("\n[B] Distribution histograms (raw vs winsorized, HAMPEL_VARS only) ...")

fig = make_subplots(rows=1, cols=len(HAMPEL_VARS), subplot_titles=HAMPEL_VARS)
for i, col in enumerate(HAMPEL_VARS, start=1):
    fig.add_trace(go.Histogram(x=raw[col], name="raw", opacity=0.55, marker_color="#4c72b0",
                                showlegend=(i == 1)), row=1, col=i)
    fig.add_trace(go.Histogram(x=clean[col], name="winsorized", opacity=0.55, marker_color="#dd8452",
                                showlegend=(i == 1)), row=1, col=i)
fig.update_layout(title="Distribution — Raw vs Winsorized (overlaid, HAMPEL_VARS only)", barmode="overlay")
out = OUTPUTS_DIR / "qc_clean_distributions_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════
# C. OUTLIER FLAG-COUNT BAR CHART
# ═══════════════════════════════════════════════════════════
print("\n[C] Outlier flag-count bar chart (per point, per HAMPEL_VAR) ...")

flag_counts = clean.groupby("point_id", observed=True)[flag_cols].sum()
flag_pct = clean.groupby("point_id", observed=True)[flag_cols].mean() * 100
flag_pct.columns = [c.replace("_outlier_flag", "") for c in flag_pct.columns]

fig = go.Figure()
for col in flag_pct.columns:
    fig.add_trace(go.Bar(x=flag_pct.index, y=flag_pct[col], name=col))
fig.add_hline(y=SYSTEMATIC_OUTLIER_PCT_FLAG, line_dash="dash", line_color="red",
              annotation_text=f"{SYSTEMATIC_OUTLIER_PCT_FLAG}% systematic-issue threshold")
fig.update_layout(title="Outlier Flag % per Point, per HAMPEL_VAR", barmode="group",
                   yaxis_title="% flagged", xaxis=dict(tickangle=90))
out = OUTPUTS_DIR / "qc_clean_outlier_flags_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")
max_flag_pct = flag_pct.values.max()
print(f"  Max single (point, var) flag %: {max_flag_pct:.2f}%  "
      f"({'below' if max_flag_pct < SYSTEMATIC_OUTLIER_PCT_FLAG else 'AT/ABOVE'} the "
      f"{SYSTEMATIC_OUTLIER_PCT_FLAG}% systematic-issue threshold)")


# ═══════════════════════════════════════════════════════════
# D. SAMPLE POINT ANNUAL TIME SERIES — raw vs winsorized, by eye
# ═══════════════════════════════════════════════════════════
print("\n[D] Sample point annual time series (raw vs winsorized) ...")

sample_point = raw["point_id"].iloc[0]
sample_year = int(raw["year"].median()) if "year" in raw.columns else raw["date"].dt.year.median()
raw_sub = raw[(raw["point_id"] == sample_point) & (raw["date"].dt.year == sample_year)
              & (raw["event"] == "noon")].sort_values("date")
clean_sub = clean[(clean["point_id"] == sample_point) & (clean["date"].dt.year == sample_year)
                  & (clean["event"] == "noon")].sort_values("date")

fig = make_subplots(rows=len(HAMPEL_VARS), cols=1, shared_xaxes=True, subplot_titles=HAMPEL_VARS)
for i, col in enumerate(HAMPEL_VARS, start=1):
    fig.add_trace(go.Scatter(x=raw_sub["date"], y=raw_sub[col], mode="lines", name="raw",
                              line=dict(color="#888", width=1), showlegend=(i == 1)), row=i, col=1)
    fig.add_trace(go.Scatter(x=clean_sub["date"], y=clean_sub[col], mode="lines", name="winsorized",
                              line=dict(color="#dd8452", width=1.5), showlegend=(i == 1)), row=i, col=1)
    flagged_dates = clean_sub.loc[clean_sub[f"{col}_outlier_flag"], "date"]
    flagged_vals = raw_sub.set_index("date").loc[flagged_dates, col] if len(flagged_dates) else pd.Series(dtype=float)
    if len(flagged_vals):
        fig.add_trace(go.Scatter(x=flagged_vals.index, y=flagged_vals.values, mode="markers",
                                  marker=dict(color="red", size=6), name="flagged", showlegend=(i == 1)),
                      row=i, col=1)
fig.update_layout(title=f"Sample Point Annual Time Series (noon event) — {sample_point}, {sample_year} — "
                         f"raw vs winsorized, flagged points marked")
out = OUTPUTS_DIR / "qc_clean_sample_timeseries_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}  ({sample_point}, {sample_year})")


# ═══════════════════════════════════════════════════════════
# E. CORRELATION HEATMAP (POST-CLEAN)
# ═══════════════════════════════════════════════════════════
print("\n[E] Correlation heatmap (post-clean, all 5 QUALITY_VARS) ...")

sample = clean[QUALITY_VARS].dropna().sample(min(50_000, len(clean)), random_state=42)
corr = sample.corr()
fig = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1, colorbar=dict(title="Pearson r"),
))
fig.update_layout(title="Correlation, post-clean (all 5 quality variables)")
out = OUTPUTS_DIR / "qc_clean_correlation_rajasthan.html"
fig.write_html(str(out))
print(f"  Saved: {out}")

print("\n" + "=" * 68)
print("  DONE — inspect the HTML files in", OUTPUTS_DIR)
print("=" * 68)
