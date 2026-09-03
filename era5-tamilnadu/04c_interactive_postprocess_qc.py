"""
04c_interactive_postprocess_qc.py
====================================
Interactive (Plotly) version of 04c_postprocess_plots.py — same checks
(A, B, D, E, F; the qc_report.txt bar chart C is trivial enough to leave
as-is in the PNG script), reading tamilnadu_cleaned_physical.csv, as
zoomable/hoverable HTML.

Requires: pip install plotly

INPUT  : data/preprocessed/tamilnadu_cleaned_physical.csv
OUTPUT : data/plots/post_preprocess_interactive/*.html

HOW TO RUN:
  python 04c_interactive_postprocess_qc.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PREPROCESSED_DIR, PLOTS_DIR

OUT_DIR = PLOTS_DIR / "post_preprocess_interactive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHYSICAL_FILE = PREPROCESSED_DIR / "tamilnadu_cleaned_physical.csv"
EVENT_ORDER = ["sunrise", "noon", "sunset"]

print("=" * 68)
print("  Interactive Post-Preprocessing QA — Tamil Nadu")
print(f"  Input  : {PHYSICAL_FILE}")
print(f"  Output : {OUT_DIR}/")
print("=" * 68)

if not PHYSICAL_FILE.exists():
    raise FileNotFoundError(f"{PHYSICAL_FILE} not found — run 04_preprocess_tamilnadu.py first.")

df = pd.read_csv(PHYSICAL_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}")

# ═══════════════════════════════════════════════════════════
# A. MISSING DATA HEATMAP (post)
# ═══════════════════════════════════════════════════════════
print("\n[A] Missing data heatmap (post-clean) ...")

check_cols = ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
              "era5_precipitation", "era5_CSI", "power_ALLSKY_SFC_SW_DWN", "power_T2M"]
check_cols = [c for c in check_cols if c in df.columns]
miss_by_point = df.groupby("point_id")[check_cols].apply(lambda g: g.isna().mean() * 100)
fig = px.imshow(miss_by_point, aspect="auto", color_continuous_scale="Reds",
                 labels=dict(color="% missing"),
                 title="% Missing Data AFTER Cleaning — should be ~0 everywhere")
fig.update_layout(height=1400)
fig.write_html(str(OUT_DIR / "A_missing_post.html"), include_plotlyjs="cdn")
print(f"  Max residual missing %: {miss_by_point.values.max():.3f}")
print("  Saved: A_missing_post.html")

# ═══════════════════════════════════════════════════════════
# B. DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════
print("\n[B] Distribution sanity (post-clean) ...")

dist_cols = [c for c in ["era5_GHI", "era5_T_amb", "era5_RHum", "era5_W_spd",
                          "era5_cloud_cover", "era5_precipitation"] if c in df.columns]
fig = make_subplots(rows=2, cols=3, subplot_titles=dist_cols)
for i, col in enumerate(dist_cols):
    r, c = divmod(i, 3)
    fig.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=60, marker_color="#2a9d8f",
                                showlegend=False), row=r + 1, col=c + 1)
fig.update_layout(title="Distributions post-cleaning (hover for bin counts)", height=650)
fig.write_html(str(OUT_DIR / "B_distributions_post.html"), include_plotlyjs="cdn")
print("  Saved: B_distributions_post.html")

# ═══════════════════════════════════════════════════════════
# D. LAG FEATURE SANITY
# ═══════════════════════════════════════════════════════════
print("\n[D] Lag feature sanity ...")

if "era5_GHI_lag7d" in df.columns:
    noon = df[df["event"] == "noon"]
    sub = noon[["era5_GHI", "era5_GHI_lag7d"]].dropna()
    sample = sub.sample(min(20_000, len(sub)), random_state=42)
    r = sub["era5_GHI"].corr(sub["era5_GHI_lag7d"])
    fig = px.scatter(sample, x="era5_GHI_lag7d", y="era5_GHI", opacity=0.15,
                      title=f"Noon GHI vs. 7-day-prior lag (r={r:.3f})")
    lims = [0, sample.max().max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="black"),
                              showlegend=False))
    fig.write_html(str(OUT_DIR / "D_lag_sanity.html"), include_plotlyjs="cdn")
    print(f"  r(GHI, GHI_lag7d) = {r:.3f}")
    print("  Saved: D_lag_sanity.html")
else:
    print("  [SKIP] era5_GHI_lag7d not in file")

# ═══════════════════════════════════════════════════════════
# E. ONE-POINT TIME SERIES
# ═══════════════════════════════════════════════════════════
print("\n[E] One-point time series with rolling mean ...")

sample_point = df["point_id"].iloc[0]
sample_year = int(df["year"].median())
sub = df[(df["point_id"] == sample_point) & (df["event"] == "noon") &
         (df["year"] == sample_year)].sort_values("date")

if len(sub) > 30 and "era5_GHI_roll7d_mean" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["era5_GHI"], mode="lines",
                              line=dict(color="#888", width=1), opacity=0.6, name="Cleaned noon GHI"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["era5_GHI_roll7d_mean"], mode="lines",
                              line=dict(color="#e76f51", width=2), name="7-occurrence rolling mean"))
    if "era5_GHI_roll30d_mean" in df.columns:
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["era5_GHI_roll30d_mean"], mode="lines",
                                  line=dict(color="#264653", width=2), name="30-occurrence rolling mean"))
    fig.update_layout(title=f"Cleaned noon GHI, {sample_point}, {sample_year}",
                       xaxis_title="date", yaxis_title="GHI (W/m²)", height=500)
    fig.write_html(str(OUT_DIR / "E_point_timeseries.html"), include_plotlyjs="cdn")
    print(f"  Saved: E_point_timeseries.html  ({sample_point}, {sample_year})")
else:
    print("  [SKIP] not enough rows / rolling columns missing")

# ═══════════════════════════════════════════════════════════
# F. CORRELATION HEATMAP (post)
# ═══════════════════════════════════════════════════════════
print("\n[F] Correlation heatmap (post-clean + engineered features) ...")

CORR_COLS = [c for c in ["era5_GHI", "era5_DNI", "era5_DHI", "era5_CSI", "era5_T_amb",
                          "era5_RHum", "era5_W_spd", "era5_cloud_cover",
                          "era5_cloud_opacity", "era5_T_depression",
                          "solar_hour_angle", "era5_GHI_delta1d", "era5_T_amb_delta1d"]
             if c in df.columns]
day_sub = df[df["is_daytime"] == 1] if "is_daytime" in df.columns else df
corr = day_sub[CORR_COLS].sample(min(50_000, len(day_sub)), random_state=42).corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                 title="Post-cleaning correlation, incl. engineered features (daytime rows)")
fig.update_layout(height=700)
fig.write_html(str(OUT_DIR / "F_correlation_post.html"), include_plotlyjs="cdn")
print("  Saved: F_correlation_post.html")

print("\n" + "=" * 68)
print("  DONE — open the .html files in", OUT_DIR)
print("=" * 68)
