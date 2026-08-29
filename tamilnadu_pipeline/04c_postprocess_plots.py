"""
04c_postprocess_plots.py
==========================
POST-PREPROCESSING QA — run this AFTER 04_preprocess_tamilnadu.py, on its
output (tamilnadu_cleaned_physical.csv). 03_plots_raw.py checked the RAW
data before cleaning; this script checks the CLEANED data after cleaning,
so you can see exactly what the 13 preprocessing steps actually did
before trusting them for Phase 3.

WHAT EACH PLOT CHECKS
-----------------------------------------------------------------------
  A. Missing-data heatmap (post)   — should be essentially all-zero; if
                                      not, step 4's imputation didn't
                                      cover something and step 13's hard
                                      gate should already have failed.
  B. Distribution sanity            — histograms of the core physical
                                      columns after outlier-NaN'ing +
                                      imputation. Watch for imputation
                                      spikes (a suspicious mode exactly
                                      at the point/zone/global median).
  C. Hampel + physical-validation   — reads qc_report.txt and plots how
     impact                          many values step 2 (physical bounds)
                                      and step 3 (Hampel/MAD) each
                                      flagged, per column.
  D. Lag-feature sanity             — GHI vs GHI_lag7d scatter, correlation
                                      should be positive and clearly
                                      structured (same sun-event, same
                                      season 7 days apart), not noise.
  E. One-point time series          — raw-looking cleaned noon GHI over
                                      one full year for one point, with
                                      the 7d/30d rolling mean overlaid,
                                      to eyeball that cleaning didn't
                                      flatten or distort the seasonal
                                      shape.
  F. Correlation heatmap (post)     — same as 03's raw version but on the
                                      cleaned+engineered columns, so you
                                      can see what feature engineering
                                      (step 6) added.

HOW TO RUN:
  python 04c_postprocess_plots.py
"""

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import PREPROCESSED_DIR, PLOTS_DIR

POST_PLOT_DIR = PLOTS_DIR / "post_preprocess"
POST_PLOT_DIR.mkdir(parents=True, exist_ok=True)

PHYSICAL_FILE = PREPROCESSED_DIR / "tamilnadu_cleaned_physical.csv"
QC_REPORT = PREPROCESSED_DIR / "qc_report.txt"

EVENT_ORDER = ["sunrise", "noon", "sunset"]

print("=" * 68)
print("  Post-Preprocessing QA Plots — Tamil Nadu")
print(f"  Input  : {PHYSICAL_FILE}")
print(f"  Output : {POST_PLOT_DIR}/")
print("=" * 68)

if not PHYSICAL_FILE.exists():
    raise FileNotFoundError(f"{PHYSICAL_FILE} not found — run 04_preprocess_tamilnadu.py first.")

print("\nLoading cleaned data ...")
df = pd.read_csv(PHYSICAL_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}  |  "
      f"Years: {df['year'].min()}-{df['year'].max()}")

# ═══════════════════════════════════════════════════════════
# A. MISSING DATA HEATMAP (post)
# ═══════════════════════════════════════════════════════════
print("\n[A] Missing data heatmap (post-cleaning, should be ~0) ...")

check_cols = ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
              "era5_precipitation", "era5_CSI", "power_ALLSKY_SFC_SW_DWN", "power_T2M"]
check_cols = [c for c in check_cols if c in df.columns]

miss_by_point = df.groupby("point_id")[check_cols].apply(lambda g: g.isna().mean() * 100)
fig, ax = plt.subplots(figsize=(9, 18))
sns.heatmap(miss_by_point, cmap="Reds", vmin=0, vmax=max(1, miss_by_point.values.max()),
            cbar_kws={"label": "% missing (post-clean)"}, ax=ax)
ax.set_title("% Missing Data AFTER Cleaning — should be ~0 everywhere")
plt.tight_layout()
plt.savefig(POST_PLOT_DIR / "A_missing_post.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  Max residual missing %: {miss_by_point.values.max():.3f}")
print("  Saved: A_missing_post.png")

# ═══════════════════════════════════════════════════════════
# B. DISTRIBUTION SANITY
# ═══════════════════════════════════════════════════════════
print("\n[B] Distribution sanity (post-cleaning) ...")

dist_cols = [c for c in ["era5_GHI", "era5_T_amb", "era5_RHum", "era5_W_spd",
                          "era5_cloud_cover", "era5_precipitation"] if c in df.columns]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, col in zip(axes, dist_cols):
    ax.hist(df[col].dropna(), bins=60, color="#2a9d8f", alpha=0.85)
    ax.set_title(col)
    ax.grid(alpha=0.3)
for ax in axes[len(dist_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(POST_PLOT_DIR / "B_distributions_post.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: B_distributions_post.png  "
      "(watch for a spike exactly at one value -> imputation artifact)")

# ═══════════════════════════════════════════════════════════
# C. QC REPORT SUMMARY (physical validation + Hampel impact)
# ═══════════════════════════════════════════════════════════
print("\n[C] Physical-validation / Hampel impact (from qc_report.txt) ...")

if QC_REPORT.exists():
    text = QC_REPORT.read_text(encoding="utf-8")
    bound_hits = re.findall(r"^\s*(\S+): ([\d,]+) out-of-range values", text, re.MULTILINE)
    hampel_hits = re.findall(r"^\s*(\S+): ([\d,]+) points flagged as Hampel outliers", text, re.MULTILINE)

    rows = []
    for col, n in bound_hits:
        rows.append({"column": col, "n_flagged": int(n.replace(",", "")), "check": "physical_bounds"})
    for col, n in hampel_hits:
        rows.append({"column": col, "n_flagged": int(n.replace(",", "")), "check": "hampel_MAD"})

    if rows:
        qc_df = pd.DataFrame(rows)
        qc_df.to_csv(POST_PLOT_DIR / "C_qc_flag_counts.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = qc_df.pivot_table(index="column", columns="check", values="n_flagged",
                                   aggfunc="sum", fill_value=0)
        pivot.plot(kind="barh", ax=ax, color=["#e76f51", "#264653"])
        ax.set_title("Values flagged NaN by physical-bounds vs. Hampel filter (steps 2 & 3)")
        ax.set_xlabel("count")
        plt.tight_layout()
        plt.savefig(POST_PLOT_DIR / "C_qc_flag_counts.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(qc_df.to_string(index=False))
        print("  Saved: C_qc_flag_counts.csv, C_qc_flag_counts.png")
    else:
        print("  [INFO] No flag counts parsed from qc_report.txt (check its format).")
else:
    print(f"  [SKIP] {QC_REPORT} not found — rerun 04_preprocess_tamilnadu.py to regenerate it.")

# ═══════════════════════════════════════════════════════════
# D. LAG FEATURE SANITY
# ═══════════════════════════════════════════════════════════
print("\n[D] Lag feature sanity (GHI vs GHI_lag7d) ...")

if "era5_GHI_lag7d" in df.columns:
    noon = df[df["event"] == "noon"]
    sub = noon[["era5_GHI", "era5_GHI_lag7d"]].dropna()
    sample = sub.sample(min(30_000, len(sub)), random_state=42)
    r = sub["era5_GHI"].corr(sub["era5_GHI_lag7d"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["era5_GHI_lag7d"], sample["era5_GHI"], s=3, alpha=0.15, color="#4c72b0")
    lims = [0, max(sample.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("GHI, 7 days prior (same event)")
    ax.set_ylabel("GHI, today")
    ax.set_title(f"Noon GHI vs. 7-day-prior lag  (r={r:.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(POST_PLOT_DIR / "D_lag_sanity.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Pearson r(GHI, GHI_lag7d) = {r:.3f}  (positive & well below 1.0 is expected/healthy)")
    print("  Saved: D_lag_sanity.png")
else:
    print("  [SKIP] era5_GHI_lag7d not in file — check step 7 ran.")

# ═══════════════════════════════════════════════════════════
# E. ONE-POINT TIME SERIES WITH ROLLING MEAN
# ═══════════════════════════════════════════════════════════
print("\n[E] One-point time series (cleaned, with rolling mean overlay) ...")

sample_point = df["point_id"].iloc[0]
sample_year = int(df["year"].median())
sub = df[(df["point_id"] == sample_point) & (df["event"] == "noon") &
         (df["year"] == sample_year)].sort_values("date")

if len(sub) > 30 and "era5_GHI_roll7d_mean" in df.columns:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sub["date"], sub["era5_GHI"], color="#888", alpha=0.5, linewidth=0.8, label="Cleaned noon GHI")
    ax.plot(sub["date"], sub["era5_GHI_roll7d_mean"], color="#e76f51", linewidth=1.8, label="7-occurrence rolling mean")
    if "era5_GHI_roll30d_mean" in df.columns:
        ax.plot(sub["date"], sub["era5_GHI_roll30d_mean"], color="#264653", linewidth=1.8, label="30-occurrence rolling mean")
    ax.set_title(f"Cleaned noon GHI, {sample_point}, {sample_year} — seasonal shape should be smooth, not flattened")
    ax.set_ylabel("GHI (W/m^2)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(POST_PLOT_DIR / "E_point_timeseries.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: E_point_timeseries.png  ({sample_point}, {sample_year})")
else:
    print("  [SKIP] Not enough rows or rolling columns missing — check steps 7-9c ran.")

# ═══════════════════════════════════════════════════════════
# F. CORRELATION HEATMAP (POST, INCLUDES ENGINEERED FEATURES)
# ═══════════════════════════════════════════════════════════
print("\n[F] Correlation heatmap (post-cleaning + engineered features) ...")

CORR_COLS = [c for c in ["era5_GHI", "era5_DNI", "era5_DHI", "era5_CSI", "era5_T_amb",
                          "era5_RHum", "era5_W_spd", "era5_cloud_cover",
                          "era5_cloud_opacity", "era5_T_depression",
                          "solar_hour_angle", "era5_GHI_delta1d", "era5_T_amb_delta1d"]
             if c in df.columns]
day_sub = df[df["is_daytime"] == 1] if "is_daytime" in df.columns else df
corr_sample = day_sub[CORR_COLS].sample(min(50_000, len(day_sub)), random_state=42)
corr = corr_sample.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            vmin=-1, vmax=1, annot_kws={"size": 7})
ax.set_title("Post-cleaning correlation, incl. engineered features (daytime rows)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(POST_PLOT_DIR / "F_correlation_post.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: F_correlation_post.png")

print("\n" + "=" * 68)
print("  DONE — inspect the PNGs in", POST_PLOT_DIR)
print("  If A shows non-zero missing, or E looks flattened/distorted,")
print("  go back and check 04_preprocess_tamilnadu.py's qc_report.txt")
print("  BEFORE running 04b_climate_signature.py.")
print("=" * 68)
