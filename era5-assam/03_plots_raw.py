"""
03_plots_raw.py
=================
RAW DATA QA — run this BEFORE any cleaning (04_preprocess_assam.py),
directly on 02_combine_assam.py's output. Purpose: catch pipeline
problems (timezone bugs, unit errors, source disagreement) while they're
still cheap to fix — before they get baked into downstream columns.

This is adapted from the Tamil Nadu reference pipeline (until phase 4/
03_plots_raw.py) for the Assam population-weighted points schema:
117 population-weighted points x 3 events/day (sunrise/noon/sunset)
x 10 years, with both ERA5 and NASA POWER columns per row.

WHAT EACH PLOT CHECKS
-----------------------------------------------------------------------
  A. Point map              — sampling design sanity: are the 117 points
                               actually covering Assam, weighted by
                               population as expected?
  B. Event profile          — timezone sanity check: GHI/T_amb must
                               peak at the "noon" event, not sunrise/sunset.
                               If noon isn't the peak, something's shifted.
  C. ERA5 vs NASA POWER     — source agreement: quantifies exactly how
     scatter + MBE/RMSE       much the two sources disagree, per variable,
                               before you decide how (or whether) to
                               bias-correct in preprocessing.
  D. Missing-data heatmap   — which points/columns have real data gaps,
                               as opposed to physically-expected zeros.
  E. Seasonal boxplots      — GHI/T_amb by season, sanity check against
                               known Assam climatology:
                               - Heavy monsoon Jun-Sep (highest India rainfall)
                               - GHI strongly suppressed Jun-Sep
                               - Cool dry winter Dec-Feb
                               - Pre-monsoon Mar-May relatively clear
  F. Multi-year trend       — spot obvious jumps/discontinuities across
                               2016-2025 that would suggest a download or
                               unit problem in a specific year.

HOW TO RUN:
  python 03_plots_raw.py

OUTPUT: data/plots/raw/*.png  (all read-only diagnostics — nothing here
writes back to climate_assam_points.csv)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import COMBINED_POINTS_FILE, PLOTS_DIR

RAW_PLOT_DIR = PLOTS_DIR / "raw"
RAW_PLOT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ORDER  = ["sunrise", "noon", "sunset"]
# Assam season order — matches SEASON_MAP in 02_combine_assam.py
SEASON_ORDER = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
EVENT_COLORS  = {"sunrise": "#f9c74f", "noon": "#f3722c", "sunset": "#577590"}
SEASON_COLORS = {
    "Winter":       "#4cc9f0",
    "Pre-Monsoon":  "#f9c74f",
    "Monsoon":      "#06d6a0",
    "Post-Monsoon": "#f3722c",
}

print("=" * 68)
print("  Raw Data QA Plots — Assam Population-Weighted Points")
print(f"  Input  : {COMBINED_POINTS_FILE}")
print(f"  Output : {RAW_PLOT_DIR}/")
print("=" * 68)

if not COMBINED_POINTS_FILE.exists():
    print(f"\n  ERROR: {COMBINED_POINTS_FILE} not found.")
    print("  Run 02_combine_assam.py first.")
    raise SystemExit(1)

print("\nLoading data (may take a minute for a large CSV) ...")
df = pd.read_csv(COMBINED_POINTS_FILE, parse_dates=["date"])
df["event"]  = pd.Categorical(df["event"],  categories=EVENT_ORDER,  ordered=True)
df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}  |  "
      f"Years: {df['year'].min()}-{df['year'].max()}")


# ===========================================================
# A. POINT MAP
# ===========================================================
print("\n[A] Point map ...")

point_meta = df.groupby("point_id").agg(
    lat=("lat", "first"), lon=("lon", "first"),
    population=("population", "first"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(
    point_meta["lon"], point_meta["lat"],
    s=20 + 60 * point_meta["population"] / point_meta["population"].max(),
    c=point_meta["population"], cmap="viridis", alpha=0.85,
    edgecolors="white", linewidths=0.5,
)
plt.colorbar(sc, ax=ax, label="Population (2020, WorldPop)")
ax.set_title(f"Assam — {len(point_meta)} Population-Weighted Sampling Points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "A_point_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: A_point_map.png  ({len(point_meta)} points, "
      f"total population covered = {point_meta['population'].sum():,.0f})")


# ===========================================================
# B. EVENT PROFILE — timezone / sun-event sanity check
# ===========================================================
print("\n[B] Event profile (sunrise/noon/sunset) ...")

era5_ghi_col   = "era5_GHI"   if "era5_GHI"   in df.columns else None
era5_tamb_col  = "era5_T_amb" if "era5_T_amb" in df.columns else None
power_ghi_col  = "power_ALLSKY_SFC_SW_DWN" if "power_ALLSKY_SFC_SW_DWN" in df.columns else None
power_t_col    = "power_T2M"  if "power_T2M"  in df.columns else None

profile_cols = [c for c in [era5_ghi_col, era5_tamb_col, power_ghi_col, power_t_col]
                if c is not None]
event_means = df.groupby("event", observed=True)[profile_cols].mean()
print(event_means.to_string())

if era5_ghi_col:
    peak_event_ghi = event_means[era5_ghi_col].idxmax()
    print(f"  Peak ERA5 GHI at event   : {peak_event_ghi}  "
          f"({'OK — noon peaks as expected' if peak_event_ghi == 'noon' else '*** CHECK — expected noon ***'})")
if era5_tamb_col:
    peak_event_t = event_means[era5_tamb_col].idxmax()
    print(f"  Peak ERA5 T_amb at event : {peak_event_t}  "
          f"(noon/afternoon peak expected)")

plot_pairs = [(c, EVENT_COLORS) for c in [era5_ghi_col, era5_tamb_col] if c]
ylabels    = ["GHI (W/m²)", "T_amb (°C)"]
fig, axes = plt.subplots(1, len(plot_pairs), figsize=(6 * len(plot_pairs), 5))
if len(plot_pairs) == 1:
    axes = [axes]
for ax, (col, _), ylabel in zip(axes, plot_pairs, ylabels):
    means = df.groupby("event", observed=True)[col].mean()
    stds  = df.groupby("event", observed=True)[col].std()
    ax.bar(means.index.astype(str), means.values,
           yerr=stds.values, capsize=5,
           color=[EVENT_COLORS[e] for e in means.index.astype(str)], alpha=0.85)
    ax.set_title(f"Mean {ylabel} by Sun Event  (± 1 std)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Assam — Event Profile (should peak at noon)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "B_event_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: B_event_profile.png")


# ===========================================================
# C. ERA5 vs NASA POWER — cross-source agreement
# ===========================================================
print("\n[C] ERA5 vs NASA POWER agreement ...")

compare_pairs = [
    ("era5_GHI",           "power_ALLSKY_SFC_SW_DWN",  "GHI (W/m²)"),
    ("era5_GHI_clearsky",  "power_CLRSKY_SFC_SW_DWN",  "Clear-sky GHI (W/m²)"),
    ("era5_T_amb",         "power_T2M",                 "T_amb (°C)"),
    ("era5_RHum",          "power_RH2M",                "RHum (%)"),
    ("era5_W_spd",         "power_WS10M",               "Wind speed (m/s)"),
]
compare_pairs = [(a, b, lbl) for a, b, lbl in compare_pairs
                 if a in df.columns and b in df.columns]

mbe_rmse_rows = []
n_pairs = len(compare_pairs)
ncols = 3
nrows = (n_pairs + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
axes = axes.flatten()

for ax, (era5_col, power_col, label) in zip(axes, compare_pairs):
    sub = df[[era5_col, power_col]].dropna()
    if sub.empty:
        ax.set_title(f"{label}: no overlapping data")
        continue
    sample = sub.sample(min(30_000, len(sub)), random_state=42)

    diff = sub[era5_col] - sub[power_col]
    mbe  = diff.mean()
    rmse = np.sqrt((diff ** 2).mean())
    corr = sub[era5_col].corr(sub[power_col])
    mbe_rmse_rows.append({
        "variable": label, "n": len(sub),
        "MBE_era5_minus_power": round(mbe, 3),
        "RMSE": round(rmse, 3), "pearson_r": round(corr, 4),
    })

    ax.scatter(sample[power_col], sample[era5_col], s=3, alpha=0.15, color="#4c72b0")
    lims = [min(sample[power_col].min(), sample[era5_col].min()),
            max(sample[power_col].max(), sample[era5_col].max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("NASA POWER")
    ax.set_ylabel("ERA5")
    ax.set_title(f"{label}\nMBE={mbe:.2f}  RMSE={rmse:.2f}  r={corr:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

for ax in axes[n_pairs:]:
    ax.axis("off")
plt.suptitle("Assam — ERA5 vs NASA POWER Cross-Check", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "C_era5_vs_power.png", dpi=150, bbox_inches="tight")
plt.close()

mbe_df = pd.DataFrame(mbe_rmse_rows)
mbe_df.to_csv(RAW_PLOT_DIR / "C_era5_vs_power_stats.csv", index=False)
print("  Saved: C_era5_vs_power.png, C_era5_vs_power_stats.csv")
if not mbe_df.empty:
    print(mbe_df.to_string(index=False))
print("  -> GHI MBE > 20 W/m² means a unit or deaccumulation problem — fix before 04.")


# ===========================================================
# D. MISSING DATA HEATMAP
# ===========================================================
print("\n[D] Missing data heatmap ...")

check_cols = [
    "era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
    "era5_precipitation", "power_ALLSKY_SFC_SW_DWN", "power_T2M",
]
check_cols = [c for c in check_cols if c in df.columns]

miss_by_point = (
    df.groupby("point_id")[check_cols]
    .apply(lambda g: g.isna().mean() * 100)
)

fig, ax = plt.subplots(figsize=(9, max(8, len(miss_by_point) * 0.15)))
sns.heatmap(miss_by_point, cmap="Reds",
            vmin=0, vmax=max(5, miss_by_point.values.max()),
            cbar_kws={"label": "% missing"}, ax=ax,
            xticklabels=True, yticklabels=False)
ax.set_title("% Missing Data — per Point × Variable (Assam, 117 points)")
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "D_missing_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()

overall_missing = df[check_cols].isna().mean() * 100
print("  Overall % missing per column:")
print(overall_missing.round(2).to_string())
print("  Saved: D_missing_heatmap.png")


# ===========================================================
# E. SEASONAL BOXPLOTS
# ===========================================================
print("\n[E] Seasonal boxplots ...")

# Use only noon rows for cleaner signal
noon = df[df["event"] == "noon"].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if "era5_GHI" in noon.columns:
    sns.boxplot(data=noon, x="season", y="era5_GHI",
                order=SEASON_ORDER, palette=SEASON_COLORS,
                ax=axes[0], showfliers=False)
    axes[0].set_title("Noon GHI by Season\n(Assam — all points, all years)")
    axes[0].set_ylabel("GHI (W/m²)")
    axes[0].set_xlabel("")
    # Expected: Monsoon should be visibly suppressed (Assam gets very heavy rain Jun-Sep)
    axes[0].annotate("Monsoon suppression expected here",
                     xy=(SEASON_ORDER.index("Monsoon"), noon["era5_GHI"].quantile(0.5)),
                     xytext=(SEASON_ORDER.index("Monsoon") + 0.4,
                             noon["era5_GHI"].quantile(0.7)),
                     fontsize=7, color="gray",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

if "era5_T_amb" in noon.columns:
    sns.boxplot(data=noon, x="season", y="era5_T_amb",
                order=SEASON_ORDER, palette=SEASON_COLORS,
                ax=axes[1], showfliers=False)
    axes[1].set_title("Noon T_amb by Season\n(Assam — all points, all years)")
    axes[1].set_ylabel("T_amb (°C)")
    axes[1].set_xlabel("")

plt.suptitle("Assam Climate Seasonality Check\n"
             "Expected: Low GHI in Monsoon, low T in Winter", fontsize=12)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "E_seasonal_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: E_seasonal_boxplots.png")
print("  Expected pattern: Monsoon (Jun-Sep) GHI lowest; Winter (Dec-Feb) T lowest")


# ===========================================================
# F. MULTI-YEAR TREND (discontinuity check)
# ===========================================================
print("\n[F] Multi-year trend (discontinuity check) ...")

yearly_cols = [c for c in ["era5_GHI", "era5_T_amb"] if c in noon.columns]
yearly = noon.groupby("year")[yearly_cols].mean()

n_cols = len(yearly_cols)
fig, axes = plt.subplots(n_cols, 1, figsize=(11, 4 * n_cols), sharex=True)
if n_cols == 1:
    axes = [axes]

colors = ["#f3722c", "#4cc9f0"]
ylabels_f = ["Mean noon GHI (W/m²)", "Mean noon T_amb (°C)"]
for ax, col, color, ylabel in zip(axes, yearly_cols, colors, ylabels_f):
    ax.plot(yearly.index, yearly[col], marker="o", color=color)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

axes[0].set_title("Assam — Year-by-Year Mean (noon event, all points)\n"
                  "Should be gently varying — a step-change means a download or unit problem")
axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "F_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: F_yearly_trend.png")
print(yearly.round(2).to_string())


print("\n" + "=" * 68)
print("  DONE — inspect the PNGs in", RAW_PLOT_DIR)
print("  STOP CRITERIA (fix before running 04_preprocess_assam.py):")
print("    B: noon must be peak GHI event — if not, timezone bug")
print("    C: ERA5 vs POWER GHI MBE should be < 20 W/m²")
print("    F: no step-change in a single year")
print("=" * 68)
