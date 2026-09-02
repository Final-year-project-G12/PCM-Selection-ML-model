"""
03_plots_raw.py
=================
RAW DATA QA — run this BEFORE any cleaning (04_preprocess_uttarakhand.py),
directly on 02_combine_uttarakhand.py's output. Purpose: catch pipeline
problems (timezone bugs, unit errors, source disagreement) while they're
still cheap to fix — before they get baked into 60+ engineered/imputed
columns downstream.

This replaces the old uniform-grid/fixed-hour pipeline's plotting script
(which assumed a continuous hourly series across a state-wide 0.25° grid).
This version is built for the new schema: population-weighted points x 3
events/day (sunrise/noon/sunset) x 10 years, with both ERA5 and NASA
POWER columns per row.

WHAT EACH PLOT CHECKS (mapped to Phase 2 acceptance checks in the plan doc)
-----------------------------------------------------------------------
  A. Point map              — sampling design sanity: are the sampling points
                               actually covering Uttarakhand, weighted by
                               population as expected?
  B. Event profile          — Table 9 check #2 (timezone): GHI/T_amb must
                               peak at the "noon" event, not sunrise/sunset.
                               If noon isn't the peak, something's shifted.
  C. ERA5 vs NASA POWER     — Table 9 check #7 (solar bias correction):
     scatter + MBE/RMSE       quantifies exactly how much the two sources
                               disagree, per variable, before you decide
                               how (or whether) to bias-correct in 04.
  D. Missing-data heatmap   — which points/columns have real data gaps,
                               as opposed to physically-expected zeros.
  E. Seasonal boxplots      — GHI/T_amb by season, sanity check against
                               known Uttarakhand climatology (hot dry
                               Apr-Jun, NE monsoon Oct-Dec).
  F. Multi-year trend       — spot obvious jumps/discontinuities across
                               2016-2025 that would suggest a download or
                               unit problem in a specific year.

HOW TO RUN:
  python 03_plots_raw.py

OUTPUT: data/plots/raw/*.png  (all read-only diagnostics — nothing here
writes back to climate_uttarakhand_points.csv)
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

EVENT_ORDER = ["sunrise", "noon", "sunset"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
EVENT_COLORS = {"sunrise": "#f9c74f", "noon": "#f3722c", "sunset": "#577590"}
SEASON_COLORS = {"Winter": "#4cc9f0", "Summer": "#f9c74f",
                  "Monsoon": "#06d6a0", "Retreat": "#f3722c"}

print("=" * 68)
print("  Raw Data QA Plots — Uttarakhand Population Points")
print(f"  Input  : {COMBINED_POINTS_FILE}")
print(f"  Output : {RAW_PLOT_DIR}/")
print("=" * 68)

print("\nLoading data (this is a ~650MB CSV — may take a minute) ...")
df = pd.read_csv(COMBINED_POINTS_FILE, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}  |  "
      f"Years: {df['year'].min()}-{df['year'].max()}")


# ═══════════════════════════════════════════════════════════
# A. POINT MAP
# ═══════════════════════════════════════════════════════════
print("\n[A] Point map ...")

point_meta = df.groupby("point_id").agg(
    lat=("lat", "first"), lon=("lon", "first"),
    population=("population", "first"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 9))
sc = ax.scatter(point_meta["lon"], point_meta["lat"],
                 s=20 + 60 * point_meta["population"] / point_meta["population"].max(),
                 c=point_meta["population"], cmap="viridis", alpha=0.85,
                 edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="Population (2020, WorldPop)")
ax.set_title(f"Uttarakhand — {len(point_meta)} Population-Weighted Sampling Points")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "A_point_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: A_point_map.png  ({len(point_meta)} points, "
      f"total population covered = {point_meta['population'].sum():,.0f})")


# ═══════════════════════════════════════════════════════════
# B. EVENT PROFILE — timezone / sun-event sanity check
# ═══════════════════════════════════════════════════════════
print("\n[B] Event profile (sunrise/noon/sunset) ...")

event_means = df.groupby("event", observed=True)[
    ["era5_GHI", "era5_T_amb", "power_ALLSKY_SFC_SW_DWN", "power_T2M"]
].mean()
print(event_means.to_string())

peak_event_ghi = event_means["era5_GHI"].idxmax()
peak_event_t = event_means["era5_T_amb"].idxmax()
print(f"  Peak ERA5 GHI at event   : {peak_event_ghi}  "
      f"({'OK — noon peaks as expected' if peak_event_ghi == 'noon' else 'CHECK — expected noon'})")
print(f"  Peak ERA5 T_amb at event : {peak_event_t}  "
      f"(afternoon/noon peak expected; sunset slightly warmer than sunrise is normal)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes, ["era5_GHI", "era5_T_amb"], ["GHI (W/m²)", "T_amb (°C)"]):
    means = df.groupby("event", observed=True)[col].mean()
    stds = df.groupby("event", observed=True)[col].std()
    ax.bar(means.index.astype(str), means.values,
           yerr=stds.values, capsize=5,
           color=[EVENT_COLORS[e] for e in means.index.astype(str)], alpha=0.85)
    ax.set_title(f"Mean {title} by Sun Event  (± 1 std)")
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "B_event_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: B_event_profile.png")


# ═══════════════════════════════════════════════════════════
# C. ERA5 vs NASA POWER — cross-source agreement
# ═══════════════════════════════════════════════════════════
print("\n[C] ERA5 vs NASA POWER agreement ...")

compare_pairs = [
    ("era5_GHI", "power_ALLSKY_SFC_SW_DWN", "GHI (W/m²)"),
    ("era5_GHI_clearsky", "power_CLRSKY_SFC_SW_DWN", "Clear-sky GHI (W/m²)"),
    ("era5_T_amb", "power_T2M", "T_amb (°C)"),
    ("era5_RHum", "power_RH2M", "RHum (%)"),
    ("era5_W_spd", "power_WS10M", "Wind speed (m/s)"),
]

mbe_rmse_rows = []
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, (era5_col, power_col, label) in zip(axes, compare_pairs):
    sub = df[[era5_col, power_col, "event"]].dropna()
    if sub.empty:
        ax.set_title(f"{label}: no overlapping data"); continue
    sample = sub.sample(min(30_000, len(sub)), random_state=42)

    diff = sub[era5_col] - sub[power_col]
    mbe = diff.mean()
    rmse = np.sqrt((diff ** 2).mean())
    corr = sub[era5_col].corr(sub[power_col])
    mbe_rmse_rows.append({"variable": label, "n": len(sub),
                           "MBE_era5_minus_power": round(mbe, 3),
                           "RMSE": round(rmse, 3), "pearson_r": round(corr, 4)})

    ax.scatter(sample[power_col], sample[era5_col], s=3, alpha=0.15, color="#4c72b0")
    lims = [min(sample[power_col].min(), sample[era5_col].min()),
            max(sample[power_col].max(), sample[era5_col].max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel(f"NASA POWER"); ax.set_ylabel(f"ERA5")
    ax.set_title(f"{label}\nMBE={mbe:.2f}  RMSE={rmse:.2f}  r={corr:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
for ax in axes[len(compare_pairs):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "C_era5_vs_power.png", dpi=150, bbox_inches="tight")
plt.close()

mbe_df = pd.DataFrame(mbe_rmse_rows)
mbe_df.to_csv(RAW_PLOT_DIR / "C_era5_vs_power_stats.csv", index=False)
print("  Saved: C_era5_vs_power.png, C_era5_vs_power_stats.csv")
print(mbe_df.to_string(index=False))
print("  -> Non-zero MBE here is exactly what 04_preprocess_uttarakhand.py's "
      "bias-correction step (Phase 2, step 7) will address.")


# ═══════════════════════════════════════════════════════════
# D. MISSING DATA HEATMAP
# ═══════════════════════════════════════════════════════════
print("\n[D] Missing data heatmap ...")

check_cols = ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
              "era5_precipitation", "power_ALLSKY_SFC_SW_DWN", "power_T2M"]
check_cols = [c for c in check_cols if c in df.columns]

miss_by_point = df.groupby("point_id")[check_cols].apply(
    lambda g: g.isna().mean() * 100)

fig, ax = plt.subplots(figsize=(9, 18))
sns.heatmap(miss_by_point, cmap="Reds", vmin=0, vmax=max(5, miss_by_point.values.max()),
            cbar_kws={"label": "% missing"}, ax=ax)
ax.set_title("% Missing Data — per Point x Variable")
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "D_missing_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()

overall_missing = df[check_cols].isna().mean() * 100
print("  Overall % missing per column:")
print(overall_missing.round(2).to_string())
print("  Saved: D_missing_heatmap.png")


# ═══════════════════════════════════════════════════════════
# E. SEASONAL BOXPLOTS
# ═══════════════════════════════════════════════════════════
print("\n[E] Seasonal boxplots ...")

noon = df[df["event"] == "noon"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=noon, x="season", y="era5_GHI", order=SEASON_ORDER,
            palette=SEASON_COLORS, ax=axes[0], showfliers=False)
axes[0].set_title("Noon GHI by Season (all points, all years)")
axes[0].set_ylabel("GHI (W/m²)")
sns.boxplot(data=noon, x="season", y="era5_T_amb", order=SEASON_ORDER,
            palette=SEASON_COLORS, ax=axes[1], showfliers=False)
axes[1].set_title("Noon T_amb by Season (all points, all years)")
axes[1].set_ylabel("T_amb (°C)")
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "E_seasonal_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: E_seasonal_boxplots.png")


# ═══════════════════════════════════════════════════════════
# F. MULTI-YEAR TREND (discontinuity check)
# ═══════════════════════════════════════════════════════════
print("\n[F] Multi-year trend (discontinuity check) ...")

yearly = noon.groupby("year")[["era5_GHI", "era5_T_amb"]].mean()
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(yearly.index, yearly["era5_GHI"], marker="o", color="#f3722c")
axes[0].set_ylabel("Mean noon GHI (W/m²)")
axes[0].set_title("Year-by-Year Mean (noon event, all points) — "
                   "should be gently varying, not a step-change")
axes[0].grid(alpha=0.3)
axes[1].plot(yearly.index, yearly["era5_T_amb"], marker="o", color="#4cc9f0")
axes[1].set_ylabel("Mean noon T_amb (°C)")
axes[1].set_xlabel("Year")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RAW_PLOT_DIR / "F_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: F_yearly_trend.png")
print(yearly.round(2).to_string())

print("\n" + "=" * 68)
print("  DONE — inspect the PNGs in", RAW_PLOT_DIR)
print("  If B shows noon ≠ peak, or C shows large MBE, or F shows a step-change")
print("  in one year, resolve that BEFORE running 04_preprocess_uttarakhand.py.")
print("=" * 68)
