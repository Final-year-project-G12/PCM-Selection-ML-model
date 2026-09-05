"""
Phase 1 (Data Collection) plots - Rajasthan
Output: PLOTSV2/phase1_data_collection/

Port of era5-uttarakhand/03_plots_raw.py and tamilnadu_pipeline/03_plots_raw.py
- same six raw-QA checks, same matplotlib style, same filenames.

Rajasthan's own 03c_plots_raw_rajasthan.py covers the same ground but writes
interactive Plotly HTML only (that pipeline's convention), so there was no PNG
to put in the curated Plots folder next to Tamil Nadu's and Uttarakhand's.
This script produces the static PNGs.

Reads climate_rajasthan_points.csv with usecols - the full file is ~1.4 GB.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
POINTS = os.path.join(BASE, "data", "processed", "climate_rajasthan_points.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_data_collection")
os.makedirs(OUT, exist_ok=True)

EVENT_ORDER = ["sunrise", "noon", "sunset"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Retreat"]
EVENT_COLORS = {"sunrise": "#f9c74f", "noon": "#f3722c", "sunset": "#577590"}
SEASON_COLORS = {"Winter": "#4cc9f0", "Summer": "#f9c74f",
                 "Monsoon": "#06d6a0", "Retreat": "#f3722c"}

COMPARE_PAIRS = [
    ("era5_GHI", "power_ALLSKY_SFC_SW_DWN", "GHI (W/m2)"),
    ("era5_GHI_clearsky", "power_CLRSKY_SFC_SW_DWN", "Clear-sky GHI (W/m2)"),
    ("era5_T_amb", "power_T2M", "T_amb (degC)"),
    ("era5_RHum", "power_RH2M", "RHum (%)"),
    ("era5_W_spd", "power_WS10M", "Wind speed (m/s)"),
]
QUALITY_VARS = ["era5_T_amb", "era5_GHI", "era5_RHum", "era5_cloud_cover",
                "era5_precipitation", "power_ALLSKY_SFC_SW_DWN", "power_T2M"]

USECOLS = sorted(set(
    ["point_id", "lat", "lon", "population", "date", "event", "season", "year"]
    + [c for pair in COMPARE_PAIRS for c in pair[:2]]
    + QUALITY_VARS
))

print("=" * 68)
print("  Phase 1 Data-Collection QA Plots - Rajasthan Population Points")
print(f"  Input  : {POINTS}")
print(f"  Output : {OUT}")
print("=" * 68)

print("\nLoading data (usecols only - this is a ~1.4GB CSV) ...")
df = pd.read_csv(POINTS, usecols=USECOLS, parse_dates=["date"])
df["event"] = pd.Categorical(df["event"], categories=EVENT_ORDER, ordered=True)
df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
print(f"  Rows: {len(df):,}  |  Points: {df['point_id'].nunique()}  |  "
      f"Years: {df['year'].min()}-{df['year'].max()}")


# ---------------------------------------------------------------- A. POINT MAP
print("\n[A] Point map ...")
point_meta = df.groupby("point_id", observed=True).agg(
    lat=("lat", "first"), lon=("lon", "first"),
    population=("population", "first"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 9))
sc = ax.scatter(point_meta["lon"], point_meta["lat"],
                s=20 + 60 * point_meta["population"] / point_meta["population"].max(),
                c=point_meta["population"], cmap="viridis", alpha=0.85,
                edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="Population (2020, WorldPop)")
ax.set_title(f"Rajasthan - {len(point_meta)} Population-Weighted Sampling Points")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A_point_map.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: A_point_map.png  ({len(point_meta)} points, "
      f"total population covered = {point_meta['population'].sum():,.0f})")


# ------------------------------------------------------------ B. EVENT PROFILE
print("\n[B] Event profile (sunrise/noon/sunset) ...")
event_means = df.groupby("event", observed=True)[
    [c for c in ["era5_GHI", "era5_T_amb", "power_ALLSKY_SFC_SW_DWN", "power_T2M"] if c in df.columns]
].mean()
print(event_means.to_string())

peak_event_ghi = event_means["era5_GHI"].idxmax()
print(f"  Peak ERA5 GHI at event   : {peak_event_ghi}  "
      f"({'OK - noon peaks as expected' if peak_event_ghi == 'noon' else 'CHECK - expected noon'})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes, ["era5_GHI", "era5_T_amb"], ["GHI (W/m2)", "T_amb (degC)"]):
    means = df.groupby("event", observed=True)[col].mean()
    stds = df.groupby("event", observed=True)[col].std()
    ax.bar(means.index.astype(str), means.values,
           yerr=stds.values, capsize=5,
           color=[EVENT_COLORS[e] for e in means.index.astype(str)], alpha=0.85)
    ax.set_title(f"Mean {title} by Sun Event  (+/- 1 std)")
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "B_event_profile.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: B_event_profile.png")


# --------------------------------------------------- C. ERA5 vs NASA POWER
print("\n[C] ERA5 vs NASA POWER agreement ...")
mbe_rmse_rows = []
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, (era5_col, power_col, label) in zip(axes, COMPARE_PAIRS):
    if era5_col not in df.columns or power_col not in df.columns:
        ax.set_title(f"{label}: column missing"); ax.axis("off"); continue
    sub = df[[era5_col, power_col]].dropna()
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
    ax.set_xlabel("NASA POWER"); ax.set_ylabel("ERA5")
    ax.set_title(f"{label}\nMBE={mbe:.2f}  RMSE={rmse:.2f}  r={corr:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
for ax in axes[len(COMPARE_PAIRS):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "C_era5_vs_power.png"), dpi=150, bbox_inches="tight")
plt.close()

mbe_df = pd.DataFrame(mbe_rmse_rows)
mbe_df.to_csv(os.path.join(OUT, "C_era5_vs_power_stats.csv"), index=False)
print("  Saved: C_era5_vs_power.png, C_era5_vs_power_stats.csv")
print(mbe_df.to_string(index=False))
print("  -> Non-zero MBE here is what 04_preprocess_rajasthan.py's quantile-mapping "
      "bias correction addresses.")


# ------------------------------------------------------ D. MISSING DATA HEATMAP
print("\n[D] Missing data heatmap ...")
check_cols = [c for c in QUALITY_VARS if c in df.columns]
miss_by_point = df.groupby("point_id", observed=True)[check_cols].apply(
    lambda g: g.isna().mean() * 100)

fig, ax = plt.subplots(figsize=(9, 18))
sns.heatmap(miss_by_point, cmap="Reds", vmin=0, vmax=max(5, miss_by_point.values.max()),
            cbar_kws={"label": "% missing"}, ax=ax)
ax.set_title("% Missing Data - per Point x Variable (Rajasthan)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "D_missing_heatmap.png"), dpi=130, bbox_inches="tight")
plt.close()

overall_missing = df[check_cols].isna().mean() * 100
print("  Overall % missing per column:")
print(overall_missing.round(2).to_string())
print("  Saved: D_missing_heatmap.png")


# ------------------------------------------------------- E. SEASONAL BOXPLOTS
print("\n[E] Seasonal boxplots ...")
noon = df[df["event"] == "noon"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=noon, x="season", y="era5_GHI", order=SEASON_ORDER,
            palette=SEASON_COLORS, ax=axes[0], showfliers=False)
axes[0].set_title("Noon GHI by Season (all points, all years)")
axes[0].set_ylabel("GHI (W/m2)")
sns.boxplot(data=noon, x="season", y="era5_T_amb", order=SEASON_ORDER,
            palette=SEASON_COLORS, ax=axes[1], showfliers=False)
axes[1].set_title("Noon T_amb by Season (all points, all years)")
axes[1].set_ylabel("T_amb (degC)")
plt.suptitle("Rajasthan - Seasonal Distribution of Noon GHI and Ambient Temperature")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "E_seasonal_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: E_seasonal_boxplots.png")


# ------------------------------------------------------- F. MULTI-YEAR TREND
print("\n[F] Multi-year trend (discontinuity check) ...")
yearly = noon.groupby("year", observed=True)[["era5_GHI", "era5_T_amb"]].mean()
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(yearly.index, yearly["era5_GHI"], marker="o", color="#f3722c")
axes[0].set_ylabel("Mean noon GHI (W/m2)")
axes[0].set_title("Rajasthan - Year-by-Year Mean (noon event, all points)\n"
                  "should be gently varying, not a step-change")
axes[0].grid(alpha=0.3)
axes[1].plot(yearly.index, yearly["era5_T_amb"], marker="o", color="#4cc9f0")
axes[1].set_ylabel("Mean noon T_amb (degC)")
axes[1].set_xlabel("Year")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "F_yearly_trend.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: F_yearly_trend.png")
print(yearly.round(2).to_string())

print("\n" + "=" * 68)
print("  DONE - PNGs in", OUT)
print("=" * 68)
