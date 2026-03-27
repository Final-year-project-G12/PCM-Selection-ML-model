import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_RAW = os.path.join(BASE_DIR, "../data_2/climate_features.csv")
INPUT_SCALED = os.path.join(BASE_DIR, "../data_2/climate_scaled.csv")
VAL_DIR = os.path.join(BASE_DIR, "../data_2/validation")

PHYSICAL_BOUNDS = {
    "GHI":            (0.0,    1400.0),
    "DNI":            (0.0,    1200.0),
    "DHI":            (0.0,     800.0),
    "avg_sdirswrf":   (0.0,    1200.0),
    "LW_down":        (100.0,   600.0),
    "T_amb":          (-10.0,   55.0),
    "T_dew":          (-20.0,   40.0),
    "RHum":           (0.0,    100.0),
    "W_spd":          (0.0,     30.0),
    "W_dir":          (0.0,    360.0),
    "precipitation":  (0.0,    200.0),
    "cloud_cover":    (0.0,      1.0),
    "P_atm":          (800.0, 1100.0),
    "CSI":            (0.0,     1.5),
    "SZA":            (0.0,    180.0),
    "ETR":            (0.0,   1500.0),
    "GHI_clearsky":   (0.0,   1400.0),
}

def step_D_validate(df_raw: pd.DataFrame, df_scaled: pd.DataFrame):
    print("\n[D] DATA VALIDATION REPORT")
    print("-" * 40)
    
    os.makedirs(VAL_DIR, exist_ok=True)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

    # 1. Missing data summary
    numeric_cols = df_raw.select_dtypes(include=np.number).columns
    missing = df_raw[numeric_cols].isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(3)
    missing_df = pd.DataFrame({"column": missing.index, "missing_pct": missing_pct.values})
    missing_df = missing_df.sort_values("missing_pct", ascending=False)
    print("\n  [D1] Missing data:")
    top_missing = missing_df[missing_df["missing_pct"] > 0]
    print(top_missing.to_string(index=False) if not top_missing.empty else "    ✓ No missing values")

    # 2. Physical range statistics
    print("\n  [D2] Physical range check (raw values, after cleaning):")
    range_rows = []
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df_raw.columns:
            continue
        s = df_raw[col].dropna()
        n_below = (s < lo).sum()
        n_above = (s > hi).sum()
        range_rows.append({
            "column": col,
            "min": round(s.min(), 3),
            "max": round(s.max(), 3),
            "mean": round(s.mean(), 3),
            "std": round(s.std(), 3),
            "expected_min": lo, "expected_max": hi,
            "n_below_min": n_below, "n_above_max": n_above,
            "status": "✓ OK" if n_below == 0 and n_above == 0 else "⚠️  OUTLIERS",
        })
    range_df = pd.DataFrame(range_rows)
    print(range_df[["column","min","max","mean","status"]].to_string(index=False))

    # 3. Correlation matrix
    print("\n  [D3] Generating correlation matrix...")
    key_cols = [c for c in ["GHI", "T_amb", "cloud_cover", "RHum", "CSI", "GHI_clearsky", "SZA", "T_set", "ETR", "RRTDHS", "W_spd", "precipitation", "P_atm"] if c in df_raw.columns]
    corr = df_raw[key_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    plt.tight_layout()
    corr_path = os.path.join(VAL_DIR, "correlation_matrix.png")
    fig.savefig(corr_path, dpi=150)
    plt.close()
    print(f"    Saved → {corr_path}")

    # 4. Distributions
    print("  [D4] Generating distribution plots...")
    dist_cols = [c for c in ["GHI", "T_amb", "RHum", "cloud_cover", "CSI", "P_atm", "W_spd", "precipitation"] if c in df_raw.columns]
    
    # Avoid creating excessive plots if not all columns are present
    if dist_cols:
        ncols = 4
        nrows = (len(dist_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()
        
        for i, col in enumerate(dist_cols):
            ax = axes_flat[i]
            data = df_raw[col].dropna()
            ax.hist(data, bins=50, edgecolor="none", alpha=0.7, color="#4C72B0")
            ax.axvline(data.mean(), color="red", lw=1.5, label=f"mean={data.mean():.1f}")
            ax.axvline(data.median(), color="green", lw=1.5, linestyle="--", label=f"median={data.median():.1f}")
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=7)
            
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
            
        fig.suptitle("Feature Distributions", fontsize=13, y=1.01)
        plt.tight_layout()
        dist_path = os.path.join(VAL_DIR, "distributions.png")
        fig.savefig(dist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {dist_path}")

    # 5. Temporal heatmap
    print("  [D5] Generating temporal heatmap...")
    for city in df_raw["city"].unique():
        sub = df_raw[df_raw["city"] == city].copy()
        sub["date"] = sub["timestamp"].dt.date
        sub["hour"] = sub["timestamp"].dt.hour
        try:
            pivot = sub.pivot_table(values="GHI", index="hour", columns="date", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)//4), 5))
            sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "GHI (W/m²)"},
                        xticklabels=max(1, len(pivot.columns)//20))
            ax.set_title(f"GHI Temporal Coverage — {city}", fontsize=12)
            plt.tight_layout()
            hm_path = os.path.join(VAL_DIR, f"temporal_heatmap_{city.lower()}.png")
            fig.savefig(hm_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"    Saved → {hm_path}")
        except Exception as e:
            print(f"    [skip temporal heatmap for {city}]: {e}")

    # 6. QC Summary Report
    print("  [D6] Writing QC report...")
    qc_summary = []
    for city in df_raw["city"].unique():
        sub = df_raw[df_raw["city"] == city]
        qc_summary.append({
            "city": city,
            "total_rows": len(sub),
            "date_from": str(sub["timestamp"].min()),
            "date_to":   str(sub["timestamp"].max()),
            "expected_hourly_rows": int((sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 3600) + 1,
            "missing_GHI_pct":   round(sub["GHI"].isnull().mean()*100, 3) if "GHI" in sub.columns else 100,
            "missing_T_amb_pct": round(sub["T_amb"].isnull().mean()*100, 3) if "T_amb" in sub.columns else 100,
            "GHI_mean_Wm2":      round(sub["GHI"].mean(), 2) if "GHI" in sub.columns else np.nan,
            "GHI_max_Wm2":       round(sub["GHI"].max(), 2) if "GHI" in sub.columns else np.nan,
            "T_amb_mean_C":      round(sub["T_amb"].mean(), 2) if "T_amb" in sub.columns else np.nan,
            "RHum_mean_pct":     round(sub["RHum"].mean(), 2) if "RHum" in sub.columns else np.nan,
            "RRTDHS_mean":       round(sub["RRTDHS"].mean(), 4) if "RRTDHS" in sub.columns else np.nan,
            "high_solar_resource_pct": round(sub["high_solar_resource"].mean()*100, 1) if "high_solar_resource" in sub.columns else np.nan,
            "spatial_lat":       sub["lat"].iloc[0],
            "spatial_lon":       sub["lon"].iloc[0],
            "temporal_freq":     "1H (hourly)",
        })
        
    qc_df = pd.DataFrame(qc_summary)
    qc_path = os.path.join(VAL_DIR, "qc_report.csv")
    qc_df.to_csv(qc_path, index=False)
    print(f"    Saved → {qc_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_RAW) or not os.path.exists(INPUT_SCALED):
        print(f"Error: Required input files not found. Run 04_normalisation.py first.")
        exit(1)
        
    df_raw = pd.read_csv(INPUT_RAW)
    df_scaled = pd.read_csv(INPUT_SCALED)
    step_D_validate(df_raw, df_scaled)
