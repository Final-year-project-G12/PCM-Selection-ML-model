"""
Grey Relational Analysis utilities for climate-adaptive PCM selection.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ZETA = 0.5
GRG_WEIGHTS = {"latent_heat": 0.35, "thermal_conductivity": 0.25, "T_melt_match": 0.25, "specific_heat": 0.15}
CRITERIA_COLS = list(GRG_WEIGHTS.keys())
GHI_COEFF = 0.02
FILTER_TOLERANCE_C = 5.0

def compute_t_proxy(df):
    return df["T_amb"] + GHI_COEFF * df["GHI"].fillna(0.0)

def aggregate_district_month_climate(climate_df):
    df = climate_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["T_proxy"] = compute_t_proxy(df)
    daily_loc = df.groupby(["district","city","month","date"], as_index=False)["T_proxy"].max().rename(columns={"T_proxy":"T_peak_day"})
    daily_dist = daily_loc.groupby(["district","month","date"], as_index=False)["T_peak_day"].mean()
    climate_stats = daily_dist.groupby(["district","month"], as_index=False).agg(T_peak_mean=("T_peak_day","mean"), T_peak_min=("T_peak_day","min"), T_peak_max=("T_peak_day","max"))
    ghi_mean = df.groupby(["district","month"], as_index=False)["GHI"].mean().rename(columns={"GHI":"GHI_mean"})
    return climate_stats.merge(ghi_mean, on=["district","month"])

def prepare_pcm_table(pcm_df):
    return pcm_df.rename(columns={"product":"PCM_name","Tm_melting":"T_melt","latent_heat_melting":"latent_heat","TC_both":"thermal_conductivity","Cp_avg":"specific_heat","density_solid":"density"}).copy()

def filter_pcms_by_climate(pcm_df, t_peak_min, t_peak_max, tolerance=FILTER_TOLERANCE_C):
    return pcm_df.loc[pcm_df["T_melt"].between(t_peak_min - tolerance, t_peak_max + tolerance)].copy()

def compute_grg(pcm_df, t_peak_mean, zeta=ZETA):
    if pcm_df.empty:
        return pcm_df.assign(GRG_score=pd.Series(dtype=float))
    df = pcm_df.copy()
    raw = pd.DataFrame({"latent_heat": df["latent_heat"].astype(float), "thermal_conductivity": df["thermal_conductivity"].astype(float), "T_melt_match": 100.0 - (df["T_melt"].astype(float) - t_peak_mean).abs(), "specific_heat": df["specific_heat"].astype(float)})
    denom = raw.max() - raw.min()
    normalized = raw.copy()
    for col in CRITERIA_COLS:
        normalized[col] = 1.0 if denom[col] == 0 else (raw[col] - raw[col].min()) / denom[col]
    delta = 1.0 - normalized
    delta_min, delta_max = float(delta.min().min()), float(delta.max().max())
    if delta_max == 0:
        for col in CRITERIA_COLS: df[f"xi_{col}"] = 1.0
        df["GRG_score"] = 1.0
        return df
    xi = (delta_min + zeta * delta_max) / (delta + zeta * delta_max)
    for col in CRITERIA_COLS: df[f"xi_{col}"] = xi[col].values
    df["GRG_score"] = sum(df[f"xi_{col}"] * w for col, w in GRG_WEIGHTS.items())
    return df

def rank_pcms_for_district_month(pcm_df, t_peak_min, t_peak_max, t_peak_mean):
    filtered = filter_pcms_by_climate(pcm_df, t_peak_min, t_peak_max)
    if filtered.empty: return filtered, pd.DataFrame()
    scored = compute_grg(filtered, t_peak_mean).sort_values("GRG_score", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored, scored.iloc[[0]]
