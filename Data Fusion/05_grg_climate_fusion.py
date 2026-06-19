"""
Climate-PCM Data Fusion with Grey Relational Analysis (GRG)
============================================================
Bridges Tamil Nadu ERA5 climate data with Rubitherm/PLUSS PCM catalog.

References: Chen 2025 (GRA), Singh 2025 (criteria), Kou 2025 (climate),
            datafusion.txt (project spec)

Usage:
    python 05_grg_climate_fusion.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from grg_utils import (
    aggregate_district_month_climate,
    prepare_pcm_table,
    rank_pcms_for_district_month,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIMATE_CSV = os.path.join(
    BASE_DIR, "..", "era5-tamilnadu-pipeline", "data", "processed", "climate_tamilnadu_all.csv"
)
PCM_CSV = os.path.join(BASE_DIR, "..", "PCM_data", "pcm_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIMATE_COLS = ["timestamp", "T_amb", "GHI", "district", "month", "city"]


def load_climate(path: str) -> pd.DataFrame:
    print(f"[LOAD] Climate: {path}")
    df = pd.read_csv(path, usecols=CLIMATE_COLS)
    print(f"       {len(df):,} hourly rows, {df['district'].nunique()} districts")
    return df


def load_pcm(path: str) -> pd.DataFrame:
    print(f"[LOAD] PCM: {path}")
    df = prepare_pcm_table(pd.read_csv(path))
    print(f"       {len(df)} PCM candidates")
    return df


def run_fusion(climate_df: pd.DataFrame, pcm_df: pd.DataFrame):
    print("[STEP 1] Computing district x month T_peak...")
    climate_stats = aggregate_district_month_climate(climate_df)
    print(f"         {len(climate_stats)} district-month combinations")

    monthly_rows = []
    ranking_rows = []

    print("[STEP 2-3] Climate filter + GRG ranking per district-month...")
    for _, row in climate_stats.iterrows():
        ranked, best = rank_pcms_for_district_month(
            pcm_df,
            row["T_peak_min"],
            row["T_peak_max"],
            row["T_peak_mean"],
        )

        if not ranked.empty:
            for _, pcm_row in ranked.iterrows():
                ranking_rows.append({
                    "district": row["district"],
                    "month": int(row["month"]),
                    "PCM_name": pcm_row["PCM_name"],
                    "GRG_score": round(pcm_row["GRG_score"], 6),
                    "rank": int(pcm_row["rank"]),
                    "T_melt": pcm_row["T_melt"],
                })

        if best.empty:
            monthly_rows.append({
                "district": row["district"],
                "month": int(row["month"]),
                "best_PCM": None,
                "GRG_score": None,
                "T_peak_mean": round(row["T_peak_mean"], 3),
                "GHI_mean": round(row["GHI_mean"], 3),
                "T_peak_min": round(row["T_peak_min"], 3),
                "T_peak_max": round(row["T_peak_max"], 3),
                "n_candidates": 0,
            })
        else:
            b = best.iloc[0]
            monthly_rows.append({
                "district": row["district"],
                "month": int(row["month"]),
                "best_PCM": b["PCM_name"],
                "GRG_score": round(b["GRG_score"], 6),
                "T_peak_mean": round(row["T_peak_mean"], 3),
                "GHI_mean": round(row["GHI_mean"], 3),
                "T_peak_min": round(row["T_peak_min"], 3),
                "T_peak_max": round(row["T_peak_max"], 3),
                "n_candidates": len(ranked),
            })

    monthly_df = pd.DataFrame(monthly_rows)
    rankings_df = pd.DataFrame(ranking_rows)

    print("[STEP 4] Building annual top-3 summary...")
    if rankings_df.empty:
        annual_df = pd.DataFrame(columns=["district", "rank", "PCM_name", "mean_GRG", "months_eligible"])
    else:
        annual = (
            rankings_df.groupby(["district", "PCM_name"], as_index=False)
            .agg(mean_GRG=("GRG_score", "mean"), months_eligible=("month", "nunique"))
        )
        annual["rank"] = annual.groupby("district")["mean_GRG"].rank(ascending=False, method="first").astype(int)
        annual_df = (
            annual[annual["rank"] <= 3]
            .sort_values(["district", "rank"])
            .reset_index(drop=True)
        )
        annual_df["mean_GRG"] = annual_df["mean_GRG"].round(6)

    return monthly_df, rankings_df, annual_df


def print_top3_summary(annual_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TOP-3 PCMs PER DISTRICT (annual mean GRG)")
    print("=" * 70)
    for district in sorted(annual_df["district"].unique()):
        subset = annual_df[annual_df["district"] == district]
        print(f"\n{district}:")
        for _, r in subset.iterrows():
            print(f"  #{int(r['rank'])}  {r['PCM_name']:<16}  GRG={r['mean_GRG']:.4f}  ({int(r['months_eligible'])} months)")


def main():
    if not os.path.exists(CLIMATE_CSV):
        print(f"ERROR: Climate file not found: {CLIMATE_CSV}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(PCM_CSV):
        print(f"ERROR: PCM file not found: {PCM_CSV}", file=sys.stderr)
        sys.exit(1)

    climate_df = load_climate(CLIMATE_CSV)
    pcm_df = load_pcm(PCM_CSV)

    monthly_df, rankings_df, annual_df = run_fusion(climate_df, pcm_df)

    monthly_path = os.path.join(OUTPUT_DIR, "district_pcm_monthly.csv")
    rankings_path = os.path.join(OUTPUT_DIR, "district_pcm_grg_rankings.csv")
    annual_path = os.path.join(OUTPUT_DIR, "district_pcm_top3_annual.csv")

    monthly_df.to_csv(monthly_path, index=False)
    rankings_df.to_csv(rankings_path, index=False)
    annual_df.to_csv(annual_path, index=False)

    print(f"\n[OUTPUT] {monthly_path}  ({len(monthly_df)} rows)")
    print(f"[OUTPUT] {rankings_path}  ({len(rankings_df)} rows)")
    print(f"[OUTPUT] {annual_path}  ({len(annual_df)} rows)")

    matched = monthly_df["best_PCM"].notna().sum()
    print(f"\n[SUMMARY] {matched}/{len(monthly_df)} district-months have a best PCM")
    print(f"[SUMMARY] GRG range: {monthly_df['GRG_score'].min():.4f} - {monthly_df['GRG_score'].max():.4f}")

    chennai_apr = monthly_df[(monthly_df["district"] == "Chennai") & (monthly_df["month"] == 4)]
    if not chennai_apr.empty:
        print(f"[CHECK] Chennai April: best_PCM={chennai_apr.iloc[0]['best_PCM']}, T_peak_mean={chennai_apr.iloc[0]['T_peak_mean']}")

    print_top3_summary(annual_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
