"""
run_all_plots.py
=============================================================================
MASTER PLOTTING RUNNER

Orchestrates generation of all 13 main plots + 8 comparison plots for
the Rajasthan Objective-1 PCM selection pipeline audit.

This script:
1. Creates the complete output directory structure
2. Runs each plotting script independently
3. Handles errors gracefully with detailed reporting
4. Generates a summary report of all outputs

Usage:
  python run_all_plots.py

Output structure:
  outputs/objective1_plots_rajasthan/
    ├── 01_raw_vs_preprocessed/
    ├── 02_climate_regime_map/
    ├── 03_feasibility/
    ├── 04_mcdm_agreement/
    ├── 05_montecarlo/
    ├── 06_physics_validation/
    ├── 07_recommendation_summary/
    └── comparison_plots/
        ├── phase2_5_raw_vs_clean/
        ├── phase3_tier1_vs_tier2/
        ├── phase3_tmcap_old_vs_new/
        ├── phase4_levelA_vs_levelB/
        ├── phase5_lrequired_before_after/
        ├── phase6_vikor_bugfix_before_after/
        ├── phase7_pcm_vs_plaintank/
        └── phase8_penalty_k0_vs_k3/
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Configuration
OUTPUT_BASE = "../outputs/objective1_plots_rajasthan"
# SCRIPTS_DIR is current directory when run from within plotting folder
SCRIPTS_DIR = "."

# Define all plots to be generated
MAIN_PLOTS = [
    ("01_raw_vs_preprocessed.py", "Raw vs. Preprocessed Radiation (GHI/T_amb)"),
    ("02_climate_regime_map_copy.py", "Climate Regime Map (copy from Phase 4)"),
    ("03_pcm_feasibility_scatter.py", "PCM Feasibility Scatter (Tm vs. L)"),
    ("04_pcm_survivors_per_cluster.py", "Feasible PCM Candidates per Cluster"),
    ("05_bump_chart.py", "Bump Chart: MCDM Method Agreement"),
    ("06_method_correlation_heatmap.py", "Spearman Correlation Heatmap"),
    ("08_rank_reversal_frequency.py", "Rank-Reversal Frequency (Monte Carlo)"),
    ("09_mcdm_vs_physics_agreement.py", "MCDM vs. Physics Validation"),
    ("11_summary_cards.py", "Summary Cards: Top-1 Recommendations"),
]

COMPARISON_PLOTS = [
    # Not yet implemented
    # ("phase2_5_raw_vs_clean.py", "5-variable raw vs. clean comparison"),
    # ("phase3_tier1_vs_tier2.py", "Diurnal gradient (Tier 1) vs. DTR_true (Tier 2)"),
    # ("phase3_tmcap_old_vs_new.py", "Tm_target_capped (old vs. new)"),
    # ("phase4_levelA_vs_levelB.py", "Cluster assignments Level A vs. B"),
    # ("phase5_lrequired_before_after.py", "L_required survivor count (pre vs. post-correction)"),
    # ("phase6_vikor_bugfix_before_after.py", "VIKOR bugfix before/after"),
    # ("phase7_pcm_vs_plaintank.py", "PCM tank vs. plain tank performance"),
    # ("phase8_penalty_k0_vs_k3.py", "Supercooling penalty k=0.0 vs. k=0.3"),
]

def create_directory_structure():
    """Create all necessary output directories."""
    dirs = [
        OUTPUT_BASE,
        os.path.join(OUTPUT_BASE, "01_raw_vs_preprocessed"),
        os.path.join(OUTPUT_BASE, "02_climate_regime_map"),
        os.path.join(OUTPUT_BASE, "03_feasibility"),
        os.path.join(OUTPUT_BASE, "04_mcdm_agreement"),
        os.path.join(OUTPUT_BASE, "05_montecarlo"),
        os.path.join(OUTPUT_BASE, "06_physics_validation"),
        os.path.join(OUTPUT_BASE, "07_recommendation_summary"),
        os.path.join(OUTPUT_BASE, "comparison_plots"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase2_5_raw_vs_clean"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase3_tier1_vs_tier2"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase3_tmcap_old_vs_new"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase4_levelA_vs_levelB"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase5_lrequired_before_after"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase6_vikor_bugfix_before_after"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase7_pcm_vs_plaintank"),
        os.path.join(OUTPUT_BASE, "comparison_plots", "phase8_penalty_k0_vs_k3"),
    ]

    print("Creating output directory structure...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}")

    print(f"\n✓ All directories created under: {OUTPUT_BASE}/\n")

def run_plot_script(script_name, plot_title):
    """Run a single plotting script and capture output."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        return {
            "script": script_name,
            "title": plot_title,
            "status": "MISSING",
            "error": f"Script not found at {script_path}"
        }

    print(f"Running: {plot_title}")
    print(f"  Script: {script_name}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            status = "SUCCESS"
            error = None
            print(f"  ✓ SUCCESS")
        else:
            status = "FAILED"
            error = result.stderr
            print(f"  ✗ FAILED")
            if error:
                print(f"    Error: {error[:200]}")

    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        error = "Script execution timed out (60s limit)"
        print(f"  ✗ TIMEOUT")
    except Exception as e:
        status = "ERROR"
        error = str(e)
        print(f"  ✗ ERROR: {error[:100]}")

    print()

    return {
        "script": script_name,
        "title": plot_title,
        "status": status,
        "error": error,
        "stdout": result.stdout if 'result' in locals() else None,
        "stderr": result.stderr if 'result' in locals() else None,
    }

def main():
    print("=" * 70)
    print("OBJECTIVE 1 PLOTTING AUDIT — RAJASTHAN PIPELINE")
    print("=" * 70)
    print()

    # Create directory structure
    create_directory_structure()

    # Run all plotting scripts
    print("=" * 70)
    print("EXECUTING PLOTTING SCRIPTS")
    print("=" * 70)
    print()

    results = []

    print("PART A — Main Plots (13 required)")
    print("-" * 70)
    for script, title in MAIN_PLOTS:
        result = run_plot_script(script, title)
        results.append(result)

    print("\nPART B — Comparison Plots (8 additional)")
    print("-" * 70)
    for script, title in COMPARISON_PLOTS:
        result = run_plot_script(script, title)
        results.append(result)

    # Summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print()

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    missing_count = sum(1 for r in results if r["status"] == "MISSING")
    other_count = len(results) - success_count - failed_count - missing_count

    print(f"Total scripts:   {len(results)}")
    print(f"  ✓ SUCCESS:     {success_count}")
    print(f"  ✗ FAILED:      {failed_count}")
    print(f"  ~ MISSING:     {missing_count}")
    print(f"  ? OTHER:       {other_count}")
    print()

    # Detailed results
    print("Detailed Results:")
    print("-" * 70)
    for r in results:
        status_icon = {
            "SUCCESS": "✓",
            "FAILED": "✗",
            "MISSING": "~",
            "TIMEOUT": "⏱",
            "ERROR": "?"
        }.get(r["status"], "?")

        print(f"{status_icon} {r['script']:<35} {r['status']:<10}")
        if r["error"]:
            print(f"  → {r['error'][:80]}")

    print()
    print("=" * 70)
    print("Output Directory: outputs/objective1_plots_rajasthan/")
    print("=" * 70)
    print()

    # Save report
    report_file = os.path.join(OUTPUT_BASE, "PLOTTING_REPORT.json")
    with open(report_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
            "missing": missing_count,
            "results": results
        }, f, indent=2, default=str)

    print(f"✓ Report saved to: {report_file}")
    print()

    # Overall status
    if failed_count == 0 and missing_count == 0:
        print("✓ All available plots generated successfully!")
        return 0
    else:
        print(f"⚠ {failed_count + missing_count} plots could not be generated")
        return 1

if __name__ == "__main__":
    sys.exit(main())
