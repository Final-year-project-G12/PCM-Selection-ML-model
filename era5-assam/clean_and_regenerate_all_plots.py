"""
clean_and_regenerate_all_plots.py
=================================
Master script to:
1. Safely remove all old/stale plot artifacts across Assam visualization directories.
2. Regenerate all authoritative, publication-quality plots from scratch using the
   corrected, audited Assam dataset.
3. Synchronize all folders (data/plots/, plots/, final_outputs/visuals/, plots_assam_ppt/).
4. Report an exhaustive audit of all newly generated figures.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Target directories to clean
CLEAN_DIRS = [
    BASE_DIR / "data" / "plots",
    BASE_DIR / "plots",
    BASE_DIR / "plots_assam_ppt",
    BASE_DIR / "phase10_visualizations",
]

def clean_old_plots():
    print("=" * 78)
    print("STEP 1: PURGING ALL OLD PLOT ARTIFACTS")
    print("=" * 78)
    deleted_count = 0
    for target_dir in CLEAN_DIRS:
        if target_dir.exists():
            for root, dirs, files in os.walk(target_dir):
                for f in files:
                    if f.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html")):
                        file_path = os.path.join(root, f)
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"  [ERROR] Deleting {file_path}: {e}")
            print(f"  Purged plot files from: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Also clean visuals in final_outputs/visuals that are plot images
    visuals_dir = BASE_DIR / "final_outputs" / "visuals"
    if visuals_dir.exists():
        for f in os.listdir(visuals_dir):
            if f.endswith((".png", ".jpg", ".html")):
                try:
                    os.remove(visuals_dir / f)
                    deleted_count += 1
                except Exception as e:
                    print(f"  [ERROR] Deleting {visuals_dir / f}: {e}")
        print(f"  Purged plot files from: {visuals_dir}")
    visuals_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DONE] Purged a total of {deleted_count} old plot files.\n")

def run_step(script_name, description):
    print("-" * 78)
    print(f"RUNNING: {script_name} — {description}")
    print("-" * 78)
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"  [SKIP] Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"  [FAIL] {script_name} failed with exit code {res.returncode}")
        print("  STDERR:\n" + res.stderr[-800:])
        return False
    else:
        # Print concise summary
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        for line in lines[-8:]:
            print(f"  {line}")
        print(f"  [SUCCESS] {script_name} finished successfully.")
        return True

def sync_plots_folder():
    print("-" * 78)
    print("SYNCHRONIZING: data/plots/ -> plots/")
    print("-" * 78)
    src = BASE_DIR / "data" / "plots"
    dst = BASE_DIR / "plots"
    copied = 0
    if src.exists():
        for root, dirs, files in os.walk(src):
            rel_path = os.path.relpath(root, src)
            target_sub = dst / rel_path
            target_sub.mkdir(parents=True, exist_ok=True)
            for f in files:
                src_file = os.path.join(root, f)
                dst_file = target_sub / f
                shutil.copy2(src_file, dst_file)
                copied += 1
    print(f"  Synchronized {copied} files into era5-assam/plots/\n")

def main():
    # 1. Clean
    clean_old_plots()

    # 2. Sequential regeneration of all plot suites
    pipeline = [
        ("generate_assam_plots.py", "Objective 1 Climate Plots (Radiation & Regimes)"),
        ("verify_01_preprocessing_assam.py", "Verification 01: Preprocessing & Data Quality"),
        ("generate_era5_nasa_comparison_plots.py", "ERA5 vs NASA POWER Satellite Agreement"),
        ("generate_missing_assam_diagrams.py", "Parity Diagrams (Population Grid, Scree, Diurnal Cycle)"),
        ("verify_02_clustering_assam.py", "Verification 02: Clustering Analysis & Validation"),
        ("generate_assam_silhouette.py", "High-Resolution Silhouette Analysis (k=3)"),
        ("verify_03_feasibility_assam.py", "Verification 03: PCM Feasibility Filtering"),
        ("verify_04_ranking_assam.py", "Verification 04: MCDM Ranking & Uncertainty"),
        ("comparison_plots_assam.py", "Cross-Pipeline Comparison Figures"),
        ("generate_phase11_figures.py", "Phase 11 Final Regimes & PCA Projections"),
        ("10_validation_comparison.py", "Phase 10 Physics Validation Comparisons"),
        ("generate_phase11_consolidation.py", "Consolidate Deliverables & Visuals"),
        ("build_plots_assam_ppt.py", "Build Complete Curated PPT Results Suite"),
    ]

    success_count = 0
    for script, desc in pipeline:
        ok = run_step(script, desc)
        if ok:
            success_count += 1

    # 3. Synchronize data/plots to plots
    sync_plots_folder()

    # 4. Final Audit
    print("=" * 78)
    print("FINAL SUMMARY AUDIT OF GENERATED FIGURES")
    print("=" * 78)
    total_new = 0
    for folder in [BASE_DIR / "data" / "plots", BASE_DIR / "final_outputs" / "visuals", BASE_DIR / "plots_assam_ppt", BASE_DIR / "phase10_visualizations"]:
        count = sum(1 for root, _, files in os.walk(folder) for f in files if f.endswith((".png", ".html")))
        total_new += count
        print(f"  {folder.relative_to(BASE_DIR)}: {count} active figures")
    print(f"\nTotal new figures generated from corrected dataset: {total_new}")
    print(f"Stages completed: {success_count}/{len(pipeline)}")
    print("=" * 78)

if __name__ == "__main__":
    main()
