"""
Run the whole PLOTSV2 plot set for Rajasthan, in pipeline order.

    python run_all_plots_v2.py            # everything, then assemble Plots/
    python run_all_plots_v2.py objective1 # just the 13 objective-1 plots
    python run_all_plots_v2.py verify     # just the four verification suites
    python run_all_plots_v2.py phases     # just the phase-1 / phase-3 figures
    python run_all_plots_v2.py comparison # just the 8 cross-step comparison plots
    python run_all_plots_v2.py plots      # just re-assemble PLOTSV2/Plots/

Each script is run as its own process so one failure does not take the rest
of the set down with it; the exit summary lists what succeeded.

Phase 1 reads the ~1.4 GB raw points CSV, so a full run takes a few minutes.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))

OBJECTIVE1 = [("13 objective-1 plots", "generate_rajasthan_plots.py")]
VERIFY = [
    ("Preprocessing verification", "verify_01_preprocessing_rajasthan.py"),
    ("Clustering verification",    "verify_02_clustering_rajasthan.py"),
    ("Feasibility verification",   "verify_03_feasibility_rajasthan.py"),
    ("Ranking verification",       "verify_04_ranking_rajasthan.py"),
]
PHASES = [
    ("Phase 1 - data collection",   "phase1_data_collection_rajasthan.py"),
    ("Phase 3 - climate signature", "phase3_climate_signature_rajasthan.py"),
]
COMPARISON = [("8 cross-step comparison plots", "comparison_plots_rajasthan.py")]
ASSEMBLE = [("Curated Plots/ folder", "build_plots_folder_rajasthan.py")]

def run(label, script):
    print("\n" + "=" * 70)
    print(f"  {label}  ->  {script}")
    print("=" * 70)
    t0 = time.time()
    rc = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE).returncode
    dt = time.time() - t0
    print(f"  [{'OK' if rc == 0 else 'FAILED'}] {script}  ({dt:.1f}s)")
    return rc == 0

if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    jobs = {
        "all":        OBJECTIVE1 + VERIFY + PHASES + COMPARISON + ASSEMBLE,
        "objective1": OBJECTIVE1,
        "verify":     VERIFY,
        "phases":     PHASES,
        "comparison": COMPARISON,
        "plots":      ASSEMBLE,
    }.get(which)
    if jobs is None:
        print(f"Unknown selection '{which}'. Use: all | objective1 | verify | phases | comparison | plots")
        raise SystemExit(2)

    results = [(label, run(label, script)) for label, script in jobs]

    print("\n" + "=" * 70)
    print("  PLOTSV2 SUMMARY")
    print("=" * 70)
    for label, ok in results:
        print(f"  [{'OK    ' if ok else 'FAILED'}] {label}")
    print(f"\n  Output root: {HERE}")
    raise SystemExit(0 if all(ok for _, ok in results) else 1)
