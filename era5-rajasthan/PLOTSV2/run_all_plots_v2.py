"""
Run the whole PLOTSV2 plot set for Rajasthan, in pipeline order. Every
script writes under ../data/plots/ (see ../../docs/PLOT_NAMING.md).

    python run_all_plots_v2.py            # everything, then assemble data/plots/ppt/
    python run_all_plots_v2.py objective1 # just the 13 objective-1 plots
    python run_all_plots_v2.py verify     # just the four verification suites
    python run_all_plots_v2.py phases     # just the phase-3 signature figures
    python run_all_plots_v2.py comparison # just the 8 cross-step comparison plots
    python run_all_plots_v2.py plots      # just re-assemble data/plots/ppt/

Each script is run as its own process so one failure does not take the rest
of the set down with it; the exit summary lists what succeeded.

The phase-1 raw-data figures come from ../03_plots_raw.py, which
run_all_rajasthan.py --with-optional runs.
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
    ("Phase 3 - climate signature", "phase3_climate_signature_rajasthan.py"),
]
COMPARISON = [
    ("8 cross-step comparison plots",  "comparison_plots_rajasthan.py"),
    ("MCDM vs physics agreement",      "09_mcdm_vs_physics_agreement.py"),
    ("Phase 3 - Tm_cap old vs new",    "comparison_phase3_tmcap_old_vs_new.py"),
    ("Phase 5 - L_required before/after", "comparison_phase5_lrequired_before_after.py"),
]
ASSEMBLE = [("Curated data/plots/ppt/ folder", "build_plots_folder_rajasthan.py")]

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
    print(f"\n  Output root: {os.path.normpath(os.path.join(HERE, '..', 'data', 'plots'))}")
    raise SystemExit(0 if all(ok for _, ok in results) else 1)
