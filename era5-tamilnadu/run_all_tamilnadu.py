"""
run_all_tamilnadu.py
=============================================================================
Runs every stage of the Tamil Nadu (population-weighted points) pipeline,
in the correct dependency order, in one invocation — via `subprocess`,
exactly as if you'd typed `python <script>.py` for each one yourself in
sequence (no shared Python process, no import side effects between
stages — each script starts fresh, same as running it manually).

Order and dependencies below are taken directly from this folder's own
PIPELINE_FILE_GUIDE.md ("Quick what do I run and in what order recap"),
cross-checked against each script's actual file I/O — not just filename
numbering (numbering is NOT always run order: `10_physics_validation.py`
must run before `09_recommendation_cards.py`, since 09 is pure aggregation
that includes 10's simulated-solar-fraction output when present).

REQUIRED LIBRARIES: none beyond the standard library (subprocess/pathlib/
argparse/time) — this file itself has no third-party dependencies. Each
individual stage script has its own requirements; this runner doesn't
install anything, it just invokes `python <script>.py` with whatever
interpreter you ran this with.

WHAT THIS DOES NOT RUN, AND WHY (excluded entirely — not part of any
group below):
  05c_explore_interactive.py
  — this is a Streamlit app. It must be launched with
  `streamlit run 05c_explore_interactive.py`, not plain `python`; running
  it via subprocess the way every other stage here is run would not do
  anything useful. Launch it yourself, separately, whenever you want the
  live exploration app.
  05_cluster_regions.py
  — per PIPELINE_FILE_GUIDE.md, this is the original 4-state (TN + RJ +
  Assam + Uttarakhand) design, "not currently used", and does not
  currently exist in this folder.

KNOWN GAP — read before running the CORE chain unattended:
  `06_build_pcm_database.py` reads
  `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` (a sibling
  of this folder, produced by a separate mini-pipeline,
  `PCM_data/01_preprocess.py`, which this runner does NOT run — it lives
  outside era5-tamilnadu-pipeline/ and tamilnadu_pipeline/). As of writing,
  that exact file does not exist anywhere in this repo (only a
  differently-named/differently-nested variant does) — so 06 will FAIL
  with a clear FileNotFoundError-style message until that file is in
  place. This is a genuine, currently-unresolved data gap, not a bug in
  this runner; everything through step 05/11 (Phases 1-4) will still
  succeed independently.

ORDER AND WHY (CORE stages — required, this runner STOPS at the first
failure since every later stage reads an earlier stage's output, EXCEPT
07b which is marked optional below — see that entry):
  1.  02_combine_tamilnadu.py       — Phase 1 final step: merges ERA5 +
                                       NASA POWER at sun-events ->
                                       climate_tamilnadu_points.csv, the
                                       single input every later script
                                       ultimately traces back to
  2.  02b_build_daily_aggregates.py — Phase 2: true daily integrals from
                                       the full NASA POWER hourly cache ->
                                       daily_aggregates_tamilnadu.csv
                                       (also what 10_physics_validation.py
                                       uses to drive its simulation)
  3.  04_preprocess_tamilnadu.py    — Phase 2: 13-step QC pipeline ->
                                       tamilnadu_cleaned_physical.csv,
                                       what every downstream script reads
  4.  04b_climate_signature.py      — Phase 3: collapses each point's
                                       record into one signature vector ->
                                       climate_signature_tamilnadu.csv,
                                       what 05 clusters
  5.  05_cluster_tamilnadu.py       — Phase 4: GMM clustering ->
                                       cluster_profiles_tamilnadu.csv
                                       (K_FINAL is a manual choice you
                                       make by editing the script after
                                       reading bic_selection_tamilnadu.csv
                                       — this runner just runs whatever
                                       K_FINAL is currently set to)
  6.  11_level_b_seasonal_analysis.py — Phase 4: seasonal Top-k re-ranking
                                       within each cluster (documented as
                                       part of Phase 4's official output
                                       alongside 05, unlike the purely
                                       interactive 05b/05c/05d below)
  7.  06_build_pcm_database.py      — Phase 5: builds the PCM candidate
                                       database -> pcm_database_
                                       tamilnadu.csv (see KNOWN GAP above)
  8.  07b_charging_feasibility.py   — Phase 5, OPTIONAL: adds a regime-
                                       capped Tm_target column that 07
                                       prefers if present. Explicitly
                                       labeled optional in
                                       PIPELINE_FILE_GUIDE.md, but must
                                       run BEFORE 07 to take effect, so it
                                       is sequenced here rather than in
                                       the after-the-fact OPTIONAL group.
                                       Its failure does NOT stop the run.
  9.  07_feasibility_filter.py      — Phase 5: hard-filters the PCM
                                       database per cluster ->
                                       feasibility_survivors_by_cluster.csv
  10. 08_mcdm_ranking.py            — Phase 6: TOPSIS/GRA/PROMETHEE II/
                                       VIKOR + Monte Carlo ->
                                       mcdm_topk_by_cluster.csv
  11. 10_physics_validation.py      — Phase 7: grey-box tank simulation ->
                                       physics_validation_results.csv
                                       (numbered 10 but runs BEFORE 09 —
                                       see module docstring intro)
  12. 09_recommendation_cards.py    — Phase 8: pure aggregation of
                                       everything above -> paste-ready
                                       recommendation_cards.md

SETUP stages (one-time raw-data ACQUISITION — excluded by default: these
hit external APIs (CDS/ERA5, NASA POWER, WorldPop/GADM), need credentials
(.cdsapirc) this runner has no way to verify, and can take hours. Pass
--include-setup if you genuinely want them run first, in this order):
  00a_build_population_grid.py  — population-weighted sample points
  00b_build_suntimes.py         — sunrise/noon/sunset times per point/day
  01_download_era5_tamilnadu.py — ERA5 download (needs suntimes.csv)
  01b_download_nasapower.py     — NASA POWER download
  00_unzip_accum.py             — fixes any ZIP-disguised .nc files;
                                   PIPELINE_FILE_GUIDE.md places this
                                   right before 02_combine, after the
                                   downloads, so it runs last within setup

OPTIONAL / DIAGNOSTIC stages (run AFTER the core chain, by default — none
of the CORE_SCRIPTS read their output, so they cannot break the core
chain; each one's failure is logged and does NOT stop the run):
  03_plots_raw.py, 03b_agreement_analysis.py, 03b_interactive_raw_qa.py,
  04c_postprocess_plots.py, 04c_interactive_postprocess_qc.py,
  04d_signature_interactive.py, 05b_cluster_interactive.py,
  05d_plots_comprehensive.py

HOW TO RUN:
  python run_all_tamilnadu.py                 # core pipeline only (default)
  python run_all_tamilnadu.py --with-optional  # core + diagnostic/plot scripts
  python run_all_tamilnadu.py --dry-run        # print the order, run nothing
  python run_all_tamilnadu.py --include-setup  # ALSO run the raw-data
                                                # acquisition scripts FIRST —
                                                # read the warning above and
                                                # in --include-setup's own
                                                # section below before using
  python run_all_tamilnadu.py --from 05_cluster_tamilnadu.py
                                                # resume the CORE chain
                                                # starting at a given script
                                                # (skips everything before
                                                # it) — useful after fixing
                                                # one stage without wanting
                                                # to re-run everything
                                                # upstream of it again

This can take a while end to end — 08_mcdm_ranking.py runs a 5,000-draw
Monte Carlo per cluster, and 10_physics_validation.py simulates a full
representative year for every feasibility survivor per cluster. Expect
anywhere from several minutes to well over 30 minutes for the full core
chain depending on machine speed; this is not a hang.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# One-time raw-data acquisition — excluded by default, see docstring.
SETUP_SCRIPTS = [
    "00a_build_population_grid.py",
    "00b_build_suntimes.py",
    "01_download_era5_tamilnadu.py",
    "01b_download_nasapower.py",
    "00_unzip_accum.py",
]

# Required, sequential, stop-on-failure — EXCEPT the one entry marked
# required=False (07b_charging_feasibility.py), which is explicitly
# documented as optional but must be sequenced before 07 to take effect
# (see module docstring). (name, required) pairs, in run order.
CORE_SCRIPTS = [
    # ("02_combine_tamilnadu.py", True),
    ("02b_build_daily_aggregates.py", True),
    ("04_preprocess_tamilnadu.py", True),
    ("04b_climate_signature.py", True),
    ("05_cluster_tamilnadu.py", True),
    ("11_level_b_seasonal_analysis.py", True),
    ("06_build_pcm_database.py", True),
    ("07b_charging_feasibility.py", False),
    ("07_feasibility_filter.py", True),
    ("08_mcdm_ranking.py", True),
    ("10_physics_validation.py", True),
    ("09_recommendation_cards.py", True),
]

# QC/plotting/diagnostic — run after the core chain, continue-on-failure,
# nothing in CORE_SCRIPTS reads any of these scripts' output.
OPTIONAL_SCRIPTS = [
    "03_plots_raw.py",
    "03b_agreement_analysis.py",
    "03b_interactive_raw_qa.py",
    "04c_postprocess_plots.py",
    "04c_interactive_postprocess_qc.py",
    "04d_signature_interactive.py",
    "05b_cluster_interactive.py",
    "05d_plots_comprehensive.py",
]


def run_script(name, stop_on_failure):
    path = BASE_DIR / name
    if not path.exists():
        print(f"  [SKIP] {name} — file not found at {path}")
        return "skipped", 0.0

    print("\n" + "=" * 68)
    print(f"  RUNNING: {name}")
    print("=" * 68)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] {name} finished in {elapsed:.1f}s")
        return "ok", elapsed

    print(f"\n  [FAILED] {name} exited with code {result.returncode} after {elapsed:.1f}s")
    if stop_on_failure:
        print(f"\n  Stopping — every required stage after {name} in the core chain reads "
              f"its output, so continuing would run against stale/missing data.")
    return "failed", elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Tamil Nadu points pipeline end-to-end, in order.")
    parser.add_argument("--with-optional", action="store_true",
                         help="Also run the QC/diagnostic/plotting scripts after the core chain.")
    parser.add_argument("--include-setup", action="store_true",
                         help="Also run the one-time raw-data download/setup scripts FIRST. "
                              "These hit external APIs and need credentials — see this file's "
                              "own docstring before using this flag.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved run order and exit without running anything.")
    parser.add_argument("--from", dest="from_script", default=None,
                         help="Resume the CORE chain starting at this script name "
                              "(e.g. 05_cluster_tamilnadu.py), skipping everything before it. "
                              "Does not affect --include-setup or --with-optional stages.")
    args = parser.parse_args()

    core = list(CORE_SCRIPTS)
    core_names = [n for n, _ in core]
    if args.from_script:
        if args.from_script not in core_names:
            print(f"ERROR: --from {args.from_script!r} is not one of the core scripts: {core_names}")
            sys.exit(2)
        core = core[core_names.index(args.from_script):]

    setup = list(SETUP_SCRIPTS) if args.include_setup else []
    optional = list(OPTIONAL_SCRIPTS) if args.with_optional else []

    print("=" * 68)
    print("  TAMIL NADU POINTS PIPELINE — RUN ORDER")
    print("=" * 68)
    if setup:
        print("\n  SETUP (one-time, external APIs — --include-setup was passed):")
        for s in setup:
            print(f"    {s}")
    print("\n  CORE (required unless marked optional; stop-on-first-required-failure):")
    for name, required in core:
        tag = "" if required else "  (optional, non-blocking)"
        print(f"    {name}{tag}")
    if optional:
        print("\n  OPTIONAL / DIAGNOSTIC (continue-on-failure):")
        for s in optional:
            print(f"    {s}")
    else:
        print("\n  (Optional/diagnostic scripts skipped — pass --with-optional to include them.)")
    print("\n  NOT RUN (excluded — see docstring): 05c_explore_interactive.py "
          "(Streamlit app, launch with `streamlit run` yourself)")

    if args.dry_run:
        print("\n--dry-run: exiting without running anything.")
        return

    t_start = time.time()
    log = []

    for name in setup:
        status, elapsed = run_script(name, stop_on_failure=True)
        log.append((name, status, elapsed))
        if status == "failed":
            print_summary(log, time.time() - t_start)
            sys.exit(1)

    for name, required in core:
        status, elapsed = run_script(name, stop_on_failure=required)
        log.append((name, status, elapsed))
        if status == "failed" and required:
            print_summary(log, time.time() - t_start)
            sys.exit(1)

    for name in optional:
        status, elapsed = run_script(name, stop_on_failure=False)
        log.append((name, status, elapsed))

    print_summary(log, time.time() - t_start)


def print_summary(log, total_elapsed):
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    for name, status, elapsed in log:
        tag = {"ok": "OK    ", "failed": "FAILED", "skipped": "SKIP  "}[status]
        print(f"  [{tag}] {name:38s} {elapsed:7.1f}s")
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s")
    n_failed = sum(1 for _, s, _ in log if s == "failed")
    if n_failed:
        print(f"  {n_failed} stage(s) FAILED — see the log above for the first failure's output.")
    else:
        print("  All stages completed (or were intentionally skipped).")
    print("=" * 68)


if __name__ == "__main__":
    main()
