"""
run_all_rajasthan.py
=============================================================================
Runs every stage of the Rajasthan pipeline, in the correct dependency
order, in one invocation — via `subprocess`, exactly as if you'd typed
`python <script>.py` for each one yourself in sequence (no shared Python
process, no import side effects between stages — each script starts fresh,
same as running it manually).

REQUIRED LIBRARIES: none beyond the standard library (subprocess/pathlib/
argparse/time) — this file itself has no third-party dependencies. Each
individual stage script has its own requirements (see that script's own
docstring); this runner doesn't install anything, it just invokes
`python <script>.py` with whatever interpreter you ran this with.

WHAT THIS DOES NOT RUN, AND WHY (excluded from the default order below):
  00_unzip_accum.py, 00a_build_population_grid.py, 00b_build_suntimes.py,
  00c_attach_elevation.py, 01_download_era5_rajasthan.py,
  01b_download_nasapower.py
  — these are ONE-TIME raw-data ACQUISITION steps: they hit external APIs
  (CDS/ERA5, NASA POWER), need credentials (.cdsapirc) this runner has no
  way to verify, can take hours, and would re-download/overwrite the raw
  archive this whole pipeline is built on if re-run casually. They are
  NOT part of the "re-run the pipeline" cycle — you run them once when
  setting up a new state/region, not every time you want fresh outputs
  from already-downloaded raw data. Pass --include-setup if you genuinely
  want them included (they will still run in their own correct order,
  first) — read that flag's own warning below before using it.

ORDER AND WHY (CORE stages — required, this runner STOPS at the first
failure since every later stage reads an earlier stage's output):
  1. 02_combine_rajasthan.py            — raw ERA5 events -> climate_
                                           rajasthan_points.csv
  2. 02b_build_daily_aggregates.py      — raw NASA POWER hourly ->
                                           daily_aggregates_rajasthan*.csv
                                           (independent of step 1 — reads
                                           raw NASA POWER directly — but
                                           sequenced after it for a single
                                           linear log to read)
  3. 03b_quality_check_rajasthan.py     — Hampel-filter + impute ->
                                           climate_rajasthan_points_clean.csv
  4. 04_climate_signature_rajasthan.py  — Phase 3: two-tier signature ->
                                           climate_signature_rajasthan.csv
  5. 05_cluster_rajasthan.py            — Phase 4: GMM clustering ->
                                           cluster_profiles_rajasthan.csv
                                           (canonical-relabeled, see that
                                           script's 2026-08-11 fix)
  6. 07_feasibility_filter_rajasthan.py — Phase 5: 8-constraint filter ->
                                           feasibility_survivors_*.csv
  7. 08_mcdm_ranking_rajasthan.py       — Phase 6: 4-method MCDM + Monte
                                           Carlo -> mcdm_rankings_*.csv
  8. 09_physics_validation_rajasthan.py — Phase 7: physics simulation ->
                                           physics_validation_*.csv
  9. 10_recommendation_cards_rajasthan.py — Phase 8: pure aggregation ->
                                           recommendation_cards_*.md

  Steps 6-9 (Phase 5-8) each independently hard-fail via provenance_lib.py
  if their input was built from a DIFFERENT on-disk cluster_profiles_
  rajasthan.csv than the one currently present — running this script
  start-to-finish in one pass is exactly what keeps that chain consistent
  (see provenance_lib.py's module docstring for the 2026-08-11 bug this
  guards against).

OPTIONAL / DIAGNOSTIC stages (run AFTER the core chain succeeds, by
default — none of steps 1-8 read their output, so they cannot break the
core chain; each one's failure is logged and does NOT stop the run,
unlike the core chain above):
  03_verify_climate_csv.py, 03_qc_plots.py, 03b_agreement_analysis.py,
  03c_plots_raw_rajasthan.py, 03b_quality_check_plots_rajasthan.py,
  03b_validate_quality_fix_rajasthan.py (LAST among optionals — it
  re-runs 04_climate_signature_rajasthan.py a SECOND time internally as
  part of its own before/after diff, which is expected, not a bug in
  this runner)

HOW TO RUN:
  python run_all_rajasthan.py                 # core pipeline only (default)
  python run_all_rajasthan.py --with-optional  # core + diagnostic/plot scripts
  python run_all_rajasthan.py --dry-run        # print the order, run nothing
  python run_all_rajasthan.py --include-setup  # ALSO run the 00/01 raw-data
                                                # acquisition scripts FIRST —
                                                # read the warning above and
                                                # in --include-setup's own
                                                # section below before using
  python run_all_rajasthan.py --from 05_cluster_rajasthan.py
                                                # resume the CORE chain
                                                # starting at a given script
                                                # (skips everything before
                                                # it) — useful after fixing
                                                # one stage without wanting
                                                # to re-run everything
                                                # upstream of it again

This can take a while end to end — 02_combine_rajasthan.py alone processes
a multi-GB combined events file, and 08/09 each run Monte Carlo / full-year
hourly simulations. Expect anywhere from several minutes to ~30+ minutes
for the full core chain depending on machine speed; this is not a hang.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# One-time raw-data acquisition — excluded by default, see docstring.
SETUP_SCRIPTS = [
    "00_unzip_accum.py",
    "00a_build_population_grid.py",
    "00b_build_suntimes.py",
    "00c_attach_elevation.py",
    "01_download_era5_rajasthan.py",
    "01b_download_nasapower.py",
]

# Required, sequential, stop-on-failure — every later stage reads an
# earlier one's output (see docstring for the full dependency reasoning).
CORE_SCRIPTS = [
    # "02_combine_rajasthan.py",
    "02b_build_daily_aggregates.py",
    "03b_quality_check_rajasthan.py",
    "04_climate_signature_rajasthan.py",
    "05_cluster_rajasthan.py",
    "07_feasibility_filter_rajasthan.py",
    "08_mcdm_ranking_rajasthan.py",
    "09_physics_validation_rajasthan.py",
    "10_recommendation_cards_rajasthan.py",
]

# QC/plotting/diagnostic — run after the core chain, continue-on-failure,
# nothing in CORE_SCRIPTS reads any of these scripts' output.
OPTIONAL_SCRIPTS = [
    "03_verify_climate_csv.py",
    "03_qc_plots.py",
    "03b_agreement_analysis.py",
    "03c_plots_raw_rajasthan.py",
    "03b_quality_check_plots_rajasthan.py",
    "03b_validate_quality_fix_rajasthan.py",   # last: re-runs 04 internally
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
        print(f"\n  Stopping — every stage after {name} in the core chain reads its output, "
              f"so continuing would run against stale/missing data.")
    return "failed", elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Rajasthan pipeline end-to-end, in order.")
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
                              "(e.g. 05_cluster_rajasthan.py), skipping everything before it. "
                              "Does not affect --include-setup or --with-optional stages.")
    args = parser.parse_args()

    core = list(CORE_SCRIPTS)
    if args.from_script:
        if args.from_script not in core:
            print(f"ERROR: --from {args.from_script!r} is not one of the core scripts: {core}")
            sys.exit(2)
        core = core[core.index(args.from_script):]

    setup = list(SETUP_SCRIPTS) if args.include_setup else []
    optional = list(OPTIONAL_SCRIPTS) if args.with_optional else []

    print("=" * 68)
    print("  RAJASTHAN PIPELINE — RUN ORDER")
    print("=" * 68)
    if setup:
        print("\n  SETUP (one-time, external APIs — --include-setup was passed):")
        for s in setup:
            print(f"    {s}")
    print("\n  CORE (required, stop-on-first-failure):")
    for s in core:
        print(f"    {s}")
    if optional:
        print("\n  OPTIONAL / DIAGNOSTIC (continue-on-failure):")
        for s in optional:
            print(f"    {s}")
    else:
        print("\n  (Optional/diagnostic scripts skipped — pass --with-optional to include them.)")

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

    for name in core:
        status, elapsed = run_script(name, stop_on_failure=True)
        log.append((name, status, elapsed))
        if status == "failed":
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
        print(f"  [{tag}] {name:45s} {elapsed:7.1f}s")
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s")
    n_failed = sum(1 for _, s, _ in log if s == "failed")
    if n_failed:
        print(f"  {n_failed} stage(s) FAILED — see the log above for the first failure's output.")
    else:
        print("  All stages completed (or were intentionally skipped).")
    print("=" * 68)


if __name__ == "__main__":
    main()
