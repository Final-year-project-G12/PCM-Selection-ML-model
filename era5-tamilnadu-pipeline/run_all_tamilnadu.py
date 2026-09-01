"""
run_all_tamilnadu.py
=============================================================================
Runs every stage of the (older, 6-script) ERA5 Tamil Nadu pipeline, in the
correct dependency order, in one invocation — via `subprocess`, exactly as
if you'd typed `python <script>.py` for each one yourself in sequence (no
shared Python process, no import side effects between stages — each script
starts fresh, same as running it manually).

REQUIRED LIBRARIES: none beyond the standard library (subprocess/pathlib/
argparse/time) — this file itself has no third-party dependencies. Each
individual stage script has its own requirements (see that script's own
docstring); this runner doesn't install anything, it just invokes
`python <script>.py` with whatever interpreter you ran this with.

WHAT THIS DOES NOT RUN, AND WHY (excluded entirely — not part of any
group below):
  03_explore_raw_tamilnadu.py
  — this is a Streamlit app. It must be launched with
  `streamlit run 03_explore_raw_tamilnadu.py`, not plain `python`; running
  it via subprocess the way every other stage here is run would not do
  anything useful (no server, no browser tab). Launch it yourself,
  separately, whenever you want the interactive explorer.

ORDER AND WHY (CORE stages — required, this runner STOPS at the first
failure since every later stage reads an earlier stage's output):
  1. 02_combine_tamilnadu.py    — raw ERA5 NetCDF -> climate_
                                   tamilnadu_all.csv (+ per-location and
                                   full-grid CSVs)
  2. 04_preprocess_tamilnadu.py — cleaning, imputation, feature
                                   engineering, train/val/test split ->
                                   data/preprocessed/*
  3. 05_plot_tamilnadu.py       — all visualisations -> data/plots/*
                                   (reads 02's output directly; also reads
                                   04's output for its D-series feature
                                   plots, so it must run after both)

SETUP stages (one-time raw-data ACQUISITION/fixup — excluded by default,
same reasoning as the Rajasthan/Tamil Nadu points-pipeline runners: these
hit an external API (CDS/ERA5), need credentials (.cdsapirc) this runner
has no way to verify, and can take a long time. Pass --include-setup if
you genuinely want them run first):
  00_unzip_accum.py             — fixes any ZIP-disguised .nc files CDS
                                   sometimes returns; run once before 02,
                                   safe to re-run (skips valid files)
  01_download_era5_tamilnadu.py — downloads the raw ERA5 NetCDF archive;
                                   safe to re-run (skips completed files)

HOW TO RUN:
  python run_all_tamilnadu.py                 # core pipeline only (default)
  python run_all_tamilnadu.py --dry-run        # print the order, run nothing
  python run_all_tamilnadu.py --include-setup  # ALSO run 00/01 raw-data
                                                # acquisition FIRST — read
                                                # the warning above before
                                                # using this flag
  python run_all_tamilnadu.py --from 05_plot_tamilnadu.py
                                                # resume the CORE chain
                                                # starting at a given script
                                                # (skips everything before
                                                # it)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# One-time raw-data acquisition/fixup — excluded by default, see docstring.
SETUP_SCRIPTS = [
    "00_unzip_accum.py",
    "01_download_era5_tamilnadu.py",
]

# Required, sequential, stop-on-failure — each later stage reads an
# earlier one's output (see docstring for the full dependency reasoning).
CORE_SCRIPTS = [
    "04_preprocess_tamilnadu.py",
    "05_plot_tamilnadu.py",
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
    parser = argparse.ArgumentParser(description="Run the (older) Tamil Nadu ERA5 pipeline end-to-end, in order.")
    parser.add_argument("--include-setup", action="store_true",
                         help="Also run the one-time raw-data download/fixup scripts FIRST. "
                              "These hit an external API and need credentials — see this file's "
                              "own docstring before using this flag.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved run order and exit without running anything.")
    parser.add_argument("--from", dest="from_script", default=None,
                         help="Resume the CORE chain starting at this script name "
                              "(e.g. 05_plot_tamilnadu.py), skipping everything before it. "
                              "Does not affect --include-setup.")
    args = parser.parse_args()

    core = list(CORE_SCRIPTS)
    if args.from_script:
        if args.from_script not in core:
            print(f"ERROR: --from {args.from_script!r} is not one of the core scripts: {core}")
            sys.exit(2)
        core = core[core.index(args.from_script):]

    setup = list(SETUP_SCRIPTS) if args.include_setup else []

    print("=" * 68)
    print("  TAMIL NADU (era5-tamilnadu-pipeline) — RUN ORDER")
    print("=" * 68)
    if setup:
        print("\n  SETUP (one-time, external API — --include-setup was passed):")
        for s in setup:
            print(f"    {s}")
    print("\n  CORE (required, stop-on-first-failure):")
    for s in core:
        print(f"    {s}")
    print("\n  NOT RUN (excluded — see docstring): 03_explore_raw_tamilnadu.py "
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

    for name in core:
        status, elapsed = run_script(name, stop_on_failure=True)
        log.append((name, status, elapsed))
        if status == "failed":
            print_summary(log, time.time() - t_start)
            sys.exit(1)

    print_summary(log, time.time() - t_start)


def print_summary(log, total_elapsed):
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    for name, status, elapsed in log:
        tag = {"ok": "OK    ", "failed": "FAILED", "skipped": "SKIP  "}[status]
        print(f"  [{tag}] {name:35s} {elapsed:7.1f}s")
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s")
    n_failed = sum(1 for _, s, _ in log if s == "failed")
    if n_failed:
        print(f"  {n_failed} stage(s) FAILED — see the log above for the first failure's output.")
    else:
        print("  All stages completed (or were intentionally skipped).")
    print("=" * 68)


if __name__ == "__main__":
    main()
